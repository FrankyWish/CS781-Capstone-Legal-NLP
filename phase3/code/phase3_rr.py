#!/usr/bin/env python3
"""CS-781 Phase 3 -- hierarchical rhetorical-role labelling on a Phase-1 trunk.

    python phase3_rr.py --model M2_LexArcED --method full  --tag m2_full
    python phase3_rr.py --model M4_LexArcP  --method lora --rank 16 --tag m4_lora
    python phase3_rr.py --model M1_SC --method full --seq none --enc single --tag m1_ctx0
    python phase3_rr.py --model M2_LexArcED --method full --keep-every 2 \
        --kd-logits results/ens_A/teacher.pt --kd-test --tag kd_m2_half

THE MODEL

    sentences of a window of S consecutive sentences
      -> Phase-1 trunk over chunks of packed sentences (--enc packed, the trunk
         sees neighbouring sentences) or one sentence per chunk (--enc single)
      -> mean-pool each sentence's tokens            -> one vector per sentence
      -> + domain embedding + position-in-document features
      -> sequence model over the sentence vectors    (--seq tf | lstm | none)
      -> per-sentence logits, optionally a linear-chain CRF (--crf)

Training samples random windows of S sentences; inference slides windows with
50% overlap and averages each sentence's emissions, then Viterbi over the WHOLE
document when a CRF is used.

THE CONTEXT ABLATION (a required analysis) is three flags on one model:
    sentence alone          --enc single --seq none
    sentence + k neighbours --enc single --seq tf --band k
    full hierarchical       --enc packed --seq tf          (default)

EVALUATION PROTOCOL

5-fold document-level CV over train+dev (split.json from phase3_data.py). Each
fold model is early-stopped on its own held-out fold, its held-out logits are
saved (out-of-fold, OOF), and its test logits are saved at the best epoch. The
submission averages the 5 fold models' test logits: every model that touches
the test set was validated, and there is no unvalidated full refit (Phase 2
lesson 6). A fold whose results file exists is skipped, so a disconnected run
resumes where it stopped.

Everything needed offline is written to results/<tag>/: OOF and test logits
per fold, CRF parameters, the per-epoch trace, and meta.json. phase3_ens.py
ensembles and decodes from those files without a GPU.
"""

from __future__ import annotations

import argparse
import collections
import copy
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from phase3_data import (P1, P3, LABELS, DOMAINS, load_all, log as _log,  # noqa: E402
                         prf, report, write_submission, ensure_split)
from phase3_peft import (apply_method, blocks_of, trunk_param_ids,       # noqa: E402
                         PrefixEncoder)
sys.path.insert(0, f"{P1}/shared/lib")

NL = len(LABELS)
DOM_ID = {"CL": 0, "IT": 1, "EXT": 2}
LOGF = None


def log(*a):
    _log(*a)
    if LOGF:
        print(*a, file=LOGF, flush=True)


# ------------------------------------------------------------------- trunk --
def load_trunk(name, ckpt, keep_every=1, path=None):
    from cs781_model import Config, LexArcD
    ck = torch.load(path or f"{P1}/{name}/ckpt/{ckpt}", map_location="cpu", weights_only=False)
    cfg_d = ck["cfg"]
    if "n_enc_layer" in cfg_d:
        from cs781_model_ed import EDConfig, LexArcED
        cfg = EDConfig(**cfg_d)
        t = LexArcED(cfg)
        t.load_state_dict(ck["model"], strict=False)
        t.dec = nn.ModuleList()                 # only the encoder is used
        for attr in ("dec_norm", "head"):
            if hasattr(t, attr):
                delattr(t, attr)
        kind = "ed"
        if keep_every > 1:
            t.enc = nn.ModuleList(list(t.enc)[::keep_every])
    else:
        cfg = Config(**cfg_d)
        if getattr(cfg, "use_pointer", False):
            from cs781_model_ptr import LexArcP
            t = LexArcP(cfg)
        else:
            t = LexArcD(cfg)
        t.load_state_dict(ck["model"], strict=False)
        for attr in ("head", "cq", "ck", "cv", "gate"):
            if hasattr(t, attr):
                delattr(t, attr)
        kind = "dec"
        if keep_every > 1:
            # Layer 0 must stay first: it produces the value-residual v0 that
            # every later layer's gate mixes in.
            t.blocks = nn.ModuleList(list(t.blocks)[::keep_every])
    return t, kind, cfg


# --------------------------------------------------------------------- CRF --
class CRF(nn.Module):
    """Linear-chain CRF over the sentence sequence, with set-valued supervision.

    nll = log Z(all paths) - log Z(paths consistent with the allowed label sets).
    With one-hot sets this is the ordinary CRF likelihood; with a coarse external
    label ("some kind of Argument") it marginalises over the labels the set
    allows; a sentence with no information allows everything and contributes no
    constraint.
    """

    def __init__(self, n):
        super().__init__()
        self.trans = nn.Parameter(torch.zeros(n, n))
        self.start = nn.Parameter(torch.zeros(n))
        self.end = nn.Parameter(torch.zeros(n))

    def _logZ(self, emit, mask):
        a = self.start + emit[:, 0]
        for t in range(1, emit.size(1)):
            nxt = torch.logsumexp(a.unsqueeze(2) + self.trans.unsqueeze(0), 1) + emit[:, t]
            a = torch.where(mask[:, t:t + 1], nxt, a)
        return torch.logsumexp(a + self.end, 1)

    def nll(self, emit, allowed, mask):
        emit = emit.float()
        con = emit.masked_fill(~allowed, -1e4)
        return self._logZ(emit, mask) - self._logZ(con, mask)


@torch.no_grad()
def viterbi(emit, trans, start, end):
    """emit (T, n) -> list of label indices, on CPU tensors."""
    a = start + emit[0]
    back = []
    for t in range(1, emit.size(0)):
        sc = a.unsqueeze(1) + trans
        best, idx = sc.max(0)
        a = best + emit[t]
        back.append(idx)
    path = [int((a + end).argmax())]
    for idx in reversed(back):
        path.append(int(idx[path[-1]]))
    return path[::-1]


# ------------------------------------------------------------------- model --
def sinusoid(pos, dim):
    inv = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=pos.device).float() / dim))
    f = pos.float()[..., None] * inv
    return torch.cat([f.sin(), f.cos()], -1)


class RRModel(nn.Module):
    def __init__(self, trunk, kind, cfg, a):
        super().__init__()
        self.trunk, self.kind, self.a = trunk, kind, a
        d = cfg.d_model
        self.d = d
        self.R = getattr(cfg, "n_register", 0) if kind == "dec" else 0
        self.bi = not a.causal
        self.P = a.n_virtual if a.method in ("prompt", "prefix") else 0
        self.prompt = None
        self.prefix = None
        if a.method == "prompt":
            # Lester et al. initialise from vocabulary embeddings, not noise.
            ids = torch.randint(256, trunk.emb.num_embeddings, (self.P,))
            self.prompt = nn.Parameter(trunk.emb.weight[ids].detach().clone())
        elif a.method == "prefix":
            self.prefix = PrefixEncoder(len(blocks_of(trunk)), self.P, d)
        h = a.seq_dim
        self.drop = nn.Dropout(a.dropout)
        self.proj = nn.Linear(d, h)
        self.dom = nn.Embedding(3, h) if a.domain_emb else None
        self.posb = nn.Embedding(a.pos_buckets, h) if a.pos_buckets else None
        if a.seq == "tf":
            layer = nn.TransformerEncoderLayer(h, a.seq_heads, 4 * h, a.dropout,
                                               batch_first=True, norm_first=True)
            self.seq = nn.TransformerEncoder(layer, a.seq_layers, enable_nested_tensor=False)
        elif a.seq == "lstm":
            self.seq = nn.LSTM(h, h // 2, num_layers=a.seq_layers, batch_first=True,
                               bidirectional=True, dropout=a.dropout if a.seq_layers > 1 else 0)
        else:
            self.seq = None
        self.out_norm = nn.LayerNorm(h)
        self.head = nn.Linear(h, NL)
        # Label-shift auxiliary head (Malik et al. 2022, the paper behind this
        # dataset): does the role change between sentence i-1 and i? Trained
        # with BCE beside the role loss; it sharpens segment boundaries, which
        # is where Fact/Ratio/RLC and ArgP/ArgR errors concentrate.
        self.shift = nn.Linear(2 * h, 1) if getattr(a, "shift_aux", 0) > 0 else None
        self.last_shift = None
        self.crf = CRF(NL) if a.crf else None

    # ---- token level
    def token_hidden(self, idx, keep):
        trunk, B = self.trunk, idx.size(0)
        x = trunk.emb(idx)
        pre = []
        if self.kind == "dec" and self.R:
            pre.append(trunk.reg.expand(B, -1, -1).to(x.dtype))
        if self.P:
            v = self.prompt if self.prompt is not None else x.new_zeros(self.P, self.d)
            pre.append(v.unsqueeze(0).expand(B, -1, -1).to(x.dtype))
        lead = sum(p.size(1) for p in pre)
        if pre:
            x = torch.cat(pre + [x], 1)
        T = x.size(1)
        from cs781_model import build_rope
        cos, sin = build_rope(T, trunk.cfg.head_dim, x.device, trunk.emb.weight.dtype,
                              trunk.cfg.rope_theta)
        kfull = torch.cat([keep.new_ones(B, lead), keep], 1) if lead else keep
        mask = None
        if self.kind == "ed" or self.bi:
            nh = trunk.cfg.n_head
            mask = kfull[:, None, None, :].expand(-1, nh, T, -1)
        pref = self.prefix() if self.prefix is not None else None
        p0 = self.R                                     # prefix sits after the registers
        # Gradient checkpointing: keep only each layer's input, recompute the
        # rest in backward. ~10x less activation memory for ~30% more compute,
        # which is what lets three jobs share one 22 GB L4.
        ckpt = self.training and self.a.grad_ckpt and torch.is_grad_enabled()

        def call(fn, *args):
            if ckpt:
                return torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=False)
            return fn(*args)

        if self.kind == "dec":
            v0 = None
            for i, blk in enumerate(trunk.blocks):
                if pref is not None:
                    x = torch.cat([x[:, :p0], pref[i].unsqueeze(0).expand(B, -1, -1).to(x.dtype),
                                   x[:, p0 + self.P:]], 1)
                if v0 is None:
                    x, v = call(lambda h, b=blk: b(h, cos, sin, None, None, mask)[:2], x)
                else:
                    x, v = call(lambda h, r, b=blk: b(h, cos, sin, r, None, mask)[:2], x, v0)
                if i == 0:
                    v0 = v
            x = trunk.norm(x)
        else:
            from cs781_model import apply_rope
            from cs781_model_ed import _apply_ffn

            def enc_layer(x, blk):
                at = blk.attn
                hN = blk.n1(x)
                q = at.wq(hN).view(B, T, at.nh, at.hd).transpose(1, 2)
                k = at.wk(hN).view(B, T, at.nkv, at.hd).transpose(1, 2)
                v = at.wv(hN).view(B, T, at.nkv, at.hd).transpose(1, 2)
                q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
                if at.cfg.qk_norm:
                    q = F.rms_norm(q, (at.hd,))
                    k = F.rms_norm(k, (at.hd,))
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)
                x = x + at.wo(y.transpose(1, 2).reshape(B, T, at.nh * at.hd))
                return x + _apply_ffn(blk.ffn, blk.n2(x))

            for i, blk in enumerate(trunk.enc):
                if pref is not None:
                    x = torch.cat([pref[i].unsqueeze(0).expand(B, -1, -1).to(x.dtype),
                                   x[:, self.P:]], 1)
                x = call(lambda h, b=blk: enc_layer(h, b), x)
            x = trunk.enc_norm(x)
        return x[:, lead:]

    # ---- sentence level
    def forward(self, bt):
        H = self.token_hidden(bt["idx"], bt["keep"])            # (C, T, d)
        flat = H.reshape(-1, self.d).float()
        ts = bt["tsent"].reshape(-1)
        sel = ts >= 0
        n = bt["n_sent"]
        vec = flat.new_zeros(n, self.d).index_add_(0, ts[sel], flat[sel])
        cnt = torch.bincount(ts[sel], minlength=n).clamp(min=1).to(vec.dtype)
        vec = vec / cnt[:, None]
        B, S = bt["valid"].shape
        X = vec.new_zeros(B * S, self.d)
        X[bt["slot"]] = vec
        X = self.proj(self.drop(X.view(B, S, self.d)))
        if self.dom is not None:
            X = X + self.dom(bt["dom"])
        if self.posb is not None:
            rel = (bt["gidx"].float() / bt["dlen"].float().clamp(min=1))
            X = X + self.posb((rel * self.a.pos_buckets).long().clamp(max=self.a.pos_buckets - 1))
        X = X + 0.1 * sinusoid(bt["gidx"], X.size(-1))
        valid = bt["valid"]
        if isinstance(self.seq, nn.TransformerEncoder):
            # One (B*heads, S, S) mask: padded keys hidden, the band applied, and
            # the diagonal always open so no query row is fully masked (a fully
            # masked row is a NaN, and a banded padded row would be one).
            ii = torch.arange(S, device=X.device)
            ok = valid[:, None, :].expand(B, S, S)
            if self.a.band >= 0:
                ok = ok & ((ii[:, None] - ii[None, :]).abs() <= self.a.band)
            ok = ok | torch.eye(S, dtype=torch.bool, device=X.device)
            am = (~ok).repeat_interleave(self.a.seq_heads, 0)
            X = self.seq(X, mask=am)
        elif isinstance(self.seq, nn.LSTM):
            lens = valid.sum(1).cpu()
            pk = nn.utils.rnn.pack_padded_sequence(X, lens, batch_first=True, enforce_sorted=False)
            out, _ = self.seq(pk)
            X, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=S)
        Z = self.drop(self.out_norm(X))
        if self.shift is not None:
            prev = torch.cat([Z[:, :1] * 0, Z[:, :-1]], 1)
            self.last_shift = self.shift(torch.cat([prev, Z], -1)).squeeze(-1)   # (B, S)
        return self.head(Z)                                     # (B, S, NL)


# -------------------------------------------------------------------- data --
def tokenize_docs(docs, a):
    """{doc: [[ids] per sentence]}, cached on Drive so it is paid once."""
    from cs781_bpe import BPETokenizer
    import hashlib
    cache_dir = f"{P3}/cache"
    os.makedirs(cache_dir, exist_ok=True)
    # One cache file per document set, written atomically: several runs start
    # at once in parallel queues and must never read a half-written file.
    key = hashlib.sha1("|".join(sorted(docs)).encode()).hexdigest()[:12]
    cp = f"{cache_dir}/tokens_{key}.pt"
    cache = {}
    if os.path.exists(cp):
        try:
            cache = torch.load(cp, weights_only=False)
        except Exception as e:                      # truncated by a killed run
            log(f"  token cache unreadable ({e}); rebuilding")
    todo = [d for d in docs if d not in cache or len(cache[d]) != len(docs[d]["text"])]
    if todo:
        tok = BPETokenizer.load(f"{P1}/shared/tokenizer/bpe_legal_32k.json")
        dot = tok.encode(" .")
        t0 = time.time()
        for d in todo:
            cache[d] = [tok.encode(" " + s.strip()) or dot for s in docs[d]["text"]]
        tmp = f"{cp}.tmp{os.getpid()}"
        torch.save(cache, tmp)
        os.replace(tmp, cp)
        log(f"  tokenised {len(todo)} documents in {time.time()-t0:.0f}s -> {cp}")
    out = {}
    m, tail = a.sent_max, a.sent_max // 4
    for d in docs:
        out[d] = [t if len(t) <= m else t[:m - tail] + t[-tail:] for t in cache[d]]
    return out


def doc_windows(n, S, stride):
    if n <= S:
        return [(0, n)]
    out, i = [], 0
    while True:
        out.append((i, min(n, i + S)))
        if i + S >= n:
            return out
        i += stride


def collate(wins, toks, docs, a, device, wlab=None, kd=None):
    chunks, cur, cur_s = [], [], []
    slot, info = [], []
    for b, (d, st, en) in enumerate(wins):
        for i, s in enumerate(range(st, en)):
            t = toks[d][s]
            k = len(info)
            info.append((b, i, d, s))
            slot.append(b * a.window + i)
            if cur and (a.enc == "single" or len(cur) + len(t) > a.chunk):
                chunks.append((cur, cur_s))
                cur, cur_s = [], []
            cur = cur + t
            cur_s = cur_s + [k] * len(t)
        if cur:
            chunks.append((cur, cur_s))
            cur, cur_s = [], []
    T = max(len(c) for c, _ in chunks)
    C = len(chunks)
    idx = torch.zeros(C, T, dtype=torch.long)
    keep = torch.zeros(C, T, dtype=torch.bool)
    tsent = torch.full((C, T), -1, dtype=torch.long)
    for j, (c, s) in enumerate(chunks):
        idx[j, :len(c)] = torch.tensor(c)
        keep[j, :len(c)] = True
        tsent[j, :len(s)] = torch.tensor(s)
    B, S = len(wins), a.window
    valid = torch.zeros(B, S, dtype=torch.bool)
    gidx = torch.zeros(B, S, dtype=torch.long)
    dlen = torch.ones(B, S, dtype=torch.long)
    dom = torch.zeros(B, S, dtype=torch.long)
    allowed = torch.ones(B, S, NL, dtype=torch.bool)
    w = torch.zeros(B, S)
    kdp = torch.zeros(B, S, NL)
    kdm = torch.zeros(B, S, dtype=torch.bool)
    for (b, i, d, s) in info:
        valid[b, i] = True
        gidx[b, i] = s
        dlen[b, i] = len(docs[d]["text"])
        dom[b, i] = DOM_ID[docs[d]["domain"]]
        al = docs[d].get("allowed")
        if al is not None:
            allowed[b, i] = al[s]
            if wlab is not None and not bool(al[s].all()):
                w[b, i] = (wlab[al[s]].mean() if al[s].sum() > 1 else wlab[al[s]].sum()) \
                    * docs[d].get("weight", 1.0)
        if kd is not None and d in kd:
            t = kd[d][s]
            if al is not None and 1 < int(al[s].sum()) < NL:
                # Coarse external label ("some Argument"): keep the teacher's
                # view only over the roles the label allows, so the student
                # learns which party / which precedent treatment, not whether
                # it is an argument at all.
                t = t.masked_fill(~al[s], -1e4)
            kdp[b, i] = t
            kdm[b, i] = True
    bt = dict(idx=idx, keep=keep, tsent=tsent, valid=valid, gidx=gidx, dlen=dlen, dom=dom,
              allowed=allowed, w=w, kdp=kdp, kdm=kdm, slot=torch.tensor(slot))
    bt = {k: v.to(device, non_blocking=True) for k, v in bt.items()}
    bt["n_sent"] = len(info)
    bt["info"] = info
    return bt


@torch.no_grad()
def predict_docs(model, ids, toks, docs, a, device):
    """{doc: emissions (n, NL) float32 CPU}, windows at 50% overlap averaged."""
    model.eval()
    jobs = [(d, st, en) for d in ids for st, en in doc_windows(len(docs[d]["text"]), a.window,
                                                                max(1, a.window // 2))]
    acc = {d: torch.zeros(len(docs[d]["text"]), NL) for d in ids}
    cnt = {d: torch.zeros(len(docs[d]["text"]), 1) for d in ids}
    bs = max(1, a.batch * 2)
    for k in range(0, len(jobs), bs):
        part = jobs[k:k + bs]
        bt = collate(part, toks, docs, a, device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            em = model(bt).float().cpu()
        for b, (d, st, en) in enumerate(part):
            acc[d][st:en] += em[b, :en - st]
            cnt[d][st:en] += 1
    model.train()
    return {d: acc[d] / cnt[d] for d in ids}


def crf_params(model):
    if model.crf is None:
        return None
    c = model.crf
    return dict(trans=c.trans.detach().float().cpu(), start=c.start.detach().float().cpu(),
                end=c.end.detach().float().cpu())


def decode(em, crfp):
    out = {}
    for d, e in em.items():
        if crfp is not None:
            out[d] = [LABELS[i] for i in viterbi(e, crfp["trans"], crfp["start"], crfp["end"])]
        else:
            out[d] = [LABELS[i] for i in e.argmax(-1).tolist()]
    return out


def gold_of(docs, ids):
    return {d: docs[d]["labels"] for d in ids}


# ------------------------------------------------------------------- train --
def build_model(a, device):
    trunk, kind, cfg = load_trunk(a.model, a.ckpt, a.keep_every, a.trunk_path)
    model = RRModel(trunk, kind, cfg, a)
    n_train, ctrl, info = apply_method(model, a)
    model = model.to(device)          # AFTER wrapping: adapters are new modules
    from phase3_peft import NF4Linear
    n_trunk = sum(p.numel() for p in model.trunk.parameters())
    n_trunk += sum(m.n for m in model.trunk.modules() if isinstance(m, NF4Linear))
    return model, ctrl, info, n_train, n_trunk


def class_weights(docs, ids, a):
    c = torch.zeros(NL)
    for d in ids:
        for y in docs[d]["labels"]:
            c[LABELS.index(y)] += 1
    w = (c.sum() / c.clamp(min=1)).pow(a.weight_pow)
    w = (w / w[c > 0].mean()).clamp(max=a.weight_cap)
    w[c == 0] = w.max()
    return w


def run_fold(fold, train_ids, eval_ids, test_ids, docs, toks, a, device, kd=None,
             extra_ids=(), unl_ids=()):
    torch.manual_seed(a.seed * 100 + fold)
    rng = random.Random(a.seed * 100 + fold)
    model, ctrl, info, n_train, n_trunk = build_model(a, device)
    if fold == 0 or a.verbose:
        log(f"  {a.method}: {n_train/1e6:.3f}M trainable | trunk {n_trunk/1e6:.1f}M "
            f"({len(blocks_of(model.trunk))} layers) | {info}")
    wlab = class_weights(docs, train_ids, a)
    pool = list(train_ids) + list(extra_ids) + list(unl_ids)

    def epoch_windows(ids):
        wins = []
        for d in ids:
            n = len(docs[d]["text"])
            k = max(1, round(a.cov * docs[d].get("cov", 1.0) * n / a.window))
            for _ in range(k):
                st = rng.randint(0, max(0, n - a.window))
                wins.append((d, st, min(n, st + a.window)))
        rng.shuffle(wins)
        return wins

    tid = trunk_param_ids(model)
    g1 = [p for p in model.parameters() if p.requires_grad and id(p) in tid]
    g2 = [p for p in model.parameters() if p.requires_grad and id(p) not in tid]
    groups = [dict(params=g2, lr=a.head_lr, base=a.head_lr)]
    if g1:
        groups.append(dict(params=g1, lr=a.lr, base=a.lr))
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.98), weight_decay=a.wd)
    phases = []
    if a.pre_epochs and extra_ids:
        phases.append(("pre", list(extra_ids), a.pre_epochs))
    phases.append(("main", pool, a.epochs))
    spe = {name: math.ceil(len(epoch_windows(ids)) / a.batch) for name, ids, _ in phases}
    total = sum(spe[n] * e for n, _, e in phases)
    if ctrl is not None:
        ctrl.total = total
        ctrl.t_init, ctrl.t_final = int(0.1 * total), int(0.3 * total)
    warm = max(1, int(0.06 * total))
    params = [p for g in groups for p in g["params"]]
    step, best, best_ep, bad = 0, -1.0, 0, 0
    best_state = best_oof = best_test = best_crf = None
    tput = None
    trace = []
    t0 = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for name, ids, n_ep in phases:
        for ep in range(n_ep):
            model.train()
            tl, tn = 0.0, 0
            wins = epoch_windows(ids)
            for k in range(0, len(wins), a.batch):
                lr_f = (step / warm if step < warm else
                        0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm))))
                for g in opt.param_groups:
                    g["lr"] = g["base"] * lr_f
                bt = collate(wins[k:k + a.batch], toks, docs, a, device, wlab, kd)
                with torch.autocast(device.type, dtype=torch.bfloat16,
                                    enabled=device.type == "cuda"):
                    em = model(bt)
                em = em.float()
                valid, allowed, w = bt["valid"], bt["allowed"], bt["w"]
                logp = F.log_softmax(em, -1)
                nll = -torch.logsumexp(logp.masked_fill(~allowed, -1e4), -1)
                lab = (w > 0) & valid
                ce = (nll * w)[lab].sum() / w[lab].sum().clamp(min=1e-6)
                loss = ce
                if model.crf is not None:
                    al = allowed & valid[..., None] | ~valid[..., None]
                    crf = model.crf.nll(em, al, valid).sum() / valid.sum()
                    loss = crf + a.crf_ce * ce
                if model.shift is not None and model.last_shift is not None:
                    single = allowed.sum(-1) == 1
                    y = allowed.float().argmax(-1)
                    m = valid & single & (w > 0)
                    m = m[:, 1:] & m[:, :-1]
                    if m.any():
                        tgt = (y[:, 1:] != y[:, :-1]).float()
                        sl = model.last_shift[:, 1:].float()
                        # changes are ~5x rarer than continuations: weight them up
                        pw = torch.tensor(4.0, device=sl.device)
                        loss = loss + a.shift_aux * F.binary_cross_entropy_with_logits(
                            sl[m], tgt[m], pos_weight=pw)
                if bt["kdm"].any() and a.kd_alpha > 0:
                    T = a.kd_temp
                    ps = F.log_softmax(em / T, -1)
                    pt = F.softmax(bt["kdp"].float() / T, -1)
                    kl = (pt * (pt.clamp(min=1e-9).log() - ps)).sum(-1)
                    m = bt["kdm"] & valid
                    kdl = kl[m].mean() * T * T
                    loss = (1 - a.kd_alpha) * loss + a.kd_alpha * kdl
                if ctrl is not None:
                    loss = loss + ctrl.orth_loss()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                if ctrl is not None:
                    ctrl.step(step)
                opt.zero_grad(set_to_none=True)
                step += 1
                tl += float(loss)
                tn += 1
            if name == "pre":
                log(f"    [fold {fold}] pre-epoch {ep+1}/{n_ep} loss {tl/max(1,tn):.4f} "
                    f"({time.time()-t0:.0f}s)")
                continue
            em = predict_docs(model, eval_ids, toks, docs, a, device)
            crfp = crf_params(model)
            res, _, _ = score_docs_(decode(em, crfp), gold_of(docs, eval_ids))
            f1 = res["macro_f1"]
            trace.append(dict(epoch=ep + 1, loss=tl / max(1, tn), f1=f1, acc=res["accuracy"],
                              t=time.time() - t0))
            extra = f" ranks {sum(ctrl.ranks())}" if ctrl is not None else ""
            if device.type == "cuda":
                extra += f" mem {torch.cuda.max_memory_allocated() / 2**30:.1f}GB"
            log(f"    [fold {fold}] epoch {ep+1:2d}/{n_ep} loss {tl/max(1,tn):.4f} "
                f"macro-F1 {f1:.4f} acc {res['accuracy']:.4f}{extra} ({time.time()-t0:.0f}s)")
            # Opened and closed every epoch so Drive syncs it: readable live
            # from another runtime or a phone, unlike the long-open log.
            with open(f"{P3}/logs/live_{a.tag}.txt", "a", encoding="utf-8") as lf:
                lf.write(f"{time.strftime('%H:%M')} fold {fold} ep {ep+1} F1 {f1:.4f}\n")
            if f1 > best:
                best, best_ep, bad = f1, ep + 1, 0
                best_oof, best_crf = em, crfp
                keys = trainable_keys(model)
                best_state = {k: (v.detach().to("cpu", torch.bfloat16, copy=True)
                                  if v.is_floating_point() else v.detach().cpu().clone())
                              for k, v in model.state_dict().items() if k in keys}
                if test_ids:
                    t1 = time.time()
                    best_test = predict_docs(model, test_ids, toks, docs, a, device)
                    n_s = sum(len(docs[d]["text"]) for d in test_ids)
                    tput = n_s / max(1e-6, time.time() - t1)
            else:
                bad += 1
                if bad >= a.patience:
                    log(f"    [fold {fold}] early stop, best epoch {best_ep}")
                    break
    peak = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
    out = dict(fold=fold, best_epoch=best_ep, best_f1=best, trace=trace,
               oof={d: v.half() for d, v in best_oof.items()},
               test={d: v.half() for d, v in (best_test or {}).items()},
               crf=best_crf, n_trainable=n_train, n_trunk=n_trunk, info=info,
               peak_gb=peak, test_sent_per_s=tput,
               ranks=ctrl.ranks() if ctrl is not None else None,
               train_s=time.time() - t0)
    return out, best_state


def trainable_keys(model):
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    names |= {n for n, _ in model.named_buffers() if n.endswith(".keep")}
    return names


def score_docs_(pred, gold):
    yt, yp = [], []
    for d in gold:
        yt += gold[d]
        yp += pred[d]
    return prf(yt, yp, LABELS), yt, yp


# --------------------------------------------------------------- external ---
def load_jsonl_docs(path, split, domain_default="EXT", weight=1.0):
    """External/unlabelled docs: {"id","domain"?,"text":[...],"allowed":[[label idx]]|"labels"}."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            did = f"{split}:{r['id']}"
            n = len(r["text"])
            al = None
            if "allowed" in r:
                al = torch.zeros(n, NL, dtype=torch.bool)
                for i, s in enumerate(r["allowed"]):
                    if s is None or len(s) == 0:
                        al[i] = True
                    else:
                        al[i, s] = True
            elif r.get("labels"):
                al = torch.zeros(n, NL, dtype=torch.bool)
                for i, y in enumerate(r["labels"]):
                    if y is None:
                        al[i] = True
                    else:
                        al[i, LABELS.index(y)] = True
            out[did] = dict(id=did, domain=r.get("domain", domain_default), split=split,
                            text=r["text"], labels=None, allowed=al, weight=weight)
    return out


# ------------------------------------------------------ intermediate train --
def intermediate(a, docs, toks, ids, device):
    """Train the whole model on external RR data only, then save the TRUNK.

    LegalSeg-scale data (thousands of judgments) is too large to mix into every
    fold run, so it is used once, as an intermediate task: the trunk learns
    rhetorical structure from it, and the fold runs then start from this trunk
    (--trunk-path) with a fresh head, trained on gold data as usual.
    """
    assert ids, "--pretrain-out needs --extra data"
    assert a.method == "full", "intermediate training updates the whole trunk: use --method full"
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    model, _, _, n_train, _ = build_model(a, device)
    wlab = torch.ones(NL)
    tid = trunk_param_ids(model)
    groups = [dict(params=[p for p in model.parameters() if p.requires_grad and id(p) not in tid],
                   lr=a.head_lr, base=a.head_lr),
              dict(params=[p for p in model.parameters() if p.requires_grad and id(p) in tid],
                   lr=a.lr, base=a.lr)]
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.98), weight_decay=a.wd)

    def windows():
        out = []
        for d in ids:
            n = len(docs[d]["text"])
            for _ in range(max(1, round(a.cov * n / a.window))):
                st = rng.randint(0, max(0, n - a.window))
                out.append((d, st, min(n, st + a.window)))
        rng.shuffle(out)
        return out

    total = a.epochs * math.ceil(len(windows()) / a.batch)
    warm, step, t0 = max(1, int(0.05 * total)), 0, time.time()
    params = [p for g in groups for p in g["params"]]
    log(f"  intermediate: {len(ids)} docs, {a.epochs} epochs, {total} steps, {n_train/1e6:.1f}M trainable")
    for ep in range(a.epochs):
        model.train()
        tl = tn = 0
        wl = windows()
        for i in range(0, len(wl), a.batch):
            wins = wl[i:i + a.batch]
            f = step / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))
            for g in opt.param_groups:
                g["lr"] = g["base"] * f
            bt = collate(wins, toks, docs, a, device, wlab)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                em = model(bt)
            em = em.float()
            logp = F.log_softmax(em, -1)
            nll = -torch.logsumexp(logp.masked_fill(~bt["allowed"], -1e4), -1)
            lab = (bt["w"] > 0) & bt["valid"]
            if not lab.any():
                continue
            loss = nll[lab].mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            tl += float(loss)
            tn += 1
            if tn % 200 == 0:
                log(f"    intermediate epoch {ep+1} step {tn} loss {tl/tn:.4f} ({time.time()-t0:.0f}s)")
        log(f"  intermediate epoch {ep+1}/{a.epochs} loss {tl/max(1,tn):.4f} ({time.time()-t0:.0f}s)")
    src = a.trunk_path or f"{P1}/{a.model}/ckpt/{a.ckpt}"
    ck = torch.load(src, map_location="cpu", weights_only=False)
    sd = dict(ck["model"])
    sd.update({k: v.detach().cpu() for k, v in model.trunk.state_dict().items()})
    os.makedirs(os.path.dirname(a.pretrain_out) or ".", exist_ok=True)
    tmp = a.pretrain_out + ".tmp"
    torch.save(dict(cfg=ck["cfg"], model=sd, intermediate=dict(vars(a))), tmp)
    os.replace(tmp, a.pretrain_out)
    log(f"  wrote trunk -> {a.pretrain_out}")


# -------------------------------------------------------------------- main --
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Phase-1 dir: M1_SC M2_LexArcED M3_SC M3_LexArcL M4_LexArcP")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--trunk-path", default=None,
                    help="load the trunk from this checkpoint instead (e.g. phase3_dapt output)")
    ap.add_argument("--method", default="full",
                    choices=["head", "full", "norm", "lora", "rslora", "qlora", "adalora",
                             "lorma", "ia3", "prompt", "prefix"])
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32.0)
    ap.add_argument("--ada-target", type=int, default=6, help="AdaLoRA final average rank")
    ap.add_argument("--targets", default="attn", choices=["qv", "attn", "all"])
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--n-virtual", type=int, default=16, help="prompt/prefix length")
    ap.add_argument("--keep-every", type=int, default=1, help="2 = half-depth student")
    ap.add_argument("--causal", action="store_true", help="keep the decoder's causal mask")
    # sentence encoding / sequence model
    ap.add_argument("--enc", default="packed", choices=["packed", "single"])
    ap.add_argument("--chunk", type=int, default=512, help="max tokens per packed chunk")
    ap.add_argument("--sent-max", type=int, default=128)
    ap.add_argument("--window", type=int, default=64, help="sentences per training window")
    ap.add_argument("--cov", type=float, default=2.0, help="windows per epoch = cov * n/window")
    ap.add_argument("--seq", default="tf", choices=["tf", "lstm", "none"])
    ap.add_argument("--seq-dim", type=int, default=384)
    ap.add_argument("--seq-layers", type=int, default=2)
    ap.add_argument("--seq-heads", type=int, default=6)
    ap.add_argument("--band", type=int, default=-1, help="tf attends to |i-j|<=band; -1 = full")
    ap.add_argument("--crf", action="store_true")
    ap.add_argument("--crf-ce", type=float, default=1.0, help="weighted CE added to the CRF nll")
    ap.add_argument("--shift-aux", type=float, default=0.0,
                    help="weight of the label-shift auxiliary loss (0 = off)")
    ap.add_argument("--no-domain-emb", dest="domain_emb", action="store_false")
    ap.add_argument("--pos-buckets", type=int, default=20, help="0 = no position feature")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--grad-ckpt", action="store_true",
                    help="recompute trunk activations in backward (memory for compute)")
    # optimisation
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--batch", type=int, default=4, help="windows per step")
    ap.add_argument("--lr", type=float, default=None, help="trunk/adapter LR (method default)")
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--weight-pow", type=float, default=0.5)
    ap.add_argument("--weight-cap", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=1)
    # data
    ap.add_argument("--folds", default="all", help="'all' or e.g. 0,1")
    ap.add_argument("--extra", nargs="*", default=[], help="external labelled jsonl")
    ap.add_argument("--extra-weight", type=float, default=0.5)
    ap.add_argument("--extra-cov", type=float, default=0.3,
                    help="sampling rate of external docs relative to gold docs per epoch")
    ap.add_argument("--pre-epochs", type=int, default=0, help="epochs on --extra alone first")
    ap.add_argument("--kd-logits", default=None, help="teacher logits .pt {doc: (n,13)}")
    ap.add_argument("--kd-alpha", type=float, default=0.7)
    ap.add_argument("--kd-temp", type=float, default=2.0)
    ap.add_argument("--kd-test", action="store_true", help="distil on test-document text too")
    ap.add_argument("--kd-unlabelled", nargs="*", default=[], help="unlabelled jsonl to distil on")
    # output
    ap.add_argument("--pretrain-out", default=None,
                    help="intermediate training: train on --extra docs only, save the trunk "
                         "(Phase-1 checkpoint format) here, and exit; use it via --trunk-path")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--submit", action="store_true", help="write a validated zip from fold-avg test logits")
    ap.add_argument("--no-ckpt", action="store_true")
    ap.add_argument("--force", action="store_true", help="retrain folds that already have results")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    if a.lr is None:
        a.lr = dict(full=3e-5, head=0.0, norm=1e-3, lora=3e-4, rslora=2e-4, qlora=3e-4,
                    adalora=5e-4, lorma=3e-4, ia3=3e-3, prompt=3e-3, prefix=1e-3)[a.method]
    assert not (a.seq != "tf" and a.band >= 0), "--band needs --seq tf"

    global LOGF
    rd = f"{P3}/results/{a.tag}"
    os.makedirs(rd, exist_ok=True)
    os.makedirs(f"{P3}/logs", exist_ok=True)
    LOGF = open(f"{P3}/logs/{a.tag}.log", "a", encoding="utf-8")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        p = torch.cuda.get_device_properties(0)
        assert p.major * 10 + p.minor >= 80, f"{p.name}: bf16 is emulated below SM80 -- use an L4/A100"
        torch.set_float32_matmul_precision("high")
        gpu = p.name
    else:
        device, gpu = torch.device("cpu"), "cpu"
    log(f"\n===== {a.tag} | {gpu} | {a.model}/{a.ckpt} method={a.method} "
        f"enc={a.enc} seq={a.seq} band={a.band} crf={a.crf} keep_every={a.keep_every} =====")
    log("  args: " + json.dumps(vars(a), sort_keys=True))

    docs, labels = load_all()
    assert labels == LABELS, labels
    for r in docs.values():
        if r["labels"]:
            al = torch.zeros(len(r["labels"]), NL, dtype=torch.bool)
            al[torch.arange(len(r["labels"])), torch.tensor([LABELS.index(y) for y in r["labels"]])] = True
            r["allowed"] = al
    split = ensure_split(docs)
    folds = split["folds"]
    test_ids = sorted(d for d, r in docs.items() if r["split"] == "test")

    extra_ids = []
    for path in a.extra:
        if not os.path.exists(path):
            log(f"  extra: {path} MISSING -- skipped")
            continue
        ext = load_jsonl_docs(path, "ext", weight=a.extra_weight)
        for r in ext.values():
            r["cov"] = a.extra_cov
        clash = set(ext) & set(docs)
        assert not clash, clash
        docs.update(ext)
        extra_ids += sorted(ext)
        log(f"  extra: {path}: {len(ext)} docs, {sum(len(r['text']) for r in ext.values())} sentences")
    kd, unl_ids = None, []
    if a.kd_logits:
        kd = {d: v.float() for d, v in torch.load(a.kd_logits, weights_only=False).items()}
        for path in a.kd_unlabelled:
            u = load_jsonl_docs(path, "unl")
            docs.update(u)
            unl_ids += sorted(u)
        if a.kd_test:
            unl_ids += test_ids
        miss = [d for d in unl_ids if d not in kd]
        assert not miss, f"teacher logits missing for {len(miss)} docs, e.g. {miss[:3]}"
        log(f"  KD: teacher logits for {len(kd)} docs; distilling on {len(unl_ids)} unlabelled docs too")
    toks = tokenize_docs(docs, a)

    if a.pretrain_out:
        intermediate(a, docs, toks, extra_ids, device)
        return

    which = range(len(folds)) if a.folds == "all" else [int(x) for x in a.folds.split(",")]
    results = []
    for k in which:
        fp = f"{rd}/fold{k}.pt"
        if os.path.exists(fp) and not a.force:
            log(f"  fold {k}: found {fp}, skipping (use --force to retrain)")
            results.append(torch.load(fp, weights_only=False))
            continue
        ev = folds[k]
        tr = [d for j, f in enumerate(folds) if j != k for d in f]
        log(f"\n--- fold {k}: train {len(tr)} docs -> eval {len(ev)} docs ---")
        out, state = run_fold(k, tr, ev, test_ids, docs, toks, a, device, kd,
                              extra_ids, [d for d in unl_ids if d not in ev])
        torch.save(out, fp)
        if not a.no_ckpt and state is not None:
            os.makedirs(f"{P3}/ckpt/{a.tag}", exist_ok=True)
            torch.save(dict(state=state, args=vars(a)), f"{P3}/ckpt/{a.tag}/fold{k}.pt")
        results.append(out)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ---- pooled out-of-fold score: the CV number
    oof = {d: v.float() for r in results for d, v in r["oof"].items()}
    pred = {}
    for r in results:
        pred.update(decode({d: v.float() for d, v in r["oof"].items()}, r["crf"]))
    gold = gold_of(docs, sorted(oof))
    res, yt, yp = score_docs_({d: pred[d] for d in gold}, gold)
    log(f"\n===== {a.tag}: pooled OOF over {len(gold)} docs ({len(results)} folds) =====")
    report(res, LABELS, "OOF", yt, yp, log_fn=log)
    per_dom = {}
    for dom in DOMAINS:
        ids = [d for d in gold if docs[d]["domain"] == dom]
        r2, _, _ = score_docs_({d: pred[d] for d in ids}, {d: gold[d] for d in ids})
        per_dom[dom] = r2["macro_f1"]
    fold_f1 = [r["best_f1"] for r in results]
    log(f"  by domain: CL {per_dom['CL']:.4f}  IT {per_dom['IT']:.4f} | per fold "
        + " ".join(f"{x:.4f}" for x in fold_f1)
        + f" | best epochs {[r['best_epoch'] for r in results]}")

    meta = dict(tag=a.tag, args=vars(a), cv_macro_f1=res["macro_f1"], cv_macro_p=res["macro_p"],
                cv_macro_r=res["macro_r"], cv_by_domain=per_dom, fold_f1=fold_f1,
                best_epochs=[r["best_epoch"] for r in results],
                per_label={l: res["per"][l][2] for l in LABELS},
                n_trainable=results[0]["n_trainable"], n_trunk=results[0]["n_trunk"],
                info=results[0]["info"], peak_gb=max(r["peak_gb"] for r in results),
                test_sent_per_s=results[0]["test_sent_per_s"],
                train_s=sum(r["train_s"] for r in results),
                ranks=results[0].get("ranks"))
    json.dump(meta, open(f"{rd}/meta.json", "w"), indent=1)
    log(f"  params: {meta['n_trainable']/1e6:.3f}M trainable / trunk {meta['n_trunk']/1e6:.1f}M | "
        f"peak {meta['peak_gb']:.1f} GB | {meta['test_sent_per_s'] or 0:.0f} sent/s inference | "
        f"train {meta['train_s']/60:.1f} min")
    log(f"  wrote {rd}/meta.json")

    if a.submit:
        if len(results) < len(folds):
            log("  NOT submitting: only some folds ran")
            return
        avg = {d: torch.stack([F.log_softmax(r["test"][d].float(), -1) for r in results]).mean(0)
               for d in test_ids}
        crfp = None
        if results[0]["crf"] is not None:
            crfp = {k: torch.stack([r["crf"][k] for r in results]).mean(0) for k in ("trans", "start", "end")}
            avg = {d: torch.stack([r["test"][d].float() for r in results]).mean(0) for d in test_ids}
        tpred = decode(avg, crfp)
        test_docs = {d: docs[d] for d in test_ids}
        write_submission(tpred, test_docs, LABELS, a.tag, cv=res["macro_f1"],
                         note=f"{a.model} {a.method} 5-fold avg")
    try:
        from phase3_ens import write_status
        write_status()
    except Exception as e:
        log(f"  (status not updated: {e})")


if __name__ == "__main__":
    main()
