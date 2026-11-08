#!/usr/bin/env python3
"""CS-781 Phase 5 -- multi-label statute identification on a Phase-1 trunk.

    python phase5_lsi.py --model M4_LexArcP  --method lorma   --tag m4_lorma
    python phase5_lsi.py --model M2_LexArcED --method partial --train-top 5 --tag m2_partial
    python phase5_lsi.py --model M1_SC       --method adapter --tag m1_adapter
    python phase5_lsi.py --model M4_LexArcP  --method full --keep-every 2 \
        --kd-teacher $P5/teacher.npz --kd-docs dev,test --tag kd_m4half

THE MODEL

    document tokens (phase5_tok.py cache, head+tail cut at 8192)
      -> consecutive chunks of --chunk tokens (default 512)
      -> Phase-1 trunk over each chunk, bidirectional attention     (C, T, d)
      -> doc-level head over ALL token states of the document's chunks:
           laat  label-wise attention (Vu et al. 2020, LAAT): each statute has
                 its own query and reads the tokens that matter for it, wherever
                 they sit in the document                   (default)
           attn  one attention pool -> linear
           mean  mean pool -> linear
      -> one logit per statute; BCE with capped positive weights, or ASL.

Training sees --train-chunks chunks per document (a random sorted subset, the
first chunk always kept with --keep-first); inference sees up to --infer-chunks.
Documents are drawn with probability ~ (rarest label's train count)^-sample_pow,
so rare statutes are revisited within the one or two epochs we can afford.

COMPUTE. Training stops at --budget-min minutes: after --probe steps the measured
step time sets the total step count and the LR schedule is laid over that, so a
run never overruns and always ends at the bottom of its schedule. Tokens/s is
logged. A resume file is written every --save-every-min minutes; rerunning the
same command after a disconnect continues from it.

OUTPUT, results/<tag>/: dev_logits.npy, test_logits.npy (fp16, file order),
dev_ids.json, test_ids.json, meta.json (dev scores, timings, params), trace.json.
Every threshold and ensemble decision is made later on CPU (phase5_ens.py).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from phase5_data import (P1, P5, load, label_space, multi_hot, score_on, decode,  # noqa: E402
                         tune_global, threshold_report, log as _log)
from phase5_peft import apply_method, blocks_of, trunk_param_ids, PrefixEncoder, NF4Linear  # noqa: E402
from phase5_tok import tok_path  # noqa: E402

sys.path.insert(0, os.environ.get("CS781_LIB", f"{P1}/shared/lib"))
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
    del ck
    return t, kind, cfg


# ------------------------------------------------------------------- model --
class LSIModel(nn.Module):
    def __init__(self, trunk, kind, cfg, a, n_labels):
        super().__init__()
        self.trunk, self.kind, self.a = trunk, kind, a
        d = cfg.d_model
        self.d, self.NL = d, n_labels
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
        self.drop = nn.Dropout(a.dropout)
        h = a.att_dim
        if a.head == "laat":
            self.W = nn.Linear(d, h)
            self.U = nn.Linear(h, n_labels, bias=False)        # one query per statute
            self.out_w = nn.Parameter(torch.empty(n_labels, d))
            nn.init.normal_(self.out_w, std=d ** -0.5)
        elif a.head == "attn":
            self.W = nn.Linear(d, h)
            self.u = nn.Linear(h, 1, bias=False)
            self.out = nn.Linear(d, n_labels)
        else:
            self.out = nn.Linear(d, n_labels)
        self.out_b = nn.Parameter(torch.zeros(n_labels))

    def init_bias(self, prior):
        """Start each statute's logit at its log-odds prior: the head does not
        spend its first hundreds of steps learning that most statutes are absent."""
        p = torch.as_tensor(prior, dtype=torch.float32).clamp(1e-5, 1 - 1e-5)
        with torch.no_grad():
            self.out_b.copy_(torch.log(p / (1 - p)))

    # ---- token level (as Phase 3)
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
        p0 = self.R
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

    # ---- document level
    def forward(self, bt):
        """bt: idx/keep (C, T) chunks; slot (C,) = doc*K + k; B docs, K chunk slots."""
        H = self.token_hidden(bt["idx"], bt["keep"])            # (C, T, d)
        B, K = bt["B"], bt["K"]
        C, T, d = H.shape
        Hd = H.new_zeros(B * K, T, d)
        Hd[bt["slot"]] = H
        Md = torch.zeros(B * K, T, dtype=torch.bool, device=H.device)
        Md[bt["slot"]] = bt["keep"]
        Hd = Hd.view(B, K * T, d)
        Md = Md.view(B, K * T)
        Hd = self.drop(Hd)
        if self.a.head == "laat":
            z = torch.tanh(self.W(Hd))                          # (B, KT, h)
            s = self.U(z).float()                               # (B, KT, NL)
            s = s.masked_fill(~Md[..., None], -1e4)
            att = torch.softmax(s, 1).to(Hd.dtype)
            V = torch.einsum("btl,btd->bld", att, Hd)            # (B, NL, d)
            return (V.float() * self.out_w.float()).sum(-1) + self.out_b
        if self.a.head == "attn":
            s = self.u(torch.tanh(self.W(Hd))).squeeze(-1).float().masked_fill(~Md, -1e4)
            v = torch.einsum("bt,btd->bd", torch.softmax(s, 1).to(Hd.dtype), Hd)
        else:
            m = Md.to(Hd.dtype)
            v = (Hd * m[..., None]).sum(1) / m.sum(1, keepdim=True).clamp(min=1)
        return self.out(v).float() + self.out_b


# -------------------------------------------------------------------- data --
class Tokens:
    def __init__(self, split):
        z = np.load(tok_path(split), allow_pickle=False)
        self.flat, self.off = z["flat"], z["off"]
        self.ids = [str(x) for x in z["ids"]]
        self.row = {d: i for i, d in enumerate(self.ids)}

    def get(self, i):
        return self.flat[self.off[i]:self.off[i + 1]]


def chunks_of(toks, L, overlap=0):
    n = len(toks)
    if n <= L:
        return [toks]
    step = L - overlap
    out = [toks[s:s + L] for s in range(0, max(1, n - overlap), step)]
    return [c for c in out if len(c)]


def collate(items, a, K, device):
    """items: list of (token array, label vec or None, teacher vec or None); K chunks max per doc."""
    chunks, slot = [], []
    for b, it in enumerate(items):
        sel = it[3]
        for k, c in enumerate(sel[:K]):
            chunks.append(c)
            slot.append(b * K + k)
    T = max(len(c) for c in chunks)
    C = len(chunks)
    idx = torch.zeros(C, T, dtype=torch.long)
    keep = torch.zeros(C, T, dtype=torch.bool)
    for j, c in enumerate(chunks):
        idx[j, :len(c)] = torch.from_numpy(c.astype(np.int64))
        keep[j, :len(c)] = True
    bt = dict(idx=idx.to(device, non_blocking=True), keep=keep.to(device, non_blocking=True),
              slot=torch.tensor(slot, device=device), B=len(items), K=K,
              n_tok=int(keep.sum()))
    ys = [it[1] for it in items]
    if any(y is not None for y in ys):
        n = next(y for y in ys if y is not None).shape[0]
        Y = torch.zeros(len(items), n)
        ym = torch.zeros(len(items), dtype=torch.bool)
        for i, y in enumerate(ys):
            if y is not None:
                Y[i] = torch.from_numpy(y.astype(np.float32))
                ym[i] = True
        bt["y"], bt["ym"] = Y.to(device), ym.to(device)
        LM = torch.ones(len(items), n)
        DW = torch.ones(len(items))
        for i, it in enumerate(items):
            if len(it) > 4 and it[4] is not None:
                LM[i] = torch.from_numpy(it[4].astype(np.float32))
            if len(it) > 5:
                DW[i] = float(it[5])
        bt["lmask"], bt["dw"] = LM.to(device), DW.to(device)
    ts = [it[2] for it in items]
    if any(t is not None for t in ts):
        n = next(t for t in ts if t is not None).shape[0]
        Tt = torch.zeros(len(items), n)
        tm = torch.zeros(len(items), dtype=torch.bool)
        for i, t in enumerate(ts):
            if t is not None:
                Tt[i] = torch.from_numpy(t.astype(np.float32))
                tm[i] = True
        bt["t"], bt["tm"] = Tt.to(device), tm.to(device)
    return bt


def pick_chunks(ch, K, rng, keep_first):
    if len(ch) <= K:
        return ch
    if keep_first:
        rest = sorted(rng.sample(range(1, len(ch)), K - 1))
        return [ch[0]] + [ch[i] for i in rest]
    return [ch[i] for i in sorted(rng.sample(range(len(ch)), K))]


@torch.no_grad()
def predict(model, tok, rows, a, device, K=None):
    """logits (len(rows), NL) float32, documents batched by chunk count."""
    model.eval()
    K = K or a.infer_chunks
    L = a.chunk
    nch = {r: min(K, len(chunks_of(tok.get(r), L, a.overlap))) for r in rows}
    order = sorted(rows, key=nch.get)
    out = {}
    i = 0
    while i < len(order):
        # ascending chunk counts: grow the batch while (docs x largest count) fits the budget
        j = i + 1
        while j < len(order) and (j - i + 1) * nch[order[j]] * L <= a.infer_tokens:
            j += 1
        part = order[i:j]
        items = []
        for r in part:
            ch = chunks_of(tok.get(r), L, a.overlap)
            if len(ch) > K:   # evenly spaced, first and last kept
                pos = np.unique(np.linspace(0, len(ch) - 1, K).round().astype(int))
                ch = [ch[p] for p in pos]
            items.append((None, None, None, ch))
        bt = collate(items, a, max(len(x[3]) for x in items), device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            lg = model(bt).float().cpu().numpy()
        for r, l in zip(part, lg):
            out[r] = l
        i += len(part)
    model.train()
    return np.stack([out[r] for r in rows])


# ------------------------------------------------------------------- train --
def build_model(a, n_labels, device):
    trunk, kind, cfg = load_trunk(a.model, a.ckpt, a.keep_every, a.trunk_path)
    model = LSIModel(trunk, kind, cfg, a, n_labels)
    n_train, ctrl, info = apply_method(model, a)
    model = model.to(device)
    n_trunk = sum(p.numel() for p in model.trunk.parameters())
    n_trunk += sum(m.n for m in model.trunk.modules() if isinstance(m, NF4Linear))
    return model, ctrl, info, n_train, n_trunk


def param_groups(model, a):
    """Head at --head-lr; trunk/adapters at --lr, with layer-wise decay for full/partial."""
    tid = trunk_param_ids(model)
    head = [p for p in model.parameters() if p.requires_grad and id(p) not in tid]
    groups = [dict(params=head, lr=a.head_lr, base=a.head_lr)]
    if a.method in ("full", "partial") and a.llrd < 1.0:
        blocks = blocks_of(model.trunk)
        L = len(blocks)
        seen = set()
        for i, b in enumerate(blocks):
            ps = [p for p in b.parameters() if p.requires_grad]
            seen |= {id(p) for p in ps}
            if ps:
                lr = a.lr * a.llrd ** (L - 1 - i)
                groups.append(dict(params=ps, lr=lr, base=lr))
        rest = [p for p in model.parameters() if p.requires_grad and id(p) in tid and id(p) not in seen]
        if rest:   # embeddings sit below block 0, final norm above the top block
            emb = [p for n, p in model.trunk.named_parameters() if p.requires_grad and n.startswith("emb")]
            top = [p for p in rest if all(p is not e for e in emb)]
            if emb:
                lr = a.lr * a.llrd ** L
                groups.append(dict(params=emb, lr=lr, base=lr))
            if top:
                groups.append(dict(params=top, lr=a.lr, base=a.lr))
    else:
        tr = [p for p in model.parameters() if p.requires_grad and id(p) in tid]
        if tr:
            groups.append(dict(params=tr, lr=a.lr, base=a.lr))
    return groups


def asl_loss(logits, y, gn=4.0, gp=0.0, clip=0.05):
    """Asymmetric loss (Ridnik et al. 2021): focus off easy negatives, shift negative probs."""
    p = torch.sigmoid(logits)
    pn = (p - clip).clamp(min=0) if clip else p
    lp = y * (1 - p).pow(gp) * torch.log(p.clamp(min=1e-8))
    ln = (1 - y) * pn.pow(gn) * torch.log((1 - pn).clamp(min=1e-8))
    return -(lp + ln)


def trainable_state(model):
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    names |= {n for n, _ in model.named_buffers() if n.endswith(".keep")}
    return {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items() if k in names}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="M1_SC M2_LexArcED M3_LexArcL M4_LexArcP")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--trunk-path", default=None)
    ap.add_argument("--method", default="full",
                    choices=["head", "full", "partial", "norm", "lora", "rslora", "qlora", "adalora",
                             "lorma", "ia3", "adapter", "prompt", "prefix"])
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32.0)
    ap.add_argument("--ada-target", type=int, default=6)
    ap.add_argument("--targets", default="attn", choices=["qv", "attn", "all"])
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--bottleneck", type=int, default=64, help="adapter width")
    ap.add_argument("--train-top", type=int, default=6, help="partial: blocks trained from the top")
    ap.add_argument("--llrd", type=float, default=0.85, help="layer-wise LR decay (full/partial)")
    ap.add_argument("--n-virtual", type=int, default=32, help="prompt/prefix length")
    ap.add_argument("--keep-every", type=int, default=1, help="2 = half-depth student")
    ap.add_argument("--causal", action="store_true")
    # document representation
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=0)
    ap.add_argument("--train-chunks", type=int, default=2)
    ap.add_argument("--infer-chunks", type=int, default=8)
    ap.add_argument("--keep-first", action="store_true")
    ap.add_argument("--head", default="laat", choices=["laat", "attn", "mean"])
    ap.add_argument("--att-dim", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--grad-ckpt", action="store_true")
    # loss
    ap.add_argument("--loss", default="bce", choices=["bce", "asl"])
    ap.add_argument("--pos-pow", type=float, default=0.5, help="BCE pos_weight = (neg/pos)^pow")
    ap.add_argument("--pos-cap", type=float, default=30.0)
    ap.add_argument("--sample-pow", type=float, default=0.3,
                    help="doc sampling weight = (train count of its rarest statute)^-pow; 0 = uniform")
    # optimisation
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--budget-min", type=float, default=40.0, help="training wall-clock cap")
    ap.add_argument("--probe", type=int, default=30, help="steps timed before fixing the schedule")
    ap.add_argument("--batch", type=int, default=8, help="documents per step")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--eval-every-min", type=float, default=10.0)
    ap.add_argument("--eval-docs", type=int, default=1500, help="dev docs for in-run monitoring")
    ap.add_argument("--infer-tokens", type=int, default=32768, help="tokens per inference batch")
    # distillation
    ap.add_argument("--kd-teacher", default=None, help=".npz with ids + logits (teacher, any docs)")
    ap.add_argument("--kd-alpha", type=float, default=0.7, help="weight of the teacher term on labelled docs")
    ap.add_argument("--kd-temp", type=float, default=2.0)
    ap.add_argument("--kd-docs", default="", help="unlabelled splits distilled on: dev,test")
    ap.add_argument("--train-splits", default="train")
    ap.add_argument("--silver", type=int, default=0, help="add up to N silver docs (external/silver.jsonl)")
    ap.add_argument("--silver-weight", type=float, default=0.5)
    # output
    ap.add_argument("--tag", required=True)
    ap.add_argument("--no-ckpt", action="store_true")
    ap.add_argument("--save-every-min", type=float, default=15.0)
    ap.add_argument("--limit", type=int, default=0, help="debug: first N docs of each split")
    a = ap.parse_args()
    if a.lr is None:
        a.lr = dict(full=3e-5, partial=5e-5, head=0.0, norm=1e-3, lora=3e-4, rslora=2e-4, qlora=3e-4,
                    adalora=5e-4, lorma=3e-4, ia3=3e-3, adapter=1e-3, prompt=3e-3, prefix=1e-3)[a.method]

    global LOGF
    rd = f"{P5}/results/{a.tag}"
    os.makedirs(rd, exist_ok=True)
    os.makedirs(f"{P5}/logs", exist_ok=True)
    os.makedirs(f"{P5}/ckpt", exist_ok=True)
    LOGF = open(f"{P5}/logs/{a.tag}.log", "a", encoding="utf-8")
    if os.path.exists(f"{rd}/meta.json"):
        log(f"{rd}/meta.json exists -- {a.tag} is done. Delete it to rerun.")
        return
    if torch.cuda.is_available():
        device = torch.device("cuda")
        p = torch.cuda.get_device_properties(0)
        assert p.major * 10 + p.minor >= 80, f"{p.name}: bf16 is emulated below SM80 -- use an L4/A100"
        torch.set_float32_matmul_precision("high")
        gpu = p.name
    else:
        device, gpu = torch.device("cpu"), "cpu"
    log(f"\n===== {a.tag} | {gpu} | {a.model} method={a.method} head={a.head} chunk={a.chunk}"
        f"x{a.train_chunks}/{a.infer_chunks} keep_every={a.keep_every} =====")
    log("  args: " + json.dumps(vars(a), sort_keys=True))
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    np_rng = np.random.RandomState(a.seed)

    NL, cnt = label_space()
    splits = ("train", "dev", "test") + (("silver",) if a.silver else ())
    toks = {s: Tokens(s) for s in splits}
    recs = {s: load(s) for s in splits}
    for s in recs:
        assert [r["id"] for r in recs[s]] == toks[s].ids, f"{s}: token cache out of order -- rerun phase5_tok.py"
    lim = (lambda xs: xs[: a.limit]) if a.limit else (lambda xs: xs)
    Y = {s: multi_hot(recs[s], NL) for s in ("train", "dev")}

    # ---- training pool: (split, row, label vec | None, teacher vec | None)
    teach = {}
    if a.kd_teacher:
        z = np.load(a.kd_teacher, allow_pickle=False)
        for d, l in zip(z["ids"], z["logits"]):
            teach[str(d)] = l.astype(np.float32)
        log(f"  KD: teacher logits for {len(teach)} docs, alpha {a.kd_alpha}, T {a.kd_temp}")
    pool = []
    for s in a.train_splits.split(","):
        for i in lim(range(len(recs[s]))):
            pool.append((s, i, Y[s][i], teach.get(recs[s][i]["id"]), None, 1.0))
    if a.silver:
        sil = sorted(range(len(recs["silver"])), key=lambda i: min(cnt[l] for l in recs["silver"][i]["labels"]))
        sil = lim(sil[: a.silver])
        for i in sil:
            r = recs["silver"][i]
            y = np.zeros(NL, dtype=np.uint8)
            y[r["labels"]] = 1
            lm = np.ones(NL, dtype=np.uint8)
            lm[r["mask"]] = 0
            pool.append(("silver", i, y, teach.get(r["id"]), lm, a.silver_weight))
        log(f"  silver: +{len(sil)} docs at weight {a.silver_weight}")
    for s in [x for x in a.kd_docs.split(",") if x]:
        miss = 0
        for i in lim(range(len(recs[s]))):
            t = teach.get(recs[s][i]["id"])
            if t is None:
                miss += 1
                continue
            pool.append((s, i, None, t, None, 1.0))
        log(f"  KD: +{len(recs[s]) - miss} unlabelled {s} docs (teacher only); {miss} without teacher")
    rare = np.array([cnt[np.nonzero(y)[0]].min() if y is not None and y.any() else
                     (cnt[np.argmax(t)] if t is not None else cnt.max()) for _, _, y, t, _, _ in pool], dtype=np.float64)
    w = np.maximum(rare, 1) ** (-a.sample_pow)
    w = w / w.sum()
    n_epoch = len(pool)

    # ---- model
    model, ctrl, info, n_train, n_trunk = build_model(a, NL, device)
    model.init_bias((cnt + 1) / (len(recs["train"]) + 2))
    log(f"  {a.method}: {n_train/1e6:.3f}M trainable | trunk {n_trunk/1e6:.1f}M "
        f"({len(blocks_of(model.trunk))} layers) | {info}")
    groups = param_groups(model, a)
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.98), weight_decay=a.wd)
    params = [p for g in groups for p in g["params"]]
    pos = cnt.astype(np.float64)
    pw = torch.tensor(np.clip(((len(recs["train"]) - pos) / np.maximum(pos, 1)) ** a.pos_pow, 1, a.pos_cap),
                      dtype=torch.float32, device=device)

    planned = max(1, int(a.epochs * n_epoch / a.batch))
    total = planned
    warm = max(1, int(a.warmup * total))
    dev_rows = list(range(len(recs["dev"])))
    mon = sorted(np_rng.choice(len(dev_rows), min(a.eval_docs, len(dev_rows)), replace=False).tolist())
    mon = lim(mon)
    step, t_train, n_tok_seen = 0, 0.0, 0
    trace = []
    rp = f"{P5}/ckpt/{a.tag}_resume.pt"
    if os.path.exists(rp):
        st = torch.load(rp, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"], strict=False)
        opt.load_state_dict(st["opt"])
        step, t_train, n_tok_seen, total, trace = st["step"], st["t_train"], st["n_tok"], st["total"], st["trace"]
        warm = max(1, int(a.warmup * total))
        rng.setstate(st["rng"])
        np_rng.set_state(st["np_rng"])
        log(f"  RESUMED at step {step}/{total} ({t_train/60:.1f} min trained)")
    if ctrl is not None:
        ctrl.total = total
        ctrl.t_init, ctrl.t_final = int(0.1 * total), int(0.3 * total)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    last_eval = last_save = time.time()
    t_seg = time.time()
    model.train()
    while step < total:
        draw = np_rng.choice(len(pool), a.batch, replace=True, p=w)
        items = []
        for j in draw:
            s, i, y, t, lm, dw = pool[j]
            ch = pick_chunks(chunks_of(toks[s].get(i), a.chunk, a.overlap), a.train_chunks, rng, a.keep_first)
            items.append((None, y, t, ch, lm, dw))
        bt = collate(items, a, a.train_chunks, device)
        lr_f = step / warm if step < warm else 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))
        for g in opt.param_groups:
            g["lr"] = g["base"] * lr_f
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(bt)
        logits = logits.float()
        # per-document loss: gold only -> g; teacher only -> k; both -> (1-alpha) g + alpha k
        nb = logits.size(0)
        ym = bt.get("ym", torch.zeros(nb, dtype=torch.bool, device=logits.device))
        tm = bt.get("tm", torch.zeros(nb, dtype=torch.bool, device=logits.device))
        per = logits.new_zeros(nb)
        if ym.any():
            if a.loss == "asl":
                g = asl_loss(logits, bt["y"])
            else:
                g = F.binary_cross_entropy_with_logits(logits, bt["y"], pos_weight=pw, reduction="none")
            g = (g * bt["lmask"]).sum(1) * bt["dw"]
            per = per + torch.where(ym, g * torch.where(tm, 1 - a.kd_alpha, 1.0), 0.0)
        if tm.any():
            T = a.kd_temp
            k = F.binary_cross_entropy_with_logits(logits / T, torch.sigmoid(bt["t"] / T),
                                                   reduction="none").sum(1) * T * T
            per = per + torch.where(tm, k * torch.where(ym, a.kd_alpha, 1.0), 0.0)
        loss = per.mean()
        if ctrl is not None:
            loss = loss + ctrl.orth_loss()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if ctrl is not None:
            ctrl.step(step)
        opt.zero_grad(set_to_none=True)
        step += 1
        n_tok_seen += bt["n_tok"]
        if step == a.probe and device.type == "cuda":
            torch.cuda.synchronize()
        now = time.time()
        if step == a.probe:
            sec = (now - t_seg) / a.probe
            fit = int((a.budget_min * 60 - t_train - (now - t_seg)) / sec) + step
            total = max(step + 1, min(planned, fit))
            warm = max(1, int(a.warmup * total))
            if ctrl is not None:
                ctrl.total = total
                ctrl.t_init, ctrl.t_final = int(0.1 * total), int(0.3 * total)
            log(f"  probe: {sec:.3f} s/step, {n_tok_seen / (now - t_seg):.0f} tok/s -> {total} steps "
                f"({total * a.batch / n_epoch:.2f} epochs of {n_epoch} docs; planned {planned})")
        if step % 100 == 0 or step == total:
            el = t_train + now - t_seg
            mem = f" mem {torch.cuda.max_memory_allocated() / 2**30:.1f}GB" if device.type == "cuda" else ""
            log(f"    step {step}/{total} loss {float(loss):.4f} lr_f {lr_f:.3f} "
                f"{n_tok_seen / max(el, 1e-6):.0f} tok/s{mem} ({el/60:.1f} min)")
        if now - last_eval > a.eval_every_min * 60 and step < total:
            S = predict(model, toks["dev"], mon, a, device, K=min(a.infer_chunks, 4))
            tg, m = tune_global(S, Y["dev"][mon])
            trace.append(dict(step=step, min=(t_train + now - t_seg) / 60, mon_macro=m))
            log(f"    [monitor] step {step}: dev-subset macro-F1 {m:.4f} (global thr {tg:.2f})")
            with open(f"{P5}/logs/live_{a.tag}.txt", "a") as lf:
                lf.write(f"{time.strftime('%H:%M')} step {step}/{total} mon-F1 {m:.4f}\n")
            last_eval = time.time()
        if now - last_save > a.save_every_min * 60 and step < total:
            t_now = t_train + time.time() - t_seg
            st = dict(model=trainable_state(model), opt=opt.state_dict(), step=step, t_train=t_now,
                      n_tok=n_tok_seen, total=total, trace=trace, rng=rng.getstate(), np_rng=np_rng.get_state())
            torch.save(st, rp + ".tmp")
            os.replace(rp + ".tmp", rp)
            last_save = time.time()
    t_train += time.time() - t_seg
    peak = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
    log(f"  trained {step} steps in {t_train/60:.1f} min, {n_tok_seen/max(t_train,1e-6):.0f} tok/s, peak {peak:.1f} GB")

    # ---- full inference
    t0 = time.time()
    dv_rows = lim(dev_rows)
    te_rows = lim(list(range(len(recs["test"]))))
    Sdv = predict(model, toks["dev"], dv_rows, a, device)
    Ste = predict(model, toks["test"], te_rows, a, device)
    t_inf = time.time() - t0
    inf_tok = sum(min(len(toks[s].get(i)), a.chunk * a.infer_chunks) for s, rows in (("dev", dv_rows), ("test", te_rows)) for i in rows)
    log(f"  inference: {len(dv_rows)} dev + {len(te_rows)} test docs in {t_inf/60:.1f} min ({inf_tok/max(t_inf,1e-6):.0f} tok/s)")
    Ydv = Y["dev"][dv_rows]
    th = threshold_report(Sdv, Ydv, log_fn=log)
    dev_glob = score_on(Ydv, decode(Sdv, th["t_global"]))
    dev_best = score_on(Ydv, decode(Sdv, th["thr"]))
    log(f"  DEV global-thr {json.dumps(dev_glob)} | {th['best']} {json.dumps(dev_best)} (CV {th['cv'][th['best']]:.4f})")
    np.save(f"{rd}/dev_logits.npy", Sdv.astype(np.float16))
    np.save(f"{rd}/test_logits.npy", Ste.astype(np.float16))
    json.dump([recs["dev"][i]["id"] for i in dv_rows], open(f"{rd}/dev_ids.json", "w"))
    json.dump([recs["test"][i]["id"] for i in te_rows], open(f"{rd}/test_ids.json", "w"))
    json.dump(trace, open(f"{rd}/trace.json", "w"))
    if not a.no_ckpt:
        torch.save(dict(state=trainable_state(model), args=vars(a)), f"{P5}/ckpt/{a.tag}.pt")
    meta = dict(tag=a.tag, kind="neural", hybrid=False, args=vars(a), model=a.model, method=a.method,
                n_labels=NL, n_trainable=n_train, n_trunk=n_trunk, info=info,
                dev_global=dev_glob, dev_best=dev_best, thr_cv=th["cv"], thr_best=th["best"],
                steps=step, train_min=t_train / 60, train_tok_s=n_tok_seen / max(t_train, 1e-6),
                infer_min=t_inf / 60, infer_tok_s=inf_tok / max(t_inf, 1e-6), peak_gb=peak,
                epochs_seen=step * a.batch / n_epoch, gpu=gpu,
                ranks=ctrl.ranks() if ctrl is not None else None)
    json.dump(meta, open(f"{rd}/meta.json", "w"), indent=1)
    if os.path.exists(rp):
        os.remove(rp)
    log(f"  wrote {rd}/meta.json")


if __name__ == "__main__":
    main()
