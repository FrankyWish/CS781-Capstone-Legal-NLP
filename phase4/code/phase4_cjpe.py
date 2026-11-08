#!/usr/bin/env python3
"""CS-781 Phase 4 -- court judgment prediction on a Phase-1 trunk.

    python phase4_cjpe.py --model M4_LexArcP  --method lorma   --tag m4_lorma
    python phase4_cjpe.py --model M2_LexArcED --method full    --tag m2_full
    python phase4_cjpe.py --model M1_SC       --method adapter --tag m1_adapter
    python phase4_cjpe.py --model M2_LexArcED --method full --keep-every 2 \
        --kd results/ens_T/teacher.pt --tag kd_m2_half
    python phase4_cjpe.py ... --bench          # 30 steps, print tokens/s and peak memory, exit

THE MODEL

    judgment (BPE ids, whole document tokenised once and cached)
      -> a VIEW picks K chunks of L tokens:
            tail      the last K*L tokens            (default; ILDC: the end carries the decision)
            head      the first K*L tokens
            headtail  1 chunk from the start + K-1 from the end
            spread    K chunks evenly spaced over the whole document
      -> Phase-1 trunk over each chunk (bidirectional attention unless --causal)
      -> masked mean over each chunk's tokens               -> one vector per chunk
      -> + embedding of the chunk's position decile in the document
      -> document pooling over chunks (--pool attn | mean | tf = 1-layer transformer + attn)
      -> LayerNorm -> dropout -> linear -> 2 logits

PROTOCOL

Train on a fixed, seeded subset of train.json (--train-docs; the same subset for
every run so runs are comparable), for --epochs passes (default 1: there are
tens of thousands of documents). Dev is evaluated --evals times over training;
the best-dev point's dev and test logits are saved. Selection metric is dev
Macro-F1 at the dev-tuned threshold. Everything the ensembler needs is in
results/<tag>/ (see phase4_data.py), so all ensembling/thresholding is CPU work.

KD (--kd teacher.pt): the student is trained on the KD pool -- train documents
the teachers NEVER saw (perm[train_docs : train_docs + kd_pool]), so the soft
labels are honest held-out predictions -- with gold CE + KL to the teacher, and
on the unlabelled test text with KL only (--kd-test).
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
from phase4_data import (P1, P4, log as _log, token_cache, summarise, save_run,  # noqa: E402
                         write_submission, best_threshold, atomic_json)
from phase4_peft import (apply_method, blocks_of, trunk_param_ids, depth_of,     # noqa: E402
                         UnfreezeController, NF4Linear)
sys.path.insert(0, f"{P1}/shared/lib")

LOGF = None
PERM_SEED = 781


def log(*a):
    _log(*a)
    if LOGF:
        print(*a, file=LOGF, flush=True)


# ------------------------------------------------------------------- trunk --
def load_trunk(name, ckpt, keep_every=1, path=None):
    from cs781_model import Config, LexArcD
    ck = torch.load(path or f"{P1}/{name}/ckpt/{ckpt}", map_location="cpu", weights_only=False)
    cfg_d = ck["cfg"]
    sd = ck["model"] if "model" in ck else ck["state"]
    if "n_enc_layer" in cfg_d:
        from cs781_model_ed import EDConfig, LexArcED
        cfg = EDConfig(**cfg_d)
        t = LexArcED(cfg)
        t.load_state_dict(sd, strict=False)
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
        t.load_state_dict(sd, strict=False)
        for attr in ("head", "cq", "ck", "cv", "gate"):
            if hasattr(t, attr):
                delattr(t, attr)
        kind = "dec"
        if keep_every > 1:
            # Layer 0 must stay first: it produces the value-residual v0 that
            # every later layer's gate mixes in.
            t.blocks = nn.ModuleList(list(t.blocks)[::keep_every])
    return t, kind, cfg


# ------------------------------------------------------------------- model --
class CJPEModel(nn.Module):
    def __init__(self, trunk, kind, cfg, a):
        super().__init__()
        self.trunk, self.kind, self.a = trunk, kind, a
        d = cfg.d_model
        self.d = d
        self.R = getattr(cfg, "n_register", 0) if kind == "dec" else 0
        self.bi = not a.causal
        self.drop = nn.Dropout(a.dropout)
        self.cpos = nn.Embedding(10, d)
        nn.init.normal_(self.cpos.weight, std=0.02)
        if a.pool == "tf":
            layer = nn.TransformerEncoderLayer(d, 8, 2 * d, a.dropout, batch_first=True, norm_first=True)
            self.mix = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        else:
            self.mix = None
        self.att = nn.Sequential(nn.Linear(d, 256), nn.Tanh(), nn.Linear(256, 1))
        self.out_norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 2)

    def token_hidden(self, idx, keep):
        trunk, B = self.trunk, idx.size(0)
        x = trunk.emb(idx)
        lead = 0
        if self.kind == "dec" and self.R:
            x = torch.cat([trunk.reg.expand(B, -1, -1).to(x.dtype), x], 1)
            lead = self.R
        T = x.size(1)
        from cs781_model import build_rope
        cos, sin = build_rope(T, trunk.cfg.head_dim, x.device, trunk.emb.weight.dtype,
                              trunk.cfg.rope_theta)
        kfull = torch.cat([keep.new_ones(B, lead), keep], 1) if lead else keep
        mask = kfull[:, None, None, :] if (self.kind == "ed" or self.bi) else None
        ckpt = self.training and self.a.grad_ckpt and torch.is_grad_enabled()

        def call(fn, *args):
            if ckpt:
                return torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=False)
            return fn(*args)

        if self.kind == "dec":
            v0 = None
            for i, blk in enumerate(trunk.blocks):
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

            for blk in trunk.enc:
                x = call(lambda h, b=blk: enc_layer(h, b), x)
            x = trunk.enc_norm(x)
        return x[:, lead:]

    def forward(self, bt):
        H = self.token_hidden(bt["idx"], bt["keep"])                 # (C, T, d)
        k = bt["keep"].unsqueeze(-1).to(H.dtype)
        vec = (H * k).sum(1).float() / k.sum(1).float().clamp(min=1)  # (C, d)
        B, K = bt["cvalid"].shape
        X = vec.new_zeros(B * K, self.d)
        X[bt["cslot"]] = vec
        X = X.view(B, K, self.d) + self.cpos(bt["cpos"]).float()
        valid = bt["cvalid"]
        if self.mix is not None:
            X = self.mix(X, src_key_padding_mask=~valid)
        if self.a.pool == "mean":
            w = valid.float() / valid.float().sum(1, keepdim=True).clamp(min=1)
        else:
            s = self.att(X).squeeze(-1).masked_fill(~valid, -1e4)
            w = torch.softmax(s, -1)
        self.last_w = w.detach()
        doc = (w.unsqueeze(-1) * X).sum(1)
        return self.head(self.drop(self.out_norm(doc)))           # (B, 2)


# -------------------------------------------------------------------- data --
def view_chunks(tok, a, train=False, rng=None):
    """[(start, end)] token spans for one document, and each span's position decile."""
    n = len(tok)
    L, K = a.chunk, a.chunks
    if n == 0:
        return [(0, 0)], [0]
    if a.view == "tail":
        spans = [(max(0, n - (j + 1) * L), n - j * L) for j in range(K) if n - j * L > 0]
        spans = spans[::-1]
    elif a.view == "head":
        spans = [(j * L, min(n, (j + 1) * L)) for j in range(K) if j * L < n]
    elif a.view == "headtail":
        spans = [(0, min(n, L))]
        rest = [(max(L, n - (j + 1) * L), n - j * L) for j in range(K - 1) if n - j * L > L]
        spans += rest[::-1]
    elif a.view == "spread":
        if n <= K * L:
            spans = [(j * L, min(n, (j + 1) * L)) for j in range(K) if j * L < n]
        else:
            starts = np.linspace(0, n - L, K).round().astype(int)
            spans = [(int(s), int(s) + L) for s in starts]
    else:
        raise ValueError(a.view)
    if train and a.jitter and rng is not None and a.view == "tail" and n > K * L:
        # shift the whole tail window back by up to `jitter` tokens: the end of a
        # judgment is formulaic, so this stops the model keying on exact offsets
        sh = rng.randint(0, min(a.jitter, n - K * L))
        spans = [(s - sh, e - sh) for s, e in spans]
    pos = [min(9, int(10 * s / max(1, n))) for s, _ in spans]
    return spans, pos


def collate(docs, toks, a, device, train=False, rng=None):
    """docs: list of indices into toks. -> batch dict on device."""
    seqs, cslot, cpos_l = [], [], []
    K = a.chunks
    for b, i in enumerate(docs):
        spans, pos = view_chunks(toks[i], a, train, rng)
        for j, ((s, e), p) in enumerate(zip(spans, pos)):
            seqs.append(toks[i][s:e])
            cslot.append(b * K + j)
            cpos_l.append((b, j, p))
    T = max(1, max(len(s) for s in seqs))
    C = len(seqs)
    idx = torch.zeros(C, T, dtype=torch.long)
    keep = torch.zeros(C, T, dtype=torch.bool)
    for j, s in enumerate(seqs):
        if len(s):
            idx[j, :len(s)] = torch.from_numpy(s.astype(np.int64))
            keep[j, :len(s)] = True
        else:
            keep[j, 0] = True
    B = len(docs)
    cvalid = torch.zeros(B, K, dtype=torch.bool)
    cpos = torch.zeros(B, K, dtype=torch.long)
    for b, j, p in cpos_l:
        cvalid[b, j] = True
        cpos[b, j] = p
    bt = dict(idx=idx, keep=keep, cvalid=cvalid, cpos=cpos, cslot=torch.tensor(cslot))
    bt = {k: v.to(device, non_blocking=True) for k, v in bt.items()}
    bt["ntok"] = int(keep.sum())
    return bt


@torch.no_grad()
def predict(model, toks, a, device, bs=None):
    """z = logit1 - logit0 for every document in toks (list of arrays)."""
    model.eval()
    bs = bs or max(1, a.batch * 2)
    order = sorted(range(len(toks)), key=lambda i: min(len(toks[i]), a.chunk * a.chunks))
    z = torch.zeros(len(toks))
    ntok, t0 = 0, time.time()
    for k in range(0, len(order), bs):
        part = order[k:k + bs]
        bt = collate(part, toks, a, device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            lo = model(bt).float().cpu()
        z[part] = lo[:, 1] - lo[:, 0]
        ntok += bt["ntok"]
    model.train()
    return z, ntok / max(1e-6, time.time() - t0)


# ------------------------------------------------------------------- train --
def build_model(a, device):
    trunk, kind, cfg = load_trunk(a.model, a.ckpt, a.keep_every, a.trunk_path)
    model = CJPEModel(trunk, kind, cfg, a)
    n_train, ctrl, info = apply_method(model, a)
    model = model.to(device)          # AFTER wrapping: adapters are new modules
    n_trunk = sum(p.numel() for p in model.trunk.parameters())
    n_trunk += sum(m.n for m in model.trunk.modules() if isinstance(m, NF4Linear))
    return model, ctrl, info, n_train, n_trunk, len(blocks_of(trunk))


def param_groups(model, a):
    """Head LR for the new modules, trunk LR for the trunk, optional layer-wise decay."""
    tid = trunk_param_ids(model)
    if a.method == "unfreeze":            # closed blocks must already be in the optimiser
        tid = {id(p) for p in model.trunk.parameters()}
    head = [p for p in model.parameters() if p.requires_grad and id(p) not in tid]
    groups = [dict(params=head, lr=a.head_lr, base=a.head_lr, name="head")]
    trunk = [p for p in model.trunk.parameters() if id(p) in tid and
             (p.requires_grad or a.method == "unfreeze")]
    if not trunk:
        return groups
    if a.llrd < 1.0:
        dep = depth_of(model)
        nb = len(blocks_of(model.trunk))
        by = {}
        for p in trunk:
            dpt = dep.get(id(p))
            f = 1.0 if dpt is None else a.llrd ** (nb - 1 - dpt)
            if dpt is None and p is model.trunk.emb.weight:
                f = a.llrd ** nb
            by.setdefault(f, []).append(p)
        for f, ps in sorted(by.items()):
            groups.append(dict(params=ps, lr=a.lr * f, base=a.lr * f, name=f"trunk x{f:.3f}"))
    else:
        groups.append(dict(params=trunk, lr=a.lr, base=a.lr, name="trunk"))
    return groups


def run(a, device, tr_toks, tr_y, dev_toks, dev_y, test_toks, kd=None, tr_w=None):
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    model, ctrl, info, n_train, n_trunk, n_blocks = build_model(a, device)
    log(f"  {a.method}: {n_train/1e6:.3f}M trainable | trunk {n_trunk/1e6:.1f}M "
        f"({n_blocks} layers) | {info}")
    groups = param_groups(model, a)
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.98), weight_decay=a.wd)
    params = [p for g in groups for p in g["params"]]
    # class weights from the training labels (gold only)
    ys = [y for y in tr_y if y is not None]
    pos = sum(ys) / max(1, len(ys))
    cw = torch.tensor([0.5 / max(1e-3, 1 - pos), 0.5 / max(1e-3, pos)]) if a.class_weight else torch.ones(2)
    cw = cw.to(device)
    log(f"  train docs {len(tr_toks)} (gold {len(ys)}, pos rate {pos:.3f}) class weights "
        f"{[round(x, 3) for x in cw.tolist()]}")

    n_docs = len(tr_toks)
    tr_w = tr_w or [1.0] * n_docs
    steps_per_ep = math.ceil(n_docs / (a.batch * a.accum))
    total = steps_per_ep * a.epochs if not a.max_steps else a.max_steps
    if isinstance(ctrl, UnfreezeController):
        ctrl.every = max(1, int(a.unfreeze_frac * total / n_blocks))
        log(f"  gradual unfreezing: one block every {ctrl.every} updates")
    elif ctrl is not None:
        ctrl.total = total
        ctrl.t_init, ctrl.t_final = int(0.1 * total), int(0.3 * total)
    warm = max(1, int(a.warmup * total))
    eval_at = sorted({max(1, round(total * (k + 1) / a.evals)) for k in range(a.evals)})
    if a.bench:
        eval_at = []                      # a benchmark measures training speed only
    log(f"  {total} updates ({a.batch}x{a.accum} docs, {a.chunks}x{a.chunk} tokens/doc), "
        f"dev evals at {eval_at}")

    best = dict(f1=-1.0)
    trace = []
    t0 = time.time()
    tok_seen, t_last, tok_last = 0, time.time(), 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    step = 0
    done = False
    for ep in range(a.epochs if not a.max_steps else 10 ** 6):
        order = list(range(n_docs))
        rng.shuffle(order)
        micro = [order[k:k + a.batch] for k in range(0, len(order), a.batch)]
        tl, tn = 0.0, 0
        for mi, part in enumerate(micro):
            bt = collate(part, tr_toks, a, device, train=True, rng=rng)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                lo = model(bt)
            lo = lo.float()
            y = torch.tensor([tr_y[i] if tr_y[i] is not None else -1 for i in part], device=device)
            lab = y >= 0
            loss = lo.new_zeros(())
            if lab.any():
                sw = torch.tensor([tr_w[i] for i in part], device=device)[lab]
                ce_i = F.cross_entropy(lo[lab], y[lab], weight=cw, label_smoothing=a.smooth,
                                       reduction="none")
                ce = (ce_i * sw).sum() / (cw[y[lab]] * sw).sum().clamp(min=1e-6)
                loss = ce
            if kd is not None:
                tz = torch.tensor([kd[i] for i in part], device=device, dtype=torch.float32)
                T = a.kd_temp
                pt = torch.sigmoid(tz / T)
                pt = torch.stack([1 - pt, pt], -1)
                ls = F.log_softmax(lo / T, -1)
                kl = (pt * (pt.clamp(min=1e-9).log() - ls)).sum(-1) * T * T
                kdl = kl.mean()
                if lab.any():
                    # labelled docs: (1-alpha) CE + alpha KL; unlabelled: KL only
                    m = lab.float()
                    kdl_l = (kl * m).sum() / m.sum()
                    kdl_u = (kl * (1 - m)).sum() / (1 - m).sum().clamp(min=1)
                    loss = (1 - a.kd_alpha) * loss + a.kd_alpha * kdl_l + \
                        a.kd_alpha * kdl_u * ((1 - m).sum() / len(part))
                else:
                    loss = kdl
            if ctrl is not None and hasattr(ctrl, "orth_loss"):
                ol = ctrl.orth_loss()
                if torch.is_tensor(ol):
                    loss = loss + ol
            (loss / a.accum).backward()
            tok_seen += bt["ntok"]
            tl += float(loss.detach())
            tn += 1
            if (mi + 1) % a.accum and mi + 1 != len(micro):
                continue
            lr_f = (step / warm if step < warm else
                    0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm))))
            for g in opt.param_groups:
                g["lr"] = g["base"] * max(lr_f, a.min_lr_frac)
            torch.nn.utils.clip_grad_norm_([p for p in params if p.grad is not None], 1.0)
            opt.step()
            if ctrl is not None:
                ctrl.step(step)
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % a.log_every == 0 or step == 10:
                now = time.time()
                tps = (tok_seen - tok_last) / max(1e-6, now - t_last)
                t_last, tok_last = now, tok_seen
                mem = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0
                eta = (total - step) * (now - t0) / step / 60
                log(f"    step {step:5d}/{total} loss {tl/max(1,tn):.4f} | {tps/1e3:.1f}k tok/s "
                    f"| peak {mem:.1f} GB | {(now-t0)/60:.1f} min, ETA {eta:.0f} min")
                with open(f"{P4}/logs/live_{a.tag}.txt", "a", encoding="utf-8") as lf:
                    lf.write(f"{time.strftime('%H:%M')} step {step}/{total} {tps/1e3:.1f}k tok/s "
                             f"loss {tl/max(1,tn):.4f} ETA {eta:.0f}m\n")
                tl, tn = 0.0, 0
            if a.bench and step >= a.max_steps:
                done = True
                break
            if step in eval_at:
                zd, inf_tps = predict(model, dev_toks, a, device)
                s = summarise(dev_y, zd.tolist())
                trace.append(dict(step=step, **s, t=time.time() - t0))
                log(f"    [eval step {step}] dev macro-F1 @0 {s['f1_at0']:.4f} | tuned {s['f1_tuned']:.4f} "
                    f"(thr {s['thr']:+.2f}) | AUC {s['auc']:.4f} | P {s['precision']:.3f} R {s['recall']:.3f}"
                    f" | {inf_tps/1e3:.1f}k tok/s inference")
                with open(f"{P4}/logs/live_{a.tag}.txt", "a", encoding="utf-8") as lf:
                    lf.write(f"{time.strftime('%H:%M')} EVAL step {step} dev F1 {s['f1_tuned']:.4f} "
                             f"AUC {s['auc']:.4f}\n")
                if s["f1_tuned"] > best["f1"]:
                    t1 = time.time()
                    zt, _ = predict(model, test_toks, a, device)
                    best = dict(f1=s["f1_tuned"], step=step, dev=zd, test=zt, summ=s,
                                inf_s=time.time() - t1, inf_tps=inf_tps)
                    keys = {n for n, p in model.named_parameters() if p.requires_grad}
                    keys |= {n for n, _ in model.named_buffers() if n.endswith(".keep")}
                    best["state"] = {k: v.detach().to("cpu", copy=True) for k, v in
                                     model.state_dict().items() if k in keys}
            if step >= total:
                done = True
                break
        if done:
            break
    peak = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
    train_s = time.time() - t0
    tps_all = tok_seen / max(1e-6, train_s)
    return model, best, trace, dict(n_trainable=n_train, n_trunk=n_trunk, n_blocks=n_blocks,
                                    info=info, peak_gb=peak, train_s=train_s,
                                    train_tok_per_s=tps_all, tokens_seen=tok_seen,
                                    ranks=ctrl.ranks() if ctrl is not None else None)


def predict_only(a, device, dv, te):
    """Re-score a finished run from its saved weights: dev for the threshold, test for the zip.

    For when the organisers replace test.json. Run it with the SAME arguments as
    the original run (the queue's command line) plus --predict-only.
    """
    src = a.ckpt_from or a.tag
    cp = f"{P4}/ckpt/{src}.pt"
    assert os.path.exists(cp), f"{cp} missing: this run was trained without --save-ckpt"
    ck = torch.load(cp, map_location="cpu", weights_only=False)
    model, _, _, _, _, _ = build_model(a, device)
    missing = model.load_state_dict({k: v.float() if v.is_floating_point() else v
                                     for k, v in ck["state"].items()}, strict=False)
    log(f"  {a.tag}: loaded {len(ck['state'])} tensors from {cp} "
        f"({len(missing.unexpected_keys)} unexpected)")
    zd, _ = predict(model, dv["tok"], a, device)
    zt, _ = predict(model, te["tok"], a, device)
    s = summarise(dv["label"], zd.tolist())
    srd = f"{P4}/results/{src}"
    meta = json.load(open(f"{srd}/meta.json")) if os.path.exists(f"{srd}/meta.json") else dict(tag=a.tag)
    meta.update(tag=a.tag, ckpt_from=src, view=a.view, chunks=a.chunks)
    meta.update(dev_f1=s["f1_tuned"], thr=s["thr"], dev_auc=s["auc"], dev_p=s["precision"], dev_r=s["recall"],
                repredicted=time.strftime("%Y-%m-%d %H:%M"), n_test=len(te["ids"]))
    save_run(a.tag, dv["ids"], zd, te["ids"], zt, meta)
    log(f"  {a.tag}: re-predicted {len(te['ids'])} test docs | dev macro-F1 {s['f1_tuned']:.4f}")
    if a.submit:
        pred = {d: int(v > s["thr"]) for d, v in zip(te["ids"], zt.tolist())}
        write_submission(a.tag, te["ids"], pred, dev=s["f1_tuned"],
                         note=f"{a.model} {a.method} weights of {src}, read as {a.view} {a.chunks}x{a.chunk}",
                         kind="model")


# -------------------------------------------------------------------- main --
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="M1_SC M2_LexArcED M3_LexArcL M4_LexArcP")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--trunk-path", default=None)
    ap.add_argument("--method", default="full",
                    choices=["head", "full", "norm", "bitfit", "partial", "unfreeze", "adapter",
                             "lora", "rslora", "qlora", "adalora", "lorma", "ia3"])
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32.0)
    ap.add_argument("--ada-target", type=int, default=6)
    ap.add_argument("--targets", default="attn", choices=["qv", "attn", "all"])
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--adapter-dim", type=int, default=64)
    ap.add_argument("--top-k", type=int, default=4, help="partial: trainable top blocks")
    ap.add_argument("--unfreeze-frac", type=float, default=0.5,
                    help="unfreeze: all blocks open after this fraction of training")
    ap.add_argument("--llrd", type=float, default=1.0, help="layer-wise LR decay per block (1 = off)")
    ap.add_argument("--keep-every", type=int, default=1, help="2 = half-depth student")
    ap.add_argument("--causal", action="store_true")
    ap.add_argument("--n-virtual", type=int, default=0)
    # document view
    ap.add_argument("--view", default="tail", choices=["tail", "head", "headtail", "spread"])
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--chunks", type=int, default=2)
    ap.add_argument("--jitter", type=int, default=0)
    ap.add_argument("--pool", default="attn", choices=["attn", "mean", "tf"])
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--grad-ckpt", action="store_true")
    # optimisation
    ap.add_argument("--train-docs", type=int, default=12000)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=8, help="documents per micro-batch")
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=float, default=0.06)
    ap.add_argument("--min-lr-frac", type=float, default=0.0)
    ap.add_argument("--smooth", type=float, default=0.0)
    ap.add_argument("--no-class-weight", dest="class_weight", action="store_false")
    ap.add_argument("--evals", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--bench", action="store_true", help="run --max-steps (default 30) updates, report speed, exit")
    ap.add_argument("--use-rest", action="store_true",
                    help="also train on every train doc after the KD pool (all gold except the KD pool)")
    ap.add_argument("--extra", default=None, help="silver jsonl from phase4_external.py")
    ap.add_argument("--extra-docs", type=int, default=0, help="0 = all")
    ap.add_argument("--extra-weight", type=float, default=0.5)
    # KD
    ap.add_argument("--kd", default=None, help="teacher file {'ids': [...], 'z': tensor} over kd pool + test")
    ap.add_argument("--kd-pool", type=int, default=8000, help="train docs after --train-docs used for KD")
    ap.add_argument("--kd-alpha", type=float, default=0.7)
    ap.add_argument("--kd-temp", type=float, default=2.0)
    ap.add_argument("--kd-test", action="store_true", help="distil on the test text too")
    ap.add_argument("--dump-kd", action="store_true",
                    help="after training, predict the KD pool with the best state (teacher candidate)")
    # output
    ap.add_argument("--tag", required=True)
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--save-ckpt", action="store_true", help="keep the best trainable weights (bf16) in ckpt/<tag>.pt")
    ap.add_argument("--ckpt-from", default=None,
                    help="predict-only: weights of this run (default: --tag); results go under --tag")
    ap.add_argument("--predict-only", action="store_true",
                    help="no training: load ckpt/<tag>.pt, re-predict dev + the CURRENT test.json, rewrite the zip")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.lr is None:
        a.lr = dict(full=3e-5, head=0.0, norm=1e-3, bitfit=1e-3, partial=5e-5, unfreeze=5e-5,
                    adapter=5e-4, lora=3e-4, rslora=2e-4, qlora=3e-4, adalora=5e-4, lorma=3e-4,
                    ia3=3e-3)[a.method]
    if a.bench and not a.max_steps:
        a.max_steps = 30
    a.n_virtual = 0

    global LOGF
    rd = f"{P4}/results/{a.tag}"
    if os.path.exists(f"{rd}/meta.json") and not a.force and not a.bench and not a.predict_only:
        log(f"{a.tag}: {rd}/meta.json exists, skipping (use --force)")
        return
    os.makedirs(f"{P4}/logs", exist_ok=True)
    LOGF = open(f"{P4}/logs/{a.tag}.log", "a", encoding="utf-8")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        p = torch.cuda.get_device_properties(0)
        assert p.major * 10 + p.minor >= 80, f"{p.name}: bf16 is emulated below SM80 -- use an L4/A100"
        torch.set_float32_matmul_precision("high")
        gpu = p.name
    else:
        device, gpu = torch.device("cpu"), "cpu"
    log(f"\n===== {a.tag} | {gpu} | {a.model}/{a.ckpt} method={a.method} view={a.view} "
        f"{a.chunks}x{a.chunk} pool={a.pool} keep_every={a.keep_every} =====")
    log("  args: " + json.dumps(vars(a), sort_keys=True))

    from phase4_data import drop_test_copies
    tr = drop_test_copies(token_cache("train"), "train")
    dv = drop_test_copies(token_cache("dev"), "dev")
    te = token_cache("test")
    if a.predict_only:
        predict_only(a, device, dv, te)
        return
    perm = list(range(len(tr["ids"])))
    random.Random(PERM_SEED).shuffle(perm)
    sub = perm[:a.train_docs]
    kd = None
    if a.kd:
        teach = torch.load(a.kd, weights_only=False)
        tz = {d: float(v) for d, v in zip(teach["ids"], teach["z"].tolist())}
        pool = perm[a.train_docs:a.train_docs + a.kd_pool]
        tr_toks = [tr["tok"][i] for i in pool]
        tr_y = [tr["label"][i] for i in pool]
        kd = [tz[tr["ids"][i]] for i in pool]
        if a.kd_test:
            tr_toks += list(te["tok"])
            tr_y += [None] * len(te["ids"])
            kd += [tz[d] for d in te["ids"]]
        miss = sum(1 for v in kd if v is None)
        assert not miss, f"teacher missing {miss} docs"
        log(f"  KD: {len(pool)} held-out train docs + {len(te['ids']) if a.kd_test else 0} test docs, "
            f"alpha {a.kd_alpha} T {a.kd_temp}")
    else:
        if a.use_rest:
            sub = sub + perm[a.train_docs + a.kd_pool:]
        tr_toks = [tr["tok"][i] for i in sub]
        tr_y = [tr["label"][i] for i in sub]
    tr_w = [1.0] * len(tr_toks)
    if a.extra:
        assert not a.kd, "--extra with --kd is not supported (silver docs have no teacher scores)"
        from phase4_data import ext_cache
        ec = ext_cache(a.extra)
        k = a.extra_docs or len(ec["ids"])
        tr_toks += list(ec["tok"][:k])
        tr_y += list(ec["label"][:k])
        tr_w += [a.extra_weight] * min(k, len(ec["ids"]))
        log(f"  extra: {min(k, len(ec['ids']))} silver docs from {a.extra} at weight {a.extra_weight}")
    log(f"  training pool: {len(tr_toks)} docs")

    model, best, trace, cost = run(a, device, tr_toks, tr_y, dv["tok"], dv["label"], te["tok"], kd, tr_w)
    if a.bench:
        os.makedirs(f"{P4}/bench", exist_ok=True)
        b = dict(tag=a.tag, model=a.model, method=a.method, chunks=a.chunks, chunk=a.chunk,
                 batch=a.batch, grad_ckpt=a.grad_ckpt, keep_every=a.keep_every,
                 tok_per_s=cost["train_tok_per_s"], peak_gb=cost["peak_gb"], gpu=gpu,
                 n_trainable=cost["n_trainable"])
        atomic_json(b, f"{P4}/bench/{a.tag}.json", indent=1)
        log(f"  BENCH {a.tag}: {cost['train_tok_per_s']/1e3:.1f}k tok/s train, peak {cost['peak_gb']:.1f} GB")
        return

    s = best["summ"]
    tok_per_doc = a.chunks * a.chunk
    est = 12000 * tok_per_doc / max(1, cost["train_tok_per_s"]) / 60
    meta = dict(tag=a.tag, args=vars(a), dev_f1=s["f1_tuned"], dev_f1_at0=s["f1_at0"], thr=s["thr"],
                dev_auc=s["auc"], dev_p=s["precision"], dev_r=s["recall"], best_step=best["step"],
                trace=trace, kind="model", technique="kd" if a.kd else a.method, arch=a.model,
                **{k: v for k, v in cost.items()},
                test_inference_s=best["inf_s"], inference_tok_per_s=best["inf_tps"],
                minutes_per_12k_docs=est)
    extra = {}
    if a.dump_kd:
        model.load_state_dict({k: v.to(device) for k, v in best["state"].items()}, strict=False)
        pool = perm[a.train_docs:a.train_docs + a.kd_pool]
        zk, _ = predict(model, [tr["tok"][i] for i in pool], a, device)
        extra["kdpool"] = dict(ids=[tr["ids"][i] for i in pool], z=zk)
        yk = [tr["label"][i] for i in pool]
        sk = summarise(yk, zk.tolist(), thr=s["thr"])
        meta["kdpool_f1"] = sk["f1_tuned"]
        log(f"  KD pool ({len(pool)} unseen train docs): macro-F1 {sk['f1_tuned']:.4f} at the dev threshold "
            f"| AUC {sk['auc']:.4f}")
    save_run(a.tag, dv["ids"], best["dev"], te["ids"], best["test"], meta, extra)
    if a.save_ckpt:
        os.makedirs(f"{P4}/ckpt", exist_ok=True)
        st = {k: (v.to(torch.bfloat16) if v.is_floating_point() else v) for k, v in best["state"].items()}
        torch.save(dict(state=st, args=vars(a)), f"{P4}/ckpt/{a.tag}.pt")
    log(f"\n===== {a.tag}: dev macro-F1 {s['f1_tuned']:.4f} (thr {s['thr']:+.2f}; @0 {s['f1_at0']:.4f}) "
        f"AUC {s['auc']:.4f} | best step {best['step']} | {cost['n_trainable']/1e6:.2f}M trainable | "
        f"peak {cost['peak_gb']:.1f} GB | train {cost['train_s']/60:.1f} min "
        f"({cost['train_tok_per_s']/1e3:.1f}k tok/s) =====")
    if a.submit:
        zt = best["test"].tolist()
        pred = {d: int(v > s["thr"]) for d, v in zip(te["ids"], zt)}
        write_submission(a.tag, te["ids"], pred, dev=s["f1_tuned"],
                         note=f"{a.model} {a.method} {a.view} {a.chunks}x{a.chunk}", kind="model")
    try:
        from phase4_ens import write_status
        write_status()
    except Exception as e:
        log(f"  (status not updated: {e})")


if __name__ == "__main__":
    main()
