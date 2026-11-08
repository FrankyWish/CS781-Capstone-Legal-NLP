#!/usr/bin/env python3
"""CS-781 Phase 7 -- Phase-1 trunks as document encoders: tokenise, chunk, embed, rank.

    python phase7_enc.py tok   --splits train,dev,test --procs 8     CPU, once
    python phase7_enc.py embed --tag dense_m2 --model M2_LexArcED --split devx,test
    python phase7_enc.py dense --tag dense_m2 --agg mean --splits devx,test

This is the encoder every neural system in the phase shares -- the zero-shot baseline
here, and the contrastive/PEFT/KD runs in phase7_emb.py, which imports from this file.

LONG DOCUMENTS.  A PCR judgment is far longer than any trunk's pretrained context
(M1/M3/M4 ctx 1024, M2 window 1280), so a document is cut into `--chunk` token chunks,
each chunk is encoded independently, and the chunk vectors are combined:

    agg=mean     one vector per document (mean of L2-normalised chunk vectors)
    agg=maxsim   keep the chunks; sim(q,c) = mean over q-chunks of max over c-chunks
                 (late interaction at chunk granularity, no cross-attention cost)

Chunks are taken as the head plus evenly spaced windows over the rest, so a 40k-word
judgment is represented by its opening (parties, facts, issues) and samples of the
body, deterministically and identically for queries and candidates.

POOLING AND ATTENTION.  M2's encoder is genuinely bidirectional (EncBlock builds its
SelfAttn with causal=False), which makes it the natural retrieval backbone; M1/M3/M4
are causal decoders, where mean pooling sees each token only in left context.  `--bidir`
passes an all-visible attention mask to a decoder trunk so every token sees the whole
chunk (LLM2Vec's conversion step); it is off by default because it moves the trunk away
from what it was pretrained with, and whether it helps is measured, not assumed.

Padding is handled exactly: `enc_forward` reimplements M2's encoder block arithmetic
with an explicit key-padding mask (the shipped `LexArcED.encode` takes no mask and
would let pad positions into the bidirectional states), and the smoke test asserts it
reproduces `lm.encode()` to 1e-4 when nothing is padded.  Decoder trunks take the mask
through the Phase-1 `_trunk(..., attn_mask=...)` argument.  Pooling always averages
real positions only.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import phase7_data as D                                      # noqa: E402
import phase7_rank as R                                      # noqa: E402

P1, P7 = D.P1, D.P7
LIB = os.environ.get("CS781_LIB", f"{P1}/shared/lib")
sys.path.insert(0, LIB)
TOKJSON = os.environ.get("CS781_TOKENIZER", f"{P1}/shared/tokenizer/bpe_legal_32k.json")
CACHE = f"{P7}/cache"
PAD_MULT = 128                       # tail chunks are padded up to a multiple of this


def log(*a):
    s = " ".join(str(x) for x in a)
    print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)


# ---------------------------------------------------------------- tokenizer --
_TOK = {}


def tokenizer():
    if "t" not in _TOK:
        from cs781_bpe import BPETokenizer
        _TOK["t"] = BPETokenizer.load(TOKJSON)
    return _TOK["t"]


def eos_id():
    return dict(tokenizer().specials)["<|endoftext|>"]


def _enc_init(path):
    from cs781_bpe import BPETokenizer
    _TOK["t"] = BPETokenizer.load(path)


def _enc(text):
    return np.asarray(_TOK["t"].encode(text), dtype=np.uint16)


def side_texts(split, side):
    """(ids, texts) for the query side or the candidate side of a split."""
    if side == "q":
        rs = D.queries(split)
        return [r["id"] for r in rs], [" ".join(r["sents"]) for r in rs]
    pool = D.candidates(split)
    ids = sorted(pool)
    return ids, [" ".join(pool[c]) for c in ids]


def tok_path(split, side):
    return f"{CACHE}/tok_{split}_{side}.npz"


def cmd_tok(a):
    from multiprocessing import Pool
    os.makedirs(CACHE, exist_ok=True)
    for split in a.splits.split(","):
        for side in ("q", "c"):
            out = tok_path(split, side)
            if os.path.exists(out):
                log(f"  {out} exists")
                continue
            ids, texts = side_texts(split, side)
            t0 = time.time()
            if a.procs > 1:
                with Pool(a.procs, initializer=_enc_init, initargs=(TOKJSON,)) as pool:
                    enc = pool.map(_enc, texts, chunksize=8)
            else:
                _enc_init(TOKJSON)
                enc = [_enc(t) for t in texts]
            flat = np.concatenate(enc) if enc else np.zeros(0, np.uint16)
            off = np.concatenate([[0], np.cumsum([len(x) for x in enc])]).astype(np.int64)
            tmp = out + ".tmp.npz"
            np.savez(tmp, flat=flat, off=off, ids=np.array(ids))
            os.replace(tmp, out)
            log(f"  {split}/{side}: {len(ids)} docs, tokens {D.pct([len(x) for x in enc])} "
                f"({time.time() - t0:.0f}s)")


class Tok:
    def __init__(self, split, side):
        z = np.load(tok_path(split, side), allow_pickle=False)
        self.ids = [str(x) for x in z["ids"]]
        self.flat, self.off = z["flat"], z["off"]
        self.pos = {k: i for i, k in enumerate(self.ids)}

    def __len__(self):
        return len(self.ids)

    def doc(self, i):
        return self.flat[self.off[i]:self.off[i + 1]].astype(np.int64)


def chunks_of(ids, chunk, max_chunks):
    """Head chunk + evenly spaced windows over the rest.  Deterministic, and identical
    for queries and candidates so the two sides are represented the same way."""
    n = len(ids)
    if n <= chunk:
        return [ids]
    total = (n + chunk - 1) // chunk
    if total <= max_chunks:
        starts = [i * chunk for i in range(total)]
    else:
        starts = [0] + [int(round(x)) for x in
                        np.linspace(chunk, n - chunk, max_chunks - 1)]
    out = []
    for s in starts:
        s = max(0, min(s, max(n - chunk, 0)))
        c = ids[s:s + chunk]
        if len(c) >= max(32, chunk // 8) or not out:
            out.append(c)
    return out[:max_chunks]


# -------------------------------------------------------------------- model --
def _cfg(cls, d):
    names = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in names})


def load_lm(name, ckpt="best.pt", keep_every=1, path=None):
    """Same loader as phase6_sum.load_lm (copied, not imported: importing phase6_sum
    drags in the Phase-6 data module and its own HF downloads into an unattended run).
    keep_every>1 drops layers for the KD student; layer 0 always stays, because the
    value residual reads it."""
    import torch
    import torch.nn as nn
    from cs781_model import Config, LexArcD
    ck = torch.load(path or f"{P1}/{name}/ckpt/{ckpt}", map_location="cpu", weights_only=False)
    cd = ck["cfg"] if isinstance(ck["cfg"], dict) else dataclasses.asdict(ck["cfg"])
    if "n_enc_layer" in cd:
        from cs781_model_ed import EDConfig, LexArcED
        cfg = _cfg(EDConfig, cd)
        m, kind = LexArcED(cfg), "ed"
    else:
        cfg = _cfg(Config, cd)
        if cfg.use_pointer:
            from cs781_model_ptr import LexArcP
            m, kind = LexArcP(cfg), "ptr"
        else:
            m, kind = LexArcD(cfg), "dec"
    miss, unexp = m.load_state_dict(ck["model"], strict=False)
    miss = [k for k in miss if not k.endswith("head.weight")]        # tied to emb
    if miss:
        log(f"  !! {name}: {len(miss)} weights missing from the checkpoint, e.g. {miss[:3]}")
    del ck
    if keep_every > 1:
        if kind == "ed":
            m.enc = nn.ModuleList(list(m.enc)[::keep_every])
            m.dec = nn.ModuleList(list(m.dec)[::keep_every])
            cfg.n_enc_layer, cfg.n_dec_layer = len(m.enc), len(m.dec)
        else:
            m.blocks = nn.ModuleList(list(m.blocks)[::keep_every])
            cfg.n_layer = len(m.blocks)
    return m, kind, cfg


def trained_len(kind, cfg):
    return max(cfg.split_points) if kind == "ed" else cfg.ctx


def ntk(cfg, factor):
    """NTK-aware RoPE scaling, as in Phase 6: one config field, so training and
    inference cannot disagree about it."""
    if factor > 1.0:
        hd = cfg.head_dim
        cfg.rope_theta = float(cfg.rope_theta * factor ** (hd / (hd - 2)))
    return cfg.rope_theta


def enc_forward(lm, cfg, idx, key_ok):
    """M2's encoder stack with an explicit key-padding mask.

    LexArcED.encode() takes no mask, and its blocks are bidirectional, so a padded
    batch would mix pad states into every real position.  This recomputes the same
    arithmetic from the blocks' own modules (so LoRA/IA3 wrappers on wq/wk/wv/wo are
    picked up unchanged) with attn_mask set.  key_ok: (B,T) bool, True on real tokens."""
    import torch.nn.functional as F
    from cs781_model import apply_rope, build_rope
    B, T = idx.shape
    cos, sin = build_rope(T, cfg.head_dim, idx.device, lm.emb.weight.dtype, cfg.rope_theta)
    mask = None if bool(key_ok.all()) else key_ok[:, None, None, :].expand(B, 1, T, T)
    x = lm.emb(idx)
    for blk in lm.enc:
        a = blk.n1(x)
        att = blk.attn
        q = att.wq(a).view(B, T, att.nh, att.hd).transpose(1, 2)
        k = att.wk(a).view(B, T, att.nkv, att.hd).transpose(1, 2)
        v = att.wv(a).view(B, T, att.nkv, att.hd).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cfg.qk_norm:
            q = F.rms_norm(q, (att.hd,))
            k = F.rms_norm(k, (att.hd,))
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)
        x = x + att.wo(y.transpose(1, 2).reshape(B, T, att.nh * att.hd))
        h = blk.n2(x)
        x = x + blk.ffn["d"](F.silu(blk.ffn["g"](h)) * blk.ffn["u"](h))
    return lm.enc_norm(x)


def dec_forward(lm, cfg, idx, key_ok, bidir=False):
    """Decoder trunk states for the real positions, registers stripped.

    bidir=False keeps the pretrained causal view (right padding means pads are never
    attended, so no mask is needed and the fast kernel runs).  bidir=True passes an
    all-visible mask, which is the LLM2Vec conversion -- cheap to try, and a departure
    from pretraining, so it is reported as its own system."""
    import torch
    from cs781_model import build_rope
    B, T = idx.shape
    Rn = cfg.n_register
    cos, sin = build_rope(T + Rn, cfg.head_dim, idx.device, lm.emb.weight.dtype,
                          cfg.rope_theta)
    mask = None
    if bidir:
        ok = torch.cat([torch.ones(B, Rn, dtype=torch.bool, device=idx.device), key_ok], 1)
        mask = ok[:, None, None, :].expand(B, 1, T + Rn, T + Rn)
    h, _ = lm._trunk(idx, cos, sin, attn_mask=mask)
    return h[:, Rn:]


def pool(states, key_ok, how="mean"):
    """(B,T,d) -> (B,d) over real positions only."""
    import torch
    m = key_ok.unsqueeze(-1).to(states.dtype)
    if how == "mean":
        return (states * m).sum(1) / m.sum(1).clamp(min=1e-6)
    if how == "max":
        return states.masked_fill(~key_ok.unsqueeze(-1), -1e4).amax(1)
    if how == "last":
        last = key_ok.float().cumsum(1).argmax(1)
        return states[torch.arange(states.size(0), device=states.device), last]
    if how == "first":
        return states[:, 0]
    raise ValueError(how)


def embed_side(lm, kind, cfg, tk, a, device):
    """-> (flat (n_chunks,d) float16, offsets (n_docs+1,)).  Length-bucketed so a pad
    never exceeds PAD_MULT-1 tokens and the real positions are pooled on their own."""
    import torch
    jobs = []                                           # (doc index, chunk token ids)
    for i in range(len(tk)):
        for c in chunks_of(tk.doc(i), a.chunk, a.max_chunks):
            jobs.append((i, c))
    order = sorted(range(len(jobs)), key=lambda j: len(jobs[j][1]))
    out = np.zeros((len(jobs), cfg.d_model), dtype=np.float16)
    eos = eos_id()
    bs = a.batch
    t0 = time.time()
    with torch.no_grad():
        s = 0
        while s < len(order):
            sel = order[s:s + bs]
            T = max(len(jobs[j][1]) for j in sel)
            T = min(int(np.ceil(T / PAD_MULT) * PAD_MULT), a.chunk)
            idx = np.full((len(sel), T), eos, dtype=np.int64)
            ok = np.zeros((len(sel), T), dtype=bool)
            for r, j in enumerate(sel):
                c = jobs[j][1][:T]
                idx[r, :len(c)] = c
                ok[r, :len(c)] = True
            it = torch.from_numpy(idx).to(device)
            okt = torch.from_numpy(ok).to(device)
            st = (enc_forward(lm, cfg, it, okt) if kind == "ed"
                  else dec_forward(lm, cfg, it, okt, bidir=a.bidir))
            v = pool(st.float(), okt, a.pool)
            v = torch.nn.functional.normalize(v, dim=-1)
            out[sel] = v.cpu().numpy().astype(np.float16)
            s += bs
            if (s // bs) % 50 == 0:
                log(f"    {s}/{len(order)} chunks ({time.time() - t0:.0f}s)")
    counts = np.zeros(len(tk), dtype=np.int64)
    for i, _ in jobs:
        counts[i] += 1
    off = np.concatenate([[0], np.cumsum(counts)])
    perm = np.argsort([j[0] for j in jobs], kind="stable")
    return out[perm], off


def emb_path(tag, split, side):
    return f"{CACHE}/emb_{tag}_{split}_{side}.npz"


def cmd_embed(a):
    import torch
    os.makedirs(CACHE, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lm, kind, cfg = load_lm(a.model, a.ckpt, a.keep_every, a.weights)
    factor = max(a.chunk / trained_len(kind, cfg), 1.0)
    theta = ntk(cfg, factor) if not a.no_ntk else cfg.rope_theta
    lm.eval().to(device)
    if device.type == "cuda":
        lm.to(torch.bfloat16)
    log(f"  {a.model} kind={kind} d={cfg.d_model} chunk={a.chunk} "
        f"rope x{factor:.2f} theta {theta:.0f} pool={a.pool} bidir={a.bidir}")
    meta = dict(tag=a.tag, kind="model", model=a.model, encoder=kind, pool=a.pool,
                bidir=bool(a.bidir), chunk=a.chunk, max_chunks=a.max_chunks,
                rope_factor=round(factor, 3), weights=a.weights or f"{a.model}/{a.ckpt}",
                keep_every=a.keep_every, d_model=cfg.d_model,
                n_params=sum(p.numel() for p in lm.parameters()))
    for split in a.split.split(","):
        for side in ("q", "c"):
            p = emb_path(a.tag, split, side)
            if os.path.exists(p) and not a.force:
                log(f"  {p} exists")
                continue
            tk = Tok(split, side)
            flat, off = embed_side(lm, kind, cfg, tk, a, device)
            tmp = p + ".tmp.npz"
            np.savez(tmp, flat=flat, off=off, ids=np.array(tk.ids))
            os.replace(tmp, p)
            log(f"  wrote {p}: {len(tk)} docs, {flat.shape[0]} chunks")
    R.save_run(a.tag, "meta", {}, None, meta)


class Emb:
    def __init__(self, tag, split, side):
        z = np.load(emb_path(tag, split, side), allow_pickle=False)
        self.flat = z["flat"].astype(np.float32)
        self.off = z["off"]
        self.ids = [str(x) for x in z["ids"]]

    def doc(self, i):
        return self.flat[self.off[i]:self.off[i + 1]]

    def mean(self):
        """One L2-normalised vector per document."""
        v = np.stack([self.doc(i).mean(0) for i in range(len(self.ids))])
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)


def dense_rank(tag, split, agg="mean", topn=R.TOPN, qbatch=32, emb_tag=None):
    """agg=mean: one vector each side.  agg=maxsim: segment reductions over a single
    chunk-by-chunk matmul -- max over the candidate's chunks, then mean over the
    query's, which is late interaction at chunk granularity."""
    et = emb_tag or tag
    q, c = Emb(et, split, "q"), Emb(et, split, "c")
    drop = R.self_excluded(split)
    cpos = {k: i for i, k in enumerate(c.ids)}
    rank, score = {}, {}
    Qm = q.mean() if agg == "mean" else None
    Cm = c.mean() if agg == "mean" else None
    cstart = c.off[:-1]
    qn = np.maximum(np.diff(q.off), 1)[:, None].astype(np.float32)
    for s in range(0, len(q.ids), qbatch):
        nsel = min(qbatch, len(q.ids) - s)
        sel = range(s, s + nsel)
        if agg == "mean":
            S = Qm[s:s + nsel] @ Cm.T
        else:
            # reduceat takes segment STARTS only: q.off[s:s+nsel], never the trailing
            # total, which would index one past the last chunk row.
            Sc = q.flat[q.off[s]:q.off[s + nsel]] @ c.flat.T
            Sc = np.maximum.reduceat(Sc, cstart, axis=1)          # (q chunks, n cands)
            S = np.add.reduceat(Sc, q.off[s:s + nsel] - q.off[s], axis=0) / qn[s:s + nsel]
        for r, i in enumerate(sel):
            qid = q.ids[i]
            col = np.array(S[r], dtype=np.float32, copy=True)
            ex = drop.get(qid)
            if ex is not None and ex in cpos:
                col[cpos[ex]] = -np.inf
            n = min(topn, len(col))
            top = np.argpartition(-col, n - 1)[:n]
            top = top[np.argsort(-col[top], kind="stable")]
            rank[qid] = [c.ids[j] for j in top]
            score[qid] = col[top].astype(np.float32)
    return rank, score


def cmd_dense(a):
    meta = dict(tag=a.tag, kind="model", agg=a.agg, emb_tag=a.emb_tag or a.tag)
    for split in a.splits.split(","):
        rank, score = dense_rank(a.tag, split, a.agg, a.topn, emb_tag=a.emb_tag)
        R.save_run(a.tag, split, rank, score)
        if split != "test":
            s = R.evaluate(rank, split)
            meta[f"{split}_scores"] = s
            R.report(f"{a.tag}/{split} ({a.agg})", s)
    R.save_run(a.tag, "meta", {}, None, meta)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("tok")
    p.add_argument("--splits", default="train,dev,test")
    p.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.set_defaults(fn=cmd_tok)
    p = sp.add_parser("embed")
    p.add_argument("--tag", required=True)
    p.add_argument("--model", default="M2_LexArcED")
    p.add_argument("--ckpt", default="best.pt")
    p.add_argument("--weights", default=None, help="a fine-tuned .pt instead of Phase 1")
    p.add_argument("--keep-every", type=int, default=1)
    p.add_argument("--split", default="devx,test")
    p.add_argument("--chunk", type=int, default=1024)
    p.add_argument("--max-chunks", type=int, default=4)
    p.add_argument("--pool", default="mean", choices=["mean", "max", "last", "first"])
    p.add_argument("--bidir", action="store_true")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--no-ntk", action="store_true")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_embed)
    p = sp.add_parser("dense")
    p.add_argument("--tag", required=True)
    p.add_argument("--agg", default="mean", choices=["mean", "maxsim"])
    p.add_argument("--emb-tag", default=None,
                   help="score another tag's cached embeddings (e.g. mean vs maxsim)")
    p.add_argument("--splits", default="devx,test")
    p.add_argument("--topn", type=int, default=R.TOPN)
    p.set_defaults(fn=cmd_dense)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
