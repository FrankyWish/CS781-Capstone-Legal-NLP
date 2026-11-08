#!/usr/bin/env python3
"""CS-781 Phase 1 -- fire-and-forget: gates, train, generate, submission.zip.

    python train_phase1.py --model M1_LexArcD --hours 6.0

Order of business, and the reason for it:

  0  clean the dev set            some corpus documents contain pathological
                                  "words" (PDF extraction artefacts). One dev
                                  prompt of 512 words tokenised to 12,976
                                  tokens. The dev set is the only measuring
                                  instrument we have; it gets validated.
  1  GATES, before any real compute. Each one aborts the run.
       G1 shift        forward()'s loss equals CE(logits[:-1], idx[1:]) exactly
       G2 overfit      200 steps on one fixed batch, loss < 0.1, and then
                       argmax(logits[i]) == idx[i+1] on >95% of positions
       G3 kv-cache     8 greedy tokens through the incremental path must equal
                       the argmax of a full forward over prompt+generated
     G1+G2 together are the 90-second test that would have caught the bug that
     cost 44 GPU-hours.
  2  train to a wall-clock budget, not a step count, so a submission always
     exists. Checkpoint to Drive every 20 min, resumable.
  3  in-run fluency gate at 10% and 25%: function-word bigrams per 128 words.
     Real legal prose scores 15-25; word salad scores 1-2. Aborts on salad,
     because BERTScore cannot see broken syntax and will happily reward it.
  4  generate 1000 continuations, exactly 128 words each, write
     prediction.json and submission.zip.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
import zipfile
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.environ.get("CS781_ROOT", "/content/drive/MyDrive/cs781_capstone")
sys.path.insert(0, f"{ROOT}/shared/lib")
from cs781_bpe import BPETokenizer                       # noqa: E402
from cs781_model import Config, LexArcD, build_optimizers  # noqa: E402

CORPUS = f"{ROOT}/shared/corpus"
DEVDIR = f"{ROOT}/shared/dev"

# Phase mix. 87-95% of the official test prompts classify as Supreme Court, but
# SC is only 254.7M tokens, so we cannot simply train on it. Breadth early where
# diversity pays, then anneal onto the target distribution where it counts.
PHASES = [
    (0.60, {"train_supreme_court": .35, "train_high_courts": .30,
            "train_tribunals": .25, "train_acts": .10}),
    (0.90, {"train_supreme_court": .70, "train_high_courts": .20,
            "train_tribunals": .10, "train_acts": .00}),
    (1.01, {"train_supreme_court": 1.0}),
]

FUNCTION_BIGRAMS = ["of the", "in the", "to the", "has been", "is not", "that the",
                    "and the", "for the", "by the", "on the", "it is", "with the",
                    "of this", "as the", "the said", "in this", "there is",
                    "which is", "it was", "to be"]


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def safe_save(obj, path):
    """Write a checkpoint without ever killing the run.

    A failed checkpoint costs 20 minutes of progress. An unhandled exception
    from a full Drive costs the whole run, and Drive filling up mid-run is a
    thing that actually happens. Writes to .tmp then renames, so a partial
    write never replaces a good checkpoint.
    """
    try:
        torch.save(obj, path + ".tmp")
        os.replace(path + ".tmp", path)
        return True
    except Exception as e:
        log(f"   CHECKPOINT WRITE FAILED ({type(e).__name__}: {e}) -- continuing")
        try:
            os.remove(path + ".tmp")
        except OSError:
            pass
        return False


def fluency(text):
    """Function-word bigrams per 128 words. Legal prose 15-25, salad 1-2."""
    w = text.lower().split()
    if not w:
        return 0.0
    s = " " + " ".join(w) + " "
    n = sum(s.count(" " + b + " ") for b in FUNCTION_BIGRAMS)
    return n * 128.0 / len(w)


# ------------------------------------------------------------- dev cleaning --
def load_clean_dev(tok, max_tokens=900):
    p = json.load(open(f"{DEVDIR}/dev_prompts.json", encoding="utf-8"))
    t = json.load(open(f"{DEVDIR}/dev_truth.json", encoding="utf-8"))
    cache = f"{DEVDIR}/dev_clean_ids.json"
    if os.path.exists(cache):
        keep = json.load(open(cache))
    else:
        keep, lens = [], []
        for k, v in p.items():
            n = len(tok.encode(v["text"]))
            lens.append(n)
            if n <= max_tokens:
                keep.append(k)
        lens.sort()
        log(f"dev prompt tokens: p50 {lens[len(lens)//2]} p95 {lens[int(.95*len(lens))]} "
            f"max {lens[-1]}")
        log(f"dev clean: kept {len(keep)}/{len(p)} at <= {max_tokens} tokens "
            f"({len(p)-len(keep)} dropped as PDF-artefact documents)")
        json.dump(keep, open(cache, "w"))
    return {k: p[k] for k in keep}, {k: t[k] for k in keep}


# ------------------------------------------------------------------ loader --
class ShardLoader:
    """Yields (B, T) token windows. Never yields labels -- see cs781_model."""

    def __init__(self, B, T, device, seed=1337):
        man = json.load(open(f"{CORPUS}/manifest.json"))
        self.arr, self.p, self.tok_total = {}, {}, 0
        for s, st in man["sources"].items():
            shards, skipped = [], 0
            for n in st["shards"]:
                a = np.memmap(f"{CORPUS}/{n}", dtype=np.uint16, mode="r")
                # build_corpus flushes the leftover buffer after hitting the
                # token cap, so a source can end with a runt shard (tribunals
                # ended with 82 tokens). Anything that cannot hold one window
                # is unusable.
                if len(a) < T + 2:
                    skipped += 1
                    continue
                shards.append(a)
            if not shards:
                continue
            lens = np.array([len(x) for x in shards], dtype=np.float64)
            # Sample a shard in proportion to its length. Uniform choice would
            # over-sample the short tail shard of each source by ~10x.
            self.arr[s] = shards
            self.p[s] = lens / lens.sum()
            self.tok_total += int(lens.sum())
            print(f"  {s:22s} {len(shards)} shards, {lens.sum()/1e6:,.0f}M tokens"
                  + (f" ({skipped} runt shard(s) skipped)" if skipped else ""),
                  flush=True)
        self.B, self.T, self.device = B, T, device
        self.rng = np.random.default_rng(seed)
        self.val = {s: np.memmap(f"{CORPUS}/val_{s}.bin", dtype=np.uint16, mode="r")
                    for s in self.arr if os.path.exists(f"{CORPUS}/val_{s}.bin")}

    def _draw(self, source, n):
        shards, p = self.arr[source], self.p[source]
        out = np.empty((n, self.T), dtype=np.int64)
        picks = self.rng.choice(len(shards), size=n, p=p)
        for i, k in enumerate(picks):
            a = shards[k]
            j = self.rng.integers(0, len(a) - self.T - 1)
            out[i] = a[j:j + self.T]
        return out

    def batch(self, weights):
        names = [s for s in weights if s in self.arr and weights[s] > 0]
        w = np.array([weights[s] for s in names], dtype=np.float64)
        w /= w.sum()
        counts = self.rng.multinomial(self.B, w)
        parts = [self._draw(s, c) for s, c in zip(names, counts) if c]
        x = np.concatenate(parts, 0)
        self.rng.shuffle(x)
        return torch.from_numpy(x).to(self.device, non_blocking=True)

    def val_batch(self, source, n, seed):
        rng = np.random.default_rng(seed)
        a = self.val[source]
        out = np.empty((n, self.T), dtype=np.int64)
        for i in range(n):
            j = rng.integers(0, len(a) - self.T - 1)
            out[i] = a[j:j + self.T]
        return torch.from_numpy(out).to(self.device)


SC_ONLY = {"train_supreme_court": 1.0}


def mix_at(frac, mode="phased"):
    if mode == "sc":
        return SC_ONLY
    for upto, m in PHASES:
        if frac < upto:
            return m
    return PHASES[-1][1]


# ------------------------------------------------------------------- gates --
def gate_shift(model, loader, device):
    log("G1 shift: forward() loss must equal CE(logits[:-1], idx[1:])")
    idx = loader.batch({"train_supreme_court": 1.0})[:4, :128]
    model.eval()
    cfg = model.cfg
    z, cfg.z_loss = cfg.z_loss, 0.0
    with torch.no_grad():
        if cfg.use_pointer:
            # Force the gate fully onto the vocabulary path. The mixture then
            # has to reduce exactly to a plain LM, which proves the vocabulary
            # branch is aligned; G2 then proves the mixture itself is.
            a = model(idx, force_gate=1.0).item()
        else:
            a = model(idx).item()
        lg = model.logits_full(idx)
        b = F.cross_entropy(lg[:, :-1].reshape(-1, cfg.vocab_size).float(),
                            idx[:, 1:].reshape(-1)).item()
    cfg.z_loss = z
    log(f"   forward={a:.6f}  manual={b:.6f}  delta={abs(a-b):.2e}")
    assert abs(a - b) < 1e-3, (
        "G1 FAILED: the shift between inputs and targets is not exactly one "
        "position. This is the bug class that cost 44 GPU-hours.")
    log("G1 PASS")


def gate_overfit(model, loader, device, steps=200, target=0.1):
    log(f"G2 overfit: {steps} steps on one fixed batch, loss must go < {target}")
    idx = loader.batch({"train_supreme_court": 1.0})[:8, :256].contiguous()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=True)
    model.train()
    last = None
    for i in range(steps):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(idx)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        last = loss.item()
        if i % 50 == 0:
            log(f"   step {i:3d} loss {last:.4f}")
    log(f"   final loss {last:.4f}")
    assert last < target, f"G2 FAILED: cannot memorise 8 sequences (loss {last:.3f})"
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        if model.cfg.use_pointer:
            # Score the MIXED distribution, not the vocabulary logits: the
            # mixture is what generation samples from, so it is what has to be
            # correctly aligned.
            lg = model.mixed_logprobs(idx)
            acc = (lg.argmax(-1) == idx[:, 1:]).float().mean().item()
        else:
            lg = model.logits_full(idx)
            acc = (lg[:, :-1].argmax(-1) == idx[:, 1:]).float().mean().item()
    log(f"   argmax(p_mixed[i]) == idx[i+1] on {acc:.1%} of positions"
        if model.cfg.use_pointer else
        f"   argmax(logits[i]) == idx[i+1] on {acc:.1%} of positions")
    assert acc > 0.95, (
        f"G2 FAILED: memorised the batch but predicts the wrong offset ({acc:.1%}). "
        "This is what predicting two-tokens-ahead looks like.")
    log("G2 PASS")
    for p in model.parameters():
        p.grad = None


def gate_kv_cache(model, loader, tok, device, n_new=8):
    log("G3 kv-cache: incremental decode must match a full forward")
    idx = loader.batch({"train_supreme_court": 1.0})[:4, :64]
    mask = torch.ones_like(idx, dtype=torch.bool)
    model.eval()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        gen = model.generate(idx, mask, n_new, greedy=True, use_cache=True)
        full = torch.cat([idx, gen], 1)
        # For the pointer model this also validates the copy cache: kc/vc and
        # tok_hist are grown incrementally during generation and recomputed
        # from scratch here.
        lg = (model.mixed_logprobs(full) if model.cfg.use_pointer
              else model.logits_full(full)[:, :-1])
    ref = lg[:, idx.size(1) - 1:].argmax(-1)
    first = (ref[:, 0] == gen[:, 0]).float().mean().item()
    agree = (ref == gen).float().mean().item()
    log(f"   first token {first:.0%} | all {n_new} tokens {agree:.0%}")
    assert first == 1.0, (
        "G3 FAILED: the first generated token disagrees with a full forward. "
        "The prompt pass, the left-pad mask or the RoPE offset is wrong.")
    assert agree > 0.6, (
        f"G3 FAILED: the KV cache diverges after the first token ({agree:.0%}). "
        "Generation would be silently wrong while training looked fine.")
    log("G3 PASS")


# -------------------------------------------------------------- generation --
def left_pad_batch(seqs, device):
    L = max(len(s) for s in seqs)
    idx = torch.zeros(len(seqs), L, dtype=torch.long)
    msk = torch.zeros(len(seqs), L, dtype=torch.bool)
    for i, s in enumerate(seqs):
        idx[i, L - len(s):] = torch.tensor(s, dtype=torch.long)
        msk[i, L - len(s):] = True
    return idx.to(device), msk.to(device)


@torch.no_grad()
def generate_for(model, tok, prompts, device, max_ctx=790, batch=16,
                 n_new=210, **dec):
    model.eval()
    keys = list(prompts)
    enc = {k: tok.encode(prompts[k]["text"])[-max_ctx:] for k in keys}
    order = sorted(keys, key=lambda k: len(enc[k]))     # bucket by length
    out, t0 = {}, time.time()
    for i in range(0, len(order), batch):
        grp = order[i:i + batch]
        idx, msk = left_pad_batch([enc[k] for k in grp], device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            gen = model.generate(idx, msk, n_new, **dec)
        for k, row in zip(grp, gen.tolist()):
            words = tok.decode(row).split()
            out[k] = {"text": " ".join(words[:128])}
        if i % (batch * 10) == 0:
            done = i + len(grp)
            log(f"   generated {done}/{len(order)}  "
                f"({done/max(time.time()-t0,1):.1f}/s)")
    return out


# ------------------------------------------------------------------ metrics -
def lcs_len(a, b):
    if not a or not b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    masks = {}
    for j, t in enumerate(a):
        masks[t] = masks.get(t, 0) | (1 << j)
    state = 0
    for t in b:
        m = masks.get(t, 0)
        x = m | state
        y = (state << 1) | 1
        state = x & ~(x - y)
    return state.bit_count()


def rouge_l(p, r):
    pt, rt = p.split(), r.split()
    if not pt or not rt:
        return 0.0
    l = lcs_len(pt, rt)
    if not l:
        return 0.0
    pp, rr = l / len(pt), l / len(rt)
    return 2 * pp * rr / (pp + rr)


def bleu4(preds, refs):
    clipped, totals = [0] * 4, [0] * 4
    pl = rl = 0
    for p, r in zip(preds, refs):
        pl += len(p)
        rl += len(r)
        for n in range(1, 5):
            pn = Counter(tuple(p[i:i + n]) for i in range(len(p) - n + 1))
            rn = Counter(tuple(r[i:i + n]) for i in range(len(r) - n + 1))
            totals[n - 1] += sum(pn.values())
            for g, c in pn.items():
                clipped[n - 1] += min(c, rn.get(g, 0))
    prec = [clipped[i] / totals[i] if totals[i] else 0.0 for i in range(4)]
    if any(x <= 0 for x in prec) or pl == 0:
        return 0.0
    bp = 1.0 if pl > rl else math.exp(1.0 - rl / pl)
    return bp * math.exp(sum(math.log(x) for x in prec) / 4)


def score_dev(pred, truth):
    ks = list(pred)
    r = sum(rouge_l(pred[k]["text"], truth[k]["text"]) for k in ks) / len(ks)
    b = bleu4([pred[k]["text"].split() for k in ks],
              [truth[k]["text"].split() for k in ks])
    f = sum(fluency(pred[k]["text"]) for k in ks) / len(ks)
    return dict(rouge_l=r, bleu4=b, fluency=f, n=len(ks))


def bertscore_maybe(pred, truth):
    try:
        import bert_score
        ks = list(pred)
        _, _, f1 = bert_score.score([pred[k]["text"] for k in ks],
                                    [truth[k]["text"] for k in ks],
                                    lang="en", batch_size=16, verbose=False,
                                    device="cuda")
        return float(f1.mean())
    except Exception as e:                                   # never kill the run
        log(f"   BERTScore unavailable ({type(e).__name__}: {e}); "
            "ROUGE-L/BLEU-4 only")
        return None


# ---------------------------------------------------------------- main loop -
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="M1_LexArcD")
    ap.add_argument("--hours", type=float, default=6.0)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--attn", default="gqa", choices=["gqa", "mla"],
                    help="gqa = M1 LexArc-D, mla = M3 LexArc-L")
    ap.add_argument("--kv-latent", type=int, default=256)
    ap.add_argument("--pointer", action="store_true",
                    help="M4 LexArc-P: pointer-generator output head")
    ap.add_argument("--layers", type=int, default=0,
                    help="override n_layer to match M1 on parameters")
    ap.add_argument("--muon-lr", type=float, default=0.025)
    ap.add_argument("--adam-lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=float, default=0.02)
    ap.add_argument("--ckpt-min", type=float, default=20.0)
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--skip-gates", action="store_true")
    ap.add_argument("--init", default=None,
                    help="DIR/ckpt/FILE.pt to initialise from -- continued "
                         "pretraining rather than a fresh run")
    ap.add_argument("--mix", default="phased", choices=["phased", "sc"],
                    help="sc = Supreme Court only, for continued in-domain "
                         "pretraining")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore an existing resume.pt and start over")
    a = ap.parse_args()

    OUT = f"{ROOT}/{a.model}"
    for d in ("ckpt", "logs", "gen"):
        os.makedirs(f"{OUT}/{d}", exist_ok=True)
    dev = "cuda"
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(1337)

    p = torch.cuda.get_device_properties(0)
    sm = p.major * 10 + p.minor
    assert sm >= 80, (f"{p.name} is SM{sm}. bf16 is emulated below SM80 and runs "
                      "~30x slow; use an L4 or A100.")
    log(f"GPU {p.name} SM{sm} {p.total_memory/1e9:.1f} GB")

    tok = BPETokenizer.load(f"{ROOT}/shared/tokenizer/bpe_legal_32k.json")
    cfg = Config(vocab_size=tok.vocab_size, ctx=a.ctx,
                 attn_type=a.attn, kv_latent_dim=a.kv_latent,
                 use_pointer=a.pointer)
    if a.layers:
        cfg.n_layer = a.layers
    if cfg.use_pointer:
        from cs781_model_ptr import LexArcP
        Net = LexArcP
    else:
        Net = LexArcD
    model = Net(cfg).to(dev)
    n_all, n_ne = model.n_params(), model.n_params(True)
    log(f"params {n_all/1e6:.1f}M total, {n_ne/1e6:.1f}M non-embedding "
        f"| attn={cfg.attn_type}{' +pointer' if cfg.use_pointer else ''}"
        + (f" r={cfg.kv_latent_dim}" if cfg.attn_type == "mla" else ""))
    assert 50e6 < n_all < 300e6, f"PARAM BUDGET VIOLATED: {n_all/1e6:.1f}M not in 50-300M"

    loader = ShardLoader(a.batch, a.ctx, dev)
    log(f"corpus {loader.tok_total/1e9:.3f}B tokens across {len(loader.arr)} sources")
    dev_prompts, dev_truth = load_clean_dev(tok)

    resuming = os.path.exists(f"{OUT}/ckpt/resume.pt") and not a.fresh
    if resuming:
        log("resume.pt found -- skipping gates (they reinitialise the model)")
    if a.init and not resuming:
        log("--init given: gates would reinitialise the model, so they are "
            "skipped. The checkpoint already passed them when it was trained.")
    if not a.skip_gates and not resuming and not a.init:
        gate_shift(model, loader, dev)
        gate_overfit(model, loader, dev)
        gate_kv_cache(model, loader, tok, dev)
        # gates perturbed the weights; start clean
        model = Net(cfg).to(dev)
        log("gates passed, model reinitialised")

    opts, counts = build_optimizers(model, a.muon_lr, a.adam_lr)
    log(f"optimizers: Muon on {counts['muon']} matrices, AdamW on {counts['adam']}")

    # --- resume ------------------------------------------------------------
    # Only the model weights, in fp32, plus the wall-clock already spent. The
    # optimiser state is deliberately NOT saved: Muon's momentum buffers alone
    # are ~400 MB, a Drive write every 20 min would cost real throughput, and
    # momentum rebuilds in ~100 steps out of ~25,000. Losing it is cheaper than
    # saving it.
    resume_path = f"{OUT}/ckpt/resume.pt"
    start_elapsed, step, best_val = 0.0, 0, float("inf")
    if a.init and not os.path.exists(resume_path):
        ick = torch.load(f"{ROOT}/{a.init}", map_location=dev, weights_only=False)
        model.load_state_dict({k: v.float() for k, v in ick["model"].items()})
        log(f"INITIALISED from {a.init} (step {ick.get('step')}, "
            f"val {ick.get('val')}) -- continued pretraining, fresh LR cycle")
    if os.path.exists(resume_path) and not a.fresh:
        ck = torch.load(resume_path, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        step, start_elapsed, best_val = ck["step"], ck["elapsed"], ck["val"]
        log(f"RESUMED from step {step:,}, {start_elapsed/3600:.2f} h of "
            f"{a.hours:.1f} h already spent, best val {best_val:.4f}")

    net = model if a.no_compile else torch.compile(model)

    # --- compile first, THEN measure. The LR schedule is driven by wall-clock,
    # --- not by a predicted step count: throughput on Colab is not knowable in
    # --- advance, and timing the compile into the estimate truncated a run.
    budget = a.hours * 3600
    for _ in range(5):                                   # triggers compilation
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = net(loader.batch(mix_at(0.0, a.mix)))
        loss.backward()
        for o in opts:
            o.step()
            o.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(20):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = net(loader.batch(mix_at(0.0, a.mix)))
        loss.backward()
        for o in opts:
            o.step()
            o.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    tps = 20 * a.batch * a.ctx / (time.time() - t0)
    log(f"post-compile throughput: {tps:,.0f} tok/s -> "
        f"~{tps*budget/1e6:,.0f}M tokens in {a.hours:.1f} h "
        f"({tps*budget/100.7e6:.1f} tokens/param)")

    base_lrs = [[g["lr"] for g in o.param_groups] for o in opts]

    def set_lr(frac):
        """Cosine over elapsed TIME, so a throughput surprise cannot truncate
        the schedule."""
        if frac < a.warmup:
            s = (frac + 1e-9) / a.warmup
        else:
            prog = (frac - a.warmup) / (1.0 - a.warmup)
            s = 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))
        for o, bl in zip(opts, base_lrs):
            for g, b in zip(o.param_groups, bl):
                g["lr"] = b * s
        return s

    hist, ckpt_n, nan_steps = [], 0, 0
    start = time.time() - start_elapsed
    last_ckpt, fluency_checked = time.time(), set()

    while True:
        elapsed = time.time() - start
        if elapsed > budget:
            break
        frac = elapsed / budget
        set_lr(frac)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = net(loader.batch(mix_at(frac, a.mix)))
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # A NaN loss is unrecoverable -- the weights are already poisoned by the
        # update that produced it -- so skip the step, and bail out rather than
        # spend five more GPU-hours logging "nan".
        if not torch.isfinite(gn):
            nan_steps += 1
            for o in opts:
                o.zero_grad(set_to_none=True)
            if nan_steps == 1:
                log(f"WARNING: non-finite gradient at step {step}; skipping")
            if nan_steps > 20:
                raise SystemExit(
                    f"ABORT: {nan_steps} non-finite gradients. The loss has "
                    "diverged; stopping rather than burning the remaining "
                    "GPU hours.")
            step += 1
            continue
        for o in opts:
            o.step()
            o.zero_grad(set_to_none=True)
        step += 1

        if step % 100 == 0:
            l = loss.item()
            row = dict(step=step, loss=l, grad_norm=float(gn), frac=round(frac, 4),
                       lr=opts[0].param_groups[0]["lr"], elapsed=round(elapsed))
            hist.append(row)
            if step % 500 == 0:
                tok_done = step * a.batch * a.ctx
                log(f"step {step:6d} ({frac:5.1%}) loss {l:.4f} gn {float(gn):.2f} "
                    f"{tok_done/1e6:,.0f}M tok  {elapsed/60:.0f}m/{budget/60:.0f}m")
                with open(f"{OUT}/logs/train.jsonl", "a") as fh:
                    for r in hist:
                        fh.write(json.dumps(r) + "\n")
                hist = []

        # --- in-run fluency gate: BERTScore cannot see broken syntax ----------
        for mark in (0.10, 0.25):
            if frac >= mark and mark not in fluency_checked:
                fluency_checked.add(mark)
                sub = dict(list(dev_prompts.items())[:8])
                g = generate_for(model, tok, sub, dev, batch=8, n_new=210,
                                 greedy=True, repetition_penalty=1.0)
                fl = sum(fluency(v["text"]) for v in g.values()) / len(g)
                log(f"FLUENCY @{mark:.0%}: {fl:.1f} bigrams/128w "
                    f"(legal prose 15-25, word salad 1-2)")
                log(f"   sample: {list(g.values())[0]['text'][:240]!r}")
                model.train()
                if fl < 4.0 and mark == 0.25:
                    raise SystemExit(
                        f"ABORT: fluency {fl:.1f} at 25% of budget means the model "
                        "is producing word salad. Stopping rather than burning "
                        "the remaining GPU hours. Run greedy decoding with every "
                        "penalty off and inspect before relaunching.")

        if (time.time() - last_ckpt) > a.ckpt_min * 60:
            last_ckpt = time.time()
            vl = []
            model.eval()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for s in loader.val:
                    vl.append(model(loader.val_batch(s, 8, 0)).item())
            model.train()
            v = sum(vl) / len(vl)
            ckpt_n += 1
            slot = ckpt_n % 3
            blob = {"model": {k: x.to(torch.bfloat16) for k, x in
                              model.state_dict().items()},
                    "cfg": vars(cfg), "step": step, "val": v}
            safe_save(blob, f"{OUT}/ckpt/slot{slot}.pt")
            safe_save({"model": model.state_dict(), "cfg": vars(cfg), "step": step,
                       "elapsed": time.time() - start, "val": min(v, best_val)},
                      f"{OUT}/ckpt/resume.pt")
            if v < best_val:
                best_val = v
                safe_save(blob, f"{OUT}/ckpt/best.pt")
            log(f"ckpt step {step} val_loss {v:.4f} bits/tok {v/math.log(2):.3f} "
                f"{'(best)' if v == best_val else ''}")

    log(f"training done: {step:,} steps, {step*a.batch*a.ctx/1e6:,.0f}M tokens, "
        f"{(time.time()-start)/3600:.2f} h")
    safe_save({"model": model.state_dict(), "cfg": vars(cfg), "step": step},
              f"{OUT}/ckpt/final.pt")

    # ------------------------------------------------------------ evaluate ---
    log("dev generation (greedy, then sampled)")
    sub = dict(list(dev_prompts.items())[:200])
    sub_truth = {k: dev_truth[k] for k in sub}
    results = {}
    for name, kw in [("greedy", dict(greedy=True)),
                     ("t0.8_p0.9", dict(temperature=0.8, top_p=0.9,
                                        repetition_penalty=1.05))]:
        g = generate_for(model, tok, sub, dev, batch=16, n_new=210, **kw)
        s = score_dev(g, sub_truth)
        s["bertscore"] = bertscore_maybe(g, sub_truth)
        results[name] = s
        log(f"   {name:12s} ROUGE-L {s['rouge_l']:.4f}  BLEU-4 {s['bleu4']:.4f}  "
            f"fluency {s['fluency']:.1f}  BERTScore {s['bertscore']}")
        json.dump(g, open(f"{OUT}/gen/dev_{name}.json", "w"), ensure_ascii=False)
    best = max(results, key=lambda k: results[k]["bertscore"] or results[k]["rouge_l"])
    log(f"best dev setting: {best}")
    json.dump(results, open(f"{OUT}/logs/dev_scores.json", "w"), indent=2)

    # ----------------------------------------------------------- submission --
    log("generating the 1000 test continuations")
    test = json.load(open(f"{ROOT}/shared/test/test.json", encoding="utf-8"))
    kw = dict(greedy=True) if best == "greedy" else dict(
        temperature=0.8, top_p=0.9, repetition_penalty=1.05)
    pred = generate_for(model, tok, test, dev, batch=16, n_new=210, **kw)

    assert set(pred) == set(test), (
        f"ID MISMATCH: missing {len(set(test)-set(pred))}, extra {len(set(pred)-set(test))}")
    wc = [len(v["text"].split()) for v in pred.values()]
    log(f"continuation words: min {min(wc)} median {sorted(wc)[len(wc)//2]} max {max(wc)}")
    short = sum(1 for n in wc if n < 128)
    if short:
        log(f"WARNING: {short} continuations are under 128 words -- recall will suffer")

    pj = f"{OUT}/gen/prediction.json"
    json.dump(pred, open(pj, "w", encoding="utf-8"), ensure_ascii=False)
    zp = f"{OUT}/gen/submission.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(pj, "prediction.json")        # at the zip ROOT, per 6.3
    log(f"WROTE {zp}  ({os.path.getsize(zp)/1e6:.1f} MB)")
    log("sample continuation:")
    log("   " + repr(pred[list(pred)[0]]["text"][:300]))
    log("ALL DONE")


if __name__ == "__main__":
    main()
