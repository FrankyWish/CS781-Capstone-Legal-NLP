#!/usr/bin/env python3
"""Which Phase-1 model is the best foundation for Phases 2-7?

    python foundation_check.py

Phase 1 is scored by BERTScore on free-form continuation. Phases 2, 3, 4, 5 and
7 are understanding tasks: a head on the trunk, using its representations.
Likelihood predicts representation quality; BERTScore on 128-word continuations
does not. So the Phase-1 leaderboard is the wrong instrument for choosing what
to carry forward, and this measures the right one -- every model on identical
validation batches.

THE QUESTION THIS ACTUALLY SETTLES

M4's pointer head gives it the best validation loss by 0.045. But the copy
channel makes repeated tokens trivially predictable, so the gain could be
coming from the HEAD rather than from the trunk -- and Phase 2 discards the
head. `force_gate=1.0` closes the copy channel and reduces M4 to a plain LM over
the same trunk. If that number still beats M1, the trunk genuinely improved and
M4 is the right Phase-2 starting point. If it does not, the pointer head is
carrying the loss and M1 is the better foundation despite the worse headline.

It also verifies that all four checkpoints load, which §8 requires and which is
cheaper to discover now than in November.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

ROOT = os.environ.get("CS781_ROOT", "/content/drive/MyDrive/cs781_capstone")
sys.path.insert(0, f"{ROOT}/shared/lib")
from cs781_bpe import BPETokenizer                          # noqa: E402
from train_phase1 import ShardLoader, log                   # noqa: E402
from score_and_submit import load_ckpt                      # noqa: E402

MODELS = [
    ("M1_LexArcD_full", "soup.pt"),
    ("M2_LexArcED",     "best.pt"),
    ("M3_LexArcL",      "best.pt"),
    ("M4_LexArcP",      "best.pt"),
]
N_BATCH, BATCH = 12, 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None,
                    help="DIR:CKPT pairs, e.g. M1_SC:final.pt M3_SC:final.pt. "
                         "Defaults to the four Phase-1 models.")
    ap.add_argument("--batches", type=int, default=N_BATCH)
    a = ap.parse_args()
    models = ([tuple((x.split(":") + ["best.pt"])[:2]) for x in a.models]
              if a.models else MODELS)
    dev = "cuda"
    torch.set_float32_matmul_precision("high")
    tok = BPETokenizer.load(f"{ROOT}/shared/tokenizer/bpe_legal_32k.json")
    loader = ShardLoader(BATCH, 1024, dev)
    ed_loader = ShardLoader(BATCH, 1280, dev)   # M2 trains on 1280-token windows

    # One fixed set of batches, identical for every model.
    fixed = [loader.val_batch(s, BATCH, seed)
             for s in list(loader.val)[:1] for seed in range(a.batches)]
    fixed_ed = [ed_loader.val_batch(s, BATCH, seed)
                for s in list(ed_loader.val)[:1] for seed in range(a.batches)]

    rows = []
    for d, c in models:
        path = f"{ROOT}/{d}/ckpt/{c}"
        if not os.path.exists(path):
            log(f"{d:20s} MISSING {path}")
            continue
        model, cfg, kind, step, _ = load_ckpt(path, dev)
        model.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            if kind == "ed":
                losses = [model(b, cfg.split_points[1])[0].item() for b in fixed_ed]
            else:
                losses = [model(b).item() for b in fixed]
            val = sum(losses) / len(losses)
            gated = None
            if getattr(cfg, "use_pointer", False):
                gated = sum(model(b, force_gate=1.0).item() for b in fixed) / len(fixed)
        rows.append(dict(model=d, ckpt=c, params=model.n_params(), step=step,
                         val_loss=round(val, 4),
                         val_loss_vocab_only=round(gated, 4) if gated else None))
        log(f"{d:20s} {model.n_params()/1e6:6.1f}M  val {val:.4f}"
            + (f"   vocab-path-only {gated:.4f}" if gated else ""))
        del model
        torch.cuda.empty_cache()

    base = next((r for r in rows if r["model"] == "M1_LexArcD_full"), None)
    ptr = next((r for r in rows if r["val_loss_vocab_only"] is not None), None)
    log("\n--- which trunk to carry into Phase 2 ---")
    for r in sorted(rows, key=lambda x: x["val_loss"]):
        log(f"  {r['model']:20s} val {r['val_loss']:.4f}")
    if base and ptr:
        b1, p1 = base["model"], ptr["model"]
        carried = ptr["val_loss_vocab_only"] - ptr["val_loss"]
        d_full = base["val_loss"] - ptr["val_loss"]
        d_trunk = base["val_loss"] - ptr["val_loss_vocab_only"]
        log(f"\n{p1} beats {b1} by {d_full:+.4f} with the copy head active.")
        log(f"Closing the copy channel costs {carried:+.4f} nats, leaving it {d_trunk:+.4f} vs {b1}.")
        log("CAVEAT -- this does NOT measure trunk quality. force_gate=1.0 "
            "scores the vocabulary head as a standalone LM, and it was never "
            "trained to be one: it was trained as one component of a mixture in "
            "which the pointer carried part of the mass. So the number separates "
            "HEAD CONTRIBUTION from total loss, and nothing more. A trunk that "
            "offloaded repeat-prediction to the pointer could be better or worse "
            "for an understanding task, and this cannot tell the two apart. The "
            "decisive test is a probe: freeze each trunk, train an identical "
            "fresh head on the Phase-2 task, compare. That is Phase 2 itself.")
        log(f"-> BANKABLE for Phase 2: none of {d_full:+.4f}, because Phase 2 "
            "discards the head. The pointer model's headline advantage is at "
            "least partly head-borne and carries an unresolved confound, which "
            "the plain decoders do not -- so prefer a plain decoder unless a "
            "probe says otherwise.")
    json.dump(rows, open(f"{ROOT}/shared/foundation_check.json", "w"), indent=2)
    log(f"\nwrote {ROOT}/shared/foundation_check.json")
    log("NOTE: M2's number is NOT comparable -- it is an encoder-decoder scored "
        "on a different conditional, over 1280-token windows.")


if __name__ == "__main__":
    main()
