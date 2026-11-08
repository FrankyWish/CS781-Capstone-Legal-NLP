#!/usr/bin/env python3
"""CS-781 Phase 5 -- tokenise every document once, on a CPU runtime.

    python phase5_tok.py                 # -> $P5/cache/tok_{train,dev,test}.npz
    python phase5_tok.py --max-tokens 8192

Each .npz holds `flat` (uint16 token ids, all documents concatenated), `off`
(int64 offsets, len N+1) and `ids` (document ids in file order). Documents are
cut at --max-tokens: head and tail are kept (3/4 head, 1/4 tail), because the
statute-bearing text of a judgment sits in the facts at the start and in the
holding at the end. The trainer reads these files and never tokenises on GPU time.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from phase5_data import P1, P5, load, log  # noqa: E402

TOK = None
MAXT = 8192


def _init(path, maxt):
    global TOK, MAXT
    lib = os.environ.get("CS781_LIB", f"{P1}/shared/lib")
    sys.path.insert(0, lib)
    from cs781_bpe import BPETokenizer
    TOK = BPETokenizer.load(path)
    MAXT = maxt


def _enc(text):
    ids = TOK.encode(text)
    if len(ids) > MAXT:
        head = MAXT * 3 // 4
        ids = ids[:head] + ids[-(MAXT - head):]
    return np.asarray(ids, dtype=np.uint16), len(ids)


def tok_path(split):
    return f"{P5}/cache/tok_{split}.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--tokenizer", default=os.environ.get("CS781_TOKENIZER",
                                                          f"{P1}/shared/tokenizer/bpe_legal_32k.json"))
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    ap.add_argument("--splits", default="train,dev,test,silver")
    a = ap.parse_args()
    os.makedirs(f"{P5}/cache", exist_ok=True)
    for split in a.splits.split(","):
        out = tok_path(split)
        if split == "silver" and not os.path.exists(f"{P5}/external/silver.jsonl"):
            log("no external/silver.jsonl yet -- silver not tokenised")
            continue
        if os.path.exists(out):
            log(f"{out} exists, skipping")
            continue
        recs = load(split)
        t0 = time.time()
        with Pool(a.procs, initializer=_init, initargs=(a.tokenizer, a.max_tokens)) as pool:
            res = pool.map(_enc, [r["text"] for r in recs], chunksize=64)
        lens = np.array([n for _, n in res])
        off = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
        flat = np.concatenate([x for x, _ in res]) if res else np.zeros(0, np.uint16)
        tmp = out + ".tmp.npz"
        np.savez(tmp, flat=flat, off=off, ids=np.array([r["id"] for r in recs]))
        os.replace(tmp, out)
        log(f"{split}: {len(recs)} docs, {off[-1]/1e6:.1f}M tokens, "
            f"tokens/doc p50 {np.median(lens):.0f} p90 {np.percentile(lens, 90):.0f} "
            f"max {lens.max()} ({time.time()-t0:.0f}s) -> {out}")


if __name__ == "__main__":
    main()
