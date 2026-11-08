#!/usr/bin/env python
"""CS-781 Phase 7, part Z round 2 -- widen the decoder grid and reach the stacks.

    python phase7_ot2.py all [--minutes 40]

Round 1 (phase7_ot.py) measured: transductive decoding gains +146..+182 train hits and
+16..+31 devx hits on every base, but (a) the chosen tau=1.0 / lam=0.5 sat on the EDGE of
the grid, (b) the stacks save only devx/test, so they were never decoded, and (c) train
queries are cited by other train queries (616/827) while devx queries almost never are
(15/118), so `dropq` loses on train and is near-free on devx.

Round 2:
  1. a wider grid (tau up to 3, lam down to 0.25, rho 0.4..0.8, depth 25..all), z only;
  2. ONE pooled config = best mean normalised train gain across every base that has
     train scores (lexical members and their z-mixes) -- chosen without devx;
  3. that config applied unchanged to every base, including the stacks and a z-mix that
     contains the stacks (devx/test only) -- devx is then a clean out-of-sample check;
  4. the same config with dropq, written as a separate zip because only devx supports it.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import phase7_data as D   # noqa: E402
import phase7_rank as R   # noqa: E402
import phase7_ot as O     # noqa: E402

P7 = D.P7
log = O.log

TRAIN_BASES = {
    "citep": ["citep"],
    "maxp2": ["maxp2"],
    "zlex2": ["citep", "maxp2"],
    "zlex4": ["citep", "maxp2", "maxp_tfidf", "rm3_tfidf"],
    "zlex6": ["citep", "maxp2", "maxp_tfidf", "rm3_tfidf", "cite", "tfidf_sqrt"],
}
EVAL_BASES = {          # devx/test only (stacks never save train)
    "stack_new": ["stack_new"],
    "stack_ce": ["stack_ce"],
    "zst3": ["stack_new", "citep", "maxp2"],
    "zst5": ["stack_new", "stack_ce", "stack_lex2", "citep", "maxp2"],
    "zst7": ["stack_new", "stack_ce", "stack_lex2", "stack_ce_lex", "citep", "maxp2",
             "maxp_tfidf"],
}


def has(tags, split):
    return all(os.path.exists(f"{R.rank_dir(t)}/{split}_rank.json") for t in tags)


def lists_for(tags, split):
    if not has(tags, split):
        return None
    if len(tags) == 1:
        return O.load_lists(tags[0], split)
    return O.zmix(tags, split)


def grid2(dropq=False):
    out = []
    for depth in (25, 40, 80, 0):
        for tau in (0.8, 1.0, 1.4, 2.0, 3.0):
            for rho in (0.4, 0.6, 0.8):
                out.append(dict(mode="sink", tr="z", depth=depth, dropq=dropq, tau=tau, rho=rho))
            for lam in (0.25, 0.35, 0.5, 0.7):
                out.append(dict(mode="dsm", tr="z", depth=depth, dropq=dropq, tau=tau, lam=lam))
    return out


NONE = dict(mode="none", tr="z", depth=0, dropq=False)


def hits(run, cfg, split):
    rk, _ = O.to_rank(run, O.decode_scores(run, cfg))
    return O.perq_hits(rk, split)


def paired(h, b):
    d = h - b
    return int(d.sum()), round(2 * d.std(ddof=1) * math.sqrt(len(d)), 1)


def cmd_all(a):
    t0 = time.time()
    deadline = t0 + a.minutes * 60
    G = grid2()
    keys = [O.name(c) for c in G]
    # ---------------------------------------------- 1. pooled train selection
    gains = {}
    runs = {}
    for b, tags in TRAIN_BASES.items():
        if time.time() > deadline:
            break
        if not all(has(tags, s) for s in ("train", "devx", "test")):
            log(f"  [{b}] members missing a split -- skipped")
            continue
        rr = {s: O.Run(s, lists_for(tags, s)) for s in ("train", "devx", "test")}
        runs[b] = rr
        base = hits(rr["train"], NONE, "train")
        g = np.array([hits(rr["train"], c, "train").sum() - base.sum() for c in G])
        gains[b] = g
        bd = hits(rr["devx"], NONE, "devx")
        i = int(np.argmax(g))
        log(f"  [{b}] base train {int(base.sum())} devx {int(bd.sum())} | best train "
            f"{keys[i]} {g[i]:+.0f}  ({(time.time() - t0) / 60:.1f} min)")
    if not gains:
        raise SystemExit("no base with train scores")
    norm = np.mean([g / max(g.max(), 1.0) for g in gains.values()], axis=0)
    order = np.argsort(-norm)
    log("\n  POOLED train ranking (mean normalised gain over bases):")
    for i in order[:10]:
        log(f"     {keys[i]:34s} {norm[i]:.3f}   " +
            "  ".join(f"{b} {gains[b][i]:+.0f}" for b in gains))
    best = G[int(order[0])]
    best_dq = dict(best, dropq=True)
    log(f"  CHOSE {O.name(best)}  (and its dropq twin) -- selected on train only")

    # ---------------------------------------------- 2. apply to every base, devx check
    allb = dict(TRAIN_BASES)
    allb.update(EVAL_BASES)
    table = []
    for b, tags in allb.items():
        if not all(has(tags, s) for s in ("devx", "test")):
            log(f"  [{b}] missing devx/test -- skipped")
            continue
        rr = runs.get(b) or {s: O.Run(s, lists_for(tags, s)) for s in ("devx", "test")}
        bd = hits(rr["devx"], NONE, "devx")
        for lab, cfg in (("p", best), ("pdq", best_dq)):
            hd = hits(rr["devx"], cfg, "devx")
            g, se = paired(hd, bd)
            f1 = R.evaluate(O.to_rank(rr["devx"], O.decode_scores(rr["devx"], cfg))[0],
                            "devx")["f1"]
            table.append(dict(base=b, lab=lab, cfg=cfg, devx=int(hd.sum()), base_devx=int(bd.sum()),
                              gain=g, se=se, f1=f1))
            log(f"  [{b:9s}] {lab:3s} devx {int(hd.sum())} (base {int(bd.sum())}, {g:+d}, 2se {se})"
                f"  F1 {f1:.4f}")
        runs[b] = rr
    table.sort(key=lambda r: r["f1"], reverse=True)

    # ---------------------------------------------- 3. zips: the top 8 by devx
    made = []
    for r in table[:a.nzip]:
        tag = f"ot2_{r['base']}_{r['lab']}"
        rr = runs[r["base"]]
        for s, run in rr.items():
            rk, sc = O.to_rank(run, O.decode_scores(run, r["cfg"]))
            R.save_run(tag, s, rk, sc)
        R.save_run(tag, "meta", None, meta={
            "tag": tag, "kind": "hybrid", "base": r["base"], "members": allb[r["base"]],
            "config": r["cfg"], "technique": "transductive decoding, pooled train config",
            "devx_scores": {"f1": r["f1"], "hits": r["devx"]}})
        R.cmd_zip(argparse.Namespace(tag=tag, kind="hybrid", note=json.dumps(r["cfg"])))
        made.append(r)
    lines = ["# Part Z round 2 -- upload order", "",
             f"Config chosen on TRAIN only (pooled over {len(gains)} bases): `{O.name(best)}`.",
             "`pdq` = same config + drop candidates that are queries (supported by devx only).",
             "Round-1 best for reference: ot_zmix_trdv devx 0.4636 (417 hits).", "",
             "| # | zip | devx F1 | devx hits | base hits | gain (2SE) |",
             "|---|---|---|---|---|---|"]
    for i, r in enumerate(made, 1):
        lines.append(f"| {i} | submission_ot2_{r['base']}_{r['lab']}.zip | {r['f1']:.4f} | "
                     f"{r['devx']} | {r['base_devx']} | {r['gain']:+d} ({r['se']}) |")
    txt = "\n".join(lines) + "\n"
    with open(f"{P7}/SUBMIT_ORDER_Z2.md", "w") as f:
        f.write(txt)
    D.atomic_json(f"{P7}/results/ot2_table.json", {"chose": best, "table": table})
    log("\n" + txt + f"\nDONE in {(time.time() - t0) / 60:.1f} min")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("all")
    s.add_argument("--minutes", type=float, default=40)
    s.add_argument("--nzip", type=int, default=8)
    a = p.parse_args()
    cmd_all(a)


if __name__ == "__main__":
    main()
