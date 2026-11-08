#!/usr/bin/env python
"""CS-781 Phase 7, part Z -- transductive decoding: rank all test queries JOINTLY.

    python phase7_ot.py all                      diagnose + sweep + zips + order (CPU, ~20-40 min)
    python phase7_ot.py diag                     the structure numbers only (1 min)
    python phase7_ot.py all --bases stack_new,citep

WHY.  Every system in this phase ranks each query on its own.  But IL-PCR builds a split's
candidate pool out of the cases its queries cite, so the pool has a global structure no
per-query ranker sees: Sigma|gold| is about the pool size (train 5,309 golds / 4,320
candidates, dev 973 / 1,023), i.e. almost every candidate is the gold of about ONE query.
A per-query ranker spends many queries' top-7 slots on the same handful of generic "hub"
judgments, while a large share of the pool is never predicted by anyone.  Decoding all
queries together under "every candidate gets about its share of citations" is free
information -- it uses no labels, only the shape of the test pool itself.

`diag` measures that structure on train and devx (in-degree histogram, the share of the
pool that is someone's gold, whether a query sitting in the pool is ever a gold) so the
premise is checked, not assumed.

DECODERS (all reduce to   score(q,c) = s(q,c)/tau + b_c,   a learned per-candidate bias):
  sink   entropic optimal transport (Sinkhorn): row sums 7, column sums = each candidate's
         share of the 7Q slots; rho < 1 relaxes the column constraint (unbalanced OT), so a
         candidate liked strongly by two queries can still take both.
  dsm    dual softmax: row softmax x column softmax^lam (mutual nearest neighbour).
  dropq  candidates that are themselves queries of the split removed (measured on train).
Score transform: `z` = per-query z-score of the saved scores, `rk` = -log(1+rank).

SELECTION.  On the 827 train queries (stack runs save cross-fitted OOF train scores, the
lexical runs learn nothing from labels, so train is out of sample), with the devx number
and a paired per-query 2-SE check printed beside it.  Test only ever gets the chosen config.
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

P7 = D.P7
K = D.K
DEFAULT_BASES = "stack_new,stack_ce,citep,stack_i,rrf_rank,maxp2,stack_lex2,ptr"
MIX = ("stack_new", "stack_ce", "stack_lex2", "citep", "maxp2")


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------ matrices --
class Run:
    """A saved run as dense-per-row arrays: V (Q,d) scores, J (Q,d) column ids, -inf pad."""

    def __init__(self, split, lists):
        self.split = split
        self.qs = [r["id"] for r in D.queries(split)]
        self.pool = [str(c) for c in D.candidates(split)]
        self.cidx = {c: i for i, c in enumerate(self.pool)}
        qset = set(self.qs)
        self.isq = np.array([c in qset for c in self.pool])
        d = max([len(lists.get(q, ([], None))[0]) for q in self.qs] + [1])
        Q = len(self.qs)
        self.V = np.full((Q, d), -np.inf)
        self.J = np.zeros((Q, d), dtype=np.int64)
        self.RK = np.full((Q, d), -np.inf)
        drop = R.self_excluded(split)
        for i, q in enumerate(self.qs):
            ids, v = lists.get(q, ([], None))
            ids = [str(c) for c in ids]
            if v is None or len(v) != len(ids):
                v = -np.arange(len(ids), dtype=np.float64)
            v = np.asarray(v, dtype=np.float64)
            ok = np.isfinite(v)
            if ok.any():
                v = np.where(ok, v, v[ok].min())
            sd = v.std() if len(v) else 1.0
            z = (v - v.mean()) / (sd if sd > 1e-12 else 1.0) if len(v) else v
            seen, col = set(), 0
            for r, (c, zz) in enumerate(zip(ids, z)):
                j = self.cidx.get(c)
                if j is None or c in seen or c == q or drop.get(q) == c:
                    continue
                seen.add(c)
                self.V[i, col], self.J[i, col] = zz, j
                self.RK[i, col] = -math.log1p(col)
                col += 1
        self.C = len(self.pool)


def load_lists(tag, split):
    rank, score = R.load_run(tag, split)
    if rank is None:
        return None
    return {q: (ids, (score or {}).get(q)) for q, ids in rank.items()}


def zmix(tags, split):
    """Mean per-query z-score over the members (missing entry = that member's row min)."""
    mem = [m for m in (load_lists(t, split) for t in tags) if m is not None]
    if len(mem) < 2:
        return None
    out = {}
    for q in set().union(*[set(m) for m in mem]):
        acc = {}
        rows = []
        for m in mem:
            ids, v = m.get(q, ([], None))
            ids = [str(c) for c in ids]
            if v is None or len(v) != len(ids):
                v = -np.arange(len(ids), dtype=np.float64)
            v = np.asarray(v, dtype=np.float64)
            if len(v) == 0:
                continue
            v = np.where(np.isfinite(v), v, np.nanmin(np.where(np.isfinite(v), v, np.nan)))
            sd = v.std()
            z = (v - v.mean()) / (sd if sd > 1e-12 else 1.0)
            rows.append((dict(zip(ids, z)), float(z.min())))
        for c in set().union(*[set(r[0]) for r in rows]):
            acc[c] = float(np.mean([r[0].get(c, r[1]) for r in rows]))
        top = sorted(acc, key=acc.get, reverse=True)[:200]
        out[q] = (top, np.array([acc[c] for c in top]))
    return out


# ------------------------------------------------------------------ decoders --
def col_lse(X, J, C):
    """log-sum-exp of each column's finite entries (sparse rows, -inf = absent)."""
    flat, jj = X.ravel(), J.ravel()
    ok = np.isfinite(flat)
    flat, jj = flat[ok], jj[ok]
    out = np.full(C, -np.inf)
    if not len(flat):
        return out
    M = flat.max()
    s = np.bincount(jj, weights=np.exp(flat - M), minlength=C)
    a = s > 0
    out[a] = M + np.log(s[a])
    lost = np.zeros(C, dtype=bool)
    lost[jj] = True
    lost &= ~a                                   # underflowed: redo those exactly
    if lost.any():
        mx = np.full(C, -np.inf)
        np.maximum.at(mx, jj, flat)
        s2 = np.zeros(C)
        np.add.at(s2, jj, np.exp(flat - mx[jj]))
        out[lost] = mx[lost] + np.log(s2[lost])
    return out


def row_lse(X):
    mx = np.max(X, axis=1, keepdims=True)
    mx = np.where(np.isfinite(mx), mx, 0.0)
    with np.errstate(divide="ignore"):
        return (mx + np.log(np.exp(X - mx).sum(axis=1, keepdims=True)))[:, 0]


def prep(run, transform, depth, dropq):
    X = (run.V if transform == "z" else run.RK).copy()
    if depth and depth < X.shape[1]:
        X[:, depth:] = -np.inf
    if dropq:
        X[run.isq[run.J]] = -np.inf
    return X


def sinkhorn(X, J, C, tau, rho, isq=None, iters=150, k=K):
    """-> per-column bias b (log v).  Rows sum to k exactly; columns to uniform mass."""
    L = X / tau
    active = np.zeros(C, dtype=bool)
    active[J[np.isfinite(L)]] = True
    if isq is not None:
        active &= ~isq
    Q = L.shape[0]
    m = np.where(active, k * Q / max(active.sum(), 1), 0.0)
    with np.errstate(divide="ignore"):
        logm = np.log(m)
    logv = np.zeros(C)
    for _ in range(iters):
        logu = math.log(k) - row_lse(L + logv[J])
        lc = col_lse(L + logu[:, None], J, C)
        new = np.zeros(C)
        a = active & np.isfinite(lc)
        new[a] = rho * (logm[a] - lc[a])
        if np.max(np.abs(new - logv)) < 1e-5:
            logv = new
            break
        logv = new
    return logv


def decode_scores(run, cfg):
    X = prep(run, cfg["tr"], cfg["depth"], cfg["dropq"])
    if cfg["mode"] == "none":
        return X
    if cfg["mode"] == "sink":
        b = sinkhorn(X, run.J, run.C, cfg["tau"], cfg["rho"],
                     run.isq if cfg["dropq"] else None)
        return X / cfg["tau"] + b[run.J]
    if cfg["mode"] == "dsm":
        L = X / cfg["tau"]
        cl = col_lse(L, run.J, run.C)[run.J]
        with np.errstate(invalid="ignore"):
            S = L - cfg["lam"] * np.where(np.isfinite(cl), cl, 0.0)
        return np.where(np.isfinite(L), S, -np.inf)
    raise ValueError(cfg["mode"])


def to_rank(run, S, keep=50):
    out, sc = {}, {}
    for i, q in enumerate(run.qs):
        o = np.argsort(-S[i], kind="stable")
        o = [j for j in o[:keep] if np.isfinite(S[i, j])]
        # back-fill from the base order so a dropped/masked row still has 7
        ids = [run.pool[run.J[i, j]] for j in o]
        if len(ids) < K:
            for j in range(run.V.shape[1]):
                c = run.pool[run.J[i, j]]
                if np.isfinite(run.V[i, j]) and c not in ids:
                    ids.append(c)
                if len(ids) >= K:
                    break
        out[q] = ids
        sc[q] = np.array([S[i, j] for j in o] + [-1e9] * (len(ids) - len(o)), dtype=np.float32)
    return out, sc


def perq_hits(rank, split):
    g = D.golds(split)
    return np.array([len(set(rank.get(q, [])[:K]) & set(g[q])) for q in g], dtype=np.float64)


def grid():
    out = [dict(mode="none", tr="z", depth=0, dropq=False)]
    for dropq in (False, True):
        out.append(dict(mode="none", tr="z", depth=0, dropq=dropq))
        for tr in ("z", "rk"):
            taus = (0.1, 0.2, 0.35, 0.6, 1.0) if tr == "z" else (0.15, 0.3, 0.6, 1.0)
            for depth in (40, 0):
                for tau in taus:
                    for rho in (1.0, 0.6, 0.3):
                        out.append(dict(mode="sink", tr=tr, depth=depth, dropq=dropq,
                                        tau=tau, rho=rho))
                    for lam in (0.5, 1.0):
                        out.append(dict(mode="dsm", tr=tr, depth=depth, dropq=dropq,
                                        tau=tau, lam=lam))
    uniq, seen = [], set()
    for c in out:
        key = json.dumps(c, sort_keys=True)
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    return uniq


def name(cfg):
    if cfg["mode"] == "none":
        return "base" + ("_dropq" if cfg["dropq"] else "")
    p = f"{cfg['mode']}_{cfg['tr']}_t{cfg['tau']}_d{cfg['depth'] or 'all'}"
    p += f"_r{cfg['rho']}" if cfg["mode"] == "sink" else f"_l{cfg['lam']}"
    return p + ("_dropq" if cfg["dropq"] else "")


# ---------------------------------------------------------------- diagnostics --
def cmd_diag(_a=None):
    rep = {}
    for split in ("train", "devx"):
        g = D.golds_in_pool(split)
        pool = set(D.candidates(split))
        qids = {r["id"] for r in D.queries(split)}
        indeg = {}
        for q, v in g.items():
            for c in set(v):
                indeg[c] = indeg.get(c, 0) + 1
        hist = {k: 0 for k in ("0", "1", "2", "3", "4+")}
        for c in pool:
            n = indeg.get(c, 0)
            hist[str(n) if n < 4 else "4+"] += 1
        qin = [c for c in pool if c in qids]
        qgold = sum(1 for c in qin if indeg.get(c, 0) > 0)
        pairs = sum(len(set(v)) for v in g.values())
        qpairs = sum(1 for v in g.values() for c in set(v) if c in qids)
        row = dict(queries=len(g), pool=len(pool), gold_pairs=pairs,
                   slots_7Q=K * len(g), pool_share_cited=round(1 - hist["0"] / len(pool), 4),
                   indegree_hist=hist, queries_in_pool=len(qin),
                   queries_in_pool_that_are_gold=qgold,
                   gold_pairs_on_query_candidates=qpairs)
        rep[split] = row
        log(f"  {split}: {json.dumps(row)}")
    D.atomic_json(f"{P7}/results/ot_diag.json", rep)
    log("  READ IT: pool_share_cited near 1 and indegree mostly 1 = the transport prior is "
        "real; gold_pairs_on_query_candidates ~0 = dropq is free.")
    return rep


# --------------------------------------------------------------------- sweep --
def sweep_base(tag, lists_fn, splits, deadline):
    runs = {}
    for sp in splits:
        L = lists_fn(sp)
        if L is None:
            return None
        runs[sp] = Run(sp, L)
    rows = []
    base = {}
    for sp in ("train", "devx"):
        rk, _ = to_rank(runs[sp], decode_scores(runs[sp], grid()[0]))
        base[sp] = perq_hits(rk, sp)
    log(f"  [{tag}] base hits train {int(base['train'].sum())}  devx {int(base['devx'].sum())}"
        f"  (devx F1 {R.evaluate(to_rank(runs['devx'], decode_scores(runs['devx'], grid()[0]))[0], 'devx')['f1']:.4f})")
    for cfg in grid()[1:]:
        if time.time() > deadline:
            log("  deadline hit -- stopping this sweep early")
            break
        r = {"cfg": cfg, "name": name(cfg)}
        for sp in ("train", "devx"):
            rk, _ = to_rank(runs[sp], decode_scores(runs[sp], cfg))
            h = perq_hits(rk, sp)
            dlt = h - base[sp]
            se = dlt.std(ddof=1) * math.sqrt(len(dlt)) if len(dlt) > 1 else 0.0
            r[sp] = int(h.sum())
            r[sp + "_gain"] = int(dlt.sum())
            r[sp + "_2se"] = round(2 * se, 1)
        rows.append(r)
    rows.sort(key=lambda r: (r["train"], r["devx"]), reverse=True)
    log(f"  [{tag}] top configs by TRAIN hits (gain vs base, 2SE bar):")
    for r in rows[:8]:
        log(f"     {r['name']:38s} train {r['train']:5d} ({r['train_gain']:+4d}, 2se {r['train_2se']:5.1f})"
            f"   devx {r['devx']:4d} ({r['devx_gain']:+3d}, 2se {r['devx_2se']:4.1f})")
    return runs, rows, base


def finish(tag, runs, cfg, label, kind="hybrid"):
    out_tag = f"ot_{tag}_{label}"
    scores = {}
    for sp, run in runs.items():
        rk, sc = to_rank(run, decode_scores(run, cfg))
        R.save_run(out_tag, sp, rk, sc)
        if sp in ("train", "devx"):
            scores[sp] = R.evaluate(rk, sp)
    dv = scores["devx"]
    R.save_run(out_tag, "meta", None, meta={
        "tag": out_tag, "kind": kind, "base": tag, "config": cfg,
        "technique": "transductive decoding (entropic OT / dual softmax) over the test pool",
        "devx_scores": {k: dv[k] for k in ("f1", "hits", "precision", "recall") if k in dv},
        "train_f1": scores["train"]["f1"]})
    R.cmd_zip(argparse.Namespace(tag=out_tag, kind=kind, note=json.dumps(cfg)))
    return out_tag, scores["train"]["f1"], dv["f1"], dv["hits"]


def cmd_all(a):
    t0 = time.time()
    deadline = t0 + a.minutes * 60
    cmd_diag()
    splits = ("train", "devx", "test")
    bases = [b for b in a.bases.split(",") if b]
    todo = [(b, (lambda sp, b=b: load_lists(b, sp))) for b in bases]
    todo.append(("zmix", lambda sp: zmix(MIX, sp)))
    made = []
    per_base_budget = max(60.0, (deadline - time.time()) / max(len(todo), 1))
    for tag, fn in todo:
        if time.time() > deadline:
            log(f"  global deadline: skipping {tag}")
            continue
        if fn("devx") is None:
            log(f"  [{tag}] no saved run -- skipped")
            continue
        log(f"\n=== base {tag}  ({(time.time() - t0) / 60:.1f} min in)")
        try:
            res = sweep_base(tag, fn, splits, min(deadline, time.time() + per_base_budget))
        except Exception as e:                       # one bad base must not kill the run
            log(f"  [{tag}] FAILED: {type(e).__name__}: {e}")
            continue
        if res is None:
            log(f"  [{tag}] missing a split -- skipped")
            continue
        runs, rows, base = res
        D.atomic_json(f"{P7}/results/ot_sweep_{tag}.json",
                      {"base": {k: int(v.sum()) for k, v in base.items()}, "rows": rows})
        if not rows:
            continue
        picks = [("tr", rows[0])]
        # a second, deliberately different pick: best train among configs whose devx gain
        # is also >= 0 (guards against a train-only artefact of in-sample stack scores)
        alt = [r for r in rows if r["devx_gain"] >= 0 and r["name"] != rows[0]["name"]]
        if alt:
            picks.append(("trdv", alt[0]))
        for label, r in picks:
            if r["train_gain"] <= 0:
                continue
            try:
                made.append((*finish(tag, runs, r["cfg"], label), r))
            except Exception as e:
                log(f"  [{tag}] zip FAILED: {type(e).__name__}: {e}")
    order(made)
    log(f"\nDONE in {(time.time() - t0) / 60:.1f} min")


def order(made):
    lines = ["# Part Z (transductive decoding) -- upload order", "",
             "Sorted by devx F1. `test ~= devx - 0.005` for stacks (measured on 10 points).",
             "Upload from the top; the zips are in submissions/_upload/.", "",
             "| # | zip | devx F1 | devx hits | devx gain (2SE) | train gain (2SE) | config |",
             "|---|---|---|---|---|---|---|"]
    made = sorted(made, key=lambda m: m[2], reverse=True)
    for i, (tag, trf, dvf, dvh, r) in enumerate(made, 1):
        lines.append(f"| {i} | submission_{tag}.zip | {dvf:.4f} | {dvh} | "
                     f"{r['devx_gain']:+d} ({r['devx_2se']}) | {r['train_gain']:+d} "
                     f"({r['train_2se']}) | {r['name']} |")
    txt = "\n".join(lines) + "\n"
    with open(f"{P7}/SUBMIT_ORDER_Z.md", "w") as f:
        f.write(txt)
    log("\n" + txt)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("diag")
    s = sub.add_parser("all")
    s.add_argument("--bases", default=DEFAULT_BASES)
    s.add_argument("--minutes", type=float, default=75)
    a = p.parse_args()
    {"diag": cmd_diag, "all": cmd_all}[a.cmd](a)


if __name__ == "__main__":
    main()
