#!/usr/bin/env python3
"""Phase 7 -- why devx does not transfer, and the scale-free way to combine systems.

THE OBSERVATION THIS EXISTS FOR.  `citep` scored devx 0.4302 (387/556 hits, the best
single system of the phase) and test 0.4105 -- a gap of -0.0197, the worst of any system
uploaded. Four stacks inside a 9-hit devx band came back in exactly REVERSED test order
(Spearman -1.0). Both facts have one candidate explanation that nothing in this phase has
measured:

    devx has 1,023 candidates.  test has 1,727.  train has 4,320.

Every system scores a query against a pool, and the pool's size changes the scores. A
short query degrades fastest: `citep`'s mask-anchored windows are ~120 words, so one
coincidental passage match can dominate, while a 6,000-word whole-judgment query averages
over far more evidence. And the learned stacker is fitted on train's 4,320-candidate pool
and applied to test's 1,727, on features that are raw retrieval scores.

Two commands:

`curve` MEASURES it, without touching a test label. The devx pool is grown with real
judgments drawn from the train pool as known non-golds -- 1,023 -> 1,727 and beyond -- and
each system is re-indexed and re-scored at each size. That is a direct simulation of what
test does to a system, and it turns "devx -> test" from one fitted regression into a
per-system degradation slope that can be extrapolated.

`rrf` is the fix implied by it. Reciprocal-rank fusion uses only each member's RANK, never
its score, so it is invariant to pool size, to score scale, and to the per-query score
inflation that makes a long query look confident. Member weights are chosen by greedy
forward selection on the 827 train queries. If pool-size sensitivity is what breaks
transfer, a rank-only combiner should transfer where the learned stacker did not.

   python phase7_scale.py curve --systems citep,maxp2,tfidf_sqrt --grow 0,350,704,1400
   python phase7_scale.py rrf   --tag rrf_best --members citep,maxp2,tfidf_sqrt,stack_new
"""
import argparse
import json
import os

import numpy as np

import phase7_data as D
import phase7_rank as R

P7 = R.P7
log = D.log


# --------------------------------------------------------------- 1. the curve --
def grown_pool(split, k, seed=D.SEED):
    """The split's own pool plus `k` real judgments from the train pool as non-golds.

    Distractors are drawn from the train candidate pool rather than generated, so they
    are genuine Indian judgments with the same vocabulary, length and citation style as
    the real pool. Anything that is a gold for this split, or is one of its queries, is
    excluded -- a distractor must be a true negative or the measurement is meaningless.
    """
    pool = dict(D.candidates(split))
    if k <= 0:
        return pool, 0
    gold = {c for v in D.golds(split).values() for c in v}
    qs = {r["id"] for r in D.queries(split)}
    tr = D.candidates("train")
    cand = sorted(c for c in tr if c not in pool and c not in gold and c not in qs)
    rng = np.random.default_rng(seed)
    take = cand if k >= len(cand) else [cand[i] for i in
                                        rng.choice(len(cand), size=k, replace=False)]
    for c in take:
        pool[c] = tr[c]
    return pool, len(take)


def with_pool(split, pool, fn):
    """Run `fn()` with `split`'s candidate pool temporarily replaced.

    phase7_data caches pools by key, and every index, self-exclusion and gold-in-pool
    helper reads them through that cache. Swapping the cache entry is what lets a system
    be re-indexed against a grown pool without touching any module that is currently
    running on Colab -- the alternative was adding a parameter to Lexical.__init__ and
    re-uploading phase7_rank.py mid-run.
    """
    key = f"c_{'dev' if split == 'devx' else split}"
    old = D._CACHE.get(key)
    D._CACHE[key] = pool
    try:
        return fn()
    finally:
        if old is None:
            D._CACHE.pop(key, None)
        else:
            D._CACHE[key] = old


def score_against(split, system, cfg):
    """Re-run one system on whatever pool is currently installed. -> ranking."""
    if system == "citep":
        import phase7_cite as CT
        ix = R.Lexical(split, max_df=cfg["max_df"], passages=cfg["passages"],
                       stride=cfg["stride"])
        W = ix.weights("tfidf", 1.2, 0.75)
        rk, _ = CT.rank_cite(split, ix, W, "tfidf", "sqrt", cfg["agg"], cfg["beta"],
                             cfg["window"], topn=cfg["topn"])
        return rk
    passages = cfg["passages"] if system in ("maxp2", "maxp_tfidf") else 0
    stride = cfg["stride"] if passages else None
    rk, _ = R.rank_lexical(split, "tfidf", 1.2, 0.75, "sqrt", cfg["max_df"],
                           topn=cfg["topn"], passages=passages, stride=stride)
    return rk


def cmd_curve(a):
    cfg = dict(max_df=a.max_df, passages=a.passages, stride=a.stride, agg=a.agg,
               beta=a.beta, window=a.window, topn=a.topn)
    base_n = len(D.candidates(a.split))
    out, rows = {}, []
    for system in [s for s in a.systems.split(",") if s]:
        out[system] = []
        for k in [int(x) for x in a.grow.split(",") if x != ""]:
            pool, added = grown_pool(a.split, k)
            rk = with_pool(a.split, pool, lambda: score_against(a.split, system, cfg))
            # Score against the ORIGINAL golds: the distractors are true negatives, so
            # the reachable ceiling is unchanged and hits are directly comparable.
            s = D.score_ranking({q: rk[q] for q in D.golds(a.split) if q in rk},
                                D.golds(a.split),
                                reachable=D.golds_in_pool(a.split))
            n = base_n + added
            out[system].append(dict(pool=n, hits=s["hits"], f1=round(s["f1"], 4)))
            rows.append((system, n, s["hits"], s["f1"]))
            log(f"  {system:14s} pool {n:6d}  hits {s['hits']:4d}  f1 {s['f1']:.4f}")
        h = [r["hits"] for r in out[system]]
        n = [r["pool"] for r in out[system]]
        if len(n) > 1:
            sl = np.polyfit(n, h, 1)[0]
            out[system] = {"points": out[system],
                           "hits_per_1000_candidates": round(float(sl) * 1000, 2),
                           "predicted_at_1727": round(float(np.polyval(
                               np.polyfit(n, h, 1), 1727)), 1)}
            log(f"  {system:14s} SLOPE {sl * 1000:+.1f} hits per 1000 extra candidates; "
                f"extrapolated to the test pool's 1727: "
                f"{out[system]['predicted_at_1727']:.0f} hits")
    D.atomic_json(f"{P7}/results/pool_curve.json",
                  {"split": a.split, "base_pool": base_n, "config": cfg,
                   "by_system": out})
    log("  -> results/pool_curve.json")
    log("  READ IT LIKE THIS: the system with the FLATTEST slope is the one whose devx "
        "number you can trust on test. A steep slope explains a devx win that did not "
        "transfer, and it is a reason to prefer a rank-only combiner (`rrf`).")
    # Deliberately returns None: `main` does `return a.fn(a) or 0`, so returning
    # the results dict made SystemExit(dict) and a successful run exited 1.
    return None


# ------------------------------------------------------- 2. rank-only fusion --
def rr_matrix(members, split, topn, k=60):
    """-> (candidate id list, members x candidates reciprocal-rank matrix) per query.

    Only ranks are read. A member that did not retrieve a candidate contributes 0 for it,
    which is the standard RRF treatment and is exactly what makes the fusion indifferent
    to how large or small a member's scores happen to be on this pool.
    """
    runs = {}
    for m in members:
        rk, _ = R.load_run(m, split)
        if rk is None:
            log(f"  skip {m}: no {split} ranking")
            continue
        runs[m] = rk
    if not runs:
        raise SystemExit(f"no members have a {split} ranking")
    qs = [r["id"] for r in D.queries(split)]
    per = {}
    for q in qs:
        ids, seen = [], {}
        for m in runs:
            for c in runs[m].get(q, [])[:topn]:
                if c not in seen:
                    seen[c] = len(ids)
                    ids.append(c)
        M = np.zeros((len(runs), len(ids)), dtype=np.float64)
        for i, m in enumerate(runs):
            for r, c in enumerate(runs[m].get(q, [])[:topn]):
                M[i, seen[c]] = 1.0 / (k + r + 1)
        per[q] = (ids, M)
    return list(runs), per


def fuse(per, w, topn):
    out = {}
    for q, (ids, M) in per.items():
        v = w @ M
        n = min(topn, len(ids))
        top = np.argpartition(-v, n - 1)[:n]
        top = top[np.argsort(-v[top], kind="stable")]
        out[q] = [ids[i] for i in top]
    return out


def cmd_rrf(a):
    members = [m for m in a.members.split(",") if m]
    sel = a.select
    names, per = rr_matrix(members, sel, a.topn, a.k)
    log(f"  {len(names)} members with a {sel} ranking: {names}")

    # Greedy forward selection on the selection split. Weights are integers (how many
    # times a member is added), which keeps the search tiny and the result readable, and
    # the metric is the official one rather than a surrogate.
    w = np.zeros(len(names))
    best = (0.0, None)
    for rnd in range(a.rounds):
        cand = []
        for i in range(len(names)):
            t = w.copy()
            t[i] += 1.0
            s = D.score_ranking({q: v for q, v in fuse(per, t, a.topn).items()
                                 if q in D.golds(sel)},
                                D.golds(sel), reachable=D.golds_in_pool(sel))
            cand.append((s["f1"], s["hits"], i))
        cand.sort(reverse=True)
        f1, hits, i = cand[0]
        if best[1] is not None and f1 <= best[0] + 1e-9:
            log(f"  round {rnd + 1}: nothing improved {sel} f1 {best[0]:.4f}; stopping")
            break
        w[i] += 1.0
        best = (f1, w.copy())
        log(f"  round {rnd + 1}: +{names[i]:16s} -> {sel} f1 {f1:.4f} hits {hits} "
            f"| weights {dict(zip(names, w.astype(int).tolist()))}")
    if best[1] is None:
        raise SystemExit("greedy selection found nothing")
    w = best[1]
    log(f"  CHOSE weights {dict(zip(names, w.astype(int).tolist()))} "
        f"({sel} f1 {best[0]:.4f})")

    for split in [s for s in a.splits.split(",") if s]:
        nm, pr = rr_matrix(members, split, a.topn, a.k)
        if nm != names:
            ww = np.array([w[names.index(m)] if m in names else 0.0 for m in nm])
            log(f"  !! {split} has a different member set {nm}; reusing the weights "
                f"it shares and 0 for the rest")
        else:
            ww = w
        rk = fuse(pr, ww, a.topn)
        R.save_run(a.tag, split, rk)
        if split in ("devx", "train"):
            R.report(f"{a.tag}/{split}", R.evaluate(rk, split))
    R.save_run(a.tag, "meta", None, meta={
        "tag": a.tag, "kind": a.kind,
        "technique": f"reciprocal-rank fusion, k={a.k}, greedy weights on {sel}",
        "members": names, "weights": w.astype(int).tolist(), "rrf_k": a.k,
        "select": sel, f"{sel}_f1": best[0]})
    D.atomic_json(f"{P7}/results/rrf_{a.tag}.json",
                  {"members": names, "weights": w.astype(int).tolist(),
                   "select": sel, "f1": best[0], "rrf_k": a.k})


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("curve", help="how fast does each system decay as the pool grows?")
    c.add_argument("--split", default="devx")
    c.add_argument("--systems", default="citep,maxp2,tfidf_sqrt")
    c.add_argument("--grow", default="0,350,704,1400",
                   help="extra real judgments added; 704 takes devx's 1023 to test's 1727")
    c.add_argument("--topn", type=int, default=100)
    c.add_argument("--max-df", type=float, default=0.2)
    c.add_argument("--passages", type=int, default=250)
    c.add_argument("--stride", type=int, default=125)
    c.add_argument("--agg", default="max")
    c.add_argument("--beta", type=float, default=0.7)
    c.add_argument("--window", type=int, default=60)
    c.set_defaults(fn=cmd_curve)

    f = sub.add_parser("rrf", help="rank-only fusion: invariant to pool size and scale")
    f.add_argument("--tag", default="rrf_best")
    f.add_argument("--kind", default="hybrid", choices=list(D.KINDS))
    f.add_argument("--members", required=True)
    f.add_argument("--splits", default="train,devx,test")
    f.add_argument("--select", default="train", choices=["train", "devx"])
    f.add_argument("--topn", type=int, default=R.TOPN)
    f.add_argument("--k", type=int, default=60, help="the RRF constant")
    f.add_argument("--rounds", type=int, default=8)
    f.set_defaults(fn=cmd_rrf)

    a = p.parse_args()
    return a.fn(a) or 0


if __name__ == "__main__":
    raise SystemExit(main())
