#!/usr/bin/env python
"""Smoke test for phase7_scale.py -- the pool-growth curve and rank-only fusion.

The checks that would otherwise show up as a wrong leaderboard score:

  * `grown_pool` must add REAL non-golds. If a distractor is a gold for the split, or is
    one of the split's own queries, the degradation measurement is meaningless -- it
    would be measuring recall against a pool we corrupted. Asserted directly.
  * `with_pool` must restore the cache even when the body raises, or every later system
    in the same process silently scores against a grown pool.
  * growing the pool must never RAISE the hit count: adding true negatives can only
    distract. A rise means the swap did not take effect.
  * RRF must read only RANKS. Multiplying a member's scores by 1000 must change nothing;
    this is the single property the whole module exists for.
  * a one-member fusion must reproduce that member's own top-7 exactly.
  * fused output must be a valid ranking: no duplicates, and ids drawn only from the
    members' own lists.
  * greedy selection must be monotone on the selection split -- it is a maximiser, and a
    round that reports a drop means the weights are not being applied.
  * `--help` must render for both subcommands.

  python smoke_p7_scale.py
"""
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("P7_SMOKE_DIR",
                      os.path.join(tempfile.gettempdir(), "p7_smoke_scale"))
sys.path.insert(0, HERE)

import smoke_p7 as S                                   # sets every CS781_* env var

import numpy as np

import phase7_data as D
import phase7_rank as R
import phase7_scale as SC

FAIL = []


def ok(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAIL.append(name)


def main():
    print("building fake data")
    S.build_data()

    # base runs for the fusion to combine
    env = dict(os.environ)
    for args in (["phase7_rank.py", "tfidf", "--tag", "tfidf_sqrt", "--qtf", "sqrt",
                  "--max-df", "0.2", "--splits", "train,devx,test", "--topn", "25"],
                 ["phase7_rank.py", "bm25", "--tag", "bm25_hi", "--k1", "3.0", "--b",
                  "1.0", "--qtf", "sqrt", "--max-df", "0.2",
                  "--splits", "train,devx,test", "--topn", "25"]):
        p = subprocess.run([sys.executable] + args, cwd=HERE, env=env,
                           capture_output=True, text=True)
        if p.returncode != 0:
            print(p.stdout[-1500:], p.stderr[-1500:])
        ok(f"base run {args[3]} built", p.returncode == 0)

    # ---- grown_pool plants only true negatives -----------------------------
    base = D.candidates("devx")
    gold = {c for v in D.golds("devx").values() for c in v}
    qs = {r["id"] for r in D.queries("devx")}
    pool, added = SC.grown_pool("devx", 12)
    extra = set(pool) - set(base)
    ok("grown_pool actually added candidates", added > 0 and len(extra) == added,
       f"added {added}")
    ok("no distractor is a gold for the split", not (extra & gold),
       f"{len(extra & gold)} golds leaked in")
    ok("no distractor is one of the split's queries", not (extra & qs))
    ok("the original pool is left intact",
       all(pool[c] == base[c] for c in base))
    ok("asking for more than exists is clamped, not an error",
       SC.grown_pool("devx", 10 ** 6)[1] <= len(D.candidates("train")))

    # ---- with_pool restores, even on a raise -------------------------------
    key = "c_dev"
    before = D._CACHE.get(key)
    try:
        SC.with_pool("devx", pool, lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    except RuntimeError:
        pass
    ok("with_pool restores the cache after an exception",
       D._CACHE.get(key) is before,
       "otherwise every later system scores against a grown pool")
    seen = SC.with_pool("devx", pool, lambda: len(D.candidates("devx")))
    ok("with_pool really installs the grown pool", seen == len(pool),
       f"{seen} vs {len(pool)}")

    # ---- growing the pool can only distract --------------------------------
    cfg = dict(max_df=0.2, passages=0, stride=None, agg="max", beta=0.7, window=60,
               topn=25)
    hits = []
    for k in (0, 10, 25):
        pl, _ = SC.grown_pool("devx", k)
        rk = SC.with_pool("devx", pl,
                          lambda: SC.score_against("devx", "tfidf_sqrt", cfg))
        s = D.score_ranking({q: rk[q] for q in D.golds("devx") if q in rk},
                            D.golds("devx"), reachable=D.golds_in_pool("devx"))
        hits.append(s["hits"])
    ok("adding true negatives never increases hits",
       all(hits[i] >= hits[i + 1] for i in range(len(hits) - 1)),
       f"hits {hits} as the pool grows")

    # ---- RRF reads ranks ONLY ----------------------------------------------
    names, per = SC.rr_matrix(["tfidf_sqrt", "bm25_hi"], "devx", 25, 60)
    ok("both members entered the fusion", len(names) == 2, str(names))
    w = np.array([1.0, 1.0])
    a = SC.fuse(per, w, 25)
    rk1, sc1 = R.load_run("tfidf_sqrt", "devx")
    R.save_run("scale_blown", "devx", rk1, {q: np.asarray(v) * 1000.0 for q, v in sc1.items()})
    names2, per2 = SC.rr_matrix(["scale_blown", "bm25_hi"], "devx", 25, 60)
    b = SC.fuse(per2, w, 25)
    same = sum(1 for q in a if a[q][:7] == b.get(q, [])[:7])
    ok("scaling a member's scores by 1000 changes nothing",
       same == len(a), f"{same}/{len(a)} queries identical -- RRF is scale-free")

    one = SC.fuse(*(lambda n, p: (p, np.array([1.0]), 25))(
        *SC.rr_matrix(["tfidf_sqrt"], "devx", 25, 60)))
    ok("a one-member fusion reproduces that member's top-7",
       all(one[q][:7] == rk1[q][:7] for q in one),
       f"{sum(1 for q in one if one[q][:7] != rk1[q][:7])} queries differ")

    ok("fused lists hold no duplicates", all(len(set(v)) == len(v) for v in a.values()))
    pooled = {q: set(rk1.get(q, [])) | set(R.load_run("bm25_hi", "devx")[0].get(q, []))
              for q in a}
    ok("fused ids all come from the members' own lists",
       all(set(a[q]) <= pooled[q] for q in a))

    # ---- the CLI -----------------------------------------------------------
    for args in (["curve", "--split", "devx", "--systems", "tfidf_sqrt",
                  "--grow", "0,10", "--topn", "25", "--passages", "0"],
                 ["rrf", "--tag", "smoke_rrf", "--members", "tfidf_sqrt,bm25_hi",
                  "--splits", "devx", "--select", "devx", "--topn", "25",
                  "--rounds", "3"]):
        p = subprocess.run([sys.executable, "phase7_scale.py"] + args, cwd=HERE,
                           env=env, capture_output=True, text=True)
        if p.returncode != 0:
            print(p.stdout[-2000:], p.stderr[-2000:])
        ok(f"CLI {args[0]} exits 0", p.returncode == 0)

    cu = json.load(open(os.path.join(S.P7, "results", "pool_curve.json")))
    ok("curve records a slope per system",
       "hits_per_1000_candidates" in cu["by_system"]["tfidf_sqrt"])
    rr = json.load(open(os.path.join(S.P7, "results", "rrf_smoke_rrf.json")))
    ok("rrf records integer weights", all(float(x).is_integer() for x in rr["weights"]),
       str(rr["weights"]))
    ok("rrf picked at least one member", sum(rr["weights"]) >= 1)

    # greedy must be monotone on the selection split
    p = subprocess.run([sys.executable, "phase7_scale.py", "rrf", "--tag", "smoke_rrf2",
                        "--members", "tfidf_sqrt,bm25_hi", "--splits", "devx",
                        "--select", "devx", "--topn", "25", "--rounds", "4"],
                       cwd=HERE, env=env, capture_output=True, text=True)
    # only the "round N: +MEMBER -> sel f1 X" lines; the stop line also says f1
    f1s = [float(x.split(" f1 ")[1].split()[0]) for x in p.stdout.splitlines()
           if "round" in x and " -> " in x and " f1 " in x]
    ok("greedy selection is monotone on the selection split",
       all(f1s[i] <= f1s[i + 1] + 1e-9 for i in range(len(f1s) - 1)), str(f1s))

    for sub in ("curve", "rrf"):
        p = subprocess.run([sys.executable, "phase7_scale.py", sub, "--help"],
                           cwd=HERE, env=env, capture_output=True, text=True)
        ok(f"--help renders for {sub}", p.returncode == 0,
           (p.stderr.strip().splitlines() or [""])[-1][:80])

    print()
    if FAIL:
        print(f"SMOKE FAILED: {len(FAIL)} check(s): {FAIL}")
        return 1
    print("SMOKE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
