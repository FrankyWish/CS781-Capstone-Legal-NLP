#!/usr/bin/env python
"""Smoke test for phase7_stack.py, on fake data, before Nithin runs it.

It reuses smoke_p7's fake Phase_7_PCR corpus, then writes synthetic member runs so the
feature builder and the cross-fit can be checked against known answers:

  * a planted strong member (golds first) -- the stacker must reach near-oracle devx F1,
    which is the end-to-end "does it actually learn" test;
  * two queries whose raw scores differ by a large offset and scale but whose ordering is
    identical -- their z features must come out equal, because that per-query
    normalisation is the only thing making train (4,320-candidate pool) features
    comparable with test (1,727);
  * a member that retrieves only part of the pool -- absent must be NaN in :z and 0.0
    in :in, never a silent zero;
  * GroupKFold must never put the same query on both sides of a fold, or the train OOF
    number is fiction;
  * the submission must still be a flat zip of exactly 7 unique in-pool ids per query,
    and a second fit must give a byte-identical ranking.

  python smoke_p7_stack.py
"""
import json
import os
import subprocess
import sys
import tempfile
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("P7_SMOKE_DIR",
                      os.path.join(tempfile.gettempdir(), "p7_smoke_stack"))
sys.path.insert(0, HERE)

import smoke_p7 as S                                   # sets every CS781_* env var

import numpy as np

import phase7_data as D
import phase7_rank as R
import phase7_stack as ST

FAIL = []


def ok(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAIL.append(name)


def run(args):
    p = subprocess.run([sys.executable] + args, cwd=HERE, env=os.environ,
                       capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stdout[-3000:])
        print(p.stderr[-3000:])
    return p


# --------------------------------------------------------------- fake members --
def write_member(tag, kind, splits=("train", "devx", "test")):
    """kind: good (golds first), rand, part (half the pool, NaN-exercising)."""
    for split in splits:
        pool = sorted(D.candidates(split))
        gp = D.golds_in_pool(split)
        rank, score = {}, {}
        for qi, rec in enumerate(D.queries(split)):
            q = rec["id"]
            golds = [c for c in gp.get(q, []) if c in pool]
            rest = [c for c in pool if c not in set(golds)]
            if kind == "good":
                ids = golds + rest
            elif kind == "part":
                ids = (golds[:1] + rest)[:max(2, len(pool) // 2)]
            else:
                rs = np.random.default_rng(1000 + qi)
                ids = list(pool)
                rs.shuffle(ids)
            ids = [c for c in ids if c != q][:40]        # never retrieve itself
            # query-dependent offset and scale: raw scores are deliberately NOT
            # comparable across queries, so only the z features can carry the signal
            off, scale = (0.0, 1.0) if qi % 2 == 0 else (1000.0, 50.0)
            rank[q] = ids
            score[q] = np.array([off + scale * (len(ids) - i) for i in range(len(ids))],
                                dtype=np.float32)
        os.makedirs(R.rank_dir(tag), exist_ok=True)
        R.save_run(tag, split, rank, score)
    R.save_run(tag, "meta", {}, meta=dict(kind=kind))


def main():
    print("building fake data")
    S.build_data()
    p = run(["phase7_data.py", "gate"])
    ok("gate runs on the fake corpus", p.returncode == 0)
    if p.returncode != 0:
        return 1

    for tag, kind in (("m_good", "good"), ("m_rand", "rand"), ("m_part", "part")):
        write_member(tag, kind)
    members = ["m_good", "m_rand", "m_part"]
    print("\nchecks")

    # ---- feature builder ----------------------------------------------------
    X, y, grp, pairs, qids = ST.build("devx", members, pool_depth=40)
    names = ST.feature_names(members)
    ok("feature count matches the names", X.shape[1] == len(names),
       f"{X.shape[1]} vs {len(names)}")
    ok("every pooled pair is labelled", len(y) == len(pairs) == X.shape[0])
    ok("some positives exist", int(y.sum()) > 0, f"{int(y.sum())} of {len(y)}")

    ipart_in = names.index("m_part:in")
    ipart_z = names.index("m_part:z")
    absent = X[:, ipart_in] == 0.0
    ok("absent member -> :in is 0.0 and :z is NaN",
       absent.any() and np.isnan(X[absent][:, ipart_z]).all(),
       f"{int(absent.sum())} absent rows")
    present = X[:, ipart_in] == 1.0
    ok("present member -> :z is finite", present.any()
       and np.isfinite(X[present][:, ipart_z]).all())

    # ---- per-query normalisation kills the offset/scale ---------------------
    igz = names.index("m_good:z")
    byq = {}
    for (q, c), row in zip(pairs, X):
        byq.setdefault(q, []).append(row[igz])
    qa, qb = qids[0], qids[1]           # built with offsets (0,1000) and scales (1,50)
    # only the candidates m_good actually retrieved carry a z; the rest are NaN by design
    va = np.sort(np.array([v for v in byq[qa] if np.isfinite(v)], dtype=np.float64))
    vb = np.sort(np.array([v for v in byq[qb] if np.isfinite(v)], dtype=np.float64))
    n = min(len(va), len(vb))
    ok("z features are offset- and scale-invariant per query",
       n > 2 and np.allclose(va[-n:], vb[-n:], atol=1e-5),
       f"max diff {float(np.max(np.abs(va[-n:] - vb[-n:]))):.2e}" if n > 2 else "")

    # ---- cross-fit grouping -------------------------------------------------
    from sklearn.model_selection import GroupKFold
    Xt, yt, gt, pt, qt = ST.build("train", members, pool_depth=40)
    folds = max(2, min(5, len(set(gt.tolist()))))
    leak = 0
    for tr, te in GroupKFold(n_splits=folds).split(Xt, yt, groups=gt):
        leak += len(set(gt[tr].tolist()) & set(gt[te].tolist()))
    ok("no query spans both sides of a fold", leak == 0, f"{leak} shared groups")

    # ---- end to end ---------------------------------------------------------
    p = run(["phase7_stack.py", "fit", "--tag", "stack_smoke", "--kind", "hybrid",
             "--members", ",".join(members), "--pool-depth", "40", "--iters", "60"])
    ok("fit completes", p.returncode == 0)
    if p.returncode != 0:
        return 1
    out = p.stdout
    rank_d, _ = R.load_run("stack_smoke", "devx")
    rank_t, _ = R.load_run("stack_smoke", "test")
    ok("devx and test rankings saved", bool(rank_d) and bool(rank_t))

    g, gp = D.golds("devx"), D.golds_in_pool("devx")
    s = D.score_ranking(rank_d, g, reachable=gp)
    cap = s["max_hits"]
    ok("the planted strong member is learned (>=70% of oracle)",
       s["hits"] >= 0.7 * cap, f"{s['hits']}/{cap} hits, f1 {s['f1']:.4f}")

    meta = json.load(open(f"{R.rank_dir('stack_smoke')}/meta.json"))
    ok("meta carries kind and devx_scores for the zipper",
       meta.get("kind") == "hybrid" and "f1" in (meta.get("devx_scores") or {}))

    # ---- determinism --------------------------------------------------------
    before = json.dumps(rank_d, sort_keys=True)
    p2 = run(["phase7_stack.py", "fit", "--tag", "stack_smoke", "--kind", "hybrid",
              "--members", ",".join(members), "--pool-depth", "40", "--iters", "60"])
    after = json.dumps(R.load_run("stack_smoke", "devx")[0], sort_keys=True)
    ok("a second fit reproduces the ranking exactly",
       p2.returncode == 0 and before == after)

    # ---- submission ---------------------------------------------------------
    p = run(["phase7_rank.py", "zip", "--tag", "stack_smoke", "--kind", "hybrid"])
    ok("zip builds and validates", p.returncode == 0)
    zp = f"{D.P7}/submissions/stack_smoke/submission.zip"
    if os.path.exists(zp):
        with zipfile.ZipFile(zp) as z:
            namelist = z.namelist()
            pred = json.loads(z.read("prediction.json"))
        ok("flat zip with one prediction.json", namelist == ["prediction.json"],
           str(namelist))
        tq = [r["id"] for r in D.queries("test")]
        pool = set(D.candidates("test"))
        ok("every test query present", set(pred) == set(tq),
           f"{len(pred)} vs {len(tq)}")
        ok("exactly 7 unique in-pool ids per query",
           all(len(v) == 7 and len(set(v)) == 7 and set(v) <= pool
               for v in pred.values()))
        ok("no query predicts itself", all(q not in v for q, v in pred.items()))
    else:
        ok("submission.zip exists", False, zp)

    # ---- half-split report --------------------------------------------------
    p = run(["phase7_stack.py", "half", "--tags", "m_rand,stack_smoke,m_good"])
    ok("half-split report runs", p.returncode == 0)
    ok("half-split prints a 2SE verdict",
       ("KEEP" in p.stdout or "noise" in p.stdout), p.stdout.strip()[-120:])

    print(f"\n{'SMOKE OK' if not FAIL else 'SMOKE FAILED: ' + ', '.join(FAIL)}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
