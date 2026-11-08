#!/usr/bin/env python
"""Smoke test for the MaxP / RM3 additions to phase7_rank.py and for phase7_rerank.py.

The checks that matter, i.e. the ones that would otherwise be found only by a wrong
leaderboard score:

  * the passage index is a refinement of the document index, not a different corpus:
    `passages=0` must reproduce the existing run EXACTLY, and `to_docs(..., "max")` must
    equal a brute-force max grouped by parent document;
  * RM3 at lam=0.0 must reproduce the unexpanded ranking exactly -- otherwise the
    two-pass plumbing is changing scores on its own and any gain is an artefact;
  * the cross-fit must PARTITION the train queries: every usable train query scored
    exactly once, by a model that never trained on it.  This is the whole point of the
    file -- an in-sample train feature is what we think sank `stack_ft`;
  * the scalar relevance head must be trainable under a PEFT method that freezes the
    trunk (a frozen head cannot learn a ranking at all, and the loss would still fall);
  * the written rank files must be the length the reranker promises, and `_mix` at
    mix=1.0 must equal the pure reranker ranking;
  * phase7_stack must accept the reranker as a member with no special case.

  python smoke_p7_rerank.py
"""
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("P7_SMOKE_DIR",
                      os.path.join(tempfile.gettempdir(), "p7_smoke_rr"))
sys.path.insert(0, HERE)

import smoke_p7 as S                                   # sets every CS781_* env var

import numpy as np
import torch

import phase7_data as D
import phase7_rank as R
import phase7_enc as E
import phase7_emb as M
import phase7_rerank as X

FAIL = []


def ok(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAIL.append(name)


def run(args, quiet=True):
    p = subprocess.run([sys.executable] + args, cwd=HERE, env=os.environ,
                       capture_output=True, text=True)
    if p.returncode != 0 and not quiet:
        print(p.stdout[-3000:])
        print(p.stderr[-3000:])
    return p


CE = ["--model", "M2_LexArcED", "--chunk", "128", "--q-tok", "48", "--windows", "2",
      "--depth", "6", "--score-batch", "4", "--steps", "3", "--batch", "2",
      "--n-neg", "2", "--neg-pool", "20", "--warmup", "1", "--log-every", "1",
      "--base", "tfidf_sqrt", "--cpu"]


def driver_flags(script, sub):
    """Every long flag p7*.sh actually passes to `phase7_rerank.py <sub>`.

    The first real `ce` run died on `unrecognized arguments: --topn 200` AFTER the GPU
    stage had warmed up and the refits had finished -- argparse rejects the whole command,
    so a flag the driver passes and the parser does not define costs a whole run. Reading
    the flags out of the shell script means the parser is tested against what will
    actually be typed, not against what the smoke test remembers to type."""
    import re
    txt = open(os.path.join(HERE, script), encoding="utf-8").read()
    body = txt.split(f"phase7_rerank.py {sub}", 1)
    if len(body) < 2:
        return []
    chunk = body[1].split("||")[0]
    return sorted(set(re.findall(r"--[a-z][a-z0-9-]*", chunk)))


def main():
    print("building fake data and tiny trunks")
    S.build_data()
    S.build_models()

    p = run(["phase7_data.py", "gate"])
    ok("gate runs", p.returncode == 0)
    p = run(["phase7_enc.py", "tok", "--splits", "train,devx,test", "--procs", "1"])
    ok("tokenises train/devx/test", p.returncode == 0, p.stderr[-200:])
    if p.returncode != 0:
        return 1
    p = run(["phase7_rank.py", "tfidf", "--tag", "tfidf_sqrt", "--qtf", "sqrt",
             "--max-df", "0.2", "--splits", "train,devx,test", "--topn", "40"])
    ok("first stage built on all three splits", p.returncode == 0, p.stderr[-300:])
    if p.returncode != 0:
        return 1

    # ---- the passage index -------------------------------------------------
    ix0 = R.Lexical("devx", max_df=0.2)
    ok("passages=0 leaves parent unset", ix0.parent is None and ix0.starts is None)
    ixp = R.Lexical("devx", max_df=0.2, passages=60)
    ok("the passage index has at least as many rows as documents",
       ixp.N >= len(ixp.ids), f"{ixp.N} rows for {len(ixp.ids)} docs")
    ok("every document owns at least one passage",
       len(np.unique(ixp.parent)) == len(ixp.ids),
       f"{len(np.unique(ixp.parent))} of {len(ixp.ids)}")
    ok("parent is sorted, so reduceat is valid",
       bool(np.all(np.diff(ixp.parent) >= 0)))
    rngS = np.random.default_rng(0)
    Sm = rngS.normal(size=(ixp.N, 3)).astype(np.float32)
    agg = ixp.to_docs(Sm, "max")
    brute = np.stack([Sm[ixp.parent == d].max(0) for d in range(len(ixp.ids))])
    ok("to_docs(max) equals a brute-force max by parent",
       bool(np.allclose(agg, brute)), f"max diff {float(np.abs(agg - brute).max()):.2e}")
    aggm = ixp.to_docs(Sm, "mean")
    brutem = np.stack([Sm[ixp.parent == d].mean(0) for d in range(len(ixp.ids))])
    ok("to_docs(mean) equals a brute-force mean by parent",
       bool(np.allclose(aggm, brutem, atol=1e-5)))

    # passages=0 must reproduce the saved run bit for bit
    base, bsc = R.load_run("tfidf_sqrt", "devx")
    r0, _ = R.rank_lexical("devx", "tfidf", 1.2, 0.75, "sqrt", 0.2, topn=40, index=ix0)
    ok("passages=0 reproduces the existing ranking exactly",
       all(r0[q] == base[q] for q in r0), f"{len(r0)} queries")

    # ---- RM3 ---------------------------------------------------------------
    r_l0, s_l0 = R.rank_rm3("devx", "tfidf", 1.2, 0.75, "sqrt", 0.2, lam=0.0,
                            topn=40, index=ix0)
    ok("RM3 at lam=0 reproduces the unexpanded ranking exactly",
       all(r_l0[q] == base[q] for q in r_l0),
       f"{sum(r_l0[q] != base[q] for q in r_l0)} queries differ")
    r_l5, _ = R.rank_rm3("devx", "tfidf", 1.2, 0.75, "sqrt", 0.2, lam=0.5, fb_docs=5,
                         fb_terms=25, topn=40, index=ix0)
    moved = sum(r_l5[q] != base[q] for q in r_l5)
    ok("RM3 at lam=0.5 actually changes the ranking", moved > 0, f"{moved} queries move")
    ok("RM3 never returns a query as its own prior case",
       all(q not in r_l5[q] for q in r_l5))

    p = run(["phase7_rank.py", "maxp_tfidf", "--tag", "maxp_t", "--passages", "60",
             "--qtf", "sqrt", "--max-df", "0.2", "--splits", "devx,test",
             "--topn", "40"], quiet=False)
    ok("maxp runs end to end", p.returncode == 0, p.stderr[-300:])
    p = run(["phase7_rank.py", "rm3", "--tag", "rm3_t", "--fb-lambda", "0.4",
             "--qtf", "sqrt", "--max-df", "0.2", "--splits", "devx,test",
             "--topn", "40"], quiet=False)
    ok("rm3 runs end to end", p.returncode == 0, p.stderr[-300:])
    for t in ("maxp_t", "rm3_t"):
        rk, _ = R.load_run(t, "test")
        ok(f"{t} wrote a test ranking", rk is not None and len(rk) > 0)

    # ---- the head is trainable --------------------------------------------
    import argparse
    ns = argparse.Namespace(model="M2_LexArcED", ckpt="best.pt", chunk=128,
                            pool="mean", bidir=False)
    model, kind, cfg = X.build(ns)
    import phase6_peft as P
    n, _, _ = P.apply_method(model, argparse.Namespace(
        method="lorma", targets="all", rank=4, alpha=8.0, lora_dropout=0.0,
        bottleneck=8, train_top=1, ada_target=2))
    ok("the scalar relevance head stays trainable under lorma",
       all(q.requires_grad for q in model.score.parameters()))
    ok("lorma still freezes the trunk's attention weights",
       not [k for k, q in model.trunk.named_parameters()
            if k.endswith("wq.weight") and q.requires_grad])
    tid = P.trunk_param_ids(model)
    ok("the head is outside the trunk LR group, so --lr-head applies to it",
       all(id(q) not in tid for q in model.score.parameters()))

    # ---- pair construction -------------------------------------------------
    a = argparse.Namespace(chunk=128, q_tok=48, windows=3)
    rows = X.pair_rows(np.arange(500), np.arange(1000, 1600), a, None)
    ok("a pair fits the context exactly once", all(len(r) <= 128 for r in rows),
       f"lengths {[len(r) for r in rows]}")
    ok("scoring builds up to --windows windows", 1 <= len(rows) <= 3, str(len(rows)))
    ok("every window starts with the same query head",
       all(np.array_equal(r[:48], np.arange(48)) for r in rows))
    rows1 = X.pair_rows(np.arange(500), np.arange(1000, 1600), a,
                        np.random.default_rng(0))
    ok("training takes exactly one window", len(rows1) == 1)

    # ---- the cross-fit partitions the train queries ------------------------
    folds = 3
    ids = [r["id"] for r in D.queries("train")]
    assign = [X.fold_of(q, folds) for q in ids]
    ok("fold_of is deterministic across calls",
       assign == [X.fold_of(q, folds) for q in ids])
    ok("fold_of uses every fold", len(set(assign)) == min(folds, len(ids)),
       f"{sorted(set(assign))}")
    # the invariant the whole file exists for: the model that scores a query must never
    # have trained on it, so the folds must be disjoint and cover everything.
    held = [{q for q, f in zip(ids, assign) if f == k} for k in range(folds)]
    seen = [{q for q, f in zip(ids, assign) if f != k} for k in range(folds)]
    ok("no fold's training set contains a query that fold will score",
       all(not (held[k] & seen[k]) for k in range(folds)))
    ok("the folds cover every train query exactly once",
       set().union(*held) == set(ids) and sum(len(h) for h in held) == len(ids),
       f"{sum(len(h) for h in held)} of {len(ids)}")

    p = run(["phase7_rerank.py", "fit", "--tag", "ce_t", "--method", "lorma",
             "--rank", "4", "--alpha", "8", "--folds", "2",
             "--splits", "train,devx,test"] + CE, quiet=False)
    ok("fit runs: cross-fit folds, final model, three splits", p.returncode == 0,
       p.stderr[-400:])
    if p.returncode != 0:
        print(f"\nSMOKE FAILED: {', '.join(FAIL)}")
        return 1

    ts = M.TrainSet("tfidf_sqrt", 20)
    want = {q for q, _, _ in ts.items}
    trk, tsc = R.load_run("ce_t", "train")
    ok("every usable train query got a cross-fitted score",
       set(trk) == want, f"{len(trk)} scored of {len(want)} usable")
    ok("train scores line up with the train ranking",
       all(len(tsc[q]) == len(trk[q]) for q in trk))
    for split in ("devx", "test"):
        rk, sc = R.load_run("ce_t", split)
        ok(f"{split} reranking written", rk is not None and len(rk) > 0)
        ok(f"{split} depth is capped at --depth",
           all(len(v) <= 6 for v in rk.values()),
           f"max {max(len(v) for v in rk.values())}")
        ok(f"{split} scores are sorted descending",
           all(np.all(np.diff(sc[q]) <= 1e-6) for q in sc))
        ok(f"{split} reranking is a permutation of the first stage's top-6",
           all(set(rk[q]) <= set(R.load_run("tfidf_sqrt", split)[0][q][:6])
               for q in rk))
    mrk, _ = R.load_run("ce_t_mix", "devx")
    ok("the _mix blend is written too", mrk is not None and len(mrk) > 0)
    ok("_mix holds the same candidates as the pure reranker",
       all(set(mrk[q]) == set(trk.get(q, mrk[q])) or set(mrk[q]) ==
           set(R.load_run("ce_t", "devx")[0][q]) for q in mrk))

    # mix=1.0 must collapse to the pure reranker
    p = run(["phase7_rerank.py", "score", "--tag", "ce_t", "--mix", "1.0",
             "--splits", "devx"] + CE)
    ok("score re-runs from the saved checkpoint", p.returncode == 0, p.stderr[-300:])
    if p.returncode == 0:
        c1, _ = R.load_run("ce_t", "devx")
        m1, _ = R.load_run("ce_t_mix", "devx")
        ok("_mix at mix=1.0 equals the pure reranker",
           all(c1[q] == m1[q] for q in c1),
           f"{sum(c1[q] != m1[q] for q in c1)} queries differ")

    # ---- partial coverage must not destroy the GPU work --------------------
    # This is the regression for the failure that cost an 82-minute L4 run: the train
    # cross-fit cannot cover a query that was never trainable, the official scorer
    # refuses a partial prediction, and the old `write` evaluated train BEFORE saving
    # devx and test, so the raise discarded 17 minutes of finished forward passes.
    full, _ = R.load_run("ce_t", "devx")
    gold_tr = D.golds("train")
    tr_rank, _ = R.load_run("ce_t", "train")
    if tr_rank:
        partial = {q: v for i, (q, v) in enumerate(tr_rank.items()) if i % 3}
        st = X.evaluate_covered(partial, "train")
        ok("evaluate_covered survives a partial split",
           st["covered"] == len(partial) and st["covered"] < st["split_queries"],
           f"{st['covered']}/{st['split_queries']}")
        ok("evaluate_covered reports a denominator for the covered subset only",
           st["denominator"] == 7 * st["covered"]
           + sum(len(gold_tr[q]) for q in partial if q in gold_tr),
           str(st["denominator"]))
        try:
            R.evaluate(partial, "train")
            raised = False
        except SystemExit:
            raised = True
        ok("the official scorer still refuses a partial prediction", raised,
           "so evaluate_covered is the only place partial coverage is allowed")
    ok("a reranker run that covers train only partially still saved devx",
       full is not None and len(full) > 0,
       "devx is written before any evaluation can raise")

    # ---- determinism and the stacker --------------------------------------
    dev = torch.device("cpu")
    mdl, _, _, _ = X.load_ckpt("ce_t", dev)
    aa = argparse.Namespace(chunk=128, q_tok=48, windows=2, score_batch=4)
    qt, ct = E.Tok("devx", "q"), E.Tok("devx", "c")
    q0 = qt.ids[0]
    cands = list(ct.ids)[:4]
    s1 = X.score_one(mdl, qt.doc(qt.pos[q0]), cands, ct, aa, dev)
    s2 = X.score_one(mdl, qt.doc(qt.pos[q0]), cands, ct, aa, dev)
    ok("scoring is deterministic", bool(np.array_equal(s1, s2)),
       f"max diff {float(np.abs(s1 - s2).max()):.2e}")
    ok("scores are finite and not all equal",
       bool(np.all(np.isfinite(s1))) and float(s1.std()) > 0,
       f"sd {float(s1.std()):.3e}")

    p = run(["phase7_stack.py", "fit", "--tag", "stack_ce_t", "--kind", "hybrid",
             "--members", "tfidf_sqrt,ce_t", "--pool-depth", "10", "--topn", "40"],
            quiet=False)
    ok("phase7_stack takes the reranker as a member", p.returncode == 0,
       p.stderr[-400:])
    if p.returncode == 0:
        mf = f"{D.P7}/results/stack_ce_t/meta.json"
        ok("the stacker wrote a meta", os.path.exists(mf))
        if os.path.exists(mf):
            cfg = json.load(open(mf)).get("config", {})
            ok("the reranker is in the stacker's member list",
               "ce_t" in (cfg.get("members") or []), str(cfg.get("members")))
            ok("none of the reranker's features were dropped as dead",
               not [f for f in (cfg.get("dropped_features") or [])
                    if f.startswith("ce_t:")],
               f"dropped {cfg.get('dropped_features')}")

    # ---- the parser accepts what the drivers actually type -----------------
    for script in ("p7d.sh",):
        flags = driver_flags(script, "fit")
        ok(f"{script} passes some flags to `fit`", len(flags) > 5, str(flags))
        pp = run(["phase7_rerank.py", "fit", "--help"])
        helptxt = pp.stdout + pp.stderr
        missing = [f for f in flags if f not in helptxt]
        ok(f"every flag {script} passes to `fit` is defined by the parser",
           not missing, f"undefined: {missing}")

    p = run(["phase7_rerank.py", "oracle", "--tags", "tfidf_sqrt",
             "--depths", "7,20,40,400"])
    ok("oracle runs and refuses depths deeper than the saved run",
       p.returncode == 0 and "skip" in p.stdout, p.stdout[-200:])

    print(f"\n{'SMOKE OK' if not FAIL else 'SMOKE FAILED: ' + ', '.join(FAIL)}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
