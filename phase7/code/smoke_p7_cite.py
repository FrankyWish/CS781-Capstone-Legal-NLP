#!/usr/bin/env python
"""Smoke test for phase7_cite.py and the phase7_stack.py additions.

The checks that would otherwise only show up as a wrong leaderboard score:

  * a mask context must NEVER contain words from past a neighbouring mask -- adjacent
    citations are common and the text past the next mask describes a different case.
    Planted here with a hand-built string and asserted word by word;
  * `beta=0` (cite) and `alpha=0` (prop) must reproduce the un-modified ranking EXACTLY,
    so any measured gain cannot be the two-pass plumbing;
  * `greedy` must return distinct candidates, one per mask, in order of mask confidence;
  * the cocite leave-one-out must be EXACT: `gmul` with a query's loo list must equal
    rebuilding the graph from scratch without that query.  This is the check that decides
    whether the train number is honest, and an approximation here would quietly inflate
    every train-side selection decision that follows;
  * the lex graph must have no self-loops, since a document is trivially its own nearest
    neighbour and would just re-add its own score;
  * `n_masks` must arrive as a live stacker feature, not be dropped as dead;
  * `--model lgbm` must fit, score, and beat nothing in particular -- just run.

  python smoke_p7_cite.py
"""
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("P7_SMOKE_DIR",
                      os.path.join(tempfile.gettempdir(), "p7_smoke_cite"))
sys.path.insert(0, HERE)

import smoke_p7 as S                                   # sets every CS781_* env var

import numpy as np

import phase7_data as D
import phase7_rank as R
import phase7_cite as C

FAIL = []


def ok(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAIL.append(name)


def run(args, quiet=True, extra_path=None):
    env = dict(os.environ)
    if extra_path:
        env["PYTHONPATH"] = extra_path + os.pathsep + env.get("PYTHONPATH", "")
    p = subprocess.run([sys.executable] + args, cwd=HERE, env=env,
                       capture_output=True, text=True)
    if p.returncode != 0 and not quiet:
        print(p.stdout[-2500:])
        print(p.stderr[-2500:])
    return p


def main():
    print("building fake data and tiny trunks")
    S.build_data()

    # ---- mask contexts, on a hand-built string ------------------------------
    txt = ("alpha beta gamma delta <CITATION> epsilon zeta eta theta "
           "<CITATION> iota kappa <CITATION> lambda mu nu xi omicron")
    ctx = C.mask_contexts(txt, window=3, min_words=2)
    ok("one context per mask", len(ctx) == 3, f"{len(ctx)}")
    ok("context 1 is the 3 words before and after mask 1",
       ctx[0] == ["beta", "gamma", "delta", "epsilon", "zeta", "eta"], str(ctx[0]))
    ok("context 2 never reaches past mask 1 or mask 3",
       ctx[1] == ["zeta", "eta", "theta", "iota", "kappa"], str(ctx[1]))
    ok("context 3 never reaches past mask 2",
       ctx[2] == ["iota", "kappa", "lambda", "mu", "nu"], str(ctx[2]))
    allw = set(w for c in ctx for w in c)
    ok("'citation' never leaks into a context from the mask itself",
       "citation" not in allw)
    ok("a mask-free text yields no contexts",
       C.mask_contexts("no citations at all here", 10) == [])
    ok("mask_counts agrees with a hand count",
       C.MASKS.findall(txt) and len(C.MASKS.findall(txt)) == 3,
       f"{len(C.MASKS.findall(txt))}")

    p = run(["phase7_data.py", "gate"])
    ok("gate runs", p.returncode == 0)
    p = run(["phase7_rank.py", "tfidf", "--tag", "tfidf_sqrt", "--qtf", "sqrt",
             "--max-df", "0.2", "--splits", "train,devx,test", "--topn", "40"])
    ok("first stage on all three splits", p.returncode == 0, p.stderr[-300:])
    if p.returncode != 0:
        return 1

    p = run(["phase7_cite.py", "masks", "--splits", "train,devx,test"], quiet=False)
    ok("masks measurement runs", p.returncode == 0, p.stderr[-300:])
    mf = f"{D.P7}/results/mask_stats.json"
    if os.path.exists(mf):
        ms = json.load(open(mf))
        tr = ms.get("train", {})
        ok("the measurement reports a mask-vs-gold correlation on labelled splits",
           "spearman_mask_vs_gold" in tr, str(tr.get("spearman_mask_vs_gold")))
        # The fixture plants the SAME number of masks in every query, so the correlation
        # is undefined and must be reported as such.  argsort(argsort(x)) would instead
        # invent a ranking for the ties and print a confident number.
        ok("a correlation over all-tied mask counts is reported as undefined, not a number",
           tr.get("mask_count_varies") is False
           and tr.get("spearman_mask_vs_gold") is None
           and tr.get("pearson_mask_vs_gold") is None,
           f"varies={tr.get('mask_count_varies')} "
           f"spearman={tr.get('spearman_mask_vs_gold')}")
        v = np.array([1.0, 1.0, 2.0, 2.0, 3.0])
        ok("tied ranks are averaged",
           list(C._tied_ranks(v)) == [0.5, 0.5, 2.5, 2.5, 4.0], str(C._tied_ranks(v)))

    # ---- beta = 0 must be a no-op ------------------------------------------
    ix = R.Lexical("devx", max_df=0.2)
    W = ix.weights("tfidf", 1.2, 0.75)
    base, _ = R.rank_lexical("devx", "tfidf", 1.2, 0.75, "sqrt", 0.2, topn=40, index=ix)
    r0, _ = C.rank_cite("devx", ix, W, "tfidf", "sqrt", agg="max", beta=0.0, window=60,
                        topn=40)
    ok("cite at beta=0 reproduces whole-document matching exactly",
       all(r0[q] == base[q] for q in r0),
       f"{sum(r0[q] != base[q] for q in r0)} queries differ")
    r1, _ = C.rank_cite("devx", ix, W, "tfidf", "sqrt", agg="greedy", beta=1.0,
                        window=60, topn=40)
    moved = sum(r1[q] != base[q] for q in r1)
    ok("cite at beta=1 changes the ranking", moved > 0, f"{moved} queries move")
    ok("cite never returns a query as its own prior case",
       all(q not in r1[q] for q in r1))

    # ---- greedy aggregation ------------------------------------------------
    rng = np.random.default_rng(0)
    Sc = rng.normal(size=(40, 4))
    gv = C.aggregate(Sc, "greedy", topn=12)
    order = np.argsort(-gv)[:4]
    ok("greedy picks distinct candidates", len(set(order.tolist())) == 4)
    strength = np.argsort(-Sc.max(0))
    want = [int(np.argmax(Sc[:, j])) for j in strength]
    ok("greedy's first picks are each context's own best, strongest context first",
       list(order[:len(set(want))]) == want[:len(set(want))] or len(set(want)) < 4,
       f"got {order.tolist()} want {want}")
    mx = C.aggregate(Sc, "max", 12)
    ok("max aggregation is the row max", bool(np.allclose(mx, Sc.max(1))))
    rr = C.aggregate(Sc, "rrf", 12)
    ok("rrf gives every ranked candidate positive mass", bool((rr >= 0).all()))

    # ---- the graphs --------------------------------------------------------
    G, deg = C.lex_graph(ix, k=5)
    ok("lex graph has no self-loop", float(abs(G.diagonal()).max()) == 0.0,
       f"max diag {float(abs(G.diagonal()).max()):.2e}")
    ok("lex graph weights are non-negative", bool((G.data >= 0).all()))
    ok("lex graph has k neighbours per row",
       bool((np.diff(G.indptr) == min(5, G.shape[0] - 1)).all()),
       f"{sorted(set(np.diff(G.indptr).tolist()))}")

    ixt = R.Lexical("train", max_df=0.2)
    Ct, degt = C.cocite_graph(ixt)
    ok("cocite graph is symmetric",
       (abs(Ct - Ct.T)).nnz == 0 or float(abs(Ct - Ct.T).max()) == 0.0)
    ok("cocite graph has no self-loop", float(abs(Ct.diagonal()).max()) == 0.0)

    # the check that decides whether the train number is honest
    gold = D.golds("train")
    q0 = next((q for q, gs in gold.items()
               if len([c for c in gs if c in ixt.pos]) >= 2), None)
    ok("found a train query with 2+ in-pool golds to test leave-one-out on",
       q0 is not None)
    if q0 is not None:
        loo = [ixt.pos[c] for c in gold[q0] if c in ixt.pos]
        seed = np.random.default_rng(1).random(len(ixt.ids))
        got = C.gmul(Ct, degt, seed, loo)
        # rebuild the graph from every train query EXCEPT q0
        from scipy import sparse
        rows, cols = [], []
        deg2 = np.zeros(len(ixt.ids), dtype=np.float64)
        for q, gs in gold.items():
            if q == q0:
                continue
            idx = sorted({ixt.pos[c] for c in gs if c in ixt.pos})
            if len(idx) < 2:
                continue
            for i in idx:
                deg2[i] += 1.0
                for j in idx:
                    if i != j:
                        rows.append(i)
                        cols.append(j)
        C2 = sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                               shape=Ct.shape)
        C2.sum_duplicates()
        want2 = C.gmul(C2, deg2, seed, None)
        dmax = float(np.abs(got - want2).max())
        ok("the cocite leave-one-out is EXACT, not an approximation", dmax < 1e-9,
           f"max |diff| {dmax:.3e}")

    # ---- alpha = 0 must be a no-op -----------------------------------------
    p0, _ = C.rank_prop("devx", ix, W, "tfidf", 1.2, 0.75, "sqrt", G, deg, 0.0, 20,
                        topn=40)
    ok("prop at alpha=0 reproduces the un-propagated ranking exactly",
       all(p0[q] == base[q] for q in p0),
       f"{sum(p0[q] != base[q] for q in p0)} queries differ")
    pa, _ = C.rank_prop("devx", ix, W, "tfidf", 1.2, 0.75, "sqrt", G, deg, 0.8, 20,
                        topn=40)
    ok("prop at alpha=0.8 changes the ranking",
       sum(pa[q] != base[q] for q in pa) > 0)
    del ix, ixt

    # ---- end to end --------------------------------------------------------
    for args, name in (
            (["cite", "--tag", "cite_t", "--agg", "greedy", "--beta", "1.0"], "cite"),
            (["prop", "--tag", "prop_lex_t", "--graph", "lex", "--alpha", "0.3",
              "--neighbours", "5"], "prop lex"),
            (["prop", "--tag", "prop_cc_t", "--graph", "cocite", "--alpha", "0.3"],
             "prop cocite")):
        p = run(["phase7_cite.py"] + args + ["--splits", "train,devx,test",
                                             "--topn", "40"], quiet=False)
        ok(f"{name} runs on all three splits", p.returncode == 0, p.stderr[-300:])
    for t in ("cite_t", "prop_lex_t", "prop_cc_t"):
        for sp in ("train", "devx", "test"):
            rk, sc = R.load_run(t, sp)
            ok(f"{t}/{sp} written", rk is not None and len(rk) > 0 and sc is not None)

    # ---- the stacker -------------------------------------------------------
    p = run(["phase7_stack.py", "fit", "--tag", "stack_ct", "--kind", "hybrid",
             "--members", "tfidf_sqrt,cite_t,prop_lex_t", "--pool-depth", "10",
             "--topn", "40"], quiet=False)
    ok("the stacker takes the new members", p.returncode == 0, p.stderr[-400:])
    sf = f"{D.P7}/results/stack_ct/meta.json"
    if os.path.exists(sf):
        cfg = json.load(open(sf)).get("config", {})
        # the fixture gives every query the same mask count, so n_masks is genuinely
        # constant there and usable_cols is RIGHT to drop it; what must hold is that the
        # mask signal reaches the matrix at all, which masks_per_1k carries.
        dropped = cfg.get("dropped_features") or []
        counts = set(C.mask_counts("devx").values())
        ok("masks_per_1k survives as a live feature", "masks_per_1k" not in dropped,
           f"dropped {dropped}")
        ok("n_masks is dropped only because the fixture ties every query",
           ("n_masks" not in dropped) or len(counts) == 1,
           f"distinct fixture mask counts {sorted(counts)}, dropped {dropped}")
        ok("the new retrievers are in the member list",
           {"cite_t", "prop_lex_t"} <= set(cfg.get("members") or []),
           str(cfg.get("members")))

    # lightgbm may live in a scratch --target dir rather than the main env
    libs = os.environ.get("P7_SMOKE_LIBS", "")
    p = run(["phase7_stack.py", "fit", "--tag", "stack_lr", "--kind", "hybrid",
             "--members", "tfidf_sqrt,cite_t,prop_lex_t", "--pool-depth", "10",
             "--topn", "40", "--model", "lgbm", "--trunc", "10"],
            quiet=False, extra_path=libs)
    if "needs lightgbm" in (p.stdout + p.stderr):
        print("  SKIP  --model lgbm: lightgbm not importable here")
    else:
        ok("--model lgbm fits and scores", p.returncode == 0, p.stderr[-400:])
        if p.returncode == 0:
            rk, _ = R.load_run("stack_lr", "test")
            ok("the lambdarank stack wrote a test ranking", rk is not None and len(rk) > 0)
            cfg = json.load(open(f"{D.P7}/results/stack_lr/meta.json"))["config"]
            ok("the meta records which model was used", cfg.get("model") == "lgbm",
               str(cfg.get("model")))

    # ---- the train-side bar ------------------------------------------------
    p = run(["phase7_stack.py", "half", "--tags",
             "tfidf_sqrt,cite_t,prop_lex_t,stack_ct"], quiet=False)
    ok("half runs", p.returncode == 0, p.stderr[-300:])
    ok("half prints the 827-query train table too", "train (" in p.stdout,
       [l for l in p.stdout.splitlines() if "queries, denominator" in l][:3])
    hf = f"{D.P7}/results/half_report.json"
    if os.path.exists(hf):
        h = json.load(open(hf))
        ok("the half report records devx halves and train",
           {"devx_half_A", "devx_half_B", "train"} <= set(h), str(sorted(h)))

    # ---- every subcommand's --help must render ------------------------------
    # A literal '%' in a help string makes argparse raise `%o format: an integer is
    # required, not dict` the moment anyone asks for help -- which is how you find out
    # about it, usually while debugging something else under time pressure.
    broken = []
    for script, subs in (("phase7_rank.py", ("bm25", "tfidf", "maxp", "maxp_tfidf",
                                             "rm3", "rm3_bm25", "fuse", "zip", "show")),
                         ("phase7_cite.py", ("masks", "cite", "prop")),
                         ("phase7_stack.py", ("fit", "half")),
                         ("phase7_rerank.py", ("fit", "score", "oracle"))):
        for sub in subs:
            if run([script, sub, "--help"]).returncode != 0:
                broken.append(f"{script} {sub}")
    ok("every subcommand's --help renders", not broken, f"broken: {broken}")

    print(f"\n{'SMOKE OK' if not FAIL else 'SMOKE FAILED: ' + ', '.join(FAIL)}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
