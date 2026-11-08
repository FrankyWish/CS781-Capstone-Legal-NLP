#!/usr/bin/env python3
"""Phase 7 -- the two signals in this data that nothing else in the phase uses.  CPU only.

    python phase7_cite.py masks                    does mask count predict gold count?
    python phase7_cite.py cite --tag cite --sweep   citation-context retrieval
    python phase7_cite.py prop --tag prop --sweep   score propagation over a document graph

-----------------------------------------------------------------------------------------
1.  CITATION-CONTEXT RETRIEVAL

Every system so far matches a whole query judgment -- about 6,000 words -- against whole
candidate judgments, and the gate measured the resulting overlap at only 1.86x idf-Jaccard
over an arbitrary pair.  That is a terrible retrieval unit, and the reason is structural:
a judgment cites a prior case for ONE proposition, in ONE sentence, and the other 5,900
words are about something else.

But IL-PCR tells us exactly where those sentences are.  The citations are masked, and the
gate counted 2,681 masks in the 237 test queries -- about 11 per query, against 8.25 gold
prior cases per dev query.  Each mask is a labelled pointer saying "a prior case was cited
HERE", and the words around it are the court's own description of why.

So: take a window either side of each mask as its own query, rank the pool with it, and
aggregate.  One noisy 6,000-word query becomes ~11 focused ones, each aimed at a single
gold.  `greedy` aggregation takes one distinct candidate per mask in order of the mask's
own confidence, which matches the structure almost exactly -- ~11 masks, ~8 golds, 7 slots.

A window must never reach across a NEIGHBOURING mask: adjacent citations
(`relied on <CITATION> and on <CITATION>`) are common, and the text past the next mask
describes a different case.  Contexts are therefore cut from the segments BETWEEN masks.

This uses only the released query text.  It is not corpus lookup and it does not touch
test labels.

-----------------------------------------------------------------------------------------
2.  SCORE PROPAGATION OVER A DOCUMENT GRAPH  (the cluster hypothesis)

The gate killed the citation-popularity PRIOR: counts are flat and "ever cited in train"
enriches at 0.89x, i.e. nothing.  But that is a MARGINAL measurement, and the useful
question is conditional: given that candidate A is relevant to this query, is B?

Two graphs answer it, and they are different measurements:

  --graph lex     candidate-candidate tf-idf cosine, top-k neighbours.  Covers 100% of
                  every pool and needs no labels at all.
  --graph cocite  how often two candidates are cited by the SAME train query.  Real
                  co-citation structure, but only over candidates that appear in train
                  golds (the gate measured 36% of the test pool).

Then score'(c) = z(score) + alpha * z(G @ seed), with seed the clipped top-m of the first
pass, so only confident candidates propagate.

The trap, handled: for the TRAIN split the cocite graph was built from train labels, so
scoring a train query with it is in-sample.  `Gmul` therefore does an EXACT leave-one-out
-- it subtracts the query's own co-citation pairs from both the counts and the degrees --
so the train ranking stays honest and can still be a stacker feature.

-----------------------------------------------------------------------------------------
SELECTION.  Both sweeps choose on TRAIN (827 labelled queries), not devx (118).  Nothing
here is learned from labels, so train is free out-of-sample evaluation data, and the
paired 2-SE bar on 827 queries is about 2.6x tighter than on 118 -- which matters a lot
now, because the remaining gains are the 0.005-size ones that a 118-query bar cannot see.
devx is kept as confirmation only.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import phase7_data as D                                      # noqa: E402
import phase7_rank as R                                      # noqa: E402

P7 = D.P7
TOPN = R.TOPN

# the mask shapes IL-PCR uses; identical to phase7_overlap's, deliberately
MASKS = re.compile(r"<[^<>]{0,40}>|\[\?CITATION\?\]|\[CITATION\]|_{4,}")


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]  " + " ".join(str(x) for x in a), flush=True)


# ------------------------------------------------------- the citation contexts -------
def mask_contexts(text, window=60, min_words=8):
    """-> one word list per mask: up to `window` words either side, cut at the
    neighbouring masks so a context never describes the wrong case."""
    segs, nmask, last = [], 0, 0
    for m in MASKS.finditer(text):
        segs.append(D.WORD.findall(text[last:m.start()].lower()))
        last = m.end()
        nmask += 1
    segs.append(D.WORD.findall(text[last:].lower()))
    out = []
    for i in range(nmask):
        ctx = segs[i][-window:] + segs[i + 1][:window]
        if len(ctx) >= min_words:
            out.append(ctx)
    return out


def mask_counts(split):
    """{qid: number of citation masks}.  Imported by phase7_stack as a global feature."""
    return {r["id"]: len(MASKS.findall(" ".join(r["sents"]))) for r in D.queries(split)}


def _tied_ranks(v):
    """Average ranks, so ties get the same rank.

    `argsort(argsort(v))` is NOT this: on tied values it hands back an arbitrary
    permutation with full variance, which makes a Spearman of all-tied data come out as a
    confident non-zero number instead of undefined.  That is exactly the measurement this
    command exists to make, so it has to be right."""
    o = np.argsort(v, kind="stable")
    r = np.empty(len(v), dtype=np.float64)
    r[o] = np.arange(len(v), dtype=np.float64)
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[o[j + 1]] == v[o[i]]:
            j += 1
        if j > i:
            r[o[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return r


def cmd_masks(a):
    """Measure the premise before anything is built on it: do the masks exist in the
    numbers the gate reported, and does their count predict the gold count?"""
    rows = {}
    for split in a.splits.split(","):
        nm = mask_counts(split)
        v = np.array(list(nm.values()), dtype=np.float64)
        row = dict(split=split, queries=len(v), masks=int(v.sum()),
                   per_query=D.pct(v), zero_mask_queries=int((v == 0).sum()))
        if split != "test":
            g = D.golds(split)
            qs = [q for q in g if q in nm]
            x = np.array([nm[q] for q in qs], dtype=np.float64)
            y = np.array([len(g[q]) for q in qs], dtype=np.float64)
            row["gold_per_query_mean"] = round(float(y.mean()), 3)
            live = bool(x.std() > 0 and y.std() > 0)   # both guards use the RAW values
            row["mask_count_varies"] = bool(x.std() > 0)
            row["pearson_mask_vs_gold"] = (round(float(np.corrcoef(x, y)[0, 1]), 4)
                                           if live else None)
            xr, yr = _tied_ranks(x), _tied_ranks(y)
            row["spearman_mask_vs_gold"] = (round(float(np.corrcoef(xr, yr)[0, 1]), 4)
                                            if live else None)
            qt = np.quantile(x, [0.25, 0.5, 0.75])
            buck = np.digitize(x, qt)
            row["mean_golds_by_mask_quartile"] = [
                round(float(y[buck == b].mean()), 2) if (buck == b).any() else None
                for b in range(4)]
        rows[split] = row
        log(f"{split}: {row['queries']} queries, {row['masks']} masks, "
            f"per-query {row['per_query']}, {row['zero_mask_queries']} with none")
        if split != "test":
            log(f"       golds/query {row['gold_per_query_mean']}, "
                f"pearson {row['pearson_mask_vs_gold']}, "
                f"spearman {row['spearman_mask_vs_gold']}, "
                f"mean golds by mask quartile {row['mean_golds_by_mask_quartile']}")
    D.atomic_json(f"{P7}/results/mask_stats.json", rows)
    log("A spearman above ~0.3 means n_masks is worth adding to the stacker; "
        "near 0 means the masks locate citations but do not count them.")
    return rows


# -------------------------------------------------------------- aggregation ----------
def aggregate(Sc, agg, topn, rrf_k=60):
    """(ndocs x n_contexts) context scores -> (ndocs,) one score per candidate."""
    if agg == "max":
        return Sc.max(1).astype(np.float64)
    if agg == "sum":
        return Sc.sum(1).astype(np.float64)
    n, m = Sc.shape
    k = int(min(topn, n))
    out = np.zeros(n, dtype=np.float64)
    orders = []
    for j in range(m):
        o = np.argpartition(-Sc[:, j], k - 1)[:k]
        orders.append(list(o[np.argsort(-Sc[o, j], kind="stable")]))
    if agg == "rrf":
        w = 1.0 / (rrf_k + np.arange(1, k + 1))
        for o in orders:
            out[np.asarray(o)] += w
        return out
    if agg == "greedy":
        # one distinct candidate per mask, most confident mask picking first
        strength = np.argsort(-Sc.max(0))
        pos = [0] * m
        taken, rung = set(), 0.0
        while len(taken) < k:
            moved = False
            for j in strength:
                while pos[j] < len(orders[j]) and orders[j][pos[j]] in taken:
                    pos[j] += 1
                if pos[j] >= len(orders[j]):
                    continue
                c = orders[j][pos[j]]
                pos[j] += 1
                rung -= 1.0
                out[c] = rung
                taken.add(c)
                moved = True
                if len(taken) >= k:
                    break
            if not moved:
                break
        if taken:                       # everything unpicked sorts below every pick
            rest = Sc.max(1).astype(np.float64)
            left = np.ones(n, dtype=bool)
            left[np.fromiter(taken, dtype=np.int64, count=len(taken))] = False
            if left.any():
                out[left] = rung - 1.0 + R.zscore(rest[left]) * 1e-3
        return out
    raise ValueError(agg)


def _finish(col, ix, qid, drop, topn):
    ex = drop.get(qid)
    if ex is not None and ex in ix.pos:
        col = col.copy()
        col[ix.pos[ex]] = -np.inf
    n = int(min(topn, len(col)))
    top = np.argpartition(-col, n - 1)[:n]
    top = top[np.argsort(-col[top], kind="stable")]
    return [ix.ids[i] for i in top], col[top].astype(np.float32)


# ------------------------------------------------------ citation-context ranking -----
def rank_cite(split, ix, W, scheme, qtf, agg="greedy", beta=1.0, window=60,
              topn=TOPN, max_ctx=0, report_every=0):
    from scipy import sparse
    drop = R.self_excluded(split)
    rank, score, noctx = {}, {}, 0
    t0 = time.time()
    qs = D.queries(split)
    for i, r in enumerate(qs):
        text = " ".join(r["sents"])
        whole = ix.to_docs(
            np.asarray((W @ ix.qvec(text, scheme, qtf)).todense(), dtype=np.float32),
            "max").ravel()
        ctxs = mask_contexts(text, window)
        if max_ctx:
            ctxs = ctxs[:max_ctx]
        if ctxs:
            Qm = sparse.hstack([ix.qvec(" ".join(c), scheme, qtf) for c in ctxs]).tocsc()
            Sc = ix.to_docs(np.asarray((W @ Qm).todense(), dtype=np.float32), "max")
            cite = aggregate(Sc, agg, topn)
        else:
            cite, noctx = whole, noctx + 1      # no masks: the whole document is all we have
        col = (R.zscore(cite) if beta >= 1.0
               else (1 - beta) * R.zscore(whole) + beta * R.zscore(cite))
        rank[r["id"]], score[r["id"]] = _finish(col.astype(np.float32), ix, r["id"],
                                                drop, topn)
        if report_every and i and i % report_every == 0:
            log(f"    {i}/{len(qs)} {split} queries ({time.time() - t0:.0f}s)")
    if noctx:
        log(f"    {noctx}/{len(qs)} {split} queries had no usable mask context")
    return rank, score


def cmd_cite(a):
    tag = a.tag or "cite"
    sel = a.select
    ixs = {}

    def index(split):
        if split not in ixs:
            ixs[split] = R.Lexical(split, max_df=a.max_df, passages=a.passages,
                                   stride=a.stride)
        return ixs[split]

    cfg = dict(agg=a.agg, beta=a.beta, window=a.window, qtf=a.qtf, max_df=a.max_df,
               passages=a.passages, stride=a.stride)
    if a.sweep:
        ix = index(sel)
        W = ix.weights(a.scheme, a.k1, a.b)
        base, _ = R.rank_lexical(sel, a.scheme, a.k1, a.b, a.qtf, a.max_df,
                                 topn=a.topn, index=ix)
        bs = R.evaluate(base, sel)
        R.report(f"whole-document baseline on {sel}", bs)
        best = (bs["f1"], None)
        for agg in [x for x in a.agg_sweep.split(",") if x]:
            for beta in [float(x) for x in a.beta_sweep.split(",") if x]:
                for win in [int(x) for x in a.window_sweep.split(",") if x]:
                    rk, _ = rank_cite(sel, ix, W, a.scheme, a.qtf, agg, beta, win,
                                      topn=a.topn)
                    st = R.evaluate(rk, sel)
                    R.report(f"cite {agg} beta={beta} win={win}", st)
                    if st["f1"] > best[0]:
                        best = (st["f1"], dict(agg=agg, beta=beta, window=win,
                                               qtf=a.qtf, max_df=a.max_df,
                                               passages=a.passages, stride=a.stride))
        if best[1] is None:
            log(f"  citation contexts never beat whole-document matching on {sel} "
                f"({bs['f1']:.4f}). Building the default config anyway so it is reported.")
        else:
            cfg = best[1]
            log(f"  BEST cite: {cfg}  {sel} f1 {best[0]:.4f} "
                f"(whole-document {bs['f1']:.4f}, {best[0] - bs['f1']:+.4f})")
        D.atomic_json(f"{P7}/results/cite_sweep.json",
                      dict(chose=cfg, select=sel, f1=best[0], whole_doc_f1=bs["f1"]))

    meta = dict(tag=tag, kind=a.kind, scheme=a.scheme, config=cfg, topn=a.topn,
                selected_on=sel)
    for split in a.splits.split(","):
        ix = index(split)
        W = ix.weights(a.scheme, a.k1, a.b)
        rk, sc = rank_cite(split, ix, W, a.scheme, cfg["qtf"], cfg["agg"], cfg["beta"],
                           cfg["window"], topn=a.topn, report_every=200)
        R.save_run(tag, split, rk, sc)
        if split != "test":
            st = R.evaluate(rk, split)
            meta[f"{split}_scores"] = {k: v for k, v in st.items() if k != "per_query"}
            R.report(f"{tag}/{split}", st)
        ixs.pop(split, None)                     # a train passage index is 150MB+
    R.save_run(tag, "meta", {}, None, meta)
    return meta


# --------------------------------------------------------------- the graphs ----------
def lex_graph(ix, k=20, block=256):
    """Top-k tf-idf-cosine neighbours of every candidate.  Rows of the tfidf weight
    matrix are already L2-normalised, so the product IS the cosine."""
    from scipy import sparse
    if ix.parent is not None:
        raise SystemExit("prop needs a document index, not a passage index")
    X = ix.weights("tfidf")
    n = X.shape[0]
    kk = int(min(k, max(n - 1, 1)))
    rows, cols, vals = [], [], []
    for s in range(0, n, block):
        B = np.asarray((X[s:s + block] @ X.T).todense(), dtype=np.float32)
        for i in range(B.shape[0]):
            B[i, s + i] = -1.0                              # no self-loop
            o = np.argpartition(-B[i], kk - 1)[:kk]
            rows.extend([s + i] * kk)
            cols.extend(o.tolist())
            vals.extend(B[i, o].tolist())
    G = sparse.csr_matrix((np.clip(np.asarray(vals, dtype=np.float32), 0, None),
                           (rows, cols)), shape=(n, n))
    log(f"    lex graph: {n} nodes, top-{kk} neighbours, {G.nnz} edges, "
        f"mean weight {float(G.data.mean()):.4f}")
    return G, np.ones(n, dtype=np.float64)


def cocite_graph(ix):
    """Co-citation counts from the TRAIN gold labels, over this split's pool."""
    from scipy import sparse
    pos, n = ix.pos, len(ix.ids)
    rows, cols = [], []
    deg = np.zeros(n, dtype=np.float64)
    gold = D.golds("train")
    covered = set()
    for q, gs in gold.items():
        idx = sorted({pos[c] for c in gs if c in pos})
        covered.update(idx)
        if len(idx) < 2:
            continue
        for i in idx:
            deg[i] += 1.0
            for j in idx:
                if i != j:
                    rows.append(i)
                    cols.append(j)
    if not rows:
        raise SystemExit("cocite: no train gold shares this split's pool")
    C = sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                          shape=(n, n))
    C.sum_duplicates()
    log(f"    cocite graph: {n} nodes, {len(covered)} ({100.0 * len(covered) / n:.0f}%) "
        f"of the pool appear in a train gold set, {C.nnz} edges")
    return C, deg


def gmul(G, deg, seed, loo=None):
    """G @ seed, with an EXACT leave-one-out of `loo`'s own co-citation pairs so a train
    query is not scored by a graph that contains its own answer key."""
    b = np.asarray(G @ seed, dtype=np.float64).ravel()
    d = deg.astype(np.float64, copy=True)
    if loo is not None and len(loo):
        li = np.asarray(sorted(loo), dtype=np.int64)
        b[li] -= (float(seed[li].sum()) - seed[li])
        d[li] -= 1.0
    return b / np.sqrt(np.maximum(d, 1.0))


def rank_prop(split, ix, W, scheme, k1, b, qtf, G, deg, alpha, seeds, topn=TOPN,
              loo_map=None):
    from scipy import sparse
    drop = R.self_excluded(split)
    rank, score = {}, {}
    for r in D.queries(split):
        col = ix.to_docs(
            np.asarray((W @ ix.qvec(" ".join(r["sents"]), scheme, qtf)).todense(),
                       dtype=np.float32), "max").ravel()
        s = R.zscore(col)
        m = int(min(seeds, len(s)))
        seed = np.zeros_like(s)
        top = np.argpartition(-s, m - 1)[:m]
        seed[top] = np.clip(s[top], 0.0, None)
        boost = gmul(G, deg, seed, (loo_map or {}).get(r["id"]))
        if boost.std() > 1e-12:
            boost = R.zscore(boost)
        rank[r["id"]], score[r["id"]] = _finish((s + alpha * boost).astype(np.float32),
                                                ix, r["id"], drop, topn)
    return rank, score


def cmd_prop(a):
    tag = a.tag or f"prop_{a.graph}"
    sel = a.select
    cfg = dict(graph=a.graph, alpha=a.alpha, seeds=a.seeds, neighbours=a.neighbours,
               qtf=a.qtf, max_df=a.max_df)

    def graph_for(split, ix):
        """-> (G, deg, loo_map).  loo_map is only non-empty where the graph could leak."""
        if a.graph == "lex":
            G, deg = lex_graph(ix, a.neighbours)
            return G, deg, None                 # no labels went into it, nothing to leak
        G, deg = cocite_graph(ix)
        loo = ({q: [ix.pos[c] for c in gs if c in ix.pos]
                for q, gs in D.golds("train").items()} if split == "train" else None)
        return G, deg, loo

    meta = dict(tag=tag, kind=a.kind, scheme=a.scheme, config=cfg, topn=a.topn,
                selected_on=sel)
    if a.sweep:
        ix = R.Lexical(sel, max_df=a.max_df)
        W = ix.weights(a.scheme, a.k1, a.b)
        base, _ = R.rank_lexical(sel, a.scheme, a.k1, a.b, a.qtf, a.max_df,
                                 topn=a.topn, index=ix)
        bs = R.evaluate(base, sel)
        R.report(f"no propagation on {sel}", bs)
        G, deg, loo = graph_for(sel, ix)
        best = (bs["f1"], None)
        for alpha in [float(x) for x in a.alpha_sweep.split(",") if x]:
            for seeds in [int(x) for x in a.seed_sweep.split(",") if x]:
                rk, _ = rank_prop(sel, ix, W, a.scheme, a.k1, a.b, a.qtf, G, deg,
                                  alpha, seeds, topn=a.topn, loo_map=loo)
                st = R.evaluate(rk, sel)
                R.report(f"prop {a.graph} alpha={alpha} seeds={seeds}", st)
                if st["f1"] > best[0]:
                    best = (st["f1"], dict(cfg, alpha=alpha, seeds=seeds))
        if best[1] is None:
            log(f"  propagation never beat the un-propagated run on {sel} "
                f"({bs['f1']:.4f}). Building the default config anyway.")
        else:
            cfg = best[1]
            log(f"  BEST prop: alpha={cfg['alpha']} seeds={cfg['seeds']}  "
                f"{sel} f1 {best[0]:.4f} (no prop {bs['f1']:.4f}, "
                f"{best[0] - bs['f1']:+.4f})")
        D.atomic_json(f"{P7}/results/prop_{a.graph}_sweep.json",
                      dict(chose=cfg, select=sel, f1=best[0], no_prop_f1=bs["f1"]))
        del ix, G

    for split in a.splits.split(","):
        ix = R.Lexical(split, max_df=cfg["max_df"])
        W = ix.weights(a.scheme, a.k1, a.b)
        G, deg, loo = graph_for(split, ix)
        rk, sc = rank_prop(split, ix, W, a.scheme, a.k1, a.b, cfg["qtf"], G, deg,
                           cfg["alpha"], cfg["seeds"], topn=a.topn, loo_map=loo)
        R.save_run(tag, split, rk, sc)
        if split != "test":
            st = R.evaluate(rk, split)
            meta[f"{split}_scores"] = {k: v for k, v in st.items() if k != "per_query"}
            R.report(f"{tag}/{split}", st)
        del ix, G
    R.save_run(tag, "meta", {}, None, meta)
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("masks")
    p.add_argument("--splits", default="train,devx,test")
    p.set_defaults(fn=cmd_masks)

    def shared(p):
        p.add_argument("--tag", default=None)
        p.add_argument("--kind", default="lexical", choices=list(D.KINDS))
        p.add_argument("--sweep", action="store_true")
        p.add_argument("--select", default="train", choices=["train", "devx"],
                       help="827 labelled train queries beat 118 devx ones as a "
                            "selection set, and nothing here learns from labels")
        p.add_argument("--scheme", default="tfidf", choices=["tfidf", "bm25"])
        p.add_argument("--k1", type=float, default=1.2)
        p.add_argument("--b", type=float, default=0.75)
        p.add_argument("--qtf", default="sqrt", choices=["raw", "binary", "log", "sqrt"])
        p.add_argument("--max-df", type=float, default=0.2)
        # Fixing BOTH sides of the retrieval unit at once: mask-anchored windows on the
        # query side, passages on the candidate side.  MaxP bought 20 hits on the
        # candidate side alone, and the gate says the query side is the weaker of the two
        # (one citation, one sentence, 6,000 words of other things).
        p.add_argument("--passages", type=int, default=0,
                       help="index candidate passages of this many words (0 = documents)")
        p.add_argument("--stride", type=int, default=None)
        p.add_argument("--splits", default="train,devx,test")
        p.add_argument("--topn", type=int, default=TOPN)
        return p

    p = shared(sp.add_parser("cite"))
    p.add_argument("--agg", default="greedy", choices=["max", "sum", "rrf", "greedy"])
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--window", type=int, default=60)
    p.add_argument("--agg-sweep", default="max,rrf,greedy,sum")
    p.add_argument("--beta-sweep", default="1.0,0.7,0.4")
    p.add_argument("--window-sweep", default="60")
    p.set_defaults(fn=cmd_cite)

    p = shared(sp.add_parser("prop"))
    p.add_argument("--graph", default="lex", choices=["lex", "cocite"])
    p.add_argument("--neighbours-block", type=int, default=256)
    p.add_argument("--alpha", type=float, default=0.3)
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--neighbours", type=int, default=20)
    p.add_argument("--alpha-sweep", default="0.15,0.3,0.6")
    p.add_argument("--seed-sweep", default="10,20,50")
    p.set_defaults(fn=cmd_prop)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
