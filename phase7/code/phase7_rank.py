#!/usr/bin/env python3
"""CS-781 Phase 7 -- lexical retrieval, fusion and submissions.  CPU only.

    python phase7_rank.py bm25  --sweep              sweep k1/b/qtf/max-df on devx
    python phase7_rank.py bm25  --tag bm25           best config -> dev + test rankings
    python phase7_rank.py tfidf --tag tfidf
    python phase7_rank.py fuse  --tag fuse --members bm25,tfidf,dense_m2
    python phase7_rank.py zip   --tag bm25 [--kind lexical]
    python phase7_rank.py show                       every finished system, ranked

Each system writes results/<tag>/{dev_rank.json, test_rank.json, meta.json}, where a
rank file is {query_id: [top-N candidate IDs]} with N=100 by default.  Everything
downstream -- fusion, the neural reranker, the KD student -- reads that one format, so
a member can be added without touching the consumers (the Phase-5/6 ensemble pattern).

Why BM25 first.  The objective is total hits in the top 7 (see phase7_data), the pool
is thousands of long judgments, and a Phase-1 trunk has never been trained to put two
documents near each other -- its cosine similarity is dominated by length and register.
BM25 over idf-weighted rare legal terms (section numbers, party names, statute cites) is
the strong baseline here, and every neural system in this phase is measured against it,
not against random.  Reported as kind='lexical'; any blend with a neural scorer is
reported as kind='hybrid' beside the model-only number, as the brief requires.

A query that is also a candidate (by ID, or by identical text) is dropped from its own
ranking: retrieving the query itself is not a prior case.
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
import phase7_data as D                                      # noqa: E402

P7 = D.P7
TOPN = int(os.environ.get("P7_TOPN", 100))


def log(*a):
    s = " ".join(str(x) for x in a)
    print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)


# ------------------------------------------------------------- the index -----
class Lexical:
    """Sparse term-document index over one split's candidate pool.

    tf is built once; BM25 and TF-IDF are two re-weightings of the same CSR data
    array, so a parameter sweep costs one vectorised pass instead of a rebuild."""

    def __init__(self, split, max_df=1.0, min_len=2, passages=0, stride=None):
        """passages>0 indexes fixed-width word windows instead of whole judgments.

        A PCR query cites a prior case because of one passage of it, but the pool's
        documents average 6,383 words: whole-document BM25 divides that evidence by the
        length of everything else in the judgment.  Indexing passages and taking the
        maximum over a document's passages (MaxP) scores a document by its single best
        matching span, which is what the citation decision actually rests on.  df, idf
        and avgdl all become passage-level, which is the standard MaxP formulation."""
        from scipy import sparse
        self.split = split
        pool = D.candidates(split)
        self.ids = sorted(pool)
        self.pos = {c: i for i, c in enumerate(self.ids)}
        self.passages = int(passages or 0)
        units = []                              # (doc index, words) -- one matrix row
        for i, c in enumerate(self.ids):
            w = D.WORD.findall(" ".join(pool[c]).lower())
            if self.passages and len(w) > self.passages:
                st = int(stride or self.passages)
                for a0 in range(0, len(w), st):
                    seg = w[a0:a0 + self.passages]
                    if a0 == 0 or len(seg) >= max(32, self.passages // 4):
                        units.append((i, seg))
            else:
                units.append((i, w))
        vocab, rows, cols, vals, lens = {}, [], [], [], []
        for r, (_, w) in enumerate(units):
            cnt = {}
            for t in w:
                if len(t) >= min_len:
                    cnt[t] = cnt.get(t, 0) + 1
            lens.append(len(w))
            for t, n in cnt.items():
                j = vocab.setdefault(t, len(vocab))
                rows.append(r)
                cols.append(j)
                vals.append(n)
        # rows are emitted in document order, so one reduceat aggregates each document's
        # passages without a scatter.
        self.parent = (np.asarray([u[0] for u in units], dtype=np.int64)
                       if self.passages else None)
        self.starts = (np.searchsorted(self.parent, np.arange(len(self.ids)))
                       if self.parent is not None else None)
        self.vocab = vocab
        self.dl = np.asarray(lens, dtype=np.float32)
        self.avgdl = float(self.dl.mean()) if len(self.dl) else 1.0
        self.tf = sparse.csr_matrix(
            (np.asarray(vals, dtype=np.float32), (rows, cols)),
            shape=(len(units), len(vocab)))
        self.df = np.asarray((self.tf > 0).sum(0)).ravel().astype(np.float32)
        self.N = len(units)
        self.keep = self.df <= max_df * self.N          # drop near-universal terms
        log(f"  index {split}: {len(self.ids)} docs"
            + (f" in {self.N} passages of {self.passages}w" if self.passages else "")
            + f", {len(vocab)} terms, {self.tf.nnz / 1e6:.1f}M postings, "
              f"avgdl {self.avgdl:.0f}, "
              f"{int((~self.keep).sum())} terms dropped at max_df={max_df}")

    def to_docs(self, S, agg="max"):
        """(rows x B) passage scores -> (docs x B).  A no-op on a document index."""
        if self.parent is None:
            return S
        if agg == "mean":
            tot = np.add.reduceat(S, self.starts, axis=0)
            cnt = np.diff(np.append(self.starts, self.N)).astype(np.float32)
            return tot / np.maximum(cnt, 1.0)[:, None]
        return np.maximum.reduceat(S, self.starts, axis=0)

    def weights(self, scheme, k1=1.2, b=0.75):
        """Document-side weight matrix W (CSR, docs x terms).  score = W @ qvec."""
        from scipy import sparse
        tf = self.tf
        if scheme == "bm25":
            idf = np.log(1 + (self.N - self.df + 0.5) / (self.df + 0.5))
            norm = k1 * (1 - b + b * self.dl / max(self.avgdl, 1e-9))
            data = tf.data.copy()
            rowidx = np.repeat(np.arange(tf.shape[0]), np.diff(tf.indptr))
            data = data * (k1 + 1) / (data + norm[rowidx])
            data = data * idf[tf.indices]
        elif scheme == "tfidf":
            idf = np.log((1 + self.N) / (1 + self.df)) + 1.0
            data = (1 + np.log(tf.data)) * idf[tf.indices]
        else:
            raise ValueError(scheme)
        data = data * self.keep[tf.indices]
        W = sparse.csr_matrix((data.astype(np.float32), tf.indices.copy(), tf.indptr.copy()),
                              shape=tf.shape)
        if scheme == "tfidf":                            # cosine: L2-normalise the rows
            n = np.sqrt(np.asarray(W.multiply(W).sum(1)).ravel())
            W = sparse.diags(1.0 / np.maximum(n, 1e-9)).dot(W).tocsr()
        return W

    def qvec(self, text, scheme, qtf="raw"):
        """Query-side sparse vector.  qtf matters here: a PCR query is a whole
        judgment, so raw counts let one repeated word dominate the score."""
        from scipy import sparse
        cnt = {}
        for t in D.WORD.findall(text.lower()):
            j = self.vocab.get(t)
            if j is not None:
                cnt[j] = cnt.get(j, 0) + 1
        if not cnt:
            return sparse.csr_matrix((len(self.vocab), 1), dtype=np.float32)
        idx = np.fromiter(cnt.keys(), dtype=np.int32, count=len(cnt))
        n = np.fromiter(cnt.values(), dtype=np.float32, count=len(cnt))
        if qtf == "binary":
            v = np.ones_like(n)
        elif qtf == "log":
            v = 1 + np.log(n)
        elif qtf == "sqrt":
            v = np.sqrt(n)
        else:
            v = n
        if scheme == "tfidf":
            idf = np.log((1 + self.N) / (1 + self.df[idx])) + 1.0
            v = v * idf
            v = v / max(float(np.sqrt((v * v).sum())), 1e-9)
        return sparse.csr_matrix((v, (idx, np.zeros_like(idx))),
                                 shape=(len(self.vocab), 1), dtype=np.float32)


def self_excluded(split):
    """{query_id: candidate_id} for queries that ARE a candidate in their own pool."""
    pool = D.candidates(split)
    bykey = {D.norm_key(v): c for c, v in pool.items()}
    out = {}
    for r in D.queries(split):
        if r["id"] in pool:
            out[r["id"]] = r["id"]
        else:
            c = bykey.get(D.norm_key(r["sents"]))
            if c:
                out[r["id"]] = c
    return out


def rank_lexical(split, scheme, k1, b, qtf, max_df, topn=TOPN, index=None, batch=64,
                 agg="max", passages=0, stride=None):
    """-> ({qid: [top-n ids]}, {qid: np.array of their scores})."""
    ix = (index if index is not None
          else Lexical(split, max_df=max_df, passages=passages, stride=stride))
    W = ix.weights(scheme, k1, b)
    drop = self_excluded(split)
    qs = D.queries(split)
    rank, score = {}, {}
    from scipy import sparse
    for s in range(0, len(qs), batch):
        chunk = qs[s:s + batch]
        Qm = sparse.hstack([ix.qvec(" ".join(r["sents"]), scheme, qtf) for r in chunk]).tocsc()
        S = np.asarray((W @ Qm).todense(), dtype=np.float32)        # rows x batch
        S = ix.to_docs(S, agg)                                      # docs x batch
        for j, r in enumerate(chunk):
            col = S[:, j]
            ex = drop.get(r["id"])
            if ex is not None and ex in ix.pos:
                col = col.copy()
                col[ix.pos[ex]] = -np.inf
            n = min(topn, len(col))
            top = np.argpartition(-col, n - 1)[:n]
            top = top[np.argsort(-col[top], kind="stable")]
            rank[r["id"]] = [ix.ids[i] for i in top]
            score[r["id"]] = col[top].astype(np.float32)
    return rank, score


# -------------------------------------------------------------- bookkeeping --
def rank_dir(tag):
    return f"{P7}/results/{tag}"


def save_run(tag, split, rank, score=None, meta=None):
    d = rank_dir(tag)
    if split != "meta":
        D.atomic_json(f"{d}/{split}_rank.json", {q: list(v) for q, v in rank.items()}, indent=0)
    if score is not None:
        os.makedirs(d, exist_ok=True)
        np.savez_compressed(f"{d}/{split}_score.npz",
                            **{q: np.asarray(v, dtype=np.float32) for q, v in score.items()})
    if meta is not None:
        old = {}
        if os.path.exists(f"{d}/meta.json"):
            with open(f"{d}/meta.json") as f:
                old = json.load(f)
        D.atomic_json(f"{d}/meta.json", {**old, **meta})


def load_run(tag, split):
    f = f"{rank_dir(tag)}/{split}_rank.json"
    if not os.path.exists(f):
        return None, None
    with open(f) as fh:
        rank = json.load(fh)
    sf = f"{rank_dir(tag)}/{split}_score.npz"
    score = None
    if os.path.exists(sf):
        z = np.load(sf)
        score = {q: z[q] for q in z.files}
    return rank, score


def evaluate(rank, split="devx"):
    """The official number on our own labelled split, plus the recall curve."""
    g, gp = D.golds(split), D.golds_in_pool(split)
    rank = {q: rank[q] for q in g if q in rank}
    s = D.score_ranking(rank, g, reachable=gp)
    s["recall_curve"] = D.recall_at(rank, gp)
    s.pop("per_query", None)
    return s


def report(tag, s, extra=""):
    log(f"  {tag:22s} {D.brief(s)} | ceiling {s['ceiling_f1']:.4f} | "
        f"r@7 {s['recall_curve']['r@7']:.3f} "
        f"r@{max(int(k.split('@')[1]) for k in s['recall_curve'])} "
        f"{max(s['recall_curve'].values()):.3f} {extra}")


# -------------------------------------------------------------- the commands --
SWEEP = dict(
    k1=[0.9, 1.2, 1.6],
    b=[0.4, 0.75, 0.9],
    qtf=["binary", "log", "raw"],
    max_df=[1.0, 0.5, 0.2],
)


def cmd_bm25(a):
    scheme = a.scheme
    if a.sweep:
        best, rows = None, []
        k1s = SWEEP["k1"] if scheme == "bm25" else [0.0]
        bs = SWEEP["b"] if scheme == "bm25" else [0.0]
        for max_df in SWEEP["max_df"]:
            ix = Lexical("devx", max_df=max_df)
            for k1 in k1s:
                for b in bs:
                    for qtf in SWEEP["qtf"]:
                        r, _ = rank_lexical("devx", scheme, k1, b, qtf, max_df,
                                            topn=TOPN, index=ix)
                        s = evaluate(r, "devx")
                        rows.append(dict(k1=k1, b=b, qtf=qtf, max_df=max_df,
                                         f1=round(s["f1"], 6), hits=s["hits"],
                                         r_at_100=s["recall_curve"].get("r@100")))
                        report(f"{scheme} k1={k1} b={b} {qtf} df<={max_df}", s)
                        if best is None or s["f1"] > best[0]:
                            best = (s["f1"], dict(k1=k1, b=b, qtf=qtf, max_df=max_df), s)
        D.atomic_json(f"{P7}/results/{scheme}_sweep.json",
                      dict(rows=rows, chose=best[1], dev_f1=best[0]))
        log(f"  BEST {scheme}: {best[1]}  devx f1 {best[0]:.4f}  hits {best[2]['hits']}")
        return best[1]
    cfg = dict(k1=a.k1, b=a.b, qtf=a.qtf, max_df=a.max_df)
    f = f"{P7}/results/{scheme}_sweep.json"
    if a.use_sweep and os.path.exists(f):
        with open(f) as fh:
            cfg = json.load(fh)["chose"]
        log(f"  using the swept config {cfg}")
    tag = a.tag or scheme
    meta = dict(tag=tag, kind="lexical", scheme=scheme, config=cfg, topn=a.topn)
    for split in a.splits.split(","):
        r, sc = rank_lexical(split, scheme, cfg["k1"], cfg["b"], cfg["qtf"], cfg["max_df"],
                             topn=a.topn)
        save_run(tag, split, r, sc)
        if split != "test":
            s = evaluate(r, split)
            meta[f"{split}_scores"] = {k: v for k, v in s.items() if k != "per_query"}
            report(f"{tag}/{split}", s)
    save_run(tag, "meta", {}, None, meta)
    return cfg


# ------------------------------------------------- MaxP and RM3 query expansion -----
def _lexrun(tag, kind, scheme, cfg, splits, topn, passages=0, agg="max", rm3=None,
            stride=None):
    """One lexical system over several splits -> results/<tag>/.  Shared by maxp/rm3."""
    meta = dict(tag=tag, kind=kind, scheme=scheme,
                config=dict(**cfg, passages=passages, stride=stride, agg=agg,
                            **(rm3 or {})), topn=topn)
    for split in splits:
        if rm3:
            r, sc = rank_rm3(split, scheme, cfg["k1"], cfg["b"], cfg["qtf"], cfg["max_df"],
                             topn=topn, passages=passages, agg=agg, stride=stride, **rm3)
        else:
            r, sc = rank_lexical(split, scheme, cfg["k1"], cfg["b"], cfg["qtf"],
                                 cfg["max_df"], topn=topn, passages=passages, agg=agg,
                                 stride=stride)
        save_run(tag, split, r, sc)
        if split != "test":
            st = evaluate(r, split)
            meta[f"{split}_scores"] = {k: v for k, v in st.items() if k != "per_query"}
            report(f"{tag}/{split}", st)
    save_run(tag, "meta", {}, None, meta)
    return meta


def rm3_expand(ix, W, q, scheme, fb_docs, fb_terms, lam):
    """RM3: a relevance model built from the first pass's top documents, interpolated
    with the original query.

    Why it could help here, when expansion usually does nothing for long queries.  A PCR
    query IS a judgment, so it already carries hundreds of terms and expansion normally
    only adds drift.  But the gate measured the lexical overlap between a query and its
    golds at 1.86x idf-Jaccard, which is very weak -- the signature of vocabulary
    mismatch, and that is precisely what a relevance model addresses: the terms shared by
    the top-ranked judgments are the terms of the legal ISSUE rather than of this case's
    own facts.  Both vectors are L2-normalised before mixing, which leaves BM25's ranking
    unchanged (a per-query constant) and makes lam comparable across schemes."""
    from scipy import sparse
    col = np.asarray((W @ q).todense(), dtype=np.float32).ravel()
    n = int(max(min(fb_docs, col.size), 1))
    rows = np.argpartition(-col, n - 1)[:n]
    w = col[rows].astype(np.float64)
    w = np.clip(w - w.min(), 0.0, None)
    w = w / w.sum() if w.sum() > 1e-12 else np.full(n, 1.0 / n)
    sub = ix.tf[rows]                                   # n x V raw counts
    rm = np.asarray((sparse.diags(w / np.maximum(ix.dl[rows], 1.0)) @ sub).sum(0),
                    dtype=np.float64).ravel()
    if scheme == "tfidf":
        rm *= np.log((1 + ix.N) / (1 + ix.df)) + 1.0
    else:
        rm *= np.log(1 + (ix.N - ix.df + 0.5) / (ix.df + 0.5))
    rm *= ix.keep
    if fb_terms and fb_terms < rm.size:
        rm[np.argpartition(-rm, fb_terms)[fb_terms:]] = 0.0
    qd = np.asarray(q.todense(), dtype=np.float64).ravel()
    nq, nr = float(np.linalg.norm(qd)), float(np.linalg.norm(rm))
    v = (1 - lam) * qd / max(nq, 1e-9) + lam * rm / max(nr, 1e-9)
    return sparse.csr_matrix(v.reshape(-1, 1).astype(np.float32))


def rank_rm3(split, scheme, k1, b, qtf, max_df, fb_docs=10, fb_terms=50, lam=0.4,
             topn=TOPN, index=None, passages=0, agg="max", stride=None):
    ix = (index if index is not None
          else Lexical(split, max_df=max_df, passages=passages, stride=stride))
    W = ix.weights(scheme, k1, b)
    drop = self_excluded(split)
    rank, score = {}, {}
    for r in D.queries(split):
        q = ix.qvec(" ".join(r["sents"]), scheme, qtf)
        q2 = rm3_expand(ix, W, q, scheme, fb_docs, fb_terms, lam)
        col = ix.to_docs(np.asarray((W @ q2).todense(), dtype=np.float32), agg).ravel()
        ex = drop.get(r["id"])
        if ex is not None and ex in ix.pos:
            col = col.copy()
            col[ix.pos[ex]] = -np.inf
        n = min(topn, len(col))
        top = np.argpartition(-col, n - 1)[:n]
        top = top[np.argsort(-col[top], kind="stable")]
        rank[r["id"]] = [ix.ids[i] for i in top]
        score[r["id"]] = col[top].astype(np.float32)
    return rank, score


def cmd_maxp(a):
    """Passage-level lexical retrieval, scored by a document's best passage.

    The first sweep offered {300, 500, 800, 1200} and chose 300 -- the BOUNDARY, which is
    the same shape as the original BM25 sweep landing on 3 of 4 boundaries, and widening
    that one was worth 19 hits. So this sweeps downward, and over overlap: non-overlapping
    windows dilute a gold passage that straddles a boundary, and a half-width stride
    catches it at the cost of twice the rows.

    Swept on devx rather than train, which is a deliberate exception to the train-side
    rule: a width-100 half-stride index over the 4,320-document train pool is ~690k rows
    and several hundred MB of postings per config, and MaxP's standalone effect (20 hits)
    is large enough for devx to resolve. The chosen config is still built and reported on
    train, so a config that only works on devx shows up in the log."""
    cfg = dict(k1=a.k1, b=a.b, qtf=a.qtf, max_df=a.max_df)
    tag = a.tag or f"maxp_{a.scheme}"
    if a.sweep:
        best, rows = None, []
        for pw in [int(x) for x in a.psweep.split(",") if x]:
            for sf in [float(x) for x in a.stride_sweep.split(",") if x]:
                st_words = max(int(round(pw * sf)), 1)
                ix = Lexical("devx", max_df=a.max_df, passages=pw, stride=st_words)
                for agg in ("max", "mean"):
                    r, _ = rank_lexical("devx", a.scheme, a.k1, a.b, a.qtf, a.max_df,
                                        topn=a.topn, index=ix, agg=agg)
                    sc = evaluate(r, "devx")
                    rows.append(dict(passages=pw, stride=st_words, agg=agg,
                                     f1=round(sc["f1"], 6), hits=sc["hits"],
                                     r_at_100=max(sc["recall_curve"].values())))
                    report(f"{a.scheme} w={pw} stride={st_words} {agg}", sc)
                    if best is None or sc["f1"] > best[0]:
                        best = (sc["f1"], pw, agg, st_words)
                del ix
        log(f"  BEST maxp: passages={best[1]} stride={best[3]} agg={best[2]} "
            f"devx f1 {best[0]:.4f}")
        edge = [x for x in (min, max) if x([int(v) for v in a.psweep.split(",") if v])
                == best[1]]
        if edge:
            log("  !! the chosen width is on the sweep BOUNDARY again -- widen --psweep")
        a.passages, a.agg, a.stride = best[1], best[2], best[3]
        D.atomic_json(f"{P7}/results/maxp_{a.scheme}_sweep.json",
                      dict(chose=dict(passages=best[1], agg=best[2], stride=best[3]),
                           dev_f1=best[0], rows=rows))
    return _lexrun(tag, a.kind, a.scheme, cfg, a.splits.split(","), a.topn,
                   passages=a.passages, agg=a.agg, stride=a.stride)


def cmd_rm3(a):
    cfg = dict(k1=a.k1, b=a.b, qtf=a.qtf, max_df=a.max_df)
    tag = a.tag or f"rm3_{a.scheme}"
    rm = dict(fb_docs=a.fb_docs, fb_terms=a.fb_terms, lam=a.fb_lambda)
    if a.sweep:
        ix = Lexical("devx", max_df=a.max_df, passages=a.passages)
        base, _ = rank_lexical("devx", a.scheme, a.k1, a.b, a.qtf, a.max_df,
                               topn=a.topn, index=ix, agg=a.agg)
        bs = evaluate(base, "devx")
        report(f"{a.scheme} no expansion", bs)
        best = (bs["f1"], None)
        for lam in [float(x) for x in a.lam_sweep.split(",") if x]:
            for fd in [int(x) for x in a.doc_sweep.split(",") if x]:
                for ft in [int(x) for x in a.term_sweep.split(",") if x]:
                    r, _ = rank_rm3("devx", a.scheme, a.k1, a.b, a.qtf, a.max_df,
                                    fb_docs=fd, fb_terms=ft, lam=lam, topn=a.topn,
                                    index=ix, agg=a.agg)
                    st = evaluate(r, "devx")
                    report(f"rm3 lam={lam} docs={fd} terms={ft}", st)
                    if st["f1"] > best[0]:
                        best = (st["f1"], dict(fb_docs=fd, fb_terms=ft, lam=lam))
        del ix
        if best[1] is None:
            log(f"  RM3 never beat the unexpanded run ({bs['f1']:.4f}); keeping "
                f"lam={a.fb_lambda} so the system is still built and reported")
        else:
            rm = best[1]
            log(f"  BEST rm3: {rm} devx f1 {best[0]:.4f} "
                f"(unexpanded {bs['f1']:.4f}, {best[0] - bs['f1']:+.4f})")
        D.atomic_json(f"{P7}/results/rm3_{a.scheme}_sweep.json",
                      dict(chose=rm, dev_f1=best[0], no_expansion_f1=bs["f1"]))
    return _lexrun(tag, a.kind, a.scheme, cfg, a.splits.split(","), a.topn,
                   passages=a.passages, agg=a.agg, rm3=rm,
                   stride=getattr(a, "stride", None))


def zscore(x):
    x = np.asarray(x, dtype=np.float64)
    sd = x.std()
    return (x - x.mean()) / sd if sd > 1e-9 else np.zeros_like(x)


def fuse_ranks(members, split, weights, mode="rrf", rrf_k=60, topn=TOPN):
    """Reciprocal-rank fusion (scale-free) or z-scored score fusion over members."""
    acc = {}
    for tag, w in zip(members, weights):
        rank, score = load_run(tag, split)
        if rank is None:
            continue
        for q, ids in rank.items():
            a = acc.setdefault(q, {})
            if mode == "rrf" or score is None or q not in score:
                for i, c in enumerate(ids):
                    a[c] = a.get(c, 0.0) + w / (rrf_k + i + 1)
            else:
                z = zscore(score[q][:len(ids)])
                for c, v in zip(ids, z):
                    a[c] = a.get(c, 0.0) + w * float(v)
    return {q: [c for c, _ in sorted(v.items(), key=lambda kv: -kv[1])[:topn]]
            for q, v in acc.items()}


def cmd_fuse(a):
    members = [m for m in a.members.split(",") if m]
    have = [m for m in members if os.path.exists(f"{rank_dir(m)}/devx_rank.json")
            or os.path.exists(f"{rank_dir(m)}/dev_rank.json")]
    if len(have) < 2:
        log(f"  fuse needs 2+ finished members; have {have}")
        return
    log(f"  fusing {have}")
    best = None
    grid = [1.0, 0.5, 2.0] if len(have) <= 3 else [1.0]
    for mode in ("rrf", "z"):
        for w0 in grid:
            weights = [w0] + [1.0] * (len(have) - 1)
            r = fuse_ranks(have, "devx", weights, mode=mode, topn=a.topn)
            s = evaluate(r, "devx")
            report(f"{mode} w0={w0}", s)
            if best is None or s["f1"] > best[0]:
                best = (s["f1"], mode, weights, s)
    _, mode, weights, s = best
    log(f"  BEST fusion: {mode} weights {weights} devx f1 {best[0]:.4f}")
    tag = a.tag or "fuse"
    meta = dict(tag=tag, kind=a.kind, members=have, mode=mode, weights=weights,
                devx_scores={k: v for k, v in s.items() if k != "per_query"})
    for split in a.splits.split(","):
        r = fuse_ranks(have, split, weights, mode=mode, topn=a.topn)
        if r:
            save_run(tag, split, r)
    save_run(tag, "meta", {}, None, meta)
    report(f"{tag}/devx", s)


def cmd_zip(a):
    rank, _ = load_run(a.tag, "test")
    if rank is None:
        raise SystemExit(f"no test ranking for {a.tag} (run it with --splits ...,test)")
    meta = {}
    if os.path.exists(f"{rank_dir(a.tag)}/meta.json"):
        with open(f"{rank_dir(a.tag)}/meta.json") as f:
            meta = json.load(f)
    dev = meta.get("devx_scores") or meta.get("dev_scores") or {}
    D.write_submission(a.tag, rank, note=a.note or json.dumps(meta.get("config", {})),
                       kind=a.kind or meta.get("kind", "lexical"),
                       dev_scores={k: dev.get(k) for k in ("f1", "hits", "precision", "recall")
                                   if k in dev})


def cmd_show(a):
    import glob
    rows = []
    for m in sorted(glob.glob(f"{P7}/results/*/meta.json")):
        with open(m) as f:
            d = json.load(f)
        sc = d.get("devx_scores") or d.get("dev_scores") or {}
        rows.append((sc.get("f1") or 0.0, d.get("tag", os.path.basename(os.path.dirname(m))),
                     d.get("kind", "?"), sc.get("hits"), sc.get("max_hits"),
                     (sc.get("recall_curve") or {}).get("r@100")))
    log(f"{'tag':24s} {'kind':9s} {'devx f1':>9s} {'hits':>12s} {'r@100':>7s}")
    for f1, tag, kind, h, mh, r100 in sorted(rows, reverse=True):
        log(f"{tag:24s} {kind:9s} {f1:9.4f} {str(h) + '/' + str(mh):>12s} "
            f"{(r100 if r100 is not None else float('nan')):7.3f}")
    up = sorted(__import__("glob").glob(f"{P7}/submissions/_upload/*.zip"))
    log(f"upload-ready zips: {len(up)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name, scheme in (("bm25", "bm25"), ("tfidf", "tfidf")):
        p = sp.add_parser(name)
        p.add_argument("--tag", default=None)
        p.add_argument("--sweep", action="store_true")
        p.add_argument("--use-sweep", action="store_true",
                       help="read the config chosen by a previous --sweep")
        p.add_argument("--k1", type=float, default=1.2)
        p.add_argument("--b", type=float, default=0.75)
        p.add_argument("--qtf", default="log", choices=["raw", "binary", "log", "sqrt"])
        p.add_argument("--max-df", type=float, default=1.0)
        p.add_argument("--splits", default="devx,test")
        p.add_argument("--topn", type=int, default=TOPN)
        p.set_defaults(fn=cmd_bm25, scheme=scheme)
    for name, scheme in (("maxp", "bm25"), ("maxp_tfidf", "tfidf")):
        p = sp.add_parser(name)
        p.add_argument("--tag", default=None)
        p.add_argument("--kind", default="lexical")
        p.add_argument("--sweep", action="store_true")
        p.add_argument("--psweep", default="80,120,180,250,350")
        p.add_argument("--stride-sweep", default="1.0,0.5",
                       help="stride as a fraction of the width; 0.5 = half-width overlap")
        p.add_argument("--stride", type=int, default=None)
        p.add_argument("--passages", type=int, default=180)
        p.add_argument("--agg", default="max", choices=["max", "mean"])
        p.add_argument("--k1", type=float, default=1.2)
        p.add_argument("--b", type=float, default=0.75)
        p.add_argument("--qtf", default="sqrt", choices=["raw", "binary", "log", "sqrt"])
        p.add_argument("--max-df", type=float, default=0.2)
        p.add_argument("--splits", default="devx,test")
        p.add_argument("--topn", type=int, default=TOPN)
        p.set_defaults(fn=cmd_maxp, scheme=scheme)
    for name, scheme in (("rm3", "tfidf"), ("rm3_bm25", "bm25")):
        p = sp.add_parser(name)
        p.add_argument("--tag", default=None)
        p.add_argument("--kind", default="lexical")
        p.add_argument("--sweep", action="store_true")
        p.add_argument("--fb-docs", type=int, default=10)
        p.add_argument("--fb-terms", type=int, default=50)
        p.add_argument("--fb-lambda", type=float, default=0.4)
        p.add_argument("--stride", type=int, default=None)
        p.add_argument("--lam-sweep", default="0.2,0.4,0.6")
        p.add_argument("--doc-sweep", default="5,10,20")
        p.add_argument("--term-sweep", default="20,50,100")
        p.add_argument("--passages", type=int, default=0)
        p.add_argument("--agg", default="max", choices=["max", "mean"])
        p.add_argument("--k1", type=float, default=1.2)
        p.add_argument("--b", type=float, default=0.75)
        p.add_argument("--qtf", default="sqrt", choices=["raw", "binary", "log", "sqrt"])
        p.add_argument("--max-df", type=float, default=0.2)
        p.add_argument("--splits", default="devx,test")
        p.add_argument("--topn", type=int, default=TOPN)
        p.set_defaults(fn=cmd_rm3, scheme=scheme)
    p = sp.add_parser("fuse")
    p.add_argument("--tag", default="fuse")
    p.add_argument("--members", required=True)
    p.add_argument("--kind", default="hybrid")
    p.add_argument("--splits", default="devx,test")
    p.add_argument("--topn", type=int, default=TOPN)
    p.set_defaults(fn=cmd_fuse)
    p = sp.add_parser("zip")
    p.add_argument("--tag", required=True)
    p.add_argument("--kind", default=None)
    p.add_argument("--note", default="")
    p.set_defaults(fn=cmd_zip)
    p = sp.add_parser("show")
    p.set_defaults(fn=cmd_show)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
