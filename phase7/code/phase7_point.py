#!/usr/bin/env python3
"""Phase 7 -- the three ideas that can change the ORDER of magnitude, not the margin.

The dev oracle says why the usual levers are exhausted. 0.53 on test is 477 of the 556
reachable hits, which is 100.6% of a perfect rerank of the top 50 and 96.3% of a perfect
rerank of the top 100. So no reranker closes it: a *better ordering* of the same evidence
is mathematically incapable of getting there. The only things that can are ones that bring
in evidence the pipeline has never looked at. There are exactly three available inside the
brief's constraints, and this module is all three.

1. `ngram` -- EXACT RARE N-GRAM POINTERS.  Every system so far is a bag of words: tf-idf,
   BM25, MaxP and the dense trunks all destroy word order. But an Indian judgment that
   cites a prior case very often *quotes its holding verbatim*, and an exact 8-word match
   between a citing sentence and a candidate judgment is not a similarity -- it is a
   near-certain pointer, because the chance of an 8-gram coinciding is vanishingly small.
   This converts ranking into identification for whichever masks carry a quotation. If
   even 40% of the ~11 masks per query resolve this way at high precision, that alone is
   ~4 hits/query = 472 devx hits = devx 0.52, which is the whole gap. `residue` measures
   whether the quotations are actually there before anything is built on them.

2. `hub` -- HUBNESS AND CAPACITY.  Every system ranks each query independently, so a
   handful of long, generic judgments win slots in hundreds of queries at once. The gold
   structure cannot look like that: a prior case is cited by a limited number of later
   ones. Correcting a candidate's score by how well it does *across* queries, or decoding
   all 237 queries jointly under a per-candidate capacity, is free information that
   per-query ranking throws away. This is the same move as Phase 2's test-volume
   calibration, which was the one thing that moved that leaderboard.

3. `where` -- the diagnostic that says which queries the 556 reachable hits sit in, so
   effort goes where hits are actually available.

Nothing here reads a test label and nothing here touches an external corpus: `ngram` is a
lexical system over the released candidate text, and `hub` is a decoder over scores we
already computed. Both are reported as `kind="lexical"` and `kind="hybrid"` respectively.

   python phase7_point.py residue --splits devx --n 8
   python phase7_point.py ngram   --tag ptr --n 8 --scope masks --splits train,devx,test
   python phase7_point.py hub     --base stack_new --mode zmean,greedy --splits devx,test
   python phase7_point.py where   --splits devx
"""
import argparse
import hashlib
import json
import os

import numpy as np

import phase7_data as D
import phase7_rank as R
import phase7_cite as CT

P7 = R.P7
log = D.log
B = np.uint64(1000003)                       # the polynomial base for the rolling hash


# ------------------------------------------------------------------ hashing --
def vocab_values(seed=D.SEED):
    """A 64-bit value per word, derived from the WORD ITSELF, not from when it was seen.

    This is the bug that made the first version of this module score exactly zero on
    every query, and it is worth spelling out because the failure was silent. The map
    was built lazily with an RNG, so a word got the value that corresponded to its
    position in the stream -- and the query side and the candidate side are hashed in
    separate passes, in different orders. The same word therefore received different
    values on the two sides and no n-gram could ever match, while every log line looked
    healthy and the system quietly returned 0 hits.

    Keyed BLAKE2b of the word fixes it by construction: the value depends on nothing but
    the word and the seed, so the two passes agree, and so do separate processes, which
    matters because the driver runs each subcommand in its own python.
    """
    key = hashlib.blake2b(str(seed).encode(), digest_size=16).digest()
    store = {}

    def val(w):
        v = store.get(w)
        if v is None:
            d = hashlib.blake2b(w.encode("utf-8"), digest_size=8, key=key).digest()
            v = store[w] = np.uint64(int.from_bytes(d, "little") | 1)
        return v

    return val, store


def hashes(words, n, val):
    """-> uint64 hash of every length-`n` window of `words`, in order.

    Order-sensitive by construction, which is the entire point: this is the one signal a
    bag-of-words index cannot represent.
    """
    if len(words) < n:
        return np.zeros(0, dtype=np.uint64)
    ids = np.fromiter((val(w) for w in words), dtype=np.uint64, count=len(words))
    m = len(ids) - n + 1
    h = np.zeros(m, dtype=np.uint64)
    for k in range(n):
        h = h * B + ids[k:k + m]
    return h


def query_grams(split, n, scope, window, min_words=8, max_words=0):
    """-> {qid: uint64 array of its n-gram hashes}, deduplicated per query.

    `scope` picks what counts as the query. `masks` uses only the text around the
    <CITATION> marks -- those sentences are where a quotation of the cited case appears,
    and the rest of a 6,000-word judgment is noise that can only create false matches.
    """
    val, _ = vocab_values()
    out = {}
    for r in D.queries(split):
        text = " ".join(r["sents"])
        chunks = []
        if scope in ("masks", "both"):
            chunks += CT.mask_contexts(text, window=window, min_words=min_words)
        if scope in ("doc", "both"):
            w = D.WORD.findall(text.lower())
            chunks.append(w[:max_words] if max_words else w)
        hs = [hashes(c, n, val) for c in chunks if len(c) >= n]
        out[r["id"]] = (np.unique(np.concatenate(hs)) if hs
                        else np.zeros(0, dtype=np.uint64))
    return out


def candidate_postings(split, n, want, max_words=0):
    """Stream the pool and keep ONLY the n-grams some query also has.

    The pool is ~10M n-grams for test and ~26M for train; the queries between them have
    at most ~1.4M. Filtering against the query side while streaming keeps the posting
    list proportional to the number of actual matches instead of to the corpus, which is
    what makes this fit in a Colab CPU runtime at all.
    """
    val, _ = vocab_values()
    pool = D.candidates(split)
    ids = list(pool)
    pos = {c: i for i, c in enumerate(ids)}
    hs_all, doc_all = [], []
    for c in ids:
        w = D.WORD.findall(" ".join(pool[c]).lower() if isinstance(pool[c], list)
                           else str(pool[c]).lower())
        if max_words:
            w = w[:max_words]
        h = hashes(w, n, val)
        if not len(h):
            continue
        h = np.unique(h)                       # df, not tf: one vote per document
        keep = h[np.isin(h, want, assume_unique=True)]
        if len(keep):
            hs_all.append(keep)
            doc_all.append(np.full(len(keep), pos[c], dtype=np.int32))
    if not hs_all:
        return ids, pos, np.zeros(0, dtype=np.uint64), np.zeros(0, dtype=np.int32)
    H = np.concatenate(hs_all)
    Dc = np.concatenate(doc_all)
    o = np.argsort(H, kind="stable")
    return ids, pos, H[o], Dc[o]


def union(qg):
    """The distinct n-gram hashes over every query in a split.

    A function rather than an inline `np.concatenate(...) or np.zeros(...)`, which is the
    bug this replaces: `or` on a numpy array raises "truth value of an array ... is
    ambiguous", so the fallback for a split where no query produced a single n-gram
    (short documents, or n larger than every mask context) crashed instead of returning
    an empty index.
    """
    parts = [v for v in qg.values() if len(v)]
    return np.unique(np.concatenate(parts)) if parts else np.zeros(0, dtype=np.uint64)


def gather(H, Dc, qh, ndocs, weight="invlog"):
    """-> (score per candidate, how many of the query's n-grams matched anything).

    Each shared n-gram contributes a weight that falls with its document frequency, so a
    phrase in 1 candidate counts far more than boilerplate in 50. df here is a true
    document frequency because the postings were deduplicated per candidate.
    """
    s = np.zeros(ndocs, dtype=np.float64)
    if not len(qh) or not len(H):
        return s, 0
    lo = np.searchsorted(H, qh, "left")
    hi = np.searchsorted(H, qh, "right")
    ln = (hi - lo).astype(np.int64)
    sel = ln > 0
    if not sel.any():
        return s, 0
    lo, ln = lo[sel], ln[sel]
    ends = np.cumsum(ln)
    idx = np.repeat(lo, ln) + (np.arange(ends[-1]) - np.repeat(ends - ln, ln))
    if weight == "inv":
        w = 1.0 / ln
    elif weight == "one":
        w = np.ones(len(ln))
    else:
        w = 1.0 / np.log1p(ln.astype(np.float64) + 1.0)
    np.add.at(s, Dc[idx], np.repeat(w, ln))
    return s, int(sel.sum())


def rank_ngram(split, n, scope, window, weight, topn=R.TOPN, max_words=0, qmax=0):
    qg = query_grams(split, n, scope, window, max_words=qmax)
    want = union(qg)
    ids, pos, H, Dc = candidate_postings(split, n, want, max_words=max_words)
    log(f"    {split}: {len(want)} distinct query {n}-grams, "
        f"{len(H)} postings over {len(ids)} candidates")
    drop = R.self_excluded(split)
    rank, score, hit = {}, {}, []
    for r in D.queries(split):
        q = r["id"]
        s, nm = gather(H, Dc, qg.get(q, np.zeros(0, dtype=np.uint64)), len(ids), weight)
        hit.append(nm)
        ex = drop.get(q)
        if ex is not None and ex in pos:
            s[pos[ex]] = -np.inf
        k = min(topn, len(s))
        top = np.argpartition(-s, k - 1)[:k]
        top = top[np.argsort(-s[top], kind="stable")]
        rank[q] = [ids[i] for i in top]
        score[q] = s[top].astype(np.float32)
    log(f"    {split}: {np.mean(hit):.1f} matching {n}-grams per query, "
        f"{float(np.mean(np.array(hit) > 0)) * 100:.0f}% of queries matched something")
    return rank, score


# ------------------------------------------------------------- 1. the premise --
def cmd_residue(a):
    """Does the text around a mask actually identify the case it cites?

    This is the measurement that decides whether idea 1 is real. If a citing sentence
    quotes the cited judgment, the gold candidates will share long rare n-grams with the
    mask contexts and the non-golds will not, and the enrichment will be enormous rather
    than the 1.86x that idf-Jaccard managed. If the enrichment is near 1, the masking
    stripped the quotations and this whole route is dead -- which is worth knowing in one
    cheap CPU run rather than after building on it.
    """
    out = {}
    for split in a.splits.split(","):
        g = D.golds_in_pool(split)
        row = {}
        for n in [int(x) for x in a.nsweep.split(",")]:
            qg = query_grams(split, n, a.scope, a.window, max_words=a.qmax)
            want = union(qg)
            ids, pos, H, Dc = candidate_postings(split, n, want, max_words=a.max_words)
            gold_s, other_s, nz, top1 = [], [], 0, 0
            rng = np.random.default_rng(D.SEED)
            for r in D.queries(split):
                q = r["id"]
                s, nm = gather(H, Dc, qg.get(q, np.zeros(0, dtype=np.uint64)),
                               len(ids), a.weight)
                gi = [pos[c] for c in g.get(q, []) if c in pos]
                if not gi:
                    continue
                nz += nm > 0
                oi = rng.choice(len(ids), size=min(len(gi) * 20, len(ids)), replace=False)
                oi = [i for i in oi if i not in set(gi)]
                gold_s.append(float(np.mean(s[gi])))
                other_s.append(float(np.mean(s[oi])) if oi else 0.0)
                if s.max() > 0 and ids[int(np.argmax(s))] in set(g.get(q, [])):
                    top1 += 1
            gm, om = float(np.mean(gold_s)), float(np.mean(other_s))
            row[n] = {"gold_mean": gm, "nongold_mean": om,
                      "enrichment": (gm / om) if om > 0 else None,
                      "queries_with_a_match": int(nz),
                      "queries": len(gold_s),
                      "top1_is_gold": int(top1),
                      "postings": int(len(H))}
            e = row[n]["enrichment"]
            log(f"  {split} n={n:2d}  gold {gm:.4g} vs non-gold {om:.4g}  "
                f"enrichment {('%.1fx' % e) if e else 'inf/na':>8s}  "
                f"matched {nz}/{len(gold_s)} queries  top1 gold {top1}")
        out[split] = row
    D.atomic_json(f"{P7}/results/ngram_residue.json",
                  {"config": {k: v for k, v in vars(a).items() if not callable(v)},
                   "by_split": out})
    log("  -> results/ngram_residue.json")
    log("  READ IT LIKE THIS: enrichment near 1 kills the pointer route; enrichment in "
        "the tens or hundreds means exact quotation survives the masking and `ngram` is "
        "the highest-value system in the phase.")


# -------------------------------------------------------- 2. the pointer system --
def cmd_ngram(a):
    best, rows = None, {}
    for n in [int(x) for x in a.nsweep.split(",")]:
        for scope in a.scope.split(","):
            rank, score = rank_ngram("devx", n, scope, a.window, a.weight,
                                     a.topn, a.max_words, a.qmax)
            s = R.evaluate(rank, "devx")
            R.report(f"n={n} {scope}", s)
            rows[f"n{n}_{scope}"] = {"f1": s["f1"], "hits": s["hits"]}
            if best is None or s["f1"] > best[0]:
                best = (s["f1"], n, scope)
    _, n, scope = best
    log(f"  chose n={n} scope={scope} (devx {best[0]:.4f})")
    for split in a.splits.split(","):
        rank, score = rank_ngram(split, n, scope, a.window, a.weight,
                                 a.topn, a.max_words, a.qmax)
        R.save_run(a.tag, split, rank, score)
        if split in ("devx", "train"):
            R.report(f"{a.tag}/{split}", R.evaluate(rank, split))
    R.save_run(a.tag, "meta", None, meta={
        "tag": a.tag, "kind": "lexical", "technique": f"exact rare {n}-gram pointers",
        "n": n, "scope": scope, "weight": a.weight, "window": a.window, "sweep": rows})


# ------------------------------------------------- 3. hubness and capacity --
def as_matrix(rank, score):
    """The saved top-N lists as (per-query candidate lists, per-query z-scored values).

    Per-query z-scoring first, because raw retrieval scores are not comparable across
    queries -- a long query has larger dot products throughout -- and every correction
    below subtracts a quantity pooled over queries.
    """
    Z, L = {}, {}
    for q, ids in rank.items():
        if score and q in score:
            v = np.asarray(score[q], dtype=np.float64)
        else:
            v = -np.arange(len(ids), dtype=np.float64)
        sd = v.std()
        Z[q] = (v - v.mean()) / (sd if sd > 1e-12 else 1.0)
        L[q] = list(ids)
    return L, Z


def hub_stats(L, Z, beta=5.0):
    """Per-candidate appearance count and shrunk mean z across the queries that list it.

    The shrinkage matters: a candidate seen by one query has a one-sample mean equal to
    its own score, and subtracting that would erase exactly the evidence we want to keep.
    Shrinking toward 0 (the global mean of a z-score) makes the penalty grow with the
    number of queries that like the candidate, which is the hubness we are correcting.
    """
    cnt, tot = {}, {}
    for q in L:
        for c, z in zip(L[q], Z[q]):
            cnt[c] = cnt.get(c, 0) + 1
            tot[c] = tot.get(c, 0.0) + float(z)
    mean = {c: tot[c] / (cnt[c] + beta) for c in cnt}
    return cnt, mean


def decode(L, Z, mode, alpha, cap, k=D.K, beta=5.0, stats=None):
    """-> {qid: reordered candidate list}.  Only the first k entries matter for F1@7.

    `stats`, when given a dict, receives `backfilled`: the number of top-k slots the
    capacity could not supply and that fell back to the base order. Read any greedy or
    sinkhorn result together with it.
    """
    cnt, mean = hub_stats(L, Z, beta)
    adj = {}
    for q in L:
        z = np.asarray(Z[q], dtype=np.float64)
        if mode == "zmean":
            z = z - alpha * np.array([mean[c] for c in L[q]])
        elif mode == "count":
            z = z - alpha * np.log1p([cnt[c] for c in L[q]])
        adj[q] = z
    if mode not in ("greedy", "sinkhorn"):
        return {q: [L[q][i] for i in np.argsort(-adj[q], kind="stable")] for q in L}

    if mode == "sinkhorn":
        # Row sums -> k, column sums -> at most cap, by alternating scaling in log space.
        u = {q: np.zeros(len(L[q])) for q in L}
        for _ in range(int(alpha) if alpha >= 1 else 30):
            col = {}
            for q in L:
                p = np.exp(adj[q] + u[q])
                p *= k / max(p.sum(), 1e-12)
                for c, pv in zip(L[q], p):
                    col[c] = col.get(c, 0.0) + float(pv)
            over = {c: v for c, v in col.items() if v > cap}
            if not over:
                break
            for q in L:
                pen = [np.log(over[c] / cap) if c in over else 0.0 for c in L[q]]
                u[q] = u[q] - np.asarray(pen)
        return {q: [L[q][i] for i in np.argsort(-(adj[q] + u[q]), kind="stable")]
                for q in L}

    # greedy: one global pass over every (query, candidate) pair, best first, honouring a
    # per-query budget of k and a per-candidate capacity.
    #
    # The capacity can be impossible to honour, and that has to be MEASURED rather than
    # asserted. A counting bound (len(L)*k slots over the distinct candidates) is
    # necessary but not sufficient: each query may only use candidates in its own
    # retrieved list, so feasibility is really a bipartite b-matching condition. When a
    # query runs out of in-capacity candidates its remaining slots are back-filled from
    # its own order, which breaks the cap -- so `stats["backfilled"]` counts exactly how
    # many slots that happened to. A sweep point with a large back-fill is not a capacity
    # result at all, it is mostly the base ranking, and the number must be read that way.
    floor = -(-len(L) * k // max(len({c for q in L for c in L[q]}), 1)) if L else 0
    if cap < floor:
        log(f"      cap {cap} cannot fill {len(L)} queries x {k} slots; raising to {floor}")
        cap = floor
    pairs = [(float(adj[q][i]), q, c) for q in L for i, c in enumerate(L[q])]
    pairs.sort(key=lambda t: -t[0])
    used, chosen, seen = {}, {q: [] for q in L}, {q: set() for q in L}
    for _, q, c in pairs:
        if len(chosen[q]) >= k or used.get(c, 0) >= cap or c in seen[q]:
            continue
        chosen[q].append(c)
        seen[q].add(c)
        used[c] = used.get(c, 0) + 1
    out, short = {}, 0
    for q in L:
        short += max(k - len(chosen[q]), 0)
        out[q] = chosen[q] + [c for c in L[q] if c not in seen[q]]
    if stats is not None:
        stats["backfilled"] = short
    if short:
        log(f"      cap {cap}: {short} of {len(L) * k} top-{k} slots back-filled past "
            f"the capacity -- that sweep point is mostly the base ranking")
    return out


def cmd_hub(a):
    base = {}
    for split in a.splits.split(","):
        rank, score = R.load_run(a.base, split)
        if rank is None:
            raise SystemExit(f"no saved run for {a.base}/{split} -- run its system first")
        base[split] = as_matrix(rank, score)

    sel = a.select if a.select in base else sorted(base)[0]
    L0, Z0 = base[sel]
    plain = R.evaluate(dict(L0), sel)
    log(f"  base {a.base} on {sel}: {D.brief(plain)}")
    best = (plain["f1"], "none", 0.0, 0)
    grid = []
    for mode in a.mode.split(","):
        caps = ([int(x) for x in a.capsweep.split(",")]
                if mode in ("greedy", "sinkhorn") else [0])
        for alpha in [float(x) for x in a.asweep.split(",")]:
            for cap in caps:
                st = {}
                out = decode(L0, Z0, mode, alpha, cap or 10 ** 9, beta=a.beta, stats=st)
                s = R.evaluate(out, sel)
                grid.append({"mode": mode, "alpha": alpha, "cap": cap,
                             "f1": s["f1"], "hits": s["hits"],
                             "backfilled": st.get("backfilled", 0)})
                log(f"    {mode:9s} alpha {alpha:5.2f} cap {cap:4d}  "
                    f"{sel} {s['f1']:.4f}  "
                    f"({s['f1'] - plain['f1']:+.4f})")
                if s["f1"] > best[0]:
                    best = (s["f1"], mode, alpha, cap)
    f1, mode, alpha, cap = best
    log(f"  BEST {mode} alpha {alpha} cap {cap} -> {sel} {f1:.4f} "
        f"({f1 - plain['f1']:+.4f} over the base)")
    rep = {"base": a.base, "select": sel, "grid": grid, "chose": None}
    if mode == "none":
        log("  nothing beat the base, so nothing is written: per-query ranking is already "
            "consistent with the gold reuse structure and this lever is dead.")
        D.atomic_json(f"{P7}/results/hub_{a.base}.json", rep)
        return
    for split, (L, Z) in base.items():
        out = decode(L, Z, mode, alpha, cap or 10 ** 9, beta=a.beta)
        R.save_run(a.tag, split, out)
        if split in ("devx", "train"):
            R.report(f"{a.tag}/{split}", R.evaluate(out, split))
    R.save_run(a.tag, "meta", None, meta={
        "tag": a.tag, "kind": "hybrid", "technique": f"{mode} hubness/capacity decoding",
        "base": a.base, "mode": mode, "alpha": alpha, "cap": cap, "select": sel})
    rep["chose"] = {"mode": mode, "alpha": alpha, "cap": cap, "f1": f1}
    D.atomic_json(f"{P7}/results/hub_{a.base}.json", rep)


# ---------------------------------------------------- 4. where the hits are --
def cmd_where(a):
    out = {}
    for split in a.splits.split(","):
        gp = D.golds_in_pool(split)
        sizes = np.array([len(v) for v in gp.values()])
        reach = np.minimum(sizes, D.K)
        order = np.argsort(-sizes)
        share = np.cumsum(reach[order]) / max(reach.sum(), 1)
        serves = {}
        for v in gp.values():
            for c in v:
                serves[c] = serves.get(c, 0) + 1
        sv = np.array(sorted(serves.values(), reverse=True)) if serves else np.zeros(1)
        tenth = max(int(0.1 * len(sizes)) - 1, 0)
        row = {"queries": int(len(sizes)),
               "reachable_hits": int(reach.sum()),
               "golds_per_query": D.pct(sizes),
               "queries_with_7plus": int((sizes >= 7).sum()),
               "share_of_hits_in_7plus": float(reach[sizes >= 7].sum()
                                               / max(reach.sum(), 1)),
               "richest_10pct_queries_hold": float(share[tenth]),
               "distinct_gold_candidates": int(len(serves)),
               "queries_served_per_gold": D.pct(sv),
               "golds_serving_one_query_only": int((sv == 1).sum()),
               "max_reuse": int(sv.max())}
        out[split] = row
        log(f"  {split}: {row['reachable_hits']} reachable hits over {row['queries']} "
            f"queries; {row['queries_with_7plus']} queries have >=7 golds and hold "
            f"{row['share_of_hits_in_7plus'] * 100:.0f}% of them; the richest 10% of "
            f"queries hold {row['richest_10pct_queries_hold'] * 100:.0f}%")
        log(f"  {split}: {row['distinct_gold_candidates']} distinct gold candidates, "
            f"{row['golds_serving_one_query_only']} cited by exactly ONE query, max "
            f"reuse {row['max_reuse']} -- that max is what `hub --capsweep` must bracket")
    D.atomic_json(f"{P7}/results/where_hits.json", out)
    log("  -> results/where_hits.json")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(q, splits="devx"):
        q.add_argument("--splits", default=splits)
        q.add_argument("--topn", type=int, default=R.TOPN)
        q.add_argument("--window", type=int, default=60,
                       help="words kept either side of a mask")
        q.add_argument("--weight", default="invlog", choices=("invlog", "inv", "one"))
        q.add_argument("--max-words", type=int, default=0,
                       help="truncate each candidate to this many words, 0 = no limit")
        q.add_argument("--qmax", type=int, default=0,
                       help="truncate the query side, 0 = no limit")

    r = sub.add_parser("residue", help="does mask context identify the cited case?")
    common(r)
    r.add_argument("--nsweep", default="6,8,12")
    r.add_argument("--scope", default="masks", choices=("masks", "doc", "both"))
    r.set_defaults(fn=cmd_residue)

    g = sub.add_parser("ngram", help="exact rare n-gram pointer retrieval")
    common(g, "train,devx,test")
    g.add_argument("--tag", default="ptr")
    g.add_argument("--nsweep", default="6,8,12")
    g.add_argument("--scope", default="masks,both")
    g.set_defaults(fn=cmd_ngram)

    h = sub.add_parser("hub", help="hubness / capacity decoding over a saved run")
    h.add_argument("--base", required=True)
    h.add_argument("--tag", default=None)
    h.add_argument("--splits", default="train,devx,test")
    h.add_argument("--select", default="train")
    h.add_argument("--mode", default="zmean,count,greedy")
    h.add_argument("--asweep", default="0.25,0.5,1.0")
    h.add_argument("--capsweep", default="3,6,12,25")
    h.add_argument("--beta", type=float, default=5.0)
    h.set_defaults(fn=cmd_hub)

    w = sub.add_parser("where", help="where the reachable hits actually sit")
    w.add_argument("--splits", default="train,devx")
    w.set_defaults(fn=cmd_where)

    a = p.parse_args()
    if a.cmd == "hub" and a.tag is None:
        a.tag = f"hub_{a.base}"
    return a.fn(a) or 0


if __name__ == "__main__":
    raise SystemExit(main())
