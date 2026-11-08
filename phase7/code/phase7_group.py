#!/usr/bin/env python
"""CS-781 Phase 7 -- citator GROUPS for queries whose own AIR key is unknown.

A Supreme Court candidate's CITATOR INFO lists the later SC judgments that cite it.
IL-PCR pools are built from the queries' citations (train pool 95% cited, devx 90%), so
a citing key K that several pool candidates share, and that is NOT any query's known
AIR key, is most likely the hidden key of some query: the group G_K is (part of) that
query's gold set.  Rule A already uses the keys we know; this assigns the rest.

Each group goes to the eligible query (no known AIR key; decided no earlier than its
latest-mentioned year) whose base ranking puts the group's members highest, if it beats
the runner-up by a margin.  One group per query (a judgment has one AIR key).
Selection: a small grid, scored by precision on train + devx labels (the rule learns
nothing from them).  Used by phase7_citator.py's `g` variants as a tier after base top 3.
"""
from __future__ import annotations

import collections

GRID = [(s, m) for s in (2, 3, 4) for m in (1.3, 2.0, 3.0)]


def assign(qs, pool, Pq, Pc, base, min_size, margin, topk=60, max_size=25):
    known = {p["air"] for p in Pq.values() if p["air"]}
    groups = collections.defaultdict(set)
    for c, p in Pc.items():
        for k in p["cit"]:
            groups[k].add(c)
    rank = {}
    for r in qs:
        q = r["id"]
        if Pq[q]["air"]:
            continue                                   # rule A covers queries with a key
        rank[q] = {str(c): i for i, c in enumerate((base.get(q) or [])[:topk])}
    best = {}
    for k, G in groups.items():
        if k in known or not (min_size <= len(G) <= max_size):
            continue
        sc = []
        for q, rk in rank.items():
            p = Pq[q]
            if p["date"] and p["date"][0] not in (k[0], k[0] - 1):
                continue                               # a dated query must match the key year
            if not p["date"] and p.get("yest") and k[0] < p["yest"]:
                continue                               # key year >= latest year it mentions
            s = sum(1.0 / (1.0 + rk[c] / 5.0) for c in G if c in rk)
            if s > 0:
                sc.append((s, q))
        if not sc:
            continue
        sc.sort(reverse=True)
        s1, q1 = sc[0]
        s2 = sc[1][0] if len(sc) > 1 else 0.0
        if s1 >= margin * s2 and (q1 not in best or s1 > best[q1][0]):
            best[q1] = (s1, k, G - {q1})
    return {q: v[2] for q, v in best.items()}


def precision(out, gold):
    n = sum(len(v) for q, v in out.items() if q in gold)
    tp = sum(len(v & set(gold[q])) for q, v in out.items() if q in gold)
    return n, tp


def assign_all(built, bases, base_for, golds, log, min_prec=0.45, force=None):
    rows = []
    for cfg in ([force] if force else GRID):
        n = tp = nq = 0
        for sp in ("train", "devx"):
            qs, pool, Pq, Pc = built[sp][:4]
            _, base = base_for(sp, bases[sp])
            if base is None:
                continue
            out = assign(qs, pool, Pq, Pc, base, *cfg)
            a, b = precision(out, golds(sp))
            n, tp, nq = n + a, tp + b, nq + len(out)
        rows.append((cfg, n, tp, nq))
        log(f"  group cfg size>={cfg[0]} margin {cfg[1]}: {nq} queries, {n} pairs, {tp} correct "
            f"(precision {tp / max(n, 1):.3f})")
    ok = [r for r in rows if r[1] and r[2] / r[1] >= min_prec]
    if not ok:
        log(f"  citator groups below precision {min_prec} -- not used")
        return {}
    cfg = max(ok, key=lambda r: (r[2], r[2] / r[1]))[0]
    log(f"  CHOSE group cfg {cfg}")
    res = {}
    for sp in ("train", "devx", "test"):
        qs, pool, Pq, Pc = built[sp][:4]
        _, base = base_for(sp, bases[sp])
        if base is None:
            continue
        out = assign(qs, pool, Pq, Pc, base, *cfg)
        res.update(out)                                 # query ids are unique across splits
        if sp != "test":
            n, tp = precision(out, golds(sp))
            log(f"  [{sp}] groups for {len(out)} queries -> {n} pairs, {tp} correct "
                f"(precision {tp / max(n, 1):.3f})")
        else:
            log(f"  [test] groups for {len(out)} queries -> {sum(len(v) for v in out.values())} pairs")
    return res
