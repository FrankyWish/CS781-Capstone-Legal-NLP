#!/usr/bin/env python
"""CS-781 Phase 7 -- probe the RAW structure of the released files for large signals.

    python phase7_probe.py

40 leaderboard entries sit above 0.4536 while every text-similarity method here plateaus
near 0.46, so this asks the dataset-artefact questions nobody asked: do candidate IDs,
file order, mask strings or dates carry the answer?  Uses train/dev labels only; from
test it reads only the (label-free) files, never IL-PCR's public test labels.
Each section is independent; a failure prints and moves on.
"""
import collections
import json
import os
import random
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import phase7_data as D   # noqa: E402

rng = random.Random(781)
MASK = re.compile(r"<[^<>]{0,60}>|\[[^\[\]]{0,30}CITATION[^\[\]]{0,30}\]", re.I)
YEAR = re.compile(r"\b(19[0-9]{2}|20[0-2][0-9])\b")
WORD = re.compile(r"[a-z0-9]+")


def p(*a):
    print(*a, flush=True)


def num(x):
    m = re.findall(r"\d+", str(x))
    return int(m[-1]) if m else None


def section(f):
    def run(*a):
        p(f"\n---- {f.__name__} " + "-" * 50)
        try:
            f(*a)
        except Exception as e:                            # noqa: BLE001
            p(f"  !! {type(e).__name__}: {e}")
    return run


def load(split):
    qs = D.queries(split)
    pool = D.candidates(split)
    return qs, list(pool), pool


@section
def ids(split, qs, order, pool):
    p(f"  {split}: query ids {[r['id'] for r in qs[:4]]}  cand ids {order[:4]}")
    qn = [num(r["id"]) for r in qs]
    cn = [num(c) for c in order]
    p(f"  numeric ids: queries {sum(x is not None for x in qn)}/{len(qn)}, "
      f"cands {sum(x is not None for x in cn)}/{len(cn)}")
    if split == "test":
        return
    d_g, d_r = [], []
    for r in qs:
        a = num(r["id"])
        if a is None:
            continue
        for c in r["gold"] or []:
            b = num(c)
            if b is not None:
                d_g.append(abs(a - b))
                d_r.append(abs(a - num(rng.choice(order)) if num(order[0]) is not None else 0))
    if d_g:
        d_g, d_r = np.array(d_g), np.array(d_r)
        p(f"  |id(q)-id(gold)| median {np.median(d_g):.0f} vs random {np.median(d_r):.0f}; "
          f"frac<=100: gold {np.mean(d_g <= 100):.3f} random {np.mean(d_r <= 100):.3f}; "
          f"gold id < query id: {np.mean([num(c) < num(r['id']) for r in qs for c in (r['gold'] or []) if num(c) is not None and num(r['id']) is not None]):.3f}")


@section
def file_order(split, qs, order, pool):
    if split == "test":
        return
    pos = {c: i for i, c in enumerate(order)}
    qi = {r["id"]: i for i, r in enumerate(qs)}
    adj_g, adj_r, means, qix = 0, 0, [], []
    spread_g, spread_r = [], []
    for r in qs:
        g = sorted(pos[c] for c in (r["gold"] or []) if c in pos)
        if len(g) >= 2:
            adj_g += sum(1 for a, b in zip(g, g[1:]) if b - a <= 3)
            rs = sorted(rng.sample(range(len(order)), len(g)))
            adj_r += sum(1 for a, b in zip(rs, rs[1:]) if b - a <= 3)
            spread_g.append(np.std(g))
            spread_r.append(np.std(rs))
        if g:
            means.append(np.mean(g))
            qix.append(qi[r["id"]])
    from scipy.stats import spearmanr
    p(f"  consecutive gold positions within 3: gold {adj_g} vs random {adj_r}")
    p(f"  per-query std of gold positions: median {np.median(spread_g):.0f} vs random "
      f"{np.median(spread_r):.0f}")
    p(f"  spearman(query file index, mean gold position) = {spearmanr(qix, means)[0]:.3f}")
    # is a query's own pool copy next to its golds?
    near = []
    for r in qs:
        if r["id"] in pos and r["gold"]:
            g = [pos[c] for c in r["gold"] if c in pos]
            if g:
                near.append(min(abs(pos[r["id"]] - x) for x in g))
    if near:
        p(f"  min |pos(query copy) - pos(gold)|: median {np.median(near):.0f} "
          f"(pool {len(order)})")


@section
def masks(split, qs, order, pool):
    cnt = collections.Counter()
    for r in qs:
        cnt.update(m.lower() for m in MASK.findall(" ".join(r["sents"])))
    p(f"  {split} query mask strings: {sum(cnt.values())} total, {len(cnt)} distinct; "
      f"top {cnt.most_common(6)}")
    dig = [m for m in cnt if re.search(r"\d", m)]
    p(f"  mask strings containing digits: {len(dig)} e.g. {dig[:6]}")
    cc = collections.Counter()
    for c in order[:400]:
        cc.update(m.lower() for m in MASK.findall(" ".join(pool[c])))
    p(f"  masks in first 400 candidates: {sum(cc.values())}, top {cc.most_common(4)}")
    if split != "test" and dig:
        hit = tot = 0
        for r in qs:
            gs = {str(num(c)) for c in (r["gold"] or [])}
            for m in MASK.findall(" ".join(r["sents"])):
                for d in re.findall(r"\d+", m):
                    tot += 1
                    hit += d in gs
        p(f"  digits in masks that equal a gold id number: {hit}/{tot}")


@section
def sample(split, qs, order, pool):
    r = next(r for r in qs if r["gold"])
    t = " ".join(r["sents"])
    ms = list(MASK.finditer(t))[:3]
    for m in ms:
        p(f"  CTX: ...{t[max(0, m.start() - 220):m.end() + 120]!r}")
    g = r["gold"][0]
    if g in pool:
        p(f"  GOLD {g} head: {' '.join(pool[g])[:400]!r}")
    p(f"  QUERY {r['id']} head: {t[:300]!r}")


def yr(text):
    ys = [int(y) for y in YEAR.findall(text[:200000])]
    return max(ys) if ys else None


def yr_head(text):
    ys = [int(y) for y in YEAR.findall(text[:1500])]
    return max(ys) if ys else None


@section
def dates(split, qs, order, pool):
    if split == "test":
        return
    for nm, f in (("max-year", yr), ("head-year", yr_head)):
        cy = {c: f(" ".join(pool[c])) for c in order}
        ok = tot = rok = rtot = 0
        lag = []
        for r in qs:
            qy = f(" ".join(r["sents"]))
            if qy is None:
                continue
            for c in r["gold"] or []:
                if cy.get(c) is not None:
                    tot += 1
                    ok += cy[c] <= qy
                    lag.append(qy - cy[c])
                rc = rng.choice(order)
                if cy.get(rc) is not None:
                    rtot += 1
                    rok += cy[rc] <= qy
        p(f"  {nm}: P(cand<=query | gold) {ok / max(tot, 1):.3f} vs random "
          f"{rok / max(rtot, 1):.3f}; median lag {np.median(lag) if lag else 'na'}")


@section
def citing_structure(split, qs, order, pool):
    if split == "test":
        return
    qids = {r["id"] for r in qs}
    g = {r["id"]: set(r["gold"] or []) for r in qs}
    # co-citation transitivity: if q cites a query-candidate q2, does q also cite q2's golds?
    tr = tot = 0
    for q, gs in g.items():
        for q2 in gs & qids:
            tot += len(g[q2])
            tr += len(g[q2] & gs)
    p(f"  q cites q2 (a query) -> q also cites q2's golds: {tr}/{tot}")
    # overlap of gold sets between pairs of queries
    inv = collections.defaultdict(set)
    for q, gs in g.items():
        for c in gs:
            inv[c].add(q)
    share = collections.Counter(len(v) for v in inv.values())
    p(f"  candidates by #citing queries: {sorted(share.items())[:8]}")


def main():
    for split in ("train", "dev", "test"):
        p(f"\n================ {split}")
        qs, order, pool = load(split)
        for f in (ids, file_order, masks, sample, dates, citing_structure):
            f(split, qs, order, pool)


if __name__ == "__main__":
    main()
