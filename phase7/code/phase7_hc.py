#!/usr/bin/env python
"""CS-781 Phase 7 -- extend the corpus lookup to the High Court and tribunal corpora.

    python phase7_hc.py [--sources train_high_courts,train_tribunals] [--minutes 110]

WHY.  The Supreme Court scan found ~half the queries; the misses are mostly High Court
judgments (62/118 devx queries carry no SC header).  The Phase-1 corpus also holds
`train_high_courts` (10.7 GB) and `train_tribunals`.  One streaming pass looks for BOTH
the devx/test queries (to recover what their masks hid) and the devx/test candidates (to
recover their real titles), and MERGES into the files the SC scan wrote:

    results/overlap/recovered_{split}.json    spans for queries not already recovered
    results/names/titles_{split}.json         titles for candidates not already titled

Results are flushed after every source and at the deadline, so a timeout loses nothing.
kind = "lookup" for anything built on it (permitted Phase-1 corpus, reported separately).
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import phase7_data as D      # noqa: E402
import phase7_overlap as O   # noqa: E402
import phase7_names as NM    # noqa: E402

P7 = D.P7
WORD = O.WORD


def log(*a):
    print(" ", *a, flush=True)


def load(path):
    return json.load(open(path)) if os.path.exists(path) else {}


SENT = "qqmaskqq"


def clean_fps(text, k, n):
    """k well-spaced n-word windows that contain NO mask.  A window spanning <NAME>,
    <ORG> or <CITATION> can never match the unmasked corpus copy -- the original scan
    ignored this, and with masks every few dozen words it silently lost ~half the docs."""
    w = WORD.findall(O.MASKS.sub(f" {SENT} ", text).lower())
    if len(w) < n + 5:
        return []
    bad = [i for i, x in enumerate(w) if x == SENT]
    ok = []
    b = 0
    for i in range(len(w) - n + 1):            # valid start: no mask in w[i:i+n], long 1st word
        while b < len(bad) and bad[b] < i:
            b += 1
        if (b >= len(bad) or bad[b] >= i + n) and len(w[i]) >= 7:
            ok.append(i)
    if not ok:
        return []
    out, used = [], set()
    for j in range(k):
        t = int((j + 0.5) * len(ok) / k)
        i = ok[min(t, len(ok) - 1)]
        if i in used:
            continue
        used.add(i)
        out.append((" ".join(w[i:i + n]), j))
    return out


def flush(splits, qdocs, found_q, found_c):
    os.makedirs(f"{P7}/results/overlap", exist_ok=True)
    os.makedirs(f"{P7}/results/names", exist_ok=True)
    for sp in splits:
        rf = f"{P7}/results/overlap/recovered_{sp}.json"
        rec = load(rf)
        added = 0
        for (s2, q), (_, text) in found_q.items():
            if s2 != sp or rec.get(q):
                continue
            spans = O.recover_masks(qdocs[(sp, q)], text)
            if spans:
                rec[q] = spans
                added += 1
        D.atomic_json(rf, rec)
        tf = f"{P7}/results/names/titles_{sp}.json"
        titles = load(tf)
        tadd = 0
        for (s2, c), (_, head) in found_c.items():
            if s2 != sp or (c in titles and titles[c].get("p2")):
                continue
            m = NM.TITLE.search(head) or NM.TITLE2.search(head)
            if m:
                titles[c] = {"p1": m.group(1), "p2": m.group(2), "year": int(m.group(3))}
            elif c not in titles:
                titles[c] = {"p1": head[:200], "p2": "", "year": None}
            tadd += 1
        D.atomic_json(tf, titles)
        log(f"{sp}: +{added} queries with recovered spans (now {len(rec)}), "
            f"+{tadd} candidate titles (now {len(titles)})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sources", default="train_supreme_court")
    p.add_argument("--splits", default="devx,test")
    p.add_argument("--needles", type=int, default=40)
    p.add_argument("--n", type=int, default=10, help="fingerprint length in words")
    p.add_argument("--min-hits", type=int, default=6)
    p.add_argument("--old-fps", action="store_true", help="original mask-blind fingerprints")
    p.add_argument("--minutes", type=float, default=110)
    a = p.parse_args()
    from huggingface_hub import hf_hub_download
    deadline = time.time() + a.minutes * 60
    splits = a.splits.split(",")
    index, first = collections.defaultdict(list), set()
    qdocs = {}
    nq = nc = 0
    NW = O.N if a.old_fps else a.n

    def fps(text):
        if a.old_fps:
            return O.fingerprints(WORD.findall(text.lower()), a.needles)
        return clean_fps(text, a.needles, NW)
    for sp in splits:
        rec = load(f"{P7}/results/overlap/recovered_{sp}.json")
        titles = load(f"{P7}/results/names/titles_{sp}.json")
        for q, t in O.query_texts(sp).items():
            if rec.get(q):
                continue                          # already recovered from the SC corpus
            qdocs[(sp, q)] = t
            nq += 1
            for fp, j in fps(t):
                index[hash(fp)].append((("Q", sp, q), j))
                first.add(fp.split(" ", 1)[0])
        for c, v in D.candidates(sp).items():
            if c in titles and titles[c].get("p2"):
                continue
            nc += 1
            for fp, j in fps(" ".join(v)):
                index[hash(fp)].append((("C", sp, c), j))
                first.add(fp.split(" ", 1)[0])
    log(f"looking for {nq} queries and {nc} candidates; {len(index)} fingerprints")
    found_q, found_c = {}, {}
    for src in a.sources.split(","):
        if time.time() > deadline:
            break
        try:
            path = hf_hub_download("Exploration-Lab/CS-781-Capstone",
                                   f"Phase_1_Pretraining/Train/{src}.json", repo_type="dataset")
        except Exception as e:                    # noqa: BLE001
            log(f"{src}: download failed ({e}) -- skipped")
            continue
        log(f"scanning {src} ({os.path.getsize(path) / 2**30:.1f} GB)")
        t0 = time.time()
        n = 0
        for n, rec in enumerate(O.stream_records(path)):
            if n % 5000 == 0 and time.time() > deadline:
                log(f"  deadline at record {n:,}")
                break
            text = O.rec_text(rec)
            w = WORD.findall(text.lower())
            hits = collections.defaultdict(set)
            for i in range(len(w) - NW + 1):
                if w[i] not in first:
                    continue
                for key, j in index.get(hash(" ".join(w[i:i + NW])), ()):
                    hits[key].add(j)
            for key, js in hits.items():
                if len(js) < a.min_hits:
                    continue
                kind, sp, i_d = key
                tgt = found_q if kind == "Q" else found_c
                if (sp, i_d) not in tgt or len(js) > tgt[(sp, i_d)][0]:
                    tgt[(sp, i_d)] = (len(js), text if kind == "Q" else text[:3000])
            if n and n % 100000 == 0:
                log(f"  {n:,} records, {len(found_q)} queries + {len(found_c)} candidates "
                    f"matched ({time.time() - t0:.0f}s)")
        log(f"{src}: {n + 1:,} records in {time.time() - t0:.0f}s; "
            f"{len(found_q)} queries, {len(found_c)} candidates matched so far")
        flush(splits, qdocs, found_q, found_c)
        for f in {os.path.realpath(path), path}:  # free Colab disk (cache blob + symlink)
            try:
                os.remove(f)
            except OSError:
                pass
    flush(splits, qdocs, found_q, found_c)


if __name__ == "__main__":
    main()
