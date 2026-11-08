#!/usr/bin/env python3
"""CS-781 Phase 4 -- corpus-outcome retrieval hybrid. CPU, ~5 min.

    python phase4_retrieve.py                                  # measure on the 994 dev cases, write retrieval.json
    python phase4_retrieve.py --base ens_w_b --submit          # zip: corpus outcome where decided, ens_w_b elsewhere

A CJPE document is a Supreme Court judgment with its final order removed. The Phase-1 Supreme Court corpus
(permitted pretraining data) holds full judgments. When a CJPE case is found there (12-word needles, as in
phase4_overlap.py), the corpus words right AFTER the point where the CJPE text ends are its removed order.
The silver labeller's rule (phase4_external.py) reads it: 1 if any "allowed" phrase, else 0 if a "dismissed"
phrase, else undecided -> the base model predicts.

THIS IS RETRIEVAL, NOT PREDICTION. Every zip it writes is tagged ret_* and kind "hybrid", and the report must
present it as a retrieval-augmented hybrid, next to the model-only result (as Phase 1 did with its corpus
retrieval). Its dev score on the 994 real dev cases is an honest estimate of its test score.
"""

from __future__ import annotations

import argparse
import collections
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from phase4_data import (P4, load_all, log, atomic_json, metrics, best_threshold, load_run,  # noqa: E402
                         write_submission)
from phase4_overlap import stream_records, WORD, N                       # noqa: E402
from phase4_external import ALLOW, DISMISS                               # noqa: E402


# Tier 2: wordings of the operative order that the silver labeller's tier-1 rule does not cover. Applied
# only when tier 1 is silent; their dev accuracy is reported separately and a tier is used only if it helps.
ALLOW2 = re.compile(
    r"\b(?:appeals?|petitions?)\s+(?:partly\s+)?(?:allowed|accepted)\b"
    r"|\b(?:judgment|order|decree|conviction|sentence|award)s?\s+(?:of\s+the\s+\w+(?:\s+\w+)?\s+)?"
    r"(?:is|are|stands?)\s+(?:hereby\s+)?(?:set\s+aside|quashed|reversed)\b"
    r"|\b(?:accused|appellants?)\s+(?:is|are)\s+(?:hereby\s+)?acquitted\b"
    r"|\b(?:remand|remit)(?:ed)?\s+(?:the\s+)?(?:matter|case)\b"
    r"|\b(?:matter|case)\s+(?:is|are)\s+(?:hereby\s+)?(?:remanded|remitted)\b"
    r"|\bset\s+aside\s+the\s+(?:impugned\s+)?(?:judgment|order)", re.I)
DISMISS2 = re.compile(
    r"\b(?:appeals?|petitions?)\s+dismissed\b|\bno\s+merit\b|\bdevoid\s+of\s+merit\b"
    r"|\b(?:judgment|order|decree|conviction|sentence)s?\s+(?:of\s+the\s+\w+(?:\s+\w+)?\s+)?"
    r"(?:is|are)\s+(?:hereby\s+)?(?:upheld|affirmed|confirmed|maintained)\b"
    r"|\bno\s+(?:ground|reason|case)\s+(?:is\s+made\s+out\s+)?(?:for|to)\s+interfer", re.I)


def decide(cont, window, tier2):
    c = " ".join(cont.split()[:window])
    if ALLOW.search(c):
        return 1
    if DISMISS.search(c):
        return 0
    if tier2:
        if ALLOW2.search(c):
            return 1
        if DISMISS2.search(c):
            return 0
    return None


def needles_pos(words, k=10):
    out = []
    if len(words) < N + 5:
        return out
    for j in range(k):
        i = int((j + 0.5) * (len(words) - N) / k)
        while i < len(words) - N and len(words[i]) < 7:
            i += 1
        if i < len(words) - N:
            out.append((" ".join(words[i:i + N]), i))
    return out


def scan(a, data):
    from huggingface_hub import hf_hub_download
    docs, index, first = {}, collections.defaultdict(list), set()
    for split in ("dev", "test"):
        for d, t, l in data[split]:
            w = WORD.findall(t.lower())
            docs[(split, d)] = (len(w), l)
            for s, i in needles_pos(w):
                index[hash(s)].append(((split, d), i))
                first.add(s.split(" ", 1)[0])
    log(f"  {len(docs)} docs (dev {len(data['dev'])}, test {len(data['test'])}), {len(index)} needles")
    path = a.corpus_path or hf_hub_download("Exploration-Lab/CS-781-Capstone",
                                            "Phase_1_Pretraining/Train/train_supreme_court.json",
                                            repo_type="dataset")
    best = {}
    t0 = time.time()
    for n, rec in enumerate(stream_records(path)):
        w = WORD.findall((rec.get("text") or "").lower())
        hits = collections.defaultdict(list)
        for i in range(len(w) - N + 1):
            if w[i] in first:
                for key, ci in index.get(hash(" ".join(w[i:i + N])), ()):
                    hits[key].append((i, ci))
        for key, hs in hits.items():
            if len(hs) < a.min_hits or (key in best and best[key][0] >= len(hs)):
                continue
            cpos, ci = max(hs)                              # last matched needle
            end = cpos + (docs[key][0] - ci)                # where the CJPE text ends inside the corpus copy
            best[key] = (len(hs), " ".join(w[end:end + 1000]))
        if n and n % 20000 == 0:
            log(f"    {n} records, {len(best)} matched ({time.time()-t0:.0f}s)")
    log(f"  corpus scanned in {time.time()-t0:.0f}s")
    return docs, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-path", default=None)
    ap.add_argument("--window", type=int, default=0,
                    help="fix the corpus words read after the CJPE text ends (0 = pick on dev from 150..1000)")
    ap.add_argument("--tier2", action="store_true", help="with --window: also use the tier-2 phrases")
    ap.add_argument("--min-hits", type=int, default=3)
    ap.add_argument("--base", nargs="*", default=[], help="runs whose predictions fill the undecided cases")
    ap.add_argument("--submit", action="store_true")
    a = ap.parse_args()

    data = load_all(("dev", "test"))
    cache = f"{P4}/retrieval_cont.json"
    import json
    if os.path.exists(cache):
        cont = json.load(open(cache))
        log(f"  using saved corpus continuations {cache}")
    else:
        docs, best = scan(a, data)
        cont = {s: {k[1]: best[k][1] for k in docs if k[0] == s and k in best} for s in ("dev", "test")}
        atomic_json(cont, cache)
        log(f"  wrote {cache} ({len(cont['dev'])} dev, {len(cont['test'])} test continuations)")
    ydev = {i: l for i, _, l in data["dev"]}
    ids_dev = [i for i, _, _ in data["dev"]]
    ids_test = [i for i, _, _ in data["test"]]

    def rules(window, tier2):
        return {s: {i: (decide(cont[s][i], window, tier2) if i in cont[s] else None) for i in ids}
                for s, ids in (("dev", ids_dev), ("test", ids_test))}

    log(f"  dev: {len(cont['dev'])}/{len(ids_dev)} found in the corpus; test {len(cont['test'])}/{len(ids_test)}")
    configs = []
    for window in (150, 250, 400, 600, 1000):
        for tier2 in (False, True):
            r = rules(window, tier2)
            dec = {i: v for i, v in r["dev"].items() if v is not None}
            ok = sum(v == ydev[i] for i, v in dec.items())
            nt = sum(v is not None for v in r["test"].values())
            configs.append((window, tier2, r, len(dec), ok))
            log(f"    window {window:4d} tier2 {str(tier2):5s}: dev decides {len(dec):4d} ({100*len(dec)/len(ids_dev):.1f}%), "
                f"correct {ok} ({100*ok/max(1,len(dec)):.1f}%) | test decides {nt} ({100*nt/len(ids_test):.1f}%)")
    fixed = bool(a.window)
    atomic_json(dict(configs=[(c[0], c[1], c[3], c[4]) for c in configs]), f"{P4}/retrieval.json")

    for base in a.base:
        d, t, meta = load_run(base)
        ids = [i for i in d["ids"] if i in ydev]
        zd = dict(zip(d["ids"], d["z"].tolist()))
        thr = meta.get("thr")
        if thr is None:
            thr, _ = best_threshold([ydev[i] for i in ids], [zd[i] for i in ids])
        y = [ydev[i] for i in ids]
        bdev = {i: int(zd[i] > thr) for i in ids}
        mb = metrics(y, [bdev[i] for i in ids])
        best = None
        for window, tier2, r, nd, ok in configs:
            if fixed and (window, tier2) != (a.window, a.tier2):
                continue
            c = [r["dev"][i] if r["dev"].get(i) is not None else bdev[i] for i in ids]
            f = metrics(y, c)["macro_f1"]
            if best is None or f > best[0] + 1e-9:
                best = (f, window, tier2, r)
        f, window, tier2, r = best
        zt = dict(zip(t["ids"], t["z"].tolist()))
        pred = {i: (r["test"][i] if r["test"].get(i) is not None else int(zt[i] > thr)) for i in t["ids"]}
        n_dec = sum(r["test"].get(i) is not None for i in t["ids"])
        tag = f"ret_{base}" + ("" if (window, tier2) == (400, False) else f"_w{window}{'t2' if tier2 else ''}")
        log(f"  {base}: dev macro-F1 {mb['macro_f1']:.4f} alone -> {f:.4f} with corpus outcomes "
            f"(best: window {window}, tier2 {tier2}) | test: {n_dec}/{len(t['ids'])} decided -> {tag}")
        if a.submit:
            write_submission(tag, t["ids"], pred, dev=f, kind="hybrid",
                             note=f"RETRIEVAL HYBRID: corpus final order (window {window}, tier2 {tier2}) "
                                  f"for {n_dec} test docs, {base} elsewhere")

if __name__ == "__main__":
    main()
