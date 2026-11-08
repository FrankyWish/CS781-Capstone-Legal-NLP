#!/usr/bin/env python3
"""CS-781 Phase 4 -- integrity check: do CJPE test judgments sit in the Phase-1 corpus?

    python phase4_overlap.py                       # Supreme Court corpus (1.3 GB), CPU, ~10-20 min
    python phase4_overlap.py --sources train_supreme_court train_high_courts

WHY. CJPE text is a judgment with its final order cut off. The Phase-1
pretraining corpus holds full Supreme Court judgments. If a test case is in
there, the corpus copy carries the sentence that says "the appeal is allowed",
and predicting from it is reading the answer, not predicting. Phase 1 used
corpus overlap openly as a hybrid; for outcome prediction that defence is much
weaker. THIS SCRIPT ONLY MEASURES. It writes no submission. Using a match in a
submission needs Nithin's explicit go-ahead after asking the TAs.

METHOD. Each CJPE document gives up to 10 needles of 12 normalised words,
evenly spaced over the document, each starting on a word of >= 7 letters so the
corpus scan can skip most positions. A corpus record that contains >= 3 needles
of one document is a match. For matched dev documents the corpus text after the
last needle is searched for an outcome phrase, and that rule's agreement with
the gold label says how much a leak would be worth.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from phase4_data import P4, load_split, log, atomic_json  # noqa: E402

WORD = re.compile(r"[a-z0-9]+")
N = 12
ALLOW = re.compile(r"(appeals?|petitions?) (is|are|stand|stands) (hereby )?(allowed|accepted)|"
                   r"we (accordingly |therefore )?allow (the|this|these)|set aside the (impugned )?(judgment|order)", re.I)
DISMISS = re.compile(r"(appeals?|petitions?) (is|are|stand|stands) (hereby )?(dismissed|rejected)|"
                     r"we (accordingly |therefore )?dismiss (the|this|these)|no merit in (the|this|these)", re.I)


def stream_records(path, bufsize=1 << 22):
    """Every record of a large JSON array without loading the file (as in Phase 1)."""
    dec = json.JSONDecoder()
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        buf = f.read(bufsize)
        pos = buf.find("{")
        if pos < 0:
            return
        while True:
            try:
                obj, end = dec.raw_decode(buf, pos)
            except ValueError:
                more = f.read(bufsize)
                if not more:
                    return
                buf = buf[pos:] + more
                pos = buf.find("{")
                if pos < 0:
                    return
                continue
            yield obj
            k = end
            while k < len(buf) and buf[k] in ", \n\r\t":
                k += 1
            if k >= len(buf) or buf[k] == "]":
                more = f.read(bufsize)
                if not more:
                    return
                buf = buf[k:] + more
                pos = buf.find("{")
                if pos < 0:
                    return
            elif len(buf) - k < (1 << 20):
                more = f.read(bufsize)
                buf = buf[k:] + (more or "")
                pos = buf.find("{")
                if pos < 0:
                    return
            else:
                pos = k


def needles(text, k=10):
    w = WORD.findall(text.lower())
    out = []
    if len(w) < N + 5:
        return out
    for j in range(k):
        i = int((j + 0.5) * (len(w) - N) / k)
        while i < len(w) - N and len(w[i]) < 7:
            i += 1
        if i < len(w) - N:
            out.append((" ".join(w[i:i + N]), j))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=["train_supreme_court"])
    ap.add_argument("--min-hits", type=int, default=3)
    ap.add_argument("--corpus-path", default=None, help="local corpus JSON instead of the HF download")
    a = ap.parse_args()
    from huggingface_hub import hf_hub_download

    docs = {}
    for split in ("test", "dev"):
        for d, t, l in load_split(split):
            docs[(split, d)] = (t, l)
    index = collections.defaultdict(list)
    first = set()
    for key, (t, _) in docs.items():
        for s, j in needles(t):
            index[hash(s)].append((key, j))
            first.add(s.split(" ", 1)[0])
    log(f"  {len(docs)} CJPE docs (test + dev), {len(index)} needles, {len(first)} distinct first words")

    found = {}                                   # key -> (source, record no, hits, tail text)
    for src in a.sources:
        path = a.corpus_path or hf_hub_download("Exploration-Lab/CS-781-Capstone",
                                                f"Phase_1_Pretraining/Train/{src}.json", repo_type="dataset")
        t0 = time.time()
        for n, rec in enumerate(stream_records(path)):
            text = rec.get("text") or ""
            w = WORD.findall(text.lower())
            hits = collections.defaultdict(set)
            last = {}
            for i in range(len(w) - N + 1):
                if w[i] not in first:
                    continue
                h = hash(" ".join(w[i:i + N]))
                for key, j in index.get(h, ()):
                    hits[key].add(j)
                    last[key] = i + N
            for key, js in hits.items():
                if len(js) >= a.min_hits and (key not in found or len(js) > found[key][2]):
                    after = " ".join(w[last[key]:])
                    found[key] = (src, n, len(js), after[-3000:], len(w) - last[key])
            if n % 20000 == 0 and n:
                log(f"    {src}: {n} records, {len(found)} matched so far ({time.time()-t0:.0f}s)")
        log(f"  {src}: done, {n+1} records in {time.time()-t0:.0f}s")

    res = {}
    for split in ("test", "dev"):
        keys = [k for k in docs if k[0] == split]
        m = [k for k in keys if k in found]
        extra = sorted(found[k][4] for k in m)
        res[split] = dict(n=len(keys), matched=len(m),
                          median_words_after_last_needle=extra[len(extra) // 2] if extra else 0)
        log(f"\n  {split}: {len(m)} of {len(keys)} documents ({100*len(m)/max(1,len(keys)):.1f}%) found in "
            f"{a.sources}; median {res[split]['median_words_after_last_needle']} corpus words after the "
            f"last matched needle")
    # what would the leak be worth? outcome phrases in the corpus text that follows, on DEV
    agree = tot = cov = 0
    for k in docs:
        if k[0] != "dev" or k not in found:
            continue
        tail = found[k][3]
        al, di = bool(ALLOW.search(tail)), bool(DISMISS.search(tail))
        tot += 1
        if al != di:
            cov += 1
            agree += int((1 if al else 0) == docs[k][1])
    if tot:
        log(f"  dev leak value: of {tot} matched dev docs, the corpus continuation states an outcome for "
            f"{cov}; that rule agrees with the gold label on {agree}/{cov} ({100*agree/max(1,cov):.1f}%)")
    res["dev_rule"] = dict(matched=tot, decided=cov, agree=agree)
    res["sources"] = a.sources
    os.makedirs(P4, exist_ok=True)
    atomic_json(res, f"{P4}/integrity.json", indent=1)
    log(f"  wrote {P4}/integrity.json -- MEASUREMENT ONLY, nothing submitted")
    if res["test"]["matched"] > 0.05 * res["test"]["n"]:
        log("  >>> LARGE OVERLAP. Stop and ask the TAs before using any of it. <<<")


if __name__ == "__main__":
    main()
