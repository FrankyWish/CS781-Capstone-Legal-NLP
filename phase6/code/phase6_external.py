#!/usr/bin/env python3
"""CS-781 Phase 6 -- does the Phase-1 corpus hold these documents, and their headnotes?

    python phase6_external.py scan --procs 3      # results/corpus_scan.json

MEASURE ONLY.  This file can build no submission.  Indian Supreme Court judgments
from 1950-1994 were published with a lawyer-written headnote, and IN-Abs references
are those headnotes.  If the Phase-1 corpus keeps them, the answer key for a test
document may be sitting in the permitted corpus (Phase 4's 0.75 -> 0.96).

  train   a corpus record holding >= 2 of a train document's 12-word needles is a
          copy of it.  Gold summaries are public there, so this is the validation:
          what share of each summary's 8-grams does the copy contain, against the
          share the document itself contains?  The difference is the headnote.
  test    only counts and lengths: how many test documents have a copy, and whether
          the copy is longer than the document.  No corpus text about a test
          document is kept past its match or written anywhere.

A lookup system built on this needs Nithin's explicit go-ahead and is reported as
lookup beside the model-only number.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import time

import numpy as np

import phase6_data as D

N = 12
MINW = 7            # needles start on a word of >= 7 chars: the scan skips shorter words cheaply
GRAM = 8
FILES = dict(supreme_court="train_supreme_court", high_courts="train_high_courts")
_W = {}


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def words(t):
    return D.WORD.findall(t.lower())


def needles(w, k):
    out = []
    if len(w) < N + 5:
        return out
    for j in range(k):
        i = int((j + 0.5) * (len(w) - N) / k)
        while i < len(w) - N and len(w[i]) < MINW:
            i += 1
        if i < len(w) - N:
            out.append((" ".join(w[i:i + N]), j))
    return out


def gram_share(s, text_words, n=GRAM):
    a = {tuple(s[i:i + n]) for i in range(len(s) - n + 1)}
    if not a:
        return 0.0
    b = {tuple(text_words[i:i + n]) for i in range(len(text_words) - n + 1)}
    return len(a & b) / len(a)


def stream_records(path, bufsize=1 << 22):
    """Every record of a large JSON array without loading the file (Phase 1/4 pattern)."""
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


def _init(index, first, summ):
    _W.update(index=index, first=first, summ=summ)


def _scan(text):
    """One corpus record -> None | [(split, row, corpus_words, summary_share | None)]."""
    w = words(text)
    index, first = _W["index"], _W["first"]
    hits = collections.defaultdict(set)
    for i in range(len(w) - N + 1):
        wi = w[i]
        if len(wi) >= MINW and wi in first:
            for key, j in index.get(" ".join(w[i:i + N]), ()):
                hits[key].add(j)
    out = []
    for (split, row), js in hits.items():
        if len(js) >= 2:
            out.append((split, row, len(w),
                        gram_share(_W["summ"][row], w) if split == "train" else None))
    return out or None


def cmd_scan(a):
    from multiprocessing import Pool
    os.makedirs(f"{D.P6}/results", exist_ok=True)
    tr, te = D.load("train"), D.load("test")
    index, first = collections.defaultdict(list), set()
    tw = {"train": [], "test": []}
    for split, recs in (("train", tr), ("test", te)):
        for i, r in enumerate(recs):
            w = words(r["doc"])
            tw[split].append(len(w))
            for nd, j in needles(w, 8):
                index[nd].append(((split, i), j))
                first.add(nd.split(" ", 1)[0])
    index = dict(index)
    summ = [words(r["summ"]) for r in tr]
    log(f"indexed {len(tr)} train + {len(te)} test docs, {len(index)} needles")
    if a.corpus_path:
        paths = [a.corpus_path]
    else:
        from huggingface_hub import hf_hub_download
        paths = [hf_hub_download(D.REPO, f"Phase_1_Pretraining/Train/{FILES[s]}.json",
                                 repo_type="dataset", local_dir=a.local_dir)
                 for s in a.sources.split(",")]
    best = {}                       # train row -> (summary share, corpus words)
    found_test = {}                 # test row -> longest copy, in words
    n = 0
    with Pool(a.procs, initializer=_init, initargs=(index, first, summ)) as pool:
        for path in paths:
            t0 = time.time()
            feed = ((rec.get("text") or "") for rec in stream_records(path))
            for k, res in enumerate(pool.imap(_scan, feed, chunksize=8)):
                n += 1
                if k % 20000 == 0 and k:
                    log(f"  {os.path.basename(path)}: {k} records, {len(best)} train / "
                        f"{len(found_test)} test found ({time.time() - t0:.0f}s)")
                for split, row, cw, share in res or ():
                    if split == "test":
                        found_test[row] = max(found_test.get(row, 0), cw)
                    elif row not in best or share > best[row][0]:
                        best[row] = (share, cw)
    rows = sorted(best)
    doc_share = [gram_share(summ[i], words(tr[i]["doc"])) for i in rows]
    copy_share = [best[i][0] for i in rows]
    longer = [best[i][1] / max(tw["train"][i], 1) for i in rows]
    # a headnote is summary text the copy has and the document lacks: the raw share
    # would also count whatever the summary quotes from the judgment itself
    gain = np.array(copy_share) - np.array(doc_share)
    has = float(np.mean((np.array(copy_share) >= 0.5) & (gain >= 0.2))) if rows else 0.0
    t_long = [c / max(tw["test"][i], 1) for i, c in found_test.items()]
    R = dict(records=n, sources=a.sources,
             train_found=len(rows), train_found_share=round(len(rows) / len(tr), 4),
             train_summary_share_in_document=D.pct(doc_share),
             train_summary_share_in_corpus_copy=D.pct(copy_share),
             train_copies_holding_the_summary=round(has, 4),
             train_copy_length_ratio=D.pct(longer),
             test_found=len(found_test), test_found_share=round(len(found_test) / len(te), 4),
             test_copy_length_ratio=D.pct(t_long),
             test_copies_over_1_1x=int(sum(x > 1.1 for x in t_long)),
             test_docs_likely_holding_the_summary=round(len(found_test) * has, 1))
    with open(f"{D.P6}/results/corpus_scan.json", "w") as f:
        json.dump(R, f, indent=1)
    log(json.dumps(R, indent=1))
    log(f"SCAN DONE: {len(found_test)}/{len(te)} test docs have a corpus copy; "
        f"{has:.1%} of train copies add summary text beyond the document -> about "
        f"{R['test_docs_likely_holding_the_summary']} test docs could be answered by lookup. "
        f"Measured only; nothing about any test document's copy was stored.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("scan")
    s.add_argument("--procs", type=int, default=3)
    s.add_argument("--sources", default="supreme_court", help="supreme_court,high_courts")
    s.add_argument("--corpus-path", default=None, help="local JSON array (smoke test)")
    s.add_argument("--local-dir", default=os.environ.get("CS781_CORPUS_DIR", "/content/p1corpus"))
    s.set_defaults(fn=cmd_scan)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
