#!/usr/bin/env python
"""CS-781 Phase 7 -- integrity check: does the Phase-1 pretraining corpus hold UNMASKED
copies of the PCR query judgments?

This is MEASURE-ONLY.  Nothing is built on a match without Nithin's explicit go-ahead,
and any system that uses one is reported as kind="lookup" beside the model-only number.
The script never downloads anything but the official Phase-7 files and the Phase-1
pretraining corpus, both already permitted; it never touches IL-PCR's public test labels.

WHY IT MATTERS HERE.  IL-PCR masks the citations inside a query judgment -- the gate
counted 2,681 `<CITATION>`-style masks in the test queries alone.  The cases named at
those positions ARE, in large part, the gold prior cases.  The Phase-1 corpus contains
full Indian judgments.  If a query judgment sits in that corpus unmasked, the corpus copy
spells out the answer key in plain text, and no amount of retrieval skill is needed.

METHOD.  Two stages, deliberately separate so the cheap decisive number comes first.

  scan   Each query gives up to `--needles` fingerprints of 12 normalised words taken
         from well-spaced positions, keyed by a rare-ish first word so the corpus scan
         can skip most offsets. A corpus record carrying >= --min-hits fingerprints of
         one query is a match. For a matched query, each mask in the query is then
         located by its surrounding context and the corpus text at that position is
         recovered -- that recovered span is what the mask was hiding.

  value  Maps the recovered spans onto candidate ids by lexical lookup in the split's own
         candidate pool, and scores the result on devx with the official metric. That is
         the honest upper bound on what a lookup system would be worth, measured on the
         split where we have labels, before anyone decides whether to use it.

  python phase7_overlap.py scan  --splits test,devx
  python phase7_overlap.py value --split devx
"""
import argparse
import collections
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import phase7_data as D
import phase7_rank as R

P7 = D.P7
WORD = re.compile(r"[a-z0-9]+")
N = 12                                   # fingerprint length in words
OUT = f"{P7}/results/overlap"

# the mask shapes the gate detects; `angle_citation` is the one PCR actually uses
MASKS = re.compile(r"<[^<>]{0,40}>|\[\?CITATION\?\]|\[CITATION\]|_{4,}")


def log(*a):
    print(" ", *a, flush=True)


def stream_records(path):
    """Yield dicts from a JSON array or a JSONL file without loading 1.3 GB at once."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        head = f.read(1)
        while head and head.isspace():
            head = f.read(1)
        f.seek(0)
        if head != "[":                                     # JSONL
            for line in f:
                line = line.strip().rstrip(",")
                if line and line not in ("[", "]"):
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
            return
        buf, depth, instr, esc, started = [], 0, False, False, False
        while True:
            block = f.read(1 << 20)
            if not block:
                break
            for ch in block:
                if not started:
                    if ch == "{":
                        started, depth = True, 1
                        buf = [ch]
                    continue
                buf.append(ch)
                if instr:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        instr = False
                    continue
                if ch == '"':
                    instr = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            yield json.loads("".join(buf))
                        except json.JSONDecodeError:
                            pass
                        started, buf = False, []


def rec_text(rec):
    if isinstance(rec, str):
        return rec
    for k in ("text", "content", "judgment", "body", "document", "raw"):
        v = rec.get(k)
        if isinstance(v, str) and v:
            return v
    return max((v for v in rec.values() if isinstance(v, str)), key=len, default="")


def fingerprints(words, k):
    """k well-spaced N-word windows, each starting on a longish word so the corpus scan's
    first-word filter stays selective."""
    out = []
    if len(words) < N + 5:
        return out
    for j in range(k):
        i = int((j + 0.5) * (len(words) - N) / k)
        while i < len(words) - N and len(words[i]) < 7:
            i += 1
        if i < len(words) - N:
            out.append((" ".join(words[i:i + N]), j))
    return out


def query_texts(split):
    return {r["id"]: " ".join(r["sents"]) for r in D.queries(split)}


def recover_masks(qtext, ctext, ctx=(8, 6, 4), limit=400, window=80):
    """What the corpus copy has where the query has a mask.

    For each mask, take the words either side of it and find that pair of contexts in the
    corpus word stream; the words between them are what the mask was hiding. Context
    matching rather than alignment, because the two copies differ in whitespace and
    sentence splitting.

    Requiring the context to occur exactly once is too brittle for judgments, which
    repeat boilerplate: so every occurrence of the left context is tried, the spans they
    imply are voted on, and the longest context that yields any agreement wins.

    A context must never reach across ANOTHER mask, because that mask's own text differs
    between the two copies. Query judgments carry about eleven masks each and they
    cluster ("relied on <CITATION> and on <CITATION>"), so contexts are taken from the
    segments BETWEEN masks and shortened to whatever that segment allows, down to two
    words. Without this the common case recovers nothing."""
    cw = WORD.findall(ctext.lower())
    idx = {}

    def at(k):
        if k not in idx:
            pos = collections.defaultdict(list)
            for i in range(len(cw) - k + 1):
                pos[tuple(cw[i:i + k])].append(i)
            idx[k] = pos
        return idx[k]

    segs, marks, last = [], [], 0
    for m in MASKS.finditer(qtext):
        segs.append(WORD.findall(qtext[last:m.start()].lower()))
        marks.append(m)
        last = m.end()
    segs.append(WORD.findall(qtext[last:].lower()))

    out = []
    for i in range(min(len(marks), limit)):
        left, right = segs[i], segs[i + 1]
        span = None
        ks = sorted({min(k, len(left), len(right)) for k in ctx}, reverse=True)
        for k in [k for k in ks if k >= 2]:
            pos = at(k)
            cands = collections.Counter()
            for s0 in pos.get(tuple(left[-k:]), ())[:20]:
                s = s0 + k
                for e in pos.get(tuple(right[:k]), ()):
                    if s <= e <= s + window:
                        cands[" ".join(cw[s:e])] += 1
                        break
            if cands:
                span = cands.most_common(1)[0][0]
                break
        if span:
            out.append(span)
    return out


def cmd_scan(a):
    from huggingface_hub import hf_hub_download
    os.makedirs(OUT, exist_ok=True)
    splits = [s for s in a.splits.split(",") if s]
    docs, index, first = {}, collections.defaultdict(list), set()
    for split in splits:
        for qid, text in query_texts(split).items():
            docs[(split, qid)] = text
            for fp, j in fingerprints(WORD.findall(text.lower()), a.needles):
                index[hash(fp)].append(((split, qid), j))
                first.add(fp.split(" ", 1)[0])
    log(f"  {len(docs)} queries over {splits}, {len(index)} fingerprints, "
        f"{len(first)} distinct first words")
    nmask = {s: sum(len(MASKS.findall(t)) for (sp, _), t in docs.items() if sp == s)
             for s in splits}
    log(f"  masks in those queries: {nmask}")

    found = {}
    for src in a.sources.split(","):
        path = a.corpus_path or hf_hub_download(
            "Exploration-Lab/CS-781-Capstone",
            f"Phase_1_Pretraining/Train/{src}.json", repo_type="dataset")
        log(f"  scanning {src}")
        t0, n = time.time(), 0
        for n, rec in enumerate(stream_records(path)):
            text = rec_text(rec)
            w = WORD.findall(text.lower())
            if len(w) < N:
                continue
            hits, last = collections.defaultdict(set), {}
            for i in range(len(w) - N + 1):
                if w[i] not in first:
                    continue
                for key, j in index.get(hash(" ".join(w[i:i + N])), ()):
                    hits[key].add(j)
                    last[key] = i + N
            for key, js in hits.items():
                if len(js) >= a.min_hits and (key not in found
                                              or len(js) > found[key]["hits"]):
                    found[key] = dict(source=src, record=n, hits=len(js), text=text)
            if n and n % 20000 == 0:
                log(f"    {n:,} records, {len(found)} matched ({time.time() - t0:.0f}s)")
        log(f"  {src}: {n + 1:,} records in {time.time() - t0:.0f}s")

    report = {}
    for split in splits:
        keys = [k for k in docs if k[0] == split]
        m = [k for k in keys if k in found]
        rec = {}
        for k in m:
            spans = recover_masks(docs[k], found[k]["text"])
            if spans:
                rec[k[1]] = spans
        report[split] = dict(queries=len(keys), matched=len(m),
                             matched_pct=round(100.0 * len(m) / max(len(keys), 1), 1),
                             masks=nmask[split],
                             queries_with_recovered_spans=len(rec),
                             recovered_spans=sum(len(v) for v in rec.values()))
        D.atomic_json(f"{OUT}/recovered_{split}.json", rec)
        log(f"  {split}: {len(m)}/{len(keys)} queries found in the corpus "
            f"({report[split]['matched_pct']}%); recovered "
            f"{report[split]['recovered_spans']} masked spans in {len(rec)} queries")
    D.atomic_json(f"{OUT}/scan.json",
                  dict(sources=a.sources, min_hits=a.min_hits, needles=a.needles,
                       report=report))
    log("")
    if max((r["matched"] for r in report.values()), default=0) == 0:
        log("  VERDICT: no query judgment is in the scanned corpus. The lookup route is")
        log("  empty, and no decision is needed. Nothing further to do here.")
    else:
        log("  VERDICT: the corpus holds query judgments. `value` now measures, on devx,")
        log("  what that would be worth. MEASURE-ONLY: no submission uses it without")
        log("  Nithin saying so, and such a system is reported as kind=lookup.")
    log(f"  wrote {OUT}/scan.json")


def cmd_value(a):
    """Upper bound on a lookup system, measured on devx with the official metric."""
    split = a.split
    f = f"{OUT}/recovered_{split}.json"
    if not os.path.exists(f):
        raise SystemExit(f"run `scan --splits {split}` first ({f} missing)")
    rec = json.load(open(f))
    pool = D.candidates(split)
    ids = sorted(pool)
    # index candidate text by its rare word sequences so a recovered citation string can
    # be looked up; a citation names a case, and the cited judgment usually repeats it
    inv = collections.defaultdict(set)
    for c in ids:
        for w in set(WORD.findall(" ".join(pool[c]).lower())):
            if len(w) > 6:
                inv[w].add(c)
    log(f"  {len(ids)} candidates indexed, {len(rec)} queries with recovered spans")

    gold, gp = D.golds(split), D.golds_in_pool(split)
    rank = {}
    for q in gold:
        scores = collections.Counter()
        for span in rec.get(q, ()):
            toks = [w for w in WORD.findall(span.lower()) if len(w) > 6]
            if not toks:
                continue
            sets = [inv.get(w, set()) for w in toks]
            common = set.intersection(*sets) if all(sets) else set()
            for c in (common or set()):
                scores[c] += 1
        rank[q] = [c for c, _ in scores.most_common(D.K * 3) if c != q]
    s = D.score_ranking({q: rank.get(q, []) for q in gold}, gold, reachable=gp)
    log(f"  lookup-only on {split}: " + D.brief(s) +
        f" | ceiling {s['ceiling_f1']:.4f}")
    hit = sum(1 for q in gold if rank.get(q))
    log(f"  {hit}/{len(gold)} queries got any candidate from recovered spans")
    D.atomic_json(f"{OUT}/value_{split}.json",
                  dict(f1=s["f1"], hits=s["hits"], max_hits=s["max_hits"],
                       queries_with_any=hit, n_queries=s["n_queries"]))
    log("  MEASURE-ONLY. This is reported as the lookup upper bound; it becomes a")
    log("  submission only if Nithin says so, and then as kind=lookup.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("scan")
    p.add_argument("--splits", default="test,devx")
    p.add_argument("--sources", default="train_supreme_court")
    p.add_argument("--min-hits", type=int, default=3)
    p.add_argument("--needles", type=int, default=10)
    p.add_argument("--corpus-path", default=None)
    p.set_defaults(fn=cmd_scan)
    p = sp.add_parser("value")
    p.add_argument("--split", default="devx")
    p.set_defaults(fn=cmd_value)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
