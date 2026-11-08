#!/usr/bin/env python
"""CS-781 Phase 7 -- rule N: match recovered case NAMES to candidates' true titles.

    python phase7_names.py scan     find devx/test candidates in the Phase-1 corpus (~6 min)
    python phase7_names.py match    recovered query names -> candidate titles, swept on devx

The corpus lookup (phase7_overlap.py) recovers what a query's masks hid, and for a
<CITATION> mask that is mostly a case NAME plus year ("union of india v delhi cloth general
mills 1963"), not a reporter citation -- so rule L found 9 pairs.  Candidate titles are
masked in the released files (PETITIONER: <NAME> Vs. ...), but if a candidate also sits in
the Phase-1 corpus, its corpus copy carries the real "PETITIONER: X Vs. RESPONDENT: Y".
Matching recovered names against those titles is the name-level citation edge.

Writes results/names/titles_{split}.json and results/names/N_{split}.json {qid: [cid]}.
kind = "lookup" (uses the permitted Phase-1 corpus; reported beside the hybrid number).
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import phase7_data as D      # noqa: E402
import phase7_overlap as O   # noqa: E402

P7 = D.P7
OUT = f"{P7}/results/names"
WORD = re.compile(r"[a-z0-9]+")
TITLE = re.compile(r"PETITIONER\s*:?\s*(.{2,200}?)\s+Vs?\.?\s*RESPONDENT\s*:?\s*(.{2,200}?)\s+"
                   r"DATE\s+OF\s+JUDGMENT\s*:?\s*\d{1,2}[/.-]\d{1,2}[/.-](\d{4})", re.I | re.S)
TITLE2 = re.compile(r"^\s*(.{3,150}?)\s+(?:vs?\.?|versus)\s+(.{3,150}?)\s+on\s+\d{1,2}\s+\w+,?\s+(\d{4})",
                    re.I | re.S)                     # indiankanoon style "A vs B on 5 May, 1990"
STOP = set("""a an and the of in on for to v vs versus ors anr others another etc ltd co
company m s mr mrs smt shri sri dr messrs pvt private limited by through its""".split())
YEARS = re.compile(r"\b(1[89]\d\d|20[0-2]\d)\b")
VPAT = re.compile(r"([a-z][a-z0-9 ]{2,200}?) vs? ([a-z][a-z ]{2,120}?)(?: (1[89]\d\d|20[0-2]\d)\b.*)?$")


def log(*a):
    print(" ", *a, flush=True)


def toks(s):
    return [w for w in WORD.findall(s.lower()) if w not in STOP and not w.isdigit() and len(w) > 1]


def cmd_scan(a):
    from huggingface_hub import hf_hub_download
    os.makedirs(OUT, exist_ok=True)
    splits = a.splits.split(",")
    docs, index, first = {}, collections.defaultdict(list), set()
    for sp in splits:
        for c, v in D.candidates(sp).items():
            t = " ".join(v)
            docs[(sp, c)] = t
            for fp, j in O.fingerprints(WORD.findall(t.lower()), a.needles):
                index[hash(fp)].append(((sp, c), j))
                first.add(fp.split(" ", 1)[0])
    log(f"{len(docs)} candidates, {len(index)} fingerprints")
    found = {}
    for src in a.sources.split(","):
        path = hf_hub_download("Exploration-Lab/CS-781-Capstone",
                               f"Phase_1_Pretraining/Train/{src}.json", repo_type="dataset")
        t0 = time.time()
        for n, rec in enumerate(O.stream_records(path)):
            text = O.rec_text(rec)
            w = WORD.findall(text.lower())
            hits = collections.defaultdict(set)
            for i in range(len(w) - O.N + 1):
                if w[i] not in first:
                    continue
                for key, j in index.get(hash(" ".join(w[i:i + O.N])), ()):
                    hits[key].add(j)
            for key, js in hits.items():
                if len(js) >= a.min_hits and (key not in found or len(js) > found[key][0]):
                    found[key] = (len(js), text[:3000])
            if n and n % 20000 == 0:
                log(f"  {n:,} records, {len(found)} candidates matched ({time.time() - t0:.0f}s)")
    for sp in splits:
        titles = {}
        for (s2, c), (_, head) in found.items():
            if s2 != sp:
                continue
            m = TITLE.search(head) or TITLE2.search(head)
            if m:
                titles[c] = {"p1": m.group(1), "p2": m.group(2), "year": int(m.group(3))}
            else:
                titles[c] = {"p1": head[:200], "p2": "", "year": None}
        D.atomic_json(f"{OUT}/titles_{sp}.json", titles)
        n_sp = sum(1 for k in docs if k[0] == sp)
        log(f"{sp}: {len(titles)}/{n_sp} candidates found in corpus; "
            f"{sum(1 for t in titles.values() if t['p2'])} with a parsed title; "
            f"e.g. {list(titles.values())[:3]}")


def spans_names(spans):
    out = []
    for sp in spans:
        if sp.startswith("@single "):           # "in X's case": one party only
            s = " ".join(WORD.findall(sp[8:].lower()))
            ys = YEARS.findall(s)
            p1 = toks(s)[-6:]
            if p1:
                out.append((p1, [], int(ys[0]) if ys else None))
            continue
        s = " ".join(WORD.findall(sp.lower()))
        if len(s) > 400:
            continue
        m = VPAT.search(s)
        if not m:
            continue
        p1, p2, y = toks(m.group(1))[-6:], toks(m.group(2)), m.group(3)
        if not y:
            ys = YEARS.findall(s)
            y = ys[0] if ys else None
        out.append((p1, p2, int(y) if y else None))
    return out


def match_scores(split, need_year, rec=None):
    """-> {qid: {cid: best name score}} over every recovered name span of the query
    (or over `rec`, e.g. names read straight from the released query text)."""
    tf = f"{OUT}/titles_{split}.json"
    rf = f"{P7}/results/overlap/recovered_{split}.json"
    if not os.path.exists(tf) or (rec is None and not os.path.exists(rf)):
        return None
    titles = json.load(open(tf))
    if rec is None:
        rec = json.load(open(rf))
    tt = {c: (set(toks(t["p1"])), set(toks(t["p2"])), t["year"]) for c, t in titles.items()}
    df = collections.Counter(w for a, b, _ in tt.values() for w in a | b)
    nt = max(len(tt), 1)
    idf = {w: math.log((nt + 1) / (n + 1)) + 1.0 for w, n in df.items()}
    unk = math.log(nt + 1) + 1.0
    inv = collections.defaultdict(set)
    for c, (a, b, _) in tt.items():
        for w in a | b:
            inv[w].add(c)

    def side(qs, cs):
        if not qs:
            return 0.0
        den = sum(idf.get(w, unk) for w in qs)
        return sum(idf[w] for w in qs if w in cs) / den

    out = {}
    for q, spans in rec.items():
        best = {}
        for p1, p2, y in spans_names(spans):
            qs1, qs2 = set(p1), set(p2)
            pool = qs1 | qs2
            cands = set().union(*[inv.get(w, set()) for w in pool]) if pool else set()
            for c in cands:
                if c == q:
                    continue
                a, b, cy = tt[c]
                if need_year and y and cy and abs(y - cy) > 1:
                    continue
                s1 = max(side(qs1, a), side(qs1, b))
                s2 = max(side(qs2, b), side(qs2, a))
                sc = (s1 + s2) / 2 if qs1 and qs2 else max(s1, s2) * 0.8
                if y and cy and abs(y - cy) <= 1:
                    sc += 0.15
                if sc > best.get(c, 0):
                    best[c] = sc
        out[q] = best
    return out


def cut(S, thr):
    return {q: sorted([c for c, v in d.items() if v >= thr], key=d.get, reverse=True)
            for q, d in S.items()}


def cmd_match(a):
    os.makedirs(OUT, exist_ok=True)
    g = D.golds_in_pool("devx")
    rows = []
    for need_year in (True, False):
        S = match_scores("devx", need_year)
        if S is None:
            raise SystemExit("run `scan` (and phase7_overlap.py scan) first")
        for thr in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
            N = cut(S, thr)
            n = sum(len(v) for q, v in N.items() if q in g)
            tp = sum(len(set(v) & set(g[q])) for q, v in N.items() if q in g)
            rows.append((thr, need_year, n, tp))
            log(f"devx N thr {thr} year {need_year}: predicted {n}, correct {tp} "
                f"(precision {tp / max(n, 1):.3f})")
    ok = [r for r in rows if r[2] and r[3] / r[2] >= a.min_precision]
    if a.thr is not None:                     # forced threshold (year not required)
        ok = [r for r in rows if r[0] == a.thr and not r[1]]
    if not ok:
        log("no threshold reaches the precision bar -- rule N not written")
        return
    thr, need_year, n, tp = max(ok, key=lambda r: (r[3], r[3] / r[2]))
    log(f"CHOSE thr {thr} year {need_year}: {tp}/{n} correct on devx")
    for sp in ("devx", "test"):
        S = match_scores(sp, need_year)
        if S is not None:
            N = cut(S, thr)
            D.atomic_json(f"{OUT}/N_{sp}.json", N)
            Nh = cut(S, max(thr, a.hi))          # strict tier for the citator's `t` variants
            D.atomic_json(f"{OUT}/Nhi_{sp}.json", Nh)
            log(f"{sp}: rule N pairs {sum(len(v) for v in N.values())} over "
                f"{sum(1 for v in N.values() if v)} queries")


# ---------------------------------------------------------------- rule M: names in the
# released query text itself.  Masking hid formal citations and most party names, but
# rule B proved it missed reporter cites; it also misses some "X v. Y" / "X's case".
VBODY = re.compile(r"([A-Z][A-Za-z.&'\- ]{2,80}?)\s+(?:v\.|vs\.?|versus|V\.|Vs\.?)\s+"
                   r"([A-Z][A-Za-z.&'\- ]{2,80}?)(?=\s*[,;:(\[]|\s+\d|\s+(?:and|AIR|SCR|SCC|"
                   r"reported|where|wherein|in|the|it|this|that|is|was|has)\b|$)")
SCASE = re.compile(r"\b(?:in|of)\s+([A-Z][A-Za-z.&'\-]+(?:\s+[A-Z][A-Za-z.&'\-]+){0,4})'s\s+[Cc]ase")
JUNK = set("""see also case cases decision decisions judgment held reported supra court
learned counsel relied reliance placed upon referred followed observed this that
as per vide decided dictum ratio""".split())


def body_spans(text):
    out = []
    for m in VBODY.finditer(text):
        p1 = [w for w in WORD.findall(m.group(1).lower()) if w not in JUNK]
        p2 = [w for w in WORD.findall(m.group(2).lower()) if w not in JUNK]
        if not p1 or not p2:
            continue
        ys = YEARS.findall(text[m.end():m.end() + 60])
        out.append(" ".join(p1) + " v " + " ".join(p2) + (f" {ys[0]}" if ys else ""))
    for m in SCASE.finditer(text):
        p1 = [w for w in WORD.findall(m.group(1).lower()) if w not in JUNK]
        if p1:
            ys = YEARS.findall(text[m.end():m.end() + 60])
            out.append("@single " + " ".join(p1) + (f" {ys[0]}" if ys else ""))
    return out


def cmd_body(a):
    os.makedirs(OUT, exist_ok=True)
    spans = {}
    for sp in ("devx", "test"):
        spans[sp] = {r["id"]: body_spans(" ".join(r["sents"])) for r in D.queries(sp)}
        n = sum(1 for v in spans[sp].values() if v)
        ex = [s for v in spans[sp].values() for s in v][:15]
        log(f"{sp}: {n}/{len(spans[sp])} queries have name spans in their own text, "
            f"{sum(len(v) for v in spans[sp].values())} spans; e.g. {ex}")
    rec = {}
    rf = f"{P7}/results/overlap/recovered_{'devx'}.json"
    if os.path.exists(rf):
        rec = json.load(open(rf))
    g = D.golds_in_pool("devx")
    old = {}
    nf = f"{OUT}/N_devx.json"
    if os.path.exists(nf):
        old = json.load(open(nf))
    rows = []
    for need_year in (True, False):
        S = match_scores("devx", need_year, rec=spans["devx"])
        if S is None:
            raise SystemExit("titles_devx.json missing -- run `scan` first")
        for thr in (0.6, 0.7, 0.8, 0.9, 1.0, 1.1):
            M = cut(S, thr)
            n = sum(len(v) for q, v in M.items() if q in g)
            tp = sum(len(set(v) & set(g[q])) for q, v in M.items() if q in g)
            new = sum(len(set(v) & set(g[q]) - set(old.get(q, []))) for q, v in M.items() if q in g)
            unc = sum(1 for q, v in M.items() if v and q not in rec)
            rows.append((thr, need_year, n, tp))
            log(f"devx M thr {thr} year {need_year}: predicted {n}, correct {tp} "
                f"(precision {tp / max(n, 1):.3f}); golds not already in N {new}; "
                f"queries without recovered spans that get pairs {unc}")
    ok = [r for r in rows if r[2] and r[3] / r[2] >= a.min_precision]
    if not ok:
        log("rule M: no threshold reaches the precision bar -- not written")
        for sp in ("devx", "test"):
            D.atomic_json(f"{OUT}/M_{sp}.json", {})
        return
    thr, need_year, n, tp = max(ok, key=lambda r: (r[3], r[3] / r[2]))
    log(f"CHOSE rule M thr {thr} year {need_year}: {tp}/{n} correct on devx")
    for sp in ("devx", "test"):
        M = cut(match_scores(sp, need_year, rec=spans[sp]), thr)
        D.atomic_json(f"{OUT}/M_{sp}.json", M)
        log(f"{sp}: rule M pairs {sum(len(v) for v in M.values())} over "
            f"{sum(1 for v in M.values() if v)} queries")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan")
    s.add_argument("--splits", default="devx,test")
    s.add_argument("--sources", default="train_supreme_court")
    s.add_argument("--needles", type=int, default=10)
    s.add_argument("--min-hits", type=int, default=3)
    m = sub.add_parser("match")
    m.add_argument("--min-precision", type=float, default=0.8)
    m.add_argument("--thr", type=float, default=None, help="force this name-score cut")
    m.add_argument("--hi", type=float, default=0.9, help="strict tier cut (Nhi_*.json)")
    b = sub.add_parser("body")
    b.add_argument("--min-precision", type=float, default=0.8)
    a = p.parse_args()
    {"scan": cmd_scan, "match": cmd_match, "body": cmd_body}[a.cmd](a)


if __name__ == "__main__":
    main()
