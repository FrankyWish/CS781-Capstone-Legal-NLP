#!/usr/bin/env python3
# LEGALSEG_CONVERTER_V3
"""CS-781 Phase 3 -- LegalSeg (Nigam et al., L-NLProc/LegalSeg_CSV) as external RR data. CPU.

    python phase3_legalseg.py --max-docs 2000

LegalSeg annotates ~7k Indian judgments with 7 roles (Facts, Issue, Arguments of
Petitioner, Arguments of Respondent, Reasoning, Decision, None). Mapped onto our
13 roles as allowed sets, never a default label:
  facts -> {Fact, RulingByLowerCourt}      (its Facts include lower-court history)
  issue -> {Issue}
  arguments of petitioner -> {ArgumentPetitioner}
  arguments of respondent -> {ArgumentRespondent}
  reasoning/analysis -> {Ratio, PrecedentReliedUpon, PrecedentNotReliedUpon,
                         PrecedentOverruled, Statute}
  decision -> {RulingByPresentCourt}
  none -> no supervision (LegalSeg's catch-all: ~50% of sentences vs <2% None in ours)
Any label not recognised is masked (no supervision) and reported.

Every document is filtered against the 10 test documents (record level 13-gram
>= 50% drops it, overlapping sentences are masked), as for MARRO and Kalamkar.
Writes external/legalseg.jsonl for `phase3_rr.py --extra`.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from phase3_data import P3, LABELS, ABBR, load_all, log, contamination_filter  # noqa: E402

IX = {l: i for i, l in enumerate(LABELS)}
REASON = ["RatioOfTheDecision", "PrecedentReliedUpon", "PrecedentNotReliedUpon",
          "PrecedentOverruled", "Statute"]


def map_label(raw):
    s = re.sub(r"[^a-z]", " ", str(raw).lower()).strip()
    if "petition" in s or "appellant" in s or s in ("aop",):
        return ["ArgumentPetitioner"]
    if "respond" in s or s in ("aor",):
        return ["ArgumentRespondent"]
    if s.startswith("fact"):
        return ["Fact", "RulingByLowerCourt"]
    if s.startswith("issue"):
        return ["Issue"]
    if "reason" in s or "analysis" in s or "ratio" in s:
        return REASON
    if "decision" in s or "ruling" in s or "judgment" in s or "conclusion" in s:
        return ["RulingByPresentCourt"]
    if s in ("none", "other", "o", "na", "preamble"):
        # LegalSeg puts ~50% of all sentences under None; ours has <2%. Its None
        # is a catch-all, so it is left unsupervised (context only), not taught.
        return []
    return None


def pick(cols, *keys):
    for k in keys:
        for c in cols:
            if k in c.lower():
                return c
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-docs", type=int, default=2000, help="random sample of documents (0 = all)")
    ap.add_argument("--splits", nargs="+", default=["train.csv", "val.csv", "test.csv"])
    ap.add_argument("--seed", type=int, default=781)
    a = ap.parse_args()

    import pandas as pd
    from huggingface_hub import hf_hub_download
    frames = []
    for f in a.splits:
        p = hf_hub_download("L-NLProc/LegalSeg_CSV", f, repo_type="dataset")
        # keep_default_na=False: LegalSeg's "None" role must stay the string
        # "None"; pandas would otherwise read it as a missing value.
        df = pd.read_csv(p, keep_default_na=False, na_values=[""])
        df["__split"] = f
        frames.append(df)
        log(f"  {f}: {len(df)} rows, columns {list(df.columns)}")
    df = pd.concat(frames, ignore_index=True)
    cols = list(df.columns)
    c_doc = pick(cols, "doc", "file", "case")
    c_txt = pick(cols, "sentence", "text", "sent")
    c_lab = pick(cols, "label", "role", "tag")
    c_idx = pick(cols, "index")
    assert c_txt and c_lab, f"cannot identify text/label columns in {cols}"
    raw = collections.Counter(df[c_lab].astype(str))
    log("  raw labels: " + ", ".join(f"{k} {v}" for k, v in raw.most_common(20)))
    log(f"  first rows: {df.head(3)[[c for c in cols if c != c_txt]].to_dict('records')}")
    if c_doc is None and c_idx and df.groupby("__split")[c_idx].nunique().sum() < len(df) / 5:
        # Index repeats across consecutive rows: it is the DOCUMENT number within
        # each split file (rows of one judgment share it), and row order is
        # sentence order. Keys below combine it with the split.
        c_doc = c_idx
        n_docs = int(df.groupby("__split")[c_idx].nunique().sum())
        log(f"  {c_idx!r} is a document id: {n_docs} documents "
            f"(mean {len(df)/max(1,n_docs):.0f} sentences each)")
        assert 200 < n_docs < 50000, f"{c_idx!r} has only {n_docs} distinct values -- cannot split documents, stop"
    if c_doc is None:
        # Otherwise Index would be the sentence position inside its judgment,
        # so a new document starts wherever it fails to increase.
        assert c_idx, f"no document id and no index column in {cols}"
        idx = df[c_idx].astype(int).tolist()
        doc, cur = [], 0
        for i, v in enumerate(idx):
            if i and v <= idx[i - 1]:
                cur += 1
            doc.append(cur)
        df["__doc"] = doc
        c_doc = "__doc"
        n_docs = cur + 1
        log(f"  documents inferred from {c_idx!r} restarts: {n_docs} "
            f"(mean {len(df)/n_docs:.0f} sentences each)")
        assert 200 < n_docs < 50000, "Index does not look like a per-document position -- stop"
    log(f"  using doc={c_doc!r} text={c_txt!r} label={c_lab!r}")
    unmapped = sorted(k for k in raw if map_label(k) is None)
    if unmapped:
        log(f"  WARNING unmapped (masked): {unmapped}")

    unmapped_ok = not unmapped
    docs_by = collections.OrderedDict()
    for split, d, t, l in zip(df["__split"], df[c_doc], df[c_txt], df[c_lab]):
        if str(l).strip().lstrip("-").isdigit() and map_label(l) is None and not unmapped_ok:
            raise SystemExit(f"labels are numeric codes ({sorted(raw)[:10]}): add the LegalSeg "
                             f"code -> name mapping before converting")
        k = f"{split}:{d}"
        docs_by.setdefault(k, ([], []))
        if isinstance(t, str) and t.strip():
            docs_by[k][0].append(t.strip())
            docs_by[k][1].append(str(l))
    keys = list(docs_by)
    random.Random(a.seed).shuffle(keys)
    if a.max_docs:
        keys = keys[:a.max_docs]
    log(f"  {len(docs_by)} documents in total; keeping {len(keys)}")

    comp, _ = load_all()
    test = {d: r for d, r in comp.items() if r["split"] == "test"}
    recs = [(f"lseg_{i}", docs_by[k][0], docs_by[k][1]) for i, k in enumerate(keys)]
    kept, dropped = contamination_filter(recs, test)

    out = f"{P3}/external/legalseg.jsonl"
    os.makedirs(f"{P3}/external", exist_ok=True)
    cnt = collections.Counter()
    n_s = n_sup = 0
    with open(out + ".tmp", "w", encoding="utf-8") as fh:
        for rid, sents, labs, hit in kept:
            hit = set(hit)
            allowed = []
            for i, l in enumerate(labs):
                m = None if i in hit else map_label(l)
                allowed.append([IX[x] for x in m] if m else [])
                n_s += 1
                n_sup += bool(m)
                cnt["/".join(ABBR[x] for x in m) if m else "(none)"] += 1
            fh.write(json.dumps(dict(id=rid, domain="EXT", text=sents, allowed=allowed),
                                ensure_ascii=False) + "\n")
    os.replace(out + ".tmp", out)
    log(f"  wrote {out}: {len(kept)} documents, {n_s} sentences, {n_sup} supervised")
    log("  target sets: " + ", ".join(f"{k} {v}" for k, v in cnt.most_common(10)))


if __name__ == "__main__":
    main()
