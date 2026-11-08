#!/usr/bin/env python3
"""CS-781 Phase 3 -- external rhetorical-role data, mapped and contamination-filtered.

    python phase3_external.py marro                  # MARRO IN set (public)
    python phase3_external.py kalamkar               # opennyaiorg/InRhetoricalRoles (gated: click "agree")
    python phase3_external.py kalamkar --analysis ratio

Writes {P3}/external/<name>.jsonl for `phase3_rr.py --extra`. Each record holds a
per-sentence ALLOWED LABEL SET, not a single label:

  a fine label that maps 1:1            -> {that label}
  a coarse label (Argument, Precedent)  -> {every label it could be}; the loss
                                           marginalises over the set, so the
                                           model learns "this is an Argument"
                                           without being told which party
  a label with no counterpart           -> {} = all 13 allowed = no supervision,
                                           but the sentence still provides
                                           sequence context

Never mapping an unmatched class onto a default label is deliberate: in Phase 2
teaching unmapped classes as a default actively unlearned working labels.

WHAT IS NOT HERE, ON PURPOSE: IL-TUR's `rr` config. Its six files are the
competition's CL/IT train/dev/test splits (same 40/5/5 documents per domain,
file sizes in a constant ratio to the competition's), and its test split carries
gold labels. Training on it, or on anything derived from it, is reading the
answer key. It adds nothing legitimate either -- its train/dev are ours.

Every record is filtered against the 10 test documents at record level (13-gram
overlap >= 50% drops the record) and sentence level (a sentence whose own
13-grams hit the test set is masked to no-supervision). The filter's log goes
in EXPERIMENTS.md.
"""

from __future__ import annotations

import argparse
import collections
import io
import json
import os
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from phase3_data import P3, LABELS, ABBR, load_all, log, contamination_filter, grams  # noqa: E402

IX = {l: i for i, l in enumerate(LABELS)}
ARG = ["ArgumentPetitioner", "ArgumentRespondent"]
PREC = ["PrecedentReliedUpon", "PrecedentNotReliedUpon", "PrecedentOverruled"]

# Bhattacharya et al. 2019 scheme (7 roles), as distributed with MARRO.
MARRO_MAP = {
    "FAC": ["Fact"], "RLC": ["RulingByLowerCourt"], "ARG": ARG, "STA": ["Statute"],
    "PRE": PREC, "Ratio": ["RatioOfTheDecision"], "RPC": ["RulingByPresentCourt"],
}
# Kalamkar et al. 2022 scheme (13 roles).
KALAMKAR_MAP = {
    "FAC": ["Fact"], "RLC": ["RulingByLowerCourt"], "ISSUE": ["Issue"],
    "ARG_PETITIONER": ["ArgumentPetitioner"], "ARG_RESPONDENT": ["ArgumentRespondent"],
    "STA": ["Statute"], "PRE_RELIED": ["PrecedentReliedUpon"],
    "PRE_NOT_RELIED": ["PrecedentNotReliedUpon"], "RATIO": ["RatioOfTheDecision"],
    "RPC": ["RulingByPresentCourt"], "NONE": ["None"],
    "PREAMBLE": ["None", "Fact"],
    "ANALYSIS": [],                     # set by --analysis
}


def hf_file(repo, name):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, name, repo_type="dataset")


def load_marro():
    recs = []
    for name in ("IN-train-set.zip", "IN-test-set.zip"):
        with zipfile.ZipFile(hf_file("subinay612/MARRO_Rhetorical-Role-Labeling", name)) as z:
            for f in sorted(z.namelist()):
                if not f.endswith(".txt"):
                    continue
                sents, labs = [], []
                for line in z.read(f).decode("utf-8", "replace").splitlines():
                    if "\t" not in line:
                        continue
                    s, l = line.rsplit("\t", 1)
                    if s.strip():
                        sents.append(s.strip())
                        labs.append(l.strip())
                if sents:
                    recs.append((f"marro_{os.path.basename(f)[:-4]}", sents, labs))
    return recs, MARRO_MAP


def load_kalamkar(analysis):
    import pandas as pd
    from huggingface_hub import list_repo_files
    repo = "opennyaiorg/InRhetoricalRoles"
    files = [f for f in list_repo_files(repo, repo_type="dataset") if f.endswith(".parquet")]
    log(f"  {repo}: {files}")
    m = dict(KALAMKAR_MAP)
    m["ANALYSIS"] = {"mask": [], "ratio": ["RatioOfTheDecision"],
                     "partial": ["RatioOfTheDecision", "None"]}[analysis]
    recs = []
    for f in files:
        df = pd.read_parquet(hf_file(repo, f))
        if f == files[0]:
            log(f"  columns: {list(df.columns)}")
        for k, row in df.iterrows():
            sents, labs = _kalamkar_row(row)
            if sents and any(l is not None for l in labs):
                rid = str(row.get("id", f"{os.path.basename(f)}_{k}"))
                recs.append((f"kal_{rid}", sents, labs))
            elif k == 0:
                log(f"  WARNING: {f} row 0 has no labels -- unlabelled split, skipped")
    return recs, m


def _kalamkar_row(row):
    """Label Studio export (annotations/result/value) or flat text/labels lists."""
    def g(x, k):
        return x.get(k) if isinstance(x, dict) else getattr(x, k, None)
    if "annotations" in row and row["annotations"] is not None:
        ann = row["annotations"]
        ann = ann[0] if len(ann) else None
        res = g(ann, "result") if ann is not None else []
        sents, labs = [], []
        for r in (res if res is not None else []):
            v = g(r, "value")
            lab = g(v, "labels")
            sents.append(str(g(v, "text")).strip())
            labs.append(str(lab[0]) if lab is not None and len(lab) else None)
        return sents, labs
    if "text" in row and "labels" in row:
        return [str(s) for s in row["text"]], [str(l) for l in row["labels"]]
    raise ValueError(f"unknown schema: {list(row.index)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", choices=["marro", "kalamkar"])
    ap.add_argument("--analysis", default="mask", choices=["mask", "ratio", "partial"],
                    help="Kalamkar ANALYSIS: no supervision / Ratio / {Ratio, None}")
    ap.add_argument("--thresh", type=float, default=0.5)
    a = ap.parse_args()

    docs, _ = load_all()
    test = {d: r for d, r in docs.items() if r["split"] == "test"}
    lab = {d: r for d, r in docs.items() if r["split"] != "test"}
    recs, mp = load_marro() if a.source == "marro" else load_kalamkar(a.analysis)
    raw = collections.Counter(l for _, _, ls in recs for l in ls)
    log(f"  {a.source}: {len(recs)} documents, {sum(raw.values())} sentences")
    log("  source labels: " + ", ".join(f"{k} {v}" for k, v in raw.most_common()))
    unknown = sorted(k for k in raw if k is not None and k not in mp)
    if unknown:
        log(f"  WARNING: unmapped source labels (masked): {unknown}")

    # how much of OUR labelled data is in this corpus -- information, not a filter
    lg = set()
    for r in lab.values():
        lg |= grams(" ".join(r["text"]))
    shared = sum(1 for _, s, _ in recs if len(grams(" ".join(s)) & lg) / max(1, len(grams(" ".join(s)))) > 0.5)
    log(f"  records that are >50% copies of a competition train/dev document: {shared}")

    kept, dropped = contamination_filter([(rid, s, l) for rid, s, l in recs], test, thresh=a.thresh)
    os.makedirs(f"{P3}/external", exist_ok=True)
    name = a.source + ("" if a.source == "marro" else f"_{a.analysis}")
    out = f"{P3}/external/{name}.jsonl"
    cnt = collections.Counter()
    n_sent = n_sup = 0
    tmp = f"{out}.tmp{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        for rid, sents, labs, hit in kept:
            allowed = []
            hit = set(hit)
            for i, l in enumerate(labs):
                tgt = [] if (i in hit or l is None) else [IX[x] for x in mp.get(l, [])]
                allowed.append(tgt)
                n_sent += 1
                n_sup += bool(tgt)
                cnt["/".join(ABBR[LABELS[j]] for j in tgt) or "(none)"] += 1
            f.write(json.dumps(dict(id=rid, domain="EXT", text=sents, allowed=allowed),
                               ensure_ascii=False) + "\n")
    os.replace(tmp, out)
    log(f"  wrote {out}: {len(kept)} documents, {n_sent} sentences, {n_sup} supervised "
        f"({100*n_sup/max(1,n_sent):.0f}%)")
    log("  target sets: " + ", ".join(f"{k} {v}" for k, v in cnt.most_common(12)))


if __name__ == "__main__":
    main()
