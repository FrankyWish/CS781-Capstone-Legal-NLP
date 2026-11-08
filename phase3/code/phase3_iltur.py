#!/usr/bin/env python3
"""CS-781 Phase 3 -- IL-TUR `rr` as training data, incl. its test split. CPU.

    python phase3_iltur.py

PERMISSION. Training on IL-TUR `rr` including its test split was explicitly
permitted by the course TA on 2026-10-15 ("Yes. you can do it, as prof clearly
mentioned in the class"), in reply to a question naming the test split. The
report must say so, and must give the model-only result (0.5458, trained
without IL-TUR) beside any number from models trained with it.

What this does:
  1. downloads the six rr parquet files (CL/IT x train/dev/test)
  2. reads the label names from the parquet's own feature metadata and VERIFIES
     the mapping against our competition train/dev documents (sentence-text
     match); aborts if agreement is low
  3. matches IL-TUR documents to competition test documents by sentence text
  4. writes external/iltur_test.jsonl (IL-TUR test-split documents, gold labels) for
     phase3_rr.py / phase3_lex.py --extra; never a prediction file.
"""

from __future__ import annotations

import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from phase3_data import P3, LABELS, load_all, log, norm, grams  # noqa: E402

REPO = "Exploration-Lab/IL-TUR"
SPLITS = [f"{d}_{s}" for d in ("CL", "IT") for s in ("train", "dev", "test")]


def label_names(path):
    import pyarrow.parquet as pq
    md = pq.read_schema(path).metadata or {}
    raw = md.get(b"huggingface")
    if raw:
        feats = json.loads(raw).get("info", {}).get("features", {})
        lab = feats.get("labels", {})
        inner = lab.get("feature", lab)
        if "names" in inner:
            return inner["names"]
    return None


ABBR_MAP = {  # labels as written in the authors' 2021 release (README.txt there)
    "fac": "Fact", "fact": "Fact", "facts": "Fact", "iss": "Issue", "issue": "Issue",
    "argp": "ArgumentPetitioner", "argumentpetitioner": "ArgumentPetitioner",
    "argr": "ArgumentRespondent", "argumentrespondent": "ArgumentRespondent",
    "sta": "Statute", "statute": "Statute", "dis": "Dissent", "dissent": "Dissent",
    "prer": "PrecedentReliedUpon", "precedentreliedupon": "PrecedentReliedUpon",
    "prenr": "PrecedentNotReliedUpon", "precedentnotreliedupon": "PrecedentNotReliedUpon",
    "preo": "PrecedentOverruled", "precedentoverruled": "PrecedentOverruled",
    "rlc": "RulingByLowerCourt", "rulingbylowercourt": "RulingByLowerCourt",
    "rod": "RatioOfTheDecision", "ratio": "RatioOfTheDecision", "ratioofthedecision": "RatioOfTheDecision",
    "rpc": "RulingByPresentCourt", "rulingbypresentcourt": "RulingByPresentCourt",
    "non": "None", "none": "None",
}


def norm_label(x):
    import re as _re
    k = _re.sub(r"[^a-z]", "", str(x).lower())
    if k in ABBR_MAP:
        return ABBR_MAP[k]
    return x


def load_local(d):
    """The 2021 release: <d>/CL/CL_train.json ... <d>/IT/IT_test.json (any nesting)."""
    import glob as _glob
    recs = []
    shapes = collections.Counter()
    for s in SPLITS:
        hits = _glob.glob(os.path.join(d, "**", f"{s}.json"), recursive=True)
        assert hits, f"{s}.json not found under {d}"
        raw = json.load(open(hits[0], encoding="utf-8"))
        items = raw.items() if isinstance(raw, dict) else enumerate(raw)
        n = 0
        for rid, r in items:
            text = r.get("sentences") or r.get("text")
            labs = r.get("complete") or r.get("labels")
            assert text is not None and labs is not None, f"{s}: unknown record keys {list(r)[:8]}"
            shape = type(text).__name__
            if isinstance(text, str):
                # one string per document: split on newlines (or the sentence
                # separator that yields exactly one sentence per label)
                for sep in ("\n", "\r\n", " ## ", "\t"):
                    parts = [t for t in text.split(sep) if t.strip()]
                    if len(parts) == len(labs):
                        break
                text = parts
            elif text and isinstance(text[0], list):
                text = [" ".join(map(str, t)) for t in text]      # tokenised sentences
            shapes[(s, shape)] += 1
            recs.append((s, str(r.get("id", rid)), [str(t) for t in text], list(labs)))
            n += 1
        log(f"  {s}: {n} documents from {hits[0]}")
    ex = recs[0]
    log(f"  example labels: {ex[3][:8]}")
    log("  sentence field shapes: " + ", ".join(f"{a}:{b} {n}" for (a, b), n in sorted(shapes.items())))
    mism = collections.Counter(r[0] for r in recs if len(r[2]) != len(r[3]))
    if mism:
        log(f"  documents whose sentence and label counts differ: {dict(mism)}")
    return recs


def key(sents):
    return tuple(norm(s) for s in sents[:3])


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-dir", default=None,
                    help="folder with the authors' 2021 release (CL/*.json, IT/*.json) instead of HF")
    args = ap.parse_args()
    docs, _ = load_all()
    comp = {}
    for d, r in docs.items():
        comp.setdefault(key(r["text"]), []).append(d)

    names = None
    recs = []
    files = {}
    if args.local_dir:
        recs = load_local(args.local_dir)
        names = LABELS
    else:
        import pandas as pd
        from huggingface_hub import hf_hub_download, list_repo_files
        files = {s: next(f for f in list_repo_files(REPO, repo_type="dataset")
                         if f.startswith(f"rr/{s}-") and f.endswith(".parquet")) for s in SPLITS}
    for s, f in files.items():
        p = hf_hub_download(REPO, f, repo_type="dataset")
        names = names or label_names(p)
        df = pd.read_parquet(p)
        if s == SPLITS[0]:
            log(f"  columns: {list(df.columns)}; label names from metadata: {names}")
        for _, row in df.iterrows():
            text = [str(x) for x in row["text"]]
            labs = list(row["labels"])
            recs.append((s, str(row.get("id", "")), text, labs))
        log(f"  {s}: {len(df)} documents")
    names = names or LABELS
    assert sorted(names) == sorted(LABELS), f"label set differs: {names}"

    def to_name(x):
        if isinstance(x, str) and not x.strip().isdigit():
            return norm_label(x)
        return names[int(x)]

    # verify the mapping on documents we already have gold labels for
    agree = tot = 0
    matched = collections.Counter()
    test_hits = []
    for s, rid, text, labs in recs:
        hit = comp.get(key(text), [])
        for d in hit:
            r = docs[d]
            matched[(s, r["split"])] += 1
            if r["labels"] and len(r["labels"]) == len(labs):
                agree += sum(to_name(a) == b for a, b in zip(labs, r["labels"]))
                tot += len(labs)
            if r["split"] == "test":
                test_hits.append((d, len(r["text"]), len(text)))
    log("  IL-TUR split -> competition split matches: " +
        ", ".join(f"{a}->{b} {n}" for (a, b), n in sorted(matched.items())))
    log(f"  label agreement on matched train/dev documents: {agree}/{tot} = {agree/max(1,tot):.4f}")
    assert tot and agree / tot > 0.98, "label mapping does not reproduce our gold labels -- stop"
    # Second pass for records the first-sentences key missed (e.g. a domain whose
    # release splits sentences differently): match by 13-gram overlap, then check
    # labels sentence by sentence wherever the normalised sentence text is equal.
    found = {id(r) for r in recs if comp.get(key(r[2]))}
    cg = {d: grams(" ".join(r["text"])) for d, r in docs.items()}
    s_agree = s_tot = 0
    pos_agree = pos_tot = 0
    realign = {}                 # id(record) -> record re-expressed on OUR sentences
    count_off = collections.Counter()
    fuzzy = collections.Counter()
    for r in recs:
        s, rid, text, labs = r
        if id(r) in found:
            continue
        g = grams(" ".join(text))
        if not g:
            continue
        best, frac = None, 0.0
        for d, gd in cg.items():
            if gd:
                f = len(g & gd) / min(len(g), len(gd))
                if f > frac:
                    best, frac = d, f
        if frac < 0.5:
            fuzzy[(s, "no match")] += 1
            continue
        fuzzy[(s, docs[best]["split"])] += 1
        # The release's labels are positional (one per sentence of ITS split).
        # When their count equals the number of sentences in our copy of the
        # same judgment, they are re-expressed on OUR sentences; checked
        # position by position on train/dev, where our gold labels exist.
        if len(labs) == len(docs[best]["text"]):
            realign[id(r)] = (s, rid, list(docs[best]["text"]), labs)
            if docs[best]["labels"]:
                pos_tot += len(labs)
                pos_agree += sum(to_name(a) == b for a, b in zip(labs, docs[best]["labels"]))
        else:
            count_off[(s, docs[best]["split"])] += 1
        if docs[best]["split"] == "test":
            test_hits.append((best, len(docs[best]["text"]), len(text)))
        if docs[best]["labels"]:
            gl = {norm(t): l for t, l in zip(docs[best]["text"], docs[best]["labels"])}
            for t, l in zip(text, labs):
                if norm(t) in gl:
                    s_tot += 1
                    s_agree += to_name(l) == gl[norm(t)]
    if fuzzy:
        log("  13-gram matches for the rest: " + ", ".join(f"{a}->{b} {n}" for (a, b), n in sorted(fuzzy.items())))
        log(f"  re-expressed on our sentences (label count == our sentence count): {len(realign)}; "
            f"count differs: {dict(count_off)}")
        log(f"  POSITIONAL label agreement on re-expressed train/dev documents: "
            f"{pos_agree}/{pos_tot} = {pos_agree/max(1,pos_tot):.4f}")
        assert pos_tot == 0 or pos_agree / pos_tot > 0.95, \
            "re-expressed labels disagree with our gold labels -- positional alignment is wrong, stop"
        log(f"  sentence-level label agreement on those: {s_agree}/{s_tot} = {s_agree/max(1,s_tot):.4f}")
        assert s_tot == 0 or s_agree / s_tot > 0.95, "labels in the fuzzily matched records disagree -- stop"
    # Records whose labels cannot be aligned (counts differ) are dropped rather
    # than padded: a shifted label sequence would teach the wrong role.
    recs = [realign.get(id(r), r) for r in recs
            if id(r) in found or id(r) in realign]
    bad = collections.Counter(to_name(x) for r in recs for x in r[3] if to_name(x) not in LABELS)
    assert not bad, f"labels outside the vocabulary after mapping: {dict(bad.most_common(8))} -- stop"
    n_test = len({d for d, _, _ in test_hits})
    same_len = sum(a == b for _, a, b in test_hits)
    log(f"  competition test documents found in IL-TUR: {n_test} of "
        f"{sum(1 for r in docs.values() if r['split'] == 'test')} (sentence counts equal for {same_len})")

    os.makedirs(f"{P3}/external", exist_ok=True)
    # Written: every record that does NOT duplicate one of our own labelled
    # (train/dev) documents. Duplicates are left out because adding them would
    # leak each CV fold's held-out documents into its training data. What remains
    # is (a) the test-split records that match our test documents (the use the
    # TA permitted) and (b) records matching nothing of ours -- in the 2021
    # release, all 50 CL documents are different judgments from the
    # competition's, i.e. plain extra labelled data.
    lab_g = {d: g for d, g in cg.items() if docs[d]["split"] != "test" and g}

    def dup_of_labelled(text):
        if any(docs[d]["split"] != "test" for d in comp.get(key(text), [])):
            return True
        g = grams(" ".join(text))
        return bool(g) and any(len(g & gd) / min(len(g), len(gd)) >= 0.5 for gd in lab_g.values())

    out = f"{P3}/external/iltur_test.jsonl"
    kinds = collections.Counter()
    with open(out + ".tmp", "w", encoding="utf-8") as fh:
        for s, rid, text, labs in recs:
            if dup_of_labelled(text):
                kinds["skipped: duplicates our train/dev"] += 1
                continue
            kinds[f"written: {s}"] += 1
            lab = [to_name(x) for x in labs]
            if len(lab) != len(text):
                # a few documents in the 2021 release have a label list that is not
                # the length of their sentence list: trim, or pad with None (no
                # supervision), so every sentence keeps its position
                kinds["label/sentence length fixed"] += 1
                lab = lab[:len(text)] + [None] * (len(text) - len(lab))
            fh.write(json.dumps(dict(id=f"{s}_{rid or len(text)}", domain=s[:2], text=text,
                                     labels=lab), ensure_ascii=False) + "\n")
    os.replace(out + ".tmp", out)
    log("  " + ", ".join(f"{k} {v}" for k, v in sorted(kinds.items())))
    log(f"  wrote {out}: {sum(v for k, v in kinds.items() if k.startswith('written'))} documents "
        f"(training data only; never written to a prediction file)")


if __name__ == "__main__":
    main()
