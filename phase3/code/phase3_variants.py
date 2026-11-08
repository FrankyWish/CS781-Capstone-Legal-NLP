#!/usr/bin/env python3
"""CS-781 Phase 3 -- ensemble variants from saved logits, for leaderboard selection. CPU, minutes.

    python phase3_variants.py              # writes submissions/submission_v01.zip ... and variants.tsv

Every finished run saved its test logits, so new ensembles cost no GPU. This
writes ~20 variants: 4 member sets x 5 global decoding settings (prior
correction tau, transition weight beta). Only GLOBAL choices are varied --
never per-label settings, which would amount to fitting the test labels through
the leaderboard. Choosing among these by leaderboard score is selection on the
test set itself; the report must say so and quote the CV figures alongside.
"""

from __future__ import annotations

import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from phase3_data import P3, LABELS, load_all, log, prf, write_submission, ensure_split  # noqa: E402
from phase3_ens import load_run, run_logprobs, stats, decode_one  # noqa: E402

SETS = {
    # members of ens_final_hyb2 (0.5458, the best so far)
    "hyb2": ["ext_M4", "kd_M4half", "lorma_M3", "lora_M4", "lex_tfidf"],
    # + rsLoRA M1 (0.5193 alone)
    "hyb2+M1": ["ext_M4", "kd_M4half", "lorma_M3", "lora_M4", "lex_tfidf", "rslora_M1"],
    # the four strongest single models on test, model-only
    "top4": ["lora_M4", "rslora_M1", "kd_M4half", "lorma_M3"],
    # every run that scored >= 0.50 alone on test, plus the lexical model
    "wide": ["lora_M4", "rslora_M1", "kd_M4half", "kd2_M4half", "lorma_M3", "ext_M4", "lex_tfidf"],
}
DECODE = [(0.5, 0.25), (0.25, 0.25), (0.75, 0.25), (0.5, 0.5), (0.5, 0.0)]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="*", default=None, help="names from SETS (default: all)")
    ap.add_argument("--add-set", action="append", default=[],
                    help="extra set as name=run1,run2,... (e.g. lseg=lora_M4_lseg,lora_M4,...)")
    ap.add_argument("--taus", nargs="*", type=float, default=None)
    ap.add_argument("--betas", nargs="*", type=float, default=None)
    ap.add_argument("--start", type=int, default=1, help="first variant number (v<start>)")
    a = ap.parse_args()
    for spec in a.add_set:
        name, runs = spec.split("=", 1)
        SETS[name] = runs.split(",")
    sets = {k: SETS[k] for k in (a.sets if a.sets else SETS)}
    decode = ([(t, b) for t in a.taus for b in a.betas] if a.taus is not None and a.betas is not None
              else DECODE)
    docs, _ = load_all()
    split = ensure_split(docs)
    folds = split["folds"]
    fold_of = {d: k for k, f in enumerate(folds) for d in f}
    labelled = sorted(fold_of)
    test_ids = sorted(d for d, r in docs.items() if r["split"] == "test")
    st_fold = {k: stats(docs, [d for d in labelled if fold_of[d] != k]) for k in range(len(folds))}
    st_all = stats(docs, labelled)

    cache = {}
    rows = []
    n = a.start - 1
    for name, members in sets.items():
        runs = []
        for t in members:
            if t not in cache:
                fr = load_run(t)
                cache[t] = run_logprobs(fr) if len(fr) == len(folds) else None
            if cache[t] is None:
                log(f"  {name}: skipping missing run {t}")
                continue
            runs.append(cache[t])
        if len(runs) < 2:
            continue
        oof = {d: sum(r[0][d] for r in runs) / len(runs) for d in labelled}
        test = {d: sum(r[1][d] for r in runs) / len(runs) for d in test_ids}
        for tau, beta in decode:
            n += 1
            tag = f"v{n:02d}"
            po = {d: decode_one(oof[d], tau, beta, st_fold[fold_of[d]]) for d in labelled}
            cv = prf([l for d in labelled for l in docs[d]["labels"]],
                     [l for d in labelled for l in po[d]], LABELS)["macro_f1"]
            pt = {d: decode_one(test[d], tau, beta, st_all) for d in test_ids}
            write_submission(pt, {d: docs[d] for d in test_ids}, LABELS, tag, cv=cv,
                             note=f"variant {name} tau={tau} beta={beta}")
            rows.append((tag, name, tau, beta, cv))
            log(f"  {tag}  {name:8s} tau={tau:<4} beta={beta:<4} CV {cv:.4f}")
    with open(f"{P3}/submissions/variants.tsv", "w") as f:
        f.write("tag\tset\ttau\tbeta\tcv\tmembers\n")
        for tag, name, tau, beta, cv in rows:
            f.write(f"{tag}\t{name}\t{tau}\t{beta}\t{cv:.4f}\t{' '.join(SETS[name])}\n")
    log(f"  wrote {len(rows)} variants; table in {P3}/submissions/variants.tsv")


if __name__ == "__main__":
    main()
