#!/usr/bin/env python3
"""CS-781 Phase 5 -- do the FACTS still name the sections? Citation rule, measured. CPU, minutes.

    python phase5_rule.py                    # report + results/rule/ (+ zips if it helps)
    python phase5_rule.py --base lex_w12     # which run the rule is added to (default: best by dev CV)

If the facts text still says "convicted under Section 302 read with 34 IPC", a
rule maps that mention to the statute with near-certainty, and rare statutes --
which decide Macro-F1 -- are where a model is weakest and a rule is not.

  keys     every section mention in the text, tagged with the act named after it
           (phase5_external.cite_keys: '302|IPC', '438|CrPC', '34|-' ...).
  map      learned on TRAIN only: key -> label it co-occurs with, kept if its
           precision on train >= --prec (and >= --min-n occurrences).
  report   on DEV: share of documents with any mapped mention, the rule alone,
           per-statute precision/recall, and the rule OR-ed into the base model's
           thresholded predictions (rule positives are added; nothing is removed).

Writes results/rule/{dev,test}_rule.npy (0/1 matrices) and, with --submit, zips
`<base>_rule` built exactly like the base model's own zip plus the rule positives.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from phase5_data import (P5, load, label_space, multi_hot, score_on, decode, f1_per_label,  # noqa: E402
                         threshold_report, write_submission, log)
from phase5_external import cite_keys  # noqa: E402


def best_run():
    best = None
    for f in glob.glob(f"{P5}/results/*/meta.json"):
        m = json.load(open(f))
        if not os.path.exists(f"{P5}/results/{m['tag']}/dev_logits.npy") or m["tag"] == "rule":
            continue
        cv = max((m.get("thr_cv") or {"x": 0}).values())
        if best is None or cv > best[0]:
            best = (cv, m["tag"])
    return best[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prec", type=float, default=0.7)
    ap.add_argument("--min-n", type=int, default=5)
    ap.add_argument("--base", default=None)
    ap.add_argument("--submit", action="store_true")
    a = ap.parse_args()
    tr, dv, te = load("train"), load("dev"), load("test")
    n, cnt = label_space()
    Ytr, Ydv = multi_hot(tr, n), multi_hot(dv, n)
    K = {s: [cite_keys(r["text"]) for r in recs] for s, recs in (("train", tr), ("dev", dv), ("test", te))}
    for s in K:
        log(f"  {s}: {np.mean([bool(k) for k in K[s]]):.3f} of documents mention a section number; "
            f"{np.mean([any(x.endswith(('|IPC', '|CrPC')) for x in k) for k in K[s]]):.3f} with an act named")
    allk = collections.Counter(x for k in K["train"] for x in k)
    log("  most frequent mention keys in train: " + ", ".join(f"{k}:{c}" for k, c in allk.most_common(25)))

    # key -> label on train
    co = collections.defaultdict(collections.Counter)
    seen = collections.Counter()
    for keys, y in zip(K["train"], Ytr):
        labs = np.nonzero(y)[0]
        for k in keys:
            seen[k] += 1
            for l in labs:
                co[k][int(l)] += 1
    kmap = {}
    for k, c in co.items():
        l, m = c.most_common(1)[0]
        if seen[k] >= a.min_n and m / seen[k] >= a.prec:
            kmap[k] = (l, m / seen[k], seen[k])
    log(f"  rule map: {len(kmap)} mention keys with train precision >= {a.prec} "
        f"(covering {len({v[0] for v in kmap.values()})} statutes)")

    def apply(keys_list):
        P = np.zeros((len(keys_list), n), dtype=np.uint8)
        for i, keys in enumerate(keys_list):
            for k in keys:
                if k in kmap:
                    P[i, kmap[k][0]] = 1
        return P

    Rdv, Rte = apply(K["dev"]), apply(K["test"])
    log(f"  dev documents the rule fires on: {Rdv.any(1).mean():.3f}; test: {Rte.any(1).mean():.3f}")
    s_rule = score_on(Ydv, Rdv)
    tp = (Ydv & Rdv).sum()
    log(f"  RULE ALONE on dev: {json.dumps(s_rule)} | precision {tp / max(1, Rdv.sum()):.3f} "
        f"recall {tp / max(1, Ydv.sum()):.3f}")
    f1r = f1_per_label(Ydv, Rdv)
    good = [(int(l), round(float(f1r[l]), 3), int(Rdv[:, l].sum()), int(Ydv[:, l].sum())) for l in np.argsort(-f1r)[:15]]
    log("  best statutes for the rule (id, F1, fired, gold): " + " ".join(map(str, good)))

    base = a.base or best_run()
    m = json.load(open(f"{P5}/results/{base}/meta.json"))
    ids_dv = json.load(open(f"{P5}/results/{base}/dev_ids.json"))
    ids_te = json.load(open(f"{P5}/results/{base}/test_ids.json"))
    pos_dv = {d: i for i, d in enumerate(ids_dv)}
    pos_te = {d: i for i, d in enumerate(ids_te)}
    Sdv = np.load(f"{P5}/results/{base}/dev_logits.npy").astype(np.float32)[[pos_dv[r["id"]] for r in dv]]
    Ste = np.load(f"{P5}/results/{base}/test_logits.npy").astype(np.float32)[[pos_te[r["id"]] for r in te]]
    th = threshold_report(Sdv, Ydv, log_fn=lambda *x: None)
    key = th["best"] if th["cv"][th["best"]] > th["cv"]["global"] + 0.002 else "global"
    thr = th["all_thr"][key]
    Pdv, Pte = decode(Sdv, thr), decode(Ste, thr)
    s_base = score_on(Ydv, Pdv)
    s_or = score_on(Ydv, Pdv | Rdv)
    log(f"  BASE {base} ({key}) on dev: {json.dumps(s_base)}")
    log(f"  BASE OR RULE on dev:          {json.dumps(s_or)}   (gain {s_or['macro_f1'] - s_base['macro_f1']:+.4f})")
    rd = f"{P5}/results/rule"
    os.makedirs(rd, exist_ok=True)
    np.save(f"{rd}/dev_rule.npy", Rdv)
    np.save(f"{rd}/test_rule.npy", Rte)
    json.dump(dict(kmap={k: list(v) for k, v in kmap.items()}, rule_dev=s_rule, base=base, base_dev=s_base,
                   base_or_rule_dev=s_or, dev_fire=float(Rdv.any(1).mean()), test_fire=float(Rte.any(1).mean())),
              open(f"{rd}/report.json", "w"), indent=1)
    if a.submit and s_or["macro_f1"] > s_base["macro_f1"] + 0.002:
        write_submission(f"{base}_rule", te, Pte | Rte, note=f"{base} + citation rule (train-learned map)",
                         hybrid=True, dev_scores=dict(s_or, cv_macro_f1=s_or["macro_f1"]))


if __name__ == "__main__":
    main()
