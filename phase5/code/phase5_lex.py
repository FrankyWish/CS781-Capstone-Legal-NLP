#!/usr/bin/env python3
"""CS-781 Phase 5 -- lexical statute model (TF-IDF + one-vs-rest logistic regression). CPU only.

    python phase5_lex.py --tag lex_w12                       # word 1-2 grams
    python phase5_lex.py --tag lex_w12c --char               # + char_wb 3-5 grams
    python phase5_lex.py --tag lex_w12 --refit-dev           # test model also sees dev labels
    python phase5_lex.py --tag lex_w12x --crossfit 4         # dev labels too, dev scores out-of-fold

Statute identification on case facts is driven by surface vocabulary --
"dowry", "kerosene", "demanded", "cheque dishonoured", "forged" -- which a
sparse linear model reads directly. It costs minutes of CPU, and its errors are
unlike a neural trunk's, which is what an ensemble member is for (Phase 3: +0.03).

One binary logistic regression per statute (binary relevance), trained on
train.json. Scores for dev and test are saved as logits; all thresholding and
ensembling happens later in phase5_ens.py. With --refit-dev the TEST scores come
from a second model fit on train+dev, while dev scores (for thresholds) still come
from the train-only model; dev Macro-F1 then slightly understates the test model.
With --crossfit K every dev document is scored by a model fit on train plus the
other K-1 dev folds, and test by a model fit on train+dev: both splits are
out-of-sample and near-identically calibrated, so the run is a valid stacker member
(a --refit-dev run is not: its dev and test scores come from different models).

RULES NOTE. No pretrained weights, but not a Phase-1 model either: any submission
that includes this model is a HYBRID and is reported beside the model-only number.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from phase5_data import (P5, load, label_space, multi_hot, score_on, decode,  # noqa: E402
                         threshold_report, log)


def featurise(a, fit_texts, *others):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from scipy.sparse import hstack
    vw = TfidfVectorizer(ngram_range=(1, a.ngram), min_df=a.min_df, max_df=0.9, max_features=a.max_feat,
                         sublinear_tf=True, dtype=np.float32, token_pattern=r"(?u)\b\w[\w/.-]*\w\b|\b\w\b",
                         lowercase=True, strip_accents="unicode")
    t0 = time.time()
    Xs = [vw.fit_transform(fit_texts)] + [vw.transform(t) for t in others]
    log(f"  word tf-idf: {Xs[0].shape[1]} features ({time.time()-t0:.0f}s)")
    if a.char:
        vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5, max_features=a.char_feat,
                             sublinear_tf=True, dtype=np.float32)
        t0 = time.time()
        Cs = [vc.fit_transform([t[: a.char_chars] for t in fit_texts])] + \
             [vc.transform([t[: a.char_chars] for t in o]) for o in others]
        log(f"  char tf-idf: {Cs[0].shape[1]} features ({time.time()-t0:.0f}s)")
        Xs = [hstack([x, c]).tocsr() for x, c in zip(Xs, Cs)]
    return Xs


def fit_ovr(a, X, Y, Xouts, M=None, W=None):
    """One LR per label, in parallel. Returns [logits (N_out, n) for each X in Xouts].

    M (N, n) {0,1}: 1 where the label is supervised for that row (silver rows mask
    labels their labeller cannot vouch for). W (N,): per-row sample weight.
    """
    from joblib import Parallel, delayed
    from sklearn.linear_model import LogisticRegression

    def one(l):
        rows = np.nonzero(M[:, l])[0] if M is not None else None
        Xl = X[rows] if rows is not None else X
        y = Y[rows, l] if rows is not None else Y[:, l]
        sw = (W[rows] if rows is not None else W) if W is not None else None
        if y.sum() == 0:
            return [np.full(x.shape[0], -20.0, dtype=np.float32) for x in Xouts]
        cw = None
        if a.balance > 0:
            pos = y.mean()
            cw = {0: 1.0, 1: float(min(a.balance_cap, ((1 - pos) / max(pos, 1e-6)) ** a.balance))}
        m = LogisticRegression(C=a.C, solver="liblinear", class_weight=cw, max_iter=200)
        m.fit(Xl, y, sample_weight=sw)
        return [m.decision_function(x).astype(np.float32) for x in Xouts]

    t0 = time.time()
    outs = Parallel(n_jobs=a.jobs, prefer="processes", verbose=0)(delayed(one)(l) for l in range(Y.shape[1]))
    log(f"  fitted {Y.shape[1]} one-vs-rest LRs in {time.time()-t0:.0f}s")
    return [np.stack([o[k] for o in outs], 1) for k in range(len(Xouts))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ngram", type=int, default=2)
    ap.add_argument("--min-df", type=int, default=3)
    ap.add_argument("--max-feat", type=int, default=400_000)
    ap.add_argument("--char", action="store_true")
    ap.add_argument("--char-feat", type=int, default=300_000)
    ap.add_argument("--char-chars", type=int, default=20_000, help="chars per doc for char n-grams")
    ap.add_argument("--C", type=float, default=4.0)
    ap.add_argument("--balance", type=float, default=0.5,
                    help="positive class weight = (neg/pos)^balance; 0 = unweighted")
    ap.add_argument("--balance-cap", type=float, default=30.0)
    ap.add_argument("--refit-dev", action="store_true")
    ap.add_argument("--crossfit", type=int, default=0, help="K folds over dev; fit on train + other folds")
    ap.add_argument("--seed", type=int, default=781)
    ap.add_argument("--silver", type=int, default=0, help="add up to N silver docs (rarest labels first)")
    ap.add_argument("--silver-weight", type=float, default=0.5)
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--limit", type=int, default=0, help="debug: first N train docs only")
    a = ap.parse_args()

    rd = f"{P5}/results/{a.tag}"
    os.makedirs(rd, exist_ok=True)
    tr, dv, te = load("train"), load("dev"), load("test")
    if a.limit:
        tr = tr[: a.limit]
    n, _ = label_space()
    Ytr, Ydv = multi_hot(tr, n), multi_hot(dv, n)
    log(f"===== {a.tag}: {len(tr)} train / {len(dv)} dev / {len(te)} test, {n} labels =====")
    log("  args: " + json.dumps(vars(a), sort_keys=True))
    t0 = time.time()
    M = W = None
    fit_recs, Yfit = tr, Ytr
    if a.silver:
        from phase5_data import label_space as _ls
        _, cnt = _ls()
        sil = load("silver")
        sil = sorted(sil, key=lambda r: min(cnt[l] for l in r["labels"]))[: a.silver]
        Ys = multi_hot(sil, n)
        Ms = np.ones_like(Ys)
        for i, r in enumerate(sil):
            Ms[i, r["mask"]] = 0
        fit_recs = tr + sil
        Yfit = np.vstack([Ytr, Ys])
        M = np.vstack([np.ones_like(Ytr), Ms])
        W = np.concatenate([np.ones(len(tr)), np.full(len(sil), a.silver_weight)]).astype(np.float64)
        log(f"  + {len(sil)} silver docs (weight {a.silver_weight}); silver positives per label, rarest 15: "
            + " ".join(f"{l}:{cnt[l]}+{int(Ys[:, l].sum())}" for l in np.argsort(cnt)[:15]))
    if a.crossfit:
        assert not (a.silver or a.refit_dev), "--crossfit stands alone"
        Xall, Xte = featurise(a, [r["text"] for r in tr + dv], [r["text"] for r in te])
        Yall, Ntr = np.vstack([Ytr, Ydv]), len(tr)
        fold = np.random.RandomState(a.seed).permutation(len(dv)) % a.crossfit
        Sdv = np.zeros((len(dv), n), dtype=np.float32)
        for f in range(a.crossfit):
            keep = np.concatenate([np.arange(Ntr), Ntr + np.nonzero(fold != f)[0]])
            (Sdv[fold == f],) = fit_ovr(a, Xall[keep], Yall[keep], [Xall[Ntr + np.nonzero(fold == f)[0]]])
            log(f"  crossfit fold {f+1}/{a.crossfit} done ({(time.time()-t0)/60:.1f} min)")
        (Ste,) = fit_ovr(a, Xall, Yall, [Xte])
    else:
        Xtr, Xdv, Xte = featurise(a, [r["text"] for r in fit_recs], [r["text"] for r in dv], [r["text"] for r in te])
        Sdv, Ste = fit_ovr(a, Xtr, Yfit, [Xdv, Xte], M, W)
    if a.refit_dev:
        from scipy.sparse import vstack
        log("  refitting on train+dev for the test scores")
        Xall, Xte2 = featurise(a, [r["text"] for r in tr + dv], [r["text"] for r in te])
        assert not a.silver, "--refit-dev with --silver not supported"
        (Ste,) = fit_ovr(a, Xall, np.vstack([Ytr, Ydv]), [Xte2])
    th = threshold_report(Sdv, Ydv)
    dev_glob = score_on(Ydv, decode(Sdv, th["t_global"]))
    dev_best = score_on(Ydv, decode(Sdv, th["thr"]))
    log(f"  dev (thresholds fit on all dev -- optimistic): global {json.dumps(dev_glob)}")
    log(f"  dev best strategy {th['best']}: {json.dumps(dev_best)} | CV {th['cv'][th['best']]:.4f}")
    np.save(f"{rd}/dev_logits.npy", Sdv.astype(np.float16))
    np.save(f"{rd}/test_logits.npy", Ste.astype(np.float16))
    json.dump([r["id"] for r in dv], open(f"{rd}/dev_ids.json", "w"))
    json.dump([r["id"] for r in te], open(f"{rd}/test_ids.json", "w"))
    meta = dict(tag=a.tag, kind="lex", hybrid=True, args=vars(a), n_labels=n,
                dev_global=dev_glob, dev_best=dev_best, thr_cv=th["cv"], thr_best=th["best"],
                wall_s=time.time() - t0)
    json.dump(meta, open(f"{rd}/meta.json", "w"), indent=1)
    log(f"  wrote {rd}  ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
