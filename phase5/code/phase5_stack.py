#!/usr/bin/env python3
"""CS-781 Phase 5 -- second-stage stacker over frozen run outputs. CPU only, minutes.

    python phase5_stack.py --tag stack --submit
    python phase5_stack.py --tag stack_nocooc --no-cooc --rounds 1 --submit
    python phase5_stack.py --tag stack_bal --balance 0.5 --submit

Every member run was trained on train only, so its dev and test scores are both
out-of-sample and calibrated alike; a model fitted on dev scores and applied to test
scores sees no calibration shift (the Phase-3 failure was CV-fold models vs
full-data models). Runs whose test scores come from a train+dev refit are left out
for that reason, and so are ensembles (their weights were chosen on dev).

Rows = (document, candidate statute) for the top-K statutes of the mean z-scored
member logits. A gradient-boosted classifier (sklearn HistGB, statute id as a
categorical feature) predicts whether the statute applies. Features:
  per member   z-scored logit, within-document rank
  agreement    mean / std / max of member z, members ranking it top-3
  statute      log train frequency, id
  co-occur.    sum_k P(j|k) p(k) / sum_k p(k) and max_k P(j|k) p(k) over the document's
               other statutes; p is the member mean (round 1) or the previous round's
               stacker (round 2 on: a learned classifier chain)
  document     expected statute count, top score, log length
K-fold over dev documents gives out-of-fold dev scores; thresholds and the CV
number are tuned on those exactly as for any run. Test scores average the K fold
models. Non-candidates get a logit of -12 and are never predicted.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from phase5_data import P5, load, label_space, multi_hot, tune_global, log  # noqa: E402
from phase5_ens import aligned, finish  # noqa: E402

NEG = -12.0


def members(spec):
    if spec:
        return [t for t in spec.split(",") if t]
    out = []
    for f in sorted(glob.glob(f"{P5}/results/*/meta.json")):
        m = json.load(open(f))
        if m.get("kind") not in ("neural", "lex"):
            continue
        if (m.get("args") or {}).get("refit_dev"):
            continue
        out.append(m["tag"])
    return out


def zscore(D, T):
    R = np.concatenate([D, T])
    mu, sd = R.mean(0, keepdims=True), R.std(0, keepdims=True) + 1e-6
    return (D - mu) / sd, (T - mu) / sd


def rank_in_doc(Z):
    return np.argsort(np.argsort(-Z, 1), 1).astype(np.float32)


def cooc(p, C, chunk=1000):
    """p (N, n) probabilities, C[j, k] = P(j | k) with zero diagonal -> (mean, max) co-occurrence support."""
    mean = (p @ C.T) / (p.sum(1, keepdims=True) + 1e-6)
    mx = np.zeros_like(p)
    for s in range(0, len(p), chunk):
        mx[s:s + chunk] = (p[s:s + chunk, None, :] * C[None]).max(2)
    return mean.astype(np.float32), mx.astype(np.float32)


def build(split_feats, cand, p, C, use_cooc):
    """Stack features for candidate rows (doc i, statute j) of one split."""
    Z, Rk, extra = split_feats
    ii, jj = cand
    cols = [Z[:, ii, jj].T, Rk[:, ii, jj].T]                      # (rows, members) each
    zz = Z[:, ii, jj]
    cols.append(np.stack([zz.mean(0), zz.std(0), zz.max(0), (Rk[:, ii, jj] < 3).sum(0)], 1))
    doc = np.stack([p.sum(1), p.max(1), extra["loglen"]], 1)[ii]
    cols.append(doc)
    cols.append(np.stack([extra["logfreq"][jj], p[ii, jj]], 1))
    if use_cooc:
        cm, cx = cooc(p, C)
        cols.append(np.stack([cm[ii, jj], cx[ii, jj]], 1))
    X = np.concatenate([c.astype(np.float32) for c in cols], 1)
    return np.concatenate([X, jj[:, None].astype(np.float32)], 1)   # statute id last (categorical)


def fit(X, y, w, seed, a):
    from sklearn.ensemble import HistGradientBoostingClassifier
    cat = np.zeros(X.shape[1], dtype=bool)
    cat[-1] = True
    m = HistGradientBoostingClassifier(max_iter=a.iters, learning_rate=a.lr, max_leaf_nodes=31,
                                       min_samples_leaf=40, l2_regularization=1.0, categorical_features=cat,
                                       early_stopping=True, validation_fraction=0.1, n_iter_no_change=30,
                                       random_state=seed)
    m.fit(X, y, sample_weight=w)
    return m


def to_full(rows_score, cand, N, n):
    S = np.full((N, n), NEG, dtype=np.float32)
    S[cand[0], cand[1]] = rows_score
    return S


def sig(x):
    return 1 / (1 + np.exp(-x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="stack")
    ap.add_argument("--runs", default="", help="comma list; default = every neural/lex run except train+dev refits")
    ap.add_argument("--k", type=int, default=25, help="candidate statutes per document")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--no-cooc", action="store_true")
    ap.add_argument("--balance", type=float, default=0.0, help="weight rows of statute j by (median/count_j)^b")
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--lr", type=float, default=0.06)
    ap.add_argument("--seed", type=int, default=781)
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--strategy", default="auto")
    ap.add_argument("--max-label", type=int, default=None)
    a = ap.parse_args()
    a.combine = "stack"
    t0 = time.time()

    tags = members(a.runs)
    log(f"[stack {a.tag}] {len(tags)} members: {','.join(tags)}")
    runs, metas = aligned(tags)
    tr, dv, te = load("train"), load("dev"), load("test")
    n, cnt = label_space()
    Ytr, Ydv = multi_hot(tr, n), multi_hot(dv, n)
    Nd, Nt = len(dv), len(te)

    Zd, Zt, Rd, Rt, Pd, Pt = [], [], [], [], [], []
    for t in tags:
        zd, zt = zscore(runs[t]["dev"], runs[t]["test"])
        Zd.append(zd); Zt.append(zt)
        Rd.append(rank_in_doc(zd)); Rt.append(rank_in_doc(zt))
        Pd.append(sig(runs[t]["dev"])); Pt.append(sig(runs[t]["test"]))
    Zd, Zt, Rd, Rt = map(np.stack, (Zd, Zt, Rd, Rt))
    p0d, p0t = np.mean(Pd, 0), np.mean(Pt, 0)
    del Pd, Pt

    co = (Ytr.T.astype(np.float32) @ Ytr.astype(np.float32))
    C = co / np.maximum(np.diag(co)[None, :], 1)
    np.fill_diagonal(C, 0)
    logfreq = np.log1p(cnt).astype(np.float32)
    ex_d = dict(loglen=np.log1p([len(r["text"]) for r in dv]).astype(np.float32), logfreq=logfreq)
    ex_t = dict(loglen=np.log1p([len(r["text"]) for r in te]).astype(np.float32), logfreq=logfreq)

    def candidates(Z):
        top = np.argsort(-Z.mean(0), 1)[:, :a.k]
        return np.repeat(np.arange(len(top)), top.shape[1]), top.reshape(-1)

    cd, ct = candidates(Zd), candidates(Zt)
    rec = Ydv[cd[0], cd[1]].sum() / max(Ydv.sum(), 1)
    log(f"  candidates: top-{a.k} per doc -> {len(cd[0])} dev rows, {len(ct[0])} test rows; "
        f"gold recall on dev {rec:.4f} (ceiling on micro recall)")
    y = Ydv[cd[0], cd[1]].astype(np.uint8)
    w = None
    if a.balance:
        wl = (np.median(cnt[cnt > 0]) / np.maximum(cnt, 1)) ** a.balance
        w = wl[cd[1]].astype(np.float32)

    rng = np.random.RandomState(a.seed)
    fold = rng.permutation(Nd) % a.folds
    row_fold = fold[cd[0]]

    pd_, pt_ = p0d, p0t
    best = None
    for r in range(a.rounds):
        Xd = build((Zd, Rd, ex_d), cd, pd_, C, not a.no_cooc)
        Xt = build((Zt, Rt, ex_t), ct, pt_, C, not a.no_cooc)
        oof = np.zeros(len(y), dtype=np.float32)
        tst = np.zeros(len(ct[0]), dtype=np.float32)
        for f in range(a.folds):
            tr_m, va_m = row_fold != f, row_fold == f
            m = fit(Xd[tr_m], y[tr_m], None if w is None else w[tr_m], a.seed + f, a)
            oof[va_m] = m.decision_function(Xd[va_m])
            tst += m.decision_function(Xt) / a.folds
            log(f"  round {r+1} fold {f+1}/{a.folds}: {m.n_iter_} trees ({(time.time()-t0)/60:.1f} min)")
        Sd, St = to_full(oof, cd, Nd, n), to_full(tst, ct, Nt, n)
        m1 = tune_global(Sd, Ydv)[1]
        log(f"  round {r+1}: OOF dev Macro-F1 (global thr, tuned on OOF) {m1:.4f}; "
            f"features {Xd.shape[1]}; test mean logit {tst.mean():.3f} vs OOF {oof.mean():.3f}")
        if best is None or m1 > best[0] + 1e-4:
            best = (m1, r + 1, Sd, St)
        pd_, pt_ = sig(Sd), sig(St)
        if a.no_cooc:
            break                        # without co-occurrence features the chain has no input
    m1, rbest, Sd, St = best
    log(f"  using round {rbest} ({m1:.4f})")
    finish(a.tag, dict(dev=Sd, test=St), Ydv, a, True,
           f"stack(HistGB, {len(tags)} members, K={a.k}, rounds={rbest}, cooc={not a.no_cooc}, bal={a.balance})",
           a.strategy, {t: 1 for t in tags})
    log(f"[stack {a.tag}] done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
