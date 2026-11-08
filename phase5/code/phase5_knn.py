#!/usr/bin/env python3
"""CS-781 Phase 5 -- near-duplicate cases: k-nearest-neighbour statute voting over train. CPU.

    python phase5_knn.py --tag knn_w12              # report + results/knn_w12 (an ensemble member)

High Court bail / quashing orders are formulaic, and co-accused of one FIR file
separate applications with near-identical facts. Where a dev/test document has a
near-copy in train, the copy's statutes are very likely its statutes. This is an
ordinary kNN classifier over the training data (no test or external labels).

  features  TF-IDF word 1-2 grams (as phase5_lex), L2-normalised: dot = cosine.
  score     s_l(d) = sum_{j in top-k} cos(d, j)^p * y_jl / sum_j cos(d, j)^p
  report    how many dev documents have a train neighbour at cosine >= .95/.9/.8/.7,
            and how often that neighbour's label SET is exactly the dev label set:
            the size of the near-duplicate lever, measured before using it.

Writes results/<tag>/{dev,test}_logits.npy (log-odds of s, clipped) + meta.json
(kind "lex", hybrid: TF-IDF features, not a Phase-1 model).
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
from phase5_data import (P5, load, label_space, multi_hot, score_on, decode, threshold_report, log)  # noqa: E402


def topk_sims(Xq, Xb, k, bs=1000):
    """(idx (N,k), sim (N,k)) of the k most similar rows of Xb for each row of Xq."""
    idx = np.zeros((Xq.shape[0], k), dtype=np.int32)
    sim = np.zeros((Xq.shape[0], k), dtype=np.float32)
    XbT = Xb.T.tocsr()
    for s in range(0, Xq.shape[0], bs):
        S = (Xq[s:s + bs] @ XbT).toarray()
        part = np.argpartition(-S, k, axis=1)[:, :k]
        ps = np.take_along_axis(S, part, 1)
        o = np.argsort(-ps, 1)
        idx[s:s + bs] = np.take_along_axis(part, o, 1)
        sim[s:s + bs] = np.take_along_axis(ps, o, 1)
    return idx, sim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="knn_w12")
    ap.add_argument("--k", type=int, default=25)
    ap.add_argument("--p", type=float, default=4.0, help="similarity sharpening exponent")
    ap.add_argument("--max-feat", type=int, default=400_000)
    a = ap.parse_args()
    from sklearn.feature_extraction.text import TfidfVectorizer
    tr, dv, te = load("train"), load("dev"), load("test")
    n, cnt = label_space()
    Ytr, Ydv = multi_hot(tr, n), multi_hot(dv, n)
    t0 = time.time()
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9, max_features=a.max_feat, sublinear_tf=True,
                          dtype=np.float32, token_pattern=r"(?u)\b\w[\w/.-]*\w\b|\b\w\b", strip_accents="unicode")
    Xtr = vec.fit_transform([r["text"] for r in tr])
    Xdv = vec.transform([r["text"] for r in dv])
    Xte = vec.transform([r["text"] for r in te])
    log(f"  tf-idf {Xtr.shape[1]} features ({time.time()-t0:.0f}s)")
    idv, sdv = topk_sims(Xdv, Xtr, a.k)
    ite, ste = topk_sims(Xte, Xtr, a.k)
    log(f"  neighbours found ({time.time()-t0:.0f}s)")

    # ---- the size of the near-duplicate lever
    top = sdv[:, 0]
    same = np.array([(Ytr[idv[i, 0]] == Ydv[i]).all() for i in range(len(dv))])
    rows = []
    for c in (0.98, 0.95, 0.9, 0.8, 0.7, 0.5):
        m = top >= c
        rows.append(f"cos>={c}: dev {m.mean():.3f} (test {(ste[:, 0] >= c).mean():.3f}), "
                    f"neighbour label set identical {same[m].mean() if m.any() else float('nan'):.3f}")
    log("  NEAR-DUPLICATES (top-1 train neighbour):\n    " + "\n    ".join(rows))

    def scores(idx, sim):
        w = np.clip(sim, 0, 1) ** a.p
        S = np.einsum("nk,nkl->nl", w, Ytr[idx].astype(np.float32)) / np.maximum(w.sum(1, keepdims=True), 1e-9)
        S = np.clip(S, 1e-4, 1 - 1e-4)
        return np.log(S / (1 - S)).astype(np.float32)

    Sdv, Ste = scores(idv, sdv), scores(ite, ste)
    th = threshold_report(Sdv, Ydv)
    key = th["best"] if th["cv"][th["best"]] > th["cv"]["global"] + 0.002 else "global"
    dev = score_on(Ydv, decode(Sdv, th["all_thr"][key]))
    log(f"  KNN dev ({key}): {json.dumps(dev)} | CV {th['cv'][key]:.4f}")
    # top-1 copy only, for documents with a very close neighbour
    rd = f"{P5}/results/{a.tag}"
    os.makedirs(rd, exist_ok=True)
    np.save(f"{rd}/dev_logits.npy", Sdv.astype(np.float16))
    np.save(f"{rd}/test_logits.npy", Ste.astype(np.float16))
    json.dump([r["id"] for r in dv], open(f"{rd}/dev_ids.json", "w"))
    json.dump([r["id"] for r in te], open(f"{rd}/test_ids.json", "w"))
    json.dump(dict(tag=a.tag, kind="lex", hybrid=True, args=vars(a), dev_global=score_on(Ydv, decode(Sdv, th["t_global"])),
                   dev_best=dev, thr_cv=th["cv"], thr_best=key, near_dup=rows), open(f"{rd}/meta.json", "w"), indent=1)
    log(f"  wrote {rd} ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
