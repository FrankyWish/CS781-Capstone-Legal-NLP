#!/usr/bin/env python3
"""CS-781 Phase 3 -- lexical baseline / hybrid member. CPU only, minutes.

    python phase3_lex.py                      # gold data only
    python phase3_lex.py --extra --tag lex_ext

Rhetorical roles carry strong surface cues: "learned counsel for the appellant
submitted", "Section 45 of the Act provides", citations ("v.", "SCC", "AIR"),
"the appeal is dismissed". A TF-IDF n-gram model sees those directly, needs no
GPU, and makes errors unlike a neural trunk's -- which is what an ensemble
member is for.

Features per sentence: word 1-2-gram and char 3-5-gram TF-IDF of the sentence,
the same for its previous and next sentence (cheap context), and position in
the document (10 buckets) x domain. Multinomial logistic regression with
balanced class weights. Same 5 folds as phase3_rr.py; writes
results/<tag>/fold*.pt + meta.json in the same format, so phase3_ens.py
ensembles it with the neural runs and adds transition decoding on top.

RULES NOTE. This model uses no pretrained weights, but it is not a Phase-1
model either. Like Phase 2's rules, any submission that includes it is a
HYBRID and is reported beside the model-only number, never instead of it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from phase3_data import P3, LABELS, DOMAINS, load_all, log, prf, report, ensure_split  # noqa: E402

IX = {l: i for i, l in enumerate(LABELS)}


def rows(docs, ids):
    """Flattened sentences with neighbours and position, in document order."""
    cur, prev, nxt, pos, dom, y, owner = [], [], [], [], [], [], []
    for d in ids:
        t = docs[d]["text"]
        n = len(t)
        for i, s in enumerate(t):
            cur.append(s)
            prev.append(t[i - 1] if i else "<start>")
            nxt.append(t[i + 1] if i + 1 < n else "<end>")
            pos.append(min(9, i * 10 // n))
            dom.append(docs[d]["domain"])
            y.append(docs[d]["labels"][i] if docs[d].get("labels") else None)
            owner.append((d, i))
    return dict(cur=cur, prev=prev, nxt=nxt, pos=pos, dom=dom, y=y, owner=owner)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="lex_tfidf")
    ap.add_argument("--C", type=float, default=4.0)
    ap.add_argument("--extra", action="store_true",
                    help="add single-label external sentences (partial sets are skipped)")
    ap.add_argument("--extra-weight", type=float, default=0.3)
    ap.add_argument("--add-docs", default=None,
                    help="jsonl of fully labelled documents added to every fold's training set")
    a = ap.parse_args()

    from scipy.sparse import hstack, csr_matrix
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    docs, _ = load_all()
    split = ensure_split(docs)
    folds = split["folds"]
    test_ids = sorted(d for d, r in docs.items() if r["split"] == "test")
    add_ids = []
    if a.add_docs:
        for line in open(a.add_docs, encoding="utf-8"):
            r = json.loads(line)
            d = f"add:{r['id']}"
            docs[d] = dict(id=d, domain=r["domain"], split="add", text=r["text"], labels=r["labels"])
            add_ids.append(d)
        log(f"  added {len(add_ids)} labelled documents to every fold's training set")

    ext = None
    if a.extra:
        es, ey = [], []
        for name in ("marro", "kalamkar_mask"):
            p = f"{P3}/external/{name}.jsonl"
            if not os.path.exists(p):
                continue
            for line in open(p, encoding="utf-8"):
                r = json.loads(line)
                for s, al in zip(r["text"], r["allowed"]):
                    if len(al) == 1:
                        es.append(s)
                        ey.append(LABELS[al[0]])
        ext = (es, ey)
        log(f"  external single-label sentences: {len(es)}")

    def featurize(fit_rows, rows_list):
        wv = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200000,
                             sublinear_tf=True, lowercase=True)
        cv = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                             max_features=200000, sublinear_tf=True)
        corpus = fit_rows["cur"] + (ext[0] if ext else [])
        wv.fit(corpus)
        cv.fit(corpus)

        def one(r, with_ctx=True):
            parts = [wv.transform(r["cur"]), cv.transform(r["cur"])]
            if with_ctx:
                parts += [0.5 * wv.transform(r["prev"]), 0.5 * wv.transform(r["nxt"])]
            P = np.zeros((len(r["cur"]), 20), dtype=np.float32)
            for k, (p, dm) in enumerate(zip(r["pos"], r["dom"])):
                P[k, p + (10 if dm == "IT" else 0)] = 1.0
            parts.append(csr_matrix(P))
            return hstack(parts).tocsr()
        return [one(r) for r in rows_list], (lambda sents: hstack(
            [wv.transform(sents), cv.transform(sents),
             csr_matrix((len(sents), wv.transform(["x"]).shape[1] * 2 + 20))]).tocsr())

    results = []
    t0 = time.time()
    for k, ev in enumerate(folds):
        tr = [d for j, f in enumerate(folds) if j != k for d in f] + add_ids
        R_tr, R_ev, R_te = rows(docs, tr), rows(docs, ev), rows(docs, test_ids)
        (X_tr, X_ev, X_te), ext_feat = featurize(R_tr, [R_tr, R_ev, R_te])
        keep = np.array([v is not None for v in R_tr["y"]])    # unlabelled sentences of added docs
        X_tr = X_tr[keep]
        y = np.array([IX[v] for v in R_tr["y"] if v is not None])
        w = np.ones(len(y))
        if ext:
            X_ex = ext_feat(ext[0])
            from scipy.sparse import vstack
            X_tr = vstack([X_tr, X_ex]).tocsr()
            y = np.concatenate([y, [IX[v] for v in ext[1]]])
            w = np.concatenate([w, np.full(len(ext[1]), a.extra_weight)])
        clf = LogisticRegression(C=a.C, max_iter=3000, class_weight="balanced")
        clf.fit(X_tr, y, sample_weight=w)
        full = np.full((X_ev.shape[0], len(LABELS)), -30.0)
        full[:, clf.classes_] = clf.predict_log_proba(X_ev)
        fte = np.full((X_te.shape[0], len(LABELS)), -30.0)
        fte[:, clf.classes_] = clf.predict_log_proba(X_te)

        def per_doc(R, M):
            out, i = {}, 0
            for d, j in R["owner"]:
                out.setdefault(d, []).append(M[i])
                i += 1
            return {d: torch.tensor(np.stack(v), dtype=torch.float32) for d, v in out.items()}
        oof, te = per_doc(R_ev, full), per_doc(R_te, fte)
        yt = [v for d in ev for v in docs[d]["labels"]]
        yp = [LABELS[i] for d in ev for i in oof[d].argmax(-1).tolist()]
        f1 = prf(yt, yp, LABELS)["macro_f1"]
        log(f"  fold {k}: macro-F1 {f1:.4f} ({time.time()-t0:.0f}s)")
        r = dict(fold=k, best_epoch=0, best_f1=f1, trace=[], oof={d: v.half() for d, v in oof.items()},
                 test={d: v.half() for d, v in te.items()}, crf=None, n_trainable=int(clf.coef_.size),
                 n_trunk=0, info={}, peak_gb=0.0, test_sent_per_s=None, ranks=None,
                 train_s=time.time() - t0)
        results.append(r)

    rd = f"{P3}/results/{a.tag}"
    os.makedirs(rd, exist_ok=True)
    for r in results:
        torch.save(r, f"{rd}/fold{r['fold']}.pt")
    ids = [d for f in folds for d in f]
    yt, yp = [], []
    pred = {}
    for r in results:
        for d, v in r["oof"].items():
            pred[d] = [LABELS[i] for i in v.float().argmax(-1).tolist()]
    for d in ids:
        yt += docs[d]["labels"]
        yp += pred[d]
    res = prf(yt, yp, LABELS)
    report(res, LABELS, f"{a.tag} OOF", yt, yp)
    by = {}
    for dom in DOMAINS:
        di = [d for d in ids if docs[d]["domain"] == dom]
        by[dom] = prf([v for d in di for v in docs[d]["labels"]], [v for d in di for v in pred[d]],
                      LABELS)["macro_f1"]
    meta = dict(tag=a.tag, args=dict(model="tfidf", method="logreg", enc="-", seq="none", crf=False,
                                     C=a.C, extra=a.extra),
                cv_macro_f1=res["macro_f1"], cv_macro_p=res["macro_p"], cv_macro_r=res["macro_r"],
                cv_by_domain=by, fold_f1=[r["best_f1"] for r in results], best_epochs=[0] * len(folds),
                per_label={l: res["per"][l][2] for l in LABELS}, n_trainable=results[0]["n_trainable"],
                n_trunk=0, info={"hybrid": True}, peak_gb=0.0, test_sent_per_s=None,
                train_s=time.time() - t0, ranks=None)
    json.dump(meta, open(f"{rd}/meta.json", "w"), indent=1)
    log(f"  by domain: CL {by['CL']:.4f}  IT {by['IT']:.4f} | wrote {rd}")


if __name__ == "__main__":
    main()
