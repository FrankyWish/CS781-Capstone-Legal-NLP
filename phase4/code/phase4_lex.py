#!/usr/bin/env python3
"""CS-781 Phase 4 -- lexical baselines (TF-IDF + logistic regression). CPU only.

    python phase4_lex.py --view all  --tag lex_all
    python phase4_lex.py --view tail --chars 6000 --tag lex_tail
    python phase4_lex.py --sections          # head / middle / tail / all, one model each: the
                                             # "which sections carry the outcome" evidence

Features: hashed word 1-2-gram TF-IDF (sublinear tf, 2^21 buckets) of the chosen view,
plus hashed char 3-5-gram TF-IDF (char_wb) when --char is on. Logistic regression, C picked on
dev Macro-F1 over a small grid, balanced class weights. Writes results/<tag>/
in the shared run format so phase4_ens.py can mix it with the neural runs.

RULES NOTE. No pretrained weights and nothing outside the CJPE data, but it is
not a Phase-1 model: any submission containing it is a HYBRID, reported beside
the model-only number, never instead of it.
"""

from __future__ import annotations

import argparse
import collections
import os
import random
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from phase4_data import P4, load_all, load_jsonl, log, summarise, save_run, write_submission  # noqa: E402


def view(t, v, n):
    if v == "all":
        return t
    if v == "head":
        return t[:n]
    if v == "tail":
        return t[-n:]
    if v == "mid":
        m = len(t) // 2
        return t[max(0, m - n // 2): m + n // 2]
    raise ValueError(v)


def fit_one(data, v, chars, use_char, tag, submit, Cs, max_feat, sw=None, note=""):
    from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
    from sklearn.linear_model import LogisticRegression
    from scipy.sparse import hstack
    t0 = time.time()
    X = {s: [view(t, v, chars) for _, t, _ in data[s]] for s in data}
    y = {s: np.array([l for _, _, l in data[s]]) for s in ("train", "dev")}

    def feats(vec, tr, dv, te):
        # hashing keeps memory flat on a 12 GB Colab CPU runtime: a fitted
        # vocabulary of every word bigram in 37k judgments would not fit
        tf = TfidfTransformer(sublinear_tf=True)
        a_ = tf.fit_transform(vec.transform(tr))
        return a_, tf.transform(vec.transform(dv)), tf.transform(vec.transform(te))

    wv = HashingVectorizer(ngram_range=(1, 2), n_features=max_feat, alternate_sign=False, norm=None,
                           dtype=np.float32, lowercase=True)
    a1, b1, c1 = feats(wv, X["train"], X["dev"], X["test"])
    Mtr, Mdv, Mte = [a1], [b1], [c1]
    if use_char:
        cv = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 5), n_features=max_feat,
                               alternate_sign=False, norm=None, dtype=np.float32)
        tl = lambda s: [view(t, "tail", min(chars, 6000)) if v == "all" else x
                        for (_, t, _), x in zip(data[s], X[s])]
        a2, b2, c2 = feats(cv, tl("train"), tl("dev"), tl("test"))
        Mtr.append(a2)
        Mdv.append(b2)
        Mte.append(c2)
    Xtr, Xdv, Xte = hstack(Mtr).tocsr(), hstack(Mdv).tocsr(), hstack(Mte).tocsr()
    log(f"  [{tag}] view={v} chars={chars} char={use_char}: {Xtr.shape[1]} features, "
        f"vectorised in {time.time()-t0:.0f}s")
    best = None
    for C in Cs:
        clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced", solver="liblinear")
        clf.fit(Xtr, y["train"], sample_weight=sw)
        zd = clf.decision_function(Xdv)
        s = summarise(y["dev"].tolist(), zd.tolist())
        log(f"    C={C:<5}: dev macro-F1 tuned {s['f1_tuned']:.4f} @0 {s['f1_at0']:.4f} AUC {s['auc']:.4f}")
        if best is None or s["f1_tuned"] > best[0]["f1_tuned"]:
            best = (s, C, zd, clf.decision_function(Xte))
    s, C, zd, zt = best
    meta = dict(tag=tag, kind="hybrid", technique="tfidf-lr", arch="lexical", view=v, chars=chars,
                char_ngrams=use_char, C=C, n_train=len(data["train"]), note=note, dev_f1=s["f1_tuned"], dev_f1_at0=s["f1_at0"], thr=s["thr"],
                dev_auc=s["auc"], dev_p=s["precision"], dev_r=s["recall"],
                n_features=int(Xtr.shape[1]), train_s=time.time() - t0, n_trainable=int(Xtr.shape[1]))
    ids_d = [r[0] for r in data["dev"]]
    ids_t = [r[0] for r in data["test"]]
    save_run(tag, ids_d, zd, ids_t, zt, meta)
    log(f"  [{tag}] BEST C={C}: dev macro-F1 {s['f1_tuned']:.4f} (thr {s['thr']:+.3f}) AUC {s['auc']:.4f} "
        f"P {s['precision']:.3f} R {s['recall']:.3f} ({time.time()-t0:.0f}s)")
    if submit:
        pred = {i: int(v_ > s["thr"]) for i, v_ in zip(ids_t, zt)}
        write_submission(tag, ids_t, pred, dev=s["f1_tuned"], kind="hybrid",
                         note=f"TF-IDF+LR view={v} chars={chars} char={use_char} n={len(data['train'])} {note}")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", default="all", choices=["all", "head", "tail", "mid"])
    ap.add_argument("--chars", type=int, default=6000)
    ap.add_argument("--char", action="store_true", help="add char 3-5-gram features")
    ap.add_argument("--sections", action="store_true")
    ap.add_argument("--C", nargs="+", type=float, default=[1.0, 4.0, 16.0])
    ap.add_argument("--max-feat", type=int, default=1 << 21)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--no-submit", dest="submit", action="store_false")
    ap.add_argument("--train-max", type=int, default=0, help="use only this many train docs (seed-781 order)")
    ap.add_argument("--extra", default=None, help="external jsonl (phase4_external.py output)")
    ap.add_argument("--extra-max", type=int, default=0)
    ap.add_argument("--extra-weight", type=float, default=1.0)
    ap.add_argument("--curve", action="store_true",
                    help="learning curve: dev F1 at 2.5k/5k/10k/20k/all gold docs (+ all + extra)")
    a = ap.parse_args()
    data = load_all()
    gold = data["train"]
    perm = list(range(len(gold)))
    random.Random(781).shuffle(perm)            # same order as phase4_cjpe.py
    ext = load_jsonl(a.extra, a.extra_max or None) if a.extra else []
    if a.extra:
        log(f"  extra: {len(ext)} docs from {a.extra}, labels {dict(collections.Counter(l for _, _, l in ext))}, "
            f"weight {a.extra_weight}")

    def with_train(n, use_ext):
        d = dict(data)
        tr = [gold[i] for i in perm[:n]] if n else list(gold)
        w = [1.0] * len(tr)
        if use_ext:
            tr += ext
            w += [a.extra_weight] * len(ext)
        d["train"] = tr
        return d, np.array(w)

    if a.sections:
        out = {}
        for v in ("head", "mid", "tail", "all"):
            out[v] = fit_one(data, v, a.chars, False, f"lexsec_{v}", False, a.C, a.max_feat)
        log("\n  section evidence (word TF-IDF, same model, different part of the judgment):")
        for v, s in out.items():
            log(f"    {v:5s} {a.chars if v != 'all' else 'all'} chars: dev macro-F1 {s['f1_tuned']:.4f} AUC {s['auc']:.4f}")
        return
    if a.curve:
        rows = []
        sizes = [n for n in (2500, 5000, 10000, 20000) if n < len(gold)] + [0]
        for n in sizes:
            d, w = with_train(n, False)
            s = fit_one(d, a.view, a.chars, False, f"lexcurve_{n or 'all'}", False, a.C, a.max_feat, w)
            rows.append((len(d["train"]), "gold", s))
        if ext:
            d, w = with_train(0, True)
            s = fit_one(d, a.view, a.chars, False, "lexcurve_all_ext", False, a.C, a.max_feat, w)
            rows.append((len(d["train"]), f"gold + {len(ext)} silver", s))
        log(f"\n  learning curve (word TF-IDF, view={a.view}):")
        for n, what, s in rows:
            log(f"    {n:6d} docs ({what}): dev macro-F1 {s['f1_tuned']:.4f} AUC {s['auc']:.4f}")
        return
    d, w = with_train(a.train_max, bool(ext))
    tag = a.tag or f"lex_{a.view}"
    fit_one(d, a.view, a.chars, a.char, tag, a.submit, a.C, a.max_feat, w,
            note=f"+{len(ext)} silver" if ext else "")


if __name__ == "__main__":
    main()
