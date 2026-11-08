#!/usr/bin/env python3
"""CS-781 Phase 3 -- stacking over finished runs, plus the dissent rule. CPU only.

    python phase3_stack.py --runs ext_M4 lorma_M3 ext_M2 base_M2 lex_tfidf lex_ext --tag stk1
    python phase3_stack.py --runs ... --no-dissent --tag stk1_nodiss
    python phase3_stack.py --dissent-only              # just report the rule on train and test

STACKING. A second-level classifier reads, for sentence i:
  - every member's probabilities at i
  - the members' mean probabilities at i-2, i-1, i+1, i+2 (what the
    neighbours are predicted to be: "the sentence after a respondent's
    argument is probably also the respondent's")
  - position in the document, domain, sentence length.
It is trained on out-of-fold member predictions only, and is itself scored
out-of-fold (stacker for fold k never sees fold k), so its CV number is honest.
Then the same two global decoding knobs as phase3_ens.py: prior correction tau
and transition Viterbi beta. Phase 2's stacker was the largest single gain on
the test set (+0.057).

Known risk, from Phase 2: members' test log-probs are averages of 5 fold models
and are smoother than their single-model OOF log-probs. Features are
probabilities (bounded) rather than log-probs to limit that shift.

DISSENT RULE. In the training data, Dissent occurs in exactly two documents,
both CCI orders, each beginning at a header "<name>, Member (dissenting)" and
running to the end of the document. The rule labels every sentence from such a
header onwards as Dissent. Its precision and recall on the 90 labelled
documents are printed, and so are its matches in the TEST text (reading test
inputs is allowed; no test labels exist here). Any submission using it is a
hybrid, reported beside the model-only number.
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from phase3_data import P3, LABELS, DOMAINS, load_all, log, prf, report, write_submission, ensure_split  # noqa: E402
from phase3_ens import load_run, run_logprobs, stats, decode_one  # noqa: E402

NL = len(LABELS)
IX = {l: i for i, l in enumerate(LABELS)}
# Only the header form seen in training: a short name/title, then "(dissenting)",
# at the START of a sentence. A looser pattern ("the dissenting opinion of X")
# would fire on precedent discussion and relabel the rest of a document.
DISS = re.compile(r"^\W*[A-Z][A-Za-z .,'-]{0,40},\s*(member|chairperson|chairman|j\.?|judge)\s*\(\s*dissent(ing)?\s*\)", re.I)


def dissent_start(sents):
    """Index of the first dissent header in the document, or None."""
    for i, s in enumerate(sents):
        if DISS.search(s):
            return i
    return None


def apply_dissent(pred, docs):
    out = {}
    for d, seq in pred.items():
        k = dissent_start(docs[d]["text"])
        out[d] = list(seq) if k is None else list(seq[:k]) + ["Dissent"] * (len(seq) - k)
    return out


def dissent_report(docs, ids, test_ids):
    tp = fp = fn = 0
    for d in ids:
        k = dissent_start(docs[d]["text"])
        y = docs[d]["labels"]
        p = [False] * len(y) if k is None else [i >= k for i in range(len(y))]
        for yi, pi in zip(y, p):
            tp += pi and yi == "Dissent"
            fp += pi and yi != "Dissent"
            fn += (not pi) and yi == "Dissent"
        if k is not None:
            log(f"    train match {d}: from sentence {k}/{len(y)}: {docs[d]['text'][k][:90]!r}")
    pr = tp / max(1, tp + fp)
    rc = tp / max(1, tp + fn)
    log(f"  dissent rule on labelled docs: P={pr:.3f} R={rc:.3f} (tp {tp} fp {fp} fn {fn})")
    hits = [(d, dissent_start(docs[d]["text"])) for d in test_ids]
    hits = [(d, k) for d, k in hits if k is not None]
    if hits:
        for d, k in hits:
            log(f"  ** TEST document {d} has a dissent header at sentence {k}/{len(docs[d]['text'])}: "
                f"{docs[d]['text'][k][:120]!r}")
    else:
        log("  no test document has a dissent header: Dissent scores 0 for everyone, the rule is inert")
    return hits


def features(lps, doc_ids, docs):
    """lps: list over members of {doc: (n, NL) log-probs}. Returns X, owners."""
    X, own = [], []
    for d in doc_ids:
        P = [lp[d].float().exp().numpy() for lp in lps]           # members' probabilities
        n = P[0].shape[0]
        mean = np.mean(P, 0)
        pad = np.zeros((2, NL), dtype=np.float32)
        mp = np.concatenate([pad, mean, pad])
        nb = [mp[2 + o: 2 + o + n] for o in (-2, -1, 1, 2)]
        pos = np.arange(n) / max(1, n - 1)
        dec = np.zeros((n, 10), dtype=np.float32)
        dec[np.arange(n), np.minimum(9, (np.arange(n) * 10) // n)] = 1
        L = np.array([len(s.split()) for s in docs[d]["text"]], dtype=np.float32)[:, None]
        dom = np.full((n, 1), 1.0 if docs[d]["domain"] == "IT" else 0.0, dtype=np.float32)
        edge = np.zeros((n, 2), dtype=np.float32)
        edge[0, 0] = edge[-1, 1] = 1
        X.append(np.concatenate(P + [mean] + nb + [pos[:, None], dec, np.log1p(L), dom, edge], 1))
        own += [(d, i) for i in range(n)]
    return np.concatenate(X).astype(np.float32), own


def to_docs(M, own, ids):
    out = {d: [] for d in ids}
    for (d, _), row in zip(own, M):
        out[d].append(row)
    return {d: torch.tensor(np.stack(v)) for d, v in out.items()}


def make_clf(kind, seed=0):
    if kind == "lr":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced")
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
                                          l2_regularization=1.0, class_weight="balanced",
                                          early_stopping=False, random_state=seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--model", default="both", choices=["hgb", "lr", "both"])
    ap.add_argument("--no-dissent", action="store_true")
    ap.add_argument("--dissent-only", action="store_true")
    ap.add_argument("--tag", default="stk")
    ap.add_argument("--submit", action="store_true")
    a = ap.parse_args()

    docs, _ = load_all()
    split = ensure_split(docs)
    folds = split["folds"]
    fold_of = {d: k for k, f in enumerate(folds) for d in f}
    labelled = sorted(fold_of)
    test_ids = sorted(d for d, r in docs.items() if r["split"] == "test")
    hits = dissent_report(docs, labelled, test_ids)
    if a.dissent_only:
        return

    oofs, tests, names = [], [], []
    for t in a.runs:
        fr = load_run(t)
        if len(fr) != len(folds):
            log(f"  skip {t}: {len(fr)}/{len(folds)} folds")
            continue
        o, te, _ = run_logprobs(fr)
        oofs.append(o)
        tests.append(te)
        names.append(t)
    assert names, "no complete runs"
    log(f"  members: {names}")
    X, own = features(oofs, labelled, docs)
    y = np.array([IX[docs[d]["labels"][i]] for d, i in own])
    fo = np.array([fold_of[d] for d, _ in own])
    Xt, own_t = features(tests, test_ids, docs)
    log(f"  stack features: {X.shape[1]} per sentence, {len(y)} labelled sentences")

    st_fold = {k: stats(docs, [d for d in labelled if fold_of[d] != k]) for k in range(len(folds))}
    st_all = stats(docs, labelled)

    def oof_probs(kind):
        P = np.zeros((len(y), NL), dtype=np.float32)
        for k in range(len(folds)):
            tr, ev = fo != k, fo == k
            clf = make_clf(kind)
            clf.fit(X[tr], y[tr])
            p = np.full((ev.sum(), NL), 1e-6, dtype=np.float32)
            p[:, clf.classes_] = clf.predict_proba(X[ev])
            P[ev] = p
        return P

    def decode_all(lp_docs, tau, beta, test=False):
        if test:
            return {d: decode_one(v, tau, beta, st_all) for d, v in lp_docs.items()}
        return {d: decode_one(v, tau, beta, st_fold[fold_of[d]]) for d, v in lp_docs.items()}

    def macro(pred):
        yt = [l for d in labelled for l in docs[d]["labels"]]
        yp = [l for d in labelled for l in pred[d]]
        return prf(yt, yp, LABELS)

    # baseline for comparison: plain mean of the members, argmax
    base = {d: torch.stack([o[d] for o in oofs]).mean(0) for d in labelled}
    log(f"  mean of members, argmax: OOF macro-F1 {macro(decode_all(base, 0, 0))['macro_f1']:.4f}")

    kinds = ["hgb", "lr"] if a.model == "both" else [a.model]
    best = None
    for kind in kinds:
        P = oof_probs(kind)
        lp = to_docs(np.log(np.clip(P, 1e-6, 1)), own, labelled)
        for tau in (0.0, 0.25, 0.5):
            for beta in (0.0, 0.25, 0.5):
                pr = decode_all(lp, tau, beta)
                f = macro(pr)["macro_f1"]
                wf = min(prf([l for d in fd for l in docs[d]["labels"]],
                             [l for d in fd for l in pr[d]], LABELS)["macro_f1"] for fd in folds)
                log(f"  {kind} tau={tau} beta={beta}: OOF macro-F1 {f:.4f} (worst fold {wf:.4f})")
                if best is None or f > best[0]:
                    best = (f, kind, tau, beta, lp)
    f, kind, tau, beta, lp = best
    pred = decode_all(lp, tau, beta)
    if not a.no_dissent:
        pred = apply_dissent(pred, docs)
    res = macro(pred)
    yt = [l for d in labelled for l in docs[d]["labels"]]
    yp = [l for d in labelled for l in pred[d]]
    log(f"\n=== stacker {kind} tau={tau} beta={beta}{'' if a.no_dissent else ' + dissent rule'} ===")
    report(res, LABELS, "stacked OOF", yt, yp)
    for dom in DOMAINS:
        di = [d for d in labelled if docs[d]["domain"] == dom]
        log(f"  {dom}: {prf([l for d in di for l in docs[d]['labels']], [l for d in di for l in pred[d]], LABELS)['macro_f1']:.4f}")

    if a.submit:
        clf = make_clf(kind)
        clf.fit(X, y)
        pt = np.full((Xt.shape[0], NL), 1e-6, dtype=np.float32)
        pt[:, clf.classes_] = clf.predict_proba(Xt)
        tlp = to_docs(np.log(np.clip(pt, 1e-6, 1)), own_t, test_ids)
        tp = decode_all(tlp, tau, beta, test=True)
        if not a.no_dissent:
            tp = apply_dissent(tp, docs)
        hybrid = any(n.startswith("lex") for n in names) or (not a.no_dissent and hits)
        write_submission(tp, {d: docs[d] for d in test_ids}, LABELS, a.tag, cv=res["macro_f1"],
                         note=f"stack {kind} {names} tau={tau} beta={beta} "
                              f"dissent={'off' if a.no_dissent else 'on'} {'HYBRID' if hybrid else 'model-only'}")


if __name__ == "__main__":
    main()
