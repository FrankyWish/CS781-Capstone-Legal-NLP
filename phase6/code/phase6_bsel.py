#!/usr/bin/env python3
"""CS-781 Phase 6 -- sentence choice aimed at BERTScore itself, not at ROUGE.

    python phase6_bsel.py --procs 10 fit --docs 1500     # results/bsel_{f,g}_model.pkl

phase6_sel supervises its scorer with each sentence's ROUGE-1 F, a proxy.  The metric
that ranks us is token-level: every token of the summary takes its best cosine against
the reference's tokens under roberta-large layer 17, and the other way round.  Those
two tables can be computed for every sentence of a training document, so:

  fit        labels from the metric, on training documents whose references are public.
             y_f = the sentence's own BERTScore F1 against the reference window.
             y_g = membership of the greedy oracle, which also sees redundancy.
             Both feed the same cheap HistGB on the same lexical features, so no
             neural network is needed at test time.  `phase6_win.py sweep --heads
             bsel_f,bsel_g --tag bsel_best` picks one on half of dev and reports it on
             the other half.  Submissions are kind='extractive'.

  consensus  sentence-level MBR.  No reference exists at test time, so the top-k
             systems stand in for it: greedily pick the sentences that maximise the
             mean BERTScore against those k summaries.  Candidates are the document's
             sentences, optionally with the best neural members' sentences.  Phase 1's
             whole-candidate MBR was worth +0.0068; this can recombine candidates
             instead of only choosing one.  roberta-large sits in this decision rule,
             so it is labelled kind='hybrid' and reported that way.

The tables score each sentence alone, while the real scorer embeds the summary as one
sequence.  That is an approximation, which is why every number that matters is the true
BERTScore on held-out documents.
"""
from __future__ import annotations

import argparse
import collections
import os
import pickle
import time

import numpy as np
import torch
import torch.nn.functional as TF

import phase6_data as D
import phase6_sel as S

P6 = D.P6
RES = f"{P6}/results"
WINDOW = 510                    # roberta's 512 minus the two special tokens
log = S.log


def enc(texts):
    from bert_score.utils import sent_encode
    tk = D.bert().s._tokenizer
    return [sent_encode(tk, t) for t in texts]


@torch.no_grad()
def embed(ids, batch=64):
    """(token vectors, sentence index of each token, tokens per sentence); specials dropped."""
    sc = D.bert().s
    dev = next(sc._model.parameters()).device
    pad = sc._tokenizer.pad_token_id
    order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
    out = [None] * len(ids)
    for b in range(0, len(order), batch):
        ix = order[b:b + batch]
        m = max(len(ids[i]) for i in ix)
        x = torch.full((len(ix), m), pad, dtype=torch.long, device=dev)
        mask = torch.zeros_like(x)
        for r, i in enumerate(ix):
            x[r, :len(ids[i])] = torch.tensor(ids[i], device=dev)
            mask[r, :len(ids[i])] = 1
        e = TF.normalize(sc._model(x, attention_mask=mask)[0], dim=-1)
        for r, i in enumerate(ix):
            out[i] = e[r, 1:max(len(ids[i]) - 1, 1)]
    n = torch.tensor([len(c) for c in out], device=dev)
    seg = torch.repeat_interleave(torch.arange(len(out), device=dev), n)
    return torch.cat(out), seg, n


def tables(E, seg, ns, R):
    """Per sentence: summed best cosine of its tokens (precision side), and its best
    cosine at each reference token (recall side)."""
    sim = E @ R.T
    p = torch.zeros(ns, device=E.device).index_add_(0, seg, sim.max(1).values)
    C = torch.full((ns, R.shape[0]), -1.0, device=E.device)
    C.scatter_reduce_(0, seg[:, None].expand(-1, R.shape[0]), sim, "amax")
    return p, C.clamp(min=0)


def greedy(Ps, Cs, n, cost, stop, allow=None):
    """Sentence order that maximises the mean F1 against every reference in Ps/Cs."""
    ok = torch.ones_like(n, dtype=torch.bool) if allow is None else allow.clone()
    nf = n.float()
    cov = [torch.zeros(C.shape[1], device=C.device) for C in Cs]
    sp = [0.0] * len(Ps)
    N, spent, order = 0.0, 0, []
    best = 0.0
    while spent < stop and ok.any():
        sc = 0.0
        for r, (P, C) in enumerate(zip(Ps, Cs)):
            rec = torch.maximum(cov[r], C).mean(1)
            pre = (sp[r] + P) / (N + nf).clamp(min=1)
            sc = sc + 2 * pre * rec / (pre + rec).clamp(min=1e-6)
        sc = sc / len(Ps)
        sc[~ok] = -1
        i = int(sc.argmax())
        best = float(sc[i])
        order.append(i)
        ok[i] = False
        spent += int(cost[i])
        N += float(nf[i])
        for r, (P, C) in enumerate(zip(Ps, Cs)):
            sp[r] += float(P[i])
            cov[r] = torch.maximum(cov[r], C[i])
    return order, best


# ----------------------------------------------------------------------- fit --
class TextHead:
    """HistGB over the lexical features plus a word-level score.  S.select hands a model
    only F, so the document's TF-IDF rows are bound first (phase6_win.build does this)."""

    def __init__(self, gb, text, Xn=None):
        self.gb, self.text, self.Xn = gb, text, Xn

    def bind(self, Xn):
        return TextHead(self.gb, self.text, Xn)

    def predict(self, F):
        return self.gb.predict(np.c_[F, self.text.predict(self.Xn)])


def text_scores(Xs, y, folds=3, alpha=10.0):
    """Ridge on sentence TF-IDF.  The 27 hand features cannot see vocabulary, and BERTScore
    rewards the headnote's register.  Out-of-fold by document, so the HistGB on top sees
    the score it will get at test time, not one fitted to these labels."""
    from scipy.sparse import vstack
    from sklearn.linear_model import Ridge
    fold = np.arange(len(Xs)) % folds
    oof = [None] * len(Xs)
    for k in range(folds):
        tr = np.where(fold != k)[0]
        m = Ridge(alpha=alpha).fit(vstack([Xs[i] for i in tr]), np.concatenate([y[i] for i in tr]))
        for i in np.where(fold == k)[0]:
            oof[i] = m.predict(Xs[i]).astype(np.float32)
    o, t = np.concatenate(oof), np.concatenate(y)
    log(f"  text score out-of-fold correlation with the label {np.corrcoef(o, t)[0, 1]:.4f}")
    return o, Ridge(alpha=alpha).fit(vstack(Xs), t)


def cmd_fit(a):
    from sklearn.ensemble import HistGradientBoostingRegressor
    S.load_model()
    fit = D.load("fit")
    rng = np.random.default_rng(D.SEED)
    sub = [fit[i] for i in rng.permutation(len(fit))[:min(a.docs, len(fit))]]
    rows = S._map(S._pred_rows, sub, a.procs)           # workers first: no fork once on the GPU
    t0 = time.time()
    X, Xs, yf, yg, orc = [], [], [], [], []
    for k, ((_, ss, F, Xn), r) in enumerate(zip(rows, sub)):
        if time.time() - t0 > a.budget_min * 60:
            log(f"  time budget reached at {k} docs")
            break
        if len(ss) > a.max_sents:           # a few huge records would dominate memory, not the fit
            continue
        E, seg, n = embed(enc(ss))
        R = embed(enc([r["summ"]]))[0]
        if not R.shape[0]:
            continue
        p, C = tables(E, seg, len(ss), R)
        pre, rec = p / n.clamp(min=1).float(), C.mean(1)
        f = (2 * pre * rec / (pre + rec).clamp(min=1e-6)).cpu().numpy()
        order, best = greedy([p], [C], n, n, min(WINDOW, R.shape[0]))
        g = np.zeros(len(ss), dtype=np.float32)
        for j, i in enumerate(order):
            g[i] = 1 - 0.5 * j / len(order)
        X.append(F)
        Xs.append(Xn)
        # same spread as phase6_sel's ROUGE labels, so MMR's lam means the same thing
        yf.append(0.05 * (f - f.mean()) / max(float(f.std()), 1e-6))
        yg.append(0.2 * g)
        orc.append(best)
        if k % 200 == 0:
            log(f"  {k}/{len(sub)} docs, {(time.time() - t0) / 60:.1f} min, surrogate oracle "
                f"F {np.mean(orc):.4f}")
    X = np.concatenate(X)
    log(f"  {X.shape[0]} sentences from {len(orc)} docs, surrogate oracle F {np.mean(orc):.4f}")
    for name, y in ((f"{a.tag}_f", yf), (f"{a.tag}_g", yg)):
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, min_samples_leaf=40,
                                          l2_regularization=1.0, early_stopping=True,
                                          validation_fraction=0.1, random_state=D.SEED)
        extra = {}
        if a.text:
            oof, extra["text"] = text_scores(Xs, y)
            m.fit(np.c_[X, oof], np.concatenate(y))
        else:
            m.fit(X, np.concatenate(y))
        with open(f"{RES}/{name}_model.pkl", "wb") as f:
            pickle.dump(dict(model=m, oracle=float(np.mean(orc)), docs=len(orc), **extra), f)
        log(f"  wrote {RES}/{name}_model.pkl ({m.n_iter_} trees)")


# ----------------------------------------------------------------- consensus --
def assemble(order, sents, wc, target, win):
    """Selection order decides what gets in; inside the window and after it, the text
    keeps candidate order, so nothing chosen early falls past the scorer's window."""
    head, tail, w = [], [], 0
    for i in order:
        if w >= target:
            break
        (head if w < win else tail).append(i)
        w += wc[i]
    text = " ".join(sents[i] for i in sorted(head) + sorted(tail))
    return " ".join(text.split()[:target])


def consensus(feats, pool, split, ranked, neural, target, rel, ks, mults, win=420, cap=250):
    """{(k, mixed, mult): {id: text}}.  `ranked` = systems by held-out score."""
    out = collections.defaultdict(dict)
    t0 = time.time()
    for done, (rid, ss, F, _) in enumerate(feats):
        keep = sorted(np.argsort(-rel.predict(F))[:cap].tolist())
        sents = [ss[i] for i in keep]
        nd = len(sents)
        for s in neural[:2]:
            sents += [x for x in D.sentences(pool[s][split][rid]) if x not in sents]
        wc = [len(x.split()) for x in sents]
        E, seg, n = embed(enc(sents))
        Ps, Cs = [], []
        for s in ranked[:max(ks)]:
            R = embed(enc([pool[s][split][rid]]))[0]
            if R.shape[0]:
                p, C = tables(E, seg, len(sents), R)
                Ps.append(p)
                Cs.append(C)
        cost = torch.tensor(wc, device=E.device)
        stop = int(target[rid] * max(mults))
        doc_only = torch.zeros(len(sents), dtype=torch.bool, device=E.device)
        doc_only[:nd] = True
        for k in ks:
            for mixed in ([False, True] if neural and len(sents) > nd else [False]):
                order, _ = greedy(Ps[:k], Cs[:k], n, cost, stop, None if mixed else doc_only)
                for m in mults:
                    out[(k, mixed, m)][rid] = assemble(order, sents, wc, int(target[rid] * m), win)
        if done % 50 == 0:
            log(f"  consensus/{split}: {done}/{len(feats)} docs, {(time.time() - t0) / 60:.1f} min")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--procs", type=int, default=3)
    sp = ap.add_subparsers(dest="cmd", required=True)
    f = sp.add_parser("fit")
    f.add_argument("--docs", type=int, default=1500)
    f.add_argument("--budget-min", type=float, default=30)
    f.add_argument("--tag", default="bsel", help="writes results/<tag>_{f,g}_model.pkl")
    f.add_argument("--text", action="store_true", help="add the word-level score")
    f.add_argument("--max-sents", type=int, default=1500)
    f.set_defaults(fn=cmd_fit)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
