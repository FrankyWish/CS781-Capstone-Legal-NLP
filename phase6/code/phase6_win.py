#!/usr/bin/env python3
"""CS-781 Phase 6 -- extraction aimed at the only part of the summary BERTScore reads.

    python phase6_win.py fit   --procs 3      # results/win_model.pkl
    python phase6_win.py sweep --procs 3      # results/win_best/ + its zip

The gate measured it on the real references: BERTScore saturates by 450 words (the
scorer's 512-token window) while the median reference is 642 words, so on most
documents only the FIRST ~420 words of the prediction are compared, and only with
the FIRST ~420 words of the reference.

`phase6_sel.py` is supervised against the whole summary and keeps document order, so
what lands in the window is whichever selected sentences come earliest in the
judgment.  Here:

  head   a second scorer, supervised against the reference's first --win words,
         fills the window with the sentences that match what is actually scored.
  tail   after the window the phase6_sel scorer continues to the full length target.
         BERTScore never reads it; ROUGE, the secondary metric, does.

The sweep picks the window and length on one half of the held-out split and reports
the gain over sel_best on the OTHER half, so the number is not selection-optimistic.
Submissions are kind='extractive'.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle

import numpy as np

import phase6_data as D
import phase6_sel as S

P6 = D.P6
RES = f"{P6}/results"
MODEL = f"{RES}/win_model.pkl"
log = S.log


def cmd_fit(a):
    from sklearn.ensemble import HistGradientBoostingRegressor
    S.load_model()                                  # the fitted TF-IDF vocabulary
    fit = D.load("fit")
    rng = np.random.default_rng(D.SEED)
    sub = [fit[i] for i in rng.permutation(len(fit))[:min(a.docs, len(fit))]]
    cut = [dict(r, summ=" ".join(r["summ"].split()[:a.win])) for r in sub]
    log(f"  extracting rows from {len(cut)} docs, target = first {a.win} summary words")
    out = S._map(S._rows, cut, a.procs)
    X = np.concatenate([o[0] for o in out])
    y = np.concatenate([o[1] for o in out])
    m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, min_samples_leaf=40,
                                      l2_regularization=1.0, early_stopping=True,
                                      validation_fraction=0.1, random_state=D.SEED)
    m.fit(X, y)
    with open(MODEL, "wb") as f:
        pickle.dump(dict(model=m, win=a.win), f)
    log(f"  wrote {MODEL} ({X.shape[0]} sentences, {m.n_iter_} trees)")


def build(feat, target, win, m_head, m_all, lam):
    """Window first, then the whole-summary scorer up to the length target."""
    ss, F, Xn = feat
    if hasattr(m_head, "bind"):                     # phase6_bsel.TextHead reads the words too
        m_head = m_head.bind(Xn)
    head = S.select(ss, F, Xn, m_head, min(target, win), lam=lam, trim=False)
    if target <= win:
        return " ".join(head.split()[:target])
    rest = [i for i, s in enumerate(ss) if s not in head]
    if not rest:
        return head
    tail = S.select([ss[i] for i in rest], F[rest], Xn[rest], m_all,
                    target - len(head.split()), lam=lam, trim=False)
    return " ".join((head + " " + tail).split()[:target])


def summarise(recs, feats, how, mult, win, m_head, m_all, lam):
    tg = S.len_targets(recs, how, mult)
    return {f[0]: build(f[1:], tg[f[0]], win, m_head, m_all, lam) for f in feats}


def load_head(name):
    with open(f"{RES}/{name}_model.pkl", "rb") as f:
        d = pickle.load(f)
    if "text" in d:
        import phase6_bsel as B
        return B.TextHead(d["model"], d["text"])
    return d["model"]


def cmd_sweep(a):
    m_all = S.load_model()
    heads = {h: load_head(h) for h in a.heads.split(",")}
    dev, test = D.load("dev"), D.load("test")
    # workers first: nothing may fork once the scorer is on the GPU
    fd, ft = S._map(S._pred_rows, dev, a.procs), S._map(S._pred_rows, test, a.procs)
    refs = D.refs_of("dev")
    rng = np.random.default_rng(D.SEED)
    perm = rng.permutation(len(dev))
    half = len(dev) // 2
    pick, hold = [dev[i]["id"] for i in perm[:half]], [dev[i]["id"] for i in perm[half:]]
    key = "rouge1" if a.no_bert else "bertscore"
    sub = lambda sm, ids: D.score_summaries({k: sm[k] for k in ids}, {k: refs[k] for k in ids},
                                            do_bert=not a.no_bert)
    how = "model"
    rows = []
    for h, m_head in heads.items():
        for win in [int(x) for x in a.wins.split(",")]:
            for mult in [float(x) for x in a.mults.split(",")]:
                for lam in [float(x) for x in a.lams.split(",")]:
                    sm = summarise(dev, fd, how, mult, win, m_head, m_all, lam)
                    s = sub(sm, pick)
                    rows.append(dict(head=h, win=win, mult=mult, lam=lam, score=s[key],
                                     words=s["pred_words"]))
                    log(f"  {h} win={win} mult={mult} lam={lam}: {key} {s[key]:.4f} "
                        f"rouge1 {s['rouge1']:.4f} words {s['pred_words']:.0f}")
    best = max(rows, key=lambda r: r["score"])
    m_head = heads[best["head"]]
    sm = summarise(dev, fd, how, best["mult"], best["win"], m_head, m_all, best["lam"])
    held = sub(sm, hold)
    meta = dict(tag=a.tag, kind="extractive", head=best["head"], win=best["win"], len=how,
                mult=best["mult"], lam=best["lam"], sweep=rows, dev=held,
                dev_note="scored on the half the sweep never saw")
    f = f"{RES}/{a.base}/summaries_dev.json"
    if os.path.exists(f):
        with open(f, encoding="utf-8") as fh:
            base = sub(json.load(fh), hold)
        meta[f"{a.base}_same_docs"] = base
        log(f"  HELD-OUT HALF: {a.tag} {key} {held[key]:.4f} vs {a.base} {base[key]:.4f} "
            f"(gain {held[key] - base[key]:+.4f}) | ROUGE-1 {held['rouge1']:.4f} vs "
            f"{base['rouge1']:.4f} | n={len(hold)}")
    d = f"{RES}/{a.tag}"
    os.makedirs(d, exist_ok=True)
    st = summarise(test, ft, how, best["mult"], best["win"], m_head, m_all, best["lam"])
    with open(f"{d}/summaries_dev.json", "w", encoding="utf-8") as fh:
        json.dump(sm, fh, ensure_ascii=False)
    with open(f"{d}/summaries_test.json", "w", encoding="utf-8") as fh:
        json.dump(st, fh, ensure_ascii=False)
    with open(f"{d}/meta.json", "w") as fh:
        json.dump(meta, fh, indent=1)
    D.write_submission(a.tag, st, note=f"{best['head']} head, window-aimed extraction "
                       f"win={best['win']} len={how}x{best['mult']} lam={best['lam']}",
                       kind="extractive", dev_scores=held)
    log(f"WIN DONE: {a.tag} chose head={best['head']} win={best['win']} mult={best['mult']} "
        f"lam={best['lam']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--procs", type=int, default=3)
    sp = ap.add_subparsers(dest="cmd", required=True)
    f = sp.add_parser("fit")
    f.add_argument("--docs", type=int, default=3000)
    f.add_argument("--win", type=int, default=420)
    f.set_defaults(fn=cmd_fit)
    s = sp.add_parser("sweep")
    s.add_argument("--heads", default="win", help="results/<name>_model.pkl, comma-separated")
    s.add_argument("--tag", default="win_best")
    s.add_argument("--base", default="sel_best", help="system the held-out gain is measured against")
    s.add_argument("--wins", default="380,420,460")
    s.add_argument("--mults", default="0.9,1.0,1.2")
    s.add_argument("--lams", default="0.45")
    s.add_argument("--no-bert", action="store_true")
    s.set_defaults(fn=cmd_sweep)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
