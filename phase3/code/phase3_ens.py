#!/usr/bin/env python3
"""CS-781 Phase 3 -- offline ensembling, decoding, tracking. CPU only.

    python phase3_ens.py ens --runs m2_full m4_lora m1_full            # score + grid
    python phase3_ens.py ens --runs m2_full m4_lora --tau 0.5 --beta 0.5 --submit --tag ens_A
    python phase3_ens.py ens --runs ... --greedy 8                     # Caruana selection
    python phase3_ens.py ens --runs ... --dump-teacher results/ens_A_teacher.pt
    python phase3_ens.py compare m2_full m4_lora                       # paired doc bootstrap
    python phase3_ens.py table                                         # every run, one row
    python phase3_ens.py track --tag ens_A --test 0.71                 # record leaderboard score

Everything reads results/<tag>/fold*.pt written by phase3_rr.py, so no idea
here costs GPU time.

DECODING. Two global knobs, never per label (per-label choices tuned on CV lost
on the leaderboard 4 times out of 4 in Phase 2):

  tau   logit adjustment: score(y) = log p(y|x) - tau * log prior(y).
        Under macro-F1 a rare label is worth as much as a common one, and the
        Bayes-optimal rule for a balanced metric divides by the prior
        (Menon et al. 2021). tau = 0 is plain argmax.
  beta  Viterbi with label-bigram transitions estimated from training folds:
        + beta * log P(y_t | y_t-1). This is the CRF's structure added after
        the fact to models trained without one, at zero GPU cost.

Transition statistics and priors for fold k come only from the other folds,
so the OOF grid is not scored on statistics it was fitted on.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import os
import random
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from phase3_data import P3, LABELS, DOMAINS, load_all, log, prf, report, write_submission  # noqa: E402

NL = len(LABELS)
IX = {l: i for i, l in enumerate(LABELS)}


def load_run(tag):
    fs = sorted(glob.glob(f"{P3}/results/{tag}/fold*.pt"))
    return [torch.load(f, weights_only=False) for f in fs]


def all_tags(exclude=()):
    """Every run with a meta.json (all folds finished), minus excluded prefixes."""
    out = []
    for m in sorted(glob.glob(f"{P3}/results/*/meta.json")):
        t = os.path.basename(os.path.dirname(m))
        if not any(t.startswith(x) for x in exclude):
            out.append(t)
    return out


def run_logprobs(folds):
    """OOF log-probs {doc: (n,NL)}, test log-probs averaged over folds, doc->fold."""
    oof, test, where = {}, collections.defaultdict(list), {}
    for r in folds:
        for d, v in r["oof"].items():
            oof[d] = F.log_softmax(v.float(), -1)
            where[d] = r["fold"]
        for d, v in r["test"].items():
            test[d].append(F.log_softmax(v.float(), -1))
    test = {d: torch.stack(v).mean(0) for d, v in test.items()}
    return oof, test, where


def stats(docs, ids):
    """log prior and log transition matrix (add-1 smoothed) from gold docs."""
    pri = torch.ones(NL)
    tr = torch.ones(NL, NL)
    st = torch.ones(NL)
    for d in ids:
        y = [IX[l] for l in docs[d]["labels"]]
        for a, b in zip(y, y[1:]):
            tr[a, b] += 1
        for a in y:
            pri[a] += 1
        st[y[0]] += 1
    return (pri / pri.sum()).log(), (tr / tr.sum(1, keepdim=True)).log(), (st / st.sum()).log()


def viterbi(score, trans, start):
    a = start + score[0]
    back = []
    for t in range(1, score.size(0)):
        best, idx = (a.unsqueeze(1) + trans).max(0)
        a = best + score[t]
        back.append(idx)
    path = [int(a.argmax())]
    for idx in reversed(back):
        path.append(int(idx[path[-1]]))
    return path[::-1]


def decode_one(lp, tau, beta, st):
    pri, tr, s0 = st
    sc = lp - tau * pri
    if beta > 0:
        return [LABELS[i] for i in viterbi(sc, beta * tr, beta * s0)]
    return [LABELS[i] for i in sc.argmax(-1).tolist()]


def score(pred, docs, ids):
    yt, yp = [], []
    for d in ids:
        yt += docs[d]["labels"]
        yp += pred[d]
    return prf(yt, yp, LABELS), yt, yp


def macro(pred, docs, ids):
    return score(pred, docs, ids)[0]["macro_f1"]


def combine(runs, weights=None):
    weights = weights or [1.0] * len(runs)
    z = sum(weights)
    oof = {d: sum(w * r[0][d] for r, w in zip(runs, weights)) / z for d in runs[0][0]}
    test = {d: sum(w * r[1][d] for r, w in zip(runs, weights)) / z for d in runs[0][1]}
    return oof, test


def cmd_ens(a):
    docs, _ = load_all()
    split = json.load(open(f"{P3}/split.json"))
    folds = split["folds"]
    fold_of = {d: k for k, f in enumerate(folds) for d in f}
    labelled = sorted(fold_of)
    st_fold = {k: stats(docs, [d for d in labelled if fold_of[d] != k]) for k in range(len(folds))}
    st_all = stats(docs, labelled)

    runs, names = [], []
    tags = all_tags(a.exclude) if a.runs == ["all"] else a.runs
    for t in tags:
        fr = load_run(t)
        if len(fr) != len(folds):
            log(f"  skip {t}: {len(fr)}/{len(folds)} folds")
            continue
        oof, test, _ = run_logprobs(fr)
        runs.append((oof, test))
        names.append(t)

    def dec(lps, tau, beta, test=False):
        if test:
            return {d: decode_one(v, tau, beta, st_all) for d, v in lps.items()}
        return {d: decode_one(v, tau, beta, st_fold[fold_of[d]]) for d, v in lps.items()}

    log("=== members (OOF macro-F1, argmax) ===")
    single = []
    for n, (oof, _) in zip(names, runs):
        f = macro(dec(oof, 0, 0), docs, labelled)
        single.append(f)
        log(f"  {n:32s} {f:.4f}")

    weights = None
    if a.greedy:
        # Caruana et al. 2004: forward selection with replacement on OOF.
        bag, cur = [], -1
        for _ in range(a.greedy):
            best = None
            for j in range(len(runs)):
                trial = bag + [j]
                w = [trial.count(i) for i in range(len(runs))]
                o, _ = combine(runs, w)
                f = macro(dec(o, a.tau or 0, a.beta or 0), docs, labelled)
                if best is None or f > best[0]:
                    best = (f, j)
            if best[0] <= cur and bag:
                break
            cur = best[0]
            bag.append(best[1])
            log(f"  greedy +{names[best[1]]:28s} -> {cur:.4f}")
        weights = [bag.count(i) for i in range(len(runs))]
        log("  weights: " + ", ".join(f"{n}={w}" for n, w in zip(names, weights) if w))
    oof, test = combine(runs, weights)

    log("\n=== decoding grid on pooled OOF (rows tau, cols beta) | worst fold in () ===")
    taus = [a.tau] if a.tau is not None else [0.0, 0.25, 0.5, 0.75, 1.0]
    betas = [a.beta] if a.beta is not None else [0.0, 0.25, 0.5, 1.0]
    grid = {}
    log("        " + "".join(f"{b:>17.2f}" for b in betas))
    for t in taus:
        row = []
        for b in betas:
            p = dec(oof, t, b)
            f = macro(p, docs, labelled)
            wf = min(macro(p, docs, f_) for f_ in folds)
            grid[(t, b)] = (f, wf)
            row.append(f"{f:.4f} ({wf:.4f})")
        log(f"  {t:5.2f} " + "".join(f"{x:>17s}" for x in row))
    tb = max(grid, key=lambda k: grid[k][0])
    log(f"  best pooled: tau={tb[0]} beta={tb[1]} -> {grid[tb][0]:.4f} (worst fold {grid[tb][1]:.4f})")
    t, b = (a.tau, a.beta) if a.tau is not None and a.beta is not None else tb

    p = dec(oof, t, b)
    res, yt, yp = score(p, docs, labelled)
    log(f"\n=== ensemble of {len(runs)} at tau={t} beta={b} ===")
    report(res, LABELS, "OOF", yt, yp)
    for dom in DOMAINS:
        ids = [d for d in labelled if docs[d]["domain"] == dom]
        log(f"  {dom}: {macro(p, docs, ids):.4f}")

    if a.dump_teacher:
        teacher = {d: v.clone() for d, v in oof.items()}
        teacher.update({d: v.clone() for d, v in test.items()})
        torch.save(teacher, a.dump_teacher)
        log(f"  teacher logits for {len(teacher)} docs -> {a.dump_teacher}")
    if a.submit:
        test_ids = sorted(test)
        tp = dec(test, t, b, test=True)
        write_submission(tp, {d: docs[d] for d in test_ids}, LABELS, a.tag, cv=res["macro_f1"],
                         note=f"ens {names} w={weights} tau={t} beta={b}")


def cmd_compare(a):
    """Paired bootstrap over documents: how often does A beat B on resampled sets?"""
    docs, _ = load_all()
    split = json.load(open(f"{P3}/split.json"))
    labelled = sorted(d for f in split["folds"] for d in f)
    preds = []
    for t in (a.a, a.b):
        oof, _, _ = run_logprobs(load_run(t))
        preds.append({d: [LABELS[i] for i in v.argmax(-1).tolist()] for d, v in oof.items()})
    rng = random.Random(0)
    base = [macro(p, docs, labelled) for p in preds]
    wins, diffs = 0, []
    # resample 10-document sets: the size of the real test set
    for _ in range(a.n):
        ids = [rng.choice(labelled) for _ in range(a.size)]
        fa, fb = macro(preds[0], docs, ids), macro(preds[1], docs, ids)
        diffs.append(fa - fb)
        wins += fa > fb
    diffs.sort()
    log(f"  {a.a} {base[0]:.4f} vs {a.b} {base[1]:.4f} (pooled OOF, diff {base[0]-base[1]:+.4f})")
    log(f"  on {a.size}-document resamples: A wins {100*wins/a.n:.0f}%, "
        f"diff 90% interval [{diffs[int(.05*a.n)]:+.4f}, {diffs[int(.95*a.n)]:+.4f}]")


def write_status():
    """STATUS.txt on Drive: the run table + tracker, readable from a phone."""
    import contextlib
    import io
    import time
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        cmd_table(None)
        p = f"{P3}/submissions/tracker.json"
        if os.path.exists(p):
            print("")
            print("submissions (zip | CV | leaderboard):")
            for t in json.load(open(p)):
                print(f"  {t['zip']:44s} {t['cv'] or 0:.4f}  {t['test'] if t.get('test') else '-'}")
    with open(f"{P3}/STATUS.txt", "w", encoding="utf-8") as f:
        f.write(time.strftime("updated %Y-%m-%d %H:%M:%S UTC", time.gmtime()) + "\n\n" + buf.getvalue())


def cmd_table(a):
    rows = []
    for m in sorted(glob.glob(f"{P3}/results/*/meta.json")):
        rows.append(json.load(open(m)))
    rows.sort(key=lambda r: -r["cv_macro_f1"])
    log(f"{'tag':30s} {'model':12s} {'method':8s} {'enc':6s} {'seq':4s} {'crf':3s} "
        f"{'train M':>8s} {'CV':>7s} {'CL':>7s} {'IT':>7s} {'worst':>7s} {'GB':>5s} {'sent/s':>7s} {'min':>5s}")
    for r in rows:
        g = r["args"]
        log(f"{r['tag'][:30]:30s} {g['model'][:12]:12s} {g['method']:8s} {g['enc']:6s} {g['seq']:4s} "
            f"{'y' if g['crf'] else '-':3s} {r['n_trainable']/1e6:8.3f} {r['cv_macro_f1']:7.4f} "
            f"{r['cv_by_domain']['CL']:7.4f} {r['cv_by_domain']['IT']:7.4f} {min(r['fold_f1']):7.4f} "
            f"{r['peak_gb']:5.1f} {r['test_sent_per_s'] or 0:7.0f} {r['train_s']/60:5.1f}")


def cmd_partial(a):
    """Submission from the folds of a run that have finished so far.

    Each fold file already holds that fold model's test logits at its best
    epoch, so a run interrupted after k folds can be submitted as a k-model
    average without waiting for the rest.
    """
    docs, _ = load_all()
    fr = load_run(a.tag)
    assert fr, f"no finished folds for {a.tag}"
    _, test, _ = run_logprobs(fr)
    test_ids = sorted(d for d, r in docs.items() if r["split"] == "test")
    assert set(test) == set(test_ids), "fold files lack test logits"
    pred = {d: [LABELS[i] for i in test[d].argmax(-1).tolist()] for d in test_ids}
    f1 = [r["best_f1"] for r in fr]
    log(f"  {a.tag}: {len(fr)} finished folds {[r['fold'] for r in fr]}, fold macro-F1 "
        + " ".join(f"{x:.4f}" for x in f1))
    # per-label picture on the documents those folds held out
    oof, _, _ = run_logprobs(fr)
    ids = sorted(oof)
    op = {d: [LABELS[i] for i in oof[d].argmax(-1).tolist()] for d in ids}
    res, yt, yp = score(op, docs, ids)
    report(res, LABELS, f"OOF over {len(ids)} held-out docs", yt, yp)
    write_submission(pred, {d: docs[d] for d in test_ids}, LABELS, f"{a.tag}_{len(fr)}fold",
                     cv=sum(f1) / len(f1), note=f"{len(fr)}-fold partial average")


def cmd_track(a):
    p = f"{P3}/submissions/tracker.json"
    tr = json.load(open(p))
    hit = [t for t in tr if t["tag"] == a.tag]
    assert hit, f"{a.tag} not in tracker"
    hit[-1]["test"] = a.test
    json.dump(tr, open(p, "w"), indent=1)
    log(f"{'tag':32s} {'CV':>7s} {'test':>7s} {'test/CV':>8s}")
    for t in tr:
        r = (t["test"] / t["cv"]) if t.get("test") and t.get("cv") else None
        log(f"{t['tag'][:32]:32s} {t['cv'] or 0:7.4f} {t['test'] or 0:7.4f} "
            f"{'' if r is None else f'{r:8.3f}'}")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("ens")
    e.add_argument("--runs", nargs="+", required=True, help="tags, or 'all' = every finished run")
    e.add_argument("--exclude", nargs="*", default=["ctx0", "band"],
                   help="tag prefixes left out of --runs all (ablations)")
    e.add_argument("--tau", type=float, default=None)
    e.add_argument("--beta", type=float, default=None)
    e.add_argument("--greedy", type=int, default=0)
    e.add_argument("--dump-teacher", default=None)
    e.add_argument("--submit", action="store_true")
    e.add_argument("--tag", default=None)
    c = sp.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--n", type=int, default=2000)
    c.add_argument("--size", type=int, default=10)
    sp.add_parser("table")
    sp.add_parser("status")
    pp = sp.add_parser("partial")
    pp.add_argument("--tag", required=True)
    t = sp.add_parser("track")
    t.add_argument("--tag", required=True)
    t.add_argument("--test", type=float, required=True)
    a = ap.parse_args()
    if a.cmd == "ens":
        assert not a.submit or a.tag, "--submit needs --tag"
        cmd_ens(a)
    elif a.cmd == "compare":
        cmd_compare(a)
    elif a.cmd == "table":
        cmd_table(a)
    elif a.cmd == "partial":
        cmd_partial(a)
    elif a.cmd == "status":
        write_status()
        print(open(f"{P3}/STATUS.txt", encoding="utf-8").read())
    else:
        cmd_track(a)


if __name__ == "__main__":
    main()
