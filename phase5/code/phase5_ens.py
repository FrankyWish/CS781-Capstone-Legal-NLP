#!/usr/bin/env python3
"""CS-781 Phase 5 -- offline ensembling, thresholds, zips, KD teacher, status. CPU only.

    python phase5_ens.py zip    --run lex_w12 --strategy global   --tag lex_global
    python phase5_ens.py zip    --run lex_w12 --strategy auto     --tag lex_perlabel
    python phase5_ens.py ens    --runs m4_lorma,m1_adapter,lex_w12 --greedy 12 --tag ens_hyb --submit
    python phase5_ens.py ens    --runs m4_lorma,m1_adapter --tag ens_model --submit --teacher $P5/teacher.npz
    python phase5_ens.py status                     # results table -> results/STATUS.md

Ensembling averages PROBABILITIES (Phase 3: stacking over CV outputs failed on
calibration shift; plain averaging transferred). --combine z instead averages
per-statute z-scored logits, which equalises models whose scales differ (the
LR model's balanced weights inflate its probabilities). Greedy forward selection
with replacement (Caruana et al. 2004) picks the weights on dev; the reported
number for the final choice is the 2-fold dev CV, which pays for the selection.

Threshold strategies (phase5_data.threshold_report): one global threshold, or
per-statute thresholds shrunk toward it by `shrink` pseudo-documents. `auto`
takes whichever has the best 2-fold CV on dev; ties go to global.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from phase5_data import (P5, load, label_space, multi_hot, score_on, decode, tune_global,  # noqa: E402
                         threshold_report, write_submission, log)


def sig(x):
    return 1 / (1 + np.exp(-x))


def load_run(tag):
    rd = f"{P5}/results/{tag}"
    meta = json.load(open(f"{rd}/meta.json"))
    dv_ids = json.load(open(f"{rd}/dev_ids.json"))
    te_ids = json.load(open(f"{rd}/test_ids.json"))
    S = dict(dev=np.load(f"{rd}/dev_logits.npy").astype(np.float32),
             test=np.load(f"{rd}/test_logits.npy").astype(np.float32))
    return meta, dict(dev=dv_ids, test=te_ids), S


def aligned(tags):
    """{tag: {'dev': (N, n), 'test': (M, n)}} in file order of dev/test, + metas."""
    dv, te = load("dev"), load("test")
    order = dict(dev=[r["id"] for r in dv], test=[r["id"] for r in te])
    out, metas = {}, {}
    for t in tags:
        meta, ids, S = load_run(t)
        metas[t] = meta
        out[t] = {}
        for s in ("dev", "test"):
            pos = {d: i for i, d in enumerate(ids[s])}
            miss = [d for d in order[s] if d not in pos]
            assert not miss, f"{t}: {len(miss)} {s} docs missing"
            out[t][s] = S[s][[pos[d] for d in order[s]]]
    return out, metas


def transform(S, how, ref=None):
    if how == "prob":
        return sig(S)
    if how == "logit":
        return S
    if how == "z":
        R = ref if ref is not None else S
        mu, sd = R.mean(0, keepdims=True), R.std(0, keepdims=True) + 1e-6
        return (S - mu) / sd
    raise ValueError(how)


def combine(runs, weights, how):
    """weighted mean of transformed scores, dev and test."""
    tot = sum(weights.values())
    out = {}
    for s in ("dev", "test"):
        acc = None
        for t, w in weights.items():
            ref = np.concatenate([runs[t]["dev"], runs[t]["test"]]) if how == "z" else None
            x = transform(runs[t][s], how, ref) * (w / tot)
            acc = x if acc is None else acc + x
        out[s] = acc
    return out


def to_logit(S, how):
    if how == "prob":
        p = np.clip(S, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))
    return S


def greedy(runs, Ydv, how, rounds, start=None):
    """Caruana forward selection with replacement on dev Macro-F1 (global threshold)."""
    w = dict(start or {})
    if not w:
        best_t = max(runs, key=lambda t: tune_global(transform(runs[t]["dev"], how), Ydv)[1])
        w = {best_t: 1}
    cur = tune_global(combine(runs, w, how)["dev"], Ydv)[1]
    log(f"  greedy start {w}: {cur:.4f}")
    for r in range(rounds):
        cand = []
        for t in runs:
            w2 = dict(w)
            w2[t] = w2.get(t, 0) + 1
            cand.append((tune_global(combine(runs, w2, how)["dev"], Ydv)[1], t))
        m, t = max(cand)
        if m <= cur + 1e-5:
            log(f"  greedy round {r+1}: no gain (best +{t} {m:.4f}), stop")
            break
        w[t] = w.get(t, 0) + 1
        cur = m
        log(f"  greedy round {r+1}: +{t} -> {m:.4f}  {w}")
    return w


def finish(tag, S, Ydv, args, hybrid, note, strategy="auto", members=None, save_run=True):
    """Threshold choice, dev numbers, results/<tag> as a run, optional zip + teacher."""
    th = threshold_report(S["dev"], Ydv)
    key = th["best"] if strategy == "auto" else strategy
    if strategy == "auto" and th["cv"]["global"] >= th["cv"][key] - 0.002:
        key = "global"            # per-label must beat global by a margin to be trusted
    thr = th["all_thr"][key]
    Pdv = decode(S["dev"], thr, min_k=0 if key == "global_nomin" else 1)
    dev = score_on(Ydv, Pdv)
    dev["cv_macro_f1"] = th["cv"][key]
    dev["strategy"] = key
    log(f"  {tag}: strategy {key}: dev {json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in dev.items()})}")
    if not save_run:            # a zip of one existing run: never touch that run's results/
        if getattr(args, "submit", False):
            te = load("test")
            Pte = decode(S["test"], thr, min_k=0 if key == "global_nomin" else 1)
            n, _ = label_space()
            write_submission(tag, te, Pte, note=note, hybrid=hybrid, dev_scores=dev,
                             max_label=min(n - 1, args.max_label if args.max_label is not None else n - 1))
            np.save(f"{P5}/submissions/{tag}/thr.npy", thr)
        return dev
    rd = f"{P5}/results/{tag}"
    os.makedirs(rd, exist_ok=True)
    np.save(f"{rd}/dev_logits.npy", S["dev"].astype(np.float16))
    np.save(f"{rd}/test_logits.npy", S["test"].astype(np.float16))
    json.dump([r["id"] for r in load("dev")], open(f"{rd}/dev_ids.json", "w"))
    json.dump([r["id"] for r in load("test")], open(f"{rd}/test_ids.json", "w"))
    meta = dict(tag=tag, kind="ensemble", hybrid=hybrid, note=note, members=members or {},
                dev_best=dev, thr_cv=th["cv"], thr_best=key, combine=getattr(args, "combine", None))
    json.dump(meta, open(f"{rd}/meta.json", "w"), indent=1)
    np.save(f"{rd}/thr.npy", thr)
    if getattr(args, "submit", False):
        te = load("test")
        Pte = decode(S["test"], thr, min_k=0 if key == "global_nomin" else 1)
        n, _ = label_space()
        write_submission(tag, te, Pte, note=note, hybrid=hybrid, dev_scores=dev,
                         max_label=min(n - 1, args.max_label if args.max_label is not None else n - 1))
    return dev


def cmd_zip(a):
    runs, metas = aligned([a.run])
    Ydv = multi_hot(load("dev"), label_space()[0])
    S = {s: runs[a.run][s] for s in ("dev", "test")}
    m = metas[a.run]
    a.submit = True
    finish(a.tag, S, Ydv, a, m.get("hybrid", False), f"{a.run} {m.get('method', m.get('kind'))}",
           a.strategy, {a.run: 1}, save_run=False)


def cmd_ens(a):
    tags = [t for t in a.runs.split(",") if t]
    runs, metas = aligned(tags)
    Ydv = multi_hot(load("dev"), label_space()[0])
    for t in tags:
        m = tune_global(transform(runs[t]["dev"], a.combine), Ydv)[1]
        log(f"  member {t:24s} dev macro-F1 (global thr) {m:.4f}")
    if a.weights:
        w = {t: float(x) for t, x in zip(tags, a.weights.split(","))}
    elif a.greedy:
        w = greedy(runs, Ydv, a.combine, a.greedy)
    else:
        w = {t: 1.0 for t in tags}
    S = combine(runs, w, a.combine)
    hybrid = any(metas[t].get("hybrid") for t in w)
    dev = finish(a.tag, S, Ydv, a, hybrid, f"ens({a.combine}) " + ",".join(f"{t}x{v:g}" for t, v in w.items()),
                 a.strategy, w)
    if a.teacher:
        dv, te = load("dev"), load("test")
        ids = [r["id"] for r in dv] + [r["id"] for r in te]
        L = np.concatenate([to_logit(S["dev"], a.combine), to_logit(S["test"], a.combine)]).astype(np.float32)
        if a.combine == "z":
            log("  NOTE: z-combined teacher logits are not log-odds; KD temperature acts on z units")
        np.savez(a.teacher, ids=np.array(ids), logits=L)
        log(f"  teacher -> {a.teacher}: {len(ids)} docs (dev+test)")
    return dev


def cmd_status(a):
    rows = []
    for mp in sorted(glob.glob(f"{P5}/results/*/meta.json")):
        m = json.load(open(mp))
        d = m.get("dev_best", {})
        g = m.get("dev_global", {})
        rows.append((d.get("macro_f1", 0), m["tag"], m.get("kind"), m.get("model", ""), m.get("method", ""),
                     g.get("macro_f1"), d.get("macro_f1"), d.get("micro_f1"), m.get("thr_best"),
                     (m.get("thr_cv") or {}).get(m.get("thr_best") or "", None),
                     "HYB" if m.get("hybrid") else "model", m.get("train_min"), m.get("train_tok_s")))
    rows.sort(reverse=True)
    f = lambda x: f"{x:.4f}" if isinstance(x, float) else ("" if x is None else str(x))
    lines = ["| run | kind | trunk | method | dev macro (global) | dev macro (chosen) | dev micro | thr | CV | type | train min | tok/s |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append("| " + " | ".join(f(x) if not isinstance(x, float) or i not in (11, 12) else f"{x:.0f}"
                                       for i, x in enumerate(r[1:], 1)) + " |")
    subs = []
    for ip in sorted(glob.glob(f"{P5}/submissions/*/info.json")):
        s = json.load(open(ip))
        subs.append(f"| {s['tag']} | {'HYBRID' if s['hybrid'] else 'model-only'} | "
                    f"{s['dev'].get('macro_f1', 0):.4f} | {s['dev'].get('cv_macro_f1', 0):.4f} | "
                    f"{s['dev'].get('micro_f1', 0):.4f} | {s['labels_per_doc']:.2f} | {s['note']} |")
    txt = "## Runs (dev)\n\n" + "\n".join(lines) + "\n\n## Zips\n\n| zip | type | dev macro | dev CV macro | dev micro | labels/doc | what |\n|---|---|---|---|---|---|---|\n" + "\n".join(subs) + "\n"
    with open(f"{P5}/results/STATUS.md", "w") as fh:
        fh.write(txt)
    print(txt)


REQUIRED = {  # zip tag -> what it covers for grading (spec 11.2)
    "m4_lorma": "LoRMA (compulsory) / M4", "m1_adapter": "Adapters / M1", "m2_partial": "Partial FT / M2",
    "m3_qlora": "QLoRA / M3", "m1_prefix": "Prefix tuning / M1", "kd_m4half": "Knowledge distillation (compulsory)",
    "m4_full": "Full FT / M4", "m3_adalora": "AdaLoRA / M3", "m2_prompt": "Prompt tuning / M2",
}


def cmd_plan(a):
    """submissions/*/info.json -> SUBMISSION_PLAN.md: best expected first, then the graded set."""
    subs = [json.load(open(p)) for p in glob.glob(f"{P5}/submissions/*/info.json")]
    for s in subs:
        s["exp"] = s["dev"].get("cv_macro_f1", s["dev"].get("macro_f1", 0))
    best = sorted(subs, key=lambda s: -s["exp"])
    order, seen = [], set()
    # 1-2: best hybrid and best model-only (the leaderboard keeps the max; the report needs both)
    for s in best:
        if s["tag"].startswith("top_"):
            REQUIRED.setdefault(s["tag"], "best single model, longer training (goal 2)" + (", gold+silver" if s["tag"].endswith("_sil") else ""))
    for pick in ([s for s in best if s["hybrid"]][:1] + [s for s in best if not s["hybrid"]][:1]):
        order.append(pick)
        seen.add(pick["tag"])
    # then every graded technique run, best first, then the rest
    for s in best:
        if s["tag"] in REQUIRED and s["tag"] not in seen:
            order.append(s)
            seen.add(s["tag"])
    for s in best:
        if s["tag"] not in seen:
            order.append(s)
            seen.add(s["tag"])
    lines = ["| # | zip | type | dev Macro-F1 (thr fit on dev) | dev Macro-F1, 2-fold CV | dev Micro-F1 | "
             "statutes/doc | covers | upload day |", "|---|---|---|---|---|---|---|---|---|"]
    days = ["Oct 22", "Oct 22", "Oct 22", "Oct 22", "Oct 22", "Oct 23", "Oct 23", "Oct 23", "Oct 23", "Oct 23"]
    for i, s in enumerate(order):
        lines.append(f"| {i+1} | `submissions/{s['tag']}/submission.zip` | {'HYBRID' if s['hybrid'] else 'model-only'} | "
                     f"{s['dev'].get('macro_f1', 0):.4f} | {s['exp']:.4f} | {s['dev'].get('micro_f1', 0):.4f} | "
                     f"{s['labels_per_doc']:.2f} | {REQUIRED.get(s['tag'], s['note'])} | "
                     f"{days[i] if i < len(days) else 'Oct 24+'} |")
    txt = "\n".join(lines) + "\n"
    with open(f"{P5}/results/PLAN_TABLE.md", "w") as fh:
        fh.write(txt)
    print(txt)


def cmd_rate(a):
    """Volume variants of a finished run/ensemble: shift all thresholds by one scalar so the TEST set gets
    factor x as many statutes per document, and zip each (Phase 2: the leaderboard peaked at a different
    volume than CV chose). Dev Macro-F1 at the shifted thresholds is reported so the cost is visible."""
    runs, metas = aligned([a.run])
    dv, te = load("dev"), load("test")
    n, _ = label_space()
    Ydv = multi_hot(dv, n)
    Sdv, Ste = runs[a.run]["dev"], runs[a.run]["test"]
    tp = next((p for p in (f"{P5}/results/{a.run}/thr.npy", f"{P5}/submissions/{a.run}/thr.npy")
               if os.path.exists(p)), None)
    thr = np.load(tp) if tp else np.full(n, tune_global(Sdv, Ydv)[0], dtype=np.float32)
    base = decode(Ste, thr).sum(1).mean()
    span = float(np.std(np.concatenate([Sdv, Ste]))) * 3 + 1e-3
    log(f"  {a.run}: base {base:.2f} statutes/doc on test, dev Macro-F1 {score_on(Ydv, decode(Sdv, thr))['macro_f1']:.4f}")
    for f in [float(x) for x in a.factors.split(",")]:
        lo, hi = -span, span                      # +delta raises thresholds -> fewer labels
        for _ in range(40):
            mid = (lo + hi) / 2
            v = decode(Ste, thr + mid).sum(1).mean()
            if v > f * base:
                lo = mid
            else:
                hi = mid
        d = (lo + hi) / 2
        Pte, Pdv = decode(Ste, thr + d), decode(Sdv, thr + d)
        dev = score_on(Ydv, Pdv)
        dev["cv_macro_f1"] = dev["macro_f1"]
        tag = f"{a.run}_v{int(round(f * 100))}"
        log(f"  {tag}: {Pte.sum(1).mean():.2f} statutes/doc on test | dev Macro-F1 {dev['macro_f1']:.4f}")
        write_submission(tag, te, Pte, note=f"{a.run} at x{f:g} test volume (leaderboard calibration)",
                         hybrid=bool(metas[a.run].get("hybrid")), dev_scores=dev)


def cmd_analyze(a):
    """Where does Macro-F1 come from? Per-statute F1 on dev by train-frequency bucket, the
    statutes stuck at 0, labels/doc predicted vs gold, and the most confused statute pairs."""
    from phase5_data import f1_per_label
    runs, metas = aligned([a.run])
    dv = load("dev")
    n, cnt = label_space()
    Y = multi_hot(dv, n)
    S = runs[a.run]["dev"]
    tp = next((p for p in (f"{P5}/results/{a.run}/thr.npy", f"{P5}/submissions/{a.run}/thr.npy")
               if os.path.exists(p)), None)
    thr = np.load(tp) if tp else tune_global(S, Y)[0]
    P = decode(S, thr)
    w = int(np.nonzero(Y.any(0))[0].max()) + 1
    f1 = f1_per_label(Y[:, :w], P[:, :w])
    log(f"  {a.run}: dev Macro-F1 {f1.mean():.4f} over {w} statutes; gold labels/doc {Y.sum(1).mean():.2f}, "
        f"predicted {P.sum(1).mean():.2f}")
    edges = [0, 50, 100, 250, 500, 1000, 3000, 10**9]
    log("  train docs  | statutes | mean F1 | share of the macro deficit")
    deficit = (1 - f1).sum()
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (cnt[:w] >= lo) & (cnt[:w] < hi)
        if m.any():
            log(f"  {lo:>5}-{hi if hi < 10**9 else 'inf':<5} | {int(m.sum()):>8} | {f1[m].mean():.3f}   | "
                f"{(1 - f1[m]).sum() / deficit:.1%}")
    zero = [int(l) for l in np.nonzero(f1 == 0)[0]]
    log(f"  statutes at F1 = 0 ({len(zero)}): " + " ".join(f"{l}(train {cnt[l]}, dev {int(Y[:, l].sum())})" for l in zero))
    log(f"  each statute is worth {1/w:.4f} Macro-F1; lifting every F1 < 0.2 statute to 0.4 would add "
        f"{np.maximum(0, 0.4 - f1[f1 < 0.2]).sum() / w:+.4f}")
    # confusions: predicted l (false positive) where gold had m but not l
    fp = (P[:, :w] == 1) & (Y[:, :w] == 0)
    fn = (P[:, :w] == 0) & (Y[:, :w] == 1)
    C = fp.T.astype(np.int64) @ fn.astype(np.int64)
    pairs = sorted(((int(C[i, j]), i, j) for i in range(w) for j in range(w) if i != j and C[i, j]), reverse=True)[:12]
    log("  most confused (predicted -> missed gold, docs): " + "  ".join(f"{i}->{j}:{c}" for c, i, j in pairs))
    json.dump(dict(run=a.run, per_label_f1=f1.round(4).tolist(), train_count=cnt[:w].tolist(), zero=zero,
                   confusions=pairs), open(f"{P5}/results/{a.run}/analysis.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    z = sp.add_parser("zip")
    z.add_argument("--run", required=True)
    z.add_argument("--tag", required=True)
    z.add_argument("--strategy", default="auto")
    z.add_argument("--max-label", type=int, default=None)
    e = sp.add_parser("ens")
    e.add_argument("--runs", required=True)
    e.add_argument("--weights", default=None)
    e.add_argument("--greedy", type=int, default=0)
    e.add_argument("--combine", default="prob", choices=["prob", "logit", "z"])
    e.add_argument("--strategy", default="auto")
    e.add_argument("--tag", required=True)
    e.add_argument("--submit", action="store_true")
    e.add_argument("--teacher", default=None)
    e.add_argument("--max-label", type=int, default=None)
    sp.add_parser("status")
    sp.add_parser("plan")
    rt = sp.add_parser("rate")
    rt.add_argument("--run", required=True)
    rt.add_argument("--factors", default="0.85,1.15")
    an = sp.add_parser("analyze")
    an.add_argument("--run", required=True)
    a = ap.parse_args()
    {"zip": cmd_zip, "ens": cmd_ens, "status": cmd_status, "plan": cmd_plan, "analyze": cmd_analyze, "rate": cmd_rate}[a.cmd](a)


if __name__ == "__main__":
    main()
