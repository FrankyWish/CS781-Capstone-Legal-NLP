#!/usr/bin/env python3
"""CS-781 Phase 4 -- offline ensembling, thresholds, KD teacher, status. CPU only.

    python phase4_ens.py status
    python phase4_ens.py ens --runs all --kind model --greedy 8 --tag ens_model
    python phase4_ens.py ens --runs m4_lorma m2_full lex_tail --tag ens_hyb_3
    python phase4_ens.py ens --runs m2_full m4_lorma --teacher        # writes results/<tag>/teacher.pt
    python phase4_ens.py zip --run m4_lorma --rate                    # one run, rate-matched threshold

Every run is put on a common scale before averaging: z' = (z - thr) / sd_dev,
so each member's own dev-tuned decision boundary sits at 0 and a confident
model does not drown the others (the Phase-3 stacker failed on exactly this
kind of calibration difference). The averaged score is re-thresholded on dev.

--rate replaces the dev-tuned threshold with one that gives the test set the
dev set's predicted-positive rate (Phase 2: calibrating output volume moved the
leaderboard more than the CV-optimal cut did).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from phase4_data import (P4, log, summarise, best_threshold, rate_threshold, save_run,  # noqa: E402
                         write_submission, load_run, metrics, token_cache, atomic_json)

REQUIRED = ["full", "partial", "unfreeze", "bitfit", "lorma", "adalora", "adapter"]
ARCHS = ["M1_SC", "M2_LexArcED", "M3_LexArcL", "M4_LexArcP"]


def all_runs():
    return sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(f"{P4}/results/*/meta.json"))


def dev_labels():
    if os.path.exists(f"{P4}/cache/tok_dev.pt"):
        dv = token_cache("dev")
        return {d: y for d, y in zip(dv["ids"], dv["label"])}
    from phase4_data import load_split
    return {d: y for d, _, y in load_split("dev")}


def scaled(tag, ydev):
    d, t, meta = load_run(tag)
    zd = d["z"].float()
    y = [ydev[i] for i in d["ids"]]
    thr, _ = best_threshold(y, zd.tolist())
    sd = float(zd.std()) or 1.0
    return (dict(zip(d["ids"], ((zd - thr) / sd).tolist())),
            dict(zip(t["ids"], ((t["z"].float() - thr) / sd).tolist())), meta)


def combine(tags, ydev, weights=None):
    S = [scaled(t, ydev) for t in tags]
    w = weights or [1.0] * len(tags)
    ids_d = list(S[0][0])
    ids_t = list(S[0][1])
    zd = [sum(wi * s[0][i] for wi, s in zip(w, S)) / sum(w) for i in ids_d]
    zt = [sum(wi * s[1][i] for wi, s in zip(w, S)) / sum(w) for i in ids_t]
    return ids_d, zd, ids_t, zt, [s[2] for s in S]


def dev_f1(ids_d, zd, ydev):
    y = [ydev[i] for i in ids_d]
    return best_threshold(y, zd)[1]


def cmd_ens(a):
    ydev = dev_labels()
    runs = all_runs() if a.runs == ["all"] else a.runs
    runs = [r for r in runs if not r.startswith("ens_") and not any(x in r for x in a.exclude)]
    if a.kind == "model":
        runs = [r for r in runs if json.load(open(f"{P4}/results/{r}/meta.json")).get("kind") == "model"]
    if a.teacher:
        runs = [r for r in runs if os.path.exists(f"{P4}/results/{r}/kdpool.pt")]
    assert runs, "no runs to ensemble"
    if a.greedy:
        # forward selection with replacement on dev Macro-F1 at the tuned threshold
        single = sorted(((dev_f1(*combine([r], ydev)[:2], ydev), r) for r in runs), reverse=True)
        log("  singles: " + ", ".join(f"{r} {f:.4f}" for f, r in single))
        chosen = [single[0][1]]
        cur = single[0][0]
        for _ in range(a.greedy - 1):
            cand = []
            for r in runs:
                f = dev_f1(*combine(chosen + [r], ydev)[:2], ydev)
                cand.append((f, r))
            f, r = max(cand)
            if f <= cur + a.min_gain:
                break
            chosen.append(r)
            cur = f
            log(f"  + {r}: dev {f:.4f}")
        runs = chosen
    w = None
    if a.weights:
        assert len(a.weights) == len(runs), "--weights needs one number per --runs entry"
        w = a.weights
    ids_d, zd, ids_t, zt, metas = combine(runs, ydev, w)
    y = [ydev[i] for i in ids_d]
    s = summarise(y, zd)
    thr = s["thr"]
    if a.rate:
        thr = rate_threshold(zt, s["pos_rate"])
    kinds = {m.get("kind", "model") for m in metas}
    kind = "model" if kinds == {"model"} else "hybrid"
    from collections import Counter
    log(f"  {a.tag}: {len(runs)} members {dict(Counter(runs))} | dev macro-F1 {s['f1_tuned']:.4f} "
        f"(thr {s['thr']:+.3f}) AUC {s['auc']:.4f} | {kind}")
    meta = dict(tag=a.tag, members=runs, weights=w, dev_f1=s["f1_tuned"], dev_auc=s["auc"], thr=thr,
                dev_p=s["precision"], dev_r=s["recall"], kind=kind, technique="ensemble",
                arch="+".join(sorted({m.get("arch", "?") for m in metas})), rate=a.rate)
    extra = {}
    if a.teacher:
        kp = None
        for r in runs:
            k = torch.load(f"{P4}/results/{r}/kdpool.pt", weights_only=False)
            d, _, _ = load_run(r)
            yd = [ydev[i] for i in d["ids"]]
            t0, _ = best_threshold(yd, d["z"].tolist())
            sd = float(d["z"].float().std()) or 1.0
            v = (k["z"].float() - t0) / sd
            kp = v if kp is None else kp + v
            kids = k["ids"]
        kp = kp / len(runs)
        # the teacher's z must be a calibrated logit for sigmoid(z/T): rescale the
        # averaged score so that its dev logistic fit is the identity
        import math
        zz = torch.tensor(zd)
        yy = torch.tensor(y, dtype=torch.float32)
        A = torch.zeros(2, requires_grad=True)
        opt = torch.optim.LBFGS([A], max_iter=100)

        def closure():
            opt.zero_grad()
            l = torch.nn.functional.binary_cross_entropy_with_logits(A[0] * zz + A[1], yy)
            l.backward()
            return l
        opt.step(closure)
        sa, sb = float(A[0]), float(A[1])
        log(f"  teacher calibration on dev: logit = {sa:.3f} * z + {sb:+.3f}")
        tz = {i: sa * v + sb for i, v in zip(kids, kp.tolist())}
        tz.update({i: sa * v + sb for i, v in zip(ids_t, zt)})
        teach = dict(ids=list(tz), z=torch.tensor(list(tz.values())))
        os.makedirs(f"{P4}/results/{a.tag}", exist_ok=True)
        torch.save(teach, f"{P4}/teacher_{a.tag}.pt")
        log(f"  wrote {P4}/teacher_{a.tag}.pt ({len(tz)} docs: KD pool + test)")
    save_run(a.tag, ids_d, zd, ids_t, zt, meta, extra)
    if a.submit:
        pred = {i: int(v > thr) for i, v in zip(ids_t, zt)}
        write_submission(a.tag, ids_t, pred, dev=s["f1_tuned"], kind=kind,
                         note=f"ensemble of {len(runs)}: {' '.join(runs)}" + (f" weights {w}" if w else "")
                         + (" rate-matched" if a.rate else ""))
    write_status()


def cmd_zip(a):
    ydev = dev_labels()
    d, t, meta = load_run(a.run)
    y = [ydev[i] for i in d["ids"]]
    s = summarise(y, d["z"].tolist())
    thr = s["thr"]
    zt = t["z"].tolist()
    if a.rate:
        thr = rate_threshold(zt, s["pos_rate"])
    tag = a.tag or (a.run + ("_rate" if a.rate else ""))
    pred = {i: int(v > thr) for i, v in zip(t["ids"], zt)}
    write_submission(tag, t["ids"], pred, dev=s["f1_tuned"], kind=meta.get("kind", "model"),
                     note=f"{a.run} thr {thr:+.3f}" + (" rate-matched" if a.rate else ""))


def write_status():
    """STATUS.md on Drive: every run, its dev score, cost, and requirement coverage."""
    rows = []
    for r in all_runs():
        m = json.load(open(f"{P4}/results/{r}/meta.json"))
        rows.append(m)
    rows.sort(key=lambda m: -m.get("dev_f1", 0))
    L = ["| run | kind | technique | arch | dev F1 | AUC | P | R | trainable M | peak GB | train min | tok/s |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in rows:
        L.append(f"| {m['tag']} | {m.get('kind','?')} | {m.get('technique','?')} | {m.get('arch','?')} | "
                 f"{m.get('dev_f1',0):.4f} | {m.get('dev_auc',float('nan')):.4f} | {m.get('dev_p',0):.3f} | "
                 f"{m.get('dev_r',0):.3f} | {m.get('n_trainable',0)/1e6:.2f} | {m.get('peak_gb',0):.1f} | "
                 f"{m.get('train_s',0)/60:.1f} | {m.get('train_tok_per_s',0)/1e3:.1f}k |")
    model_runs = [m for m in rows if m.get("kind") == "model" and m.get("technique") != "ensemble"]
    tech = {m["technique"] for m in model_runs}
    tech |= {"kd" for m in model_runs if (m.get("args") or {}).get("kd")}
    arch = {m["arch"] for m in model_runs}
    L.append("")
    L.append(f"techniques covered: {sorted(tech)}; missing from the phase list: "
             f"{[t for t in REQUIRED + ['kd'] if t not in tech]}")
    L.append(f"architectures covered: {sorted(arch)}; missing: {[x for x in ARCHS if x not in arch]}")
    sub = f"{P4}/submissions/tracker.json"
    if os.path.exists(sub):
        tr = json.load(open(sub))
        L.append("")
        L.append("| zip | kind | dev F1 | test pos-rate | note |")
        L.append("|---|---|---|---|---|")
        for t in sorted(tr, key=lambda t: -(t.get("dev") or 0)):
            L.append(f"| {t['zip']} | {t['kind']} | {t.get('dev') or 0:.4f} | {t['pos_rate']:.3f} | {t['note']} |")
    with open(f"{P4}/STATUS.md", "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    return "\n".join(L)


def write_order(top=15):
    """SUBMIT_ORDER.txt: every distinct zip (by prediction hash), best dev first."""
    tr = json.load(open(f"{P4}/submissions/tracker.json"))
    seen, rows = set(), []
    for t in sorted(tr, key=lambda t: -(t.get("dev") or 0)):
        if t["sha"] in seen:
            continue                     # identical predictions never count twice on Codabench
        seen.add(t["sha"])
        rows.append(f"{len(rows)+1:2d}. {t['zip']:28s} {t['kind']:6s} dev {t.get('dev') or 0:.4f}  "
                    f"test pos-rate {t['pos_rate']:.3f}  {t['note']}")
    with open(f"{P4}/SUBMIT_ORDER.txt", "w") as f:
        f.write("\n".join(rows) + "\n")
    print("\n".join(rows[:top]))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("ens")
    e.add_argument("--runs", nargs="+", default=["all"])
    e.add_argument("--exclude", nargs="*", default=[])
    e.add_argument("--kind", choices=["any", "model"], default="any")
    e.add_argument("--greedy", type=int, default=0)
    e.add_argument("--weights", nargs="*", type=float, default=None, help="fixed weight per --runs entry")
    e.add_argument("--min-gain", type=float, default=0.0005)
    e.add_argument("--rate", action="store_true")
    e.add_argument("--teacher", action="store_true")
    e.add_argument("--no-submit", dest="submit", action="store_false")
    e.add_argument("--tag", required=True)
    z = sp.add_parser("zip")
    z.add_argument("--run", required=True)
    z.add_argument("--rate", action="store_true")
    z.add_argument("--tag", default=None)
    sp.add_parser("status")
    sp.add_parser("order")
    a = ap.parse_args()
    if a.cmd == "ens":
        if not a.tag.startswith("ens_"):
            a.tag = "ens_" + a.tag
        if a.teacher:
            a.submit = False             # a teacher is an input to KD, not a submission
        cmd_ens(a)
    elif a.cmd == "zip":
        cmd_zip(a)
    elif a.cmd == "order":
        write_order()
    else:
        print(write_status())


if __name__ == "__main__":
    main()
