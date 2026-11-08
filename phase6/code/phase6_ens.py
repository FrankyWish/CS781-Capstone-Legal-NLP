#!/usr/bin/env python3
"""CS-781 Phase 6 -- stage 3: pool every member's candidates, pick per document.

    python phase6_ens.py all        pool -> MBR (ROUGE utility, + BERTScore utility)
                                    -> dev-trained reranker with out-of-fold CV
                                    -> every zip -> SUBMIT_ORDER.md
    python phase6_ens.py zips       just the per-run zips (required coverage)

Phase 1's MBR over pooled candidates was worth +0.0068 BERTScore; Phase 5's
dev-trained stacker +0.0066.  Both are rebuilt here for generation:

  MBR      y* = argmax_y mean_c U(y, c) over a document's candidates.  The
           primary utility is ROUGE-1 F (no pretrained weights anywhere); the
           `_bert` variant uses BERTScore F1, i.e. roberta-large inside the
           decision rule, and is labelled so in info.json and the report.
  rerank   a HistGB regressor predicts each candidate's BERTScore against the
           (unseen) reference from reference-free features: which system, its
           centrality in the pool, its length against the target, how much of
           it is copied from the document, how repetitive it is.  Trained on the
           decoded dev subset, scored OUT-OF-FOLD by document (5 folds), then
           refit on all of it for test.  The OOF number is the honest one.

Kinds: a pool of neural runs only is `model` (choosing among the models' own
outputs uses no outside text); a pool that includes extractive members is
`hybrid`.  Length: neural members are trimmed from their long raw generations to
the calibrated per-document target, so the pool is length-consistent.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np

import phase6_data as D
import phase6_sel as S

P6 = D.P6
RES = f"{P6}/results"
REQUIRED = ["m4_lorma", "m2_lora", "m1_houlsby", "m3_qlora", "m2_adalora", "m3_pfeiffer",
            "kd_m1half"]
EXTRACTIVE = ["bsel_best", "win_best", "sel_best", "sel_short", "sel_long"]


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def jload(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def length_setting():
    f = f"{RES}/length_calib.json"
    if os.path.exists(f):
        c = jload(f)["chose"]
        return c["len"], float(c["mult"])
    return "model", 1.0


def trim(t, n):
    w = t.split()
    return " ".join(w[:n]) if len(w) > n else t


# --------------------------------------------------------------------- pool --
def neural_runs():
    out = []
    for m in sorted(glob.glob(f"{RES}/*/meta.json")):
        d = jload(m)
        if d.get("kind") == "model" and os.path.exists(f"{RES}/{d['tag']}/gen_{d['decode']}_test.json"):
            out.append(d)
    return out


# at a scaled length, an extractive member is replaced by the selection built nearest
# that length: shorter = trim the same selection, longer = the next-longer selection
LONGER = {"sel_short": "sel_best", "sel_best": "sel_long", "sel_long": "sel_long",
          "win_best": "win_best", "bsel_best": "bsel_best"}


def build_pool(scale=1.0, with_extractive=True):
    """{system: {"dev": {id: text}, "test": {id: text}}} on the decoded dev subset.

    System NAMES are the same at every scale, so a selector chosen at 1.0x can be
    replayed at 0.85x / 1.15x for the leaderboard length bracket."""
    how, mult = length_setting()
    dev_ids = set(D.gen_dev_ids(10 ** 9))
    w_dev = S.len_targets(D.load("dev"), how, mult * scale)
    w_test = S.len_targets(D.load("test"), how, mult * scale)
    pool, kinds = {}, {}
    for d in neural_runs():
        t, dc = d["tag"], d["decode"]
        gd = jload(f"{RES}/{t}/gen_{dc}_dev.json")
        gt = jload(f"{RES}/{t}/gen_{dc}_test.json")
        pool[t] = dict(dev={k: trim(v, w_dev[k]) for k, v in gd.items()},
                       test={k: trim(v, w_test[k]) for k, v in gt.items()})
        kinds[t] = "model"
    if with_extractive:
        for t in EXTRACTIVE:
            src = LONGER[t] if scale > 1.0 else t
            f = f"{RES}/{src}/summaries_test.json"
            if not os.path.exists(f) or not os.path.exists(f"{RES}/{t}/summaries_test.json"):
                continue
            dv, te = jload(f"{RES}/{src}/summaries_dev.json"), jload(f)
            if scale < 1.0:
                dv = {k: trim(v, int(len(v.split()) * scale)) for k, v in dv.items()}
                te = {k: trim(v, int(len(v.split()) * scale)) for k, v in te.items()}
            pool[t] = dict(dev=dv, test=te)
            kinds[t] = "extractive"
    # restrict dev to documents every member decoded
    common = set.intersection(*[set(v["dev"]) for v in pool.values()]) & dev_ids if pool else set()
    for v in pool.values():
        v["dev"] = {k: v["dev"][k] for k in sorted(common)}
    return pool, kinds, sorted(common)


# ---------------------------------------------------------------------- MBR --
def mbr(pool, systems, split, ids, util="rouge"):
    pick = {}
    if util == "bert":
        sc = D.bert()
    for k in ids:
        cands = [pool[s][split][k] for s in systems]
        m = len(cands)
        U = np.zeros((m, m))
        if util == "rouge":
            tk = [D.toks(c) for c in cands]
            for i in range(m):
                for j in range(i + 1, m):
                    U[i, j] = U[j, i] = D.rouge1_f(tk[i], tk[j])
        else:
            pr = [(i, j) for i in range(m) for j in range(i + 1, m)]
            f = sc.f1([cands[i] for i, _ in pr], [cands[j] for _, j in pr]) if pr else []
            for (i, j), v in zip(pr, f):
                U[i, j] = U[j, i] = v
        pick[k] = systems[int(np.argmax(U.sum(1)))]
    return pick


# ------------------------------------------------------------------- rerank --
def features(pool, systems, split, ids, docs):
    """Reference-free features per (doc, candidate)."""
    X, rows = [], []
    for k in ids:
        cands = [pool[s][split][k] for s in systems]
        tk = [D.toks(c) for c in cands]
        dt = D.toks(docs[k])
        m = len(cands)
        cen = np.zeros(m)
        for i in range(m):
            cen[i] = np.mean([D.rouge1_f(tk[i], tk[j]) for j in range(m) if j != i]) if m > 1 else 0
        for i, s in enumerate(systems):
            oh = [float(s == x) for x in systems]
            w = len(cands[i].split())
            X.append(oh + [cen[i], cen[i] - cen.max(), w, np.log1p(w),
                           D.copy_rate(tk[i], dt, 8), D.novel_rate(tk[i], dt, 1),
                           D.novel_rate(tk[i], dt, 2), repeat(cands[i])])
            rows.append((k, s))
    return np.asarray(X, dtype=np.float32), rows


def repeat(t, n=4):
    w = t.lower().split()
    g = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    return (len(g) - len(set(g))) / max(len(g), 1)


def targets(pool, systems, ids, refs, use_bert):
    pr = [pool[s]["dev"][k] for k in ids for s in systems]
    rf = [refs[k] for k in ids for _ in systems]
    if use_bert:
        return D.bert().f1(pr, rf)
    return np.array([D.rouge1_f(D.toks(p), D.toks(r)) for p, r in zip(pr, rf)])


def rerank(pool, systems, ids, refs, test_ids, use_bert, seed=D.SEED):
    from sklearn.ensemble import HistGradientBoostingRegressor
    docs_dev = {r["id"]: r["doc"] for r in D.load("dev")}
    docs_test = {r["id"]: r["doc"] for r in D.load("test")}
    X, rows = features(pool, systems, "dev", ids, docs_dev)
    y = targets(pool, systems, ids, refs, use_bert)
    m = len(systems)
    Y = y.reshape(len(ids), m)
    rng = np.random.default_rng(seed)
    fold = rng.permutation(len(ids)) % 5
    oof = np.zeros(len(ids) * m)
    mk = lambda: HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_depth=3,
                                               min_samples_leaf=10, random_state=seed)
    for f in range(5):
        tr = np.repeat(fold != f, m)
        te = np.repeat(fold == f, m)
        if tr.sum() < 20 or te.sum() == 0:
            continue
        oof[te] = mk().fit(X[tr], y[tr]).predict(X[te])
    choice = oof.reshape(len(ids), m).argmax(1)
    oof_score = float(Y[np.arange(len(ids)), choice].mean())
    model = mk().fit(X, y)
    Xt, _ = features(pool, systems, "test", test_ids, docs_test)
    ct = model.predict(Xt).reshape(len(test_ids), m).argmax(1)
    pick_test = {k: systems[int(c)] for k, c in zip(test_ids, ct)}
    pick_dev = {k: systems[int(c)] for k, c in zip(ids, choice)}
    per_sys = {s: float(Y[:, i].mean()) for i, s in enumerate(systems)}
    oracle = float(Y.max(1).mean())
    return pick_dev, pick_test, oof_score, per_sys, oracle


# --------------------------------------------------------------------- zips --
def write(tag, summaries_test, kind, note, dev_scores):
    d = f"{RES}/{tag}"
    os.makedirs(d, exist_ok=True)
    with open(f"{d}/summaries_test.json", "w", encoding="utf-8") as f:
        json.dump(summaries_test, f, ensure_ascii=False)
    with open(f"{d}/ens.json", "w") as f:
        json.dump(dict(tag=tag, kind=kind, note=note, dev=dev_scores), f, indent=1)
    D.write_submission(tag, summaries_test, note=note, kind=kind, dev_scores=dev_scores)


def score_pick(pool, pick, refs, use_bert):
    sm = {k: pool[s]["dev"][k] for k, s in pick.items()}
    return D.score_summaries(sm, {k: refs[k] for k in pick}, do_bert=use_bert)


def cmd_zips(a):
    """Every neural run at the final length + the extractive family: required coverage."""
    how, mult = length_setting()
    w_test = S.len_targets(D.load("test"), how, mult)
    for d in neural_runs():
        t = d["tag"]
        gt = jload(f"{RES}/{t}/gen_{d['decode']}_test.json")
        D.write_submission(t, {k: trim(v, w_test[k]) for k, v in gt.items()}, kind="model",
                           note=f"{d['model']} {d['method']} view={d['view']} decode={d['decode']} "
                                f"len={how}x{mult}", dev_scores=d.get("dev"))
    for t in EXTRACTIVE:
        if os.path.exists(f"{RES}/{t}/summaries_test.json") and \
                not os.path.exists(f"{P6}/submissions/{t}/submission.zip"):
            m = jload(f"{RES}/{t}/meta.json")
            D.write_submission(t, jload(f"{RES}/{t}/summaries_test.json"), kind="extractive",
                               note=json.dumps({k: v for k, v in m.items() if k != "dev"}),
                               dev_scores=m.get("dev"))


def cmd_all(a):
    cmd_zips(a)
    use_bert = not a.no_bert
    refs = D.refs_of("dev")
    report = {}
    pool, kinds, ids = build_pool(1.0)
    test_ids = [r["id"] for r in D.load("test")]
    neural = [s for s in pool if kinds[s] == "model"]
    if len(pool) < 2 or not ids:
        log(f"pool has {len(pool)} systems over {len(ids)} dev docs -- nothing to ensemble")
        return
    log(f"pool: {len(pool)} systems ({len(neural)} neural) over {len(ids)} decoded dev docs")
    singles = {s: D.score_summaries(pool[s]["dev"], {k: refs[k] for k in ids}, do_bert=use_bert)
               for s in pool}
    key = "bertscore" if use_bert else "rouge1"
    for s, v in sorted(singles.items(), key=lambda x: -x[1][key]):
        log(f"  single {s:16s} {kinds[s]:10s} {key} {v[key]:.4f} words {v['pred_words']:.0f}")
    report["singles"] = singles

    # MBR is a majority vote: a pool of many weak members outvotes a few good ones,
    # so it runs over the TOP_K systems by held-out score (Phase 1 pooled its best
    # models only, for the same reason)
    rank_ = sorted(pool, key=lambda s: -singles[s][key])
    top_all = rank_[:a.top_k]
    top_model = [s for s in rank_ if kinds[s] == "model"][:a.top_k]
    variants = [("mbr_all", top_all, "hybrid" if any(kinds[s] != "model" for s in top_all) else "model")]
    if len(top_model) >= 2:
        variants.append(("mbr_model", top_model, "model"))
    for tag, systems, kind in variants:
        pd = mbr(pool, systems, "dev", ids)
        pt = mbr(pool, systems, "test", test_ids)
        sc = score_pick(pool, pd, refs, use_bert)
        log(f"  {tag}: {key} {sc[key]:.4f}  (picks: { {str(k): int(v) for k, v in zip(*np.unique(list(pd.values()), return_counts=True))} })")
        write(tag, {k: pool[s]["test"][k] for k, s in pt.items()}, kind,
              f"MBR, ROUGE-1 utility, over {systems}", sc)
        report[tag] = sc
    if use_bert and len(pool) >= 3:
        top = sorted(pool, key=lambda s: -singles[s][key])[:6]
        pd = mbr(pool, top, "dev", ids, util="bert")
        pt = mbr(pool, top, "test", test_ids, util="bert")
        sc = score_pick(pool, pd, refs, use_bert)
        log(f"  mbr_bert: {key} {sc[key]:.4f}")
        kind = "model" if all(kinds[s] == "model" for s in top) else "hybrid"
        write("mbr_bert", {k: pool[s]["test"][k] for k, s in pt.items()}, kind,
              f"MBR, BERTScore utility (roberta-large in the decision rule), over {top}", sc)
        report["mbr_bert"] = sc

    for tag, systems, kind in [("rank_all", list(pool), "hybrid")] + \
            ([("rank_model", neural, "model")] if len(neural) >= 2 else []):
        pd, pt, oof, per, orc = rerank(pool, systems, ids, refs, test_ids, use_bert)
        best1 = max(per.values())
        log(f"  {tag}: OOF {key} {oof:.4f} vs best single {best1:.4f} "
            f"(gain {oof - best1:+.4f}) | per-doc oracle {orc:.4f}")
        sc = score_pick(pool, pd, refs, False)          # words + ROUGE of the OOF picks
        sc.update({key: oof, "best_single": best1, "oracle": orc})
        write(tag, {k: pool[s]["test"][k] for k, s in pt.items()}, kind,
              f"dev-trained reranker (HistGB, 5-fold OOF by document) over {systems}", sc)
        report[tag] = dict(oof=oof, best_single=best1, oracle=orc)

    for scale, sfx in ((0.85, "_short"), (1.15, "_long")):      # leaderboard length bracket
        p2, k2, ids2 = build_pool(scale)
        systems = [s for s in top_all if s in p2]                 # the SAME members as mbr_all
        if len(systems) < 2 or not ids2:
            continue
        pt = mbr(p2, systems, "test", test_ids)
        pd = mbr(p2, systems, "dev", ids2)
        sc = score_pick(p2, pd, refs, use_bert)
        write(f"mbr_all{sfx}", {k: p2[s]["test"][k] for k, s in pt.items()}, variants[0][2],
              f"mbr_all's members {systems} at {scale}x the calibrated length", sc)
        report[f"mbr_all{sfx}"] = sc
        log(f"  mbr_all{sfx}: {key} {sc[key]:.4f} words {sc['pred_words']:.0f}")
    with open(f"{RES}/ens_report.json", "w") as f:
        json.dump(report, f, indent=1)
    order()


def order():
    """SUBMIT_ORDER.md: required coverage first (the phase is graded on it), then
    everything else by its held-out score."""
    rows = []
    for p in sorted(glob.glob(f"{P6}/submissions/*/info.json")):
        i = jload(p)
        dv = i.get("dev") or {}
        sc = dv.get("bertscore", dv.get("rouge1"))
        rows.append((i["tag"], i["kind"], sc, dv.get("pred_words"), i.get("note", "")[:90]))
    req = [r for t in REQUIRED for r in rows if r[0] == t]
    rest = sorted([r for r in rows if r[0] not in REQUIRED], key=lambda r: -(r[2] or 0))
    L = ["# Phase 6 -- upload order (generated by phase6_ens.py)", "",
         "Upload `submissions/_upload/submission_<tag>.zip`. Required coverage first, then by",
         "held-out score. Dev scores of different systems are on the same decoded dev subset",
         "only within the pool; the per-run numbers are each run's own decoded subset.", "",
         "| # | tag | kind | dev | words | note |", "|---|---|---|---|---|---|"]
    for n, (t, k, sc, w, note) in enumerate(req + rest, 1):
        L.append(f"| {n} | `{t}` | {k} | {sc if sc is None else round(sc, 4)} | "
                 f"{w if w is None else round(w)} | {note} |")
    missing = [t for t in REQUIRED if t not in {r[0] for r in rows}]
    if missing:
        L += ["", f"**Missing required runs:** {', '.join(missing)} -- see logs/queue.txt."]
    with open(f"{P6}/SUBMIT_ORDER.md", "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    log(f"wrote {P6}/SUBMIT_ORDER.md ({len(rows)} zips; missing required: {missing or 'none'})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["all", "zips", "order"])
    ap.add_argument("--no-bert", action="store_true")
    ap.add_argument("--top-k", type=int, default=6, help="systems in each MBR pool")
    a = ap.parse_args()
    {"all": cmd_all, "zips": cmd_zips, "order": lambda _: order()}[a.cmd](a)


if __name__ == "__main__":
    main()
