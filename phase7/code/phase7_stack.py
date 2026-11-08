#!/usr/bin/env python
"""Phase 7 -- cross-fitted feature stacker / reranker over the saved system runs.

Micro-F1@7 is linear in total top-7 hits (SUBMISSION_PLAN.md s1), so the objective is
expected hits: score every pooled candidate with P(relevant), sort, cut at 7.  Nothing
listwise beats that here, and there is no K to calibrate.

Why a stacker rather than a cross-encoder.  The devx oracle says a perfect reordering of
fuse's top 50 is worth 478 hits against the 339 the best single system gets, so ordering
is where the phase is won.  A true cross-encoder over these documents would be 237 test
queries x 50 candidates x ~10k tokens a pair, which does not fit the schedule.  Phase 6
was won the same way: a cross-fitted depth-3 HistGB over several cheap scorers.

Every feature is normalised WITHIN its query.  The train pool is 4,320 candidates against
devx's 1,023 and test's 1,727, and `tfidf_sqrt` scores 0.2875 on train vs 0.3769 on devx
purely because of that: a raw score, and anything derived from pool size such as a raw
rank, is not comparable across splits, so a model fitted on raw train features would be
miscalibrated on test.  Absent features are left NaN, which HistGradientBoosting splits on
natively -- "this system did not retrieve this candidate" is information, not a zero.

Kinds, per the brief: a stack whose members include BM25/TF-IDF is a `hybrid` and is
reported beside the model-only stack over the dense trunks alone.

  python phase7_stack.py fit  --tag stack       --kind hybrid
  python phase7_stack.py fit  --tag stack_model --kind model --members-model
  python phase7_stack.py half --tags stack,tfidf_sqrt
"""
import argparse
import hashlib
import json
import os
import re

import numpy as np

import phase7_data as D
import phase7_rank as R

P7 = D.P7
SEED = D.SEED
K = D.K

# Permissive on purpose: the max year in a judgment is only a date proxy, and the
# measured check (23% of devx golds postdate their query under this proxy) is why the
# predate filter was rejected and only the signed difference survives as a feature.
YEAR = re.compile(r"\b(1[89]\d\d|20[0-2]\d)\b")

MEMBERS = ("tfidf_sqrt", "bm25_hi", "bm25", "tfidf",
           "dense_m2_c8_max", "dense_m2_c8", "dense_m2_max", "dense_m2",
           "dense_m1", "dense_m3", "dense_m4")
MODEL_ONLY = ("dense_m2_c8_max", "dense_m2_c8", "dense_m2_max", "dense_m2",
              "dense_m1", "dense_m3", "dense_m4")

# n_masks is the count of citation masks in the query.  It is not a relevance signal for
# a candidate -- it is a per-query signal of HOW MANY prior cases were cited, which lets
# the tree learn how deep in a ranking to keep trusting: for a query that cited 20 cases a
# candidate at rank 30 is far likelier to be gold than for one that cited 2.
GLOBAL_FEATS = ("n_sys", "z_mean", "z_max", "rr_sum", "best_rank",
                "cand_logw", "qry_logw", "logw_diff", "year_diff", "cand_year_ok",
                "n_masks", "masks_per_1k")


def log(*a):
    print(" ", *a, flush=True)


def available(members, splits):
    """Members with a saved ranking on every split we need.  A missing member is skipped
    loudly rather than silently filled, so a failed embed step cannot quietly change the
    feature set between runs."""
    ok = []
    for m in members:
        miss = [s for s in splits if R.load_run(m, s)[0] is None]
        if miss:
            log(f"skip {m}: no run for {','.join(miss)}")
        else:
            ok.append(m)
    if not ok:
        raise SystemExit(f"no member has runs on all of {splits}")
    return ok


def doc_stats(split):
    """(word count, max year) for every candidate and query of a split."""
    def one(sents):
        txt = " ".join(sents)
        ys = YEAR.findall(txt)
        return len(txt.split()), (float(max(int(y) for y in ys)) if ys else np.nan)
    cand = {c: one(v) for c, v in D.candidates(split).items()}
    qry = {r["id"]: one(r["sents"]) for r in D.queries(split)}
    return cand, qry


def feature_names(members):
    names = []
    for m in members:
        names += [f"{m}:z", f"{m}:rr", f"{m}:in"]
    return names + list(GLOBAL_FEATS)


def build(split, members, pool_depth, labels=True):
    """-> X, y, group index, [(qid, cand)], [qid].

    The candidate pool for a query is the union of every member's top `pool_depth`, so the
    reranker sees everything any retriever thought was plausible."""
    runs = {m: R.load_run(m, split) for m in members}
    cstat, qstat = doc_stats(split)
    import phase7_cite as CT
    nmask = CT.mask_counts(split)
    gold = D.golds_in_pool(split) if labels else {}
    qids = [r["id"] for r in D.queries(split)]
    nm = len(members)
    nf = 3 * nm + len(GLOBAL_FEATS)
    X, y, grp, pairs = [], [], [], []
    for gi, q in enumerate(qids):
        look, pool = {}, []
        for m in members:
            rk, sc = runs[m]
            ids = (rk or {}).get(q, [])[:pool_depth]
            sv = (sc or {}).get(q)
            d = {}
            for i, c in enumerate(ids):
                d[c] = (i, float(sv[i]) if sv is not None and i < len(sv) else np.nan)
            look[m] = d
            pool.extend(ids)
        pool = list(dict.fromkeys(pool))
        if not pool:
            continue
        # per-query standardisation of each member's own retrieved scores
        zs = {}
        for m in members:
            v = np.array([t[1] for t in look[m].values()], dtype=np.float64)
            v = v[np.isfinite(v)]
            mu = float(v.mean()) if v.size else 0.0
            sd = float(v.std()) if v.size > 1 else 1.0
            zs[m] = (mu, sd if sd > 1e-12 else 1.0)
        qw, qy = qstat.get(q, (0, np.nan))
        nmk = float(nmask.get(q, 0))
        gset = set(gold.get(q, ()))
        for c in pool:
            row = np.full(nf, np.nan, dtype=np.float32)
            zv, ranks, rr, npres = [], [], 0.0, 0
            k = 0
            for m in members:
                hit = look[m].get(c)
                if hit is None:
                    row[k + 2] = 0.0            # explicit: not retrieved by this system
                else:
                    i, s = hit
                    mu, sd = zs[m]
                    z = (s - mu) / sd if np.isfinite(s) else np.nan
                    row[k] = z
                    row[k + 1] = 1.0 / (60.0 + i)
                    row[k + 2] = 1.0
                    if np.isfinite(z):
                        zv.append(z)
                    rr += 1.0 / (60.0 + i)
                    ranks.append(i)
                    npres += 1
                k += 3
            cw, cy = cstat.get(c, (0, np.nan))
            row[k + 0] = float(npres)
            row[k + 1] = float(np.mean(zv)) if zv else np.nan
            row[k + 2] = float(np.max(zv)) if zv else np.nan
            row[k + 3] = rr
            row[k + 4] = float(min(ranks)) if ranks else np.nan
            row[k + 5] = float(np.log1p(cw))
            row[k + 6] = float(np.log1p(qw))
            row[k + 7] = float(np.log1p(cw) - np.log1p(qw))
            row[k + 8] = (qy - cy) if (np.isfinite(qy) and np.isfinite(cy)) else np.nan
            row[k + 9] = 1.0 if np.isfinite(cy) else 0.0
            row[k + 10] = nmk
            row[k + 11] = 1000.0 * nmk / max(qw, 1.0)
            X.append(row)
            grp.append(gi)
            pairs.append((q, c))
            if labels:
                y.append(1 if c in gset else 0)
    X = np.vstack(X) if X else np.zeros((0, nf), dtype=np.float32)
    return X, np.array(y, dtype=np.int8), np.array(grp, dtype=np.int32), pairs, qids


def usable_cols(X):
    """Columns with at least one finite value and some variation.

    An all-NaN column makes HistGradientBoosting's binner raise "window shape cannot be
    larger than input array shape", which is an obscure way to say it cannot find a
    percentile of nothing.  That happens for real: `year_diff` is all-NaN on any corpus
    without parseable years, and every column of a member whose scores failed to save
    would be too.  The mask is computed on train and reused verbatim for devx and test, so
    the model never sees a different feature layout than it was fitted on."""
    keep = np.zeros(X.shape[1], dtype=bool)
    for j in range(X.shape[1]):
        v = X[:, j]
        f = v[np.isfinite(v)]
        keep[j] = f.size > 0 and float(f.min()) != float(f.max())
    return keep


class Ranker:
    """LightGBM lambdarank behind the same fit/predict_proba surface HistGB exposes.

    Why bother.  The objective is "hits in the top 7", and a pointwise logloss spends its
    capacity separating candidate 400 from candidate 401 -- which cannot change the score
    at all.  lambdarank weights each swap by how much it would move the ranking metric at
    a truncation, so the gradient goes where the metric is.  `label_gain=[0, 1]` makes the
    gain linear in hits, which is what Micro-F1@7 is."""

    def __init__(self, a):
        try:
            import lightgbm as lgb
        except ImportError:
            raise SystemExit(
                "--model lgbm needs lightgbm, which is not importable here. Install it "
                "(`pip install -q lightgbm`) or drop the flag to use the HistGB model.")
        self.m = lgb.LGBMRanker(
            objective="lambdarank", n_estimators=a.iters, learning_rate=a.lr,
            num_leaves=a.leaves, max_depth=(a.depth if a.depth and a.depth > 0 else -1),
            min_child_samples=a.min_leaf, reg_lambda=a.l2,
            lambdarank_truncation_level=a.trunc, label_gain=[0, 1],
            random_state=SEED, verbose=-1, n_jobs=-1)

    def fit(self, X, y, grp):
        o = np.argsort(grp, kind="stable")          # lambdarank needs contiguous groups
        _, counts = np.unique(grp[o], return_counts=True)
        self.m.fit(X[o], y[o], group=counts)
        return self

    def predict_proba(self, X):
        v = np.asarray(self.m.predict(X), dtype=np.float64)
        return np.stack([-v, v], axis=1)            # only column 1 is ever read


def new_model(a):
    if getattr(a, "model", "hgb") == "lgbm":
        return Ranker(a)
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        max_depth=a.depth, learning_rate=a.lr, max_leaf_nodes=a.leaves,
        max_iter=a.iters, l2_regularization=a.l2, min_samples_leaf=a.min_leaf,
        early_stopping=False, random_state=SEED)


def fit_model(a, X, y, grp):
    """One call site for both models: the ranker needs the query groups, HistGB does not
    (and would read a third positional as sample_weight)."""
    m = new_model(a)
    if isinstance(m, Ranker):
        m.fit(X, y, grp)
    else:
        m.fit(X, y)
    return m


def crossfit(X, y, grp, folds, a):
    """Out-of-fold predictions, grouped by QUERY so no candidate is scored by a model that
    saw another candidate of the same query.  Without this the train number is fiction."""
    from sklearn.model_selection import GroupKFold
    folds = max(2, min(folds, len(set(grp.tolist()))))
    oof = np.zeros(len(y), dtype=np.float64)
    for tr, te in GroupKFold(n_splits=folds).split(X, y, groups=grp):
        m = fit_model(a, X[tr], y[tr], grp[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return oof


def to_rank(pairs, scores, qids, topn):
    """Ties break on the candidate id, so a re-run gives a byte-identical submission."""
    by = {}
    for (q, c), s in zip(pairs, scores):
        by.setdefault(q, []).append((float(s), c))
    rank, score = {}, {}
    for q in qids:
        v = sorted(by.get(q, []), key=lambda t: (-t[0], t[1]))[:topn]
        rank[q] = [c for _, c in v]
        score[q] = np.array([s for s, _ in v], dtype=np.float32)
    return rank, score


def importance(model, X, y, names, n=20000, repeats=3):
    from sklearn.inspection import permutation_importance
    rs = np.random.default_rng(SEED)
    idx = rs.choice(len(y), size=min(n, len(y)), replace=False)
    r = permutation_importance(model, X[idx], y[idx], n_repeats=repeats,
                               random_state=SEED, scoring="average_precision")
    order = np.argsort(-r.importances_mean)
    log("top features by permutation importance (average precision):")
    for i in order[:12]:
        log(f"    {names[i]:24s} {r.importances_mean[i]:+.5f} +- {r.importances_std[i]:.5f}")


def cmd_fit(a):
    members = list(MODEL_ONLY if a.members_model else
                   [m for m in a.members.split(",") if m])
    members = available(members, ["train", "devx", "test"])
    log(f"members ({len(members)}): {','.join(members)}")

    Xtr, ytr, gtr, ptr, qtr = build("train", members, a.pool_depth)
    log(f"train rows {Xtr.shape[0]:,} x {Xtr.shape[1]} features | "
        f"positives {int(ytr.sum()):,} ({100.0 * ytr.mean():.2f}%)")
    if ytr.sum() == 0:
        raise SystemExit("no positive training rows: check the train rankings")

    names = feature_names(members)
    cols = usable_cols(Xtr)
    dropped = [n for n, k in zip(names, cols) if not k]
    if dropped:
        log(f"dropping {len(dropped)} dead features: {','.join(dropped)}")
    if not cols.any():
        raise SystemExit("every feature is constant or missing")
    Xtr = Xtr[:, cols]
    names = [n for n, k in zip(names, cols) if k]

    oof = crossfit(Xtr, ytr, gtr, a.folds, a)
    rk, _ = to_rank(ptr, oof, qtr, K)
    s_tr = D.score_ranking(rk, D.golds("train"), reachable=D.golds_in_pool("train"))
    log("train OOF   " + D.brief(s_tr) + f" | ceiling {s_tr['ceiling_f1']:.4f}")

    model = fit_model(a, Xtr, ytr, gtr)
    if a.importance and not isinstance(model, Ranker):
        importance(model, Xtr, ytr, names)

    out = {}
    for split in ("devx", "test"):
        X, _, _, pairs, qids = build(split, members, a.pool_depth, labels=False)
        X = X[:, cols]
        p = model.predict_proba(X)[:, 1] if len(X) else np.zeros(0)
        rank, score = to_rank(pairs, p, qids, a.topn)
        R.save_run(a.tag, split, rank, score)
        out[split] = rank
        log(f"{split}: {X.shape[0]:,} rows, {np.mean([len(v) for v in rank.values()]):.0f} "
            f"ranked per query")

    sd = R.evaluate(out["devx"], "devx")
    R.report(f"{a.tag}/devx", sd)
    cfg = dict(members=members, pool_depth=a.pool_depth, depth=a.depth, lr=a.lr,
               leaves=a.leaves, iters=a.iters, l2=a.l2, min_leaf=a.min_leaf,
               model=getattr(a, "model", "hgb"), folds=a.folds,
               train_oof_f1=round(s_tr["f1"], 6),
               n_features=int(cols.sum()), dropped_features=dropped)
    R.save_run(a.tag, "meta", {}, meta=dict(
        kind=a.kind, config=cfg,
        devx_scores={k: sd[k] for k in ("f1", "hits", "precision", "recall")}))
    log(f"saved {a.tag}: devx f1 {sd['f1']:.4f} hits {sd['hits']}/{sd['max_hits']} "
        f"-- zip it with:  python phase7_rank.py zip --tag {a.tag} --kind {a.kind}")


def paired(per, tags, base, g, qs, label):
    """One paired table: f1, hits, and the 2-SE verdict against `base` over `qs`."""
    den = K * len(qs) + sum(len(g[q]) for q in qs)
    log(f"{label}: {len(qs)} queries, denominator {den}")
    hb = np.array([per[base][q][0] for q in qs], dtype=np.float64)
    out = {}
    for t in per:
        h = np.array([per[t][q][0] for q in qs], dtype=np.float64)
        f1 = 2 * h.sum() / den
        if t == base:
            log(f"    {t:22s} f1 {f1:.4f} hits {int(h.sum()):5d}   (baseline)")
            out[t] = dict(f1=round(f1, 4), hits=int(h.sum()))
            continue
        d = h - hb
        se = float(d.std(ddof=1)) * np.sqrt(len(d)) / den * 2 if len(d) > 1 else 0.0
        df1 = 2 * d.sum() / den
        verdict = "KEEP" if abs(df1) > 2 * se else "noise"
        log(f"    {t:22s} f1 {f1:.4f} hits {int(h.sum()):5d}   "
            f"delta {df1:+.4f} +- {se:.4f} (2SE {2 * se:.4f}) {verdict}")
        out[t] = dict(f1=round(f1, 4), hits=int(h.sum()), delta=round(df1, 5),
                      two_se=round(2 * se, 5), verdict=verdict)
    return out


def cmd_half(a):
    """Phase 6's selection protocol: choose on half A, report on half B, paired SE.

    Per-query hits are integers and the metric is linear in them, so the paired SE is
    exact here rather than a bootstrap estimate.  A gain under 2 SE is not kept.

    Plus the change that matters now: the same table over the 827 TRAIN queries.  devx has
    118 queries, which puts the 2-SE bar at about 11 hits (0.012 F1) -- so every real gain
    of 0.005 we are now chasing is invisible there by construction.  Train is 7x larger,
    and for a system that learns nothing from labels (every lexical retriever, the
    citation-context runs, the lex graph) its ranking is genuinely out of sample, so the
    bar is ~2.6x tighter.  A gain that clears train AND points the right way on devx half B
    is worth banking; one that clears only devx is still noise."""
    g, gp = D.golds("devx"), D.golds_in_pool("devx")
    side = {q: int(hashlib.md5(f"{q}:{SEED}".encode()).hexdigest(), 16) % 2 for q in g}
    tags = [t for t in a.tags.split(",") if t]
    base = tags[0]
    per = {}
    for t in tags:
        rk, _ = R.load_run(t, "devx")
        if rk is None:
            log(f"skip {t}: no devx ranking")
            continue
        per[t] = D.score_ranking(rk, g, reachable=gp)["per_query"]
    if base not in per:
        raise SystemExit(f"baseline {base} has no devx ranking")
    report = {}
    for name, want in (("A (choose)", 0), ("B (report)", 1)):
        qs = [q for q in g if side[q] == want]
        report[f"devx_half_{name[0]}"] = paired(per, tags, base, g, qs, f"half {name}")

    if a.train:
        gt, gpt = D.golds("train"), D.golds_in_pool("train")
        pt = {}
        for t in tags:
            rk, _ = R.load_run(t, "train")
            if rk is None or any(q not in rk for q in gt):
                log(f"skip {t} on train: no complete train ranking")
                continue
            pt[t] = D.score_ranking(rk, gt, reachable=gpt)["per_query"]
        if base in pt and len(pt) > 1:
            report["train"] = paired(pt, list(pt), base, gt, list(gt),
                                     "train (827 queries, tighter bar)")
        else:
            log(f"train table skipped: the baseline {base} has no train ranking")
    D.atomic_json(f"{P7}/results/half_report.json", report)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("fit")
    p.add_argument("--tag", default="stack")
    p.add_argument("--kind", default="hybrid", choices=list(D.KINDS))
    p.add_argument("--members", default=",".join(MEMBERS))
    p.add_argument("--members-model", action="store_true",
                   help="dense trunks only, for the model-only number")
    p.add_argument("--pool-depth", type=int, default=50)
    p.add_argument("--topn", type=int, default=100)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--leaves", type=int, default=15)
    p.add_argument("--lr", type=float, default=0.06)
    p.add_argument("--iters", type=int, default=400)
    p.add_argument("--l2", type=float, default=1.0)
    p.add_argument("--min-leaf", type=int, default=40)
    p.add_argument("--model", default="hgb", choices=["hgb", "lgbm"],
                   help="hgb = pointwise HistGB; lgbm = LightGBM lambdarank")
    p.add_argument("--trunc", type=int, default=20,
                   help="lambdarank truncation level (lgbm only)")
    p.add_argument("--importance", action="store_true")
    p.set_defaults(fn=cmd_fit)
    p = sp.add_parser("half")
    p.add_argument("--tags", required=True,
                   help="first tag is the baseline every other is compared against")
    p.add_argument("--no-train", dest="train", action="store_false",
                   help="skip the 827-query train table")
    p.set_defaults(fn=cmd_half, train=True)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
