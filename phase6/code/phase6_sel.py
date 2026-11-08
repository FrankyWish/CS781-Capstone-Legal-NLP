#!/usr/bin/env python3
"""CS-781 Phase 6 -- sentence selection: the CPU half of the phase.

Two jobs, one model:

 1. INPUT VIEW for the neural summarisers.  A Phase-1 trunk sees 1024-1280 tokens;
    these judgments are far longer.  "Importance-based context selection" from the
    brief means: score every sentence, keep the best ones in document order, and
    hand the model a window that actually contains the headnote material.  `view`.

 2. A STRONG EXTRACTIVE BASELINE, which on an unrescaled-BERTScore task is not a
    toy.  Phase 1 ended with retrieval beating generation by 0.024; the same
    asymmetry is likely here, because a 126M model trained from scratch on a few
    thousand pairs writes worse English than the judgment already contains.
    Submitted with kind='extractive' and reported as not abstractive.  `predict`.

The scorer is a HistGradientBoostingRegressor over cheap lexical/positional
features, supervised by each sentence's own ROUGE-1 F against the reference
summary.  No pretrained weights anywhere, so nothing here touches the phase rules.

    python phase6_sel.py fit    [--procs 12] [--docs 3000]
    python phase6_sel.py predict --tag sel_m10 --split dev,test [--len model] [--mult 1.0]
    python phase6_sel.py baselines --split dev
    python phase6_sel.py calib --tag sel_m10 [--bert 200]
    python phase6_sel.py view --name sel1024 --words 850 --split fit,dev,test
    python phase6_sel.py zip --tag sel_m10
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import time

import numpy as np

import phase6_data as D

P6 = D.P6
RES = f"{P6}/results"
VIEWS = f"{P6}/views"
MODEL = f"{RES}/sel_model.pkl"

CUES = {
    "held": r"\b(held|holding|we hold|it was held)\b",
    "decide": r"\b(therefore|accordingly|hence|thus|consequently|in the result)\b",
    "outcome": r"\b(allow(ed|ing)?|dismiss(ed|ing)?|set aside|quash(ed)?|remand(ed)?|acquit|convict)\w*\b",
    "statute": r"\b(section|sections|s\.|art(icle)?s?\.?|act|rule|clause|sub-section)\b",
    "court": r"\b(high court|supreme court|tribunal|trial court|appellate)\b",
    "party": r"\b(appellant|respondent|petitioner|plaintiff|defendant|accused|assessee)\b",
    "question": r"\b(whether|question (is|arises)|issue (is|arises)|point for consideration)\b",
    "opinion": r"\b(in our (opinion|view)|we are of the|it is clear that|no doubt)\b",
    "evidence": r"\b(evidence|witness|testimony|document|exhibit|affidavit)\b",
    "cite": r"\b(air|scc|scr|ilr|v\.|vs\.?|reported in|supra)\b",
    "proc": r"\b(appeal no|civil appeal|criminal appeal|writ petition|special leave|slp)\b",
    "date": r"\b(19|20)\d{2}\b",
}
CUE_RE = {k: re.compile(v, re.IGNORECASE) for k, v in CUES.items()}
FEAT = (["pos", "pos2", "is_first", "is_top3", "is_last", "is_bot3", "words", "logw",
         "sim", "simrank", "digits", "caps", "quote", "nsent", "docwords"]
        + [f"cue_{k}" for k in CUES])

BOILER = re.compile(
    r"^(?:(?:civil|criminal)\s+appeal|writ petition|special leave|slp|appeal\s+no|"
    r"petition\s+no|c\.?a\.?\s*no|coram|before\s*:|from the judgment|"
    r"the judgment of the court was delivered by|judgment|judgement|order)\b", re.IGNORECASE)


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


# ------------------------------------------------------------------ features --
def sent_feats(doc, vec):
    """(sentences, feature matrix).  One TF-IDF transform per document."""
    ss = [s for s in D.sentences(doc) if not BOILER.match(s)]
    if not ss:
        ss = D.sentences(doc) or [" ".join(doc.split()[:60]) or "empty"]
    X = vec.transform(ss)
    n = np.asarray(X.multiply(X).sum(1)).ravel() ** 0.5
    Xn = X.multiply(1 / np.maximum(n, 1e-9)[:, None]).tocsr()
    cen = np.asarray(Xn.mean(0)).ravel()
    cen /= max(np.linalg.norm(cen), 1e-9)
    sim = Xn @ cen
    order = np.argsort(np.argsort(-sim)) / max(len(ss) - 1, 1)
    dw = sum(D.nwords(s) for s in ss)
    F = np.zeros((len(ss), len(FEAT)), dtype=np.float32)
    for i, s in enumerate(ss):
        w = D.nwords(s)
        low = s.lower()
        row = [i / max(len(ss) - 1, 1), (i / max(len(ss) - 1, 1)) ** 2, float(i == 0),
               float(i < 3), float(i == len(ss) - 1), float(i >= len(ss) - 3), w, np.log1p(w),
               float(sim[i]), float(order[i]),
               sum(c.isdigit() for c in s) / max(len(s), 1),
               sum(c.isupper() for c in s) / max(len(s), 1),
               float('"' in s or "'" in s), np.log1p(len(ss)), np.log1p(dw)]
        row += [min(len(CUE_RE[k].findall(low)), 4) for k in CUES]
        F[i] = row
    return ss, F, Xn


def build_vec(recs, n_sent=200_000, seed=D.SEED):
    from sklearn.feature_extraction.text import TfidfVectorizer
    rng = np.random.default_rng(seed)
    pool = []
    for i in rng.permutation(len(recs)):
        pool.extend(D.sentences(recs[i]["doc"])[:120])
        if len(pool) >= n_sent:
            break
    vec = TfidfVectorizer(sublinear_tf=True, min_df=3, max_features=120_000,
                          dtype=np.float32, token_pattern=r"[a-zA-Z][a-zA-Z0-9]+")
    vec.fit(pool)
    log(f"  tfidf idf fitted on {len(pool)} sentences, {len(vec.vocabulary_)} terms")
    return vec


_W = {}


def _init(vec):
    _W["vec"] = vec


def _rows(rec):
    """Training rows for one document: features + each sentence's own ROUGE-1 F."""
    ss, F, _ = sent_feats(rec["doc"], _W["vec"])
    rt = D.toks(rec["summ"])
    y = np.array([D.rouge1_f(D.toks(s), rt) for s in ss], dtype=np.float32)
    return F, y


def _pred_rows(rec):
    ss, F, Xn = sent_feats(rec["doc"], _W["vec"])
    return rec["id"], ss, F, Xn


def _map(fn, recs, procs):
    if procs <= 1:
        _init(_W.get("vec"))
        return [fn(r) for r in recs]
    import multiprocessing as mp
    with mp.Pool(procs, initializer=_init, initargs=(_W["vec"],)) as pool:
        return pool.map(fn, recs, chunksize=8)


def cmd_fit(a):
    from sklearn.ensemble import HistGradientBoostingRegressor
    os.makedirs(RES, exist_ok=True)
    fit = D.load("fit")
    rng = np.random.default_rng(D.SEED)
    sub = [fit[i] for i in rng.permutation(len(fit))[:min(a.docs, len(fit))]]
    _W["vec"] = build_vec(fit)
    log(f"  extracting rows from {len(sub)} docs on {a.procs} procs")
    out = _map(_rows, sub, a.procs)
    X = np.concatenate([o[0] for o in out])
    y = np.concatenate([o[1] for o in out])
    log(f"  {X.shape[0]} sentences, target mean {y.mean():.4f}")
    m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, max_depth=None,
                                      min_samples_leaf=40, l2_regularization=1.0,
                                      early_stopping=True, validation_fraction=0.1,
                                      random_state=D.SEED)
    m.fit(X, y)
    with open(MODEL, "wb") as f:
        pickle.dump(dict(model=m, vec=_W["vec"], feat=FEAT), f)
    log(f"  wrote {MODEL} ({m.n_iter_} trees)")


def load_model():
    with open(MODEL, "rb") as f:
        d = pickle.load(f)
    _W["vec"] = d["vec"]
    return d["model"]


# ------------------------------------------------------------------- lengths --
def len_targets(recs, how="model", mult=1.0, floor=40, cap=2500):
    """Words of summary to aim for, per document.  F1 is symmetric: this is the
    dominant dial in the phase, which is why it is a first-class argument.

    No clipping at the scorer's 512-token window.  Above that window BERTScore
    stops seeing the text, so extra words neither help nor hurt the primary
    metric while they still help ROUGE -- the dial is one-sided up there, which
    is why the calibration grid runs well past 1.0."""
    g = {}
    f = f"{RES}/gate.json"
    if os.path.exists(f):
        with open(f) as fh:
            g = json.load(fh)
    med = (g.get("train_summ_words") or {}).get("p50")
    lm = g.get("len_model") or {}
    if med is None:
        med = float(np.median([D.nwords(r["summ"]) for r in D.load("fit")]))
    out = {}
    for r in recs:
        if how == "model" and lm:
            t = np.exp(lm["log_intercept"] + lm["log_slope"] * np.log(max(D.nwords(r["doc"]), 1)))
        else:
            t = med
        out[r["id"]] = int(np.clip(t * mult, floor, cap))
    return out


def select(ss, F, Xn, model, budget, lam=0.45, trim=True):
    """Greedy MMR over predicted sentence relevance, document order preserved,
    trimmed to exactly `budget` words (length beats fluency on this metric)."""
    rel = model.predict(F)
    chosen, words = [], 0
    simsel = np.zeros(len(ss), dtype=np.float32)
    for _ in range(len(ss)):
        sc = rel - lam * simsel
        sc[chosen] = -1e9
        i = int(np.argmax(sc))
        if sc[i] <= -1e8:
            break
        chosen.append(i)
        words += D.nwords(ss[i])
        row = Xn[i] @ Xn.T
        simsel = np.maximum(simsel, np.asarray(row.todense() if hasattr(row, "todense")
                                               else row).ravel())
        if words >= budget:
            break
    text = " ".join(ss[i] for i in sorted(chosen))
    w = text.split()
    if trim and len(w) > budget:
        text = " ".join(w[:budget])
    return text or " ".join(ss[0].split()[:budget])


def _summarise(split, model, a, recs=None, feats=None):
    recs = recs if recs is not None else D.load(split)
    tg = len_targets(recs, a.len, a.mult)
    out = feats if feats is not None else _map(_pred_rows, recs, a.procs)
    sums = {}
    for rid, ss, F, Xn in out:
        sums[rid] = select(ss, F, Xn, model, tg[rid], lam=a.lam, trim=not a.no_trim)
    return sums


def cmd_predict(a):
    model = load_model()
    tag = a.tag
    d = f"{RES}/{tag}"
    os.makedirs(d, exist_ok=True)
    meta = dict(tag=tag, kind="extractive", len=a.len, mult=a.mult, lam=a.lam,
                trim=not a.no_trim)
    done = {}
    for split in a.split.split(","):
        sums = done[split] = _summarise(split, model, a)
        with open(f"{d}/summaries_{split}.json", "w", encoding="utf-8") as f:
            json.dump(sums, f, ensure_ascii=False)
        log(f"  {tag}/{split}: {len(sums)} summaries, "
            f"{np.mean([D.nwords(v) for v in sums.values()]):.0f} words mean")
    if "dev" in done:             # after every worker pool: never fork once the scorer is on GPU
        s = D.score_summaries(done["dev"], D.refs_of("dev"), do_bert=not a.no_bert)
        meta["dev"] = s
        log(f"  {tag} dev: " + " ".join(f"{k} {v:.4f}" for k, v in s.items()
                                        if isinstance(v, float)))
    with open(f"{d}/meta.json", "w") as f:
        json.dump(meta, f, indent=1)


def cmd_baselines(a):
    """Lead-k and TF-IDF centroid, at the same length targets: the floor every
    neural run has to clear before it is worth a submission slot."""
    recs = D.load(a.split)
    _W["vec"] = build_vec(D.load("fit"), n_sent=60_000)
    tg = len_targets(recs, a.len, a.mult)
    res = {}
    lead, cent = {}, {}
    for r in recs:
        ss, F, Xn = sent_feats(r["doc"], _W["vec"])
        b = tg[r["id"]]
        acc, w = [], 0
        for s in ss:
            acc.append(s)
            w += D.nwords(s)
            if w >= b:
                break
        lead[r["id"]] = " ".join(" ".join(acc).split()[:b])
        sim = F[:, FEAT.index("sim")]
        order = np.argsort(-sim)
        pick, w = [], 0
        for i in order:
            pick.append(i)
            w += D.nwords(ss[i])
            if w >= b:
                break
        cent[r["id"]] = " ".join(" ".join(ss[i] for i in sorted(pick)).split()[:b])
    refs = D.refs_of(a.split)
    for nm, sm in (("lead", lead), ("centroid", cent)):
        res[nm] = D.score_summaries(sm, refs, do_bert=not a.no_bert)
        log(f"  {nm}: " + " ".join(f"{k} {v:.4f}" for k, v in res[nm].items()
                                   if isinstance(v, float)))
        os.makedirs(f"{RES}/base_{nm}", exist_ok=True)
        with open(f"{RES}/base_{nm}/summaries_{a.split}.json", "w", encoding="utf-8") as f:
            json.dump(sm, f, ensure_ascii=False)
    with open(f"{RES}/baselines_{a.split}.json", "w") as f:
        json.dump(res, f, indent=1)


def cmd_calib(a):
    """Sweep the length multiplier.  ROUGE-1 over the whole held-out set is free;
    BERTScore -- the metric that actually ranks us -- is confirmed on a subset,
    because the two do not always agree about length."""
    model = load_model()
    dev = D.load("dev")
    refs = D.refs_of("dev")
    rng = np.random.default_rng(D.SEED)
    sub = [dev[i] for i in rng.permutation(len(dev))[:a.bert]] if a.bert else []
    rows = []
    feats = _map(_pred_rows, dev, a.procs)      # once: no fork after the scorer reaches the GPU
    for how in a.lens.split(","):
        for mult in [float(x) for x in a.mults.split(",")]:
            a.len, a.mult = how, mult
            sm = _summarise("dev", model, a, feats=feats)
            r = D.score_summaries(sm, refs, do_bert=False)
            row = dict(len=how, mult=mult, rouge1=r["rouge1"], rougeL=r["rougeL"],
                       words=r["pred_words"])
            if sub:
                ss = {x["id"]: sm[x["id"]] for x in sub}
                rr = {x["id"]: refs[x["id"]] for x in sub}
                row["bertscore_sub"] = D.score_summaries(ss, rr, do_bert=True)["bertscore"]
            rows.append(row)
            log("  " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                for k, v in row.items()))
    key = "bertscore_sub" if sub else "rouge1"
    best = max(rows, key=lambda r: r[key])
    ms = sorted({r["mult"] for r in rows})
    if len(ms) > 1 and best["mult"] in (ms[0], ms[-1]):
        log(f"  !! the winning multiplier {best['mult']} is at the edge of the grid "
            f"{ms} -- rerun calib with a wider --mults before trusting it")
    with open(f"{RES}/length_calib.json", "w") as f:
        json.dump(dict(rows=rows, chose=best, key=key), f, indent=1)
    log(f"  BEST by {key}: {best}")


def cmd_view(a):
    """Write the selected-sentence input views the neural trainer reads."""
    model = load_model()
    os.makedirs(VIEWS, exist_ok=True)
    for split in a.split.split(","):
        recs = D.load(split)
        out = _map(_pred_rows, recs, a.procs)
        v = {}
        for rid, ss, F, Xn in out:
            v[rid] = select(ss, F, Xn, model, a.words, lam=a.lam, trim=False)
        with open(f"{VIEWS}/{a.name}_{split}.json", "w", encoding="utf-8") as f:
            json.dump(v, f, ensure_ascii=False)
        log(f"  {VIEWS}/{a.name}_{split}.json: {len(v)} docs, "
            f"{np.mean([D.nwords(x) for x in v.values()]):.0f} words mean")


def cmd_zip(a):
    with open(f"{RES}/{a.tag}/summaries_test.json", encoding="utf-8") as f:
        sm = json.load(f)
    meta = {}
    if os.path.exists(f"{RES}/{a.tag}/meta.json"):
        with open(f"{RES}/{a.tag}/meta.json") as f:
            meta = json.load(f)
    D.write_submission(a.tag, sm, note=a.note or json.dumps(
        {k: v for k, v in meta.items() if k != "dev"}), kind=meta.get("kind", "extractive"),
        dev_scores=meta.get("dev"))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    sp = ap.add_subparsers(dest="cmd", required=True)

    f = sp.add_parser("fit")
    f.add_argument("--docs", type=int, default=3000)
    f.set_defaults(fn=cmd_fit)

    def common(p):
        p.add_argument("--len", default="model", choices=["model", "const"])
        p.add_argument("--mult", type=float, default=1.0)
        p.add_argument("--lam", type=float, default=0.45)
        p.add_argument("--no-trim", action="store_true")
        p.add_argument("--no-bert", action="store_true")

    p = sp.add_parser("predict")
    p.add_argument("--tag", required=True)
    p.add_argument("--split", default="dev,test")
    common(p)
    p.set_defaults(fn=cmd_predict)

    b = sp.add_parser("baselines")
    b.add_argument("--split", default="dev")
    common(b)
    b.set_defaults(fn=cmd_baselines)

    c = sp.add_parser("calib")
    c.add_argument("--tag", default="calib")
    c.add_argument("--lens", default="model,const")
    c.add_argument("--mults", default="0.6,0.75,0.9,1.0,1.1,1.25,1.45,1.7")
    c.add_argument("--bert", type=int, default=200, help="0 = ROUGE only")
    common(c)
    c.set_defaults(fn=cmd_calib)

    v = sp.add_parser("view")
    v.add_argument("--name", required=True)
    v.add_argument("--words", type=int, default=850)
    v.add_argument("--split", default="fit,dev,test")
    v.add_argument("--lam", type=float, default=0.45)
    v.set_defaults(fn=cmd_view)

    z = sp.add_parser("zip")
    z.add_argument("--tag", required=True)
    z.add_argument("--note", default="")
    z.set_defaults(fn=cmd_zip)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
