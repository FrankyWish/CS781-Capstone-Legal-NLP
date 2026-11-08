#!/usr/bin/env python3
"""CS-781 Phase 6 -- the final day's batch: every useful zip, numbered in upload order.

    python phase6_batch.py --procs 10 all

Runs after `p6.sh ens`.  Adds, on top of the per-run zips and the ensembles:

  cons_*    sentence-level BERTScore consensus of the top systems (phase6_bsel), at
            three lengths, from document sentences alone or mixed with neural ones
  stitch_*  a neural opening followed by the best extraction
  *_xNNN    the best extractive systems across the length dial

Every new variant is scored with the real BERTScore on the same decoded dev documents,
identical predictions are dropped (Codabench does not count them, and they teach us
nothing), and submissions/_today/NN_<tag>.zip holds everything in upload order: the
required runs first, then the best model-only systems, then held-out score.  With 100
test documents the per-doc SE is ~0.003, so neighbouring ranks are ties: the list is a
priority order, not a promise, and the leaderboard settles it.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import shutil
import time
import traceback

import phase6_bsel as B
import phase6_data as D
import phase6_ens as E
import phase6_sel as S
import phase6_win as W

P6 = D.P6
RES = f"{P6}/results"
SUBS = f"{P6}/submissions"
log = S.log


def digest(sm):
    return hashlib.md5(json.dumps(sorted(sm.items()), ensure_ascii=False).encode()).hexdigest()


def existing():
    seen = {}
    for p in glob.glob(f"{SUBS}/*/prediction.json"):
        seen[digest({r["id"]: r["summary"] for r in E.jload(p)})] = os.path.basename(os.path.dirname(p))
    return seen


class Batch:
    def __init__(self, ids, refs, use_bert, minutes):
        self.ids, self.refs, self.use_bert = ids, refs, use_bert
        self.stop = time.time() + minutes * 60
        self.seen = existing()
        self.key = "bertscore" if use_bert else "rouge1"
        self.added = 0

    def add(self, tag, kind, note, dev, test):
        h = digest(test)
        if h in self.seen:
            log(f"  same prediction as {self.seen[h]}: {tag} dropped")
            return
        sc = None
        if time.time() < self.stop:
            sc = D.score_summaries({k: dev[k] for k in self.ids},
                                   {k: self.refs[k] for k in self.ids}, do_bert=self.use_bert)
            log(f"  {tag:26s} {kind:10s} {self.key} {sc[self.key]:.4f} words {sc['pred_words']:.0f}")
        else:
            log(f"  {tag}: scoring budget spent, zip written without a dev score")
        D.write_submission(tag, test, note=note, kind=kind, dev_scores=sc)
        self.seen[h] = tag
        self.added += 1


def scale_tag(t, x):
    return f"{t}_x{int(round(x * 100)):03d}"


def extractive_bracket(bt, m_all, recs, feats):
    """The window-aimed systems and sel_best, regenerated across the length dial."""
    for t in ("bsel_best", "win_best"):
        f = f"{RES}/{t}/meta.json"
        if not os.path.exists(f):
            continue
        m = E.jload(f)
        head = W.load_head(m.get("head", "win"))
        for x in (0.8, 0.9, 1.1, 1.25):
            mk = lambda r, fs: W.summarise(r, fs, "model", m["mult"] * x, m["win"], head, m_all, m["lam"])
            bt.add(scale_tag(t, x), "extractive", f"{t} at {x}x its chosen length",
                   mk(recs["dev"], feats["dev"]), mk(recs["test"], feats["test"]))
    how, mult = E.length_setting()
    for x in (0.7, 1.3):
        arg = argparse.Namespace(len=how, mult=mult * x, lam=0.45, no_trim=False, procs=1)
        bt.add(scale_tag("sel", x), "extractive", f"sel_best at {x}x the calibrated length",
               S._summarise("dev", m_all, arg, recs["dev"], feats["dev"]),
               S._summarise("test", m_all, arg, recs["test"], feats["test"]))


def stitched(bt, pool, kinds, ranked, tg_dev, tg_test):
    """The headnote's register from a model, the case's facts from the document."""
    neural = [s for s in ranked if kinds[s] == "model"][:3]
    extra = [s for s in ranked if kinds[s] == "extractive"][:2]

    def join(n, x, target, ratio, win=420):
        h = int(min(target, win) * ratio)
        return " ".join(n.split()[:h] + x.split()[:max(target - h, 0)])

    for n in neural:
        for x in extra:
            for ratio in (0.25, 0.5):
                mk = lambda split, tg: {k: join(pool[n][split][k], pool[x][split][k], tg[k], ratio)
                                        for k in pool[n][split]}
                bt.add(f"stitch_{n}_{x}_{int(ratio * 100)}", "hybrid",
                       f"first {ratio:.0%} of the window from {n}, the rest from {x}",
                       mk("dev", tg_dev), mk("test", tg_test))


def consensus_family(bt, pool, kinds, ranked, m_all, tg, feats):
    neural = [s for s in ranked if kinds[s] == "model"]
    ks, mults = (3, 6), (0.85, 1.0, 1.2)
    dev = B.consensus(feats["dev"], pool, "dev", ranked, neural, tg["dev"], m_all, ks, mults)
    test = B.consensus(feats["test"], pool, "test", ranked, neural, tg["test"], m_all, ks, mults)
    for (k, mixed, m), sm in sorted(dev.items(), key=lambda kv: (kv[0][2] != 1.0, kv[0])):
        src = "mix" if mixed else "doc"
        bt.add(f"cons_k{k}_{src}_x{int(round(m * 100)):03d}", "hybrid",
               f"sentence-level BERTScore consensus of {ranked[:k]}, {src} sentences, "
               f"{m}x length (roberta-large in the decision rule)", sm, test[(k, mixed, m)])


def top_up(bt, m_all, recs, feats, want):
    """Only if a thin run left fewer than `want` distinct zips."""
    how, mult = E.length_setting()
    for lam in (0.2, 0.7, 0.3, 0.6):
        for x in (1.0, 0.9, 1.1, 1.25):
            if count_zips() >= want:
                return
            arg = argparse.Namespace(len=how, mult=mult * x, lam=lam, no_trim=False, procs=1)
            bt.add(f"sel_l{int(lam * 100):02d}_x{int(round(x * 100)):03d}", "extractive",
                   f"sel_best with MMR lam={lam} at {x}x length",
                   S._summarise("dev", m_all, arg, recs["dev"], feats["dev"]),
                   S._summarise("test", m_all, arg, recs["test"], feats["test"]))


def count_zips():
    return len(glob.glob(f"{SUBS}/*/info.json"))


def final_order(singles, key, per_day):
    rows = []
    for p in sorted(glob.glob(f"{SUBS}/*/info.json")):
        i = E.jload(p)
        dv = i.get("dev") or {}
        # a system in the pool has a score on the shared decoded documents: prefer it
        sc = (singles.get(i["tag"]) or dv).get(key)
        w = (singles.get(i["tag"]) or dv).get("pred_words")
        rows.append(dict(tag=i["tag"], kind=i["kind"], dev=sc, words=w, note=i.get("note", "")[:90]))
    by = {r["tag"]: r for r in rows}
    first = [t for t in E.REQUIRED if t in by]
    models = sorted((r for r in rows if r["kind"] == "model" and r["tag"] not in first),
                    key=lambda r: -(r["dev"] or 0))
    first += [r["tag"] for r in models[:2]]
    rest = sorted((r for r in rows if r["tag"] not in first), key=lambda r: -(r["dev"] or 0))
    ranked = [by[t] for t in first] + rest
    today = f"{SUBS}/_today"
    os.makedirs(today, exist_ok=True)
    for f in glob.glob(f"{today}/*.zip"):
        os.remove(f)
    lines = ["# Phase 6 -- upload order for the last day (written by phase6_batch.py)", "",
             f"Upload `submissions/_today/NN_<tag>.zip` from 01 upwards. {len(ranked)} zips; the cap "
             f"is {per_day} a day, so anything past {per_day} is spare. Rows 1-{len(first)} are "
             "required coverage and the best model-only systems, the rest are by held-out BERTScore.",
             "With 100 test documents, gaps under 0.006 are mostly luck: trust the leaderboard.",
             "**Codabench shows the submission you push, not your best: push every new best.**", "",
             "| # | tag | kind | dev | words | note |", "|---|---|---|---|---|---|"]
    for n, r in enumerate(ranked, 1):
        shutil.copyfile(f"{SUBS}/_upload/submission_{r['tag']}.zip", f"{today}/{n:02d}_{r['tag']}.zip")
        d = "-" if r["dev"] is None else f"{r['dev']:.4f}"
        w = "-" if r["words"] is None else f"{r['words']:.0f}"
        lines.append(f"| {n} | `{r['tag']}` | {r['kind']} | {d} | {w} | {r['note']} |")
    missing = [t for t in E.REQUIRED if t not in by]
    if missing:
        lines += ["", f"**Missing required runs:** {', '.join(missing)} -- see logs/queue.txt."]
    with open(f"{P6}/SUBMIT_ORDER.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return ranked, missing


def cmd_all(a):
    m_all = S.load_model()
    pool, kinds, ids = E.build_pool(1.0)
    if not pool:
        raise SystemExit("no systems in the pool: nothing to build on")
    ids = ids[:a.n_dev]
    refs = D.refs_of("dev")
    use_bert = not a.no_bert
    want = set(ids)
    recs = dict(dev=[r for r in D.load("dev") if r["id"] in want], test=D.load("test"))
    # workers first: nothing may fork once the scorer is on the GPU
    feats = {s: S._map(S._pred_rows, recs[s], a.procs) for s in recs}
    how, mult = E.length_setting()
    tg = {s: S.len_targets(recs[s], how, mult) for s in recs}
    for s in pool:
        pool[s]["dev"] = {k: pool[s]["dev"][k] for k in ids}
    key = "bertscore" if use_bert else "rouge1"
    rep = f"{RES}/ens_report.json"
    singles = E.jload(rep).get("singles", {}) if os.path.exists(rep) else {}
    for s in pool:
        if key not in (singles.get(s) or {}):
            singles[s] = D.score_summaries(pool[s]["dev"], {k: refs[k] for k in ids}, do_bert=use_bert)
    ranked = sorted(pool, key=lambda s: -singles[s][key])
    log(f"pool of {len(pool)} over {len(ids)} docs; best: " +
        ", ".join(f"{s} {singles[s][key]:.4f}" for s in ranked[:6]))
    bt = Batch(ids, refs, use_bert, a.score_min)
    families = [("consensus", lambda: consensus_family(bt, pool, kinds, ranked, m_all, tg, feats)
                 if use_bert and len(pool) >= 3 else None),
                ("lengths", lambda: extractive_bracket(bt, m_all, recs, feats)),
                ("stitch", lambda: stitched(bt, pool, kinds, ranked, tg["dev"], tg["test"])),
                ("top-up", lambda: top_up(bt, m_all, recs, feats, a.min_zips))]
    for name, build in families:
        try:                        # one broken family must not cost the numbered list
            build()
        except Exception:
            log(f"  !! family {name} failed, carrying on:\n{traceback.format_exc()}")
    ranked_zips, missing = final_order(singles, key, a.per_day)
    log(f"added {bt.added} zips, {len(ranked_zips)} in submissions/_today, missing required: "
        f"{missing or 'none'}")
    for n, r in enumerate(ranked_zips[:12], 1):
        d = "-" if r["dev"] is None else f"{r['dev']:.4f}"
        log(f"  {n:2d} {r['tag']:26s} {r['kind']:10s} {d}")
    log(f"BATCH DONE: {len(ranked_zips)} zips ready")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["all"])
    ap.add_argument("--procs", type=int, default=3)
    ap.add_argument("--n-dev", type=int, default=200)
    ap.add_argument("--score-min", type=float, default=45, help="BERTScore budget for new variants")
    ap.add_argument("--min-zips", type=int, default=45)
    ap.add_argument("--per-day", type=int, default=40)
    ap.add_argument("--no-bert", action="store_true")
    a = ap.parse_args()
    cmd_all(a)


if __name__ == "__main__":
    main()
