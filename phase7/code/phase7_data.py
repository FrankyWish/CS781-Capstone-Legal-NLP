#!/usr/bin/env python3
"""CS-781 Phase 7 -- Prior Case Retrieval: data, integrity gate, scoring, submissions.

    python phase7_data.py gate                    measure the data and every integrity risk
    python phase7_data.py score --pred F [--split dev]
    python phase7_data.py ids --split dev         candidate_ids.json for the official scorer

WHAT THE METRIC ACTUALLY IS.  The scorer forces exactly K=7 unique in-pool IDs per
query, so for Q queries with gold sets g_q:

    TP = hits,   FP = 7Q - TP,   FN = sum|g_q| - TP
    Micro-F1@7 = 2TP / (2TP + FP + FN) = 2TP / (7Q + sum|g_q|)

The denominator does not depend on the prediction at all.  Consequences that drive
the whole phase:

  * Micro-F1@7 is an AFFINE FUNCTION OF TOTAL HITS.  So are Micro-P@7 (TP/7Q) and
    Micro-R@7 (TP/sum|g|).  All three rank systems identically; there is one number.
  * There is NO precision/recall trade-off and NO K to calibrate.  Fewer than 7 IDs is
    rejected by the validator, and a 7th guess that is wrong costs exactly what an
    empty slot costs.  Always fill all 7.
  * The objective is exactly "expected number of golds in the top 7", so ranking by
    P(relevant) and cutting at 7 is optimal.  A query with many golds is worth more
    hits than a query with one, so a ranker that is good on long-gold queries scores
    higher at identical mean average precision.
  * The reachable ceiling is 2*sum(min(|g_q|,7)) / (7Q + sum|g_q|), printed by the
    gate.  Queries with more than 7 golds cap what any system can reach.

INTEGRITY.  The public source is almost certainly IL-TUR `pcr` / IL-PCR (Joshi et al.
2023), whose test relevance labels are public.  NEVER download them.  This module only
MEASURES leaks: dev queries that are copies of test queries (Phases 4 and 5 both
shipped that), queries sitting inside a candidate pool, train->test pool overlap, and
the citation-popularity prior available from the train labels.  Anything built on the
Phase-1 corpus is a `lookup` system (phase7_overlap.py measures it) and is not
submitted without Nithin's explicit go-ahead.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import re
import shutil
import subprocess
import sys
import zipfile

import numpy as np

BASE = os.environ.get("CS781_BASE", "/content/drive/MyDrive/cs781_capstone")
P1 = os.environ.get("CS781_P1", f"{BASE}/phase1")          # read-only
P7 = os.environ.get("CS781_P7", f"{BASE}/phase7")          # everything we write
LOCAL = os.environ.get("CS781_P7_DATA")                    # local copy of Phase_7_PCR
REPO = "Exploration-Lab/CS-781-Capstone"
SUB = "Phase_7_PCR"
HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.normpath(os.path.join(HERE, "..", "..", "tools"))

K = 7
SEED = 781
WORD = re.compile(r"[a-z0-9]+")
KINDS = ("model", "lexical", "hybrid", "prior", "lookup")

# Citation masks: IL-PCR replaces the cited-case text inside a query.  Whichever of
# these appears in the queries and not in the candidates is the mask; the gate reports
# the counts rather than assuming a particular token.
MASK_PATS = {
    "angle_citation": re.compile(r"<\s*/?\s*citation[^>]*>", re.I),
    "bracket_citation": re.compile(r"\[\s*\?*\s*citation\s*\?*\s*\]", re.I),
    "masked_word": re.compile(r"\bmasked\s*(citation|case|reference)\b", re.I),
    "underscores": re.compile(r"_{4,}"),
    "section_tag": re.compile(r"\[\s*\?*\s*(fact|issue|argument)s?\s*\?*\s*\]", re.I),
}


def log(*a):
    print(*a, flush=True)


def atomic_json(path, obj, indent=1):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)
    os.replace(tmp, path)


# --------------------------------------------------------------------- load --
def fetch(name):
    """Phase-7 file at the spec's path, or found by name if that path moved."""
    if LOCAL:
        for p in (os.path.join(LOCAL, name), os.path.join(LOCAL, os.path.basename(name))):
            if os.path.exists(p):
                return p
    from huggingface_hub import hf_hub_download
    try:
        return hf_hub_download(REPO, f"{SUB}/{name}", repo_type="dataset")
    except Exception as e:
        from huggingface_hub import HfApi
        want = os.path.basename(name).lower()
        files = [f for f in HfApi().list_repo_files(REPO, repo_type="dataset")
                 if any(k in f.lower() for k in ("phase_7", "phase7", "pcr"))]
        hit = [f for f in files if os.path.basename(f).lower() == want]
        if not hit:
            raise SystemExit(f"{SUB}/{name} not found ({e}); Phase-7 files in the repo: {files}")
        log(f"  {SUB}/{name} not at the spec's path; using {hit[0]}")
        return hf_hub_download(REPO, hit[0], repo_type="dataset")


def _sents(v):
    """The `text` field: a list of sentences, or a string that gets split into them."""
    if isinstance(v, list):
        return [" ".join(str(s).split()) for s in v if str(s).strip()]
    if isinstance(v, dict):
        return _sents(max(v.values(), key=lambda x: len(str(x)), default=""))
    s = "" if v is None else str(v)
    return [" ".join(x.split()) for x in re.split(r"(?<=[.;:?!])\s+|\n+", s) if x.strip()]


_TEXT_KEYS = ("text", "document", "doc", "sentences", "sents", "judgement", "judgment", "body")
_GOLD_KEYS = ("relevant_candidates", "relevant", "candidates", "gold", "relevant_cases",
              "citations", "labels")
_ID_KEYS = ("id", "query_id", "case_id", "doc_id", "name")


def _records(path):
    """-> [(id, dict|str)] from either an object keyed by ID or a list of records."""
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    try:
        x = json.loads(raw)
    except json.JSONDecodeError:
        x = [json.loads(l) for l in raw.splitlines() if l.strip()]
    if isinstance(x, dict):
        return [(str(k), v) for k, v in x.items()]
    out = []
    for i, r in enumerate(x):
        if not isinstance(r, dict):
            raise SystemExit(f"{path}: record {i} is not an object")
        low = {str(k).lower(): v for k, v in r.items()}
        idk = next((k for k in _ID_KEYS if k in low), None)
        if idk is None:
            raise SystemExit(f"{path}: record {i} has no id field (keys {sorted(low)})")
        out.append((str(low[idk]), low))
    return out


def _pick_text(v, path):
    """-> (sentences, lowercased record).  The field is found in the file, so a renamed
    one cannot stop an unattended run; only a record with no text at all does."""
    if not isinstance(v, dict):
        return _sents(v), {}
    low = {str(k).lower(): x for k, x in v.items()}
    for k in _TEXT_KEYS:
        if low.get(k):
            return _sents(low[k]), low
    cand = [(len(str(x)), k) for k, x in low.items()
            if isinstance(x, (str, list)) and k not in _ID_KEYS]
    if not cand:
        raise SystemExit(f"{path}: no text field (keys {sorted(low)})")
    return _sents(low[max(cand)[1]]), low


def _gold(low):
    if not isinstance(low, dict):
        return None
    for k in _GOLD_KEYS:
        v = low.get(k)
        if isinstance(v, list):
            return [str(x) for x in v if str(x).strip()]
    return None


_CACHE = {}


def read_queries(split):
    """[{id, sents, gold|None}] in file order.  test gold is always None."""
    path = fetch({"train": "Train/train.json", "dev": "Dev/dev.json",
                  "test": "Test/test.json"}[split])
    recs = _records(path)
    out, extra = [], set()
    for qid, v in recs:
        sents, low = _pick_text(v, path)
        out.append(dict(id=qid, sents=sents, gold=None if split == "test" else _gold(low)))
        extra |= set(low)
    if split == "test":
        leftover = sorted(extra - set(_TEXT_KEYS) - set(_ID_KEYS))
        if leftover:
            log(f"  !! test.json has extra fields {leftover}: names only, values never read")
        if any(_gold(v) is not None for _, v in recs):
            log("  !! test.json APPEARS TO CARRY RELEVANCE LABELS -- they are never read. "
                "Report this; every system stays label-free.")
    else:
        n_g = sum(r["gold"] is not None for r in out)
        if n_g < len(out):
            log(f"  !! {len(out) - n_g}/{len(out)} {split} queries have no gold field")
    return out


def read_candidates(split):
    """{id: [sentences]} for the split's own pool."""
    path = fetch({"train": "Train/train_candidates.json", "dev": "Dev/dev_candidates.json",
                  "test": "Test/test_candidates.json"}[split])
    return {cid: _pick_text(v, path)[0] for cid, v in _records(path)}


def queries(split):
    """train/dev/test as released; `devx` is dev with every contaminated query dropped."""
    if split == "devx":
        if "devx" not in _CACHE:
            bad = contaminated_dev_ids()
            _CACHE["devx"] = [r for r in queries("dev") if r["id"] not in bad]
        return _CACHE["devx"]
    key = f"q_{split}"
    if key not in _CACHE:
        _CACHE[key] = read_queries(split)
    return _CACHE[key]


def candidates(split):
    key = f"c_{'dev' if split == 'devx' else split}"
    if key not in _CACHE:
        _CACHE[key] = read_candidates("dev" if split == "devx" else split)
    return _CACHE[key]


def golds(split):
    """{qid: [gold ids]} exactly as released, INCLUDING any label outside the split's
    own candidate pool.  The official scorer validates predictions against the pool but
    not the truth, so an unreachable label still lands in FN and still sits in the
    denominator -- filtering it here would quietly inflate every dev number we report.
    `golds_in_pool` is the reachable subset, for diagnostics only."""
    return {r["id"]: list(r["gold"] or []) for r in queries(split)}


def golds_in_pool(split):
    pool = set(candidates(split))
    return {q: [c for c in v if c in pool] for q, v in golds(split).items()}


def texts(split):
    """{id: joined text} for queries, and the same for the pool."""
    return {r["id"]: " ".join(r["sents"]) for r in queries(split)}


def nwords(t):
    return len(WORD.findall(t.lower()))


def norm_key(sents, n=3000):
    t = sents if isinstance(sents, str) else " ".join(sents)
    return " ".join(WORD.findall(t.lower()))[:n]


def pct(a, qs=(0, 5, 25, 50, 75, 95, 100)):
    a = np.asarray(a, dtype=float)
    if not a.size:
        return {}
    return {f"p{q}": round(float(np.percentile(a, q)), 3) for q in qs} | \
           {"mean": round(float(a.mean()), 3), "n": int(a.size)}


def contaminated_dev_ids():
    """Dev queries that are a copy of a TEST query.  Selecting on them is selecting on
    the test set, and Phase 4's and Phase 5's dev.json both did exactly this.  Dropped
    from `devx`, which every selection decision uses, and counted in the report."""
    f = f"{P7}/results/dev_contaminated.json"
    if os.path.exists(f):
        with open(f) as fh:
            return set(json.load(fh)["ids"])
    tk = {norm_key(r["sents"]) for r in queries("test")}
    bad = sorted(r["id"] for r in queries("dev") if norm_key(r["sents"]) in tk)
    atomic_json(f, dict(n=len(bad), ids=bad))
    if bad:
        log(f"  !! {len(bad)} dev queries are copies of test queries -> dropped from devx")
    return set(bad)


# ------------------------------------------------------------------ scoring --
def score_ranking(pred, gold, k=K, reachable=None):
    """score() of evaluate_phase7.py, plus the hit count, the ceiling and per-query hits.

    pred: {qid: ranked list}.  gold: {qid: [ids]} as released.  `reachable` is the
    in-pool subset when some labels are unreachable, so max_hits is the real ceiling
    rather than min(|gold|, 7).  Every gold qid must be predicted."""
    miss = [q for q in gold if q not in pred]
    if miss:
        raise SystemExit(f"score: {len(miss)} gold queries have no prediction (first {miss[:3]})")
    tp = fp = fn = 0
    per = {}
    for q, labels in gold.items():
        g = set(labels)
        p = set(pred[q][:k])
        h = len(g & p)
        tp += h
        fp += len(p - g)
        fn += len(g - p)
        per[q] = [h, len(g)]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * tp) / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    reach = reachable if reachable is not None else gold
    cap = sum(min(len(reach.get(q, gold[q])), k) for q in gold)
    den = k * len(gold) + sum(len(v) for v in gold.values())
    return dict(f1=f1, precision=precision, recall=recall, tp=tp, fp=fp, fn=fn,
                n_queries=len(gold), hits=tp, max_hits=cap, denominator=den,
                ceiling_f1=(2 * cap / den if den else 0.0), per_query=per)


def brief(s):
    return (f"f1 {s['f1']:.4f} (P {s['precision']:.4f} R {s['recall']:.4f}) "
            f"hits {s['hits']}/{s['max_hits']} over {s['n_queries']} queries")


def recall_at(ranked, gold, ks=(1, 5, 7, 10, 20, 50, 100)):
    """Diagnostics only: where the golds sit in a full ranking.  recall@100 is the
    reranking headroom -- a reranker can only reach what the retriever brought back."""
    out = {}
    for k in ks:
        tot = hit = 0
        for q, g in gold.items():
            if not g:
                continue
            tot += len(g)
            hit += len(set(g) & set(ranked.get(q, [])[:k]))
        out[f"r@{k}"] = round(hit / tot, 4) if tot else 0.0
    return out


def write_candidate_ids(split):
    """candidate_ids.json for the official evaluator (a flat JSON list)."""
    p = f"{P7}/results/{split}_candidate_ids.json"
    atomic_json(p, sorted(candidates(split)), indent=0)
    return p


def official_score(pred_path, split="dev"):
    """Run spec/evaluation_scripts/evaluate_phase7.py so our number IS the number."""
    ev = os.path.normpath(os.path.join(HERE, "..", "..", "spec", "evaluation_scripts",
                                       "evaluate_phase7.py"))
    if not os.path.exists(ev):
        log(f"  (official evaluator not at {ev})")
        return None
    truth = f"{P7}/results/{split}_truth.json"
    atomic_json(truth, golds(split), indent=0)
    cmd = [sys.executable, "-I", ev, "--truth", truth, "--prediction", pred_path,
           "--candidate-ids", write_candidate_ids(split)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    out = (r.stdout + r.stderr).strip()
    if r.returncode != 0:
        log("  OFFICIAL EVALUATOR FAILED:\n" + out)
        return None
    m = re.search(r"Micro-F1@7:\s*([0-9.]+)", out)
    return float(m.group(1)) if m else None


# -------------------------------------------------------------- submissions --
def popularity_order(pool_split, label_splits=("train", "dev")):
    """Pool candidates ordered by how often the labelled splits cite them.

    A MEASURED prior, not a model: if a few leading judgments are cited by many
    queries, `always return the 7 most-cited` is a real floor.  Any system that uses
    it is reported as kind='prior'; the gate prints what it is worth on dev."""
    pool = set(candidates(pool_split))
    cnt = collections.Counter()
    for s in label_splits:
        for g in golds(s).values():
            cnt.update(c for c in g if c in pool)
    ranked = [c for c, _ in cnt.most_common()]
    return ranked + sorted(pool - set(ranked))


def write_submission(tag, ranking, note="", kind="model", dev_scores=None, out_root=None):
    """{qid: ranked ids} -> submissions/<tag>/{prediction.json, submission.zip} + _upload.

    Every query gets exactly 7 unique in-pool IDs.  Short rankings are backfilled from
    the most-cited pool candidates: the scorer rejects the whole file over one
    wrong-length list, and a wrong 7th guess costs exactly what a blank slot costs."""
    if kind not in KINDS:
        raise SystemExit(f"kind must be one of {KINDS}")
    pool = candidates("test")
    fill = popularity_order("test")
    qs = queries("test")
    preds, patched, dropped = {}, 0, 0
    for r in qs:
        seen, out = set(), []
        for c in (ranking.get(r["id"]) or []):
            c = str(c)
            if c not in pool:
                dropped += 1
                continue
            if c in seen:
                continue
            seen.add(c)
            out.append(c)
            if len(out) == K:
                break
        if len(out) < K:
            patched += 1
            for c in fill:
                if c not in seen:
                    seen.add(c)
                    out.append(c)
                if len(out) == K:
                    break
        assert len(out) == K == len(set(out)), (r["id"], out)
        preds[r["id"]] = out
    if dropped:
        log(f"  !! {dropped} predicted IDs were outside the test pool and were dropped")
    if patched:
        log(f"  !! {patched}/{len(qs)} queries were backfilled up to {K} candidates")
    out_root = out_root or f"{P7}/submissions"
    d = os.path.join(out_root, tag)
    os.makedirs(d, exist_ok=True)
    pj = os.path.join(d, "prediction.json")
    atomic_json(pj, preds, indent=1)
    ok = validate(pj)
    zp = os.path.join(d, "submission.zip")
    with zipfile.ZipFile(zp, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(pj, arcname="prediction.json")
    with zipfile.ZipFile(zp) as z:                     # the zip must be flat
        assert z.namelist() == ["prediction.json"]
        assert json.loads(z.read("prediction.json")) == preds
    up = os.path.join(out_root, "_upload")
    os.makedirs(up, exist_ok=True)
    shutil.copyfile(zp, os.path.join(up, f"submission_{tag}.zip"))
    use = collections.Counter(c for v in preds.values() for c in v)
    info = dict(tag=tag, note=note, kind=kind, n_queries=len(preds), validated=ok,
                distinct_candidates=len(use), backfilled=patched, out_of_pool=dropped,
                top_candidate_share=round(use.most_common(1)[0][1] / len(preds), 4) if use else 0.0,
                dev=dev_scores or {})
    atomic_json(os.path.join(d, "info.json"), info)
    log(f"  WROTE {zp}  ({len(preds)} queries, {len(use)} distinct candidates, "
        f"{kind}, validated={ok})")
    return zp


def validate(pj):
    tool = os.path.join(TOOLS, "cs781_submit.py")
    if not os.path.exists(tool):
        log(f"  (validator not at {tool}; format checked in-process only)")
        return None
    cmd = [sys.executable, "-I", tool, "validate", "--phase", "7", "--prediction", pj]
    try:
        cmd += ["--candidates", fetch("Test/test_candidates.json"),
                "--test", fetch("Test/test.json")]
    except Exception:
        pass
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        log("  VALIDATION FAILED:\n" + (r.stdout + r.stderr).strip())
        return False
    return True


# -------------------------------------------------------------- the gate ----
def gate(a):
    os.makedirs(f"{P7}/results", exist_ok=True)
    G = {}
    splits = ["train", "dev", "test"]
    Q = {s: queries(s) for s in splits}
    C = {s: candidates(s) for s in splits}

    # --- 1. shape -----------------------------------------------------------
    G["n"] = {s: dict(queries=len(Q[s]), candidates=len(C[s])) for s in splits}
    for s in splits:
        qw = pct([nwords(" ".join(r["sents"])) for r in Q[s]])
        qs = pct([len(r["sents"]) for r in Q[s]])
        cw = pct([nwords(" ".join(v)) for v in C[s].values()])
        G[f"{s}_query_words"], G[f"{s}_query_sents"], G[f"{s}_cand_words"] = qw, qs, cw
        log(f"{s:5s}: {len(Q[s]):5d} queries (median {qw.get('p50', 0):.0f} words, "
            f"{qs.get('p50', 0):.0f} sents) | {len(C[s]):5d} candidates "
            f"(median {cw.get('p50', 0):.0f} words)")
    log(f"  scoring cost per system: {len(Q['test'])} test queries x {len(C['test'])} "
        f"candidates = {len(Q['test']) * len(C['test']):,} pairs")

    # --- 2. gold sets: the ceiling, and what 7 is worth ---------------------
    for s in ("train", "dev"):
        pool = set(C[s])
        g, gp = golds(s), golds_in_pool(s)
        sz = [len(v) for v in g.values()]
        oos = sum(len(g[q]) - len(gp[q]) for q in g)
        over = sum(1 for v in g.values() if len(v) > K)
        cap = sum(min(len(gp[q]), K) for q in g)
        den = K * len(g) + sum(sz)
        G[f"{s}_gold"] = dict(sizes=pct(sz), total=int(sum(sz)), queries=len(g),
                              over_K=over, out_of_pool=oos,
                              zero=int(sum(1 for n in sz if n == 0)),
                              ceiling_f1=round(2 * cap / den, 6) if den else 0.0,
                              expected_random_f1=round(
                                  2 * sum(sz) * K / max(len(pool), 1) / den, 6) if den else 0.0)
        gg = G[f"{s}_gold"]
        log(f"{s:5s} gold: {gg['total']} labels, {pct(sz).get('mean', 0):.2f}/query "
            f"(p50 {pct(sz).get('p50', 0):.0f}, p95 {pct(sz).get('p95', 0):.0f}, "
            f"max {pct(sz).get('p100', 0):.0f}) | {over} queries have >{K} golds | "
            f"{gg['zero']} have none")
        log(f"{s:5s} CEILING Micro-F1@7 = {gg['ceiling_f1']:.4f} (unbeatable); "
            f"random 7 ~ {gg['expected_random_f1']:.4f}")
        if oos:
            log(f"  !! {oos} {s} gold labels are NOT in the split's own pool: unreachable, "
                f"and already counted against the ceiling")

    # --- 3. contamination between the splits --------------------------------
    qkey = {s: {norm_key(r["sents"]): r["id"] for r in Q[s]} for s in splits}
    ckey = {s: {norm_key(v): c for c, v in C[s].items()} for s in splits}
    ov = {}
    for a_, b_ in (("dev", "test"), ("train", "test"), ("train", "dev")):
        sq = sorted(set(qkey[a_]) & set(qkey[b_]))
        ov[f"{a_}_queries_in_{b_}_queries"] = dict(
            n=len(sq), examples=[[qkey[a_][k], qkey[b_][k]] for k in sq[:8]])
        sc = sorted(set(ckey[a_]) & set(ckey[b_]))
        ov[f"{a_}_cands_in_{b_}_cands"] = dict(
            n=len(sc), frac_of_b=round(len(sc) / max(len(C[b_]), 1), 4),
            by_id=len(set(C[a_]) & set(C[b_])),
            examples=[[ckey[a_][k], ckey[b_][k]] for k in sc[:8]])
    for s in splits:
        hit = sorted(set(qkey[s]) & set(ckey[s]))
        ov[f"{s}_queries_in_{s}_pool"] = dict(
            n=len(hit), by_id=len(set(x["id"] for x in Q[s]) & set(C[s])),
            examples=[[qkey[s][k], ckey[s][k]] for k in hit[:8]])
    G["overlap"] = ov
    log(f"contamination: dev queries copying a test query "
        f"{ov['dev_queries_in_test_queries']['n']} | train queries copying a test query "
        f"{ov['train_queries_in_test_queries']['n']} | train copying dev "
        f"{ov['train_queries_in_dev_queries']['n']}")
    tc = ov["train_cands_in_test_cands"]
    log(f"pool overlap: the train pool covers {tc['frac_of_b'] * 100:.1f}% of the test pool "
        f"({tc['n']} by text, {tc['by_id']} by ID) | dev pool covers "
        f"{ov['dev_cands_in_test_cands']['frac_of_b'] * 100:.1f}%")
    sm = ov["test_queries_in_test_pool"]
    log(f"self-match: {sm['n']} test queries are also test candidates by text, "
        f"{sm['by_id']} by ID")
    if ov["dev_queries_in_test_queries"]["n"]:
        log("  !! dev is contaminated with test queries -- devx drops them and every "
            "selection number is computed on devx")
    if sm["n"] or sm["by_id"]:
        log("  !! a test query is also a test CANDIDATE: phase7_rank drops exact "
            "self-matches from a query's own ranking")
    contaminated_dev_ids()

    # --- 4. the citation mask ----------------------------------------------
    G["masks"] = {}
    for s in splits:
        qt = " ".join(" ".join(r["sents"]) for r in Q[s][:400])
        ct = " ".join(" ".join(v) for v in list(C[s].values())[:400])
        found = {nm: dict(query=len(p.findall(qt)), candidate=len(p.findall(ct)))
                 for nm, p in MASK_PATS.items() if p.search(qt) or p.search(ct)}
        G["masks"][s] = found
        if found:
            log(f"{s:5s} masks: {found}")
    if any(G["masks"].values()):
        log("  !! citations are masked in the queries.  An UNMASKED copy of a query "
            "judgment in the Phase-1 corpus would name the cases it cites, i.e. hold the "
            "answer key.  phase7_overlap.py MEASURES that; nothing is built on it without "
            "Nithin's explicit go-ahead, and such a system is reported as kind='lookup'.")

    # --- 5. the popularity prior, measured on dev ---------------------------
    dk, dkp = golds("devx"), golds_in_pool("devx")
    pop_dev = popularity_order("dev", label_splits=("train",))
    s_pop = score_ranking({q: pop_dev[:K] for q in dk}, dk, reachable=dkp)
    rng = np.random.default_rng(SEED)
    pool_dev = sorted(C["dev"])
    s_rnd = score_ranking(
        {q: [pool_dev[i] for i in rng.choice(len(pool_dev), K, replace=False)] for q in dk},
        dk, reachable=dkp)
    cnt = collections.Counter(c for g in golds("train").values() for c in g)
    in_test = sum(1 for c in cnt if c in C["test"])
    G["prior"] = dict(top7_from_train_f1=round(s_pop["f1"], 6),
                      random_f1=round(s_rnd["f1"], 6), dev_queries_scored=len(dk),
                      train_citation_counts=pct(list(cnt.values())) if cnt else {},
                      distinct_cited_in_train=len(cnt),
                      cited_in_train_and_in_test_pool=in_test,
                      top10=[[c, n] for c, n in cnt.most_common(10)])
    log(f"prior: the 7 most-cited train candidates score {brief(s_pop)} on devx")
    log(f"random 7 from the dev pool: {brief(s_rnd)}")
    log(f"  {len(cnt)} distinct candidates are cited in train ({in_test} of them are also "
        f"in the TEST pool); counts {G['prior']['train_citation_counts']}")

    # --- 6. is there lexical signal at all? ---------------------------------
    gold_dev = {q: v for q, v in dkp.items() if v}
    sample = sorted(gold_dev)[:a.sample]
    if sample:
        qtext = {r["id"]: r["sents"] for r in Q["dev"]}
        docs = {c: set(WORD.findall(" ".join(v).lower())) for c, v in C["dev"].items()}
        df = collections.Counter()
        for t in docs.values():
            df.update(t)
        N = max(len(docs), 1)
        idf = {t: math.log(1 + (N - n + 0.5) / (n + 0.5)) for t, n in df.items()}
        w = lambda ts: sum(idf.get(t, 0.0) for t in ts)
        jg, jr = [], []
        others = sorted(docs)
        for i, q in enumerate(sample):
            qt = set(WORD.findall(" ".join(qtext[q]).lower()))
            for c in gold_dev[q][:3]:
                if c in docs:
                    jg.append(w(qt & docs[c]) / max(w(qt | docs[c]), 1e-9))
            for c in others[i % max(len(others) - 3, 1):][:3]:
                if c not in gold_dev[q]:
                    jr.append(w(qt & docs[c]) / max(w(qt | docs[c]), 1e-9))
        G["lexical_signal"] = dict(gold_idf_jaccard=pct(jg), other_idf_jaccard=pct(jr))
        log(f"idf-weighted Jaccard: gold pairs {pct(jg).get('mean', 0):.4f} vs arbitrary "
            f"pairs {pct(jr).get('mean', 0):.4f}  (a lexical ranker has signal iff these differ)")

    # --- 7. fingerprint: identify the source, then never touch its labels ---
    G["fingerprint"] = [dict(id=r["id"], n_sents=len(r["sents"]), n_gold=len(r["gold"] or []),
                             head=" ".join(r["sents"])[:300]) for r in Q["train"][:3]]
    G["fingerprint_cand"] = [dict(id=c, head=" ".join(v)[:220])
                             for c, v in list(C["train"].items())[:2]]
    atomic_json(f"{P7}/results/gate.json", G)
    log("\n--- source fingerprint (identify the dataset; its test labels are NOT downloaded) ---")
    for r in G["fingerprint"]:
        log(f"[{r['id']}] {r['n_sents']} sents, {r['n_gold']} golds: {r['head'][:200]}")
    for r in G["fingerprint_cand"]:
        log(f"cand [{r['id']}]: {r['head'][:200]}")
    log(f"\nwrote {P7}/results/gate.json")
    return G


def cmd_score(a):
    with open(a.pred, encoding="utf-8") as f:
        p = json.load(f)
    s = score_ranking(p, golds(a.split))
    log(brief(s) + f" | ceiling {s['ceiling_f1']:.4f}")
    if not a.no_official:
        o = official_score(a.pred, a.split)
        if o is not None:
            log(f"official evaluator Micro-F1@7 {o:.10f} vs ours {s['f1']:.10f} -> "
                f"{'MATCH' if abs(o - s['f1']) < 1e-9 else 'MISMATCH'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    g = sp.add_parser("gate")
    g.add_argument("--sample", type=int, default=120, help="dev queries for the lexical probe")
    g.set_defaults(fn=gate)
    s = sp.add_parser("score")
    s.add_argument("--pred", required=True)
    s.add_argument("--split", default="dev")
    s.add_argument("--no-official", action="store_true")
    s.set_defaults(fn=cmd_score)
    i = sp.add_parser("ids")
    i.add_argument("--split", default="dev")
    i.set_defaults(fn=lambda a: log(write_candidate_ids(a.split)))
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
