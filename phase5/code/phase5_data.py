#!/usr/bin/env python3
"""CS-781 Phase 5 (legal statute identification) -- data, gate, scoring, decoding, zips.

    python phase5_data.py gate            # stats + integrity checks -> results/gate.json
    python phase5_data.py score --pred prediction.json --split dev

RUN `gate` FIRST. Everything else imports from here. CPU only.

Scoring is a re-implementation of spec/evaluation_scripts/evaluate_phase5.py:
the label dimension is max(gold label) + 1 over the REFERENCE set, and Macro-F1
averages over every ID in 0..n-1 -- including IDs absent from the reference,
which score 0 for everybody (sklearn zero_division=0). Two consequences that
decide how predictions are made:

  * A predicted ID >= n makes the official scorer FAIL the whole submission.
    n comes from the private test gold. We never emit an ID above the highest ID
    seen in train+dev (MAX_LABEL), and `safe_max` can lower that further.
  * A statute absent from the test gold contributes 0 whatever we predict, so
    predicting it costs nothing in Macro-F1 (only Micro-F1). A statute present
    in the gold but never predicted is a guaranteed 0: under Macro-F1 every
    statute should be predicted for at least a few documents.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import re
import shutil
import sys
import zipfile

import numpy as np

BASE = os.environ.get("CS781_BASE", "/content/drive/MyDrive/cs781_capstone")
P1 = os.environ.get("CS781_P1", f"{BASE}/phase1")        # read-only
P5 = os.environ.get("CS781_P5", f"{BASE}/phase5")        # everything we write
LOCAL = os.environ.get("CS781_P5_DATA")                  # optional local copy of Phase_5_LSI
REPO = "Exploration-Lab/CS-781-Capstone"
SUB = "Phase_5_LSI"
HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.normpath(os.path.join(HERE, "..", "..", "tools"))


WORD = re.compile(r"[a-z0-9]+")


def log(*a):
    print(*a, flush=True)


# --------------------------------------------------------------------- load --
def fetch(name):
    if LOCAL:
        for p in (os.path.join(LOCAL, name), os.path.join(LOCAL, os.path.basename(name))):
            if os.path.exists(p):
                return p
    from huggingface_hub import hf_hub_download
    return hf_hub_download(REPO, f"{SUB}/{name}", repo_type="dataset")


def _text(r):
    t = r.get("text")
    if isinstance(t, list):                      # sentence lists -> one string
        return "\n".join(str(s) for s in t)
    return str(t)


def read_silver():
    """external/silver.jsonl (phase5_external.py): [{id, text, labels, mask}]; mask = unsupervised label IDs."""
    out = []
    with open(f"{P5}/external/silver.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out.append(dict(id=r["id"], text=r["text"], labels=r["labels"], mask=r.get("mask", [])))
    return out


def norm_key(text):
    return " ".join(WORD.findall(text.lower()))[:3000]


def read_split(split):
    """[{id, text(str), labels(list[int]) | None}] in file order. Test labels are never read.

    dev is returned WITHOUT any record whose text is a copy of a test record
    (Phase 4's dev.json held labelled copies of every test document; selecting
    thresholds on those would be tuning on the test set).
    """
    if split == "silver":
        return read_silver()
    if split == "dev":
        tk = {norm_key(r["text"]) for r in load("test")}
        dv = read_split("dev_raw")
        keep = [r for r in dv if norm_key(r["text"]) not in tk]
        if len(keep) < len(dv):
            log(f"  dev: dropped {len(dv) - len(keep)} records that are copies of test documents")
        return keep
    name = {"train": "Train/train.json", "dev_raw": "Train/dev.json", "test": "Test/test.json"}[split]
    with open(fetch(name), encoding="utf-8") as f:
        raw = f.read()
    try:
        x = json.loads(raw)
    except json.JSONDecodeError:
        x = [json.loads(l) for l in raw.splitlines() if l.strip()]
    if isinstance(x, dict):
        x = [dict(v, id=k) if isinstance(v, dict) else {"id": k, "text": v} for k, v in x.items()]
    out = []
    for r in x:
        lab = None
        if split != "test":
            lab = sorted({int(v) for v in r["labels"]})
        out.append(dict(id=str(r["id"]), text=_text(r), labels=lab))
    return out


_CACHE = {}


def load(split):
    if split not in _CACHE:
        _CACHE[split] = read_split(split)
    return _CACHE[split]


def label_space():
    """(n_labels, train counts). n = max ID over train+dev + 1."""
    tr, dv = load("train"), load("dev")
    mx = max(max(r["labels"], default=-1) for r in tr + dv)
    n = mx + 1
    cnt = np.zeros(n, dtype=np.int64)
    for r in tr:
        for y in r["labels"]:
            cnt[y] += 1
    return n, cnt


def multi_hot(recs, n):
    Y = np.zeros((len(recs), n), dtype=np.uint8)
    for i, r in enumerate(recs):
        for y in r["labels"]:
            if y < n:
                Y[i, y] = 1
    return Y


# ------------------------------------------------------------------ scoring --
def f1_per_label(Y, P):
    """Y, P: (N, n) {0,1}. sklearn's F1 with zero_division=0, per label."""
    Y = Y.astype(bool)
    P = P.astype(bool)
    tp = (Y & P).sum(0).astype(np.float64)
    fp = (~Y & P).sum(0)
    fn = (Y & ~P).sum(0)
    den = 2 * tp + fp + fn
    return np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0)


def scores(Y, P):
    """Exactly the three numbers evaluate_phase5.py prints (n taken from Y's width).

    Y's width must be the reference label dimension (max gold ID + 1), as in the
    official scorer; callers pass Y[:, :n_ref].
    """
    Yb, Pb = Y.astype(bool), P.astype(bool)
    f1 = f1_per_label(Yb, Pb)
    tp = (Yb & Pb).sum()
    fp = (~Yb & Pb).sum()
    fn = (Yb & ~Pb).sum()
    micro = 2 * tp / max(1, 2 * tp + fp + fn)
    subset = float((Yb == Pb).all(1).mean())
    return dict(macro_f1=float(f1.mean()), micro_f1=float(micro), subset_acc=subset)


def ref_width(Y):
    """The evaluator's label dimension for this reference set: max gold ID + 1."""
    cols = np.nonzero(Y.any(0))[0]
    return int(cols.max()) + 1 if len(cols) else Y.shape[1]


def score_on(Y, P):
    n = ref_width(Y)
    return scores(Y[:, :n], P[:, :n])


# --------------------------------------------------------------- decoding ----
def decode(S, thr, min_k=1, max_k=None):
    """Scores S (N, n) -> predictions (N, n) uint8.

    thr: scalar or (n,) vector. min_k: if fewer than min_k labels clear their
    threshold, take the top labels by margin (S - thr) until min_k are set; every
    training document has >= 1 statute, so an empty prediction is always wrong.
    max_k caps the number of labels per document by margin.
    """
    thr = np.broadcast_to(np.asarray(thr, dtype=np.float32), S.shape[1:])
    M = S - thr
    P = (M >= 0)
    if min_k:
        k = P.sum(1)
        need = np.nonzero(k < min_k)[0]
        if len(need):
            top = np.argsort(-M[need], 1)[:, :min_k]
            for r, i in enumerate(need):
                P[i, top[r]] = True
    if max_k:
        over = np.nonzero(P.sum(1) > max_k)[0]
        for i in over:
            keep = np.argsort(-M[i])[:max_k]
            P[i] = False
            P[i, keep] = True
    return P.astype(np.uint8)


def tune_global(S, Y, grid=None, min_k=1):
    """Best single threshold for Macro-F1 on (S, Y)."""
    if grid is None:
        lo, hi = np.quantile(S, [0.5, 0.9999])
        grid = np.unique(np.concatenate([np.linspace(lo, hi, 80), np.quantile(S, np.linspace(0.8, 0.999, 60))]))
    best = (-1.0, None)
    for t in grid:
        m = score_on(Y, decode(S, t, min_k))["macro_f1"]
        if m > best[0]:
            best = (m, float(t))
    return best[1], best[0]


def tune_per_label(S, Y, t_glob, shrink=5.0, min_pos=1):
    """Per-label thresholds shrunk toward the global one.

    For label l with n_l positives in (S, Y), t_l* maximises that label's own F1
    over candidate cut points; the returned threshold is
        (n_l * t_l* + shrink * t_glob) / (n_l + shrink),
    so a label with 2 dev positives barely moves and one with 300 is free.
    Labels with < min_pos positives keep t_glob.
    """
    n = S.shape[1]
    thr = np.full(n, t_glob, dtype=np.float32)
    for l in range(n):
        y = Y[:, l].astype(bool)
        npos = int(y.sum())
        if npos < min_pos:
            continue
        s = S[:, l]
        order = np.argsort(-s)
        ys = y[order]
        tp = np.cumsum(ys)
        k = np.arange(1, len(s) + 1)
        f1 = 2 * tp / (k + npos)
        j = int(np.argmax(f1[: max(npos * 20, 50)]))
        cut = s[order[j]]
        nxt = s[order[j + 1]] if j + 1 < len(s) else cut - 1e-3
        t_star = 0.5 * (cut + nxt)
        thr[l] = (npos * t_star + shrink * t_glob) / (npos + shrink)
    return thr


def threshold_report(S, Y, shrinks=(0, 2, 5, 10, 25), seed=781, log_fn=log):
    """Honest comparison of thresholding strategies by 2-fold CV over the dev documents.

    Tuning on dev and scoring on the same dev rewards overfitting, and per-label
    thresholds overfit rare labels most. Each strategy is tuned on one half of
    dev and scored on the other, both ways, and the two Macro-F1s averaged.
    Returns {strategy: cv_macro_f1} plus the strategy's thresholds refit on all of dev.
    """
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(S))
    halves = [idx[: len(idx) // 2], idx[len(idx) // 2:]]
    res = collections.defaultdict(list)
    for a, b in ((0, 1), (1, 0)):
        A, B = halves[a], halves[b]
        tg, _ = tune_global(S[A], Y[A])
        res["global"].append(score_on(Y[B], decode(S[B], tg))["macro_f1"])
        res["global_nomin"].append(score_on(Y[B], decode(S[B], tg, min_k=0))["macro_f1"])
        for sh in shrinks:
            th = tune_per_label(S[A], Y[A], tg, shrink=sh)
            res[f"perlabel_s{sh}"].append(score_on(Y[B], decode(S[B], th))["macro_f1"])
    cv = {k: float(np.mean(v)) for k, v in res.items()}
    tg, full_g = tune_global(S, Y)
    fits = {"global": np.full(S.shape[1], tg, dtype=np.float32)}
    fits["global_nomin"] = fits["global"]
    for sh in shrinks:
        fits[f"perlabel_s{sh}"] = tune_per_label(S, Y, tg, shrink=sh)
    best = max(cv, key=cv.get)
    log_fn("  threshold strategies (2-fold CV on dev, Macro-F1): "
           + "  ".join(f"{k}={v:.4f}" for k, v in sorted(cv.items(), key=lambda kv: -kv[1])))
    return dict(cv=cv, best=best, thr=fits[best], all_thr=fits, t_global=tg)


# ------------------------------------------------------------ submissions ----
def write_submission(tag, test_recs, P, note="", hybrid=False, dev_scores=None, out_root=None,
                     max_label=None):
    """P (N_test, n) {0,1} in test_recs order -> submissions/<tag>/{prediction.json, submission.zip}.

    Validates with tools/cs781_submit.py before zipping. Never calls `pack`: the
    ledger's daily cap is about uploads, and these zips are built days early.
    """
    out_root = out_root or os.path.join(P5, "submissions")
    d = os.path.join(out_root, tag)
    os.makedirs(d, exist_ok=True)
    if max_label is None:
        max_label = P.shape[1] - 1
    preds = []
    for r, row in zip(test_recs, P):
        ids = [int(i) for i in np.nonzero(row)[0] if i <= max_label]
        preds.append({"id": r["id"], "pred": ids})
    assert len({p["id"] for p in preds}) == len(preds)
    pj = os.path.join(d, "prediction.json")
    with open(pj, "w", encoding="utf-8") as f:
        json.dump(preds, f)
    ok = validate(pj, max_label)
    zp = os.path.join(d, "submission.zip")
    with zipfile.ZipFile(zp, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(pj, arcname="prediction.json")
    with zipfile.ZipFile(zp) as z:
        assert z.namelist() == ["prediction.json"]
        back = json.loads(z.read("prediction.json"))
        assert back == preds
    # upload-ready copy, named by run, in one flat folder (download single files from Drive: no wrapper zip)
    up = os.path.join(out_root, "_upload")
    os.makedirs(up, exist_ok=True)
    shutil.copyfile(zp, os.path.join(up, f"submission_{tag}.zip"))
    per_doc = float(np.mean([len(p["pred"]) for p in preds]))
    info = dict(tag=tag, note=note, hybrid=hybrid, n_docs=len(preds), labels_per_doc=per_doc,
                distinct_labels=int(len({i for p in preds for i in p["pred"]})),
                validated=ok, dev=dev_scores or {}, max_label=int(max_label))
    with open(os.path.join(d, "info.json"), "w") as f:
        json.dump(info, f, indent=1)
    log(f"  WROTE {zp}  ({per_doc:.2f} statutes/doc, {info['distinct_labels']} distinct, "
        f"{'HYBRID' if hybrid else 'model-only'}, validated={ok})")
    return zp


def validate(pj, max_label):
    """Run the shared validator; returns True if it passed."""
    import subprocess
    tool = os.path.join(TOOLS, "cs781_submit.py")
    if not os.path.exists(tool):
        log(f"  (validator not found at {tool}; format checked in-process only)")
        return None
    cmd = [sys.executable, tool, "validate", "--phase", "5", "--prediction", pj,
           "--max-label", str(max_label)]
    try:
        cmd += ["--test", fetch("Test/test.json")]
    except Exception:
        pass
    r = subprocess.run(cmd, capture_output=True, text=True)
    out = (r.stdout + r.stderr).strip()
    if r.returncode != 0:
        log("  VALIDATION FAILED:\n" + out)
        return False
    return True


# ---------------------------------------------------------------- the gate ---
# Statute mentions as they appear in Indian judgments: "Section 302", "Sec. 498-A",
# "S. 34", "u/s 307", "Sections 302, 307 and 34", "Article 21".
SEC_RE = re.compile(
    r"\b(?:sections?|secs?\.?|ss?\.|u/s\.?|u/ss\.?|articles?|arts?\.)\s*"
    r"((?:\d{1,3}\s*[-‑]?\s*[A-Z]{0,2}\b(?:\s*\(\s*\w{1,4}\s*\))*\s*(?:,|/|&|and|or|r/w|read with)?\s*)+)",
    re.IGNORECASE)
NUM_RE = re.compile(r"(\d{1,3})\s*[-‑]?\s*([A-Z]{0,2})\b")
MASK_RE = re.compile(r"<[A-Z_]{3,}>|\[[A-Z_]{3,}\]|\b(?:SECTION|STATUTE|ACT)_?\d*\b")


def mentions(text):
    """Set of section-number strings mentioned in text, e.g. {'302', '498A', '34'}."""
    out = set()
    for m in SEC_RE.finditer(text):
        for n, suf in NUM_RE.findall(m.group(1)):
            out.add(n + suf.upper())
    return out


def ntoks(text):
    return len(WORD.findall(text.lower()))


def pct(a, qs=(0, 10, 25, 50, 75, 90, 99, 100)):
    a = np.asarray(a)
    return {f"p{q}": float(np.percentile(a, q)) for q in qs}


def gate(args):
    os.makedirs(f"{P5}/results", exist_ok=True)
    tr, dv, te = load("train"), load("dev"), load("test")
    n, cnt = label_space()
    G = {}
    G["n_docs"] = dict(train=len(tr), dev=len(dv), test=len(te))
    G["n_labels"] = n
    dcnt = multi_hot(dv, n).sum(0)
    G["labels_unseen_in_train"] = [int(i) for i in np.nonzero(cnt == 0)[0]]
    G["labels_unseen_in_dev"] = [int(i) for i in np.nonzero(dcnt == 0)[0]]
    G["train_label_freq"] = {int(i): int(c) for i, c in enumerate(cnt)}
    G["dev_label_freq"] = {int(i): int(c) for i, c in enumerate(dcnt)}
    for k in (5, 10, 20, 50, 100):
        G[f"labels_lt{k}_train"] = int((cnt < k).sum())
    G["labels_per_doc"] = {s: pct([len(r["labels"]) for r in recs]) | {"mean": float(np.mean([len(r["labels"]) for r in recs]))}
                           for s, recs in (("train", tr), ("dev", dv))}
    G["empty_label_docs"] = {s: int(sum(not r["labels"] for r in recs)) for s, recs in (("train", tr), ("dev", dv))}
    G["words_per_doc"] = {s: pct([ntoks(r["text"]) for r in recs]) for s, recs in (("train", tr), ("dev", dv), ("test", te))}
    G["chars_total_M"] = {s: sum(len(r["text"]) for r in recs) / 1e6 for s, recs in (("train", tr), ("dev", dv), ("test", te))}

    log(f"docs: {G['n_docs']}   labels: n={n}  unseen in train {G['labels_unseen_in_train']}  "
        f"unseen in dev {len(G['labels_unseen_in_dev'])}")
    top = np.argsort(-cnt)
    log("train label freq, head: " + " ".join(f"{i}:{cnt[i]}" for i in top[:15]))
    log("train label freq, tail: " + " ".join(f"{i}:{cnt[i]}" for i in top[-15:]))
    log("labels with < 5/10/20/50/100 train docs: "
        + " / ".join(str(G[f'labels_lt{k}_train']) for k in (5, 10, 20, 50, 100)))
    log(f"labels per doc: {json.dumps(G['labels_per_doc'])}")
    log(f"words per doc: {json.dumps(G['words_per_doc'])}")
    log(f"chars (M): {G['chars_total_M']}")

    # co-occurrence
    Y = multi_hot(tr, n).astype(np.float32)
    C = Y.T @ Y
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if C[i, j] > 0:
                pairs.append((C[i, j] / max(1, min(C[i, i], C[j, j])), int(C[i, j]), i, j))
    pairs.sort(reverse=True)
    G["top_pairs_by_count"] = sorted(((int(c), i, j) for _, c, i, j in pairs), reverse=True)[:30]
    G["top_pairs_by_containment"] = [(round(float(r), 3), c, i, j) for r, c, i, j in pairs if c >= 20][:30]
    sets = collections.Counter(tuple(r["labels"]) for r in tr)
    G["distinct_label_sets"] = len(sets)
    G["top_label_sets"] = [(list(k), v) for k, v in sets.most_common(20)]
    dsets = collections.Counter(tuple(r["labels"]) for r in dv)
    G["dev_sets_seen_in_train_frac"] = float(sum(v for k, v in dsets.items() if k in sets) / len(dv))
    log(f"co-occurrence: top pairs {G['top_pairs_by_count'][:10]}")
    log(f"distinct label sets {len(sets)}; dev docs whose exact set occurs in train: "
        f"{G['dev_sets_seen_in_train_frac']:.3f}")

    # statute mentions left in the text?
    smp = tr[:5000]
    has = [bool(mentions(r["text"])) for r in smp]
    G["frac_docs_with_section_mention"] = float(np.mean(has))
    masks = collections.Counter()
    for r in smp[:2000]:
        masks.update(MASK_RE.findall(r["text"]))
    G["mask_tokens"] = masks.most_common(15)
    log(f"docs with a 'Section N' mention: {G['frac_docs_with_section_mention']:.3f}; "
        f"mask-like tokens: {G['mask_tokens'][:8]}")
    for r in tr[:3]:
        log(f"--- {r['id']} labels={r['labels']} ---\n{r['text'][:700]}\n...")

    # exact / near duplicates across splits
    def key(t):
        return hash(" ".join(WORD.findall(t.lower()))[:2000])
    ktr = {key(r["text"]): r for r in tr}
    G["test_exact_dup_of_train"] = int(sum(key(r["text"]) in ktr for r in te))
    G["dev_exact_dup_of_train"] = int(sum(key(r["text"]) in ktr for r in dv))
    dup_agree = [ktr[key(r["text"])]["labels"] == r["labels"] for r in dv if key(r["text"]) in ktr]
    G["dev_dup_label_agreement"] = float(np.mean(dup_agree)) if dup_agree else None
    dv_raw = read_split("dev_raw")
    kte = {norm_key(r["text"]) for r in te}
    G["dev_raw_records"] = len(dv_raw)
    G["dev_records_copying_test"] = int(sum(norm_key(r["text"]) in kte for r in dv_raw))
    G["dev_unique_texts"] = len({norm_key(r["text"]) for r in dv_raw})
    G["train_unique_texts"] = len({norm_key(r["text"]) for r in tr})
    log(f"dev.json: {len(dv_raw)} records, {G['dev_unique_texts']} unique texts, "
        f"{G['dev_records_copying_test']} copies of TEST documents (dropped from dev everywhere); "
        f"train unique texts {G['train_unique_texts']}/{len(tr)}")
    G["id_overlap"] = dict(train_test=len({r["id"] for r in tr} & {r["id"] for r in te}),
                           train_dev=len({r["id"] for r in tr} & {r["id"] for r in dv}))
    log(f"exact dups (first 2000 normalised chars): test->train {G['test_exact_dup_of_train']}, "
        f"dev->train {G['dev_exact_dup_of_train']} (label agreement {G['dev_dup_label_agreement']}); "
        f"id overlap {G['id_overlap']}")
    G["id_examples"] = dict(train=[r["id"] for r in tr[:5]], test=[r["id"] for r in te[:5]])
    with open(f"{P5}/results/gate.json", "w") as f:
        json.dump(G, f, indent=1)
    log(f"wrote {P5}/results/gate.json")


def cmd_score(args):
    recs = load(args.split)
    pred = {r["id"]: r["pred"] for r in json.load(open(args.pred))}
    n, _ = label_space()
    Y = multi_hot(recs, n)
    P = np.zeros_like(Y)
    for i, r in enumerate(recs):
        for y in pred[r["id"]]:
            P[i, y] = 1
    log(json.dumps(score_on(Y, P)))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("gate")
    s = sp.add_parser("score")
    s.add_argument("--pred", required=True)
    s.add_argument("--split", default="dev")
    a = ap.parse_args()
    {"gate": gate, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
