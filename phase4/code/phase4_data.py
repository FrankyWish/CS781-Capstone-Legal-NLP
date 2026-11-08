#!/usr/bin/env python3
"""CS-781 Phase 4 (CJPE) -- data, token cache, metrics, thresholds, submissions.

    python phase4_data.py gate            # stats + leakage checks (CPU, no tokenizer needed)
    python phase4_data.py tokenize        # BPE token cache for the GPU runs (CPU, all cores)

Everything else imports from here. Paths come from the environment so the same
code runs on Colab (Drive) and locally:

    CS781_BASE   default /content/drive/MyDrive/cs781_capstone
    CS781_P4     where results/, submissions/, cache/ and logs/ go
    CS781_P4_DATA  optional directory holding train.json / dev.json / test.json

Run-file format shared by every model (GPU runs, lexical, ensembles), so the
ensembler never needs to know what produced a file:

    results/<tag>/dev.pt   {"ids": [...], "z": float32 tensor}   z = logit(P(label=1))
    results/<tag>/test.pt  same, test ids
    results/<tag>/meta.json  dev metrics, threshold, params, cost
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import re
import sys
import time
import zipfile

BASE = os.environ.get("CS781_BASE", "/content/drive/MyDrive/cs781_capstone")
P1 = os.environ.get("CS781_P1", f"{BASE}/phase1")
P4 = os.environ.get("CS781_P4", f"{BASE}/phase4")
LOCAL = os.environ.get("CS781_P4_DATA")
REPO = "Exploration-Lab/CS-781-Capstone"
SUB = "Phase_4_CJPE"
FILES = {"train": "Train/train.json", "dev": "Train/dev.json", "test": "Test/test.json"}


def log(*a):
    print(*a, flush=True)


# --------------------------------------------------------------------- load --
def fetch(split):
    name = FILES[split]
    for root in [LOCAL, f"{P4}/data"]:
        if root:
            for p in (os.path.join(root, name), os.path.join(root, os.path.basename(name))):
                if os.path.exists(p):
                    return p
    from huggingface_hub import hf_hub_download
    return hf_hub_download(REPO, f"{SUB}/{name}", repo_type="dataset")


def read_records(path):
    """A JSON list, a dict id -> record, or JSON lines."""
    with open(path, encoding="utf-8") as f:
        head = f.read(1)
        f.seek(0)
        if head == "[" or head == "{":
            try:
                raw = json.load(f)
            except json.JSONDecodeError:
                f.seek(0)
                raw = [json.loads(l) for l in f if l.strip()]
        else:
            raw = [json.loads(l) for l in f if l.strip()]
    if isinstance(raw, dict):
        if "id" in raw and "text" in raw:
            raw = [raw]
        else:
            raw = [dict(v, id=k) if isinstance(v, dict) else {"id": k, "text": v}
                   for k, v in raw.items()]
    return raw


def as_label(v):
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float)):
        return int(v)
    s = str(v).strip().lower()
    return {"0": 0, "1": 1, "rejected": 0, "accepted": 1}[s]


def load_split(split):
    """[(id, text, label|None)] in file order, one record per id. Test labels are never read.

    The files merge ILDC's single- and multi-appeal sets, so the same case id can
    appear twice. Repeats are dropped (first copy kept) and counted: same text,
    different text, and -- for labelled splits -- different label.
    """
    out, seen = [], {}
    dup = {"same_text": 0, "diff_text": 0, "diff_label": 0}
    for r in read_records(fetch(split)):
        did = str(r["id"])
        text = r["text"]
        if isinstance(text, list):
            text = "\n".join(map(str, text))
        text = str(text)
        lab = None
        if split != "test":
            lab = as_label(r["label"])
        if did in seen:
            t0, l0 = seen[did]
            dup["same_text" if t0 == text else "diff_text"] += 1
            if split != "test" and l0 != lab:
                dup["diff_label"] += 1
            continue
        seen[did] = (text, lab)
        out.append((did, text, lab))
    if any(dup.values()):
        log(f"  {split}: {sum(dup.values()) - dup['diff_label']} repeated ids dropped {dup} -> {len(out)} unique")
    return out


def text_key(t):
    return hashlib.sha1(norm(t).encode()).hexdigest()


def test_copies():
    """{"train": [ids], "dev": [ids]}: labelled documents whose text IS a test document.

    The released dev.json holds ILDC's 994 dev cases AND its 1,517 test cases with
    their labels (2,511 = 994 + 1,517; every test text has an exact copy there).
    Selecting checkpoints or thresholds on those copies would be tuning on the
    test labels, so they are removed from every labelled split before use.
    Cached in cache/test_copies.json after the first call.
    """
    if os.environ.get("CS781_P4_KEEP_TEST_COPIES") == "1":
        return {"train": [], "dev": []}           # only if the TAs confirm Codabench scores a different set
    cp = f"{P4}/cache/test_copies.json"
    if os.path.exists(cp):
        return json.load(open(cp))
    data = {s: load_split(s) for s in FILES}
    tk = {text_key(t) for _, t, _ in data["test"]}
    out = {s: sorted(d for d, t, _ in data[s] if text_key(t) in tk) for s in ("train", "dev")}
    os.makedirs(os.path.dirname(cp), exist_ok=True)
    atomic_json(out, cp)
    log(f"  test copies inside labelled data: train {len(out['train'])}, dev {len(out['dev'])} -> {cp}")
    return out


def load_all(splits=("train", "dev", "test"), clean=True):
    """clean=True drops labelled documents that are copies of test documents (see test_copies)."""
    data = {s: load_split(s) for s in splits}
    if clean:
        drop = test_copies()
        for s in ("train", "dev"):
            if s in data and drop.get(s):
                bad = set(drop[s])
                n0 = len(data[s])
                data[s] = [r for r in data[s] if r[0] not in bad]
                log(f"  {s}: removed {n0 - len(data[s])} copies of test documents -> {len(data[s])}")
    return data


def drop_test_copies(c, split):
    """Same filter for a token cache dict."""
    bad = set(test_copies().get(split, []))
    if not bad:
        return c
    keep = [i for i, d in enumerate(c["ids"]) if d not in bad]
    log(f"  {split}: removed {len(c['ids']) - len(keep)} copies of test documents -> {len(keep)}")
    return dict(ids=[c["ids"][i] for i in keep], tok=[c["tok"][i] for i in keep],
                label=[c["label"][i] for i in keep])


def load_jsonl(path, limit=None):
    """External labelled docs, one {"id","text","label"} per line -> [(id, text, label)]."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out.append((str(r["id"]), r["text"], as_label(r["label"])))
                if limit and len(out) >= limit:
                    break
    return out


# -------------------------------------------------------------- token cache --
def _tok_worker(args):
    path, texts = args
    sys.path.insert(0, f"{P1}/shared/lib")
    from cs781_bpe import BPETokenizer
    import numpy as np
    tok = BPETokenizer.load(path)
    return [np.asarray(tok.encode(t), dtype=np.uint16) for t in texts]


def ext_cache(path, procs=None):
    """Token cache for an external jsonl (cached under its basename)."""
    name = "ext_" + os.path.splitext(os.path.basename(path))[0]
    return token_cache(name, recs=None if os.path.exists(f"{P4}/cache/tok_{name}.pt") else load_jsonl(path),
                       procs=procs)


def token_cache(split, recs=None, procs=None):
    """{"ids": [...], "tok": [np.uint16 arrays], "label": [...]}, cached on Drive.

    Whole documents are tokenised once (the BPE is pure Python); every view
    (tail, head+tail, spread chunks) is cut from these arrays at train time.
    """
    import numpy as np
    import torch
    d = f"{P4}/cache"
    os.makedirs(d, exist_ok=True)
    cp = f"{d}/tok_{split}.pt"
    if os.path.exists(cp):
        try:
            return torch.load(cp, weights_only=False)
        except Exception as e:                      # truncated by a killed run
            log(f"  token cache {cp} unreadable ({e}); rebuilding")
    recs = recs if recs is not None else load_split(split)
    path = f"{P1}/shared/tokenizer/bpe_legal_32k.json"
    procs = procs or max(1, os.cpu_count() or 1)
    t0 = time.time()
    texts = [" " + t.strip() for _, t, _ in recs]
    # contiguous shards keep each worker's chunk cache warm on similar text
    k = max(1, math.ceil(len(texts) / (procs * 8)))
    shards = [(path, texts[i:i + k]) for i in range(0, len(texts), k)]
    if procs > 1:
        import multiprocessing as mp
        with mp.get_context("spawn" if os.name == "nt" else "fork").Pool(procs) as pool:
            parts = pool.map(_tok_worker, shards)
    else:
        parts = [_tok_worker(s) for s in shards]
    toks = [a for p in parts for a in p]
    n = sum(len(a) for a in toks)
    out = dict(ids=[r[0] for r in recs], tok=toks, label=[r[2] for r in recs])
    tmp = f"{cp}.tmp{os.getpid()}"
    torch.save(out, tmp)
    os.replace(tmp, cp)
    log(f"  tokenised {split}: {len(toks)} docs, {n/1e6:.1f}M tokens in {time.time()-t0:.0f}s "
        f"({n/max(1e-6, time.time()-t0)/1e3:.0f}k tok/s, {procs} procs) -> {cp}")
    return out


# ------------------------------------------------------------------ metrics --
def macro_f1(y, p):
    """sklearn f1_score(average='macro', labels=[0,1], zero_division=0)."""
    tp = sum(1 for a, b in zip(y, p) if a == 1 and b == 1)
    tn = sum(1 for a, b in zip(y, p) if a == 0 and b == 0)
    fp = sum(1 for a, b in zip(y, p) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(y, p) if a == 1 and b == 0)
    f1 = lambda t, f_p, f_n: 2 * t / (2 * t + f_p + f_n) if t else 0.0
    return (f1(tp, fp, fn) + f1(tn, fn, fp)) / 2


def metrics(y, p):
    tp = sum(1 for a, b in zip(y, p) if a == 1 and b == 1)
    fp = sum(1 for a, b in zip(y, p) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(y, p) if a == 1 and b == 0)
    return dict(macro_f1=macro_f1(y, p), precision=tp / (tp + fp) if tp + fp else 0.0,
                recall=tp / (tp + fn) if tp + fn else 0.0,
                acc=sum(a == b for a, b in zip(y, p)) / max(1, len(y)),
                pos_rate=sum(p) / max(1, len(p)))


def best_threshold(y, z):
    """Threshold on z maximising dev Macro-F1. Ties -> the one closest to 0."""
    order = sorted(range(len(z)), key=lambda i: -z[i])
    P = sum(y)
    N = len(y) - P
    best = (macro_f1(y, [1 if v > 0 else 0 for v in z]), 0.0)
    tp = fp = 0
    for r, i in enumerate(order):
        tp += y[i]
        fp += 1 - y[i]
        fn, tn = P - tp, N - fp
        f = ((2 * tp / (2 * tp + fp + fn) if tp else 0) + (2 * tn / (2 * tn + fn + fp) if tn else 0)) / 2
        nxt = z[order[r + 1]] if r + 1 < len(order) else z[i] - 1.0
        thr = (z[i] + nxt) / 2
        if f > best[0] + 1e-12 or (abs(f - best[0]) < 1e-12 and abs(thr) < abs(best[1])):
            best = (f, thr)
    return best[1], best[0]


def rate_threshold(z_test, rate):
    """Threshold giving the test set the same positive rate as `rate` (dev's)."""
    s = sorted(z_test, reverse=True)
    k = int(round(rate * len(s)))
    if k <= 0:
        return s[0] + 1.0
    if k >= len(s):
        return s[-1] - 1.0
    return (s[k - 1] + s[k]) / 2


def auc(y, z):
    pos = [v for v, t in zip(z, y) if t == 1]
    neg = [v for v, t in zip(z, y) if t == 0]
    if not pos or not neg:
        return float("nan")
    allv = sorted((v, t) for v, t in zip(z, y))
    rank, i, rs = 0, 0, 0.0
    while i < len(allv):
        j = i
        while j < len(allv) and allv[j][0] == allv[i][0]:
            j += 1
        avg = (i + j + 1) / 2
        rs += avg * sum(1 for k in range(i, j) if allv[k][1] == 1)
        i = j
    return (rs - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def summarise(y, z, thr=None):
    t, fbest = best_threshold(y, z)
    thr = t if thr is None else thr
    m05 = metrics(y, [1 if v > 0 else 0 for v in z])
    mt = metrics(y, [1 if v > thr else 0 for v in z])
    return dict(f1_at0=m05["macro_f1"], f1_tuned=mt["macro_f1"], thr=thr, auc=auc(y, z),
                precision=mt["precision"], recall=mt["recall"], acc=mt["acc"],
                pos_rate=mt["pos_rate"])


# -------------------------------------------------------------- run files ---
def save_run(tag, dev_ids, dev_z, test_ids, test_z, meta, extra=None):
    import torch
    rd = f"{P4}/results/{tag}"
    os.makedirs(rd, exist_ok=True)
    torch.save(dict(ids=list(dev_ids), z=torch.as_tensor(dev_z, dtype=torch.float32)), f"{rd}/dev.pt")
    torch.save(dict(ids=list(test_ids), z=torch.as_tensor(test_z, dtype=torch.float32)), f"{rd}/test.pt")
    for k, v in (extra or {}).items():
        torch.save(v, f"{rd}/{k}.pt")
    atomic_json(meta, f"{rd}/meta.json", indent=1)
    return rd


def load_run(tag):
    import torch
    rd = f"{P4}/results/{tag}"
    d = torch.load(f"{rd}/dev.pt", weights_only=False)
    t = torch.load(f"{rd}/test.pt", weights_only=False)
    meta = json.load(open(f"{rd}/meta.json"))
    return d, t, meta


def atomic_json(obj, path, **kw):
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, **kw)
    os.replace(tmp, path)


# --------------------------------------------------------------- submission --
def validate_prediction(pred, test_ids):
    """Every rule evaluate_phase4.py rejects on."""
    assert isinstance(pred, list) and pred, "prediction must be a non-empty JSON list"
    seen = set()
    for n, r in enumerate(pred):
        assert isinstance(r, dict) and "id" in r and "pred" in r, f"item {n} needs id and pred"
        assert isinstance(r["id"], str) and r["id"].strip(), f"item {n}: bad id"
        assert not isinstance(r["pred"], bool) and r["pred"] in (0, 1) and isinstance(r["pred"], int), \
            f"item {n}: pred must be int 0/1"
        assert r["id"] not in seen, f"duplicate id {r['id']}"
        seen.add(r["id"])
    want = set(test_ids)
    assert seen == want, f"id mismatch: missing {len(want - seen)}, extra {len(seen - want)}"
    return True


def write_submission(tag, test_ids, pred, dev=None, note="", kind="model"):
    """pred: {id: 0/1}. Writes submissions/<tag>.zip and appends to the tracker.

    kind: "model" (only Phase-1-derived models) or "hybrid" (anything else in
    the mix, e.g. TF-IDF). The report must keep the two apart.
    """
    out = [{"id": d, "pred": int(pred[d])} for d in test_ids]
    validate_prediction(out, test_ids)
    sd = f"{P4}/submissions"
    os.makedirs(sd, exist_ok=True)
    pj = f"{sd}/prediction_{tag}.json"
    with open(pj, "w", encoding="utf-8") as f:
        json.dump(out, f)
    zp = f"{sd}/{tag}.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(pj, "prediction.json")
    with zipfile.ZipFile(zp) as z:                       # re-read what we ship
        assert z.namelist() == ["prediction.json"], z.namelist()
        validate_prediction(json.loads(z.read("prediction.json")), test_ids)
    h = hashlib.sha1(json.dumps(out, sort_keys=True).encode()).hexdigest()[:12]
    tr_path = f"{sd}/tracker.json"
    tr = json.load(open(tr_path)) if os.path.exists(tr_path) else []
    dup = [t["tag"] for t in tr if t.get("sha") == h and t["tag"] != tag]
    tr = [t for t in tr if t["tag"] != tag]
    tr.append(dict(tag=tag, zip=os.path.basename(zp), sha=h, kind=kind,
                   dev=dev, pos_rate=sum(out[i]["pred"] for i in range(len(out))) / len(out),
                   note=note, test=None, time=time.strftime("%Y-%m-%d %H:%M")))
    atomic_json(tr, tr_path, indent=1)
    pr = sum(r["pred"] for r in out) / len(out)
    log(f"  WROTE {zp}  sha {h}  test pos-rate {pr:.3f}"
        + (f"  -- IDENTICAL to {dup}, will not count" if dup else ""))
    return zp


# --------------------------------------------------------------------- gate --
WORD = re.compile(r"[a-z0-9]+")

OUTCOME = [  # phrases that state the outcome outright; ILDC is supposed to strip them
    r"appeals? (?:is|are|stands?|shall stand) (?:hereby )?(?:allowed|dismissed|accepted|rejected|disposed)",
    r"petitions? (?:is|are|stands?) (?:hereby )?(?:allowed|dismissed|accepted|rejected|disposed)",
    r"(?:we|i) (?:therefore |accordingly )?(?:allow|dismiss) (?:the|this|these) (?:appeal|petition)",
    r"in the result,? the appeal",
    r"leave (?:is|was) granted",
]


def norm(t):
    return " ".join(WORD.findall(t.lower()))


def gate(a):
    data = load_all(clean=False)
    for s, recs in data.items():
        n = len(recs)
        lens = sorted(len(t) for _, t, _ in recs)
        wl = sorted(len(t.split()) for _, t, _ in recs)
        q = lambda v, p: v[min(len(v) - 1, int(p * len(v)))]
        lab = collections.Counter(l for _, _, l in recs)
        log(f"{s:5s}: {n:6d} docs | words median {q(wl,.5)} p90 {q(wl,.9)} max {wl[-1]} | "
            f"chars median {q(lens,.5)} | labels {dict(lab)}")
    # ids: ILDC-style ids tell us the source
    for s, recs in data.items():
        log(f"  {s} id examples: {[r[0] for r in recs[:4]]}")

    # outcome phrases in the last 1500 chars
    pats = [re.compile(p, re.I) for p in OUTCOME]
    for s, recs in data.items():
        hit = collections.Counter()
        by_lab = collections.defaultdict(collections.Counter)
        for _, t, l in recs:
            tail = t[-1500:]
            for k, p in enumerate(pats):
                if p.search(tail):
                    hit[k] += 1
                    by_lab[k][l] += 1
        log(f"  {s}: outcome phrases in last 1500 chars: " +
            ", ".join(f"p{k}={hit[k]} {dict(by_lab[k])}" for k in range(len(pats))))
    log("\n  last 300 chars of 3 train docs per label:")
    shown = collections.Counter()
    for _, t, l in data["train"]:
        if shown[l] < 3:
            shown[l] += 1
            log(f"   [label {l}] ...{t[-300:]!r}")

    # exact and 13-gram overlap between test and train/dev
    def key(t):
        return hashlib.sha1(norm(t).encode()).hexdigest()

    lab_keys = {key(t): (s, d, l) for s in ("train", "dev") for d, t, l in data[s]}
    ex = [(d, lab_keys[key(t)]) for d, t, _ in data["test"] if key(t) in lab_keys]
    log(f"\n  exact (normalised) duplicates test->train/dev: {len(ex)} of {len(data['test'])}")
    dd = [(d, lab_keys[key(t)]) for d, t, _ in data["dev"] if key(t) in lab_keys and lab_keys[key(t)][0] == "train"]
    log(f"  exact duplicates dev->train: {len(dd)} of {len(data['dev'])}")
    tr_keys = collections.defaultdict(set)
    for d, t, l in data["train"]:
        tr_keys[key(t)].add(l)
    conflict = sum(1 for v in tr_keys.values() if len(v) > 1)
    dup_train = len(data["train"]) - len(tr_keys)
    log(f"  duplicate texts inside train: {dup_train}; with conflicting labels: {conflict}")

    # 13-gram near-duplicates: sample shingles of each test doc's tail
    n = 13
    idx = collections.defaultdict(list)
    for s in ("train", "dev"):
        for j, (d, t, l) in enumerate(data[s]):
            w = norm(t[-6000:]).split()
            for i in range(0, max(0, len(w) - n + 1), 25):
                idx[hash(" ".join(w[i:i + n]))].append((s, j))
    near = 0
    for d, t, _ in data["test"]:
        w = norm(t[-6000:]).split()
        c = collections.Counter()
        for i in range(max(0, len(w) - n + 1)):
            for h in idx.get(hash(" ".join(w[i:i + n])), ()):
                c[h] += 1
        if c and c.most_common(1)[0][1] >= 20:
            near += 1
    log(f"  test docs whose tail shares >=20 sampled 13-grams with one train/dev doc: {near}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gate", "tokenize"])
    ap.add_argument("--procs", type=int, default=None)
    a = ap.parse_args()
    if a.cmd == "gate":
        gate(a)
    else:
        for s in ("dev", "test", "train"):
            c = token_cache(s, procs=a.procs)
            L = sorted(len(x) for x in c["tok"])
            q = lambda p: L[min(len(L) - 1, int(p * len(L)))]
            log(f"  {s}: BPE tokens/doc median {q(.5)} p75 {q(.75)} p90 {q(.9)} max {L[-1]} "
                f"| >512 {sum(x > 512 for x in L)/len(L):.2f} >2048 {sum(x > 2048 for x in L)/len(L):.2f} "
                f">4096 {sum(x > 4096 for x in L)/len(L):.2f}")


if __name__ == "__main__":
    main()
