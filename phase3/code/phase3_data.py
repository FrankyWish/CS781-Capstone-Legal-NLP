#!/usr/bin/env python3
"""CS-781 Phase 3 (rhetorical roles) -- data, gate, CV split, validator, scoring.

    python phase3_data.py                 # stats, overlap checks, writes split.json
    python phase3_data.py --folds 5 --seed 781

RUN THIS FIRST. Everything else imports from here.

What it reports, and why each number matters:

  label distribution per domain/split   Macro-F1 over 13 labels: a label never
                                        predicted is a flat 0 of 1/13. A label
                                        absent from the test gold is ALSO a flat
                                        0 (sklearn zero_division=0), whatever we do.
  H(y) vs H(y_t | y_t-1), persistence   how much the previous sentence's label
                                        tells us. This is the case for a CRF /
                                        sequence model, measured before building one.
  label by document decile              how much absolute position tells us.
  13-gram overlap test vs train/dev     near-duplicate documents across splits.

The split: 5-fold document-level CV over train+dev pooled (90 documents),
stratified by domain. The organisers' dev set is 10 documents; macro-F1 on 10
documents with 13 labels is mostly noise, so it is not used on its own.
Every fold model is early-stopped on its own held-out fold, and the submission
averages the fold models' test logits -- no unvalidated full refit.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import random
import re
import sys
import zipfile

BASE = os.environ.get("CS781_BASE", "/content/drive/MyDrive/cs781_capstone")
P1 = os.environ.get("CS781_P1", f"{BASE}/phase1")       # read-only
P3 = os.environ.get("CS781_P3", f"{BASE}/phase3")       # everything we write
LOCAL = os.environ.get("CS781_P3_DATA")                 # optional local copy
REPO = "Exploration-Lab/CS-781-Capstone"
SUB = "Phase_3_RR"

LABELS = ["Fact", "Issue", "ArgumentPetitioner", "ArgumentRespondent",
          "Statute", "Dissent", "PrecedentReliedUpon", "PrecedentNotReliedUpon",
          "PrecedentOverruled", "RulingByLowerCourt", "RatioOfTheDecision",
          "RulingByPresentCourt", "None"]
DOMAINS = ("CL", "IT")


def log(*a):
    print(*a, flush=True)


# --------------------------------------------------------------------- load --
def fetch(name):
    """Path to Train/... or Test/... -- local copy first, else the HF repo."""
    if LOCAL:
        p = os.path.join(LOCAL, name)
        if os.path.exists(p):
            return p
        p = os.path.join(LOCAL, os.path.basename(name))
        if os.path.exists(p):
            return p
    from huggingface_hub import hf_hub_download
    return hf_hub_download(REPO, f"{SUB}/{name}", repo_type="dataset")


def strip_txt(d):
    d = str(d)
    return d[:-4] if d.endswith(".txt") else d


def _records(raw, path):
    """Accept a list of records, a dict id -> record, or JSON lines."""
    if isinstance(raw, dict):
        if "id" in raw and "text" in raw:
            return [raw]
        out = []
        for k, v in raw.items():
            r = dict(v) if isinstance(v, dict) else {"text": v}
            r.setdefault("id", k)
            out.append(r)
        return out
    if isinstance(raw, list):
        return raw
    raise ValueError(f"{path}: unsupported JSON shape {type(raw).__name__}")


def read_split(name):
    path = fetch(name)
    with open(path, encoding="utf-8") as f:
        txt = f.read()
    try:
        raw = json.loads(txt)
    except json.JSONDecodeError:
        raw = [json.loads(l) for l in txt.splitlines() if l.strip()]
    return _records(raw, path)


def load_label_vocab():
    with open(fetch("Train/label_vocab.json"), encoding="utf-8") as f:
        v = json.load(f)
    if isinstance(v, dict):
        vals = list(v.values())
        if all(isinstance(x, int) and not isinstance(x, bool) for x in vals):
            labels = [k for k, _ in sorted(v.items(), key=lambda kv: kv[1])]
        else:
            labels = [v[k] for k in sorted(v, key=lambda k: int(k))]
    else:
        labels = list(v)
    assert len(labels) == 13 and set(labels) == set(LABELS), (
        f"label_vocab.json differs from the brief: {labels}")
    return labels


def load_all(with_test=True):
    """{doc_id: dict(id, domain, split, text[list], labels[list]|None)}.

    Labels are normalised to strings even if the file stores integer ids, using
    the vocabulary order from label_vocab.json.
    """
    labels = load_label_vocab()
    docs = {}
    names = [("train", "Train/{d}_train.json"), ("dev", "Train/{d}_dev.json")]
    if with_test:
        names.append(("test", "Test/{d}_test.json"))
    for split, pat in names:
        for dom in DOMAINS:
            for r in read_split(pat.format(d=dom)):
                did = strip_txt(r["id"])
                text = r.get("text") or r.get("sentences")
                assert isinstance(text, list) and text, f"{did}: no sentence list"
                # Test labels are never read, even if the file carries a field.
                lab = r.get("labels") if split != "test" else None
                if lab is not None:
                    lab = [labels[x] if isinstance(x, int) else x for x in lab]
                    assert len(lab) == len(text), f"{did}: {len(lab)} labels / {len(text)} sents"
                    bad = set(lab) - set(labels)
                    assert not bad, f"{did}: unknown labels {bad}"
                assert did not in docs, f"duplicate document id {did}"
                docs[did] = dict(id=did, raw_id=r["id"], domain=dom, split=split,
                                 text=[str(s) for s in text], labels=lab)
    return docs, labels


# ------------------------------------------------------------------ overlap --
WORD = re.compile(r"[a-z0-9]+")


def norm(t):
    return " ".join(WORD.findall(t.lower()))


def grams(t, n=13):
    w = norm(t).split()
    return {" ".join(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}


def sent_key(s):
    return hashlib.sha1(norm(s).encode()).hexdigest()[:16]


def contamination_filter(records, test_docs, n=13, thresh=0.5, log_fn=log):
    """Record-level filter for EXTERNAL data. Keep this; never skip it.

    `records` is a list of (rid, list_of_sentences, payload). A record is dropped
    if >= `thresh` of its 13-grams occur anywhere in the test documents, and
    every individual sentence whose own 13-grams hit the test set is masked out
    of the records that survive. Phase 2's document-level check rated a verbatim
    copied passage at 0.4%; only the per-record filter caught it.
    """
    tg = set()
    tsent = set()
    for d in test_docs.values():
        tg |= grams(" ".join(d["text"]), n)
        tsent |= {sent_key(s) for s in d["text"] if len(norm(s).split()) >= 6}
    kept, dropped, masked = [], [], 0
    for rid, sents, payload in records:
        g = grams(" ".join(sents), n)
        frac = len(g & tg) / len(g) if g else 0.0
        if frac >= thresh:
            dropped.append((rid, frac))
            continue
        hit = []
        for i, s in enumerate(sents):
            sg = grams(s, n)
            if sent_key(s) in tsent or (sg and len(sg & tg) / len(sg) >= thresh):
                hit.append(i)
        masked += len(hit)
        kept.append((rid, sents, payload, hit))
    log_fn(f"  contamination: dropped {len(dropped)} records "
           f"({', '.join(f'{r}={100*f:.0f}%' for r, f in dropped[:6])}"
           f"{' ...' if len(dropped) > 6 else ''}); masked {masked} sentences in kept records")
    return kept, dropped


# ---------------------------------------------------------------- scoring ---
def prf(y_true, y_pred, labels):
    """Per-label (P, R, F1, support, n_pred) and macro averages.

    Matches sklearn f1_score(..., labels=labels, average='macro',
    zero_division=0): a label with no gold and no predictions scores 0, so it
    still drags the macro mean down.
    """
    tp = collections.Counter()
    npred = collections.Counter(y_pred)
    ngold = collections.Counter(y_true)
    for t, p in zip(y_true, y_pred):
        if t == p:
            tp[t] += 1
    per = {}
    for l in labels:
        p = tp[l] / npred[l] if npred[l] else 0.0
        r = tp[l] / ngold[l] if ngold[l] else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        per[l] = (p, r, f, ngold[l], npred[l])
    k = len(labels)
    mp = sum(v[0] for v in per.values()) / k
    mr = sum(v[1] for v in per.values()) / k
    mf = sum(v[2] for v in per.values()) / k
    present = [l for l in labels if ngold[l]]
    mf_present = (sum(per[l][2] for l in present) / len(present)) if present else 0.0
    return dict(macro_f1=mf, macro_p=mp, macro_r=mr, macro_f1_present=mf_present,
                accuracy=sum(tp.values()) / max(1, len(y_true)), per=per)


def score_docs(pred, gold, labels, ids=None):
    ids = list(ids if ids is not None else gold)
    yt, yp = [], []
    for d in ids:
        assert len(pred[d]) == len(gold[d]), d
        yt += gold[d]
        yp += pred[d]
    return prf(yt, yp, labels), yt, yp


def confusion(yt, yp, labels):
    ix = {l: i for i, l in enumerate(labels)}
    m = [[0] * len(labels) for _ in labels]
    for t, p in zip(yt, yp):
        m[ix[t]][ix[p]] += 1
    return m


ABBR = {"Fact": "Fact", "Issue": "Iss", "ArgumentPetitioner": "ArgP",
        "ArgumentRespondent": "ArgR", "Statute": "Stat", "Dissent": "Diss",
        "PrecedentReliedUpon": "PreR", "PrecedentNotReliedUpon": "PreN",
        "PrecedentOverruled": "PreO", "RulingByLowerCourt": "RLC",
        "RatioOfTheDecision": "Ratio", "RulingByPresentCourt": "RPC", "None": "None"}


def report(res, labels, title="", yt=None, yp=None, log_fn=log):
    log_fn(f"  {title} macro-F1 {res['macro_f1']:.4f}  P {res['macro_p']:.4f}  "
           f"R {res['macro_r']:.4f}  | acc {res['accuracy']:.4f}  "
           f"| macro-F1 over labels present {res['macro_f1_present']:.4f}")
    for l in labels:
        p, r, f, n, k = res["per"][l]
        flag = ""
        if n and not k:
            flag = "  <-- NEVER PREDICTED"
        elif not n:
            flag = "  (no gold: scores 0 whatever we do)"
        log_fn(f"    {l:23s} P={p:.3f} R={r:.3f} F1={f:.3f}  gold={n:5d} pred={k:5d}{flag}")
    if yt is not None:
        m = confusion(yt, yp, labels)
        log_fn("  confusion (rows gold, cols pred):")
        log_fn("    " + " " * 6 + "".join(f"{ABBR[l]:>6s}" for l in labels))
        for l, row in zip(labels, m):
            log_fn("    " + f"{ABBR[l]:>6s}" + "".join(f"{x:6d}" for x in row))
        off = [(row[j], labels[i], labels[j]) for i, row in enumerate(m)
               for j in range(len(labels)) if i != j and row[j]]
        off.sort(reverse=True)
        log_fn("  top confusions: " + "; ".join(f"{a}->{b} {n}" for n, a, b in off[:8]))


# --------------------------------------------------------------- submission --
def validate_prediction(pred, test_docs, labels):
    """Every rule evaluate_phase3.py rejects on. One violation fails the upload."""
    allowed = set(labels)
    norm_ids = {}
    for raw in pred:
        n = strip_txt(raw)
        assert n not in norm_ids, f"duplicate id after .txt normalisation: {raw}"
        norm_ids[n] = raw
    want = {strip_txt(d) for d in test_docs}
    miss, extra = want - set(norm_ids), set(norm_ids) - want
    assert not miss and not extra, f"id mismatch: missing {sorted(miss)[:5]} extra {sorted(extra)[:5]}"
    for d in want:
        seq = pred[norm_ids[d]]
        assert isinstance(seq, list), f"{d}: not a list"
        assert len(seq) == len(test_docs[d]["text"]), (
            f"{d}: {len(seq)} labels for {len(test_docs[d]['text'])} sentences")
        bad = [x for x in seq if x not in allowed]
        assert not bad, f"{d}: unknown label {bad[0]!r}"
    unused = sorted(allowed - {x for v in pred.values() for x in v})
    if unused:
        log(f"  WARNING: never predicted {unused} -- each is a flat 0 of 1/13")
    return True


def write_submission(pred, test_docs, labels, tag, cv=None, note=""):
    """Validate, write prediction.json + zip, and append to the CV->test tracker."""
    validate_prediction(pred, test_docs, labels)
    out = {test_docs[d]["raw_id"] if d in test_docs else d: pred[d] for d in pred}
    sd = f"{P3}/submissions"
    os.makedirs(sd, exist_ok=True)
    pj = f"{sd}/prediction_{tag}.json"
    with open(pj, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    zp = f"{sd}/submission_{tag}.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(pj, "prediction.json")
    with zipfile.ZipFile(zp) as z:                   # re-read what we ship
        assert z.namelist() == ["prediction.json"], z.namelist()
        validate_prediction(json.loads(z.read("prediction.json")), test_docs, labels)
    h = hashlib.sha1(json.dumps(out, sort_keys=True).encode()).hexdigest()[:12]
    tr_path = f"{sd}/tracker.json"
    tr = json.load(open(tr_path)) if os.path.exists(tr_path) else []
    dup = [t["tag"] for t in tr if t.get("sha") == h]
    tr.append(dict(tag=tag, zip=os.path.basename(zp), sha=h, cv=cv, test=None, note=note))
    json.dump(tr, open(tr_path, "w"), indent=1)
    dist = collections.Counter(x for v in pred.values() for x in v)
    log(f"  WROTE {zp}  sha {h}" + (f"  -- IDENTICAL to {dup}, will not count" if dup else ""))
    log("  predicted distribution: " + ", ".join(f"{ABBR[l]} {dist[l]}" for l in labels))
    return zp


# --------------------------------------------------------------------- gate --
def entropy(c):
    n = sum(c.values())
    return -sum(v / n * math.log2(v / n) for v in c.values() if v)


def make_split(docs, k, seed):
    rng = random.Random(seed)
    folds = [[] for _ in range(k)]
    for dom in DOMAINS:
        ids = sorted(d for d, r in docs.items()
                     if r["split"] != "test" and r["labels"] and r["domain"] == dom)
        rng.shuffle(ids)
        # balance sentence counts too: longest first into the lightest fold
        ids.sort(key=lambda d: -len(docs[d]["text"]))
        load = [0] * k
        for d in ids:
            j = min(range(k), key=lambda i: (load[i], rng.random()))
            folds[j].append(d)
            load[j] += len(docs[d]["text"])
    return [sorted(f) for f in folds]


def atomic_json(obj, path, **kw):
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, **kw)
    os.replace(tmp, path)


def ensure_split(docs, k=5, seed=781):
    """split.json, created deterministically if missing (same folds as the gate)."""
    sp = f"{P3}/split.json"
    if not os.path.exists(sp):
        os.makedirs(P3, exist_ok=True)
        folds = make_split(docs, k, seed)
        atomic_json(dict(seed=seed, folds=folds,
                         test=sorted(d for d, r in docs.items() if r["split"] == "test"),
                         labels=LABELS, domain={d: r["domain"] for d, r in docs.items()}),
                    sp, indent=1)
    return json.load(open(sp))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=781)
    ap.add_argument("--tokenizer", action="store_true",
                    help="also report BPE tokens per sentence (needs Phase-1 lib)")
    a = ap.parse_args()

    docs, labels = load_all()
    log(f"label vocab ({len(labels)}): {labels}")
    by = collections.defaultdict(list)
    for d, r in docs.items():
        by[(r["split"], r["domain"])].append(d)
    log("\n=== documents / sentences ===")
    for (sp, dom), ids in sorted(by.items()):
        n = [len(docs[d]["text"]) for d in ids]
        ch = [sum(len(s) for s in docs[d]["text"]) / len(docs[d]["text"]) for d in ids]
        log(f"  {sp:5s} {dom}: {len(ids):3d} docs  {sum(n):6d} sents  "
            f"sents/doc min {min(n)} med {sorted(n)[len(n)//2]} max {max(n)}  "
            f"chars/sent {sum(ch)/len(ch):.0f}")

    log("\n=== label distribution (sentences) ===")
    hdr = "  " + f"{'':23s}" + "".join(f"{s+'-'+d:>10s}" for s in ("train", "dev") for d in DOMAINS)
    log(hdr + f"{'all':>8s}{'%':>7s}{'docs w/':>9s}")
    cnt = {k: collections.Counter() for k in by}
    docs_with = collections.Counter()
    for d, r in docs.items():
        if r["labels"]:
            cnt[(r["split"], r["domain"])].update(r["labels"])
            docs_with.update(set(r["labels"]))
    tot = collections.Counter()
    for c in cnt.values():
        tot.update(c)
    N = sum(tot.values())
    for l in labels:
        row = "".join(f"{cnt.get((s, d), collections.Counter())[l]:10d}"
                      for s in ("train", "dev") for d in DOMAINS)
        log(f"  {l:23s}{row}{tot[l]:8d}{100*tot[l]/N:6.2f}%{docs_with[l]:9d}")
    n_lab = sum(1 for r in docs.values() if r["labels"])
    log(f"  ({n_lab} labelled documents; a test set of 10 documents will miss labels "
        f"present in few documents)")

    log("\n=== sequence structure ===")
    trans = collections.Counter()
    prev_c = collections.Counter()
    same = total = 0
    seg_len = []
    for r in docs.values():
        y = r["labels"]
        if not y:
            continue
        run = 1
        for a_, b_ in zip(y, y[1:]):
            trans[(a_, b_)] += 1
            prev_c[a_] += 1
            same += a_ == b_
            total += 1
            if a_ == b_:
                run += 1
            else:
                seg_len.append(run)
                run = 1
        seg_len.append(run)
    h_y = entropy(tot)
    h_cond = 0.0
    for a_ in prev_c:
        row = collections.Counter({b_: n for (x, b_), n in trans.items() if x == a_})
        h_cond += prev_c[a_] / total * entropy(row)
    log(f"  H(y) = {h_y:.3f} bits   H(y_t | y_t-1) = {h_cond:.3f} bits   "
        f"-> the previous label removes {100*(1-h_cond/h_y):.0f}% of the uncertainty")
    log(f"  label persistence P(y_t == y_t-1) = {same/total:.3f};  "
        f"segments: {len(seg_len)}, mean length {sum(seg_len)/len(seg_len):.1f} sentences")
    # oracle-ish baselines: persistence and majority
    log("  most likely successor of each label:")
    for a_ in labels:
        row = [(n, b_) for (x, b_), n in trans.items() if x == a_ and b_ != a_]
        row.sort(reverse=True)
        if prev_c[a_]:
            log(f"    {a_:23s} stays {trans[(a_, a_)]/prev_c[a_]:.2f}; else -> "
                + ", ".join(f"{ABBR[b_]} {n}" for n, b_ in row[:4]))

    log("\n=== label share by document decile (row = decile) ===")
    dec = [collections.Counter() for _ in range(10)]
    for r in docs.values():
        if r["labels"]:
            n = len(r["labels"])
            for i, l in enumerate(r["labels"]):
                dec[min(9, i * 10 // n)][l] += 1
    log("     " + "".join(f"{ABBR[l]:>6s}" for l in labels))
    for i, c in enumerate(dec):
        s = sum(c.values())
        log(f"  {i:2d} " + "".join(f"{100*c[l]/s:6.1f}" for l in labels))
    h_pos = sum(sum(c.values()) / N * entropy(c) for c in dec)
    log(f"  H(y | decile) = {h_pos:.3f} bits (vs H(y) {h_y:.3f})")

    log("\n=== overlap checks ===")
    test = {d: r for d, r in docs.items() if r["split"] == "test"}
    lab = {d: r for d, r in docs.items() if r["split"] != "test"}
    tg = {d: grams(" ".join(r["text"])) for d, r in test.items()}
    worst = []
    for d, r in lab.items():
        g = grams(" ".join(r["text"]))
        for t, gt in tg.items():
            if g and gt:
                f = len(g & gt) / min(len(g), len(gt))
                if f > 0.02:
                    worst.append((f, d, t))
    worst.sort(reverse=True)
    log(f"  labelled-vs-test document pairs sharing >2% of 13-grams: {len(worst)}")
    for f, d, t in worst[:8]:
        log(f"    {100*f:5.1f}%  {d} ~ {t}")
    ts = collections.Counter(sent_key(s) for r in test.values() for s in r["text"]
                             if len(norm(s).split()) >= 6)
    ls = {sent_key(s) for r in lab.values() for s in r["text"] if len(norm(s).split()) >= 6}
    hit = sum(n for k, n in ts.items() if k in ls)
    log(f"  test sentences (>=6 words) found verbatim in train/dev: {hit} of {sum(ts.values())}")
    # label consistency of exact duplicate sentences inside the labelled data
    dup = collections.defaultdict(collections.Counter)
    for r in lab.values():
        for s, l in zip(r["text"], r["labels"]):
            if len(norm(s).split()) >= 6:
                dup[sent_key(s)][l] += 1
    multi = [c for c in dup.values() if sum(c.values()) > 1]
    agree = sum(max(c.values()) for c in multi) / max(1, sum(sum(c.values()) for c in multi))
    log(f"  repeated sentences inside train/dev: {len(multi)}; label agreement {agree:.3f}")

    if a.tokenizer:
        sys.path.insert(0, f"{P1}/shared/lib")
        from cs781_bpe import BPETokenizer
        tok = BPETokenizer.load(f"{P1}/shared/tokenizer/bpe_legal_32k.json")
        lens = sorted(len(tok.encode(s)) for r in docs.values() for s in r["text"])
        q = lambda p: lens[min(len(lens) - 1, int(p * len(lens)))]
        log(f"\n=== BPE tokens per sentence ===\n  median {q(.5)}  p90 {q(.9)}  "
            f"p99 {q(.99)}  max {lens[-1]}  total {sum(lens)}  "
            f"(>128: {sum(x > 128 for x in lens)}, >256: {sum(x > 256 for x in lens)})")

    folds = make_split(docs, a.folds, a.seed)
    os.makedirs(P3, exist_ok=True)
    split = dict(seed=a.seed, folds=folds,
                 test=sorted(test), labels=labels,
                 domain={d: r["domain"] for d, r in docs.items()})
    sp = f"{P3}/split.json"
    if os.path.exists(sp):
        old = json.load(open(sp))
        if old["folds"] != folds:
            log(f"\nWARNING: {sp} exists with DIFFERENT folds; not overwriting. "
                f"Delete it deliberately if you mean to re-split.")
            return
    atomic_json(split, sp, indent=1)
    log(f"\nwrote {sp}")
    for i, f in enumerate(folds):
        log(f"  fold {i}: {len(f)} docs ({sum(docs[d]['domain']=='CL' for d in f)} CL), "
            f"{sum(len(docs[d]['text']) for d in f)} sentences")


if __name__ == "__main__":
    main()
