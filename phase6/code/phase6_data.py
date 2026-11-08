#!/usr/bin/env python3
"""CS-781 Phase 6 -- data, held-out split, official-identical scoring, submissions.

    python phase6_data.py gate        measure the data and every integrity risk
    python phase6_data.py split       write the fixed held-out ids (results/split.json)
    python phase6_data.py score --pred <file> [--split dev]

The scoring here is byte-identical to spec/evaluation_scripts/evaluate_phase6.py:
ROUGE-1 F and ROUGE-L F over `rouge_score`'s DefaultTokenizer(use_stemmer=True),
and BERTScore F1 from `bert_score.score(..., lang='en')` -- roberta-large layer 17,
idf=False, rescale_with_baseline=False.  Consequences of that last line drive the
whole phase:

  * unrescaled BERTScore F1 for fluent English of the right length sits near 0.80;
    the leaderboard will be decided in the third decimal (Phase 1: 0.8485 .. 0.8338
    over ten places).
  * F1 is symmetric, so LENGTH COUNTS TWICE.  Short output kills recall, long
    output kills precision.  Matching the reference length distribution is the
    single biggest dial in this phase, exactly as it was in Phase 1.
  * no idf: boilerplate and function words score as much as the holding of the
    case.  Register matching beats being right.

Integrity: Phase-6 test references are private here, but the public source of this
data (IL-TUR `summ` / IN-Abs) ships its test summaries.  Never download them.  The
gate below only MEASURES leaks (summary text inside the input document, train
documents that duplicate test documents); it never builds a submission from one.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import zipfile

import numpy as np

BASE = os.environ.get("CS781_BASE", "/content/drive/MyDrive/cs781_capstone")
P1 = os.environ.get("CS781_P1", f"{BASE}/phase1")          # read-only
P6 = os.environ.get("CS781_P6", f"{BASE}/phase6")          # everything we write
LOCAL = os.environ.get("CS781_P6_DATA")                    # optional local copy of Phase_6_Summ
REPO = "Exploration-Lab/CS-781-Capstone"
SUB = "Phase_6_Summ"
HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.normpath(os.path.join(HERE, "..", "..", "tools"))

N_HELDOUT = int(os.environ.get("P6_HELDOUT", 600))
SEED = 781

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
    try:
        return hf_hub_download(REPO, f"{SUB}/{name}", repo_type="dataset")
    except Exception as e:
        # The spec's path may not be the repo's: find the Phase-6 file by name
        # rather than dying in minute one of an unattended run.
        from huggingface_hub import HfApi
        want = os.path.basename(name).lower()
        files = [f for f in HfApi().list_repo_files(REPO, repo_type="dataset")
                 if "phase_6" in f.lower() or "phase6" in f.lower()]
        hit = [f for f in files if os.path.basename(f).lower() == want] or \
              [f for f in files if want.split(".")[0] in os.path.basename(f).lower()
               and f.lower().endswith((".json", ".jsonl"))]
        if not hit:
            raise SystemExit(f"{SUB}/{name} not found ({e}); Phase-6 files in the repo: {files}")
        log(f"  {SUB}/{name} not at the spec's path; using {hit[0]}")
        return hf_hub_download(REPO, hit[0], repo_type="dataset")


def _str(x):
    if isinstance(x, dict):                      # nested record: its longest text value
        x = max((_str(v) for v in x.values()), key=len, default="")
    if isinstance(x, list):
        return chr(10).join(_str(s) for s in x)
    return "" if x is None else str(x)


_DOC_KEYS = ("document", "text", "doc", "judgement", "judgment", "input", "source")
_SUM_KEYS = ("summary", "summaries", "reference_summary", "reference", "abstract", "headnote",
             "target", "output")
_ID_KEYS = ("id", "doc_id", "document_id", "case_id", "name")


def _text_stats(recs, skip=()):
    """{field: (share of records non-empty, mean words when non-empty)}."""
    seen = {}
    for r in recs:
        for k, v in r.items():
            if k in skip or not isinstance(v, (str, list, dict)):
                continue
            n = nwords(_str(v))
            a = seen.setdefault(k, [0, 0])
            a[0] += bool(n)
            a[1] += n
    return {k: (round(a[0] / len(recs), 3), round(a[1] / max(a[0], 1), 1)) for k, a in seen.items()}


def _choose(recs, names, skip, what, min_words):
    """A listed field name that is mostly filled, else the longest mostly-filled
    text field.  The field is found from the file, so a renamed one cannot stop
    an unattended run; only a file with no such field at all does."""
    st = _text_stats(recs, skip)
    for k in names:
        if st.get(k, (0, 0))[0] >= 0.5:
            return k, st
    other = [k for k, (share, words) in st.items() if share >= 0.5 and words >= min_words]
    if not other:
        raise SystemExit(f"no {what} field; fields (non-empty share, mean words): {st}")
    return max(other, key=lambda k: st[k][1]), st


def read_split(split):
    """[{id, doc, summ|None}] in file order.  Only train's summary field is read.
    Any other field in test.json is reported by NAME and its values never touched."""
    name = {"train": "Train/train.json", "test": "Test/test.json"}[split]
    with open(fetch(name), encoding="utf-8") as f:
        raw = f.read()
    try:
        x = json.loads(raw)
    except json.JSONDecodeError:
        x = [json.loads(l) for l in raw.splitlines() if l.strip()]
    if isinstance(x, dict):
        x = [dict(v, id=k) if isinstance(v, dict) else {"id": k, "document": v}
             for k, v in x.items()]
    x = [{str(k).lower(): v for k, v in r.items()} for r in x]
    idk = next((k for k in _ID_KEYS if k in x[0]), None)
    if idk is None:
        raise SystemExit(f"{split}: no id field in record keys {sorted(x[0])}")
    dk = next((k for k in _DOC_KEYS if sum(bool(r.get(k)) for r in x) >= 0.5 * len(x)), None)
    if split == "test":
        if dk is None:
            raise SystemExit(f"test: no document field in record keys {sorted(x[0])}")
        extra = sorted({k for r in x for k in r} - {idk, dk})
        if extra:
            log(f"  !! test.json has extra fields {extra}: names only, values never read")
        return [dict(id=str(r[idk]), doc=_str(r.get(dk)), summ=None) for r in x]
    if dk is None:
        dk, _ = _choose(x, (), (idk,), "document", 200)
    sk, st = _choose(x, _SUM_KEYS, (idk, dk), "summary", 15)
    log(f"  train fields: document={dk!r} summary={sk!r} | other text fields "
        f"(non-empty share, mean words): { {k: v for k, v in st.items() if k != sk} }")
    out = [dict(id=str(r[idk]), doc=_str(r.get(dk)), summ=_str(r.get(sk))) for r in x]
    keep = [r for r in out if nwords(r["doc"]) and nwords(r["summ"])]
    if len(keep) < len(out):
        log(f"  !! dropped {len(out) - len(keep)} train records with an empty document or summary")
    return keep


_CACHE = {}


def load(split):
    """train / test as released; 'fit' and 'dev' are our own split of train, with
    every answer-key record removed (see answer_key_ids)."""
    if split in ("fit", "dev"):
        if split not in _CACHE:
            ids, bad = heldout_ids(), answer_key_ids()
            want = (split == "dev")
            _CACHE[split] = [r for r in load("train")
                             if (r["id"] in ids) == want and r["id"] not in bad]
        return _CACHE[split]
    if split not in _CACHE:
        _CACHE[split] = read_split(split)
    return _CACHE[split]


def answer_key_ids():
    """Train ids whose document is a copy of a TEST document.  Such a record pairs a
    test input with its gold summary, so training or selecting on it is training on
    the test set -- Phase 4's dev.json and Phase 5's dev.json both did this.  They
    are dropped from `fit` and `dev` and counted in the report; using them is a
    `lookup` submission and needs Nithin's explicit go-ahead."""
    f = f"{P6}/results/answer_keys.json"
    if os.path.exists(f):
        with open(f) as fh:
            return set(json.load(fh)["ids"])
    tk = {norm_key(r["doc"]) for r in load("test")}
    bad = sorted(r["id"] for r in load("train") if norm_key(r["doc"]) in tk)
    os.makedirs(f"{P6}/results", exist_ok=True)
    with open(f, "w") as fh:
        json.dump(dict(n=len(bad), ids=bad), fh)
    if bad:
        log(f"  !! {len(bad)} train records are copies of test documents -> excluded "
            f"from fit and dev (results/answer_keys.json)")
    return set(bad)


def heldout_ids():
    """Deterministic held-out set: the N ids with the smallest hash.  Stable across
    machines and reruns, and independent of file order, so every module agrees."""
    f = f"{P6}/results/split.json"
    if os.path.exists(f):
        with open(f) as fh:
            return set(json.load(fh)["dev"])
    recs = load("train")
    rank = sorted(recs, key=lambda r: hashlib.sha1(f"{SEED}:{r['id']}".encode()).hexdigest())
    return {r["id"] for r in rank[:min(N_HELDOUT, len(recs) // 5)]}


def gen_dev_ids(n):
    """The fixed subset of dev that every neural run decodes, so the candidate
    pools of all members line up document for document in stage 3."""
    ids = sorted((r["id"] for r in load("dev")),
                 key=lambda k: hashlib.sha1(f"gen:{k}".encode()).hexdigest())
    return ids[:n]


def write_split():
    os.makedirs(f"{P6}/results", exist_ok=True)
    ids = sorted(heldout_ids())
    with open(f"{P6}/results/split.json", "w") as f:
        json.dump(dict(seed=SEED, n=len(ids), dev=ids), f)
    log(f"  held out {len(ids)} of {len(load('train'))} train docs -> results/split.json")
    return ids


def nwords(t):
    return len(WORD.findall(t.lower()))


def norm_key(t, n=3000):
    return " ".join(WORD.findall(t.lower()))[:n]


def pct(a, qs=(0, 5, 10, 25, 50, 75, 90, 95, 100)):
    a = np.asarray(a, dtype=float)
    if not a.size:
        return {}
    return {f"p{q}": round(float(np.percentile(a, q)), 3) for q in qs} | \
           {"mean": round(float(a.mean()), 3)}


# ------------------------------------------------------------------ scoring --
_TOKENIZER = {"obj": None, "exact": None}


def _tokenizer():
    """rouge_score's own tokenizer when it is installed (then our ROUGE equals the
    official scorer's to the last digit); a stemmer-free fallback otherwise, which
    ranks the same but is not the official number -- flagged in every report."""
    if _TOKENIZER["obj"] is None:
        try:
            from rouge_score import tokenizers
            _TOKENIZER["obj"] = tokenizers.DefaultTokenizer(use_stemmer=True)
            _TOKENIZER["exact"] = True
        except Exception:
            class _Fallback:
                def tokenize(self, text):
                    return [t for t in re.split(r"\s+", re.sub(r"[^a-z0-9]+", " ", text.lower())) if t]
            _TOKENIZER["obj"] = _Fallback()
            _TOKENIZER["exact"] = False
    return _TOKENIZER["obj"]


def toks(text):
    return _tokenizer().tokenize(text)


def _f1(overlap, np_, nr):
    if not np_ or not nr or not overlap:
        return 0.0
    p, r = overlap / np_, overlap / nr
    return 2 * p * r / (p + r)


def rouge1_f(pred_t, ref_t):
    import collections
    cp, cr = collections.Counter(pred_t), collections.Counter(ref_t)
    ov = sum(min(c, cr[t]) for t, c in cp.items())
    return _f1(ov, len(pred_t), len(ref_t))


def lcs_len(a, b):
    """The scorer's own bit-parallel LCS.  O(|a|*|b|/64)."""
    if len(a) > len(b):
        a, b = b, a
    masks = {}
    for i, t in enumerate(a):
        masks[t] = masks.get(t, 0) | (1 << i)
    s = 0
    for t in b:
        m = masks.get(t, 0)
        x = m | s
        y = (s << 1) | 1
        s = x & ~(x - y)
    return bin(s).count("1")


def rougeL_f(pred_t, ref_t):
    if not pred_t or not ref_t:
        return 1.0 if not pred_t and not ref_t else 0.0
    return _f1(lcs_len(pred_t, ref_t), len(pred_t), len(ref_t))


class Bert:
    """One BERTScorer, reused.  Loading roberta-large per call dominates everything."""

    def __init__(self, device=None, batch=16):
        import bert_score
        import torch
        self.batch = batch
        self.s = bert_score.BERTScorer(lang="en", rescale_with_baseline=False,
                                       device=device or ("cuda" if torch.cuda.is_available() else "cpu"))

    def f1(self, preds, refs):
        _, _, f = self.s.score(list(preds), list(refs), batch_size=self.batch, verbose=False)
        return np.asarray(f, dtype=float)


_BERT = {}


def bert(device=None):
    if "s" not in _BERT:
        _BERT["s"] = Bert(device)
    return _BERT["s"]


def score_summaries(preds, refs, do_bert=True, per_doc=False):
    """preds/refs: {id: text}.  Returns the three official numbers over the shared ids."""
    ids = [k for k in refs if k in preds]
    miss = [k for k in refs if k not in preds]
    if miss:
        raise SystemExit(f"score: {len(miss)} reference ids have no prediction (first: {miss[:3]})")
    r1, rl, wl = [], [], []
    for k in ids:
        pt, rt = toks(preds[k].strip()), toks(refs[k].strip())
        r1.append(rouge1_f(pt, rt))
        rl.append(rougeL_f(pt, rt))
        wl.append(nwords(preds[k]))
    out = dict(n=len(ids), rouge1=float(np.mean(r1)), rougeL=float(np.mean(rl)),
               pred_words=float(np.mean(wl)), rouge_exact=bool(_TOKENIZER["exact"]))
    if do_bert:
        f = bert().f1([preds[k].strip() for k in ids], [refs[k].strip() for k in ids])
        out["bertscore"] = float(f.mean())
        if per_doc:
            out["per_doc_bert"] = {k: float(v) for k, v in zip(ids, f)}
    if per_doc:
        out["per_doc_rouge1"] = dict(zip(ids, map(float, r1)))
    return out


def refs_of(split="dev"):
    return {r["id"]: r["summ"] for r in load(split)}


# ------------------------------------------------------------ submissions ----
KINDS = ("model", "extractive", "hybrid", "lookup")


def write_submission(tag, summaries, note="", kind="model", dev_scores=None, out_root=None,
                     test_recs=None):
    """{id: summary} -> submissions/<tag>/{prediction.json, submission.zip} + _upload copy.

    `kind` lands in info.json and in the report: 'extractive' means the text was
    lifted from the test document itself (legitimate, but not abstractive);
    'lookup' means anything read out of the Phase-1 corpus, which may only be
    submitted after Nithin's explicit go-ahead and must be reported as lookup.
    """
    if kind not in KINDS:
        raise SystemExit(f"kind must be one of {KINDS}")
    test_recs = test_recs if test_recs is not None else load("test")
    out_root = out_root or os.path.join(P6, "submissions")
    d = os.path.join(out_root, tag)
    os.makedirs(d, exist_ok=True)
    preds, short = [], 0
    for r in test_recs:
        s = (summaries.get(r["id"]) or "").strip()
        if not s:                                   # the scorer rejects empty summaries
            s = " ".join(r["doc"].split()[:120]) or "No summary available."
            short += 1
        preds.append({"id": r["id"], "summary": s})
    if short:
        log(f"  !! {short} empty summaries were backfilled from the document head")
    assert len({p["id"] for p in preds}) == len(preds) == len(test_recs)
    pj = os.path.join(d, "prediction.json")
    with open(pj, "w", encoding="utf-8") as f:
        json.dump(preds, f, ensure_ascii=False)
    ok = validate(pj)
    zp = os.path.join(d, "submission.zip")
    with zipfile.ZipFile(zp, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(pj, arcname="prediction.json")
    with zipfile.ZipFile(zp) as z:
        assert z.namelist() == ["prediction.json"]
        assert json.loads(z.read("prediction.json")) == preds
    up = os.path.join(out_root, "_upload")
    os.makedirs(up, exist_ok=True)
    shutil.copyfile(zp, os.path.join(up, f"submission_{tag}.zip"))
    wl = [nwords(p["summary"]) for p in preds]
    info = dict(tag=tag, note=note, kind=kind, n_docs=len(preds), words=pct(wl),
                validated=ok, dev=dev_scores or {})
    with open(os.path.join(d, "info.json"), "w") as f:
        json.dump(info, f, indent=1)
    log(f"  WROTE {zp}  ({len(preds)} docs, {np.mean(wl):.0f} words mean, {kind}, validated={ok})")
    return zp


def validate(pj):
    import subprocess
    tool = os.path.join(TOOLS, "cs781_submit.py")
    if not os.path.exists(tool):
        log(f"  (validator not found at {tool}; format checked in-process only)")
        return None
    cmd = [sys.executable, tool, "validate", "--phase", "6", "--prediction", pj]
    try:
        cmd += ["--test", fetch("Test/test.json")]
    except Exception:
        pass
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        log("  VALIDATION FAILED:\n" + (r.stdout + r.stderr).strip())
        return False
    return True


# ---------------------------------------------------------------- the gate ---
SENT_END = re.compile(r"(?<=[.;:?!])\s+(?=[\"'(\[]?[A-Z0-9])|\n{2,}")
HEAD_MARK = re.compile(r"\b(HEADNOTE|HEAD\s?NOTE|SUMMARY|JUDGMENT|JUDGEMENT|ORDER|"
                       r"CIVIL APPELLATE JURISDICTION|CRIMINAL APPELLATE JURISDICTION)\b")
YEAR = re.compile(r"\b(19[0-9]{2}|20[0-2][0-9])\b")


def sentences(text, min_words=4):
    out = []
    for part in re.split(r"\n{2,}", text):
        for s in SENT_END.split(part):
            s = " ".join(s.split())
            if nwords(s) >= min_words:
                out.append(s)
    return out


def ngrams(tk, n):
    return {tuple(tk[i:i + n]) for i in range(len(tk) - n + 1)}


def copy_rate(summ_t, doc_t, n=8):
    """Fraction of the summary's n-grams that occur verbatim in the document.
    High values mean the reference is sitting inside the input text."""
    s, d = ngrams(summ_t, n), ngrams(doc_t, n)
    return len(s & d) / len(s) if s else 0.0


def novel_rate(summ_t, doc_t, n=1):
    s, d = ngrams(summ_t, n), ngrams(doc_t, n)
    return 1 - len(s & d) / len(s) if s else 0.0


def oracle_extractive(doc_sents, ref, budget):
    """Greedy ROUGE-1 oracle over the document's own sentences, stopped at `budget`
    words.  The ceiling any extractive system can reach on this document."""
    rt = toks(ref)
    cand = [toks(s) for s in doc_sents]
    chosen, cur, used = [], [], set()
    best = 0.0
    while True:
        gain, pick = 0.0, -1
        for i, c in enumerate(cand):
            if i in used or len(cur) + len(c) > budget * 1.25:
                continue
            v = rouge1_f(cur + c, rt)
            if v > gain:
                gain, pick = v, i
        if pick < 0 or gain <= best + 1e-6:
            break
        used.add(pick)
        chosen.append(pick)
        cur = cur + cand[pick]
        best = gain
    return best, [doc_sents[i] for i in sorted(chosen)]


def gate(a):
    os.makedirs(f"{P6}/results", exist_ok=True)
    tr, te = load("train"), load("test")
    G = {"n_docs": dict(train=len(tr), test=len(te))}
    log(f"docs: train {len(tr)}  test {len(te)}")

    # --- 1. shape: how long is the input, how long must the output be ---------
    dw = np.array([nwords(r["doc"]) for r in tr])
    sw = np.array([nwords(r["summ"]) for r in tr])
    tw = np.array([nwords(r["doc"]) for r in te])
    G["train_doc_words"], G["train_summ_words"] = pct(dw), pct(sw)
    G["test_doc_words"] = pct(tw)
    G["compression"] = pct(sw / np.maximum(dw, 1))
    log(f"train doc words:   {G['train_doc_words']}")
    log(f"train summ words:  {G['train_summ_words']}")
    log(f"test  doc words:   {G['test_doc_words']}")
    log(f"compression ratio: {G['compression']}")
    # length model: is the reference length predictable from the input length?
    ld, ls = np.log(np.maximum(dw, 1)), np.log(np.maximum(sw, 1))
    A = np.vstack([ld, np.ones_like(ld)]).T
    coef, *_ = np.linalg.lstsq(A, ls, rcond=None)
    pred = np.exp(A @ coef)
    G["len_model"] = dict(log_slope=round(float(coef[0]), 4), log_intercept=round(float(coef[1]), 4),
                          r=round(float(np.corrcoef(ld, ls)[0, 1]), 4),
                          median_abs_rel_err=round(float(np.median(np.abs(pred - sw) / sw)), 4),
                          const_median_abs_rel_err=round(
                              float(np.median(np.abs(np.median(sw) - sw) / sw)), 4))
    log(f"length model: {G['len_model']}  "
        f"(use the fitted length only if its error beats the constant's)")

    # --- 2. is the reference text inside the input? --------------------------
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(tr), size=min(a.sample, len(tr)), replace=False)
    cr8, cr1, nov1, nov2, blk = [], [], [], [], 0
    for i in idx:
        st, dt = toks(tr[i]["summ"]), toks(tr[i]["doc"])
        cr8.append(copy_rate(st, dt, 8))
        cr1.append(copy_rate(st, dt, 1))
        nov1.append(novel_rate(st, dt, 1))
        nov2.append(novel_rate(st, dt, 2))
        if cr8[-1] > 0.5:
            blk += 1
    G["summary_in_document"] = dict(ngram8_copy_rate=pct(cr8), unigram_copy_rate=pct(cr1),
                                    novel_unigram=pct(nov1), novel_bigram=pct(nov2),
                                    docs_over_half_copied=int(blk), sampled=len(idx))
    log(f"summary inside the document: 8-gram copy rate {G['summary_in_document']['ngram8_copy_rate']}")
    log(f"  novel unigrams {G['summary_in_document']['novel_unigram']['mean']:.3f} | "
        f"novel bigrams {G['summary_in_document']['novel_bigram']['mean']:.3f} | "
        f"{blk}/{len(idx)} docs have >50% of the summary verbatim in the input")
    if blk > 0.2 * len(idx):
        log("  !! THE ANSWER IS LARGELY IN THE INPUT -- an extraction rule may beat any model")

    # --- 3. structural markers (pre-1994 SC judgments carry a headnote) ------
    G["markers"] = {}
    for nm, recs in (("train", tr), ("test", te)):
        c = {}
        for r in recs[:4000]:
            for m in set(HEAD_MARK.findall(r["doc"][:6000].upper())):
                c[m] = c.get(m, 0) + 1
        G["markers"][nm] = {k: round(v / min(len(recs), 4000), 4) for k, v in sorted(c.items())}
    log(f"markers in the first 6k chars: train {G['markers']['train']}")
    log(f"                               test  {G['markers']['test']}")
    yrs = [int(y) for r in tr[:3000] for y in YEAR.findall(r["doc"][:4000])[:4]]
    G["years"] = pct([y for y in yrs if 1900 < y < 2030]) if yrs else {}
    log(f"years mentioned early in train docs: {G['years']}")

    # --- 4. duplicates and answer keys --------------------------------------
    tk = {}
    for r in te:
        tk.setdefault(norm_key(r["doc"]), []).append(r["id"])
    dup_tt = [(r["id"], tk[norm_key(r["doc"])]) for r in tr if norm_key(r["doc"]) in tk]
    seen, dup_tr = {}, 0
    for r in tr:
        k = norm_key(r["doc"])
        if k in seen:
            dup_tr += 1
        seen[k] = r["id"]

    G["duplicates"] = dict(train_docs_copying_a_test_doc=len(dup_tt),
                           examples=[[x, y] for x, y in dup_tt[:10]],
                           train_internal_duplicate_docs=dup_tr,
                           distinct_train_docs=len(seen),
                           duplicate_ids_train=len(tr) - len({r["id"] for r in tr}),
                           duplicate_ids_test=len(te) - len({r["id"] for r in te}))
    log(f"duplicates: {len(dup_tt)} train docs are copies of a test doc "
        f"(each one is a labelled answer key) | {dup_tr} internal train duplicates")
    if dup_tt:
        log("  !! those train records are excluded from our held-out split and flagged in the report")

    # --- 5. what extraction can reach at all --------------------------------
    sub = idx[:min(a.oracle, len(idx))]
    orc, lead, nsent = [], [], []
    for i in sub:
        ss = sentences(tr[i]["doc"])
        nsent.append(len(ss))
        if not ss:
            continue
        b = int(nwords(tr[i]["summ"]))
        o, _ = oracle_extractive(ss[:250], tr[i]["summ"], b)
        orc.append(o)
        head, cur = [], 0
        for s in ss:
            head.append(s)
            cur += nwords(s)
            if cur >= b:
                break
        lead.append(rouge1_f(toks(" ".join(head)), toks(tr[i]["summ"])))
    G["extractive"] = dict(oracle_rouge1=pct(orc), lead_rouge1=pct(lead),
                           sentences_per_doc=pct(nsent), sampled=len(orc))
    log(f"extractive oracle ROUGE-1 {np.mean(orc):.4f} | lead baseline {np.mean(lead):.4f} "
        f"| {np.mean(nsent):.0f} sentences/doc  (n={len(orc)})")

    # --- 6. where the scorer stops looking ----------------------------------
    # bert_score encodes with truncation at roberta-large's 512-token window, so
    # for long references only the first ~380 words of EITHER side are scored.
    # This is measured, not assumed: score a reference's own prefixes against it.
    # If BERTScore saturates at k words while ROUGE keeps rising, the length dial
    # is one-sided above k -- overshooting is free on the primary metric.
    G["bert_window"] = {}
    try:
        longs = [r for r in tr if nwords(r["summ"]) > 600][:a.window]
        if longs:
            ks = [150, 250, 350, 450, 550, 700, 900]
            rows = []
            for k in ks:
                pr = [" ".join(r["summ"].split()[:k]) for r in longs]
                rf = [r["summ"] for r in longs]
                b = float(bert().f1(pr, rf).mean())
                r1 = float(np.mean([rouge1_f(toks(p), toks(x)) for p, x in zip(pr, rf)]))
                rows.append(dict(words=k, bertscore=round(b, 5), rouge1=round(r1, 5)))
                log(f"  window probe: first {k:4d} words of the reference vs the whole "
                    f"reference -> bert {b:.5f} rouge1 {r1:.5f}")
            sat = next((r["words"] for r in rows if r["bertscore"] > 0.995), None)
            G["bert_window"] = dict(rows=rows, n=len(longs), saturates_at_words=sat)
            if sat:
                log(f"  !! BERTScore saturates at {sat} words: text past that point is "
                    f"not scored at all, in the prediction or in the reference")
        else:
            G["bert_window"] = dict(note="no reference over 600 words; the 512-token "
                                         "window never binds", n=0)
            log("  window probe: no reference over 600 words -- the scorer's 512-token "
                "window never binds, so length is two-sided as in Phase 1")
    except Exception as e:                      # bert_score absent (local smoke tests)
        G["bert_window"] = dict(error=str(e)[:200])
        log(f"  window probe skipped: {str(e)[:120]}")

    # --- 7. our split and a fingerprint of the source ------------------------
    dev_ids = write_split()
    bad = [i for i, _ in dup_tt]
    G["heldout"] = dict(n=len(dev_ids), excluded_answer_keys=len(set(bad) & set(dev_ids)))
    G["fingerprint"] = [dict(id=r["id"], doc_head=" ".join(r["doc"].split())[:300],
                             summ_head=" ".join(r["summ"].split())[:300],
                             doc_words=nwords(r["doc"]), summ_words=nwords(r["summ"]))
                        for r in tr[:3]]
    G["rouge_exact"] = bool(_TOKENIZER["exact"])
    with open(f"{P6}/results/gate.json", "w") as f:
        json.dump(G, f, indent=1)
    log("\n--- source fingerprint (identify the dataset, then check if its test refs are public) ---")
    for r in G["fingerprint"]:
        log(f"[{r['id']}] doc({r['doc_words']}w): {r['doc_head'][:200]}")
        log(f"          summ({r['summ_words']}w): {r['summ_head'][:200]}")
    log(f"\nwrote {P6}/results/gate.json")
    return G


def cmd_score(a):
    with open(a.pred, encoding="utf-8") as f:
        p = json.load(f)
    if isinstance(p, list):
        p = {r["id"]: r["summary"] for r in p}
    s = score_summaries(p, refs_of(a.split), do_bert=not a.no_bert)
    log(json.dumps(s, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    g = sp.add_parser("gate")
    g.add_argument("--sample", type=int, default=1500, help="docs for the copy-rate scan")
    g.add_argument("--oracle", type=int, default=60, help="docs for the extractive oracle")
    g.add_argument("--window", type=int, default=24, help="long refs for the BERTScore window probe")
    g.set_defaults(fn=gate)
    s = sp.add_parser("split")
    s.set_defaults(fn=lambda a: write_split())
    c = sp.add_parser("score")
    c.add_argument("--pred", required=True)
    c.add_argument("--split", default="dev")
    c.add_argument("--no-bert", action="store_true")
    c.set_defaults(fn=cmd_score)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
