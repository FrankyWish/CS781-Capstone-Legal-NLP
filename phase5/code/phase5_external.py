#!/usr/bin/env python3
"""CS-781 Phase 5 -- silver statute labels from the Phase-1 judgment corpus (distant supervision). CPU.

    python phase5_external.py scan       # one pass over SC + HC judgments -> external/candidates.jsonl
    python phase5_external.py finalize   # learn cite->label map on train copies -> external/silver.jsonl + report
    python phase5_external.py scan --corpus-path X.json   # local file instead of HF (smoke test)

IDEA. ILSI (IL-TUR lsi, which this phase is) was built by distant supervision:
Indian Kanoon judgments, labelled with the target sections they cite, and only
the facts kept. The Phase-1 corpus (permitted; external data is allowed if the
report describes it) holds hundreds of thousands of SC/HC judgments. Redoing the
construction on it gives silver training documents, and many of them for the rare
statutes that decide Macro-F1.

  cites   every "Section(s) N[A][(x)] ... of the <act>" mention, tagged with the
          act named right after the list (IPC / CrPC / other / none). The label
          list mixes IPC sections with numbers that are CrPC provisions in
          practice (156, 161, 164, 173, 438, 482 ...), so the act tag is a feature
          and the map from (number, act) to label ID is LEARNED, not assumed.
  map     learned in `finalize` from corpus judgments that contain a TRAIN
          document's facts (>= 2 of its 12-word needles): for each cite key, the
          label it co-occurs with; kept if precision >= --key-prec.
  labels  per statute l the labeller's precision and recall are measured on those
          train copies. Positives are used only where precision >= --pos-prec;
          absence is used as a NEGATIVE only where recall >= --neg-rec; everything
          else is MASKED (no loss), so an unreliable labeller cannot teach that a
          statute is absent.
  text    citing sentences (any section/act/article mention) are deleted, so the
          model cannot learn to read section numbers that the facts-only test
          text does not show; the rest is cut to the region where the train facts
          were found in their judgments (measured, `facts_end_frac` p90).

EXCLUSIONS -- what keeps this legitimate:
  * a judgment containing a DEV or TEST document's facts is never written. For
    TEST matches nothing but the count is kept (no cites stored): there is no
    answer-key artifact on disk. DEV matches are scored once, as the size of the
    leak the brief asked to measure, then dropped.
  * judgments matching TRAIN are not written (duplicates of gold) but are the
    labeller's validation set.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from phase5_data import P5, load, label_space, multi_hot, score_on, f1_per_label, log  # noqa: E402

EXT = f"{P5}/external"
WORD = re.compile(r"[a-z0-9]+")
N = 12
MINW = 7            # needles start on a word of >= 7 chars: the scan skips every shorter word cheaply
CRIM = re.compile(r"\bI\.?\s?P\.?\s?C\b|Penal\s+Code|\bCr\.?\s?P\.?\s?C\b|Criminal\s+Procedure", re.I)
SOURCES = dict(supreme_court="train_supreme_court", high_courts="train_high_courts")

# The 100 target sections as the IL-TUR card names them. Used ONLY to pre-filter
# judgments and to bias sampling toward rare ones; the label IDs themselves are
# learned from train copies in `finalize`.
CARD = ("2 3 4 5 13 34 107 109 114 120 120B 143 147 148 149 155 156 161 164 173 174A 186 188 190 193 200 201 228 "
        "229A 279 294 294(b) 299 300 302 304 304A 304B 306 307 308 313 320 323 324 325 326 332 336 337 338 341 342 "
        "353 354 363 364 365 366 366A 375 376 376(2) 379 380 384 389 392 394 395 397 406 409 411 415 417 419 420 "
        "427 436 437 438 447 448 450 452 457 465 467 468 471 482 494 498 498A 500 504 506 509 511").split()
CARD_NUMS = {re.match(r"\d+", s).group(0) for s in CARD}

SEC = re.compile(
    r"\b(?:sections?|secs?\.|sec\b|ss\.|s\.|u/s\.?|u/ss\.?|under\s+s\.)\s*"
    r"((?:\d{1,3}\s*-?\s*[A-Za-z]?\b(?:\s*\(\s*[0-9a-z]{1,4}\s*\))*\s*(?:,|/|&|and|or|r/w|read\s+with|to)?\s*){1,12})",
    re.I)
NUMS = re.compile(r"(\d{1,3})\s*-?\s*([A-Za-z]?)\b((?:\s*\(\s*[0-9a-z]{1,4}\s*\))?)")
ACT = [("IPC", re.compile(r"^\W{0,3}(?:of\s+(?:the\s+)?)?(?:I\.?\s?P\.?\s?C|Indian\s+Penal\s+Code|Penal\s+Code)", re.I)),
       ("CrPC", re.compile(r"^\W{0,3}(?:of\s+(?:the\s+)?)?(?:Cr\.?\s?P\.?\s?C|Code\s+of\s+Criminal\s+Procedure|Criminal\s+Procedure\s+Code)", re.I)),
       ("other", re.compile(r"^\W{0,3}of\s+(?:the\s+)?[A-Z][\w.,()' ]{2,60}?(?:Act|Code|Rules?|Order)\b"))]
ANY_CITE = re.compile(r"\b(?:sections?|secs?\.|u/s|articles?|I\.?P\.?C|Cr\.?P\.?C|Penal\s+Code|Act,?\s+\d{4})\b", re.I)
SENT = re.compile(r"(?<=[.?!;])\s+(?=[A-Z(\"'])|\n+")


def cite_keys(text):
    """Counter of 'N[S]|ACT' keys, e.g. '302|IPC', '438|CrPC', '34|-' and '376(2)|IPC'."""
    out = collections.Counter()
    for m in SEC.finditer(text):
        after = text[m.end(): m.end() + 80]
        act = next((name for name, rx in ACT if rx.search(after)), "-")
        for num, suf, sub in NUMS.findall(m.group(1)):
            base = num + suf.upper()
            out[f"{base}|{act}"] += 1
            sub = re.sub(r"\s", "", sub).lower()
            if sub:
                out[f"{base}{sub}|{act}"] += 1
    return out


def clean_text(text, max_chars):
    """Drop every sentence that cites a provision; keep the first max_chars of the rest."""
    keep, n = [], 0
    for s in SENT.split(text):
        s = s.strip()
        if not s or ANY_CITE.search(s):
            continue
        keep.append(s)
        n += len(s) + 1
        if n >= max_chars:
            break
    return " ".join(keep)


def needles(text, k):
    w = WORD.findall(text.lower())
    out = []
    if len(w) < N + 5:
        return out
    for j in range(k):
        i = int((j + 0.5) * (len(w) - N) / k)
        while i < len(w) - N and len(w[i]) < MINW:
            i += 1
        if i < len(w) - N:
            out.append((" ".join(w[i:i + N]), j))
    return out


def stream_records(path, bufsize=1 << 22):
    """Every record of a large JSON array without loading the file (Phase 1/4 pattern)."""
    dec = json.JSONDecoder()
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        buf = f.read(bufsize)
        pos = buf.find("{")
        if pos < 0:
            return
        while True:
            try:
                obj, end = dec.raw_decode(buf, pos)
            except ValueError:
                more = f.read(bufsize)
                if not more:
                    return
                buf = buf[pos:] + more
                pos = buf.find("{")
                if pos < 0:
                    return
                continue
            yield obj
            k = end
            while k < len(buf) and buf[k] in ", \n\r\t":
                k += 1
            if k >= len(buf) or buf[k] == "]":
                more = f.read(bufsize)
                if not more:
                    return
                buf = buf[k:] + more
                pos = buf.find("{")
                if pos < 0:
                    return
            elif len(buf) - k < (1 << 20):
                more = f.read(bufsize)
                buf = buf[k:] + (more or "")
                pos = buf.find("{")
                if pos < 0:
                    return
            else:
                pos = k


# ------------------------------------------------------------------- scan ----
_W = {}


def _init_worker(index, first, rare_nums, common_keep, max_chars, min_words):
    _W.update(index=index, first=first, rare=rare_nums, keep=common_keep, max_chars=max_chars, min_words=min_words)


def _scan_one(arg):
    """One judgment -> ('skip', reason) | ('test', ids) | ('dev', rows, cites) | ('train', [(row, pos)], cites, n)
    | ('cand', cites, rare, body, n). Runs in a worker process."""
    k, text = arg
    if not CRIM.search(text):
        return ("skip", "not_criminal")
    cites = cite_keys(text)
    nums = {re.match(r"\d+", c).group(0) for c in cites}
    if not (nums & CARD_NUMS):
        return ("skip", "no_target_cite")
    index, first = _W["index"], _W["first"]
    w = WORD.findall(text.lower())
    hits = collections.defaultdict(set)
    pos0 = {}
    for i in range(len(w) - N + 1):
        wi = w[i]
        if len(wi) >= MINW and wi in first:
            for key, j in index.get(" ".join(w[i:i + N]), ()):
                hits[key].add(j)
                pos0.setdefault(key, i / max(1, len(w)))
    m = [key for key, js in hits.items() if len(js) >= 2]
    if any(key[0] == "test" for key in m):
        return ("test", [key[1] for key in m if key[0] == "test"])
    if any(key[0] == "dev" for key in m):
        return ("dev", [key[1] for key in m if key[0] == "dev"], dict(cites))
    tr = [key for key in m if key[0] == "train"]
    if tr:
        return ("train", [(key[1], pos0[key]) for key in tr], dict(cites), len(w))
    rare = bool(nums & _W["rare"])
    if not rare and random.Random(k).random() > _W["keep"]:
        return ("skip", "common_dropped")
    body = clean_text(text, _W["max_chars"])
    if len(body.split()) < _W["min_words"]:
        return ("skip", "too_short")
    return ("cand", dict(cites), rare, body, len(w))


def cmd_scan(a):
    from multiprocessing import Pool
    os.makedirs(EXT, exist_ok=True)
    recs = {s: load(s) for s in ("train", "dev", "test")}
    index = collections.defaultdict(list)
    first = set()
    for s in recs:
        for i, r in enumerate(recs[s]):
            for nd, j in needles(r["text"], 10 if s != "train" else 4):
                index[nd].append(((s, i), j))
                first.add(nd.split(" ", 1)[0])
    index = dict(index)
    log(f"  indexed {sum(len(v) for v in recs.values())} docs, {len(index)} needles, {len(first)} start words")
    # rarity of card sections, assuming the competition IDs follow the card order (checked in finalize)
    n, cnt = label_space()
    rare_nums = {re.match(r"\d+", CARD[l]).group(0) for l in range(min(n, len(CARD))) if cnt[l] < a.rare_below}
    log(f"  rare (< {a.rare_below} train docs) section numbers: {sorted(rare_nums, key=int)}")

    out = open(f"{EXT}/candidates.jsonl.tmp", "w", encoding="utf-8")
    pairs = open(f"{EXT}/train_pairs.jsonl.tmp", "w", encoding="utf-8")
    st = collections.Counter()
    dev_leak = []
    test_matched = set()
    from huggingface_hub import hf_hub_download
    paths = [a.corpus_path] if a.corpus_path else [
        hf_hub_download("Exploration-Lab/CS-781-Capstone", f"Phase_1_Pretraining/Train/{SOURCES[s]}.json",
                        repo_type="dataset", local_dir=a.local_dir) for s in a.sources.split(",")]
    log(f"  scanning with {a.procs} worker processes")
    with Pool(a.procs, initializer=_init_worker,
              initargs=(index, first, rare_nums, a.common_keep, a.max_chars, a.min_words)) as pool:
        for path in paths:
            t0 = time.time()
            src = os.path.basename(path).replace(".json", "")
            feed = ((k, rec.get("text") or "") for k, rec in enumerate(stream_records(path)))
            k = -1
            for k, res in enumerate(pool.imap(_scan_one, feed, chunksize=8)):
                st["records"] += 1
                if k % 50000 == 0 and k:
                    log(f"    {src}: {k} records, {st['written']} written, {st['train_copy']} train / "
                        f"{st['dev_copy']} dev / {st['test_copy']} test copies ({time.time()-t0:.0f}s)")
                kind = res[0]
                if kind == "skip":
                    st[res[1]] += 1
                elif kind == "test":
                    st["test_copy"] += 1
                    test_matched |= set(res[1])
                elif kind == "dev":
                    st["dev_copy"] += 1
                    dev_leak += [dict(row=r, cites=res[2]) for r in res[1]]
                elif kind == "train":
                    st["train_copy"] += 1
                    for row, pos in res[1]:
                        pairs.write(json.dumps(dict(row=row, cites=res[2], pos=pos, n_words=res[3])) + "\n")
                else:
                    _, cites, rare, body, nw = res
                    out.write(json.dumps(dict(id=f"{src}:{k}", text=body, cites=cites, rare=rare,
                                              n_words=nw)) + "\n")
                    st["written"] += 1
                    st["written_rare"] += int(rare)
            log(f"  {src}: {k+1} records in {time.time()-t0:.0f}s")
    out.close()
    pairs.close()
    os.replace(f"{EXT}/candidates.jsonl.tmp", f"{EXT}/candidates.jsonl")
    os.replace(f"{EXT}/train_pairs.jsonl.tmp", f"{EXT}/train_pairs.jsonl")
    json.dump(dev_leak, open(f"{EXT}/dev_leak.json", "w"))
    res = dict(stats=dict(st), test_docs_found_in_corpus=len(test_matched),
               test_frac_found=len(test_matched) / max(1, len(recs["test"])),
               dev_docs_found_in_corpus=len({d["row"] for d in dev_leak}),
               dev_frac_found=len({d["row"] for d in dev_leak}) / max(1, len(recs["dev"])))
    json.dump(res, open(f"{EXT}/scan_stats.json", "w"), indent=1)
    log(json.dumps(res, indent=1))
    log("  test matches: only the count is kept. Nothing about them is on disk.")


# --------------------------------------------------------------- finalize ----
def cmd_finalize(a):
    tr, dv = load("train"), load("dev")
    n, cnt = label_space()
    pairs = [json.loads(l) for l in open(f"{EXT}/train_pairs.jsonl", encoding="utf-8")]
    log(f"  {len(pairs)} corpus copies of train documents")
    # 1. key -> label map
    co = collections.defaultdict(collections.Counter)
    seen = collections.Counter()
    for p in pairs:
        gold = set(tr[p["row"]]["labels"])
        for key in p["cites"]:
            seen[key] += 1
            for g in gold:
                co[key][g] += 1
    kmap = {}
    for key, c in co.items():
        lab, k = c.most_common(1)[0]
        if seen[key] >= a.key_min and k / seen[key] >= a.key_prec:
            kmap[key] = int(lab)
    # card-order check: does '302|IPC' map to the card position of 302?
    agree = [(key, lab, CARD[lab] if lab < len(CARD) else None) for key, lab in kmap.items() if key.endswith("|IPC")]
    card_ok = np.mean([key.split("|")[0] == c for key, lab, c in agree]) if agree else float("nan")
    log(f"  cite-key map: {len(kmap)} keys; IPC keys agreeing with the card's label order: {card_ok:.3f}")

    def labeller(cites):
        y = np.zeros(n, dtype=np.uint8)
        for key in cites:
            if key in kmap:
                y[kmap[key]] = 1
        return y

    # 2. per-statute reliability of the labeller on train copies
    Yg = np.stack([multi_hot([tr[p["row"]]], n)[0] for p in pairs]) if pairs else np.zeros((0, n), np.uint8)
    Yl = np.stack([labeller(p["cites"]) for p in pairs]) if pairs else np.zeros((0, n), np.uint8)
    tp = (Yg & Yl).sum(0)
    prec = tp / np.maximum(1, Yl.sum(0))
    rec = tp / np.maximum(1, Yg.sum(0))
    pos_ok = (prec >= a.pos_prec) & (Yl.sum(0) >= 3)
    neg_ok = (rec >= a.neg_rec) & (Yg.sum(0) >= 3)
    log(f"  labeller on train copies: micro P {tp.sum()/max(1,Yl.sum()):.3f} R {tp.sum()/max(1,Yg.sum()):.3f} | "
        f"statutes usable as positive {int(pos_ok.sum())}/{n}, absence usable as negative {int(neg_ok.sum())}/{n}")
    pos = np.array([p["pos"] for p in pairs]) if pairs else np.array([0.3])
    frac_end = float(np.clip(np.percentile(pos, 90) + 0.15, 0.3, 1.0))
    log(f"  where train facts start inside their judgments: p10 {np.percentile(pos,10):.2f} "
        f"p50 {np.percentile(pos,50):.2f} p90 {np.percentile(pos,90):.2f} -> silver text cut at {frac_end:.2f}")

    # 3. the leak, measured on dev once (brief: measure and report, never use)
    leak = json.load(open(f"{EXT}/dev_leak.json"))
    if leak:
        rows = sorted({d["row"] for d in leak})
        P = np.zeros((len(dv), n), dtype=np.uint8)
        for d in leak:
            P[d["row"]] |= labeller(d["cites"])
        Yd = multi_hot(dv, n)
        leak_scores = dict(on_matched=score_on(Yd[rows], P[rows]), all_dev=score_on(Yd, P), n=len(rows))
    else:
        leak_scores = {}
    log(f"  LEAK (dev docs whose judgment is in the corpus, labels read off its cites): {json.dumps(leak_scores)}")

    # 4. silver set
    out = open(f"{EXT}/silver.jsonl.tmp", "w", encoding="utf-8")
    wrote, per = 0, np.zeros(n, dtype=np.int64)
    for line in open(f"{EXT}/candidates.jsonl", encoding="utf-8"):
        c = json.loads(line)
        y = labeller(c["cites"])
        if not (y & pos_ok).any():
            continue
        y = y & pos_ok
        mask = [int(l) for l in range(n) if not y[l] and not neg_ok[l]]
        text = c["text"]
        text = text[: max(2000, int(len(text) * frac_end))] if a.cut else text
        out.write(json.dumps(dict(id=c["id"], text=text, labels=[int(l) for l in np.nonzero(y)[0]],
                                  mask=mask)) + "\n")
        wrote += 1
        per += y
    out.close()
    os.replace(f"{EXT}/silver.jsonl.tmp", f"{EXT}/silver.jsonl")
    rep = dict(n_pairs=len(pairs), key_map=kmap, card_order_agreement=card_ok, frac_end=frac_end,
               labeller_prec=prec.round(3).tolist(), labeller_rec=rec.round(3).tolist(),
               pos_ok=np.nonzero(pos_ok)[0].tolist(), neg_ok=np.nonzero(neg_ok)[0].tolist(),
               silver_docs=wrote, silver_per_label=per.tolist(), gold_per_label=cnt.tolist(),
               dev_leak=leak_scores)
    json.dump(rep, open(f"{EXT}/silver_report.json", "w"), indent=1)
    gain = [(int(l), int(cnt[l]), int(per[l])) for l in np.argsort(cnt)[:25]]
    log(f"  WROTE {wrote} silver docs -> {EXT}/silver.jsonl")
    log("  rarest 25 statutes (id, gold train docs, silver docs): " + " ".join(f"{l}:{g}+{s}" for l, g, s in gain))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("scan")
    s.add_argument("--sources", default="supreme_court,high_courts")
    s.add_argument("--corpus-path", default=None)
    s.add_argument("--local-dir", default=os.environ.get("CS781_CORPUS_DIR", "/content/p1corpus"))
    s.add_argument("--rare-below", type=int, default=300, help="statutes with fewer train docs are 'rare'")
    s.add_argument("--common-keep", type=float, default=0.15, help="keep rate for judgments citing no rare section")
    s.add_argument("--max-chars", type=int, default=12000)
    s.add_argument("--min-words", type=int, default=80)
    s.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 2),
                   help="worker processes (default: all cores but two, which the GPU job keeps)")
    f = sp.add_parser("finalize")
    f.add_argument("--key-min", type=int, default=5)
    f.add_argument("--key-prec", type=float, default=0.5)
    f.add_argument("--pos-prec", type=float, default=0.6)
    f.add_argument("--neg-rec", type=float, default=0.6)
    f.add_argument("--no-cut", dest="cut", action="store_false")
    a = ap.parse_args()
    {"scan": cmd_scan, "finalize": cmd_finalize}[a.cmd](a)


if __name__ == "__main__":
    main()
