#!/usr/bin/env python3
"""CS-781 Phase 2 -- inspect an external NER corpus and test it for
contamination against fold3 BEFORE any of it is used for training.

    python phase2_external.py --inspect

CPU, a few minutes.

WHY THIS EXISTS

The professor relaxed the data restriction on 2026-10-10: external and
synthetic data may now be used at any training stage, described in the report.
That makes `opennyaiorg/InLegalNER` (Kalamkar et al., NLLP 2022) available --
46,545 human annotations over Indian court judgments, with a label set that
maps almost one-to-one onto this phase's twelve.

It also creates a hazard the rules were incidentally protecting against.
Phase-2's labels come from that same benchmark lineage, and InLegalNER splits
by time: train 1950-2017, dev/test 2018-2022. Phase-2 documents span both eras
-- there are judgments dated 16/11/1953 and 26/03/1976 alongside
W.P.No.15591/2020. If the organisers' fold3 documents sit inside InLegalNER's
TRAIN split, then fine-tuning on it is training on the test set. The
leaderboard would move and the number would mean nothing.

So this script does two things and trains nothing:

  1. prints the actual schema -- field names, label inventory, counts -- so a
     converter is written against what is there rather than a guess
  2. measures overlap against fold1, fold2 and especially fold3, by exact
     prefix hash AND by rare-13-gram containment, which catches reformatted
     or re-whitespaced duplicates that a hash misses

Phase 1 established the habit the hard way: test prompts appeared verbatim in
the permitted corpus, and the overlap check is what turned that into a
deliberate decision instead of an accident. Same check, opposite purpose --
here an overlap is a reason NOT to use the data.
"""

import argparse
import collections
import hashlib
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase2_data import P2, fetch, load_json, log, parse_gold  # noqa: E402

WORD = re.compile(r"\w+")


def norm(t):
    return " ".join(WORD.findall(t.lower()))


def prefix_hash(t, n=3000):
    return hashlib.sha1(norm(t)[:n].encode("utf-8", "ignore")).hexdigest()[:16]


def grams(t, n=13, cap=4000):
    w = norm(t).split()
    step = max(1, len(w) // cap)
    return {" ".join(w[i:i + n]) for i in range(0, max(0, len(w) - n), step)}


def load_parquet(name, paths):
    """Read the repo's parquet shards directly.

    opennyaiorg/InLegalNER is gated, so load_dataset() fails until the terms
    are accepted on the Hub; once they are, the parquet files download with an
    ordinary authenticated request and skip the loader script entirely.
    """
    import pandas as pd
    from huggingface_hub import hf_hub_download
    recs = []
    for rel in paths:
        f = hf_hub_download(name, rel, repo_type="dataset")
        df = pd.read_parquet(f)
        log("  %s -> %d rows, columns %s" % (rel, len(df), list(df.columns)))
        if len(df):
            log("  first row: %s" % str(df.iloc[0].to_dict())[:1200])
        for i, r in df.iterrows():
            d = r.to_dict()
            txt = d.get("text") or d.get("content") or ""
            if not txt and isinstance(d.get("data"), dict):
                txt = d["data"].get("text", "")
            anns = (d.get("annotations") or d.get("entities")
                    or d.get("spans") or d.get("label") or [])
            if hasattr(anns, "tolist"):
                anns = anns.tolist()
            recs.append((d.get("id", i), txt, anns))
    return recs


def load_external(name, split):
    """Return [(id, text, [(start, end, label)])] from whatever shape it uses."""
    recs = []
    try:
        from datasets import load_dataset
        ds = load_dataset(name, split=split)
        log("  loaded via datasets: %s" % ds)
        cols = ds.column_names
        log("  columns: %s" % cols)
        for i, r in enumerate(ds):
            txt = r.get("text") or r.get("data", {}).get("text") \
                or r.get("content") or ""
            anns = (r.get("annotations") or r.get("entities")
                    or r.get("spans") or r.get("label") or [])
            recs.append((r.get("id", i), txt, anns))
            if i == 0:
                log("  first record keys: %s" % sorted(r))
                log("  first record sample: %s"
                    % json.dumps(r, default=str)[:1200])
    except Exception as e:                                   # noqa: BLE001
        log("  load_dataset failed: %r" % e)
        log("  -> pass --files to point at raw JSON instead")
    return recs




# --------------------------------------------------------------- converter --
# InLegalNER (Kalamkar et al., NLLP 2022) carries 14 types; this phase scores
# 12. Ten map directly. The other four are MASKED (-100) rather than taught as
# O, for the same reason COURT and PREC are masked in the weak pipeline:
#
#   LAWYER        one type where we need A.COUNSEL vs R.COUNSEL split by side.
#                 The caption rules make that distinction at 0.74/0.84 exact
#                 precision, so the rules supply it and the external data stays
#                 silent rather than guessing.
#   PROVISION     'Section 166' -- our gold labels the ACT ('MV Act') and not
#                 the provision, so teaching PROVISION as STAT would be wrong
#                 and teaching it as O would suppress a real entity.
#   GPE           'State of Haryana' is RESP in our data and GPE in theirs.
#   OTHER_PERSON  could be any of WIT / APP / RESP / JUDGE here.
MAP = {
    "COURT": "COURT", "PETITIONER": "APP", "RESPONDENT": "RESP",
    "JUDGE": "JUDGE", "DATE": "DATE", "ORG": "AUTH", "STATUTE": "STAT",
    "PRECEDENT": "PREC", "CASE_NUMBER": "CASENO", "CASENUMBER": "CASENO",
    "CASE-NUMBER": "CASENO", "WITNESS": "WIT",
}
MASK = {"LAWYER", "GPE", "PROVISION", "OTHER_PERSON", "OTHER-PERSON", "TYPE"}


def spans_of_record(anns):
    """Label-Studio nesting: annotations[i].result[j].value{start,end,labels}."""
    out = []
    for blk in (anns if isinstance(anns, (list, tuple)) else []):
        res = blk.get("result") if isinstance(blk, dict) else None
        if res is None:
            continue
        for r in res:
            v = r.get("value") if isinstance(r, dict) else None
            if not isinstance(v, dict):
                continue
            labs = v.get("labels")
            if hasattr(labs, "tolist"):
                labs = labs.tolist()
            if not labs:
                continue
            out.append((int(v["start"]), int(v["end"]), str(labs[0]).upper()))
    return out


def drop_contaminated(recs, test_text, thresh=0.5):
    """Remove the individual passages that share text with fold3.

    A document-level threshold is the wrong instrument: one InLegalNER test
    record overlapping one of our test documents shows up as 4.7% of that
    document and slips under any sane document-level cutoff, while the record
    itself may be a verbatim copy. Filtering per record keeps the other 4,500
    and removes exactly the ones that would leak.

    13-grams are rare enough that sharing half of them is copying, not
    coincidence; shared boilerplate ("in the supreme court of india ...")
    rarely reaches that fraction of a 200-character passage.
    """
    t3 = set()
    for t in test_text.values():
        t3 |= grams(t)
    out, dropped = [], []
    for rid, text, anns in recs:
        g = grams(text) if text else set()
        if g and len(g & t3) / len(g) >= thresh:
            dropped.append((rid, len(g & t3) / len(g)))
            continue
        out.append((rid, text, anns))
    if dropped:
        log("  DROPPED %d records overlapping fold3: %s" % (
            len(dropped),
            "  ".join("%s=%.0f%%" % (r, 100 * f) for r, f in dropped[:8])))
    else:
        log("  no individual record overlaps fold3 at >=%.0f%%" % (100 * thresh))
    return out


def convert(recs, tok, labels, label_to_id, a):
    """Pack passages into realistic windows and encode partial BIO tags.

    The passages have a median length of 192 characters against our documents'
    26,640, so one passage per window would train the model on 50-token inputs
    and evaluate it on 1024-token ones. Consecutive passages are therefore
    concatenated with a blank line between them until they approach the context
    length, and every span offset is shifted by its passage's position in the
    pack. Unmapped types become -100 so the model learns nothing about regions
    the external annotation disagrees with us about.
    """
    import torch
    from phase2_data import align
    from phase2_weaksup import encode_weak
    from phase2_ner import windows

    kept = collections.Counter()
    dropped = collections.Counter()
    data, buf, buf_spans, buf_len = [], [], [], 0
    SEP = "\n\n"

    def flush():
        nonlocal buf, buf_spans, buf_len
        if not buf:
            return
        text = SEP.join(buf)
        ids, spans, _ = align(tok, text)
        if ids:
            tags = encode_weak(spans, buf_spans, label_to_id, a.radius)
            for mlo, mhi in mask_ranges:
                for i, (ts, te) in enumerate(spans):
                    if ts < mhi and te > mlo:
                        tags[i] = -100
            for st, wi, wt, sk in windows(ids, tags, a.ctx, a.ctx // 2):
                if wt is not None and any(x > 0 for x in wt):
                    data.append(dict(ids=wi, tags=wt))
        buf, buf_spans, buf_len = [], [], 0
        mask_ranges.clear()

    mask_ranges = []
    for _, text, anns in recs:
        if not text:
            continue
        base = buf_len
        for st, en, lb in spans_of_record(anns):
            if en <= st or en > len(text):
                continue
            tgt = MAP.get(lb)
            if tgt in label_to_id:
                buf_spans.append((base + st, base + en, tgt))
                kept[tgt] += 1
            else:
                mask_ranges.append((base + st, base + en))
                dropped[lb] += 1
        buf.append(text)
        buf_len += len(text) + len(SEP)
        if buf_len > a.pack_chars:
            flush()
    flush()

    log("")
    log("  MAPPED (supervised):   " + "  ".join(
        "%s=%d" % kv for kv in kept.most_common()))
    log("  MASKED (-100, not O):  " + "  ".join(
        "%s=%d" % kv for kv in dropped.most_common()))
    sup = sum(sum(1 for x in c["tags"] if x >= 0) for c in data)
    tot = sum(len(c["tags"]) for c in data)
    log("  %d windows, %d of %d positions supervised (%.1f%%)"
        % (len(data), sup, tot, 100 * sup / max(tot, 1)))
    out = "%s/results/external_%s.pt" % (P2, a.tag)
    os.makedirs("%s/results" % P2, exist_ok=True)
    torch.save(data, out)
    log("  wrote %s" % out)
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--convert", action="store_true",
                    help="write results/external_<tag>.pt for phase2_weaksup")
    ap.add_argument("--tag", default="inlegalner")
    ap.add_argument("--radius", type=int, default=24)
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--drop-thresh", type=float, default=0.5,
                    help="drop a record if this fraction of its 13-grams "
                         "appears in fold3")
    ap.add_argument("--pack-chars", type=int, default=20000,
                    help="concatenate passages up to this many characters")
    ap.add_argument("--name", default="opennyaiorg/InLegalNER")
    ap.add_argument("--split", default="train")
    ap.add_argument("--files", nargs="*", default=None,
                    help="raw JSON paths, if load_dataset cannot read it")
    ap.add_argument("--parquet", nargs="*", default=None, metavar="REL",
                    help="repo-relative parquet paths, e.g. data/train-*.parquet")
    a = ap.parse_args()

    # --- ours, including the test documents we must protect ---------------
    text, gold = {}, {}
    for fold in ("fold1", "fold2"):
        text.update({str(k): v
                     for k, v in load_json("Train/%s-text.json" % fold).items()})
        gold.update(parse_gold(load_json("Train/%s-gold.json" % fold)))
    test_text = {str(k): v
                 for k, v in load_json("Test/fold3-text.json").items()}
    split_f = json.load(open("%s/split.json" % P2))
    log("  ours: %d train documents, %d test documents"
        % (len(text), len(test_text)))

    recs = []
    if a.parquet:
        recs = load_parquet(a.name, a.parquet)
    elif a.files:
        for f in a.files:
            raw = json.load(open(f, encoding="utf-8"))
            log("  %s -> %s, %d items"
                % (f, type(raw).__name__, len(raw)))
            it = raw.values() if isinstance(raw, dict) else raw
            for i, r in enumerate(it):
                if isinstance(r, str):
                    recs.append((i, r, []))
                else:
                    recs.append((r.get("id", i),
                                 r.get("text") or r.get("data", {}).get("text", ""),
                                 r.get("annotations") or r.get("entities") or []))
    else:
        recs = load_external(a.name, a.split)

    if not recs:
        log("  nothing loaded -- rerun with --files <paths>")
        return
    log("  external: %d records" % len(recs))

    # --- label inventory ---------------------------------------------------
    lab = collections.Counter()
    for _, _, anns in recs:
        for x in anns:
            if isinstance(x, dict):
                v = x.get("label") or x.get("labels") or x.get("type")
                if isinstance(v, list):
                    for y in v:
                        lab[str(y)] += 1
                elif v is not None:
                    lab[str(v)] += 1
            elif isinstance(x, (list, tuple)) and len(x) >= 3:
                lab[str(x[2])] += 1
    log("")
    log("  EXTERNAL LABEL INVENTORY (maps onto our 12?)")
    for k, v in lab.most_common(30):
        log("    %-18s %7d" % (k, v))

    lens = sorted(len(t) for _, t, _ in recs if t)
    if lens:
        log("")
        log("  text length: min %d median %d p95 %d max %d"
            % (lens[0], lens[len(lens) // 2], lens[int(.95 * len(lens))],
               lens[-1]))
        log("  (ours: median 26,640 -- if theirs is ~2,000 these are")
        log("   sentences or preamble blocks, not whole judgments)")

    # --- CONTAMINATION -----------------------------------------------------
    log("")
    log("  " + "=" * 68)
    log("  CONTAMINATION CHECK -- an overlap with fold3 disqualifies the data")
    log("  " + "=" * 68)
    ext_hash = {}
    for rid, t, _ in recs:
        if t:
            ext_hash.setdefault(prefix_hash(t), []).append(rid)
    ext_grams = set()
    for _, t, _ in recs:
        if t:
            ext_grams |= grams(t)
    log("  external: %d distinct prefix hashes, %d sampled 13-grams"
        % (len(ext_hash), len(ext_grams)))

    for name, docs in (("fold1", {d: text[d] for d in split_f["fold1"]}),
                       ("fold2", {d: text[d] for d in split_f["fold2"]}),
                       ("fold3 (TEST)", test_text)):
        hits_h, hits_g, worst = 0, 0, []
        for d, t in docs.items():
            if prefix_hash(t) in ext_hash:
                hits_h += 1
            g = grams(t)
            if g:
                frac = len(g & ext_grams) / len(g)
                if frac > 0.05:
                    hits_g += 1
                worst.append((frac, d))
        worst.sort(reverse=True)
        log("")
        log("  %-14s %2d/%2d exact-prefix, %2d/%2d >5%% 13-gram overlap"
            % (name, hits_h, len(docs), hits_g, len(docs)))
        log("                 worst documents: %s"
            % "  ".join("%s=%.1f%%" % (d, 100 * f) for f, d in worst[:5]))

    if a.convert:
        if any(f > 0.05 for f, _ in
               [(len(grams(t) & ext_grams) / max(len(grams(t)), 1), d)
                for d, t in test_text.items()]):
            log("")
            log("  REFUSING to convert: fold3 documents appear in this data.")
            log("  Training on it would be training on the test set.")
            return
        recs = drop_contaminated(recs, test_text, a.drop_thresh)
        sys.path.insert(0, "%s/code" % P2)
        from cs781_bpe import BPETokenizer
        P1 = os.environ.get("CS781_P1", os.path.dirname(P2) + "/phase1")
        sys.path.insert(0, "%s/shared/lib" % P1)
        with open(fetch("Train/labels.txt"), encoding="utf-8") as f:
            labs = [x.strip() for x in f if x.strip()]
        tok = BPETokenizer.load("%s/shared/tokenizer/bpe_legal_32k.json" % P1)
        convert(recs, tok, labs, {l: i for i, l in enumerate(labs)}, a)

    log("")
    log("  READING THIS: fold3 at >5%% 13-gram overlap means those documents")
    log("  are in the external data and must be removed before training, or")
    log("  the leaderboard score is measuring memorisation. fold1/fold2")
    log("  overlap is harmless -- we already have those labels.")


if __name__ == "__main__":
    main()
