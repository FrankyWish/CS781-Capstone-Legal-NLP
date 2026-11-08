#!/usr/bin/env python3
"""CS-781 Phase 2 (L-NER) -- data, character-offset alignment, and its self-test.

    python phase2_data.py                 # inspect, align, self-test, report
    python phase2_data.py --overlap       # also check fold3 against fold1/2

RUN THIS BEFORE ANY TRAINING. It is the gate.

Phase 2 is scored on STRICT character spans: a prediction with the right entity
and a one-character boundary error is counted wrong. Every prediction we make
travels characters -> tokens -> BIO labels -> tokens -> characters, and if any
leg of that is off by one, all six experiments score near zero and we lose a day
finding out. So this file does the alignment and then proves it, by encoding the
gold spans to BIO and decoding them straight back.

TWO NUMBERS THIS PRODUCES THAT NOTHING ELSE WILL

1. THE ALIGNMENT CEILING. A gold span whose boundary falls inside a token can
   never be recovered exactly by a token-level model -- the span we decode will
   snap outward to the token edge and strict scoring will reject it. The
   round-trip measures what fraction of gold spans survive. That is a hard cap
   on recall for ANY token-classification approach, known before training rather
   than inferred afterwards from a disappointing score, and it is the number
   that decides whether to use a span-extraction head instead.

2. PER-LABEL FREQUENCY. The metric is strict MACRO-F1 over 12 labels, so every
   label is worth 1/12 regardless of how rare it is, and a label we never predict
   contributes a flat zero. The rare labels decide the ranking, not the common
   ones.

WHY THE ALIGNMENT IS NOT A ONE-LINER

The tokenizer is byte-level BPE with no offset support. Offsets are rebuilt from
two facts: `LEGAL_PAT.finditer` gives each chunk's character span, and inside a
chunk every BPE token is a string of GPT-2 byte-proxy characters, so its length
IS its byte length. Walking those byte lengths against the chunk's UTF-8
encoding gives each token a character span.

One trap worth stating: `LEGAL_PAT` matches letters, digit runs, punctuation and
whitespace, but `_` is in none of those classes -- `\\w` includes it so the
punctuation branch excludes it, and the letter branch excludes it too. So the
pattern does NOT tile the text, `findall` silently drops those characters, and
anything assuming contiguity drifts. `finditer` is used here and the gaps are
counted and reported.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys

# Drive layout is one folder per phase. Phase 2 READS Phase 1's tokenizer and
# shared library -- the tokenizer is irreproducible without a retrain, and the
# whole point of the capstone is that each phase builds on the Phase-1
# foundation -- and WRITES only inside its own folder.
BASE = os.environ.get("CS781_BASE", "/content/drive/MyDrive/cs781_capstone")
P1 = os.environ.get("CS781_P1", f"{BASE}/phase1")       # read-only from here
P2 = os.environ.get("CS781_P2", f"{BASE}/phase2")       # everything we write
sys.path.insert(0, f"{P1}/shared/lib")
from cs781_bpe import BPETokenizer, LEGAL_PAT          # noqa: E402

REPO = "Exploration-Lab/CS-781-Capstone"


def log(*a):
    print(*a, flush=True)


# --------------------------------------------------------------------- load --
def fetch(name):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(REPO, f"Phase_2_LNER/{name}", repo_type="dataset")


def load_json(name):
    with open(fetch(name), encoding="utf-8") as f:
        return json.load(f)


def parse_gold(raw):
    """{doc_id: [(start, end, label), ...]} from whichever shape the file uses.

    The spec says each annotation carries start, end, label and entity_text but
    not how they are packed, so the plausible encodings are all accepted and the
    one actually found is printed.
    """
    out, shape = {}, None
    for doc, anns in raw.items():
        if isinstance(anns, dict):
            anns = (anns.get("entities") or anns.get("annotations")
                    or anns.get("spans") or [])
        spans = []
        for a in anns:
            if isinstance(a, (list, tuple)) and len(a) >= 3:
                shape = shape or "list [start, end, label, ...]"
                spans.append((int(a[0]), int(a[1]), str(a[2])))
            elif isinstance(a, dict):
                shape = shape or f"dict {sorted(a)[:5]}"
                s = a.get("start", a.get("start_char", a.get("begin")))
                e = a.get("end", a.get("end_char", a.get("stop")))
                lb = a.get("label", a.get("entity", a.get("type", a.get("tag"))))
                if s is None or e is None or lb is None:
                    continue
                spans.append((int(s), int(e), str(lb)))
        out[str(doc)] = spans
    log(f"  gold annotation shape: {shape}")
    return out


# ---------------------------------------------------------------- alignment --
def align(tok, text):
    """text -> (ids, char_spans, n_dropped_chars).

    char_spans[i] is the (start, end) character range that token i covers.
    """
    ids, spans, covered = [], [], 0
    for m in LEGAL_PAT.finditer(text):
        chunk, c0 = m.group(), m.start()
        covered += len(chunk)
        # Byte offset -> enclosing character index. Not a boundary lookup: a
        # byte-level BPE token can start or end INSIDE a multi-byte character
        # (routine for Devanagari in this corpus), and an earlier version
        # dropped those tokens, which silently lost text and broke the span
        # arithmetic. Every byte maps to the character that contains it.
        floor = []
        for ci, ch in enumerate(chunk):
            floor.extend([ci] * len(ch.encode("utf-8")))
        floor.append(len(chunk))
        bpos = 0
        for sym in tok._bpe_chunk(chunk):
            tid = tok.stoi.get(sym)
            nbytes = len(sym)                  # one proxy char == one byte
            s, e = bpos, bpos + nbytes
            bpos = e
            if tid is None or e == s:
                continue
            cs = floor[s]
            ce = floor[min(e - 1, len(floor) - 1)] + 1
            ids.append(tid)
            spans.append((c0 + cs, c0 + min(ce, len(chunk))))
    return ids, spans, len(text) - covered


def encode_bio(spans, gold, label_to_id):
    """Token-level BIO ids. 0 = O, then 2k+1 = B-label_k, 2k+2 = I-label_k."""
    tags = [0] * len(spans)
    for s, e, lb in gold:
        k = label_to_id.get(lb)
        if k is None:
            continue
        inside = [i for i, (ts, te) in enumerate(spans) if ts < e and te > s]
        for j, i in enumerate(inside):
            tags[i] = (2 * k + 1) if j == 0 else (2 * k + 2)
    return tags


def trim(text, s, e):
    """Pull a decoded span in off surrounding whitespace.

    LEGAL_PAT's word branch attaches the space BEFORE a word, so a token carries that space
    BEFORE its word. An entity starting at a word boundary therefore decodes one
    character early -- (7, 29) comes back as (6, 29) -- and strict scoring
    rejects that outright. Gold entities do not begin or end on whitespace, so
    trimming recovers the exact boundary. This single line is worth more than
    any modelling decision in the phase.
    """
    while s < e and text[s].isspace():
        s += 1
    while e > s and text[e - 1].isspace():
        e -= 1
    return s, e


def decode_bio(tags, spans, labels, text=None, merge_gap=0):
    """BIO ids + token char spans -> [(start, end, label)], whitespace-trimmed."""
    out, cur = [], None
    for i, t in enumerate(tags):
        if t == 0:
            if cur:
                out.append(cur)
                cur = None
            continue
        k, is_b = (t - 1) // 2, (t % 2 == 1)
        if is_b or cur is None or cur[2] != labels[k]:
            if cur:
                out.append(cur)
            cur = [spans[i][0], spans[i][1], labels[k]]
        else:
            cur[1] = spans[i][1]
    if cur:
        out.append(cur)
    if text is None:
        return [tuple(c) for c in out]
    res, seen = [], set()
    for st, en, lb in out:
        st, en = trim(text, st, en)
        # Two different token runs can trim onto the SAME character span, and
        # evaluate_phase2.py rejects the entire submission on one duplicate --
        # not the entity, the upload. Dedupe here, once, for every caller.
        if en > st and (st, en, lb) not in seen:
            seen.add((st, en, lb))
            res.append((st, en, lb))
    return merge_fragments(res, text, merge_gap) if merge_gap else res


def n_tags_for(labels, scheme):
    return (4 if scheme == "bioes" else 2) * len(labels) + 1


def encode_tags(spans, gold, label_to_id, scheme="bio"):
    """Token-level tag ids under BIO or BIOES.

    BIO gives the model no way to say "this token ENDS an entity" -- the last
    token is only identified by the absence of a continuation on the next one.
    Measured on this data, that is where the whole phase is being lost: the
    detector reaches 0.58 overlap-F1 and 0.29 strict, so nearly every entity is
    found and almost none is bounded correctly. BIOES adds explicit End and
    Single tags so the boundary is a thing the model predicts directly rather
    than something inferred from what comes after.
    """
    if scheme == "bio":
        return encode_bio(spans, gold, label_to_id)
    tags = [0] * len(spans)
    for s, e, lb in gold:
        k = label_to_id.get(lb)
        if k is None:
            continue
        inside = [i for i, (ts, te) in enumerate(spans) if ts < e and te > s]
        if not inside:
            continue
        if len(inside) == 1:
            tags[inside[0]] = 4 * k + 4                 # S
        else:
            tags[inside[0]] = 4 * k + 1                 # B
            for i in inside[1:-1]:
                tags[i] = 4 * k + 2                     # I
            tags[inside[-1]] = 4 * k + 3                # E
    return tags


def merge_fragments(spans, text, max_gap=3):
    """Join adjacent same-label spans separated only by punctuation or space.

    The boundary diagnostic showed the model tags SOME subword pieces of a
    multi-token entity and not others -- 'MUKHERJEA, B.K.' comes back as 'EA',
    'KRISHNAIYER, V.R.' as 'KRISHNAIYER,'. ALL-CAPS party names fragment into
    many rare pieces under a tokenizer trained on mixed-case legal prose, so the
    model sees them as a string of unfamiliar tokens and commits to only some.

    Where two predictions of the same label are separated by nothing but a comma,
    a full stop or a space, they are almost always one entity that was split.
    Merging them costs nothing when the model was right and recovers the whole
    span when it fragmented. `max_gap` is deliberately small: a larger window
    starts swallowing the gap between two genuinely distinct judges.
    """
    out = []
    for sp in sorted(spans, key=lambda x: (x[2], x[0])):
        s0, e0, lb = sp
        if out and out[-1][2] == lb and 0 <= s0 - out[-1][1] <= max_gap                 and not text[out[-1][1]:s0].strip(" ,.;:-()[]"):
            out[-1] = (out[-1][0], max(out[-1][1], e0), lb)
        else:
            out.append((s0, e0, lb))
    return sorted(set(out))


def decode_tags(tags, spans, labels, text=None, scheme="bio", merge_gap=0):
    """Tag ids -> [(start, end, label)], whitespace-trimmed."""
    if scheme == "bio":
        return decode_bio(tags, spans, labels, text, merge_gap)
    out, cur = [], None
    for i, t in enumerate(tags):
        if t == 0:
            if cur:
                out.append(cur)
                cur = None
            continue
        k, r = (t - 1) // 4, (t - 1) % 4                # 0=B 1=I 2=E 3=S
        lab = labels[k]
        if r == 3:
            if cur:
                out.append(cur)
                cur = None
            out.append([spans[i][0], spans[i][1], lab])
        elif r == 0:
            if cur:
                out.append(cur)
            cur = [spans[i][0], spans[i][1], lab]
        elif r == 1:
            if cur and cur[2] == lab:
                cur[1] = spans[i][1]
            else:
                if cur:
                    out.append(cur)
                cur = [spans[i][0], spans[i][1], lab]
        else:
            if cur and cur[2] == lab:
                cur[1] = spans[i][1]
                out.append(cur)
                cur = None
            else:
                if cur:
                    out.append(cur)
                out.append([spans[i][0], spans[i][1], lab])
    if cur:
        out.append(cur)
    if text is None:
        return [tuple(c) for c in out]
    res, seen = [], set()
    for st, en, lb in out:
        st, en = trim(text, st, en)
        # Two different token runs can trim onto the SAME character span, and
        # evaluate_phase2.py rejects the entire submission on one duplicate --
        # not the entity, the upload. Dedupe here, once, for every caller.
        if en > st and (st, en, lb) not in seen:
            seen.add((st, en, lb))
            res.append((st, en, lb))
    return merge_fragments(res, text, merge_gap) if merge_gap else res


# -------------------------------------------------------------------- main --
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overlap", action="store_true",
                    help="also check whether fold3 documents appear in fold1/2")
    ap.add_argument("--max-docs", type=int, default=0, help="0 = all")
    a = ap.parse_args()

    tok = BPETokenizer.load(f"{P1}/shared/tokenizer/bpe_legal_32k.json")

    with open(fetch("Train/labels.txt"), encoding="utf-8") as f:
        labels = [x.strip() for x in f if x.strip()]
    log(f"labels ({len(labels)}): {labels}")
    label_to_id = {l: i for i, l in enumerate(labels)}

    text, gold, fold_of = {}, {}, {}
    for fold in ("fold1", "fold2"):
        t = load_json(f"Train/{fold}-text.json")
        g = parse_gold(load_json(f"Train/{fold}-gold.json"))
        text.update({str(k): v for k, v in t.items()})
        gold.update(g)
        for k in t:
            fold_of[str(k)] = fold
        log(f"{fold}: {len(t)} documents, {sum(len(v) for v in g.values())} entities")
    test_text = load_json("Test/fold3-text.json")
    log(f"fold3 (test): {len(test_text)} documents")

    docs = sorted(text)
    if a.max_docs:
        docs = docs[:a.max_docs]

    lens = sorted(len(text[d]) for d in docs)
    log(f"\ndocument characters: min {lens[0]:,} median {lens[len(lens)//2]:,} "
        f"p95 {lens[int(len(lens)*0.95)]:,} max {lens[-1]:,}")

    freq = collections.Counter(lb for d in docs for _, _, lb in gold.get(d, []))
    log(f"\nper-label frequency (macro-F1 weights every one of these equally):")
    for lb in labels:
        n = freq.get(lb, 0)
        bar = "#" * min(40, n // max(1, max(freq.values()) // 40 or 1))
        log(f"  {lb:12s} {n:6d}  {bar}")
    missing = [l for l in labels if l not in freq]
    if missing:
        log(f"  !! labels with ZERO training examples: {missing}")
    rare = [l for l, n in sorted(freq.items(), key=lambda x: x[1])[:4]]
    log(f"  rarest four -- these decide the ranking: {rare}")

    # ---- THE GATE: encode gold to BIO, decode back, compare exactly ---------
    log(f"\n--- alignment self-test over {len(docs)} documents ---")
    total, exact, dropped_chars, tok_total = 0, 0, 0, 0
    per_label = collections.Counter()
    per_label_ok = collections.Counter()
    for d in docs:
        ids, spans, drop = align(tok, text[d])
        dropped_chars += drop
        tok_total += len(ids)
        g = gold.get(d, [])
        if not g:
            continue
        back = set(decode_bio(encode_bio(spans, g, label_to_id), spans,
                              labels, text[d]))
        for s, e, lb in g:
            total += 1
            per_label[lb] += 1
            if (s, e, lb) in back:
                exact += 1
                per_label_ok[lb] += 1

    log(f"  tokens produced      {tok_total:,}")
    log(f"  characters dropped   {dropped_chars:,} "
        f"({100*dropped_chars/sum(len(text[d]) for d in docs):.4f}% -- "
        f"LEGAL_PAT does not match '_')")
    log(f"\n  ALIGNMENT CEILING    {exact}/{total} = {100*exact/max(1,total):.2f}% "
        f"of gold spans survive the char->token->char round trip")
    log(f"  worst labels:")
    for lb in sorted(per_label, key=lambda l: per_label_ok[l] / max(1, per_label[l]))[:5]:
        log(f"    {lb:12s} {per_label_ok[lb]:5d}/{per_label[lb]:<5d} "
            f"{100*per_label_ok[lb]/max(1,per_label[lb]):6.2f}%")

    if exact / max(1, total) < 0.97:
        log("\n  -> CEILING IS LOW. Gold boundaries fall inside tokens often "
            "enough that BIO tagging cannot reach them under strict scoring. "
            "A span-extraction head predicting character offsets directly is "
            "the better primary approach, and this number is the justification.")
    else:
        log("\n  -> alignment is sound. BIO token classification can reach "
            f"{100*exact/total:.1f}% recall at best; the rest is modelling.")

    # ---- the split: TWO-FOLD CV on the organisers' own folds ---------------
    # 70 documents is far too few for one held-out split to select models on.
    # The rarest label (R.COUNSEL, 184 instances) would put ~29 examples in a
    # 15% dev set, and macro-F1 weights that label exactly as heavily as DATE
    # with 1575 -- selection noise would swamp every effect we care about.
    #
    # The organisers supplied two folds of 35. Using them as CV folds -- train
    # on fold1 score fold2, train on fold2 score fold1, average -- doubles the
    # evaluation set, costs one extra training run per experiment (minutes at
    # this scale), and needs no arbitrary split point to defend. The final
    # submission trains on all 70.
    split = {"fold1": sorted(d for d in docs if fold_of.get(d) == "fold1"),
             "fold2": sorted(d for d in docs if fold_of.get(d) == "fold2"),
             "scheme": "2-fold CV on the organisers' folds; final model trains on both"}
    os.makedirs(P2, exist_ok=True)
    json.dump(split, open(f"{P2}/split.json", "w"), indent=2)
    log("")
    log(f"split: 2-fold CV, {len(split['fold1'])} + {len(split['fold2'])} documents. "
        f"One 15% dev set would hold ~29 of the rarest label, which macro-F1 "
        f"weights as heavily as the one with 1575.")

    if a.overlap:
        log("\n--- fold3 overlap against fold1/2 ---")
        import hashlib
        seen = {hashlib.sha1(v[:3000].encode()).hexdigest(): k for k, v in text.items()}
        hit = sum(1 for v in test_text.values()
                  if hashlib.sha1(v[:3000].encode()).hexdigest() in seen)
        log(f"  exact first-3000-character duplicates: {hit}/{len(test_text)}")
        log("  (if this is non-zero, gold spans transfer directly for those "
            "documents and that bounds what the modelling is worth)")

    json.dump({"labels": labels, "alignment_ceiling": exact / max(1, total),
               "label_freq": dict(freq), "n_train_docs": len(docs)},
              open(f"{P2}/data_report.json", "w"), indent=2)
    log(f"\nwrote {P2}/split.json and data_report.json")


if __name__ == "__main__":
    main()
