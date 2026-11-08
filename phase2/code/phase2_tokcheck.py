#!/usr/bin/env python3
"""Is the TOKENIZER the reason APP, RESP, JUDGE and WIT fail? CPU, ~2 min.

    python phase2_tokcheck.py

THE HYPOTHESIS

Every boundary example this phase has produced is an ALL-CAPS name shredded
into rare subword pieces:

    'MUKHERJEA, B.K.'      -> 'EA, B.K. BOSE, VIVIAN'
    'KRISHNAIYER, V.R.'    -> 'NAIYER, V.R.'
    'ACHUTANANDA PUROHIT'  -> 'CHUTANANDA PUR'
    'BHAGWATI, NATWARLAL H.' -> 'WARLAL H.'

The Phase-1 BPE was trained on running legal prose, which is mostly mixed
case. ALL-CAPS text shares almost no merges with it, so 'MUKHERJEA' may cost
six rare tokens where 'Mukherjea' costs two common ones. A model with 70
training documents cannot learn to place a B- tag on a token it has seen a
handful of times, which would explain why the labels that live in ALL-CAPS
caption blocks -- APP 0.12, RESP 0.11, WIT 0.06, JUDGE in BENCH lines -- are
exactly the ones that fail, while DATE (digits, case-free) scores 0.90.

If that holds, the fix needs no retraining of Phase 1: case-fold the input
before tokenizing. Character offsets are unchanged by case folding, so the
whole alignment layer still works and gold spans still land where they did.

WHAT THIS MEASURES

  1. tokens per character for gold entity strings, split by case shape
  2. the same string ALL-CAPS vs title-case, head to head
  3. per label: share of gold spans that are ALL-CAPS, and the per-label
     fertility -- set against the per-label F1 the ensemble actually gets
  4. how much of the vocabulary is reachable at all from upper-case text

A null result here is worth as much as a positive one: it closes off the last
cheap explanation and says the detector is simply undertrained.
"""

import collections
import json
import os
import sys

BASE = os.environ.get("CS781_BASE", "/content/drive/MyDrive/cs781_capstone")
P1 = os.environ.get("CS781_P1", "%s/phase1" % BASE)
P2 = os.environ.get("CS781_P2", "%s/phase2" % BASE)
sys.path.insert(0, "%s/shared/lib" % P1)
sys.path.insert(0, "%s/code" % P2)

from phase2_data import fetch, load_json, log, parse_gold  # noqa: E402
from cs781_bpe import BPETokenizer                          # noqa: E402

# Ensemble per-label strict F1, mean of the two folds, from ens6_m4b.
ENS_F1 = {
    "DATE": 0.901, "COURT": 0.745, "STAT": 0.586, "JUDGE": 0.562,
    "CASENO": 0.370, "R.COUNSEL": 0.275, "PREC": 0.259, "A.COUNSEL": 0.253,
    "AUTH": 0.247, "APP": 0.123, "RESP": 0.113, "WIT": 0.056,
}


def shape(s):
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return "no letters"
    if all(c.isupper() for c in letters):
        return "ALL CAPS"
    if s == s.title() or (letters[0].isupper() and not s.isupper()):
        return "Mixed case"
    return "lower case"


def main():
    tok = BPETokenizer.load("%s/shared/tokenizer/bpe_legal_32k.json" % P1)
    with open(fetch("Train/labels.txt"), encoding="utf-8") as f:
        labels = [x.strip() for x in f if x.strip()]
    text, gold = {}, {}
    for fold in ("fold1", "fold2"):
        text.update({str(k): v
                     for k, v in load_json("Train/%s-text.json" % fold).items()})
        gold.update(parse_gold(load_json("Train/%s-gold.json" % fold)))
    split = json.load(open("%s/split.json" % P2))
    docs = split["fold1"] + split["fold2"]

    def ntok(s):
        return len(list(tok._bpe_chunk(s))) if s else 0

    # 1. fertility by case shape
    by_shape = collections.defaultdict(lambda: [0, 0, 0])   # toks, chars, n
    per_label = collections.defaultdict(lambda: [0, 0, 0, 0])  # toks, chars, n, caps
    pairs = []
    for d in docs:
        t = text[d]
        for a, b, l in gold[d]:
            st = t[a:b]
            n = ntok(st)
            sh = shape(st)
            by_shape[sh][0] += n
            by_shape[sh][1] += len(st)
            by_shape[sh][2] += 1
            per_label[l][0] += n
            per_label[l][1] += len(st)
            per_label[l][2] += 1
            if sh == "ALL CAPS":
                per_label[l][3] += 1
                pairs.append(st)

    log("")
    log("  1. TOKENS PER CHARACTER by case shape (lower is better: the")
    log("     tokenizer recognises the string instead of spelling it out)")
    log("     %-12s %8s %9s %9s" % ("shape", "spans", "tok/char", "tok/span"))
    for sh in sorted(by_shape, key=lambda k: -by_shape[k][2]):
        tk, ch, n = by_shape[sh]
        log("     %-12s %8d %9.3f %9.2f" % (sh, n, tk / max(ch, 1), tk / max(n, 1)))

    # 2. the same text, both ways
    log("")
    log("  2. THE SAME STRING, ALL CAPS vs title case")
    tot_u = tot_t = 0
    worst = []
    for st in pairs:
        u, ti = ntok(st), ntok(st.title())
        tot_u += u
        tot_t += ti
        if u - ti > 0:
            worst.append((u - ti, st, u, ti))
    if pairs:
        log("     %d ALL-CAPS gold spans: %d tokens as written, %d title-cased"
            % (len(pairs), tot_u, tot_t))
        log("     ratio %.2fx -- every extra token is one more place for a"
            % (tot_u / max(tot_t, 1)))
        log("     B- or I- tag to land in the wrong spot")
        worst.sort(reverse=True)
        log("     biggest differences:")
        for dlt, st, u, ti in worst[:10]:
            log("       %2d -> %2d  (%+d)  %r" % (u, ti, -dlt, st[:44]))

    # 3. per label, against what the ensemble actually scores
    log("")
    log("  3. PER LABEL: case shape, fertility, and the F1 the ensemble gets")
    log("     %-11s %6s %8s %9s %8s" % ("label", "spans", "%ALLCAPS",
                                        "tok/char", "ens F1"))
    rows = []
    for l in labels:
        tk, ch, n, caps = per_label[l]
        if not n:
            continue
        rows.append((ENS_F1.get(l, 0.0), l, n, 100 * caps / n, tk / max(ch, 1)))
    for f1, l, n, pc, fert in sorted(rows, reverse=True):
        log("     %-11s %6d %7.1f%% %9.3f %8.3f" % (l, n, pc, fert, f1))

    # Does fertility actually track the failures, or is this a dead end?
    if len(rows) > 2:
        xs = [r[4] for r in rows]
        ys = [r[0] for r in rows]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        vx = sum((x - mx) ** 2 for x in xs) ** 0.5
        vy = sum((y - my) ** 2 for y in ys) ** 0.5
        r = cov / (vx * vy) if vx and vy else 0.0
        log("")
        log("     correlation(tokens per character, ensemble F1) = %+.3f" % r)
        log("     strongly negative => the tokenizer is the constraint and")
        log("     case folding is worth a GPU run; near zero => it is not,")
        log("     and the detector is simply undertrained on 70 documents.")

    # 4. is upper case even representable?
    vocab_up = sum(1 for s in tok.stoi if any(c.isupper() for c in s))
    log("")
    log("  4. %d of %d vocabulary entries contain an upper-case character "
        "(%.1f%%)" % (vocab_up, len(tok.stoi), 100 * vocab_up / len(tok.stoi)))


if __name__ == "__main__":
    main()
