#!/usr/bin/env python3
"""CS-781 Phase 2 -- show me the data. CPU ONLY, ~1 minute.

    python phase2_inspect.py

Everything so far has been designed from aggregate statistics -- transfer
rates, repetition rates, boundary deltas. That was enough to find that the
deficit is span boundaries and that per-document string matching scores 0.9863
if the strings are right. It is not enough to WRITE the thing that gets the
strings right, because four of the twelve labels are defined by a structured
caption block and three more are format-regular citations, and neither can be
handled without looking at the actual surface forms.

This prints, per label:

  * the most frequent gold strings, which show the recurring forms
  * sampled gold spans in context, which show the delimiters a rule must cut on
  * character-class skeletons (A=letters, 9=digits), which show how regular a
    label really is -- CASENO at overlap 0.706 and strict 0.249 is a boundary
    problem, and a skeleton that collapses to a handful of shapes means a
    pattern can fix it outright

and for the caption labels, the head of a few documents with the gold spans
that fall inside it, because APP/RESP/A.COUNSEL/R.COUNSEL all derive from that
one block and propagation carries it through the rest of the document.

Output is deliberately compact: this gets pasted back, not read on Colab.
"""

import collections
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase2_data import P2, fetch, load_json, log, parse_gold  # noqa: E402

CAPTION = ("APP", "RESP", "A.COUNSEL", "R.COUNSEL")
REGULAR = ("CASENO", "PREC", "DATE", "STAT", "AUTH")


def skeleton(s):
    """'AIR 1976 SC 1207' -> 'AAA 9999 AA 9999'. Collapses surface detail."""
    out = []
    for c in s:
        out.append("9" if c.isdigit() else "A" if c.isalpha() else c)
    return "".join(out)


def main():
    random.seed(0)
    with open(fetch("Train/labels.txt"), encoding="utf-8") as f:
        labels = [x.strip() for x in f if x.strip()]
    text, gold = {}, {}
    for fold in ("fold1", "fold2"):
        text.update({str(k): v
                     for k, v in load_json("Train/%s-text.json" % fold).items()})
        gold.update(parse_gold(load_json("Train/%s-gold.json" % fold)))
    split = json.load(open("%s/split.json" % P2))
    docs = split["fold1"] + split["fold2"]

    by = collections.defaultdict(list)
    for d in docs:
        for s, e, l in gold[d]:
            by[l].append((d, s, e))

    for l in labels:
        items = by[l]
        strings = collections.Counter(text[d][s:e] for d, s, e in items)
        log("")
        log("=" * 74)
        log("  %s   %d spans, %d distinct, median length %d"
            % (l, len(items), len(strings),
               sorted(e - s for _, s, e in items)[len(items) // 2]))
        log("=" * 74)

        log("  most frequent forms:")
        for st, n in strings.most_common(10):
            log("    %4d  %r" % (n, st[:70]))

        if l in REGULAR:
            sk = collections.Counter(skeleton(text[d][s:e]) for d, s, e in items)
            log("  skeletons (A=letter 9=digit) -- %d distinct over %d spans:"
                % (len(sk), len(items)))
            for st, n in sk.most_common(12):
                log("    %4d  %s" % (n, st[:70]))

        log("  in context (| marks the gold boundaries):")
        for d, s, e in random.sample(items, min(6, len(items))):
            t = text[d]
            log("    ...%s|%s|%s..."
                % (t[max(0, s - 60):s].replace("\n", "\\n"),
                   t[s:e].replace("\n", "\\n"),
                   t[e:e + 40].replace("\n", "\\n")))

    # The caption block, which defines four of the twelve labels.
    log("")
    log("#" * 74)
    log("  DOCUMENT HEADS -- the block APP/RESP/A.COUNSEL/R.COUNSEL come from")
    log("#" * 74)
    for d in docs[:3]:
        t = text[d]
        log("")
        log("  --- %s, first 1100 characters ---" % d)
        log("  " + t[:1100].replace("\n", "\n  "))
        log("  --- gold spans inside the first 2500 characters ---")
        for s, e, l in sorted(x for x in gold[d] if x[0] < 2500):
            log("    %6d %-11s %r" % (s, l, t[s:e][:60]))

    # Where in the document each label actually lives. If the caption labels
    # are genuinely caption-defined the mass sits near 0; §2.1 measured APP at
    # median 0.153 but p90 0.890, which refuted a caption-only reading once
    # already and needs to be checked per label before writing a parser.
    log("")
    log("#" * 74)
    log("  RELATIVE POSITION IN DOCUMENT (0 = start, 1 = end)")
    log("#" * 74)
    log("  %-11s %6s %6s %6s %6s %6s" % ("label", "p10", "p50", "p90", "first", "n"))
    for l in labels:
        pos = sorted((s / max(len(text[d]), 1)) for d, s, e in by[l])
        if not pos:
            continue
        # how often the FIRST occurrence of the label is in the caption
        firsts = []
        for d in docs:
            ss = [s for s, e, lb in gold[d] if lb == l]
            if ss:
                firsts.append(min(ss) / max(len(text[d]), 1))
        log("  %-11s %6.3f %6.3f %6.3f %6.3f %6d"
            % (l, pos[len(pos) // 10], pos[len(pos) // 2],
               pos[9 * len(pos) // 10],
               sorted(firsts)[len(firsts) // 2] if firsts else -1, len(pos)))


if __name__ == "__main__":
    main()
