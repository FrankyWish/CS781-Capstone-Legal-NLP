#!/usr/bin/env python3
"""CS-781 Phase 2 -- gazetteer audit and boundary snapping. CPU ONLY.

    python phase2_gazetteer.py --audit             # measure the lever, ~2 min
    python phase2_gazetteer.py --snap ens6_m4      # apply it to a submission

WHY THIS EXISTS

The ensemble scores 0.4608 strict on test and roughly 0.77 overlap. Every
entity is being FOUND and then mis-delimited. Eight modelling interventions
(BIOES, CRF, fragment merging, class reweighting, caption prepending, lower LR,
LoRA, LoRMA) failed to move the strict/overlap ratio at all, which says the
boundary information is not present in the representation and no decoder on top
of it can recover it.

A gazetteer does not need it to be. Supreme Court judgments reuse the same
courts, the same statutes, the same precedents and a small pool of judges. A
string annotated once in training carries its EXACT character extent to every
later occurrence. That converts "detected but mis-bounded" -- the whole deficit
-- into "exact", without touching the model.

Three numbers decide whether to spend any more GPU on this phase:

  1. TRANSFER RATE. Of the gold spans in the held-out fold, what fraction have
     their exact string annotated with the same label in the training fold?
     That is the hard ceiling on the recall this lever can contribute.
  2. GAZETTEER-ONLY F1. String matching with no model at all, scored by the
     official strict rule. Directly comparable to the model's 0.375 CV.
  3. PRECISION. A gazetteer that fires everywhere destroys macro-F1 through
     false positives on the rare labels. Every string is therefore filtered by
     its measured ANNOTATION RATE in training -- of all the places this string
     occurs in the training text, how often was it actually annotated? A string
     that appears 40 times and is annotated twice is a trap, not an entity.

The filter in (3) is the whole difference between a useful gazetteer and noise,
and it is estimated from data rather than guessed.
"""

import argparse
import collections
import json
import os
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase2_data import P2, fetch, load_json, log, parse_gold  # noqa: E402


# ------------------------------------------------------------- string match --
def is_word(c):
    return c.isalnum() or c == "_"


def occurrences(text, s):
    """Every occurrence of `s` in `text` that is not inside a larger word.

    Without the boundary guard a gazetteer entry like "Act" matches inside
    "Acting" and every such match is a false positive on a rare label.
    """
    out, i = [], text.find(s)
    while i != -1:
        a, b = i, i + len(s)
        okl = not (is_word(s[0]) and a > 0 and is_word(text[a - 1]))
        okr = not (is_word(s[-1]) and b < len(text) and is_word(text[b]))
        if okl and okr:
            out.append((a, b))
        i = text.find(s, i + 1)
    return out


# --------------------------------------------------------------- gazetteer ---
def build(docs, text, gold, min_len=4, dominance=0.9, min_rate=0.5):
    """{string: (label, annotation_rate, n_seen)} from the training documents."""
    lab = collections.defaultdict(collections.Counter)
    for d in docs:
        t = text[d]
        for s, e, l in gold[d]:
            st = t[s:e].strip()
            # A gold span whose text does not round-trip (leading whitespace in
            # the annotation) would match at the wrong offset. Drop it.
            if len(st) >= min_len and st == t[s:e]:
                lab[st][l] += 1

    gaz, drop = {}, collections.Counter()
    for st, c in lab.items():
        top, n = c.most_common(1)[0]
        seen = sum(c.values())
        if n / seen < dominance:
            drop["ambiguous label"] += 1
            continue
        # count() is the C-level approximation of occurrences(); the boundary
        # guard only ever removes matches, so this under-estimates the rate and
        # the filter stays conservative.
        occ = sum(text[d].count(st) for d in docs)
        if occ == 0:
            drop["unmatchable"] += 1
            continue
        rate = seen / occ
        if rate < min_rate:
            drop["low annotation rate"] += 1
            continue
        gaz[st] = (top, rate, seen)
    return gaz, drop


def tag(text, gaz):
    """Every gazetteer match in one document, longest-first, non-overlapping."""
    hits = []
    for st, (l, rate, _) in gaz.items():
        for a, b in occurrences(text, st):
            hits.append((b - a, a, b, l, rate))
    hits.sort(key=lambda h: (-h[0], -h[4], h[1]))
    out, taken = [], []
    for _, a, b, l, _r in hits:
        if any(a < y and b > x for x, y in taken):
            continue
        taken.append((a, b))
        out.append((a, b, l))
    return out


def snap(pred, gaz_spans, add_missing):
    """Replace mis-bounded model spans with the gazetteer's exact extent.

    A gazetteer match that overlaps a model prediction of the same label is
    treated as the corrected boundary for it. A match the model did not find at
    all is only added when `add_missing` -- that trades precision for recall and
    is measured separately.
    """
    out = list(pred)
    for (a, b, l) in gaz_spans:
        hit = [i for i, p in enumerate(out)
               if p is not None and p[2] == l and p[0] < b and p[1] > a]
        if hit:
            for i in hit:
                out[i] = None
            out.append((a, b, l))
        elif add_missing:
            out.append((a, b, l))
    seen, fin = set(), []
    for p in out:
        if p is None:
            continue
        k = (int(p[0]), int(p[1]), str(p[2]))
        if k in seen:          # the official scorer rejects the whole upload
            continue           # on one duplicate span
        seen.add(k)
        fin.append(k)
    return fin


# ------------------------------------------------------------------ scoring --
def score(pred, truth, labels):
    """Official strict macro-F1, plus the overlap diagnostic beside it."""
    st = {l: [0, 0, 0] for l in labels}
    rel = {l: [0, 0, 0, 0] for l in labels}
    for d in truth:
        g = set(map(tuple, truth[d]))
        p = set(tuple(x) for x in pred.get(d, []))
        for l in labels:
            gs = {x for x in g if x[2] == l}
            ps = {x for x in p if x[2] == l}
            st[l][0] += len(gs & ps)
            st[l][1] += len(ps - gs)
            st[l][2] += len(gs - ps)
            rel[l][1] += len(ps)
            rel[l][3] += len(gs)
            rel[l][0] += sum(1 for a in ps
                             if any(a[0] < b[1] and a[1] > b[0] for b in gs))
            rel[l][2] += sum(1 for b in gs
                             if any(a[0] < b[1] and a[1] > b[0] for a in ps))
    per = {}
    for l in labels:
        tp, fp, fn = st[l]
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
        rp = rel[l][0] / rel[l][1] if rel[l][1] else 0.0
        rr = rel[l][2] / rel[l][3] if rel[l][3] else 0.0
        rf = 2 * rp * rr / (rp + rr) if rp + rr else 0.0
        per[l] = (f, pr, rc, rf, st[l][0] + st[l][2])
    macro = sum(v[0] for v in per.values()) / len(labels)
    overlap = sum(v[3] for v in per.values()) / len(labels)
    return macro, overlap, per


# -------------------------------------------------------------------- audit --
def audit(text, gold, split, labels, args):
    for tr, ev in (("fold1", "fold2"), ("fold2", "fold1")):
        log("")
        log("=" * 72)
        log("  gazetteer from %s  ->  scored on %s" % (tr, ev))
        log("=" * 72)
        gaz, drop = build(split[tr], text, gold,
                          min_len=args.min_len, dominance=args.dominance,
                          min_rate=args.min_rate)
        log("  entries kept %d   dropped %s"
            % (len(gaz), dict(drop) or "{}"))

        # 1. transfer rate -- the ceiling
        hit = collections.Counter()
        tot = collections.Counter()
        for d in split[ev]:
            t = text[d]
            for s, e, l in gold[d]:
                tot[l] += 1
                if gaz.get(t[s:e], (None,))[0] == l:
                    hit[l] += 1
        n_hit, n_tot = sum(hit.values()), sum(tot.values())
        log("")
        log("  1. TRANSFER RATE  %d of %d gold spans (%.1f%%) have their exact"
            % (n_hit, n_tot, 100 * n_hit / max(n_tot, 1)))
        log("     string annotated with the same label in %s" % tr)
        for l in sorted(labels, key=lambda l: -tot[l]):
            log("       %-11s %4d of %4d  %5.1f%%"
                % (l, hit[l], tot[l], 100 * hit[l] / max(tot[l], 1)))

        # 2. gazetteer alone, official rule
        pred = {d: tag(text[d], gaz) for d in split[ev]}
        truth = {d: gold[d] for d in split[ev]}
        macro, overlap, per = score(pred, truth, labels)
        log("")
        log("  2. GAZETTEER-ONLY strict macro-F1 %.4f   overlap %.4f"
            % (macro, overlap))
        log("     (model full fine-tune on the same split: 0.2852 strict)")
        log("     %-11s %6s %6s %6s %6s" % ("label", "F1", "P", "R", "gold"))
        for l in sorted(labels, key=lambda l: -per[l][0]):
            f, pr, rc, rf, ng = per[l]
            log("     %-11s %6.3f %6.3f %6.3f %6d" % (l, f, pr, rc, ng))
        npred = sum(len(v) for v in pred.values())
        log("     emitted %d spans against %d gold" % (npred, n_tot))

    # Is the operation word_snap() relies on actually safe? A gold span that
    # starts or ends mid-word would be widened past its true edge.
    mis = collections.Counter()
    tot = collections.Counter()
    for d in split["fold1"] + split["fold2"]:
        t = text[d]
        for a, b, l in gold[d]:
            tot[l] += 1
            if (a > 0 and is_word(t[a - 1]) and is_word(t[a])) or                (b < len(t) and is_word(t[b]) and is_word(t[b - 1])):
                mis[l] += 1
    log("")
    log("  GOLD SPANS THAT START OR END MID-WORD (word_snap would widen these")
    log("  past their true edge -- the cost of the rule):")
    for l in sorted(labels, key=lambda x: -tot[x]):
        log("    %-11s %4d of %4d  %5.2f%%"
            % (l, mis[l], tot[l], 100 * mis[l] / max(tot[l], 1)))
    log("    %-11s %4d of %4d  %5.2f%%"
        % ("ALL", sum(mis.values()), sum(tot.values()),
           100 * sum(mis.values()) / max(sum(tot.values()), 1)))

    # the gazetteer that would actually ship
    log("")
    log("=" * 72)
    gaz, drop = build(split["fold1"] + split["fold2"], text, gold,
                      min_len=args.min_len, dominance=args.dominance,
                      min_rate=args.min_rate)
    log("  SHIPPING gazetteer from all 70 documents: %d entries" % len(gaz))
    by = collections.Counter(v[0] for v in gaz.values())
    log("  " + "  ".join("%s=%d" % kv for kv in by.most_common()))
    json.dump({k: v[0] for k, v in gaz.items()},
              open("%s/gazetteer.json" % P2, "w", encoding="utf-8"),
              ensure_ascii=False, indent=0)
    log("  wrote %s/gazetteer.json" % P2)


# --------------------------------------------------------------------- snap --
def do_snap(text, gold, split, labels, args):
    gaz, _ = build(split["fold1"] + split["fold2"], text, gold,
                   min_len=args.min_len, dominance=args.dominance,
                   min_rate=args.min_rate)
    log("  gazetteer %d entries" % len(gaz))

    src = "%s/submissions/prediction_%s.json" % (P2, args.snap)
    pred = json.load(open(src, encoding="utf-8"))
    log("  loaded %s  (%d documents, %d spans)"
        % (src, len(pred), sum(len(v) for v in pred.values())))

    test_text = {str(k): v for k, v in load_json("Test/fold3-text.json").items()}
    out, n_snap, n_add = {}, 0, 0
    for d, spans in pred.items():
        t = test_text[d]
        gs = tag(t, gaz)
        before = set((int(a), int(b), str(l)) for a, b, l in spans)
        after = snap([(int(a), int(b), str(l)) for a, b, l in spans],
                     gs, args.add_missing)
        out[d] = [list(x) for x in after]
        n_snap += len(before - set(after))
        n_add += len(set(after) - before)
    log("  %d spans replaced, %d net new, %d -> %d total"
        % (n_snap, n_add, sum(len(v) for v in pred.values()),
           sum(len(v) for v in out.values())))

    tag_out = "%s_snap%s" % (args.snap, "_add" if args.add_missing else "")
    pj = "%s/submissions/prediction_%s.json" % (P2, tag_out)
    json.dump(out, open(pj, "w", encoding="utf-8"), ensure_ascii=False)
    zp = "%s/submissions/submission_%s.zip" % (P2, tag_out)
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(pj, "prediction.json")
    log("  wrote %s" % zp)


# ------------------------------------------------- within-document reuse --
def word_snap(spans, t, skip=()):
    """Push each span outward to the enclosing word.

    A gold span always sits on a word boundary -- the data audit measures this
    directly. A PREDICTED span need not: BPE tokens are sub-word, so a model
    that stops one token early stops mid-word, which is how
    'ACHUTANANDA PUROHIT' is returned as 'CHUTANANDA PUR' and '26/03/1976' as
    '6/03/1976'. Both edges of both examples are mid-word, and expanding to the
    enclosing alphanumeric run recovers the gold span exactly. The operation
    can only widen, so it is safe to the extent that gold spans really are
    word-aligned -- which is why `--audit` now reports that rate instead of
    assuming it.
    """
    out, seen = [], set()
    for a, b, l in spans:
        a, b = int(a), int(b)
        if l in skip:
            # Measured per label: PREC gold spans end mid-token 6.76% of the
            # time against <=1.7% for every other label, because a citation
            # legitimately stops inside one. Widening those costs more than it
            # recovers, so the rule is withheld from exactly that label.
            k = (a, b, str(l))
            if k not in seen:
                seen.add(k)
                out.append(k)
            continue
        while a > 0 and is_word(t[a - 1]) and is_word(t[a]):
            a -= 1
        while b < len(t) and is_word(t[b]) and is_word(t[b - 1]):
            b += 1
        k = (a, b, str(l))
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def self_consistent(pred, text, min_len=5, min_support=1, max_expand=48,
                    propagate=0, wordsnap=False, skip=()):
    """Propagate each document's longest predicted form of an entity.

    The cross-document gazetteer only reaches labels whose strings recur
    between judgments -- COURT, STAT, AUTH, which are already the best-scoring
    labels. The broken ones (PREC, CASENO, APP, RESP, WIT, both COUNSELs) are
    document-specific, so no training data carries their extents. They do
    repeat heavily INSIDE a document, though: a party named once in full is
    named twenty more times, and the boundary report shows the model getting
    the full form somewhere and clipping it elsewhere.

    Frequency is the wrong weight here and the first version of this got it
    backwards: the clipped form is the COMMON prediction and the full form the
    rare one, so a majority vote spreads the error. The rule is longest-wins --
    a predicted span is expanded to the longest other predicted form of the
    same label that contains it at that position. Expansion only, capped at
    `max_expand` characters so one over-merged prediction cannot swallow a
    paragraph.
    """
    out = {}
    for d, spans in pred.items():
        t = text[d]
        if wordsnap:
            spans = word_snap(spans, t, skip)
        forms = collections.defaultdict(collections.Counter)
        for a, b, l in spans:
            if b - a >= min_len:
                forms[l][t[a:b]] += 1
        cover = collections.defaultdict(list)
        for l, cnt in forms.items():
            for st, n in cnt.items():
                if n >= min_support:
                    cover[l].extend(occurrences(t, st))
        new, grown = [], 0
        for a, b, l in spans:
            x0, y0 = a, b
            for x, y in cover[l]:
                if x <= a and y >= b and (y - x) > (y0 - x0)                         and (y - x) - (b - a) <= max_expand:
                    x0, y0 = x, y
            grown += (x0, y0) != (a, b)
            new.append((x0, y0, l))
        # Propagation. Expansion alone only repairs spans the model already
        # emitted; the oracle says per-document string matching scores 1.000
        # precision, so an entity string the model is confident about should
        # also be tagged at the occurrences it missed. Support is required
        # because a single spurious prediction would otherwise be stamped
        # across the whole document.
        if propagate:
            sup = collections.Counter((t[a:b], l) for a, b, l in new)
            have = [(a, b) for a, b, _ in new]
            for (st, l), n in sup.items():
                if n < propagate or len(st) < min_len:
                    continue
                for x, y in occurrences(t, st):
                    if not any(x < b and y > a for a, b in have):
                        new.append((x, y, l))
                        have.append((x, y))
        seen, fin = set(), []
        for p in new:
            k = (int(p[0]), int(p[1]), str(p[2]))
            if k not in seen:
                seen.add(k)
                fin.append(k)
        out[d] = fin
    return out


def repeat_audit(text, gold, split, labels, args):
    """How much headroom within-document reuse can possibly have."""
    rep = collections.Counter()
    tot = collections.Counter()
    for d in split["fold1"] + split["fold2"]:
        t = text[d]
        c = collections.Counter(t[s:e] for s, e, _ in gold[d])
        for s, e, l in gold[d]:
            tot[l] += 1
            if c[t[s:e]] > 1:
                rep[l] += 1
    log("")
    log("  IN-DOCUMENT REPETITION -- gold spans sharing their exact string")
    log("  with another gold span in the same document:")
    for l in sorted(labels, key=lambda x: -tot[x]):
        log("    %-11s %4d of %4d  %5.1f%%"
            % (l, rep[l], tot[l], 100 * rep[l] / max(tot[l], 1)))
    log("    %-11s %4d of %4d  %5.1f%%"
        % ("ALL", sum(rep.values()), sum(tot.values()),
           100 * sum(rep.values()) / max(sum(tot.values()), 1)))

    # Ceiling: if we knew each document's entity STRINGS but nothing about
    # where they occur, how well would string matching recover the spans?
    docs = split["fold1"] + split["fold2"]
    pred = {}
    for d in docs:
        g = {text[d][a:b]: (l, 1.0, 1) for a, b, l in gold[d]
             if b - a >= args.min_len}
        pred[d] = tag(text[d], g)
    macro, overlap, per = score(pred, {d: gold[d] for d in docs}, labels)
    log("")
    log("  ORACLE per-document string matching: strict %.4f  overlap %.4f"
        % (macro, overlap))
    log("  (gold strings, model-free -- the ceiling of within-document reuse)")
    for l in sorted(labels, key=lambda x: -per[x][0]):
        log("    %-11s F1=%.3f P=%.3f R=%.3f" % (l, per[l][0], per[l][1],
                                                 per[l][2]))


def evaluate(text, gold, split, labels, args):
    """Score saved held-out CV predictions under each post-process. CPU, free.

    Every boundary idea from here on is a few seconds of CPU against this file
    rather than a GPU fold, which is the only way to search the space in the
    time left.
    """
    pred = {d: [(int(a), int(b), str(l)) for a, b, l in v]
            for d, v in json.load(open(args.evaluate, encoding="utf-8")).items()}
    other = {}
    for tr, ev in (("fold1", "fold2"), ("fold2", "fold1")):
        g, _ = build(split[tr], text, gold, args.min_len, args.dominance,
                     args.min_rate)
        for d in split[ev]:
            other[d] = g
    truth = {d: gold[d] for d in pred}

    def per_fold(p):
        """Both folds separately. One number over 70 documents and 72 configs
        is a selection-bias generator; a config that wins on one fold and loses
        on the other has been fitted to the split, not to the problem."""
        return [score({d: p[d] for d in split[f] if d in p},
                      {d: truth[d] for d in split[f] if d in p}, labels)[0]
                for f in ("fold1", "fold2")]

    def report(name, p):
        m, o, per = score(p, truth, labels)
        worst = sorted(labels, key=lambda l: per[l][0])[:4]
        f1, f2 = per_fold(p)
        log("  %-22s strict %.4f  overlap %.4f  folds %.4f/%.4f  worst: %s"
            % (name, m, o, f1, f2,
               " ".join("%s=%.2f" % (l, per[l][0]) for l in worst)))
        return m

    log("")
    log("  post-processing, scored on the organisers' folds:")
    base = report("raw", pred)
    cross = {d: snap(v, tag(text[d], other[d]), args.add_missing)
             for d, v in pred.items()}
    report("+ cross-doc gazetteer", cross)
    sc = self_consistent(pred, text, args.min_len,
                         args.min_support, args.max_expand, args.propagate,
                          args.wordsnap, set(args.wordsnap_skip))
    report("+ self-consistency", sc)
    both = {d: snap(v, tag(text[d], other[d]), args.add_missing)
            for d, v in sc.items()}
    m = report("+ both", both)
    log("")
    log("  best gain over raw: %+.4f" % (max(m, base) - base))

    if not args.sweep:
        return
    # Over-merging and truncation pull the expansion rule in opposite
    # directions and the logs show both, so the parameters are searched rather
    # than argued about. Free: no GPU, no submission burned per hypothesis.
    log("")
    log("  sweep (ranked by the WORSE fold, not the mean -- a config that")
    log("  wins on one fold and loses on the other is fitted to the split):")
    rows = []
    for ws in (False, True):
        for ms in (1, 2):
            for me in (24, 48, 96):
                for pr in (0, 2, 3):
                    q = self_consistent(pred, text, args.min_len, ms, me, pr,
                                        ws, set(args.wordsnap_skip))
                    for cross in (False, True):
                        r = ({d: snap(v, tag(text[d], other[d]), False)
                              for d, v in q.items()} if cross else q)
                        f1, f2 = per_fold(r)
                        rows.append((min(f1, f2), (f1 + f2) / 2, ws, ms, me,
                                     pr, cross))
    rows.sort(reverse=True)
    log("   worse   mean   wordsnap  minsup  maxexp  propagate  crossdoc")
    for r in rows[:12]:
        log("  %.4f %.4f   %-8s  %-6d  %-6d  %-9d  %s"
            % (r[0], r[1], r[2], r[3], r[4], r[5], r[6]))
    log("")
    log("  raw baseline %.4f on folds %.4f/%.4f"
        % (base, *per_fold(pred)))


def do_selfconsist(text, gold, split, labels, args):
    src = "%s/submissions/prediction_%s.json" % (P2, args.selfconsist)
    pred = {d: [(int(a), int(b), str(l)) for a, b, l in v]
            for d, v in json.load(open(src, encoding="utf-8")).items()}
    test_text = {str(k): v for k, v in load_json("Test/fold3-text.json").items()}
    n0 = sum(len(v) for v in pred.values())
    out = self_consistent(pred, test_text, args.min_len,
                          args.min_support, args.max_expand, args.propagate,
                          args.wordsnap, set(args.wordsnap_skip))
    changed = sum(len(set(out[d]) - set(pred[d])) for d in pred)
    log("  %d of %d spans rewritten by within-document reuse" % (changed, n0))
    log("  %d -> %d total" % (n0, sum(len(v) for v in out.values())))

    if args.also_gazetteer:
        g, _ = build(split["fold1"] + split["fold2"], text, gold,
                     args.min_len, args.dominance, args.min_rate)
        out = {d: snap(v, tag(test_text[d], g), args.add_missing)
               for d, v in out.items()}
        log("  then cross-doc gazetteer -> %d total"
            % sum(len(v) for v in out.values()))

    t = "%s_sc%s%s%s" % (args.selfconsist,
                       "w" if args.wordsnap else "",
                       "p%d" % args.propagate if args.propagate else "",
                       "g" if args.also_gazetteer else "")
    pj = "%s/submissions/prediction_%s.json" % (P2, t)
    json.dump({d: [list(x) for x in v] for d, v in out.items()},
              open(pj, "w", encoding="utf-8"), ensure_ascii=False)
    zp = "%s/submissions/submission_%s.zip" % (P2, t)
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(pj, "prediction.json")
    log("  wrote %s" % zp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--snap", default=None, metavar="TAG",
                    help="snap submissions/prediction_TAG.json")
    ap.add_argument("--add-missing", action="store_true",
                    help="also emit gazetteer matches the model never found")
    ap.add_argument("--also-gazetteer", action="store_true",
                    help="with --selfconsist, apply both")
    ap.add_argument("--repeat-audit", action="store_true",
                    help="within-document repetition stats and its ceiling")
    ap.add_argument("--selfconsist", default=None, metavar="TAG",
                    help="propagate each document's own repeated extents")
    ap.add_argument("--evaluate", default=None, metavar="CVPRED",
                    help="score results/cvpred_*.json under each post-process")
    ap.add_argument("--min-len", type=int, default=4)
    ap.add_argument("--min-support", type=int, default=1)
    ap.add_argument("--max-expand", type=int, default=48)
    ap.add_argument("--sweep", action="store_true",
                    help="with --evaluate, search the post-process grid")
    ap.add_argument("--wordsnap-skip", nargs="*", default=["PREC"],
                    metavar="LABEL",
                    help="labels exempt from --wordsnap (default: PREC)")
    ap.add_argument("--wordsnap", action="store_true",
                    help="widen every span to the enclosing word")
    ap.add_argument("--propagate", type=int, default=0, metavar="N",
                    help="also tag missed occurrences of strings "
                         "predicted at least N times (0 = off)")
    ap.add_argument("--dominance", type=float, default=0.9)
    ap.add_argument("--min-rate", type=float, default=0.5)
    a = ap.parse_args()

    os.makedirs("%s/submissions" % P2, exist_ok=True)
    with open(fetch("Train/labels.txt"), encoding="utf-8") as f:
        labels = [x.strip() for x in f if x.strip()]
    text, gold = {}, {}
    for fold in ("fold1", "fold2"):
        text.update({str(k): v
                     for k, v in load_json("Train/%s-text.json" % fold).items()})
        gold.update(parse_gold(load_json("Train/%s-gold.json" % fold)))
    split = json.load(open("%s/split.json" % P2))
    log("  %d labels, %d train documents" % (len(labels), len(text)))

    did = False
    if a.audit:
        audit(text, gold, split, labels, a); did = True
    if a.repeat_audit:
        repeat_audit(text, gold, split, labels, a); did = True
    if a.evaluate:
        evaluate(text, gold, split, labels, a); did = True
    if a.snap:
        do_snap(text, gold, split, labels, a); did = True
    if a.selfconsist:
        do_selfconsist(text, gold, split, labels, a); did = True
    if not did:
        ap.error("pass --audit, --repeat-audit, --evaluate, --snap or "
                 "--selfconsist")


if __name__ == "__main__":
    main()
