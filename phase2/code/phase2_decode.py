#!/usr/bin/env python3
"""CS-781 Phase 2 -- decode saved emissions offline. CPU ONLY.

    python phase2_decode.py --emis results/emis_ext6e --sweep
    python phase2_decode.py --emis results/emis_ext6e --write ctx:mass:1.0 avg:argmax:0
    python phase2_decode.py --emis results/emis_ext6e --write ctx:mass:1.0:0.6/0.2

The optional fourth field is the document-level string vote, add/drop -- see
vote_doc().

Reads what `phase2_weaksup.py` now saves per ensemble member: held-out CV
log-probs for both folds and test log-probs, each under two window rules. Every
decoding idea below becomes seconds of CPU instead of a six-hour GPU run, and
each one is scored through the SAME post-processing that ships (word-snap +
longest-wins, then rules snap-and-add), because tuning against a proxy of the
submitted system is what lost 0.068, 0.027, 0.036 and 0.029 on the leaderboard.

THE THREE AXES

window   avg | ctx -- see phase2_weaksup.emissions2. Causal trunks see almost
         no left context in the first half of a window; `ctx` keeps only the
         copy of each token that had the most.
decoder  argmax   per-token argmax over all 25 tags, what has always shipped
         mass     two-stage: a token is inside SOME entity when P(not O) beats
                  P(O); a run of such tokens splits only where B-mass beats
                  I-mass; each piece takes the label with the highest summed
                  log P(B-l)+P(I-l) over its tokens
         massl    as mass, also splitting where the per-token label changes
bias     log-odds shift toward O. Emission is 139 spans/document against 120
         gold and precision is 0.25-0.40 on half the labels, so the default
         operating point is probably on the wrong side.

WHY `mass` IS THE ONE WITH A REASON BEHIND IT

The ensemble emits 2.03 PREC and 2.39 CASENO spans per gold entity at 0.74
overlap: one entity, several pieces. Per-token argmax over 25 tags produces
exactly that. A token inside a citation with P(O)=0.40, P(I-PREC)=0.35,
P(B-PREC)=0.25 is argmaxed to O and splits the entity, although the model
puts 0.60 on "inside a PREC". Likewise a token split 0.3/0.3 between I-APP and
I-RESP breaks one party name into two labels. Deciding entity-vs-O first and
the label per span second removes both failure shapes without retraining.

Selection is by the WORSE of the two folds. The grid is 2 x 3 x ~7 = 42
configurations over 70 documents; ranking by the mean would fit the split.
"""

import argparse
import glob
import json
import os
import re
import sys
import zipfile

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase2_data import P2, decode_tags, fetch, load_json, log, parse_gold  # noqa: E402
from phase2_gazetteer import score, self_consistent, snap                    # noqa: E402
from phase2_gazetteer import build as gaz_build, tag as gaz_tag             # noqa: E402
from phase2_rules import proposals                                           # noqa: E402

BIAS = {"argmax": (-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0),
        "mass": (-0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0),
        "massl": (0.0, 0.5, 1.0, 1.5, 2.0)}


# ----------------------------------------------------------------- loading --
def load_members(dirs, pattern, window):
    """Mean log-prob over every member file matching `pattern` in `dirs`."""
    files = sorted(f for d in dirs for f in glob.glob(os.path.join(d, pattern)))
    if not files:
        return None, []
    pool = None
    for f in files:
        em = torch.load(f, weights_only=False)[window]
        em = {d: v.float() for d, v in em.items()}
        pool = em if pool is None else {k: pool[k] + em[k] for k in pool}
    return {k: v / len(files) for k, v in pool.items()}, files


def load_spans(dirs, name):
    for d in dirs:
        f = os.path.join(d, "spans_%s.pt" % name)
        if os.path.exists(f):
            return torch.load(f, weights_only=False)
    raise FileNotFoundError("spans_%s.pt in %s" % (name, dirs))


# ---------------------------------------------------------------- decoding --
def decode_doc(lp, spans, text, labels, mode, bias):
    if lp.size(0) == 0:
        return []
    if mode == "argmax":
        x = lp.clone()
        x[:, 0] += bias
        tags = x.argmax(-1).tolist()
        return [tuple(s) for s in decode_tags(tags, spans, labels, text, "bio")]

    p = torch.softmax(lp, -1)
    p_o = p[:, 0].clamp(1e-6, 1 - 1e-6)
    ent = (torch.log1p(-p_o) - torch.log(p_o)) > bias
    pb, pi = p[:, 1::2], p[:, 2::2]                 # B-l at 2k+1, I-l at 2k+2
    lab_mass = torch.log((pb + pi).clamp(min=1e-9))
    start_here = (pb.sum(-1) > pi.sum(-1)).tolist()
    tok_lab = lab_mass.argmax(-1).tolist()

    segs, cur, prev = [], None, -2
    for t in ent.nonzero().flatten().tolist():
        split = (cur is None or t != prev + 1 or start_here[t]
                 or (mode == "massl" and tok_lab[t] != tok_lab[prev]))
        if split:
            if cur:
                segs.append(cur)
            cur = [t]
        else:
            cur.append(t)
        prev = t
    if cur:
        segs.append(cur)

    tags = [0] * lp.size(0)
    for s in segs:
        k = int(lab_mass[s].sum(0).argmax())
        tags[s[0]] = 2 * k + 1
        for t in s[1:]:
            tags[t] = 2 * k + 2
    return [tuple(x) for x in decode_tags(tags, spans, labels, text, "bio")]


def decode_all(pool, spans, text, labels, mode, bias):
    return {d: decode_doc(pool[d], spans[d], text[d], labels, mode, bias)
            for d in pool}


def vote_doc(pred, lp, tspans, text, labels, add, drop, min_len=4):
    """Decide each distinct entity STRING once per document, from all of its
    occurrences' emissions, then apply that decision to every occurrence.

    The oracle in EXPERIMENTS 5.2: tagging each document by string matching on
    its own gold strings scores 0.9863 strict. The task is "find each entity
    string once, copy it everywhere", and APP/RESP/WIT -- three of the four
    labels the detector cannot find -- repeat 77-81% within a document.

    Propagation by predicted COUNT lost 0.0058 because it stamped spurious
    strings everywhere. This gates on the model's own probability instead:
      score(s) = mean over every occurrence of s of mean P(not O) on its tokens
      label(s) = argmax of summed P(B-l)+P(I-l) over every occurrence
    score >= add   -> every occurrence is tagged, all with label(s)
    score <  drop  -> every prediction of s is removed as a false positive
    otherwise      -> predictions kept, relabelled to label(s) for consistency
    """
    import bisect
    from phase2_gazetteer import occurrences
    if not pred or lp.size(0) == 0:
        return pred
    p = torch.softmax(lp, -1)
    ent = 1 - p[:, 0]
    lab = p[:, 1::2] + p[:, 2::2]
    starts = [s for s, _ in tspans]

    def toks(a, b):
        i = max(0, bisect.bisect_right(starts, a) - 1)
        out = []
        while i < len(tspans) and tspans[i][0] < b:
            if tspans[i][1] > a:
                out.append(i)
            i += 1
        return out

    strings = {}
    for a, b, l in pred:
        s = text[a:b]
        if len(s) >= min_len and s not in strings:
            occ = [(x, y) for x, y in occurrences(text, s)]
            tk = [toks(x, y) for x, y in occ]
            tk = [t for t in tk if t]
            if not tk:
                continue
            sc = sum(float(ent[t].mean()) for t in tk) / len(tk)
            lm = sum(lab[t].sum(0) for t in tk)
            strings[s] = (sc, labels[int(lm.argmax())], occ)

    out, taken = [], []
    for a, b, l in pred:
        s = text[a:b]
        if s in strings:
            sc, vl, _ = strings[s]
            if sc < drop:
                continue
            l = vl
        out.append((a, b, l))
        taken.append((a, b))
    # longest strings first, so a full party name claims its occurrences
    # before a fragment of it can
    for s, (sc, vl, occ) in sorted(strings.items(), key=lambda kv: -len(kv[0])):
        if sc < add:
            continue
        for x, y in occ:
            if not any(x < bb and y > aa for aa, bb in taken):
                out.append((x, y, vl))
                taken.append((x, y))
    seen, fin = set(), []
    for a, b, l in out:
        k = (int(a), int(b), str(l))
        if k not in seen:
            seen.add(k)
            fin.append(k)
    return fin


def postprocess(pred, text, rules, gaz, vote=None, pool=None, tspans=None,
                labels=None):
    """The shipped chain, exactly as wsens6_scwg_rpadd (0.5577) was built:

        phase2_gazetteer.py --selfconsist TAG --wordsnap --also-gazetteer
        phase2_rules.py     --apply TAG_scwg --propagate --add

    `gaz` is the cross-document gazetteer; for CV it must be built from the
    OTHER fold only, for test from all 70 documents.

    With `vote=(add, drop)` the document-level string vote runs after
    word-snap and longest-wins have fixed the extents it votes on, and before
    the gazetteer and rules, which have the last word on boundaries.
    """
    sc = self_consistent(pred, text, 4, 1, 48, 0, True, {"PREC"})
    if vote:
        sc = {d: vote_doc(v, pool[d], tspans[d], text[d], labels, *vote)
              for d, v in sc.items()}
    sc = {d: snap(v, gaz_tag(text[d], gaz), False) for d, v in sc.items()}
    return {d: snap(sc[d], rules[d], True) for d in sc}


def validate(pred, labels):
    """Every rule evaluate_phase2.py rejects on. One violation fails the lot."""
    allowed = set(labels)
    for d, spans in pred.items():
        seen = set()
        for a, b, l in spans:
            assert isinstance(a, int) and isinstance(b, int), d
            assert 0 <= a < b, (d, a, b)
            assert l in allowed, (d, l)
            assert (a, b, l) not in seen, (d, a, b, l)
            seen.add((a, b, l))


def rule_spans(text, docs):
    out = {}
    for d in docs:
        p = proposals(text[d], None, 0.40, True)        # --propagate
        out[d] = [(a, b, l) for l, v in p.items() for a, b in v]
    return out


VOTES = ((0.5, 0.0), (0.6, 0.0), (0.7, 0.0), (0.5, 0.2), (0.6, 0.2),
         (0.7, 0.3), (1.1, 0.2), (1.1, 0.3))     # add=1.1: relabel/drop only


def parse_cfg(s):
    """'ctx:mass:1.0' or 'ctx:mass:1.0:0.6/0.2' (vote add/drop)."""
    parts = s.split(":")
    w, m, b = parts[:3]
    assert w in ("avg", "ctx") and m in BIAS, s
    vote = tuple(float(x) for x in parts[3].split("/")) if len(parts) > 3 \
        else None
    return w, m, float(b), vote


def cfg_name(w, m, b, vote):
    return "%s:%s:%g%s" % (w, m, b, ":%g/%g" % vote if vote else "")


# ------------------------------------------------------------------- sweep --
def sweep(a, dirs, labels, text, gold, split):
    cv = {}
    for w in ("avg", "ctx"):
        for fold in ("fold1", "fold2"):
            pool, files = load_members(dirs, "cv_%s_*.pt" % fold, w)
            if pool is None:
                sys.exit("no cv_%s_*.pt in %s -- has the CV half finished?"
                         % (fold, dirs))
            cv[(w, fold)] = pool
        log("  %s: %d member files per fold" % (w, len(files)))
    spans = {**load_spans(dirs, "fold1"), **load_spans(dirs, "fold2")}
    docs = split["fold1"] + split["fold2"]
    rules = rule_spans(text, docs)
    truth = {d: gold[d] for d in docs}
    # held-out gazetteer: scoring fold2 may only use strings from fold1
    gaz = {"fold2": gaz_build(split["fold1"], text, gold)[0],
           "fold1": gaz_build(split["fold2"], text, gold)[0]}

    def run(w, m, b, vote=None):
        raw, pp = {}, {}
        for fold in ("fold1", "fold2"):
            pool = cv[(w, fold)]
            p = decode_all(pool, spans, text, labels, m, b)
            raw[fold] = p
            pp[fold] = postprocess(p, text, rules, gaz[fold], vote, pool,
                                   spans, labels)
        r = [score(raw[f], {d: truth[d] for d in split[f]}, labels)[0]
             for f in ("fold1", "fold2")]
        q = [score(pp[f], {d: truth[d] for d in split[f]}, labels)[0]
             for f in ("fold1", "fold2")]
        n = sum(len(v) for f in raw for v in raw[f].values()) / len(docs)
        return r, q, n, {**pp["fold1"], **pp["fold2"]}

    rows = []
    log("")
    log("  window decoder  bias  | raw f1/f2      | post f1/f2      worse   "
        "spans/doc")
    for w in ("avg", "ctx"):
        for m, grid in BIAS.items():
            for b in grid:
                r, q, n, _ = run(w, m, b)
                rows.append((min(q), sum(q) / 2, w, m, b, r, q, n))
                log("  %-6s %-7s %5.1f | %.4f/%.4f  | %.4f/%.4f  %.4f  %6.1f%s"
                    % (w, m, b, r[0], r[1], q[0], q[1], min(q), n,
                       "   <- what ships today"
                       if (w, m, b) == ("avg", "argmax", 0.0) else ""))
    rows.sort(key=lambda x: (-x[0], -x[1]))
    base = next(x for x in rows if x[2:5] == ("avg", "argmax", 0.0))
    log("")
    log("  TOP 8 by the worse fold (gold averages 119.9 spans/doc):")
    for x in rows[:8]:
        log("    %s:%s:%g   worse %.4f  mean %.4f  (%+.4f vs shipped)  %.1f/doc"
            % (x[2], x[3], x[4], x[0], x[1], x[1] - base[1], x[7]))

    # The document-level vote only on the three best decoders and the
    # shipped one, to keep the number of configurations searched small.
    log("")
    log("  document-level string vote (add/drop) on the top decoders:")
    vrows = []
    heads = rows[:3] + ([base] if base not in rows[:3] else [])
    for x in heads:
        for v in VOTES:
            _, q, n, _ = run(x[2], x[3], x[4], v)
            vrows.append((min(q), sum(q) / 2, x[2], x[3], x[4], v, q))
            log("    %-28s post %.4f/%.4f  worse %.4f  (%+.4f vs no vote)"
                % (cfg_name(x[2], x[3], x[4], v), q[0], q[1], min(q),
                   min(q) - x[0]))
    vrows.sort(key=lambda r: (-r[0], -r[1]))
    v = vrows[0]
    log("  best with vote: %s  worse %.4f  mean %.4f  (%+.4f vs shipped)"
        % (cfg_name(*v[2:6]), v[0], v[1], v[1] - base[1]))

    best = rows[0] if rows[0][0] >= v[0] else (v[0], v[1], *v[2:5])
    bvote = None if rows[0][0] >= v[0] else v[5]
    _, _, _, pb = run(*best[2:5], bvote)
    _, _, _, p0 = run("avg", "argmax", 0.0)
    _, _, per_b = score(pb, truth, labels)
    _, _, per_0 = score(p0, truth, labels)
    log("")
    log("  per label, post-processed, both folds pooled:")
    log("  %-11s shipped  %s" % ("", cfg_name(*best[2:5], bvote)))
    for l in labels:
        log("  %-11s  %.3f    %.3f   %+.3f"
            % (l, per_0[l][0], per_b[l][0], per_b[l][0] - per_0[l][0]))


# ------------------------------------------------------------------- write --
def write(a, dirs, labels, cfgs):
    test_text = {str(k): v for k, v in load_json("Test/fold3-text.json").items()}
    spans = load_spans(dirs, "test")
    rules = rule_spans(test_text, list(test_text))
    text, gold = {}, {}
    for fold in ("fold1", "fold2"):
        text.update({str(k): v for k, v in
                     load_json("Train/%s-text.json" % fold).items()})
        gold.update(parse_gold(load_json("Train/%s-gold.json" % fold)))
    gaz = gaz_build(list(text), text, gold)[0]      # all 70, as --also-gazetteer
    pat = "test_*_e%d.pt" % a.test_epochs if a.test_epochs else "test_*.pt"
    os.makedirs("%s/submissions" % P2, exist_ok=True)
    for c in cfgs:
        w, m, b, vote = parse_cfg(c)
        pool, files = load_members(dirs, pat, w)
        if pool is None:
            sys.exit("no %s in %s -- has the submit half finished?" % (pat, dirs))
        eps = sorted({re.search(r"_e(\d+)\.pt$", f).group(1) for f in files})
        if len(eps) > 1:
            sys.exit("test files from different gold-epoch counts %s are "
                     "mixed; pass --test-epochs" % eps)
        raw = decode_all(pool, spans, test_text, labels, m, b)
        if not vote:
            # raw test spans, so phase2_gazetteer/phase2_rules can rebuild the
            # _scwg and _scwg_rpadd stages that phase2_stack.py takes with '*'
            json.dump({d: [list(x) for x in raw.get(d, [])] for d in test_text},
                      open("%s/submissions/prediction_%s.json"
                           % (P2, wtag(a.tag, w, m, b)), "w", encoding="utf-8"),
                      ensure_ascii=False)
        pp = postprocess(raw, test_text, rules, gaz, vote, pool, spans, labels)
        pp = {d: [[int(x), int(y), str(l)] for x, y, l in pp.get(d, [])]
              for d in test_text}
        validate({d: [tuple(s) for s in v] for d, v in pp.items()}, labels)
        got = {s[2] for v in pp.values() for s in v}
        miss = [l for l in labels if l not in got]
        tag = "%s%s_pp" % (wtag(a.tag, w, m, b),
                           ("_v%g-%g" % vote).replace(".", "p") if vote else "")
        pj = "%s/submissions/prediction_%s.json" % (P2, tag)
        json.dump(pp, open(pj, "w", encoding="utf-8"), ensure_ascii=False)
        zp = "%s/submissions/submission_%s.zip" % (P2, tag)
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(pj, "prediction.json")
        nr = sum(len(v) for v in raw.values()) / len(test_text)
        npp = sum(len(v) for v in pp.values()) / len(test_text)
        log("  %-22s %d members  raw %.1f/doc  post %.1f/doc%s"
            % (c, len(files), nr, npp,
               "  NEVER PREDICTS %s" % miss if miss else ""))
        log("    WROTE %s" % zp)


def wtag(tag, w, m, b):
    return "%s_%s%s%s" % (tag, w, m, ("%+.1f" % b).replace(".", "p"))


def dump_cv(a, dirs, labels, cfgs):
    """Raw (pre-post-processing) held-out predictions for phase2_stack.py.

    Written as results/cvpred_<tag>_<w><dec><bias>.json; the matching test
    file is the one --write produces for the same config (same name + _pp).
    """
    text = {}
    for fold in ("fold1", "fold2"):
        text.update({str(k): v for k, v in
                     load_json("Train/%s-text.json" % fold).items()})
    spans = {**load_spans(dirs, "fold1"), **load_spans(dirs, "fold2")}
    for c in cfgs:
        w, m, b, vote = parse_cfg(c)
        assert vote is None, "stacking post-processes the raw spans itself"
        out = {}
        for fold in ("fold1", "fold2"):
            pool, _ = load_members(dirs, "cv_%s_*.pt" % fold, w)
            out.update(decode_all(pool, spans, text, labels, m, b))
        p = "%s/results/cvpred_%s.json" % (P2, wtag(a.tag, w, m, b))
        json.dump({d: [list(x) for x in v] for d, v in out.items()},
                  open(p, "w", encoding="utf-8"), ensure_ascii=False)
        log("  wrote %s  (%.1f raw spans/doc)"
            % (p, sum(len(v) for v in out.values()) / len(out)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-cv", nargs="*", default=None, metavar="W:DEC:BIAS",
                    help="write raw CV predictions for phase2_stack.py")
    ap.add_argument("--emis", nargs="+", required=True,
                    help="emission dirs from phase2_weaksup.py; members from "
                         "all of them are pooled")
    ap.add_argument("--sweep", action="store_true",
                    help="score every window x decoder x bias on CV")
    ap.add_argument("--write", nargs="*", default=None, metavar="W:DEC:BIAS",
                    help="write post-processed test submissions, e.g. "
                         "ctx:mass:1.0")
    ap.add_argument("--test-epochs", type=int, default=None,
                    help="only use test_*_e<N>.pt")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    dirs = [d if os.path.isabs(d) else os.path.join(P2, d) for d in a.emis]
    a.tag = a.tag or os.path.basename(dirs[0].rstrip("/")).replace("emis_", "")

    with open(fetch("Train/labels.txt"), encoding="utf-8") as f:
        labels = [x.strip() for x in f if x.strip()]
    if a.sweep:
        text, gold = {}, {}
        for fold in ("fold1", "fold2"):
            text.update({str(k): v for k, v in
                         load_json("Train/%s-text.json" % fold).items()})
            gold.update(parse_gold(load_json("Train/%s-gold.json" % fold)))
        split = json.load(open("%s/split.json" % P2))
        gold = {d: [tuple(x) for x in v] for d, v in gold.items()}
        sweep(a, dirs, labels, text, gold, split)
    if a.write:
        write(a, dirs, labels, a.write)
    if a.dump_cv:
        dump_cv(a, dirs, labels, a.dump_cv)
    if not a.sweep and not a.write and not a.dump_cv:
        ap.error("pass --sweep, --write and/or --dump-cv")


if __name__ == "__main__":
    main()
