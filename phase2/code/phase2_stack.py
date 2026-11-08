#!/usr/bin/env python3
"""CS-781 Phase 2 -- learn the span vote across systems. CPU ONLY.

    python phase2_stack.py --cv wsens6 ext6 ens6_m4 \
        --test wsens6_scwg_rpadd ext6_scwg_rpadd ens6_m4_scwg_rpadd --tag stack3

WHAT THE LEADERBOARD SAID

    exact-span vote, 2 of 3 systems     129.6 spans/doc   0.5763   best
    overlap vote, base system's extent  ~138 spans/doc    0.5631
    unanimous, 3 of 3                    97.9 spans/doc   0.5626
    best single system                  140.4 spans/doc   0.5577

Exact agreement is a boundary-correctness filter: where systems find the same
entity but disagree on its extent, the base system's extent is usually wrong.
A fixed k is one point on a curve, and it is the same k for DATE, which every
system bounds correctly, and for APP, which none do.

WHAT THIS DOES

Every span any system emits is a candidate. A gradient-boosted classifier
predicts whether that exact span is a gold span, from features that say how
much the systems agree on it, how many competing extents exist, whether a
validated rule or the gazetteer produced it, and how the string behaves in the
document. Spans above a threshold are kept, the highest-scoring extent wins
where kept spans of one label overlap.

It is trained on the held-out CV predictions of the SAME systems (cvpred_*.json),
post-processed through the exact shipped chain with a held-out gazetteer, so the
measurement and the submission are the same system up to the 35- vs 70-document
refit. Each fold's classifier is trained on the other fold; the threshold is
chosen by the WORSE fold; the comparison row is the exact 2-of-k vote, which is
what scored 0.5763.

Features are counts and flags rather than model probabilities on purpose: the
CV systems are 35-document models and the test systems are 70-document ones,
and probability calibration is exactly what moves between them.
"""

import argparse
import bisect
import collections
import json
import os
import re
import sys
import zipfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase2_data import P2, fetch, load_json, log, parse_gold        # noqa: E402
from phase2_gazetteer import (build as gaz_build, occurrences, score,  # noqa: E402
                              self_consistent, snap, tag as gaz_tag)
from phase2_rules import merge_same, proposals                      # noqa: E402

TH = [round(0.05 + 0.025 * i, 3) for i in range(27)]                # .05-.70


def stages(raw, text, rules, gaz):
    """(scwg, full): after --selfconsist --wordsnap --also-gazetteer, and after
    --apply --propagate --add on top of that."""
    sc = self_consistent(raw, text, 4, 1, 48, 0, True, {"PREC"})
    sc = {d: snap(v, gaz_tag(text[d], gaz), False) for d, v in sc.items()}
    return sc, {d: snap(sc[d], rules[d], True) for d in sc}


def postprocess(raw, text, rules, gaz):
    return stages(raw, text, rules, gaz)[1]


def as_set(v):
    return {(int(a), int(b), str(l)) for a, b, l in v}


VERSUS = re.compile(r"\b(?:versus|vs\.?|v/s)\b", re.I)
PREV = ["mr", "mrs", "ms", "shri", "sri", "smt", "dr", "km", "kum", "m/s",
        "advocate", "counsel", "sr", "learned", "pw", "dw", "witness", "the",
        "section", "and", "of", "by", "j", "justice", "hon'ble", "no", "dated",
        "on", "in", "vs", "v", "versus", "appellant", "respondent"]
NEXT = [r"appellant|petitioner|applicant|plaintiff", r"respondent|defendant",
        r"for the", r"counsel|advocate", r"\bp\.?w|\bd\.?w|witness|deposed",
        r"section|\bact\b|article", r"court", r"\bj\.|justice", r"\bno\.|case",
        r"dated|\bon\b", r"\bv\.|\bvs|versus", r"\(", r"&|\bors\b|\banr\b"]
NEXT = [re.compile(x) for x in NEXT]
INSIDE = [re.compile(x, re.I) for x in (
    r"\bv\.|\bvs\.?\b|versus", r"\bno\.", r"\bact\b", r"section|article",
    r"court|tribunal|commission", r"\bj\.", r"&|\bors\b|\banr\b",
    r"state of|union of|government", r"\bair\b|\bscc\b|\bscr\b|\(\d{4}\)",
    r"\bmr\b|\bshri\b|\bsri\b", r"\bpw|\bdw")]


def context(t, a, b, s, vpos):
    """Where the span sits and what surrounds it -- the evidence APP, RESP,
    WIT and the counsel labels are actually defined by."""
    left = t[max(0, a - 40):a].lower()
    right = t[b:b + 60].lower()
    pw = re.findall(r"[a-z/'.]+", left)
    prev = pw[-1].strip(".") if pw else ""
    first = t.find(s)
    return [
        PREV.index(prev) + 1 if prev in PREV else 0,
        *[int(bool(r.search(right))) for r in NEXT],
        *[int(bool(r.search(left[-25:]))) for r in NEXT[:5]],
        *[int(bool(r.search(s))) for r in INSIDE],
        int(vpos >= 0 and vpos - 1500 <= first < vpos),
        int(vpos >= 0 and vpos < first <= vpos + 1500),
        int(first < 6000), min(first, 200000) / 1000.0,
        (first - vpos) / 1000.0 if vpos >= 0 else 0.0,
        sum(ch.isupper() for ch in s) / max(1, sum(ch.isalpha() for ch in s)),
    ]


SWAP = (("APP", "RESP", "WIT"), ("A.COUNSEL", "R.COUNSEL"))


# ------------------------------------------------- extent perturbation --
# Built from the held-out misses of stack18 (error analysis, 2026-10-11):
#   PREC   'Kashiram & Ors.['           -> gold stops before '['
#          'State of T.N'               -> gold runs on to '(1991) 3 SCC 471'
#          'Rana Pratap'                -> gold '... and others vs. State of Haryana'
#   CASENO '272'                        -> gold '(2009) 2 SCC 272'   (start late x68)
#          'Regular Civil Suit No. 921' -> gold 'No. 921 of 1966'    (start early)
#   APP    'JAYA GOKUL EDUCATIONAL'     -> gold '... TRUST'          (caps run cut)
#   RESP   'the State of Jharkhand'     -> gold 'State of Jharkhand'
# Annotation conventions, not document layout -- the kind of regularity that
# should carry from 70 training documents to fold3, unlike the context cues
# that cost 0.025-0.030 on the leaderboard.
PERT_TYPES = ["none", "trim_the", "trim_lead_punct", "trim_trail", "start-1w",
              "start+1w", "end-1w", "end+1w", "bracket_close", "cite_tail",
              "caps_run", "title_run", "vs_ext", "start-2w", "start-3w",
              "caps_run_nodot", "start_no", "start+2w", "rep_page"]
CAPS_STOP = {"APPELLANT", "APPELLANTS", "APPELLANT(S)", "RESPONDENT",
             "RESPONDENTS", "RESPONDENT(S)", "PETITIONER", "PETITIONERS",
             "PETITIONER(S)", "VS", "V", "VERSUS", "AND", "ANR", "ORS", "DATE",
             "BETWEEN", "WITH", "IN", "THE"}
TITLE_STOP = {"The", "In", "On", "Mr", "Mrs", "Ms", "Shri", "Sri", "Smt", "And",
              "For", "Vs", "Versus", "Appellant", "Respondent", "Petitioner"}
CITE = re.compile(r"[\s,.]*(?:AIR\s+\d{4}\s+[A-Za-z.]+\s*\d+"
                  r"|[\[(]{0,2}\d{4}[\])]{0,2}\s*(?:\(\d+\)\s*)?\d*\s*"
                  r"[A-Z][A-Za-z.]*(?:\s+[A-Z][A-Za-z.]*)?\s+\d+[\])]{0,2})")
VSEXT = re.compile(r"(?:\s+(?:and|&)\s+(?:others|another|ors\.?|anr\.?))?"
                   r"\s+(?:[Vv][Ss]?\.?|[Vv]ersus)\s+(?:(?:[A-Z][\w.&'()/-]*"
                   r"|of|the|and)\s+){0,6}(?:[A-Z][\w.&'()/-]*)")


REPPAGE = re.compile(r"\s+[A-Z][A-Za-z]*\.?\s*\d+")


def _caps_tok(w):
    core = w.strip(".,;:()[]")
    return (bool(re.fullmatch(r"[A-Z0-9&().,'/-]+", w))
            and any(ch.isalpha() for ch in w) and core.upper() not in CAPS_STOP)


def variants(t, ws, wstart, a, b):
    """[(a', b', type)] boundary variants of one span."""
    out = []
    s = t[a:b]
    i = max(0, bisect.bisect_right(wstart, a) - 1)
    j = max(0, bisect.bisect_right(wstart, b - 1) - 1)
    m = re.match(r"(?:the|The|THE)\s+", s)
    if m and m.end() < len(s):
        out.append((a + m.end(), b, "trim_the"))
    k = a
    while k < b and not t[k].isalnum():
        k += 1
    if a < k < b:
        out.append((k, b, "trim_lead_punct"))
    e = b
    while e > a + 1 and t[e - 1] in "[(.,;:& ":
        e -= 1
        out.append((a, e, "trim_trail"))
    if i > 0:
        out.append((ws[i - 1][0], b, "start-1w"))
    if i > 1:
        out.append((ws[i - 2][0], b, "start-2w"))
    if i > 2:
        out.append((ws[i - 3][0], b, "start-3w"))
    if i + 1 <= j:
        out.append((ws[i + 1][0], b, "start+1w"))
    if j - 1 >= i:
        out.append((a, ws[j - 1][1], "end-1w"))
    if j + 1 < len(ws):
        out.append((a, ws[j + 1][1], "end+1w"))
    for op, cl in (("(", ")"), ("[", "]")):
        if s.count(op) > s.count(cl):
            q = t.find(cl, b, b + 60)
            if q >= 0:
                out.append((a, q + 1, "bracket_close"))
    m = CITE.match(t, b)
    if m and m.end() - b <= 60:
        out.append((a, m.end(), "cite_tail"))
    if re.search(r"\d{4}$", s):                 # '..., AIR 1975' + ' Guj. 120'
        m = REPPAGE.match(t, b)
        if m:
            out.append((a, m.end(), "rep_page"))
    m = re.search(r"\b[Nn][Oo][Ss]?\.", s)        # 'Regular Civil Suit No. 921'
    if m and m.start() > 0:
        out.append((a + m.start(), b, "start_no"))
    if i + 2 <= j:
        out.append((ws[i + 2][0], b, "start+2w"))
    m = VSEXT.match(t, b)
    if m and m.end() - b <= 90:
        out.append((a, m.end(), "vs_ext"))
    if any(ch.isupper() for ch in s) and s == s.upper():
        x, y = i, j
        while x > 0 and _caps_tok(t[ws[x - 1][0]:ws[x - 1][1]]):
            x -= 1
        while y + 1 < len(ws) and _caps_tok(t[ws[y + 1][0]:ws[y + 1][1]]):
            y += 1
        if (x, y) != (i, j):
            ca, cb = ws[x][0], ws[y][1]
            while ca < cb - 1 and not t[ca].isalnum():
                ca += 1
            while cb > ca + 1 and t[cb - 1] in ",;:":
                cb -= 1
            out.append((ca, cb, "caps_run"))
            if t[cb - 1] == ".":
                out.append((ca, cb - 1, "caps_run_nodot"))
    y, end = j, b
    while y + 1 < len(ws) and y - j < 3:
        w = t[ws[y + 1][0]:ws[y + 1][1]]
        core = w.rstrip(".,;:")
        if not re.fullmatch(r"[A-Z][a-z]+", core) or core in TITLE_STOP:
            break
        y, end = y + 1, ws[y + 1][0] + len(core)
        if core != w:                              # 'Chouguley.' ends the name
            break
    if y != j:
        out.append((a, end, "title_run"))
    return [(x, y, k) for x, y, k in out if 0 <= x < y <= len(t)
            and t[x:y].strip() == t[x:y]]


NO_PERTURB = {"DATE", "STAT", "COURT"}       # ceilings already 0.87-0.97
CH_CLASSES = "[(.,)]&;:"


def _chclass(c):
    """0 none, 1 upper, 2 lower, 3 digit, 4 space, 5.. one of CH_CLASSES."""
    if not c:
        return 0
    if c.isupper():
        return 1
    if c.islower():
        return 2
    if c.isdigit():
        return 3
    if c.isspace():
        return 4
    k = CH_CLASSES.find(c)
    return 5 + k if k >= 0 else 5 + len(CH_CLASSES)


def shape(t, a, b, s):
    """What the span itself looks like at its edges. Stack18's misses differ
    from gold in exactly these: a trailing '[' footnote marker, an unclosed
    bracket, a missing reporter page, a leading 'the'. Character-level form
    of the span, not neighbouring words or position in the document."""
    return [
        s.count("(") - s.count(")"), s.count("[") - s.count("]"),
        _chclass(s[:1]), _chclass(s[-1:]),
        _chclass(t[a - 1:a] if a > 0 else ""), _chclass(t[b:b + 1]),
        int(s[-1:].isdigit()), int(s[:4].lower() == "the "),
        int(bool(re.search(r"\s(?:v|vs|versus)\.?\s", s, re.I))),
    ]


def _overlap(A, B, sa, sb):
    """n x m boolean: candidate i overlaps span j."""
    if len(sa) == 0:
        return np.zeros((len(A), 0), dtype=bool)
    return (A[:, None] < sb[None, :]) & (B[:, None] > sa[None, :])


def features(d, t, systems, rules_d, gaz_d, labels, expand=False, rich=False,
             swap=False, perturb=False):
    """Candidates and their feature rows for one document.

    With `expand`, candidates also include every other occurrence of every
    string a system emitted (with the label it was emitted with) and every
    gazetteer match. With `perturb`, every system span (except DATE, already
    at 0.96) also proposes the boundary variants above, each tagged with the
    kind of change. Overlap counts are vectorised because perturbation
    multiplies the candidates roughly tenfold.
    """
    sets = [as_set(s.get(d, [])) for s in systems]
    rset, gset = as_set(rules_d), as_set(gaz_d)
    emit = collections.Counter()            # (string, label) -> emissions
    emit_any = collections.Counter()        # string -> emissions, any label
    for S in sets:
        for a, b, l in S:
            emit[(t[a:b], l)] += 1
            emit_any[t[a:b]] += 1
    base = set().union(*sets)
    pool = set(base)
    if expand:
        for (s, l) in list(emit):
            if len(s) >= 4 and s == s.strip():
                pool |= {(x, y, l) for x, y in occurrences(t, s)}
        pool |= gset
    pert = {}
    if perturb:
        ws = [(m.start(), m.end()) for m in re.finditer(r"\S+", t)]
        wstart = [x for x, _ in ws]
        for a, b, l in sorted(base):
            if l in NO_PERTURB:
                continue
            for x, y, k in variants(t, ws, wstart, a, b):
                c = (x, y, l)
                if c not in pool:
                    pool.add(c)
                    pert[c] = (PERT_TYPES.index(k), x - a, y - b)
    if swap:
        for x, y, l in list(pool):
            for g in SWAP:
                if l in g:
                    pool |= {(x, y, o) for o in g if o in labels}
    ext_sets = [{(x, y) for x, y, _ in S} for S in sets]
    cands = sorted(pool)
    if not cands:
        return [], []
    n = len(cands)
    A = np.array([c[0] for c in cands], dtype=np.int64)
    B = np.array([c[1] for c in cands], dtype=np.int64)
    lab_i = {l: i for i, l in enumerate(labels)}
    Lc = np.array([lab_i[c[2]] for c in cands], dtype=np.int64)

    def arr(S):
        v = sorted(S)
        return (np.array([x[0] for x in v], dtype=np.int64),
                np.array([x[1] for x in v], dtype=np.int64),
                np.array([lab_i[x[2]] for x in v], dtype=np.int64))

    EX = np.array([[c in S for S in sets] for c in cands], dtype=np.int64)
    OV = np.zeros((n, len(sets)), dtype=np.int64)
    OA = np.zeros((n, len(sets)), dtype=np.int64)
    for k, S in enumerate(sets):
        sa, sb, sl = arr(S)
        M = _overlap(A, B, sa, sb)
        OA[:, k] = M.any(1)
        OV[:, k] = (M & (Lc[:, None] == sl[None, :])).any(1)
    ra, rb, rl = arr(rset)
    RV = (_overlap(A, B, ra, rb) & (Lc[:, None] == rl[None, :])).any(1)
    ncomp = np.zeros(n, dtype=np.int64)
    longest = np.ones(n, dtype=np.int64)
    shortest = np.ones(n, dtype=np.int64)
    LEN = B - A
    for li in np.unique(Lc):
        g = np.where(Lc == li)[0]
        M = _overlap(A[g], B[g], A[g], B[g])
        np.fill_diagonal(M, False)
        cnt = M.sum(1)
        mx = np.where(M, LEN[g][None, :], -1).max(1)
        mn = np.where(M, LEN[g][None, :], 1 << 30).min(1)
        ncomp[g] = cnt
        longest[g] = (cnt == 0) | (LEN[g] >= mx)
        shortest[g] = (cnt == 0) | (LEN[g] <= mn)

    str_lab = collections.defaultdict(set)
    str_cnt = collections.Counter()
    for a, b, l in cands:
        str_lab[t[a:b]].add(l)
        str_cnt[(t[a:b], l)] += 1
    occ_cache = {}
    L = len(t) or 1
    vm = VERSUS.search(t[:8000])
    vpos = vm.start() if vm else -1
    rows = []
    for i, (a, b, l) in enumerate(cands):
        s = t[a:b]
        if s not in occ_cache:
            occ_cache[s] = len(occurrences(t, s)) if s.strip() else 0
        n_occ = occ_cache[s]
        ex, ov, oa = EX[i].tolist(), OV[i].tolist(), OA[i].tolist()
        sx = sum(ex)
        rows.append(ex + ov + oa + [
            emit[(s, l)], emit_any[s] - emit[(s, l)],
            emit[(s, l)] / max(1, n_occ * len(sets)),
            int(sx == 0),
            sx, sum(ov), sum(oa), sum(ov) - sx,
            lab_i[l],
            b - a, len(s.split()),
            int((a, b, l) in rset), int(RV[i]), int((a, b, l) in gset),
            min(n_occ, 50), str_cnt[(s, l)], len(str_lab[s]),
            int(ncomp[i]), int(longest[i]), int(shortest[i]),
            int(s.isupper()), int(s[:1].isupper()),
            int(any(ch.isdigit() for ch in s)), s.count(","),
            int(s[-1:] in ".,;:)"), a / L,
        ] + (context(t, a, b, s, vpos) if rich else [])
          + ([int((a, b) in E) for E in ext_sets]
             + [sum((a, b) in E for E in ext_sets),
                sx / max(1, sum((a, b) in E for E in ext_sets))]
             if swap else [])
          + (list(pert.get((a, b, l), (0, 0, 0))) + shape(t, a, b, s)
             if perturb else []))
    return cands, rows


def table(docs, text, systems, rules, gazs, labels, gold=None, expand=False,
          rich=False, swap=False, perturb=False):
    X, y, idx = [], [], []
    for d in docs:
        g = as_set(gold[d]) if gold else set()
        gz = gaz_tag(text[d], gazs[d])
        cands, rows = features(d, text[d], systems, rules[d], gz, labels,
                               expand, rich, swap, perturb)
        X += rows
        y += [int(c in g) for c in cands]
        idx += [(d, c) for c in cands]
    return np.array(X, dtype=np.float32), np.array(y), idx


def select(p, idx, th, cross=False):
    """Keep p >= th; overlapping kept spans -> highest p wins, within a label
    or (`cross`) across labels."""
    per = collections.defaultdict(list)
    lab_th = th if isinstance(th, dict) else None
    for (d, c), q in zip(idx, p):
        if q >= (lab_th.get(c[2], 1.01) if lab_th is not None else th):
            per[d].append((q, c))
    out = {}
    for d, v in per.items():
        v.sort(reverse=True)
        kept = []
        for q, (a, b, l) in v:
            if not any((cross or k[2] == l) and k[0] < b and k[1] > a
                       for k in kept):
                kept.append((a, b, l))
        out[d] = kept
    return out


def model():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
        min_samples_leaf=40, l2_regularization=1.0, random_state=0)


def fit_predict(Xt, yt, idx_t, Xe, bag, seed0=0):
    """One classifier, or the mean of `bag` classifiers each trained on a
    bootstrap resample of the training DOCUMENTS (rows weighted by how often
    their document was drawn). Documents, not rows, because candidates within
    one document are strongly correlated -- resampling rows would leave every
    document in every bag."""
    if bag <= 1:
        return model().fit(Xt, yt).predict_proba(Xe)[:, 1]
    rdocs = np.array([d for d, _ in idx_t])
    uniq = sorted(set(rdocs))
    out = np.zeros(len(Xe))
    for b in range(bag):
        rng = np.random.default_rng(seed0 + b)
        draw = collections.Counter(rng.choice(uniq, size=len(uniq), replace=True))
        w = np.array([draw.get(d, 0) for d in rdocs], dtype=np.float64)
        keep = w > 0
        m = model()
        m.set_params(random_state=seed0 + b)
        m.fit(Xt[keep], yt[keep], sample_weight=w[keep])
        out += m.predict_proba(Xe)[:, 1]
    return out / bag


def merged(pred, gap):
    """Each label's spans with fragments <= gap characters apart joined.

    The ensemble emits 2.03 PREC and 2.39 CASENO spans per gold entity, and
    their candidate ceilings are 0.547 / 0.675: the full citation is never
    proposed, only its pieces. A merged copy of each system proposes it; the
    classifier, not a fixed gap, decides when the join is right."""
    out = {}
    for d, v in pred.items():
        v = [tuple(x) for x in v]
        r = []
        for l in {x[2] for x in v}:
            r += merge_same(v, l, gap)
        out[d] = r
    return out


def label_thresholds(p, idx, cross, glob_th, targets):
    """Per-label thresholds whose output matches `targets` {label: spans}.

    Leaderboard secondary metrics put the gap to ranks 1-2 in PRECISION
    (0.606 against 0.673-0.690) with recall already level (0.690 against
    0.693). One global threshold sets every label's volume at once; this
    sets each label's volume from the TRAINING gold label mix instead, so a
    label the classifier over-scores cannot take spans from one it
    under-scores. Overlap suppression couples the labels, so each label's
    pre-suppression target is scaled by its survival rate at the global
    solution."""
    sel = select(p, idx, glob_th, cross)
    post = collections.Counter(c[2] for v in sel.values() for c in v)
    by_lab = collections.defaultdict(list)
    for (d, c), q in zip(idx, p):
        by_lab[c[2]].append(q)
    th = {}
    for l, qs in by_lab.items():
        qs.sort(reverse=True)
        pre = sum(1 for q in qs if q >= glob_th)
        surv = post[l] / pre if pre else 1.0
        k = int(round(targets.get(l, 0) / max(surv, 0.05)))
        t_l = 1.01 if k <= 0 else (qs[k - 1] if k <= len(qs) else qs[-1])
        # A label short of good candidates must not reach down into junk to
        # fill its quota: never below a third of the global threshold.
        th[l] = max(t_l, glob_th / 3)
    return th


def vote_k(systems, docs, k):
    out = {}
    for d in docs:
        cnt = collections.Counter()
        for s in systems:
            cnt.update(as_set(s.get(d, [])))
        out[d] = [c for c, n in cnt.items() if n >= k]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv", nargs="+", required=True,
                    help="X for each results/cvpred_X.json (raw held-out CV)")
    ap.add_argument("--test", nargs="+", default=None,
                    help="X for each submissions/prediction_X.json, same "
                         "systems in the same order, already post-processed")
    ap.add_argument("--rich", action="store_true",
                    help="add context features: caption side of 'versus', neighbouring cue words, span content patterns")
    ap.add_argument("--swap", action="store_true",
                    help="offer party/counsel extents under every label of their group (APP/RESP/WIT, A./R.COUNSEL); use with --rich")
    ap.add_argument("--bag", type=int, default=7,
                    help="also try the mean of N document-bootstrap classifiers (1 disables)")
    ap.add_argument("--rates", nargs="*", type=int, default=[124, 128, 132],
                    help="also write test zips whose threshold gives this many spans/doc")
    ap.add_argument("--merge", nargs="*", type=int, default=[],
                    help="for * systems, also add copies with same-label fragments <= N chars apart merged, e.g. --merge 3 8 20")
    ap.add_argument("--perturb", action="store_true",
                    help="every system span also proposes boundary variants "
                         "(trim '[' / 'the', citation tail, caps run, start "
                         "at 'No.', +-words), tagged by kind of change")
    ap.add_argument("--pick", default=None, metavar="EXPAND,CROSS,BAG",
                    help="force a configuration, e.g. 1,1,1 = stack9's "
                         "(expand, cross-label NMS, single classifier)")
    ap.add_argument("--label-prior", action="store_true",
                    help="also set each label's volume from the training "
                         "gold label mix; scored on CV, written as _lp<rate>")
    ap.add_argument("--tag", default="stack")
    a = ap.parse_args()
    assert not a.test or len(a.test) == len(a.cv), "--test must match --cv"
    # '*' = all three stages; the test tag is then the RAW prediction tag and
    # <tag>_scwg / <tag>_scwg_rpadd are loaded beside it

    with open(fetch("Train/labels.txt"), encoding="utf-8") as f:
        labels = [x.strip() for x in f if x.strip()]
    text, gold = {}, {}
    for fold in ("fold1", "fold2"):
        text.update({str(k): v for k, v in
                     load_json("Train/%s-text.json" % fold).items()})
        gold.update(parse_gold(load_json("Train/%s-gold.json" % fold)))
    gold = {d: [tuple(x) for x in v] for d, v in gold.items()}
    split = json.load(open("%s/split.json" % P2))
    docs = split["fold1"] + split["fold2"]
    other = {"fold1": "fold2", "fold2": "fold1"}

    rules = {}
    for d in docs:
        p = proposals(text[d], None, 0.40, True)
        rules[d] = [(x, y, l) for l, v in p.items() for x, y in v]
    gz_fold = {f: gaz_build(split[other[f]], text, gold)[0] for f in other}
    gazs = {d: gz_fold[f] for f in other for d in split[f]}

    # A tag ending in '*' enters at all three stages -- raw, scwg, full --
    # as three separate systems. Post-processing moves extents, sometimes
    # breaking one the raw model had right; as separate inputs the
    # classifier learns which stage to trust for which kind of span.
    systems, full_systems, names = [], [], []
    for tg in a.cv:
        staged, t0 = tg.endswith("*"), tg.rstrip("*")
        raw = json.load(open("%s/results/cvpred_%s.json" % (P2, t0),
                             encoding="utf-8"))
        raw = {d: [tuple(x) for x in raw[d]] for d in docs if d in raw}
        miss = [d for d in docs if d not in raw]
        assert not miss, "cvpred_%s lacks %d documents" % (t0, len(miss))
        scg, pp = {}, {}
        for f in other:
            s1, s2 = stages({d: raw[d] for d in split[f]}, text, rules,
                            gz_fold[f])
            scg.update(s1)
            pp.update(s2)
        if staged:
            systems += [raw, scg, pp]
            names += [t0 + ":raw", t0 + ":scwg", t0]
            for g in a.merge:
                systems.append(merged(pp, g))
                names.append("%s:m%d" % (t0, g))
        else:
            systems.append(pp)
            names.append(t0)
        full_systems.append(pp)

    def by_fold(pred):
        return [score({d: pred.get(d, []) for d in split[f]},
                      {d: gold[d] for d in split[f]}, labels)[0]
                for f in ("fold1", "fold2")]

    log("")
    log("  CV, held-out gazetteer (stacking inputs):")
    for tg, s in zip(names, systems):
        f = by_fold(s)
        log("    %-30s %.4f / %.4f   %.1f spans/doc"
            % (tg, f[0], f[1], sum(len(v) for v in s.values()) / len(docs)))
    for k in range(1, len(full_systems) + 1):
        v = vote_k(full_systems, docs, k)
        f = by_fold(v)
        log("    exact vote %d of %d (full)      %.4f / %.4f   %.1f spans/doc%s"
            % (k, len(full_systems), f[0], f[1],
               sum(len(x) for x in v.values()) / len(docs),
               "   <- 0.5763 on test" if k == 2 and len(full_systems) == 3
               else ""))
    base_k = vote_k(full_systems, docs, 2)
    base_f = by_fold(base_k)

    # cross-fitted classifier, with and without candidate expansion
    tabs, probs, rows = {}, {}, []
    for ex in (False, True):
        tab = {f: table(split[f], text, systems, rules, gazs, labels, gold, ex,
                        a.rich, a.swap, a.perturb)
               for f in other}
        log("")
        log("  expand=%s  candidates: %s  (gold-exact %s of %d gold)"
            % (ex, {f: len(tab[f][1]) for f in tab},
               {f: int(tab[f][1].sum()) for f in tab},
               sum(len(gold[d]) for d in docs)))
        tabs[ex] = tab
        for bag in sorted({1, a.bag}):
            prob = {}
            for f in other:
                Xt, yt, it = tab[other[f]]
                prob[f] = fit_predict(Xt, yt, it, tab[f][0], bag)
            probs[(ex, bag)] = prob
            for th in TH:
                for cross in (False, True):
                    sel = {}
                    for f in other:
                        sel.update(select(prob[f], tab[f][2], th, cross))
                    sc = by_fold(sel)
                    n = sum(len(v) for v in sel.values()) / len(docs)
                    rows.append((min(sc), sum(sc) / 2, th, ex, cross, sc, n,
                                 bag))
    pool_rows = rows
    if a.pick:
        # Force one configuration (expand, cross, bag) instead of the CV
        # choice -- for leaderboard experiments that must hold everything but
        # one factor fixed. The threshold within it is still CV's.
        pe, pc, pb = (int(x) for x in a.pick.split(","))
        pool_rows = [r for r in rows if (int(r[3]), int(r[4]), r[7]) == (pe, pc, pb)]
        assert pool_rows, "no rows for --pick %s (is --bag set?)" % a.pick
    best = max(pool_rows, key=lambda r: (r[0], r[1]))
    _, _, bth, bex, bcross, _, _, bbag = best
    log("")
    log("  config                          th     fold1 / fold2    worse   "
        "mean    /doc")
    for ex in (False, True):
        for cross in (False, True):
            for bag in sorted({1, a.bag}):
                r = max((x for x in rows if x[3] == ex and x[4] == cross
                         and x[7] == bag), key=lambda r: (r[0], r[1]))
                log("  expand=%-5s cross=%-5s bag=%-2d  %.3f  %.4f / %.4f  "
                    "%.4f  %.4f  %5.1f%s"
                    % (ex, cross, bag, r[2], r[5][0], r[5][1], r[0], r[1],
                       r[6], "  <-" if r is best else ""))
    log("")
    log("  threshold curve of the chosen config:")
    for r in sorted((x for x in rows if x[3] == bex and x[4] == bcross
                     and x[7] == bbag), key=lambda r: r[2]):
        log("    th %.3f   %.4f / %.4f   worse %.4f   %5.1f/doc"
            % (r[2], r[5][0], r[5][1], r[0], r[6]))
    log("")
    log("  STACKED  expand=%s cross=%s bag=%d th=%.3f  worse %.4f  mean %.4f"
        "  %.1f/doc" % (bex, bcross, bbag, bth, best[0], best[1], best[6]))
    log("  vote 2-of-%d       worse %.4f  mean %.4f"
        % (len(full_systems), min(base_f), sum(base_f) / 2))
    log("  gain over the 0.5763 rule on CV: %+.4f (worse fold)  %+.4f (mean)"
        % (best[0] - min(base_f), best[1] - sum(base_f) / 2))
    tab, prob = tabs[bex], probs[(bex, bbag)]
    sel = {}
    for f in other:
        sel.update(select(prob[f], tab[f][2], bth, bcross))
    _, _, pb = score(sel, {d: gold[d] for d in docs}, labels)
    cvp = "%s/results/stackcv_%s.json" % (P2, a.tag)
    json.dump({d: [list(x) for x in v] for d, v in sel.items()},
              open(cvp, "w", encoding="utf-8"), ensure_ascii=False)
    log("  wrote %s (held-out stacked predictions, for error analysis)" % cvp)
    _, _, pv = score(base_k, {d: gold[d] for d in docs}, labels)
    log("")
    # Ceiling = share of gold spans that some candidate matches exactly. A
    # label far below its ceiling needs better SELECTION; a label whose
    # ceiling is itself low needs better CANDIDATES, and no stacker fixes it.
    gcount = collections.Counter(l for d in docs for _, _, l in gold[d])
    reach = collections.Counter()
    for f in other:
        for (d, c), yy in zip(tab[f][2], tab[f][1]):
            if yy:
                reach[c[2]] += 1
    log("  per label     vote2   stacked  gain   | stacked P    R   | "
        "candidate ceiling")
    for l in labels:
        log("    %-11s %.3f   %.3f   %+.3f  |  %.3f  %.3f  |  %.3f  (%d/%d)"
            % (l, pv[l][0], pb[l][0], pb[l][0] - pv[l][0], pb[l][1], pb[l][2],
               reach[l] / max(1, gcount[l]), reach[l], gcount[l]))

    # ---- label-prior volumes, scored on CV against the global threshold
    # at the SAME total volume. Fold f's targets come from the OTHER fold's
    # gold label mix -- the eval fold's own labels are never used.
    if a.label_prior:
        log("")
        log("  label-prior volumes vs global threshold, same total spans:")
        for f in other:
            tr = split[other[f]]
            mix = collections.Counter(l for d in tr for _, _, l in gold[d])
            tot = sum(mix.values())
            gsel = select(prob[f], tab[f][2], bth, bcross)
            n_out = sum(len(v) for v in gsel.values())
            targets = {l: n_out * mix[l] / tot for l in labels}
            lth = label_thresholds(prob[f], tab[f][2], bcross, bth, targets)
            lsel = select(prob[f], tab[f][2], lth, bcross)
            truth = {d: gold[d] for d in split[f]}
            g_f = score({d: gsel.get(d, []) for d in split[f]}, truth, labels)[0]
            l_f = score({d: lsel.get(d, []) for d in split[f]}, truth, labels)[0]
            log("    eval %s   global %.4f   label-prior %.4f   (%+.4f)   "
                "%.1f vs %.1f spans/doc"
                % (f, g_f, l_f, l_f - g_f, n_out / len(split[f]),
                   sum(len(v) for v in lsel.values()) / len(split[f])))

    if not a.test:
        return
    test_text = {str(k): v for k, v in load_json("Test/fold3-text.json").items()}
    tdocs = sorted(test_text)
    def readp(t):
        return json.load(open("%s/submissions/prediction_%s.json" % (P2, t),
                              encoding="utf-8"))
    tsys, tfull = [], []
    for tg, cg in zip(a.test, a.cv):
        assert tg.endswith("*") == cg.endswith("*"), (tg, cg)
        t0 = tg.rstrip("*")
        if tg.endswith("*"):
            full = readp(t0 + "_scwg_rpadd")
            tsys += [readp(t0), readp(t0 + "_scwg"), full]
            tsys += [merged(full, g) for g in a.merge]
        else:
            full = readp(t0)
            tsys.append(full)
        tfull.append(full)
    trules = {}
    for d in tdocs:
        p = proposals(test_text[d], None, 0.40, True)
        trules[d] = [(x, y, l) for l, v in p.items() for x, y in v]
    gz_all = gaz_build(docs, text, gold)[0]
    Xa = np.concatenate([tabs[bex][f][0] for f in other])
    ya = np.concatenate([tabs[bex][f][1] for f in other])
    ia = [x for f in other for x in tabs[bex][f][2]]
    Xs, _, idx = table(tdocs, test_text, tsys, trules,
                       {d: gz_all for d in tdocs}, labels, None, bex, a.rich,
                       a.swap, a.perturb)
    ps = fit_predict(Xa, ya, ia, Xs, bbag)
    os.makedirs("%s/submissions" % P2, exist_ok=True)
    # Thresholds by OUTPUT VOLUME on test. The leaderboard measured stack9 at
    # 132.0 / 124.9 / 119.2 spans/doc -> 0.6425 / 0.6421 / 0.6268: flat from
    # ~125 to ~132, falling below ~120, and CV-chosen thresholds land on the
    # falling side because test candidates come from 70-document models and
    # the classifier, trained on 35-document ones, underrates them. Matching
    # a span count uses no test label -- only how much we emit -- and makes
    # configurations comparable whatever their probability scale.
    def rate(th):
        return sum(len(v) for v in select(ps, idx, th, bcross).values()) \
            / len(tdocs)

    def at_rate(r):
        lo_, hi_ = 0.0, 1.0                  # rate(lo_) > r >= rate(hi_)
        for _ in range(30):
            mid = (lo_ + hi_) / 2
            if rate(mid) > r:
                lo_ = mid
            else:
                hi_ = mid
        return min((lo_, hi_), key=lambda t: abs(rate(t) - r))

    plan = [(bth, "")] + [(at_rate(r), "_r%d" % r) for r in a.rates]
    if a.label_prior:
        mix = collections.Counter(l for d in docs for _, _, l in gold[d])
        tot = sum(mix.values())
        for r in a.rates:
            th0 = at_rate(r)
            targets = {l: r * len(tdocs) * mix[l] / tot for l in labels}
            plan.append((label_thresholds(ps, idx, bcross, th0, targets),
                         "_lp%d" % r))
    for th, name in plan:
        sel = select(ps, idx, th, bcross)
        out = {d: [[int(x), int(y), str(l)] for x, y, l in sorted(sel.get(d, []))]
               for d in tdocs}
        for d, v in out.items():
            assert len({tuple(s) for s in v}) == len(v), d
        tag = "%s%s" % (a.tag, name)
        pj = "%s/submissions/prediction_%s.json" % (P2, tag)
        json.dump(out, open(pj, "w", encoding="utf-8"), ensure_ascii=False)
        zp = "%s/submissions/submission_%s.zip" % (P2, tag)
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(pj, "prediction.json")
        log("  th=%s  %.1f spans/doc  WROTE %s"
            % ("per-label" if isinstance(th, dict) else "%.3f" % th,
               sum(len(v) for v in out.values()) / len(tdocs), zp))
    v2 = vote_k(tfull, tdocs, 2)
    log("  (test exact 2-of-%d vote: %.1f spans/doc)"
        % (len(tfull), sum(len(v) for v in v2.values()) / len(tdocs)))


if __name__ == "__main__":
    main()
