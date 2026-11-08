#!/usr/bin/env python3
"""CS-781 Phase 2 -- domain patterns, each one measured before it is trusted.

    python phase2_rules.py --validate              # score every rule on gold
    python phase2_rules.py --apply ens6_m4_scwg    # build a submission

CPU ONLY.

WHY RULES AT ALL

Strict macro-F1 weights all twelve labels equally, and the deficit is
concentrated in six that the model FINDS and mis-bounds: R.COUNSEL 0.100 at
overlap 0.654, PREC 0.186/0.688, CASENO 0.249/0.706, JUDGE 0.341/0.779. Each
is worth 1/12 = 0.083 outright. Per-document string matching scores 0.9863
strict when the strings are right, so the whole problem is choosing extents --
and for a citation, a date or a case number the extent is defined by a format,
not by anything a 126M model trained on 70 documents is going to infer.

WHY EVERY RULE IS SCORED SEPARATELY

A pattern that looks obvious in six sampled contexts is a guess. The
cross-document gazetteer looked obvious too and scored -0.0002 on the
leaderboard. `--validate` runs each rule over all 70 training documents and
reports exact precision, exact recall and overlap precision against gold, so a
rule earns its place or is deleted. Overlap precision separates the two ways a
rule fails: low exact but high overlap means the pattern finds the right thing
with the wrong edges, low both means it is firing on the wrong text entirely.

HOW THEY ARE APPLIED

Rules do not replace the model. The model is the detector -- it is at ~0.78
overlap and nothing here beats that. A rule match that OVERLAPS a model span of
the same label replaces its boundaries (`--apply`); a match the model missed is
only added under `--add`, which is measured separately because the model
already over-emits.
"""

import argparse
import collections
import json
import os
import re
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase2_data import P2, fetch, load_json, log, parse_gold  # noqa: E402
from phase2_gazetteer import score, snap  # noqa: E402

RULES = []

# Exact precision and recall measured by --validate over all 70 training
# documents on 2026-10-08. Kept in the source because --apply filters on it:
# a rule whose extent is wrong most of the time must never be allowed to
# overwrite a model boundary, however sensible the pattern looks.
MEASURED = {
    "DATE numeric":                         (0.976, 0.695),
    "DATE 19th June, 2009":                 (0.957, 0.128),
    "DATE May 1, 1987":                     (0.966, 0.090),
    "CASENO abbrev+No.+year":               (0.680, 0.336),
    "CASENO Civil Appeal No. N of YYYY":    (0.636, 0.055),
    "CASENO YYYY AIR N":                    (0.134, 0.017),   # precedents
    "PREC X v. Y[, citation]":              (0.066, 0.037),   # ovl .797
    "STAT <Name> Act[, YYYY]":              (0.271, 0.395),   # 1840 fires
    "STAT abbreviations":                   (0.740, 0.248),
    "STAT Constitution":                    (0.688, 0.086),
    "WIT PW-N <name>":                      (0.800, 0.086),
    "WIT <name> (PW-N)":                    (0.783, 0.096),
    "JUDGE HON'BLE JUSTICE X":              (0.308, 0.016),
    "JUDGE X, J.":                          (0.436, 0.069),
    "JUDGE BENCH: line split":              (0.857, 0.122),   # was .174
    "A.COUNSEL honorific + for appellant":  (0.741, 0.103),
    "R.COUNSEL honorific + for respondent": (0.839, 0.141),
    "APP PETITIONER: X Vs.":                (0.588, 0.025),   # ovl .941
    "APP X ..Appellant":                    (0.700, 0.034),
    "RESP RESPONDENT: X":                   (0.571, 0.031),   # ovl .810
    "RESP Versus X ..Respondent":           (0.643, 0.024),
    # Measured 2026-10-09. The 1.000 seeds these replace meant no threshold
    # filtered them, so the run that produced these numbers had the useless
    # WIT rule active throughout and WIT read 0.396 instead of 0.567.
    "APP caption before versus":            (0.472, 0.042),   # ovl .694
    "RESP caption after versus":            (0.400, 0.042),   # ovl .725
    "AUTH common authorities":              (0.661, 0.462),   # AUTH 0 -> .50
    "AUTH <Name> Board/Ministry/Tribunal":  (0.232, 0.110),   # drop
    "WIT witness/deposed <name>":           (0.011, 0.005),   # 94 fires, 1 hit
}

# Labels whose entities repeat inside a document often enough that one
# high-precision match carries the rest: APP 77.1%, RESP 80.3%, WIT 81.3%,
# A.COUNSEL 50.8%, JUDGE 47.3%, R.COUNSEL 43.5% (measured by
# phase2_gazetteer.py --repeat-audit). These are exactly the rules with high
# precision and recall near 0.1 -- they fire once, at the caption, and stop.
PROPAGATE = ("APP", "RESP", "WIT", "A.COUNSEL", "R.COUNSEL", "JUDGE")


def rule(name, label):
    def deco(f):
        RULES.append((name, label, f))
        return f
    return deco


def spans_of(pat, text, group=0):
    for m in pat.finditer(text):
        if m.group(group) is not None:
            yield m.start(group), m.end(group)


# --------------------------------------------------------------------- DATE --
MONTH = (r"(?:January|February|March|April|May|June|July|August|September"
         r"|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept|Sep"
         r"|Oct|Nov|Dec)")
# 585 spans are dd.mm.yyyy, 167 d.mm.yyyy, 95 dd/mm/yyyy, 71 d.m.yyyy,
# 43 dd.mm.yy -- one pattern covers all of them.
DATE_NUM = re.compile(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b")
DATE_DMY = re.compile(r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:of\s+)?" + MONTH +
                      r",?\s+\d{4}\b")
DATE_MDY = re.compile(r"\b" + MONTH + r"\s+\d{1,2},\s*\d{4}\b")


@rule("DATE numeric", "DATE")
def _d1(t):
    return spans_of(DATE_NUM, t)


@rule("DATE 19th June, 2009", "DATE")
def _d2(t):
    return spans_of(DATE_DMY, t)


@rule("DATE May 1, 1987", "DATE")
def _d3(t):
    return spans_of(DATE_MDY, t)


# ------------------------------------------------------------------- CASENO --
# Skeletons: AAA.AA.999/9999 (RFA.No.708/2006), A.A.A.AA.99999 AA 9999
# (C.M.P.No.15283 of 2019), A.A. AA. 999 AA 9999 (C.S. No. 369 of 2014).
CASE_ABBR = re.compile(
    r"\b(?:[A-Z][A-Za-z]{0,4}\.?\s?){1,4}(?:No|NO|no)\.?\s*\d+\s*"
    r"(?:/|of\s|OF\s|\.)\s*\d{2,4}\b")
CASE_LONG = re.compile(
    r"\b(?:CIVIL|Civil|CRIMINAL|Criminal|WRIT|Writ|SPECIAL|Special)\s+"
    r"(?:APPEAL|Appeal|PETITION|Petition|LEAVE|Leave)s?\s*(?:\([A-Za-z]+\)\s*)?"
    r"(?:No|NO)\.?\s*\d+\s*(?:of|OF|/)\s*\d{4}\b")
CASE_CITE = re.compile(r"\b\d{4}\s+(?:AIR|SCR|SCC|SC)\s*\d+\b")


@rule("CASENO abbrev+No.+year", "CASENO")
def _c1(t):
    return spans_of(CASE_ABBR, t)


@rule("CASENO Civil Appeal No. N of YYYY", "CASENO")
def _c2(t):
    return spans_of(CASE_LONG, t)


@rule("CASENO YYYY AIR N", "CASENO")
def _c3(t):
    return spans_of(CASE_CITE, t)


# --------------------------------------------------------------------- PREC --
# Commas sit INSIDE a case name -- "Wealth-tax Commissioner, A.P. v. Courts
# of Wards" -- so the run has to cross them. Capitalised "Vs." is dropped:
# it is the caption separator, and allowing it matched
# "SREE SREE ISHWAR SRIDHAR JEW Vs. RESPONDENT" as a precedent.
NAMEY = r"[A-Z][\w.&'()-]*(?:[\s,]+[A-Za-z][\w.&'()-]*){0,6}"
PREC_V = re.compile(NAMEY + r"\s+(?:v\.|vs\.)\s+" + NAMEY +
                    r"(?:,\s*(?:\(\d{4}\)|\[\d{4}\]|AIR)[^;\n]{0,40})?")


@rule("PREC X v. Y[, citation]", "PREC")
def _p1(t):
    return spans_of(PREC_V, t)


# --------------------------------------------------------------------- STAT --
STAT_ACT = re.compile(r"\b(?:[A-Z][\w.&'-]*\s+){0,4}(?:Act|Code|Rules)"
                      r"(?:,?\s*\d{4})?\b")
STAT_ABBR = re.compile(r"\b(?:IPC|CPC|Cr\.P\.C\.|CrPC|PMLA|NDPS|TADA|POTA"
                       r"|SARFAESI|MV Act|UGC Act|AICTE)\b")
STAT_CONST = re.compile(r"\bConstitution(?:\s+of\s+India)?\b")


@rule("STAT <Name> Act[, YYYY]", "STAT")
def _s1(t):
    return spans_of(STAT_ACT, t)


@rule("STAT abbreviations", "STAT")
def _s2(t):
    return spans_of(STAT_ABBR, t)


@rule("STAT Constitution", "STAT")
def _s3(t):
    return spans_of(STAT_CONST, t)


# ---------------------------------------------------------------------- WIT --
# 'LW-3 Pathan Basheerunnisa', 'PW-5 Jagidsh', 'the complainant Jagdish (PW-5)'
# A capitalised run stops at a connector, or 'HON'BLE JUSTICE BISWANATH
# SOMADDER AND THE HON'BLE...' captures 'BISWANATH SOMADDER AND THE'.
PTOK = (r"(?!(?:AND|THE|OF|WITH|VERSUS|VS|FOR|IN|AT|BY|And|The|Of)\b)"
        r"[A-Z][\w.'-]*")
PERSON = PTOK + r"(?:\s+" + PTOK + r"){0,3}"
WIT_PRE = re.compile(r"\b(?:PW|LW|DW|CW)[-. ]?\d+\s+(" + PERSON + r")")
WIT_POST = re.compile(r"\b(" + PERSON + r")\s*\(\s*(?:PW|LW|DW|CW)[-. ]?\d+\s*\)")


@rule("WIT PW-N <name>", "WIT")
def _w1(t):
    return spans_of(WIT_PRE, t, 1)


@rule("WIT <name> (PW-N)", "WIT")
def _w2(t):
    return spans_of(WIT_POST, t, 1)


# -------------------------------------------------------------------- JUDGE --
JUDGE_HON = re.compile(r"(?:HON'BLE|Hon'ble)\s+(?:MR\.?|MRS\.?|MS\.?|Mr\.?"
                       r"|Mrs\.?|Ms\.?)?\s*(?:JUSTICE|Justice)\s+(" +
                       PERSON + r")")
JUDGE_J = re.compile(r"\b([A-Z][A-Z.]+(?:\s+[A-Z][A-Z.]+)*(?:,\s*[A-Z.]+)?)"
                     r"\s*,\s*(?:J\.|C\.J\.)")
JUDGE_BENCH = re.compile(r"BENCH:\s*([^\n]{2,80}?)(?=\s*(?:BENCH:|CITATION:"
                         r"|ACT:|\n))")
# Measured at exact 0.174 but OVERLAP 0.913: the region is right and the span
# is not, because one BENCH line holds several judges --
# "BHAGWATI, NATWARLAL H. MUKHERJEA, B.K. BOSE, VIVIAN" came back as one span.
# This is the same adjacent-entity merge the model makes and BIO cannot fix.
ONE_JUDGE = re.compile(r"[A-Z][A-Za-z]+,\s*[A-Z][A-Za-z.]*"
                       r"(?:\s+(?![A-Z][A-Za-z]+\s*,)[A-Z][A-Za-z.]*)?")


@rule("JUDGE HON'BLE JUSTICE X", "JUDGE")
def _j1(t):
    return spans_of(JUDGE_HON, t, 1)


@rule("JUDGE X, J.", "JUDGE")
def _j2(t):
    return spans_of(JUDGE_J, t, 1)


@rule("JUDGE BENCH: line split", "JUDGE")
def _j3(t):
    for m in JUDGE_BENCH.finditer(t):
        base = m.start(1)
        for j in ONE_JUDGE.finditer(m.group(1)):
            yield base + j.start(), base + j.end()


# ------------------------------------------------------------------ COUNSEL --
HON = r"(?:Mr|Mrs|Ms|Shri|Smt|Dr|Miss)\.?\s+"
COUNSEL = re.compile(r"\b" + HON + r"(" + PERSON + r")")
FOR_APP = re.compile(r"appellant|petitioner|Appellant|Petitioner")
FOR_RESP = re.compile(r"respondent|Respondent|State|defendant")


def _counsel(t, want):
    """An honorific-led name is counsel for whichever side the clause names.

    'Mr. Abhinav Mukerji, learned counsel appearing for the appellant' and
    'Shri Ankit Premchandani, Panel Lawyer for the respondents' differ only in
    the trailing clause, so the side is read from the next 140 characters and
    the match is discarded when neither side appears or both do.
    """
    for m in COUNSEL.finditer(t):
        tail = t[m.end(1):m.end(1) + 140]
        if "counsel" not in tail.lower() and "advocate" not in tail.lower() \
                and "lawyer" not in tail.lower() and "general" not in tail.lower():
            continue
        a, r = bool(FOR_APP.search(tail)), bool(FOR_RESP.search(tail))
        if a == r:
            continue
        if (want == "A") == a:
            yield m.start(1), m.end(1)


@rule("A.COUNSEL honorific + for appellant", "A.COUNSEL")
def _ac(t):
    return _counsel(t, "A")


@rule("R.COUNSEL honorific + for respondent", "R.COUNSEL")
def _rc(t):
    return _counsel(t, "R")


# ----------------------------------------------------------------- APP/RESP --
CAP_PET = re.compile(r"PETITIONER:\s*(.{2,70}?)\s*(?:Vs\.|VERSUS|versus|V\.)",
                     re.S)
CAP_RES = re.compile(r"RESPONDENT:\s*(.{2,70}?)\s*(?:DATE OF JUDGMENT|BENCH:)",
                     re.S)
CAP_APP = re.compile(r"([A-Z][A-Za-z.&'@ -]{2,58}?)\s*\.{2,}\s*"
                     r"(?:Appellants?|Petitioners?|APPELLANTS?)")
CAP_RSP = re.compile(r"(?:Versus|VERSUS|versus)\s+([A-Za-z.&'@ -]{2,58}?)"
                     r"\s*\.{2,}\s*(?:Respondents?|RESPONDENTS?)")


@rule("APP PETITIONER: X Vs.", "APP")
def _a1(t):
    return spans_of(CAP_PET, t, 1)


@rule("APP X ..Appellant", "APP")
def _a2(t):
    return spans_of(CAP_APP, t, 1)


@rule("RESP RESPONDENT: X", "RESP")
def _r1(t):
    return spans_of(CAP_RES, t, 1)


@rule("RESP Versus X ..Respondent", "RESP")
def _r2(t):
    return spans_of(CAP_RSP, t, 1)


# The existing caption rules need a literal 'PETITIONER:' or '..Appellant',
# and only 37 of the 70 training documents carry one. Every caption has a
# versus token, though, and the formats differ only around it:
#
#   PETITIONER: SREE SREE ISHWAR SRIDHAR JEW Vs. RESPONDENT: ...
#   Writ Petition No.15591/2020 Dharmendra Jatav Vs. State of M.P., and others
#   MERLIN PROJECTS LIMITED VERSUS PAWAN KUMAR AGARWAL
#   Mathew Thomas Petitioner(s) versus Union of India and ...
#   Pathan Hussain Basha ... Appellant Versus State of A.P. ... Respondent
#
# So: find the first versus token in the head of the document, and take the
# capitalised run that ends just before it and the one that begins just after.
# One hit per document instead of one per two documents, and propagation
# carries it from there -- APP recurs 77.1% within a document, RESP 80.3%.
VS_TOK = re.compile(r"\s(?:Vs\.|VS\.|V/s|V\.|VERSUS|Versus|versus|vs\.|v\.)\s")
# 'State of M.P.' and 'Union of India' need lowercase joiners inside the run,
# so interior tokens may be of/the/and/& as well as capitalised words.
# 'With' and 'and' are capitalised-or-not continuations that belong to the
# NEXT clause, not the party: 'PAWAN KUMAR AGARWAL With A.P.D.' and
# 'Union of India and others' both over-ran. 'of' and 'the' must stay, for
# 'State of M.P.' and 'Union of India'.
_STOP = r"(?:With|WITH|And|AND|Through|THROUGH|Versus|VERSUS|In|IN)"
_JOIN = r"(?:(?!" + _STOP + r"\b)[A-Z][A-Za-z.&'@-]*|of|the|OF|THE|&)"
PARTY_L = re.compile(r"([A-Z][A-Za-z.&'@-]*(?:\s+" + _JOIN + r"){0,6})\s*$")
PARTY_R = re.compile(r"^\s*([A-Z][A-Za-z.&'@-]*(?:\s+" + _JOIN + r"){0,6})")
HEAD = 2500
# Markers that belong to the form, not to the party.
NOISE = re.compile(r"^(?:PETITIONER|RESPONDENT|Petitioner|Respondent|Appellant"
                   r"|Appellants|APPELLANT|APPELLANTS|RESPONDENTS)\b[:\s]*")
# 'Mathew Thomas Petitioner(s) versus ...' -- the role marker sits between the
# party and the versus token, so it has to come off before the run is matched.
LTRIM = re.compile(r"[\s.,:;]*(?:Petitioner\(s\)|Petitioners?|Appellant\(s\)"
                   r"|Appellants?|PETITIONERS?|APPELLANTS?)[\s.,:;]*$")
RTRIM = re.compile(r"[\s,]+(?:and|the|of|with|through|AND|OF|THE)\s*$")


def _caption(t, side):
    m = VS_TOK.search(t, 0, HEAD)
    if not m:
        return
    if side == "L":
        lo = max(0, m.start() - 90)
        # LTRIM only strips a SUFFIX, so offsets measured from `lo` stay valid.
        seg = LTRIM.sub("", t[lo:m.start()])
        g = PARTY_L.search(seg)
        if not g:
            return
        a, b = lo + g.start(1), lo + g.end(1)
    else:
        lo = m.end()
        g = PARTY_R.search(t[lo:lo + 90])
        if not g:
            return
        a, b = lo + g.start(1), lo + g.end(1)
    r = RTRIM.search(t[a:b])
    if r:
        b = a + r.start()
    # 'RESPONDENT: SUSHILA BALA DASI' -- drop the form label, keep the party.
    n = NOISE.match(t[a:b])
    if n:
        a += n.end()
    if b - a >= 3:
        yield a, b


@rule("APP caption before versus", "APP")
def _a3(t):
    return _caption(t, "L")


@rule("RESP caption after versus", "RESP")
def _r3(t):
    return _caption(t, "R")


# AUTH is 1003 spans over only 125 distinct strings and the common ones are
# generic across judgments -- 'State Government' x119, 'Government' x111,
# 'Special Court' x92, 'Central Government' x64. The model scores 0.342 at
# overlap 0.455, so this is a detection gap, not a boundary one.
AUTH_FIX = re.compile(r"\b(?:State Government|Central Government|Government"
                      r"|Special Court|Railway Board|Appellate Board"
                      r"|Compensation Officer|State Govt\.?)\b")
AUTH_SUFF = re.compile(r"\b(?:[A-Z][A-Za-z]+\s+){1,3}"
                       r"(?:Board|Ministry|Tribunal|Commission|Authority"
                       r"|Department|Corporation|Council|Committee)\b")


@rule("AUTH common authorities", "AUTH")
def _au1(t):
    return spans_of(AUTH_FIX, t)


@rule("AUTH <Name> Board/Ministry/Tribunal", "AUTH")
def _au2(t):
    return spans_of(AUTH_SUFF, t)


# WIT rules fire 43 times against 187 gold spans. A witness is also named
# where the deposition is described.
WIT_DEP = re.compile(r"\b(?:witness|deposed by|examined as)\s+(" + PERSON +
                     r")", re.I)
WIT_NUM = re.compile(r"\b(?:PW|LW|DW|CW)\s*[-. ]?\s*\d+\b")


@rule("WIT witness/deposed <name>", "WIT")
def _w3(t):
    return spans_of(WIT_DEP, t, 1)


# ----------------------------------------------------------------- harness --
TAIL = re.compile(r"\s*(?:&|and|AND)\s*(?:ORS|Ors|ors|ANR|Anr|anr|OTHERS"
                  r"|Others|others)\.?\s*$")


def strip_party(t, a, b):
    """Gold stops at the party, not at '... AND OTHERS'.

    'RESPONDENT: SUSHILA BALA DASI AND OTHERS.' is annotated as
    'SUSHILA BALA DASI', and 'JAGE RAM & ORS.' as 'Jage Ram'.
    """
    m = TAIL.search(t[a:b])
    return (a, a + m.start()) if m else (a, b)


def occurrences_ci(t, s):
    """Case-insensitive, word-bounded occurrences.

    The caption writes 'JAGE RAM & ORS.' and the body writes 'Jage Ram', and
    only the body mentions are annotated. Matching case-sensitively would
    propagate nothing in exactly the documents where propagation is worth most.
    """
    low, ls = t.lower(), s.lower()
    out, i = [], low.find(ls)
    while i != -1:
        a, b = i, i + len(s)
        okl = not (s[0].isalnum() and a > 0 and t[a - 1].isalnum())
        okr = not (s[-1].isalnum() and b < len(t) and t[b].isalnum())
        if okl and okr:
            out.append((a, b))
        i = low.find(ls, i + 1)
    return out


def proposals(t, keep=None, min_prec=0.0, propagate=False):
    out = collections.defaultdict(list)
    for name, label, f in RULES:
        if keep is not None and name not in keep:
            continue
        if MEASURED.get(name, (1.0, 0.0))[0] < min_prec:
            continue
        for a, b in f(t):
            if label in ("APP", "RESP"):
                a, b = strip_party(t, a, b)
            while b > a and t[b - 1].isspace():
                b -= 1
            while a < b and t[a].isspace():
                a += 1
            if b > a:
                out[label].append((a, b))
    if propagate:
        # One trustworthy match gives the string; string matching gives the
        # rest of the document at 1.000 precision (phase2_gazetteer oracle).
        for label in PROPAGATE:
            seen = {t[a:b] for a, b in out.get(label, [])}
            for st in seen:
                if len(st) < 5:
                    continue
                for a, b in occurrences_ci(t, st):
                    out[label].append((a, b))
        for label in out:
            out[label] = sorted(set(out[label]))
    return out


def validate(text, gold, docs, labels):
    log("")
    log("  Each rule over all %d training documents. 'exact P' is the share of"
        % len(docs))
    log("  its matches that are a gold span of that label, byte for byte;")
    log("  'ovl P' allows any overlap. High ovl, low exact = right thing,")
    log("  wrong edges. Low both = firing on the wrong text.")
    log("")
    log("  %-38s %6s %7s %7s %7s %6s"
        % ("rule", "fires", "exact P", "ovl P", "recall", "gold"))
    gold_by = collections.defaultdict(lambda: collections.defaultdict(set))
    for d in docs:
        for s, e, l in gold[d]:
            gold_by[l][d].add((s, e))
    for name, label, f in RULES:
        n = hit = ovl = 0
        found = 0
        for d in docs:
            t = text[d]
            g = gold_by[label][d]
            got = set()
            for a, b in f(t):
                if label in ("APP", "RESP"):
                    a, b = strip_party(t, a, b)
                while b > a and t[b - 1].isspace():
                    b -= 1
                while a < b and t[a].isspace():
                    a += 1
                if b <= a:
                    continue
                n += 1
                got.add((a, b))
                if (a, b) in g:
                    hit += 1
                if any(a < y and b > x for x, y in g):
                    ovl += 1
            found += len(got & g)
        ng = sum(len(v) for v in gold_by[label].values())
        log("  %-38s %6d %7s %7s %7s %6d"
            % (name[:38], n,
               "%.3f" % (hit / n) if n else "  -  ",
               "%.3f" % (ovl / n) if n else "  -  ",
               "%.3f" % (found / ng) if ng else "  -  ", ng))

    # All rules together, as a model-free tagger, scored the official way.
    for mp, prop in ((0.0, True), (0.40, True), (0.45, True),
                     (0.55, True), (0.70, True)):
        pred = {}
        for d in docs:
            p = proposals(text[d], None, mp, prop)
            pred[d] = [(a, b, l) for l, v in p.items() for a, b in v]
        macro, overlap, per = score(pred, {d: gold[d] for d in docs}, labels)
        log("")
        log("  RULES ALONE  min_prec=%.2f propagate=%-5s  strict %.4f  "
            "overlap %.4f" % (mp, prop, macro, overlap))
        for l in sorted(labels, key=lambda x: -per[x][0]):
            log("    %-11s F1=%.3f P=%.3f R=%.3f"
                % (l, per[l][0], per[l][1], per[l][2]))
    log("")
    log("  (the ensemble scores 0.5200 on test; rules alone are not expected")
    log("   to beat it -- they are there to fix its edges and, where they are")
    log("   precise, to carry a string across the document it appears in)")


def apply_rules(text, gold, labels, args):
    src = "%s/submissions/prediction_%s.json" % (P2, args.apply)
    pred = {d: [(int(a), int(b), str(l)) for a, b, l in v]
            for d, v in json.load(open(src, encoding="utf-8")).items()}
    tt = {str(k): v for k, v in load_json("Test/fold3-text.json").items()}
    keep = set(args.only) if args.only else None
    rep = set(args.replace or ())
    modes = json.load(open(args.modes)) if args.modes else None
    n0 = sum(len(v) for v in pred.values())
    out = {}
    for d, spans in pred.items():
        p = proposals(tt[d], keep, args.min_prec, args.propagate)
        rs = [(a, b, l) for l, v in p.items() for a, b in v]
        if modes:
            # Exactly what --evaluate chose, per label.
            v = []
            for l in set([x[2] for x in spans] + [x[2] for x in rs]):
                ml = [x for x in spans if x[2] == l]
                rl = [x for x in rs if x[2] == l]
                mo = modes.get(l, "model")
                if mo == "model":
                    v += [tuple(x) for x in ml]
                elif mo.startswith("merge"):
                    v += merge_same(ml, l, int(mo[5:]))
                elif mo == "snap":
                    v += snap(ml, rl, False)
                elif mo == "snapadd":
                    v += snap(ml, rl, True)
                else:
                    v += [tuple(x) for x in rl]
            seen, fin = set(), []
            for x in v:
                k = (int(x[0]), int(x[1]), str(x[2]))
                if k not in seen:
                    seen.add(k)
                    fin.append(k)
            out[d] = fin
        elif rep:
            # For a label where the rules measurably beat the model, a UNION
            # is the wrong combination: it keeps the model spans that are
            # worse and lets them cost precision. Replace instead.
            spans = [x for x in spans if x[2] not in rep]
            keepr = [x for x in rs if x[2] in rep]
            rs = [x for x in rs if x[2] not in rep]
            out[d] = snap(spans, rs, args.add) + [tuple(x) for x in keepr]
            seen, fin = set(), []
            for x in out[d]:
                k = (int(x[0]), int(x[1]), str(x[2]))
                if k not in seen:
                    seen.add(k)
                    fin.append(k)
            out[d] = fin
        else:
            out[d] = snap(spans, rs, args.add)
    log("  %d -> %d spans (%s)"
        % (n0, sum(len(v) for v in out.values()),
           "snapping and adding" if args.add else "snapping only"))
    t = "%s_r%s%s%s%s" % (args.apply,
                        "M" if modes else
                        ("X" + "".join(x[0] for x in sorted(rep))) if rep else "",
                        "p" if args.propagate else "",
                        "add" if args.add else "",
                        ("%02d" % int(args.min_prec * 100))
                        if args.min_prec != 0.40 else "")
    pj = "%s/submissions/prediction_%s.json" % (P2, t)
    json.dump({d: [list(x) for x in v] for d, v in out.items()},
              open(pj, "w", encoding="utf-8"), ensure_ascii=False)
    zp = "%s/submissions/submission_%s.zip" % (P2, t)
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(pj, "prediction.json")
    log("  wrote %s" % zp)


def merge_same(spans, label, gap):
    """Join same-label spans separated by at most `gap` characters.

    The ensemble emits 2.03 PREC spans and 2.39 CASENO spans per gold entity
    while scoring 0.74 overlap, which is fragmentation: one entity covered by
    several predicted pieces, each counting as a false positive while the
    entity itself counts as a miss. Merging was tried once at gap=3 on the
    weakest single model and dropped; it was never swept, never tried on the
    ensemble, and never decided per label -- which it must be, because JUDGE
    fails in the opposite direction (one BENCH line, three judges, one span).
    """
    sel = sorted([x for x in spans if x[2] == label])
    out = []
    for a, b, l in sel:
        if out and a - out[-1][1] <= gap:
            out[-1] = (out[-1][0], max(out[-1][1], b), l)
        else:
            out.append((a, b, l))
    return [tuple(x) for x in out]


# ------------------------------------------------- offline whole pipeline --
def evaluate_pipeline(text, gold, split, labels, args):
    """Score model -> post-process -> rules on held-out CV predictions.

    Every combination up to now cost one leaderboard submission to test, which
    is how `--replace WIT R.COUNSEL A.COUNSEL CASENO` got sent: the single-model
    per-label numbers said the rules were better on all four, and it lost
    0.068, because the ensemble plus post-processing is far above the single
    model those numbers came from. With held-out predictions on disk the whole
    pipeline scores in seconds, and the replacement set is CHOSEN by
    measurement instead of argued from a proxy.
    """
    from phase2_gazetteer import self_consistent

    pred = {d: [(int(a), int(b), str(l)) for a, b, l in v]
            for d, v in json.load(open(args.evaluate, encoding="utf-8")).items()}
    truth = {d: gold[d] for d in pred if d in gold}
    pred = {d: v for d, v in pred.items() if d in truth}
    log("  %d documents, %d model spans"
        % (len(pred), sum(len(v) for v in pred.values())))

    def rep(name, p, quiet=False):
        m, o, per = score(p, truth, labels)
        if not quiet:
            worst = sorted(labels, key=lambda l: per[l][0])[:4]
            log("  %-34s strict %.4f  overlap %.4f   %s"
                % (name, m, o,
                   " ".join("%s=%.2f" % (l, per[l][0]) for l in worst)))
        return m, per

    base, per0 = rep("model raw", pred)

    sc = self_consistent(pred, text, 5, args.min_support, args.max_expand,
                         0, True, {"PREC"})
    after_pp, _ = rep("+ wordsnap + longest-wins", sc)

    rules = {d: proposals(text[d], None, args.min_prec, args.propagate)
             for d in pred}
    flat = {d: [(a, b, l) for l, v in r.items() for a, b in v]
            for d, r in rules.items()}
    uni = {d: snap(sc[d], flat[d], False) for d in sc}
    rep("+ rules (snap only)", uni)
    uni_add = {d: snap(sc[d], flat[d], True) for d in sc}
    cur, per_u = rep("+ rules (snap + add)", uni_add)

    # Per-label arbitration, and it is EXACT rather than greedy: macro-F1 is
    # the mean of the twelve per-label F1 scores, and a label's F1 depends
    # only on that label's spans, so optimising each independently optimises
    # the total. Four sources per label, where every experiment so far forced
    # one global choice across all twelve.
    #
    # Each label is also decided on each fold separately. A label whose two
    # folds disagree has been fitted to the split, and its vote is not worth
    # acting on -- that is how --replace CASENO DATE WIT reached the
    # leaderboard and cost 0.027.
    def build_mode(l, mode, d):
        ml = [x for x in sc[d] if x[2] == l]
        rl = [x for x in flat[d] if x[2] == l]
        if mode == "model":
            return [tuple(x) for x in ml]
        if mode.startswith("merge"):
            return merge_same(ml, l, int(mode[5:]))
        if mode == "snap":
            return snap(ml, rl, False)
        if mode == "snapadd":
            return snap(ml, rl, True)
        return [tuple(x) for x in rl]

    def label_f1(l, mode, docs):
        p1 = {d: build_mode(l, mode, d) for d in docs}
        t1 = {d: [x for x in truth[d] if x[2] == l] for d in docs}
        return score(p1, t1, [l])[0]

    MODES = ("model", "merge4", "merge12", "merge30",
             "snap", "snapadd", "rules")
    f1s = [d for d in pred if d in split["fold1"]]
    f2s = [d for d in pred if d in split["fold2"]]
    chosen_mode, unstable = {}, []
    log("")
    log("  per-label source (exact, not greedy -- macro-F1 decomposes):")
    log("  model_scale %.2f applied to the rules column before comparison"
        % args.model_scale)
    log("  %-11s %s   %-8s %s"
        % ("label", " ".join("%6s" % m for m in MODES), "pick", "folds"))
    for l in sorted(labels):
        sc_all = {m: label_f1(l, m, list(pred)) for m in MODES}
        # The model column is measured on a 35-document fold; the submitted
        # model is refit on all 70, and the measured CV->test ratio is 1.23.
        # Rules are fixed patterns and do not improve with data. Comparing the
        # two unscaled is why "replace" won on CV and lost on the leaderboard
        # three times (-0.068, -0.027, -0.036). Anything rule-only must beat
        # the model by that margin to be believed.
        adj = dict(sc_all)
        adj["rules"] = sc_all["rules"] / args.model_scale
        best = max(MODES, key=lambda m: adj[m])
        def fold_best(ds):
            v = {m: label_f1(l, m, ds) for m in MODES}
            v["rules"] /= args.model_scale
            return max(MODES, key=lambda m: v[m])

        b1 = fold_best(f1s) if f1s else best
        b2 = fold_best(f2s) if f2s else best
        agree = b1 == b2
        if not agree:
            unstable.append(l)
            best = "model" if "model" in (b1, b2) else best
        chosen_mode[l] = best
        log("  %-11s %s   %-8s %s"
            % (l, " ".join("%6.3f" % sc_all[m] for m in MODES), best,
               "ok" if agree else "DISAGREE %s/%s -> safe" % (b1, b2)))

    best_pred = {}
    for d in pred:
        v = []
        for l in labels:
            v += build_mode(l, chosen_mode[l], d)
        best_pred[d] = sorted(set(v))
    cur, _ = rep("per-label arbitrated", best_pred)
    if unstable:
        log("  fold-unstable, forced to the safe source: %s"
            % " ".join(unstable))

    mf = "%s/results/label_modes.json" % P2
    json.dump(chosen_mode, open(mf, "w"), indent=1)
    log("  wrote %s" % mf)
    best = best_pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--evaluate", default=None, metavar="CVPRED",
                    help="score the whole pipeline on results/cvpred_*.json")
    ap.add_argument("--model-scale", type=float, default=1.23, metavar="R",
                    help="CV->test ratio; the rules column is divided by it "
                         "before arbitration, because the CV model sees 35 "
                         "documents and the submitted model sees 70 while "
                         "rules do not improve with data (default 1.23, "
                         "1.0 disables)")
    ap.add_argument("--min-support", type=int, default=1)
    ap.add_argument("--max-expand", type=int, default=48)
    ap.add_argument("--apply", default=None, metavar="TAG")
    ap.add_argument("--add", action="store_true",
                    help="also emit rule matches the model never found")
    ap.add_argument("--only", nargs="*", default=None, metavar="RULE",
                    help="restrict to these rule names")
    ap.add_argument("--min-prec", type=float, default=0.40, metavar="P",
                    help="drop rules whose measured exact precision is below "
                         "this (default 0.6, which removes PREC v., "
                         "STAT <Name> Act, CASENO YYYY AIR N and the JUDGE "
                         "rules)")
    ap.add_argument("--modes", default=None, metavar="JSON",
                    help="results/label_modes.json written by --evaluate")
    ap.add_argument("--replace", nargs="*", default=None, metavar="LABEL",
                    help="for these labels discard the model entirely and "
                         "emit only rule matches")
    ap.add_argument("--propagate", action="store_true",
                    help="spread each matched string over its other "
                         "occurrences in the same document")
    a = ap.parse_args()

    with open(fetch("Train/labels.txt"), encoding="utf-8") as f:
        labels = [x.strip() for x in f if x.strip()]
    text, gold = {}, {}
    for fold in ("fold1", "fold2"):
        text.update({str(k): v
                     for k, v in load_json("Train/%s-text.json" % fold).items()})
        gold.update(parse_gold(load_json("Train/%s-gold.json" % fold)))
    split = json.load(open("%s/split.json" % P2))
    docs = split["fold1"] + split["fold2"]

    if a.validate:
        validate(text, gold, docs, labels)
    if a.evaluate:
        evaluate_pipeline(text, gold, split, labels, a)
    if a.apply:
        apply_rules(text, gold, labels, a)
    if not a.validate and not a.apply and not a.evaluate:
        ap.error("pass --validate, --evaluate CVPRED or --apply TAG")


if __name__ == "__main__":
    main()
