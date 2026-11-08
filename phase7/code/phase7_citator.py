#!/usr/bin/env python
"""CS-781 Phase 7 -- the citation graph that the released texts still carry.

    python phase7_citator.py all

FOUND BY phase7_probe.py (7 Nov night).  Supreme Court judgments in IL-PCR keep their
judis header, which the masking did not touch:

    CITATION: 1964 AIR 1545 1964 SCR (3) 573
    CITATOR INFO : R 1970 SC 765 (5) RF 1954 SC 411 (4) E 1963 SC1417 (21) ...
    DATE OF JUDGMENT: 07/05/1963

CITATOR INFO lists the LATER judgments that cited this one, by AIR citation.  So if a
query q cites candidate c, c's CITATOR INFO should contain q's own AIR citation.  That is
the citation edge itself, in the input text -- no labels, no external data.

Rules (each measured for precision / recall on train and devx before use):
  A  q's AIR key  in  citator(c)            -> c is (very likely) a prior case of q
  B  AIR / SCR cites left unmasked in q's body  match  c's own header keys
  R  c's AIR key  in  citator(q)            -> c CITES q (reverse; measured, expect ~0)
  D  header dates: a candidate decided after q is demoted (only when both dates parse)

Final ranking = A matches, then B matches, then the base ranking (the best transductive
run, ot2_zst3_p on devx/test), optionally date-demoted.  kind = "hybrid".
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import phase7_data as D   # noqa: E402
import phase7_rank as R   # noqa: E402

P7 = D.P7
K = D.K
WORD = re.compile(r"[a-z0-9]+")

HDR_AIR = re.compile(r"CITATION\s*:?\s*(\d{4})\s+AIR\s+(\d{1,4})\b", re.I)
HDR_SCR = re.compile(r"(\d{4})\s+SCR\s*(?:\(\s*(\d)\s*\))?\s*(\d{1,4})\b", re.I)
CIT_HEAD = re.compile(r"CITATOR\s+INFO\s*:?", re.I)
CIT_END = re.compile(r"\bACT\s*:|\bHEADNOTE\s*:|\bJUDGMENT\s*:", re.I)
CIT_KEY = re.compile(r"\b(\d{4})\s*SC\s*(\d{1,4})\b")          # "1963 SC1417", "1955 SC 610"
BODY_AIR = re.compile(r"\bA\.?\s*I\.?\s*R\.?\s*\(?(\d{4})\)?\s*S\.?\s*C\.?\s*(\d{1,4})\b", re.I)
BODY_AIR2 = re.compile(r"\b(\d{4})\s+AIR\s+(\d{1,4})\b", re.I)
BODY_SCR = re.compile(r"[\[(](\d{4})[\])]\s*(?:\(?(\d)\)?\s*)?S\.?\s*C\.?\s*R\.?\s*(\d{1,4})\b", re.I)
YEAR_ANY = re.compile(r"\b(19[4-9]\d|20[0-2]\d)\b")
DATE = re.compile(r"DATE\s+OF\s+JUDGMENT\s*:?\s*(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})", re.I)


def log(*a):
    print(*a, flush=True)


NK_AIR1 = re.compile(r"\ba ?i ?r (\d{4}) (?:s ?c|supreme court) (\d{1,4})\b")
NK_AIR2 = re.compile(r"\b(\d{4}) air (\d{1,4})\b")
NK_REP = re.compile(r"\b(\d{4}) (?:(\d{1,2}) )?(s ?c ?r|s ?c ?c) (?:(\d{1,2}) )?(\d{1,4})\b")
HDR_SEG = re.compile(r"\bcitation (.{0,400}?)(?:\bcitator info\b|\bact\b|\bheadnote\b|$)")


def norm(text):
    return " ".join(WORD.findall(text.lower()))


def nkeys(s):
    """Reporter citations in normalised text -> {("AIR"|"SCR"|"SCC", year, page)}."""
    out = {("AIR", int(y), int(pg)) for y, pg in NK_AIR1.findall(s)}
    out |= {("AIR", int(y), int(pg)) for y, pg in NK_AIR2.findall(s)}
    for y, _v1, rep, _v2, pg in NK_REP.findall(s):
        out.add((rep.replace(" ", "").upper(), int(y), int(pg)))
    return out


def parse(text):
    head = text[:8000]
    m = HDR_AIR.search(head)
    air = (int(m.group(1)), int(m.group(2))) if m else None
    cit = set()
    ch = CIT_HEAD.search(head)
    if ch:
        seg = head[ch.end():ch.end() + 6000]
        e = CIT_END.search(seg)
        seg = seg[:e.start()] if e else seg
        cit = {(int(y), int(pg)) for y, pg in CIT_KEY.findall(seg)}
    d = DATE.search(head)
    date = (int(d.group(3)), int(d.group(2)), int(d.group(1))) if d else None
    hs = HDR_SEG.search(norm(head[:3000]))
    hk = nkeys(hs.group(1)) if hs else set()
    if not air:
        ak = sorted(k for k in hk if k[0] == "AIR")
        air = (ak[0][1], ak[0][2]) if ak else None
    if air:
        hk.add(("AIR",) + air)
    bk = nkeys(norm(text[3000:])) - hk
    ys = [int(y) for y in YEAR_ANY.findall(text) if 1940 <= int(y) <= 2023]
    return dict(air=air, cit=cit, date=date, hk=hk, bk=bk, yest=max(ys) if ys else None)


def build(split):
    qs = D.queries(split)
    pool = D.candidates(split)
    Pq = {r["id"]: parse(" ".join(r["sents"])) for r in qs}
    Pc = {c: parse(" ".join(v)) for c, v in pool.items()}
    by_cit, by_key = {}, {}
    for c, p in Pc.items():
        for k in p["cit"]:
            by_cit.setdefault(k, set()).add(c)
        for k in p["hk"]:
            by_key.setdefault(k, set()).add(c)
    rec = {}
    rf = f"{P7}/results/overlap/recovered_{split}.json"
    if os.path.exists(rf):
        with open(rf) as f:
            rec = json.load(f)
    A, B, L, Rv = {}, {}, {}, {}
    lk_only = {}
    nk_l = 0
    for r in qs:
        q, p = r["id"], Pq[r["id"]]
        A[q] = set(by_cit.get(p["air"], ())) - {q} if p["air"] else set()
        B[q] = set().union(*[by_key.get(k, set()) for k in p["bk"]]) - {q} if p["bk"] else set()
        lk = nkeys(norm(" ".join(rec.get(q, [])))) - p["hk"]
        nk_l += len(lk)
        L[q] = set().union(*[by_key.get(k, set()) for k in lk]) - {q} if lk else set()
        lk_only[q] = set(L[q])
        Rv[q] = {c for c, pc in Pc.items() if pc["air"] and pc["air"] in p["cit"]} - {q}
    nf = f"{P7}/results/names/N_{split}.json"
    if os.path.exists(nf):                     # rule N (phase7_names.py) joins the lookup set
        with open(nf) as f:
            Nn = json.load(f)
        add = 0
        for q in L:
            extra = set(Nn.get(q, [])) & set(pool) - {q}
            add += len(extra - L[q])
            L[q] |= extra
        log(f"  [{split}] rule N (corpus title match) added {add} pairs to the lookup set")
        hf = f"{P7}/results/names/Nhi_{split}.json"
        if os.path.exists(hf):                 # loose band = N pairs below the strict cut
            with open(hf) as f:
                Nh = json.load(f)
            nb = 0
            for q in L:
                band = set(Nn.get(q, [])) - set(Nh.get(q, [])) - lk_only[q]
                BAND[q] = band
                nb += len(band)
            log(f"  [{split}] loose N band (below strict cut): {nb} pairs")
    mf = f"{P7}/results/names/M_{split}.json"
    if os.path.exists(mf):                     # rule M: names in the released query text
        with open(mf) as f:
            Mm = json.load(f)
        add = 0
        for q in L:
            extra = set(Mm.get(q, [])) & set(pool) - {q}
            add += len(extra - L[q])
            L[q] |= extra
        log(f"  [{split}] rule M (names in query text) added {add} pairs to the lookup set")
    if rec:
        ex = [sp for v in list(rec.values())[:40] for sp in v if any(ch.isdigit() for ch in sp)][:12]
        log(f"  [{split}] recovered spans for {len(rec)} queries, {nk_l} reporter keys; "
            f"digit spans e.g. {ex}")
    return qs, pool, Pq, Pc, A, B, L, Rv


def measure(split, qs, Pq, Pc, A, B, L, Rv):
    g = D.golds_in_pool(split)
    pairs = sum(len(v) for v in g.values())
    log(f"  [{split}] queries with AIR key {sum(1 for p in Pq.values() if p['air'])}/{len(Pq)}; "
        f"candidates with CITATOR INFO {sum(1 for p in Pc.values() if p['cit'])}/{len(Pc)}; "
        f"with dates q {sum(1 for p in Pq.values() if p['date'])} c "
        f"{sum(1 for p in Pc.values() if p['date'])}")
    for nm, M in (("A q-key in cit(c)", A), ("B body cites", B), ("L corpus lookup", L),
                  ("R reverse", Rv), ("A|B|L union", {q: A[q] | B[q] | L[q] for q in A})):
        tp = sum(len(M[q] & set(g[q])) for q in g)
        n = sum(len(M[q]) for q in g)
        cov = sum(1 for q in g if M[q])
        log(f"  [{split}] rule {nm:18s} predicted {n:5d} pairs, correct {tp:5d} "
            f"(precision {tp / max(n, 1):.3f}, recall {tp / max(pairs, 1):.3f}), queries {cov}")
    ok = tot = 0
    for q in g:
        dq = Pq[q]["date"]
        for c in g[q]:
            dc = Pc[c]["date"]
            if dq and dc:
                tot += 1
                ok += dc <= dq
    log(f"  [{split}] header dates: gold decided on/before the query {ok}/{tot}")


def combine(qs, pool, Pq, Pc, A, B, L, base, use_a, use_b, use_d, use_l=0, use_x=0, keep=60,
            use_v=0, use_t=0, tier_k=3, use_g=0):
    out = {}
    order_all = list(pool)
    claim = {}
    if use_x:                       # candidates a precise rule already gave to ANOTHER query
        for r in qs:
            for c in (A[r["id"]] | B[r["id"]] | (L[r["id"]] if use_l else set())):
                claim.setdefault(c, set()).add(r["id"])
    for r in qs:
        q = r["id"]
        bl = [str(c) for c in (base.get(q) or [])]
        pos = {c: i for i, c in enumerate(bl)}

        def key(c):
            return (pos.get(c, 10 ** 6), c)    # deterministic ties (set order varied by run)
        ranked = []
        if use_v:                   # vote: a candidate several rules agree on goes first
            rs = (A[q] if use_a else set()) | (B[q] if use_b else set()) | (L[q] if use_l else set())

            def vote(c):
                return -(2 * (use_a and c in A[q]) + 2 * (use_b and c in B[q])
                         + (use_l and c in L[q]))
            ranked += sorted(rs, key=lambda c: (vote(c), key(c)))
        if use_a and not use_v:
            ranked += sorted(A[q], key=key)
        if use_b and not use_v:
            ranked += sorted(B[q] - set(ranked), key=key)
        if use_l and not use_v:
            ranked += sorted(L[q] - set(ranked), key=key)
        rest = [c for c in bl if c not in set(ranked) and c != q]
        if use_d:
            dq = Pq[q]["date"]
            if dq:
                late = [c for c in rest if Pc.get(c, {}).get("date") and Pc[c]["date"] > dq]
                ls = set(late)
                rest = [c for c in rest if c not in ls] + late
        if use_x:
            taken = [c for c in rest if claim.get(c, set()) - {q}]
            ts = set(taken)
            rest = [c for c in rest if c not in ts] + taken
        if use_t and use_l and BAND.get(q):    # loose band: after the top-k base candidates
            ab = (A[q] if use_a else set()) | (B[q] if use_b else set())
            bd = [c for c in sorted(BAND[q], key=key) if c not in ab and c != q]
            bs = set(bd)
            ranked = [c for c in ranked if c not in bs]
            rest = [c for c in rest if c not in bs]
            rest = rest[:tier_k] + bd + rest[tier_k:]
        if use_g and GROUPS.get(q):            # citator group: after the top-k base candidates
            tier_k = G_TIER
            top = rest[:tier_k]                # members already in the base top-k stay there
            gd = [c for c in sorted(GROUPS[q], key=key)
                  if c not in set(ranked) and c not in set(top) and c != q]
            gs = set(gd)
            rest = top + gd + [c for c in rest[tier_k:] if c not in gs]
        ranked += rest
        if len(ranked) < K:
            ranked += [c for c in order_all if c not in set(ranked) and c != q][:K]
        out[q] = ranked[:keep]
    return out


VARIANTS = {          # (A, B, D, L)
    "a": (1, 0, 0, 0), "ab": (1, 1, 0, 0), "ad": (1, 0, 1, 0), "abd": (1, 1, 1, 0),
    "d": (0, 0, 1, 0), "al": (1, 0, 0, 1), "abl": (1, 1, 0, 1), "abdl": (1, 1, 1, 1),
    "abdx": (1, 1, 1, 0), "abdlx": (1, 1, 1, 1),        # x = exclusivity demotion
    "abdlv": (1, 1, 1, 1), "ablv": (1, 1, 0, 1),        # v = rules vote on the order
    "abdlt": (1, 1, 1, 1), "abdlvt": (1, 1, 1, 1),      # t = loose N band after base top 3
    "abdlg": (1, 1, 1, 1), "abdlvg": (1, 1, 1, 1),      # g = citator groups after base top 3
}


def base_for(split, tags):
    for t in tags:
        rk, _ = R.load_run(t, split)
        if rk is not None:
            return t, rk
    return None, None


def infer_keys(qs, Pq, Pc, B, L, base, cfg, simulate=False, topk=40, est=False):
    """For SC queries whose own AIR key is unknown, find the AIR key their likely golds
    all list in CITATOR INFO (year = the query's decision year or the next).
    simulate=True runs it on queries whose key IS known, hiding it, to measure precision."""
    min_sup, margin = cfg
    out = {}
    for r in qs:
        q, p = r["id"], Pq[r["id"]]
        if bool(p["air"]) != simulate:
            continue
        if est:     # headerless queries: year estimated from the text, base-only evidence
            if (p["date"] and not simulate) or not p["yest"]:
                continue
            years = {p["yest"], p["yest"] + 1}
        else:
            if not p["date"]:
                continue
            years = {p["date"][0], p["date"][0] + 1}
        ev = collections.Counter()
        for c in (set() if est else (B[q] | L[q])):
            ev[c] += 3.0
        for i, c in enumerate((base.get(q) or [])[:topk]):
            ev[str(c)] += 1.0 / (1.0 + i / 5.0)
        score, sup = collections.Counter(), collections.Counter()
        for c, w in ev.items():
            pc = Pc.get(c)
            if not pc:
                continue
            for k in pc["cit"]:
                if k[0] in years:
                    score[k] += w
                    sup[k] += 1
        ranked = [k for k, _ in score.most_common() if sup[k] >= min_sup]
        if not ranked:
            continue
        k1 = ranked[0]
        s2 = score[ranked[1]] if len(ranked) > 1 else 0.0
        if score[k1] >= margin * s2:
            out[q] = k1
    return out


EST_INFER = False
BAND = {}                   # qid -> N pairs between the loose and the strict cut
GROUPS = {}                 # qid -> citator group assigned by phase7_group.py
G_TIER = 3                  # groups enter after this many base candidates
INFER_GRID = [(s, m) for s in (2, 3) for m in (1.0, 1.5, 2.0, 3.0)]


def infer_all(built, bases):
    res = infer_mode(built, bases, est=False)
    if not EST_INFER:      # measured 8 Nov: hidden-key precision 0.24-0.29 -> off by default
        return res
    log("  -- estimated-year inference (queries with no header date; base-only evidence)")
    res2 = infer_mode(built, bases, est=True)
    for sp, d in res2.items():
        for q, v in d.items():
            res.setdefault(sp, {}).setdefault(q, set())
            res[sp][q] = res[sp][q] | v
    return res


def infer_mode(built, bases, est):
    """Choose the inference config on HIDDEN known keys (train + devx), then apply it to
    queries without a key.  Returns {split: {qid: set(cands)}} or {} if it is not precise."""
    best = None
    for cfg in INFER_GRID:
        ok = tot = 0
        for sp in ("train", "devx"):
            qs, pool, Pq, Pc, A, B, L, Rv = built[sp]
            _, base = base_for(sp, bases[sp])
            if base is None:
                continue
            got = infer_keys(qs, Pq, Pc, B, L, base, cfg, simulate=True, est=est)
            tot += len(got)
            ok += sum(1 for q, k in got.items() if k == Pq[q]["air"])
        prec = ok / max(tot, 1)
        log(f"  infer cfg sup>={cfg[0]} margin {cfg[1]}: hidden keys recovered {ok}/{tot} "
            f"(precision {prec:.3f})")
        if prec >= 0.80 and (best is None or ok > best[1]):
            best = (cfg, ok, prec)
    if best is None:
        log("  key inference not precise enough -- not used")
        return {}
    cfg = best[0]
    log(f"  CHOSE infer cfg {cfg}")
    res = {}
    for sp in ("train", "devx", "test"):
        qs, pool, Pq, Pc, A, B, L, Rv = built[sp]
        _, base = base_for(sp, bases[sp])
        if base is None:
            continue
        by_cit = {}
        for c, p in Pc.items():
            for k in p["cit"]:
                by_cit.setdefault(k, set()).add(c)
        got = infer_keys(qs, Pq, Pc, B, L, base, cfg, simulate=False, est=est)
        res[sp] = {q: by_cit.get(k, set()) - {q} for q, k in got.items()}
        if sp != "test":
            g = D.golds_in_pool(sp)
            n = sum(len(v) for v in res[sp].values())
            tp = sum(len(v & set(g.get(q, []))) for q, v in res[sp].items())
            log(f"  [{sp}] inferred keys for {len(got)} queries -> {n} pairs, {tp} correct "
                f"(precision {tp / max(n, 1):.3f})")
            if est and sp == "train" and n and tp / n < 0.6:
                log("  train precision of estimated-year pairs < 0.6 -- NOT used")
                return {}
            if sp == "devx" and n and tp / n < 0.7:
                log("  devx precision of inferred pairs < 0.7 -- inference NOT used")
                return {}
        else:
            log(f"  [test] inferred keys for {len(got)} queries -> "
                f"{sum(len(v) for v in res[sp].values())} pairs")
    return res


def cmd_all(a):
    res = {}
    built = {}
    for split in ("train", "devx", "test"):
        built[split] = build(split)
        qs, pool, Pq, Pc, A, B, L, Rv = built[split]
        if split != "test":
            measure(split, qs, Pq, Pc, A, B, L, Rv)
        else:
            log(f"  [test] queries with AIR key {sum(1 for p in Pq.values() if p['air'])}/{len(Pq)}; "
                f"candidates with CITATOR {sum(1 for p in Pc.values() if p['cit'])}/{len(Pc)}; "
                f"rule-A pairs {sum(len(v) for v in A.values())} over "
                f"{sum(1 for v in A.values() if v)} queries")
    bases = {"train": a.train_bases.split(","), "devx": a.bases.split(","),
             "test": a.bases.split(",")}
    if not a.no_infer:
        inf = infer_all(built, bases)
        for sp, extra in inf.items():
            A = built[sp][4]
            for q, v in extra.items():
                A[q] = A[q] | v                    # inferred-key matches join rule A
    global G_TIER
    G_TIER = a.group_tier
    want_g = any("g" in v for v in (a.variants.split(",") if a.variants else VARIANTS))
    if want_g:
        import phase7_group as G
        force = tuple(float(x) for x in a.group_cfg.split(",")) if a.group_cfg else None
        if force:
            force = (int(force[0]), force[1])
        GROUPS.update(G.assign_all(built, bases, base_for, D.golds_in_pool, log,
                                   min_prec=a.group_min_prec, force=force))
    rows = []
    want = set(a.variants.split(",")) if a.variants else set(VARIANTS)
    for vn, (ua, ub, ud, ul) in VARIANTS.items():
        if vn not in want:
            continue
        row = {"variant": vn}
        for split in ("train", "devx"):
            t, base = base_for(split, bases[split])
            if base is None:
                continue
            qs, pool, Pq, Pc, A, B, L, Rv = built[split]
            s0 = R.evaluate(base, split)
            s = R.evaluate(combine(qs, pool, Pq, Pc, A, B, L, base, ua, ub, ud, ul, "x" in vn, use_v="v" in vn, use_t="t" in vn, use_g="g" in vn), split)
            row[split] = s["f1"]
            row[split + "_hits"] = s["hits"]
            row[split + "_base"] = (t, s0["f1"], s0["hits"])
        rows.append(row)
        log(f"  variant {vn:4s} " + "  ".join(
            f"{sp} {row.get(sp, float('nan')):.4f} ({row.get(sp + '_hits')} hits; base "
            f"{row[sp + '_base'][0]} {row[sp + '_base'][1]:.4f})" for sp in ("train", "devx")
            if sp in row))
    rows.sort(key=lambda r: (r.get("devx", 0), r.get("train", 0)), reverse=True)
    try:                                        # where the remaining devx misses are
        vb = rows[0]["variant"]
        ua, ub, ud, ul = VARIANTS[vb]
        qs, pool, Pq, Pc, A, B, L, Rv = built["devx"]
        _, bb = base_for("devx", bases["devx"])
        rk = combine(qs, pool, Pq, Pc, A, B, L, bb, ua, ub, ud, ul, "x" in vb, use_v="v" in vb, use_t="t" in vb, use_g="g" in vb)
        g = D.golds_in_pool("devx")
        cat = collections.Counter()
        wrong_rule = 0
        for q, gs in g.items():
            top = rk[q][:K]
            rules = A[q] | B[q] | (L[q] if ul else set())
            wrong_rule += sum(1 for c in top if c in rules and c not in gs)
            bl = [str(c) for c in (bb.get(q) or [])]
            for c in gs:
                if c in top:
                    continue
                if c in rules:
                    cat["rule-matched, squeezed out of top 7"] += 1
                elif c in bl[:20]:
                    cat["base rank <= 20"] += 1
                elif c in bl:
                    cat["base rank 21-%d" % len(bl)] += 1
                else:
                    cat["never retrieved"] += 1
            if len(gs) > K:
                cat["(gold beyond 7 slots, unreachable)"] += len(gs) - K
        log(f"  MISS BREAKDOWN devx ({vb}): {dict(cat)}; wrong rule matches inside top 7: {wrong_rule}")
    except Exception as e:                      # noqa: BLE001
        log(f"  miss breakdown failed: {e}")
    lines = ["# Citator-graph submissions -- upload order", "",
             "Variants with l use the Phase-1 corpus lookup (kind=lookup); the rest are hybrid.", "",
             "| # | zip | devx F1 | devx hits | train F1 |", "|---|---|---|---|---|"]
    t, tb = base_for("test", bases["test"])
    pick = rows[:a.nzip]
    nolk = [r for r in rows if not VARIANTS[r["variant"]][3]]
    if nolk and nolk[0] not in pick:
        pick.append(nolk[0])                     # always ship the best non-lookup variant
    for i, row in enumerate(pick, 1):
        ua, ub, ud, ul = VARIANTS[row["variant"]]
        qs, pool, Pq, Pc, A, B, L, Rv = built["test"]
        rk = combine(qs, pool, Pq, Pc, A, B, L, tb, ua, ub, ud, ul, "x" in row["variant"], use_v="v" in row["variant"], use_t="t" in row["variant"], use_g="g" in row["variant"])
        tag = f"cit_{row['variant']}_{t}{a.suffix}"
        R.save_run(tag, "test", rk)
        for sp in ("devx",):
            _, bb = base_for(sp, bases[sp])
            q2, p2, Pq2, Pc2, A2, B2, L2, _ = built[sp]
            R.save_run(tag, sp, combine(q2, p2, Pq2, Pc2, A2, B2, L2, bb, ua, ub, ud, ul, "x" in row["variant"], use_v="v" in row["variant"], use_t="t" in row["variant"], use_g="g" in row["variant"]))
        R.save_run(tag, "meta", None, meta={
            "tag": tag, "kind": "lookup" if ul else "hybrid", "base": t, "variant": row["variant"],
            "technique": "citation graph from judis CITATOR INFO headers + base ranking",
            "devx_scores": {"f1": row.get("devx"), "hits": row.get("devx_hits")}})
        R.cmd_zip(argparse.Namespace(tag=tag, kind="lookup" if ul else "hybrid", note=row["variant"]))
        lines.append(f"| {i} | submission_{tag}.zip | {row.get('devx', 0):.4f} | "
                     f"{row.get('devx_hits')} | {row.get('train', float('nan')):.4f} |")
    if a.suffix:
        lines.insert(1, f"Run suffix `{a.suffix}`.")
    txt = "\n".join(lines) + "\n"
    with open(f"{P7}/SUBMIT_ORDER_CIT{a.suffix}.md", "w") as f:
        f.write(txt)
    log("\n" + txt)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("all")
    s.add_argument("--bases", default="ot2_zst3_p,ot2_zst5_p,ot_zmix_trdv,stack_new")
    s.add_argument("--train-bases", default="ot2_zlex2_p,ot_zmix_tr,citep")
    s.add_argument("--nzip", type=int, default=5)
    s.add_argument("--no-infer", action="store_true")
    s.add_argument("--suffix", default="", help="appended to zip tags, e.g. _n90")
    s.add_argument("--group-cfg", default="", help="force group cfg 'min_size,margin'")
    s.add_argument("--group-min-prec", type=float, default=0.45)
    s.add_argument("--group-tier", type=int, default=3, help="groups enter after k base cands")
    s.add_argument("--variants", default="", help="comma list to evaluate (default all)")
    a = p.parse_args()
    cmd_all(a)


if __name__ == "__main__":
    main()
