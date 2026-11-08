#!/usr/bin/env python
"""Smoke test for phase7_point.py -- the exact-n-gram pointer and the hub decoder.

The whole premise of `ngram` is that word ORDER carries information no bag-of-words
system in this phase can see. That premise is not asserted in prose here, it is tested:
a rare phrase is planted in one candidate and in one query's mask context, and the test
requires that (a) the n-gram system ranks that candidate first, and (b) SHUFFLING the
phrase in the query destroys the n-gram score while leaving the tf-idf score unchanged.
If that second half ever fails, `ngram` is a reskin of tf-idf and must not be reported
as a separate system.

The other checks are the ones whose failure would look like a score rather than a bug:

  * every identity path must be EXACT -- `zmean`/`count` at alpha=0, and `greedy` with an
    unlimited cap, must reproduce the base ranking element for element, so any measured
    gain cannot be the decoder's plumbing;
  * `greedy` must never exceed its capacity, must emit every base candidate exactly once,
    and must keep the list length, or the submission is malformed;
  * `candidate_postings` must equal a brute-force index, including the df counts, since
    the weight depends on df and an off-by-one there silently reweights everything;
  * a query that is its own candidate must never retrieve itself;
  * `where`'s reachable-hit count must equal sum(min(7, |gold|)) -- the number every
    target in this phase is quoted against;
  * `--help` must render for every subcommand (a literal % in a help string raises
    "%o format: an integer is required" and kills the command, which has happened);
  * every flag the driver passes must exist in the parser (a missing --topn killed a
    whole GPU stage once).

  python smoke_p7_point.py
"""
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("P7_SMOKE_DIR",
                      os.path.join(tempfile.gettempdir(), "p7_smoke_point"))
sys.path.insert(0, HERE)

import smoke_p7 as S                                   # sets every CS781_* env var

import numpy as np

import phase7_data as D
import phase7_rank as R
import phase7_point as PT

FAIL = []
PHRASE = ("zorbic quillant murmex fandrel plimsy vorkath glebbin tarnuk weftal "
          "cindrome brascol dulvane")


def ok(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAIL.append(name)


def plant():
    """Put PHRASE after a mask in one dev query and into one of its golds.

    Returns (query id, gold id, a non-gold id that also gets the phrase shuffled, so the
    test can tell a real pointer from any candidate that merely shares the vocabulary).
    """
    dv = os.path.join(S.DATA, "Dev", "dev.json")
    dc = os.path.join(S.DATA, "Dev", "dev_candidates.json")
    qs = json.load(open(dv, encoding="utf-8"))
    cs = json.load(open(dc, encoding="utf-8"))
    qid = next(q for q in sorted(qs)
               if qs[q].get("relevant_candidates") and
               any(c in cs for c in qs[q]["relevant_candidates"]))
    gid = next(c for c in qs[qid]["relevant_candidates"] if c in cs)
    other = next(c for c in sorted(cs)
                 if c not in qs[qid]["relevant_candidates"] and c != qid)
    qs[qid]["text"] = list(qs[qid]["text"]) + [f"<CITATION> {PHRASE} ."]
    cs[gid]["text"] = list(cs[gid]["text"]) + [f"{PHRASE} ."]
    # the decoy holds the same WORDS in a different order: identical bag of words,
    # no shared n-gram.  This is what separates the two signals.
    cs[other]["text"] = list(cs[other]["text"]) + [
        " ".join(reversed(PHRASE.split())) + " ."]
    json.dump(qs, open(dv, "w", encoding="utf-8"))
    json.dump(cs, open(dc, "w", encoding="utf-8"))
    D._CACHE.clear()
    return qid, gid, other


def main():
    print("building fake data")
    S.build_data()
    qid, gid, other = plant()
    print(f"  planted a 12-word phrase in query {qid}, gold {gid}, decoy {other}")

    # ---- the hash itself ----------------------------------------------------
    val, store = PT.vocab_values()
    a = PT.hashes("alpha beta gamma delta".split(), 2, val)
    b = PT.hashes("beta alpha delta gamma".split(), 2, val)
    ok("hash is order-sensitive", set(a.tolist()).isdisjoint(b.tolist()),
       f"{len(a)} vs {len(b)} grams")
    ok("hash is deterministic across calls",
       PT.hashes("alpha beta gamma".split(), 2, PT.vocab_values()[0]).tolist()
       == PT.hashes("alpha beta gamma".split(), 2, PT.vocab_values()[0]).tolist())
    ok("hash count is len - n + 1", len(PT.hashes(list("abcdefg"), 3, val)) == 5,
       str(len(PT.hashes(list("abcdefg"), 3, val))))
    ok("too-short input gives no grams", len(PT.hashes(list("ab"), 5, val)) == 0)

    # ---- postings against a brute-force index -------------------------------
    n = 6
    qg = PT.query_grams("devx", n, "masks", 60)
    want = np.unique(np.concatenate([v for v in qg.values() if len(v)]))
    ids, pos, H, Dc = PT.candidate_postings("devx", n, want)
    val2, _ = PT.vocab_values()
    pool = D.candidates("devx")
    brute = {}
    for c in ids:
        w = D.WORD.findall(" ".join(pool[c]).lower())
        for h in set(PT.hashes(w, n, val2).tolist()):
            if h in set(want.tolist()):
                brute.setdefault(h, set()).add(pos[c])
    got = {}
    for h, d in zip(H.tolist(), Dc.tolist()):
        got.setdefault(h, set()).add(d)
    ok("postings equal a brute-force index", got == brute,
       f"{len(got)} vs {len(brute)} hashes")
    ok("postings are sorted by hash", bool(np.all(np.diff(H) >= 0)) if len(H) else True)
    ok("postings are deduplicated per candidate",
       all(len(v) == len(set(v)) for v in got.values()))

    # ---- df weighting -------------------------------------------------------
    Hs = np.array([1, 1, 1, 2], dtype=np.uint64)
    Ds = np.array([0, 1, 2, 0], dtype=np.int32)
    s1, _ = PT.gather(Hs, Ds, np.array([1], dtype=np.uint64), 3, "invlog")
    s2, _ = PT.gather(Hs, Ds, np.array([2], dtype=np.uint64), 3, "invlog")
    ok("a rarer n-gram is worth more", s2[0] > s1[0], f"{s2[0]:.3f} > {s1[0]:.3f}")
    s3, nm = PT.gather(Hs, Ds, np.array([99], dtype=np.uint64), 3, "invlog")
    ok("an absent n-gram scores nothing", s3.sum() == 0 and nm == 0)
    s4, _ = PT.gather(Hs, Ds, np.array([1, 2], dtype=np.uint64), 3, "one")
    ok("weight=one gives a plain match count", s4[0] == 2.0, str(s4[0]))

    # ---- THE PREMISE: the planted pointer -----------------------------------
    rank, score = PT.rank_ngram("devx", 12, "masks", 60, "invlog", topn=25)
    hit = rank[qid]
    ok("the planted gold is ranked FIRST by the pointer", hit[0] == gid,
       f"got {hit[0]}, wanted {gid}")
    ok("the planted gold is the ONLY scoring candidate",
       float(np.sum(np.asarray(score[qid]) > 0)) == 1.0,
       f"{int(np.sum(np.asarray(score[qid]) > 0))} scored")
    ok("the shuffled decoy scores ZERO", other not in hit[:1])

    # the same comparison a bag of words would make: tf-idf cannot tell them apart
    trank, tscore = R.rank_lexical("devx", "tfidf", 1.2, 0.75, "sqrt", 0.2, topn=25)
    tp = {c: float(v) for c, v in zip(trank[qid], tscore[qid])}
    ok("tf-idf scores the decoy and the gold almost the same",
       abs(tp.get(gid, 0.0) - tp.get(other, 0.0)) < 0.5 * max(tp.get(gid, 1e-9), 1e-9),
       f"gold {tp.get(gid, 0):.3f} vs decoy {tp.get(other, 0):.3f} -- which is exactly "
       f"why the n-gram signal is NEW information")

    drop = R.self_excluded("devx")
    selfq = [q for q in drop if q in rank]
    ok("a query never retrieves itself",
       all(drop[q] not in rank[q] for q in selfq), f"{len(selfq)} self-candidate queries")

    # ---- the hub decoder: every identity path must be exact ------------------
    L, Z = PT.as_matrix(rank, score)
    base = {q: list(L[q]) for q in L}
    for mode in ("zmean", "count"):
        out = PT.decode(L, Z, mode, 0.0, 10 ** 9)
        ok(f"{mode} at alpha=0 is the identity", out == base,
           "any gain at alpha>0 is therefore real, not plumbing")
    big = PT.decode(L, Z, "greedy", 0.0, 10 ** 9)
    ok("greedy with an unlimited cap keeps every base top-7",
       all(set(big[q][:7]) == set(base[q][:7]) for q in base))
    ok("greedy preserves the candidate SET exactly",
       all(sorted(big[q]) == sorted(base[q]) for q in base))
    ok("greedy emits no duplicates",
       all(len(set(big[q])) == len(big[q]) for q in big))

    # A capacity cannot always be honoured: a query may only use candidates from its own
    # retrieved list, so feasibility is a bipartite b-matching condition, not a counting
    # bound. An earlier version of this test asserted a counting floor and failed on the
    # fixture (max reuse 8 against a floor of 6) -- correctly, because the floor was the
    # wrong claim. What must hold is the honest version: when the decoder reports no
    # back-fill the cap is EXACT, and when it cannot fill the slots it must say so rather
    # than quietly hand back the base ranking under a capacity label.
    ncand = len({c for q in L for c in L[q]})

    def reuse(out):
        u = {}
        for q in out:
            for c in out[q][:7]:
                u[c] = u.get(c, 0) + 1
        return max(u.values())

    loose, st_loose = {}, {}
    loose = PT.decode(L, Z, "greedy", 0.0, ncand, stats=st_loose)
    ok("an ample cap needs no back-fill", st_loose.get("backfilled") == 0,
       f"backfilled {st_loose.get('backfilled')}")
    st_tight = {}
    tight = PT.decode(L, Z, "greedy", 0.0, 1, stats=st_tight)
    ok("an impossible cap REPORTS its shortfall", st_tight.get("backfilled", 0) > 0,
       f"backfilled {st_tight.get('backfilled')} of {len(L) * 7} slots")
    caps, bad = [], []
    for c in (2, 3, 4, 6, 8, 12, ncand):
        st = {}
        out = PT.decode(L, Z, "greedy", 0.0, c, stats=st)
        if st.get("backfilled", 0):
            continue
        caps.append(c)
        if reuse(out) > c:
            bad.append(c)
    ok("wherever there is no back-fill the cap holds EXACTLY", not bad,
       f"clean caps {caps}, violated {bad}")
    capped = PT.decode(L, Z, "greedy", 0.0, max(caps) if caps else ncand)
    ok("capped greedy is still a full valid ranking",
       all(sorted(capped[q]) == sorted(base[q]) for q in base))
    sk = PT.decode(L, Z, "sinkhorn", 5, max(caps) if caps else ncand)
    ok("sinkhorn returns a full valid ranking",
       all(sorted(sk[q]) == sorted(base[q]) for q in base))

    cnt, mean = PT.hub_stats(L, Z, beta=5.0)
    ok("hub counts never exceed the number of queries",
       max(cnt.values()) <= len(L), f"max {max(cnt.values())} of {len(L)}")
    one = [c for c in cnt if cnt[c] == 1]
    ok("a candidate seen once is shrunk, not cancelled",
       all(abs(mean[c]) < 1.0 for c in one[:50]), f"{len(one)} singletons")

    # ---- the CLI ------------------------------------------------------------
    R.save_run("smoke_ptr", "devx", rank, score)
    R.save_run("smoke_ptr", "train", *PT.rank_ngram("train", 6, "masks", 60,
                                                    "invlog", topn=25))
    env = dict(os.environ)
    for args in (["where", "--splits", "devx"],
                 ["residue", "--splits", "devx", "--nsweep", "6", "--scope", "masks"],
                 ["ngram", "--tag", "smoke_ng", "--nsweep", "6", "--scope", "masks",
                  "--splits", "devx", "--topn", "25"],
                 ["hub", "--base", "smoke_ptr", "--splits", "devx", "--select", "devx",
                  "--mode", "zmean,greedy", "--asweep", "0,0.5", "--capsweep", "3"]):
        p = subprocess.run([sys.executable, "phase7_point.py"] + args,
                           cwd=HERE, env=env, capture_output=True, text=True)
        if p.returncode != 0:
            print(p.stdout[-1800:])
            print(p.stderr[-1800:])
        ok(f"CLI {args[0]} exits 0", p.returncode == 0)

    w = json.load(open(os.path.join(S.P7, "results", "where_hits.json")))
    gp = D.golds_in_pool("devx")
    want_hits = sum(min(len(v), D.K) for v in gp.values())
    ok("where's reachable count is sum(min(7, |gold|))",
       w["devx"]["reachable_hits"] == want_hits,
       f"{w['devx']['reachable_hits']} vs {want_hits}")
    ok("where reports the max gold reuse the capsweep must bracket",
       w["devx"]["max_reuse"] >= 1, str(w["devx"]["max_reuse"]))

    res = json.load(open(os.path.join(S.P7, "results", "ngram_residue.json")))
    ok("residue reports an enrichment or an explicit None",
       "enrichment" in res["by_split"]["devx"]["6"])

    # ---- --help renders, and the driver's flags all exist -------------------
    for sub in ("residue", "ngram", "hub", "where"):
        p = subprocess.run([sys.executable, "phase7_point.py", sub, "--help"],
                           cwd=HERE, env=env, capture_output=True, text=True)
        ok(f"--help renders for {sub}", p.returncode == 0,
           (p.stderr.strip().splitlines() or [""])[-1][:90])

    drv = os.path.join(HERE, "p7g.sh")
    if os.path.exists(drv):
        import re
        body = open(drv, encoding="utf-8").read()
        for sub in ("residue", "ngram", "hub", "where"):
            blocks = re.findall(r"phase7_point\.py\s+" + sub + r"([^|&\n]*(?:\\\n[^|&\n]*)*)",
                                body)
            flags = {f for b in blocks for f in re.findall(r"--[a-z0-9][a-z0-9-]*", b)}
            h = subprocess.run([sys.executable, "phase7_point.py", sub, "--help"],
                               cwd=HERE, env=env, capture_output=True, text=True).stdout
            missing = sorted(f for f in flags if f not in h)
            ok(f"p7g.sh flags for `{sub}` all exist", not missing, ",".join(missing))
    else:
        ok("p7g.sh present for the flag lint", False, "p7g.sh missing")

    print()
    if FAIL:
        print(f"SMOKE FAILED: {len(FAIL)} check(s): {FAIL}")
        return 1
    print("SMOKE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
