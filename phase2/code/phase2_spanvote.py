#!/usr/bin/env python3
"""CS-781 Phase 2 -- exact-span voting across finished submissions. CPU ONLY.

    python phase2_spanvote.py --tags A B C --k 2
    python phase2_spanvote.py --tags A B C --base A --k 2

Strict scoring makes exact-span agreement a meaningful vote: two systems
built on different weak data that emit the SAME (start, end, label) are far
more often right than either alone, and a span only one of them emits is where
their false positives live. Logit averaging is not available across these runs
because none of them saved emissions; their prediction files are on disk.

--k K            keep every span emitted by at least K of the systems
--base A --k K   keep A's spans that at least K-1 OTHER systems also emit, plus
                 any span at least K of the others agree on that A missed
"""

import argparse
import collections
import json
import os
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase2_data import P2, log  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True,
                    help="X for each submissions/prediction_X.json")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--base", default=None)
    ap.add_argument("--overlap", action="store_true",
                    help="with --base: another system SUPPORTS a base span if "
                         "it emits any overlapping span of the same label, "
                         "not only the identical one. Recovers entities all "
                         "systems found but bounded differently, keeping the "
                         "base system's extent")
    ap.add_argument("--no-add", action="store_true",
                    help="with --base: only drop unsupported base spans, "
                         "never add the other systems' spans")
    ap.add_argument("--add-k", type=int, default=0,
                    help="with --base: add spans the base lacks that at least "
                         "this many other systems emit exactly, replacing an "
                         "overlapping base span of the same label")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    assert not a.overlap or a.base, "--overlap needs --base"
    preds = {t: json.load(open("%s/submissions/prediction_%s.json" % (P2, t),
                               encoding="utf-8")) for t in a.tags}
    docs = sorted(set().union(*[set(p) for p in preds.values()]))
    out = {}
    for d in docs:
        cnt = collections.Counter()
        for t, p in preds.items():
            cnt.update({(int(x), int(y), str(l)) for x, y, l in p.get(d, [])})
        if a.base:
            base = {(int(x), int(y), str(l)) for x, y, l in preds[a.base].get(d, [])}
            if a.overlap:
                others = [{(int(x), int(y), str(l)) for x, y, l in p.get(d, [])}
                          for t, p in preds.items() if t != a.base]

                def sup(s):
                    return sum(any(o[2] == s[2] and o[0] < s[1] and o[1] > s[0]
                                   for o in os_) for os_ in others)
                keep = {s for s in base if sup(s) >= a.k - 1}
            else:
                keep = {s for s in base if cnt[s] - 1 >= a.k - 1}
            add = {s for s, n in cnt.items() if s not in base and n >= a.k}
            if a.no_add:
                add = set()          # precision trim only: never add spans
            if a.add_k:
                # Near-unanimous spans the base lacks. Exact agreement across
                # systems has been the most reliable boundary signal all day,
                # so where it contradicts the base's extent for the same
                # entity, the agreed extent replaces the base's.
                add = {s for s, n in cnt.items() if s not in base and n >= a.add_k}
                keep = {k for k in keep
                        if not any(s[2] == k[2] and s[0] < k[1] and s[1] > k[0]
                                   for s in add)}
            if a.overlap:
                # one extent per entity: the base system's wins
                add = {s for s in add
                       if not any(k[2] == s[2] and k[0] < s[1] and k[1] > s[0]
                                  for k in keep)}
            keep |= add
        else:
            keep = {s for s, n in cnt.items() if n >= a.k}
        out[d] = [list(s) for s in sorted(keep)]
    for t, p in preds.items():
        log("  %-40s %6.1f spans/doc" % (t, sum(len(v) for v in p.values()) / len(docs)))
    n = sum(len(v) for v in out.values())
    tag = a.out or "vote%d%s%s_%s" % (a.k, "b" if a.base else "",
                                      "o" if a.overlap else "",
                                    "-".join(x[:10] for x in a.tags))
    log("  -> %s  %.1f spans/doc" % (tag, n / len(docs)))
    pj = "%s/submissions/prediction_%s.json" % (P2, tag)
    json.dump(out, open(pj, "w", encoding="utf-8"), ensure_ascii=False)
    zp = "%s/submissions/submission_%s.zip" % (P2, tag)
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(pj, "prediction.json")
    log("  WROTE %s" % zp)


if __name__ == "__main__":
    main()
