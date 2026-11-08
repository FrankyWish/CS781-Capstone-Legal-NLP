#!/usr/bin/env python3
"""CS-781 Phase 2 -- confident pseudo-labels on the fold3 TEXT, as a weak cache.

    python phase2_pseudo.py --pred ext6e_ctxmass+1p0_pp --emis results/emis_ext6e

CPU, a few minutes. Writes results/pseudo_<pred>.pt, which phase2_weaksup.py
takes through --weak-cache alongside the external caches.

WHY

Every training-data addition so far came from a DIFFERENT distribution than
the test set: Supreme Court corpus labelled by rules, InLegalNER passages. The
CV->test ratio fell from 1.23 to 1.02 as they were added -- they substitute for
gold rather than teaching the 70-document model anything about fold3. The one
unlabelled text that is exactly in-distribution is the test documents
themselves. Self-training on them (Yarowsky 1995; McClosky et al. 2006 for
parsing) teaches the trunk fold3's own party names, counsel lines and citation
formats from the spans the system is already confident about, and the gold
fine-tuning pass afterwards keeps the final model fitted to real annotation.

WHAT IS AND IS NOT USED

Only the test TEXT, which every participant is given, and this system's own
predictions on it. No test label exists anywhere in this pipeline. It must be
described in the report as transductive self-training.

GATING

A predicted span becomes a pseudo-label only if the pooled ensemble's mean
P(not O) over its tokens is at least --min-conf; everything outside the kept
spans and a small --radius of O around them is -100, exactly the partial
supervision the rule-labelled corpus uses, so the model is never told that an
entity it missed is O.
"""

import argparse
import bisect
import glob
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase2_data import P1, P2, align, fetch, load_json, log  # noqa: E402
from phase2_ner import windows                                # noqa: E402
from phase2_weaksup import encode_weak                        # noqa: E402
from cs781_bpe import BPETokenizer                            # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True,
                    help="TAG of submissions/prediction_TAG.json -- the best "
                         "post-processed test prediction")
    ap.add_argument("--emis", default=None,
                    help="emission dir with test_*.pt; gates on P(not O)")
    ap.add_argument("--agree", default=None, metavar="TAG",
                    help="second system's prediction_TAG.json; keep only spans "
                         "BOTH systems emit exactly. Usable before any "
                         "emissions exist")
    ap.add_argument("--window", default="ctx", choices=["avg", "ctx"])
    ap.add_argument("--min-conf", type=float, default=0.6)
    ap.add_argument("--radius", type=int, default=8)
    ap.add_argument("--repeat", type=int, default=3,
                    help="copies of each window; the external caches are far "
                         "larger and would otherwise drown 35 documents")
    ap.add_argument("--ctx", type=int, default=1024)
    a = ap.parse_args()
    assert a.emis or a.agree, "pass --emis and/or --agree"

    with open(fetch("Train/labels.txt"), encoding="utf-8") as f:
        labels = [x.strip() for x in f if x.strip()]
    label_to_id = {l: i for i, l in enumerate(labels)}
    test_text = {str(k): v for k, v in load_json("Test/fold3-text.json").items()}

    def read(tag):
        return json.load(open("%s/submissions/prediction_%s.json" % (P2, tag),
                              encoding="utf-8"))
    pred = read(a.pred)
    if a.agree:
        # Two systems built on different weak data agreeing on the exact
        # character span and label is a far stronger signal than either one:
        # each scores ~0.56 strict, and their disagreements are where the
        # errors concentrate.
        other = read(a.agree)
        n0 = sum(len(v) for v in pred.values())
        pred = {d: [s for s in v
                    if tuple(s) in {tuple(x) for x in other.get(d, [])}]
                for d, v in pred.items()}
        log("  --agree %s: %d of %d spans emitted by both systems"
            % (a.agree, sum(len(v) for v in pred.values()), n0))
    pool = None
    if a.emis:
        emis = a.emis if os.path.isabs(a.emis) else os.path.join(P2, a.emis)
        files = sorted(glob.glob(os.path.join(emis, "test_*.pt")))
        assert files, "no test_*.pt in %s" % emis
        for f in files:
            em = {d: v.float() for d, v in
                  torch.load(f, weights_only=False)[a.window].items()}
            pool = em if pool is None else {k: pool[k] + em[k] for k in pool}
        pool = {k: torch.softmax(v / len(files), -1) for k, v in pool.items()}
        log("  %d members pooled from %s" % (len(files), emis))

    tok = BPETokenizer.load("%s/shared/tokenizer/bpe_legal_32k.json" % P1)
    data, kept, total = [], {l: 0 for l in labels}, 0
    for d, t in sorted(test_text.items()):
        ids, spans, _ = align(tok, t)
        if not ids:
            continue
        p_ent = 1 - pool[d][:, 0] if pool is not None else None
        starts = [s for s, _ in spans]
        weak = []
        for x, y, l in pred.get(d, []):
            total += 1
            i = max(0, bisect.bisect_right(starts, x) - 1)
            tk = []
            while i < len(spans) and spans[i][0] < y:
                if spans[i][1] > x:
                    tk.append(i)
                i += 1
            if tk and (p_ent is None
                       or float(p_ent[tk].mean()) >= a.min_conf):
                weak.append((x, y, l))
                kept[l] += 1
        tags = encode_weak(spans, weak, label_to_id, a.radius)
        for _, wi, wt, _ in windows(ids, tags, a.ctx, a.ctx // 2):
            if wt is not None and any(x > 0 for x in wt):
                data.extend([dict(ids=wi, tags=wt)] * a.repeat)
    n = sum(kept.values())
    log("  kept %d of %d candidate spans%s" % (
        n, total, " at P(not O) >= %.2f" % a.min_conf if pool is not None
        else ""))
    log("  per label: " + "  ".join("%s=%d" % kv for kv in kept.items()))
    log("  %d windows (x%d repeat)" % (len(data), a.repeat))
    out = "%s/results/pseudo_%s%s.pt" % (P2, a.pred,
                                         "_x_" + a.agree if a.agree else "")
    torch.save(data, out)
    log("  wrote %s" % out)


if __name__ == "__main__":
    main()
