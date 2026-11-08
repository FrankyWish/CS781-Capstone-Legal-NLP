#!/usr/bin/env python3
"""CS-781 Phase 4 -- silver-labelled training data from the Phase-1 Supreme Court corpus. CPU.

    python phase4_external.py                      # -> phase4/external/sc_silver.jsonl + stats
    python phase4_external.py --corpus-path X.json # local file instead of the HF download (smoke test)

IDEA. A full Supreme Court judgment ends with its order: "the appeal is
allowed", "the appeals are dismissed". ILDC, which CJPE is built from, was
labelled exactly that way and then had the order removed. This script does the
same to the Phase-1 corpus (permitted data; the professor allows external data
if the report describes it):

  label  1 if the decision region (last 20% of the text, at least 3000 chars)
         contains an "allowed / partly allowed / we allow" phrase, else 0 if it
         contains a "dismissed / fails / we dismiss" phrase, else skipped.
         (ILDC's rule for multi-appeal cases: 1 if any appeal is allowed.)
  cut    the text is cut at the start of the sentence holding the FIRST outcome
         phrase in the decision region, so the model never sees the order.

EXCLUSIONS -- what makes this legitimate rather than a leak:
  * any corpus judgment matching a TEST or DEV document (>= 2 of its 12-word
    needles) is dropped and never written. Its order would be the answer.
  * judgments matching a TRAIN document are not written either (they are
    duplicates of gold data) but are used to MEASURE the labeller: agreement of
    the silver label with the gold label, and how close the cut lands to where
    the organisers' text ends. Those two numbers go in the report.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from phase4_data import P4, load_split, log, atomic_json            # noqa: E402
from phase4_overlap import stream_records, needles, WORD, N           # noqa: E402

SUBJ = r"(?:the\s+|these\s+|this\s+|both\s+(?:the\s+)?|all\s+(?:the\s+)?)?" \
       r"(?:civil\s+|criminal\s+|special\s+leave\s+|writ\s+|review\s+|transfer\s+)?" \
       r"(?:appeals?|petitions?)"
ADV = r"(?:\s+(?:are|is|stands?|shall\s+stand|must\s+be|deserves?\s+to\s+be|ought\s+to\s+be|should\s+be))" \
      r"(?:\s+(?:hereby|accordingly|therefore|consequently|thus))?(?:\s+(?:partly|partially))?"
ALLOW = re.compile(
    rf"\b{SUBJ}{ADV}\s+(?:allowed|accepted)\b|\b{SUBJ}\s+succeeds?\b|"
    r"\bwe\s+(?:accordingly\s+|therefore\s+|would\s+|hereby\s+)?(?:partly\s+)?allow\s+(?:the|this|these|both|all)\b|"
    r"\b(?:allowed|accepted)\s+in\s+part\b|\bpartly\s+allowed\b|\brule\s+(?:is\s+)?made\s+absolute\b", re.I)
DISMISS = re.compile(
    rf"\b{SUBJ}{ADV}\s+(?:dismissed|rejected)\b|\b{SUBJ}\s+(?:fails?|is\s+devoid\s+of\s+merit)\b|"
    r"\bwe\s+(?:accordingly\s+|therefore\s+|would\s+|hereby\s+)?dismiss\s+(?:the|this|these|both|all)\b|"
    r"\brule\s+(?:is\s+)?discharged\b", re.I)


def label_and_cut(text, frac=0.2, min_tail=3000):
    """-> (label, cut_index) or (None, reason)."""
    start = max(0, min(len(text) - min_tail, int(len(text) * (1 - frac))))
    region = text[start:]
    al = [m.start() for m in ALLOW.finditer(region)]
    di = [m.start() for m in DISMISS.finditer(region)]
    if not al and not di:
        return None, "no_outcome_phrase"
    lab = 1 if al else 0
    first = start + min(al + di)
    # back up to the start of that sentence / paragraph
    cut = max(text.rfind(". ", 0, first), text.rfind("\n", 0, first))
    cut = cut + 1 if cut >= 0 else first
    return lab, cut


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=["train_supreme_court"])
    ap.add_argument("--corpus-path", default=None)
    ap.add_argument("--min-words", type=int, default=300)
    ap.add_argument("--out", default=f"{P4}/external/sc_silver.jsonl")
    a = ap.parse_args()
    from huggingface_hub import hf_hub_download

    docs = {}
    for split in ("train", "dev", "test"):
        for d, t, l in load_split(split):
            docs[(split, d)] = (t, l)
    index = collections.defaultdict(list)
    first = set()
    for key, (t, _) in docs.items():
        for s, j in needles(t, k=10 if key[0] != "train" else 4):
            index[hash(s)].append((key, j))
            first.add(s.split(" ", 1)[0])
    log(f"  {len(docs)} CJPE docs indexed ({len(index)} needles)")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    tmp = a.out + ".tmp"
    out = open(tmp, "w", encoding="utf-8")
    st = collections.Counter()
    val_agree, val_n, cut_ratio, lab_count = 0, 0, [], collections.Counter()
    for src in a.sources:
        path = a.corpus_path or hf_hub_download("Exploration-Lab/CS-781-Capstone",
                                                f"Phase_1_Pretraining/Train/{src}.json", repo_type="dataset")
        t0 = time.time()
        for n, rec in enumerate(stream_records(path)):
            text = rec.get("text") or ""
            st["records"] += 1
            lab, cut = label_and_cut(text)
            if lab is None:
                st[cut] += 1
                continue
            stripped = text[:cut].rstrip()
            if len(stripped.split()) < a.min_words:
                st["too_short_after_cut"] += 1
                continue
            # which CJPE documents is this judgment?
            w = WORD.findall(text.lower())
            hits = collections.defaultdict(set)
            for i in range(len(w) - N + 1):
                if w[i] in first:
                    for key, j in index.get(hash(" ".join(w[i:i + N])), ()):
                        hits[key].add(j)
            m = [k for k, js in hits.items() if len(js) >= 2]
            if any(k[0] in ("test", "dev") for k in m):
                st["excluded_matches_test_or_dev"] += 1
                continue
            tr = [k for k in m if k[0] == "train"]
            if tr:
                st["duplicate_of_train"] += 1
                g_text, g_lab = docs[tr[0]]
                val_n += 1
                val_agree += int(g_lab == lab)
                cut_ratio.append(len(stripped) / max(1, len(g_text)))
                continue
            out.write(json.dumps({"id": f"{src}:{n}", "text": stripped, "label": lab}) + "\n")
            st["written"] += 1
            lab_count[lab] += 1
            if n % 20000 == 0 and n:
                log(f"    {src}: {n} records, {st['written']} written ({time.time()-t0:.0f}s)")
        log(f"  {src}: {n+1} records in {time.time()-t0:.0f}s")
    out.close()
    os.replace(tmp, a.out)
    cut_ratio.sort()
    q = lambda p: cut_ratio[min(len(cut_ratio) - 1, int(p * len(cut_ratio)))] if cut_ratio else float("nan")
    res = dict(stats=dict(st), labels=dict(lab_count),
               train_validation=dict(n=val_n, label_agreement=val_agree / max(1, val_n),
                                     cut_len_ratio_p10=q(.1), cut_len_ratio_median=q(.5), cut_len_ratio_p90=q(.9)))
    atomic_json(res, a.out.replace(".jsonl", "_stats.json"), indent=1)
    log(f"\n  {dict(st)}")
    log(f"  WRITTEN {st['written']} silver docs -> {a.out}  labels {dict(lab_count)}")
    log(f"  labeller vs gold on {val_n} corpus copies of TRAIN docs: agreement {100*val_agree/max(1,val_n):.1f}% | "
        f"our cut / organisers' length: p10 {q(.1):.2f} median {q(.5):.2f} p90 {q(.9):.2f} (1.00 = same cut)")
    log("  test/dev matches were dropped and never written.")


if __name__ == "__main__":
    main()
