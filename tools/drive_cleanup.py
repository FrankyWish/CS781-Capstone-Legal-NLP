#!/usr/bin/env python3
"""Report, and optionally reclaim, redundant files in the capstone Drive folder.

    python drive_cleanup.py              # report only, deletes nothing
    python drive_cleanup.py --delete     # delete the SAFE categories

WHY THIS IS NOT A ONE-LINER

Phases 2-7 require starting from the Phase-1 models and exercising all four
architectures. A deleted checkpoint is 6 GPU-hours and ~12 compute units to
recreate, and there is no undo on a Drive mounted in Colab. So this script has
an explicit allowlist of what may go, refuses everything else, and defaults to
reporting rather than deleting.

SAFE to delete, and why:

    */ckpt/resume.pt    mid-run resume state (fp32 weights + elapsed seconds).
                        Only useful while a run is in flight. All Phase-1 runs
                        have finished, so these are dead weight.
    */ckpt/slot*.pt     the rotating checkpoints that `soup.pt` was averaged
                        from. The soup is already built and saved; the slots
                        are only needed to build a DIFFERENT soup, which no
                        remaining phase calls for.
    MBR_homog/          the homogeneous-pool control run, killed before its test
                        stage. Its dev result is recorded in EXPERIMENTS.md SS8.2.
    */gen/prediction.json   the loose copy beside a submission zip, which
                        already contains the same file.

NEVER deleted, and the script will not do it even with --delete:

    */ckpt/best.pt, final.pt, soup.pt    the models themselves
    shared/tokenizer/                    irreproducible without a retrain
    shared/dev/                          the held-out split every comparison used
    corpus/*.bin                         40 minutes to rebuild, and the Phase-1
                                         corpus is still the thing later phases
                                         check their test sets against
    */gen/*.zip                          the submissions
    *.log                                the run logs the report is written from
"""

from __future__ import annotations

import argparse
import os
import re

ROOT = os.environ.get("CS781_ROOT", "/content/drive/MyDrive/cs781_capstone")

SAFE = [
    (re.compile(r"/ckpt/resume\.pt$"),        "mid-run resume state, run finished"),
    (re.compile(r"/ckpt/slot\d+\.pt$"),       "soup ingredient, soup.pt already built"),
    (re.compile(r"/MBR_homog/"),              "killed control run, result in EXPERIMENTS SS8.2"),
    (re.compile(r"/gen/prediction\.json$"),   "duplicated inside the submission zip"),
]
PROTECTED = re.compile(
    r"/ckpt/(best|final|soup)\.pt$"
    r"|/shared/(tokenizer|dev)/"
    r"|/corpus/.*\.bin$"
    r"|\.zip$"
    r"|\.log$"
)


def walk():
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d != ".ipynb_checkpoints"]
        for fn in filenames:
            p = os.path.join(dirpath, fn)
            try:
                yield p, os.path.getsize(p)
            except OSError:
                continue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delete", action="store_true",
                    help="actually remove the safe categories (default: report only)")
    a = ap.parse_args()

    total, hits, protected_hits = 0, [], 0
    for p, sz in walk():
        total += sz
        rel = p.replace("\\", "/")
        for pat, why in SAFE:
            if pat.search(rel):
                if PROTECTED.search(rel):
                    protected_hits += 1          # belt and braces; should be 0
                    break
                hits.append((p, sz, why))
                break

    hits.sort(key=lambda x: -x[1])
    reclaim = sum(s for _, s, _ in hits)
    print(f"{ROOT}")
    print(f"  total on disk        {total/1e9:6.2f} GB")
    print(f"  reclaimable          {reclaim/1e9:6.2f} GB across {len(hits)} files")
    if protected_hits:
        print(f"  {protected_hits} file(s) matched a safe rule but are PROTECTED -- skipped")
    print()
    by_reason = {}
    for _, sz, why in hits:
        n, b = by_reason.get(why, (0, 0))
        by_reason[why] = (n + 1, b + sz)
    for why, (n, b) in sorted(by_reason.items(), key=lambda x: -x[1][1]):
        print(f"  {b/1e9:6.2f} GB  {n:4d} files  {why}")

    if not a.delete:
        print("\nREPORT ONLY. Nothing was deleted. Re-run with --delete to reclaim.")
        print("Largest candidates:")
        for p, sz, _ in hits[:12]:
            print(f"  {sz/1e6:8.1f} MB  {p[len(ROOT)+1:]}")
        return

    freed, failed = 0, 0
    for p, sz, _ in hits:
        try:
            os.remove(p)
            freed += sz
        except OSError as e:
            failed += 1
            print(f"  could not remove {p}: {e}")
    print(f"\nfreed {freed/1e9:.2f} GB" + (f", {failed} failures" if failed else ""))
    print("Checkpoints (best/final/soup), tokenizer, dev split, corpus shards, "
          "submissions and logs were not touched.")


if __name__ == "__main__":
    main()
