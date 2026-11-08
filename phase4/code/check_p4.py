#!/usr/bin/env python3
"""CS-781 Phase 4 -- independent check with the OFFICIAL scripts before uploading. CPU, seconds.

    python check_p4.py ens_push_hyb m4_spread8 kd_m4half ...      # default: the Day-1 ten

For each run:
  1. dev: writes truth / prediction JSON in the official format for the 994 real dev cases (prediction =
     the run's dev scores at the threshold its zip used) and scores them with evaluate_phase4.py, next to
     our own number. The two must agree to 4 decimals.
  2. zip: unpacks submissions/<run>.zip and runs tools/cs781_submit.py validate --phase 4 against test.json.

Needs evaluate_phase4.py and cs781_submit.py copied next to this file (spec/evaluation_scripts/ and tools/).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from phase4_data import P4, fetch, load_run, metrics, best_threshold, log  # noqa: E402

DAY1 = ["ens_push_hyb", "m4_spread8", "kd_m4half", "m4_full", "m1_adapter", "m3_adalora",
        "m2_unfreeze", "m1_bitfit", "m2_partial", "ens_push_hyb_rate"]


def dev_labels():
    from phase4_ens import dev_labels as dl
    return dl()


def main():
    tags = sys.argv[1:] or DAY1
    out = f"{P4}/localeval"
    os.makedirs(out, exist_ok=True)
    ydev = dev_labels()
    test_path = fetch("test")

    rows = []
    for tag in tags:
        d, t, meta = load_run(tag)
        ids = [i for i in d["ids"] if i in ydev]
        z = dict(zip(d["ids"], d["z"].tolist()))
        y = [ydev[i] for i in ids]
        thr, _ = best_threshold(y, [z[i] for i in ids])
        if meta.get("rate"):                      # rate-matched ensembles: zip threshold differs from dev's
            thr = meta["thr"]
        truth = [{"id": i, "gold": int(ydev[i])} for i in ids]
        pred = [{"id": i, "pred": int(z[i] > thr)} for i in ids]
        tp, pp = f"{out}/dev_truth.json", f"{out}/dev_pred_{tag}.json"
        json.dump(truth, open(tp, "w"))
        json.dump(pred, open(pp, "w"))
        ours = metrics(y, [p["pred"] for p in pred])["macro_f1"]
        r = subprocess.run([sys.executable, f"{HERE}/evaluate_phase4.py", "--truth", tp, "--prediction", pp],
                           capture_output=True, text=True)
        off = next((l.split()[-1] for l in r.stdout.splitlines() if l.startswith("Macro-F1")), "ERROR " + r.stderr[-200:])
        # zip validation with the course tool
        zp = f"{P4}/submissions/{tag}.zip"
        val = "no zip"
        if os.path.exists(zp):
            with zipfile.ZipFile(zp) as zf:
                names = zf.namelist()
                zf.extract("prediction.json", f"{out}/unz_{tag}")
            v = subprocess.run([sys.executable, f"{HERE}/cs781_submit.py", "validate", "--phase", "4",
                                "--prediction", f"{out}/unz_{tag}/prediction.json", "--test", test_path],
                               capture_output=True, text=True)
            ok = "Validation passed" in v.stdout
            val = ("PASS" if ok else "FAIL: " + (v.stdout + v.stderr).strip().splitlines()[-1]) + \
                  ("" if names == ["prediction.json"] else f" (zip holds {names})")
        rows.append((tag, ours, off, val))
        log(f"  {tag:22s} ours {ours:.4f} | official {off} | zip {val}")
    def clean(r):
        try:
            return abs(float(r[2]) - r[1]) < 5e-5 and r[3] == "PASS"
        except ValueError:
            return False
    bad = [r[0] for r in rows if not clean(r)]
    log(f"\n  {len(rows) - len(bad)}/{len(rows)} clean (official score == ours, zip valid)"
        + (f"; PROBLEMS: {bad}" if bad else ""))


if __name__ == "__main__":
    main()
