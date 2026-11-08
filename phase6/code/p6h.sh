#!/bin/bash
# Phase 6, the Phase-1 sentence scorer -- ONE foreground cell, about 4 h on an L4.
#
#   bash p6h.sh
#
# Caches the metric's sentence labels, separates bsel2's two changes, fine-tunes two
# Phase-1 trunks to score sentences, then writes ranked zips to submissions/_final2.
# Each step is skipped if its output exists, so after a disconnect re-run the SAME cell.
# Nothing here touches submissions/_today, _final or SUBMIT_ORDER.md.
set -u
cd "$(dirname "$0")"
P6=${CS781_P6:-/content/drive/MyDrive/cs781_capstone/phase6}
L=$P6/logs
Q=$L/queue.txt
mkdir -p "$L"
say() { echo "[$(date +%H:%M)] $*" | tee -a "$Q"; }
J=${P6_PROCS:-$(nproc)}
CJ=$(( J > 3 ? J - 2 : 1 ))

step() {            # step <marker-file> <log-name> <command...>
  local mark=$1 name=$2; shift 2
  if [ -e "$mark" ]; then say "skip  $name (exists)"; return 0; fi
  say "run   $name"
  if "$@" > "$L/$name.log" 2>&1; then
    say "done  $name"
  else
    say "FAIL  $name -- tail of $L/$name.log:"
    tail -20 "$L/$name.log" | tee -a "$Q"
    return 1
  fi
}

say "======== p6h.sh | Phase-1 sentence scorer"
if [ -z "${P6_NO_PIP:-}" ] && ! python -c "import rouge_score, bert_score" 2>/dev/null; then
  say "pip   rouge-score bert-score"
  pip -q install rouge-score bert-score > "$L/pip.log" 2>&1
fi

step "$P6/cache/nsel_labels.done" 70_labels \
     python phase6_nsel.py --procs "$CJ" labels --budget-min "${P6H_LAB_MIN:-85}" || exit 1
grep -a 'surrogate oracle F' "$L/70_labels.log" | tail -2 | cut -c1-200 | while read -r l; do say "nsel  $l"; done
step "$P6/results/bsel3_f_model.pkl" 71_gb python phase6_nsel.py --procs "$CJ" gb || exit 1

MEM=""
for pair in ${P6H_MODELS:-m1:M1_SC m2:M2_LexArcED}; do
  t=nsel_${pair%%:*}
  MEM=$MEM${MEM:+,}$t
  step "$P6/results/$t/meta.json" "72_$t" \
       python phase6_nsel.py --procs "$CJ" train --tag "$t" --model "${pair##*:}" \
              --budget-min "${P6H_MIN:-60}" ${P6H_TRAIN_ARGS:-}
  grep -a -E '^\[.*\] (DONE|.*probe:)' "$L/72_$t.log" | cut -c1-200 | while read -r l; do say "nsel  $l"; done
done

step "$P6/results/nsel_sweep.json" 73_sweep python phase6_nsel.py --procs "$CJ" sweep --members "$MEM" || exit 1
grep -a -E 'CONTROL|HELD-OUT HALF' "$L/73_sweep.log" | cut -c1-200 | while read -r l; do say "nsel  $l"; done
Z=$(ls "$P6/submissions/_final2/"*.zip 2>/dev/null | wc -l)
say "======== SCORER DONE: $Z zips in submissions/_final2, best held-out gain first."
