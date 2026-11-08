#!/bin/bash
# Phase 6, M3 and M4 as two more sentence scorers -- ONE foreground cell, about 2.5 h on an L4.
#
#   bash p6i.sh
#
# Needs p6h's outputs (cached labels, bsel3, nsel_m1, nsel_m2). Trains nsel_m3 and nsel_m4,
# then re-sweeps over all four trunks. Zips land in submissions/_final3 as nsel4_*, ranked by
# held-out gain, each also scored against nsel_stack (test 0.8596) on the same 300 docs.
# Each step is skipped if its output exists, so after a disconnect re-run the SAME cell.
# Nothing here touches _final2 or any nsel_* result.
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

say "======== p6i.sh | M3 + M4 sentence scorers"
if [ -z "${P6_NO_PIP:-}" ] && ! python -c "import rouge_score, bert_score" 2>/dev/null; then
  say "pip   rouge-score bert-score"
  pip -q install rouge-score bert-score > "$L/pip.log" 2>&1
fi
for need in cache/nsel_labels.done results/bsel3_f_model.pkl results/nsel_m1/scores.pkl \
            results/nsel_m2/scores.pkl results/nsel_stack/summaries_dev.json; do
  [ -e "$P6/$need" ] || { say "FAIL  missing $need (run p6h.sh first)"; exit 1; }
done

MEM=nsel_m1,nsel_m2
for pair in ${P6I_MODELS:-m3:M3_LexArcL m4:M4_LexArcP}; do
  t=nsel_${pair%%:*}
  step "$P6/results/$t/meta.json" "74_$t" \
       python phase6_nsel.py --procs "$CJ" train --tag "$t" --model "${pair##*:}" \
              --budget-min "${P6I_MIN:-55}" ${P6I_TRAIN_ARGS:-}
  [ -e "$P6/results/$t/scores.pkl" ] && MEM=$MEM,$t
  grep -a -E '^\[.*\] (DONE|.*probe:)' "$L/74_$t.log" 2>/dev/null | cut -c1-200 | while read -r l; do say "nsel  $l"; done
done

say "sweep members: $MEM"
step "$P6/results/nsel4_sweep.json" 75_sweep python phase6_nsel.py --procs "$CJ" sweep \
     --members "$MEM" --prefix nsel4 --out _final3 --versus nsel_stack || exit 1
grep -a -E 'CONTROL|VERSUS|HELD-OUT HALF|VS nsel' "$L/75_sweep.log" | cut -c1-200 | while read -r l; do say "nsel  $l"; done
Z=$(ls "$P6/submissions/_final3/"*.zip 2>/dev/null | wc -l)
say "======== M3M4 DONE: $Z zips in submissions/_final3, best held-out gain first."
