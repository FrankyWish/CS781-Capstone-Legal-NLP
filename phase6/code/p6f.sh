#!/bin/bash
# Phase 6, last day -- ONE foreground cell, picks up from whatever already exists.
#
#   P6F_HOURS=7 bash p6f.sh     hours until you are back; the zips are ready before that
#
# 1. finishes the main run if it did not finish (p6.sh all, resumes, required runs first)
# 2. corpus scan, measure only, in the background
# 3. window-aimed extraction, then extraction labelled by BERTScore itself
# 4. neural members trained on the summary's first ~430 words, while time remains
# 5. ensembles over everything, then phase6_batch.py: numbered zips in submissions/_today
#
# Every step is skipped if its output exists, so after a disconnect re-run the SAME cell.
# Stops starting GPU work 2 h before P6F_HOURS: stage 5 always runs.
set -u
cd "$(dirname "$0")"
P6=${CS781_P6:-/content/drive/MyDrive/cs781_capstone/phase6}
L=$P6/logs
mkdir -p "$L"
Q=$L/queue.txt
say() { echo "[$(date +%H:%M)] $*" | tee -a "$Q"; }
J=${P6_PROCS:-$(nproc)}
CJ=$(( J > 3 ? J - 2 : 1 ))
NB=""; [ -n "${P6_NO_BERT:-}" ] && NB="--no-bert"
B=${BUDGET:-25}
T0=$(date +%s)
H=${P6F_HOURS:-7}
STOP=$(( T0 + $(python -c "print(max(0, int(($H - 2) * 3600)))") ))
late() { [ "$(date +%s)" -gt "$STOP" ]; }

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

more() {            # more <tag> <args...>
  local tag=$1; shift
  if [ -f "$P6/results/$tag/meta.json" ]; then say "skip  $tag (done)"; return 0; fi
  if late; then say "skip  $tag (GPU deadline passed)"; return 0; fi
  say "start $tag"
  python phase6_sum.py train --tag "$tag" $COMMON "$@" >> "$L/q_$tag.out" 2>&1
  if [ -f "$P6/results/$tag/meta.json" ]; then
    say "done  $tag | $(grep -a 'DONE ' "$L/q_$tag.out" | tail -1 | cut -c12-200)"
  else
    say "FAIL  $tag -- tail of logs/q_$tag.out:"; tail -8 "$L/q_$tag.out" | tee -a "$Q"
  fi
}

main_finished() {
  [ -f "$P6/SUBMIT_ORDER.md" ] || return 1
  for t in m4_lorma m2_lora m1_houlsby m3_qlora m2_adalora m3_pfeiffer kd_m1half; do
    [ -f "$P6/results/$t/meta.json" ] || return 1
  done
}

say "======== p6f.sh | $H h until you are back, GPU work stops at +$(( (STOP - T0) / 60 )) min"
if [ -z "${P6_NO_PIP:-}" ] && ! python -c "import rouge_score, bert_score" 2>/dev/null; then
  say "pip   rouge-score bert-score"
  pip -q install rouge-score bert-score > "$L/pip.log" 2>&1
fi

if main_finished; then
  say "skip  main run (all 7 required runs and SUBMIT_ORDER.md exist)"
else
  say "run   main run (resumes where it stopped)"
  P6_DEADLINE=$STOP bash p6.sh all
fi
if [ ! -f "$P6/results/sel_model.pkl" ]; then
  say "STOP: results/sel_model.pkl is missing, the main run failed. Paste logs/queue.txt."
  exit 1
fi

scan() {             # measure only: it may never hold back the zips
  local left=$(( STOP - $(date +%s) )); [ "$left" -lt 1800 ] && left=1800
  step "$P6/results/corpus_scan.json" 40_corpus_scan timeout "$left" \
       python phase6_external.py scan --procs 3 ${P6F_SCAN_ARGS:-} || return 0
  grep -a 'SCAN DONE' "$L/40_corpus_scan.log" | cut -c1-260 | while read -r l; do say "scan  $l"; done
}
scan &
SCAN=$!

step "$P6/results/win_model.pkl" 41_win_fit python phase6_win.py --procs "$CJ" fit
if [ -f "$P6/results/win_model.pkl" ]; then
  step "$P6/results/win_best/meta.json" 42_win_sweep \
       python phase6_win.py --procs "$CJ" sweep --tag win_best --heads win \
              --mults "0.9,1.0,1.15,1.3" $NB
  grep -a 'HELD-OUT HALF' "$L/42_win_sweep.log" | cut -c1-200 | while read -r l; do say "win   $l"; done
fi

if [ -z "$NB" ]; then
  step "$P6/results/bsel_g_model.pkl" 43_bsel_fit \
       python phase6_bsel.py --procs "$CJ" fit --docs "${P6F_BSEL_DOCS:-1500}"
  grep -a 'surrogate oracle' "$L/43_bsel_fit.log" | tail -1 | cut -c1-200 | while read -r l; do say "bsel  $l"; done
fi
if [ -f "$P6/results/bsel_g_model.pkl" ]; then
  step "$P6/results/bsel_best/meta.json" 44_bsel_sweep \
       python phase6_win.py --procs "$CJ" sweep --tag bsel_best --heads bsel_f,bsel_g \
              --wins 420 --mults "1.0,1.15,1.3" --lams "0.15,0.45" $NB
  grep -a 'HELD-OUT HALF' "$L/44_bsel_sweep.log" | cut -c1-200 | while read -r l; do say "bsel  $l"; done
fi

# the summary's first ~430 words are all BERTScore reads: spend the model on those
WS=${P6F_SUM_TOKENS:-520}
COMMON="--budget-min $B --sum-tokens $WS --doc-tokens ${P6_DOC_TOKENS:-1152} --gen-dev ${P6_GEN_DEV:-200} $NB ${P6_ARGS:-}"
more m4_win --model M4_LexArcP  --method lorma   --rank 16
more m2_win --model M2_LexArcED --method lora    --rank 16 --targets attn
more m3_win --model M3_LexArcL  --method qlora   --rank 16
more m1_win --model M1_SC       --method houlsby --bottleneck 64

say "waiting for the corpus scan"
wait $SCAN
bash p6.sh ens
step "$P6/results/batch.done.$T0" 50_batch python phase6_batch.py --procs "$CJ" all $NB \
     && touch "$P6/results/batch.done.$T0"
grep -a -E '^\[.*\]\s+[0-9 ][0-9] [a-z]|added |BATCH DONE' "$L/50_batch.log" | cut -c1-200 | while read -r l; do say "batch $l"; done
bash p6.sh status
Z=$(ls "$P6/submissions/_today/"*.zip 2>/dev/null | wc -l)
say "======== READY: $Z numbered zips in submissions/_today, order in SUBMIT_ORDER.md"
