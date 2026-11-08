#!/bin/bash
# Phase 4 second push, unattended (~4.5 GPU-h). Acts on the Codabench test scores of 2026-10-18:
#   * big M4 models transfer to test (m4_final dev 0.691 -> test 0.704, m4_spread8 0.719 -> 0.708)
#   * TF-IDF does not (lex_all_ext 0.721 -> 0.692), so ensembles should lean on more, different neural members
#
#   1. m4_full_spread   M4 FULL fine-tuning, all gold + silver, 4 chunks spread over the judgment
#                       (full FT lost at 10k docs; it has 4.6x the data here and errs differently from LoRMA)
#   2. m1_lorma_spread  M1 LoRMA, same data and view: a second architecture for the ensemble
#   2b. m4_spread6     M4 LoRMA trained on 6 spread chunks (training sees more of the judgment)
#   each + an 8-chunk re-read of its weights (no training)
#   3. ensembles: greedy hybrid, greedy model-only, and fixed neural-heavy mixes -> zips, SUBMIT_ORDER.txt
# A deadline (P4_HOURS, default 6) stops new GPU work.

cd "$(dirname "$0")"
P4=${CS781_P4:-/content/drive/MyDrive/cs781_capstone/phase4}
L=$P4/logs
Q=$L/queue.txt
SILVER=$P4/external/sc_silver.jsonl
mkdir -p "$L"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
say() { echo "[$(date +%H:%M)] $*" | tee -a "$Q"; }
DEADLINE=$(( $(date +%s) + ${P4_HOURS:-10} * 3600 ))
late() { [ "$(date +%s)" -gt "$DEADLINE" ]; }
say "push2: started (deadline in ${P4_HOURS:-10} h) | $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"

ALL="--train-docs 100000 --kd-pool 0 --extra $SILVER --extra-weight 0.5 --chunk 512 --evals 3 --submit --save-ckpt"

run() {   # run <tag> <args>
  local tag=$1; shift
  if [ -f "$P4/results/$tag/meta.json" ]; then say "skip  $tag (done)"; return; fi
  if late; then say "skip  $tag (deadline)"; return; fi
  say "start $tag"
  python phase4_cjpe.py --tag "$tag" $ALL "$@" ${P4_ARGS:-} >> "$L/q_$tag.out" 2>&1 \
    || { grep -aq "out of memory\|OutOfMemoryError" "$L/q_$tag.out" && say "OOM   $tag -- retrying with --grad-ckpt" \
         && python phase4_cjpe.py --tag "$tag" $ALL "$@" ${P4_ARGS:-} --grad-ckpt >> "$L/q_$tag.out" 2>&1; }
  if [ -f "$P4/results/$tag/meta.json" ]; then
    say "done  $tag | $(grep -a '=====.*dev macro-F1' "$L/q_$tag.out" | tail -1 | sed 's/^=* *//')"
  else
    say "FAIL  $tag -- see logs/q_$tag.out"; tail -n 5 "$L/q_$tag.out" | tee -a "$Q"
  fi
}

reread8() {   # reread8 <src tag> <args used to train it>
  local src=$1; shift
  [ -f "$P4/ckpt/$src.pt" ] || return
  [ -f "$P4/results/${src}8/meta.json" ] && return
  python phase4_cjpe.py --tag "${src}8" --ckpt-from "$src" --predict-only $ALL "$@" ${P4_ARGS:-} \
      --view spread --chunks 8 >> "$L/q_${src}8.out" 2>&1
  say "done  ${src}8 | $(grep -a 're-predicted' "$L/q_${src}8.out" | tail -1 | sed 's/^ *//')"
}

F="--model M4_LexArcP --method full --view spread --chunks 4 --batch 4 --accum 4"
M="--model M1_SC --method lorma --rank 16 --view spread --chunks 4 --batch 8 --accum 2"
run m4_full_spread $F
reread8 m4_full_spread $F
run m1_lorma_spread $M
reread8 m1_lorma_spread $M
S6="--model M4_LexArcP --method lorma --rank 16 --view spread --chunks 6 --batch 4 --accum 4"
run m4_spread6 $S6
reread8 m4_spread6 $S6

# ---- ensembles (CPU, seconds)
ens() { local tag=$1; shift
  python phase4_ens.py ens --tag "$tag" "$@" > "$L/$tag.out" 2>&1
  say "ens   $(grep -a 'dev macro-F1\|Error' "$L/$tag.out" | tail -1 | sed 's/^ *//')"
}
X="--exclude lexcurve lexsec bench m4_tail_e2"
ens ens_p2_hyb --greedy 8 $X
ens ens_p2_model --kind model --greedy 8 $X
# neural-heavy fixed mixes: every strong neural member at weight 1, lex_all_ext at weight 1
NEU=""
for r in m4_spread8 m4_final m4_spread m4_full_spread8 m1_lorma_spread8 m4_spread68; do
  [ -f "$P4/results/$r/meta.json" ] && NEU="$NEU $r"
done
W=$(for r in $NEU; do printf "1 "; done)
ens ens_p2_neu --runs $NEU lex_all_ext --weights $W 1
ens ens_p2_neu_rate --runs $NEU lex_all_ext --weights $W 1 --rate
ens ens_p2_neu_model --runs $NEU --weights $W
# corpus-retrieval versions of the new ensembles (cached corpus text, seconds)
if [ -f "$P4/retrieval_cont.json" ]; then
  RB=""
  for r in ens_p2_hyb ens_p2_neu ens_p2_neu_rate ens_p2_neu_model; do
    [ -f "$P4/results/$r/meta.json" ] && RB="$RB $r"
  done
  [ -n "$RB" ] && python phase4_retrieve.py --base $RB --submit > "$L/ret_p2.out" 2>&1
  grep -a "with corpus outcomes" "$L/ret_p2.out" | sed 's/^ *//' | while read -r l; do say "ret   $l"; done
fi
python phase4_ens.py status > /dev/null 2>&1
python phase4_ens.py order
say "PUSH2 DONE -- see phase4/SUBMIT_ORDER.txt"
