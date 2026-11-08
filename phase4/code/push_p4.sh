#!/bin/bash
# Phase 4 score push, unattended (~4.5 GPU-h). Builds on the finished run_p4.sh + final_p4.sh results.
#
# Evidence it acts on (dev = 994 real dev cases):
#   * neural score still rising with data and with training (m4_final best at its last step):
#       m4_lorma 10k 0.634 -> m4_rest 24k 0.676 -> m4_final 46k 0.691
#   * TF-IDF by section: head 0.631, middle 0.645, end 0.675, WHOLE document 0.707,
#     but every neural run so far reads only the last 1,024 tokens.
#
#   1. m4_spread    M4 LoRMA, all gold + silver, 4 chunks of 512 spread over the WHOLE judgment
#   2. m4_spread8   m4_spread's weights reading 8 chunks at test time (no training, minutes)
#   3. m4_tail_e2   m4_final's setup (tail 2x512) trained for 2 epochs instead of 1
#   CPU, alongside: lex_all_ext (whole-document TF-IDF + silver), lex_all_char_ext (+ char n-grams)
#   4. ensembles -> zips, STATUS.md, SUBMIT_ORDER.txt
#
# A deadline (P4_HOURS, default 6) stops new GPU work.

cd "$(dirname "$0")"
P4=${CS781_P4:-/content/drive/MyDrive/cs781_capstone/phase4}
L=$P4/logs
Q=$L/queue.txt
SILVER=$P4/external/sc_silver.jsonl
mkdir -p "$L"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
say() { echo "[$(date +%H:%M)] $*" | tee -a "$Q"; }
DEADLINE=$(( $(date +%s) + ${P4_HOURS:-6} * 3600 ))
late() { [ "$(date +%s)" -gt "$DEADLINE" ]; }
say "push: started (deadline in ${P4_HOURS:-6} h) | $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"

# ---- CPU: stronger lexical members, in the background
(
  lex() { local tag=$1; shift
    [ -f "$P4/results/$tag/meta.json" ] && return
    python phase4_lex.py "$@" --tag "$tag" > "$L/$tag.log" 2>&1
    say "lex   $tag | $(grep -a 'BEST' "$L/$tag.log" | tail -1 | sed 's/^ *//')"
  }
  lex lex_all_ext --view all --extra "$SILVER"
  lex lex_all_char_ext --view all --char --extra "$SILVER"
) &
LEXPID=$!

ALL="--model M4_LexArcP --method lorma --rank 16 --train-docs 100000 --kd-pool 0 --extra $SILVER --extra-weight 0.5
     --chunk 512 --batch 8 --accum 2 --evals 3 --submit --save-ckpt ${P4_ARGS:-}"

run() {
  local tag=$1; shift
  if [ -f "$P4/results/$tag/meta.json" ]; then say "skip  $tag (done)"; return; fi
  if late; then say "skip  $tag (deadline)"; return; fi
  say "start $tag"
  python phase4_cjpe.py --tag "$tag" $ALL "$@" >> "$L/q_$tag.out" 2>&1 \
    || { grep -aq "out of memory\|OutOfMemoryError" "$L/q_$tag.out" && say "OOM   $tag -- retrying with --grad-ckpt" \
         && python phase4_cjpe.py --tag "$tag" $ALL "$@" --grad-ckpt >> "$L/q_$tag.out" 2>&1; }
  if [ -f "$P4/results/$tag/meta.json" ]; then
    say "done  $tag | $(grep -a '=====.*dev macro-F1' "$L/q_$tag.out" | tail -1 | sed 's/^=* *//')"
  else
    say "FAIL  $tag -- see logs/q_$tag.out"; tail -n 5 "$L/q_$tag.out" | tee -a "$Q"
  fi
}

run m4_spread --view spread --chunks 4
if [ -f "$P4/ckpt/m4_spread.pt" ] && [ ! -f "$P4/results/m4_spread8/meta.json" ]; then
  python phase4_cjpe.py --tag m4_spread8 --ckpt-from m4_spread --predict-only $ALL --view spread --chunks 8 \
      >> "$L/q_m4_spread8.out" 2>&1
  say "done  m4_spread8 | $(grep -a 're-predicted' "$L/q_m4_spread8.out" | tail -1 | sed 's/^ *//')"
fi
run m4_tail_e2 --view tail --chunks 2 --epochs 2

# ---- ensembles
wait $LEXPID
ens() { local tag=$1; shift
  python phase4_ens.py ens --tag "$tag" --exclude lexcurve lexsec bench "$@" > "$L/$tag.out" 2>&1
  say "ens   $(grep -a 'dev macro-F1' "$L/$tag.out" | tail -1 | sed 's/^ *//')"
}
ens ens_push_model --kind model --greedy 8
ens ens_push_hyb --greedy 8
ens ens_push_model_rate --kind model --greedy 8 --rate
ens ens_push_hyb_rate --greedy 8 --rate
python phase4_ens.py status > /dev/null 2>&1
python phase4_ens.py order
say "PUSH DONE -- see phase4/SUBMIT_ORDER.txt"
