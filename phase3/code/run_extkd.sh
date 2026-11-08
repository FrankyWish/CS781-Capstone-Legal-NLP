#!/bin/bash
# Teacher-refined external data.
#
#   1. teacher_best.pt = out-of-fold + test logits of the members of the best
#      leaderboard ensemble (ens_final_hyb2, 0.5458) plus rsLoRA M1
#   2. teacher_ext.pt  = that, plus the adapter runs' fold checkpoints run over
#      the 495 external judgments
#   3. students trained on gold + external + test text, where each coarse
#      external label ("Argument", "Precedent") is refined by the teacher's
#      distribution over the roles that label allows
#        kdx_M4_lora   LoRA on M4                       ~4 h
#        kdx_M4half    half-depth M4 student            ~2 h
#   4. final ensembles over fixed member lists, equal weights
# Re-running is safe: finished steps are skipped.
#
# The 128-window + aux-head config tested worse on the leaderboard three times out
# of three (0.4929 / 0.4814 / 0.5052 against 0.5223 / 0.5140), so the students use
# the plain config. Greedy selection on CV picked worse test members twice
# (ens_push 0.5319 vs equal-weight ens_final_hyb2 0.5458), so the final ensembles
# use fixed member lists.

cd "$(dirname "$0")"
P3=${CS781_P3:-/content/drive/MyDrive/cs781_capstone/phase3}
L=$P3/logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=$(nproc)
EXT="--extra $P3/external/marro.jsonl $P3/external/kalamkar_mask.jsonl"
NEW="--batch 2 --epochs 12 --patience 3 --submit ${P3_ARGS:-}"
BEST="ext_M4 kd_M4half lorma_M3 lora_M4 rslora_M1"
Q=$L/queues.txt
say() { echo "[$(date +%H:%M)] $*" | tee -a "$Q"; }

run() {
  local tag=$1; shift
  if [ -f "$P3/results/$tag/meta.json" ]; then say "skip $tag (done)"; return; fi
  for attempt in 1 2 3; do
    say "start $tag"
    local n0=$(cat "$L/q_$tag.out" 2>/dev/null | wc -l)
    python phase3_rr.py --tag "$tag" "$@" >> "$L/q_$tag.out" 2>&1
    if [ -f "$P3/results/$tag/meta.json" ]; then
      say "done  $tag | $(grep -a 'OOF macro-F1' "$L/q_$tag.out" | tail -1 | sed 's/^ *//')"; return
    fi
    if tail -n +$((n0 + 1)) "$L/q_$tag.out" | grep -aq "OutOfMemoryError\|CUDA out of memory"; then
      say "OOM   $tag (attempt $attempt) -- retrying in 1 min"; sleep 60
    else
      say "FAIL  $tag -- see logs/q_$tag.out"; tail -5 "$L/q_$tag.out" | tee -a "$Q"; return
    fi
  done
}

if [ ! -f "$P3/teacher_best.pt" ]; then
  python phase3_ens.py ens --runs $BEST --dump-teacher "$P3/teacher_best.pt" > "$L/teacher_best.out" 2>&1
  grep -a "best pooled\|teacher" "$L/teacher_best.out" | tee -a "$Q"
fi
[ -f "$P3/teacher_best.pt" ] || { say "FAIL teacher_best -- see logs/teacher_best.out"; exit 1; }

if [ ! -f "$P3/teacher_ext.pt" ]; then
  say "teacher on external judgments"
  python phase3_predict.py --runs lora_M4 rslora_M1 lorma_M3 \
      --docs "$P3/external/marro.jsonl" "$P3/external/kalamkar_mask.jsonl" \
      --merge "$P3/teacher_best.pt" --out "$P3/teacher_ext.pt" >> "$L/q_teacher_ext.out" 2>&1
  say "$(tail -1 "$L/q_teacher_ext.out")"
fi
[ -f "$P3/teacher_ext.pt" ] || { say "FAIL teacher_ext -- see logs/q_teacher_ext.out"; exit 1; }

run kdx_M4_lora --model M4_LexArcP --method lora --rank 16 --kd-logits "$P3/teacher_ext.pt" --kd-test $EXT $NEW
run kdx_M4half  --model M4_LexArcP --method full --no-ckpt --keep-every 2 \
    --kd-logits "$P3/teacher_ext.pt" --kd-test $EXT $NEW

for spec in "ens_x_model $BEST kdx_M4_lora kdx_M4half" "ens_x_hyb $BEST kdx_M4_lora kdx_M4half lex_tfidf"; do
  set -- $spec; tag=$1; shift
  python phase3_ens.py ens --runs "$@" --submit --tag "$tag" > "$L/$tag.out" 2>&1
  grep -a "best pooled\|OOF macro\|WROTE\|Error\|Traceback" "$L/$tag.out" | tee -a "$Q"
done
python phase3_ens.py status > /dev/null 2>&1
say "EXTKD DONE"
