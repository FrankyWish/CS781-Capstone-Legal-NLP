#!/bin/bash
# Phase 3 final push, ONE job at a time (alone on the L4 is fastest and cannot
# OOM). Steps are ordered by expected Macro-F1 gain per GPU-hour, so if compute
# runs out part-way, what finished is the most valuable part. Re-running is
# safe: finished steps are skipped, a half-finished run resumes at its next fold.
#
#   1. lora_M4_aux       LoRA on M4 (best single, 0.5223) + label-shift auxiliary
#                        head + 128-sentence windows                      ~2.0 h
#   2. DAPT M4           continued pretraining on in-domain text           ~0.5 h
#   3. lora_M4_dapt_aux  step 1 again, from the continued-pretrained trunk ~2.0 h
#   4. ensemble -> teacher for round-2 distillation                      minutes
#   5. kd2_M4half        half-depth M4 student, distilled from that teacher
#                        on train + test text                              ~1.2 h
#   6. final ensembles (model-only and hybrid) -> zips                   minutes

cd "$(dirname "$0")"
P3=${CS781_P3:-/content/drive/MyDrive/cs781_capstone/phase3}
L=$P3/logs
mkdir -p "$L"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=$(nproc)
NEW="--batch 2 --epochs 12 --patience 3 --window 128 --shift-aux 0.5 --grad-ckpt --submit ${P3_ARGS:-}"
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
      say "done  $tag | $(grep -a 'OOF macro-F1' "$L/q_$tag.out" | tail -1 | sed 's/^ *//')"
      return
    fi
    if tail -n +$((n0 + 1)) "$L/q_$tag.out" | grep -aq "OutOfMemoryError\|CUDA out of memory"; then
      say "OOM   $tag (attempt $attempt) -- retrying in 1 min"; sleep 60
    else
      say "FAIL  $tag -- see logs/q_$tag.out"; tail -5 "$L/q_$tag.out" | tee -a "$Q"; return
    fi
  done
}

ens() {   # ens <tag> <extra args>
  local tag=$1; shift
  python phase3_ens.py ens --runs all --greedy 15 --submit --tag "$tag" "$@" > "$L/$tag.out" 2>&1
  grep -a "weights\|best pooled\|OOF macro\|teacher\|WROTE\|Error\|Traceback" "$L/$tag.out" | tee -a "$Q"
}

say "push: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"

run lora_M4_aux --model M4_LexArcP --method lora --rank 16 $NEW

if [ ! -f "$P3/dapt/M4_LexArcP/best.pt" ]; then
  say "start DAPT M4"
  python phase3_dapt.py --model M4_LexArcP --tokens 40e6 >> "$L/q_dapt_M4.out" 2>&1
  say "DAPT: $(grep -a 'held-out masked-LM loss .* ->' "$L/q_dapt_M4.out" | tail -1 | sed 's/^ *//')"
fi
if [ -f "$P3/dapt/M4_LexArcP/best.pt" ]; then
  run lora_M4_dapt_aux --model M4_LexArcP --trunk-path "$P3/dapt/M4_LexArcP/best.pt" \
      --method lora --rank 16 $NEW
fi

ens ens_push1 --exclude ctx0 band lex --dump-teacher "$P3/teacher_push.pt"
if [ -f "$P3/teacher_push.pt" ]; then
  run kd2_M4half --model M4_LexArcP --method full --no-ckpt --keep-every 2 \
      --kd-logits "$P3/teacher_push.pt" --kd-test $NEW
fi

ens ens_push_model --exclude ctx0 band lex
ens ens_push_hyb --exclude ctx0 band
python phase3_ens.py status > /dev/null 2>&1
say "PUSH DONE"
