#!/bin/bash
# Phase 5 GPU queue, ONE job at a time (alone on the L4 is fastest and cannot OOM).
# Ordered by what grading needs first, then expected value, so if compute runs
# out part-way, what finished is the most valuable part. Re-running is safe:
# finished runs are skipped (results/<tag>/meta.json), an interrupted run resumes
# from ckpt/<tag>_resume.pt.
#
#   BUDGET=35 bash run_p5.sh            # minutes of TRAINING per run (inference extra)
#   SILVER=40000 BUDGET=35 bash run_p5.sh   # + up to 40k silver docs per run (only if the CPU
#                                           # lex comparison lex_w12 vs lex_w12_sil says silver helps)
#   ONLY="m4_lorma" bash run_p5.sh      # one run
#
#   1. m4_lorma     M4 pointer-decoder, LoRMA (compulsory)
#   2. m1_adapter   M1 GQA decoder, Houlsby adapters
#   3. m2_partial   M2 encoder (of the enc-dec), top-5-of-10 blocks + layer-wise LR decay
#   4. m3_qlora     M3 MLA decoder, NF4 trunk + LoRA on every linear
#   5. m1_prefix    M1, prefix tuning (trunk frozen)
#   6. teacher      CPU: greedy probability average of the neural runs -> teacher.npz (dev+test)
#   7. kd_m4half    half-depth M4 student, full FT, gold on train + teacher on dev/test text
#   8. m4_full      M4 full fine-tuning (only if compute remains)

cd "$(dirname "$0")"
P5=${CS781_P5:-/content/drive/MyDrive/cs781_capstone/phase5}
L=$P5/logs
mkdir -p "$L"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
B=${BUDGET:-35}
SIL=""
[ -n "$SILVER" ] && [ "$SILVER" != "0" ] && SIL="--silver $SILVER --silver-weight ${SILVER_W:-0.5}"
COMMON="$SIL --budget-min $B --chunk 512 --train-chunks 2 --infer-chunks ${INFER_CHUNKS:-6} --batch 8 --keep-first ${P5_ARGS:-}"
Q=$L/queue.txt
say() { echo "[$(date +%H:%M)] $*" | tee -a "$Q"; }
DEADLINE=${P5_DEADLINE:-0}          # epoch seconds; set by final_p5.sh. 0 = none
late() { [ "$DEADLINE" -gt 0 ] && [ "$(date +%s)" -gt "$DEADLINE" ]; }

run() {
  local tag=$1; shift
  if [ -n "$ONLY" ] && [[ ",$ONLY," != *",$tag,"* ]]; then return; fi
  if [ -f "$P5/results/$tag/meta.json" ]; then say "skip $tag (done)"; return; fi
  if late; then say "skip $tag (deadline)"; return; fi
  for attempt in 1 2 3; do
    say "start $tag"
    local n0=$(cat "$L/q_$tag.out" 2>/dev/null | wc -l)
    python phase5_lsi.py --tag "$tag" $COMMON "$@" >> "$L/q_$tag.out" 2>&1
    if [ -f "$P5/results/$tag/meta.json" ]; then
      say "done  $tag | $(grep -a 'DEV global' "$L/q_$tag.out" | tail -1 | cut -c1-160)"
      return
    fi
    if tail -n +$((n0 + 1)) "$L/q_$tag.out" | grep -aq "OutOfMemoryError\|CUDA out of memory"; then
      say "OOM   $tag (attempt $attempt) -- retrying with --grad-ckpt in 1 min"; sleep 60
      set -- "$@" --grad-ckpt
    else
      say "FAIL  $tag -- see logs/q_$tag.out"; tail -n 5 "$L/q_$tag.out" | tee -a "$Q"; return
    fi
  done
}

say "queue: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null) budget ${B} min/run"

run m4_lorma   --model M4_LexArcP  --method lorma   --rank 16
run m1_adapter --model M1_SC       --method adapter --bottleneck 64
run m2_partial --model M2_LexArcED --method partial --train-top 5 --llrd 0.85
run m3_qlora   --model M3_LexArcL  --method qlora   --rank 16
run m1_prefix  --model M1_SC       --method prefix  --n-virtual 32

if [ -z "$ONLY" ] || [[ ",$ONLY," == *",kd_m4half,"* ]]; then
  if [ ! -f "$P5/teacher.npz" ]; then
    # neural members only: a teacher containing the TF-IDF model would make the
    # student's submission a hybrid by inheritance.
    members=$(for t in m4_lorma m1_adapter m2_partial m3_qlora m1_prefix m4_full; do
                [ -f "$P5/results/$t/meta.json" ] && printf "%s," "$t"; done)
    say "teacher from: $members"
    python phase5_ens.py ens --runs "$members" --greedy 10 --tag ens_teacher --teacher "$P5/teacher.npz" \
        > "$L/ens_teacher.out" 2>&1
    grep -a "greedy\|strategy\|teacher" "$L/ens_teacher.out" | tail -n 4 | tee -a "$Q"
  fi
  run kd_m4half --model M4_LexArcP --method full --keep-every 2 --no-ckpt \
      --kd-teacher "$P5/teacher.npz" --kd-docs dev,test --kd-alpha 0.5
fi

run m4_full --model M4_LexArcP --method full --no-ckpt
python phase5_ens.py status > /dev/null 2>&1
say "QUEUE DONE"
