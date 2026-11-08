#!/bin/bash
# Day-1 restart. Three jobs on one L4 ran at ~9% of the GPU's throughput (6 min
# per epoch each), so this runs TWO queues, caps CPU threads per job, drops
# gradient checkpointing (two jobs fit without it; results are bit-identical),
# and stops earlier on runs that have not started yet. Runs already in progress
# resume at their next missing fold with their original arguments.
#
#   bash run_day1b.sh          # both queues, then ensemble + distillation

cd "$(dirname "$0")"
P3=${CS781_P3:-/content/drive/MyDrive/cs781_capstone/phase3}
L=$P3/logs
mkdir -p "$L"
NT=$(( $(nproc) / 2 )); [ $NT -lt 1 ] && NT=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=$NT MKL_NUM_THREADS=$NT
EXT="--extra $P3/external/marro.jsonl $P3/external/kalamkar_mask.jsonl"
COMMON="--batch 2 ${P3_ARGS:-}"
FAST="--epochs 12 --patience 3"
Q=$L/queues.txt

say() { echo "[$(date +%H:%M)] $*" | tee -a "$Q"; }

run() {
  local tag=$1; shift
  if [ -f "$P3/results/$tag/meta.json" ]; then say "skip $tag (done)"; return; fi
  for attempt in 1 2 3; do
    say "start $tag"
    local n0=$(cat "$L/q_$tag.out" 2>/dev/null | wc -l)
    python phase3_rr.py --tag "$tag" --submit $COMMON "$@" >> "$L/q_$tag.out" 2>&1
    if [ -f "$P3/results/$tag/meta.json" ]; then
      say "done  $tag | $(grep -a 'OOF macro-F1' "$L/q_$tag.out" | tail -1 | sed 's/^ *//')"
      return
    fi
    if tail -n +$((n0 + 1)) "$L/q_$tag.out" | grep -aq "OutOfMemoryError\|CUDA out of memory"; then
      say "OOM   $tag (attempt $attempt) -- waiting 10 min"; sleep 600
    else
      say "FAIL  $tag -- see logs/q_$tag.out"; tail -5 "$L/q_$tag.out" | tee -a "$Q"; return
    fi
  done
}

say "restart: 2 queues, $NT CPU threads each, $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"

qa() {
  run base_M2  --model M2_LexArcED --method full --no-ckpt            # resumes, original args
  run base_M4  --model M4_LexArcP  --method full --no-ckpt $FAST
  run base_M1  --model M1_SC       --method full --no-ckpt $FAST
  run base_M3  --model M3_LexArcL  --method full --no-ckpt $FAST
}
qb() {
  run ext_M2   --model M2_LexArcED --method full --no-ckpt $EXT       # resumes, original args
  run lorma_M3 --model M3_LexArcL  --method lorma --rank 16           # resumes, original args (compulsory)
  run ext_M4   --model M4_LexArcP  --method full --no-ckpt $EXT $FAST
}

qa & qb & wait

say "ensemble of all finished runs"
python phase3_ens.py ens --runs all --greedy 12 --dump-teacher "$P3/teacher_day1.pt" \
  --submit --tag ens_day1 > "$L/ens_day1.out" 2>&1 && grep -a "best pooled\|OOF macro\|WROTE" "$L/ens_day1.out" | tee -a "$Q"
python phase3_ens.py status > /dev/null 2>&1
say "ALL DONE"
