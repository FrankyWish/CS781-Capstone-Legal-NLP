#!/bin/bash
# Phase 3 day-1 pipeline, unattended, on ONE L4 runtime.
#
#   gate -> external data -> 3 queues in parallel on the same GPU
#        -> ensemble of everything finished -> distilled half-depth student
#        -> final ensemble -> STATUS.txt
#
# Re-running it is safe: finished runs are skipped, half-finished runs resume at
# the next missing fold. Every run writes a validated submission zip.
#
#   bash run_day1.sh                 # everything
#   bash run_day1.sh q1 q2 final     # a subset (e.g. q3 in another runtime)

cd "$(dirname "$0")"
P3=${CS781_P3:-/content/drive/MyDrive/cs781_capstone/phase3}
L=$P3/logs
mkdir -p "$L"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
STEPS=${@:-prep q1 q2 q3 final}
EXT="--extra $P3/external/marro.jsonl $P3/external/kalamkar_mask.jsonl"   # missing files are skipped
COMMON="--batch 2 --grad-ckpt ${P3_ARGS:-}"   # 3 jobs share 24 GB; ~6 GB each at batch 2
Q=$L/queues.txt

say() { echo "[$(date +%H:%M)] $*" | tee -a "$Q"; }

run() {   # run <tag> <args...>
  local tag=$1; shift
  if [ -f "$P3/results/$tag/meta.json" ]; then say "skip $tag (done)"; return; fi
  for attempt in 1 2 3 4 5 6; do
    say "start $tag"
    python phase3_rr.py --tag "$tag" --submit $COMMON "$@" >> "$L/q_$tag.out" 2>&1
    if [ -f "$P3/results/$tag/meta.json" ]; then
      say "done  $tag | $(grep -a 'OOF macro-F1' "$L/q_$tag.out" | tail -1 | sed 's/^ *//')"
      return
    fi
    if grep -aq "OutOfMemoryError\|CUDA out of memory" "$L/q_$tag.out"; then
      say "OOM   $tag (attempt $attempt) -- waiting 10 min, then resuming at the missing fold"
      sleep 600
    else
      say "FAIL  $tag -- see logs/q_$tag.out"; tail -5 "$L/q_$tag.out" | tee -a "$Q"; return
    fi
  done
}

q1() {   # trunk comparison, gold data only, full fine-tuning
  run base_M2 --model M2_LexArcED --method full --no-ckpt
  run base_M4 --model M4_LexArcP  --method full --no-ckpt
  run base_M1 --model M1_SC       --method full --no-ckpt
  run base_M3 --model M3_LexArcL  --method full --no-ckpt
  run ext_M1  --model M1_SC       --method full --no-ckpt $EXT
}
q2() {   # external data, CRF, and the context ablation
  run ext_M2   --model M2_LexArcED --method full --no-ckpt $EXT
  run ext_M4   --model M4_LexArcP  --method full --no-ckpt $EXT
  run crf_M2   --model M2_LexArcED --method full --no-ckpt --crf
  run ctx0_M2  --model M2_LexArcED --method full --no-ckpt --enc single --seq none
  run band2_M2 --model M2_LexArcED --method full --no-ckpt --enc single --band 2
}
q3() {   # the required adaptation techniques, across all four architectures
  run lorma_M3   --model M3_LexArcL  --method lorma --rank 16
  run lora_M4    --model M4_LexArcP  --method lora  --rank 16
  run rslora_M1  --model M1_SC       --method rslora --rank 64
  run qlora_M2   --model M2_LexArcED --method qlora --rank 16
  run adalora_M3 --model M3_LexArcL  --method adalora --rank 16
  run ia3_M4     --model M4_LexArcP  --method ia3
  run prefix_M1  --model M1_SC       --method prefix
  run prompt_M2  --model M2_LexArcED --method prompt
}

for s in $STEPS; do
  case $s in
    prep)
      rm -f "$L"/q?.done
      say "gate"
      python phase3_data.py --tokenizer > "$L/00_gate.log" 2>&1 || { say "GATE FAILED"; tail -30 "$L/00_gate.log"; exit 1; }
      tail -12 "$L/00_gate.log"
      say "external: marro"
      python phase3_external.py marro > "$L/01_marro.log" 2>&1 && tail -4 "$L/01_marro.log" || say "marro FAILED (continuing without it)"
      say "external: kalamkar"
      python phase3_external.py kalamkar > "$L/02_kalamkar.log" 2>&1 && tail -4 "$L/02_kalamkar.log" \
        || say "kalamkar FAILED -- accept the dataset terms on HF; continuing without it"
      ;;
    q1|q2|q3)
      ( $s; date > "$L/$s.done" ) &
      [ "$s" = q1 ] && sleep ${Q1_HEAD_START:-150}  # q1 builds the gold token cache before the others start
      ;;
    final)
      wait
      # Queues may run in another runtime (bash run_day1.sh q3 there): wait for
      # their markers, at most 8 hours, then ensemble whatever has finished.
      for q in q1 q2 q3; do
        t=0
        while [ ! -f "$L/$q.done" ] && [ $t -lt 480 ]; do sleep 60; t=$((t+1)); done
      done
      say "ensemble of all finished runs"
      python phase3_ens.py ens --runs all --greedy 12 --dump-teacher "$P3/teacher_day1.pt" \
        --submit --tag ens_day1 > "$L/ens_day1.out" 2>&1 && grep -a "best pooled\|OOF macro\|WROTE" "$L/ens_day1.out" | tee -a "$Q"
      if [ -f "$P3/teacher_day1.pt" ]; then
        # COMPULSORY distillation: ensemble teacher -> half-depth M2 student,
        # soft targets on gold docs (out-of-fold) and on the test documents' text
        run kd_M2half --model M2_LexArcED --method full --keep-every 2 \
          --kd-logits "$P3/teacher_day1.pt" --kd-test $EXT
      fi
      python phase3_ens.py ens --runs all --greedy 12 --submit --tag ens_day1_kd \
        > "$L/ens_day1_kd.out" 2>&1 && grep -a "best pooled\|OOF macro\|WROTE" "$L/ens_day1_kd.out" | tee -a "$Q"
      ;;
  esac
done
wait
python phase3_ens.py status
say "ALL DONE"
