#!/bin/bash
# Phase 4 GPU queue, ONE job at a time on the L4 (two jobs split the GPU, three
# OOM'd in Phase 3). Ordered by what the grade and the leaderboard need first,
# so if compute runs out part-way, what finished is the most valuable part.
# Re-running is safe: a run whose meta.json exists is skipped.
#
#   0. bench        30 updates each of M4 full / M4 LoRMA / M2 full: measured tokens/s
#                   -> projected minutes for every run printed before it starts
#   1. m4_lorma     LoRMA on M4 (COMPULSORY) + KD-pool predictions (teacher candidate)
#   2. m4_full      full fine-tuning of M4 + KD-pool predictions (teacher candidate)
#   3. kd_m4half    half-depth M4 student distilled from ens(m4_full, m4_lorma) on
#                   8k held-out train docs + the test text (COMPULSORY KD)
#   4. m1_adapter   Houlsby adapters on M1
#   5. m3_adalora   AdaLoRA on M3
#   6. m2_unfreeze  gradual unfreezing of the M2 encoder
#   7. m1_bitfit    BitFit on M1
#   8. m2_partial   top-6 blocks of M2 + layer-wise LR decay
#   9. ensembles -> zips (CPU work, seconds)
#  10. m4_rest      better M4 method on ALL gold docs except the KD pool (~29k docs)
#  11. m4_silver    same on 10k gold + 15k silver docs from the Phase-1 corpus (if built)
#  12. final ensembles
#
# Knobs (environment): P4_DOCS (train docs per run, default 10000),
# P4_ARGS (extra args for every run), P4_ONLY (space-separated tags to run).

cd "$(dirname "$0")"
P4=${CS781_P4:-/content/drive/MyDrive/cs781_capstone/phase4}
L=$P4/logs
mkdir -p "$L"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=$(nproc)
DOCS=${P4_DOCS:-10000}
COMMON="--train-docs $DOCS --kd-pool 8000 --chunks 2 --chunk 512 --batch 8 --accum 2 --evals 3 --submit --save-ckpt ${P4_ARGS:-}"
Q=$L/queue.txt
say() { echo "[$(date +%H:%M)] $*" | tee -a "$Q"; }

project() {   # project <bench tag>: minutes for one run at the measured speed
  python - "$P4/bench/$1.json" "$DOCS" <<'EOF'
import json, sys, os
p, docs = sys.argv[1], int(sys.argv[2])
if os.path.exists(p):
    b = json.load(open(p))
    m = docs * 1024 / b["tok_per_s"] / 60
    print(f"{b['tok_per_s']/1e3:.1f}k tok/s, peak {b['peak_gb']:.1f} GB -> ~{m:.0f} min training + ~5 min eval")
else:
    print("no bench")
EOF
}

run() {
  local tag=$1; shift
  if [ -n "${P4_ONLY:-}" ] && [[ " $P4_ONLY " != *" $tag "* ]]; then return; fi
  if [ -f "$P4/results/$tag/meta.json" ]; then say "skip $tag (done)"; return; fi
  local extra=""
  for attempt in 1 2 3; do
    say "start $tag $extra"
    local n0=$(cat "$L/q_$tag.out" 2>/dev/null | wc -l)
    python phase4_cjpe.py --tag "$tag" $COMMON "$@" $extra >> "$L/q_$tag.out" 2>&1
    if [ -f "$P4/results/$tag/meta.json" ]; then
      say "done  $tag | $(grep -a '=====.*dev macro-F1' "$L/q_$tag.out" | tail -1 | sed 's/^=* *//')"
      return
    fi
    if tail -n +$((n0 + 1)) "$L/q_$tag.out" | grep -aq "OutOfMemoryError\|CUDA out of memory"; then
      say "OOM   $tag (attempt $attempt) -- retrying with --grad-ckpt"
      extra="--grad-ckpt"; sleep 20
    else
      say "FAIL  $tag -- see logs/q_$tag.out"; tail -n 5 "$L/q_$tag.out" | tee -a "$Q"; return
    fi
  done
}

bench() {
  local tag=$1; shift
  [ -f "$P4/bench/$tag.json" ] && return
  python phase4_cjpe.py --tag "$tag" $COMMON --bench --max-steps 30 "$@" >> "$L/q_bench.out" 2>&1 \
    || python phase4_cjpe.py --tag "$tag" $COMMON --bench --max-steps 30 --grad-ckpt "$@" >> "$L/q_bench.out" 2>&1
  say "bench $tag: $(project $tag)"
}

ens() {   # ens <tag> <args>
  local tag=$1; shift
  python phase4_ens.py ens --tag "$tag" "$@" > "$L/$tag.out" 2>&1
  grep -a "dev macro-F1\|WROTE\|teacher\|Error\|Traceback" "$L/$tag.out" | tee -a "$Q"
}

say "queue: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null) | $DOCS docs/run"
python phase4_data.py tokenize >> "$L/q_tokenize.out" 2>&1 || { say "tokenize FAILED"; tail -n 5 "$L/q_tokenize.out"; exit 1; }

bench bench_m4_full  --model M4_LexArcP --method full
bench bench_m4_lorma --model M4_LexArcP --method lorma
bench bench_m2_full  --model M2_LexArcED --method full

run m4_lorma    --model M4_LexArcP  --method lorma --rank 16 --dump-kd
run m4_full     --model M4_LexArcP  --method full --dump-kd
if [ ! -f "$P4/teacher_ens_T1.pt" ]; then
  ens ens_T1 --runs m4_full m4_lorma --teacher
fi
if [ -f "$P4/teacher_ens_T1.pt" ]; then
  run kd_m4half --model M4_LexArcP --method full --keep-every 2 --kd "$P4/teacher_ens_T1.pt" --kd-test
fi
run m1_adapter  --model M1_SC       --method adapter --adapter-dim 64
run m3_adalora  --model M3_LexArcL  --method adalora --rank 16 --ada-target 6
run m2_unfreeze --model M2_LexArcED --method unfreeze --llrd 0.9
run m1_bitfit   --model M1_SC       --method bitfit
run m2_partial  --model M2_LexArcED --method partial --top-k 6 --llrd 0.9

ens ens_model_all --kind model
ens ens_model_greedy --kind model --greedy 8
ens ens_hyb_greedy --greedy 8
ens ens_model_rate --kind model --greedy 8 --rate
# ---- top-score extras (after every required run): more data on the better M4 method
BEST=$(python - "$P4" <<'EOF2'
import json, os, sys
p = sys.argv[1]
f = lambda t: json.load(open(f"{p}/results/{t}/meta.json"))["dev_f1"] if os.path.exists(f"{p}/results/{t}/meta.json") else -1
print("lorma --rank 16" if f("m4_lorma") > f("m4_full") else "full")
EOF2
)
say "extras use M4 method: $BEST"
run m4_rest --model M4_LexArcP --method $BEST --use-rest          # all gold except the KD pool (~3x data)
SILVER=$P4/external/sc_silver.jsonl
if [ -f "$SILVER" ]; then
  run m4_silver --model M4_LexArcP --method $BEST --extra "$SILVER" --extra-docs 15000 --extra-weight 0.5
else
  say "skip m4_silver: $SILVER not built (run phase4_external.py)"
fi
ens ens_final_model --kind model --greedy 8
ens ens_final_hyb --greedy 8
ens ens_final_rate --kind model --greedy 8 --rate
python phase4_ens.py status > /dev/null 2>&1
say "QUEUE DONE"
