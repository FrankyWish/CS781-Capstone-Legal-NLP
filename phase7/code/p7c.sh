#!/usr/bin/env bash
# Phase 7, part C -- the graded GPU work: seven adaptation techniques on M2's encoder,
# the compulsory LoRMA and KD runs, the efficiency table, then a stacker that can finally
# see TRAINED dense members.
#
#   bash p7c.sh all        train -> kd -> bench -> stack -> zips      (~2h on an L4)
#   bash p7c.sh train      the seven technique runs only
#   bash p7c.sh kd         distillation from the best trained teacher
#   bash p7c.sh stack      re-stack including the trained members
#   bash p7c.sh status
#
# Skip-if-done on each output, so re-running after a Colab drop continues. Each training
# run also embeds train/devx/test, because a stacker member needs a ranking on all three.
set -u

P7=${CS781_P7:-/content/drive/MyDrive/cs781_capstone/phase7}
L=$P7/logs
Q=$L/queuec.txt
STEPS=${P7_STEPS:-600}
BATCH=${P7_BATCH:-4}
NNEG=${P7_NNEG:-3}
CHUNK=${P7_CHUNK:-1024}
MAXCH=${P7_MAXCH:-8}
EB=${P7_EB:-16}
TOPN=${P7_TOPN:-200}
SPLITS=train,devx,test
mkdir -p "$L" "$P7/results" "$P7/models"

say() { printf '[%s] %s\n' "$(date +%H:%M)" "$*" | tee -a "$Q"; }

step() {
  local m=$1 n=$2; shift 2
  if [ -e "$m" ]; then say "skip  $n"; return 0; fi
  say "run   $n"
  if "$@" >"$L/$n.log" 2>&1; then
    say "done  $n"
    grep -aE 'trainable of|/devx |student:|teacher |final loss|ms/query|step .*/' \
         "$L/$n.log" | tail -4 | cut -c1-200 | while read -r l; do say "      $l"; done
    return 0
  fi
  say "FAIL  $n -- tail of $L/$n.log:"
  tail -25 "$L/$n.log" | tee -a "$Q"
  return 1
}

gpu_check() {
  python - <<'PY' || return 1
import sys
import torch
if not torch.cuda.is_available():
    sys.exit("STOP: no GPU. The training runs need an L4.")
cap = torch.cuda.get_device_capability(0)
name = torch.cuda.get_device_name(0)
if cap[0] < 8:
    sys.exit(f"STOP: {name} is SM{cap[0]}{cap[1]}. bf16 is emulated below SM80 and runs "
             f"~30x slow -- switch the runtime to an L4.")
print(f"  {name} SM{cap[0]}{cap[1]}, "
      f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GiB")
PY
}

trainrun() {                   # trainrun <tag> <method> [extra args...]
  local tag=$1 method=$2; shift 2
  step "$P7/cache/emb_${tag}_test_c.npz" "c_${tag}" \
       python phase7_emb.py train --tag "$tag" --method "$method" \
              --splits "$SPLITS" --steps "$STEPS" --batch "$BATCH" --n-neg "$NNEG" \
              --chunk "$CHUNK" --max-chunks "$MAXCH" --embed-batch "$EB" "$@" || return 1
  step "$P7/results/$tag/test_rank.json" "c_${tag}_rank" \
       python phase7_enc.py dense --tag "$tag" --agg maxsim \
              --splits "$SPLITS" --topn "$TOPN" || return 1
}

train() {
  say "======== training: $STEPS steps, batch $BATCH, $NNEG hard negatives, ${MAXCH}x$CHUNK"
  # LoRMA is compulsory, so it goes first and nothing else can starve it of GPU time.
  trainrun ft_lorma_m2   lorma   --rank 8 --alpha 16 || true
  trainrun ft_qlora_m2   qlora   --rank 8 --alpha 16 || true
  trainrun ft_ia3_m2     ia3                        || true
  trainrun ft_partial_m2 partial --train-top 4      || true
  trainrun ft_full_m2    full    --lr 3e-5          || true
  trainrun ft_lora_m2    lora    --rank 8 --alpha 16 || true
  trainrun ft_norm_m2    norm    --lr 5e-4          || true
}

best_teacher() {
  python - "$P7" <<'PY'
import glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phase7_rank as R
best, bf1 = None, -1.0
for d in sorted(glob.glob(f"{sys.argv[1]}/results/ft_*")):
    tag = os.path.basename(d)
    rk, _ = R.load_run(tag, "devx")
    if not rk:
        continue
    try:
        f1 = R.evaluate(rk, "devx")["f1"]
    except Exception:
        continue
    if f1 > bf1:
        best, bf1 = tag, f1
print(best or "ft_lorma_m2")
PY
}

kd() {
  local t
  t=$(best_teacher | tail -1)
  say "======== KD: teacher $t (best trained devx F1), student = every 2nd layer"
  step "$P7/cache/emb_kd_half_test_c.npz" c_kd_half \
       python phase7_emb.py kd --tag kd_half --teacher "$t" --keep-every 2 \
              --method lorma --rank 8 --alpha 16 --kd-alpha 0.7 \
              --splits "$SPLITS" --steps "$STEPS" --batch "$BATCH" --n-neg "$NNEG" \
              --chunk "$CHUNK" --max-chunks "$MAXCH" --embed-batch "$EB" || return 1
  step "$P7/results/kd_half/test_rank.json" c_kd_half_rank \
       python phase7_enc.py dense --tag kd_half --agg maxsim \
              --splits "$SPLITS" --topn "$TOPN" || return 1
  step "$P7/results/kd_efficiency.json" c_bench \
       python phase7_emb.py bench --tags "$t,kd_half" --chunk "$CHUNK" --reps 20 || true
}

stack() {
  say "======== stacking with trained members"
  local trained lex m
  trained=$(python - "$P7" <<'PY'
import glob, os, sys
out = []
for d in sorted(glob.glob(f"{sys.argv[1]}/results/*")):
    t = os.path.basename(d)
    if (t.startswith("ft_") or t.startswith("kd_")) and \
       os.path.exists(f"{d}/train_rank.json") and os.path.exists(f"{d}/test_rank.json"):
        out.append(t)
print(",".join(out))
PY
)
  say "trained members: ${trained:-none}"
  if [ -z "$trained" ]; then say "no trained member has all three splits; skipping"; return 0; fi
  lex=tfidf_sqrt,bm25_hi,bm25,tfidf
  m="$lex,dense_m2_c8_max,dense_m2_c8,dense_m2,dense_m1,dense_m3,dense_m4,$trained"
  step "$P7/results/stack_ft/test_rank.json" c_stack_ft \
       python phase7_stack.py fit --tag stack_ft --kind hybrid --members "$m" \
              --pool-depth 50 --topn "$TOPN" --importance || true
  # model-only, as the brief requires beside every hybrid
  step "$P7/results/stack_ft_model/test_rank.json" c_stack_ft_model \
       python phase7_stack.py fit --tag stack_ft_model --kind model \
              --members "dense_m2_c8_max,dense_m2_c8,dense_m2,dense_m1,dense_m3,dense_m4,$trained" \
              --pool-depth 50 --topn "$TOPN" || true
  step "$P7/results/stack_ft_d20/test_rank.json" c_stack_ft_d20 \
       python phase7_stack.py fit --tag stack_ft_d20 --kind hybrid --members "$m" \
              --pool-depth 20 --topn "$TOPN" || true
  say "half-split selection report"
  python phase7_stack.py half \
         --tags "stack_d100,stack_ft,stack_ft_d20,stack_ft_model,$trained" \
         2>&1 | tee -a "$Q" || true
}

zips() {
  say "======== zips"
  for t in $(python - "$P7" <<'PY'
import glob, os, sys
print(" ".join(os.path.basename(d) for d in sorted(glob.glob(f"{sys.argv[1]}/results/*"))
                if (os.path.basename(d).startswith(("ft_", "kd_", "stack_ft")))
                and os.path.exists(f"{d}/test_rank.json")))
PY
); do
    k=model
    case "$t" in stack_ft|stack_ft_d20) k=hybrid ;; esac
    step "$P7/submissions/$t/submission.zip" "z_$t" \
         python phase7_rank.py zip --tag "$t" --kind "$k" || true
  done
  bash p7b.sh zips 2>&1 | tail -45 | tee -a "$Q"
}

status() {
  python phase7_rank.py show 2>&1 | tee -a "$Q" || true
  say "zips: $(ls -1 "$P7"/submissions/*/submission.zip 2>/dev/null | wc -l)"
}

case "${1:-all}" in
  train)  gpu_check && train ;;
  kd)     gpu_check && kd ;;
  stack)  stack ;;
  zips)   zips ;;
  status) status ;;
  all)    gpu_check || exit 1
          say "======== p7c.sh all | steps $STEPS | chunk ${MAXCH}x$CHUNK | topn $TOPN"
          train; kd; stack; zips; status
          say "======== PART C DONE" ;;
  *)      echo "usage: bash p7c.sh [all|train|kd|stack|zips|status]"; exit 2 ;;
esac
