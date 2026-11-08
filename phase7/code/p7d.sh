#!/usr/bin/env bash
# Phase 7, part D -- the three levers left, in the order of cost.
#
#   bash p7d.sh all      everything below, one blocking run        (~2.5h on an L4)
#   bash p7d.sh cpu      deep oracle + MaxP + RM3                  (~40min, no GPU)
#   bash p7d.sh refit    the two technique runs that OOM'd in p7c  (~25min, GPU)
#   bash p7d.sh ce       the cross-encoder reranker, cross-fitted  (~70min, GPU)
#   bash p7d.sh stack    re-stack with the new members             (~10min, CPU)
#   bash p7d.sh zips     rebuild every zip + SUBMIT_ORDER.md
#   bash p7d.sh status
#
# Why these three and nothing else.  The 27 paired leaderboard scores put test at
# devx - 0.0095 (sd 0.015) and the devx oracle@100 at 0.5481 against an achieved 0.4158.
# So there are ~100 reachable hits inside the pool we already retrieve, in the wrong
# order, and roughly 60 more outside it.  MaxP and RM3 go after the second number
# (retrieval), the cross-encoder after the first (ordering).  The refits only close the
# technique table the report needs -- they are not expected to move the score.
#
# Skip-if-done on each output, so a Colab drop continues where it stopped.
set -u

P7=${CS781_P7:-/content/drive/MyDrive/cs781_capstone/phase7}
L=$P7/logs
Q=$L/queued.txt
TOPN=${P7_TOPN:-200}
# 1000 steps x batch 2 is ~2.5 epochs over the 816 usable train queries.  The scalar
# relevance head starts from nothing -- unlike the bi-encoder, which started from an
# already usable cosine space -- so it needs more steps than p7c's 600, not fewer.
CE_STEPS=${P7_CE_STEPS:-1000}
CE_DEPTH=${P7_CE_DEPTH:-50}
CE_FOLDS=${P7_CE_FOLDS:-3}
CE_BASE=${P7_CE_BASE:-tfidf_sqrt}
SPLITS=train,devx,test
mkdir -p "$L" "$P7/results" "$P7/models" "$P7/submissions"

# leak-free members only.  At matched pool depth the six trained bi-encoders added 0 hits
# (depth 20) and +4 (depth 50) against an 11-hit 2-SE bar, so they buy nothing here -- and
# their train-side columns are in-sample, since they were trained on the queries the
# stacker is fitted on.  The cross-encoder is cross-fitted, so it is the one trained
# member that may join this list.
CLEAN=tfidf_sqrt,bm25_hi,bm25,tfidf,dense_m2_c8_max,dense_m2_c8,dense_m2_max,dense_m2,dense_m1,dense_m3,dense_m4
LEX4=tfidf_sqrt,bm25_hi,bm25,tfidf
NEWLEX=maxp_tfidf,maxp_bm25,rm3_tfidf,rm3_maxp

say() { printf '[%s] %s\n' "$(date +%H:%M)" "$*" | tee -a "$Q"; }

step() {
  local m=$1 n=$2; shift 2
  if [ -e "$m" ]; then say "skip  $n"; return 0; fi
  say "run   $n"
  if "$@" >"$L/$n.log" 2>&1; then
    say "done  $n"
    grep -aE 'BEST|/devx |/train |train OOF|oracle@|trainable of|final loss|top1|scored ' \
         "$L/$n.log" | tail -5 | cut -c1-200 | while read -r l; do say "      $l"; done
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
cap, name = torch.cuda.get_device_capability(0), torch.cuda.get_device_name(0)
if cap[0] < 8:
    sys.exit(f"STOP: {name} is SM{cap[0]}{cap[1]}. bf16 is emulated below SM80 and runs "
             f"~30x slow -- switch the runtime to an L4.")
print(f"  {name} SM{cap[0]}{cap[1]}, "
      f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GiB")
PY
}

chosen_passages() {                # the width maxp's sweep picked, 500 until it has run
  python - "$P7" <<'PY'
import json, os, sys
f = f"{sys.argv[1]}/results/maxp_tfidf_sweep.json"
try:
    print(int(json.load(open(f))["chose"]["passages"]))
except Exception:
    print(500)
PY
}

# ------------------------------------------------------------------------- CPU work --
cpu() {
  say "======== deep oracle: how much headroom is left, and at what depth"
  # the saved runs are only 200 deep, so oracle@500/1000 needs a deeper run first
  step "$P7/results/tfidf_deep/devx_rank.json" d_tfidf_deep \
       python phase7_rank.py tfidf --tag tfidf_deep --qtf sqrt --max-df 0.2 \
              --splits devx --topn 1000 || true
  step "$P7/results/oracle_depth.json" d_oracle \
       python phase7_rerank.py oracle --tags tfidf_deep,tfidf_sqrt,stack_d100,bm25_hi \
              --depths 7,20,50,100,200,500,1000 || true

  say "======== MaxP: score a judgment by its best passage, not its average"
  step "$P7/results/maxp_tfidf/test_rank.json" d_maxp_tfidf \
       python phase7_rank.py maxp_tfidf --tag maxp_tfidf --sweep --qtf sqrt \
              --max-df 0.2 --splits "$SPLITS" --topn "$TOPN" || true
  local pw
  pw=$(chosen_passages | tail -1)
  say "      passage width chosen: $pw"
  step "$P7/results/maxp_bm25/test_rank.json" d_maxp_bm25 \
       python phase7_rank.py maxp --tag maxp_bm25 --passages "$pw" --k1 4.0 --b 1.0 \
              --qtf sqrt --max-df 0.2 --splits "$SPLITS" --topn "$TOPN" || true

  say "======== RM3: expand the query with the relevance model of its own top hits"
  step "$P7/results/rm3_tfidf/test_rank.json" d_rm3_tfidf \
       python phase7_rank.py rm3 --tag rm3_tfidf --sweep --qtf sqrt --max-df 0.2 \
              --splits "$SPLITS" --topn "$TOPN" || true
  step "$P7/results/rm3_maxp/test_rank.json" d_rm3_maxp \
       python phase7_rank.py rm3 --tag rm3_maxp --sweep --passages "$pw" --qtf sqrt \
              --max-df 0.2 --splits "$SPLITS" --topn "$TOPN" || true
}

# ----------------------------------------------------- the two p7c runs that OOM'd ---
refit() {
  say "======== refits: full and lora OOM'd at batch 4; batch 2 halves the activations"
  say "         (reported with that caveat -- the other five ran at batch 4)"
  for spec in "ft_full_m2 full 3e-5" "ft_lora_m2 lora 1e-4"; do
    set -- $spec
    local tag=$1 method=$2 lr=$3
    step "$P7/cache/emb_${tag}_test_c.npz" "d_${tag}" \
         python phase7_emb.py train --tag "$tag" --method "$method" --lr "$lr" \
                --splits "$SPLITS" --steps 600 --batch 2 --n-neg 3 \
                --chunk 1024 --max-chunks 8 --embed-batch 16 \
                $([ "$method" = lora ] && echo --rank 8 --alpha 16) || continue
    step "$P7/results/$tag/test_rank.json" "d_${tag}_rank" \
         python phase7_enc.py dense --tag "$tag" --agg maxsim \
                --splits "$SPLITS" --topn "$TOPN" || true
  done
}

# --------------------------------------------------------------- the cross-encoder ---
# Re-score devx and test from the SAVED final checkpoint, without retraining.
#
# The first ce run trained all four models (26 min), scored devx and test in full (17
# min), and then died in `write` because the cross-fitted TRAIN ranking cannot cover the
# 11 train queries that were never trainable -- and `write` evaluated train before saving
# devx and test. The checkpoint and the train ranking were already on disk, so only the
# scoring needs redoing: ~17 min instead of ~82. `write` now saves every split before it
# evaluates anything, so this cannot recur.
cescore() {
  say "======== re-scoring devx+test from models/ce_lorma.pt -- no retraining"
  step "$P7/results/ce_lorma/test_rank.json" d_ce_score        python phase7_rerank.py score --tag ce_lorma --method lorma --rank 8 --alpha 16               --base "$CE_BASE" --depth "$CE_DEPTH"               --q-tok 384 --windows 3 --chunk 1024 --score-batch 24               --splits devx,test --topn "$TOPN" || return 1
}

ce() {
  say "======== cross-encoder: $CE_FOLDS cross-fit folds + 1 final model over $CE_BASE top-$CE_DEPTH"
  say "         the folds exist so the stacker's train-side feature is OUT of sample"
  say "         hard negatives are mined from $CE_BASE too: a reranker should be trained"
  say "         against the distractors its own first stage will actually hand it"
  step "$P7/results/ce_lorma/test_rank.json" d_ce_lorma \
       python phase7_rerank.py fit --tag ce_lorma --method lorma --rank 8 --alpha 16 \
              --base "$CE_BASE" --neg-tag "$CE_BASE" \
              --depth "$CE_DEPTH" --folds "$CE_FOLDS" \
              --q-tok 384 --windows 3 --chunk 1024 --score-batch 24 \
              --steps "$CE_STEPS" --batch 2 --n-neg 7 --temp 1.0 \
              --splits "$SPLITS" --topn "$TOPN" || return 1
}

ce2() {                            # only worth it if MaxP beat tfidf_sqrt
  local b=${1:-maxp_tfidf}
  say "======== second cross-encoder over $b (run this only if $b won on devx)"
  step "$P7/results/ce_lorma2/test_rank.json" d_ce_lorma2 \
       python phase7_rerank.py fit --tag ce_lorma2 --method lorma --rank 8 --alpha 16 \
              --base "$b" --depth "$CE_DEPTH" --folds "$CE_FOLDS" \
              --q-tok 384 --windows 3 --chunk 1024 --score-batch 24 \
              --steps "$CE_STEPS" --batch 2 --n-neg 7 --temp 1.0 \
              --splits "$SPLITS" --topn "$TOPN" || true
}

# ------------------------------------------------------------------------ stacking ---
stack() {
  say "======== stacking: leak-free members, plus the cross-fitted reranker"
  local have ce new
  have=$(python - "$P7" "$NEWLEX,ce_lorma,ce_lorma_mix,ce_lorma2" <<'PY'
import os, sys
P7 = sys.argv[1]
out = [t for t in sys.argv[2].split(",") if t and all(
    os.path.exists(f"{P7}/results/{t}/{s}_rank.json") for s in ("train", "devx", "test"))]
print(",".join(out))
PY
)
  say "new members with all three splits: ${have:-none}"
  new=$(python - "$have" <<'PY'
import sys
print(",".join(t for t in sys.argv[1].split(",") if t.startswith(("maxp", "rm3"))))
PY
)
  ce=$(python - "$have" <<'PY'
import sys
print(",".join(t for t in sys.argv[1].split(",") if t.startswith("ce_")))
PY
)
  # 1. new lexical only -- isolates what MaxP and RM3 add with no neural member at all
  [ -n "$new" ] && step "$P7/results/stack_lex2/test_rank.json" d_stack_lex2 \
       python phase7_stack.py fit --tag stack_lex2 --kind lexical \
              --members "$LEX4,$new" --pool-depth 50 --topn "$TOPN" --importance || true
  # 2. the stack_d100 member set plus the new lexical systems
  [ -n "$new" ] && step "$P7/results/stack_new/test_rank.json" d_stack_new \
       python phase7_stack.py fit --tag stack_new --kind hybrid \
              --members "$CLEAN,$new" --pool-depth 100 --topn "$TOPN" || true
  # 3. everything leak-free, including the reranker
  if [ -n "$ce" ]; then
    step "$P7/results/stack_ce/test_rank.json" d_stack_ce \
         python phase7_stack.py fit --tag stack_ce --kind hybrid \
                --members "$CLEAN${new:+,$new},$ce" --pool-depth 100 --topn "$TOPN" \
                --importance || true
    step "$P7/results/stack_ce_lex/test_rank.json" d_stack_ce_lex \
         python phase7_stack.py fit --tag stack_ce_lex --kind hybrid \
                --members "$LEX4${new:+,$new},$ce" --pool-depth 50 --topn "$TOPN" || true
  fi
  say "half-split selection report (choose on A, report on B, paired 2 SE bar)"
  python phase7_stack.py half \
         --tags "stack_d100,stack_lex2,stack_new,stack_ce,stack_ce_lex${have:+,$have}" \
         2>&1 | tee -a "$Q" || true
}

# ---------------------------------------------------------------------------- zips ---
zips() {
  say "======== zips"
  for t in maxp_tfidf maxp_bm25 rm3_tfidf rm3_maxp stack_lex2; do
    [ -d "$P7/results/$t" ] || continue
    step "$P7/submissions/$t/submission.zip" "z_$t" \
         python phase7_rank.py zip --tag "$t" --kind lexical || true
  done
  for t in ce_lorma ce_lorma_mix ce_lorma2 ce_lorma2_mix stack_new stack_ce \
           stack_ce_lex; do
    [ -d "$P7/results/$t" ] || continue
    step "$P7/submissions/$t/submission.zip" "z_$t" \
         python phase7_rank.py zip --tag "$t" --kind hybrid || true
  done
  for t in ft_full_m2 ft_lora_m2; do
    [ -d "$P7/results/$t" ] || continue
    step "$P7/submissions/$t/submission.zip" "z_$t" \
         python phase7_rank.py zip --tag "$t" --kind model || true
  done
  bash p7b.sh zips 2>&1 | tail -60 | tee -a "$Q"
}

status() {
  python phase7_rank.py show 2>&1 | tee -a "$Q" || true
  say "zips: $(ls -1 "$P7"/submissions/*/submission.zip 2>/dev/null | wc -l)"
}

case "${1:-all}" in
  cpu)    cpu ;;
  refit)  gpu_check && refit ;;
  ce)      gpu_check && ce ;;
  cescore) gpu_check && cescore ;;
  ce2)    gpu_check && ce2 "${2:-maxp_tfidf}" ;;
  stack)  stack ;;
  zips)   zips ;;
  status) status ;;
  all)    say "======== p7d.sh all | ce $CE_STEPS steps over $CE_BASE top-$CE_DEPTH, $CE_FOLDS folds"
          cpu
          if gpu_check; then refit; ce; else say "no L4: skipping the GPU stages"; fi
          stack; zips; status
          say "======== PART D DONE" ;;
  *)      echo "usage: bash p7d.sh [all|cpu|refit|ce|cescore|ce2|stack|zips|status]"; exit 2 ;;
esac
