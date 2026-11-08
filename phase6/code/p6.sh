#!/bin/bash
# Phase 6 -- the whole phase in ONE foreground cell on ONE L4.
#
#   bash p6.sh all       everything, unattended, ends with STATUS + SUBMIT_ORDER.md   ~6-7 h
#   bash p6.sh status    what exists, what the numbers are
#   bash p6.sh stage1    CPU-only part (gate, extractive system, length calibration)
#   bash p6.sh gpu       just the GPU queue        bash p6.sh ens   just stage 3
#
# Every step is skipped if its output exists and every GPU run resumes from its
# last 10-minute checkpoint, so after a Colab disconnect you re-run the SAME cell
# and it carries on.  No setsid / background jobs (Phase 5 lost four runs to them);
# the one `&` below is a child of this script and the script waits for it.
#
# Order of `all`:
#   prep (CPU, serial, ~40 min)   gate -> sentence scorer -> input views -> tokenise
#   then, side by side:
#     GPU  the 7 required runs (LoRMA/M4, LoRA/M2, Houlsby/M1, QLoRA/M3, AdaLoRA/M2,
#          Pfeiffer/M3, KD into a half-depth M1 from the best of them), then extras
#          while time remains (full FT on M4, the long-document comparison)
#     CPU  baselines, length calibration, extractive zips, the other views
#   ens  MBR + reranker + length bracket + every zip + SUBMIT_ORDER.md
#
# Knobs (env): BUDGET train minutes per run (25) · P6_HOURS GPU-queue deadline (6.5)
#   P6_GEN_DEV decoded dev docs per run (200) · P6_NO_BERT=1 ROUGE only · P6_NO_PIP=1
set -u
cd "$(dirname "$0")"
P6=${CS781_P6:-/content/drive/MyDrive/cs781_capstone/phase6}
L=$P6/logs
mkdir -p "$L" "$P6/results" "$P6/views" "$P6/submissions" "$P6/cache" "$P6/ckpt"
Q=$L/queue.txt
say() { echo "[$(date +%H:%M)] $*" | tee -a "$Q"; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
J=${P6_PROCS:-$(nproc)}
CJ=$(( J > 3 ? J - 2 : 1 ))          # leave two cores to feed the GPU
NB=""; [ -n "${P6_NO_BERT:-}" ] && NB="--no-bert"
B=${BUDGET:-25}

step() {            # step <marker-file> <log-name> <command...>
  local mark=$1 name=$2; shift 2
  if [ -e "$mark" ]; then say "skip  $name (exists)"; return 0; fi
  say "run   $name"
  if "$@" > "$L/$name.log" 2>&1; then
    say "done  $name"
  else
    say "FAIL  $name -- tail of $L/$name.log:"
    tail -20 "$L/$name.log" | tee -a "$Q"
    return 1
  fi
}

deps() {
  if [ -n "${P6_NO_PIP:-}" ]; then say "pip   skipped (P6_NO_PIP)"; return 0; fi
  if ! python -c "import rouge_score, bert_score" 2>/dev/null; then
    say "pip   rouge-score bert-score (bert_score downloads roberta-large, ~1.4 GB)"
    pip -q install rouge-score bert-score > "$L/pip.log" 2>&1
  fi
}

gate_summary() {
  for k in 'train fields' 'docs:' 'train doc words' 'train summ words' 'test  doc words' 'summary inside' \
           'novel unigrams' 'duplicates:' 'extractive oracle' 'length model' 'window probe' ; do
    grep -a -m7 "$k" "$L/00_gate.log" | cut -c1-230 | while read -r l; do say "gate  $l"; done
  done
  grep -a '!!' "$L/00_gate.log" | while read -r l; do say "gate  $l"; done
}

# ------------------------------------------------------------------- prep ---
prep() {
  deps
  step "$P6/results/gate.json" 00_gate python phase6_data.py gate || return 1
  gate_summary
  step "$P6/results/sel_model.pkl" 01_sel_fit \
       python phase6_sel.py --procs "$J" fit --docs "${P6_FIT_DOCS:-3000}" || return 1
  step "$P6/views/sel1024_test.json" 02_views \
       python phase6_sel.py --procs "$J" view --name sel1024 --words "${P6_VIEW_WORDS:-850}" \
              --split fit,dev,test || return 1
  step "$P6/cache/tok_sel1024_test.npz" 03_tok \
       python phase6_sum.py tok --view sel1024 --procs "$J" || return 1
  grep -a "^\[.*\] fit/" "$L/03_tok.log" | cut -c1-230 | while read -r l; do say "tok   $l"; done
}

# ------------------------------------------------------------- CPU branch ---
cpu_rest() {
  step "$P6/results/baselines_dev.json" 10_baselines \
       python phase6_sel.py --procs "$CJ" baselines --split dev $NB || true
  grep -a -E '^\[.*\]\s+(lead|centroid):' "$L/10_baselines.log" | cut -c1-200 | while read -r l; do say "base  $l"; done
  step "$P6/results/length_calib.json" 11_calib \
       python phase6_sel.py --procs "$CJ" calib --bert "${P6_BERT:-200}" $NB || return 1
  grep -a 'BEST by\|!!' "$L/11_calib.log" | cut -c1-230 | while read -r l; do say "calib $l"; done
  read -r BLEN BMULT <<< "$(python - "$P6" <<'EOF'
import json, sys
c = json.load(open(sys.argv[1] + "/results/length_calib.json"))["chose"]
print(c["len"], c["mult"])
EOF
)"
  say "calib chose len=$BLEN mult=$BMULT"
  step "$P6/results/sel_best/summaries_test.json" 12_sel_best \
       python phase6_sel.py --procs "$CJ" predict --tag sel_best --split dev,test \
              --len "$BLEN" --mult "$BMULT" $NB || return 1
  grep -a 'sel_best dev' "$L/12_sel_best.log" | cut -c1-200 | while read -r l; do say "sel   $l"; done
  # one shorter and one longer variant: dev picks a point, the leaderboard settles length
  SHORT=$(python -c "print(round($BMULT*0.85,3))")
  LONG=$(python -c "print(round($BMULT*1.15,3))")
  for spec in "sel_short|$SHORT" "sel_long|$LONG"; do
    t=${spec%%|*}; m=${spec#*|}
    step "$P6/results/$t/summaries_test.json" "13_$t" \
         python phase6_sel.py --procs "$CJ" predict --tag "$t" --split dev,test \
                --len "$BLEN" --mult "$m" --no-bert || true
  done
  for t in sel_best sel_short sel_long ; do
    [ -f "$P6/results/$t/summaries_test.json" ] || continue
    step "$P6/submissions/$t/submission.zip" "14_zip_$t" python phase6_sel.py zip --tag "$t" || true
  done
  # views for the long-document comparison (extras at the end of the GPU queue)
  step "$P6/views/sel2048_test.json" 15_views_long \
       python phase6_sel.py --procs "$CJ" view --name sel2048 --words 1700 --split fit,dev,test || true
  step "$P6/cache/tok_sel2048_test.npz" 16_tok_long python phase6_sum.py tok --view sel2048 --procs "$CJ" || true
  step "$P6/cache/tok_trunc_test.npz" 17_tok_trunc python phase6_sum.py tok --view trunc --procs "$CJ" || true
  say "CPU BRANCH DONE"
}

# -------------------------------------------------------------- GPU queue ---
sum_tokens() {      # summary token budget = p90 of the reference length, 256..1024
  python - "$P6" <<'EOF'
import json, sys, numpy as np
z = np.load(sys.argv[1] + "/cache/tok_sel1024_fit.npz")
p90 = json.loads(str(z["meta"]))["summ_tok"]["p90"]
print(int(min(1024, max(256, 64 * round(p90 / 64)))))
EOF
}

late() { [ "$(date +%s)" -gt "$DEADLINE" ]; }

run() {             # run <tag> <args...>
  local tag=$1; shift
  if [ -f "$P6/results/$tag/meta.json" ]; then say "skip  $tag (done)"; return 0; fi
  if late; then say "skip  $tag (GPU deadline passed)"; return 0; fi
  local extra=""
  for attempt in 1 2 3; do
    say "start $tag (attempt $attempt)"
    python phase6_sum.py train --tag "$tag" $COMMON "$@" $extra >> "$L/q_$tag.out" 2>&1
    if [ -f "$P6/results/$tag/meta.json" ]; then
      say "done  $tag | $(grep -a 'DONE ' "$L/q_$tag.out" | tail -1 | cut -c12-200)"
      return 0
    fi
    if tail -50 "$L/q_$tag.out" | grep -aq "OutOfMemoryError\|CUDA out of memory"; then
      say "OOM   $tag -- retrying with half the micro-batch"; extra="--batch 2 --accum 8"; sleep 30
    else
      say "FAIL  $tag -- tail of logs/q_$tag.out:"; tail -8 "$L/q_$tag.out" | tee -a "$Q"; return 1
    fi
  done
}

best_teacher() {    # strongest finished required run by held-out score
  python - "$P6" <<'EOF'
import json, os, sys
p = sys.argv[1]
best = None
for t in ["m4_lorma", "m2_lora", "m1_houlsby", "m3_qlora", "m2_adalora", "m3_pfeiffer"]:
    f = f"{p}/results/{t}/meta.json"
    if os.path.exists(f) and os.path.exists(f"{p}/ckpt/{t}_final.pt"):
        d = json.load(open(f))["dev"]
        s = d.get("bertscore", d.get("rouge1"))
        if best is None or s > best[1]:
            best = (t, s)
print(best[0] if best else "")
EOF
}

gpu() {
  local gpu_name
  gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
  say "gpu   $gpu_name"
  case "$gpu_name" in *T4*) say "STOP: T4 runtime (bf16 emulated, ~30x slow). Switch to L4."; return 1 ;; esac
  DEADLINE=${P6_DEADLINE:-$(( $(date +%s) + $(python -c "print(int(${P6_HOURS:-6.5}*3600))") ))}
  local S; S=${P6_SUM_TOKENS:-$(sum_tokens)}
  local DT=${P6_DOC_TOKENS:-1152}
  say "gpu   summary budget $S tokens (p90 of the references), prompt $DT tokens, $B train-min/run"
  COMMON="--budget-min $B --sum-tokens $S --doc-tokens $DT --gen-dev ${P6_GEN_DEV:-200} $NB ${P6_ARGS:-}"
  run m4_lorma    --model M4_LexArcP  --method lorma    --rank 16
  run m2_lora     --model M2_LexArcED --method lora     --rank 16 --targets attn
  run m1_houlsby  --model M1_SC       --method houlsby  --bottleneck 64
  run m3_qlora    --model M3_LexArcL  --method qlora    --rank 16
  run m2_adalora  --model M2_LexArcED --method adalora  --rank 12 --ada-target 6
  run m3_pfeiffer --model M3_LexArcL  --method pfeiffer --bottleneck 128
  local T; T=$(best_teacher)
  if [ -n "$T" ]; then
    say "kd    teacher = $T (best held-out score of the finished required runs)"
    run kd_m1half --model M1_SC --method full --keep-every 2 --teacher "$T" --batch 2 --accum 8
  else
    say "kd    no finished teacher -- KD skipped"
  fi
  # extras, only while the deadline allows: a spare technique and the long-document axis
  run m4_full     --model M4_LexArcP  --method full
  # the long-document views are tokenised by the CPU branch; wait for it (<=30 min)
  # rather than skip the comparison because the GPU got here first
  for _ in $(seq 360); do
    { [ -f "$P6/cache/tok_sel2048_test.npz" ] && [ -f "$P6/cache/tok_trunc_test.npz" ]; } && break
    grep -aq "CPU BRANCH DONE" "$Q" && break
    late && break
    sleep 5
  done
  [ -f "$P6/cache/tok_sel2048_test.npz" ] && \
    run m4_lorma_2048  --model M4_LexArcP --method lorma --rank 16 --view sel2048 --doc-tokens $(( DT * 16 / 9 ))
  [ -f "$P6/cache/tok_trunc_test.npz" ] && \
    run m4_lorma_trunc --model M4_LexArcP --method lorma --rank 16 --view trunc
  say "GPU QUEUE DONE"
}

ens() {
  step "$P6/SUBMIT_ORDER.md.done" 30_ens python phase6_ens.py all $NB || true
  [ -f "$P6/SUBMIT_ORDER.md" ] && touch "$P6/SUBMIT_ORDER.md.done"
  grep -a -E 'single|mbr_|rank_|wrote' "$L/30_ens.log" | cut -c1-200 | while read -r l; do say "ens   $l"; done
}

all() {
  say "======== p6.sh all | $J cores"
  prep || { say "PREP FAILED -- nothing else can run; fix and re-run the same cell"; return 1; }
  cpu_rest > "$L/cpu_branch.out" 2>&1 &
  local CPU=$!
  gpu
  say "waiting for the CPU branch"
  wait $CPU
  rm -f "$P6/SUBMIT_ORDER.md.done"          # new members -> rebuild the ensembles
  ens
  status
  say "======== ALL DONE"
}

stage1() { prep && cpu_rest && status; }

status() {
  python - "$P6" <<'EOF'
import glob, json, os, sys
P6 = sys.argv[1]
g = f"{P6}/results/gate.json"
if os.path.exists(g):
    G = json.load(open(g))
    print(f"gate: train {G['n_docs']['train']} test {G['n_docs']['test']} | "
          f"summ words p10/p50/p90 {G['train_summ_words']['p10']:.0f}/{G['train_summ_words']['p50']:.0f}/"
          f"{G['train_summ_words']['p90']:.0f} | doc words p50 {G['train_doc_words']['p50']:.0f}")
    s = G["summary_in_document"]
    print(f"leak: 8-gram copy p50 {s['ngram8_copy_rate']['p50']:.3f} | >50% copied in "
          f"{s['docs_over_half_copied']}/{s['sampled']} docs | train copies of test docs "
          f"{G['duplicates']['train_docs_copying_a_test_doc']}")
    print(f"extractive oracle R1 {G['extractive']['oracle_rouge1']['mean']:.4f} | "
          f"lead R1 {G['extractive']['lead_rouge1']['mean']:.4f} | "
          f"BERTScore saturates at {G.get('bert_window', {}).get('saturates_at_words')} words")
else:
    print("gate: not run yet")
rows = []
for m in glob.glob(f"{P6}/results/*/meta.json") + glob.glob(f"{P6}/results/*/ens.json"):
    d = json.load(open(m))
    dv = d.get("dev") or {}
    rows.append((dv.get("bertscore"), dv.get("rouge1"), d.get("tag"), d.get("kind"), dv.get("pred_words")))
print("held-out scores (neural runs: their decoded 200-doc subset; extractive: all of dev):")
for b, r1, tag, kind, w in sorted(rows, key=lambda r: -((r[0] or 0) * 10 + (r[1] or 0))):
    fb = "  -   " if b is None else f"{b:.4f}"
    fr = "  -   " if r1 is None else f"{r1:.4f}"
    print(f"  {tag:18s} {kind:10s} bert {fb} rouge1 {fr} words {w and round(w)}")
z = sorted(glob.glob(f"{P6}/submissions/_upload/*.zip"))
print(f"upload-ready zips: {len(z)}  (order: {P6}/SUBMIT_ORDER.md)")
EOF
}

case "${1:-}" in
  all) all ;;
  stage1) stage1 ;;
  gpu) prep && gpu ;;
  ens) rm -f "$P6/SUBMIT_ORDER.md.done"; ens ;;
  status) status ;;
  *) sed -n '2,25p' "$0" ;;
esac
