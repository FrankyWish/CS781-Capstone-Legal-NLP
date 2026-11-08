#!/bin/bash
# Phase 5 -- everything on ONE L4 runtime. Each command starts itself in the background
# (setsid, survives the cell) and returns at once; progress goes to phase5/logs/queue.txt.
#
#   bash p5.sh goal1    REQUIRED SET. GPU: 6 fine-tuning experiments on gold data
#                       (LoRMA/M4, Adapters/M1, Partial FT/M2, QLoRA/M3, Prefix/M1, KD half-depth student;
#                       + Full FT/M4 if the CPU side is still busy, instead of an idle GPU).
#                       CPU cores, at the same time: data gate, silver scan of the Phase-1 judgments,
#                       TF-IDF gold vs gold+silver. Ends with every zip + SUBMIT_ORDER.md.     ~4.5 h
#   bash p5.sh goal2    TOP SCORE. Retrains the best goal-1 method on gold + silver (if silver won on
#                       dev) with a longer budget, then the final ensembles and zips.          ~2 h
#   bash p5.sh all      goal1 then goal2, unattended.                                   ~6-7 h
#   bash p5.sh round2   2nd long model (M3 QLoRA) + TF-IDF refit on train+dev + char n-grams
#                       + final ensembles + volume variants.                              <2 h
#   bash p5.sh status   what is done, what is running, dev scores so far.
#   bash p5.sh wait     blocks until the running goal finishes (then put runtime.unassign()).
#
# Knobs (env): BUDGET (train minutes per goal-1 run, 25), BUDGET2 (goal-2 run, 90), P5_HOURS (deadline, 6).

cd "$(dirname "$0")"
P5=${CS781_P5:-/content/drive/MyDrive/cs781_capstone/phase5}
L=$P5/logs
mkdir -p "$L"
Q=$L/queue.txt
say() { echo "[$(date +%H:%M)] $*" | tee -a "$Q"; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
J=$(nproc)
CJ=$(( J > 3 ? J - 2 : 1 ))      # leave two cores for the GPU job

cpu_branch() {       # everything that needs no GPU; safe to call twice (skips finished steps)
  pip -q install pyahocorasick > /dev/null 2>&1
  if [ ! -f "$P5/external/candidates.jsonl" ]; then
    say "cpu   silver scan started (Phase-1 SC + HC judgments, ~12 GB to /content)"
    python phase5_external.py scan ${P5_SCAN_ARGS:-} > "$L/01_silver_scan.log" 2>&1
    say "cpu   scan | $(grep -a 'test_frac_found\|dev_frac_found' "$L/01_silver_scan.log" | tr -d ' \n')"
  fi
  if [ -f "$P5/external/candidates.jsonl" ] && [ ! -f "$P5/external/silver.jsonl" ]; then
    python phase5_external.py finalize > "$L/02_silver_finalize.log" 2>&1
    grep -a "labeller on train\|LEAK\|WROTE" "$L/02_silver_finalize.log" | cut -c1-250 | while read -r l; do say "cpu   $l"; done
  fi
  python phase5_tok.py --splits silver --procs "$CJ" >> "$L/03_tok.log" 2>&1
  for spec in "lex_w12|" "lex_w12_sil|--silver 40000 --silver-weight 0.5"; do
    tag=${spec%%|*}; args=${spec#*|}
    [ -f "$P5/results/$tag/meta.json" ] && continue
    [ "$tag" = lex_w12_sil ] && [ ! -f "$P5/external/silver.jsonl" ] && continue
    python phase5_lex.py --tag "$tag" --jobs "$CJ" $args > "$L/$tag.log" 2>&1
    say "cpu   $tag | $(grep -a 'dev best strategy' "$L/$tag.log" | tail -1 | sed 's/^ *//' | cut -c1-160)"
  done
  python - "$P5" <<'EOF' | while read -r l; do say "cpu   $l"; done
import json, os, sys
p = sys.argv[1]
def cv(t):
    f = f"{p}/results/{t}/meta.json"
    return max(json.load(open(f))["thr_cv"].values()) if os.path.exists(f) else None
g, s = cv("lex_w12"), cv("lex_w12_sil")
use = g is not None and s is not None and s >= g + 0.005
open(f"{p}/silver_decision.txt", "w").write("40000" if use else "0")
print(f"silver decision: gold {g} vs gold+silver {s} (dev CV Macro-F1) -> SILVER={'40000' if use else '0'}")
EOF
  say "CPU BRANCH DONE"
}

goal1() {
  say "goal1 started on $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null), $J CPUs"
  [ -f "$P5/results/gate.json" ] || python phase5_data.py gate > "$L/00_gate.log" 2>&1
  say "gate  $(grep -a '^docs:' "$L/00_gate.log" | head -1 | cut -c1-160)"
  say "gate  $(grep -a '^dev.json' "$L/00_gate.log" | head -1 | cut -c1-200)"
  python phase5_tok.py --splits train,dev,test --procs "$J" >> "$L/03_tok.log" 2>&1
  say "tok   $(grep -a '^train:' "$L/03_tok.log" | tail -1 | cut -c1-110)"
  cpu_branch &                                   # CPU cores, alongside the GPU
  CPU=$!
  export P5_DEADLINE=$(( $(date +%s) + ${P5_HOURS:-6} * 3600 ))
  SILVER=0 BUDGET=${BUDGET:-25} ONLY="m4_lorma,m1_adapter,m2_partial,m3_qlora,m1_prefix,kd_m4half" bash run_p5.sh
  if kill -0 $CPU 2>/dev/null; then              # never leave the GPU idle while the CPU side finishes
    say "goal1: CPU branch still running -> Full FT on M4 meanwhile (an extra technique, not idle GPU)"
    SILVER=0 BUDGET=${BUDGET:-25} ONLY="m4_full" bash run_p5.sh
  fi
  say "goal1 GPU runs finished; waiting for the CPU branch"
  wait $CPU
  bash make_zips.sh > "$L/05_zips.log" 2>&1
  cp "$P5/results/STATUS.md" "$P5/STATUS.md" 2>/dev/null
  cp "$P5/results/PLAN_TABLE.md" "$P5/SUBMIT_ORDER.md" 2>/dev/null
  say "GOAL1 DONE -- zips in phase5/submissions, order in phase5/SUBMIT_ORDER.md"
}

goal2() {
  say "goal2 started"
  grep -q "CPU BRANCH DONE" "$Q" 2>/dev/null || cpu_branch
  SIL=$(cat "$P5/silver_decision.txt" 2>/dev/null || echo 0)
  [ -f "$P5/cache/tok_silver.npz" ] || SIL=0
  # best goal-1 neural run by dev CV Macro-F1 -> its trunk + method (+ the method's own flags)
  read -r MODEL METHOD EXTRA <<< "$(python - "$P5" <<'EOF' | tr -d '\r'
import glob, json, sys
best = None
for f in glob.glob(f"{sys.argv[1]}/results/*/meta.json"):
    m = json.load(open(f))
    if m.get("kind") != "neural" or m.get("args", {}).get("keep_every", 1) != 1 or m["tag"].startswith("top_"):
        continue
    cv = max(m["thr_cv"].values())
    if best is None or cv > best[0]:
        best = (cv, m)
m = best[1]
a = m["args"]
extra = {"lorma": f"--rank {a['rank']}", "qlora": f"--rank {a['rank']}", "adapter": f"--bottleneck {a['bottleneck']}",
         "partial": f"--train-top {a['train_top']} --llrd {a['llrd']}", "prefix": f"--n-virtual {a['n_virtual']}"}
print(m["model"], m["method"], extra.get(m["method"], ""))
EOF
)"
  export P5_DEADLINE=$(( $(date +%s) + ${P5_HOURS:-6} * 3600 ))
  TAG="top_${MODEL%%_*}_${METHOD}"
  [ "$SIL" != "0" ] && TAG="${TAG}_sil"
  say "goal2: best goal-1 method = $MODEL $METHOD $EXTRA; SILVER=$SIL -> $TAG (${BUDGET2:-90} min)"
  if [ ! -f "$P5/results/$TAG/meta.json" ]; then
    SA=""; [ "$SIL" != "0" ] && SA="--silver $SIL --silver-weight 0.5"
    python phase5_lsi.py --tag "$TAG" --model "$MODEL" --method "$METHOD" $EXTRA $SA --budget-min "${BUDGET2:-90}" \
        --epochs 3 --chunk 512 --train-chunks 2 --infer-chunks 8 --batch 8 --keep-first >> "$L/q_$TAG.out" 2>&1
    say "done  $TAG | $(grep -a 'DEV global' "$L/q_$TAG.out" | tail -1 | cut -c1-160)"
  fi
  bash make_zips.sh > "$L/06_zips.log" 2>&1
  cp "$P5/results/STATUS.md" "$P5/STATUS.md" 2>/dev/null
  cp "$P5/results/PLAN_TABLE.md" "$P5/SUBMIT_ORDER.md" 2>/dev/null
  say "GOAL2 DONE -- order in phase5/SUBMIT_ORDER.md"
}

all() { goal1; goal2; say "ALL DONE"; }

round2() {           # under 2 h on one L4: a 2nd long-trained model + stronger TF-IDF + volume variants
  say "round2 started on $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null), $J CPUs"
  local T0=$(date +%s)
  local LEXMIN=${LEXMIN:-85}           # CPU jobs are killed after this many minutes
  python phase5_tok.py --splits train,dev,test --procs "$J" >> "$L/03_tok.log" 2>&1
  lexjob() { local tag=$1; shift
    [ -f "$P5/results/$tag/meta.json" ] && return
    timeout "${LEXMIN}m" python phase5_lex.py --tag "$tag" --jobs $(( CJ / 2 > 0 ? CJ / 2 : 1 )) "$@" > "$L/$tag.log" 2>&1
    say "cpu   $tag | $(grep -a 'dev best strategy' "$L/$tag.log" | tail -1 | sed 's/^ *//' | cut -c1-150)"
  }
  lexjob lex_w12r --refit-dev &        # same TF-IDF, test scores from a train+dev refit (+23% data)
  local C1=$!
  lexjob lex_char --char &             # + char_wb 3-5 grams
  local C2=$!
  if [ ! -f "$P5/results/top_M3_qlora/meta.json" ]; then
    say "start top_M3_qlora (${BUDGET2:-50} min training)"
    python phase5_lsi.py --tag top_M3_qlora --model M3_LexArcL --method qlora --rank 16 --budget-min "${BUDGET2:-50}" \
        --epochs 3 --chunk 512 --train-chunks 2 --infer-chunks 8 --batch 8 --keep-first >> "$L/q_top_M3_qlora.out" 2>&1
    say "done  top_M3_qlora | $(grep -a 'DEV global' "$L/q_top_M3_qlora.out" | tail -1 | cut -c1-150)"
  fi
  wait $C1 $C2
  say "round2: all models done after $(( ($(date +%s) - T0) / 60 )) min; building ensembles"
  GREEDY=8 bash make_zips.sh > "$L/09_zips_r2.log" 2>&1
  for r in ens_hyb_z ens_model; do
    [ -f "$P5/results/$r/meta.json" ] && python phase5_ens.py rate --run $r --factors 0.85,1.15 >> "$L/09_zips_r2.log" 2>&1
  done
  python phase5_ens.py status > /dev/null 2>&1
  python phase5_ens.py plan > /dev/null 2>&1
  cp "$P5/results/STATUS.md" "$P5/STATUS.md" 2>/dev/null
  cp "$P5/results/PLAN_TABLE.md" "$P5/SUBMIT_ORDER.md" 2>/dev/null
  grep -a "ens_hyb_z:\|ens_model:\|ens_hyb:\|_v85\|_v115" "$L/09_zips_r2.log" | grep -a "dev\|statutes" | cut -c1-170 | while read -r l; do say "zips  $l"; done
  say "ROUND2 DONE after $(( ($(date +%s) - T0) / 60 )) min -- order in phase5/SUBMIT_ORDER.md"
}

case "$1" in
  goal1|goal2|all|round2)
    if [ -z "$P5_CHILD" ]; then
      P5_CHILD=1 setsid nohup bash "$0" "$1" > "$L/$1_stdout.txt" 2>&1 < /dev/null &
      echo "$1 started in the background (pid $!). Watch: bash p5.sh status"
      exit 0
    fi
    echo $$ > /tmp/p5_job.pid                      # wait polls exactly this process
    "$1" ;;
  status)
    tail -n 30 "$Q" 2>/dev/null
    for f in $(ls -t "$L"/live_*.txt 2>/dev/null | head -2); do echo "-- $(basename "$f"): $(tail -n 1 "$f")"; done
    ps aux | grep -q "[p]hase5_lsi.py" && echo "-- GPU job running: $(ps aux | grep '[p]hase5_lsi.py' | grep -o 'tag [^ ]*' | head -1)"
    python phase5_ens.py status 2>/dev/null | sed -n '3,40p' ;;
  wait)              # blocks while the started goal is alive (cap 9 h), then prints the end of the log
    sleep 20
    pid=$(cat /tmp/p5_job.pid 2>/dev/null)
    if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
      echo "WAIT: no running p5.sh job found (pid file: ${pid:-missing}). Not waiting."
      tail -n 8 "$Q" 2>/dev/null
      exit 1
    fi
    echo "waiting on pid $pid (checks every 2 min, cap 9 h)"
    t0=$(date +%s)
    while kill -0 "$pid" 2>/dev/null && [ $(( $(date +%s) - t0 )) -lt 32400 ]; do sleep 120; done
    tail -n 15 "$Q"
    [ -f "$P5/SUBMIT_ORDER.md" ] && cat "$P5/SUBMIT_ORDER.md"
    exit 0 ;;
  *) sed -n '2,15p' "$0" ;;
esac
