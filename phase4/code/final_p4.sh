#!/bin/bash
# Phase 4 unattended finish. Run AFTER run_p4.sh has been started (it waits for it):
#
#   1. TF-IDF zips on CPU, in the background (lex_all, lex_tail, lex_tail_ext, section evidence)
#   2. wait for run_p4.sh to finish; if it died before its final ensembles, restart it once (resumes)
#   3. m4_final: the better M4 method (LoRMA or full, by dev) on ALL gold docs + silver docs
#   4. final ensembles (greedy model-only, greedy hybrid, rate-matched) -> zips
#   5. STATUS.md + SUBMIT_ORDER.txt on Drive
#
# A deadline (P4_HOURS, default 7) stops new GPU work, so a stuck run cannot eat the compute units.

cd "$(dirname "$0")"
P4=${CS781_P4:-/content/drive/MyDrive/cs781_capstone/phase4}
L=$P4/logs
Q=$L/queue.txt
SILVER=$P4/external/sc_silver.jsonl
mkdir -p "$L"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
say() { echo "[$(date +%H:%M)] $*" | tee -a "$Q"; }
DEADLINE=$(( $(date +%s) + ${P4_HOURS:-7} * 3600 ))
late() { [ "$(date +%s)" -gt "$DEADLINE" ]; }
say "final: started (deadline in ${P4_HOURS:-7} h)"

# ---- 1. lexical zips, CPU, alongside the GPU queue
(
  lex() { local tag=$1; shift
    [ -f "$P4/results/$tag/meta.json" ] && return
    python phase4_lex.py "$@" --tag "$tag" > "$L/$tag.log" 2>&1
    say "lex   $tag | $(grep -a 'BEST' "$L/$tag.log" | tail -1 | sed 's/^ *//')"
  }
  lex lex_all --view all
  lex lex_tail --view tail --chars 6000 --char
  [ -f "$SILVER" ] && lex lex_tail_ext --view tail --chars 6000 --char --extra "$SILVER"
  [ -f "$P4/results/lexsec_all/meta.json" ] || python phase4_lex.py --sections > "$L/lex_sections.log" 2>&1
) &
LEXPID=$!

# ---- 2. the main queue
while ps aux | grep -q "[r]un_p4.sh"; do
  late && break
  sleep 60
done
if [ ! -f "$P4/results/ens_final_model/meta.json" ] && ! late; then
  say "final: main queue stopped before its end -- restarting it once (finished runs are skipped)"
  bash run_p4.sh >> "$L/queue_stdout.txt" 2>&1
fi

# ---- 3. the all-data model
BEST=$(python - "$P4" <<'EOF'
import json, os, sys
p = sys.argv[1]
f = lambda t: json.load(open(f"{p}/results/{t}/meta.json"))["dev_f1"] if os.path.exists(f"{p}/results/{t}/meta.json") else -1
print("lorma --rank 16" if f("m4_lorma") > f("m4_full") else "full")
EOF
)
EXTRA=""
[ -f "$SILVER" ] && EXTRA="--extra $SILVER --extra-weight 0.5"
if [ ! -f "$P4/results/m4_final/meta.json" ] && ! late; then
  say "start m4_final (M4 $BEST, all gold docs ${EXTRA:+ + silver})"
  python phase4_cjpe.py --tag m4_final --model M4_LexArcP --method $BEST --train-docs 100000 --kd-pool 0 \
      $EXTRA --chunks 2 --chunk 512 --batch 8 --accum 2 --evals 3 --submit --save-ckpt ${P4_ARGS:-} \
      >> "$L/q_m4_final.out" 2>&1 \
    || python phase4_cjpe.py --tag m4_final --model M4_LexArcP --method $BEST --train-docs 100000 --kd-pool 0 \
      $EXTRA --chunks 2 --chunk 512 --batch 8 --accum 2 --evals 3 --submit --save-ckpt --grad-ckpt ${P4_ARGS:-} \
      >> "$L/q_m4_final.out" 2>&1
  say "done  m4_final | $(grep -a '=====.*dev macro-F1' "$L/q_m4_final.out" | tail -1 | sed 's/^=* *//')"
fi

# ---- 4. final ensembles (CPU, seconds)
wait $LEXPID
ens() { local tag=$1; shift
  python phase4_ens.py ens --tag "$tag" --exclude lexcurve lexsec bench "$@" > "$L/$tag.out" 2>&1
  say "ens   $(grep -a 'dev macro-F1' "$L/$tag.out" | tail -1 | sed 's/^ *//')"
}
ens ens_best_model --kind model --greedy 8
ens ens_best_hyb --greedy 8
ens ens_best_model_rate --kind model --greedy 8 --rate
ens ens_best_hyb_rate --greedy 8 --rate

# ---- 5. what to upload
python phase4_ens.py status > /dev/null 2>&1
python - "$P4" <<'EOF'
import json, sys
p = sys.argv[1]
tr = json.load(open(f"{p}/submissions/tracker.json"))
seen, rows = set(), []
for t in sorted(tr, key=lambda t: -(t.get("dev") or 0)):
    if t["sha"] in seen:
        continue                     # identical predictions never count twice on Codabench
    seen.add(t["sha"])
    rows.append(f"{len(rows)+1:2d}. {t['zip']:28s} {t['kind']:6s} dev {t.get('dev') or 0:.4f}  "
                f"test pos-rate {t['pos_rate']:.3f}  {t['note']}")
open(f"{p}/SUBMIT_ORDER.txt", "w").write("\n".join(rows) + "\n")
print("\n".join(rows[:15]))
EOF
say "FINAL DONE -- see phase4/SUBMIT_ORDER.txt and phase4/STATUS.md"
