#!/usr/bin/env bash
# Phase 7, part I -- everything that has never been run, ordered by what could actually
# reach 0.54.  CPU ONLY.  One blocking invocation; nothing here forks or backgrounds.
#
#   bash p7i.sh all          the whole thing, unattended, ~3-4 h
#   bash p7i.sh overlap      THE GATE for 0.54 -- run this first, alone, if time is short
#   bash p7i.sh rrf          rank-only fusion: invariant to pool size and score scale
#   bash p7i.sh curve        how fast each system decays as the pool grows
#   bash p7i.sh pointers     part H: where / residue / ngram
#   bash p7i.sh hub          hubness + capacity decoding over every finished run
#   bash p7i.sh prop         part E: graph score propagation
#   bash p7i.sh csum         the one citep aggregation never measured
#   bash p7i.sh combine      stack + half + zips + order
#
# WHY THIS ORDER.  0.54 on test is 486 of the 556 reachable devx hits -- 87% of every gold
# that exists, and 98% of a PERFECT rerank of the top 100 (measured oracle@100 = 495).
# Ranking cannot get there: three days of it moved 299 -> 387 hits. So `overlap` runs
# first, because recovering a masked citation from the permitted Phase-1 corpus is the
# only mechanism of the right magnitude, and `value` prices it with the official metric
# before anything is built on it.
#
# Then the transfer problem, which is the reason today's devx gains kept evaporating:
# devx has 1,023 candidates, test has 1,727, train has 4,320. `citep` won devx by 26 hits
# and lost test by 0.0197; four stacks inside a 9-hit devx band came back in exactly
# reversed test order. `curve` measures that decay directly by growing the devx pool with
# real train judgments as true negatives, and `rrf` is the fix it implies -- fusion on
# RANKS only, so pool size and score scale cannot touch it.
set -u

P7=${CS781_P7:-/content/drive/MyDrive/cs781_capstone/phase7}
L=$P7/logs
Q=$L/queuei.txt
TOPN=${P7_TOPN:-200}
SEL=${P7_SELECT:-train}
MARGIN=${P7_UPLOAD_MARGIN:-0.02}
SPLITS=train,devx,test
mkdir -p "$L" "$P7/results" "$P7/submissions"

CLEAN=tfidf_sqrt,bm25_hi,bm25,tfidf,dense_m2_c8_max,dense_m2_c8,dense_m2_max,dense_m2,dense_m1,dense_m3,dense_m4
FUSEM=citep,maxp2,maxp_tfidf,tfidf_sqrt,bm25_hi,rm3_tfidf,stack_new,stack_ce_lex,stack_lex2,ce_lorma_mix,dense_m2_c8_max,ft_lorma_m2,ptr,cite
OLD=maxp_tfidf,maxp_bm25,rm3_tfidf,maxp2,citep,cite,ce_lorma,ce_lorma_mix
NEW=ptr,citeps,rrf_rank,rrf_lex
HUBB=${P7_HUB_BASES:-citep,stack_new,stack_ce_lex,rrf_rank,maxp2,maxp_tfidf}

say() { printf '[%s] %s\n' "$(date +%H:%M)" "$*" | tee -a "$Q"; }

step() {
  local m=$1 n=$2; shift 2
  if [ -n "$m" ] && [ -e "$m" ]; then say "skip  $n"; return 0; fi
  say "run   $n"
  if "$@" >"$L/$n.log" 2>&1; then
    say "done  $n"
    grep -aE 'BEST|CHOSE|chose|SLOPE|enrichment|recoverable|upper bound|lookup|/devx |/train |nothing beat|READ IT' \
         "$L/$n.log" | tail -10 | cut -c1-210 | while read -r l; do say "      $l"; done
    return 0
  fi
  say "FAIL  $n -- tail of $L/$n.log:"
  tail -20 "$L/$n.log" | tee -a "$Q"
  return 1
}

have() {
  python - "$P7" "$1" <<'PY'
import os, sys
P7 = sys.argv[1]
print(",".join(t for t in sys.argv[2].split(",") if t and all(
    os.path.exists(f"{P7}/results/{t}/{s}_rank.json") for s in ("train", "devx", "test"))))
PY
}

# ===================================================== 1. THE GATE FOR 0.54 ====
overlap() {
  say "======== THE GATE: can a masked citation be recovered from the Phase-1 corpus?"
  say "         0.54 on test = 486 of 556 reachable devx hits = 98% of a PERFECT rerank"
  say "         of the top 100. Ranking cannot reach it. Recovery is the only mechanism"
  say "         of the right size, and `value` prices it with the official metric."
  # Hard time cap. Every other stage is `|| true`, so a FAILURE cannot cost the
  # rest -- but a slow corpus stream would simply block them, and this runs
  # unattended for hours. 45 min is enough to stream the Supreme Court corpus;
  # past that it will not finish and the remaining stages are worth more.
  step "$P7/results/overlap/recovered_devx.json" i_scan \
       timeout ${P7_SCAN_SECONDS:-2700} python phase7_overlap.py scan \
              --splits test,devx \
              --sources train_supreme_court --needles 10 --min-hits 3 || true
  step "" i_value python phase7_overlap.py value --split devx || true
  say "      MEASURE-ONLY. Nothing is built on a match without Nithin's explicit"
  say "      go-ahead, and any such system is reported as kind=lookup beside the"
  say "      model-only number."
}

# ====================================== 2. the fix for the transfer problem ====
rrf() {
  local mem
  mem=$(have "$FUSEM")
  say "======== rank-only fusion over: ${mem:-none}"
  say "         RRF reads RANKS, never scores, so pool size (devx 1023 vs test 1727 vs"
  say "         train 4320) and per-query score inflation cannot affect it. The learned"
  say "         stacker is fitted on train's pool and applied to test's -- this is not."
  [ -z "$mem" ] && { say "no members; skipping"; return 0; }
  step "$P7/results/rrf_rank/test_rank.json" i_rrf \
       python phase7_scale.py rrf --tag rrf_rank --kind hybrid --members "$mem" \
              --select "$SEL" --splits "$SPLITS" --topn "$TOPN" --rounds 10 || true
  local lex
  lex=$(have "citep,maxp2,maxp_tfidf,tfidf_sqrt,bm25_hi,bm25,tfidf,rm3_tfidf,cite,ptr")
  [ -z "$lex" ] && return 0
  step "$P7/results/rrf_lex/test_rank.json" i_rrf_lex \
       python phase7_scale.py rrf --tag rrf_lex --kind lexical --members "$lex" \
              --select "$SEL" --splits "$SPLITS" --topn "$TOPN" --rounds 10 || true
}

curve() {
  local sys
  sys=$(have "citep,maxp2,tfidf_sqrt,stack_new")
  say "======== pool-growth decay: devx 1023 grown with real train judgments to 2423"
  say "         704 extra takes devx to exactly the test pool's 1727"
  [ -z "$sys" ] && { say "no systems; skipping"; return 0; }
  step "$P7/results/pool_curve.json" i_curve \
       python phase7_scale.py curve --split devx --systems "$sys" \
              --grow 0,350,704,1400 --topn 100 \
              --passages 250 --stride 125 --agg max --beta 0.7 --window 60 || true
}

# ======================================================= 3. part H, never run ==
pointers() {
  say "======== part H: where the hits are, then the exact-n-gram pointer gate"
  step "" i_where python phase7_point.py where --splits train,devx || true
  step "$P7/results/ngram_residue.json" i_residue \
       python phase7_point.py residue --splits devx --nsweep 6,8,12,20 --scope masks || true
  step "" i_residue_doc \
       python phase7_point.py residue --splits devx --nsweep 8,12 --scope doc || true
  step "$P7/results/ptr/test_rank.json" i_ngram \
       python phase7_point.py ngram --tag ptr --nsweep 6,8,12,20 \
              --scope masks,both --splits "$SPLITS" --topn "$TOPN" || true
}

hub() {
  local bases b
  bases=$(have "$HUBB")
  say "======== hubness / capacity decoding over: ${bases:-none}"
  [ -z "$bases" ] && return 0
  for b in $(echo "$bases" | tr ',' ' '); do
    step "$P7/results/hub_$b/test_rank.json" "i_hub_$b" \
         python phase7_point.py hub --base "$b" --splits "$SPLITS" --select "$SEL" \
                --mode zmean,count,greedy,sinkhorn \
                --asweep 0.1,0.25,0.5,1.0 --capsweep 3,6,12,25,50 || true
  done
}

prop() {
  say "======== part E: graph score propagation (lex kNN and train co-citation)"
  step "$P7/results/prop_lex/test_rank.json" i_prop_lex \
       python phase7_cite.py prop --tag prop_lex --graph lex --sweep --select "$SEL" \
              --splits "$SPLITS" --topn "$TOPN" || true
  step "$P7/results/prop_cocite/test_rank.json" i_prop_cocite \
       python phase7_cite.py prop --tag prop_cocite --graph cocite --sweep \
              --select "$SEL" --splits "$SPLITS" --topn "$TOPN" || true
}

csum() {
  say "======== citep with agg=sum -- the one aggregation the stopped sweep never reached"
  step "$P7/results/citeps/test_rank.json" i_citeps \
       python phase7_cite.py cite --tag citeps --agg sum --beta 0.7 --window 60 \
              --passages 250 --stride 125 --select "$SEL" \
              --splits "$SPLITS" --topn "$TOPN" || true
}

# ============================================================== 4. combine ====
combine() {
  local new old mem hubs
  new=$(have "$NEW")
  old=$(have "$OLD")
  hubs=$(have "$(echo "$HUBB" | sed 's/[^,]*/hub_&/g')")
  say "======== stacking | new: ${new:-none} | hub: ${hubs:-none}"
  mem="$CLEAN${old:+,$old}${new:+,$new}${hubs:+,$hubs}"
  step "$P7/results/stack_i/test_rank.json" i_stack \
       python phase7_stack.py fit --tag stack_i --kind hybrid --members "$mem" \
              --pool-depth 100 --topn "$TOPN" --importance || true
  # and a rank-only fusion ON TOP of everything, including the stacks
  local allm
  allm=$(have "$FUSEM,$NEW,stack_i,$(echo "$HUBB" | sed 's/[^,]*/hub_&/g')")
  [ -n "$allm" ] && step "$P7/results/rrf_final/test_rank.json" i_rrf_final \
       python phase7_scale.py rrf --tag rrf_final --kind hybrid --members "$allm" \
              --select "$SEL" --splits "$SPLITS" --topn "$TOPN" --rounds 12 || true
  say "======== selection: devx halves AND the 827-query train table"
  python phase7_stack.py half --tags \
    "stack_new,$(have "stack_i,rrf_rank,rrf_lex,rrf_final,citeps,$NEW,$OLD,$(echo "$HUBB" | sed 's/[^,]*/hub_&/g')")" \
    2>&1 | tee -a "$Q" || true
  say "======== zips"
  for t in $(have "rrf_rank,rrf_lex,rrf_final,citeps,ptr,stack_i,$(echo "$HUBB" | sed 's/[^,]*/hub_&/g')" | tr ',' ' '); do
    case "$t" in
      rrf_lex|citeps|ptr) k=lexical ;;
      *)                  k=hybrid ;;
    esac
    step "$P7/submissions/$t/submission.zip" "z_$t" \
         python phase7_rank.py zip --tag "$t" --kind "$k" || true
  done
  say "======== gated SUBMIT_ORDER.md (gate: best devx minus $MARGIN)"
  P7_UPLOAD_MARGIN=$MARGIN bash p7b.sh zips 2>&1 | tail -80 | tee -a "$Q"
  python phase7_rank.py show 2>&1 | tee -a "$Q" || true
  say "======== THE ZIP TO UPLOAD"
  python - "$P7" <<'PYZ' 2>&1 | tee -a "$Q"
import glob, json, os, sys
P7 = sys.argv[1]
rows = []
for m in glob.glob(f"{P7}/results/*/meta.json"):
    t = os.path.basename(os.path.dirname(m))
    if not os.path.exists(f"{P7}/submissions/{t}/submission.zip"):
        continue
    try:
        d = json.load(open(m))
    except Exception:
        continue
    f1 = (d.get("devx_scores") or {}).get("f1")
    if f1 is None:
        for v in d.values():
            if isinstance(v, dict) and "f1" in v:
                f1 = v["f1"]
                break
    if f1 is not None:
        rows.append((float(f1), t, str(d.get("kind", "?"))))
rows.sort(reverse=True)
print(f"{len(rows)} built zips have a devx score. Best first:")
for f1, t, k in rows[:8]:
    print(f"   devx {f1:.4f}  {k:8s}  phase7/submissions/{t}/submission.zip")
if rows:
    print()
    print(f"UPLOAD FIRST: phase7/submissions/{rows[0][1]}/submission.zip "
          f"(devx {rows[0][0]:.4f})")
    print("Then the next two or three down the list.")
    print("Right-click the submission.zip FILE, never the folder: a folder download")
    print("is a wrapper zip and Codabench rejects it.")
    print("Nothing below devx 0.428 can beat the 0.4241 already on the board.")
PYZ
}

case "${1:-all}" in
  overlap)  overlap ;;
  rrf)      rrf ;;
  curve)    curve ;;
  pointers) pointers ;;
  hub)      hub ;;
  prop)     prop ;;
  csum)     csum ;;
  combine)  combine ;;
  all)      say "======== p7i.sh all | select $SEL | topn $TOPN | gate $MARGIN | CPU only"
            overlap
            rrf
            curve
            pointers
            hub
            prop
            csum
            combine
            say "======== PART I DONE"
            say "         READ IN THIS ORDER: results/overlap/*.json and the i_value log"
            say "         (the only route to 0.54), then results/pool_curve.json (why devx"
            say "         did not transfer), then the half-split table above." ;;
  *)        echo "usage: bash p7i.sh [all|overlap|rrf|curve|pointers|hub|prop|csum|combine]"
            exit 2 ;;
esac
