#!/usr/bin/env bash
# Phase 7, part F -- push on the ONE thing that demonstrably worked, and stop uploading
# systems that cannot win.  CPU ONLY, so it runs beside the cross-encoder on the L4.
#
#   bash p7f.sh all      maxp2 -> citep -> stack -> half -> zips -> order   (~50 min)
#   bash p7f.sh maxp2    the widened MaxP sweep: smaller widths AND overlap
#   bash p7f.sh citep    mask-anchored query windows x candidate passages
#   bash p7f.sh stack    restack over whatever is finished
#   bash p7f.sh half     devx halves + the 827-query train table
#   bash p7f.sh order    gated SUBMIT_ORDER.md -- run ONCE, after every other part
#
# WHY THESE TWO.  Of everything built in parts A-E, exactly one single-system change
# cleared the 11-hit bar: MaxP, +20 hits, by scoring a judgment on its best 300-word
# passage instead of its 6,383-word average. And its sweep chose 300 out of
# {300,500,800,1200} -- the BOUNDARY. The original BM25 sweep landed on 3 of 4 boundaries
# and widening it was worth 19 hits, so the same move is owed here: smaller widths, and
# overlapping windows, because a non-overlapping grid dilutes a gold passage that
# straddles a boundary.
#
# `citep` then fixes the OTHER side of the same problem. MaxP cut up the candidate; the
# query is still a whole judgment. The ~11 citation masks say exactly which sentences
# cite a prior case, so mask-anchored query windows against candidate passages is the
# full version of the only idea that has paid.
set -u

P7=${CS781_P7:-/content/drive/MyDrive/cs781_capstone/phase7}
L=$P7/logs
Q=$L/queuef.txt
TOPN=${P7_TOPN:-200}
SEL=${P7_SELECT:-train}
MARGIN=${P7_UPLOAD_MARGIN:-0.02}
SPLITS=train,devx,test
mkdir -p "$L" "$P7/results" "$P7/submissions"

CLEAN=tfidf_sqrt,bm25_hi,bm25,tfidf,dense_m2_c8_max,dense_m2_c8,dense_m2_max,dense_m2,dense_m1,dense_m3,dense_m4
LEX4=tfidf_sqrt,bm25_hi,bm25,tfidf
MAYBE=maxp_tfidf,maxp_bm25,rm3_tfidf,ce_lorma,ce_lorma_mix,cite,prop_lex,prop_cocite
NEWF=maxp2,citep

say() { printf '[%s] %s\n' "$(date +%H:%M)" "$*" | tee -a "$Q"; }

step() {
  local m=$1 n=$2; shift 2
  if [ -e "$m" ]; then say "skip  $n"; return 0; fi
  say "run   $n"
  if "$@" >"$L/$n.log" 2>&1; then
    say "done  $n"
    grep -aE 'BEST|BOUNDARY|/devx |/train |train OOF|index .*passages' \
         "$L/$n.log" | tail -6 | cut -c1-200 | while read -r l; do say "      $l"; done
    return 0
  fi
  say "FAIL  $n -- tail of $L/$n.log:"
  tail -25 "$L/$n.log" | tee -a "$Q"
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

chose() {                          # chose <key> -> the widened sweep's pick
  python - "$P7" "$1" <<'PY'
import json, sys
try:
    print(json.load(open(f"{sys.argv[1]}/results/maxp_tfidf_sweep.json"))["chose"][sys.argv[2]])
except Exception:
    print({"passages": 180, "stride": 90, "agg": "max"}[sys.argv[2]])
PY
}

# --------------------------------------------------------- 1. the widened MaxP ------
maxp2() {
  say "======== MaxP, swept properly: widths 80-350 and half-width overlap"
  say "         the first sweep chose 300 of {300,500,800,1200} -- the boundary"
  # the sweep itself overwrites maxp_tfidf_sweep.json, so force it by removing the marker
  rm -f "$P7/results/maxp_tfidf_sweep.json"
  step "$P7/results/maxp2/test_rank.json" f_maxp2 \
       python phase7_rank.py maxp_tfidf --tag maxp2 --sweep \
              --psweep 80,120,180,250,350 --stride-sweep 1.0,0.5 \
              --qtf sqrt --max-df 0.2 --splits "$SPLITS" --topn "$TOPN" || return 1
  say "      chosen: width $(chose passages) stride $(chose stride) agg $(chose agg)"
}

# ------------------------------------- 2. both sides of the retrieval unit at once ---
citep() {
  local pw st
  pw=$(chose passages | tail -1)
  st=$(chose stride | tail -1)
  say "======== citep: mask-anchored query windows x ${pw}w candidate passages"
  say "         MaxP fixed the candidate side; the ~11 masks fix the query side"
  step "$P7/results/citep/test_rank.json" f_citep \
       python phase7_cite.py cite --tag citep --sweep --select "$SEL" \
              --passages "$pw" --stride "$st" \
              --splits "$SPLITS" --topn "$TOPN" || true
  # and the plain-document version, so the passage contribution is separable
  step "$P7/results/cite/test_rank.json" f_cite \
       python phase7_cite.py cite --tag cite --sweep --select "$SEL" \
              --splits "$SPLITS" --topn "$TOPN" || true
}

# ------------------------------------------------------------------- 3. stacking ----
stack() {
  local new maybe mem
  new=$(have "$NEWF")
  maybe=$(have "$MAYBE")
  say "======== stacking | new: ${new:-none} | already finished: ${maybe:-none}"
  if [ -z "$new$maybe" ]; then say "nothing to add; skipping"; return 0; fi
  mem="$CLEAN${maybe:+,$maybe}${new:+,$new}"
  step "$P7/results/stack_f/test_rank.json" f_stack_f \
       python phase7_stack.py fit --tag stack_f --kind hybrid --members "$mem" \
              --pool-depth 100 --topn "$TOPN" --importance || true
  local lexmem
  lexmem=$(have "maxp_tfidf,rm3_tfidf,cite,citep,maxp2")
  step "$P7/results/stack_f_lex/test_rank.json" f_stack_f_lex \
       python phase7_stack.py fit --tag stack_f_lex --kind lexical \
              --members "$LEX4${lexmem:+,$lexmem}" \
              --pool-depth 50 --topn "$TOPN" || true
  python -c "import lightgbm" 2>/dev/null || { [ -d /content ] && pip install -q lightgbm; }
  step "$P7/results/stack_f_rank/test_rank.json" f_stack_f_rank \
       python phase7_stack.py fit --tag stack_f_rank --kind hybrid --members "$mem" \
              --pool-depth 100 --topn "$TOPN" --model lgbm --trunc 20 || true
}

half() {
  say "======== selection: devx halves AND the 827-query train table"
  python phase7_stack.py half --tags \
    "stack_new,$(have "stack_f,stack_f_lex,stack_f_rank,stack_cite,stack_rank,$NEWF,$MAYBE")" \
    2>&1 | tee -a "$Q" || true
}

zips() {
  say "======== zips"
  for t in maxp2 citep cite stack_f_lex; do
    [ -d "$P7/results/$t" ] || continue
    step "$P7/submissions/$t/submission.zip" "z_$t" \
         python phase7_rank.py zip --tag "$t" --kind lexical || true
  done
  for t in stack_f stack_f_rank; do
    [ -d "$P7/results/$t" ] || continue
    step "$P7/submissions/$t/submission.zip" "z_$t" \
         python phase7_rank.py zip --tag "$t" --kind hybrid || true
  done
}

order() {
  say "======== gated SUBMIT_ORDER.md (upload gate: best devx minus $MARGIN)"
  P7_UPLOAD_MARGIN=$MARGIN bash p7b.sh zips 2>&1 | tail -70 | tee -a "$Q"
}

status() {
  python phase7_rank.py show 2>&1 | tee -a "$Q" || true
  say "zips: $(ls -1 "$P7"/submissions/*/submission.zip 2>/dev/null | wc -l)"
}

case "${1:-all}" in
  maxp2)  maxp2 ;;
  citep)  citep ;;
  stack)  stack ;;
  half)   half ;;
  zips)   zips ;;
  order)  order ;;
  status) status ;;
  all)    say "======== p7f.sh all | select $SEL | topn $TOPN | gate $MARGIN | CPU only"
          maxp2 && citep
          stack; half; zips; order; status
          say "======== PART F DONE" ;;
  *)      echo "usage: bash p7f.sh [all|maxp2|citep|stack|half|zips|order|status]"
          exit 2 ;;
esac
