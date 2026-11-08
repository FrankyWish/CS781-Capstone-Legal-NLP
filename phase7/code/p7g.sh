#!/usr/bin/env bash
# Phase 7, part H -- the only two ideas that can change the ORDER of magnitude.
# CPU ONLY, so it runs beside a GPU part.  ~45 min.
#
#   bash p7g.sh all        where -> residue -> ngram -> hub -> stack -> half -> zips
#   bash p7g.sh where      where the 556 reachable hits actually sit
#   bash p7g.sh residue    THE GATE: does mask context identify the cited case?
#   bash p7g.sh ngram      exact rare n-gram pointer retrieval
#   bash p7g.sh hub        hubness / capacity decoding over the best saved runs
#   bash p7g.sh stack      restack including the new members
#   bash p7g.sh half       devx halves + the 827-query train table
#   bash p7g.sh zips       submission zips
#
# WHY ONLY THESE TWO.  The measured dev oracle says 0.53 on test is 477 of the 556
# reachable hits: 100.6% of a PERFECT rerank of the top 50 and 96.3% of the top 100.
# So a better ordering of the same evidence cannot reach it -- that is arithmetic, not
# pessimism.  Only new evidence can, and inside the brief's constraints (Phase-1 trunks
# only, no external weights or corpora, no LLM APIs) there are two sources left:
#
#   1. WORD ORDER.  Every system in the phase is a bag of words.  An exact 8-word match
#      between a citing sentence and a candidate judgment is not a similarity score, it
#      is a pointer, because judgments quote the holdings they cite.  `residue` measures
#      whether the quotations survived IL-PCR's masking; `ngram` is the system.
#   2. THE JOINT STRUCTURE ACROSS QUERIES.  Every system ranks each query alone, so a
#      few long generic judgments win slots everywhere.  Gold structure cannot look like
#      that.  `hub` corrects for it and decodes all queries together under a capacity.
#      Same move as Phase 2's test-volume calibration -- the one thing that worked there.
#
# Read `residue` FIRST.  Enrichment near 1 means the masking stripped the quotations and
# idea 1 is dead; enrichment in the tens or hundreds means it is the best lever in the
# phase and deserves the rest of the day.
set -u

P7=${CS781_P7:-/content/drive/MyDrive/cs781_capstone/phase7}
L=$P7/logs
Q=$L/queueg.txt
TOPN=${P7_TOPN:-200}
SEL=${P7_SELECT:-train}
MARGIN=${P7_UPLOAD_MARGIN:-0.02}
SPLITS=train,devx,test
NSWEEP=${P7_NSWEEP:-6,8,12,20}
mkdir -p "$L" "$P7/results" "$P7/submissions"

CLEAN=tfidf_sqrt,bm25_hi,bm25,tfidf,dense_m2_c8_max,dense_m2_c8,dense_m2_max,dense_m2,dense_m1,dense_m3,dense_m4
LEX4=tfidf_sqrt,bm25_hi,bm25,tfidf
OLD=maxp_tfidf,maxp_bm25,rm3_tfidf,maxp2,citep,cite,prop_lex,prop_cocite,ce_lorma,ce_lorma_mix
NEW=ptr
BASES=${P7_HUB_BASES:-stack_new,stack_f,stack_lex2,maxp2,citep,maxp_tfidf,ptr}

say() { printf '[%s] %s\n' "$(date +%H:%M)" "$*" | tee -a "$Q"; }

step() {
  local m=$1 n=$2; shift 2
  if [ -n "$m" ] && [ -e "$m" ]; then say "skip  $n"; return 0; fi
  say "run   $n"
  if "$@" >"$L/$n.log" 2>&1; then
    say "done  $n"
    grep -aE 'BEST|chose|enrichment|reachable hits|distinct gold|matching|/devx |/train |nothing beat' \
         "$L/$n.log" | tail -8 | cut -c1-200 | while read -r l; do say "      $l"; done
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

# ------------------------------------------------------- 0. free diagnostics --
where() {
  say "======== where the 556 reachable hits sit (free, and it aims everything else)"
  step "" g_where python phase7_point.py where --splits train,devx || true
}

# ------------------------------------------------------------- 1. THE GATE ----
residue() {
  say "======== THE GATE: does the text around a mask identify the case it cites?"
  say "         enrichment ~1 -> masking stripped the quotations, pointer route dead"
  say "         enrichment >>1 -> exact quotation survives, and it is the best lever left"
  step "$P7/results/ngram_residue.json" g_residue \
       python phase7_point.py residue --splits devx --nsweep "$NSWEEP" --scope masks || true
  # and the same question for the whole judgment, not just the mask windows: a quotation
  # can sit anywhere, and `doc` scope costs nothing extra to measure.
  step "" g_residue_doc \
       python phase7_point.py residue --splits devx --nsweep 8,12 --scope doc || true
}

# ----------------------------------------------------- 2. the pointer system --
ngram() {
  say "======== exact rare n-gram pointers (word order -- the one signal nothing else uses)"
  step "$P7/results/ptr/test_rank.json" g_ngram \
       python phase7_point.py ngram --tag ptr --nsweep "$NSWEEP" \
              --scope masks,both --splits "$SPLITS" --topn "$TOPN" || true
}

# --------------------------------------------- 3. hubness + capacity decoding --
hub() {
  local bases b
  bases=$(have "$BASES")
  say "======== hubness / capacity decoding over: ${bases:-none available}"
  say "         selection on $SEL -- train is 827 queries, a ~2.6x tighter paired"
  say "         bar than devx's 118, and out of sample for a decoder that learns nothing"
  [ -z "$bases" ] && { say "no base runs present; skipping"; return 0; }
  for b in $(echo "$bases" | tr ',' ' '); do
    step "$P7/results/hub_$b/test_rank.json" "g_hub_$b" \
         python phase7_point.py hub --base "$b" --splits "$SPLITS" --select "$SEL" \
                --mode zmean,count,greedy,sinkhorn \
                --asweep 0.1,0.25,0.5,1.0 --capsweep 3,6,12,25,50 || true
  done
}

# ------------------------------------------------------------------ 4. stack --
stack() {
  local new old mem hubs
  new=$(have "$NEW")
  old=$(have "$OLD")
  hubs=$(have "$(echo "$BASES" | sed 's/[^,]*/hub_&/g')")
  say "======== stacking | new: ${new:-none} | hub: ${hubs:-none} | old: ${old:-none}"
  if [ -z "$new$hubs" ]; then say "nothing new to add; skipping"; return 0; fi
  mem="$CLEAN${old:+,$old}${new:+,$new}${hubs:+,$hubs}"
  step "$P7/results/stack_g/test_rank.json" g_stack_g \
       python phase7_stack.py fit --tag stack_g --kind hybrid --members "$mem" \
              --pool-depth 100 --topn "$TOPN" --importance || true
  local lexmem
  lexmem=$(have "maxp_tfidf,maxp2,citep,cite,rm3_tfidf,ptr")
  step "$P7/results/stack_g_lex/test_rank.json" g_stack_g_lex \
       python phase7_stack.py fit --tag stack_g_lex --kind lexical \
              --members "$LEX4${lexmem:+,$lexmem}" \
              --pool-depth 50 --topn "$TOPN" || true
}

# the decoder is the LAST thing applied, so it also gets a shot at the new stack
restack_hub() {
  local s
  s=$(have "stack_g,stack_g_lex")
  [ -z "$s" ] && return 0
  say "======== decoding the new stacks too"
  for b in $(echo "$s" | tr ',' ' '); do
    step "$P7/results/hub_$b/test_rank.json" "g_hub_$b" \
         python phase7_point.py hub --base "$b" --splits "$SPLITS" --select "$SEL" \
                --mode zmean,count,greedy,sinkhorn \
                --asweep 0.1,0.25,0.5,1.0 --capsweep 3,6,12,25,50 || true
  done
}

half() {
  say "======== selection: devx halves AND the 827-query train table"
  python phase7_stack.py half --tags \
    "stack_new,$(have "stack_g,stack_g_lex,stack_f,stack_f_lex,stack_f_rank,ptr,$(echo "$BASES,stack_g,stack_g_lex" | sed 's/[^,]*/hub_&/g')")" \
    2>&1 | tee -a "$Q" || true
}

zips() {
  say "======== zips"
  for t in ptr stack_g_lex $(have "$(echo "$BASES,stack_g,stack_g_lex" | sed 's/[^,]*/hub_&/g')" | tr ',' ' '); do
    [ -d "$P7/results/$t" ] || continue
    case "$t" in
      ptr|stack_g_lex) k=lexical ;;
      *)               k=hybrid ;;
    esac
    step "$P7/submissions/$t/submission.zip" "z_$t" \
         python phase7_rank.py zip --tag "$t" --kind "$k" || true
  done
  [ -d "$P7/results/stack_g" ] && step "$P7/submissions/stack_g/submission.zip" z_stack_g \
       python phase7_rank.py zip --tag stack_g --kind hybrid || true
}

order() {
  say "======== gated SUBMIT_ORDER.md (upload gate: best devx minus $MARGIN)"
  P7_UPLOAD_MARGIN=$MARGIN bash p7b.sh zips 2>&1 | tail -80 | tee -a "$Q"
}

status() {
  python phase7_rank.py show 2>&1 | tee -a "$Q" || true
  say "zips: $(ls -1 "$P7"/submissions/*/submission.zip 2>/dev/null | wc -l)"
}

case "${1:-all}" in
  where)   where ;;
  residue) residue ;;
  ngram)   ngram ;;
  hub)     hub ;;
  stack)   stack ;;
  restack) restack_hub ;;
  half)    half ;;
  zips)    zips ;;
  order)   order ;;
  status)  status ;;
  all)     say "======== p7g.sh all | select $SEL | topn $TOPN | nsweep $NSWEEP | CPU only"
           where; residue; ngram; hub
           stack; restack_hub; half; zips; status
           say "======== PART H DONE -- read results/ngram_residue.json FIRST" ;;
  *)       echo "usage: bash p7g.sh [all|where|residue|ngram|hub|stack|restack|half|zips|order|status]"
           exit 2 ;;
esac
