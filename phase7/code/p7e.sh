#!/usr/bin/env bash
# Phase 7, part E -- the two signals in this data that nothing else in the phase uses,
# plus the two methodology fixes that let us bank gains too small for devx to see.
# CPU ONLY: it never touches a GPU, so it can run on a second free Colab runtime at the
# same time as p7d.sh on the L4.
#
#   bash p7e.sh all      masks -> cite -> prop -> stack -> half -> zips   (~45 min, CPU)
#   bash p7e.sh masks    the premise check: do mask counts predict gold counts?
#   bash p7e.sh cite     citation-context retrieval, swept on TRAIN
#   bash p7e.sh prop     score propagation over the lex and cocite graphs
#   bash p7e.sh stack    stacks including the new members, incl. a lambdarank variant
#   bash p7e.sh zips     zip the new systems only (does NOT rebuild SUBMIT_ORDER.md)
#   bash p7e.sh order    rebuild SUBMIT_ORDER.md over everything -- run this ONCE, last
#   bash p7e.sh status
#
# RUNNING IT ALONGSIDE p7d.sh: safe, because the two write disjoint files -- but only
# ONE of them may run `zips`/`order`, since both regenerate SUBMIT_ORDER.md. So let p7d
# finish its own zips, then run `bash p7e.sh zips` and `bash p7e.sh order` here.
#
# WHY THESE.  Test = devx - 0.0095 and the test oracle@100 is ~0.538, so better ordering
# of the pool we retrieve cannot reach 0.53. Everything here is either a better retrieval
# unit or information we are not using at all:
#   * the ~11 citation masks per query are labelled pointers at the exact sentences that
#     cite a prior case, and every system so far matched the whole 6,000-word judgment;
#   * the candidate-candidate graph answers the CONDITIONAL question the dead popularity
#     prior could not: given A is relevant, is B?
#   * selection moves to the 827 train queries, because the 118-query devx bar is ~11
#     hits and cannot see the 0.005-size gains that are all that is left.
set -u

P7=${CS781_P7:-/content/drive/MyDrive/cs781_capstone/phase7}
L=$P7/logs
Q=$L/queuee.txt
TOPN=${P7_TOPN:-200}
SEL=${P7_SELECT:-train}
SPLITS=train,devx,test
mkdir -p "$L" "$P7/results" "$P7/submissions"

CLEAN=tfidf_sqrt,bm25_hi,bm25,tfidf,dense_m2_c8_max,dense_m2_c8,dense_m2_max,dense_m2,dense_m1,dense_m3,dense_m4
LEX4=tfidf_sqrt,bm25_hi,bm25,tfidf
# anything p7d may have finished by now; available() drops whatever is missing, loudly
MAYBE=maxp_tfidf,maxp_bm25,rm3_tfidf,rm3_maxp,ce_lorma,ce_lorma_mix
NEW=cite,prop_lex,prop_cocite

say() { printf '[%s] %s\n' "$(date +%H:%M)" "$*" | tee -a "$Q"; }

step() {
  local m=$1 n=$2; shift 2
  if [ -e "$m" ]; then say "skip  $n"; return 0; fi
  say "run   $n"
  if "$@" >"$L/$n.log" 2>&1; then
    say "done  $n"
    grep -aE 'BEST|/devx |/train |train OOF|spearman|pearson|per-query|graph:|never beat' \
         "$L/$n.log" | tail -6 | cut -c1-200 | while read -r l; do say "      $l"; done
    return 0
  fi
  say "FAIL  $n -- tail of $L/$n.log:"
  tail -25 "$L/$n.log" | tee -a "$Q"
  return 1
}

have() {                           # have <tag,tag,...> -> those with all three splits
  python - "$P7" "$1" <<'PY'
import os, sys
P7 = sys.argv[1]
print(",".join(t for t in sys.argv[2].split(",") if t and all(
    os.path.exists(f"{P7}/results/{t}/{s}_rank.json") for s in ("train", "devx", "test"))))
PY
}

# ------------------------------------------------------------- 1. the premise check --
masks() {
  say "======== do the citation masks exist, and do they predict the gold count?"
  step "$P7/results/mask_stats.json" e_masks \
       python phase7_cite.py masks --splits "$SPLITS" || true
  python - "$P7" <<'PY' 2>&1 | tee -a "$Q"
import json, sys
try:
    d = json.load(open(f"{sys.argv[1]}/results/mask_stats.json"))
except Exception as e:
    sys.exit(f"  no mask_stats.json ({e})")
for s, r in d.items():
    print(f"  {s:6s} {r['queries']:4d} queries, {r['masks']:5d} masks, "
          f"{r['zero_mask_queries']} with none, per-query {r['per_query']}")
    if r.get("spearman_mask_vs_gold") is not None:
        print(f"         golds/query {r.get('gold_per_query_mean')}  "
              f"spearman(masks, golds) {r['spearman_mask_vs_gold']}  "
              f"mean golds by mask quartile {r.get('mean_golds_by_mask_quartile')}")
    elif s != "test":
        print("         mask count does not vary, so it cannot predict the gold count")
t = d.get("train", {})
sp = t.get("spearman_mask_vs_gold")
print("  VERDICT:", "n_masks is worth its stacker feature" if (sp or 0) > 0.3 else
      "n_masks is weak as a COUNT -- the masks still locate the citations, which is "
      "what `cite` uses, and that does not depend on this number")
PY
}

# -------------------------------------------------- 2. citation-context retrieval ----
cite() {
  say "======== citation-context retrieval: ~11 focused sub-queries instead of 1 long one"
  say "         swept on $SEL (827 queries beats 118), confirmed on devx"
  step "$P7/results/cite/test_rank.json" e_cite \
       python phase7_cite.py cite --tag cite --sweep --select "$SEL" \
              --splits "$SPLITS" --topn "$TOPN" || true
  # the best aggregation blended back against whole-document matching, as a safety net
  step "$P7/results/cite_mix/test_rank.json" e_cite_mix \
       python phase7_cite.py cite --tag cite_mix --agg greedy --beta 0.5 \
              --splits "$SPLITS" --topn "$TOPN" || true
}

# ------------------------------------------------------- 3. graph propagation --------
prop() {
  say "======== score propagation: the cluster hypothesis, on two different graphs"
  step "$P7/results/prop_lex/test_rank.json" e_prop_lex \
       python phase7_cite.py prop --tag prop_lex --graph lex --sweep --select "$SEL" \
              --splits "$SPLITS" --topn "$TOPN" || true
  say "         cocite is built from TRAIN labels; its train ranking uses an exact"
  say "         leave-one-out, so it stays honest as a stacker feature"
  step "$P7/results/prop_cocite/test_rank.json" e_prop_cocite \
       python phase7_cite.py prop --tag prop_cocite --graph cocite --sweep \
              --select "$SEL" --splits "$SPLITS" --topn "$TOPN" || true
}

# ---------------------------------------------------------------- 4. stacking --------
stack() {
  say "======== stacking with the new members"
  local new maybe mem
  new=$(have "$NEW")
  maybe=$(have "$MAYBE")
  say "new members: ${new:-none}"
  say "p7d members already finished: ${maybe:-none}"
  if [ -z "$new" ]; then say "nothing new to stack; skipping"; return 0; fi
  mem="$CLEAN${maybe:+,$maybe},$new"
  step "$P7/results/stack_cite/test_rank.json" e_stack_cite \
       python phase7_stack.py fit --tag stack_cite --kind hybrid --members "$mem" \
              --pool-depth 100 --topn "$TOPN" --importance || true
  step "$P7/results/stack_cite_lex/test_rank.json" e_stack_cite_lex \
       python phase7_stack.py fit --tag stack_cite_lex --kind lexical \
              --members "$LEX4,$new" --pool-depth 50 --topn "$TOPN" || true
  # lambdarank: the objective is hits@7, and pointwise logloss optimises neither the
  # ordering nor the cutoff.  Needs lightgbm; the step fails cleanly without it.
  # Only ever install inside Colab -- a driver must not mutate someone's local
  # environment as a side effect of being run there.
  if ! python -c "import lightgbm" 2>/dev/null; then
    if [ -d /content ]; then
      say "installing lightgbm for the lambdarank variant"
      pip install -q lightgbm 2>&1 | tail -2 | tee -a "$Q"
    else
      say "lightgbm not importable and this is not Colab: skipping the lambdarank stacks"
      return 0
    fi
  fi
  step "$P7/results/stack_rank/test_rank.json" e_stack_rank \
       python phase7_stack.py fit --tag stack_rank --kind hybrid --members "$mem" \
              --pool-depth 100 --topn "$TOPN" --model lgbm --trunc 20 || true
  step "$P7/results/stack_rank_d50/test_rank.json" e_stack_rank_d50 \
       python phase7_stack.py fit --tag stack_rank_d50 --kind hybrid --members "$mem" \
              --pool-depth 50 --topn "$TOPN" --model lgbm --trunc 10 || true
}

half() {
  say "======== selection: devx halves AND the 827-query train table"
  local tags
  tags="stack_d100,$(have "stack_cite,stack_cite_lex,stack_rank,stack_rank_d50,$NEW,$MAYBE")"
  python phase7_stack.py half --tags "$tags" 2>&1 | tee -a "$Q" || true
  say "A gain that clears the TRAIN bar and points the right way on devx half B is worth"
  say "banking. One that clears only devx half A is the stack_d20 mistake again."
}

zips() {
  say "======== zips for the new systems only (SUBMIT_ORDER.md is rebuilt by 'order')"
  for t in cite cite_mix prop_lex prop_cocite stack_cite_lex; do
    [ -d "$P7/results/$t" ] || continue
    step "$P7/submissions/$t/submission.zip" "z_$t" \
         python phase7_rank.py zip --tag "$t" --kind lexical || true
  done
  for t in stack_cite stack_rank stack_rank_d50; do
    [ -d "$P7/results/$t" ] || continue
    step "$P7/submissions/$t/submission.zip" "z_$t" \
         python phase7_rank.py zip --tag "$t" --kind hybrid || true
  done
}

order() { say "======== rebuilding SUBMIT_ORDER.md over every system"
          bash p7b.sh zips 2>&1 | tail -70 | tee -a "$Q"; }

status() {
  python phase7_rank.py show 2>&1 | tee -a "$Q" || true
  say "zips: $(ls -1 "$P7"/submissions/*/submission.zip 2>/dev/null | wc -l)"
}

case "${1:-all}" in
  masks)  masks ;;
  cite)   cite ;;
  prop)   prop ;;
  stack)  stack ;;
  half)   half ;;
  zips)   zips ;;
  order)  order ;;
  status) status ;;
  all)    say "======== p7e.sh all | select on $SEL | topn $TOPN | CPU only"
          masks; cite; prop; stack; half; zips; status
          say "======== PART E DONE -- run 'bash p7e.sh order' once p7d has finished too" ;;
  *)      echo "usage: bash p7e.sh [all|masks|cite|prop|stack|half|zips|order|status]"
          exit 2 ;;
esac
