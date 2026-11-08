#!/bin/bash
# Phase 6, the last refit -- ONE foreground cell, about 2 h on an L4.
#
#   bash p6g.sh
#
# Refits the only lever that moved the score (bsel) on every fit document instead of
# 1,500, with a word-level score added, then writes one zip per label.  Each step is
# skipped if its output exists, so after a disconnect re-run the SAME cell.
# Nothing here touches submissions/_today or SUBMIT_ORDER.md.
set -u
cd "$(dirname "$0")"
P6=${CS781_P6:-/content/drive/MyDrive/cs781_capstone/phase6}
L=$P6/logs
Q=$L/queue.txt
say() { echo "[$(date +%H:%M)] $*" | tee -a "$Q"; }
J=${P6_PROCS:-$(nproc)}
CJ=$(( J > 3 ? J - 2 : 1 ))
NB=""; [ -n "${P6_NO_BERT:-}" ] && NB="--no-bert"

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

say "======== p6g.sh | last refit: bsel on all fit documents + word-level score"
if [ -z "${P6_NO_PIP:-}" ] && ! python -c "import rouge_score, bert_score" 2>/dev/null; then
  say "pip   rouge-score bert-score"
  pip -q install rouge-score bert-score > "$L/pip.log" 2>&1
fi

step "$P6/results/bsel2_g_model.pkl" 60_bsel2_fit \
     python phase6_bsel.py --procs "$CJ" fit --tag bsel2 --text \
            --docs "${P6G_DOCS:-7000}" --budget-min "${P6G_MIN:-75}" || exit 1
grep -a -E 'surrogate oracle F|correlation' "$L/60_bsel2_fit.log" | tail -3 | cut -c1-200 \
  | while read -r l; do say "bsel2 $l"; done

for h in f g; do
  step "$P6/results/bsel2_$h/meta.json" "61_bsel2_$h" \
       python phase6_win.py --procs "$CJ" sweep --tag "bsel2_$h" --heads "bsel2_$h" --base bsel_best \
              --wins 420 --mults "1.0,1.25" --lams "0.15,0.45" $NB
  grep -a 'HELD-OUT HALF' "$L/61_bsel2_$h.log" | cut -c1-200 | while read -r l; do say "bsel2 $l"; done
done

Z=$(ls "$P6/submissions/_upload/submission_bsel2_"*.zip 2>/dev/null | wc -l)
say "======== REFIT DONE: $Z zips in submissions/_upload (submission_bsel2_f.zip, submission_bsel2_g.zip). Upload one only if its gain over bsel_best is at least +0.0010."
