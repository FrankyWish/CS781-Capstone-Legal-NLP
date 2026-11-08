#!/usr/bin/env bash
# Phase 7, part B -- everything that needs only CPU, so the GPU quota stays free for
# training: wider lexical configs, two fusions, eight stacker variants, and one zip per
# system.  ~27 upload-ready submissions.
#
#   bash p7b.sh all          everything below, in order
#   bash p7b.sh lex          lexical variants + fusions only
#   bash p7b.sh stack        stacker variants only
#   bash p7b.sh zips         (re)build every zip and the upload order
#   bash p7b.sh status
#
# Every step is skip-if-done on its output file, so re-running after a Colab disconnect
# continues where it stopped.  Nothing here touches a GPU.
set -u

P7=${CS781_P7:-/content/drive/MyDrive/cs781_capstone/phase7}
L=$P7/logs
Q=$L/queueb.txt
TOPN=${P7_TOPN:-200}
POOL=${P7_POOL:-50}
mkdir -p "$L" "$P7/results" "$P7/submissions"

# the members that have train+devx+test runs; phase7_stack skips any that do not
ALL=tfidf_sqrt,bm25_hi,bm25,tfidf,dense_m2_c8_max,dense_m2_c8,dense_m2_max,dense_m2,dense_m1,dense_m3,dense_m4
LEXM=tfidf_sqrt,bm25_hi,bm25,tfidf
C8M=tfidf_sqrt,bm25_hi,dense_m2_c8_max,dense_m2_c8
TOPM=tfidf_sqrt,bm25_hi,dense_m2_c8_max,dense_m2_c8

say() { printf '[%s] %s\n' "$(date +%H:%M)" "$*" | tee -a "$Q"; }

step() {                       # step <marker> <name> <cmd...>
  local m=$1 n=$2; shift 2
  if [ -e "$m" ]; then say "skip  $n"; return 0; fi
  say "run   $n"
  if "$@" >"$L/$n.log" 2>&1; then
    say "done  $n"
    grep -aE '/devx |train OOF|dropping|skip ' "$L/$n.log" | cut -c1-200 | \
      while read -r l; do say "      $l"; done
    return 0
  fi
  say "FAIL  $n -- tail of $L/$n.log:"
  tail -20 "$L/$n.log" | tee -a "$Q"
  return 1
}

lexrun() {                     # lexrun <tag> <scheme> <args...>
  local tag=$1 scheme=$2; shift 2
  step "$P7/results/$tag/test_rank.json" "b_$tag" \
       python phase7_rank.py "$scheme" --tag "$tag" "$@" \
              --splits devx,test --topn "$TOPN" || true
}

stk() {                        # stk <tag> <kind> <args...>
  local tag=$1 kind=$2; shift 2
  step "$P7/results/$tag/test_rank.json" "b_$tag" \
       python phase7_stack.py fit --tag "$tag" --kind "$kind" \
              --pool-depth "$POOL" --topn "$TOPN" "$@" || true
}

# --------------------------------------------------------------- lexical + fusion ---
lex() {
  say "======== lexical variants and fusions"
  lexrun tfidf_s03  tfidf --qtf sqrt --max-df 0.3
  lexrun tfidf_s015 tfidf --qtf sqrt --max-df 0.15
  lexrun tfidf_bin  tfidf --qtf binary --max-df 0.2
  lexrun bm25_x     bm25  --k1 4.0 --b 1.0 --qtf sqrt --max-df 0.2
  lexrun bm25_lo    bm25  --k1 1.2 --b 0.75 --qtf sqrt --max-df 0.2
  step "$P7/results/fuse_top/test_rank.json" b_fuse_top \
       python phase7_rank.py fuse --tag fuse_top --members "$TOPM" \
              --kind hybrid --splits devx,test --topn "$TOPN" || true
  step "$P7/results/fuse_all/test_rank.json" b_fuse_all \
       python phase7_rank.py fuse --tag fuse_all \
              --members tfidf_sqrt,bm25_hi,dense_m2_c8_max,dense_m2_c8,dense_m2_max,dense_m2,dense_m1,dense_m3,dense_m4 \
              --kind hybrid --splits devx,test --topn "$TOPN" || true
}

# ------------------------------------------------------------------------ stackers ---
stack() {
  say "======== stacker variants (pool depth $POOL unless stated)"
  stk stack       hybrid  --members "$ALL" --importance
  stk stack_model model   --members-model
  stk stack_lex   lexical --members "$LEXM"
  stk stack_c8    hybrid  --members "$C8M"
  step "$P7/results/stack_d20/test_rank.json" b_stack_d20 \
       python phase7_stack.py fit --tag stack_d20 --kind hybrid --members "$ALL" \
              --pool-depth 20 --topn "$TOPN" || true
  step "$P7/results/stack_d100/test_rank.json" b_stack_d100 \
       python phase7_stack.py fit --tag stack_d100 --kind hybrid --members "$ALL" \
              --pool-depth 100 --topn "$TOPN" || true
  stk stack_deep  hybrid  --members "$ALL" --depth 5 --leaves 31
  stk stack_slow  hybrid  --members "$ALL" --lr 0.02 --iters 1500
  say "half-split selection report (choose on A, report on B, 2 SE bar)"
  python phase7_stack.py half \
         --tags tfidf_sqrt,stack,stack_model,stack_lex,stack_c8,stack_d20,stack_d100,stack_deep,stack_slow,fuse_top,fuse_all \
         2>&1 | tee -a "$Q" || true
}

# ---------------------------------------------------------------------------- zips ---
zips() {
  say "======== zips"
  for t in tfidf_sqrt bm25_hi tfidf_s03 tfidf_s015 tfidf_bin bm25_x bm25_lo \
           bm25 tfidf; do
    [ -d "$P7/results/$t" ] || continue
    step "$P7/submissions/$t/submission.zip" "z_$t" \
         python phase7_rank.py zip --tag "$t" --kind lexical || true
  done
  for t in dense_m2_c8_max dense_m2_c8 dense_m2_max dense_m2 dense_m1 dense_m3 \
           dense_m4 dense_m1_bidir stack_model; do
    [ -d "$P7/results/$t" ] || continue
    step "$P7/submissions/$t/submission.zip" "z_$t" \
         python phase7_rank.py zip --tag "$t" --kind model || true
  done
  for t in fuse fuse2 fuse_top fuse_all stack stack_c8 stack_d20 stack_d100 \
           stack_deep stack_slow; do
    [ -d "$P7/results/$t" ] || continue
    step "$P7/submissions/$t/submission.zip" "z_$t" \
         python phase7_rank.py zip --tag "$t" --kind hybrid || true
  done
  [ -d "$P7/results/stack_lex" ] && \
    step "$P7/submissions/stack_lex/submission.zip" z_stack_lex \
         python phase7_rank.py zip --tag stack_lex --kind lexical || true
  order
}

order() {
  python - "$P7" <<'PY' | tee "$P7/SUBMIT_ORDER.md" | tee -a "$Q"
import glob, json, os, sys
P7 = sys.argv[1]
rows = []
for m in sorted(glob.glob(f"{P7}/results/*/meta.json")):
    tag = os.path.basename(os.path.dirname(m))
    if not os.path.exists(f"{P7}/submissions/{tag}/submission.zip"):
        continue
    try:
        d = json.load(open(m))
    except Exception:
        continue
    s = d.get("devx_scores") or d.get("dev_scores") or {}
    f1 = s.get("f1")
    rows.append((f1 if isinstance(f1, (int, float)) else -1.0,
                 tag, d.get("kind", "?"), s.get("hits")))
rows.sort(key=lambda r: -r[0])
GATE = float(os.environ.get("P7_UPLOAD_MARGIN", "0.02"))
best = rows[0][0] if rows else 0.0
cut = best - GATE
print("# Phase 7 - upload order (best devx Micro-F1@7 first)\n")
print("Ceiling on devx is 0.6181. Upload the single `submission.zip` from each")
print("`submissions/<tag>/` - right-clicking a FOLDER in Drive gives a wrapper zip that")
print("Codabench rejects. 40 per day, 160 per phase; identical resubmissions still count.")
print("The leaderboard shows the PUSHED submission, not the best one.\n")
print(f"**Upload gate: devx >= {cut:.4f}** (best {best:.4f} minus {GATE:.2f}). The "
      f"dev-to-test regression is test = devx - 0.0095 with sd 0.015, so a system more "
      f"than ~2 sd below the best devx score cannot plausibly come out on top of the "
      f"leaderboard, and uploading it buys nothing but a slot. 27 paired points already "
      f"pin the regression, so the calibration argument for uploading broadly is spent.\n")
up = [r for r in rows if r[0] >= cut]
skip = [r for r in rows if r[0] < cut]
print(f"## Upload these {len(up)}\n")
print("| # | tag | kind | devx F1@7 | hits |")
print("|---|-----|------|-----------|------|")
for i, (f1, tag, kind, hits) in enumerate(up, 1):
    print(f"| {i} | `{tag}` | {kind} | {f1:.4f} | {hits if hits is not None else '-'} |")
print(f"\n## Do NOT upload ({len(skip)} below the gate; devx numbers are already in "
      f"EXPERIMENTS.md, which is what the brief asks to be reported)\n")
print("| tag | kind | devx F1@7 | hits |")
print("|-----|------|-----------|------|")
for f1, tag, kind, hits in skip:
    print(f"| `{tag}` | {kind} | {f1:.4f} | {hits if hits is not None else '-'} |")
print(f"\n{len(rows)} zips built, {len(up)} worth a submission slot.")
PY
}

status() {
  python phase7_rank.py show 2>&1 | tee -a "$Q" || true
  say "zips: $(ls -1 "$P7"/submissions/*/submission.zip 2>/dev/null | wc -l)"
}

case "${1:-all}" in
  lex)    lex ;;
  stack)  stack ;;
  zips)   zips ;;
  status) status ;;
  all)    say "======== p7b.sh all | topn $TOPN | pool $POOL | CPU only"
          lex; stack; zips; status
          say "======== PART B DONE" ;;
  *)      echo "usage: bash p7b.sh [all|lex|stack|zips|status]"; exit 2 ;;
esac
