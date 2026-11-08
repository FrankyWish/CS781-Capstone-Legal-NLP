#!/bin/bash
# Phase 7 -- Prior Case Retrieval.  ONE foreground cell on ONE L4.
#
#   bash p7.sh stage1    the whole day-1 baseline, unattended           ~50-80 min
#   bash p7.sh status    what exists and what the numbers are
#   bash p7.sh gate      just the integrity gate (CPU)
#
# Every step is skipped if its output already exists, so after a Colab disconnect you
# re-run the SAME cell and it carries on.  No setsid, no background jobs (Phase 5 lost
# four runs to them).
#
# stage1 does, in order:
#   gate      shapes, gold-set sizes, the F1@7 CEILING, dev/test contamination,
#             train->test pool overlap, the citation mask, the popularity prior
#   tok       BPE-tokenise every query and candidate once (CPU, all cores)
#   bm25      sweep k1/b/qtf/max-df on devx, then the best config -> dev + test
#   tfidf     cosine TF-IDF, same index
#   dense     M2's bidirectional encoder and M1's decoder, mean-pooled over chunks,
#             zero-shot from Phase 1 (the number contrastive training has to beat)
#   fuse      RRF / z-score fusion, weights chosen on devx           -> kind=hybrid
#   zips      submissions/_upload/*.zip + SUBMIT_ORDER.md
#
# Knobs (env): P7_TOPN retrieved depth (100) · P7_CHUNK tokens per chunk (1024)
#   P7_MAXCH chunks per document (4) · P7_EMB_BATCH chunks per forward (16)
#   P7_NO_PIP=1 skip pip · P7_PROCS tokeniser processes
set -u
cd "$(dirname "$0")"
P7=${CS781_P7:-/content/drive/MyDrive/cs781_capstone/phase7}
L=$P7/logs
mkdir -p "$L" "$P7/results" "$P7/submissions" "$P7/cache"
Q=$L/queue.txt
say() { echo "[$(date +%H:%M)] $*" | tee -a "$Q"; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
J=${P7_PROCS:-$(nproc)}
TOPN=${P7_TOPN:-100}
CHUNK=${P7_CHUNK:-1024}
MAXCH=${P7_MAXCH:-4}
EB=${P7_EMB_BATCH:-16}
export P7_TOPN=$TOPN

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
  [ -n "${P7_NO_PIP:-}" ] && { say "pip   skipped (P7_NO_PIP)"; return 0; }
  python -c "import scipy, huggingface_hub" 2>/dev/null && return 0
  say "pip   scipy huggingface_hub"
  pip -q install scipy huggingface_hub > "$L/pip.log" 2>&1
}

gpu_check() {
  local n; n=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
  say "gpu   ${n:-none}"
  case "$n" in
    *T4*) say "STOP: T4 runtime (bf16 is emulated on SM75, ~30x slow). Switch to an L4."
          return 1 ;;
    "")   say "WARN  no GPU: the lexical systems still run, the dense ones will crawl" ;;
  esac
  return 0
}

gate_summary() {
  for k in 'queries (' 'scoring cost' 'gold:' 'CEILING' 'contamination' 'pool overlap' \
           'self-match' 'masks' 'prior:' 'random 7' 'distinct candidates are cited' \
           'idf-weighted' ; do
    grep -a -m8 "$k" "$L/00_gate.log" | cut -c1-230 | while read -r l; do say "gate  $l"; done
  done
  grep -a '!!' "$L/00_gate.log" | cut -c1-230 | while read -r l; do say "gate  $l"; done
}

# -------------------------------------------------------------------- CPU ---
cpu() {
  deps
  step "$P7/results/gate.json" 00_gate python phase7_data.py gate || return 1
  gate_summary
  step "$P7/cache/tok_test_c.npz" 01_tok \
       python phase7_enc.py tok --splits devx,test --procs "$J" || return 1
  grep -a ': .* docs, tokens' "$L/01_tok.log" | cut -c1-200 | while read -r l; do say "tok   $l"; done
  step "$P7/results/bm25_sweep.json" 02_bm25_sweep \
       python phase7_rank.py bm25 --sweep || return 1
  grep -a 'BEST bm25' "$L/02_bm25_sweep.log" | while read -r l; do say "bm25  $l"; done
  step "$P7/results/bm25/test_rank.json" 03_bm25 \
       python phase7_rank.py bm25 --tag bm25 --use-sweep --splits devx,test --topn "$TOPN" || return 1
  step "$P7/results/tfidf/test_rank.json" 04_tfidf \
       python phase7_rank.py tfidf --tag tfidf --splits devx,test --topn "$TOPN" || true
  for f in 03_bm25 04_tfidf; do
    grep -a -E '/(devx|dev) ' "$L/$f.log" | cut -c1-200 | while read -r l; do say "lex   $l"; done
  done
}

# -------------------------------------------------------------------- GPU ---
dense() {        # dense <tag> <model> [extra args...]
  local tag=$1 model=$2; shift 2
  step "$P7/cache/emb_${tag}_test_c.npz" "1${tag}_embed" \
       python phase7_enc.py embed --tag "$tag" --model "$model" --split devx,test \
              --chunk "$CHUNK" --max-chunks "$MAXCH" --batch "$EB" "$@" || return 1
  step "$P7/results/$tag/test_rank.json" "1${tag}_rank" \
       python phase7_enc.py dense --tag "$tag" --agg mean --splits devx,test --topn "$TOPN" || return 1
  grep -a "$tag/devx" "$L/1${tag}_rank.log" | cut -c1-200 | while read -r l; do say "dense $l"; done
}

gpu() {
  gpu_check || return 1
  dense dense_m2 M2_LexArcED || return 1          # bidirectional encoder: the one to beat
  dense dense_m1 M1_SC || true                    # causal decoder, mean-pooled
  # same embeddings, chunk-level late interaction instead of one mean vector
  step "$P7/results/dense_m2_max/test_rank.json" 1dense_m2_max \
       python phase7_enc.py dense --tag dense_m2_max --emb-tag dense_m2 --agg maxsim \
              --splits devx,test --topn "$TOPN" || true
  grep -a 'dense_m2_max/devx' "$L/1dense_m2_max.log" | cut -c1-200 | \
    while read -r l; do say "dense $l"; done
}

# ------------------------------------------------------------- fuse + zip ---
fuse_and_zip() {
  local members=""
  for t in bm25 tfidf dense_m2 dense_m1 dense_m2_max; do
    [ -f "$P7/results/$t/test_rank.json" ] && members="${members:+$members,}$t"
  done
  say "fuse  members: $members"
  step "$P7/results/fuse/test_rank.json" 20_fuse \
       python phase7_rank.py fuse --tag fuse --members "$members" --kind hybrid \
              --splits devx,test --topn "$TOPN" || true
  grep -a -E 'BEST fusion|fuse/devx' "$L/20_fuse.log" | cut -c1-200 | \
    while read -r l; do say "fuse  $l"; done
  # one zip per system, so every number on the leaderboard has a local twin
  for t in bm25 tfidf dense_m2 dense_m1 dense_m2_max fuse; do
    [ -f "$P7/results/$t/test_rank.json" ] || continue
    case "$t" in
      bm25|tfidf) k=lexical ;;
      fuse)       k=hybrid ;;
      *)          k=model ;;
    esac
    step "$P7/submissions/$t/submission.zip" "21_zip_$t" \
         python phase7_rank.py zip --tag "$t" --kind "$k" || true
  done
  python - "$P7" <<'EOF' | tee "$P7/SUBMIT_ORDER.md" | while read -r l; do say "order $l"; done
import glob, json, os, sys
P7 = sys.argv[1]
rows = []
for m in sorted(glob.glob(f"{P7}/results/*/meta.json")):
    d = json.load(open(m))
    sc = d.get("devx_scores") or d.get("dev_scores") or {}
    if not os.path.exists(f"{P7}/submissions/{d.get('tag','')}/submission.zip"):
        continue
    rows.append((sc.get("f1") or 0.0, d.get("tag"), d.get("kind", "?"), sc.get("hits"),
                 sc.get("max_hits"), sc.get("ceiling_f1")))
rows.sort(reverse=True)
print("# Phase 7 - upload order (best devx Micro-F1@7 first)")
print()
print("| # | tag | kind | devx F1@7 | hits | ceiling |")
print("|---|-----|------|-----------|------|---------|")
for i, (f1, tag, kind, h, mh, ceil) in enumerate(rows, 1):
    print(f"| {i} | `{tag}` | {kind} | {f1:.4f} | {h}/{mh} | {ceil and round(ceil,4)} |")
print()
print("Upload `submissions/_upload/submission_<tag>.zip` in this order. Right-click the")
print("single file in Drive -- a folder download is a wrapper zip. Check the daily and")
print("total caps on the Codabench page first; identical resubmissions still count.")
print("The leaderboard shows the PUSHED submission, not the best one.")
EOF
}

status() {
  python - "$P7" <<'EOF'
import glob, json, os, sys
P7 = sys.argv[1]
g = f"{P7}/results/gate.json"
if os.path.exists(g):
    G = json.load(open(g))
    n = G["n"]
    print(f"gate: queries train/dev/test {n['train']['queries']}/{n['dev']['queries']}/"
          f"{n['test']['queries']} | pools {n['train']['candidates']}/"
          f"{n['dev']['candidates']}/{n['test']['candidates']}")
    for s in ("train", "dev"):
        d = G.get(f"{s}_gold", {})
        if d:
            print(f"  {s} gold {d['total']} labels, {d['sizes'].get('mean')}/query, "
                  f"{d['over_K']} queries over 7 | CEILING F1@7 {d['ceiling_f1']} | "
                  f"random {d['expected_random_f1']}")
    o = G.get("overlap", {})
    if o:
        print(f"  dev queries copying a test query: "
              f"{o['dev_queries_in_test_queries']['n']} | train pool covers "
              f"{o['train_cands_in_test_cands']['frac_of_b']*100:.1f}% of the test pool | "
              f"test queries inside the test pool {o['test_queries_in_test_pool']['n']}")
    p = G.get("prior", {})
    if p:
        print(f"  prior: top-7 most-cited scores {p['top7_from_train_f1']} on devx "
              f"(random {p['random_f1']})")
else:
    print("gate: not run yet")
rows = []
for m in sorted(glob.glob(f"{P7}/results/*/meta.json")):
    d = json.load(open(m))
    sc = d.get("devx_scores") or d.get("dev_scores") or {}
    rc = sc.get("recall_curve") or {}
    rows.append((sc.get("f1") or 0.0, d.get("tag", "?"), d.get("kind", "?"),
                 sc.get("hits"), sc.get("max_hits"), rc.get("r@7"), rc.get("r@100"),
                 os.path.exists(f"{P7}/submissions/{d.get('tag','')}/submission.zip")))
print(f"\n{'tag':20s} {'kind':8s} {'devx F1@7':>10s} {'hits':>11s} {'r@7':>6s} {'r@100':>6s} zip")
for f1, tag, kind, h, mh, r7, r100, z in sorted(rows, reverse=True):
    f = lambda x: "  -  " if x is None else f"{x:5.3f}"
    print(f"{tag:20s} {kind:8s} {f1:10.4f} {str(h)+'/'+str(mh):>11s} {f(r7)} {f(r100)} "
          f"{'yes' if z else '-'}")
z = sorted(glob.glob(f"{P7}/submissions/_upload/*.zip"))
print(f"\nupload-ready zips: {len(z)}  (order: {P7}/SUBMIT_ORDER.md)")
EOF
}

stage1() {
  say "======== p7.sh stage1 | $J cores | topn $TOPN | chunk $CHUNK x $MAXCH"
  cpu || { say "CPU STAGE FAILED -- nothing downstream can run; fix and re-run the cell"; return 1; }
  gpu || say "GPU stage incomplete -- the lexical systems and their zips still follow"
  fuse_and_zip
  status
  say "======== STAGE1 DONE"
}

case "${1:-}" in
  stage1|all) stage1 ;;
  gate) deps; step "$P7/results/gate.json" 00_gate python phase7_data.py gate; gate_summary ;;
  cpu) cpu ;;
  gpu) gpu ;;
  zip) fuse_and_zip ;;
  status) status ;;
  *) sed -n '2,25p' "$0" ;;
esac
