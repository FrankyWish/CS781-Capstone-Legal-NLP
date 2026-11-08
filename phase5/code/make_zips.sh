#!/bin/bash
# Phase 5: build every submission zip from whatever runs exist. CPU only, minutes.
# Safe to re-run after more GPU runs finish: zips are rebuilt from results/.
#
#   bash make_zips.sh
#
# Model-only zips: one per finished neural run, plus ens_model (greedy average of
# the neural runs). Hybrid zips: lex_global, lex_perlabel, ens_hyb (neural + TF-IDF),
# ens_hyb_z (same members, z-scored logits instead of probabilities).

cd "$(dirname "$0")"
P5=${CS781_P5:-/content/drive/MyDrive/cs781_capstone/phase5}
have() { [ -f "$P5/results/$1/meta.json" ]; }
E="python phase5_ens.py"

lex=""
if have lex_w12; then
  $E zip --run lex_w12 --strategy global       --tag lex_global
  $E zip --run lex_w12 --strategy perlabel_s5  --tag lex_perlabel
  lex="lex_w12"
fi
if have lex_w12_sil; then
  $E zip --run lex_w12_sil --strategy auto     --tag lex_silver
  lex="${lex:+$lex,}lex_w12_sil"
fi
# every lexical-type run (TF-IDF LR, kNN ...) joins the hybrid ensembles
lex=$(python - "$P5" <<'EOF3' | tr -d '\r'
import glob, json, sys
print(",".join(sorted(json.load(open(f))["tag"] for f in glob.glob(f"{sys.argv[1]}/results/*/meta.json")
                      if json.load(open(f)).get("kind") == "lex")))
EOF3
)
neural=""
for t in $(python - "$P5" <<'EOF2' | tr -d '\r'
import glob, json, os, sys
for f in sorted(glob.glob(f"{sys.argv[1]}/results/*/meta.json")):
    m = json.load(open(f))
    if m.get("kind") == "neural":
        print(m["tag"])
EOF2
); do
  $E zip --run $t --strategy auto --tag $t
  neural="$neural$t,"
done
if [ -n "$neural" ]; then
  $E ens --runs "$neural" --greedy ${GREEDY:-12} --tag ens_model --submit
  if [ -n "$lex" ]; then
    $E ens --runs "${neural}${lex}" --greedy ${GREEDY:-12} --tag ens_hyb   --submit
    $E ens --runs "${neural}${lex}" --greedy ${GREEDY:-12} --tag ens_hyb_z --submit --combine z
  fi
fi
$E status > /dev/null
$E plan
