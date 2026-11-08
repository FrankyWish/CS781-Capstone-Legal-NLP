# Phase 5 — how to run (one L4 runtime, plain commands)

Steps 1-4 are how rounds 1-2 ran (required models, TF-IDF, ensembles). The final-day stacker runs that produced
the best scores are in step 5.

**1. Upload (laptop).** The files inside `E:\cs781\capstone_Project\phase5\code\` (all 13) to
`MyDrive/cs781_capstone/phase5/code/`, replacing what is there.

**2. New Colab notebook, Runtime → L4. Setup cell:**
```python
import os
from google.colab import drive
if not os.path.ismount('/content/drive'):
    drive.mount('/content/drive')
BASE = '/content/drive/MyDrive/cs781_capstone'
os.environ.update(CS781_BASE=BASE, CS781_P1=f'{BASE}/phase1', CS781_P5=f'{BASE}/phase5',
                  CS781_LIB=f'{BASE}/phase1/shared/lib', CS781_CORPUS_DIR='/content/p1corpus', PYTHONUNBUFFERED='1')
os.environ['HF_TOKEN'] = open(f'{BASE}/.hf_token').read().strip()
os.chdir(f'{BASE}/phase5/code')
!sed -i 's/\r$//' *.sh *.py && ls && nvidia-smi --query-gpu=name --format=csv,noheader
```

**Going out? One command does both goals (~6–7 h):** `!bash p5.sh all`, then the last cell below.

**3. Goal 1 — the required fine-tuned models (~4.5 h, background):**
```
!bash p5.sh goal1
```
GPU: LoRMA/M4 · Adapters/M1 · Partial FT/M2 · QLoRA/M3 · Prefix/M1 · KD half-depth student.
Spare CPU cores at the same time: data gate, silver data from the Phase-1 judgments, TF-IDF models.

**4. Goal 2 — top score (~2 h, background), after `GOAL1 DONE`:**
```
!bash p5.sh goal2
```
Retrains the best goal-1 method on gold + silver data (if silver won on dev), then the final ensembles.

**Any time:** `!bash p5.sh status`  ·  **Last cell** (blocks until the running goal is done, then stops billing):
```
!bash p5.sh wait
from google.colab import runtime; runtime.unassign()
```

Results on Drive: `phase5/SUBMIT_ORDER.md` (zips best-first), `phase5/STATUS.md`, `phase5/submissions/<tag>/submission.zip`,
`phase5/logs/queue.txt`. Knobs: `!BUDGET=25 bash p5.sh goal1` (train minutes per run), `!BUDGET2=90 bash p5.sh goal2`.

**5. Final day (Oct 28): stacker and dev-cross-fitted members, CPU on the same L4 runtime.**
Run in the **foreground** (one blocking cell): Colab recycled `setsid` background jobs four times that day. Every step
skips itself when `results/<tag>/meta.json` already exists, so a lost runtime is resumed by rerunning the same cell.
```python
open('/content/r8.sh', 'w').write(r'''cd /content/drive/MyDrive/cs781_capstone/phase5/code
sed -i 's/\r$//' *.py
Q=../logs/queue.txt; L=../logs/r8.log; R=../results
say() { echo "[$(date +%H:%M)] $*" | tee -a $Q; }
have() { [ -f $R/$1/meta.json ]; }
st() { if have $2; then say "skip $2 (done)"; return; fi; python phase5_stack.py --submit --k 35 --lr 0.04 --iters 900 "$@" >> $L 2>&1; say "$(grep -a ': strategy' $L | tail -1 | cut -c1-150)"; }
ens() { t=$1; shift; if have $t; then say "skip $t (done)"; return; fi; python phase5_ens.py ens --tag $t --submit --combine logit "$@" >> $L 2>&1; say "$(grep -a "$t: strategy" $L | tail -1 | cut -c1-150)"; }
have lex_w12x || python phase5_lex.py --tag lex_w12x --crossfit 4 --jobs 10 >> $L 2>&1
have lex_w13x || python phase5_lex.py --tag lex_w13x --ngram 3 --max-feat 800000 --crossfit 4 --jobs 11 >> $L 2>&1
for s in 0 1 2 3 4 5 6 7; do st --tag s6_s$s --seed $((781+100*s)); done
ens s6_bag --runs s6_s0,s6_s1,s6_s2,s6_s3,s6_s4,s6_s5,s6_s6,s6_s7 --weights 1,1,1,1,1,1,1,1
say "DONE"
''')
!bash /content/r8.sh
from google.colab import runtime; runtime.unassign()
```
Timings on the L4's 12 CPUs: `lex_w12x` ~60 min (4 folds + final fit), `lex_w13x` ~120 min, one stacker run
~1.5-6 min (longer on a fresh runtime: data download). `phase5_stack.py` uses every neural/lex run that has a
`meta.json`, except `--refit-dev` runs. Volume variants: `python phase5_ens.py rate --run <tag> --factors 0.95,1.05`.
