# Phase 6 — how each run was done (one L4, one cell, unattended)

**Phase closed, rank 1 at 0.8606.** This file records how every script was run, newest first. All five ran
for real on a Colab L4, in this order:

| script | ran (IST) | what it produced |
|---|---|---|
| `p6.sh all` | Nov 1 14:53 – Nov 2 | gate, extraction, the 7 required runs, MBR and the reranker |
| `p6f.sh` | Nov 2, done 06:56 | `bsel_best`, `win_best`, `m*_win`, `cons_*`, `stitch_*`, 60 numbered zips |
| `p6g.sh` | Nov 2, done 11:22 | `bsel2_f`, `bsel2_g` |
| `p6h.sh` | Nov 2 22:03 – Nov 3 01:37 | cached labels, `bsel3`, `nsel_m1`, `nsel_m2`, 7 `nsel_*` zips |
| `p6i.sh` | Nov 3 14:16 – 16:19 | `nsel_m3`, `nsel_m4`, 7 `nsel4_*` zips, including the final `nsel4_best` |

Paths below are on Drive (`MyDrive/cs781_capstone/phase6/`). The zips that were downloaded now live on this
laptop in `submissions/<family>/<tag>.zip` (see `submissions/README.md`). The `_today`, `_final`, `_final2`,
`_final3` and `_upload` folders named below exist only on Drive.

## M3 + M4 scorers: `p6i.sh` (about 2 h 30 min, Nov 3 afternoon)

Trains M3 and M4 as two more sentence scorers, the same recipe as `nsel_m1` and `nsel_m2`
(55 min each), then re-sweeps over all four trunks. With 3 or more trunks the stack gets each
trunk's score as its own feature. Zips land in `submissions/_final3/` as `nsel4_*`, and each
`VS nsel_stack` line is the paired held-out gain over the current best (test 0.8596). It needs
p6h's outputs on Drive and writes nothing into `_final2`. Smoke-tested on fake data with tiny M3
and M4 trunks, including the skip-only re-run. Ran for real Nov 3, 14:16–16:19 IST: control 600/600.

Upload `p6i.sh` and `phase6_nsel.py` to `phase6/code/`, replacing what is there. Run cell 1
from section 2 below, then:

**Check cell. It must print `CODE OK`:**
```python
import hashlib
want = {'p6i.sh': 'cdddff9eff3fcb4878b6726c7cb46b80', 'phase6_nsel.py': 'b00df866d744783c534d40fb45fd19be',
        'phase6_bsel.py': '0aab87efdda9fd7567bc2288021d7b4a', 'phase6_win.py': '7eea8f85ed204c507fbc93a43a17cbff'}
bad = [f for f, h in want.items() if not os.path.exists(f) or hashlib.md5(open(f, 'rb').read()).hexdigest() != h]
print('CODE', 'OK' if not bad else f'STOP, wrong or missing: {bad}')
```

**Run cell:**
```python
!bash p6i.sh
from google.colab import runtime; runtime.unassign()
```
The last line of `logs/queue.txt` is `M3M4 DONE`. If one trunk fails, the sweep still runs on
the trunks that finished. Upload a `nsel4_*` zip only if its `VS nsel_stack` gain is positive.

## Phase-1 sentence scorer: `p6h.sh` (about 4 h 15 min, Nov 2 night)

Caches the metric's sentence labels, refits `bsel` without the word-level score, fine-tunes
the top 4 blocks of M1 and of M2's encoder to score sentences, then writes zips ranked by
held-out gain to `submissions/_final2/`. Smoke-tested on fake data with tiny trunks,
including skip and resume. Ran for real Nov 2 22:03 – Nov 3 01:37 IST: control 600/600.

Upload `p6h.sh` and `phase6_nsel.py` to `phase6/code/`. Run cell 1 from section 2 below, then:

**Check cell. It must print `CODE OK`:**
```python
import hashlib
want = {'p6h.sh': 'fa7caad8b41c0d86a4f5c27eb5fea276', 'phase6_nsel.py': 'fdd5208ef3932e72384565c8f9cab430',
        'phase6_bsel.py': '0aab87efdda9fd7567bc2288021d7b4a', 'phase6_win.py': '7eea8f85ed204c507fbc93a43a17cbff'}
bad = [f for f, h in want.items() if not os.path.exists(f) or hashlib.md5(open(f, 'rb').read()).hexdigest() != h]
print('CODE', 'OK' if not bad else f'STOP, wrong or missing: {bad}')
```

**Run cell:**
```python
!bash p6h.sh
from google.colab import runtime; runtime.unassign()
```
The last line of `logs/queue.txt` is `SCORER DONE`. After a disconnect, re-run the same cells:
labels are saved every 10 minutes and training resumes from its last checkpoint. The
`CONTROL` line must say 600/600 or close to it, otherwise the held-out gains are not comparable.

## Last refit: `p6g.sh` (about 1 h 45 min, Nov 2)

Refits `bsel`, the only lever that moved the score, on all fit documents with a
word-level score added. It writes `submissions/_upload/submission_bsel2_f.zip` and
`submission_bsel2_g.zip`, and leaves `submissions/_today` alone. Smoke-tested on fake data,
then ran for real on Nov 2 (done 11:22 IST).

Upload `p6g.sh`, `phase6_bsel.py` and `phase6_win.py` to `phase6/code/`, replacing what is
there. Run cell 1 from section 2 below, then:

**Check cell. It must print `CODE OK`:**
```python
import hashlib
want = {'p6g.sh': '12245cca5c273987d566fe198ca5ae4d', 'phase6_bsel.py': '0aab87efdda9fd7567bc2288021d7b4a',
        'phase6_win.py': '7eea8f85ed204c507fbc93a43a17cbff'}
bad = [f for f, h in want.items() if not os.path.exists(f) or hashlib.md5(open(f, 'rb').read()).hexdigest() != h]
print('CODE', 'OK' if not bad else f'STOP, wrong or missing: {bad}')
```

**Run cell:**
```python
!bash p6g.sh
from google.colab import runtime; runtime.unassign()
```
The last line of `logs/queue.txt` is `REFIT DONE`. Upload a `bsel2` zip only if its
`HELD-OUT HALF` line shows a gain of at least +0.0010 over `bsel_best`. After a disconnect,
re-run the same cells.

## Last day: `p6f.sh` (Nov 2)

`p6f.sh` picks up from whatever exists on Drive. It finishes the main run if needed,
adds the final-day systems, rebuilds the ensembles, and writes numbered zips to
`submissions/_today/`.

Upload these six to `phase6/code/`, replacing what is there: `p6f.sh`,
`phase6_batch.py`, `phase6_bsel.py`, `phase6_win.py`, `phase6_ens.py`,
`phase6_external.py`. Leave `p6.sh` alone. Run cell 1 from section 2 below, then:

**Check cell, takes seconds.** It must print `CODE OK`:
```python
import hashlib
want = {'p6f.sh': '736bcbf7dd01795c23670414173975df', 'phase6_batch.py': 'ae8ad3e6fc65713e4f7eb2501569b06f',
        'phase6_bsel.py': '819b62ff1d32d3e9c9d31fc1b608f4de', 'phase6_win.py': 'd7bd28cbc26db871b33aa7ae4b8288c3',
        'phase6_ens.py': 'f57a5a7f0ccdac1b4964ff4bc1921755', 'phase6_external.py': '8e012d90aad746c4d2af7aeb86ce6032',
        'p6.sh': '3b363a4c95530e35cf6d3220928e609c'}
bad = [f for f, h in want.items() if not os.path.exists(f) or hashlib.md5(open(f, 'rb').read()).hexdigest() != h]
print('CODE', 'OK' if not bad else f'STOP, wrong or missing: {bad}')
!tail -4 ../logs/queue.txt
```

**Run cell:**
```python
!P6F_HOURS=7 bash p6f.sh
from google.colab import runtime; runtime.unassign()
```
`P6F_HOURS` is the hours until you are back. GPU work stops starting 2 h before it, so
the ensembles and zips always finish in time. After a disconnect, re-run the same cells.
The laptop must stay awake with the tab open. Progress is in `logs/queue.txt`, and the
last line is `READY: N numbered zips`.

## The original full run

`bash p6.sh all` does the whole phase: integrity gate → extractive system → the 7
required fine-tuning runs (+ extras while time remains) → MBR / reranker → every
zip → `SUBMIT_ORDER.md`, then the cell releases the runtime.

## 1. Upload (laptop, ~2 min)

In Google Drive, open `MyDrive/cs781_capstone/` and create `phase6/code/` if it
doesn't exist. Upload these six files from `E:\cs781\capstone_Project\phase6\code\`
into it, replacing anything already there:

```
p6.sh  phase6_data.py  phase6_sel.py  phase6_sum.py  phase6_ens.py  phase6_peft.py
```

Also check `MyDrive/cs781_capstone/tools/cs781_submit.py` exists (it did for Phase 5).
If it's missing, upload `E:\cs781\capstone_Project\tools\cs781_submit.py` there — without
it the zips are still written, just not run through the validator.

## 2. Colab (~1 min)

New notebook → **Runtime → Change runtime type → L4 GPU** → Save.

**Cell 1 — setup:**
```python
import os
from google.colab import drive
if not os.path.ismount('/content/drive'):
    drive.mount('/content/drive')
BASE = '/content/drive/MyDrive/cs781_capstone'
os.environ.update(CS781_BASE=BASE, CS781_P1=f'{BASE}/phase1', CS781_P6=f'{BASE}/phase6',
                  CS781_LIB=f'{BASE}/phase1/shared/lib', PYTHONUNBUFFERED='1')
os.environ['HF_TOKEN'] = open(f'{BASE}/.hf_token').read().strip()
os.chdir(f'{BASE}/phase6/code')
!sed -i 's/\r$//' *.sh *.py && ls && nvidia-smi --query-gpu=name --format=csv,noheader
```
The last line must print `NVIDIA L4`. If it says T4, change the runtime type — the
script refuses to run on a T4 anyway.

**Preflight — run once, takes seconds.** Both lines must say `OK` and the file list
must show `Phase_6_Summ` train and test files before you start cell 2:
```python
import hashlib
from huggingface_hub import HfApi
P1 = os.environ['CS781_P1']
want = {'p6.sh': '3b363a4c95530e35cf6d3220928e609c', 'phase6_data.py': '93b7d32aefea78cdce91e4979291c640',
        'phase6_ens.py': 'f358566f8b0a6a3647c09cddad6ca7a6', 'phase6_peft.py': '2e81b3893797d2ab09562868b8b3497a',
        'phase6_sel.py': '9d34dbdca2f72bcfc4d14a674116dbf2', 'phase6_sum.py': '7c7b3b69852fa7346573fc68ae300d7e'}
bad = [f for f, h in want.items() if not os.path.exists(f) or hashlib.md5(open(f, 'rb').read()).hexdigest() != h]
need = [f'{P1}/{m}/ckpt/best.pt' for m in ('M1_SC', 'M2_LexArcED', 'M3_LexArcL', 'M4_LexArcP')]
need += [f'{P1}/shared/tokenizer/bpe_legal_32k.json', f'{BASE}/tools/cs781_submit.py']
need += [f'{P1}/shared/lib/{f}.py' for f in ('cs781_bpe', 'cs781_model', 'cs781_model_ed', 'cs781_model_ptr')]
miss = [f for f in need if not os.path.exists(f)]
print('CODE  ', 'OK' if not bad else f'STOP, wrong or missing: {bad}')
print('ASSETS', 'OK' if not miss else f'STOP, missing: {miss}')
print([f for f in HfApi().list_repo_files('Exploration-Lab/CS-781-Capstone', repo_type='dataset') if 'phase_6' in f.lower()])
```

**Cell 2 — everything, then release the GPU (run only after preflight):**
```python
!bash p6.sh all
from google.colab import runtime; runtime.unassign()
```

Leave the tab open and the laptop awake and plugged in. Closing the tab disconnects
the runtime on a non-Pro+ plan.

## 3. While you're away

Progress goes to `MyDrive/cs781_capstone/phase6/logs/queue.txt` — readable from the
Drive app on your phone. Rough timeline from the moment cell 2 starts:

| after | what is finished |
|---|---|
| ~40 min | gate, sentence scorer, input views, tokenisation |
| ~1.5 h | extractive zips (`sel_best`, `sel_short`, `sel_long`), length calibration |
| ~1 h → ~5 h | the 7 required runs, ~35 min each (25 train + ~10 decode) |
| ~5–6.5 h | extras if the 6.5 h GPU deadline allows: `m4_full`, `m4_lorma_2048`, `m4_lorma_trunc` |
| last ~20 min | MBR, reranker, length bracket, every zip, `SUBMIT_ORDER.md` |

## 4. If Colab disconnects

Open the notebook again, run **cell 1, then cell 2 unchanged**. Every finished step
is skipped and an interrupted training run resumes from its last 10-minute
checkpoint. Nothing is lost except the minutes since that checkpoint.

## 5. When you're back

In a Colab terminal (no `!`):
```
cd /content/drive/MyDrive/cs781_capstone/phase6/code && bash p6.sh status
tail -40 /content/drive/MyDrive/cs781_capstone/phase6/logs/queue.txt
```
Or, if the runtime is gone, open `phase6/logs/queue.txt` and `phase6/SUBMIT_ORDER.md`
in Drive. Paste both to Claude.

Uploads: `phase6/submissions/_upload/submission_<tag>.zip`, in `SUBMIT_ORDER.md`
order. Caps are 40/day and 160 total.

## Knobs (prefix the command, e.g. `!BUDGET=35 bash p6.sh all`)

`BUDGET=25` training minutes per run · `P6_HOURS=6.5` GPU-queue deadline (runs that
would start after it are skipped, then stage 3 still runs) · `P6_GEN_DEV=200` decoded
held-out docs per run · `P6_BERT=200` held-out docs scored with BERTScore in the
length sweep · `P6_NO_BERT=1` ROUGE only (much faster, not the real metric).

Other entry points: `bash p6.sh stage1` (CPU part only), `bash p6.sh gpu`,
`bash p6.sh ens` (rebuild ensembles + zips after new runs), `bash p6.sh status`.

## Corpus scan (measure only, runs beside the main cell)

Answers one question: does the Phase-1 corpus hold the test documents with their
headnotes? Upload `phase6_external.py`, `phase6_win.py` and the updated `phase6_ens.py` to `phase6/code/`. Once `queue.txt` shows
`done  12_sel_best`, open Colab's Terminal on the same runtime (no `!`):
```
cd /content/drive/MyDrive/cs781_capstone/phase6/code
export CS781_BASE=/content/drive/MyDrive/cs781_capstone CS781_P1=$CS781_BASE/phase1 CS781_P6=$CS781_BASE/phase6 PYTHONUNBUFFERED=1
export HF_TOKEN=$(cat $CS781_BASE/.hf_token)
sed -i 's/\r$//' phase6_external.py
python phase6_external.py scan --procs 3 2>&1 | tee ../logs/40_corpus_scan.log
python phase6_win.py --procs 3 fit 2>&1 | tee ../logs/41_win.log
python phase6_win.py --procs 3 sweep 2>&1 | tee -a ../logs/41_win.log
```
The last two lines are the window-aimed extraction (`phase6_win.py`), which needs `done  12_sel_best` in `queue.txt` first.
Writes `results/corpus_scan.json`. It can build no submission and keeps only counts
and lengths for test documents.
