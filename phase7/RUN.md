# Phase 7 — how to run every part

## Final pipeline: reproduce the 0.5157 submission (restart, 7–8 Nov)

Everything below this section is the 3–7 Nov text-retrieval work. It produced the saved
base runs (`stack_new`, `citep`, `maxp2`, …) that this pipeline builds on, and is kept for
history. The final system needs **CPU only**, about 25 minutes plus the corpus download,
and these steps in order. Run them from `cs781_capstone/phase7/code` on Drive after the
setup and gated-data auth cells below:

```python
# 1. transductive decoding of the saved base runs (Sinkhorn, config chosen on train only)
!python phase7_ot.py all            # -> ot_zmix_*, results/ot_diag.json
!python phase7_ot2.py all           # -> ot2_zst3_p (the base under the rules), SUBMIT_ORDER_Z2.md
# 2. corpus lookup (Phase-1 corpus, kind=lookup): recover what the queries' masks hid
!python phase7_overlap.py scan --splits test,devx --sources train_supreme_court
!python phase7_names.py scan        # candidates' real titles from their corpus copies
!python phase7_hc.py                # mask-free rescan of the SC corpus (merges, never overwrites)
# optional, +2 devx / +7 test queries in 70 min:
# !python phase7_hc.py --sources train_high_courts,train_tribunals --minutes 150
# 3. rules + names -> submissions
!python phase7_names.py match       # rule N, cut chosen on devx (0.8, precision 0.905)
!python phase7_citator.py all       # rules A/B/D/L(+N) over ot2_zst3_p -> SUBMIT_ORDER_CIT.md
```

The final zip is `cit_abdl_ot2_zst3_p`. Since 8 Nov, ties among rule matches outside the
base list are broken by candidate ID. That ordering scores **0.5151**. The banked 0.5157
came from one luckier ordering among those ties, so a re-run reproduces 0.5151, not 0.5157.
The banked zip itself is `submissions/00_FINAL_best_0.5157/`.

Experiment switches added on 8 Nov. Every one of them was measured, and none beat 0.5157:
`phase7_names.py match --thr X` (forced name cut), `phase7_names.py body` (rule M),
`phase7_citator.py all --variants … --suffix …`, `--bases <run>`,
`--group-cfg/--group-min-prec/--group-tier` (citator groups, `phase7_group.py`). The results
are in EXPERIMENTS.md, "What was tried and rejected". Smoke tests for all of it are in
`code/smoke_restart/` and run locally against synthetic data, without the gated files.

**Integrity.** IL-PCR's public test labels were never downloaded. Every rule is chosen on
train/devx labels only, and anything built on the Phase-1 corpus is reported as
`kind=lookup` beside the model-only number.

---

Four parts, each one blocking Colab cell that releases the runtime when it finishes.
Every step is skip-if-done on its output file, so re-running the same cell after a Colab
drop continues where it stopped.

| part | driver | needs | time | what it does |
|---|---|---|---|---|
| 1 | `p7.sh stage1` | L4 | 50–80 min | gate, BM25/TF-IDF sweeps, zero-shot Phase-1 embeddings, fusion, zips |
| B | `p7b.sh all` | CPU | ~25 min | wider lexical configs, 8 stacker variants, half-split report |
| C | `p7c.sh all` | L4 | ~2 h | the 7 adaptation techniques, compulsory LoRMA + KD, efficiency table, trained-member stacks |
| D | `p7d.sh all` | L4 | ~2.5 h | deep oracle, MaxP, RM3, the two OOM refits, the cross-encoder, new stacks |
| E | `p7e.sh all` | CPU | ~45 min | citation-context retrieval, graph propagation, lambdarank stacks, train-side bar |
| F | `p7d.sh ce` | L4 | ~1 h 25 | the cross-encoder, re-run after the `--topn` fix, over MaxP's top 100 |
| G | `p7f.sh all` | CPU | ~50 min | MaxP swept properly (width + overlap), mask windows x passages, gated upload order |
| H | `p7g.sh all` | CPU | ~45 min | exact rare n-gram pointers, hubness/capacity decoding, new stacks |
| I | `p7i.sh all` | CPU | ~3-4 h | **everything never run**, overlap gate first, then rank-only fusion and the pool-growth curve |

Parts 1/B/C/D have all run. Part E has not. Part D's cross-encoder stage failed on a
driver/parser mismatch and is re-run in part F below.

---

## Cells common to every part

### Setup (run first in any new runtime)

```python
from google.colab import drive
drive.mount('/content/drive', force_remount=True)       # never rmtree the drive
import os
os.chdir('/content/drive/MyDrive/cs781_capstone/phase7/code')
!sed -i 's/\r$//' *.py *.sh                              # strip CRLF after the upload
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
print(sorted(os.listdir('.')))
```

### Gated-data auth — needed on EVERY fresh runtime, before any driver

Colab's secret store is readable only from the notebook kernel, so the drivers' `python`
subprocesses see no token unless it is exported and the six files are already cached.
This cell does both, reading the token from your own Colab secret.

```python
from google.colab import userdata
from huggingface_hub import login, hf_hub_download
import os
os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')
login(os.environ['HF_TOKEN'], add_to_git_credential=False)
for s in ('Train/train', 'Train/train_candidates', 'Dev/dev', 'Dev/dev_candidates',
          'Test/test', 'Test/test_candidates'):
    p = hf_hub_download('Exploration-Lab/CS-781-Capstone',
                        f'Phase_7_PCR/{s}.json', repo_type='dataset')
    print(f'{os.path.getsize(p) / 2**20:7.1f} MB  {s}')
```

**If a run dies with `GatedRepoError: 401`, this cell was not run in that runtime.**
`runtime.unassign()` at the end of a run cell releases the runtime, so the next cell you
execute starts a fresh one with no token — re-run setup and this cell first.

### Checksums — must print `CODE OK` before any run cell

```python
import hashlib, os
want = {
        'p7.sh':                 'c7e9a8e61c13fb0df91d23aa49739bba',
        'p7b.sh':                'ff8081b9befa187c7bea7a9a15e4ecca',
        'p7c.sh':                '509cacbcc8d7ee08b3baeca96d393091',
        'p7d.sh':                '616de7cd51947f47242539a8f7e9f511',
        'p7e.sh':                '5c0cf118c7fa1cfdd374376c68494ba3',
        'p7f.sh':                '0257b4de1612161242bb8eaaf85d42f4',
        'phase7_cite.py':        'f22cd96f109984963c1f1aa059b1ffc8',
        'phase7_data.py':        'ce2ec49a576e0a8112ae7eca5cfc1d06',
        'phase7_emb.py':         'f1ea9369abeb0b475946e6b58b78dcb4',
        'phase7_enc.py':         '7d5193d93ef611da1241bac3e4e03bb3',
        'phase7_overlap.py':     'b655cac8899b534f90674c73d2a6d83a',
        'phase7_scale.py':       'f285c1cd162ecb9a4d9e230fbe88f4df',
        'p7i.sh':                '8b3026ebd60ed517c43acd80c2fc7013',
        'p7g.sh':                '9dc60de0cd9a238e3bf8721294fed338',
        'phase7_point.py':       '06ff68ab034b50188ad89dc053b418e9',
        'phase7_rank.py':        'd38c0a3694fa3c175bc2f8961ed6e0fa',
        'phase7_rerank.py':      '232a8cc235da13daa33241b44e11d5aa',
        'phase7_stack.py':       '1b7dcab28b4a0237a7f975a16b9e5768',
}
bad = [f for f, h in want.items()
       if not os.path.exists(f) or hashlib.md5(open(f, 'rb').read()).hexdigest() != h]
print('CODE', 'OK' if not bad else f'STOP, wrong or missing: {bad}')
```

A file listed as wrong means the CRLF strip did not run or that upload is stale.
Re-upload it, re-run setup, re-run this cell. Do not start a run cell on a `STOP`.

---

## Part 1 — `p7.sh stage1` (L4, done)

```python
!bash p7.sh stage1
from google.colab import runtime; runtime.unassign()
```

Last line of `logs/queue.txt` is `======== STAGE1 DONE`. Refuses the dense systems on a
T4 (bf16 is emulated on SM75 and runs ~30× slow) and says so; the lexical systems still
finish.

Knobs: `P7_CHUNK` (1024) tokens per chunk, `P7_MAXCH` (4) chunks per document,
`P7_EMB_BATCH` (16) chunks per forward, `P7_TOPN` (100) retrieved depth saved,
`P7_PROCS` tokeniser processes, `P7_NO_PIP=1` to skip pip.
Example: `!P7_MAXCH=2 P7_EMB_BATCH=8 bash p7.sh stage1`

## Part B — `p7b.sh all` (CPU, done)

```python
!bash p7b.sh all
```

## Part C — `p7c.sh all` (L4, done)

```python
!bash p7c.sh all
from google.colab import runtime; runtime.unassign()
```

## Part D — `p7d.sh all` (L4, done except the cross-encoder)

```python
!bash p7d.sh all
from google.colab import runtime; runtime.unassign()
```

Stages: `cpu` (deep oracle, MaxP width swept over 300/500/800/1200 × max/mean, RM3 over
λ × feedback docs × feedback terms) → `refit` (`ft_full_m2`, `ft_lora_m2` at batch 2) →
`ce` (the cross-encoder) → `stack` → `zips`. Last line of `logs/queued.txt` is
`======== PART D DONE`.

Knobs: `P7_CE_BASE` (first stage), `P7_CE_DEPTH` (candidates reranked),
`P7_CE_STEPS`, `P7_CE_FOLDS`.

## Part E — `p7e.sh all` (CPU, NOT YET RUN)

Touches no GPU, so it can run on a second free CPU runtime at the same time as a GPU
part. The two write disjoint files. **Only one notebook may run `zips`/`order`**, since
both regenerate `SUBMIT_ORDER.md`.

```python
!bash p7e.sh all
from google.colab import runtime; runtime.unassign()
```

Stages: `masks` (the premise check — do mask counts predict gold counts?) → `cite`
(citation-context retrieval, swept on the 827 train queries) → `prop` (score propagation
over the lex and cocite graphs) → `stack` (including two lambdarank variants) → `half`
(devx halves **and** the 827-query train table) → `zips`.

## Part G — `p7f.sh all` (CPU, NOT YET RUN)

Pushes on the one single-system change that cleared the bar, and stops uploading systems
that cannot win.

```python
!bash p7f.sh all
from google.colab import runtime; runtime.unassign()
```

Stages: `maxp2` (MaxP swept over widths 80/120/180/250/350 x stride 1.0/0.5 x max/mean —
the first sweep chose 300 out of {300,500,800,1200}, the boundary) → `citep`
(mask-anchored query windows against the chosen candidate passages: both sides of the
retrieval unit at once) → `stack` → `half` → `zips` → `order`.

`order` writes a **gated** `SUBMIT_ORDER.md`: an "upload these" table for everything
within `P7_UPLOAD_MARGIN` (default 0.02) of the best devx score, and a "do NOT upload"
table for the rest. test = devx - 0.0095 with sd 0.015, so a system more than ~2 sd below
the best cannot plausibly top the leaderboard. Change the margin with
`!P7_UPLOAD_MARGIN=0.03 bash p7f.sh order`.

## Part I — `p7i.sh all` (CPU, the unattended catch-up)

One blocking cell. Runs every unrun idea, ordered by what could actually reach 0.54.

```python
!bash p7i.sh all
from google.colab import runtime; runtime.unassign()
```

**Why this order.** 0.54 on test is **486 of the 556 reachable devx hits** — 87% of every
gold in the pool, and **98% of a perfect rerank of the top 100** (measured oracle@100 =
495). Ranking cannot reach it: three days of ranking work moved 299 → 387 hits. So:

1. **`overlap` — the gate.** Recovering a masked citation from the permitted Phase-1
   corpus is the only mechanism of the right magnitude. `scan` finds the test and devx
   queries in the corpus; `value` prices the resulting lookup system on devx with the
   official metric. **Measure-only** — nothing is built on a match without an explicit
   go-ahead, and any such system is reported as `kind="lookup"` beside the model-only
   number. This had never been run; it 401'd once and then sat.
2. **`rrf` — the fix for the transfer failure.** devx has 1,023 candidates, test 1,727,
   train 4,320. `citep` won devx by 26 hits and lost test by 0.0197; four stacks inside a
   9-hit devx band came back in exactly reversed test order. Reciprocal-rank fusion reads
   only ranks, so pool size and score scale cannot touch it, and weights are chosen
   greedily on the 827 train queries. The learned stacker is fitted on train's pool and
   applied to test's; this is not.
3. **`curve` — the measurement that should have come first.** The devx pool is grown with
   real train judgments as true negatives, 1,023 → 2,423 (704 extra lands exactly on
   test's 1,727), and each system is re-indexed and re-scored at each size. Output is a
   per-system decay slope in hits per 1,000 extra candidates. The flattest slope is the
   system whose devx number can be trusted.
4. `pointers`, `hub`, `prop`, `csum` — part H, part E, and the `sum` aggregation the
   stopped `citep` sweep never reached. All built, none ever run.
5. `combine` — a stack and a rank-only fusion over everything, the half-split table, the
   zips, and the gated `SUBMIT_ORDER.md`.

Every stage is `|| true` and skip-if-done, so one failure cannot cost the rest, and
re-running the cell after a Colab drop continues where it stopped.

Read in this order: the `i_value` log and `results/overlap/*.json`, then
`results/pool_curve.json`, then the half-split table.

## Part H — `p7g.sh all` (CPU, NOT YET RUN)

The only two ideas left that can change the order of magnitude rather than the margin.
CPU only, so it runs beside a GPU part.

```python
!bash p7g.sh all
from google.colab import runtime; runtime.unassign()
```

Stages: `where` (where the 556 reachable hits sit) → **`residue`** → `ngram` → `hub` →
`stack` → `restack` → `half` → `zips` → `status`.

**`residue` is the gate, and it is the first thing to read.** 0.53 on test is 477 of the
556 reachable hits, which is 100.6% of a *perfect* rerank of the top 50 and 96.3% of the
top 100 — so no reordering of the current evidence reaches it. Only new evidence can, and
inside the brief's constraints there are two sources:

1. **Word order.** Every system in the phase is a bag of words. An exact 8-word match
   between a citing sentence and a candidate judgment is not a similarity, it is a
   pointer, because judgments quote the holdings they cite. `residue` measures whether
   those quotations survived IL-PCR's masking:
   - enrichment near 1 → the masking stripped them, and `ngram` is dead. Say so and stop.
   - enrichment in the tens or hundreds → it is the best lever in the phase. If even 40%
     of the ~11 masks per query resolve this way, that alone is ~4 hits/query = 472 devx
     hits = devx 0.52, which is the entire gap.
2. **The joint structure across queries.** Every system ranks each query alone, so a few
   long generic judgments take slots everywhere; gold structure cannot look like that.
   `hub` corrects a candidate's score by how it does across queries (`zmean`, `count`) and
   decodes all queries together under a per-candidate capacity (`greedy`, `sinkhorn`).
   Same move as Phase 2's test-volume calibration. It runs over every finished base run
   and over the new stacks, selecting on the 827 train queries.

Knobs: `P7_NSWEEP` (6,8,12,20), `P7_HUB_BASES`, `P7_SELECT` (train), `P7_TOPN`.

Read `results/ngram_residue.json`, `results/where_hits.json` and `results/hub_*.json`.
A `hub` sweep point also reports `backfilled`: the number of top-7 slots the capacity
could not supply. A point with a large back-fill is mostly the base ranking, not a
capacity result.

## Part F — the cross-encoder (L4): the fit ran, then `write` threw it away

The first `ce` run on 7 Nov trained all four models (26 min) and scored devx and test in
full (17 min), then died at the last step:

```
score: 11 gold queries have no prediction (first ['0000081544', ...])
```

Two bugs, both now fixed, and the second is the expensive one:

1. The cross-fitted **train** ranking cannot cover every gold query. `TrainSet` drops the
   contaminated query, every gold-less one, and any query whose golds fall outside the
   tokenised pool, so those 11 belong to no fold and are never scored. The official
   scorer is right to refuse a partial prediction; `evaluate_covered` now scores the
   covered subset and prints the coverage, because an F1 over 816 of 827 queries has its
   own denominator.
2. `write` **evaluated train before saving devx and test**, so the raise discarded 17
   minutes of finished forward passes. It now saves every split first and evaluates
   afterwards: nothing that can fail belongs between the GPU work and the file it writes.

**The checkpoint and the train ranking survived**, so the recovery costs ~17 min, not
~82. Re-upload `phase7_rerank.py` and `p7d.sh`, then:

```python
!bash p7d.sh cescore
```

Check first that the pieces are really there — all three should exist:

```python
!ls -la /content/drive/MyDrive/cs781_capstone/phase7/models/ce_lorma*.pt && ls /content/drive/MyDrive/cs781_capstone/phase7/results/ce_lorma/
```

Then re-stack and zip, which is where `ce_lorma` and `ce_lorma_mix` join:

```python
!bash p7d.sh stack && bash p7d.sh zips
```

Only if the checkpoint is missing does the full `ce` stage need re-running:

```python
!P7_CE_BASE=maxp_tfidf P7_CE_DEPTH=100 bash p7d.sh ce
```

## The corpus-overlap scan (measure-only, CPU, needs the auth cell first)

```python
!python phase7_overlap.py scan --splits test,devx --sources train_supreme_court
```

Nothing is built on a match without an explicit go-ahead, and any such system is reported
as `kind="lookup"` beside the model-only number.

---

## What to send back after any part

```python
P = '/content/drive/MyDrive/cs781_capstone/phase7'
import glob, os
print(open(sorted(glob.glob(f'{P}/logs/queue*.txt'), key=os.path.getmtime)[-1]).read()[-9000:])
for f in ('results/kd_efficiency.json', 'results/oracle_depth.json',
          'results/mask_stats.json', 'results/half_report.json'):
    if os.path.exists(f'{P}/{f}'):
        print(f'\n==== {f}\n' + open(f'{P}/{f}').read())
```

## Then, locally, before uploading

```bash
python capstone_Project/tools/cs781_submit.py status --phase 7
```

Locally, every uploaded zip now lives in `phase7/submissions/<family>/`, named
`<test score>__<tag>__<run>.zip` and listed in `submissions/INDEX.md`. Downloads holds none.

Upload order is `phase7/SUBMIT_ORDER.md`, best devx F1@7 first. Right-click the single
`submission.zip` inside `submissions/<tag>/` — downloading the folder gives a wrapper zip
that Codabench rejects. 40/day and 160/phase; identical resubmissions still count. The
leaderboard shows the **pushed** submission, not the best one.
