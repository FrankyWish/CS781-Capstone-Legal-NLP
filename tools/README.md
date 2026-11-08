# CS-781 Capstone — tooling

Three standard-library scripts. None of them needs a GPU except
`cs781_submit.py eval`, which shells out to the official evaluators.

| Script | Does |
|---|---|
| `cs781_submit.py` | Validate, package and track Codabench submissions |
| `make_dev_split.py` | Carve a held-out Phase-1 dev split from the training corpus |
| `param_budget.py` | Count parameters against the 50M–300M limit before training |

## make_dev_split.py

Phase 1 ships no dev set, so model selection has to run against a split you
build. This writes `dev_prompts.json` and `dev_truth.json` in exactly the
shapes the official test input and `evaluate_phase1.py` expect, plus a
manifest of held-out records so training can exclude them.

```bash
python make_dev_split.py \
    --train Train/train_acts.json Train/train_high_courts.json \
            Train/train_supreme_court.json Train/train_tribunals.json \
    --out-dir dev_split --n 200 --per-source
```

`--per-source` draws equally from each corpus so the dev set is balanced
across Acts/HC/SC/Tribunals. Windows are cut from a random offset rather than
the document start, because these documents open with near-identical
boilerplate headers and a start-anchored dev set is much easier than the real
test set.

Excluding the held-out records from training is not optional — otherwise the
local BERTScore measures memorization. The manifest keys records by
`(source_file, record_index)` and also stores a content hash, so the holdout
survives the files being reordered.

## param_budget.py

```bash
python param_budget.py --sweep --ffn-type swiglu --norm rms --pos rope --n-kv-head 4
python param_budget.py --d-model 768 --n-layer 12 --n-head 12 --n-kv-head 4 \
    --ffn-mult 2.67 --ffn-type swiglu --norm rms --pos rope \
    --tokens 500e6 --throughput 10000
```

Breaks the count down by component, flags the 50M–300M bounds, reports the KV
cache footprint with and without GQA, and converts a token budget into
GPU-hours at a throughput you measured. `--target-params` solves for the FFN
multiplier that hits a given size.

## cs781_submit.py

Sits between "my model produced a prediction.json" and
"I uploaded submission.zip to Codabench". It validates the file against the
rules the platform scorer enforces, packages the archive with the exact layout
Codabench expects, and keeps a ledger of every submission so the phase
minimums, the daily cap and the no-identical-resubmissions rule are tracked
in one place instead of in your head.

Standard library only. The same file runs on Windows, Kaggle and Colab.

## Why bother

Codabench takes 15–30 minutes per submission and the phase windows are 4–5
days. A rejected upload over a missing test ID or a one-character offset bug
costs half an hour you do not have, and the brief says delay excuses are not
entertained. Every check here is something that would otherwise cost a round
trip, or silently score zero.

## Workflow

```bash
# 1. sanity-check the file your notebook just wrote
python cs781_submit.py validate --phase 1 \
    --prediction prediction.json \
    --test /path/to/Phase_1_Pretraining/Test/test.json

# 2. score it on your own held-out split
python cs781_submit.py eval --phase 1 \
    --prediction dev_prediction.json --truth dev_truth.json

# 3. package and log it
python cs781_submit.py pack --phase 1 \
    --prediction prediction.json \
    --test /path/to/test.json \
    --method "rotary-decoder-only" \
    --base-model "arch-A-560d-8L" \
    --notes "top-p 0.9, repetition penalty 1.1" \
    --score 0.6412

# 4. upload the printed submission.zip, then record what came back
python cs781_submit.py score --seq 3 --leaderboard 0.6388

# 5. check where you stand
python cs781_submit.py status
python cs781_submit.py report --phase 1   # markdown table for the report
```

## Per-phase arguments

Validation is much stronger when it can see the official test file, because
ID mismatches are the most common cause of a rejected submission.

| Phase | Required | Strongly recommended |
|-------|----------|----------------------|
| 1 | — | `--test Test/test.json` |
| 2 | `--labels labels.txt` | `--test Test/fold3-text.json` |
| 3 | `--labels label_vocab.json` | `--test-cl CL_test.json --test-it IT_test.json` |
| 4 | — | `--test Test/test.json` |
| 5 | — | `--test Test/test.json --max-label <max ID in train+dev>` |
| 6 | — | `--test Test/test.json` |
| 7 | `--candidates Test/test_candidates.json` | `--test Test/test.json` |

Phase 2 passes the document text to the validator, so if you include the
optional fourth `entity_text` element in each span it will tell you when your
character offsets are shifted — the single easiest way to lose an entire
phase to strict span matching.

Phase 3 passes sentence counts, so a document where
`len(predictions) != len(document["text"])` is caught before upload rather
than by the scorer.

## What `validate` checks

Everything the corresponding `evaluate_phaseN.py` checks, plus:

- Strict JSON loading — duplicate object keys and `NaN`/`Infinity` are
  rejected exactly as the scorer rejects them.
- Missing and unknown test IDs against the official test file.
- Phase 1: continuation length against the 128-word target, prompt echoed
  into the output, degenerate repetition.
- Phase 2: offsets past the end of the document, `entity_text` disagreeing
  with `document[start:end]`, labels never predicted at all.
- Phase 3: sentence-count mismatch, `.txt` ID normalization, unused labels.
- Phase 4: single-class predictions (Macro-F1 caps near 0.33).
- Phase 5: statute IDs above the reference label space, empty predictions.
- Phase 6: very short or repetitive summaries.
- Phase 7: a candidate returned for most queries, which usually means the
  ranker has collapsed.

Errors block packaging. Warnings do not — they are things worth a look before
you spend one of your 40 submissions.

## What the ledger tracks

`submissions/ledger.json`, one entry per packaged archive:

- **Duplicate detection.** The content is hashed after parsing and sorting,
  so re-dumping the same predictions with different indentation still counts
  as a duplicate. Identical resubmissions do not count toward the phase
  minimum, and `pack` refuses them rather than letting you burn a slot.
- **Caps.** 40 per phase, 10 per day; the daily count is checked in both UTC
  and local time and the stricter one wins.
- **Minimums.** 4 for phase 1, 6 for phases 2–7, counting only
  non-duplicate submissions.
- **Method tags.** Phases 2–7 require six methodologically distinct runs,
  including LORMA and knowledge distillation. `status` flags either one if no
  submission is tagged for it, and warns when fewer than four distinct
  Phase-1 base models have been used.
- **Scores.** Local and leaderboard, so `report` emits the experiment table
  the final report needs without reconstructing it from memory in November.

`--force` overrides any refusal. `--dry-run` runs every check and writes
nothing.

## Setup

```bash
pip install -r requirements.txt   # only needed for the `eval` subcommand
```

Keep the ledger on persistent storage. On Kaggle, point `--ledger` at a
dataset or Drive path, or export it before the session ends — `/kaggle/working`
does not survive:

```bash
export CS781_LEDGER=/content/drive/MyDrive/cs781/submissions
```
