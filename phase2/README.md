# Phase 2: legal named-entity recognition

**Task.** Tag 12 entity types (COURT, PETITIONER/APP, RESPONDENT/RESP, JUDGE, A.COUNSEL,
R.COUNSEL, STAT, PREC, CASENO, DATE, WIT, AUTH) as exact character spans in full judgments.
70 training documents (organiser folds 1 and 2, 8,393 entities), 35 test documents (fold 3).
Metric: **strict** Macro-F1 over the 12 labels; a span one character off scores zero.

**Result.** Rank 6 at **0.6515** (P 0.6252, R 0.6835), `submission_r3_rescue.zip`. Best
model-only system: 0.4873 (`ext6`, an ensemble of fine-tuned Phase-1 trunks with InLegalNER
added). Everything between those two numbers is combination of systems, boundary
post-processing and measured domain rules.

## Architecture

A Phase-1 trunk reads 1,024-token windows of the judgment and a linear head emits BIO tags per
token. Character offsets are recovered by mapping each byte of each BPE token back to its
enclosing character (byte-level BPE splits Devanagari characters across tokens) and trimming
the leading space that the legal pre-tokenizer attaches to words. 98.33% of gold spans survive
the char-to-token-to-char round trip. Loss is class-weighted cross-entropy (weight power 0.5,
tested in both directions).

The graded system has four layers:

1. **Members.** Full fine-tunes of all four trunks (M4 was the best single model at 0.329 CV,
   above `M1_SC`, which Phase-1 likelihood had ranked first), averaged as log-probabilities,
   in three training-data variants: gold only (`ens6_m4`), plus rule-labelled Supreme Court,
   High Court and tribunal text (`wsens6`), plus InLegalNER (`ext6`, `ext6e`).
2. **Post-processing.** Word-snap (widen a span to whole words, except PREC), longest-form
   expansion of the same label, and 26 regex rules, each gated on its exact precision against
   all 8,393 gold spans (6 failed the gate and are off).
3. **Stacking** (`phase2_stack.py`). Every span any system emits, at three post-processing
   stages, is a candidate. A gradient-boosted classifier predicts whether the exact span is
   gold from agreement counts, string repetition and shape features. It is cross-fitted on the
   held-out CV predictions of the same systems, selected by the worse fold.
4. **Precision trim and rescue** (`phase2_spanvote.py`). Keep a stack span only if another
   stack also emits it exactly; then add back spans that two systems agree on exactly
   (`--base <previous> --k 2`), repeated for four rounds (`r1` to `r4`; `r3` was graded).

## What worked and what did not

| step | test |
|---|---|
| M1_SC fine-tune, bidirectional | 0.3854 |
| ensemble of 4 / 6 / 6 with M4 tripled | 0.4435 / 0.4522 / 0.4608 |
| + word-snap + longest-wins expansion | 0.5200 |
| + domain rules (snap and add) | 0.5334 |
| weak-supervision ensemble + full post-processing | 0.5577 |
| exact 2-of-3 span vote | 0.5763 |
| stacker over 3 systems x 3 stages (`stack9`) | 0.6421 |
| + output volume 136 spans/doc | 0.6436 |
| + precision trim (`s9trim`) | 0.6444 |
| + rescue rounds (`r3_rescue`) | **0.6515** |

Interventions that failed, each with a measured number in `EXPERIMENTS.md`: linear probe
(0.056 CV), lower learning rate, weaker and stronger class weights, caption prepending, BIOES,
CRF, fragment merging, a cross-document gazetteer (-0.0002 test), occurrence propagation
(-0.0058), per-label source arbitration (lost four times out of four on test), context
features and bagging in the stacker (both rose on CV and fell on test). LoRMA on M3 scored
0.211 CV against 0.285 for full fine-tuning: with 70 documents the trunk has to move.
Knowledge distillation into a half-depth M4 student kept 88% of strict F1 at 60% of the
parameters and 1.94x the throughput.

The diagnostic that redirected the phase: strict F1 0.30 against overlap F1 0.58 on CV, i.e.
entities were found and mis-bounded. A per-document string-matching oracle on gold strings
scores 0.9863, so the task is mostly "find each distinct entity once, then copy it".

## External data

The professor allowed external data on 2026-10-10. InLegalNER (Kalamkar et al., 2022) maps
10 of its 14 types onto ours; the rest are masked from the loss. Every record was checked
against the 35 test documents with rare 13-gram containment and 11 records that copy test
passages were dropped.

## How to run

Environment as in the top-level README, with `CS781_P2` pointing at the phase folder.

```bash
python phase2_data.py --overlap                                   # gate: alignment self-test, fold overlap
python phase2_ner.py --model M4_LexArcP --method full --epochs 8 --submit      # one member
python phase2_ensemble.py --specs M1_SC:full M2_LexArcED:full M3_LexArcL:full M4_LexArcP:full --submit
python phase2_external.py --inspect                               # InLegalNER mapping + contamination filter
python phase2_weaksup.py --n-docs 2000 --submit --tag ws2000      # weak-supervision members
python phase2_gazetteer.py --snap ens6_m4                         # word-snap + expansion
python phase2_rules.py --validate && python phase2_rules.py --apply ens6_m4_scwg
python phase2_stack.py --cv wsens6 ext6 ens6_m4 \
    --test wsens6_scwg_rpadd ext6_scwg_rpadd ens6_m4_scwg_rpadd --tag stack9 --rates 136
python phase2_spanvote.py --tags s9x_r136 stack12_r136 stack27_r136 --base s9x_r136 --k 2 --out s9trim
python phase2_spanvote.py --tags <stack> <student> ... --base r1_rescue --k 2 --out r3_rescue
```

Every script prints its own options with `--help`. CPU is enough for everything after the
member training.

| file | role |
|---|---|
| `phase2_data.py` | loading, offset alignment and its self-test, strict and overlap scoring |
| `phase2_ner.py` | trainer: full / head / LoRA / LoRMA / BitFit, BIO or BIOES, CV and refit |
| `phase2_ensemble.py` | log-probability ensembling across trunks |
| `phase2_distil.py` | knowledge distillation into a half-depth student |
| `phase2_external.py`, `phase2_weaksup.py`, `phase2_pseudo.py` | external, weak and pseudo labels |
| `phase2_gazetteer.py`, `phase2_rules.py` | boundary snapping and gated domain rules |
| `phase2_decode.py` | offline decoding of saved emissions |
| `phase2_stack.py`, `phase2_spanvote.py` | span stacker, exact-span voting, trim and rescue |
| `phase2_tokcheck.py`, `phase2_inspect.py` | the tokenizer hypothesis test, data viewer |

Full log: [`EXPERIMENTS.md`](EXPERIMENTS.md). Every upload: [`submissions/README.md`](submissions/README.md).
