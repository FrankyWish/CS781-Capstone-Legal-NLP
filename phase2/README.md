# Phase 2: legal named-entity recognition

**Task.** The task was to identify and tag 12 types of entities (COURT, PETITIONER/APP, RESPONDENT/RESP, JUDGE, A.COUNSEL, R.COUNSEL, STAT, PREC, CASENO, DATE, WIT, AUTH) as exact character spans in full legal judgments. There were 70 training documents (organiser folds 1 and 2, with 8,393 entities) and 35 test documents (fold 3). The evaluation metric was **strict** Macro-F1 over all 12 labels. Even a one-character difference in a predicted span gives a zero score.

**Result.** Got Rank 7 with **0.6515** (P 0.6252, R 0.6835) using `submission_r3_rescue.zip`. The best model-only system scored 0.4873 (`ext6`), which was an ensemble of fine-tuned Phase-1 models with InLegalNER added. The improvement from there came from combining multiple systems, fixing entity boundaries and adding domain-specific rules based on the experiments.

## Architecture

The basic setup uses a Phase-1 model to read 1,024-token windows from a judgment. A linear head then predicts BIO tags for each token.

Character offsets are recovered by mapping each byte of a BPE token back to the character it belongs to. This was needed because byte-level BPE can split Devanagari characters across multiple tokens. I also had to remove the leading spaces that the legal pre-tokenizer attaches to words. Around 98.33% of the gold spans survived the complete character-to-token-to-character conversion.

The training loss is class-weighted cross-entropy, with weight power 0.5. I tested both directions for the weighting as well.

The final system has four main parts:

1. **Members**

   I fine-tuned all four Phase-1 models completely. M4 performed best as a single model with 0.329 CV, even though `M1_SC` had ranked first based on Phase-1 likelihood. Their predictions were combined using average log-probabilities.

   I tested three versions with different training data:
   - Gold training data only (`ens6_m4`).
   - Gold data plus rule-labelled Supreme Court, High Court and tribunal text (`wsens6`).
   - The above data with InLegalNER added (`ext6`, `ext6e`).

2. **Post-processing**

   I added some fixes for entity boundaries:
   - Word-snap expands a predicted span to cover complete words, except for PREC.
   - Longest-form expansion selects the longest matching form of the same label.
   - 26 regex rules were tested. Each rule had to pass a precision check against all 8,393 gold spans. Six rules failed this check and were disabled.

3. **Stacking** (`phase2_stack.py`)

   I collected every span predicted by any system at three different post-processing stages. A gradient-boosted classifier then predicts whether each exact span is likely to be correct.

   It uses features such as agreement between systems, repeated strings and the shape of the entity. The classifier was cross-fitted using held-out CV predictions from the same systems, and the final choice was based on the worse fold.

4. **Precision trim and rescue** (`phase2_spanvote.py`)

   First, a span from the stacker is kept only if another stack also predicts exactly the same span. After this, spans agreed on by two systems are added back using `--base <previous> --k 2`.

   This was repeated for four rounds (`r1` to `r4`). The `r3` version was used for the final graded submission.

## What worked and what did not

| **step** | **test** |
|---|---:|
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

I also tried quite a few things that did not help. These are all recorded in `EXPERIMENTS.md` with their measured results.

Some of the failed experiments were:

- Linear probe (0.056 CV).
- Lower learning rate.
- Weaker and stronger class weights.
- Caption prepending.
- BIOES tagging.
- CRF.
- Fragment merging.
- Cross-document gazetteer (-0.0002 test).
- Occurrence propagation (-0.0058).
- Per-label source arbitration, which lost all four times on test.
- Context features and bagging in the stacker. Both improved CV but performed worse on test.

LoRMA on M3 also scored only 0.211 CV compared to 0.285 for full fine-tuning. With only 70 training documents, the model seemed to need more freedom to change its weights.

I also tried knowledge distillation into a half-depth M4 student. It retained 88% of strict F1 while using 60% of the parameters and getting 1.94x the throughput.

One useful observation during this phase was the difference between strict F1 (0.30) and overlap F1 (0.58) on CV. This showed that the model was often finding the correct entities but getting their boundaries wrong.

Another test was a per-document string-matching oracle using gold entity strings. It reached 0.9863, which suggested that most of the problem was finding each distinct entity and then copying its exact span correctly.

## External data

InLegalNER (Kalamkar et al., 2022) was also used as an additional training source. It has 14 entity types, of which 10 could be mapped to the labels used in this phase. The remaining types were masked from the loss.

I checked every record against the 35 test documents using rare 13-gram containment. Eleven records containing copied test passages were removed.

## How to run

The environment setup is the same as the top-level README, with `CS781_P2` pointing to the Phase-2 folder.

```bash
python phase2_data.py --overlap
# alignment self-test and fold overlap check

python phase2_ner.py --model M4_LexArcP --method full --epochs 8 --submit
# one member

python phase2_ensemble.py --specs M1_SC:full M2_LexArcED:full M3_LexArcL:full M4_LexArcP:full --submit

python phase2_external.py --inspect
# InLegalNER mapping + contamination filter

python phase2_weaksup.py --n-docs 2000 --submit --tag ws2000
# weak-supervision members

python phase2_gazetteer.py --snap ens6_m4
# word-snap + expansion

python phase2_rules.py --validate && python phase2_rules.py --apply ens6_m4_scwg

python phase2_stack.py --cv wsens6 ext6 ens6_m4 \
    --test wsens6_scwg_rpadd ext6_scwg_rpadd ens6_m4_scwg_rpadd \
    --tag stack9 --rates 136

python phase2_spanvote.py --tags s9x_r136 stack12_r136 stack27_r136 \
    --base s9x_r136 --k 2 --out s9trim

python phase2_spanvote.py --tags <stack> <student> ... \
    --base r1_rescue --k 2 --out r3_rescue
```

Every script shows its available options using `--help`. GPU is only needed for training the individual models. Everything after member training can be run on CPU.

| **file** | **role** |
|---|---|
| `phase2_data.py` | loading, offset alignment and its self-test, strict and overlap scoring |
| `phase2_ner.py` | trainer: full / head / LoRA / LoRMA / BitFit, BIO or BIOES, CV and refit |
| `phase2_ensemble.py` | log-probability ensembling across trunks |
| `phase2_distil.py` | knowledge distillation into a half-depth student |
| `phase2_external.py`, `phase2_weaksup.py`, `phase2_pseudo.py` | external, weak and pseudo labels |
| `phase2_gazetteer.py`, `phase2_rules.py` | boundary snapping and gated domain rules |
| `phase2_decode.py` | offline decoding of saved emissions |
| `phase2_stack.py`, `phase2_spanvote.py` | span stacker, exact-span voting, trim and rescue |
| `phase2_tokcheck.py`, `phase2_inspect.py` | tokenizer hypothesis test and data viewer |

The complete experiment log is available in [`EXPERIMENTS.md`](https://github.com/FrankyWish/CS781-Capstone-Legal-NLP/blob/main/phase2/EXPERIMENTS.md).

All submissions are listed in [`submissions/README.md`](https://github.com/FrankyWish/CS781-Capstone-Legal-NLP/blob/main/phase2/submissions/README.md).
