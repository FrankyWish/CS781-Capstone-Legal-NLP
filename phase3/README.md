# Phase 3: rhetorical role prediction

**Task.** The task was to assign one of 13 rhetorical roles to every sentence in a legal judgment. The labels are Fact, Issue, ArgumentPetitioner, ArgumentRespondent, Statute, PrecedentReliedUpon, PrecedentNotReliedUpon, PrecedentOverruled, RulingByLowerCourt, RatioOfTheDecision, RulingByPresentCourt, Dissent and None.

There were 90 labelled documents for training and development, containing 18,987 sentences. The test set had 10 documents (5 Competition Law and 5 Income Tax). The brief mentioned Corporate Law / IT, but CL actually stands for Competition Law. The evaluation metric was Macro-F1 over all 13 labels.

**Results (test).**

| **system** | **uses** | **test** |
|---|---|---:|
| `lora_M4_lsegall`: M4 after LegalSeg intermediate training, LoRA r16 | Phase-1 model only | 0.5637 |
| `v54`: 8-member ensemble including a TF-IDF model and transition decoding | genuine hybrid | 0.5845 |
| `v46`: 3-member ensemble | genuine hybrid | 0.5879 |
| `v68`: LoRA M4 (LegalSeg) + TF-IDF, trained with 5 additional IL-TUR `rr` documents | additional labelled data | **0.6707**, rank 4 (graded entry) |

The best result without the additional documents was 0.5879 using `v46`. The best model-only result was 0.5637 using `lora_M4_lsegall`.

The data comes from IL-TUR rr (Malik et al., 2022). Runs that add the 5 extra IL-TUR `rr` documents to training are kept separate from the main results; the graded entry `v68` is one of them, and none of the other rows uses them.

## Architecture

The model uses a hierarchical architecture. First, a Phase-1 model encodes packed sentence chunks of up to 512 tokens. Each sentence is then mean-pooled into a sentence representation.

After that, a 2-layer transformer processes the sequence of sentence representations along with position and domain features. A linear head finally predicts the rhetorical role for each sentence.

Training uses windows of 64 sentences.

For decoding, I added a prior correction (`tau`) and a learned label-transition matrix (`beta`). Viterbi decoding is then applied over the complete document. This helps account for the fact that some rhetorical roles are more likely to appear after certain other roles.

For external data with coarse labels, I used allowed-set training. The loss is calculated as the negative log of the summed probabilities of all allowed roles. This avoids forcing an uncertain external label into one default class.

### Evaluation

I used 5-fold document-level cross-validation over the 90 documents, with the folds stratified by domain.

Each fold model was early-stopped using its own validation fold. For the final submission, the test logits from all five fold models were averaged. This also made it possible to check how each model performed before combining them.

## What worked and what did not

**Adapters performed better than full fine-tuning.**

LoRA on M4 scored 0.5223 test, rsLoRA on M1 scored 0.5193 and LoRMA on M3 scored 0.5045. Full fine-tuning of M4, even with three times the data, only reached 0.4912.

It looks like full fine-tuning was memorising the 90 documents. The training loss went down to 0.05, but the validation F1 stopped improving after around epoch 7.

**LegalSeg intermediate training helped the most.**

I tried training on LegalSeg first, using 2,000 and then 7,117 judgments with 7 coarse roles. This was the most useful improvement for the model-only results. LoRA M4 improved from 0.5223 to 0.5636 test.

**Knowledge distillation (KD).**

An 8-layer student model based on M4 reached 0.5140. This was better than every full-size single model except the LoRA-based runs.

**Trunk comparison.**

The model order was M4 > M1 ~ M3 > M2. This was similar to what happened in Phase 2, but different from the ranking based on Phase-1 likelihood.

**Decoding improvements.**

I tested different decoding settings and selected them based on the leaderboard. The final settings were `tau 0.25` and `beta 0.25 to 0.5`, using global settings only. These changes were worth around +0.02, although this was based on only 10 test documents.

**Things that did not work:**

- Stacking scored 0.4478 OOF compared to 0.4644 for the normal mean ensemble. It also performed very poorly on ArgumentRespondent in the test set.
- External MARRO / Kalamkar data improved CV by only +0.017 and did not give a reliable test improvement.
- The label-shift auxiliary head lost on test in all three attempts.
- In-domain continued pretraining did not help.
- Greedy CV member selection did not work well.
- IA³ and QLoRA performed poorly, with scores of 0.4696 and 0.4442 respectively.

### Data limitations

There were 496 sentences repeated within the train/dev data, and their labels agreed only 77.1% of the time. This makes the task difficult for models that only classify individual sentences.

PrecedentOverruled had only 3 sentences across the 90 documents. Dissent was also absent from the test set. Because of this, both labels ended up with a score of zero for all systems.

## How to run

```bash
python phase3_data.py --folds 5 --seed 781
# gate + split.json

python phase3_legalseg.py --max-docs 7117
# LegalSeg -> allowed-set jsonl

python phase3_rr.py --model M4_LexArcP --method lora --rank 16 --tag lora_M4

# Intermediate stage: one pass over external/legalseg.jsonl
# (run tag inter_M4)
# Then the gold run starts from that trunk:

python phase3_rr.py --model M4_LexArcP --method lora --rank 16 \
    --trunk-path <inter_M4 checkpoint> --tag lora_M4_lsegall

python phase3_rr.py --model M1_SC --method rslora --rank 64 --tag rslora_M1

python phase3_rr.py --model M3_LexArcL --method lorma --rank 16 --tag lorma_M3

python phase3_lex.py
# TF-IDF member

python phase3_ens.py ens --runs ... --dump-teacher results/ens_teacher.pt

python phase3_rr.py --model M4_LexArcP --method full --keep-every 2 \
    --kd-logits results/ens_teacher.pt --kd-test --tag kd_M4half

python phase3_variants.py
# ensembles + decoding grid -> submissions/
```

`run_day1.sh`, `run_day1b.sh`, `run_extkd.sh` and `run_push.sh` contain the experiment run sequences used to produce the different results.

Every run saves its out-of-fold logits, test logits, CRF parameters and per-epoch training trace under `results/<tag>/`. Once the runs are completed, all the ensembling can be done on CPU.

## Files

| **file** | **role** |
|---|---|
| `phase3_data.py` | data checks, CV split, validator and scoring |
| `phase3_rr.py` | hierarchical trainer, 11 adaptation methods, 5-fold CV and KD |
| `phase3_peft.py` | LoRA, rsLoRA, QLoRA, AdaLoRA, LoRMA, IA³, prompt and prefix tuning |
| `phase3_external.py`, `phase3_legalseg.py` | MARRO, Kalamkar and LegalSeg as allowed-set data |
| `phase3_iltur.py` | the 5 additional IL-TUR `rr` documents as training data (used by `v68`) |
| `phase3_lex.py` | TF-IDF + logistic regression member |
| `phase3_ens.py`, `phase3_variants.py` | ensembling, decoding, KD teacher and leaderboard variants |
| `phase3_stack.py`, `phase3_dapt.py`, `phase3_predict.py` | stacking (failed), continued pretraining (failed), teacher labelling |

The full experiment log, including the failed experiments and their results, is available in `EXPERIMENTS.md`.
