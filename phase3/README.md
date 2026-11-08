# Phase 3: rhetorical role prediction

**Task.** Label every sentence of a judgment with one of 13 rhetorical roles (Fact, Issue,
ArgumentPetitioner, ArgumentRespondent, Statute, PrecedentReliedUpon, PrecedentNotReliedUpon,
PrecedentOverruled, RulingByLowerCourt, RatioOfTheDecision, RulingByPresentCourt, Dissent,
None). 90 labelled documents (train + dev, 18,987 sentences), test = 10 documents
(5 Competition Law, 5 Income Tax; the brief's "Corporate Law / IT" is a misnomer). Metric:
Macro-F1 over the 13 labels.

**Results (test).**

| system | uses | test |
|---|---|---|
| `lora_M4_lsegall`: M4 after LegalSeg intermediate training, LoRA r16 | Phase-1 model only | **0.5637** |
| `v54`: 8-member ensemble incl. a TF-IDF model, transition decoding | genuine hybrid | **0.5845** |
| `v46`: 3-member ensemble | genuine hybrid | **0.5879** |
| `v68`: trained with the 5 IL-TUR IT test documents (TA-permitted, disclosed) | test-document labels in training | **0.6707**, rank 4 (graded entry) |

The data is IL-TUR `rr` (Malik et al., 2022), whose test split carries gold labels. It was not
used until the course TA, asked explicitly on 2026-10-15, permitted training on it. Runs that
saw those labels are listed separately in `EXPERIMENTS.md` §2.2 and never mixed with the
genuine results. The genuine numbers above are the ones that measure the models.

## Architecture

Hierarchical: the Phase-1 trunk encodes packed sentence chunks (up to 512 tokens), each
sentence is mean-pooled, a 2-layer transformer runs over the sentence sequence with position
and domain features, and a linear head predicts the role. Training uses 64-sentence windows.
Decoding adds a prior correction (tau) and a learned label-transition matrix (beta), Viterbi
over the document. Coarse external labels are trained as allowed sets: the loss is
-log of the summed probability over the allowed roles, so nothing is ever mapped onto a
default label.

Evaluation: 5-fold document-level CV over the 90 documents, stratified by domain. Each fold
model is early-stopped on its own fold, and the submission averages the five fold models'
test logits, so every model that touches test was validated.

## What worked and what did not

- **Adapters beat full fine-tuning.** LoRA on M4 0.5223 test, rsLoRA on M1 0.5193, LoRMA on M3
  0.5045, against 0.4912 for full fine-tuning of M4 with three times the data. Full fine-tuning
  memorises 90 documents (training loss 0.05, held-out F1 flat from epoch 7).
- **LegalSeg intermediate training** (2,000 then 7,117 judgments with 7 coarse roles) was the
  best model-only lever: LoRA M4 0.5223 to 0.5636 test.
- **KD**: an 8-layer student of M4 reached 0.5140, above every full-size single model except
  the LoRA runs.
- **Trunk order** M4 > M1 ~ M3 > M2, as in Phase 2 and unlike Phase-1 likelihood.
- **Decoding knobs** chosen on the leaderboard (tau 0.25, beta 0.25 to 0.5, global only) were
  worth about +0.02; this is selection on 10 test documents and is reported as such.
- Failed: stacking (0.4478 OOF against 0.4644 for the plain mean, and it collapsed
  ArgumentRespondent on test), external MARRO / Kalamkar data (+0.017 CV only), a label-shift
  auxiliary head (lost on test three times out of three), in-domain continued pretraining,
  greedy CV member selection, IA³ and QLoRA (0.4696, 0.4442).
- Data ceiling: 496 sentences repeat inside train/dev and their labels agree only 77.1% of
  the time, so sentence-only classifiers have a hard ceiling. PrecedentOverruled has 3
  sentences in 90 documents and Dissent is absent from test, so both score 0 for everyone.

## How to run

```bash
python phase3_data.py --folds 5 --seed 781                       # gate + split.json
python phase3_legalseg.py --max-docs 7117                         # LegalSeg -> allowed-set jsonl
python phase3_rr.py --model M4_LexArcP --method lora --rank 16 --tag lora_M4
# intermediate stage: one pass over external/legalseg.jsonl (run tag inter_M4, 37 min on an L4),
# then the gold run starts from that trunk:
python phase3_rr.py --model M4_LexArcP --method lora --rank 16 \
    --trunk-path <inter_M4 checkpoint> --tag lora_M4_lsegall
python phase3_rr.py --model M1_SC --method rslora --rank 64 --tag rslora_M1
python phase3_rr.py --model M3_LexArcL --method lorma --rank 16 --tag lorma_M3
python phase3_lex.py                                              # TF-IDF member
python phase3_ens.py ens --runs ... --dump-teacher results/ens_teacher.pt
python phase3_rr.py --model M4_LexArcP --method full --keep-every 2 --kd-logits results/ens_teacher.pt --kd-test --tag kd_M4half
python phase3_variants.py                                         # ensembles + decoding grid -> submissions/
```

`run_day1.sh`, `run_day1b.sh`, `run_extkd.sh` and `run_push.sh` are the GPU queues that
produced the runs. Every run saves out-of-fold logits, test logits, CRF parameters and its
per-epoch trace under `results/<tag>/`, so all ensembling is CPU-only.

| file | role |
|---|---|
| `phase3_data.py` | data gate, CV split, validator, scoring |
| `phase3_rr.py` | hierarchical trainer, 11 adaptation methods, 5-fold CV, KD |
| `phase3_peft.py` | LoRA, rsLoRA, QLoRA, AdaLoRA, LoRMA, IA³, prompt and prefix tuning |
| `phase3_external.py`, `phase3_legalseg.py` | MARRO, Kalamkar and LegalSeg as allowed-set data |
| `phase3_iltur.py` | IL-TUR `rr` test split as training data (TA-permitted, disclosed) |
| `phase3_lex.py` | TF-IDF + logistic regression member |
| `phase3_ens.py`, `phase3_variants.py` | ensembling, decoding, KD teacher, leaderboard variants |
| `phase3_stack.py`, `phase3_dapt.py`, `phase3_predict.py` | stacking (failed), continued pretraining (failed), teacher labelling |

Full log: [`EXPERIMENTS.md`](EXPERIMENTS.md).
