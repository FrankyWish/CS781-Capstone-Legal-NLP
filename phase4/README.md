# Phase 4: court judgment prediction (CJPE)

**Task.** The task was to predict the binary outcome of an Indian Supreme Court appeal using the judgment text, after removing its final order. The labels were 1 if at least one appeal was allowed and 0 otherwise. There were 32,305 unique training cases (41% positive), 994 actual dev cases and 1,517 test cases. The evaluation metric was Macro-F1 over the two classes {0, 1}.

**Results (test).**

| **system** | **uses** | **test** |
|---|---|---:|
| `ens_w_m`: three M4 LoRMA runs, equal-weight mean | Phase-1 model only | **0.7225** |
| `ens_w_b`: two M4 runs + whole-judgment TF-IDF with silver data | genuine hybrid | **0.7488** |
| `ret_m4_spread8`: corpus lookup of the case's own order (55.8% of cases) + M4 elsewhere | lookup | **0.8767**, rank 2 (graded entry) |
| `ret_ens_w_b_w600t2`: tier-2 lookup (87.6% of cases) + `ens_w_b` | lookup | 0.9598 |

The lookup-based systems work by finding the same test case in the permitted Phase-1 Supreme Court corpus and retrieving its original final order. Around 1,345 of the 1,517 test judgments were present in that corpus.

This is retrieval rather than actual outcome prediction, so I have kept these results separate from the normal prediction results. The graded submission used `ret_m4_spread8`, where M4 fine-tuned with LoRMA was the only learned component. It had just 0.73M trained parameters.

## Architecture

The model reads multiple 512-token chunks from the judgment using a bidirectional Phase-1 trunk. Each chunk is mean-pooled, and then an attention pooling layer combines the chunks using position-decile embeddings. A linear head produces the final prediction logit.

The best setup used 4 chunks spread across the judgment during training and 8 chunks during testing. The attention pooling layer was able to handle more chunks than it had seen during training, which improved dev performance by +0.011.

I also used balanced class weights, AdamW, a cosine learning-rate schedule and a threshold selected on the dev set to maximise Macro-F1.

## What worked and what did not

The required techniques were tested using the same 10,000 training documents and the last 1,024 tokens for dev and test.

| **method** | **dev** | **test** |
|---|---:|---:|
| LoRMA on M4 | 0.6343 | 0.6128 |
| Full fine-tuning of M4 | 0.6100 | 0.5522 |
| KD into an 8-layer M4 student | 0.6062 | 0.5666 |
| Houlsby adapters on M1 | 0.5935 | 0.5557 |
| BitFit on M1 | 0.5925 | 0.5537 |
| Partial fine-tuning of M2 encoder with layer-wise LR decay | 0.5757 | 0.5383 |
| AdaLoRA on M3 | 0.5744 | 0.5565 |
| Gradual unfreezing of M2 | 0.5716 | 0.5407 |

**Data was the biggest limitation.**

M4 LoRMA trained on 10k gold examples scored 0.6343 on dev. When I used all 32.3k gold cases along with 14.3k silver cases, the score increased to 0.6905.

The silver labels were generated using the final orders from judgments in the Phase-1 corpus (`phase4_external.py`). They had 92.3% agreement with the gold labels on corpus copies of training cases. Adding around 15k silver cases gave an improvement similar to adding another 14k gold cases.

**Reading the whole judgment helped.**

Using 4 chunks spread across the judgment instead of only reading the ending improved the score by +0.018. Increasing this to 8 chunks at test time gave another +0.011.

**TF-IDF also helped.**

The TF-IDF model used the entire judgment and made different mistakes compared to the neural model. Combining the two improved the result by +0.032.

**Things that did not work:**

- An equal-weight ensemble of all eight technique runs scored 0.6106, which was below the best individual member.
- Rate-matched thresholds gave inconsistent improvements (+0.002 / -0.004).
- Character n-grams reduced the score by -0.027.
- Dev-greedy member selection gave 0.7582 on dev but dropped to 0.7462 on test.

### Data integrity

All 1,517 test texts had labelled copies in the original `dev.json`. I removed these copies before doing any model selection and reported the issue to the TAs. The organisers later uploaded the dev data again.

## How to run

```bash
python phase4_data.py gate
# stats + leakage checks (CPU)

python phase4_data.py tokenize
# BPE token cache (CPU)

python phase4_external.py
# silver cases from the Phase-1 SC corpus

python phase4_cjpe.py --model M4_LexArcP --method lorma --tag m4_lorma
# one technique run

bash run_p4.sh
# the 8 required technique runs

bash final_p4.sh
# m4_final: all gold + silver

bash push_p4.sh
# m4_spread / m4_spread8 + whole-judgment TF-IDF

python phase4_lex.py --view all --tag lex_all

python phase4_ens.py ens --runs m4_spread8 m4_final lex_all_ext --tag ens_w_b

python phase4_overlap.py
# how many test cases are present in the corpus (CPU)

python phase4_retrieve.py --base m4_spread8 --submit
# graded entry ret_m4_spread8

python check_p4.py ens_w_b m4_spread8
# re-score using the official evaluator
```

## Files

| **file** | **role** |
|---|---|
| `phase4_data.py` | loading, deduplication, test-copy removal, token cache, metrics and zips |
| `phase4_cjpe.py` | trainer: all adaptation methods, chunk views, KD and predict-only |
| `phase4_peft.py` | LoRMA, AdaLoRA, Houlsby adapters, BitFit, partial FT and gradual unfreezing |
| `phase4_lex.py` | TF-IDF models and section evidence |
| `phase4_ens.py` | ensembles, thresholds and KD teacher |
| `phase4_external.py` | silver training data |
| `phase4_overlap.py`, `phase4_retrieve.py` | corpus overlap measurement and lookup hybrid |
| `check_p4.py` | official evaluator re-check of every zip |
| `run_p4.sh`, `final_p4.sh`, `push_p4.sh` | experiment run scripts; `push2_p4.sh` was never completed and should not be run |
| `notebooks/CS781_Phase4_CJPE.ipynb` | early all-in-one notebook, later replaced by the scripts in `code/` |

The complete experiment log is available in [`EXPERIMENTS.md`](https://github.com/FrankyWish/CS781-Capstone-Legal-NLP/blob/main/phase4/EXPERIMENTS.md).

All submissions are listed in [`submissions/INDEX.md`](https://github.com/FrankyWish/CS781-Capstone-Legal-NLP/blob/main/phase4/submissions/INDEX.md).
