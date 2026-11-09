# Phase 5: legal statute identification

**Task.** Given the facts of a case, with entities, sections and acts masked, predict which IPC/CrPC statutes apply from 100 possible labels. There are 42,750 training documents, 10,181 dev documents (128 labelled copies of test documents were removed), and 13,019 test documents. The metric is Macro-F1 over statute IDs. The dataset is IL-TUR `lsi`; its public test split was never downloaded.

**Results (test).**

| **system** | **uses** | **test** |
| --- | --- | ---: |
| `top_M1_adapter`: Houlsby adapters on M1, ~1.5 epochs | single Phase-1 model | 0.3830 |
| `ens_model`: greedy average of neural runs | Phase-1 models only | 0.3949 |
| `ens_hyb`: neural + TF-IDF, probability average | hybrid | 0.4387 |
| `s6_bag`: stacker, mean of 8 seeds (honest method number) | hybrid | **0.4622** |
| `s5_s1`: one stacker seed, best of 13 uploaded | hybrid | **0.4632**, rank 6 (graded entry) |

The 0.4632 score is the best result from 13 seeds, whose test scores ranged from 0.4551 to 0.4632. So it also includes some leaderboard selection. The more honest number for the method is 0.4622.

## Architecture

**Neural members.** Each document is cut to 8,192 tokens, keeping 3/4 from the beginning and 1/4 from the end. During training, two 512-token chunks are read, while inference uses up to six chunks. A label-wise attention head has one query vector for each statute and uses it over the trunk's token states. This gives 100 separate sigmoid outputs.

The runs included LoRMA on M4 (compulsory), Houlsby adapters on M1, partial fine-tuning of M2 (top 5 of 10 blocks, LLRD 0.85), QLoRA on M3, prefix tuning on M1, and a half-depth M4 student distilled from the ensemble (compulsory).

**Lexical members.** TF-IDF word 1-2 and 1-3 grams were used with one-vs-rest logistic regression. The final members were cross-fitted on dev. Each dev document was scored by a model trained on the training data plus the other three dev folds. For test, the model used train plus all of dev. This way, dev labels could also be used for training without a document scoring itself.

**Stacker** (`phase5_stack.py`). A histogram gradient-boosted classifier was trained on (document, candidate statute) rows, using the top 25 to 35 candidates for each document (`--k`). Its features included each member's per-label z-score and rank, agreement between members, statute frequency, and co-occurrence support P(j|k)p(k) from the other candidates.

It used 4-fold out-of-fold predictions on dev. For test, the predictions from the fold models were averaged.

**Thresholds.** One global threshold was used along with per-statute thresholds shrunk towards it:

`t_l = (n_l t_l* + s t_glob) / (n_l + s)`, with `s = 25`.

These were selected using 2-fold CV on dev, tuning on one half and scoring on the other. Every document was made to have at least one predicted statute.

## What worked and what did not

The main improvements on test came from stacking the members (+0.007 over `ens_hyb`), cross-fitting the TF-IDF members on dev (+0.006), and averaging eight stacker seeds with one extra member (+0.005). Using per-label thresholds shrunk towards the global threshold also helped every model by around 0.02 to 0.03.

Some other ideas did not help. These included silver labels taken from Phase-1 corpus judgments (labeller precision 0.58 and recall 0.33, with no CV gain), a citation rule (facts were masked and it only fired on 2.6% of dev), and kNN over near-duplicate cases (0.353 CV). Longer neural training also did not help: it scored 0.382 after 2.65 epochs compared to 0.394 after 1.5 epochs. Output-volume scaling, class balancing in the stacker, and removing the co-occurrence features were also tested and did not improve things. After around 0.56 epoch, the four trunks were already within 0.03 of each other.

All these experiments and their numbers are in `EXPERIMENTS.md`.

## How to run

The full Colab procedure, including the setup cell, is available in [`RUN.md`](https://github.com/FrankyWish/CS781-Capstone-Legal-NLP/blob/main/phase5/RUN.md).

```bash
python phase5_data.py gate                       # stats + integrity checks (CPU)
python phase5_tok.py --max-tokens 8192           # token cache (CPU)
bash p5.sh goal1                                 # required neural runs
bash p5.sh goal2                                 # longer best run + first ensembles

python phase5_lex.py --tag lex_w12x --crossfit 4 --jobs 10
python phase5_lex.py --tag lex_w13x --ngram 3 --max-feat 800000 --crossfit 4 --jobs 11

for s in 0 1 2 3 4 5 6 7; do
  python phase5_stack.py --submit --k 35 --lr 0.04 --iters 900 --tag s6_s$s --seed $((781+100*s))
done

python phase5_ens.py ens --tag s6_bag --submit --combine logit \
    --runs s6_s0,s6_s1,s6_s2,s6_s3,s6_s4,s6_s5,s6_s6,s6_s7 \
    --weights 1,1,1,1,1,1,1,1
```

| **file** | **role** |
| --- | --- |
| `phase5_data.py` | data gate, official-identical scoring, thresholds, zips |
| `phase5_tok.py` | tokenisation cache |
| `phase5_lsi.py` | neural trainer, label-wise attention head, KD |
| `phase5_peft.py` | LoRMA, adapters, partial FT, QLoRA, prefix tuning |
| `phase5_lex.py`, `phase5_knn.py` | TF-IDF (with dev cross-fitting) and kNN members |
| `phase5_ens.py` | ensembles, thresholds, KD teacher, status |
| `phase5_stack.py` | the second-stage stacker |
| `phase5_external.py`, `phase5_rule.py` | silver labels and citation rule (both refuted) |
| `p5.sh`, `run_p5.sh`, `make_zips.sh` | run scripts |

Full log: [`EXPERIMENTS.md`](https://github.com/FrankyWish/CS781-Capstone-Legal-NLP/blob/main/phase5/EXPERIMENTS.md). Every upload: [`submissions/SCORES.md`](https://github.com/FrankyWish/CS781-Capstone-Legal-NLP/blob/main/phase5/submissions/SCORES.md).
