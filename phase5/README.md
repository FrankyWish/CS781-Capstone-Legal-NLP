# Phase 5: legal statute identification

**Task.** Given the facts of a case (entities, sections and acts masked), predict the set of
IPC/CrPC statutes that apply, from 100 labels. 42,750 train, 10,181 dev (128 labelled copies
of test documents dropped), 13,019 test documents. Metric: Macro-F1 over statute IDs.
The data is IL-TUR `lsi`; its public test split was never downloaded.

**Results (test).**

| system | uses | test |
|---|---|---|
| `top_M1_adapter`: Houlsby adapters on M1, ~1.5 epochs | single Phase-1 model | 0.3830 |
| `ens_model`: greedy average of the neural runs | Phase-1 models only | 0.3949 |
| `ens_hyb`: neural + TF-IDF, probability average | hybrid | 0.4387 |
| `s6_bag`: stacker, mean of 8 seeds (the honest method number) | hybrid | **0.4622** |
| `s5_s1`: one stacker seed, best of 13 uploaded | hybrid | **0.4632**, rank 6 (graded entry) |

The 0.4632 is the best of 13 seeds whose test scores spread 0.4551 to 0.4632, so it includes
leaderboard selection; 0.4622 is the method's number.

## Architecture

**Neural members.** Each document is cut to 8,192 tokens (3/4 head, 1/4 tail); training reads
two 512-token chunks, inference up to six. A label-wise attention head (one query vector per
statute over the trunk's token states) feeds 100 independent sigmoids. Runs: LoRMA on M4
(compulsory), Houlsby adapters on M1, partial fine-tuning of M2 (top 5 of 10 blocks, LLRD 0.85),
QLoRA on M3, prefix tuning on M1, and a half-depth M4 student distilled from the ensemble
(compulsory).

**Lexical members.** TF-IDF word 1-2 and 1-3 grams with one-vs-rest logistic regression.
The final members are cross-fitted on dev: each dev document is scored by a model fitted on
train plus the other three dev folds, and test by train plus all of dev, so dev labels add
training data without any dev document scoring itself.

**Stacker** (`phase5_stack.py`). A histogram gradient-boosted classifier over (document,
candidate statute) rows for each document's top 25 to 35 candidates (`--k`). Features: every member's
per-label z-score and rank, member agreement, statute frequency, and co-occurrence support
P(j|k)p(k) from the other candidates. 4-fold out-of-fold on dev; test uses the mean of the
fold models.

**Thresholds.** One global threshold against per-statute thresholds shrunk toward it,
`t_l = (n_l t_l* + s t_glob) / (n_l + s)` with s = 25, chosen by 2-fold CV over dev (tune on
one half, score the other). Every document gets at least one statute.

## What worked and what did not

What moved test, in order: stacking over members (+0.007 over `ens_hyb`), cross-fitting the
TF-IDF members on dev (+0.006), bagging eight stacker seeds with one more member (+0.005).
Per-label shrunk thresholds helped every model by 0.02 to 0.03.

Refuted, each with its number in `EXPERIMENTS.md`: silver labels read off Phase-1 corpus
judgments (labeller precision 0.58, recall 0.33; no CV gain), a citation rule (the facts are
masked; fires on 2.6% of dev), kNN over near-duplicate cases (0.353 CV), longer neural training
(0.382 after 2.65 epochs vs 0.394 after 1.5), output-volume scaling, class balancing in the
stacker, removing the co-occurrence features. After ~0.56 epoch the four trunks sat within 0.03
of each other.

## How to run

The full Colab procedure, including the setup cell, is in [`RUN.md`](RUN.md).

```bash
python phase5_data.py gate                       # stats + integrity checks (CPU)
python phase5_tok.py --max-tokens 8192           # token cache (CPU)
bash p5.sh goal1                                 # the required neural runs (~4.5 h, L4)
bash p5.sh goal2                                 # longer best run + first ensembles
python phase5_lex.py --tag lex_w12x --crossfit 4 --jobs 10
python phase5_lex.py --tag lex_w13x --ngram 3 --max-feat 800000 --crossfit 4 --jobs 11
for s in 0 1 2 3 4 5 6 7; do
  python phase5_stack.py --submit --k 35 --lr 0.04 --iters 900 --tag s6_s$s --seed $((781+100*s))
done
python phase5_ens.py ens --tag s6_bag --submit --combine logit \
    --runs s6_s0,s6_s1,s6_s2,s6_s3,s6_s4,s6_s5,s6_s6,s6_s7 --weights 1,1,1,1,1,1,1,1
```

| file | role |
|---|---|
| `phase5_data.py` | data gate, official-identical scoring, thresholds, zips |
| `phase5_tok.py` | tokenisation cache |
| `phase5_lsi.py` | neural trainer, label-wise attention head, KD |
| `phase5_peft.py` | LoRMA, adapters, partial FT, QLoRA, prefix tuning |
| `phase5_lex.py`, `phase5_knn.py` | TF-IDF (with dev cross-fitting) and kNN members |
| `phase5_ens.py` | ensembles, thresholds, KD teacher, status |
| `phase5_stack.py` | the second-stage stacker |
| `phase5_external.py`, `phase5_rule.py` | silver labels and the citation rule (both refuted) |
| `p5.sh`, `run_p5.sh`, `make_zips.sh` | the run queues |

Full log: [`EXPERIMENTS.md`](EXPERIMENTS.md). Every upload: [`submissions/SCORES.md`](submissions/SCORES.md).
