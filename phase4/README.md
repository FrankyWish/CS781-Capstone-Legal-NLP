# Phase 4: court judgment prediction (CJPE)

**Task.** Predict the binary outcome of an Indian Supreme Court appeal (1 = at least one
appeal allowed) from the judgment text with its final order removed. 32,305 unique training
cases (41% positive), 994 real dev cases, 1,517 test cases. Metric: Macro-F1 over {0, 1}.

**Results (test).**

| system | uses | test |
|---|---|---|
| `ens_w_m`: three M4 LoRMA runs, equal-weight mean | Phase-1 model only | **0.7225** |
| `ens_w_b`: two M4 runs + whole-judgment TF-IDF with silver data | genuine hybrid | **0.7488** |
| `ret_m4_spread8`: corpus lookup of the case's own order (55.8% of cases) + M4 elsewhere | lookup | **0.8767**, rank 1 of 101 (graded entry) |
| `ret_ens_w_b_w600t2`: tier-2 lookup (87.6% of cases) + `ens_w_b` | lookup | 0.9598 |

The lookup rows read each test case's removed final order from its copy in the permitted
Phase-1 Supreme Court corpus (1,345 of 1,517 test judgments are in it). That is retrieval, not
prediction, and is reported beside the prediction results rather than in place of them. The
graded entry's only learned component is M4 fine-tuned with LoRMA: 0.73M trained parameters.

## Architecture

The trunk reads several 512-token chunks of the judgment bidirectionally. Each chunk is
mean-pooled, then an attention pool over chunks with a position-decile embedding produces one
vector and a linear head gives the logit. The best view reads 4 chunks spread evenly over the
judgment in training and 8 at test time; the attention pool generalises to more chunks than it
was trained on (+0.011 dev). Balanced class weights, AdamW, cosine schedule, threshold tuned on
dev for Macro-F1.

## What worked and what did not

Required techniques, same 10,000 training documents, last 1,024 tokens (dev / test):
LoRMA on M4 0.6343 / 0.6128, full fine-tuning of M4 0.6100 / 0.5522, KD into an 8-layer M4
student 0.6062 / 0.5666, Houlsby adapters on M1 0.5935 / 0.5557, BitFit on M1 0.5925 / 0.5537,
partial fine-tuning of M2's encoder with layer-wise LR decay 0.5757 / 0.5383, AdaLoRA on M3
0.5744 / 0.5565, gradual unfreezing of M2 0.5716 / 0.5407.

- **Data was the binding constraint.** M4 LoRMA on 10k gold 0.6343 dev, on all 32.3k gold plus
  14.3k silver cases 0.6905. Silver labels come from the final orders of corpus judgments
  (`phase4_external.py`, 92.3% agreement with gold on corpus copies of train cases); 15k silver
  cases were worth about as much as 14k extra gold ones.
- **Reading the whole judgment** beat reading its tail: +0.018 from 4 spread chunks, +0.011 more
  from 8 chunks at test time.
- **TF-IDF over the whole judgment** errs differently from the neural model: together +0.032.
- Failed: an equal-weight ensemble of all eight technique runs (0.6106, below the best member),
  rate-matched thresholds (+0.002 / -0.004), char n-grams (-0.027), dev-greedy member selection
  (0.7582 dev to 0.7462 test).
- Integrity: all 1,517 test texts had labelled copies in the original `dev.json`; they were
  dropped before any selection and reported to the TAs (the organisers later re-uploaded dev).

## How to run

```bash
python phase4_data.py gate                     # stats + leakage checks (CPU)
python phase4_data.py tokenize                 # BPE token cache (CPU)
python phase4_external.py                      # silver cases from the Phase-1 SC corpus
python phase4_cjpe.py --model M4_LexArcP --method lorma --tag m4_lorma          # one technique run
bash run_p4.sh                                 # the 8 required technique runs
bash final_p4.sh                               # m4_final: all gold + silver
bash push_p4.sh                                # m4_spread / m4_spread8 + whole-judgment TF-IDF
python phase4_lex.py --view all --tag lex_all
python phase4_ens.py ens --runs m4_spread8 m4_final lex_all_ext --tag ens_w_b
python phase4_overlap.py                       # how many test cases sit in the corpus (CPU)
python phase4_retrieve.py --base m4_spread8 --submit      # graded entry ret_m4_spread8
python check_p4.py ens_w_b m4_spread8          # re-score with the official evaluator
```

| file | role |
|---|---|
| `phase4_data.py` | loading, dedup, test-copy removal, token cache, metrics, zips |
| `phase4_cjpe.py` | trainer: all adaptation methods, chunk views, KD, predict-only |
| `phase4_peft.py` | LoRMA, AdaLoRA, Houlsby adapters, BitFit, partial FT, gradual unfreezing |
| `phase4_lex.py` | TF-IDF models and section evidence |
| `phase4_ens.py` | ensembles, thresholds, KD teacher |
| `phase4_external.py` | silver training data |
| `phase4_overlap.py`, `phase4_retrieve.py` | corpus overlap measurement and the lookup hybrid |
| `check_p4.py` | official-evaluator re-check of every zip |
| `run_p4.sh`, `final_p4.sh`, `push_p4.sh` | GPU queues; `push2_p4.sh` was never completed, do not run |
| `notebooks/CS781_Phase4_CJPE.ipynb` | early all-in-one notebook, superseded by `code/` |

Full log: [`EXPERIMENTS.md`](EXPERIMENTS.md). Every upload: [`submissions/INDEX.md`](submissions/INDEX.md).
