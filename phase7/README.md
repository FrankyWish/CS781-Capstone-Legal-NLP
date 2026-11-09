# Phase 7: prior case retrieval

**Task.** For each query judgment, rank the candidate cases and return exactly 7 prior cases that it cites. The training set has 827 queries and 4,320 candidates, dev has 118 queries and 1,023 candidates, and test has 237 queries and 1,727 candidates. The median document length is around 4,600 words.

The metric is Micro-F1@7. Since K is fixed, it reduces to `2 · hits / (7Q + Σ|gold|)`, which means it is directly related to the number of correct gold cases in the top 7. The dev ceiling is 0.6181 (556 reachable hits). The dataset is IL-PCR, and its public test labels were never downloaded.

**Results (test).**

| **system** | **kind** | **test** |
| --- | --- | ---: |
| `ft_lorma_m2`: M2 encoder, contrastive LoRMA bi-encoder (compulsory) | Phase-1 model only | 0.3325 |
| `kd_half`: half-depth student of it (compulsory) | Phase-1 model only | 0.3154 |
| `stack_new`: gradient-boosted stack over lexical, passage and dense members | hybrid | 0.4241 |
| `ot2_zst3_p`: transductive Sinkhorn decoding of the saved stacks | hybrid | 0.4536 |
| `cit_abd`: + citator-graph rules from the judgments' own headers | hybrid, no corpus lookup | 0.4979 |
| `cit_abdl_ot2_zst3_p`: + Phase-1 corpus lookup of masked citations | lookup | **0.5157**, rank 2 (graded entry) |

## Architecture

**1. Candidate generation and base ranking.** Several methods were used to rank the candidate cases:

- TF-IDF with square-root query weighting and `max_df 0.2`.
- BM25.
- Passage-level MaxP, using 300-word passages and taking the maximum score over passages.
- Citation-context queries built from the text around each `<CITATION>` mask (`citep`), which was the best single system.
- Dense retrieval using the Phase-1 trunks, with 8 chunks of 1,024 tokens and maximum similarity over chunks.

**2. Stacker** (`phase7_stack.py`). A HistGB model was trained over the top 100 candidates per query. Each member's score was normalised within its query, and the predictions were cross-fitted by query.

**3. Transductive decoding** (`phase7_ot.py`, `phase7_ot2.py`). IL-PCR builds each split's candidate pool using citations from its queries, so the candidates have some global structure. All test queries are ranked together using entropic optimal transport (Sinkhorn).

The rows represent queries with 7 slots each, while the column mass is based on each candidate's share of all slots. This is relaxed using an unbalanced penalty. The configuration was selected using train data only. It gave +35 devx hits over `stack_new`.

**4. Citator rules** (`phase7_citator.py`). Supreme Court judgments keep their JUDIS headers, including details like `CITATION: 1964 AIR 1545`, `DATE OF JUDGMENT` and `CITATOR INFO`.

The following rules were used:

- **Rule A:** Check whether the query's own AIR key appears in a candidate's CITATOR INFO. Devx precision was 0.981.
- **Rule B:** Match AIR/SCR/SCC citations that the masking missed against candidate header keys. Precision was 0.978.
- **Rule D:** Demote candidates whose dates are later than the query. Header dates were consistent for all 352/352 devx cases.
- **Rules L and N:** Look up the query in the Phase-1 corpus to recover information hidden by the masks, mostly case names and years, and match these against the candidates' actual titles. Precision was 0.905 at a cutoff of 0.8.

## Trained retrieval

`phase7_emb.py` fine-tunes the trunks using contrastive learning (InfoNCE), with in-batch negatives and 3 hard negatives, for 600 steps.

LoRMA was the best out of seven techniques. It got 0.3524 devx and recall@100 of 0.727, the highest recall@100 among the single systems. It performed better than QLoRA, norm-only, full fine-tuning, LoRA, IA³ and partial fine-tuning.

Training mainly improved recall and did not improve top-7 precision by much. KD used every second layer and KL loss over the in-batch softmax. It reduced the score by 6%, but also reduced latency by 47%.

## What worked and what did not

The main improvements on test came from transductive decoding (+0.030), the citator rules (+0.044), and the corpus lookup (+0.018).

Several other approaches were tested but did not work:

- Zero-shot bidirectional conversion of a decoder reduced devx hits by 50.
- RM3 query expansion added noise and also interfered with MaxP.
- A temporal predate filter would have removed 23% of devx gold cases. The header dates were later fixed and used instead.
- The "cited in train" popularity prior gave only 0.89x enrichment.
- Cross-encoder reranking was barely trained.
- High Court corpus scans added only 2 devx hits.
- Key inference for queries without headers had precision of only 0.24 to 0.29.

Among the top stacks, the devx rank and test rank were exactly reversed (Spearman -1.0). This was also what the paired half-split 2-SE bar had predicted.

## How to run

The final pipeline runs on CPU and takes around 25 minutes, plus the time needed to download the corpus. The base runs (`stack_new`, `citep`, etc.) come from the earlier drivers `p7.sh` to `p7i.sh`. [`RUN.md`](https://github.com/FrankyWish/CS781-Capstone-Legal-NLP/blob/main/phase7/RUN.md) contains the details for running them.

```bash
python phase7_ot.py all                     # transductive decoding of the saved base runs
python phase7_ot2.py all                    # -> ot2_zst3_p
python phase7_overlap.py scan --splits test,devx --sources train_supreme_court
python phase7_names.py scan                 # candidates' real titles from their corpus copies
python phase7_hc.py                         # mask-free rescan of the SC corpus
python phase7_names.py match                # rule N, cut chosen on devx
python phase7_citator.py all                # rules A/B/D/L(+N) over ot2_zst3_p -> zips
```

A rerun gives 0.5151. Since 8 Nov, ties between rule matches outside the base list are broken using candidate ID. The saved 0.5157 result came from a more favourable ordering of those ties. Smoke tests for all parts run on synthetic data (`code/smoke_*.py`, `code/smoke_restart/`).

| **file** | **role** |
| --- | --- |
| `phase7_data.py` | gate, scoring, submissions |
| `phase7_rank.py` | lexical and passage systems |
| `phase7_enc.py`, `phase7_emb.py` | trunks as encoders; contrastive PEFT and KD |
| `phase7_cite.py`, `phase7_point.py`, `phase7_scale.py` | citation-context retrieval and other signals |
| `phase7_stack.py`, `phase7_rerank.py` | stacker, cross-encoder |
| `phase7_ot.py`, `phase7_ot2.py` | transductive decoding |
| `phase7_probe.py`, `phase7_citator.py`, `phase7_group.py` | header discovery, citator rules, citator groups |
| `phase7_overlap.py`, `phase7_names.py`, `phase7_hc.py` | corpus lookup (kind = lookup) |
| `p7*.sh` | run drivers |

Full log: [`EXPERIMENTS.md`](https://github.com/FrankyWish/CS781-Capstone-Legal-NLP/blob/main/phase7/EXPERIMENTS.md). Every upload: [`submissions/INDEX.md`](https://github.com/FrankyWish/CS781-Capstone-Legal-NLP/blob/main/phase7/submissions/INDEX.md).
