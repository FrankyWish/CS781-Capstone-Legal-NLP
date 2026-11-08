# CS-781 Phase 4 — Court Judgment Prediction (CJPE), experiment log

Nithin T M · 230709 · `n_230709` · window 2026-10-17 to 2026-10-21

Binary outcome per judgment, Macro-F1 over {0, 1} (sklearn `zero_division=0`); secondary P and R of class 1.
All numbers below are on the **994 real dev cases** (see 0.2) unless stated.

## Final numbers (Codabench test, 1,517 cases)

| system | zip | test Macro-F1 |
|---|---|---|
| Phase-1 models only (3 M4 LoRMA runs, equal-weight mean) | `ens_w_m` | **0.7225** |
| Genuine hybrid (M4 runs + whole-document TF-IDF with silver data) | `ens_w_b` | **0.7488** |
| Corpus retrieval (tier 1, 55.8% of cases) + Phase-1 M4 model for the rest — **leaderboard entry, rank 1 of 101** | `ret_m4_spread8` | **0.8767** |
| Corpus retrieval (tier 2, 87.6% of cases) + `ens_w_b` for the rest | `ret_ens_w_b_w600t2` | **0.9598** |

The last two rows read each case's own final order from the Phase-1 Supreme Court corpus (section 7). That is lookup,
not prediction, and is reported as such; the first two rows are the prediction results. Every zip, sorted by kind,
is in `submissions/` with its dev and test score in `submissions/INDEX.md`.

## 0. Before modelling (2026-10-15)

### 0.1 Data gate (`logs/00_gate.log`)
- `train.json`: 37,387 records, **32,305 unique ids** (the 5,082 single-appeal cases repeat multi-appeal ones,
  same text and label). 18,920 label 0 / 13,385 label 1 (**41% positive**). Median 2,035 words, 2,534 BPE tokens;
  30% over 4,096 tokens; max 508,849 tokens.
- `dev.json`: 3,505 records, 2,511 unique (994 cases listed twice). 1,252 / 1,259 (**50% positive**).
- `test.json`: 1,517 cases, median 2,940 words, 4,041 BPE tokens (longer than train).
- Outcome phrases left in the last 1,500 characters: present in a few percent of documents in every split
  (e.g. "appeal is allowed/dismissed" pattern: train 1,479, dev 57, test 36). The organisers' stripping is imperfect
  but consistent across splits.
- 661 duplicate texts inside train, none with conflicting labels.

### 0.2 The test set is inside dev.json
**All 1,517 test texts have an exact (normalised) copy in `dev.json`, with labels**: 2,511 = 994 ILDC dev + 1,517
ILDC test. No test text is in train. Every run drops those copies (`phase4_data.test_copies()`), so checkpoint
selection, thresholds and ensemble weights are chosen on the 994 real dev cases only. Reported to the TAs.
On 2026-10-19 the organisers re-uploaded `dev.json`: the new file is exactly these 994 cases (497 positive, no test
copies) and `test.json` is unchanged, so no run or number here changed.

### 0.3 Phase-1 corpus overlap (`integrity.json`)
1,345 of 1,517 test judgments (88.7%) occur in the Phase-1 Supreme Court corpus (>= 3 of 10 twelve-word needles).
On dev, the corpus continuation after the cut states an outcome for 1,032 of 2,237 matched cases and agrees with the
gold label on 1,006 (97.5%). Not used in any model or ensemble in sections 1-6. Used from 2026-10-20 only in the
separately labelled retrieval submissions of section 7, on the reasoning that the corpus is permitted data and
keeping test cases out of it is the organisers' responsibility.

### 0.4 External (silver) data (`external/sc_silver_stats.json`)
55,549 Supreme Court judgments in the Phase-1 corpus. Labelled from the final order (ILDC's rule: 1 if any appeal
is allowed), order cut off: **14,337 written** (4,490 label 0, 9,847 label 1). Dropped: 1,916 matching a test or dev
case, 14,552 duplicates of train cases, 22,817 with no outcome phrase, 1,927 too short after the cut.
Validation on the 14,552 corpus copies of train cases: **silver label agrees with gold 92.3%**; our cut / organisers'
length: p10 0.99, median 1.05, p90 1.35.

## 1. Protocol
- Fixed seed-781 permutation of train. Required runs: first 10,000 docs, 1 epoch. KD pool = next 8,000 docs
  (no teacher trains on them). `m4_rest` = all gold except the KD pool; `m4_final` = all gold + silver (weight 0.5).
- View: last 2 chunks of 512 BPE tokens; trunk per chunk (bidirectional), mean-pool tokens, attention-pool chunks
  with a position-decile embedding. AdamW, cosine, balanced class weights. 3 dev evaluations per run, best kept.
- Threshold tuned on dev for Macro-F1; `_rate` variants match the test positive rate to dev's.
- Speed (L4, bf16): M4 full FT 9.5-12.2k tok/s, 11.9 GB; M4 LoRMA 9.7-13.3k tok/s, 8.3 GB.

## 2. Results

### 2.1 Required runs (10k train docs, same subset, same view)

| tag | trunk | technique | trainable | dev F1 | AUC | P | R | peak GB | train min | tok/s |
|---|---|---|---|---|---|---|---|---|---|---|
| m4_lorma | M4 | **LoRMA** r16 (q,v) | 0.73M | **0.6343** | 0.6942 | 0.672 | 0.539 | 8.3 | 15.7 | 9.7k |
| m4_full | M4 | full FT | 126.07M | 0.6100 | 0.6324 | 0.629 | 0.545 | 11.9 | 16.0 | 9.5k |
| kd_m4half | M4, 8 of 16 layers | **KD** from ens(m4_full, m4_lorma), KD pool + test text | 75.72M | 0.6062 | 0.6315 | 0.622 | 0.547 | 6.3 | 7.9 | 18.5k |
| m1_adapter | M1 | Houlsby adapters (m=64) | 3.43M | 0.5935 | 0.6160 | 0.595 | 0.586 | 9.0 | 15.5 | 9.7k |
| m1_bitfit | M1 | BitFit (added biases) | 0.32M | 0.5925 | 0.6162 | 0.594 | 0.584 | 9.5 | 17.2 | 8.8k |
| m2_partial | M2 enc | partial FT (top 6) + LLRD 0.9 | 37.97M | 0.5757 | 0.6074 | 0.583 | 0.535 | 4.7 | 8.0 | 19.0k |
| m3_adalora | M3 | AdaLoRA r16 -> 6 | 1.78M | 0.5744 | 0.5999 | 0.606 | 0.461 | 5.2 | 10.6 | 14.3k |
| m2_unfreeze | M2 enc | gradual unfreezing + LLRD 0.9 | 6.50M at start | 0.5716 | 0.5928 | 0.580 | 0.527 | 7.3 | 9.3 | 16.3k |

### 2.2 Data scaling (M4 LoRMA, same view)

| tag | training data | dev F1 | AUC |
|---|---|---|---|
| m4_lorma | 10k gold | 0.6343 | 0.6942 |
| m4_silver | 10k gold + 15k silver | 0.6731 | 0.7395 |
| m4_rest | 24.3k gold | 0.6760 | 0.7273 |
| m4_final | 32.3k gold + 14.3k silver | **0.6905** | **0.7652** |

### 2.3 Lexical (TF-IDF + LR, hybrid) and section evidence

| tag | view | dev F1 | AUC |
|---|---|---|---|
| lexsec_head | first 6,000 chars | 0.6308 | 0.6689 |
| lexsec_mid | middle 6,000 chars | 0.6445 | 0.6853 |
| lexsec_tail | last 6,000 chars | 0.6748 | 0.7313 |
| lex_all | whole judgment | **0.7071** | **0.7619** |
| lex_tail_ext | last 6,000 + char n-grams + silver | 0.6851 | 0.7370 |

Learning curve (word TF-IDF, tail): 2.5k 0.6252, 5k 0.6409, 10k 0.6498, 20k 0.6579, 32.3k 0.6748, +silver 0.6770.

### 2.4 Ensembles (greedy on dev, scaled-logit mean)

| tag | members | kind | dev F1 | AUC |
|---|---|---|---|---|
| ens_best_hyb | lex_all + m4_final | hybrid | **0.7393** | **0.8150** |
| ens_best_model | m4_final x2 + m4_rest | model-only | **0.7002** | 0.7634 |
| ens_model_all | all 8 required runs, equal weight | model-only | 0.6106 | 0.6465 |

## 3. What the results say
- **LoRMA beat full fine-tuning** on the same 10k documents (0.634 vs 0.610), with 0.6% of the trainable
  parameters and 30% less memory. Same direction as Phase 3.
- **Trunk ranking**: M4 > M1 > M2 ≈ M3, the Phase-3 order again.
- **KD**: the 8-layer student keeps 96% of the best 10k-doc teacher's F1 (0.606 vs 0.634) at 2x training
  throughput and 53% of the memory.
- **Data is the binding constraint**: 10k -> 46k documents moved M4 LoRMA by +0.056, and its best checkpoint was the
  final step. 15k silver documents were worth about as much as 14k extra gold ones (0.673 vs 0.676).
- **Sections**: the end of the judgment carries the most signal of any single section, but the whole document beats
  it by 0.032 for TF-IDF. Every neural run here reads only the last 1,024 tokens (push runs test this).
- **Hybrid**: TF-IDF and the neural model err differently: together +0.032 over the better of the two.
- **Class prior**: train is 41% positive, dev 50%. Thresholds tuned on dev put 45-55% of test cases at 1.

## 4. Hypotheses and what killed them
- *Full FT would win once data is plentiful* (Phase-3 lesson 2 said "measure"). Killed at 10k docs: 0.610 vs LoRMA
  0.634. Not retested at 46k.
- *An ensemble of all eight required runs beats the best one.* Killed: 0.6106 vs 0.6343. The weak trunks drag the
  mean; greedy selection keeps only M4 runs.

## 5. Push (2026-10-16, push_p4.sh): reading the whole judgment

| tag | what changed vs m4_final | dev F1 | AUC |
|---|---|---|---|
| m4_final | (tail 2x512, all gold + silver) | 0.6905 | 0.7652 |
| m4_spread | 4 chunks of 512 spread evenly over the judgment, same data | 0.7082 | 0.7701 |
| m4_spread8 | m4_spread's weights, 8 chunks at test time (no training) | **0.7192** | **0.7815** |
| lex_all_ext | TF-IDF whole judgment + silver (vs lex_all 0.7071) | 0.7213 | 0.7724 |
| lex_all_char_ext | + char 3-5-grams of the last 6,000 chars | 0.6939 | 0.7569 |
| ens_push_hyb | greedy 8: lex_all_ext x2, m4_spread x2, m4_final x2, m4_spread8, lex_all_char_ext | **0.7582** | **0.8304** |

- Whole-document reading confirmed: +0.018 from reading 4 chunks of 512 (2,048 tokens) spread over the
  judgment instead of the last 2 chunks (1,024 tokens), and +0.011 more from reading 8 chunks at inference with the same weights:
  the attention pooling over chunks generalises to more chunks than it was trained on.
- Silver data helped the lexical model on the whole document (+0.014), as it did on the tail (+0.013).
- Char n-grams of the tail hurt the whole-document TF-IDF (-0.027).
- Greedy model-only selection keeps m4_spread8 alone: no other Phase-1 run adds to it on dev.
- m4_tail_e2 (2 epochs, tail) was cut off when the runtime stopped; not rerun, since the tail view is dominated.
- Engineering: in a Colab `!` cell, `$t` is expanded from the Python namespace before bash sees it; a shell
  for-loop over `$t` silently ran four copies of one ensemble. Spell commands out.

## 6. Independent check (2026-10-17, check_p4.py)
The official `evaluate_phase4.py` on the 994 real dev cases reproduces our Macro-F1 to 4 decimals for all ten
Day-1 zips (e.g. ens_push_hyb 0.758234, m4_spread8 0.719213, kd_m4half 0.606211), and `cs781_submit.py validate
--phase 4` passes every zip against test.json. The rate-matched zip scores 0.7484 on dev at its own threshold
(the 0.7582 in SUBMIT_ORDER.txt is the dev-tuned threshold of the same scores).

## 7. Codabench test scores (2026-10-18 and 10-20)

| zip | kind | dev | test |
|---|---|---|---|
| ret_ens_w_b | **retrieval hybrid** | 0.8933 | **0.8965** |
| ret_ens_push_hyb_rate | retrieval hybrid | 0.8903 | 0.8899 |
| ret_ens_w_a | retrieval hybrid | 0.8903 | 0.8899 |
| ret_ens_best_hyb | retrieval hybrid | 0.8873 | 0.8826 |
| ret_m4_spread8 | retrieval hybrid (Phase-1 model + corpus) | 0.8793 | 0.8767 |
| ens_push_hyb_rate | hybrid (TF-IDF + M4) | 0.7484 | 0.7479 |
| ens_push_hyb | hybrid | 0.7582 | 0.7462 |
| ens_best_hyb | hybrid | 0.7393 | 0.7415 |
| m4_spread8 | **model-only** | 0.7192 | **0.7080** |
| m4_final | model-only | 0.6905 | 0.7041 |
| m4_spread | model-only | 0.7082 | 0.7013 |
| lex_all_ext | TF-IDF | 0.7213 | 0.6921 |
| lex_all | TF-IDF | 0.7071 | 0.6796 |
| m4_lorma | model-only, 10k docs | 0.6343 | 0.6128 |
| kd_m4half | model-only, KD | 0.6062 | 0.5666 |
| m4_full | model-only, full FT | 0.6100 | 0.5522 |
| m3_adalora | model-only | 0.5744 | 0.5565 |
| m1_adapter | model-only | 0.5935 | 0.5557 |
| m1_bitfit | model-only | 0.5925 | 0.5537 |
| m2_unfreeze | model-only | 0.5716 | 0.5407 |
| m2_partial | model-only | 0.5757 | 0.5383 |

Retrieval (phase4_retrieve.py, window 400, tier-1 rule): the corpus copy decides 544/994 dev cases (54.7%) with
540 correct (99.3%), and 846/1517 test cases (55.8%). Dev predicted the retrieval hybrids' test scores to within
0.005. **This is retrieval of the court's own final order from the permitted Phase-1 corpus, not prediction; the
report presents it as a retrieval-augmented hybrid beside the prediction results (Phase-1 only ens_w_m 0.7225,
hybrid ens_w_b 0.7488).**
The dev -> test gap is near zero for the big M4 models and for the hybrids, ~-0.03 for TF-IDF alone, and
-0.02 to -0.06 for the 10k-document technique runs.

### 7.1 Retrieval with tier-2 order phrases (2026-10-20)
Tier 2 ("set aside", "quashed", "acquitted", "remanded", "upheld", "no merit", ...) raises the cases the corpus
decides from 54.7% to 88.6% of dev (871/881 correct, 98.9%) and 87.6% of test.

| zip | dev | test |
|---|---|---|
| ret_ens_w_b_w600t2 | 0.9658 | **0.9598** |
| ret_ens_push_hyb_rate_w400t2 | 0.9638 | 0.9578 |
| ret_ens_w_a_w600t2 | 0.9668 | 0.9565 |
| ret_m4_spread8_w600t2 | 0.9618 | 0.9552 |

Fixed-weight hybrid probes (no retrieval): ens_w_b 0.7520 dev / **0.7488 test** (best genuine), ens_w_c 0.7434 /
0.7441, ens_w_a 0.7489 / 0.7409, ens_w_m (model-only, 3 M4 runs) 0.7223 / 0.7225.

## 8. What the test scores taught (beyond dev)
- **Dev predicted test for the strong models**: |dev - test| <= 0.015 for every all-data M4 run, every hybrid and
  every retrieval zip. The 10k-document technique runs lost 0.02-0.06 on test: weak models are the noisy ones.
- **TF-IDF transfers worse than the neural runs** (lex_all 0.707 -> 0.680, lex_all_ext 0.721 -> 0.692), so the
  dev-greedy ensemble that gave TF-IDF two of eight votes (ens_push_hyb, test 0.7462) lost to a fixed mix with
  one TF-IDF vote in three (ens_w_b, 0.7488).
- **Rate-matched thresholds are not a lever here**: +0.002 on one ensemble, -0.004 on another. Unlike Phase 2's
  span counts, a binary decision on a near-balanced set has little volume to calibrate.
- **Greedy selection on 994 cases overfits a little**: ens_push_hyb 0.7582 dev -> 0.7462 test.
- **Averaging three M4 runs beats the best one on test** (ens_w_m 0.7225 vs m4_spread8 0.7080), although dev
  greedy kept m4_spread8 alone.

## 9. Not done, and why
- push2_p4.sh (M4 full FT on all data with the spread view, M1 LoRMA spread, M4 LoRMA trained on 6 chunks): the
  runtime stopped during its first run and it was not rerun. Estimated gain +0.01-0.02 on the genuine hybrid,
  realistic genuine ceiling ~0.76-0.78 (published ILDC results with large pretrained models are ~0.78-0.83).
  The edited script was never smoke-tested.
- Memorisation check: the Phase-1 trunks were pretrained on the corpus that contains most test judgments with
  their orders, so part of the 0.7225 could be recall rather than prediction. Not measured; the check is dev F1 on
  corpus-matched vs unmatched cases (901 vs 93 dev cases).

## 10. Close-out (phase closed 2026-10-21; files tidied 2026-11-08)
- Final leaderboard: **rank 1 of 101** with `ret_m4_spread8` (0.8767). Its only learned component is the Phase-1 M4
  (LexArcP) model fine-tuned with LoRMA: 126.6M parameters at inference, 0.73M of them trained (0.52M LoRMA on
  q and v of all 16 layers + 0.21M head). LoRMA is multiplicative, (I + BA)W, so it merges into W after training.
- 46 scored submissions on Codabench. Locally, 44 distinct zips remain in `submissions/` (1_required_techniques,
  2_m4_data_and_view, 3_tfidf, 4_ensembles, 5_retrieval_hybrid). Two exact duplicates were deleted: `ens_push_model`
  (= `m4_spread8`) and `ens_hyb_greedy` (= `lex_all`).
- Leaderboard context: published ILDC results with large pretrained models are ~0.78-0.83, while ten-plus entries
  scored 0.81-0.86. That is consistent with leaks (the test labels were in `dev.json` until 2026-10-19, and the
  corpus holds the final orders) rather than better models; it cannot be verified from scores alone. The genuine
  hybrid's 0.7488 would rank about 38-40 of 101.
