# Phase 4 submissions — every zip, its folder and its score

44 zips, one per distinct set of predictions (two exact duplicates, `ens_push_model` = `m4_spread8` and
`ens_hyb_greedy` = `lex_all`, were deleted on 2026-11-08; both had been uploaded and scored the same as the kept one).
Dev = the 994 real dev cases. Test = Codabench score. `*_rate` = same scores as the zip it names, threshold set so the
test positive rate matches dev's. Leaderboard entry: **`ret_m4_spread8`, test 0.8767, rank 1 of 101**.

## 1_required_techniques — the graded technique comparison (10k train docs, last 1,024 tokens)

| zip | trunk | technique | dev | test |
|---|---|---|---|---|
| m4_lorma | M4 | LoRMA r16 (compulsory) | 0.6343 | 0.6128 |
| m4_full | M4 | full fine-tuning | 0.6100 | 0.5522 |
| kd_m4half | M4, 8 of 16 layers | knowledge distillation (compulsory) | 0.6062 | 0.5666 |
| m1_adapter | M1 | Houlsby adapters | 0.5935 | 0.5557 |
| m1_bitfit | M1 | BitFit | 0.5925 | 0.5537 |
| m2_partial | M2 encoder | partial / layer-wise FT | 0.5757 | 0.5383 |
| m3_adalora | M3 | AdaLoRA | 0.5744 | 0.5565 |
| m2_unfreeze | M2 encoder | gradual unfreezing | 0.5716 | 0.5407 |

## 2_m4_data_and_view — M4 LoRMA with more data and the whole-judgment view

| zip | change | dev | test |
|---|---|---|---|
| m4_rest | 24.3k gold docs | 0.6760 | 0.6638 |
| m4_silver | 10k gold + 15k silver | 0.6731 | 0.6731 |
| m4_final | all 32.3k gold + 14.3k silver | 0.6905 | 0.7041 |
| m4_spread | same data, 4 chunks spread over the judgment | 0.7082 | 0.7013 |
| m4_spread8 | m4_spread's weights reading 8 chunks | 0.7192 | 0.7080 |

## 3_tfidf — TF-IDF + logistic regression (hybrid members, not Phase-1 models)

| zip | view | dev | test |
|---|---|---|---|
| lex_all_ext | whole judgment + silver | 0.7213 | 0.6921 |
| lex_all | whole judgment | 0.7071 | 0.6796 |
| lex_all_char_ext | whole + char n-grams + silver | 0.6939 | 0.6591 |
| lex_tail_ext | last 6,000 chars + char n-grams + silver | 0.6851 | 0.6592 |
| lex_tail | last 6,000 chars + char n-grams | 0.6720 | 0.6539 |

## 4_ensembles — genuine prediction (no corpus lookup)

| zip | members | kind | dev | test |
|---|---|---|---|---|
| **ens_w_b** | m4_spread8, m4_final, lex_all_ext (equal) | hybrid | 0.7520 | **0.7488** (best genuine) |
| ens_push_hyb_rate | greedy 8 incl. TF-IDF, rate threshold | hybrid | 0.7484 | 0.7479 |
| ens_push_hyb | greedy 8 incl. TF-IDF | hybrid | 0.7582 | 0.7462 |
| ens_w_c | m4_spread8 x2, m4_final x2, m4_spread, lex_all_ext | hybrid | 0.7434 | 0.7441 |
| ens_best_hyb | lex_all + m4_final | hybrid | 0.7393 | 0.7415 |
| ens_w_a | m4_spread8, m4_final, m4_spread, lex_all_ext | hybrid | 0.7489 | 0.7409 |
| ens_best_hyb_rate | ens_best_hyb, rate threshold | hybrid | | 0.7375 |
| **ens_w_m** | m4_spread8, m4_final, m4_spread (equal) | model-only | 0.7223 | **0.7225** (best Phase-1 only) |
| ens_final_hyb | lexsec_all, m4_silver x2, lex_tail_ext | hybrid | 0.7364 | 0.7211 |
| ens_push_model_rate | m4_spread8, rate threshold | model-only | | 0.7079 |
| ens_best_model_rate | ens_best_model, rate threshold | model-only | | 0.6934 |
| ens_best_model | m4_final x2 + m4_rest | model-only | 0.7002 | 0.6898 |
| ens_final_rate | ens_final_model, rate threshold | model-only | | 0.6716 |
| ens_final_model | m4_rest + m4_silver x2 | model-only | 0.6870 | 0.6685 |
| ens_model_greedy | m4_lorma + kd_m4half | model-only | 0.6413 | 0.5984 |
| ens_model_rate | ens_model_greedy, rate threshold | model-only | | 0.5978 |
| ens_model_all | all 8 required runs, equal | model-only | 0.6106 | 0.5669 |

## 5_retrieval_hybrid — final order read from the Phase-1 corpus (lookup, not prediction)

The corpus copy of the same judgment decides 55.8% (tier-1 phrases) or 87.6% (tier-2, `_w*t2`) of test cases;
the named ensemble predicts the rest. Report these as retrieval, beside the genuine numbers above.

| zip | base for undecided cases | dev | test |
|---|---|---|---|
| ret_ens_w_b_w600t2 | ens_w_b | 0.9658 | **0.9598** |
| ret_ens_push_hyb_rate_w400t2 | ens_push_hyb_rate | 0.9638 | 0.9578 |
| ret_ens_w_a_w600t2 | ens_w_a | 0.9668 | 0.9565 |
| ret_m4_spread8_w600t2 | m4_spread8 (Phase-1 only) | 0.9618 | 0.9552 |
| ret_ens_w_b | ens_w_b | 0.8933 | 0.8965 |
| ret_ens_push_hyb_rate | ens_push_hyb_rate | 0.8903 | 0.8899 |
| ret_ens_w_a | ens_w_a | 0.8903 | 0.8899 |
| ret_ens_best_hyb | ens_best_hyb | 0.8873 | 0.8826 |
| **ret_m4_spread8** | m4_spread8 (Phase-1 only) | 0.8793 | **0.8767** (leaderboard entry) |
