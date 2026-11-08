# Phase 6 submissions

**Final: `6_nsel_4trunk/nsel4_best.zip`, BERTScore F1 0.8606, rank 1** (Codabench ID 1030632; screenshot `../leaderboard_final.png`).
Every zip is flat with `prediction.json` at the root. Scores are the Codabench test scores. Names are the tag
only: the `NN_` upload prefixes came from two different numberings and are dropped.

Moved here from Downloads on 2026-11-08. Exact duplicates went to the Recycle Bin.

Uploaded straight from Drive, so not in this folder (copies stay in Drive `cs781_capstone/phase6/submissions/`):
`m3_pfeiffer` 0.8350, `nsel4_stack` 0.8606, `nsel4_nn` 0.8604, `mbr_all` 0.8379, `mbr_all_long` 0.8374,
`mbr_all_short` 0.8382, `win_best` 0.8410, `m1_win` 0.8370, `m2_win` 0.8249, `m3_win` 0.8348, `m4_win` 0.8317,
`sel_short` 0.8388, `sel_long` 0.8374. Never uploaded on purpose: every `cons_*` zip and `mbr_bert`, which run
roberta-large on test documents (spec §12.2).

## `1_required/`: The 7 required runs (§12.2): 6 PEFT techniques + KD, all four Phase-1 trunks, abstractive

| zip | what | test |
|---|---|---|
| `m4_lorma.zip` | LoRMA on M4 pointer-generator (compulsory) | 0.8343 |
| `m2_lora.zip` | LoRA, M2 encoder-decoder | 0.8250 |
| `m1_houlsby.zip` | Houlsby adapters, M1 | 0.8364 |
| `m3_qlora.zip` | QLoRA, M3 MLA | 0.8344 |
| `m2_adalora.zip` | AdaLoRA, M2 | 0.8164 |
| `kd_m1half.zip` | KD into half-depth M1 (compulsory) | 0.8277 |

## `2_model_only/`: Best Phase-1-model-only (abstractive) systems

| zip | what | test |
|---|---|---|
| `rank_model.zip` | HistGB reranker over model candidates | 0.8391 |
| `mbr_model.zip` | MBR over model candidates | 0.8391 |
| `m4_lorma_trunc.zip` | LoRMA M4, truncated input | 0.8390 |
| `m4_lorma_2048.zip` | LoRMA M4, 2048-token view | 0.8374 |
| `m4_full.zip` | full fine-tune M4 | 0.8343 |

## `3_extractive/`: Sentence selection from the test document (not abstractive)

| zip | what | test |
|---|---|---|
| `sel_best.zip` | HistGB on ROUGE labels | 0.8379 |
| `bsel_best.zip` | HistGB on BERTScore labels | 0.8546 |
| `bsel_best_x080.zip` | length x0.8 | 0.8542 |
| `bsel_best_x090.zip` | length x0.9 | 0.8543 |
| `bsel_best_x110.zip` | length x1.1 | 0.8547 |
| `bsel_best_x125.zip` | length x1.25 | 0.8543 |
| `bsel2_f.zip` | all fit docs + word-level score (dev gain did not transfer) | 0.8539 |

## `4_hybrid/`: Neural + extractive mixes

| zip | what | test |
|---|---|---|
| `rank_all.zip` | reranker over all candidates | 0.8528 |
| `stitch_m1_houlsby_bsel_best_25.zip` | 25% neural opening + bsel_best | 0.8502 |
| `stitch_m1_houlsby_bsel_best_50.zip` | 50% neural opening | 0.8458 |
| `stitch_m4_lorma_2048_bsel_best_25.zip` | 25% | 0.8497 |
| `stitch_m4_lorma_2048_bsel_best_50.zip` | 50% | 0.8449 |
| `stitch_m4_lorma_trunc_bsel_best_25.zip` | 25% | 0.8518 |
| `stitch_m4_lorma_trunc_bsel_best_50.zip` | 50% | 0.8471 |

## `5_nsel_2trunk/`: p6h: M1 + M2 fine-tuned as sentence scorers (extractive)

| zip | what | test |
|---|---|---|
| `nsel_stack.zip` | stacked with bsel features | 0.8596 |
| `nsel_best.zip` | best window/citation setting | 0.8596 |
| `nsel_nn.zip` | trunks only | 0.8593 |
| `nsel_mix75.zip` | 75% trunks / 25% bsel | 0.8595 |
| `nsel_mix50.zip` | 50/50 | 0.8591 |
| `nsel_mix25.zip` | 25% trunks | 0.8578 |

## `6_nsel_4trunk/`: p6i: M1-M4 as sentence scorers (extractive). **nsel4_best is the final pushed entry, rank 1**

| zip | what | test |
|---|---|---|
| `nsel4_best.zip` | **FINAL, pushed, rank 1** | 0.8606 |
| `nsel4_mix75.zip` | 75% trunks | 0.8599 |
| `nsel4_mix50.zip` | 50/50 | 0.8592 |
| `nsel4_mix25.zip` | 25% trunks | 0.8578 |
| `nsel4_bsel3.zip` | bsel refit on all docs, no trunks; byte-identical predictions to nsel_bsel3 | 0.8529 |
