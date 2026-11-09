# Phase 3 submissions

Codabench shows 92 valid submissions for this phase. This index lists the 89 zip files kept locally
(`submission_<tag>.zip`; the zips themselves are not in git). Test scores are given where they are recorded in
[`EXPERIMENTS.md`](../EXPERIMENTS.md); "–" means the score is not recorded here. All CV numbers and the
method details are in `EXPERIMENTS.md` §2.

## Single models (Phase-1 trunk + hierarchical sentence head, 5-fold CV, test logits averaged)

| tag | trunk and method | test |
|---|---|---:|
| `base_M2`, `base_M2_3fold`, `base_M2_5fold` | M2 encoder-decoder, full fine-tune (first runs, 1/3/5 folds) | 0.4726 (`base_M2`) |
| `ext_M4` | M4, full fine-tune, gold + external (MARRO, Kalamkar) data | 0.4912 |
| `lorma_M3` | M3, LoRMA r16 (compulsory) | 0.5045 |
| `lora_M4` | M4, LoRA r16 | 0.5223 |
| `rslora_M1` | M1_SC, rsLoRA r64 | 0.5193 |
| `ia3_M3` | M3, IA³ | 0.4696 |
| `qlora_M2` | M2, QLoRA (NF4 + double quantisation) r16 | 0.4442 |
| `lora_M4_aux` | M4, LoRA + label-shift auxiliary head, 128-sentence windows | 0.4929 |
| `lora_M4_dapt_aux` | as `lora_M4_aux`, after in-domain continued pretraining | 0.4814 |
| `kd_M4half` | 8-layer M4 student distilled from the model-only ensemble (compulsory) | 0.5140 |
| `kd2_M4half` | second-round distillation with the auxiliary head | 0.5052 |
| `lora_M4_lseg` | M4 after LegalSeg intermediate training (2,000 judgments), LoRA r16 | 0.5636 |
| `lora_M4_lsegall` | M4 after intermediate training on all LegalSeg (7,117) + MARRO + Kalamkar, LoRA r16 | **0.5637** (best model-only) |
| `lex_tfidf_5fold` | TF-IDF n-grams + logistic regression (hybrid member) | – |

## Ensembles (mean of member logits + prior correction and transition decoding)

| tag | members | test |
|---|---|---:|
| `ens_day1_model` | ext_M4 x3 + lorma_M3 | 0.5032 |
| `ens_day1` | ext_M4 x2 + lorma_M3 + lex_tfidf (hybrid) | 0.5356 |
| `ens_final_model` | kd_M4half x4 + ext_M4 x2 + qlora_M2, greedy mean | 0.5447 |
| `ens_final_hyb2` | ext_M4, kd_M4half, lorma_M3, lora_M4, lex_tfidf, equal weights (hybrid) | 0.5458 |
| `ens_final3`, `ens_final_hyb`, `ens_push1`, `ens_push_hyb`, `ens_push_model` | other member sets from the same days | – |

## Leaderboard variants from `phase3_variants.py` (`v01` to `v76`)

Each `vNN` is an ensemble of saved runs with one decoding setting (prior correction tau, transition weight beta).
The range includes both genuine ensembles and ensembles that use the additional IL-TUR `rr` documents; the rows
below are the documented ones.

| tag | members | test |
|---|---|---:|
| `v07` | ens_final_hyb2 + rslora_M1, beta 0.25 | 0.5631 |
| `v30` | lora_M4, rslora_M1, kd_M4half, kd2_M4half, lorma_M3, ext_M4, lex_tfidf, beta 0.5 | 0.5653 |
| `v53` | lora_M4_lsegall, lora_M4_lseg, lora_M4, rslora_M1, kd_M4half, lorma_M3, ext_M4, lex_tfidf, beta 0.25 | 0.5820 |
| `v54` | same 8 members, beta 0.5 | 0.5845 |
| `v46` | 3-member group G3, beta 0.5 | **0.5879** (best without the additional documents) |
| `v55` | lora_M4_lsegall_it5 + TF-IDF trained the same way, with the additional documents | 0.6681 |
| `v68` | lora_M4_lsegall_it5 x2 + TF-IDF trained the same way, with the additional documents | **0.6707** (graded entry) |
| all other `vNN` | the same members under other decoding settings or member subsets | – |

## Runs with additional IL-TUR `rr` documents in training

Kept separate from the main results (see `EXPERIMENTS.md` §2.2).

| tag | what | test |
|---|---|---:|
| `lora_M4_lsegall_it5` | full-LegalSeg trunk, LoRA M4, + 5 additional IT documents | 0.6270 |
| `lex_it_5fold`, `lex_il10_5fold`, `lora_M4_lseg_il10` | TF-IDF / LoRA M4 with 5 or 10 additional documents | – |
| `lex_k2_5fold`, `lex_k6_5fold`, `lex_k8_5fold`, `nn_k6_1fold`, `nn_k8_1fold` | built for a study of how scores change with the number of added documents; not submitted | – |
