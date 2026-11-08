# Phase 5 submissions — Codabench test Macro-F1

Every zip holds one `prediction.json` at its root (upload as is). Scores are from the Codabench submissions page,
2026-10-28. "—" = built, never uploaded. Final leaderboard: **rank 5, 0.4632** (`s5_s1`).

## 01_required_models — single fine-tuned Phase-1 models (the graded minimums)

| zip | method | test |
|---|---|---|
| `submission_m4_lorma.zip` | LoRMA, M4 (compulsory) | 0.3479 |
| `submission_kd_m4half.zip` | KD half-depth M4 student (compulsory) | 0.3464 |
| `submission_m1_adapter.zip` | adapters, M1 | 0.3610 |
| `submission_m1_prefix.zip` | prefix tuning, M1 | 0.3349 |
| `submission_m2_partial.zip` | partial FT + LLRD, M2 | 0.3368 |
| `submission_m3_qlora.zip` | QLoRA, M3 | 0.3556 |
| `submission_top_M1_adapter.zip` | adapters, M1, 3x training | 0.3830 |
| `submission_top_M3_qlora.zip` | QLoRA, M3, longer | 0.3673 |

## 02_tfidf — TF-IDF + one-vs-rest LR (hybrid)

| zip | test |
|---|---|
| `submission_lex_global.zip` | 0.4098 |
| `submission_lex_perlabel.zip` | 0.4134 |
| `submission_lex_silver.zip` | 0.4143 |

## 03_ensembles — greedy averages (`v85`/`v115` = 0.85x / 1.15x statutes per doc)

| zip | test |
|---|---|
| `submission_ens_model.zip` (neural only) | 0.3949 |
| `submission_ens_model_v85.zip` | 0.3895 |
| `submission_ens_model_v115.zip` | 0.3954 |
| `submission_ens_hyb.zip` (neural + TF-IDF) | 0.4387 |
| `submission_ens_hyb_z.zip` | 0.4321 |
| `submission_ens_hyb_z_v85.zip` | 0.4286 |
| `submission_ens_hyb_z_v115.zip` | 0.4308 |

## 04_stack_v1 — first stacker (`phase5_stack.py`, train-only members)

| zip | test |
|---|---|
| `submission_stack_r2.zip` | 0.4453 |
| `submission_stack_s1.zip` | 0.4490 |
| `submission_stack_s2.zip` | — |
| `submission_stack_s3.zip` | 0.4475 |
| `submission_stack_bal.zip` | 0.4484 |
| `submission_stack_nocooc.zip` | 0.4488 |
| `submission_stack_k35.zip` | 0.4506 |
| `submission_stack_mix.zip` | 0.4511 |
| `submission_stack_v110.zip` | 0.4487 |
| `submission_stack_bag.zip` | — |
| `submission_stack_bag_v95.zip` | 0.4484 |
| `submission_stack_bag_v105.zip` | — |
| `submission_stack_bag_v110.zip` | 0.4506 |

## 05_stack_s5 — stacker + dev-cross-fitted TF-IDF (`lex_w12x`)

| zip | test |
|---|---|
| `submission_s5a.zip` (predictions identical to `s5`, which also scored 0.4570) | 0.4570 |
| `submission_s5a_v90.zip` / `v95` / `v105` / `v110` | 0.4540 / 0.4562 / 0.4570 / 0.4569 |
| `submission_s5_s1.zip` | **0.4632** (best, final leaderboard entry) |
| `submission_s5_s2.zip` | 0.4579 |
| `submission_s5_s3.zip` | 0.4570 |
| `submission_s5_bag.zip` | 0.4567 |
| `submission_s5_mix.zip` | 0.4576 |

## 06_stack_s6_final — + `lex_w13x`, 8 seeds

| zip | test |
|---|---|
| `submission_s6_s0.zip` … `s6_s7.zip` | 0.4624, 0.4574, 0.4587, 0.4624, 0.4551, 0.4581, 0.4553, 0.4616 |
| `submission_s6_bag.zip` (mean of 8 seeds; honest method number) | 0.4622 |
| `submission_s6_mix.zip` | 0.4612 |

## 99_unsubmitted_round1

`01_ens_hyb_z.zip`, `02_ens_model.zip`, `10_ens_hyb.zip`: round-1 builds of the ensembles (Oct 26), superseded by
the Oct 28 rebuilds in `03_ensembles`; never uploaded. `UPLOAD_ORDER.txt` is that round's plan.
