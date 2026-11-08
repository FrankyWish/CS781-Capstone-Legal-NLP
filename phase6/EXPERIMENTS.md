# Phase 6 — experiments

**Result: rank 1 at BERTScore F1 0.8606** (`nsel4_best`, extractive, driven by four fine-tuned Phase-1
trunks). The best abstractive (model-only) score is 0.8391. Both numbers go in the report.

Chronology, every number as measured. Dev = our own 600-document held-out split of
`train` (`results/split.json`), with answer-key records removed. Test = Codabench.
Phase 5's dev CV predicted Codabench within about ±0.005; rank by held-out score,
never by in-sample fit. In this phase, held-out-to-test gaps were: neural runs -0.0004 to -0.0033;
`bsel_best` 0; the winning `nsel_*` gains reached test at about two thirds of their held-out size.

`results/` and `logs/` live on Drive (`cs781_capstone/phase6/`); they were never copied to this laptop.
Every number below was read from those files or from Codabench at the time.

ROUGE here equals the official scorer's to the last digit **only when `rouge_score`
is installed** (`gate.json: rouge_exact`). Every table says which.

## 0. Timeline

| when | what |
|---|---|
| 2026-10-29 | Phase identified as abstractive summarization on BERTScore F1; design written; stage 1 (CPU: gate, sentence scorer, extractive baselines, length calibration, input views) built and smoke-tested on fake data |
| 2026-11-01 | Real run started on the L4 at 14:53 IST |
| 2026-11-02 02:30 IST | Leaderboard around 0.86. Final-day driver `p6f.sh` built: BERTScore-labelled selection, sentence-level consensus, window-trained neural members, and a numbered 40+ zip batch |
| 2026-11-02 06:56 IST | `p6f.sh` finished: 60 numbered zips, no required run missing, best held-out `bsel_best` 0.8549, best model-only `rank_model` 0.8425 |
| 2026-11-02 09:10 IST | Last refit `p6g.sh` built (bsel on all fit documents plus a word-level score); smoke-tested on fake data only |
| 2026-11-02 11:22 IST | `p6g.sh` finished on the real data: `bsel2_f` held-out 0.8574, the new best. Fit took 69 min. Text score out-of-fold correlation with the label: 0.6882 (f), 0.2096 (g) |
| 2026-11-02 14:02 IST | First test scores: best 0.8547 (`bsel_best_x110`). Held-out to test: neural runs -0.0004 to -0.0033, mean -0.0023; `bsel_best` unchanged at 0.8546; `bsel2_f` -0.0035 |
| 2026-11-02 21:28 IST | 32 of 160 submissions used. Best test still 0.8547 (`bsel_best_x110`); best model-only test 0.8391 (`rank_model`, `mbr_model`) |
| 2026-11-02 21:43 IST | Last build `p6h.sh`: M1 and M2's encoder fine-tuned to score sentences, target test above 0.86. Smoke-tested on fake data only |
| 2026-11-03 01:37 IST | `p6h.sh` finished (started 22:03 IST): labels 74 min, M1 42 train-min, M2 37, sweep 14 min. Control rebuilt `bsel_best` on 600/600 dev docs. Best held-out `nsel_stack` 0.8624, +0.0075 over `bsel_best` on the same 300 docs. 7 zips in `submissions/_final2` |
| 2026-11-03 13:50 IST | `_final2` test scores: `nsel_stack` = `nsel_best` 0.8596 (new best, +0.0049), `nsel_nn` 0.8593 from the Phase-1 trunks alone. 39 of 160 submissions used. Weekly Claude usage 64%, past the 50% stop rule, so no further build without his override |
| 2026-11-03 14:15 IST | He overrode the usage rule. Last build `p6i.sh`: M3 and M4 as two more scorers, four-trunk sweep with each trunk's score fed to the stack separately, paired gate against `nsel_stack`. Smoke-tested on fake data only |
| 2026-11-03 16:19 IST | `p6i.sh` finished (started 14:16 IST). M3 0.8197 and M4 0.8245 correlation with y_f, the same as M1 and M2. Control 600/600. Against `nsel_stack` on the held-out 300: `nsel4_nn` +0.0001 ± 0.0004, `nsel4_stack` -0.0003, mix75 -0.0009. Four trunks tie two |
| 2026-11-03 18:19 IST | Test: `nsel4_stack` 0.8606, `nsel4_nn` 0.8604, first scores above 0.86. Pushed `nsel4_stack`. 42 of 160 used. Phase 6 modelling closed |
| 2026-11-03 22:50 IST | Report-coverage uploads (21:05–22:50): `nsel4_best` 0.8606 (pushed at 21:06, ties `nsel4_stack`), mix75 0.8599, mix50 0.8592, mix25 0.8578, bsel3 0.8529; `mbr_all` 0.8379, long 0.8374, short 0.8382; `win_best` 0.8410; `m1_win` 0.8370, `m2_win` 0.8249, `m3_win` 0.8348, `m4_win` 0.8317; `sel_short` 0.8388, `sel_long` 0.8374. 57 of 160 used |
| 2026-11-04 00:18 IST | **Phase 6 closed: rank 1 of the leaderboard at 0.8606** (ROUGE-1 0.5801, ROUGE-L 0.2961). Second 0.8570, so the lead is +0.0036, about one test SE |
| 2026-11-08 | Zips moved from Downloads into `submissions/<family>/<tag>.zip`, duplicates removed, catalogue in `submissions/README.md` |

## 1. The gate — what the data is

Measured 2026-11-01. **BERTScore saturates by 450 words** (reference prefix vs whole: 350w 0.981, 450w 1.000, while ROUGE-1 keeps rising, 0.609 at 450w, 0.892 at 900w), and the median reference is 642 words, so most documents are scored only on the first ~420 words of each side. This is why `phase6_win.py` exists.

| measurement | value | why it matters |
|---|---|---|
| train / test documents | 7,028 (2 empty dropped) / 100, = IN-Abs | |
| document words p10/p50/p90 | 1,358 / 3,147 / 7,902 | how far past the 1024–1280-token window we are |
| summary words p10/p50/p90 | 293 / 642 / 1,505 | the length target; F1 is symmetric so this is the first dial |
| length model vs constant (median abs rel err) | 0.324 vs 0.398, r 0.67: per-document target wins | per-document target only if it beats the constant |
| 8-gram copy rate of summary in document | p50 0.323, mean 0.324; 232/1500 docs above 0.5 | if high, the phase is extraction, not generation |
| novel unigrams / bigrams | 0.086 / 0.262 | how abstractive the references really are |
| train records copying a test document | 0 (28 internal train duplicates) | answer keys; excluded from fit and dev |
| extractive oracle ROUGE-1 | 0.7715 (n=60) | the ceiling any selection system can reach |
| lead baseline ROUGE-1 | 0.5848 (n=60) | the floor every neural run must clear |

## 2. Stage 1 — selection and length

Scored on the shared 200 decoded dev documents. Lead-k and TF-IDF centroid numbers are in `results/baselines_dev.json` on Drive; they were never copied here, so those rows stay empty rather than guessed.

| system | dev BERTScore | dev ROUGE-1 | dev ROUGE-L | words | note |
|---|---|---|---|---|---|
| lead-k at the calibrated length | not copied | | | | `baselines_dev.json` on Drive |
| TF-IDF centroid | not copied | | | | `baselines_dev.json` on Drive |
| supervised sentence scorer (`sel_best`) | 0.8426 | 0.5732 | — | 892 | test 0.8379 |
| `sel_short` / `sel_long` | 0.8423 / 0.8422 | 0.5640 / 0.5734 | | 763 / 1020 | ±15% length: BERTScore barely moves, because the window has saturated. Test 0.8388 / 0.8374 |

Length sweep: `results/length_calib.json`.

## 3. Stage 2 — the required fine-tuning runs

Dev = the 200 decoded held-out documents. All means about 800 words.

Trainable-parameter counts are printed in each run's log on Drive and were not copied here.

| tag | technique | arch | view | dev BERTScore | dev ROUGE-1 | test | note |
|---|---|---|---|---|---|---|---|
| `m4_lorma` | LoRMA | M4 | sel1024 | 0.8347 | 0.4854 | 0.8343 | compulsory |
| `m2_lora` | LoRA | M2 | sel1024 | 0.8268 | 0.4452 | 0.8250 | |
| `m1_houlsby` | Houlsby | M1 | sel1024 | 0.8394 | 0.4951 | 0.8364 | best required run, KD teacher |
| `m3_qlora` | QLoRA | M3 | sel1024 | 0.8366 | 0.4840 | 0.8344 | |
| `m3_pfeiffer` | Pfeiffer | M3 | sel1024 | 0.8380 | 0.4885 | 0.8350 | |
| `m2_adalora` | AdaLoRA | M2 | sel1024 | 0.8197 | 0.4265 | 0.8164 | |
| `kd_m1half` | distillation | M1/2 | sel1024 | 0.8306 | 0.4518 | 0.8277 | compulsory; 0.0088 below its teacher on dev, 0.0087 on test (`m1_houlsby` 0.8364) |
| `m4_full` | full fine-tuning | M4 | sel1024 | 0.8345 | 0.4762 | 0.8343 | LoRMA matches full fine-tuning here (0.8347), and on test (both 0.8343) |

Long-document axis, one technique held fixed:

| view | dev BERTScore | test | note |
|---|---|---|---|
| `m4_lorma_trunc` raw document, head 70% + tail 30% | 0.8403 | 0.8390 | beat the selected view at the same prompt size, ROUGE-1 equal (0.4854); selection did not help the neural model |
| `m4_lorma` selected sentences, 1152-token prompt | 0.8347 | 0.8343 | |
| `m4_lorma_2048` selected sentences, 2048-token prompt | 0.8401 | 0.8374 | best ROUGE-1 of the neural runs (0.5086) |

## 4. Stage 3 — MBR, reranker, fusion

| system | kind | dev BERTScore | dev ROUGE-1 | note |
|---|---|---|---|---|
| `rank_model` | model | 0.8425 | 0.5067 | OOF; +0.0022 over the best model. **Best model-only system; test 0.8391**, level with `m4_lorma_trunc` (0.8390), so the +0.0022 did not transfer |
| `mbr_model` | model | 0.8415 | 0.5054 | ROUGE utility, +0.0012. Test 0.8391 |
| `rank_all` | hybrid | 0.8541 | 0.5754 | OOF; before `bsel` it gained +0.0012, after it -0.0005. **Test 0.8528** |
| `mbr_all` | hybrid | 0.8426 | 0.5650 | picked `sel_best` on 188 of 200 documents, `bsel_best` on 1: the ROUGE utility cannot see what `bsel` adds. Test 0.8379; ±15% length 0.8382 / 0.8374 |
| `mbr_bert` | hybrid | 0.8425 | 0.5641 | roberta-large at test time, see §6; never uploaded |
| per-document oracle over all systems | | 0.8611 | | ceiling for choosing between these candidates |

## 4b. Final-day levers

Gains are on the 300-document half the sweep never saw, except `cons_*` and `stitch_*`,
which come from the 200 shared documents. The test number comes from Codabench.

| lever | hypothesis | held-out gain | test |
|---|---|---|---|
| `win_best` | a scorer supervised on the reference's first 420 words beats whole-summary selection | -0.0004 vs `sel_best` (0.8421 vs 0.8425) | 0.8410 vs `sel_best` 0.8379: +0.0031 on test after -0.0004 on dev, about one test SE |
| `bsel_best` | labels from the metric's own token tables beat ROUGE labels | **+0.0124** vs `sel_best` (0.8549 vs 0.8425), ROUGE-1 also up, 0.5886 vs 0.5738 | **0.8546**; x1.1 0.8547 (best until `nsel`), x1.25 0.8543, x0.9 0.8543, x0.8 0.8542 |
| `cons_*` | sentence-level consensus beats choosing one whole candidate (`mbr_all`) | none above `bsel_best`: the held-out ranking in `SUBMIT_ORDER.md` put every `cons_*` below it | never uploaded: roberta-large at test time (§6) |
| `stitch_*` | a neural opening plus extraction beats either alone | none above `bsel_best` on the 200 shared dev documents | 25% from the model: trunc 0.8518, m1 0.8502, 2048 0.8497. 50%: trunc 0.8471, m1 0.8458, 2048 0.8449. All below `bsel_best`, and worse as the neural share grows |
| `m*_win` | training only on what BERTScore reads beats training on the whole summary | m1 -0.0006, m2 +0.0001, m3 +0.0004, m4 -0.0016 against the same technique on the whole summary | m1 0.8370 (houlsby 0.8364), m2 0.8249 (lora 0.8250), m3 0.8348 (qlora 0.8344), m4 0.8317 (lorma 0.8343): the same null as dev, M4 worse on both |
| `bsel2_f` | all fit documents (6,416, was 1,500) and a word-level score beat `bsel_best` | **+0.0025** (0.8574 vs 0.8549), ROUGE-1 0.5926 vs 0.5886, n=300. Both changes were made together, so the gain cannot be split between them | **0.8539: below `bsel_best` by 0.0007, so the held-out gain did not reach test** |
| `bsel2_g` | the same, on the greedy-oracle label | -0.0019 (0.8531 vs 0.8549), ROUGE-1 0.5875 vs 0.5886, n=300 | not uploaded: failed the gate |
| `nsel_*` (`p6h.sh`) | a fine-tuned Phase-1 trunk sees what 27 hand features cannot. M1 and M2's encoder, top 4 blocks, blended and stacked with `bsel`; window 360/390 and citation removal tested on the best head | **`nsel_stack` +0.0075 ± 0.0007 (0.8624 vs 0.8549)**, `nsel_best` +0.0075, `nsel_nn` (the two trunks alone, no `bsel`) +0.0074 ± 0.0007, mix75 +0.0068, mix50 +0.0054, mix25 +0.0029; ROUGE-1 up too (0.5966 vs 0.5886); n=300, paired SE. Per-document correlation with y_f on dev: M1 0.8201, M2 0.8252, `bsel_f` 0.7436 | **`nsel_stack` 0.8596 = `nsel_best` 0.8596 (+0.0049 over 0.8547)**, mix75 0.8595, `nsel_nn` 0.8593, mix50 0.8591, mix25 0.8578. Held-out to test: -0.0028 on the winner, so about two thirds of the dev gain reached test; the order is kept |
| `nsel4_*` (`p6i.sh`) | M3 and M4 as two more scorers add what M1 and M2 miss; the stack weighs each trunk | **none: `nsel4_nn` +0.0001 ± 0.0004 against `nsel_stack` (0.8625 vs 0.8624)**, `nsel4_stack` -0.0003 ± 0.0004, mix75 -0.0009, mix50 -0.0024. Each trunk alone correlates 0.82 with y_f (M3 0.8197, M4 0.8245), so the four trunks make the same errors: same labels, same data, same Phase-1 corpus | **`nsel4_stack` 0.8606, `nsel4_best` 0.8606 (the final pushed entry), `nsel4_nn` 0.8604**, mix75 0.8599, mix50 0.8592, mix25 0.8578 (Phase-1 trunks only): +0.0010 and +0.0011 over their two-trunk versions on test, against +0.0001 and -0.0003 on held-out dev. Test SE on a difference is about 0.001, so part of this is chosen-on-test luck |
| `bsel3` | all 6,416 fit documents without the word-level score | +0.0003 ± 0.0005: more documents add nothing, so `bsel2`'s held-out +0.0025 came from the word-level score, which did not reach test | 0.8529, below `bsel_best` |

`bsel_best` length, held-out: x1.1 and x1.25 both give 0.8548 against 0.8546 at x1.0; on test every length from x0.8 to x1.25 landed within 0.0005. Past the window
these are the same system on the primary metric. Surrogate greedy-oracle F on the training
documents is 0.8853, so there is about 0.03 between the scorer and that ceiling.

## 5. Refuted hypotheses

*Kept deliberately — a refuted lever saves the next phase a day.*

| hypothesis | test | result |
|---|---|---|
| Aiming a ROUGE-supervised scorer at the window helps | `win_best` vs `sel_best`, held-out half | No, -0.0004. The labels were the problem, not the window |
| Training neural models on only the first ~430 summary words helps | `m*_win` vs the matching whole-summary run | No, within ±0.0016 in both directions. ROUGE-1 fell at about 540 words, BERTScore stayed flat |
| Hybrids beat the best extraction | `mbr_*`, `rank_all`, `cons_*`, `stitch_*` vs `bsel_best` | No. Before this run I predicted hybrid 0.855-0.87; the best hybrid was 0.8541 |
| Selected input sentences help the neural model | `m4_lorma` vs `m4_lorma_trunc` | No, 0.8347 vs 0.8403 at the same prompt size |
| A word-level score and all fit documents improve `bsel` | `bsel2_f` vs `bsel_best`, test; `bsel3` held-out | Not on test: 0.8539 vs 0.8546, after +0.0025 held-out. `bsel3` (all documents, no word-level score) is +0.0003 ± 0.0005, so the held-out gain was the word-level score, and it did not transfer |
| A neural opening improves extraction | `stitch_*` vs `bsel_best`, test | No. 0.8518 at best, and 0.8449-0.8471 at a 50% share |
| Generation-only reaches 0.81-0.83 | the neural runs | Prediction too low: individual runs 0.8197-0.8403 on dev, 0.8164-0.8390 on test |
| Four trunks beat two as sentence scorers | `nsel4_*` vs `nsel_stack`, held-out half | No on dev: +0.0001 ± 0.0004. Test showed +0.0010, about one SE, so it can't be claimed as a finding. M3 and M4 alone correlate 0.82 with the label, like M1 and M2: they make the same errors |
| A ROUGE-utility MBR over all candidates beats its best member | `mbr_all` vs `sel_best` | No: it picked `sel_best` on 188 of 200 documents. Test 0.8379 vs 0.8379 |

## 6. Integrity log

| risk | measured | decision |
|---|---|---|
| public test references (IL-TUR `summ` / IN-Abs ships test summaries) | — | never downloaded |
| train records that duplicate a test document | `results/answer_keys.json` | dropped from fit and dev |
| reference summary present inside the input document | 8-gram copy rate p50 0.323 (§1) | partly: extraction is a strong floor but not the answer |
| Phase-1 corpus containing the test documents with their headnotes | **100/100 test documents have a corpus copy; 99.2% of train copies add summary text beyond the document, so about 99 test documents could be answered by lookup.** The test figure is inferred from train; no test copy was read or stored | go-ahead never given, so no lookup system was built or submitted |
| roberta-large as the MBR utility | — | primary MBR variant uses a ROUGE-only utility; `mbr_bert` runs roberta on test documents, so do not push it |
| roberta-large in the `bsel` labels and `cons_*` selection | `bsel`: labels only, nothing neural at test time. `cons_*`: in the decision rule at test time. Spec §12.2 says "You must not use another pretrained language model" | `cons_*` never uploaded. `bsel_*` and `nsel_*` use roberta-large only to label train and dev sentences, so nothing but Phase-1 trunks and HistGB runs at test time. Both stay labelled extractive: it is the official metric run on training data, but it does not start from a Phase-1 model, so it never counts toward the 6 required experiments |
| leaderboard probing with 40+ submissions | the best of many test scores is selection-optimistic | 57 of 160 used. Report held-out numbers beside any test number; model-only best stated separately; say the final pick was made on test among systems tied on dev |

## Final standing (for the report)

Rank 1 of the Phase-6 leaderboard at BERTScore F1 0.8606 (`nsel4_best`, ID 1030632), screenshot in
`leaderboard_final.png`. Second is 0.8570, so the lead is +0.0036, about one test SE.

**The lead is on the primary metric only.** On ROUGE-1 we have 0.5801, against 0.5819 for rank 3 and 0.6001
for rank 5. On ROUGE-L we have 0.2961, against 0.3231 for rank 5 and 0.3234 for rank 7. This is the
trade-off we chose: length and sentence choice were tuned for what BERTScore reads (the first 512
roberta tokens of each side, which is about 420 words). Past that window, extra text neither helps nor
hurts BERTScore, but it does help ROUGE, because ROUGE rewards covering the whole reference. A system
tuned for ROUGE would write longer summaries and give up some BERTScore.

**The winning system is extractive.** It chooses sentences from the test document using fine-tuned Phase-1
trunks; it does not generate text. The best abstractive (model-only) test score is 0.8391 (`rank_model`,
`mbr_model`), and the best single generator is 0.8390 (`m4_lorma_trunc`).
