# CS-781 Phase 5 — Legal Statute Identification, experiment log

Nithin T M · 230709 · `n_230709` · window 2026-10-22 to 2026-10-28 23:59 IST (data public from 2026-10-15) · **final: rank 5, test 0.4632**

Primary metric: Macro-F1 over statute IDs 0..n-1, where n = max gold ID + 1 in the private test reference
(`evaluate_phase5.py`, sklearn `zero_division=0`). Secondary: Micro-F1, subset accuracy.

---

## 0. Before any modelling (2026-10-15)

### 0.1 Provenance: the data is IL-TUR `lsi`, whose test labels are public

| file | competition | IL-TUR `lsi` parquet | ratio |
|---|---|---|---|
| train | 299,396,974 B | 145,199,662 B | 2.06 |
| dev | 79,377,066 B | 38,615,535 B | 2.06 |
| test | 98,061,196 B | 48,307,271 B | 2.03 (no labels in ours) |

The constant ratio is the same fingerprint Phase 3 had with IL-TUR `rr`. The IL-TUR card (read; no data files
downloaded) gives 42,750 / 10,181 / 13,019 documents, **100 IPC sections** as labels, and says the text is
**the facts only**, cut from Indian Kanoon Supreme Court and High Court judgments (Paul et al., ILSI / LeSICiN).

Consequences:

- **IL-TUR `lsi/test` carries the gold labels. It is never downloaded in this project** and no loader for it
  exists. IL-TUR's train/dev are our train/dev, so it offers nothing legitimate.
- **The label space is 100 statutes.** A predicted ID above the private reference's maximum makes the official
  scorer reject the whole file, so no zip contains an ID above the train+dev maximum (`--max-label`).

### 0.2 Integrity scan: facts vs the Phase-1 pretraining corpus

The Phase-1 corpus is Supreme Court and High Court judgments. If a test case's facts sit inside a full judgment in
that corpus, the rest of the judgment usually names the sections ("convicted under Section 302 read with Section 34
IPC"). `code/phase5_external.py scan` measures this (match rate per split, and the dev Macro-F1 of "labels = IPC sections
cited in the matched judgment"). **It is a measurement. Nothing from it is submitted without Nithin's explicit
go-ahead.** Results: §0.4 (21 % of test found; not an answer key; not used).

### 0.2b External data (allowed if described — professor, Discord)

A Hub search (2026-10-16) found no other labelled data for this task worth using. AILA-2019 has 50 queries; the
"Indian Penal Code" datasets are section texts. The large source is the **Phase-1 corpus itself (SC + HC
judgments) used as distant supervision**, which is how ILSI was built. `code/phase5_external.py`:

1. `scan`: every judgment citing a target section number. Cites are tagged by the act named after them
   (IPC/CrPC/other/none), because several label numbers (156, 161, 164, 173, 438, 482) are CrPC provisions in practice.
2. Judgments containing a dev or test document's facts (≥ 2 of its 12-word needles) are **never written**. For test,
   only the count is kept. Judgments containing a train document are the labeller's validation set.
3. `finalize`: the cite → label-ID map is learned on those train copies. Per statute, positives are used only where
   the labeller's precision ≥ 0.6, and absence counts as a negative only where its recall ≥ 0.6. All other
   labels are masked from the loss. Citing sentences are deleted from the silver text so the model cannot read
   section numbers the facts-only test text does not show.
4. The value is measured on CPU first: `lex_w12` (gold) vs `lex_w12_sil` (gold + 40k silver, weight 0.5) on dev.
   The GPU queue gets `SILVER=40000` only if the second wins.

The same scan reports the integrity number (§0.2): the fraction of dev/test documents whose full judgment is in the
corpus, and the dev Macro-F1 of reading labels off that judgment's cites.

### 0.3 Protocol

- Models train on `train.json`; `dev.json` is used for thresholds, ensemble weights and model selection.
- Every run saves dev and test logits (`results/<tag>/`); thresholds and ensembles are fit on CPU afterwards.
- Thresholds: one global threshold vs per-statute thresholds shrunk toward it
  (`t_l = (n_l t_l* + s t_glob) / (n_l + s)`). The two are compared by **2-fold CV over the dev documents**
  (tune on one half, score the other, both ways), because tuning and scoring on the same dev rewards overfitting,
  and per-statute thresholds overfit rare statutes most. Per-label must beat global by 0.002 CV to be used.
  Every document gets at least one statute (no training document has none).
- Model-only and hybrid (anything containing the TF-IDF model) are always reported separately.

### 0.4 Data gate (Colab, 2026-10-16)

- 42,750 train (42,151 unique texts), 10,181 dev records (10,098 unique), 13,019 test; 100 statutes, all present in
  train and dev, **every statute has >= 50 train documents**.
- **dev.json holds 128 labelled copies of test documents.** They are dropped from dev everywhere (`phase5_data`), so no
  threshold or weight ever sees test labels. To be reported to the TAs.
- BPE tokens per document: median 637, p90 4,125, max cut 8,192 (head 3/4 + tail 1/4); 64.3M train tokens.
- **The facts are masked**: `<ENTITY>` (125k in 2k docs), `<SECTION>` (11k), `<ACT>` (11k). Only 2.9–4.1 % of documents
  keep a raw section number.
- Exact duplicates: 105 dev and 230 test documents copy a train document (dev label agreement 0.74).
- Integrity scan (Phase-1 SC + HC corpus): the judgment of 21.4 % of test and 13.5 % of dev documents is in the corpus;
  reading labels off its citations scores Macro-F1 0.237 on the matched dev documents (0.068 over all dev). Not an
  answer key, and not used.

---

## 1. Results (dev, 10,053 documents; CV = 2-fold dev cross-validation of the threshold choice)

L4, 9.1–9.8k training tokens/s; required runs 25 min training (~0.56 epoch), 2 x 512-token chunks per document,
label-wise attention head, inference over up to 6 x 512 tokens, ~23 min per run for dev + test.

| run | technique | trunk | dev Macro-F1 (global thr) | dev Macro-F1 (chosen thr) | CV | type |
|---|---|---|---|---|---|---|
| `lex_w12` | TF-IDF word 1-2g + one-vs-rest LR | — | 0.403 | 0.418 (per-label, s=25) | 0.408 | hybrid |
| `lex_w12_sil` | + 17,164 silver docs (weight 0.5) | — | — | 0.422 | 0.408 | hybrid |
| `knn_w12` | kNN over train TF-IDF (k=25, cos^4) | — | — | 0.365 | 0.353 | hybrid |
| `m4_lorma` | LoRMA r16 (compulsory) | M4 | 0.336 | 0.363 | 0.347 | model |
| `m1_adapter` | Houlsby adapters, 64 | M1 | 0.353 | 0.374 | 0.363 | model |
| `m2_partial` | top 5/10 blocks + LLRD 0.85 | M2 | 0.330 | 0.349 | 0.337 | model |
| `m3_qlora` | NF4 trunk + LoRA r16 | M3 | 0.350 | 0.370 | 0.359 | model |
| `m1_prefix` | prefix tuning, 32 | M1 | — | 0.351 | 0.338 | model |
| `kd_m4half` | half-depth M4, full FT, KD from ensemble teacher (0.384) on dev+test text | M4/2 | 0.338 | 0.357 | 0.344 | model |
| `top_M1_adapter` | adapters, 90 min (~1.5 epochs) | M1 | 0.377 | 0.394 | 0.382 | model |
| `ens_model` | greedy prob. average of neural runs | — | — | 0.407 | 0.395 | model |
| `ens_hyb` | neural + TF-IDF, prob. average | — | — | 0.425 | 0.424 | hybrid |
| `ens_hyb_z` | neural + TF-IDF, z-scored logits | — | — | **0.443** | **0.425** | hybrid |

Observations: after ~0.56 epoch the four architectures are within 0.03 of each other; 3x the training time on the best
one (M1 adapters) gave +0.02 CV. Per-label thresholds help every model (+0.02–0.03) and want heavy shrinkage (s=25).
The z-scored ensemble beats probability averaging on dev but not on CV (0.4245 vs 0.4239): within noise.

### 1b. Members added in rounds 2-5 (dev numbers from the run logs)

| run | what | dev Macro-F1 | type |
|---|---|---|---|
| `top_M3_qlora` | QLoRA, M3, longer budget (Codabench test 0.3673) | — | model |
| `lex_w12r` | `lex_w12` with test scores from a train+dev refit (excluded from the stacker: dev and test from different models) | — | hybrid |
| `lex_char` | TF-IDF char_wb 3-5 grams | — | hybrid |
| `top2_M1_adapter` | adapters, M1, 2.65 epochs (150 min) | 0.382 (global thr) | model |
| `top2_M3_qlora` | QLoRA, M3, 120 min (resumed after a runtime loss) | 0.385 (global thr) | model |
| `lex_w12x` | `lex_w12` **cross-fitted on dev**: each dev doc scored by a model fit on train + the other 3 dev folds, test by train + all dev | 0.444 (per-label s=25) | hybrid |
| `lex_w13x` | same, word 1-3 grams, 800k features | 0.439 (per-label s=25) | hybrid |

Longer training did not make single neural models stronger (0.382 after 2.65 epochs vs 0.394 dev for
`top_M1_adapter` after ~1.5); cross-fitting the TF-IDF model on dev labels gave +0.026 dev.

## 2. Refuted hypotheses

| hypothesis | number that killed it |
|---|---|
| Silver labels from the Phase-1 judgments help rare statutes | labeller P 0.58 / R 0.33 on corpus copies of train docs; TF-IDF CV 0.4083 gold vs 0.4078 gold+silver |
| A citation rule reads statutes off the facts | facts are masked; rule fires on 2.6 % of dev; gain +0.0012 |
| Near-duplicate cases give a large kNN gain | only 4.2 % of dev has a train neighbour at cos >= 0.9 (70 % same label set); kNN alone CV 0.353 |
| The corpus holds an answer key for test | cite-lookup Macro-F1 0.237 on the 13.5 % of dev it finds |
| A local dev score near 0.7 is reachable honestly | same TF-IDF + LR (hashed 1-2 grams), scored on dev: trained on train only **0.399**; trained on train + dev **0.981** (2026-10-16, local CPU). Dev scores near 0.7+ come from evaluating on documents the model trained on. |

## 3. Final day (2026-10-28): stacking + cross-fitted members, Codabench test scores

Test = Codabench Macro-F1 on the 13,019 hidden test documents. CV = 2-fold dev CV of the threshold choice on
out-of-fold stacker scores (honest: no dev label ever scores its own document).

| step | what changed | dev CV | test |
|---|---|---|---|
| `ens_hyb` | best plain ensemble (round 2), greedy prob. average | 0.424 | 0.4387 |
| `stack_r2` | `phase5_stack.py`: HistGB over (doc, top-25 candidate statute) rows; member z-scores and ranks, agreement, statute frequency, co-occurrence support P(j\|k)·p(k); 4-fold OOF on dev; test = mean of fold models | 0.4369 | 0.4453 |
| `stack` / `stack_mix` | + `top2_M1_adapter` (2.65 epochs); greedy logit mix of 8 stacks | 0.4417 / 0.4433 | — / 0.4511 |
| `s5a` | + `lex_w12x`: TF-IDF LR **cross-fitted on dev** (dev doc scored by a model fit on train + the other 3 dev folds; test by train + all dev), + `top2_M3_qlora` | 0.4543 | 0.4570 |
| `s6_bag` | + `lex_w13x` (word 1-3 grams, cross-fitted); mean of 8 stacker seeds | **0.4591** | **0.4622** |
| best single upload | `s5_s1` (stacker seed 881 of the `s5a` configuration) | — | **0.4632*** |

\* The 0.4632 is the best of 13 seeds of two stacker configurations; their test spread was
0.4551–0.4632 (s5: 0.4570, 0.4632, 0.4579, 0.4570; s6: 0.4624, 0.4574, 0.4587, 0.4624, 0.4551, 0.4581, 0.4553, 0.4616).
The honest method number is the 8-seed bag, **0.4622**; 0.4632 is leaderboard selection over seeds.

What moved test, in order: stacking over members (+0.007 over `ens_hyb`), letting the TF-IDF members learn from dev
via cross-fitting (+0.006; `lex_w12x` dev 0.444 vs `lex_w12` 0.418), bagging 8 seeds with more members (+0.005).
What did not: test volume (`s5a` at x0.90/0.95/1.00/1.05/1.10 of its statutes per doc: 0.4540/0.4562/0.4570/0.4570/0.4569),
class balancing in the stacker (`stack_bal` 0.4484), removing co-occurrence features (`stack_nocooc` 0.4488 vs
`stack_k35` 0.4506: within seed noise), a 4-seed bag of near-identical stacks (`s5_bag` 0.4567).

Not tried for lack of time: cross-fitting the neural members on dev (2+ GPU trainings per member), `lex_charx`
(char n-gram cross-fit: >2 h on 12 CPUs, killed by runtime recycling). Colab recycled background (setsid) jobs four
times on the last day; foreground `!bash script.sh` with skip-if-done steps fixed it.

**Final leaderboard (after the deadline):** 1st 0.4727, 2nd 0.4717, 3rd 0.4651, 4th 0.4635, **5th n_230709 0.4632**,
6th 0.4624. Every uploaded zip and its score: `submissions/SCORES.md`.
