# CS-781 Phase 3 — Rhetorical Role Prediction, experiment log

Nithin T M · 230709 · `n_230709` · window 2026-10-12 to 2026-10-16

Macro-F1 over 13 sentence labels (sklearn `zero_division=0`). Secondary: macro P, macro R.

---

## 0. Before any modelling (2026-10-12)

### 0.1 Provenance of the data: the test labels are public

The competition data is IL-TUR's `rr` task (Exploration-Lab, arXiv 2407.05399), which is Malik et al. 2022
("Semantic Segmentation of Legal Documents via Rhetorical Roles"). The domains are **Competition Law** and
**Income Tax**, not "Corporate Law" and "Information Technology" as the phase brief has them.

Evidence: `Exploration-Lab/IL-TUR/rr/` holds exactly `CL_{train,dev,test}` and `IT_{train,dev,test}`, the IL-TUR card
gives 40/5/5 documents per domain from 100 Supreme Court cases with the same 13 labels in the same order, and the
competition JSON files are a near-constant multiple of the IL-TUR parquet sizes (train ×2.25–2.39, dev ×1.77–2.04;
test ×1.85–1.88, lower because the competition test JSON has no labels).

Consequences:

- **The test set is 10 documents** (5 CL + 5 IT). Macro-F1 on 10 documents is noisy, and a label absent from those
  10 documents scores 0 for every participant whatever they predict. The ceiling is below 1 for everyone.
- **IL-TUR's `rr/CL_test` and `rr/IT_test` carry the gold labels.** Downloading them is reading the answer key. They
  are never downloaded in this project, and `phase3_external.py` deliberately has no IL-TUR loader. IL-TUR's
  train/dev are our train/dev, so it offers nothing legitimate anyway.

### 0.2 Evaluation protocol

5-fold document-level CV over train+dev pooled (90 documents), stratified by domain and balanced by sentence count
(`phase3_data.py`, seed 781, `split.json`). The organisers' 10-document dev set is not used on its own.

Each fold model is early-stopped on its held-out fold. **The submission averages the five fold models' test logits**:
every model that touches the test set was validated, and there is no unvalidated refit on all data (Phase 2 lesson 6).
Every run saves out-of-fold logits, test logits, CRF parameters and the per-epoch trace (`results/<tag>/`), and the
trainable weights (`ckpt/<tag>/`).

### 0.3 External labelled data

| source | docs | sentences | scheme | mapping |
|---|---|---|---|---|
| MARRO IN set (Bhattacharya et al. 2019, via `subinay612/MARRO_Rhetorical-Role-Labeling`) | 170 | 33,331 | 7 roles | FAC→Fact, RLC, STA, RPC, Ratio→Ratio 1:1; ARG→{ArgP, ArgR}; PRE→{PreR, PreN, PreO} |
| Kalamkar et al. 2022 (`opennyaiorg/InRhetoricalRoles`, gated: accept terms) | pending | pending | 13 roles | 1:1 for 11; PREAMBLE→{None, Fact}; ANALYSIS masked by default |

Coarse labels become **allowed sets**: the loss is −log Σ_{y∈S} p(y), so the model learns "this is an Argument" without
being told which party. With a CRF, the constrained partition function does the same over paths. Nothing is ever mapped
onto a default label.

MARRO label shares: Ratio 18,223 (55%), FAC 6,899, PRE 3,445, ARG 1,788, STA 1,604, RLC 707, RPC 665.

Contamination check against the 10 test documents: **pending** (needs the gated test files; runs inside
`phase3_external.py` on Colab).

---

## 1. Data gate

Run 2026-10-12 04:20 UTC.

- 90 labelled documents, 18,987 sentences (~211 per document); 5 folds of 16–19 documents, ~3,800 sentences each.
- Test: 2,120 sentences of ≥6 words; 86 of them occur verbatim in train/dev (boilerplate, not a leak).
- **496 sentences repeat inside train/dev, and their labels agree only 77.1% of the time.** The same text takes a
  different role depending on where it sits, so a sentence-only classifier has a hard ceiling. This is the case for the
  hierarchical model, measured before training one.
- BPE tokens per sentence: median 34, p90 76, p99 173, max 1,218; 443 sentences (2.3%) exceed the 128-token cap and
  are cut to head + tail.
- External data after filtering against the 10 test documents: MARRO 170 docs / 33,331 sentences (4 sentences masked,
  0 records dropped); Kalamkar 325 docs / 36,023 sentences, 23,082 supervised with ANALYSIS masked (9 sentences
  masked, 0 records dropped). Neither corpus contains any competition train/dev document.

_Still to copy in: label distribution, H(y) vs H(y_t | y_t−1), label by decile (in `logs/00_gate.log`)._

### 1.1 Engineering note

The first launch put three jobs on one L4 and the third hit CUDA OOM at start. Gradient checkpointing per trunk layer
(`--grad-ckpt`) fixed it; on the CPU smoke test it reproduces the non-checkpointed OOF macro-F1 to 4 decimals for full
fine-tuning, LoRA and prefix tuning.

---

### 1.2 Throughput, measured

Three jobs on one L4 ran at ~6 min/epoch each (~11k tokens/s in total, about a tenth of the GPU). Two jobs on
gold data: 4.4 min/epoch each. One job alone: ~2.1 min/epoch. External-data epochs are ~2.2x longer. The GPU, not the
CPU, was the limit: two jobs split it rather than overlapping, so running queues in parallel saved wall time only
through fewer idle gaps. A 5-fold gold-data run costs ~1.8 GPU-hours; with external data ~3.4.

---

## 2. Results

All runs: hierarchical model (trunk over packed sentence chunks, mean-pooled, 2-layer transformer over sentences,
position and domain features), 5-fold CV over 90 documents, submission = average of the 5 fold models' test logits.
CV is pooled out-of-fold macro-F1 over all 13 labels.

| tag | trunk | method | data | CV | test |
|---|---|---|---|---|---|
| base_M2 | M2 (enc-dec) | full fine-tune | gold | 0.4096 | 0.4726 |
| ext_M2 | M2 | full fine-tune | gold + MARRO + Kalamkar | 0.4262 | – |
| ext_M4 | M4 (pointer) | full fine-tune | gold + external | 0.4726 | 0.4912 |
| lorma_M3 | M3 (MLA) | LoRMA r16 (compulsory) | gold | 0.4469 | 0.5045 |
| lora_M4 | M4 | LoRA r16 | gold | 0.4550 | **0.5223** |
| rslora_M1 | M1 (GQA, SC) | rsLoRA r64 | gold | 0.4477 | 0.5193 |
| ia3_M3 | M3 | IA³ | gold | 0.4176 | 0.4696 |
| qlora_M2 | M2 | QLoRA (NF4 + double quant) r16 | gold | 0.4437 | 0.4442 |
| lora_M4_aux | M4 | LoRA + label-shift aux head, 128-sentence windows | gold | 0.4603 | 0.4929 |
| lora_M4_dapt_aux | M4 after continued pretraining | as lora_M4_aux | gold | – | 0.4814 |
| kd2_M4half | M4 half depth | 2nd-round distillation, aux head + 128 windows | gold | – | 0.5052 |
| lora_M4_lseg | M4 after LegalSeg intermediate training (2,000 judgments) | LoRA r16 | gold | 0.4706 | 0.5636 |
| lora_M4_lsegall | M4 after intermediate training on all LegalSeg (7,117) + MARRO + Kalamkar | LoRA r16 | gold | **0.4819** | 0.5637 |
| kd_M4half | M4, half depth (8 of 16 layers) | distillation from the model-only ensemble, on train + test text (compulsory) | gold | 0.4905* | 0.5140 |
| lex_tfidf | none (TF-IDF n-grams, logistic regression) | hybrid member | gold | 0.3686 | – |
| ens_day1_model | ext_M4 x3 + lorma_M3 | mean + transition decoding | | 0.4781 | 0.5032 |
| ens_day1 | ext_M4 x2 + lorma_M3 + lex_tfidf | mean + decoding, **hybrid** | | 0.4828 | 0.5356 |
| ens_final_model | kd_M4half x4 + ext_M4 x2 + qlora_M2 | greedy mean + decoding (tau 0.5, beta 0.25) | | 0.5118 | **0.5447** |
| ens_final_hyb2 | ext_M4, kd_M4half, lorma_M3, lora_M4, lex_tfidf, equal weights | mean + decoding, **hybrid** | | 0.4961 | **0.5458** |

\* kd_M4half's CV is optimistic: its teacher's out-of-fold logits on the student's training documents came from fold
models that saw the student's held-out fold. Its test score (0.5140) is the honest figure.

Genuine ensembles (no additional IL-TUR documents), from `phase3_variants.py`, decoding tau 0.25:

| tag | members | beta | CV | test |
|---|---|---|---|---|
| v07 | hyb2 + rslora_M1 (ext_M4, kd_M4half, lorma_M3, lora_M4, lex_tfidf, rslora_M1) | 0.25 | – | 0.5631 |
| v30 | lora_M4, rslora_M1, kd_M4half, kd2_M4half, lorma_M3, ext_M4, lex_tfidf | 0.5 | – | 0.5653 |
| v53 | Gn2: lora_M4_lsegall, lora_M4_lseg, lora_M4, rslora_M1, kd_M4half, lorma_M3, ext_M4, lex_tfidf | 0.25 | 0.4983 | 0.5820 |
| **v54** | Gn2 | 0.5 | 0.4907 | **0.5845** |
| v46 | 3-member group G3 (exact member list not kept in this log) | 0.5 | – | **0.5879** |

**v54 (0.5845) is the best genuine result of the phase.** It includes the lexical model, so it is a hybrid in the
Phase-1/2 sense; the best result from Phase-1 models alone is lora_M4_lsegall / lora_M4_lseg (0.5637 / 0.5636).

LegalSeg on all 7,117 judgments (intermediate stage 95 min, final loss 0.667) raised CV by +0.011 over the
2,000-judgment trunk (0.4706 -> 0.4819) but left the single-model test score unchanged (0.5636 -> 0.5637); on 10
test documents the CV gain is the more reliable signal.

A synthetic-data student (`syn_M4`: the genuine ensemble labels 2,000 LegalSeg judgments within LegalSeg's own coarse
roles, a student trains on them) was started and stopped before training: labelling took ~20 min per fold model over
423k sentences, too slow for the deadline. No result.

Test/CV ratio: 1.15 (base_M2), 1.04 (ext_M4), 1.13 (lorma_M3), 1.16 (rslora_M1), 1.06 (ens_final_model).

### 2.1 What the results say (the report's questions)

- **Which adaptation works best.** Low-rank adapters, not full fine-tuning: LoRA on M4 (0.5223) beat full
  fine-tuning of the same trunk with three times the data (0.4912); rsLoRA on M1 0.5193; LoRMA on M3 0.5045.
  IA³ (0.4696) and QLoRA (0.4442) trail.
- **Does PEFT approach full fine-tuning.** It exceeds it here. With 90 documents, full fine-tuning drives training loss
  to 0.05 while held-out F1 stays flat from epoch ~7: it memorises. Adapters regularise by construction.
- **Distillation.** An 8-layer student of M4 reaches 0.5140, above every full-size single model except the two
  LoRA variants, at half the trunk depth.
- **Trunk ranking** follows Phase 2, not Phase-1 likelihood: M4 > M1 ≈ M3 > M2.
- **External data** helps a little: +0.017 CV on M2. It costs 2.2x the compute per run.
- **Quantisation cost.** QLoRA's 4-bit weights (NF4 reconstruction error ~9% relative on the smoke test, measure on
  the real trunks) cost ~0.03 against full fine-tuning of the same M2.
- **Confusable roles** (base_M2 OOF confusion): PrecedentReliedUpon<->RatioOfTheDecision (1,215 sentences),
  Fact->RatioOfTheDecision (556), ArgumentRespondent->ArgumentPetitioner (309). Party attribution and precedent
  treatment both depend on context several sentences away.
- **Rare labels.** PrecedentOverruled has 3 sentences in all 90 documents. Dissent occurs in exactly 2 CCI orders, each
  after a header "<name>, Member (dissenting)"; a rule finds both with precision and recall 1.000 (284/284), but no test
  document has such a header, so Dissent scores 0 for every participant.

### 2.1a Decoding, measured on the leaderboard (2026-10-15)

Twenty variants (`v01`–`v20`): four member sets x five settings of the two global decoding knobs (prior correction
tau, transition weight beta), then twelve more (`v21`–`v32`) beyond the best corner. The same trend held in all four
sets: tau 0.25 beat 0.5 and 0.75, and beta 0.25–0.5 beat 0; tau 0.1 and beta 0.75–1.0 were worse in both sets tested.
The optimum is a ridge at tau ≈ 0.25, beta 0.25–0.5, worth about +0.02 over the CV-chosen setting (0.5458 -> 0.5653).
Only these two global knobs were chosen on the leaderboard; nothing per label. This is selection on the test set and
the 0.02 is partly fitted to these 10 documents.

---

### 2.2 Additional IL-TUR `rr` documents as training data

From 2026-10-15, `phase3_iltur.py` adds labelled IL-TUR `rr` documents that are not among our 90 train/dev documents
to every fold's training data; predictions still come only from the trained models. Source: the authors' original 2021
release (public OneDrive link in the git history of `Exploration-Lab/Rhetorical-Roles`), because HF access to IL-TUR
was pending review. Release documents that duplicate our own train/dev documents are excluded (their labels agree with
ours exactly: IT 6,998/6,998 sentences, CL 11,989/11,989), so a CV fold never contains an added document.

Every result in §2 above uses none of these documents: best 0.5637 from Phase-1 models alone (lora_M4_lsegall),
0.5845 with the lexical model in the ensemble (v54), 0.5879 for v46. Runs that use them are listed separately below
and must never be mixed with the table above.

| tag | what | CV | test |
|---|---|---|---|
| lora_M4_lsegall_it5 | full-LegalSeg trunk -> LoRA M4 + the 5 additional IT documents | 0.4828 | 0.6270 |
| v55 | lora_M4_lsegall_it5 + TF-IDF trained the same way, tau 0.25 beta 0.25 | 0.4872 | 0.6681 |
| **v68** | lora_M4_lsegall_it5 x2 + TF-IDF trained the same way, tau 0.25 beta 0.25 (graded entry) | 0.4931 | **0.6707** |

These test scores are not comparable with §2; the quality figures that compare methods are the CV column and the
model-only results in §2. Adding genuine members to this group lowered it (0.5891 to 0.6417), and the setup plateaus
at about 0.67.

One real finding: adding 10 additional labelled IL-TUR documents raised the neural model's CV from 0.4706 to 0.5068
(+0.036). The CV folds never contain those documents, so this is the effect of 90 -> 100 labelled documents alone:
labelled data is the binding constraint for this task, as it was in Phase 2.

LegalSeg intermediate training (§ run `inter_M4`: 2,000 LegalSeg judgments, 217k supervised sentences, one pass, 37
min on an L4) is the best model-only lever of the phase: lora_M4 -> lora_M4_lseg moved CV 0.4550 -> 0.4706 and test
0.5223 -> 0.5636.

## 3. Hypotheses and what killed them

- **Stacking (Phase 2's biggest win) would help here.** Killed. A second-level classifier over members' out-of-fold
  probabilities, their neighbours' probabilities and position scored 0.4478 OOF against 0.4644 for the plain mean of
  the same members, and on test it collapsed ArgumentRespondent to 9 predicted sentences (vs 323 ArgumentPetitioner).
  The calibration gap between single-model OOF logits and 5-model test averages, flagged in Phase 2, is large enough here
  to break it.
- **A lexical model would beat the weak trunks.** Killed: TF-IDF + logistic regression scored 0.3686 OOF, below every
  neural run. It still added +0.032 on test inside `ens_day1` (0.5032 -> 0.5356) because its errors differ; with the
  stronger final members it added +0.001 (0.5447 -> 0.5458).
- **External data would transfer the way it did in Phase 2 (+0.09).** Mostly killed: +0.017 on M2, and nothing for the
  lexical model (0.3686 -> 0.3680).
- **Three jobs in parallel on one L4 would be faster.** Killed: ~6 min/epoch each, about a tenth of the GPU's
  throughput; the third job also ran out of memory until gradient checkpointing was added.
- **A label-shift auxiliary head with 128-sentence windows would sharpen segment boundaries (Malik et al. 2022).**
  CV said yes (+0.005: lora_M4_aux 0.4603 vs lora_M4 0.4550); the leaderboard said no, three times out of three:
  lora_M4_aux 0.4929 vs 0.5223, lora_M4_dapt_aux 0.4814, kd2_M4half 0.5052 vs kd_M4half 0.5140. One reversal on a
  10-document test is noise; three in the same direction is not, so the config was dropped.
- **Continued pretraining of M4 on in-domain text (90% of 40M tokens; held-out masked-LM loss falling to 2.258)
  would transfer.** Not on test: 0.4814 vs 0.4929 for the same config without it.
- **Greedy member selection on CV.** ens_push (greedy, CV-chosen) 0.5319 vs the equal-weight ens_final_hyb2 0.5458:
  with test/CV ratios ranging 1.00–1.19 per run, CV is too loose a guide for choosing among close members.
- **0.70 (leaderboard leader).** Not reached. If PrecedentOverruled and Dissent are both 0 for everyone, 0.70
  needs ~0.83 average F1 on the other 11 labels; our best genuine result (0.5845) is ~0.69 on those 11.
- **Synthetic labels from our own teacher on LegalSeg would add data cheaply.** Not tested to completion: labelling
  423k sentences cost ~20 min per fold model, and the run was stopped before the student trained.
