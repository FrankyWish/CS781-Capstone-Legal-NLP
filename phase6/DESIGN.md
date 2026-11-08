# Phase 6 — Abstractive Summarization: design

Window 2026-10-27 → 2026-11-03 23:59 IST (moved from the spec's Nov 2). Written 2026-10-29, two days
in, nothing built yet. Primary metric **BERTScore F1**; secondary ROUGE-1 F, ROUGE-L F.

Sections 1–5 are the plan as written on Oct 29 and are left unchanged. Section 6 compares it with
what actually happened. **Outcome: rank 1 at 0.8606.**

## 1. What the metric actually rewards

`evaluate_phase6.py` calls `bert_score.score(preds, refs, lang='en')`. That is
roberta-large layer 17, **`idf=False`, `rescale_with_baseline=False`** — the same
configuration Phase 1 was scored on, and the same four consequences hold:

1. **The usable range is tiny.** Unrescaled BERTScore F1 for fluent English of
   roughly the right length sits near 0.80–0.82. Phase 1's leaderboard spanned
   0.8485 (#1) to 0.8338 (#10). We are fighting over the third decimal.
2. **F1 is symmetric, so length counts twice.** Short output destroys recall, long
   output destroys precision. In Phase 1 the references were exactly 128 words and
   matching that exactly was the single largest controllable gain. Here reference
   length varies per document, so the equivalent is a **per-document length target**
   fitted on train and then calibrated against the leaderboard. This is the first
   dial, built on day one (`phase6_sel.py calib`, `results/length_calib.json`).
3. **No idf.** Procedural boilerplate, party names and function words score exactly
   as much as the holding of the case. Register matching beats being right.
4. **BERTScore cannot see broken syntax** and will reward content-word soup. Every
   decoding decision is gated on dev BERTScore *and* a repetition check, never one
   alone. A small from-scratch model's characteristic failure is a loop, and a loop
   can score well on BERTScore while being indefensible at the viva.

A fifth consequence is specific to summarization: because precision is half the
score, **a summary made of sentences the judgment already contains starts from a
very high floor**. Extraction is not a weak baseline on this metric. Phase 1 ended
with retrieved text beating generated text by 0.024 BERTScore, and the same
asymmetry is likely here for the same reason — a 126M model trained from scratch on
a few thousand pairs writes worse English than the judgment already contains.

## 2. Where the leaderboard will land

Prediction, to be checked against the first submissions:

| system | expected BERTScore F1 |
|---|---|
| naive full-document truncation, no length control | 0.79–0.81 |
| our Phase-1 models fine-tuned, generation only | 0.81–0.83 |
| length-calibrated supervised extractive selection | 0.845–0.86 |
| best hybrid + MBR + reranker | 0.855–0.87 |
| **likely top-3 band** | **0.855–0.875** |

Two things would break that prediction, both measured before any modelling:

* **If the reference summary is largely inside the input document** (the gate's
  `summary_in_document` 8-gram copy rate), the ceiling is far higher and the phase
  is an extraction problem, not a generation problem.
* **If the Phase-1 corpus contains the test documents with their headnotes.**
  Indian Supreme Court judgments from 1950–1994 were published *with* a
  lawyer-written headnote, and this phase's references are almost certainly those
  headnotes. If the organisers derived the documents by stripping the headnote from
  the same text that is in the permitted pretraining corpus, then the answer key is
  in the corpus, and someone will find it — Phase 4 went from 0.75 to 0.96 that way.
  `phase6_external.py` measures this and nothing else; a submission built on it is
  a **lookup hybrid**, needs Nithin's explicit go-ahead, and is reported as lookup
  beside the model-only number, exactly as Phases 1 and 4 were.

What top 3 takes, assuming no corpus leak: per-document length calibration, a
supervised extractive member that is genuinely strong, MBR over a pooled candidate
set, and a dev-trained reranker. Those last two were worth +0.0068 and +0.007 in
Phases 1 and 5 respectively. The required fine-tuning runs are a requirement, not
a plan to win — the honest model-only number will be a long way below the hybrid,
and both go in the report.

## 3. The required six, and how they cover everything

Every run starts from a Phase-1 checkpoint. No pretrained weights, no LLM APIs.

| run | technique | Phase-1 architecture | long-document view |
|---|---|---|---|
| `m4_lorma` | **LoRMA** (compulsory) | M4 `LexArc-P`, pointer-generator | selected sentences, 1152 tok |
| `m2_lora` | LoRA (attention projections, encoder + decoder) | M2 `LexArc-ED`, encoder-decoder | selected sentences |
| `m1_houlsby` | Houlsby adapters (after attention and FFN) | M1 `LexArc-D`, GQA decoder | selected sentences |
| `m3_qlora` | QLoRA (NF4 base + LoRA) | M3 `LexArc-L`, MLA decoder | selected sentences |
| `m2_adalora` | AdaLoRA (rank 12 -> 6, budget reallocated) | M2 | selected sentences |
| `m3_pfeiffer` | Pfeiffer adapters (after FFN only) | M3 | selected sentences |
| `kd_m1half` | **knowledge distillation** (compulsory) | half-depth M1 student, teacher = best of the six | selected sentences |

Seven runs, seven techniques, every one from the brief's list (§12.2.1): LoRMA,
LoRA, Houlsby, QLoRA, AdaLoRA, Pfeiffer, KD. All four architectures. Extras run
only if the GPU deadline allows: `m4_full` (full fine-tuning, the ceiling the
PEFT methods are compared against) and the long-document axis on M4 LoRMA —
`m4_lorma_2048` (selection into a 2048-token prompt) and `m4_lorma_trunc`
(raw document, head 70% + tail 30%).

**Why no prefix or prompt tuning.** Both change the input sequence (learned
states or embeddings), and the Phase-1 models' `generate()` takes token ids, so
supporting them means rewriting four KV-cached decoders. Pfeiffer adapters and
AdaLoRA are on the same list and need nothing changed, so they replace them at
no cost to the requirement.

**Distillation design.** Teacher = the best fine-tuned model (or the MBR consensus
over the pool, which is strictly stronger). Student = M1 with every second block
dropped, trained on (a) teacher-generated summaries for the *fit* split as extra
targets and (b) token-level KL against the teacher's next-token distribution.
Sequence-level KD is the generation-specific signal the brief asks for; the
token-KL term is what makes it distillation rather than self-training.

**Long-document handling** is a separate axis from the adaptation technique, and the
report compares them on one fixed technique: truncation, importance-based selection
(`views/sel1024`), the same selection at 2048 tokens, and two-pass hierarchical
(chunk summaries concatenated, then summarised). Documents here run to thousands of
words against a 1024–1280-token pretraining window, so this axis decides more than
the choice of adapter does.

**Input format.** Decoder-only models are trained on
`<selected document text>\n\nHEADNOTE:\n<summary>` with the loss on the summary
only. No new special token: pre-1994 Supreme Court judgments carry the literal word
HEADNOTE, so the Phase-1 tokenizer and trunk have both seen it in context. M2 takes
the document view through the encoder and generates the summary in the decoder.

## 4. Stages

**Stage 1 (CPU runtime, no GPU quota).** Gate — counts, length distributions, the
fitted length model, the copy-rate and duplicate scans. Then the supervised sentence
scorer, lead and centroid baselines, the length sweep, the extractive submissions,
and the input views stage 2 reads. Output: a submittable system and every number the
rest of the phase is parameterised by.

**Stage 2 (one L4).** The seven runs above, time-budgeted and resumable, each saving
dev and test candidate summaries plus its decoding sweep.

**Stage 3 (CPU + short L4).** MBR over the pooled candidates (utility = BERTScore,
plus a ROUGE-only utility variant that uses no external weights at inference),
the dev-trained reranker with honest out-of-fold CV, extractive/abstractive fusion,
and ±15% length variants of the best system for leaderboard calibration.

## 5. Decisions already taken, and their costs

* **Held-out split, not cross-validation, for member selection.** 600 train
  documents held out; members train on the rest and are scored out-of-sample. Cost:
  no cross-fitted members (Phase 5's +0.006 lever) unless time allows a second fold.
  Reason: four days left, and k-fold multiplies GPU cost by k.
* **Train records that duplicate a test document are dropped from fit and dev.**
  Phase 4's dev.json held labelled copies of every test document and Phase 5's held
  128; selecting on those is selecting on the test set. Counted in
  `results/answer_keys.json` and reported.
* **BERTScore as the MBR utility uses roberta-large.** That is pretrained weights
  inside the pipeline, which the phase rules ban for *the model*. The organisers'
  own scorer ships it and Phase 1's rank-1 system used it, so it stays available —
  but the primary MBR variant uses a ROUGE-only utility, which is defensible without
  argument and costs about 0.001.
* **Extractive submissions are labelled `extractive` in `info.json` and in the
  report.** They are legitimate (the text comes from the test input, which we are
  given) but they are not abstractive and do not count towards the required six.

## 6. What happened, against this design (added 2026-11-08)

| design claim | outcome |
|---|---|
| Length is the first dial | True, but only up to the 512-token window. BERTScore saturates by about 450 words, so past the window length stopped mattering: every `bsel_best` length from x0.8 to x1.25 landed within 0.0005 on test |
| Generation-only lands at 0.81–0.83 | A little better: 0.8164–0.8390 on test. The best model-only system was 0.8391 (`rank_model`, `mbr_model`) |
| Length-calibrated extraction lands at 0.845–0.86 | ROUGE-labelled extraction (`sel_best`) was only 0.8379. What reached that band was labelling sentences with the metric itself (`bsel_best`, 0.8546) |
| Best hybrid + MBR + reranker lands at 0.855–0.87 | No. Every hybrid scored below plain extraction (best `rank_all` 0.8528). The ROUGE-utility MBR could not see what BERTScore-labelled extraction adds |
| Likely top-3 band 0.855–0.875 | Rank 1 at 0.8606. Second was 0.8570 |
| MBR and a reranker are the levers that win | They were not. The winner was a lever this design did not have: the top 4 blocks of each Phase-1 trunk fine-tuned as a sentence scorer on metric-derived labels (`phase6_nsel.py`), stacked with the hand-feature scorer. Two trunks gave +0.0049 on test, four gave +0.0010 more |
| Selected input sentences help the neural models | No: raw head+tail truncation beat selection at the same prompt size (0.8390 vs 0.8343 on test) |
| Two-pass hierarchical summarization as a long-document view | Never built. The four days went to the required runs and extraction |
| Corpus leak possible | Confirmed: 100/100 test documents have a corpus copy, about 99 with the headnote. Lookup never got a go-ahead, so nothing read from the corpus was built or submitted |
| roberta-large stays out of the test-time model | Held. It labelled train and dev sentences only. `cons_*` and `mbr_bert`, which use it at test time, were never uploaded |

The lesson to carry forward: **on a metric this saturated, the lever is labels aligned with the metric
plus the strongest scorer you can train on them**, not more candidates to choose between.
