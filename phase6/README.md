# Phase 6: summarization of Indian Supreme Court judgments

**Task.** Summarise a full judgment (median 3,147 words) into a headnote-style summary (median 642 words). There are 7,028 training pairs and 100 test documents from IN-Abs. The primary metric is BERTScore F1 (`roberta-large`, raw), with ROUGE-1 and ROUGE-L as secondary metrics.

**Results (test).**

| **system** | **kind** | **test** |
| --- | --- | ---: |
| `m4_lorma_trunc`: M4 + LoRMA, head 70% / tail 30% of the judgment | abstractive, single model | 0.8390 |
| `rank_model` / `mbr_model`: reranker or MBR over abstractive candidates | abstractive | **0.8391** |
| `bsel_best`: sentence selection, gradient boosting on BERTScore-derived labels | extractive | 0.8546 |
| `nsel4_best`: sentence selection scored by four fine-tuned Phase-1 trunks + `bsel` | extractive | **0.8606**, rank 1 (graded entry) |

The graded system selects sentences directly from the test document instead of generating new text. The best abstractive score is 0.8391, and both results are included in the report. Second place scored 0.8570, so the difference of +0.0036 is around one test standard error.

## Architecture

**Abstractive members** (`phase6_sum.py`). For M1, M3 and M4, decoder-only continuation was used, where the prompt contains the document and the model continues with the summary. M2 used an encoder-decoder setup. Every prompt was padded to exactly the same length so that the summary starts at the same position in every row.

The required runs included LoRMA on M4 (compulsory), LoRA and AdaLoRA on M2, Houlsby adapters on M1, QLoRA and Pfeiffer adapters on M3, and KD into a half-depth M1 (compulsory). Full fine-tuning of M4 was also run as a reference.

**Why extraction.** BERTScore truncates both sides at 512 `roberta-large` tokens. Because of this, the score mostly stops changing after around 450 words. For example, taking only the first 450 words of a reference and comparing it with the full reference gives BERTScore 1.000 and ROUGE-1 0.609. The median reference is 642 words, so only around the first 420 words of each summary are actually compared.

The 8-gram copy rate of references from their documents is 0.32, and the extractive oracle gets ROUGE-1 of 0.77. This made sentence extraction worth trying.

**`bsel`** (`phase6_bsel.py`). For every training sentence, token-level cosine tables were calculated against the reference using the same model and layer as the scorer, which is `roberta-large` layer 17. This gives a BERTScore-aligned label for each sentence. `roberta-large` is only used for creating training labels. At test time, only Phase-1 trunks and gradient boosting are used.

**`nsel`** (`phase6_nsel.py`). The top 4 blocks of each Phase-1 trunk (M1, M2's encoder, M3 and M4) are fine-tuned to predict those sentence labels using sentence-in-context windows. Their scores are combined with the 27 hand-made features from `bsel`. The highest-scoring sentences are then selected and placed in their original document order, up to a target length for each document.

A length model was also tested and worked better than using a fixed target length. Its median absolute relative error was 0.324 compared to 0.398 for the constant target.

## What worked and what did not

| **lever** | **held-out gain** | **test** |
| --- | ---: | ---: |
| `bsel` labels from the metric's own token tables, vs ROUGE labels (`sel_best`) | +0.0124 | 0.8379 to 0.8546 |
| `nsel` (M1 + M2 trunks as sentence scorers) | +0.0075 ± 0.0007 | 0.8596 |
| `nsel4` (all four trunks) | +0.0001 ± 0.0004 | 0.8606 |

Several other approaches did not work as well. Hybrids of neural and extractive output got a best dev score of 0.8541. A neural opening stitched onto extraction moved test from 0.8449 to 0.8518. Selecting input sentences for the neural model scored 0.8347, compared to 0.8403 for plain truncation.

Training only on the part of the summary that BERTScore reads also did not help. ROUGE-utility MBR over all candidates mostly selected the extractive member, doing so for 188 out of 200 documents. A word-level score in `bsel` gave +0.0025 on held-out data but -0.0007 on test.

LoRMA and full fine-tuning gave exactly the same test score (0.8343). KD also lost 0.0087 compared to its teacher.

**Integrity.** The Phase-1 corpus contains all 100 test judgments, and around 99 of these copies also contain the headnote. No lookup system was built or submitted. `cons_*` and `mbr_bert`, which would use `roberta-large` at test time, were never uploaded.

## How to run

The scripts were run unattended from Colab using the provided run commands. [`RUN.md`](https://github.com/FrankyWish/CS781-Capstone-Legal-NLP/blob/main/phase6/RUN.md) contains the exact cells and the code-hash check used for each run.

```bash
bash p6.sh all      # gate, extraction baselines, required abstractive runs, MBR, reranker
bash p6f.sh         # bsel_best, window scorer, stitch / consensus experiments
bash p6g.sh         # bsel2 refits (did not transfer)
bash p6h.sh         # cached labels, bsel3, nsel_m1 + nsel_m2, nsel_* zips
bash p6i.sh         # nsel_m3 + nsel_m4, four-trunk sweep -> nsel4_best (graded)
```

| **file** | **role** |
| --- | --- |
| `phase6_data.py` | gate, held-out split, scoring identical to the official evaluator |
| `phase6_sel.py` | input views for the neural models, ROUGE-supervised selection |
| `phase6_sum.py`, `phase6_peft.py` | abstractive trainer and the adaptation methods |
| `phase6_ens.py` | candidate pooling, MBR, reranker |
| `phase6_bsel.py` | BERTScore-labelled sentence selection |
| `phase6_nsel.py` | Phase-1 trunks fine-tuned as sentence scorers, stack, sweep |
| `phase6_win.py`, `phase6_batch.py` | window-aimed selection, final-day batch |
| `phase6_external.py` | corpus overlap measurement (measure only, builds no submission) |

Full log: [`EXPERIMENTS.md`](https://github.com/FrankyWish/CS781-Capstone-Legal-NLP/blob/main/phase6/EXPERIMENTS.md). Design: [`DESIGN.md`](https://github.com/FrankyWish/CS781-Capstone-Legal-NLP/blob/main/phase6/DESIGN.md). Every upload: [`submissions/README.md`](https://github.com/FrankyWish/CS781-Capstone-Legal-NLP/blob/main/phase6/submissions/README.md).
