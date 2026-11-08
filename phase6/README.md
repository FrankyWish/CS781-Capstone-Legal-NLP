# Phase 6: summarization of Indian Supreme Court judgments

**Task.** Summarise a full judgment (median 3,147 words) into a headnote-style summary
(median 642 words). 7,028 training pairs, 100 test documents (IN-Abs). Primary metric:
BERTScore F1 (`roberta-large`, raw); secondary ROUGE-1, ROUGE-L.

**Results (test).**

| system | kind | test |
|---|---|---|
| `m4_lorma_trunc`: M4 + LoRMA, head 70% / tail 30% of the judgment | abstractive, single model | 0.8390 |
| `rank_model` / `mbr_model`: reranker or MBR over the abstractive candidates | abstractive | **0.8391** |
| `bsel_best`: sentence selection, gradient boosting on BERTScore-derived labels | extractive | 0.8546 |
| `nsel4_best`: sentence selection scored by four fine-tuned Phase-1 trunks + `bsel` | extractive | **0.8606**, rank 1 (graded entry) |

The graded system chooses sentences from the test document; it does not generate text. The
best abstractive result is 0.8391. Both go in the report. Second place was 0.8570, so the
lead (+0.0036) is about one test standard error.

## Architecture

**Abstractive members** (`phase6_sum.py`). Decoder-only continuation for M1, M3, M4 (prompt =
the document view, then the summary) and encoder-decoder for M2. Every prompt is padded to
exactly the same length so the summary starts at the same position in every row. Seven
required runs: LoRMA on M4 (compulsory), LoRA and AdaLoRA on M2, Houlsby adapters on M1,
QLoRA and Pfeiffer adapters on M3, KD into a half-depth M1 (compulsory), plus full
fine-tuning of M4 as the reference.

**Why extraction.** BERTScore truncates both sides at 512 `roberta-large` tokens, so it
saturates by about 450 words (reference prefix of 450 words against the whole reference:
BERTScore 1.000, ROUGE-1 0.609). The median reference is 642 words, so only the first ~420
words of each summary are compared. The 8-gram copy rate of references from their documents
is 0.32, and the extractive oracle reaches ROUGE-1 0.77.

**`bsel`** (`phase6_bsel.py`). For every training sentence, compute its token-level cosine
tables against the reference under the scorer's own model and layer (roberta-large layer 17).
That yields a BERTScore-aligned label per sentence. `roberta-large` is used only to label
training data; nothing but Phase-1 trunks and gradient boosting runs at test time.

**`nsel`** (`phase6_nsel.py`). The top 4 blocks of each Phase-1 trunk (M1, M2's encoder, M3,
M4) are fine-tuned to regress those labels from sentence-in-context windows. Their scores are
stacked with the 27 hand features of `bsel`, and the top-scoring sentences are emitted in
document order up to a per-document length target (a length model beat a constant target:
median absolute relative error 0.324 against 0.398).

## What worked and what did not

| lever | held-out gain | test |
|---|---|---|
| `bsel` labels from the metric's own token tables, vs ROUGE labels (`sel_best`) | +0.0124 | 0.8379 to 0.8546 |
| `nsel` (M1 + M2 trunks as sentence scorers) | +0.0075 ± 0.0007 | 0.8596 |
| `nsel4` (all four trunks) | +0.0001 ± 0.0004 | 0.8606 |

Refuted: hybrids of neural and extractive output (best 0.8541 dev), a neural opening stitched
onto extraction (0.8449 to 0.8518 test), selected input sentences for the neural model (0.8347
against 0.8403 for plain truncation), training only on the part of the summary BERTScore
reads, ROUGE-utility MBR over all candidates (picked the extractive member on 188 of 200
documents), a word-level score in `bsel` (+0.0025 held-out, -0.0007 test). LoRMA matched full
fine-tuning exactly on test (0.8343 both). KD cost 0.0087 against its teacher.

Integrity: the Phase-1 corpus holds all 100 test judgments, and about 99 of those copies
include the headnote. No lookup system was built or submitted. `cons_*` and `mbr_bert`, which
would run `roberta-large` at test time, were never uploaded.

## How to run

Every script ran unattended on one Colab L4 from a single cell; [`RUN.md`](RUN.md) has the
exact cells, and the code-hash check each run used.

```bash
bash p6.sh all      # gate, extraction baselines, the required abstractive runs, MBR, reranker
bash p6f.sh         # bsel_best, window scorer, stitch / consensus experiments
bash p6g.sh         # bsel2 refits (did not transfer)
bash p6h.sh         # cached labels, bsel3, nsel_m1 + nsel_m2, nsel_* zips
bash p6i.sh         # nsel_m3 + nsel_m4, four-trunk sweep -> nsel4_best (graded)
```

| file | role |
|---|---|
| `phase6_data.py` | gate, held-out split, scoring identical to the official evaluator |
| `phase6_sel.py` | input views for the neural models, ROUGE-supervised selection |
| `phase6_sum.py`, `phase6_peft.py` | abstractive trainer and the adaptation methods |
| `phase6_ens.py` | candidate pooling, MBR, reranker |
| `phase6_bsel.py` | BERTScore-labelled sentence selection |
| `phase6_nsel.py` | Phase-1 trunks fine-tuned as sentence scorers, stack, sweep |
| `phase6_win.py`, `phase6_batch.py` | window-aimed selection, final-day batch |
| `phase6_external.py` | corpus overlap measurement (measure only, builds no submission) |

Full log: [`EXPERIMENTS.md`](EXPERIMENTS.md). Design: [`DESIGN.md`](DESIGN.md). Every upload:
[`submissions/README.md`](submissions/README.md).
