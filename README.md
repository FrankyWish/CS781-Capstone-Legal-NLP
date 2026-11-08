# CS781 Capstone: an in-house legal NLP stack built from scratch

Nithin T M (roll 230709), Codabench `n_230709`, IIT Kanpur, CS781, Sept to Nov 2026.

This repository holds the code and experiment logs for all seven phases of the CS781
capstone, an individual Codabench competition on Indian legal NLP. Phase 1 pretrains four
language models from scratch (126M to 151M parameters) on the permitted legal corpus.
Phases 2 to 7 start from those four models and adapt them to six downstream tasks. No
external pretrained weights and no LLM APIs are used anywhere in a submitted system.

| phase | task | metric | best model-only (test) | final graded entry (test) | rank |
|---|---|---|---|---|---|
| 1 | Legal LM pretraining (512-word prompt, 128-word continuation) | BERTScore F1 | 0.8449 (MBR over 3 models) | **0.8689** (MBR + corpus retrieval) | 1 |
| 2 | Legal NER, 12 labels | strict Macro-F1 | 0.4873 (ensemble) | **0.6515** (stacked systems + rules) | 7 |
| 3 | Rhetorical roles, 13 labels | Macro-F1 | 0.5637 (LoRA M4, LegalSeg) | **0.6707** (trained with 5 TA-permitted IL-TUR test documents; best genuine 0.5879) | 4 |
| 4 | Court judgment prediction | Macro-F1 | 0.7225 (3 x M4 LoRMA) | **0.8767** (corpus lookup + M4) | 2 |
| 5 | Statute identification, 100 labels | Macro-F1 | 0.3949 (neural ensemble) | **0.4632** (stacker over neural + TF-IDF) | 6 |
| 6 | Summarization | BERTScore F1 | 0.8391 (abstractive reranker) | **0.8606** (Phase-1 trunks as sentence scorers) | 1 |
| 7 | Prior case retrieval | Micro-F1@7 | 0.3325 (LoRMA bi-encoder) | **0.5157** (citator rules + OT decoding) | 2 |

Ranks as shown on the Codabench leaderboards on 9 November 2026.

Several graded entries are hybrids: they combine the Phase-1 models with lexical models,
rules, or a lookup in the permitted Phase-1 corpus. Every phase README reports the
model-only number beside the graded one, and says which is which.

## Layout

```
capstone_Project/
├── README.md            this file
├── tools/               submission validator + ledger, dev-split builder, parameter counter
├── phase1/              pretraining: tokenizer, corpus, four architectures, decoding, MBR, retrieval
├── phase2/ … phase7/    one folder per downstream task
│   ├── README.md        architecture, method, what worked, how to run the best system
│   ├── EXPERIMENTS.md   the full experiment log, every number as measured
│   ├── RUN.md / DESIGN.md  run instructions and design notes, where they exist
│   ├── code/            every script used in the phase, as run on Colab
│   └── submissions/     index of every Codabench upload and its score (zips not in git)
```

Start with a phase's `README.md`. `EXPERIMENTS.md` is the long version and records the
failed ideas as well as the successful ones.

## The Phase-1 foundation in one paragraph

All four models share one tokenizer (my own byte-level BPE, 32,768 tokens, legal-aware
pre-tokenisation), one 1.255B-token corpus built from the four permitted sources, and one
training recipe (Muon on 2-D weights, AdamW elsewhere, QK-norm with z-loss, value residual,
6 wall-clock hours on an L4 each). The architectures are M1 `LexArc-D`, a GQA decoder
(125.9M); M2 `LexArc-ED`, an encoder-decoder with a joint next-token and masked-LM objective
(151.0M); M3 `LexArc-L`, a decoder with multi-head latent attention (129.0M); and M4
`LexArc-P`, a GQA decoder with a pointer-generator head (126.0M). `M1_SC` is M1 after four
more hours of Supreme-Court-only pretraining. Every later phase loads these checkpoints.

## Common setup

All runs were on Google Colab (one NVIDIA L4, bf16) with Google Drive as storage. The code
reads its paths from environment variables, so the same scripts run on Colab and locally:

```python
import os
BASE = '/content/drive/MyDrive/cs781_capstone'
os.environ.update(
    CS781_BASE=BASE,
    CS781_P1=f'{BASE}/phase1',          # Phase-1 checkpoints, tokenizer, corpus shards
    CS781_LIB=f'{BASE}/phase1/shared/lib',  # cs781_bpe.py and the model files
    CS781_P2=f'{BASE}/phase2',          # one CS781_P<n> per phase
    HF_TOKEN='<your token>',            # the dataset Exploration-Lab/CS-781-Capstone is gated
)
```

Python 3.10+, PyTorch 2.x, numpy, scikit-learn, `bert-score` and `rouge-score` for the
official metrics, `pyahocorasick` for the Phase-1 retrieval. `tools/requirements.txt` lists
the evaluator dependencies.

Do not train on a T4. `torch.cuda.is_bf16_supported()` returns True on SM75, but bf16 is
emulated there and runs about 30x slower. Every training script prints the GPU name first.

## Data and integrity

The only data sources are the competition's Hugging Face dataset
(`Exploration-Lab/CS-781-Capstone`), the permitted Phase-1 corpus, and external data that
the course staff allowed for Phases 2 onward, each described in its phase README. Several
competition datasets are public IL-TUR tasks whose test labels are also public; those test
files were never downloaded, with one disclosed exception in Phase 3, where the course TA
permitted training on the IL-TUR `rr` test split (`phase3/README.md`). Every phase checks
its train/dev data for copies of test documents before training and drops them.

Datasets, checkpoints, caches and submission zips are not in this repository. They are on
the author's Google Drive (`MyDrive/cs781_capstone/`).
