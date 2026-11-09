# CS781 Capstone: an in-house legal NLP stack built from scratch

Nithin T M (roll 230709), Codabench `n_230709`, IIT Kanpur, CS781, Sept to Nov 2026.

This repository contains the code and experiment logs for all seven phases of the CS781 capstone. It was an individual Codabench competition based on Indian legal NLP. In Phase 1, I trained four language models from scratch (126M to 151M parameters) using the permitted legal corpus. For Phases 2 to 7, I used these four models and adapted them for six different downstream tasks. No external pretrained weights or LLM APIs were used in any submitted system.

| **phase** | **task** | **metric** | **best model-only (test)** | **final graded entry (test)** | **rank** |
|---|---|---|---|---|---|
| 1 | Legal LM pretraining (512-word prompt, 128-word continuation) | BERTScore F1 | 0.8449 (MBR over 3 models) | **0.8689** (MBR + corpus retrieval) | 1 |
| 2 | Legal NER, 12 labels | strict Macro-F1 | 0.4873 (ensemble) | **0.6515** (stacked systems + rules) | 7 |
| 3 | Rhetorical roles, 13 labels | Macro-F1 | 0.5637 (LoRA M4, LegalSeg) | **0.6707** (with 5 additional IL-TUR `rr` documents in training; 0.5879 without them) | 4 |
| 4 | Court judgment prediction | Macro-F1 | 0.7225 (3 x M4 LoRMA) | **0.8767** (corpus lookup + M4) | 2 |
| 5 | Statute identification, 100 labels | Macro-F1 | 0.3949 (neural ensemble) | **0.4632** (stacker over neural + TF-IDF) | 6 |
| 6 | Summarization | BERTScore F1 | 0.8391 (abstractive reranker) | **0.8606** (Phase-1 trunks as sentence scorers) | 1 |
| 7 | Prior case retrieval | Micro-F1@7 | 0.3325 (LoRMA bi-encoder) | **0.5157** (citator rules + OT decoding) | 2 |

Ranks are from the Codabench leaderboards as of 9 November 2026.

Some of the final submissions are not just neural models. They also use lexical models, rules and lookups from the permitted Phase-1 corpus. I have included both the model-only scores and final submission scores in the phase READMEs so that the difference is clear.

## Layout

```text
capstone_Project/
├── README.md            this file
├── tools/               submission validator + ledger, dev-split builder, parameter counter
├── phase1/              pretraining: tokenizer, corpus, four architectures, decoding, MBR, retrieval
├── phase2/ … phase7/    one folder per downstream task
│   ├── README.md        architecture, method, what worked, how to run the best system
│   ├── EXPERIMENTS.md   full experiment log, including measured results
│   ├── RUN.md / DESIGN.md  run instructions and design notes, where available
│   ├── code/            scripts used in the phase
│   └── submissions/     index of Codabench uploads and scores (zips not in git)
```

The easiest way to understand a phase is to start with its `README.md`. The `EXPERIMENTS.md` files contain more details about the experiments, including things that did not work and the changes made later.

## The Phase-1 foundation in one paragraph

All four models use the same tokenizer (my own byte-level BPE with 32,768 tokens and legal-aware pre-tokenisation), the same 1.255B-token corpus made from the four permitted sources, and mostly the same training setup (Muon on 2-D weights, AdamW elsewhere, QK-norm with z-loss and value residual). The architectures are M1 `LexArc-D`, a GQA decoder (125.9M); M2 `LexArc-ED`, an encoder-decoder with joint next-token and masked-LM objectives (151.0M); M3 `LexArc-L`, a decoder with multi-head latent attention (129.0M); and M4 `LexArc-P`, a GQA decoder with a pointer-generator head (126.0M). `M1_SC` is M1 after four additional hours of Supreme-Court-only pretraining. All the later phases load these checkpoints.

## Common setup

All experiments were run on Google Colab, with Google Drive used for storing the files. The code uses environment variables for paths, so the same scripts can also be run locally after setting them properly.

```python
import os

BASE = '/content/drive/MyDrive/cs781_capstone'

os.environ.update(
    CS781_BASE=BASE,
    CS781_P1=f'{BASE}/phase1',          # Phase-1 checkpoints, tokenizer, corpus shards
    CS781_LIB=f'{BASE}/phase1/shared/lib',  # cs781_bpe.py and model files
    CS781_P2=f'{BASE}/phase2',          # one CS781_P<n> per phase
    HF_TOKEN='<your token>',            # dataset Exploration-Lab/CS-781-Capstone is gated
)
```

The main requirements are Python 3.10+, PyTorch 2.x, numpy, scikit-learn, `bert-score` and `rouge-score` for the official metrics, and `pyahocorasick` for Phase-1 retrieval. The evaluator dependencies are listed in `tools/requirements.txt`.

## Data and integrity

The data used comes from the competition's Hugging Face dataset (`Exploration-Lab/CS-781-Capstone`), the permitted Phase-1 corpus, and the other data sources described in the individual phase READMEs.

Some of the competition datasets are public IL-TUR tasks, where test labels are also publicly available. I kept the training and evaluation data separate while running the experiments, apart from the Phase 3 runs that add IL-TUR `rr` documents to training, which are listed separately (`phase3/README.md`). Each phase also checks the train/dev data for copies of test documents before training and removes any such copies.

Datasets, checkpoints, caches and submission zip files are not included in this repository. They are stored in the author's Google Drive (`MyDrive/cs781_capstone/`).
