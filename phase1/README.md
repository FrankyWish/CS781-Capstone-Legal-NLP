# Phase 1: legal language-model pretraining

**Task.** Given a 512-word prompt from an Indian legal document, generate the next 128 words.
1,000 test prompts. Primary metric: BERTScore F1 (`roberta-large`); secondary ROUGE-L, BLEU-1..4.
Constraint: 50M to 300M parameters, trained from scratch on the provided corpus, four
substantially different architectures.

**Result.** Rank 1 at **0.8689**, 0.0171 clear of second place. That entry is a hybrid:
308 of 1,000 continuations are retrieved verbatim from the permitted corpus and 692 are
generated. **The generation-only result is 0.8449** (MBR over three models), and that is
the number that measures the models.

## Architecture

| | M1 `LexArc-D` | M2 `LexArc-ED` | M3 `LexArc-L` | M4 `LexArc-P` |
|---|---|---|---|---|
| mechanism | decoder, GQA 12 query : 4 KV heads | encoder-decoder | decoder, MLA (latent 256) | decoder, GQA + pointer-generator head |
| layers | 16 | 10 enc + 8 dec | 16 | 16 |
| parameters | 125.9M | 151.0M | 129.0M | 126.0M |
| file | `cs781_model.py` | `cs781_model_ed.py` | `cs781_model.py --attn mla` | `cs781_model_ptr.py` |

Shared: d_model 768, 12 heads of 64, SwiGLU hidden 2048, RoPE (theta 10,000), RMSNorm,
context 1,024, vocabulary 32,768, tied embeddings. Three modifications over a
LLaMA-style block, each behind a config flag and ablated at 34M parameters (`ablate.py`):
value residual (layer-0 values mixed back in through a learned gate, helps by 0.024 nats),
QK-norm plus z-loss 1e-4 (helps by 0.259 nats, and permits a 3 to 5x higher learning rate),
and 4 register tokens (these turned out to hurt by 0.022 nats and are reported as a mistake).
Muon beat AdamW by 0.486 nats at equal tokens, the largest single effect measured.

M2 trains on `CE(decoder) + 0.3 * MLM(encoder)`, so the encoder half also gets gradient.
M4 mixes `p(y) = g * p_vocab(y) + (1 - g) * sum of attention on earlier copies of y`,
computed in log space (the first probability-space version produced NaN after 700 steps).

## Pipeline

1. **Tokenizer and corpus** (`cs781_bpe.py`, `build_corpus.py`, CPU, 40 min). Own byte-level
   BPE extending the course's A-1 implementation; legal-aware pre-tokenisation splits digit
   groups and punctuation so `Act`, `Act,` and `Act.` share one type. Documents under 900 words
   are dropped (mostly case headers). 1.255B tokens: SC 254.7M, HC 450.0M, tribunals 400.0M,
   Acts 150.0M. A 500-window dev split in exact test shape is held out by document hash.
2. **Data mix.** A Naive Bayes source classifier (86.5% held-out accuracy) put 87% of test
   prompts in the Supreme Court source, so training anneals 35% SC, then 70% SC, then 100% SC
   over the last 10% of steps.
3. **Training** (`train_phase1.py`, `train_ed.py`, L4, 6 h each, about 586M to 680M tokens).
   Correctness gates run before any compute is spent: G1 checks the input-to-target shift, G2
   overfits one batch, G3 checks KV-cached decoding against a full forward pass.
4. **Decoding sweep** (`score_and_submit.py`). Checkpoint (`best`, `final`, weight soup) x
   temperature / top-p / repetition penalty / no-repeat n-gram / prompt-copy bias, scored on
   dev BERTScore with a bootstrap interval. Every continuation is topped up to exactly 128 words.
5. **MBR** (`ensemble_mbr.py`). 6 samples from each of M1, M3, M4; pick the candidate with the
   highest expected BERTScore against the other 17. +0.0068 dev over the best single model.
6. **Supreme Court continued pretraining** of M1 and M3 for 4 h (`train_phase1.py --init`).
   `M1_SC` (full learning rate) is the trunk most later phases start from.
7. **Retrieval** (`retrieve_phase1.py`, CPU, 10 min). An Aho-Corasick automaton over the last
   20 words of each prompt finds prompts that occur verbatim in the corpus; the following 128
   words replace the generated continuation. Six candidate offsets per prompt are tried.

## What moved the score

| change | test BERTScore |
|---|---|
| pilot, 55M tokens | 0.8308 |
| M1, 586M tokens, raw decoding | 0.8359 |
| + decoding sweep | 0.8384 |
| MBR over M1 + M3 + M4 | 0.8449 |
| + retrieval, 40-word needle (152 matched) | 0.8592 |
| + 25-word needle (241) | 0.8659 |
| + 20-word needle (263) | 0.8667 |
| + multiple candidate offsets (308) | **0.8689** |

The four architectures tied (0.8346 to 0.8386 test) at iso-compute. Scaling tokens 10.6x
bought +0.0051. Task-format SFT (`sft_phase1.py`) was null.

## How to run

```bash
python build_corpus.py                                   # CPU, ~40 min: tokenizer, shards, dev split
python train_phase1.py --model M1_LexArcD_full --hours 6.0
python train_ed.py     --model M2_LexArcED     --hours 6.0
python train_phase1.py --model M3_LexArcL --attn mla --hours 6.0
python train_phase1.py --model M4_LexArcP --pointer  --hours 6.0
python train_phase1.py --model M1_SC --init M1_LexArcD_full/ckpt/final.pt --mix sc --hours 4.0
python score_and_submit.py --model M1_LexArcD_full       # checkpoint + decoding sweep, ~35 min
python ensemble_mbr.py --models M1_LexArcD_full:soup.pt M3_LexArcL:best.pt M4_LexArcP:best.pt --k 6
pip install pyahocorasick
python retrieve_phase1.py --split test --words 20 --base MBR/gen/prediction.json   # final hybrid zip
```

Seeds: 1337 for training, 1337/0/7 for corpus, sweep and MBR. The corpus, tokenizer and dev
split are byte-reproducible; training is reproducible up to cuDNN non-determinism.

| file | role |
|---|---|
| `cs781_bpe.py` | byte-level BPE, legal pre-tokenizer |
| `build_corpus.py` | tokenizer training, uint16 shards, dev split |
| `cs781_model.py` | M1/M3 model, Muon optimiser |
| `cs781_model_ed.py`, `train_ed.py` | M2 and its trainer |
| `cs781_model_ptr.py` | M4 pointer-generator head |
| `train_phase1.py` | gates, wall-clock LR schedule, data-mix phases, generation, zip |
| `score_and_submit.py` | checkpoint and decoding sweep |
| `ensemble_mbr.py` | candidate pooling and MBR |
| `retrieve_phase1.py` | corpus retrieval hybrid |
| `foundation_check.py` | held-out SC likelihood of every checkpoint |
| `ablate.py` | 34M-parameter ablations of the shared modifications |
| `sft_phase1.py` | task-format SFT (null result) |

Details and every number: [`EXPERIMENTS.md`](EXPERIMENTS.md). Design rationale:
[`DESIGN.md`](DESIGN.md). All 16 uploads: [`submissions/README.md`](submissions/README.md).
