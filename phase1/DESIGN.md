# Phase 1 — clean-room design (v2)

Written 2026-10-02. Deadline 2026-10-06 23:59. Design derived from the capstone
spec and the Phase-1 scorer only; no reference to the earlier runs.

## 1. What the metric actually rewards

`evaluate_phase1.py` calls `bert_score.score(pred, ref, lang='en')`. That means
roberta-large, layer 17, **`idf=False`, `rescale_with_baseline=False`**.
Consequences that drive every design choice below:

1. **The usable range is tiny.** Unrescaled BERTScore F1 for any fluent English
   of the right length sits around 0.80-0.82. The leaderboard spans 0.8485 (#1)
   to 0.8338 (#10) — a 0.0147 band. We are optimising the third decimal place,
   not the first.
2. **F1 is symmetric, so length matters twice.** Short outputs destroy recall,
   long ones destroy precision. References are exactly 128 words. Our output
   must be **exactly 128 words**, every time, no exceptions.
3. **No IDF weighting.** Function words, punctuation and legal boilerplate
   count as much as the holding of the case. Register matching is worth more
   than being right.
4. The leaderboard's secondary numbers (ROUGE-L ~0.22, BLEU-4 ~0.10 at the top)
   are *high* for free-form generation. BLEU-4 at 0.10 implies real 4-gram
   overlap, which in this domain means **boilerplate phrasing plus terminology
   lifted from the prompt**. Copying from the prompt is a first-class
   objective, not a side effect.
5. BERTScore **cannot see broken syntax** and will happily reward content-word
   soup. Every decoding decision is gated on dev BERTScore **and** a
   function-word-bigram fluency count. Never one alone.

## 2. Task shape

Input 512 words (~700 BPE tokens with a legal tokenizer), output 128 words
(~180 tokens). This is a **long-read / short-write** task. That asymmetry is
the thing to exploit architecturally — almost nothing about it is "generic
causal LM".

## 3. Shared substrate (built once, used by every model)

| Decision | Choice | Why |
|---|---|---|
| Tokenizer | new byte-level BPE, vocab **32768** | GPT-2 BPE fragments Indian legal vocabulary (`Magistrate`, `Sub-Divisional`, section cites). A legal 32k BPE is ~12-15% denser, so 12-15% more effective tokens per GPU-second, and the 512-word prompt fits in ~700 tokens instead of ~790. 32k costs 25.2M embedding params (tied, counted once); 64k would spend 50M of the 300M budget for a few % more compression. |
| Corpus format | pre-tokenized **uint16 memmap `.bin` shards** on Drive | vocab < 65536 so uint16 is lossless. Zero JSON parsing and zero tokenization inside the training loop. Also deletes a whole class of loader bugs by making the loader four lines long. |
| Mixing | SC 35% / HC 35% / Tribunals 20% / Acts 10% | test prompts are judgment prose. Statutory language in `train_acts.json` is a different register: useful for vocabulary, harmful as a majority. |
| Final anneal | last 15% of steps, LR to 0, SC+HC judgment bodies only | cheap, consistently large gain in the final checkpoint's generation quality. |
| Dev set | 500 held-out windows in exact test shape (512-word prompt / 128-word reference), cut from random offsets, **excluded from training** | without this we cannot tune decoding, and decoding is where the last 0.005-0.010 lives. `tools/make_dev_split.py` already does this. |
| Precision | bf16 on SM80+, fp16 + GradScaler on T4 | T4 is SM75: no bf16, no FlashAttention-2. Probed, not assumed. |
| Attention kernel | `F.scaled_dot_product_attention`, `is_causal=True`, **no materialised mask** | flash kernel on SM80+, mem-efficient on T4. Passing an explicit mask silently drops to the slow path. |
| Compile | `torch.compile(model)`, fixed shapes | 1.3-1.8x on models this size. Requires stable shapes or it recompiles forever. |
| Optimiser | fused AdamW, betas (0.9, 0.95), wd 0.1 (none on norms/embeddings), grad clip 1.0 | |
| Stability | **QK-norm + z-loss 1e-4** | lets us run 3-5x the usual LR without spikes. On a wall-clock budget, learning rate *is* throughput. |
| Logging | accumulate loss on-device, `.item()` once per N steps | an `.item()` per step forces a GPU sync and costs real throughput. |

## 4. The two models

Compared at **iso-compute** (same wall-clock, tokenizer, corpus, optimiser),
not iso-parameter. On Colab the budget is wall-clock, so that is the comparison
that matters. Stated explicitly in the report.

### M1 — `LexGQA-126M`: decoder-only, throughput-maximal

The safe bet. Everything is spent on tokens seen per second.

```
vocab 32768 | d_model 768 | n_layer 16 | n_head 12 | n_kv_head 4 (GQA 3:1)
head_dim 64 | SwiGLU hidden 2048 | ctx 1024 | RoPE theta 10000
RMSNorm pre-norm | QK-norm | tied embeddings | no biases | z-loss 1e-4
params: 100.7M non-embedding + 25.2M embedding = 125.9M   (50M < x < 300M OK)
```

Two things that are *not* boilerplate:

- **Document-boundary attention masking.** Standard packing concatenates
  unrelated judgments into one 1024-token window and lets attention bleed
  across the seam, training the model to condition on noise. We block-mask at
  document boundaries (FlexAttention `BlockMask` if the GPU supports it, else
  one document per window). Test prompts are single coherent passages; the
  training distribution should be too.
- **4 learned register tokens** prepended to every window. Gives attention
  somewhere to dump probability mass instead of hijacking token 0, a known
  small-scale pathology.

### M2 — `LexSeq2Seq-151M`: encoder-decoder, shaped to the task

The differentiated bet, and architecture #2 of the four that §7.3 requires
("Encoder-decoder Transformer").

```
encoder 10 layers, bidirectional, d_model 768, GQA 12/4, SwiGLU 2048
decoder  8 layers, causal self-attn + cross-attn, same width
RoPE, RMSNorm, QK-norm, embeddings tied across enc/dec/output
params: 62.9M (enc) + 62.9M (dec) + 25.2M (emb) = 151.0M   OK
```

Why this is the right second bet and not just a different label:

- The prompt is 512 words we only ever **read**. A bidirectional encoder gives
  every prompt token full two-sided context for the same parameters. A causal
  model cannot do that at any width.
- All three attention calls are flash-friendly: encoder `is_causal=False`,
  decoder self `is_causal=True`, cross-attn unmasked. **No materialised masks
  anywhere**, so we keep the fast kernel. This is why I chose encoder-decoder
  over a prefix-LM, whose mask would have forced the slow path.
- Training windows: 1280 tokens, split at p in {576, 704, 832} chosen per step.
  Encoder gets `[0:p]`, decoder gets `[p:1280]`. Three fixed shapes, so three
  compile variants, then stable. p is centred on the real test prompt length
  (~700 tokens), so inference is in-distribution.

**The honest problem with encoder-decoder here, and the fix.**
Loss only lands on the decoder half, so ~45% of tokens carry gradient against
M1's 100%. At iso-compute that is a real ~2x data-efficiency penalty. So M2
trains on a joint objective:

```
L = CE_decoder(next-token) + 0.3 * CE_encoder(masked-LM on 15% of encoder tokens)
```

The encoder half stops being free context and starts being supervision. Cost is
one extra softmax over ~100 positions per step, under 2% wall-clock. This is the
one genuinely novel piece in the design, and it is deliberately assembled from
two mechanisms that are individually boring, because what cost 44 GPU-hours last
time was a clever-looking pipeline nobody unit-tested.

### Models 3 and 4 (not scoped now)

Decided after M1/M2 report back: a prefix-LM decoder, and either a gated
linear-attention / SSM hybrid or a sparse-FFN (MoE) variant. §8 requires all
four to be genuinely usable in Phase 2, so none can be a throwaway.

## 5. Compute plan

80 units plus PAYG. Two Colab sessions, one model each.

| GPU | units/h | est. tok/s @126M ctx1024 | 8 h of units | tokens |
|---|---|---|---|---|
| T4 (SM75, fp16, no FA2) | ~1.9 | 10-15k | 42 h | — |
| **L4 (SM89, bf16, FA2)** | ~4.8 | 28-35k | 16.6 h | ~0.9B / model |
| A100 40G (SM80) | ~11.8 | 70-90k | 6.8 h | ~1.0B / model |

**Recommendation: L4 for both, ~7 h each (~67 units), PAYG top-up for
generation and the decoding sweep.** L4 and A100 land in the same place on
total tokens; L4 is cheaper per hour, more available, and less painful when a
session dies. ~0.9B tokens on a 126M model is ~7 tokens/param, under
Chinchilla's 20, which is correct when wall-clock is the binding constraint.

Checkpoint to Drive every 20 minutes. Colab sessions die.

## 6. Order of operations

- **Stage A — shared setup, ~50 min, one session, sequential.** GPU probe,
  stream the corpus (one file is 10.67 GB, never `json.load` it), train the
  tokenizer, write uint16 `.bin` shards and the dev split to Drive.
- **Stage B — gates, ~25 min.** Non-negotiable, and the reason this rebuild
  exists:
  1. **Teacher-forcing alignment assert.** `argmax(logits[:, i])` is trained
     against `input_ids[:, i+1]`. Asserted in code, not eyeballed. For M2 the
     same assert runs on the decoder stream.
  2. **Overfit one batch** — 200 steps, one fixed batch, loss must go < 0.1.
  3. **Toy model writes English** — ~2M params, ~5M tokens, 10 min, print 5
     continuations and read them. Crude but recognisable English, or stop.
  4. **Tokenizer round-trip** `decode(encode(x)) == x` on 100 real documents.
- **Stage C — launch M1 and M2 in parallel**, two sessions.
- **Stage D — generation, decoding sweep, submit.**

When something looks wrong, the first test is always **greedy decoding with
every penalty off**, because it isolates the model from all decoding knobs at
once.

## 7. Where the last 0.01 comes from (cheap, after training)

Ranked by value per GPU-minute. All tuned on the 500-example dev split, never
guessed:

1. **Exactly 128 words.** Generate ~170, truncate on a word boundary, never
   emit fewer. Pure accounting, costs nothing, protects recall.
2. **Decoding sweep.** Beam search (4-8, length-normalised) vs low-temperature
   nucleus. Beam is the prior favourite here: BERTScore rewards
   high-probability boilerplate, which is exactly what beam produces. Must be
   measured, and gated on the fluency bigram count, because BERTScore will
   happily reward degenerate output.
3. **Prompt-copy logit bias.** Additive boost to tokens already present in the
   prompt. Zero training risk, one tunable scalar, targets the exact thing the
   high BLEU-4 scores imply. If this is the big lever, model 3 gets a trained
   pointer-generator head.
4. **Checkpoint soup.** Average the last 3-5 checkpoints. Free, usually small
   positive.
5. **Logit ensemble / MBR** across M1 and M3 (same tokenizer) for a late
   submission. Only if the dev split says it is worth a slot.

## 8. What this leaves for Phase 2

§8 requires fine-tuning **all four** Phase-1 architectures, so each needs a
saved tokenizer, config, weights and a loadable entry point. Build that
interface now, not in November.

---

## 9. Revision v2.1 — after the professor's and TA's clarifications (2026-10-02)

Four clarifications arrived after v2 was written. They override §3 and §4 where
they conflict. Recorded here rather than silently edited, because the report
needs the audit trail.

### 9.1 "Use your own BPE implementation, not SentencePiece"

The A-1 / A-3 `BPETokenizer` is reused: byte-level, heap-driven merge selection
with incremental pair-count updates and a per-chunk encode cache. It is already
the right algorithm and it scales.

Two changes are mine and both are measurable, so both are ablations:

1. **Legal-aware pre-tokenization.** A-1 splits on `\s*\S+|\s+`, so
   `AIR 1973 SC 1461`, `S.138/1881` and `(2)(a)` each become single chunks and
   burn vocabulary on citation strings that never recur. The replacement regex
   splits digit runs into groups of at most 3 and separates punctuation from
   word characters. Reported as bytes/token and words/token on held-out legal
   text, against the A-1 regex at equal vocab size.
2. **Vocabulary size swept at 16k / 32k / 48k** on the same held-out sample,
   choosing on words-per-token against embedding-parameter cost.

The scaling problem is encoding, not training: a pure-Python encoder cannot run
over billions of tokens inside a training loop. It does not have to. Chunks are
Zipfian, so the corpus is encoded **once**, offline, through a
`chunk -> token-ids` dictionary built from the unique chunk types, with
`multiprocessing` over the source files, into uint16 `.bin` shards. Training
then never calls the tokenizer at all. This is also what makes the loader four
lines long, which is the real point.

### 9.2 "Your architecture must not be a reimplementation of an existing model"

Inspiration from LLaMA/Mistral/DeepSeek papers is allowed; a renamed copy is
not. Required: manual implementation, meaningful changes of my own, stated
justification, and ablations.

So each model is specified as **backbone + named modifications**, and every
modification gets a 30M-parameter / 50M-token ablation (~20 min each) rather
than a claim.

**M1 `LexArc-D`** — decoder-only backbone (RoPE, GQA, SwiGLU, RMSNorm, SDPA).
My modifications:

| # | Modification | Why, for *this* task | Ablation |
|---|---|---|---|
| D1 | **Value residual** — every layer mixes in layer-1's values through a learned per-layer gate, `v_l = (1-a_l)v_l + a_l v_1` | first-layer value information is destroyed by depth at small scale; restoring it is known to speed early convergence, and we are deeply token-limited | on/off at 30M |
| D2 | **QK-norm + z-loss 1e-4** | not a quality trick — it buys a 3-5x higher stable LR, and on a wall-clock budget learning rate *is* throughput | LR sweep with and without |
| D3 | **4 learned register tokens** | stops attention hijacking token 0 to dump probability mass | on/off at 30M |
| D4 | **Copy-bias attention in the final 2 layers** — an additive, learned score bonus on key positions whose token id matches a token already in the query's context | directly targets the prompt-copying that the leaderboard's BLEU-4 of 0.10 implies the leaders are doing; confined to 2 layers so the other 14 keep the flash kernel | on/off, measured on dev ROUGE-L and BLEU-4 |
| D5 | **Document-boundary masking** in packing | test prompts are single coherent passages; packed training windows should be too | on/off at 30M |

**M2 `LexArc-ED`** — encoder-decoder, as in §4, whose own contribution is the
joint objective: `CE_decoder(next-token) + 0.3 * CE_encoder(MLM on 15%)`. This
is not T5 span corruption and not BART denoising; it exists to recover the
gradient signal that a plain seq2seq objective throws away on the encoder half,
which is the specific weakness of encoder-decoder at a fixed wall-clock budget.
Ablation: lambda in {0, 0.3} at 30M.

**M3 / M4, later.** A-3 already has `MultiHeadLatentAttention`, so M3 is an
MLA-latent-KV decoder (distinct memory mechanism, and a much smaller KV cache
for generating 1000 continuations). M4 is a local/global or linear-attention
hybrid. Both are short runs: they exist to satisfy §7.3 and to leave Phase-2
checkpoints, not to win.

### 9.3 Optimiser: Muon

A-3 has a working `Muon` with `newton_schulz_orthogonalize` and
`build_param_groups_for_muon`. Muon on the 2-D hidden weight matrices, AdamW on
embeddings, norms and the output head.

Reported speedups to equal validation loss are around 1.3-1.5x. On a 7-hour run
that is 2+ GPU-hours recovered for free, which is worth more than any single
architectural change on this list. It is also an honest ablation the professor
asked for. **Verified at toy scale in Gate 1 before any real run** — Muon vs
AdamW on the same overfit-one-batch and the same 10-minute toy model.

### 9.4 External and synthetic data are now permitted

The TA's message permits generic corpora for pretraining, constructed
prompt-response pairs for SFT, preference data for alignment, and synthetic
data, provided the approach is described in the report.

**What I am not doing:** mixing in Wikipedia or C4. There are 4.5B tokens of
in-domain legal text and a budget of roughly 0.9B. Diluting an already
token-starved run with out-of-domain text costs in-domain fit for a fluency
gain we do not need. Worth one 30M ablation, not a design decision.

**What this actually unlocks, in value order:**

1. **Task-format SFT (high value, low risk).** Build (512-word prompt ->
   128-word continuation) pairs from held-in corpus text and fine-tune for the
   last ~10% of budget with loss on the continuation only. This trains on
   exactly the test distribution instead of on generic windows. Previously
   ambiguous under §7.6; now explicitly sanctioned.
2. **Rejection-sampling fine-tuning (RAFT).** Sample k=8 continuations per
   prompt, keep the best, fine-tune on the winners. This is the only technique
   available that optimises the *evaluation surface* rather than a proxy, and
   the leaderboard band is 0.0147 wide.
   - **Selector: ROUGE-L against the gold continuation, plus the function-word
     bigram fluency count.** Both are pure string metrics. No pretrained model
     is involved, so §7.6 is untouched.
   - Using BERTScore itself as the selector would be stronger and is a **rule
     question**, since it means training against roberta-large's judgments.
     Ask the TA before doing it; do not assume.
3. **Synthetic data: skip.** Generating it well needs a model better than the
   one we are training. Noted in the report as considered and rejected, with
   the reason.

### 9.5 Scope, stated plainly

Four and a half days. Four fully-trained architectures plus a complete ablation
suite does not fit, and pretending otherwise is the known failure mode. What
fits:

| | |
|---|---|
| Oct 2 | Gate 0, corpus streaming, BPE training (CPU, runs unattended) |
| Oct 3 | Encode to `.bin`, Gates 1-4, launch M1 and M2 in parallel |
| Oct 4 | Training completes, generation, decoding sweep, submissions 1-2 |
| Oct 5 | SFT/RAFT on M1, submissions 3-4, 30M ablation sweep, short M3 run |
| Oct 6 | Buffer only. Codabench takes 15-30 min per submission and the brief says delays are not excused. |

Ablations run at 30M parameters on 50M tokens, roughly 20 minutes each. That is
the correct scale for them anyway: an ablation at full scale that you can only
afford to run once is not an ablation.

---

## 10. Measured facts (2026-10-02, Gate 0)

Replacing estimates with measurements as they arrive.

### 10.1 Hardware: T4 reports bf16 support and does not have it

`torch.cuda.is_bf16_supported()` returns **True** on a Tesla T4 (SM75). It is
emulated. Measured 8192-cube matmul: **2.2 TFLOPS in bf16**, against a ~65
TFLOPS fp16 tensor-core peak — roughly 30x slow.

Rules that follow:

- **T4 => fp16 + GradScaler. Never bf16.** L4/A100 (SM80+) => bf16.
- `is_bf16_supported()` is not a capability check, it is a compatibility check.
  Benchmark the dtype, do not ask about it.
- SDPA on T4: FLASH unavailable (SM80-121 only), **MEM_EFFICIENT works**
  (2.21 ms at B8/H12/T1024/d64). `flex_attention` imports, speed unverified.

### 10.2 GPU choice

| GPU | realistic tok/s @126M ctx1024 | tokens in 7 h | tokens/unit | tokens/param |
|---|---|---|---|---|
| T4 (fp16, no FA2, ~1.84 u/h) | 8-12k | ~250-300M | ~20M | 2.3 |
| **L4 (bf16, FA2, ~4.8 u/h)** | 30-35k | ~800M | ~24M | **6.4** |
| A100 40G (~11.8 u/h) | 75-85k | ~1.1B in 4 h | ~24M | 8.6 |

Per unit the three are within 20%. Per wall-clock hour L4 is 3x a T4 and the
deadline, not the unit balance, is binding. **M1/M2 on L4 or A100; gates and
30M ablations on T4.**

### 10.3 Test set

1000 examples, ids `"1"`-`"1000"` (contiguous, numeric), **every prompt exactly
512 words**. Confirms the ~700-token prompt estimate, so M1 at ctx 1024 has
~150 tokens of headroom and M2's encoder split points `{576, 704, 832}` are
centred correctly. Test file is `Phase_1_Pretraining/Test/test.json`, not the
`Phase1- Test.json` named in the spec.

### 10.4 Corpus

| file | GB | approx tokens |
|---|---|---|
| `train_high_courts.json` | 10.67 | ~2.4B |
| `train_tribunals.json` | 6.48 | ~1.4B |
| `train_supreme_court.json` | 1.32 | ~0.29B |
| `train_acts.json` | 0.73 | ~0.16B |
| total | 19.20 | ~4.3B |

Supreme Court is smaller than the target 35% mix implies, so the mix becomes
all of SC, all of Acts, and sampled HC/Tribunals, rather than the ratios in §3.

## 11. Storage layout (Drive, survives session death)

```
MyDrive/cs781_capstone/
  shared/
    lib/        cs781_bpe.py, cs781_core.py (Muon, RoPE, MLA from A-3)
    tokenizer/  bpe_legal_32k.json, tokenizer_stats.json
    corpus/     train_*.bin (uint16), val.bin, manifest.json
    dev/        dev_prompts.json, dev_truth.json, holdout_manifest.json
    test/       test.json
  M1_LexArcD/   config.json ckpt/ logs/ gen/ EXPERIMENTS.md
  M2_LexArcED/  config.json ckpt/ logs/ gen/ EXPERIMENTS.md
  ablations/
  submissions/  ledger.json, submission_*.zip
```

One folder per model so two browser tabs never write to the same path.
Checkpoints every 20 minutes, atomic (write `.tmp`, then rename), keep the last
3 plus the best-by-val-loss. Corpus shards written one at a time with a
manifest so a dead session resumes instead of restarting.

---

## 12. Measured facts, round 2 (2026-10-02)

### 12.1 L4 throughput, optimised

Same 125.9M config, bf16, ctx 1024:

| variant | B | tok/s | peak | tokens in 8 h |
|---|---|---|---|---|
| cell-1b benchmark (fp32 logits copy, repeat_interleave K/V, unfused RoPE) | 16 | 20,751 | 21.6 GB | 523M |
| eager, those three removed | 16 | 23,331 | 18.9 GB | 672M |
| **+ torch.compile** | 32 | **35,083** | 21.6 GB | **1,010M** |

A 1.69x end-to-end gain from deleting waste and compiling. The largest single
item was `logits.float()`, which materialised a 2.1 GB fp32 tensor per step on a
card with only 300 GB/s of bandwidth -- less than a T4's 320. L4 is compute-rich
and bandwidth-poor, so fusion matters more here than on a T4.

**Real runs use B=24, not 32.** 21.6 of 24 GB leaves no headroom for periodic
eval and checkpointing, and losing an 8-hour run to an OOM at hour 7 costs more
than the ~6% throughput.

Inductor logs `Not enough SMs to use max_autotune_gemm` (L4 has 58); default
mode is used.

### 12.2 Prefix sampling of an ordered corpus is a biased sample

Cell 2b read the first 20k records of each file and reported High Courts at 431
words/doc. Random-offset sampling of the same file reports **941**. The files
are ordered by year and court, so the prefix was unrepresentative. The mixing
ratio would have been set from that number.

Corrected length distributions (random offsets, n = 27k-62k docs per source):

| source | mean | p50 | p95 | >=640 w | >=900 w |
|---|---|---|---|---|---|
| Acts | 6,138 | 2,711 | 21,298 | 84.2% | 78.4% |
| Supreme Court | 3,722 | 2,462 | 10,632 | 86.1% | 80.2% |
| Tribunals | 1,435 | 700 | 4,700 | 53.3% | 41.3% |
| High Courts | 941 | 323 | 3,526 | 27.3% | 20.6% |

### 12.3 Short High Court documents are the wrong register

28% of sampled HC documents are under 200 words, and inspection shows they are
case headers -- cause number, date of decision, coram, counsel appearances --
not judgment prose. The test prompts are mid-judgment reasoning with citations.

**Corpus filter: documents >= 900 words.** They also cannot fill a 1024-token
window, so nothing is lost on the packing side either.

### 12.4 Records carry metadata

```
SC    title, date, court, citations, bench, headnote, text
Acts  title, category, year, url, text
HC    court, court_id, year, bench, filepath, page_count, word_count, char_count, text
Trib  text
```

- HC ships a precomputed `word_count`: 4M documents can be length-filtered
  without parsing their bodies.
- `filepath` on HC is a dedup key.
- SC carries a `headnote` -- a human summary of the judgment. Not used in Phase
  1; it is directly relevant to Phase 6 (summarisation) and should not be
  forgotten.
- Metadata-conditioned pretraining (prepending court/year with dropout) is
  possible but creates a train/test mismatch, since test prompts are bare
  512-word excerpts. Considered, not adopted.

### 12.5 Mixing ratio: measure the test set instead of guessing

The SC-35/HC-35/Trib-20/Acts-10 ratio in section 3 was intuition. Replaced by a
measurement: a 4-way Naive Bayes classifier over word unigrams, trained on
512-word windows cut from each source, applied to the 1000 real test prompts.
The predicted source composition of the test set sets the training mix and the
dev-split composition. Held-out source accuracy is reported first -- if the
classifier cannot separate the sources, its test predictions mean nothing.

### 12.6 Pre-tokenizer: verdict deferred

On a 20 MB probe, legal-aware vs the A-1 pattern:

| pattern | chunks | distinct types | chars/chunk |
|---|---|---|---|
| A-1 `\s*\S+|\s+` | 3,405,211 | 115,852 | 5.87 |
| legal-aware | 4,034,038 | 52,515 | 4.96 |

Distinct types halve, which is the intent: `Act`, `Act,` and `Act.` stop being
three unrelated types. But chunk count rises 18.5%, because splitting
punctuation and digits creates more pieces, and a chunk the A-1 pattern can
merge into one token may now cost two.

**The decision metric is tokens/word after 32k merges on held-out text, not
chunk statistics.** Both tokenizers get trained; the A-1 variant runs on CPU
for free while the GPUs are busy. If the A-1 pattern wins, it is used.

---

## 13. The fluency gate was fooled by the thing it exists to catch

At 10% of M1's budget the in-run gate reported **9.3 function-word bigrams per
128 words** -- comfortably above the 4.0 abort threshold and above the 8.0
floor used for decoding selection. The sample it printed:

> "...the words used in the clause must be given a restricted meaning and that
> the words used in the clause must be given a restricted meaning and that the
> words used in the clause must b"

A greedy repetition loop. It passed because **a loop manufactures exactly the
bigrams the metric counts**. The detector for degeneracy is inflated by
degeneracy.

Added a second, orthogonal gate: `repeat_rate`, the fraction of 4-grams in a
continuation that are not unique. Measured on the two real samples we have:

| sample | fluency | repeat4 | verdict |
|---|---|---|---|
| looping, M1 @10% greedy | 9.3 | **0.44** | rejected (ceiling 0.35) |
| pilot final, sampled | 12.3 | **0.00** | accepted |

Both gates are now required in `score_and_submit.py`: BERTScore chooses the
decoding setting, but only among settings that clear the fluency floor *and*
the repetition ceiling. The two failure modes are different -- the floor
catches content-word soup, the ceiling catches loops -- and loops are
precisely what pass the floor.

The general lesson, which belongs in the report: a quality proxy built from
surface statistics will be gamed by any degenerate text whose statistics
happen to match. One proxy is not a gate. This is the same shape as the
original BERTScore failure, one level down.

Greedy looping at 10% of training is not itself alarming -- the pilot did not
loop at the end of its schedule -- but the gate that was supposed to notice is
now able to.

---

## 14. Leaderboard calibration (submission 1, 2026-10-02)

Pilot model: 2,250 steps, **55M tokens**, 36 minutes on an L4, decoding
t0.8/p0.9. Codabench, Phase 1, user `n_230709`:

**Rank 15 -- BERTScore 0.8308, ROUGE-L 0.1883, BLEU-4 0.0394.**

### 14.1 The dev split is a trustworthy proxy

| metric | our dev (200 ex) | official test (1000) | offset |
|---|---|---|---|
| ROUGE-L | 0.1927 | 0.1883 | -0.0044 |
| BLEU-4 | 0.0419 | 0.0394 | -0.0025 |

Optimistic by 2-6% relative, and consistently so on both metrics. This is the
single most useful result so far: it means a decoding setting that wins on dev
by more than the bootstrap noise will also win on the leaderboard. Without it
every sweep result would be a guess. Expect dev BERTScore to read ~0.002-0.004
above the Codabench figure.

Why it holds: the dev split was cut from random byte offsets, in the exact test
shape (512-word prompt / 128-word continuation), with source proportions
matched to the classifier's estimate of the test set, and the source documents
excluded from every training shard.

### 14.2 We are losing on the primary metric, not on overlap

| rank | BERTScore | ROUGE-L | BLEU-4 |
|---|---|---|---|
| 1 | 0.8485 | 0.2288 | **0.1057** |
| 9 | 0.8351 | 0.1875 | 0.0542 |
| 10 | 0.8343 | 0.1818 | 0.0441 |
| 14 | 0.8310 | 0.1783 | 0.0345 |
| **15 (ours)** | **0.8308** | **0.1883** | 0.0394 |

We beat ranks 10 and 14 on ROUGE-L while scoring below both on BERTScore, so
the loss is something word overlap cannot see. The prime suspect is length:
**28 of 1000 continuations were under 128 words, minimum 36**, caused by the
legal-aware tokenizer splitting digit runs so a citation-dense passage yields
far fewer words per token than the 1.335 corpus average. BERTScore F1 is
symmetric against a 128-word reference, so short output is a direct recall
hit -- roughly 0.0015 here, which at this leaderboard density is 2-3 places.
Fixed by the sliding-window top-up in `score_and_submit.py`.

### 14.3 Rank 1 is an outlier in a specific, explicable way

BLEU-4 0.1057 against the field's ~0.05 -- double -- while the ROUGE-L lead is
proportionally much smaller. Doubling 4-gram precision specifically means long
exact phrase matches. Given §12 measured that **32% of test prompts appear
verbatim in the provided training corpus**, the most economical explanation is
a model trained long enough on Supreme Court text to reproduce those passages
from memory. Nothing to copy: it is what more in-domain training produces
anyway, and the three-phase mix anneals onto Supreme Court data at the end for
independent reasons.

---

## 15. Results: M1 vs M2, iso-compute (2026-10-03)

Identical tokenizer, corpus, 6-hour wall-clock budget, optimiser and decoding
sweep protocol. Each model evaluated under *its own* best decoding, on 250
held-out dev examples, BERTScore with bootstrap 95% intervals.

| | params | tokens seen | supervised tokens | dev BERTScore | ROUGE-L | BLEU-4 |
|---|---|---|---|---|---|---|
| **M1 `LexArc-D`** (decoder-only) | 125.9M | 586M | ~586M | **0.8402** | 0.2078 | 0.0589 |
| M2 `LexArc-ED` (encoder-decoder) | 151.0M | 680M | ~360M | 0.8306 | 0.1960 | 0.0390 |

**The decoder-only wins by 0.0096** with fewer parameters and fewer raw tokens,
because every token it sees carries gradient. The encoder-decoder's joint
MLM+CLM objective recovered part of the ~47% of tokens a plain seq2seq
objective wastes, and it was not enough. BLEU-4 is where the gap is widest
(0.0589 vs 0.0390), consistent with fewer supervised tokens producing weaker
long-range phrase modelling.

### 15.1 The headline decoding result: prompt-copy bias replicates

| | no copy bias | copy_bias 2.0 | gain |
|---|---|---|---|
| M1 | 0.8374 | 0.8402 | **+0.0028** |
| M2 | 0.8281 | 0.8306 | **+0.0025** |

Near-identical magnitude on two architecturally unrelated models, both outside
the bootstrap noise. Prompt-copying is a general deficiency in how these models
generate, and an untrained uniform logit bias recovers a quarter of a percent
of BERTScore. This was predicted from the leaderboard: rank 1's BLEU-4 of
0.1057, double the field, implied long exact phrase matches.

It also makes the case for a **learned** copy mechanism -- a pointer-generator
head, which can decide *when* and *what* to copy rather than biasing every
prompt token uniformly -- as one of the remaining architectures. That is now an
evidence-backed design decision rather than a hunch.

### 15.2 Greedy decoding is a trap, quantified

| | greedy BERTScore | repeat4 | best sampled | cost of greedy |
|---|---|---|---|---|
| M1 | 0.8178 | 0.62 | 0.8402 | **-0.0224** |
| M2 | 0.8021 | 0.70 | 0.8306 | **-0.0285** |

Greedy also scored the *highest* function-word fluency of any setting on both
models. Without the repetition ceiling, a selector using fluency plus ROUGE-L
-- which is what the training script had -- could have shipped it. The entire
leaderboard spans 0.0188; one decoding knob is worth more than that.

### 15.3 Null result: checkpoint selection barely matters

M1 picked `soup3` (0.8361) over `final` (0.8357) and `best`-by-val-loss
(0.8353). M2 picked `best` (0.8282) over `final` (0.8281) and `soup3` (0.8273).
Opposite winners, all three inside each other's confidence intervals. Averaging
checkpoints and selecting on validation loss are both worth approximately
nothing here. Reported because a null result measured is better than a
convention followed.
