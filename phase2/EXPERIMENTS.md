# CS-781 Phase 2 — Legal NER, experiment log

Nithin T M · 230709 · `n_230709` · window 2026-10-07 to 2026-10-11

Strict Macro-F1 over 12 labels. A correct entity with a one-character boundary
error scores zero.

---

## 1. What the data forced

`phase2_data.py` measured everything below before any model was trained, and
each number changed the design.

| | |
|---|---|
| train | **70 documents** (fold1 35 + fold2 35), 8,393 entities |
| test | 35 documents (fold3) |
| document size | median 26,640 chars — **7–28× the 1024-token window** |
| tokens | 472,327 total |
| label balance | R.COUNSEL 184 … DATE 1,575 (8.5×) |
| token balance | ~95% of tokens are `O` |
| **alignment ceiling** | **98.33%** of gold spans survive char→token→char |
| fold3 overlap with fold1/2 | **0 of 35** — no retrieval shortcut this phase |

**70 documents** is the fact that shapes the phase. It also means a single
held-out split is unusable for selection: a 15% dev set would hold ~29 instances
of the rarest label, which macro-F1 weights exactly as heavily as the one with
1,575. The split is therefore **2-fold CV on the organisers' own folds**, with
the submitted model refit on all 70.

That refit is worth a lot, and the CV numbers below systematically understate it:
the same configuration scored **0.2941 on CV (35 docs) and 0.3854 on test (70
docs)**. The model is data-starved, not capacity-starved.

### 1.1 Two alignment bugs caught before training

Both would have been invisible — the model would have looked like it was
learning while every entity scored zero.

- Byte-level BPE emits tokens that **straddle a multi-byte character** (routine
  for the Devanagari in this corpus). The first implementation skipped them,
  silently losing text and shifting every later offset. Fixed by mapping each
  byte to its *enclosing* character rather than requiring a boundary.
- `LEGAL_PAT` attaches the leading space to a word, so an entity starting at a
  word boundary decoded one character early — a gold `(7, 29)` came back as
  `(6, 29)`. Strict scoring rejects all of those. Whitespace trimming on decode
  recovers the exact boundary. **That one line is worth more than any modelling
  decision in the phase.**

`LEGAL_PAT` also matches no `_` at all, so `findall` does not tile the text;
`finditer` is used instead and the gap is counted (27 characters over 70
documents).

---

## 2. Results

CV = 2-fold on the organisers' folds. Test = Codabench, model refit on all 70.

| run | model | method | scheme | CV | test |
|---|---|---|---|---|---|
| linear probe | M1_SC | head only | BIO | 0.0558 | |
| **baseline** | M1_SC | full FT | BIO | **0.2852** | |
| lower LR | M1_SC | full FT, lr 5e-5 | BIO | 0.2143 | |
| weaker class weights | M1_SC | full FT, wp 0.25 | BIO | 0.2077 | |
| bidirectional | M1_SC | full FT, no causal mask | BIO | 0.2941 | **0.3854** |
| caption prepend | M1_SC | full FT, header 192 | BIO | 0.2859 | |
| **LoRMA** | M3_LexArcL | multiplicative, r=8 | BIO | 0.2109 | |

Leaderboard at the time of the 0.3854 submission: rank 1 **0.6249**, rank 5
0.4843. Sixth.

### 2.1 Four hypotheses that failed

Recorded because the brief grades design decisions, and a measured refutation is
a decision.

**Frozen representations are enough.** No: 0.0558 against 0.2852. The Phase-1
trunk carries little NER signal on its own and has to move.

**The learning rate is too high.** No. Training loss hit 0.0004, which looked
like memorisation, but at 5e-5 dev F1 was still *climbing* at epoch 10 and the
run scored 0.2143. Selecting the best epoch already handles the late decline, so
the faster schedule reaches a better optimum. 3e-4 stands.

**Class weighting is over-predicting.** Directionally true — the model emits 153
entities per document against 120 in training — but weakening it to `wp 0.25`
scored *worse* (0.2077). Not the lever.

**The weak labels are caption-defined.** APP, RESP and both COUNSEL labels all
score ~0.05 and depend on the "X versus Y" line, which only the first window
contains. Measuring the position distribution refuted it: APP has median 0.153
but **p90 0.890**, and only 30% fall in the first 5% of a document. Prepending
the caption to every window changed nothing (0.2859 vs 0.2852).

Bidirectional attention was the one directional success (+0.009) but sits inside
noise and costs 3× the runtime, because an explicit `attn_mask` disables the
fused causal kernel. Not adopted.

### 2.2 The finding that matters

An overlap-based score was added beside the strict one — a prediction counts if
it touches the right entity at all. The gap between them separates *failing to
find* entities from *failing to bound* them, and those need opposite fixes.

```
            strict    overlap
fold2       0.3019    0.5819
fold1       0.2698    0.5889
```

**The detector is already at ~0.58. Rank 1 on the leaderboard is 0.6249.**
Essentially the entire deficit is span boundaries, not detection. Per label:

| | strict | overlap | gap |
|---|---|---|---|
| R.COUNSEL | 0.100 | 0.654 | **+0.555** |
| PREC | 0.186 | 0.688 | **+0.502** |
| CASENO | 0.249 | 0.706 | +0.457 |
| JUDGE | 0.341 | 0.779 | +0.438 |
| COURT | 0.742 | 0.869 | +0.128 |
| DATE | 0.849 | 0.931 | +0.082 |

Only DATE and COURT — short, format-regular entities — are tightly bounded.
Everything with variable extent is found and mis-delimited.

The direct attack is **BIOES tagging**. BIO never lets the model say "this token
ENDS an entity"; the final token is identified only by the absence of a
continuation on the next one. BIOES adds explicit End and Single tags so the
boundary becomes something predicted rather than inferred. §8.1 lists span
decoding strategy as a legitimate axis of investigation, so this is an
experiment and not a hyperparameter change.

---

## 3. Required experiments

Six methodologically distinct runs across all four Phase-1 architectures, with
LORMA and knowledge distillation compulsory.

| # | model | technique | status |
|---|---|---|---|
| 1 | M1_SC | full fine-tuning | done |
| 2 | M1_SC | linear probe (head only) | done |
| 3 | M3_LexArcL | **LoRMA** (compulsory) | done |
| 4 | M3_LexArcL | LoRA | queued |
| 5 | M4_LexArcP | BitFit | queued |
| 6 | M4_LexArcP | **knowledge distillation** (compulsory) | done |

BitFit needed adapting: the Phase-1 blocks are bias-free by design (11 ×
`bias=False`), so pure BitFit would have nothing to train. It trains RMSNorm
gains instead, which is the equivalent sparse knob in this architecture.

### 3.1 LoRMA, implemented from the paper

LoRMA (Bihany, Patel & Modi 2025, arXiv 2506.07621) replaces LoRA's additive
reparametrisation with a multiplicative one:

```
LoRA    h = (W0 + (alpha/r)·BA) x
LoRMA+  h = (I_d + (alpha/r)·BA) · W0 x          eq. 8
```

The low-rank term transforms the layer's *output*, so `A` maps from the output
dimension rather than the input one. Two details carry the method, and both are
implemented:

**Rank inflation.** `BA` has rank at most r, so a bare multiplicative update
collapses the pretrained weight. Verified numerically: at the π-variant's
initialisation `rank(BA) = 1`, and after inflation `rank = 48 of 48`.

**Operation reordering.** Forming `(I + BA)W0` explicitly costs O(d²k); the
identity distributes, so it evaluates as `y = W0 x` then `y + (alpha/r)·B(A·y)`
— O(dkb), the same as LoRA.

Both variants initialise to exactly the identity, so fine-tuning begins at the
pretrained function. LoRMA-π was verified to produce the identity at max
deviation `0.00e+00`.

**Result: 0.2109 CV against full fine-tuning's 0.2852**, with overlap 0.4841
against 0.5819. At 0.17% of parameters the *detector* is weaker, not only the
boundaries. That contradicts the prior expectation that parameter-efficient
methods would win on 70 documents: at this data size the trunk still has to move.

### 3.2 Knowledge distillation

Teacher: the fine-tuned 126M model. Student: **every second layer kept** — the
DistilBERT construction. Half the depth, teacher embeddings and final norm,
student layer *i* initialised from teacher layer *2i*. The phase forbids any
pretrained weights but our own, and training a small model from scratch on 70
documents would be hopeless, so the student has to be derived from the teacher.

```
L = alpha·T²·KL(student/T || teacher/T) + (1 − alpha)·CE(student, gold)
```

The T² factor restores the gradient magnitude that dividing logits by T removes
(Hinton et al. 2015, §2.1). Masked positions are excluded from both terms —
imitating the teacher on padding is free loss reduction and no learning.

The script reports strict F1, overlap F1, parameter count and measured
tokens/second for teacher and student, because §8.2.2 asks for the efficiency
trade-off rather than just a score.

**Result.** The teacher is M4_LexArcP, not M1_SC: §4.2 had already shown the
Phase-1 likelihood ranking was wrong, and distilling from the weaker trunk
would have measured the wrong thing.

The run that produced `submission_distil.zip`:

| | strict CV | overlap | params | tok/s |
|---|---|---|---|---|
| teacher | **0.3231** | 0.607 | 125.9M | 79,986 |
| student | **0.2849** | 0.575 | 75.5M | 154,740 |

**88.2% of strict F1 retained at 60% of the parameters and 1.94x the
throughput**, and the folds agree closely on it: 88.0% on fold2, 88.5% on
fold1.

An earlier run of the identical configuration gave teacher 0.3245, student
0.2819, 86.9% retained, and split 90.4% / 83.2% across the two folds. Same
script, same seeds exposed, different shuffle order. The retention ratio moves
by 1.3 points and the per-fold spread by seven between runs, which bounds how
much any single distillation number here should be trusted and matches the
0.02 spread seen across the three M4 seeds in the ensemble. The conclusion
that survives both runs is the one worth reporting: roughly 88% of strict F1
for 60% of the parameters and double the throughput.


---

## 4. The ensemble, and the base model being wrong

### 4.1 Results after the full sweep

| run | CV | note |
|---|---|---|
| **ensemble of 4 architectures** | **0.3606** | overlap 0.606 / 0.629 |
| M4_LexArcP full | 0.3294 | **best single model** |
| M3_LexArcL full | 0.2956 | |
| M1_SC full | 0.2938 | the baseline everything was tuned on |
| M2_LexArcED full | 0.2920 | |
| M1_SC + fragment merge | 0.2897 | no effect |
| M1_SC unweighted loss | 0.2567 | worse |
| M1_SC + CRF | 0.2601 | worse, and 3x slower |

### 4.2 The base model was wrong the whole time

**M4_LexArcP beats M1_SC by +0.05 as a single model**, and every experiment
before the ensemble was run on M1_SC. That choice came from Phase 1's
`foundation_check.py`, which ranked trunks by validation loss on held-out
Supreme Court text and put M1_SC first.

The caveat written into that script at the time turns out to be the load-bearing
sentence: `force_gate=1.0` scores M4's vocabulary head as a standalone LM, which
it was never trained to be, so the measurement **could not** separate trunk
quality from head contribution and said so. Ranking trunks by language-model
likelihood simply does not predict which one fine-tunes best for token
classification. The only reliable instrument was running the downstream task.

### 4.3 Why ensembling worked where everything else failed

Eight single-model interventions were tried and seven lost: BIOES, lower LR,
unweighted loss, reduced class weighting, caption prepending, fragment merging,
and a CRF. Averaging four architectures gained **+0.031 over the best member and
+0.067 over the baseline**, and lifted *overlap*-F1 as well as strict -- so it
improved detection, not just span boundaries, which is the ceiling everything
else was stuck under.

Emissions are averaged as **log-probabilities**, not logits: the models calibrate
differently and a larger logit scale would otherwise dominate the average
regardless of confidence. Averaging happens before decoding because decoded
spans cannot be averaged -- two models disagreeing by one token give
incompatible spans and a vote between them is arbitrary.


---

## 5. The boundary diagnosis, and what the data allows

Ensembling stopped paying: 4 members 0.3606 CV / 0.4435 test, 6 members 0.3674
/ 0.4522, 6 with M4 tripled 0.3751 / 0.4608. **+0.009 per doubling.** The
CV->test map is tight at x1.23 across all three, so the extrapolation is
reliable and it says this axis finishes near 0.48. Rank 1 was 0.6249.

The number that reframes the phase was already on the page and had not been
read correctly:

```
test strict 0.4608      test overlap ~0.77
```

Every entity is being FOUND and then mis-delimited. If the spans already
detected were bounded exactly the score would be 0.77, above rank 1. The
deficit is not detection, and no amount of ensembling addresses it.

### 5.1 What the boundary errors actually are

`boundary_report` splits each inexact span by which edge moved. Across four
runs the split is ~38% start, ~41% end, ~21% both -- near-even, which
**refutes** the hypothesis that causal masking is the cause. A decoder blind to
its right context would fail predominantly at the end edge.

Reading the delta table correctly matters, and the legend was wrong until now
("+ means the predicted span is wider" is true of the end delta and backwards
for the start delta). `start+9 end+0` means the prediction *begins nine
characters late*; `start+0 end-9` means it *stops nine characters early*. Those
are the two largest cells at 15% and 18%, and **both are truncation**:

```
'ACHUTANANDA PUROHIT'          -> 'CHUTANANDA PUR'
'SREE SREE ISHWAR SRIDHAR JEW' -> 'SREE SREE'
'BHAGWATI, NATWARLAL H.'       -> 'BHAGWATI,'
'26/03/1976'                   -> '6/03/1976'
```

The opposite failure exists too and is structural rather than noisy: BIO cannot
separate two adjacent entities of the same label, so a BENCH line collapses --
both `'KRISHNAIYER, V.R.'` and `'CHANDRACHUD, Y.V.'` come back as
`'KRISHNAIYER, V.R. CHANDRACHUD'`. This is the known BIO limitation that BIOES
exists to fix, and BIOES did not fix it here.

### 5.2 Three measurements that decided the intervention

| | measured | verdict |
|---|---|---|
| cross-document string transfer | 19.5% / 12.5% | **dead end** |
| in-document repetition | **67.4%** | the lever |
| gold spans starting/ending mid-word | **1.66%** | widening is safe |

**Cross-document fails for a structural reason, not a tuning one.** Transfer is
83%/70% on COURT, 29%/13% on STAT, 25%/19% on AUTH -- and **0.0-1.1% on CASENO,
APP, RESP, A.COUNSEL, WIT and PREC.** A gazetteer built from training gold
repairs the labels that already score well and cannot touch the six destroying
macro-F1. The shipping gazetteer says why: 971 distinct DATE strings, 791 PREC,
492 CASENO, roughly one unique string per entity. Those labels are
document-specific. A gazetteer alone scores 0.1575 / 0.1218 strict.

**The ceiling that reframes the task.** Building a gazetteer from each
document's own gold strings and tagging that document by string matching
alone -- no model -- scores:

```
strict macro-F1 0.9863,  precision 1.000 on ten of twelve labels
```

So the task is not "tag every token correctly". It is **"identify each distinct
entity string once per document, then copy it to every occurrence"**, and the
copying step is nearly lossless. That is the strongest single finding of the
phase and it should have been measured on day one, before the eight failed
modelling interventions.

### 5.3 The three transforms

Each targets a measured error mode rather than a hypothesis.

1. **word-snap** -- widen every predicted span to the enclosing word. Gold
   spans are word-aligned; predicted spans need not be, because BPE tokens are
   sub-word, which is exactly how `'ACHUTANANDA PUROHIT'` returns as
   `'CHUTANANDA PUR'`. Cost measured per label: 1.66% overall, but **6.76% on
   PREC** against <=1.7% everywhere else, since a citation legitimately stops
   mid-token. PREC is exempted on that evidence.
2. **longest-wins expansion** -- replace a span with the longest other
   predicted form of the same label containing it at that position, capped at
   48 characters. The first implementation weighted by frequency and got this
   exactly backwards: the clipped form is the COMMON prediction and the full
   form the rare one, so a majority vote spreads the error.
3. **propagation** -- also tag occurrences the model missed, for strings
   predicted at least N times. Justified by the 1.000 oracle precision, but
   that was measured with gold strings; with model strings it raises emission
   from 139 to 155 spans/document against ~120 in gold, so it is submitted
   separately rather than assumed.

Transforms 2 and 3 pull against the over-merging in §5.1 -- expansion will
happily grow a correct span onto a merged one. The asymmetry to exploit is that
a merged span typically occurs once (the BENCH line) while a correct full form
repeats, so raising `min_support` should kill merge targets preferentially.
Rather than argue it, `phase2_ner.py` now dumps held-out CV predictions and the
grid is searched offline on CPU, ranked by the **worse** fold: 72 configurations
over 70 documents will otherwise just fit the split.


---

## 6. Domain patterns, and the submission ladder

### 6.1 The ceiling that ended the post-processing work

Per-label overlap-F1 is, by construction, the score a label would reach with
perfect boundaries. At the best post-processed setting:

```
DATE 0.950  COURT 0.828  STAT 0.784  JUDGE 0.699  PREC 0.699  R.COUNSEL 0.606
CASENO 0.578  A.COUNSEL 0.561  AUTH 0.455  WIT 0.365  APP 0.301  RESP 0.285
```

Mean **0.593**. Perfect boundary repair on the entities currently detected
caps below 0.6, so no further post-processing could reach the leaderboard
head. The bottom four are not boundary failures at all -- APP, RESP, WIT and
AUTH are simply not being found, and they are 4/12 = 0.33 of the metric.
That reframed the remaining work from fixing edges to raising detection.

### 6.2 Twenty-six patterns, each measured before use

`phase2_rules.py --validate` scores every rule against all 8,393 gold spans
and reports exact precision, overlap precision and recall. Rules are gated on
measured exact precision, so a pattern cannot overwrite a model boundary
unless it has earned the right. Six of twenty-six failed and were gated out:

| rule | exact P | why it failed |
|---|---|---|
| PREC `X v. Y` | 0.066 | overlap **0.797** -- finds the citation, almost never its extent |
| CASENO `YYYY AIR N` | 0.134 | matches precedent citations, not case numbers |
| WIT `witness/deposed X` | 0.011 | 94 fires, 1 correct |
| AUTH `<Name> Board` | 0.232 | over-fires on ordinary capitalised pairs |
| STAT `<Name> Act` | 0.271 | 1,840 fires against 1,260 gold spans |
| JUDGE `HON'BLE JUSTICE X` | 0.308 | superseded by the BENCH splitter |

Two corrections were worth more than any new pattern. `JUDGE BENCH: line`
scored exact 0.174 at overlap 0.913 -- the region was right and the span was
not, because one BENCH line holds several judges and came back as a single
entity. That is the same adjacent-entity merge the model makes and that BIO
cannot express. Splitting on the surname-comma-initials boundary took it to
**0.857**. Separately, the five rules added in the last round were seeded at
precision 1.000 so the gate could not hide them before scoring -- which meant
no threshold filtered them, and the useless WIT rule depressed WIT from 0.567
to 0.396 in every row of that run until the real numbers were recorded.

### 6.3 Repetition is the lever, not the gazetteer

| | measured |
|---|---|
| cross-document string transfer | 19.5% / 12.5% |
| in-document repetition | **67.4%** |
| per-document string matching on gold strings | **0.9863 strict, P=1.000** |

A rule that fires once per document with high precision therefore covers the
document, and the rules with precision 0.7-0.8 all had recall near 0.1 because
they fire at the caption and stop. Propagating each match case-insensitively
over its other occurrences lifted rules-alone from 0.2265 to 0.3012; the
generalised caption parser and the AUTH lexicon took it to **0.3788**.

The caption parser is the clearest instance. The original rules needed a
literal `PETITIONER:` or `..Appellant` and fired on 37 of 70 documents. Every
caption has a versus token and the formats differ only around it, so matching
the capitalised run on each side of the first versus token in the document
head covers all six observed layouts. Case-insensitive propagation matters
here specifically: the caption writes `JAGE RAM & ORS.` and only the body
mentions `Jage Ram` are annotated.

### 6.4 The ladder, and three failures

| submission | test | |
|---|---|---|
| M1_SC bidirectional | 0.3854 | the configuration everything was tuned on |
| ens4 / ens6 / ens6_m4 | 0.4435 / 0.4522 / 0.4608 | **+0.009 per doubling** |
| + cross-doc gazetteer | 0.4606 | **-0.0002** |
| + longest-wins expansion | 0.5066 | |
| + word-snap | **0.5200** | |
| + occurrence propagation | 0.5142 | **-0.0058** |
| + rules, snap only | 0.5298 | |
| + rules, snap and add | 0.5334 | |
| + replace DATE | 0.5337 | +0.0003 |
| + replace CASENO DATE WIT | 0.5062 | **-0.027** |

Three interventions lost, and each was predicted or diagnosed rather than
discovered:

- **Propagation on the model's own spans** was predicted to fail before
  submission, because it raised emission from 139 to 155 spans/document
  against ~120 in gold and the model already over-predicts. It lost 0.0058.
- **The cross-document gazetteer** transfers at 83%/70% on COURT and
  0.0-1.1% on CASENO, APP, RESP, A.COUNSEL, WIT and PREC. It repairs the
  labels that already score well and cannot touch the six that do not.
- **Per-label replacement** was chosen from single-model CV numbers while the
  submitted system was a six-model ensemble plus post-processing. The proxy
  said +0.009; the real system lost 0.027. The lesson is specific: the
  measurement and the submission have to be the same system, which is why
  both `phase2_ner.py` and `phase2_ensemble.py` now dump held-out predictions
  and the whole pipeline is scored on CPU before anything is uploaded.

### 6.5 Per-label arbitration

Macro-F1 is the mean of twelve per-label F1 scores and a label's F1 depends
only on that label's spans, so choosing the source **per label** is exactly
optimal rather than greedy. Four sources are available for each -- model,
model snapped to rules, snapped and added, rules alone -- where every
experiment above forced one global choice across all twelve. Each label is
also decided on each fold separately, and a label whose folds disagree is
returned to the model, because a 12x4 search over 70 documents will otherwise
fit the split.

### 6.6 This is a hybrid system

As in Phase 1, the graded number is not the model's. Both are reported:

| | |
|---|---|
| ensemble, model only | **0.4608** |
| with boundary post-processing and domain rules | **0.5337** |


---

## 7. Two refutations that closed the phase

Both were run because the alternative was to keep buying +0.01 from
post-processing. Each cost under an hour and each produced a clean negative,
which is the point: they bound what was left rather than leaving it open.

### 7.1 The tokenizer is not the constraint

Every boundary failure in this phase looks like an ALL-CAPS name shredded into
rare subword pieces -- `'MUKHERJEA, B.K.'` returned as `'EA'`,
`'KRISHNAIYER, V.R.'` as `'NAIYER, V.R.'`, `'BHAGWATI, NATWARLAL H.'` as
`'WARLAL H.'`. The Phase-1 BPE was trained on running legal prose, which is
mixed case, so the obvious hypothesis was that upper-case text shares few
merges with it and costs several times the tokens -- making the labels that
live in ALL-CAPS caption blocks undertrainable by construction. Case folding
would have fixed it without retraining anything, since case folding does not
move character offsets.

`phase2_tokcheck.py` measured it:

```
ALL CAPS     0.457 tok/char   4.55 tok/span
Mixed case   0.280 tok/char   5.83 tok/span
929 ALL-CAPS gold spans: 4226 tokens as written, 4539 title-cased -> 0.93x
correlation(tokens per character, ensemble F1) = +0.375
77.8% of the 32,768 vocabulary entries contain an upper-case character
```

**Upper case tokenizes more efficiently, not less**, and the correlation has
the opposite sign to the prediction. The decisive case is WIT: **0.0%
ALL-CAPS and the worst label in the phase at 0.056.** Case has nothing to do
with it. The shredded names in the error log are a symptom of a weak detector,
not of the vocabulary.

### 7.2 Class weighting was already at its optimum

Earlier, `--weight-pow 0.25` scored 0.2077 against 0.2852 and this file
recorded "not the lever". That conclusion was reached from one side only: the
weaker direction was tested and the stronger one never was. Since macro-F1
weights WIT (187 spans) exactly as heavily as DATE (1,575), and three of the
four worst labels are among the four rarest, stronger weighting was the
direction with a reason behind it.

| `--weight-pow` | CV macro-F1 |
|---|---|
| 0.25 | 0.2077 (M1_SC) |
| **0.50, default** | **0.3258** |
| 0.75 | 0.3219 |
| 1.00 | 0.3094 |

Monotone decline, and the per-label numbers show why: heavier weighting buys
recall and loses more precision than it gains. WIT on fold1 moves from R=0.269
to R=0.463 at `wp 0.75` while precision falls to 0.100, and its F1 ends at
0.164. The default was already optimal. Tested in both directions now, which
is how the earlier claim should have been made.

### 7.3 Where the ceiling actually is

Per-label overlap-F1 is the score each label would reach with perfect
boundaries. Scaled to test by the measured CV->test ratio of 1.23:

```
STAT ~1.0   DATE ~0.98  COURT ~0.98  CASENO ~0.96  JUDGE ~0.96  PREC ~0.89
R.COUNSEL ~0.74  A.COUNSEL ~0.72  AUTH ~0.49  APP ~0.43  RESP ~0.40  WIT ~0.35
```

Mean **~0.74**. Four labels cap below 0.5 because the detector does not find
them, and no post-processing can raise a ceiling. Reaching the leaderboard
head would require a better detector, which means a stronger Phase-1 trunk --
70 labelled documents against a 126M model pretrained from scratch is the
binding constraint of this phase, and it was fixed before Phase 2 began.

The honest summary of the phase: **ensembling bought +0.009 per doubling and
stopped; post-processing bought +0.073 and stopped; the detector is the
limit.** Everything that moved the score after 0.4608 was CPU work costing no
GPU time at all.

### 7.4 Final per-label arbitration

Measured on held-out ensemble predictions, with the rules column divided by
1.23 to correct for the CV model seeing 35 documents against the submitted
model's 70 -- the correction that three leaderboard failures paid for.

```
model raw                   0.3755
+ wordsnap + longest-wins   0.4411
+ rules (snap only)         0.4550
+ rules (snap and add)      0.4817
+ per-label arbitrated      0.4934
```

Merging was selected for PREC, RESP, STAT and APP, confirming fragmentation:
the ensemble emits 2.03 PREC spans and 2.39 CASENO spans per gold entity while
scoring 0.74 overlap, so one entity is covered by several predicted pieces,
each counting as a false positive while the entity itself counts as a miss.
Only WIT went to rules alone; six labels tripped the fold-disagreement guard
and fell back to the safer source.


---

## 8. The last day: from 0.5577 to 0.6515

Starting point on 2026-10-11: weak-supervision ensemble plus post-processing
at **0.5577** test, with the InLegalNER variant (`ext6`) at 0.4873 raw.
Leaderboard rank 1 was 0.7005 at the time. Finishing point: **0.6515**, from a
system that trains nothing new over the morning's models -- it combines them.
That was rank 3 when submitted and **rank 6 at the close** (section 8.9): four
entries above it landed in the last seven hours, which is the same pattern
Phase 1 showed and the reason a target set against a mid-window board is
always too low.

### 8.1 The problem the day started with

```
                CV       test     ratio
plain ensemble  0.3739   0.4610   1.228
+ weak          0.4395   0.4847   1.103
+ InLegalNER    0.4765   0.4873   1.023
```

CV gains had stopped transferring. CV models train on 35 gold documents and
the submitted model on 70, so extra training data stands in for gold that the
CV model lacks and the submitted model already has. Adding data was no longer
buying test score. The rest of the day worked on *combining* systems rather
than training better ones.

A tooling gap had to close first: no run saved its fine-tuned weights or
outputs, so every idea cost a full retrain. `phase2_weaksup.py` now saves
every ensemble member's per-token log-probabilities (CV and test, under two
window rules) and reuses them on a rerun, so a crash costs one member, not the
run. `--skip-cv` / `--cv-folds` split one run across two GPUs.

### 8.2 Exact-span voting across finished systems

Strict scoring makes exact agreement meaningful. Two systems built on
different weak data that emit the same `(start, end, label)` are right far
more often than either alone. `phase2_spanvote.py` over the three best
post-processed systems (`wsens6` 0.5577, `ext6` 0.5527, `ens6_m4` 0.5337):

| rule | spans/doc | test |
|---|---|---|
| best single system | 140.4 | 0.5577 |
| **exact 2 of 3** | 129.6 | **0.5763** |
| exact 3 of 3 | 97.9 | 0.5626 |
| 2 of 3 by *overlap*, base system's extent | ~138 | 0.5631 |

The overlap row is the informative one. Where systems find the same entity
and disagree on its boundaries, the base system's boundary is usually wrong.
Exact agreement works as a **boundary-correctness filter**, and that is where
the +0.019 came from.

### 8.3 Stacking: a classifier over candidate spans

A fixed k treats DATE (every system bounds it correctly) and APP (none do)
alike. `phase2_stack.py` learns the vote instead. Every span any system emits
is a candidate, and a gradient-boosted classifier predicts whether that exact
span is gold. Its features are counts, not model probabilities:

- for each system: exact match, same-label overlap, any-label overlap
- how often the string is emitted elsewhere in the document
- competing extents, and rule and gazetteer hits
- shape: length, case, digits, trailing punctuation

It is trained on the held-out CV predictions (`cvpred_*.json`) of the *same*
systems, post-processed through the exact shipped chain with a held-out
gazetteer, and cross-fitted: fold1's classifier trains on fold2's candidates
and the reverse. The test model trains on both. Selection is by the worse
fold. Probabilities were avoided on purpose: they come from 35-document CV
models and 70-document test models, and calibration is exactly what moves
between those.

**The jump came from post-processing stages.** Each system enters three times:

- raw model output
- after word-snap and gazetteer (`scwg`)
- after rules (`scwg_rpadd`)

Post-processing moves extents, sometimes breaking one the raw model had right.
As separate inputs, the classifier learns which stage to trust for which kind
of span.

| | inputs | candidate ceiling | CV worse fold | test |
|---|---|---|---|---|
| exact 2-of-3 vote | 3 | – | 0.5590 | 0.5763 |
| stack3b | 3 systems | 6,009 / 8,393 (72%) | 0.5939 | 0.5847 |
| **stack9** | 3 systems × 3 stages | 6,655 (79%, with expansion) | **0.6155** | **0.6421** |

The candidate ceiling is the share of gold spans that some candidate matches
exactly, which caps any selection. Stack9 is the only change of the day whose
test gain (+0.057) *exceeded* its CV gain (+0.022). Per label against the
vote, the largest stack9 gains were R.COUNSEL +0.084, CASENO +0.083, RESP
+0.068 and JUDGE +0.056. APP moved +0.006.

### 8.4 Output volume, calibrated on the leaderboard

Thresholds chosen on CV landed too high for test. Holding stack9's classifier
fixed and moving only the threshold:

| spans/doc | 119.2 | 124.9 | 132.0 | **136.0** | 140.0 |
|---|---|---|---|---|---|
| test | 0.6268 | 0.6421 | 0.6425 | **0.6436** | 0.6384 |

Flat from about 125 to 136, falling on either side. Gold averages ~120 per
document, so the optimum over-emits. The likely mechanism is that test
candidates come from 70-document models and are better than the CV
candidates the classifier learned from, so it underrates them.
`--rates N` now writes zips whose threshold yields N spans/doc on test. This
uses no test label, only output volume, but the target (136) was **chosen by
leaderboard feedback**, and that is reported as such.

### 8.5 What rose on CV and fell on test

Every change below improved CV, usually on fold1 alone, and lost on the
leaderboard. Rows marked "matched" were also re-scored at the same output
volume, to separate the change from its threshold.

| change | CV worse fold | CV mean | test | matched |
|---|---|---|---|---|
| stack9 (reference) | 0.6155 | 0.6164 | 0.6421–0.6436 | |
| + context features (caption side of "versus", neighbouring cue words) | 0.6171 | 0.6247 | 0.6177 | 0.6316 @132 |
| + context + label swap (APP/RESP/WIT, A./R.COUNSEL) | 0.6141 | 0.6276 | 0.6122 | |
| + extent perturbation + span-shape features | 0.6113 | 0.6246 | 0.6284 @128 | |
| + bagging (7 document-bootstrap classifiers) | 0.6236 | 0.6247 | 0.6236 | 0.6306 @132 |
| + 4th model `ext6e`, mass decoder (stack12) | 0.6268 | 0.6280 | 0.6322 @136 | |
| + 4th model, six decoders × 3 stages (stack27) | 0.6269 | 0.6304 | 0.6318 @136 | |
| merged-fragment copies of each system | 0.6117 | 0.6139 | not submitted | |
| per-label volumes from the training label mix | −0.0022 / −0.0035 vs global | | not submitted | |

Three readings, each supported by more than one row:

1. **A gain on one fold is not a gain.** Context features, label swap and
   perturbation each lifted fold1 by 2–3 points and left fold2 flat, and all
   three lost on test. The worse-fold rule held every time it was tested.
2. **Features tied to document layout or model calibration do not cross the
   35 → 70 document gap.** Agreement counts do. The context cues encode where
   things sit in fold1/fold2 documents. For the 4th model the cause is not
   established. One candidate: its CV members are early-stopped on the
   evaluated fold while its test members train a fixed six epochs, and the
   mass decoder it brings depends on exactly that calibration.
3. **Bagging improved both folds on CV (+0.008 each) and still lost 0.012 on
   test at matched volume.** Balanced CV improvement is necessary, not
   sufficient.

Two results stand apart. The extent-perturbation generator did its own job:
it raised the candidate ceiling from 79% to 85% (PREC 0.547 → 0.691, CASENO
0.675 → 0.807), recovering 16 of 16 real boundary misses from the error
analysis. The classifier could not pick the right variant without features
that overfit. And the two-stage span decoder (entity-vs-O first, label per
span second) was a real decoding gain on `ext6e`: worse-fold CV 0.5542 →
0.5697, with both folds up and the gains on A.COUNSEL (+0.067) and
R.COUNSEL (+0.048). It did not survive stacking on test.

Of the inference-only decoding ideas tested on saved emissions, best-context
window selection gave nothing over plain averaging, and the document-level
string vote lost 0.003–0.014 on CV.

### 8.6 Precision trim: the final submission

The leaderboard's secondary metrics put the gap to ranks 1–2 in what look
like precision (0.606 against 0.673–0.690). On what looks like recall it is
level with rank 2 (0.690 against 0.693) and below rank 1 (0.736). The final step trims the best system's own spans: keep a stack9 span
only if at least one of the two 4-system stacks also emits it exactly.

| | spans/doc | test |
|---|---|---|
| stack9 at 136 (`s9x_r136`) | 136.0 | 0.6436 |
| **trim, backed by stack12 or stack27 (`s9trim`)** | 127.5 | **0.6444** |
| trim plus near-unanimous adds from 6 stacks (`t3`) | 135.6 | 0.6440 |
| trim applied to a 152-spans/doc base | 131.2 | 0.6376 |

+0.0008 is inside the noise of a 35-document test set and is reported as a
tie, not a gain. The last row is informative: spans below stack9's own
threshold, even when backed by the other stacks, were mostly wrong.

### 8.7 External data and self-training: what was used

The professor relaxed §17.2 on 2026-10-10: external and synthetic data may be
used at any training stage if described in the report.

**InLegalNER** (Kalamkar et al., NLLP 2022). Ten of its fourteen types map
onto ours. `LAWYER`, `OTHER_PERSON`, `GPE` and `PROVISION` are masked (−100),
not taught as O. Before any training, every record was tested against fold3
by rare-13-gram containment and dropped at ≥50% overlap. That removed 11
records that copy test passages verbatim, which a document-level check had
rated at 0.4%. It is used through the systems `ext6`, `ext6e` and `ext6st`.

**Self-training on the fold3 text (`ext6st`).** Pseudo-labels are the spans
on which two finished systems (`wsens6`, `ext6`) agree exactly on the test
documents: 3,929 of 4,914, all twelve labels present. They were added as 472
partially supervised windows to the *gold* phase, alongside 891 gold windows.
Everything outside a pseudo-span and an 8-token margin is masked. Only the
test documents' text and this system's own predictions on it are used; no
test annotation exists anywhere in the pipeline.

Effect: alone, one decoder at the `scwg` stage scored **0.5644**, above every
earlier single system (best 0.5577 with full post-processing). It emits 131.8
spans/doc against 138.1 for the same configuration without pseudo-labels.
Swapped into a stacker trained on another model's CV outputs it fell to
0.6053, because the classifier had learned the other model's habits. It is
not part of the final submission.

**Best trimmed stack (`s9trim`, 0.6444)** combines: `wsens6` (rule-labelled
Supreme Court / High Court / tribunal text), `ext6` (+ InLegalNER) and
`ens6_m4` (gold only), stacked at three post-processing stages. It is trimmed
by support from two stacks that also include `ext6e` (+ InLegalNER).
Self-training enters none of them.

### 8.8 This phase, honestly

| | test |
|---|---|
| best ensemble, model only | 0.4873 |
| + boundary post-processing and domain rules | 0.5577 |
| + exact-span voting | 0.5763 |
| + stacking over systems × stages | 0.6421 |
| + output volume tuned on the leaderboard | 0.6436 |
| + precision trim (`s9trim`) | 0.6444 |
| **recall recovery (`r3_rescue`, selected)** | **0.6515** |

The last three hours ran a recall-recovery pass over the trimmed stack. The
full tail, in submission order:

| time | submission | test |
|---|---|---|
| 15:03 | `s9trim` | 0.6444 |
| 15:09 | `t3` | 0.6440 |
| 15:27 | `s9trim152` | 0.6376 |
| 19:21 | `r1_rescue` | 0.6488 |
| 19:22 | `r2_strim` | 0.6372 |
| 19:39 | `v1_wide` | 0.6463 |
| **21:21** | **`r3_rescue`** | **0.6515**  ← selected |
| 21:48 | `r3s2_rescue` | 0.6515 |
| 23:12 | `r4_rescue` | 0.6514 |

Three independent configurations land within 0.0001 of each other, which is
where this system saturates: `r3_rescue`, `r3s2_rescue` and `r4_rescue` differ
in construction and agree on the score to four decimal places. Nothing in the
final three hours moved it further.

Everything above 0.4873 is a combination of systems, rules and
post-processing over the same Phase-1 trunks, and the last three rows use
leaderboard feedback. About 25 submissions on the final day served as
validation, all within the published limits.

### 8.9 Final standing

| # | participant | Macro-F1 | P | R |
|---|---|---|---|---|
| 1 | (other participant) | 0.7019 | 0.7036 | 0.7176 |
| 2 | (other participant) | 0.7005 | 0.6904 | 0.7360 |
| 3 | (other participant) | 0.6938 | 0.6862 | 0.7078 |
| 4 | (other participant) | 0.6629 | 0.7142 | 0.6488 |
| 5 | (other participant) | 0.6626 | 0.6912 | 0.6748 |
| **6** | **n_230709** | **0.6515** | **0.6252** | **0.6835** |
| 7 | (other participant) | 0.6433 | 0.6408 | 0.6770 |

**Sixth of the field, 0.0504 behind the winner.** The secondary metrics say
where that gap lives: this system runs **recall 0.6835 against precision
0.6252**, the only entry in the top seven whose recall materially exceeds its
precision. Rank 4 is the mirror image (P 0.7142, R 0.6488) and scores
0.0114 higher on a *lower* recall. The deficit is precision, and it traces
straight back to §6.1 -- the detector emits around 130-140 spans per document
against 119.9 in training, and every post-processing stage from §5 onward was
built to recover boundaries rather than to suppress false positives.

What this phase actually demonstrated, in order of size:

1. **Data, not architecture.** Six architectures, five adaptation methods,
   class weighting swept both ways, BIO vs BIOES, CRF and bidirectional
   attention together bought less than one afternoon of adding training data.
   Ensembling returned **+0.009 per doubling**; external annotated data
   returned **+0.09 per model**.
2. **Combination over any single system.** 0.4873 to 0.6444 came entirely
   from voting and stacking across systems that already existed.
3. **Measurement before intervention.** The strict/overlap gap, the boundary
   delta shapes, the 26 individually scored rules, the 0.9863 per-document
   string-matching oracle -- each of these redirected the work, and three
   interventions were predicted to fail before submission and did.
4. **Contamination filtering at record level.** InLegalNER contained eleven
   verbatim passages from the test documents; a document-level check rated the
   worst at 0.4% and would have passed all of them.

What would have closed the gap, on this evidence: a better detector, or a
stacker trained on more than the 70 documents CV can supply. Not more features
over the same 70.

### 8.10 Carried into Phase 3

- Check for external data and run the record-level contamination filter on
  **day one**, not day four.
- **Save checkpoints and save logits.** Neither was saved this phase, so every
  idea cost a full retrain and ensembling could only vote over decoded spans
  rather than average probabilities.
- Expect CV->test transfer to degrade as training data grows: measured 1.228,
  1.103, 1.023 across three data additions.
- Per-label selection on CV overfits -- it lost four times out of four on the
  leaderboard (-0.068, -0.027, -0.036, -0.029). Prefer one global
  configuration.
- Track precision and recall separately from the start. Phase 2 optimised a
  single number and finished unbalanced in the direction that costs most.
