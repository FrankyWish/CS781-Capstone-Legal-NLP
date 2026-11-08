#!/usr/bin/env python3
"""CS-781 Phase 2 (L-NER) -- fine-tune a Phase-1 trunk for strict span tagging.

    python phase2_ner.py --model M1_SC --method head        --epochs 8
    python phase2_ner.py --model M2_LexArcED --method full   --epochs 8
    python phase2_ner.py --model M1_SC --method lora --rank 8 --epochs 12
    python phase2_ner.py --model M1_SC --method full --bidirectional --submit

WHAT THE DATA SAYS, AND WHAT FOLLOWS FROM IT

phase2_data.py measured the things that decide this phase:

  70 training documents, 8,393 entities, 472,327 tokens
  documents median 26,640 chars -- 7 to 28 times the 1024-token window
  ~95% of tokens are O
  alignment ceiling 98.33% (PREC only 93.41%)
  fold3 overlaps fold1/2 on 0 of 35 documents -- no retrieval shortcut here

Four consequences are built into this file:

1. 70 DOCUMENTS IS TINY. Full fine-tuning 126M parameters on it will overfit
   almost immediately, so the parameter-efficient methods are not a box to tick,
   they are the expected winners. `--method head` (train only the classifier)
   is the honest floor every other method must beat.

2. MACRO-F1 OVER 12 LABELS. Each label is 1/12 of the score no matter how rare,
   and the official scorer gives F1 = 0 for a label with no predictions. Never
   predicting R.COUNSEL costs 0.083 outright -- more than most modelling wins.
   Hence class-weighted loss, and a per-label report after every run.

3. DOCUMENTS DO NOT FIT. Windows of `--ctx` tokens at 50% stride; every token is
   scored in each window containing it and the logits are AVERAGED, so a token
   near a window edge still gets a prediction informed by the window where it
   sat centrally.

4. STRICT CHARACTER SPANS. Decoding goes back through phase2_data's trimming,
   and `validate()` enforces every rule the official scorer rejects on --
   integer offsets, end > start, known labels, no duplicate spans -- because one
   violation fails the entire submission rather than costing one entity.

THE SIX REQUIRED EXPERIMENTS

The phase needs 6 methodologically distinct runs across all four Phase-1
architectures, with LORMA and knowledge distillation compulsory:

  1  M1_SC        --method full                  full fine-tune, causal
  2  M2_LexArcED  --method full                  bidirectional MLM-pretrained encoder
  3  M1_SC        --method full --bidirectional   causal mask dropped (ablation vs 1)
  4  M3_LexArcL   --method lorma --rank 8        COMPULSORY
  5  M4_LexArcP   --method bitfit                sparse, biases only
  6  distillation                                COMPULSORY, see phase2_distil.py

Experiment 3 is the interesting one. M1/M3/M4 are causal decoders and token
classification wants both sides of the context; `attn_mask` overrides
`is_causal` in the Phase-1 attention, so dropping the mask converts a decoder
into a bidirectional encoder for fine-tuning. It tests directly against M2,
which already has a bidirectional MLM-trained encoder -- the model that came
last in Phase 1 may well come first here.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

BASE = os.environ.get("CS781_BASE", "/content/drive/MyDrive/cs781_capstone")
P1 = os.environ.get("CS781_P1", f"{BASE}/phase1")
P2 = os.environ.get("CS781_P2", f"{BASE}/phase2")
sys.path.insert(0, f"{P1}/shared/lib")
sys.path.insert(0, f"{P2}/code")
from cs781_bpe import BPETokenizer                                   # noqa: E402
from cs781_model import Config, LexArcD, build_rope                  # noqa: E402
from phase2_data import (align, encode_tags, decode_tags, n_tags_for,  # noqa: E402
                         parse_gold, fetch, load_json, log)


# ------------------------------------------------------------------- model --
class CRF(nn.Module):
    """Linear-chain CRF over the tag sequence.

    WHY THIS AND NOT MORE TAGS

    The boundary diagnostic showed the failure is FRAGMENTATION, not detection:
    the detector reaches ~0.58 overlap-F1 while strict scoring gets 0.29, and the
    errors are large truncations rather than off-by-one --

        'KRISHNAIYER, V.R.'            -> 'KRISHNAIYER'
        'SREE SREE ISHWAR SRIDHAR JEW' -> 'SREE SREE'

    Softmax-per-token has no way to express "this entity continues across the
    comma". Each position is classified independently and the span ends wherever
    one token happens to fall below threshold. BIOES was tried first and made it
    worse (0.2539 vs 0.2938) -- doubling the tag count dilutes the signal from
    70 documents without adding any coupling between neighbours.

    A CRF adds exactly that coupling: a learned transition matrix scored jointly
    with the emissions, so the whole tag sequence is optimised rather than each
    position in isolation, and Viterbi decoding cannot produce a path the
    transitions make unlikely.
    """

    def __init__(self, n):
        super().__init__()
        self.n = n
        self.trans = nn.Parameter(torch.zeros(n, n))     # trans[i, j]: i -> j
        self.start = nn.Parameter(torch.zeros(n))
        self.end = nn.Parameter(torch.zeros(n))

    def _path_score(self, emit, tags, mask):
        B, T, _ = emit.shape
        m = mask.float()
        sc = self.start[tags[:, 0]] + emit[:, 0].gather(1, tags[:, :1]).squeeze(1)
        for t in range(1, T):
            step = (self.trans[tags[:, t - 1], tags[:, t]]
                    + emit[:, t].gather(1, tags[:, t:t + 1]).squeeze(1))
            sc = sc + step * m[:, t]
        last = (mask.sum(1).long() - 1).clamp(min=0)
        return sc + self.end[tags.gather(1, last.unsqueeze(1)).squeeze(1)]

    def _logZ(self, emit, mask):
        B, T, _ = emit.shape
        a = self.start + emit[:, 0]
        for t in range(1, T):
            nxt = torch.logsumexp(a.unsqueeze(2) + self.trans.unsqueeze(0), 1) + emit[:, t]
            a = torch.where(mask[:, t:t + 1], nxt, a)
        return torch.logsumexp(a + self.end, 1)

    def nll(self, emit, tags, mask):
        safe = tags.clamp(min=0)
        return (self._logZ(emit, mask) - self._path_score(emit, safe, mask)).mean()

    @torch.no_grad()
    def viterbi(self, emit):
        """emit (1, T, n) -> list[int] of length T. Used on a whole document."""
        T = emit.size(1)
        a = self.start + emit[:, 0]
        back = []
        for t in range(1, T):
            sc = a.unsqueeze(2) + self.trans.unsqueeze(0)
            best, idx = sc.max(1)
            a = best + emit[:, t]
            back.append(idx)
        path = [int((a + self.end).argmax(1))]
        for idx in reversed(back):
            path.append(int(idx[0, path[-1]]))
        return path[::-1]


class NERModel(nn.Module):
    """A Phase-1 trunk with its LM head replaced by a tag classifier."""

    def __init__(self, trunk, d_model, n_tags, kind, n_register, bidirectional,
                 crf=False):
        super().__init__()
        self.trunk, self.kind, self.R = trunk, kind, n_register
        self.bi = bidirectional
        self.crf = CRF(n_tags) if crf else None
        self.drop = nn.Dropout(0.1)
        self.head = nn.Linear(d_model, n_tags)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def hidden(self, idx, pad):
        if self.kind == "ed":
            # M2's encoder is already bidirectional and carries no registers.
            return self.trunk.encode(idx)
        cfg = self.trunk.cfg
        R = self.R
        cos, sin = build_rope(idx.size(1) + R, cfg.head_dim, idx.device,
                              self.trunk.emb.weight.dtype, cfg.rope_theta)
        mask = None
        if self.bi:
            # Any non-pad key is visible from any query. Passing attn_mask at
            # all is what turns off is_causal in the Phase-1 attention, so this
            # one tensor is the whole decoder -> encoder conversion.
            keep = torch.cat([pad.new_ones(pad.size(0), R), pad], 1)
            mask = keep[:, None, None, :].expand(-1, cfg.n_head, keep.size(1), -1)
        h, _ = self.trunk._trunk(idx, cos, sin, attn_mask=mask)
        return h[:, R:]

    def forward(self, idx, pad):
        return self.head(self.drop(self.hidden(idx, pad)))


# ------------------------------------------------------- adaptation methods --
class LoRALinear(nn.Module):
    """W x + (alpha/r) * B(A x). The base weight is frozen; only A and B train."""

    def __init__(self, base: nn.Linear, r: int, alpha: float = 16.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.A = nn.Parameter(torch.zeros(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.normal_(self.A, std=1 / math.sqrt(base.in_features))
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.A), self.B) * self.scale


class LoRMALinear(nn.Module):
    """LoRMA+ : h = (I_d + (alpha/r) B A) W_0 x   -- eq. 8 of arXiv 2506.07621.

    LoRMA (Bihany, Patel & Modi 2025) shifts LoRA's ADDITIVE reparametrisation
    to a MULTIPLICATIVE one. Where LoRA learns a low-rank matrix added to the
    frozen weight, LoRMA learns a low-rank transform applied to that weight:

        LoRA    h = (W_0 + (alpha/r) B A) x
        LoRMA+  h = (I_d + (alpha/r) B A) W_0 x

    Two details from the paper are load-bearing and both are implemented here.

    RANK INFLATION. B A has rank at most r, so a bare multiplicative update
    would collapse the weight to rank r and destroy the pretrained matrix. The
    identity term is the inflation: rank(I + BA) >= d - r. The paper's other
    variant inflates by cyclically rotating row i of B A by i positions, which
    raises the rank without spending parameters -- that is `lorma_pi` below.

    OPERATION REORDERING. Forming (I + BA) W_0 explicitly would cost O(d^2 k).
    Because the identity makes the expression distribute, it is evaluated as
    y = W_0 x first, then y + (alpha/r) B (A y) -- O(dkb), the same as LoRA.
    Note where the low-rank term sits: it transforms the layer's OUTPUT, so A
    maps from the output dimension, not the input one.

    INITIALISATION. B = 0 and A random, so I(BA) = I at step 0 and fine-tuning
    begins exactly at the pretrained function.
    """

    def __init__(self, base: nn.Linear, r: int, alpha: float = 16.0):
        super().__init__()
        self.base = base
        for q in self.base.parameters():
            q.requires_grad_(False)
        d = base.out_features
        self.A = nn.Parameter(torch.empty(r, d))
        self.B = nn.Parameter(torch.zeros(d, r))
        nn.init.normal_(self.A, std=1 / math.sqrt(d))
        self.scale = alpha / r

    def forward(self, x):
        y = self.base(x)
        return y + F.linear(F.linear(y, self.A), self.B) * self.scale


class LoRMAPiLinear(nn.Module):
    """LoRMA-pi : h = I_pi(B A) W_0 x, inflating rank by cyclic row rotation.

    Row i of B A is rolled by i positions. That costs no parameters and raises
    the rank, but it breaks associativity -- the rotated matrix cannot be
    factored back into B and A -- so the d x d product is materialised and
    applied to y. At d=768 that is one 768x768 matmul per adapted layer, which
    is affordable at this scale and is the honest cost of the variant.

    Initialised so I_pi(B A) = I: first column of B all ones, A[0,0] = 1 and the
    rest of A zero, which places a single 1 in each row at the position the
    rotation carries to the diagonal.
    """

    def __init__(self, base: nn.Linear, r: int):
        super().__init__()
        self.base = base
        for q in self.base.parameters():
            q.requires_grad_(False)
        d = base.out_features
        self.A = nn.Parameter(torch.zeros(r, d))
        self.B = nn.Parameter(torch.randn(d, r) * 0.02)
        with torch.no_grad():
            self.A[0, 0] = 1.0
            self.B[:, 0] = 1.0
        idx = (torch.arange(d)[None, :] - torch.arange(d)[:, None]) % d
        self.register_buffer("roll", idx, persistent=False)

    def forward(self, x):
        y = self.base(x)
        m = (self.B @ self.A).gather(1, self.roll)      # row i rotated by i
        return torch.einsum("...j,ij->...i", y, m)


def apply_method(model, method, rank):
    """Freeze/unfreeze, returning the number of trainable parameters."""
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.head.parameters():
        p.requires_grad_(True)
    if model.crf is not None:
        for p in model.crf.parameters():
            p.requires_grad_(True)

    if method == "head":
        pass                                     # classifier only -- the floor
    elif method == "full":
        for p in model.trunk.parameters():
            p.requires_grad_(True)
    elif method == "bitfit":
        for n, p in model.trunk.named_parameters():
            if n.endswith(".bias"):
                p.requires_grad_(True)
        # The Phase-1 blocks are bias-free by design, so BitFit alone would have
        # nothing to train. Norm gains are the equivalent sparse knob here.
        for n, p in model.trunk.named_parameters():
            if ".n1." in n or ".n2." in n or n.startswith("norm"):
                p.requires_grad_(True)
    elif method == "lora":
        wrap(model.trunk, lambda lin: LoRALinear(lin, rank),
             ("wq", "wk", "wv", "wo", "kv_down", "k_up", "v_up"))
    elif method == "lorma":
        # The paper applies it to the query and value projections at this model
        # scale (RoBERTa / GPT-2); the wider MLP targeting is for 2B+ models.
        wrap(model.trunk, lambda lin: LoRMALinear(lin, rank), ("wq", "wv"))
    elif method == "lorma_pi":
        wrap(model.trunk, lambda lin: LoRMAPiLinear(lin, rank), ("wq", "wv"))
    else:
        raise ValueError(method)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def wrap(trunk, make, targets):
    """Replace the named attention projections in every block via `make`."""
    for block in getattr(trunk, "blocks", None) or []:
        att = block.attn
        for name in targets:
            lin = getattr(att, name, None)
            if isinstance(lin, nn.Linear):
                setattr(att, name, make(lin))
    for group in ("enc", "dec"):                 # M2
        for block in getattr(trunk, group, []) or []:
            att = getattr(block, "self", getattr(block, "attn", None))
            for name in targets:
                lin = getattr(att, name, None) if att is not None else None
                if isinstance(lin, nn.Linear):
                    setattr(att, name, make(lin))


# --------------------------------------------------------------- windowing --
def windows(ids, tags, ctx, stride, header=0):
    """[(start, ids, tags, skip)] covering every token.

    `header` prepends the document's first `header` tokens to every window.
    APP, RESP, A.COUNSEL and R.COUNSEL are defined by the case caption -- who
    the appellant is depends entirely on the "X versus Y" line at the top of the
    document -- but only the first window contains it. Every other window is
    asked to identify the appellant with the defining evidence removed, which is
    exactly the set of labels scoring ~0.05 while DATE and COURT, which are
    locally recognisable, score 0.75+.

    The prepended tokens get tag -100 so they are never trained on and never
    predicted from; `skip` tells the caller how many leading positions to drop.
    """
    body = ctx - header
    out, i = [], 0
    while True:
        j = min(i + body, len(ids))
        wi, wt = ids[i:j], (tags[i:j] if tags else None)
        skip = 0
        if header and i >= header:
            wi = ids[:header] + wi
            wt = ([-100] * header + wt) if wt is not None else None
            skip = header
        out.append((i, wi, wt, skip))
        if j >= len(ids):
            break
        i += stride
    return out


def build(tok, docs, text, gold, label_to_id, ctx, stride,
          with_tags=True, header=0, scheme="bio"):
    data = []
    for d in docs:
        ids, spans, _ = align(tok, text[d])
        tags = (encode_tags(spans, gold.get(d, []), label_to_id, scheme)
                if with_tags else None)
        for st, wi, wt, sk in windows(ids, tags, ctx, stride, header):
            data.append(dict(doc=d, start=st, ids=wi, tags=wt, skip=sk,
                             n=len(ids), spans=spans))
    return data


def batches(data, bs, device, shuffle, n_tags):
    order = torch.randperm(len(data)).tolist() if shuffle else range(len(data))
    buf = list(order)
    for k in range(0, len(buf), bs):
        chunk = [data[i] for i in buf[k:k + bs]]
        w = max(len(c["ids"]) for c in chunk)
        idx = torch.zeros(len(chunk), w, dtype=torch.long)
        tag = torch.full((len(chunk), w), -100, dtype=torch.long)
        pad = torch.zeros(len(chunk), w, dtype=torch.bool)
        for i, c in enumerate(chunk):
            n = len(c["ids"])
            idx[i, :n] = torch.tensor(c["ids"])
            pad[i, :n] = True
            if c["tags"] is not None:
                tag[i, :n] = torch.tensor(c["tags"])
        yield (idx.to(device), tag.to(device), pad.to(device), chunk)


# ---------------------------------------------------------------- scoring ---
def strict_macro_f1(pred, truth, labels):
    """Byte-for-byte the official rule: TP/FP/FN pooled per label, mean of F1."""
    stats = {l: [0, 0, 0] for l in labels}
    # Alongside the official strict counts, an OVERLAP-based tally. A prediction
    # that finds the right entity but clips a title ("Shri", "M/s") scores zero
    # under strict matching, so the gap between the two says whether the model
    # is failing to FIND entities or failing to BOUND them -- and those call for
    # completely different fixes. Not the graded metric; a diagnostic.
    rel = {l: [0, 0, 0, 0] for l in labels}      # hit_p, n_pred, hit_g, n_gold
    for d in truth:
        g, p = set(map(tuple, truth[d])), set(map(tuple, pred.get(d, [])))
        for l in labels:
            gs = {x for x in g if x[2] == l}
            ps = {x for x in p if x[2] == l}
            stats[l][0] += len(gs & ps)
            stats[l][1] += len(ps - gs)
            stats[l][2] += len(gs - ps)
            rel[l][1] += len(ps)
            rel[l][3] += len(gs)
            rel[l][0] += sum(1 for a in ps if any(a[0] < b[1] and a[1] > b[0] for b in gs))
            rel[l][2] += sum(1 for b in gs if any(a[0] < b[1] and a[1] > b[0] for a in ps))
    relax = {l: (rel[l][0] / rel[l][1] if rel[l][1] else 0.0,
                 rel[l][2] / rel[l][3] if rel[l][3] else 0.0) for l in labels}
    per = {}
    for l in labels:
        tp, fp, fn = stats[l]
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
        rp, rr = relax[l]
        rf = 2 * rp * rr / (rp + rr) if rp + rr else 0.0
        per[l] = (pr, rc, f, tp, fp, fn, rf)
    return sum(v[2] for v in per.values()) / len(labels), per


def boundary_report(pred, truth, text, labels, n_examples=6):
    """Exactly HOW the spans are wrong, not just that they are.

    The overlap-vs-strict gap says the detector works and the boundaries do not.
    That is only actionable once we know the shape of the error: a systematic
    "+1 character at the end" is a one-line fix worth a quarter of the metric,
    while a scattered distribution means the model genuinely cannot find edges
    and needs a different head. For every gold span, this takes the predicted
    span of the same label with the largest overlap and records how far off each
    edge is.
    """
    import collections
    deltas = collections.Counter()
    edge = collections.Counter()
    examples = []
    for d in truth:
        ps = [tuple(x) for x in pred.get(d, [])]
        for g in truth[d]:
            gs, ge, gl = tuple(g)
            cands = [p for p in ps if p[2] == gl and p[0] < ge and p[1] > gs]
            if not cands:
                continue
            best = max(cands, key=lambda p: min(p[1], ge) - max(p[0], gs))
            ds, de = best[0] - gs, best[1] - ge
            if (ds, de) == (0, 0):
                continue
            deltas[(max(-9, min(9, ds)), max(-9, min(9, de)))] += 1
            edge["start" if ds and not de else "end" if de and not ds else "both"] += 1
            if len(examples) < n_examples and d in text:
                examples.append((gl, text[d][gs:ge], text[d][best[0]:best[1]]))
    tot = sum(deltas.values())
    if not tot:
        return
    log("")
    log(f"  --- boundary error shape ({tot} overlapping-but-inexact spans) ---")
    log(f"  wrong edge: " + "  ".join(f"{k}={v} ({100*v/tot:.0f}%)"
                                      for k, v in edge.most_common()))
    log("  most common (start, end) deltas; start+ begins LATE and end- stops"
        "  EARLY, so BOTH signs mean the prediction is too narrow:")
    for (ds, de), n in deltas.most_common(10):
        log(f"    start{ds:+3d} end{de:+3d}  {n:5d}  {100*n/tot:5.1f}%")
    log("  examples (gold -> predicted):")
    for lb, g, p in examples:
        log(f"    {lb:11s} {g[:54]!r}")
        log(f"    {'':11s} {p[:54]!r}")


def validate(pred, labels):
    """Every rule evaluate_phase2.py rejects on. One violation fails the lot."""
    allowed = set(labels)
    for d, spans in pred.items():
        seen = set()
        for s in spans:
            assert len(s) == 3, f"{d}: entity must be [start, end, label], got {s}"
            a, b, l = s
            assert isinstance(a, int) and isinstance(b, int), f"{d}: offsets must be int"
            assert 0 <= a < b, f"{d}: require 0 <= start < end, got {(a, b)}"
            assert l in allowed, f"{d}: unknown label {l!r}"
            assert (a, b, l) not in seen, f"{d}: duplicate span {(a, b, l)}"
            seen.add((a, b, l))


# ---------------------------------------------------------------- predict ---
@torch.no_grad()
def predict(model, data, labels, text, device, bs, n_tags, scheme="bio",
            merge_gap=0):
    """Average logits across overlapping windows, then decode to char spans."""
    model.eval()
    acc = {}
    for d in {c["doc"] for c in data}:
        n = next(c["n"] for c in data if c["doc"] == d)
        acc[d] = torch.zeros(n, n_tags)
    for idx, _, pad, chunk in batches(data, bs, device, False, n_tags):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = model(idx, pad).float().cpu()
        for i, c in enumerate(chunk):
            k = c.get("skip", 0)
            n = len(c["ids"]) - k          # body positions only; the prepended
            acc[c["doc"]][c["start"]:c["start"] + n] += lg[i, k:k + n]
    out = {}
    for d, a in acc.items():
        spans = next(c["spans"] for c in data if c["doc"] == d)
        if a.size(0) == 0:                 # document produced no tokens at all
            out[d] = []
            continue
        if getattr(model, "crf", None) is not None:
            # Viterbi over the ENTIRE document, after the window emissions have
            # been averaged -- so a transition is never cut at a window edge.
            tags = model.crf.viterbi(a.unsqueeze(0).to(next(model.parameters()).device))
        else:
            tags = a.argmax(-1).tolist()
        out[d] = [list(x) for x in decode_tags(tags, spans, labels, text[d],
                                              scheme, merge_gap)]
    model.train()
    return out


# ------------------------------------------------------------------- train --
def run_fold(trunk_fn, train_docs, eval_docs, text, gold, labels, a, device):
    label_to_id = {l: i for i, l in enumerate(labels)}
    n_tags = n_tags_for(labels, a.scheme)
    tok = BPETokenizer.load(f"{P1}/shared/tokenizer/bpe_legal_32k.json")

    tr = build(tok, train_docs, text, gold, label_to_id, a.ctx, a.ctx // 2,
                  header=a.header, scheme=a.scheme)
    ev = build(tok, eval_docs, text, gold, label_to_id, a.ctx, a.ctx // 2,
                  header=a.header, scheme=a.scheme)

    model, kind = trunk_fn(n_tags)
    n_train = apply_method(model, a.method, a.rank)
    # .to() AFTER the wrapping: LoRA/LoRMA build new Linear-replacing modules,
    # and moving the model first leaves their parameters and buffers on the CPU.
    model = model.to(device)
    total = sum(p.numel() for p in model.parameters())
    log(f"  {a.method}: {n_train/1e6:.2f}M trainable of {total/1e6:.1f}M "
        f"({100*n_train/total:.2f}%)")

    # Class weights. ~95% of tokens are O; without this the model predicts O
    # everywhere, every label scores F1 0, and macro-F1 is 0.
    freq = torch.zeros(n_tags)
    for c in tr:
        for t in c["tags"]:
            if t >= 0:                  # -100 marks prepended header positions
                freq[t] += 1
    w = (freq.sum() / freq.clamp(min=1)).pow(a.weight_pow)
    w = (w / w.mean()).clamp(max=a.weight_cap).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95), weight_decay=0.01)
    steps = a.epochs * max(1, math.ceil(len(tr) / a.batch))
    warm = max(1, int(0.1 * steps))
    best, best_pred, best_ep, step = -1.0, None, 1, 0
    t0 = time.time()
    for ep in range(a.epochs):
        for idx, tag, pad, _ in batches(tr, a.batch, device, True, n_tags):
            lr = a.lr * (step / warm if step < warm else
                         0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, steps - warm))))
            for g in opt.param_groups:
                g["lr"] = lr
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(idx, pad)
                if model.crf is not None:
                    # The CRF scores whole sequences, so class weighting has no
                    # place in it; the transition matrix is what stops the model
                    # collapsing to all-O, and it learns that from the data.
                    loss = model.crf.nll(logits.float(), tag, pad & (tag != -100))
                else:
                    loss = F.cross_entropy(logits.reshape(-1, n_tags).float(),
                                           tag.reshape(-1), weight=w,
                                           ignore_index=-100)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
        pred = predict(model, ev, labels, text, device, a.batch, n_tags, a.scheme, a.merge_gap)
        truth = {d: [list(x) for x in gold.get(d, [])] for d in eval_docs}
        f1, per = strict_macro_f1(pred, truth, labels)
        log(f"  epoch {ep+1:2d}/{a.epochs}  loss {loss.item():.4f}  "
            f"macro-F1 {f1:.4f}  ({time.time()-t0:.0f}s)")
        if f1 > best:
            best, best_pred, best_per, best_ep = f1, pred, per, ep + 1
    log(f"  best macro-F1 {best:.4f}")
    for l in labels:
        pr, rc, f, tp, fp, fn, rf = best_per[l]
        flag = "  <-- NEVER PREDICTED, costs 1/12 outright" if tp + fp == 0 else ""
        gap = rf - f
        note = "  BOUNDARIES" if gap > 0.15 else ""
        log(f"    {l:11s} P={pr:.3f} R={rc:.3f} F1={f:.3f} | overlap-F1={rf:.3f} "
            f"(+{gap:.3f}){note}  tp={tp:4d} fp={fp:4d} fn={fn:4d}{flag}")
    boundary_report(best_pred, {d: [list(x) for x in gold.get(d, [])]
                                for d in eval_docs}, text, labels)
    sm = sum(v[2] for v in best_per.values()) / len(labels)
    rm = sum(v[6] for v in best_per.values()) / len(labels)
    log(f"    macro: strict {sm:.4f} | overlap {rm:.4f} -- the gap is pure "
        f"boundary error, recoverable without a better detector")
    return best, best_per, model, best_ep, best_pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Phase-1 dir, e.g. M1_SC")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--method", default="head",
                    choices=["head", "full", "bitfit", "lora", "lorma", "lorma_pi"])
    ap.add_argument("--bidirectional", action="store_true",
                    help="drop the causal mask -- decoder becomes an encoder")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--scheme", default="bio", choices=["bio", "bioes"],
                    help="bioes adds explicit End/Single tags so the model "
                         "predicts boundaries directly")
    ap.add_argument("--merge-gap", type=int, default=0,
                    help="join adjacent same-label spans separated by at most "
                         "this many punctuation/space characters")
    ap.add_argument("--crf", action="store_true",
                    help="linear-chain CRF over the tags; the fix for fragmented spans")
    ap.add_argument("--header", type=int, default=0,
                    help="prepend this many document-initial tokens to every "
                         "window, so body windows still see the case caption")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-pow", type=float, default=0.5)
    ap.add_argument("--weight-cap", type=float, default=20.0)
    ap.add_argument("--submit", action="store_true",
                    help="refit on all 70 docs and write a fold3 submission")
    ap.add_argument("--tag", default="", help="name for the results file")
    a = ap.parse_args()
    assert not (a.crf and a.header), (
        "--crf assumes every valid position is contiguous from index 0; "
        "--header puts masked positions in front of them. Header did not help "
        "(0.2859 vs 0.2852), so drop it rather than complicating the CRF.")

    device = "cuda"
    torch.set_float32_matmul_precision("high")
    p = torch.cuda.get_device_properties(0)
    assert p.major * 10 + p.minor >= 80, f"{p.name}: bf16 emulated below SM80"
    log(f"GPU {p.name} | {a.model}/{a.ckpt} | method={a.method}"
        f"{' +bidirectional' if a.bidirectional else ''}")

    with open(fetch("Train/labels.txt"), encoding="utf-8") as f:
        labels = [x.strip() for x in f if x.strip()]
    text, gold = {}, {}
    for fold in ("fold1", "fold2"):
        text.update({str(k): v for k, v in load_json(f"Train/{fold}-text.json").items()})
        gold.update(parse_gold(load_json(f"Train/{fold}-gold.json")))
    split = json.load(open(f"{P2}/split.json"))

    ck = torch.load(f"{P1}/{a.model}/ckpt/{a.ckpt}", map_location="cpu",
                    weights_only=False)
    cfg_d = ck["cfg"]
    is_ed = "n_enc_layer" in cfg_d

    def trunk_fn(n_tags):
        if is_ed:
            from cs781_model_ed import EDConfig, LexArcED
            cfg = EDConfig(**cfg_d)
            t = LexArcED(cfg)
            t.load_state_dict(ck["model"], strict=False)
            return NERModel(t, cfg.d_model, n_tags, "ed", 0, True, a.crf), "ed"
        cfg = Config(**cfg_d)
        if getattr(cfg, "use_pointer", False):
            from cs781_model_ptr import LexArcP
            t = LexArcP(cfg)
        else:
            t = LexArcD(cfg)
        t.load_state_dict(ck["model"], strict=False)
        return NERModel(t, cfg.d_model, n_tags, "dec", cfg.n_register,
                        a.bidirectional, a.crf), "dec"

    # 2-fold CV on the organisers' folds -- see phase2_data.py for why.
    scores, best_eps, cv_pred = [], [], {}
    for tr_fold, ev_fold in (("fold1", "fold2"), ("fold2", "fold1")):
        log(f"\n--- train {tr_fold} -> score {ev_fold} ---")
        f1, per, _, ep, fpred = run_fold(trunk_fn, split[tr_fold],
                                         split[ev_fold], text, gold,
                                         labels, a, device)
        scores.append(f1)
        best_eps.append(ep)
        # held-out predictions, so boundary post-processing can be tuned
        # on CPU afterwards instead of costing a GPU run per idea
        cv_pred.update({d: [list(x) for x in v] for d, v in fpred.items()})
    cv = sum(scores) / len(scores)
    log(f"\n2-fold CV macro-F1 {cv:.4f}  (folds {scores[0]:.4f} / {scores[1]:.4f})")

    os.makedirs(f"{P2}/results", exist_ok=True)
    cvp = f"{P2}/results/cvpred_{a.tag or a.model}.json"
    json.dump(cv_pred, open(cvp, "w", encoding="utf-8"), ensure_ascii=False)
    log(f"wrote {cvp}  ({sum(len(v) for v in cv_pred.values())} spans)")

    if a.submit:
        # CV trained on 35 documents per fold; the submitted model gets all 70,
        # so it should beat the CV number rather than match it. There is no
        # held-out set left to early-stop on, so it runs for the epoch count
        # that won on average during CV.
        ep = max(1, round(sum(best_eps) / len(best_eps)))
        all_docs0 = split["fold1"] + split["fold2"]
        log("")
        log(f"--- final fit on all {len(all_docs0)} documents for {ep} "
            f"epochs (CV best epochs: {best_eps}) ---")
        test_text = {str(k): v for k, v in load_json("Test/fold3-text.json").items()}
        tok = BPETokenizer.load(f"{P1}/shared/tokenizer/bpe_legal_32k.json")
        label_to_id = {l: i for i, l in enumerate(labels)}
        n_tags = n_tags_for(labels, a.scheme)
        all_docs = all_docs0

        tr = build(tok, all_docs, text, gold, label_to_id, a.ctx, a.ctx // 2,
                      header=a.header, scheme=a.scheme)
        model, _ = trunk_fn(n_tags)
        apply_method(model, a.method, a.rank)          # before .to(), see run_fold
        model = model.to(device)
        freq = torch.zeros(n_tags)
        for c in tr:
            for t in c["tags"]:
                if t >= 0:              # -100 marks prepended header positions
                    freq[t] += 1
        w = (freq.sum() / freq.clamp(min=1)).pow(a.weight_pow)
        w = (w / w.mean()).clamp(max=a.weight_cap).to(device)
        params = [q for q in model.parameters() if q.requires_grad]
        opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95), weight_decay=0.01)
        steps = ep * max(1, math.ceil(len(tr) / a.batch))
        warm, step = max(1, int(0.1 * steps)), 0
        for e in range(ep):
            for idx, tagt, pad, _ in batches(tr, a.batch, device, True, n_tags):
                lr = a.lr * (step / warm if step < warm else
                             0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, steps - warm))))
                for gp in opt.param_groups:
                    gp["lr"] = lr
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = F.cross_entropy(model(idx, pad).reshape(-1, n_tags).float(),
                                           tagt.reshape(-1), weight=w, ignore_index=-100)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
            log(f"  final epoch {e+1}/{ep}  loss {loss.item():.4f}")

        te = build(tok, sorted(test_text), test_text, {}, label_to_id,
                   a.ctx, a.ctx // 2, with_tags=False, header=a.header,
                   scheme=a.scheme)
        pred = predict(model, te, labels, test_text, device, a.batch, n_tags, a.scheme, a.merge_gap)
        pred = {d: pred.get(d, []) for d in test_text}
        validate(pred, labels)          # one bad span fails the whole upload
        n_ent = sum(len(v) for v in pred.values())
        empty = [d for d, v in pred.items() if not v]
        log(f"  {n_ent} entities over {len(pred)} documents "
            f"({n_ent/len(pred):.1f} per document; training averages "
            f"{sum(len(v) for v in gold.values())/len(gold):.1f})")
        if empty:
            log(f"  WARNING: {len(empty)} documents have no predictions at all")
        got = {l for v in pred.values() for _, _, l in v}
        miss = [l for l in labels if l not in got]
        if miss:
            log(f"  WARNING: never predicted {miss} -- each costs 1/12 of macro-F1")
        tag0 = a.tag or f"{a.model}_{a.method}"
        os.makedirs(f"{P2}/submissions", exist_ok=True)
        pj = f"{P2}/submissions/prediction_{tag0}.json"
        json.dump(pred, open(pj, "w", encoding="utf-8"), ensure_ascii=False)
        import zipfile
        zp = f"{P2}/submissions/submission_{tag0}.zip"
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(pj, "prediction.json")
        log(f"  WROTE {zp}")

    tag = a.tag or f"{a.model}_{a.method}{'_bi' if a.bidirectional else ''}"
    os.makedirs(f"{P2}/results", exist_ok=True)
    json.dump({"tag": tag, "model": a.model, "ckpt": a.ckpt, "method": a.method,
               "bidirectional": a.bidirectional, "rank": a.rank,
               "epochs": a.epochs, "lr": a.lr, "cv_macro_f1": cv, "folds": scores},
              open(f"{P2}/results/{tag}.json", "w"), indent=2)
    log(f"wrote {P2}/results/{tag}.json")


if __name__ == "__main__":
    main()
