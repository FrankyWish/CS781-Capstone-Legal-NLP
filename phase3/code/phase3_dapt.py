#!/usr/bin/env python3
"""CS-781 Phase 3 -- continued pretraining of a Phase-1 trunk on in-domain text.

    python phase3_dapt.py --model M2_LexArcED --tokens 60e6
    python phase3_dapt.py --model M4_LexArcP  --tokens 60e6
    python phase3_rr.py --model M2_LexArcED --trunk-path $CS781_P3/dapt/M2_LexArcED/best.pt ...

WHY. The Phase-1 trunks were pretrained on general Indian legal text. The task
is Competition Law and Income Tax Supreme Court judgments with 90 labelled
documents. Continuing pretraining on the task's own domain before fine-tuning
(DAPT/TAPT, Gururangan et al. 2020) helps most exactly when labelled data is
this small. The result is still a Phase-1 model: same weights, further trained
on permitted text.

TEXT. All competition documents (train, dev AND test sentences -- unlabelled
text only, no labels are read), the MARRO and Kalamkar judgments, and any
--corpus files (e.g. CL/IT judgments from the Phase-1 corpus, keyword-filtered).

OBJECTIVE, chosen to match how the trunk is used downstream:
  M2 (encoder)  masked LM on the encoder, 15% of tokens, BERT 80/10/10.
  decoders      the fine-tuning drops the causal mask, so plain next-token LM
                would adapt the wrong attention pattern. Instead: masked
                next-token prediction with bidirectional attention (MNTP,
                LLM2Vec, BehnamGhader et al. 2024) -- mask 20% of tokens, predict
                token t from the hidden state at t-1, i.e. through the same
                output head the model was pretrained with.

Writes {P3}/dapt/<model>/best.pt in the Phase-1 checkpoint format (cfg + full
state dict, untouched parts copied from the original), plus held-out masked-LM
loss before and after, which goes in the report.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import re
import sys
import time
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from phase3_data import P1, P3, load_all, log  # noqa: E402
from phase3_rr import RRModel, load_trunk     # noqa: E402


def corpus_docs(paths, pattern, min_hits):
    """Documents from .jsonl/.json ({"text": ...}) or .txt (one document per file
    or per blank-line block), kept if the domain pattern matches >= min_hits times."""
    rx = re.compile(pattern, re.I)
    out = []
    for p in paths:
        for f in sorted(glob.glob(p)):
            if f.endswith((".jsonl", ".json")):
                with open(f, encoding="utf-8", errors="replace") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            r = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        t = r.get("text") if isinstance(r, dict) else None
                        if isinstance(t, list):
                            t = " ".join(map(str, t))
                        if t and len(rx.findall(t)) >= min_hits:
                            out.append(t)
            else:
                t = open(f, encoding="utf-8", errors="replace").read()
                for part in (re.split(r"\n\s*\n\s*\n", t) if len(t) > 2e6 else [t]):
                    if len(rx.findall(part)) >= min_hits:
                        out.append(part)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--tokens", type=float, default=60e6, help="training token budget")
    ap.add_argument("--block", type=int, default=512)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--mask-rate", type=float, default=None, help="default .15 (M2) / .20 (decoders)")
    ap.add_argument("--corpus", nargs="*", default=[], help="extra text files/globs")
    ap.add_argument("--domain-re", default=r"competition act|income[- ]tax|assessee|cartel|"
                                           r"dominant position|assessment year|commissioner of income")
    ap.add_argument("--min-hits", type=int, default=5)
    ap.add_argument("--no-test-text", action="store_true", help="exclude test documents' text")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    random.seed(a.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(0)
        assert p.major >= 8, f"{p.name}: bf16 emulated, use an L4"
    out_dir = f"{P3}/dapt/{a.model}"
    os.makedirs(out_dir, exist_ok=True)
    logf = open(f"{P3}/logs/dapt_{a.model}.log", "a", encoding="utf-8")

    def say(*x):
        log(*x)
        print(*x, file=logf, flush=True)

    # ---- text
    docs, _ = load_all()
    texts = [" ".join(r["text"]) for r in docs.values()
             if not (a.no_test_text and r["split"] == "test")]
    n_comp = len(texts)
    for path in glob.glob(f"{P3}/external/*.jsonl"):
        with open(path, encoding="utf-8") as f:
            texts += [" ".join(json.loads(l)["text"]) for l in f if l.strip()]
    n_ext = len(texts) - n_comp
    extra = corpus_docs(a.corpus, a.domain_re, a.min_hits)
    texts += extra
    say(f"DAPT {a.model}: {n_comp} competition docs, {n_ext} external docs, "
        f"{len(extra)} corpus docs matching /{a.domain_re}/ >= {a.min_hits}x")

    from cs781_bpe import BPETokenizer
    tok = BPETokenizer.load(f"{P1}/shared/tokenizer/bpe_legal_32k.json")
    eot, mask_id = tok.specials["<|endoftext|>"], tok.specials["<|mask|>"]
    cache = f"{P3}/cache/dapt_stream_{len(texts)}_{int(a.no_test_text)}.pt"
    if os.path.exists(cache):
        stream = torch.load(cache)
    else:
        t0 = time.time()
        ids = []
        for t in texts:
            ids.extend(tok.encode(t))
            ids.append(eot)
        stream = torch.tensor(ids, dtype=torch.long)
        torch.save(stream, cache)
        say(f"  tokenised in {time.time()-t0:.0f}s")
    nb = stream.numel() // a.block
    blocks = stream[:nb * a.block].view(nb, a.block)
    perm = torch.randperm(nb)
    n_val = max(16, nb // 50)
    val, train = blocks[perm[:n_val]], blocks[perm[n_val:]]
    say(f"  {stream.numel()/1e6:.1f}M tokens -> {len(train)} train / {len(val)} held-out blocks of {a.block}")

    # ---- model: the same trunk wrapper the fine-tuning uses, so the attention
    # pattern trained here is exactly the one used downstream
    trunk, kind, cfg = load_trunk(a.model, a.ckpt)
    ns = SimpleNamespace(method="full", n_virtual=0, causal=False, seq_dim=64, dropout=0.0,
                         domain_emb=False, pos_buckets=0, seq="none", seq_heads=4, seq_layers=1,
                         crf=False, grad_ckpt=True, band=-1)
    model = RRModel(trunk, kind, cfg, ns)
    ck = torch.load(f"{P1}/{a.model}/ckpt/{a.ckpt}", map_location="cpu", weights_only=False)
    sd0 = ck["model"]
    tied = "head.weight" not in sd0 or torch.equal(sd0["head.weight"], sd0["emb.weight"])
    head = None
    if not tied:
        head = nn.Linear(cfg.d_model, sd0["head.weight"].size(0), bias=False)
        head.weight.data.copy_(sd0["head.weight"])
    model.trunk.to(device)
    if head is not None:
        head.to(device)
    model.train()
    rate = a.mask_rate or (0.15 if kind == "ed" else 0.20)
    V = cfg.vocab_size

    def lm_loss(x, gen=None):
        x = x.to(device)
        B, T = x.shape
        sel = torch.rand(B, T, device=device, generator=gen) < rate
        sel &= x != eot
        if kind != "ed":
            sel[:, 0] = False                       # predicted from t-1
        r = torch.rand(B, T, device=device, generator=gen)
        inp = x.clone()
        inp[sel & (r < 0.8)] = mask_id
        rnd = torch.randint(0, V - 8, (B, T), device=device, generator=gen)
        swap = sel & (r >= 0.8) & (r < 0.9)
        inp[swap] = rnd[swap]
        keep = torch.ones_like(x, dtype=torch.bool)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            h = model.token_hidden(inp, keep)
            if kind != "ed":
                h = torch.cat([h[:, :1] * 0, h[:, :-1]], 1)      # h[t-1] predicts x[t]
            hs = h[sel]
            W = model.trunk.emb.weight if head is None else head.weight
            logits = F.linear(hs, W).float()
        return F.cross_entropy(logits, x[sel])

    @torch.no_grad()
    def evaluate():
        model.eval()
        g = torch.Generator(device=device).manual_seed(0)
        tot = 0.0
        for k in range(0, len(val), a.batch):
            tot += float(lm_loss(val[k:k + a.batch], g)) * len(val[k:k + a.batch])
        model.train()
        return tot / len(val)

    params = list(model.trunk.parameters()) + (list(head.parameters()) if head is not None else [])
    params = [q for q in params if q.requires_grad]
    for q in params:
        q.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.98), weight_decay=0.01)
    steps = max(1, int(a.tokens // (a.batch * a.block)))
    warm = max(1, steps // 20)
    base = evaluate()
    say(f"  held-out masked-LM loss before: {base:.4f}  | {steps} steps, rate {rate}, lr {a.lr}")
    best, t0 = base, time.time()
    for step in range(steps):
        f = step / warm if step < warm else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, steps - warm)))
        for g in opt.param_groups:
            g["lr"] = a.lr * f
        idx = torch.randint(0, len(train), (a.batch,))
        loss = lm_loss(train[idx])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if (step + 1) % max(1, steps // 10) == 0 or step + 1 == steps:
            v = evaluate()
            say(f"  step {step+1}/{steps} train {float(loss):.4f} held-out {v:.4f} ({time.time()-t0:.0f}s)")
            if v < best:
                best = v
                sd = dict(sd0)
                sd.update({k: t.detach().cpu() for k, t in model.trunk.state_dict().items()})
                if head is not None:
                    sd["head.weight"] = head.weight.detach().cpu()
                tmp = f"{out_dir}/best.pt.tmp"
                torch.save(dict(cfg=ck["cfg"], model=sd, dapt=dict(vars(a), base_loss=base, loss=v,
                                                                       step=step + 1)), tmp)
                os.replace(tmp, f"{out_dir}/best.pt")
    say(f"  DAPT {a.model}: held-out masked-LM loss {base:.4f} -> {best:.4f}; wrote {out_dir}/best.pt")


if __name__ == "__main__":
    main()
