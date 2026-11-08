#!/usr/bin/env python3
"""CS-781 Phase 1 -- ablations of the shared architectural decisions.

    python ablate.py --tokens 40e6

The TA's clarification asks for "ablation experiments wherever applicable to
demonstrate the effect of your architectural decisions". The four full models
ablate the *architecture* against each other. This ablates the modifications
they all share, which until now were justified from reading rather than from
measurement:

    value_residual   layer-0 values mixed back in through a learned gate
    qk_norm + z_loss claimed to buy a 3-5x higher stable LR
    n_register       4 learned tokens to absorb attention mass
    Muon             claimed 1.3-1.5x fewer steps than AdamW
    attn_type=mla    M3's mechanism, at small scale
    use_pointer      M4's mechanism, at small scale

SCALE, AND ITS CAVEAT

These run at ~34M parameters on ~40M tokens, roughly 12 minutes each, against
the full models' 126M on 586M. An effect measured here does not transfer
quantitatively to full scale, and a small effect here could be large there or
vice versa. It is reported as directional evidence, which is what is affordable
and is also the correct scale to ablate at: an ablation you can only afford to
run once is not an ablation. Each variant differs from the baseline in exactly
one thing and shares the seed, the data order and the token budget.

Validation loss is the metric. At 40M tokens generation quality is too noisy to
separate variants, and val loss is what these mechanisms claim to affect.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ROOT = os.environ.get("CS781_ROOT", "/content/drive/MyDrive/cs781_capstone")
sys.path.insert(0, f"{ROOT}/shared/lib")
from cs781_bpe import BPETokenizer                                   # noqa: E402
from cs781_model import Config, LexArcD, Muon, build_optimizers      # noqa: E402
from train_phase1 import ShardLoader, mix_at, log                    # noqa: E402

# ~34M parameters: 16.8M embedding + 16.9M non-embedding.
SMALL = dict(d_model=512, n_layer=6, n_head=8, n_kv_head=2, ffn_hidden=1408,
             ctx=1024)

VARIANTS = [
    ("baseline",         {}),
    ("no_value_residual", {"value_residual": False}),
    ("no_qk_norm_zloss",  {"qk_norm": False, "z_loss": 0.0}),
    ("no_registers",      {"n_register": 0}),
    ("attn_mla",          {"attn_type": "mla", "kv_latent_dim": 128}),
    ("pointer",           {"use_pointer": True}),
]


def build(cfg_over, vocab):
    cfg = Config(vocab_size=vocab, **SMALL)
    for k, v in cfg_over.items():
        setattr(cfg, k, v)
    if cfg.use_pointer:
        from cs781_model_ptr import LexArcP
        return LexArcP(cfg), cfg
    return LexArcD(cfg), cfg


@torch.no_grad()
def validate(model, loader, n=8):
    model.eval()
    vals = []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for s in loader.val:
            vals.append(model(loader.val_batch(s, n, 0)).item())
    model.train()
    return sum(vals) / len(vals)


def run(name, cfg_over, optimiser, loader, vocab, budget_tokens, batch, ctx,
        muon_lr, adam_lr, no_compile=False):
    torch.manual_seed(1337)
    model, cfg = build(cfg_over, vocab)
    n = model.n_params()
    if optimiser == "muon":
        opts, counts = build_optimizers(model, muon_lr, adam_lr)
    else:
        # Everything on AdamW, at the LR AdamW actually wants -- comparing Muon
        # at 0.025 against AdamW at 0.025 would be a strawman.
        opts = [torch.optim.AdamW(model.parameters(), lr=3e-4,
                                  betas=(0.9, 0.95), weight_decay=0.1, fused=True)]
        counts = {"muon": 0, "adam": sum(1 for _ in model.parameters())}
    model = model.to("cuda").train()
    # Compiled, not eager. Eager materialises the fp32 upcast inside
    # logsumexp over the (B*T, 32768) logits -- a 4 GiB allocation that OOMs.
    # Inductor fuses it, so compiling is both cheaper in memory and faster;
    # the ~2 min compile per variant pays for itself.
    net = model if no_compile else torch.compile(model)

    total = int(budget_tokens / (batch * ctx))
    warm = max(1, int(0.02 * total))
    base = [[g["lr"] for g in o.param_groups] for o in opts]

    def set_lr(step):
        s = (step + 1) / warm if step < warm else \
            0.5 * (1 + math.cos(math.pi * min((step - warm) / max(1, total - warm), 1.0)))
        for o, bl in zip(opts, base):
            for g, b in zip(o.param_groups, bl):
                g["lr"] = b * s

    t0, nan_steps = time.time(), 0
    for step in range(total):
        set_lr(step)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = net(loader.batch(mix_at(step / total)))
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gn):
            nan_steps += 1
            for o in opts:
                o.zero_grad(set_to_none=True)
            continue
        for o in opts:
            o.step()
            o.zero_grad(set_to_none=True)
    val = validate(model, loader)
    dt = time.time() - t0
    log(f"{name:20s} {optimiser:5s} params {n/1e6:5.1f}M  val {val:.4f}  "
        f"{total*batch*ctx/1e6:.0f}M tok  {dt/60:4.1f} min"
        + (f"  ({nan_steps} nan steps skipped)" if nan_steps else ""))
    del model, net, opts
    torch.cuda.empty_cache()
    return dict(name=name, optimiser=optimiser, params=n, val_loss=val,
                tokens=total * batch * ctx, minutes=round(dt / 60, 1),
                nan_steps=nan_steps, overrides=cfg_over)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=float, default=40e6)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--muon-lr", type=float, default=0.025)
    ap.add_argument("--adam-lr", type=float, default=1e-3)
    ap.add_argument("--out", default="ablations")
    ap.add_argument("--only", nargs="*", default=None,
                    help="run only these variant names")
    a = ap.parse_args()

    OUT = f"{ROOT}/{a.out}"
    os.makedirs(OUT, exist_ok=True)
    torch.set_float32_matmul_precision("high")
    p = torch.cuda.get_device_properties(0)
    assert p.major * 10 + p.minor >= 80, f"{p.name}: bf16 emulated below SM80"
    log(f"GPU {p.name} | ablations at {SMALL['d_model']}d/{SMALL['n_layer']}L, "
        f"{a.tokens/1e6:.0f}M tokens each, batch {a.batch}, "
        f"{'eager' if a.no_compile else 'compiled'}")

    tok = BPETokenizer.load(f"{ROOT}/shared/tokenizer/bpe_legal_32k.json")
    loader = ShardLoader(a.batch, SMALL["ctx"], "cuda")

    results = []
    todo = [v for v in VARIANTS if not a.only or v[0] in a.only]
    for name, over in todo:
        results.append(run(name, over, "muon", loader, tok.vocab_size,
                           a.tokens, a.batch, SMALL["ctx"], a.muon_lr,
                           a.adam_lr, a.no_compile))
        json.dump(results, open(f"{OUT}/ablations.json", "w"), indent=2)

    # the optimiser claim, measured against the baseline architecture
    if not a.only or "baseline" in (a.only or []):
        results.append(run("baseline", {}, "adamw", loader, tok.vocab_size,
                           a.tokens, a.batch, SMALL["ctx"], a.muon_lr,
                           a.adam_lr, a.no_compile))
        json.dump(results, open(f"{OUT}/ablations.json", "w"), indent=2)

    base = next((r for r in results
                 if r["name"] == "baseline" and r["optimiser"] == "muon"), None)
    log("\n--- ablations, delta vs baseline (lower val loss is better) ---")
    for r in results:
        d = (r["val_loss"] - base["val_loss"]) if base else float("nan")
        # delta > 0 means val loss got WORSE without the feature, i.e. the
        # feature helps. The first version of this line said the opposite.
        if d > 0.005:
            verdict = f"the feature HELPS (-{d:.4f} val loss)"
        elif d < -0.005:
            verdict = f"the feature HURTS (+{-d:.4f} val loss)"
        else:
            verdict = "no effect at this scale"
        log(f"  {r['name']:20s} {r['optimiser']:5s} val {r['val_loss']:.4f}  "
            f"delta {d:+.4f}  -> {verdict}"
            if r is not base else
            f"  {r['name']:20s} {r['optimiser']:5s} val {r['val_loss']:.4f}  "
            f"(reference)")
    log(f"\nwrote {OUT}/ablations.json")
    log("NOTE: a 'delta' here is one run, not a mean over seeds. Treat anything "
        "under ~0.005 as unresolved at this scale.")


if __name__ == "__main__":
    main()
