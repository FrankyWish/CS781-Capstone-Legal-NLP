#!/usr/bin/env python3
"""Parameter budget calculator for the Phase-1 50M-300M constraint.

Counts parameters analytically, so configurations can be explored on a laptop
without building the model or touching a GPU. Breaks the total down by
component, reports the KV cache footprint at generation time, and estimates
how long a token budget takes at a measured throughput.

  python param_budget.py --d-model 768 --n-layer 12 --n-head 12 --n-kv-head 4
  python param_budget.py --sweep
  python param_budget.py --d-model 768 --n-layer 12 --target-params 150e6

Standard library only.
"""

from __future__ import annotations

import argparse
import sys

MIN_PARAMS = 50_000_000
MAX_PARAMS = 300_000_000


def count(vocab, d_model, n_layer, n_head, n_kv_head, ffn_mult,
          ffn_type="swiglu", norm="rms", pos="rope", n_ctx=2048, tied=True):
    """Analytic parameter count for a decoder-only transformer block stack."""
    if d_model % n_head:
        raise ValueError(f"d_model {d_model} not divisible by n_head {n_head}.")
    if n_head % n_kv_head:
        raise ValueError(f"n_head {n_head} not divisible by n_kv_head {n_kv_head}.")

    head_dim = d_model // n_head
    # Round the FFN hidden size to a multiple of 64; odd sizes waste tensor cores.
    d_ffn = int(round(d_model * ffn_mult / 64)) * 64

    embedding = vocab * d_model
    positional = 0 if pos == "rope" else n_ctx * d_model

    # Attention: Q is full width, K and V are narrowed to the KV groups.
    q = d_model * (n_head * head_dim)
    k = d_model * (n_kv_head * head_dim)
    v = d_model * (n_kv_head * head_dim)
    o = (n_head * head_dim) * d_model
    attn_per_layer = q + k + v + o

    # SwiGLU needs three matrices (gate, up, down); GeLU needs two.
    ffn_per_layer = (3 if ffn_type == "swiglu" else 2) * d_model * d_ffn

    # RMSNorm carries a scale only; LayerNorm carries scale and bias.
    norm_per_layer = 2 * d_model * (1 if norm == "rms" else 2)

    per_layer = attn_per_layer + ffn_per_layer + norm_per_layer
    blocks = per_layer * n_layer
    final_norm = d_model * (1 if norm == "rms" else 2)
    head = 0 if tied else d_model * vocab

    total = embedding + positional + blocks + final_norm + head
    return {
        "total": total,
        "embedding": embedding,
        "positional": positional,
        "blocks": blocks,
        "attention": attn_per_layer * n_layer,
        "ffn": ffn_per_layer * n_layer,
        "norms": norm_per_layer * n_layer + final_norm,
        "lm_head": head,
        "per_layer": per_layer,
        "head_dim": head_dim,
        "d_ffn": d_ffn,
        "non_embedding": total - embedding - positional - head,
    }


def kv_cache_bytes(n_layer, n_kv_head, head_dim, seq_len, batch, dtype_bytes=2):
    """K and V, per layer, per token, per sequence."""
    return 2 * n_layer * n_kv_head * head_dim * seq_len * batch * dtype_bytes


def human(n):
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return str(n)


def describe(args):
    stats = count(
        args.vocab, args.d_model, args.n_layer, args.n_head, args.n_kv_head,
        args.ffn_mult, args.ffn_type, args.norm, args.pos, args.n_ctx, not args.untied,
    )
    total = stats["total"]

    print(f"d_model {args.d_model} | layers {args.n_layer} | heads {args.n_head} "
          f"({args.n_kv_head} KV) | head_dim {stats['head_dim']} | "
          f"d_ffn {stats['d_ffn']} | ctx {args.n_ctx}")
    print(f"{args.ffn_type} FFN, {args.norm}norm, {args.pos} positions, "
          f"{'untied' if args.untied else 'tied'} embeddings\n")

    rows = [
        ("token embedding", stats["embedding"]),
        ("positional", stats["positional"]),
        ("attention", stats["attention"]),
        ("feed-forward", stats["ffn"]),
        ("norms", stats["norms"]),
        ("lm head", stats["lm_head"]),
    ]
    for name, value in rows:
        if value:
            print(f"  {name:<18} {human(value):>9}  ({100 * value / total:4.1f}%)")
    print(f"  {'-' * 18} {'-' * 9}")
    print(f"  {'TOTAL':<18} {human(total):>9}")
    print(f"  {'non-embedding':<18} {human(stats['non_embedding']):>9}\n")

    if total < MIN_PARAMS:
        print(f"UNDER the 50M floor by {human(MIN_PARAMS - total)}.")
    elif total > MAX_PARAMS:
        print(f"OVER the 300M ceiling by {human(total - MAX_PARAMS)}.")
    else:
        headroom = MAX_PARAMS - total
        print(f"Within the 50M-300M budget. Headroom: {human(headroom)}.")

    # The rules cap total parameters, and embeddings are a large share of a
    # small model, so a big vocabulary is paid for out of the layer budget.
    share = 100 * stats["embedding"] / total
    if share > 35:
        print(f"NOTE: embeddings are {share:.0f}% of the budget. A smaller "
              "vocabulary would buy real depth.")

    # Generation memory ---------------------------------------------------
    print()
    for batch in (8, 32):
        cache = kv_cache_bytes(args.n_layer, args.n_kv_head, stats["head_dim"],
                               args.gen_len, batch)
        full = kv_cache_bytes(args.n_layer, args.n_head, stats["head_dim"],
                              args.gen_len, batch)
        line = (f"KV cache @ batch {batch:<3} seq {args.gen_len}: "
                f"{cache / 1e9:.2f} GB")
        if args.n_kv_head != args.n_head:
            line += f"  (full MHA would be {full / 1e9:.2f} GB)"
        print(line)

    weights = total * (4 + 4 + 4 + 4) / 1e9  # fp32 master + grads + Adam m,v
    print(f"\nTraining state (fp32 weights + grads + Adam moments): ~{weights:.2f} GB")
    print("Activations are on top of that and scale with batch x context.")

    # Time to train -------------------------------------------------------
    if args.tokens:
        # Standard approximation: ~6 FLOPs per parameter per token for a
        # forward+backward pass, counting non-embedding parameters.
        flops = 6 * stats["non_embedding"] * args.tokens
        print(f"\nTo train on {human(args.tokens)} tokens: "
              f"~{flops / 1e18:.2f} EFLOPs of compute.")
        if args.throughput:
            hours = args.tokens / args.throughput / 3600
            print(f"At {args.throughput:,.0f} tokens/sec -> {hours:.1f} GPU-hours.")
            print(f"  x4 architectures -> {4 * hours:.1f} GPU-hours "
                  f"({4 * hours / 30:.1f} weeks of one Kaggle account's 30h quota)")
        else:
            print("Pass --throughput <tokens/sec>, measured on your own GPU, "
                  "for a wall-clock estimate.")
    return 0


def sweep(args):
    """Print configurations that land inside the budget."""
    print(f"vocab {args.vocab}, {args.ffn_type} FFN x{args.ffn_mult}, "
          f"{'untied' if args.untied else 'tied'} embeddings\n")
    print(f"{'d_model':>8} {'layers':>7} {'heads':>6} {'kv':>4} "
          f"{'d_ffn':>6} {'params':>9} {'non-emb':>9}")
    print("-" * 56)
    for d_model in (512, 640, 768, 896, 1024):
        for n_layer in (8, 12, 16, 20, 24):
            n_head = d_model // 64
            if n_head == 0:
                continue
            n_kv_head = max(1, n_head // 3) if args.n_kv_head == 0 else args.n_kv_head
            while n_head % n_kv_head:
                n_kv_head -= 1
            try:
                stats = count(args.vocab, d_model, n_layer, n_head, n_kv_head,
                              args.ffn_mult, args.ffn_type, args.norm, args.pos,
                              args.n_ctx, not args.untied)
            except ValueError:
                continue
            total = stats["total"]
            if not (MIN_PARAMS <= total <= MAX_PARAMS):
                continue
            print(f"{d_model:>8} {n_layer:>7} {n_head:>6} {n_kv_head:>4} "
                  f"{stats['d_ffn']:>6} {human(total):>9} "
                  f"{human(stats['non_embedding']):>9}")
    print("\nDeeper generally beats wider at this scale for a fixed budget, "
          "but that is a claim to verify on your own validation loss, not to "
          "take on faith.")
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vocab", type=int, default=50257,
                        help="Tokenizer vocabulary size (default: GPT-2's 50257).")
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--n-layer", type=int, default=12)
    parser.add_argument("--n-head", type=int, default=12)
    parser.add_argument("--n-kv-head", type=int, default=0,
                        help="KV groups for GQA. 0 means same as --n-head (plain MHA).")
    parser.add_argument("--ffn-mult", type=float, default=4.0,
                        help="d_ffn as a multiple of d_model (SwiGLU usually ~2.67).")
    parser.add_argument("--ffn-type", choices=("swiglu", "gelu"), default="gelu")
    parser.add_argument("--norm", choices=("rms", "layer"), default="layer")
    parser.add_argument("--pos", choices=("rope", "learned"), default="learned")
    parser.add_argument("--n-ctx", type=int, default=2048)
    parser.add_argument("--untied", action="store_true",
                        help="Separate LM head instead of tying it to the embedding.")
    parser.add_argument("--gen-len", type=int, default=1100,
                        help="Sequence length for the KV cache estimate "
                             "(Phase 1 generates ~384 tokens from ~700-token prompts).")
    parser.add_argument("--tokens", type=float, default=None,
                        help="Training token budget, for a compute estimate.")
    parser.add_argument("--throughput", type=float, default=None,
                        help="Measured tokens/sec, for a wall-clock estimate.")
    parser.add_argument("--target-params", type=float, default=None,
                        help="Solve for the d_ffn multiplier that hits this count.")
    parser.add_argument("--sweep", action="store_true",
                        help="List configurations inside the 50M-300M budget.")
    return parser


def solve_for_target(args):
    """Binary-search the FFN multiplier that lands on --target-params."""
    target = int(args.target_params)
    low, high = 0.5, 16.0
    for _ in range(60):
        mid = (low + high) / 2
        total = count(args.vocab, args.d_model, args.n_layer, args.n_head,
                      args.n_kv_head, mid, args.ffn_type, args.norm, args.pos,
                      args.n_ctx, not args.untied)["total"]
        if total < target:
            low = mid
        else:
            high = mid
    args.ffn_mult = round((low + high) / 2, 3)
    print(f"Solved: --ffn-mult {args.ffn_mult} lands nearest {human(target)}\n")
    return args


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.n_kv_head == 0:
        args.n_kv_head = args.n_head
    if args.sweep:
        return sweep(args)
    if args.target_params:
        args = solve_for_target(args)
    try:
        return describe(args)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
