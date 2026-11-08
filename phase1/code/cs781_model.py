"""CS-781 Phase 1 -- LexArc-D, and the Muon optimiser.

Backbone is decoder-only in the LLaMA/Mistral family (RoPE, GQA, SwiGLU,
RMSNorm) implemented here rather than imported. On top of that, four
modifications of my own, each one motivated and each one ablatable by a config
flag:

  D1 value_residual   every layer mixes layer-0's values back in through a
                      learned per-layer gate. First-layer value information is
                      washed out by depth at small scale, and we are badly
                      token-limited, so anything that speeds early convergence
                      is worth more here than at Chinchilla scale.
  D2 qk_norm+z_loss   not a quality trick. Together they buy a 3-5x higher
                      stable learning rate, and on a wall-clock budget the
                      learning rate *is* throughput.
  D3 n_register       learned register tokens prepended to every window, so
                      attention has somewhere to dump probability mass instead
                      of hijacking token 0.
  D4 copy_bias        (off by default, kept for the ablation) additive score
                      bonus on key positions whose token id already occurred in
                      the context. Targets prompt-copying directly. Confined to
                      the last layers because it needs an explicit mask and
                      therefore loses the flash kernel.

ONE STRUCTURAL RULE, and it is the reason this file exists:

    forward() takes `idx` and nothing else. There is no `labels` argument and
    no `y`. The shift from inputs to targets happens here, once, on line
    `targets = idx[:, 1:]`. A data loader cannot shift the labels because it is
    never asked for labels. The double-shift bug that cost 44 GPU-hours is not
    "fixed" -- it is unrepresentable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 32768
    d_model: int = 768
    n_layer: int = 16
    n_head: int = 12
    n_kv_head: int = 4
    ffn_hidden: int = 2048
    ctx: int = 1024
    n_register: int = 4
    rope_theta: float = 10000.0
    z_loss: float = 1e-4
    value_residual: bool = True
    qk_norm: bool = True
    copy_bias_layers: int = 0          # D4, ablation only
    tie_embeddings: bool = True
    attn_type: str = "gqa"             # "gqa" (M1) or "mla" (M3)
    kv_latent_dim: int = 256           # MLA only: width of the shared KV code
    copy_dim: int = 64                 # M4 only: pointer-generator head width
    use_pointer: bool = False          # M4: pointer-generator output head

    @property
    def head_dim(self):
        return self.d_model // self.n_head


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return self.w * F.rms_norm(x, (x.shape[-1],), eps=self.eps)


def build_rope(T, hd, device, dtype, theta=10000.0, offset=0):
    inv = 1.0 / (theta ** (torch.arange(0, hd, 2, device=device).float() / hd))
    pos = torch.arange(offset, offset + T, device=device).float()
    f = torch.outer(pos, inv)
    e = torch.cat([f, f], -1)
    return e.cos().to(dtype)[None, None], e.sin().to(dtype)[None, None]


def _rot_half(x):
    a, b = x.chunk(2, -1)
    return torch.cat([-b, a], -1)


def apply_rope(x, cos, sin):
    return x * cos + _rot_half(x) * sin


class Attention(nn.Module):
    def __init__(self, cfg: Config, layer_idx: int):
        super().__init__()
        self.cfg, self.layer_idx = cfg, layer_idx
        hd, nh = cfg.head_dim, cfg.n_head
        self.mla = (cfg.attn_type == "mla")
        # In MLA every head gets its own K and V, reconstructed from one shared
        # low-rank code, so there is no head sharing and nkv == nh.
        nkv = nh if self.mla else cfg.n_kv_head
        self.nh, self.nkv, self.hd = nh, nkv, hd
        self.wq = nn.Linear(cfg.d_model, nh * hd, bias=False)
        if self.mla:
            r = cfg.kv_latent_dim
            # C = RMSNorm(X W_down); K = C W_k_up, V = C W_v_up. Both read the
            # SAME latent -- that is what makes it low-rank compression rather
            # than head sharing, and it is the one thing that differs from M1.
            # The norm on the latent is mine, not in the A-3 version: without it
            # the up-projections see an unnormalised bottleneck and the high
            # learning rate Muon runs at becomes unstable.
            self.kv_down = nn.Linear(cfg.d_model, r, bias=False)
            self.kv_norm = RMSNorm(r)
            self.k_up = nn.Linear(r, nh * hd, bias=False)
            self.v_up = nn.Linear(r, nh * hd, bias=False)
        else:
            self.wk = nn.Linear(cfg.d_model, nkv * hd, bias=False)
            self.wv = nn.Linear(cfg.d_model, nkv * hd, bias=False)
        self.wo = nn.Linear(nh * hd, cfg.d_model, bias=False)
        # D1: per-layer gate on the layer-0 value residual, starts near 0 so the
        # model begins as a plain transformer and opts in.
        if cfg.value_residual and layer_idx > 0:
            self.v_gate = nn.Parameter(torch.full((1, 1, nkv * hd), -2.0))
        else:
            self.v_gate = None

    def forward(self, x, cos, sin, v0=None, cache=None, attn_mask=None):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.nh, self.hd).transpose(1, 2)
        if self.mla:
            c = self.kv_norm(self.kv_down(x))
            k = self.k_up(c).view(B, T, self.nkv, self.hd).transpose(1, 2)
            v_flat = self.v_up(c)
        else:
            k = self.wk(x).view(B, T, self.nkv, self.hd).transpose(1, 2)
            v_flat = self.wv(x)
        if self.v_gate is not None and v0 is not None:
            g = torch.sigmoid(self.v_gate).to(v_flat.dtype)
            v_flat = v_flat + g * (v0 - v_flat)          # lerp toward layer-0 v
        v = v_flat.view(B, T, self.nkv, self.hd).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        if self.cfg.qk_norm:                              # D2
            q = F.rms_norm(q, (self.hd,))
            k = F.rms_norm(k, (self.hd,))

        if cache is not None:
            pk, pv = cache
            if pk is not None:
                k = torch.cat([pk, k], dim=2)
                v = torch.cat([pv, v], dim=2)
            new_cache = (k, v)
        else:
            new_cache = None

        causal = attn_mask is None and cache is None
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=causal, enable_gqa=True)
        y = self.wo(y.transpose(1, 2).reshape(B, T, self.nh * self.hd))
        return y, v_flat, new_cache


class Block(nn.Module):
    def __init__(self, cfg: Config, layer_idx: int):
        super().__init__()
        self.n1, self.n2 = RMSNorm(cfg.d_model), RMSNorm(cfg.d_model)
        self.attn = Attention(cfg, layer_idx)
        h = cfg.ffn_hidden
        self.w_gate = nn.Linear(cfg.d_model, h, bias=False)
        self.w_up = nn.Linear(cfg.d_model, h, bias=False)
        self.w_down = nn.Linear(h, cfg.d_model, bias=False)

    def forward(self, x, cos, sin, v0=None, cache=None, attn_mask=None):
        a, v, new_cache = self.attn(self.n1(x), cos, sin, v0, cache, attn_mask)
        x = x + a
        h = self.n2(x)
        x = x + self.w_down(F.silu(self.w_gate(h)) * self.w_up(h))
        return x, v, new_cache


class LexArcD(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.reg = (nn.Parameter(torch.zeros(1, cfg.n_register, cfg.d_model))
                    if cfg.n_register else None)       # D3
        self.blocks = nn.ModuleList(Block(cfg, i) for i in range(cfg.n_layer))
        self.norm = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.head.weight = self.emb.weight
        self.apply(self._init)
        if self.reg is not None:
            nn.init.normal_(self.reg, std=0.02)
        # depth-scaled init on the residual projections
        for n, p in self.named_parameters():
            if n.endswith("wo.weight") or n.endswith("w_down.weight"):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def n_params(self, non_embedding=False):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.emb.weight.numel()
        return n

    def _trunk(self, idx, cos, sin, caches=None, attn_mask=None):
        B, T = idx.shape
        x = self.emb(idx)
        R = self.cfg.n_register
        if R and caches is None:                 # registers only on a full pass
            x = torch.cat([self.reg.expand(B, -1, -1).to(x.dtype), x], 1)
        v0, new_caches = None, []
        for i, blk in enumerate(self.blocks):
            c = caches[i] if caches is not None else None
            x, v, nc = blk(x, cos, sin, v0, c, attn_mask)
            if i == 0:
                v0 = v
            new_caches.append(nc)
        return self.norm(x), new_caches

    def forward(self, idx, loss_start=1):
        """idx -> scalar loss. There is no labels argument, by design.

        loss_start: index of the first TARGET token to score. 1 (the default)
            is ordinary causal LM over the whole window. A larger value scores
            only the tail, which is how task-format SFT is done here: the first
            630 tokens are prompt context the model may read but is never
            graded on, and the last ~170 are the continuation -- the same split
            the competition evaluates. The shift still happens exactly once,
            right here.
        """
        R = self.cfg.n_register
        cos, sin = build_rope(idx.size(1) + R, self.cfg.head_dim, idx.device,
                              self.emb.weight.dtype, self.cfg.rope_theta)
        h, _ = self._trunk(idx, cos, sin)
        h = h[:, R:]                                    # drop register positions
        st = max(1, int(loss_start))
        logits = self.head(h[:, st - 1:-1])             # position i predicts i+1
        targets = idx[:, st:]
        # No .float() on the logits: under autocast cross_entropy upcasts
        # internally, so an explicit cast only materialises a 3.2 GB fp32
        # tensor every step on a card with 300 GB/s of bandwidth.
        flat = logits.reshape(-1, self.cfg.vocab_size)
        loss = F.cross_entropy(flat, targets.reshape(-1))
        if self.cfg.z_loss:
            lse = torch.logsumexp(flat, -1).float()
            loss = loss + self.cfg.z_loss * (lse ** 2).mean()
        return loss

    @torch.no_grad()
    def logits_full(self, idx):
        """Every position's logits. Used by the alignment gate, not by training."""
        R = self.cfg.n_register
        cos, sin = build_rope(idx.size(1) + R, self.cfg.head_dim, idx.device,
                              self.emb.weight.dtype, self.cfg.rope_theta)
        h, _ = self._trunk(idx, cos, sin)
        return self.head(h[:, R:])

    # ------------------------------------------------------------- generation
    @torch.no_grad()
    def generate(self, idx, pad_mask, max_new_tokens, temperature=0.8, top_p=0.9,
                 top_k=0, repetition_penalty=1.0, copy_bias=0.0, greedy=False,
                 eos_id=None, use_cache=True, no_repeat_ngram=0,
                 copy_scope="prompt"):
        """idx: (B, T) LEFT-padded. pad_mask: (B, T) True where real.

        Left padding, because a continuation must start immediately after the
        last real prompt token. RoPE positions are offset per sequence so the
        first real token is always at position 0.
        """
        cfg = self.cfg
        B, T = idx.shape
        dev = idx.device
        R = cfg.n_register
        n_pad = (~pad_mask).sum(1)                        # (B,)

        # positions: registers 0..R-1, then 0..len-1 for the real tokens
        pos = (torch.arange(T, device=dev)[None] - n_pad[:, None]).clamp(min=0)
        if R:
            reg_pos = torch.arange(R, device=dev)[None].expand(B, -1)
            pos = torch.cat([reg_pos, pos + R], 1)
        full_valid = pad_mask
        if R:
            full_valid = torch.cat([torch.ones(B, R, dtype=torch.bool, device=dev),
                                    pad_mask], 1)

        prompt_tokens = idx.clone()
        out = []
        caches = None
        cur = idx

        # per-sequence RoPE tables, since each sequence has its own offset
        def rope_for(positions):
            hd = cfg.head_dim
            inv = 1.0 / (cfg.rope_theta ** (torch.arange(0, hd, 2, device=dev).float() / hd))
            f = positions.float()[..., None] * inv            # (B, L, hd/2)
            e = torch.cat([f, f], -1)
            return e.cos()[:, None].to(self.emb.weight.dtype), \
                   e.sin()[:, None].to(self.emb.weight.dtype)

        Lf = T + R
        cos, sin = rope_for(pos)
        # (B,1,Lf,Lf): causal AND key must be a real token
        causal = torch.ones(Lf, Lf, dtype=torch.bool, device=dev).tril()
        mask = causal[None, None] & full_valid[:, None, None, :]
        h, caches = self._trunk_gen(cur, cos, sin, None, mask)
        logits = self.head(h[:, -1])

        # copy_scope="prompt": bias only tokens that occur in the PROMPT.
        # "all" also accumulates generated tokens, which conflates
        # prompt-copying with self-repetition -- two different behaviours that
        # deserve separate knobs. "prompt" is the intended semantics.
        seen = torch.zeros(B, cfg.vocab_size, dtype=torch.bool, device=dev)
        seen.scatter_(1, prompt_tokens, True)
        accumulate_seen = (copy_scope == "all")
        generated = torch.zeros(B, 0, dtype=torch.long, device=dev)
        done = torch.zeros(B, dtype=torch.bool, device=dev)
        next_pos = pos[:, -1] + 1
        gen_list = [[] for _ in range(B)]
        seen_ng = [{} for _ in range(B)]

        for step in range(max_new_tokens):
            lg = logits.float()
            if copy_bias:
                lg = lg + copy_bias * seen.float()
            if no_repeat_ngram:
                n = no_repeat_ngram
                for b in range(B):
                    if len(gen_list[b]) >= n - 1:
                        ban = seen_ng[b].get(tuple(gen_list[b][-(n - 1):]))
                        if ban:
                            lg[b, list(ban)] = float("-inf")
            if repetition_penalty != 1.0 and generated.numel():
                prev = torch.zeros_like(seen)
                prev.scatter_(1, generated, True)
                lg = torch.where(prev, lg / repetition_penalty, lg)
            if greedy:
                nxt = lg.argmax(-1)
            else:
                lg = lg / max(temperature, 1e-5)
                if top_k:
                    kth = lg.topk(top_k, -1).values[:, -1:]
                    lg = lg.masked_fill(lg < kth, float("-inf"))
                if top_p < 1.0:
                    s, si = lg.sort(-1, descending=True)
                    cp = s.softmax(-1).cumsum(-1)
                    drop = cp - s.softmax(-1) > top_p
                    s = s.masked_fill(drop, float("-inf"))
                    lg = torch.full_like(lg, float("-inf")).scatter_(1, si, s)
                nxt = torch.multinomial(lg.softmax(-1), 1).squeeze(1)
            if eos_id is not None:
                nxt = torch.where(done, torch.full_like(nxt, eos_id), nxt)
                done = done | (nxt == eos_id)
            generated = torch.cat([generated, nxt[:, None]], 1)
            if accumulate_seen:
                seen.scatter_(1, nxt[:, None], True)
            if no_repeat_ngram:
                n = no_repeat_ngram
                nx = nxt.tolist()
                for b in range(B):
                    gen_list[b].append(nx[b])
                    if len(gen_list[b]) >= n:
                        seen_ng[b].setdefault(
                            tuple(gen_list[b][-n:-1]), set()).add(gen_list[b][-1])
            if done.all():
                break
            cos1, sin1 = rope_for(next_pos[:, None])
            kmask = torch.cat([full_valid,
                               torch.ones(B, generated.size(1), dtype=torch.bool, device=dev)], 1)
            h, caches = self._trunk_gen(nxt[:, None], cos1, sin1, caches,
                                        kmask[:, None, None, :])
            logits = self.head(h[:, -1])
            next_pos = next_pos + 1
        return generated

    def _trunk_gen(self, idx, cos, sin, caches, attn_mask):
        B, T = idx.shape
        x = self.emb(idx)
        R = self.cfg.n_register
        if R and caches is None:
            x = torch.cat([self.reg.expand(B, -1, -1).to(x.dtype), x], 1)
        v0, new_caches = None, []
        for i, blk in enumerate(self.blocks):
            c = caches[i] if caches is not None else (None, None)
            x, v, nc = blk(x, cos, sin, v0, c, attn_mask)
            if i == 0:
                v0 = v
            new_caches.append(nc)
        return self.norm(x), new_caches


# ------------------------------------------------------------------- Muon ---
def _newton_schulz(G, steps=5, eps=1e-7):
    """Orthogonalise a STACK of equally-shaped matrices, G: (N, R, C).

    Batched rather than looped. The model has 112 hidden matrices in only four
    distinct shapes, so running them one at a time meant ~1,680 small
    sequential matmuls per optimizer step -- launch-bound, and measured at
    roughly 0.4 s/step on an L4. Bucketed by shape this is four bmm calls.
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """Momentum-orthogonalised SGD for 2-D hidden weights.

    Not for embeddings, the output head, norms or any 1-D parameter -- those go
    to AdamW. Reported ~1.3-1.5x fewer steps to equal validation loss, which on
    a fixed wall-clock budget is worth more than any single architectural
    change in this model.
    """

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 ns_steps=5, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov,
                                      ns_steps=ns_steps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr, mom, wd = group["lr"], group["momentum"], group["weight_decay"]
            buckets = {}
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if "m" not in st:
                    st["m"] = torch.zeros_like(g)
                buf = st["m"]
                buf.lerp_(g, 1.0 - mom)
                d = g.lerp(buf, mom) if group["nesterov"] else buf
                buckets.setdefault(tuple(d.shape), []).append((p, d))
            for shape, items in buckets.items():
                O = _newton_schulz(torch.stack([d for _, d in items]),
                                   group["ns_steps"])
                scale = max(1.0, shape[0] / shape[1]) ** 0.5
                for (p, _), o in zip(items, O):
                    if wd:
                        p.mul_(1.0 - lr * wd)
                    p.add_(o, alpha=-lr * scale)


def build_optimizers(model, muon_lr=0.025, adam_lr=1e-3, weight_decay=0.1):
    """2-D hidden matrices -> Muon. Everything else -> AdamW."""
    muon_p, adam_decay, adam_plain = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2 and "emb" not in n and "head" not in n:
            muon_p.append(p)
        elif p.ndim >= 2:
            adam_decay.append(p)
        else:
            adam_plain.append(p)
    opts = [
        Muon(muon_p, lr=muon_lr, weight_decay=0.0),
        torch.optim.AdamW([
            {"params": adam_decay, "weight_decay": weight_decay},
            {"params": adam_plain, "weight_decay": 0.0},
        ], lr=adam_lr, betas=(0.9, 0.95), eps=1e-8, fused=True),
    ]
    return opts, dict(muon=len(muon_p), adam=len(adam_decay) + len(adam_plain))
