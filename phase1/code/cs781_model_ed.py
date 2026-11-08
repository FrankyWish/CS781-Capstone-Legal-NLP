"""CS-781 Phase 1 -- LexArc-ED: encoder-decoder, shaped to the task.

Architecture #2 of the four §7.3 requires ("Encoder-decoder Transformer"), and
deliberately not a second decoder with different hyperparameters.

WHY AN ENCODER-DECODER HERE, SPECIFICALLY

The task is 512 words in, 128 words out. The prompt is text we only ever
*read*. A bidirectional encoder gives every prompt token two-sided context for
the same parameters; a causal model cannot do that at any width. That is a real
inductive-bias argument for this task shape, not a label change.

WHY NOT A PREFIX-LM, which would get the same bidirectional prompt more cheaply:
a prefix mask must be materialised and passed to SDPA, which drops off the
flash kernel. Here all three attention calls run maskless --
encoder is_causal=False, decoder self is_causal=True, cross unmasked -- so we
keep the fast path everywhere. The architecture was chosen partly by what the
kernel supports, which is the kind of decision that only shows up when you
benchmark before you design.

THE HONEST PROBLEM, AND MY FIX

A plain seq2seq objective puts loss only on the decoder half, so ~45% of tokens
carry gradient against M1's 100%. At iso-compute that is a real ~2x
data-efficiency penalty and it is the reason encoder-decoders lose these
comparisons. So the objective is joint:

    L = CE_decoder(next-token) + lambda * CE_encoder(masked-LM on 15%)

The encoder half stops being free context and becomes supervision, for one
extra softmax over ~100 positions, under 2% of wall-clock. This is not T5 span
corruption and not BART denoising -- it exists to recover gradient signal that
the seq2seq objective throws away. lambda=0 is the ablation.

Same structural rule as LexArcD: forward() takes token tensors and builds its
own decoder inputs and targets internally. No labels argument anywhere.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from cs781_model import RMSNorm, build_rope, apply_rope  # shared, already gated


@dataclass
class EDConfig:
    vocab_size: int = 32768
    d_model: int = 768
    n_enc_layer: int = 10
    n_dec_layer: int = 8
    n_head: int = 12
    n_kv_head: int = 4
    ffn_hidden: int = 2048
    window: int = 1280                       # enc_len + dec_len
    split_points: tuple = (576, 704, 832)    # centred on the ~624-token test prompt
    rope_theta: float = 10000.0
    z_loss: float = 1e-4
    qk_norm: bool = True
    mlm_lambda: float = 0.3                  # 0.0 is the ablation
    mlm_rate: float = 0.15
    mask_id: int = 32767
    bos_id: int = 32764                      # <|endoftext|> doubles as decoder BOS

    @property
    def head_dim(self):
        return self.d_model // self.n_head


class SelfAttn(nn.Module):
    def __init__(self, cfg, causal):
        super().__init__()
        hd, nh, nkv = cfg.head_dim, cfg.n_head, cfg.n_kv_head
        self.cfg, self.causal = cfg, causal
        self.nh, self.nkv, self.hd = nh, nkv, hd
        self.wq = nn.Linear(cfg.d_model, nh * hd, bias=False)
        self.wk = nn.Linear(cfg.d_model, nkv * hd, bias=False)
        self.wv = nn.Linear(cfg.d_model, nkv * hd, bias=False)
        self.wo = nn.Linear(nh * hd, cfg.d_model, bias=False)

    def forward(self, x, cos, sin, cache=None):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.nh, self.hd).transpose(1, 2)
        k = self.wk(x).view(B, T, self.nkv, self.hd).transpose(1, 2)
        v = self.wv(x).view(B, T, self.nkv, self.hd).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if self.cfg.qk_norm:
            q = F.rms_norm(q, (self.hd,))
            k = F.rms_norm(k, (self.hd,))
        new_cache = None
        if cache is not None:
            pk, pv = cache
            if pk is not None:
                k, v = torch.cat([pk, k], 2), torch.cat([pv, v], 2)
            new_cache = (k, v)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=(self.causal and cache is None and T > 1),
            enable_gqa=True)
        return self.wo(y.transpose(1, 2).reshape(B, T, self.nh * self.hd)), new_cache


class CrossAttn(nn.Module):
    """Decoder queries, encoder keys/values. No RoPE: the two position spaces
    are unrelated, and cross-attention is a lookup, not a sequence model."""

    def __init__(self, cfg):
        super().__init__()
        hd, nh, nkv = cfg.head_dim, cfg.n_head, cfg.n_kv_head
        self.nh, self.nkv, self.hd = nh, nkv, hd
        self.wq = nn.Linear(cfg.d_model, nh * hd, bias=False)
        self.wk = nn.Linear(cfg.d_model, nkv * hd, bias=False)
        self.wv = nn.Linear(cfg.d_model, nkv * hd, bias=False)
        self.wo = nn.Linear(nh * hd, cfg.d_model, bias=False)

    def kv(self, enc):
        B, S, _ = enc.shape
        k = self.wk(enc).view(B, S, self.nkv, self.hd).transpose(1, 2)
        v = self.wv(enc).view(B, S, self.nkv, self.hd).transpose(1, 2)
        return k, v

    def forward(self, x, kv, enc_mask=None):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.nh, self.hd).transpose(1, 2)
        k, v = kv
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=enc_mask,
                                           enable_gqa=True)
        return self.wo(y.transpose(1, 2).reshape(B, T, self.nh * self.hd))


def _ffn(cfg):
    return nn.ModuleDict(dict(
        g=nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False),
        u=nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False),
        d=nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)))


def _apply_ffn(m, h):
    return m["d"](F.silu(m["g"](h)) * m["u"](h))


class EncBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n1, self.n2 = RMSNorm(cfg.d_model), RMSNorm(cfg.d_model)
        self.attn = SelfAttn(cfg, causal=False)
        self.ffn = _ffn(cfg)

    def forward(self, x, cos, sin):
        a, _ = self.attn(self.n1(x), cos, sin)
        x = x + a
        return x + _apply_ffn(self.ffn, self.n2(x))


class DecBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n1 = RMSNorm(cfg.d_model)
        self.n2 = RMSNorm(cfg.d_model)
        self.n3 = RMSNorm(cfg.d_model)
        self.attn = SelfAttn(cfg, causal=True)
        self.cross = CrossAttn(cfg)
        self.ffn = _ffn(cfg)

    def forward(self, x, cos, sin, enc_kv, cache=None, enc_mask=None):
        a, nc = self.attn(self.n1(x), cos, sin, cache)
        x = x + a
        x = x + self.cross(self.n2(x), enc_kv, enc_mask)
        return x + _apply_ffn(self.ffn, self.n3(x)), nc


class LexArcED(nn.Module):
    def __init__(self, cfg: EDConfig):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.enc = nn.ModuleList(EncBlock(cfg) for _ in range(cfg.n_enc_layer))
        self.dec = nn.ModuleList(DecBlock(cfg) for _ in range(cfg.n_dec_layer))
        self.enc_norm = RMSNorm(cfg.d_model)
        self.dec_norm = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.emb.weight                     # tied everywhere
        self.apply(self._init)
        for n, p in self.named_parameters():
            if n.endswith("wo.weight") or n.endswith("ffn.d.weight"):
                nn.init.normal_(p, std=0.02 / math.sqrt(
                    2 * (cfg.n_enc_layer + cfg.n_dec_layer)))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def n_params(self, non_embedding=False):
        n = sum(p.numel() for p in self.parameters())
        return n - self.emb.weight.numel() if non_embedding else n

    def encode(self, enc_in):
        cfg = self.cfg
        cos, sin = build_rope(enc_in.size(1), cfg.head_dim, enc_in.device,
                              self.emb.weight.dtype, cfg.rope_theta)
        x = self.emb(enc_in)
        for blk in self.enc:
            x = blk(x, cos, sin)
        return self.enc_norm(x)

    def _mask_for_mlm(self, enc_in):
        """BERT-style 80/10/10, so the encoder is not only ever shown <|mask|>."""
        cfg = self.cfg
        sel = torch.rand_like(enc_in, dtype=torch.float) < cfg.mlm_rate
        inp = enc_in.clone()
        r = torch.rand_like(enc_in, dtype=torch.float)
        inp[sel & (r < 0.8)] = cfg.mask_id
        rnd = torch.randint_like(enc_in, 0, cfg.vocab_size)
        swap = sel & (r >= 0.8) & (r < 0.9)
        inp[swap] = rnd[swap]
        return inp, sel

    def forward(self, window, split):
        """window: (B, L) raw tokens. split: int. Targets are built here, once."""
        cfg = self.cfg
        enc_in, dec_tgt = window[:, :split], window[:, split:]

        if cfg.mlm_lambda > 0 and self.training:
            enc_inp, sel = self._mask_for_mlm(enc_in)
        else:
            enc_inp, sel = enc_in, None
        enc_out = self.encode(enc_inp)

        # decoder input is BOS + targets[:-1]; targets are the raw window tail
        bos = torch.full((window.size(0), 1), cfg.bos_id, device=window.device,
                         dtype=window.dtype)
        dec_in = torch.cat([bos, dec_tgt[:, :-1]], 1)
        cos, sin = build_rope(dec_in.size(1), cfg.head_dim, window.device,
                              self.emb.weight.dtype, cfg.rope_theta)
        enc_kv = [blk.cross.kv(enc_out) for blk in self.dec]
        x = self.emb(dec_in)
        for blk, kv in zip(self.dec, enc_kv):
            x, _ = blk(x, cos, sin, kv)
        logits = self.head(self.dec_norm(x))
        flat = logits.reshape(-1, cfg.vocab_size)
        loss = F.cross_entropy(flat, dec_tgt.reshape(-1))
        if cfg.z_loss:
            loss = loss + cfg.z_loss * (torch.logsumexp(flat, -1).float() ** 2).mean()

        mlm = torch.zeros((), device=loss.device)
        if sel is not None and sel.any():
            mh = self.head(enc_out[sel])
            mlm = F.cross_entropy(mh, enc_in[sel])
            loss = loss + cfg.mlm_lambda * mlm
        return loss, mlm.detach()

    @torch.no_grad()
    def logits_full(self, window, split):
        cfg = self.cfg
        enc_in, dec_tgt = window[:, :split], window[:, split:]
        enc_out = self.encode(enc_in)
        bos = torch.full((window.size(0), 1), cfg.bos_id, device=window.device,
                         dtype=window.dtype)
        dec_in = torch.cat([bos, dec_tgt[:, :-1]], 1)
        cos, sin = build_rope(dec_in.size(1), cfg.head_dim, window.device,
                              self.emb.weight.dtype, cfg.rope_theta)
        enc_kv = [blk.cross.kv(enc_out) for blk in self.dec]
        x = self.emb(dec_in)
        for blk, kv in zip(self.dec, enc_kv):
            x, _ = blk(x, cos, sin, kv)
        return self.head(self.dec_norm(x))

    @torch.no_grad()
    def generate(self, enc_in, enc_mask, max_new_tokens, temperature=0.8,
                 top_p=0.9, top_k=0, repetition_penalty=1.0, copy_bias=0.0,
                 greedy=False, eos_id=None, no_repeat_ngram=0,
                 copy_scope="prompt", **_):
        """enc_in: (B, S) LEFT-padded prompt. enc_mask: (B, S) True where real.

        The encoder runs once. Cross-attention K/V are computed once. Only the
        decoder's own self-attention needs an incremental cache -- which is the
        structural reason this architecture generates cheaply.
        """
        cfg = self.cfg
        B = enc_in.size(0)
        dev = enc_in.device
        enc_out = self.encode(enc_in)
        enc_kv = [blk.cross.kv(enc_out) for blk in self.dec]
        xmask = enc_mask[:, None, None, :]                 # (B,1,1,S)

        seen = torch.zeros(B, cfg.vocab_size, dtype=torch.bool, device=dev)
        seen.scatter_(1, enc_in, True)
        accumulate_seen = (copy_scope == "all")
        cur = torch.full((B, 1), cfg.bos_id, device=dev, dtype=torch.long)
        caches = [(None, None)] * cfg.n_dec_layer
        generated = torch.zeros(B, 0, dtype=torch.long, device=dev)
        gen_list = [[] for _ in range(B)]
        seen_ng = [{} for _ in range(B)]

        for step in range(max_new_tokens):
            cos, sin = build_rope(1, cfg.head_dim, dev, self.emb.weight.dtype,
                                  cfg.rope_theta, offset=step)
            x = self.emb(cur)
            new_caches = []
            for blk, kv, c in zip(self.dec, enc_kv, caches):
                x, nc = blk(x, cos, sin, kv, c, xmask)
                new_caches.append(nc)
            caches = new_caches
            lg = self.head(self.dec_norm(x))[:, -1].float()

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
                    lg = lg.masked_fill(lg < lg.topk(top_k, -1).values[:, -1:],
                                        float("-inf"))
                if top_p < 1.0:
                    s, si = lg.sort(-1, descending=True)
                    pr = s.softmax(-1)
                    drop = pr.cumsum(-1) - pr > top_p
                    s = s.masked_fill(drop, float("-inf"))
                    lg = torch.full_like(lg, float("-inf")).scatter_(1, si, s)
                nxt = torch.multinomial(lg.softmax(-1), 1).squeeze(1)
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
            cur = nxt[:, None]
        return generated
