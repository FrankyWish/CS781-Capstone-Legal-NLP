"""CS-781 Phase 1 -- LexArc-P: decoder with a learned pointer-generator head.

Architecture #4, and the only one in this project whose design came from a
measurement rather than from reading.

THE EVIDENCE

A crude, untrained logit bias toward tokens present in the prompt was worth
+0.0028 BERTScore on M1 and +0.0025 on M2 -- two architecturally unrelated
models, near-identical magnitude, both outside the bootstrap noise. It
saturates around bias 3.0 and degrades past 4.0, because it is indiscriminate:
it pushes *every* prompt token up by the same amount at *every* step, whether
or not copying is the right move there.

So the models systematically under-copy, and the ceiling of the crude fix is
about +0.002. A head that learns *when* to copy and *which* token to copy
should do better, and that is the hypothesis this architecture tests.

THE MECHANISM

At each position the next-token distribution is a learned mixture:

    p(y) = g * p_vocab(y)  +  (1 - g) * sum_{j <= t, x_j = y} a_tj

p_vocab is the ordinary softmax over the tied embedding. a is a dedicated
single-head attention over everything seen so far, and its mass on position j
becomes probability mass on the token that sits at j. g is a learned gate
driven by the hidden state and the attended context vector, so the model can
open and close the copy channel per token.

This is See et al.'s pointer-generator, implemented here rather than imported,
adapted to causal language modelling: the copy source is the model's own left
context rather than a separate encoder, which makes it an explicit, trainable
induction head.

COST: 148,288 parameters -- three d_model x 64 projections and a gate -- on top
of M1's 125.9M. **M1 and M4 are iso-parameter to within 0.12%**, differ in
exactly one mechanism, and share every other component, hyperparameter and byte
of training data. That is as clean as an architectural comparison gets.

NUMERICS: the mixture is formed in log space with logaddexp. Mixing two
probabilities and then taking a log invites underflow when p_vocab(y) is ~1e-10,
which it routinely is early in training.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from cs781_model import Config, LexArcD, build_rope


class LexArcP(LexArcD):
    """LexArcD plus a pointer-generator output head. Trunk is untouched."""

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        dc = cfg.copy_dim
        self.cq = nn.Linear(cfg.d_model, dc, bias=False)
        self.ck = nn.Linear(cfg.d_model, dc, bias=False)
        self.cv = nn.Linear(cfg.d_model, dc, bias=False)
        self.gate = nn.Linear(cfg.d_model + dc, 1)
        nn.init.normal_(self.cq.weight, std=0.02)
        nn.init.normal_(self.ck.weight, std=0.02)
        nn.init.normal_(self.cv.weight, std=0.02)
        nn.init.zeros_(self.gate.weight)
        # Start the gate wide open on the vocabulary path: sigmoid(3) = 0.95.
        # The model begins as an ordinary LM and has to learn to open the copy
        # channel, so a broken copy path cannot poison early training.
        nn.init.constant_(self.gate.bias, 3.0)

    # ------------------------------------------------------------------ core
    def _hidden(self, idx):
        R = self.cfg.n_register
        cos, sin = build_rope(idx.size(1) + R, self.cfg.head_dim, idx.device,
                              self.emb.weight.dtype, self.cfg.rope_theta)
        h, _ = self._trunk(idx, cos, sin)
        return h[:, R:]

    NEG = -1.0e4   # a finite stand-in for -inf: see the note in forward()

    def _copy_terms(self, hs, h, idx, tgt, q_offset):
        """Returns (log P(copy target), log g, log(1-g), attn).

        Everything stays in log space. The first version of this computed
        probabilities and then took logs with an epsilon guard, and it produced
        NaN after ~700 steps: copy_p is EXACTLY zero whenever the target token
        does not occur in the left context, which is most positions, and
        d/dx log(x + 1e-9) at x = 0 is 1e9. Every such position pushed a
        billion-scale gradient into the copy attention until something
        overflowed.
        """
        B, Q, _ = hs.shape
        T = h.size(1)
        dc = self.cq.out_features
        qc, kc, vc = self.cq(hs), self.ck(h), self.cv(h)
        scores = (qc @ kc.transpose(-1, -2)).float() / math.sqrt(dc)   # (B,Q,T)
        qpos = torch.arange(q_offset, q_offset + Q, device=idx.device)[:, None]
        kpos = torch.arange(T, device=idx.device)[None, :]
        # Finite, not -inf, so log_softmax never differentiates an infinity.
        scores = scores.masked_fill(kpos > qpos, self.NEG)
        log_attn = F.log_softmax(scores, -1)                           # (B,Q,T)
        match = (idx[:, None, :] == tgt[:, :, None])                   # (B,Q,T)
        # logsumexp over the matching positions IS log(sum of their attention).
        # A row with no match lands at ~NEG rather than -inf, so logaddexp
        # later returns the vocabulary term with a vanishing gradient on this
        # branch instead of a NaN.
        log_copy = torch.logsumexp(log_attn.masked_fill(~match, self.NEG), -1)
        attn = log_attn.exp()
        ctx = attn.to(vc.dtype) @ vc                                   # (B,Q,dc)
        gl = self.gate(torch.cat([hs, ctx], -1)).float().squeeze(-1)   # (B,Q)
        return log_copy, F.logsigmoid(gl), F.logsigmoid(-gl), attn

    def forward(self, idx, loss_start=1, force_gate=None):
        """idx -> scalar loss. force_gate=1.0 reduces this to a plain LM, which
        is what gate G1 uses to prove the vocabulary path is aligned."""
        cfg = self.cfg
        V = cfg.vocab_size
        h = self._hidden(idx)
        st = max(1, int(loss_start))
        hs = h[:, st - 1:-1]
        tgt = idx[:, st:]
        B, Q, _ = hs.shape

        logits = self.head(hs)
        # cross_entropy upcasts internally under autocast; an explicit .float()
        # here would materialise a multi-GB tensor every step.
        lp_tgt = -F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1),
                                  reduction="none").view(B, Q)

        if force_gate is not None:
            loss = -lp_tgt.mean()
        else:
            log_copy, log_g, log_1mg, _ = self._copy_terms(hs, h, idx, tgt, st - 1)
            loss = -torch.logaddexp(log_g + lp_tgt, log_1mg + log_copy).mean()
        if cfg.z_loss:
            loss = loss + cfg.z_loss * (torch.logsumexp(logits, -1).float() ** 2).mean()
        return loss

    @torch.no_grad()
    def mixed_logprobs(self, idx, loss_start=1):
        """Full mixed log-distribution at every query position. Gate G2 only --
        this materialises (B, Q, V) and must not be called during training."""
        cfg, V = self.cfg, self.cfg.vocab_size
        h = self._hidden(idx)
        st = max(1, int(loss_start))
        hs = h[:, st - 1:-1]
        B, Q, _ = hs.shape
        lp_vocab = F.log_softmax(self.head(hs).float(), -1)           # (B,Q,V)
        dc = self.cq.out_features
        qc, kc, vc = self.cq(hs), self.ck(h), self.cv(h)
        scores = (qc @ kc.transpose(-1, -2)).float() / math.sqrt(dc)
        qpos = torch.arange(st - 1, st - 1 + Q, device=idx.device)[:, None]
        kpos = torch.arange(h.size(1), device=idx.device)[None, :]
        attn = scores.masked_fill(kpos > qpos, self.NEG).softmax(-1)
        ctx = attn.to(vc.dtype) @ vc
        g = torch.sigmoid(self.gate(torch.cat([hs, ctx], -1)).float())
        copy = torch.zeros(B, Q, V, device=idx.device, dtype=torch.float32)
        copy.scatter_add_(2, idx[:, None, :].expand(B, Q, idx.size(1)), attn)
        return torch.log(g * lp_vocab.exp() + (1 - g) * copy + 1e-9)

    # ------------------------------------------------------------ generation
    @torch.no_grad()
    def generate(self, idx, pad_mask, max_new_tokens, temperature=0.8, top_p=0.9,
                 top_k=0, repetition_penalty=1.0, copy_bias=0.0, greedy=False,
                 eos_id=None, use_cache=True, no_repeat_ngram=0,
                 copy_scope="prompt"):
        """Same contract as LexArcD.generate. The copy head needs the hidden
        state of every past position, so those are cached alongside the KV."""
        cfg = self.cfg
        V, dc = cfg.vocab_size, self.cq.out_features
        B, T = idx.shape
        dev = idx.device
        R = cfg.n_register
        n_pad = (~pad_mask).sum(1)

        pos = (torch.arange(T, device=dev)[None] - n_pad[:, None]).clamp(min=0)
        if R:
            pos = torch.cat([torch.arange(R, device=dev)[None].expand(B, -1),
                             pos + R], 1)
        full_valid = pad_mask
        if R:
            full_valid = torch.cat(
                [torch.ones(B, R, dtype=torch.bool, device=dev), pad_mask], 1)

        def rope_for(p_):
            inv = 1.0 / (cfg.rope_theta ** (torch.arange(0, cfg.head_dim, 2,
                                                         device=dev).float() / cfg.head_dim))
            f = p_.float()[..., None] * inv
            e = torch.cat([f, f], -1)
            dt = self.emb.weight.dtype
            return e.cos()[:, None].to(dt), e.sin()[:, None].to(dt)

        Lf = T + R
        cos, sin = rope_for(pos)
        causal = torch.ones(Lf, Lf, dtype=torch.bool, device=dev).tril()
        mask = causal[None, None] & full_valid[:, None, None, :]
        h_all, caches = self._trunk_gen(idx, cos, sin, None, mask)
        h_all = h_all[:, R:]                                 # (B,T,D) real only

        tok_hist = idx.clone()                               # (B,T)
        valid = pad_mask.clone()                             # (B,T)
        kc_hist, vc_hist = self.ck(h_all), self.cv(h_all)    # (B,T,dc)
        h_cur = h_all[:, -1:]                                # (B,1,D)

        prompt_seen = torch.zeros(B, V, dtype=torch.bool, device=dev)
        prompt_seen.scatter_(1, idx, True)
        accumulate = (copy_scope == "all")
        generated = torch.zeros(B, 0, dtype=torch.long, device=dev)
        gen_list, seen_ng = [[] for _ in range(B)], [{} for _ in range(B)]
        next_pos = pos[:, -1] + 1

        for step in range(max_new_tokens):
            p_vocab = self.head(h_cur[:, -1]).float().softmax(-1)      # (B,V)
            qc = self.cq(h_cur[:, -1:])                                # (B,1,dc)
            sc = (qc @ kc_hist.transpose(-1, -2)).float() / math.sqrt(dc)
            sc = sc.masked_fill(~valid[:, None, :], self.NEG)
            attn = sc.softmax(-1)                                      # (B,1,Tc)
            ctx = attn.to(vc_hist.dtype) @ vc_hist                     # (B,1,dc)
            g = torch.sigmoid(self.gate(torch.cat([h_cur[:, -1:], ctx], -1)).float())
            copy = torch.zeros(B, V, device=dev, dtype=torch.float32)
            copy.scatter_add_(1, tok_hist, attn.squeeze(1))
            probs = g.squeeze(1) * p_vocab + (1 - g.squeeze(1)) * copy
            lg = torch.log(probs.clamp_min(1e-20))

            if copy_bias:
                lg = lg + copy_bias * (prompt_seen.float() if not accumulate
                                       else prompt_seen.float())
            if repetition_penalty != 1.0 and generated.numel():
                prev = torch.zeros(B, V, dtype=torch.bool, device=dev)
                prev.scatter_(1, generated, True)
                lg = torch.where(prev, lg / repetition_penalty, lg)
            if no_repeat_ngram:
                n = no_repeat_ngram
                for b in range(B):
                    if len(gen_list[b]) >= n - 1:
                        ban = seen_ng[b].get(tuple(gen_list[b][-(n - 1):]))
                        if ban:
                            lg[b, list(ban)] = float("-inf")
            if greedy:
                nxt = lg.argmax(-1)
            else:
                lg = lg / max(temperature, 1e-5)
                if top_k:
                    lg = lg.masked_fill(lg < lg.topk(top_k, -1).values[:, -1:],
                                        float("-inf"))
                if top_p < 1.0:
                    s_, si = lg.sort(-1, descending=True)
                    pr = s_.softmax(-1)
                    s_ = s_.masked_fill(pr.cumsum(-1) - pr > top_p, float("-inf"))
                    lg = torch.full_like(lg, float("-inf")).scatter_(1, si, s_)
                nxt = torch.multinomial(lg.softmax(-1), 1).squeeze(1)

            generated = torch.cat([generated, nxt[:, None]], 1)
            if accumulate:
                prompt_seen.scatter_(1, nxt[:, None], True)
            if no_repeat_ngram:
                n, nx = no_repeat_ngram, nxt.tolist()
                for b in range(B):
                    gen_list[b].append(nx[b])
                    if len(gen_list[b]) >= n:
                        seen_ng[b].setdefault(tuple(gen_list[b][-n:-1]),
                                              set()).add(gen_list[b][-1])

            cos1, sin1 = rope_for(next_pos[:, None])
            kmask = torch.cat([full_valid, torch.ones(B, generated.size(1),
                                                      dtype=torch.bool, device=dev)], 1)
            h_new, caches = self._trunk_gen(nxt[:, None], cos1, sin1, caches,
                                            kmask[:, None, None, :])
            h_cur = h_new
            kc_hist = torch.cat([kc_hist, self.ck(h_new)], 1)
            vc_hist = torch.cat([vc_hist, self.cv(h_new)], 1)
            tok_hist = torch.cat([tok_hist, nxt[:, None]], 1)
            valid = torch.cat([valid, torch.ones(B, 1, dtype=torch.bool, device=dev)], 1)
            next_pos = next_pos + 1
        return generated
