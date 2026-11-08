"""CS-781 Phase 6 -- adaptation methods (Phase 5 module; + Pfeiffer adapters, decoder blocks of M2).

Every method here is implemented directly on the Phase-1 modules (no peft
library: the trunks are custom nn.Modules, and prefix/prompt tuning in peft only
works on HF transformers classes).

    method    trainable in the trunk                         reference
    head      nothing (frozen trunk, train the RR head)      floor
    full      everything                                     ceiling to compare against
    lora      B A added to frozen W, scale alpha/r           Hu et al. 2021
    rslora    same, scale alpha/sqrt(r), higher rank         Kalajdzievski 2023
    qlora     frozen W stored as NF4 (+ double-quantised     Dettmers et al. 2023
              absmax), LoRA on every linear layer
    adalora   W + P diag(E) Q, rank budget reallocated by    Zhang et al. 2023
              sensitivity-based importance during training
    lorma     (I + alpha/r B A) W -- multiplicative          Bihany, Patel & Modi 2025
    ia3       learned rescaling of k, v and FFN activation   Liu et al. 2022
    prompt    P soft tokens prepended to the input           Lester et al. 2021
    prefix    P learned states at every layer (MLP reparam)  Li & Liang 2021
    norm      RMSNorm gains only (BitFit analogue; the       Zaken et al. 2022
              Phase-1 blocks have no biases)
    adapter   bottleneck MLP after each attention and FFN    Houlsby et al. 2019
              sublayer output (before the residual add)
    partial   top-K blocks + final norm trained, rest        Howard & Ruder 2018
              frozen; layer-wise LR decay inside the K       (discriminative FT)

`prompt` and `prefix` are handled in the model's forward pass (they change the
sequence, not a weight); this module only builds their parameters.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------- LoRA ------
class LoRALinear(nn.Module):
    """y = W x + s * B(A x);  s = alpha/r (LoRA) or alpha/sqrt(r) (rsLoRA).

    rsLoRA's point: with alpha/r the update's scale shrinks as r grows, so a
    larger rank learns no faster and the extra capacity goes unused. alpha/sqrt(r)
    keeps the learning dynamics rank-stable, which is what makes r=64 worth
    trying at all.
    """

    def __init__(self, base, r, alpha=16.0, rs=False, dropout=0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale = alpha / (math.sqrt(r) if rs else r)
        self.drop = nn.Dropout(dropout) if dropout else nn.Identity()

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(self.drop(x), self.A), self.B) * self.scale


class LoRMALinear(nn.Module):
    """LoRMA+ : h = (I + (alpha/r) B A) W0 x, evaluated as y + s B(A y), y = W0 x.

    Multiplicative instead of additive. The identity term is the rank inflation
    (rank(I + BA) >= d - r), and computing W0 x first keeps the cost O(dkr)
    like LoRA. A maps from the OUTPUT dimension. B = 0 so step 0 is the
    pretrained function. Same implementation as Phase 2 (arXiv 2506.07621 eq. 8).
    """

    def __init__(self, base, r, alpha=16.0):
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


# ----------------------------------------------------------------- QLoRA -----
NF4 = torch.tensor([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
    0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
    0.7229568362236023, 1.0])


class NF4Linear(nn.Module):
    """Frozen weight held as 4-bit NormalFloat codes, two per byte.

    Block-wise absmax (64 weights per block), and the absmax constants are
    themselves quantised to int8 in blocks of 256 around their mean -- QLoRA's
    "double quantisation" -- so the overhead is ~0.127 bits/weight rather than
    0.5. Dequantised on the fly into the compute dtype for each matmul. Storage
    is genuinely 4-bit; compute is bf16, as in the paper.
    """

    BLOCK, DQ_BLOCK = 64, 256

    def __init__(self, base: nn.Linear):
        super().__init__()
        w = base.weight.detach().float().cpu()
        self.out_features, self.in_features = w.shape
        flat = w.flatten()
        n = flat.numel()
        pad = (-n) % self.BLOCK
        if pad:
            flat = torch.cat([flat, flat.new_zeros(pad)])
        blocks = flat.view(-1, self.BLOCK)
        absmax = blocks.abs().amax(1).clamp(min=1e-12)
        normed = blocks / absmax[:, None]
        mid = (NF4[1:] + NF4[:-1]) / 2
        codes = torch.bucketize(normed, mid).to(torch.uint8).flatten()
        packed = (codes[0::2] << 4) | codes[1::2]
        # double quantisation of absmax
        mean = absmax.mean()
        c = absmax - mean
        dpad = (-c.numel()) % self.DQ_BLOCK
        cpad = torch.cat([c, c.new_zeros(dpad)]).view(-1, self.DQ_BLOCK)
        dscale = cpad.abs().amax(1).clamp(min=1e-12) / 127
        q8 = torch.round(cpad / dscale[:, None]).clamp(-127, 127).to(torch.int8)
        self.register_buffer("packed", packed)
        self.register_buffer("q8", q8)
        self.register_buffer("dscale", dscale)
        self.register_buffer("mean", mean.reshape(1))
        self.register_buffer("levels", NF4.clone())
        self.n, self.nblk = n, blocks.size(0)
        deq = self.dequant(torch.float32)
        self.rel_err = float((deq - w).norm() / w.norm())

    def dequant(self, dtype):
        hi = (self.packed >> 4).long()
        lo = (self.packed & 0x0F).long()
        codes = torch.stack([hi, lo], 1).flatten()
        absmax = (self.q8.float() * self.dscale[:, None]).flatten()[:self.nblk] + self.mean
        w = self.levels[codes].view(-1, self.BLOCK) * absmax[:, None]
        return w.flatten()[:self.n].view(self.out_features, self.in_features).to(dtype)

    def forward(self, x):
        return F.linear(x, self.dequant(x.dtype))


# --------------------------------------------------------------- AdaLoRA -----
class SVDLinear(nn.Module):
    """y = W x + (alpha/r) P diag(E) Q x. E starts at 0; pruning zeroes entries of E."""

    def __init__(self, base, r, alpha=16.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.P = nn.Parameter(torch.randn(base.out_features, r) * 0.02)
        self.E = nn.Parameter(torch.zeros(r))
        self.Q = nn.Parameter(torch.randn(r, base.in_features) * 0.02)
        self.scale = alpha / r
        self.register_buffer("keep", torch.ones(r))

    def forward(self, x):
        e = self.E * self.keep
        return self.base(x) + F.linear(F.linear(x, self.Q) * e, self.P) * self.scale


class AdaLoRAController:
    """Sensitivity-smoothed importance and a cubic rank-budget schedule.

    Importance of triplet i in a layer (paper eq. 6-9):
        I(w)  = |w * dL/dw|,   Ibar <- b1 Ibar + (1-b1) I,
        U     = |I - Ibar|,    Ubar <- b2 Ubar + (1-b2) U,   s(w) = Ibar * Ubar
        S_i   = s(E_i) + mean_j s(P_ji) + mean_j s(Q_ij)
    The total budget falls cubically from r_init * n_layers to r_target * n_layers
    between t_init and T - t_final; every `every` steps the lowest-scoring
    triplets across ALL layers are masked, so layers that matter keep more rank.
    Orthogonality of P and Q is encouraged with the paper's regulariser.
    """

    def __init__(self, mods, r_init, r_target, total, t_init_frac=0.1,
                 t_final_frac=0.3, every=10, b1=0.85, b2=0.85, orth=0.1):
        self.mods = mods
        self.r_init, self.r_target = r_init, r_target
        self.total = total
        self.t_init = int(t_init_frac * total)
        self.t_final = int(t_final_frac * total)
        self.every, self.b1, self.b2, self.orth = every, b1, b2, orth
        self.state = {}
        self.budget_log = []

    def orth_loss(self):
        reg = 0.0
        for m in self.mods:
            r = m.E.numel()
            eye = torch.eye(r, device=m.P.device)
            reg = reg + ((m.P.T @ m.P - eye) ** 2).sum() + ((m.Q @ m.Q.T - eye) ** 2).sum()
        return self.orth * reg / max(1, len(self.mods))

    def budget(self, step):
        b0 = self.r_init * len(self.mods)
        bt = self.r_target * len(self.mods)
        if step <= self.t_init:
            return b0
        if step >= self.total - self.t_final:
            return bt
        frac = (step - self.t_init) / max(1, self.total - self.t_final - self.t_init)
        return int(bt + (b0 - bt) * (1 - frac) ** 3)

    @torch.no_grad()
    def step(self, step):
        """Call after loss.backward(), before zero_grad."""
        scores = []
        for k, m in enumerate(self.mods):
            for name in ("P", "E", "Q"):
                p = getattr(m, name)
                if p.grad is None:
                    continue
                ipt = (p * p.grad).abs()
                key = (k, name)
                if key not in self.state:
                    self.state[key] = [torch.zeros_like(ipt), torch.zeros_like(ipt)]
                ib, ub = self.state[key]
                ib.mul_(self.b1).add_(ipt, alpha=1 - self.b1)
                ub.mul_(self.b2).add_((ipt - ib).abs(), alpha=1 - self.b2)
            if (k, "E") in self.state:
                s = lambda n: self.state[(k, n)][0] * self.state[(k, n)][1]
                scores.append(s("E") + s("P").mean(0) + s("Q").mean(1))
            else:
                scores.append(None)
        if step <= self.t_init or step % self.every or any(s is None for s in scores):
            return
        b = self.budget(step)
        allsc = torch.cat(scores)
        if b >= allsc.numel():
            return
        thr = torch.kthvalue(allsc.float(), allsc.numel() - b).values
        for m, s in zip(self.mods, scores):
            m.keep.copy_((s > thr).float())
        self.budget_log.append((step, b))

    def ranks(self):
        return [int(m.keep.sum()) for m in self.mods]


# ------------------------------------------------------------------- IA3 -----
class ScaleOut(nn.Module):
    """IA3 on keys/values: y = l * (W x)."""

    def __init__(self, base):
        super().__init__()
        self.base = base
        self.l = nn.Parameter(torch.ones(base.out_features))

    def forward(self, x):
        return self.base(x) * self.l


class ScaleIn(nn.Module):
    """IA3 on the FFN: rescale the intermediate activation before W_down."""

    def __init__(self, base):
        super().__init__()
        self.base = base
        self.l = nn.Parameter(torch.ones(base.in_features))

    def forward(self, x):
        return self.base(x * self.l)


# --------------------------------------------------------------- Adapters ----
class AdapterLinear(nn.Module):
    """Houlsby adapter on a sublayer's output projection: y = W x;  y + U(gelu(D y)).

    Wrapping the attention output projection (wo) and the FFN down projection
    puts one adapter after each sublayer, before its residual add -- the
    placement in Houlsby et al. Up-projection starts at zero, so step 0 is the
    pretrained function.
    """

    def __init__(self, base, bottleneck=64):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        d = base.out_features
        self.down = nn.Linear(d, bottleneck)
        self.up = nn.Linear(bottleneck, d)
        nn.init.normal_(self.down.weight, std=1 / math.sqrt(d))
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features

    def forward(self, x):
        y = self.base(x)
        return y + self.up(F.gelu(self.down(y)))


# ------------------------------------------------------------ prefix/prompt --
class PrefixEncoder(nn.Module):
    """P learned states for every layer, through Li & Liang's MLP reparametrisation.

    The states are written into the first P positions of each layer's INPUT, so
    every real token attends to keys/values computed from them. With
    [W_k; W_v] full row rank, any key/value pair is reachable from some state,
    so this is prefix-tuning, not shallow prompt-tuning; it differs from the
    original only in that RoPE and qk-norm also apply to the prefix keys.
    """

    def __init__(self, n_layer, n_prefix, d, hidden=512):
        super().__init__()
        self.n_layer, self.n_prefix, self.d = n_layer, n_prefix, d
        self.emb = nn.Parameter(torch.randn(n_prefix, d) * 0.02)
        self.mlp = nn.Sequential(nn.Linear(d, hidden), nn.Tanh(),
                                 nn.Linear(hidden, n_layer * d))
        nn.init.normal_(self.mlp[2].weight, std=0.02)
        nn.init.zeros_(self.mlp[2].bias)

    def forward(self):
        return self.mlp(self.emb).view(self.n_prefix, self.n_layer, self.d).transpose(0, 1)


# ------------------------------------------------------------------ apply -----
def blocks_of(trunk):
    """Decoder blocks, or for the encoder-decoder BOTH stacks: Phase 6 generates,
    so the decoder (deleted in Phases 3-5) is where most of the adaptation goes."""
    if getattr(trunk, "blocks", None) is not None:
        return list(trunk.blocks)
    return list(getattr(trunk, "enc", [])) + list(getattr(trunk, "dec", []))


def slots(block):
    """(parent, attr, role) for every linear layer in one block."""
    out = []
    att = block.attn
    for name, role in (("wq", "q"), ("wk", "k"), ("wv", "v"), ("wo", "o"),
                       ("kv_down", "kvd"), ("k_up", "k"), ("v_up", "v")):
        if isinstance(getattr(att, name, None), nn.Module):
            out.append((att, name, role))
    if hasattr(block, "w_gate"):
        out += [(block, "w_gate", "gate"), (block, "w_up", "up"), (block, "w_down", "down")]
    elif hasattr(block, "ffn"):
        out += [(block.ffn, "g", "gate"), (block.ffn, "u", "up"), (block.ffn, "d", "down")]
    return out


def _get(parent, attr):
    return parent[attr] if isinstance(parent, nn.ModuleDict) else getattr(parent, attr)


def _set(parent, attr, mod):
    if isinstance(parent, nn.ModuleDict):
        parent[attr] = mod
    else:
        setattr(parent, attr, mod)


TARGETS = {
    "qv": {"q", "v"},
    "attn": {"q", "k", "v", "o", "kvd"},
    "all": {"q", "k", "v", "o", "kvd", "gate", "up", "down"},
}


def apply_method(model, a):
    """Freeze the trunk, insert adapters, and return (n_trainable, controller, info)."""
    trunk = model.trunk
    for p in trunk.parameters():
        p.requires_grad_(False)
    m = a.method
    ctrl, info = None, {}
    blocks = blocks_of(trunk)
    tg = TARGETS[a.targets]

    if m == "full":
        for p in trunk.parameters():
            p.requires_grad_(True)
    elif m in ("head", "prompt", "prefix"):
        pass
    elif m == "norm":
        for n, p in trunk.named_parameters():
            if ".n1." in n or ".n2." in n or ".n3." in n or n.startswith("norm") \
                    or n.startswith("enc_norm") or "kv_norm" in n:
                p.requires_grad_(True)
    elif m in ("lora", "rslora"):
        for b in blocks:
            for par, at, role in slots(b):
                if role in tg:
                    _set(par, at, LoRALinear(_get(par, at), a.rank, a.alpha,
                                             rs=(m == "rslora"), dropout=a.lora_dropout))
    elif m == "qlora":
        errs = []
        for b in blocks:
            for par, at, role in slots(b):
                q = NF4Linear(_get(par, at))
                errs.append(q.rel_err)
                _set(par, at, LoRALinear(q, a.rank, a.alpha, dropout=a.lora_dropout)
                     if role in TARGETS["all"] else q)
        info["nf4_rel_err_mean"] = sum(errs) / len(errs)
    elif m == "adalora":
        mods = []
        for b in blocks:
            for par, at, role in slots(b):
                if role in tg:
                    s = SVDLinear(_get(par, at), a.rank, a.alpha)
                    _set(par, at, s)
                    mods.append(s)
        ctrl = AdaLoRAController(mods, a.rank, a.ada_target, total=1)  # total set by trainer
    elif m == "lorma":
        for b in blocks:
            for par, at, role in slots(b):
                if role in TARGETS["qv"]:
                    _set(par, at, LoRMALinear(_get(par, at), a.rank, a.alpha))
    elif m in ("adapter", "houlsby", "pfeiffer"):
        # Houlsby: after attention AND FFN.  Pfeiffer: after the FFN only
        # (Pfeiffer et al. 2020's AdapterFusion configuration) -- half the
        # adapters, and the comparison the brief's technique list asks for.
        where = ("down",) if m == "pfeiffer" else ("o", "down")
        for b in blocks:
            for par, at, role in slots(b):
                if role in where:
                    _set(par, at, AdapterLinear(_get(par, at), a.bottleneck))
        info["bottleneck"] = a.bottleneck
        info["adapter_at"] = list(where)
    elif m == "partial":
        k = min(a.train_top, len(blocks))
        for b in blocks[len(blocks) - k:]:
            for p in b.parameters():
                p.requires_grad_(True)
        for n, p in trunk.named_parameters():
            if n.startswith(("norm", "enc_norm")):
                p.requires_grad_(True)
        info["train_top"] = k
    elif m == "ia3":
        for b in blocks:
            for par, at, role in slots(b):
                if role in ("k", "v"):
                    _set(par, at, ScaleOut(_get(par, at)))
                elif role == "down":
                    _set(par, at, ScaleIn(_get(par, at)))
    else:
        raise ValueError(m)

    n_trunk = sum(p.numel() for p in trunk.parameters() if p.requires_grad)
    n_virt = sum(p.numel() for n, p in model.named_parameters()
                 if p.requires_grad and n.startswith(("prompt", "prefix")))
    n_head = sum(p.numel() for n, p in model.named_parameters()
                 if p.requires_grad and not n.startswith(("trunk.", "prompt", "prefix")))
    info.update(trainable_trunk=n_trunk + n_virt, trainable_head=n_head)
    return n_trunk + n_virt + n_head, ctrl, info


def trunk_param_ids(model):
    """ids of trainable parameters that belong to the trunk/adapters, for the LR split."""
    ids = {id(p) for p in model.trunk.parameters() if p.requires_grad}
    for name in ("prompt", "prefix"):
        mod = getattr(model, name, None)
        if mod is not None:
            ids |= {id(p) for p in (mod.parameters() if isinstance(mod, nn.Module) else [mod])}
    return ids
