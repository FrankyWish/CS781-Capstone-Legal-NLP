#!/usr/bin/env python3
"""CS-781 Phase 6 -- abstractive summarisation on the four Phase-1 models.

    python phase6_sum.py tok   --view sel1024              # CPU: tokenise views + summaries once
    python phase6_sum.py train --tag m4_lorma --model M4_LexArcP --method lorma
    python phase6_sum.py train --tag kd_m1half --model M1_SC --method full --keep-every 2 --teacher m4_lorma

Layout.  Every prompt is EXACTLY --doc-tokens long, so the summary starts at the
same position in every row and no attention mask is needed in training:

    decoders (M1 GQA, M3 MLA, M4 pointer):
        [eos .. eos][selected document text][\\n\\nHEADNOTE:\\n] summary eos [eos pad]
         left pad    head 70% + tail 30% if too long             ^ loss from here
    encoder-decoder (M2):   encoder <- [eos .. eos][document]   decoder <- bos summary eos

The left pad is <|endoftext|>, the token the trunks saw between packed documents
in pretraining, and generation uses the identical prompt with pad_mask all True,
so train and decode see the same thing.  No new special token: pre-1994 Supreme
Court judgments carry the literal word HEADNOTE, so the trunk has seen it.

Context.  Prompt + summary (~2k tokens) is past the 1024-1280 tokens the trunks
were pretrained on; RoPE base is raised NTK-style (theta * s^(d/(d-2))) by the
overshoot factor s, which is one config field and so applies identically in
training and in the models' own generate().

Generation bans every special token and runs to ~1.6x the per-document length
target; the raw text is saved, and trimming to a length happens at scoring time.
Greedy decoding is prefix-consistent, so one long generation yields every
shorter length for free -- stage 3 recalibrates length without re-generating.

Methods (phase6_peft): lorma lora rslora qlora adalora houlsby pfeiffer ia3 norm
partial full.  Prompt/prefix tuning are not offered here: they change the
sequence, and the Phase-1 generate() takes token ids, not embeddings.  M4's copy
head (cq/ck/cv/gate, 0.15M params) is trained under every method: it is the
task-specific output layer, the analogue of a classification head.

Knowledge distillation (--teacher <tag>): the teacher is rebuilt from its saved
args + weights and frozen; the student minimises
    (1 - a) * CE(gold) + a * T^2 * KL(p_teacher^T || p_student^T)
over the summary positions.  For an M4 teacher p_teacher is the full
pointer-generator mixture, not just its vocabulary head.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import phase6_data as D                                   # noqa: E402
from phase6_peft import apply_method                      # noqa: E402

P1, P6 = D.P1, D.P6
LIB = os.environ.get("CS781_LIB", f"{P1}/shared/lib")
sys.path.insert(0, LIB)
TOKJSON = os.environ.get("CS781_TOKENIZER", f"{P1}/shared/tokenizer/bpe_legal_32k.json")
CACHE = f"{P6}/cache"
CKPT = f"{P6}/ckpt"
SEP = "\n\nHEADNOTE:\n"
LOGF = None

LR = dict(full=1e-4, partial=1e-4, lora=5e-4, rslora=3e-4, lorma=5e-4, qlora=5e-4,
          adalora=5e-4, houlsby=1e-3, adapter=1e-3, pfeiffer=1e-3, ia3=3e-3, norm=1e-3)
DECODE = {
    "g": dict(greedy=True, no_repeat_ngram=4, repetition_penalty=1.15),
    "s": dict(greedy=False, temperature=0.7, top_p=0.9, no_repeat_ngram=4,
              repetition_penalty=1.1),
}


def log(*a):
    s = " ".join(str(x) for x in a)
    print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)
    if LOGF:
        print(f"[{time.strftime('%H:%M:%S')}] {s}", file=LOGF, flush=True)


# ---------------------------------------------------------------- tokenizer --
_TOK = {}


def tokenizer():
    if "t" not in _TOK:
        from cs781_bpe import BPETokenizer
        _TOK["t"] = BPETokenizer.load(TOKJSON)
    return _TOK["t"]


def specials():
    t = tokenizer()
    sp = dict(t.specials)
    return sp["<|endoftext|>"], sorted(set(sp.values()))


def _enc_init(path):
    from cs781_bpe import BPETokenizer
    _TOK["t"] = BPETokenizer.load(path)


def _enc(text):
    return np.asarray(_TOK["t"].encode(text), dtype=np.uint16)


def view_texts(view, split):
    recs = D.load(split)
    if view == "trunc":                       # raw document; head+tail kept at batch time
        return [r["id"] for r in recs], [" ".join(r["doc"].split()[:2500]) + " " +
                                          " ".join(r["doc"].split()[-700:]) for r in recs]
    with open(f"{P6}/views/{view}_{split}.json", encoding="utf-8") as f:
        v = json.load(f)
    return [r["id"] for r in recs], [v[r["id"]] for r in recs]


def cache_path(view, split):
    return f"{CACHE}/tok_{view}_{split}.npz"


def cmd_tok(a):
    from multiprocessing import Pool
    os.makedirs(CACHE, exist_ok=True)
    for split in a.splits.split(","):
        out = cache_path(a.view, split)
        if os.path.exists(out):
            log(f"{out} exists")
            continue
        ids, docs = view_texts(a.view, split)
        summs = [r["summ"] or "" for r in D.load(split)]
        t0 = time.time()
        if a.procs > 1:
            with Pool(a.procs, initializer=_enc_init, initargs=(TOKJSON,)) as pool:
                dt = pool.map(_enc, docs, chunksize=16)
                st = pool.map(_enc, summs, chunksize=16) if split != "test" else []
        else:
            _enc_init(TOKJSON)
            dt = [_enc(x) for x in docs]
            st = [_enc(x) for x in summs] if split != "test" else []
        pack = lambda xs: (np.concatenate(xs) if xs else np.zeros(0, np.uint16),
                           np.concatenate([[0], np.cumsum([len(x) for x in xs])]).astype(np.int64))
        df, do = pack(dt)
        sf, so = pack(st)
        meta = {}
        if st:
            sw = np.array([max(D.nwords(s), 1) for s in summs])
            sl = np.array([len(x) for x in st])
            meta = dict(tok_per_word=float(np.median(sl / sw)), summ_tok=D.pct(sl))
        tmp = out + ".tmp.npz"
        np.savez(tmp, doc=df, doc_off=do, summ=sf, summ_off=so, ids=np.array(ids),
                 meta=json.dumps(meta))
        os.replace(tmp, out)
        log(f"{split}/{a.view}: {len(ids)} docs, doc tokens {D.pct([len(x) for x in dt])}, "
            f"{meta}  ({time.time() - t0:.0f}s)")


class Data:
    def __init__(self, view, split):
        z = np.load(cache_path(view, split), allow_pickle=False)
        self.ids = [str(x) for x in z["ids"]]
        self.df, self.do = z["doc"], z["doc_off"]
        self.sf, self.so = z["summ"], z["summ_off"]
        self.meta = json.loads(str(z["meta"]))
        self.pos = {k: i for i, k in enumerate(self.ids)}

    def __len__(self):
        return len(self.ids)

    def doc(self, i):
        return self.df[self.do[i]:self.do[i + 1]].astype(np.int64).tolist()

    def summ(self, i):
        return self.sf[self.so[i]:self.so[i + 1]].astype(np.int64).tolist() if len(self.so) > 1 else []


# -------------------------------------------------------------------- model --
def _cfg(cls, d):
    names = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in names})


def load_lm(name, ckpt="best.pt", keep_every=1, path=None):
    from cs781_model import Config, LexArcD
    ck = torch.load(path or f"{P1}/{name}/ckpt/{ckpt}", map_location="cpu", weights_only=False)
    cd = ck["cfg"] if isinstance(ck["cfg"], dict) else dataclasses.asdict(ck["cfg"])
    if "n_enc_layer" in cd:
        from cs781_model_ed import EDConfig, LexArcED
        cfg = _cfg(EDConfig, cd)
        m, kind = LexArcED(cfg), "ed"
    else:
        cfg = _cfg(Config, cd)
        if cfg.use_pointer:
            from cs781_model_ptr import LexArcP
            m, kind = LexArcP(cfg), "ptr"
        else:
            m, kind = LexArcD(cfg), "dec"
    miss, unexp = m.load_state_dict(ck["model"], strict=False)
    miss = [k for k in miss if not k.endswith("head.weight")]       # tied to emb
    if miss:
        log(f"  !! {name}: {len(miss)} weights missing from the checkpoint, e.g. {miss[:3]}")
    del ck
    if keep_every > 1:            # half-depth student; layer 0 stays (value residual v0)
        if kind == "ed":
            m.enc = nn.ModuleList(list(m.enc)[::keep_every])
            m.dec = nn.ModuleList(list(m.dec)[::keep_every])
            cfg.n_enc_layer, cfg.n_dec_layer = len(m.enc), len(m.dec)
        else:
            m.blocks = nn.ModuleList(list(m.blocks)[::keep_every])
            cfg.n_layer = len(m.blocks)
    return m, kind, cfg


def trained_len(kind, cfg):
    if kind == "ed":
        return max(cfg.split_points), cfg.window - min(cfg.split_points)
    return cfg.ctx, None


def ntk(cfg, factor):
    """Raise the RoPE base so `factor` x the pretrained length maps onto the
    trained frequency range (NTK-aware scaling).  Returns the new theta."""
    if factor > 1.0:
        hd = cfg.head_dim
        cfg.rope_theta = float(cfg.rope_theta * factor ** (hd / (hd - 2)))
    return cfg.rope_theta


class Wrap(nn.Module):
    """apply_method() expects `.trunk`; for generation the trunk IS the model."""

    def __init__(self, lm):
        super().__init__()
        self.trunk = lm


def build(a, device):
    lm, kind, cfg = load_lm(a.model, a.ckpt, a.keep_every)
    enc_len, dec_len = trained_len(kind, cfg)
    if kind == "ed":
        factor = max(a.doc_tokens / enc_len, (a.sum_tokens + 1) / dec_len, 1.0)
    else:
        factor = max((a.doc_tokens + a.sum_tokens + 1) / enc_len, 1.0)
    theta = ntk(cfg, factor) if not a.no_ntk else cfg.rope_theta
    w = Wrap(lm)
    a.targets = getattr(a, "targets", "attn")
    n_train, ctrl, info = apply_method(w, a)
    if kind == "ptr":                         # the copy head is the task head
        for n in ("cq", "ck", "cv", "gate"):
            for p in getattr(lm, n).parameters():
                p.requires_grad_(True)
    if device.type == "cuda":                 # frozen weights in bf16, trainable in fp32
        for p in lm.parameters():
            if not p.requires_grad and p.is_floating_point():
                p.data = p.data.to(torch.bfloat16)
    lm.to(device)
    n_train = sum(p.numel() for p in lm.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in lm.parameters())
    n_all += sum(getattr(m, "n", 0) for m in lm.modules() if type(m).__name__ == "NF4Linear")
    info.update(kind=kind, rope_factor=round(factor, 3), rope_theta=round(theta, 1),
                n_trainable=n_train, n_params=n_all,
                n_layers=len(getattr(lm, "blocks", [])) or (len(lm.enc) + len(lm.dec)))
    return lm, kind, cfg, ctrl, info


# ------------------------------------------------------------------ batches --
def fit_doc(d, n):
    """Head 70% + tail 30% when too long: the holding sits at the end."""
    if len(d) <= n:
        return d
    h = int(n * 0.7)
    return d[:h] + d[len(d) - (n - h):]


def batch(kind, docs, summs, a, eos, sep, bos=None):
    """-> dict of LongTensors (CPU).  Every prompt is exactly a.doc_tokens long."""
    Dt, S = a.doc_tokens, a.sum_tokens
    rows, tg, mk = [], [], []
    for d, s in zip(docs, summs):
        s = (s[:S] + [eos]) if s is not None else [eos]
        if kind == "ed":
            d = fit_doc(d, Dt)
            rows.append([eos] * (Dt - len(d)) + d)
        else:
            d = fit_doc(d, Dt - len(sep))
            rows.append([eos] * (Dt - len(sep) - len(d)) + d + sep + s
                        + [eos] * (S + 1 - len(s)))
        tg.append(s + [eos] * (S + 1 - len(s)))
        mk.append([1] * len(s) + [0] * (S + 1 - len(s)))
    out = dict(idx=torch.tensor(rows), tgt=torch.tensor(tg), mask=torch.tensor(mk, dtype=torch.float))
    if kind == "ed":
        out["dec_in"] = torch.cat([torch.full((len(rows), 1), eos if bos is None else bos),
                                   out["tgt"][:, :-1]], 1)
    return out


def forward(lm, kind, cfg, b, Dt, want_dist=False):
    """-> (nll (B,Q) over the summary positions, log-dist (B,Q,V) or None)."""
    from cs781_model import build_rope
    if kind == "ed":
        enc_out = lm.encode(b["idx"])
        kv = [blk.cross.kv(enc_out) for blk in lm.dec]
        cos, sin = build_rope(b["dec_in"].size(1), cfg.head_dim, b["idx"].device,
                              lm.emb.weight.dtype, cfg.rope_theta)
        x = lm.emb(b["dec_in"])
        for blk, k in zip(lm.dec, kv):
            x, _ = blk(x, cos, sin, k)
        logits = lm.head(lm.dec_norm(x))
        nll = F.cross_entropy(logits.flatten(0, 1), b["tgt"].flatten(), reduction="none").view_as(b["tgt"]).float()
        return nll, (F.log_softmax(logits.float(), -1) if want_dist else None)
    idx = b["idx"]
    R = cfg.n_register
    cos, sin = build_rope(idx.size(1) + R, cfg.head_dim, idx.device, lm.emb.weight.dtype,
                          cfg.rope_theta)
    h, _ = lm._trunk(idx, cos, sin)
    h = h[:, R:]
    hs = h[:, Dt - 1:-1]
    tgt = idx[:, Dt:]
    logits = lm.head(hs)
    lp_tgt = -F.cross_entropy(logits.flatten(0, 1), tgt.flatten(), reduction="none").view_as(tgt).float()
    if kind == "dec":
        return -lp_tgt, (F.log_softmax(logits.float(), -1) if want_dist else None)
    log_copy, log_g, log_1mg, attn = lm._copy_terms(hs, h, idx, tgt, Dt - 1)
    nll = -torch.logaddexp(log_g + lp_tgt, log_1mg + log_copy)
    dist = None
    if want_dist:
        B, Q, T = attn.shape
        cp = torch.zeros(B, Q, cfg.vocab_size, device=idx.device)
        cp.scatter_add_(2, idx[:, None, :].expand(B, Q, T), attn.float())
        g = log_g.exp()[..., None]
        dist = torch.log(g * F.softmax(logits.float(), -1) + (1 - g) * cp + 1e-9)
    return nll, dist


# --------------------------------------------------------------- generation --
class BanHead(nn.Module):
    """Wraps the LM head at decode time: special tokens can never be emitted, so
    every summary runs to its token budget and length is set by trimming."""

    def __init__(self, head, ban):
        super().__init__()
        self.h, self.ban = head, ban
        self.weight = head.weight

    def forward(self, x):
        y = self.h(x)
        y[..., self.ban] = -1e4
        return y


def strip_specials(text):
    for s in ("<|endoftext|>", "<|pad|>", "<|reg|>", "<|mask|>"):
        text = text.replace(s, " ")
    return " ".join(text.split())


@torch.no_grad()
def generate(lm, kind, cfg, data, ids, tok_budget, a, eos, sep, ban, device, dec_cfg, seed=0):
    """{id: raw untrimmed text}.  Rows sorted by budget so a batch wastes little."""
    lm.eval()
    tok = tokenizer()
    old = lm.head
    lm.head = BanHead(old, ban)
    order = sorted(ids, key=lambda k: tok_budget[k])
    out = {}
    torch.manual_seed(seed)
    try:
        for i in range(0, len(order), a.gen_batch):
            chunk = order[i:i + a.gen_batch]
            docs = [data.doc(data.pos[k]) for k in chunk]
            b = batch(kind, docs, [None] * len(chunk), a, eos, sep)
            Dt = a.doc_tokens
            prompt = b["idx"][:, :Dt].to(device)
            mask = torch.ones_like(prompt, dtype=torch.bool)
            n_new = int(max(tok_budget[k] for k in chunk))
            ctx = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" \
                else torch.autocast("cpu", enabled=False)
            with ctx:
                g = lm.generate(prompt, mask, n_new, eos_id=None, **dec_cfg)
            for k, row in zip(chunk, g.tolist()):
                out[k] = strip_specials(tok.decode(row))
    finally:
        lm.head = old
    return out


def budgets(ids, words, tok_per_word, a):
    """Token budget per document: the length target x headroom for later trims."""
    cap = int(a.sum_tokens * 1.3)
    return {k: int(min(cap, max(48, words[k] * tok_per_word * a.gen_headroom))) for k in ids}


def length_setting():
    f = f"{P6}/results/length_calib.json"
    if os.path.exists(f):
        with open(f) as fh:
            c = json.load(fh)["chose"]
        return c["len"], float(c["mult"])
    return "model", 1.0


def trim(text, n):
    w = text.split()
    return " ".join(w[:n]) if len(w) > n else text


# --------------------------------------------------------------------- train --
def cmd_train(a):
    global LOGF
    import phase6_sel as S
    os.makedirs(f"{P6}/results/{a.tag}", exist_ok=True)
    os.makedirs(CKPT, exist_ok=True)
    LOGF = open(f"{P6}/results/{a.tag}/train.log", "a")
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and "T4" in torch.cuda.get_device_name(0):
        raise SystemExit("T4: bf16 is emulated on SM75 at ~30x slowdown. Use an L4.")
    eos, ban = specials()
    sep = tokenizer().encode(SEP)

    teacher = None
    if a.teacher:                              # KD: student inherits the teacher's layout
        with open(f"{P6}/results/{a.teacher}/meta.json") as f:
            tm = json.load(f)
        ta = argparse.Namespace(**tm["args"])
        a.view, a.doc_tokens, a.sum_tokens = ta.view, ta.doc_tokens, ta.sum_tokens
        t_lm, t_kind, t_cfg, _, t_info = build(ta, device)
        sd = torch.load(f"{CKPT}/{a.teacher}_final.pt", map_location=device, weights_only=False)
        t_lm.load_state_dict(sd, strict=False)
        t_lm.eval()
        for p in t_lm.parameters():
            p.requires_grad_(False)
        teacher = (t_lm, t_kind, t_cfg, ta)
        log(f"teacher {a.teacher}: {t_info['n_params']/1e6:.1f}M params ({t_kind})")

    a.method = {"adapter": "houlsby"}.get(a.method, a.method)
    lm, kind, cfg, ctrl, info = build(a, device)
    n_tok = len(tokenizer().itos)                 # ids the tokenizer cannot decode are banned too
    ban = sorted(set(ban) | set(range(n_tok, cfg.vocab_size)))
    log(f"{a.tag}: {a.model} {kind} {a.method} | {info['n_trainable']/1e6:.3f}M trainable of "
        f"{info['n_params']/1e6:.1f}M | {info['n_layers']} layers | rope x{info['rope_factor']}")

    fit, dev, test = Data(a.view, "fit"), Data(a.view, "dev"), Data(a.view, "test")
    if a.limit:
        fit.ids = fit.ids[:a.limit]
    tpw = fit.meta.get("tok_per_word", 1.35)
    bos = getattr(cfg, "bos_id", eos)
    params = [p for p in lm.parameters() if p.requires_grad]
    lr = a.lr or LR.get(a.method, 5e-4)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=a.wd, betas=(0.9, 0.95))

    eff = a.batch * a.accum
    steps_epoch = max(1, len(fit) // eff)
    total = max(1, int(a.epochs * steps_epoch))
    step, t_train, losses = 0, 0.0, []
    resume = f"{CKPT}/{a.tag}_resume.pt"
    if os.path.exists(resume):
        r = torch.load(resume, map_location=device, weights_only=False)
        lm.load_state_dict(r["model"], strict=False)
        if r.get("opt"):
            opt.load_state_dict(r["opt"])
        step, total, t_train = r["step"], r["total"], r["t_train"]
        log(f"  resumed at step {step}/{total} ({t_train/60:.1f} train-min done)")
    if ctrl is not None:
        ctrl.total = total
        ctrl.t_init, ctrl.t_final = int(0.1 * total), int(0.3 * total)

    rng = np.random.default_rng(a.seed)
    perm = np.concatenate([rng.permutation(len(fit)) for _ in range(int(math.ceil(a.epochs)) + 1)])
    ac = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if device.type == "cuda" \
        else (lambda: torch.autocast("cpu", enabled=False))
    budget_s = a.budget_min * 60
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    last_save, probed = time.time(), step > 0
    lm.train()
    while step < total:
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        tot_loss = 0.0
        for k in range(a.accum):
            j = perm[(step * eff + k * a.batch) % len(perm):][:a.batch]
            j = [int(x) % len(fit) for x in j]
            docs, summs = [fit.doc(i) for i in j], [fit.summ(i) for i in j]
            b = {kk: v.to(device) for kk, v in batch(kind, docs, summs, a, eos, sep, bos).items()}
            with ac():
                nll, sd = forward(lm, kind, cfg, b, a.doc_tokens, want_dist=teacher is not None)
                m = b["mask"]
                loss = (nll * m).sum() / m.sum()
                if teacher is not None:
                    t_lm, t_kind, t_cfg, ta = teacher
                    tb = {kk: v.to(device) for kk, v in batch(t_kind, docs, summs, ta, eos, sep,
                                                              getattr(t_cfg, "bos_id", eos)).items()}
                    with torch.no_grad():
                        _, td = forward(t_lm, t_kind, t_cfg, tb, ta.doc_tokens, want_dist=True)
                    T = a.kd_temp
                    kl = (F.softmax(td / T, -1) * (F.log_softmax(td / T, -1)
                                                   - F.log_softmax(sd / T, -1))).sum(-1)
                    loss = (1 - a.kd_alpha) * loss + a.kd_alpha * T * T * (kl * m).sum() / m.sum()
                if ctrl is not None:
                    loss = loss + ctrl.orth_loss()
            (loss / a.accum).backward()
            tot_loss += float(loss.detach()) / a.accum
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        if ctrl is not None:
            ctrl.step(step)
        warm = max(1, int(a.warmup * total))
        f = step / warm if step < warm else 0.1 + 0.9 * 0.5 * (1 + math.cos(
            math.pi * min(1.0, (step - warm) / max(1, total - warm))))
        for g in opt.param_groups:
            g["lr"] = lr * f
        opt.step()
        step += 1
        t_train += time.time() - t0
        losses.append(tot_loss)
        if not probed and step == a.probe:     # fix the schedule to the time budget
            per = t_train / step
            total = max(step + 1, min(total, int(budget_s / per)))
            if ctrl is not None:
                ctrl.total = total
                ctrl.t_init, ctrl.t_final = int(0.1 * total), int(0.3 * total)
            probed = True
            log(f"  probe: {per:.2f}s/step, {eff * (a.doc_tokens + a.sum_tokens) / per:.0f} tok/s "
                f"-> {total} steps ({total * eff / len(fit):.2f} epochs) in {a.budget_min:.0f} min")
        if step % 25 == 0 or step == total:
            log(f"  step {step}/{total} loss {np.mean(losses[-25:]):.4f} lr {lr * f:.2e} "
                f"{t_train / 60:.1f} min")
        if t_train > budget_s * 1.05:
            log("  time budget reached")
            break
        if time.time() - last_save > a.save_every_min * 60 and not a.no_ckpt:
            sd_ = {k: v for k, v in lm.state_dict().items() if k in _trainable_keys(lm)}
            small = info["n_trainable"] < 20e6
            torch.save(dict(model=sd_, opt=opt.state_dict() if small else None, step=step,
                            total=total, t_train=t_train), resume + ".tmp")
            os.replace(resume + ".tmp", resume)
            last_save = time.time()
    peak = torch.cuda.max_memory_allocated() / 2 ** 30 if device.type == "cuda" else 0.0
    keys = _trainable_keys(lm)
    torch.save({k: v for k, v in lm.state_dict().items() if k in keys}, f"{CKPT}/{a.tag}_final.pt")
    if teacher is not None:
        del teacher
        torch.cuda.empty_cache() if device.type == "cuda" else None

    # ---------------------------------------------------------- decode + score
    how, mult = length_setting()
    gen_dev = D.gen_dev_ids(a.gen_dev)
    gen_dev = [k for k in gen_dev if k in dev.pos]
    w_dev = S.len_targets([r for r in D.load("dev") if r["id"] in set(gen_dev)], how, mult)
    w_test = S.len_targets(D.load("test"), how, mult)
    refs = {k: v for k, v in D.refs_of("dev").items() if k in set(gen_dev)}
    res = {}
    tg0 = time.time()
    for name in a.decode.split(","):
        dc = DECODE[name]
        gd = generate(lm, kind, cfg, dev, gen_dev, budgets(gen_dev, w_dev, tpw, a), a, eos, sep,
                      ban, device, dc, a.seed)
        sm = {k: trim(v, w_dev[k]) for k, v in gd.items()}
        sc = D.score_summaries(sm, refs, do_bert=not a.no_bert)
        sc["repeat4"] = repeat_rate(sm.values())
        res[name] = sc
        with open(f"{P6}/results/{a.tag}/gen_{name}_dev.json", "w", encoding="utf-8") as f:
            json.dump(gd, f, ensure_ascii=False)
        log(f"  DEV {a.tag}/{name}: " + " ".join(f"{k} {v:.4f}" for k, v in sc.items()
                                                 if isinstance(v, float)))
    key = "bertscore" if not a.no_bert else "rouge1"
    best = max(res, key=lambda n: res[n][key])
    test_ids = test.ids
    gt = generate(lm, kind, cfg, test, test_ids, budgets(test_ids, w_test, tpw, a), a, eos, sep,
                  ban, device, DECODE[best], a.seed)
    with open(f"{P6}/results/{a.tag}/gen_{best}_test.json", "w", encoding="utf-8") as f:
        json.dump(gt, f, ensure_ascii=False)
    with open(f"{P6}/results/{a.tag}/summaries_test.json", "w", encoding="utf-8") as f:
        json.dump({k: trim(v, w_test[k]) for k, v in gt.items()}, f, ensure_ascii=False)
    with open(f"{P6}/results/{a.tag}/gen_{best}_dev.json", encoding="utf-8") as f:
        gd = json.load(f)
    with open(f"{P6}/results/{a.tag}/summaries_dev.json", "w", encoding="utf-8") as f:
        json.dump({k: trim(v, w_dev[k]) for k, v in gd.items()}, f, ensure_ascii=False)
    if "s" in D._BERT:
        del D._BERT["s"]
    meta = dict(tag=a.tag, kind="model", model=a.model, method=a.method, view=a.view,
                teacher=a.teacher, info=info, steps=step, train_min=round(t_train / 60, 2),
                gen_min=round((time.time() - tg0) / 60, 2), peak_gb=round(peak, 2),
                final_loss=float(np.mean(losses[-25:])) if losses else None,
                length=dict(how=how, mult=mult), decode=best, dev_by_decode=res,
                dev=res[best], n_gen_dev=len(gen_dev), args=vars(a))
    with open(f"{P6}/results/{a.tag}/meta.json", "w") as f:
        json.dump(meta, f, indent=1, default=str)
    if os.path.exists(resume):
        os.remove(resume)
    log(f"DONE {a.tag}: dev {key} {res[best][key]:.4f} ({best}) | {meta['train_min']} train-min, "
        f"{meta['gen_min']} gen-min, peak {meta['peak_gb']} GB")


def _trainable_keys(lm):
    names = {n for n, p in lm.named_parameters() if p.requires_grad}
    keep = set()
    for k in lm.state_dict():
        if k in names or k.endswith(".keep"):      # AdaLoRA rank masks are buffers
            keep.add(k)
    return keep


def repeat_rate(texts, n=4):
    """Share of word 4-grams that are repeats inside their own summary: the
    redundancy number the brief asks about, and the loop detector."""
    rep = tot = 0
    for t in texts:
        w = t.lower().split()
        g = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
        tot += len(g)
        rep += len(g) - len(set(g))
    return rep / max(tot, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    t = sp.add_parser("tok")
    t.add_argument("--view", default="sel1024")
    t.add_argument("--splits", default="fit,dev,test")
    t.add_argument("--procs", type=int, default=os.cpu_count())
    t.set_defaults(fn=cmd_tok)

    r = sp.add_parser("train")
    r.add_argument("--tag", required=True)
    r.add_argument("--model", required=True, help="M1_SC M2_LexArcED M3_LexArcL M4_LexArcP")
    r.add_argument("--ckpt", default="best.pt")
    r.add_argument("--method", default="lorma")
    r.add_argument("--rank", type=int, default=16)
    r.add_argument("--alpha", type=float, default=32.0)
    r.add_argument("--targets", default="attn", choices=["qv", "attn", "all"])
    r.add_argument("--lora-dropout", type=float, default=0.05)
    r.add_argument("--ada-target", type=int, default=6)
    r.add_argument("--bottleneck", type=int, default=64)
    r.add_argument("--train-top", type=int, default=6)
    r.add_argument("--keep-every", type=int, default=1)
    r.add_argument("--view", default="sel1024", help="sel1024 | sel2048 | trunc")
    r.add_argument("--doc-tokens", type=int, default=1152)
    r.add_argument("--sum-tokens", type=int, default=768)
    r.add_argument("--no-ntk", action="store_true")
    r.add_argument("--batch", type=int, default=4)
    r.add_argument("--accum", type=int, default=4)
    r.add_argument("--epochs", type=float, default=2.0)
    r.add_argument("--budget-min", type=float, default=25.0)
    r.add_argument("--probe", type=int, default=10)
    r.add_argument("--lr", type=float, default=None)
    r.add_argument("--wd", type=float, default=0.01)
    r.add_argument("--warmup", type=float, default=0.05)
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--teacher", default=None)
    r.add_argument("--kd-alpha", type=float, default=0.5)
    r.add_argument("--kd-temp", type=float, default=2.0)
    r.add_argument("--decode", default="g,s")
    r.add_argument("--gen-dev", type=int, default=200)
    r.add_argument("--gen-batch", type=int, default=12)
    r.add_argument("--gen-headroom", type=float, default=1.6)
    r.add_argument("--no-bert", action="store_true")
    r.add_argument("--no-ckpt", action="store_true")
    r.add_argument("--save-every-min", type=float, default=10.0)
    r.add_argument("--limit", type=int, default=0)
    r.set_defaults(fn=cmd_train)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
