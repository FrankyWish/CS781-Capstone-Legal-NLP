#!/usr/bin/env python
"""Phase 7 -- contrastive fine-tuning of the Phase-1 trunks for prior-case retrieval,
with the adaptation methods the brief requires and the compulsory KD pair.

Why this exists.  Zero-shot, the four trunks sit at 0.27-0.32 on devx and, worse, the
stacker shows they carry almost nothing the lexical systems do not already have: a stack
over lexical members alone reaches 0.4113 against 0.4158 for a stack over all eleven.
So the dense side has to be *trained* or there is no model-based path in this phase at
all -- and the brief grades model-based work.

What it trains.  InfoNCE over (query, gold prior case, hard negatives), where the
negatives are mined from a lexical ranking of the train pool, because random negatives
from a 4,320-document pool are trivially separable and teach nothing.  One randomly
placed window per document per step: judgments run to 500k tokens, so a step cannot see
a whole one, and resampling the window each epoch covers the document over training.

How it reaches the leaderboard.  After training it writes the SAME chunk-embedding cache
that `phase7_enc.py embed` writes, by calling `phase7_enc.embed_side` with the trained
trunk.  So the inference path is byte-for-byte the one every zero-shot dense system used,
`phase7_enc.py dense` ranks it unchanged, and `phase7_stack.py` can take it as a member.
Nothing is re-implemented, and adapters are picked up automatically because `embed_side`
calls the blocks' own modules.

Integrity: train queries whose text copies a test query are dropped (the gate found one;
this recomputes the check rather than trusting that the gate ran), and so are the 10
train queries with no gold labels.

  python phase7_emb.py train --tag ft_lorma_m2 --method lorma
  python phase7_emb.py kd    --tag kd_half --teacher ft_lorma_m2 --keep-every 2
  python phase7_emb.py bench --tags ft_lorma_m2,kd_half
"""
import argparse
import dataclasses
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import phase7_data as D
import phase7_rank as R
import phase7_enc as E

# phase6_peft implements LoRA, rsLoRA, LoRMA, QLoRA (NF4 + double quant), AdaLoRA, IA3,
# adapters, partial and norm-only on exactly these Phase-1 modules, and is reused
# unchanged -- which is why five techniques is a day of GPU time and not a week.
PEFT_DIR = os.environ.get(
    "CS781_PEFT", os.path.normpath(os.path.join(HERE, "..", "..", "phase6", "code")))
if PEFT_DIR not in sys.path:
    sys.path.insert(0, PEFT_DIR)
import phase6_peft as P

P7 = D.P7
MODELS = f"{P7}/models"
SEED = D.SEED

# Methods whose adapters live inside the blocks' Linear slots, so embed_side picks them
# up with no merge step.  Prefix and prompt tuning would need the forward pass to accept
# injected states, which enc_forward/dec_forward do not, so they are out of scope here;
# the brief needs five techniques and LoRMA + KD, and these give seven.
METHODS = ("full", "partial", "norm", "lora", "rslora", "lorma", "qlora", "ia3")


def log(*a):
    print(" ", *a, flush=True)


def gpu_or_die(allow_cpu=False):
    if not torch.cuda.is_available():
        if allow_cpu:
            log("  !! no GPU: running on CPU, which is only useful for a smoke test")
            return torch.device("cpu")
        raise SystemExit("no GPU")
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    if cap[0] < 8:
        raise SystemExit(f"STOP: {name} is SM{cap[0]}{cap[1]}; bf16 is emulated below "
                         f"SM80 and runs ~30x slow. Use an L4.")
    log(f"  {name} SM{cap[0]}{cap[1]}")
    return torch.device("cuda")


# ------------------------------------------------------------------ the retriever ---
class Retriever(nn.Module):
    """Wraps a Phase-1 trunk so phase6_peft.apply_method can patch `.trunk`, and returns
    the same pooled, L2-normalised vector that phase7_enc.embed_side produces -- so the
    model is trained through exactly the function that will later index with it."""

    def __init__(self, lm, kind, cfg, pool="mean", bidir=False):
        super().__init__()
        self.trunk = lm
        self.kind, self.cfg = kind, cfg
        self.pool_how, self.bidir = pool, bidir

    def forward(self, idx, ok):
        st = (E.enc_forward(self.trunk, self.cfg, idx, ok) if self.kind == "ed"
              else E.dec_forward(self.trunk, self.cfg, idx, ok, bidir=self.bidir))
        return F.normalize(E.pool(st.float(), ok, self.pool_how), dim=-1)


def build(a, keep_every=1):
    lm, kind, cfg = E.load_lm(a.model, a.ckpt, keep_every)
    factor = max(a.chunk / E.trained_len(kind, cfg), 1.0)
    theta = E.ntk(cfg, factor)
    log(f"  {a.model} kind={kind} d={cfg.d_model} chunk={a.chunk} "
        f"rope x{factor:.2f} theta {theta:.0f} keep_every={keep_every}")
    return Retriever(lm, kind, cfg, a.pool, a.bidir), kind, cfg


# ------------------------------------------------------------------------- the data --
def contaminated_train_ids():
    """Train queries whose normalised text copies a test query.  Their gold labels are a
    test query's answer key, so training on them is training on leaked test labels.  The
    gate reported one; this recomputes it so the trainer cannot depend on the gate having
    been run first."""
    test_keys = {D.norm_key(r["sents"]) for r in D.queries("test")}
    return {r["id"] for r in D.queries("train")
            if D.norm_key(r["sents"]) in test_keys}


class TrainSet:
    def __init__(self, neg_tag, neg_pool):
        self.gold = D.golds_in_pool("train")
        self.qtok, self.ctok = E.Tok("train", "q"), E.Tok("train", "c")
        rank, _ = R.load_run(neg_tag, "train")
        if rank is None:
            raise SystemExit(f"no train ranking for {neg_tag}: run it with --splits train")
        bad = contaminated_train_ids()
        self.items, nogold = [], 0
        for q, gs in self.gold.items():
            if q in bad or q not in self.qtok.pos:
                continue
            if not gs:
                nogold += 1
                continue
            gset = set(gs)
            pos = [c for c in gs if c in self.ctok.pos]
            negs = [c for c in rank.get(q, [])[:neg_pool]
                    if c not in gset and c != q and c in self.ctok.pos]
            if pos and negs:
                self.items.append((q, pos, negs))
        log(f"  train: {len(self.items)} usable queries "
            f"({len(bad)} contaminated and {nogold} gold-less dropped), "
            f"{np.mean([len(p) for _, p, _ in self.items]):.1f} positives and "
            f"{np.mean([len(n) for _, _, n in self.items]):.0f} hard negatives each")

    def window(self, tok, i, chunk, rng):
        d = tok.doc(i)
        if len(d) <= chunk:
            return d
        s = int(rng.integers(0, len(d) - chunk + 1))
        return d[s:s + chunk]

    def batch(self, idxs, chunk, n_neg, rng, device):
        qs, docs, labels = [], [], []
        for j in idxs:
            q, pos, negs = self.items[j]
            qs.append(self.window(self.qtok, self.qtok.pos[q], chunk, rng))
            labels.append(len(docs))
            p = pos[int(rng.integers(len(pos)))]
            docs.append(self.window(self.ctok, self.ctok.pos[p], chunk, rng))
            k = min(n_neg, len(negs))
            for c in rng.choice(negs, size=k, replace=False):
                docs.append(self.window(self.ctok, self.ctok.pos[str(c)], chunk, rng))
        return (pad(qs, chunk, device), pad(docs, chunk, device),
                torch.tensor(labels, dtype=torch.long, device=device))


def pad(seqs, chunk, device):
    T = max(len(s) for s in seqs)
    T = min(int(np.ceil(T / E.PAD_MULT) * E.PAD_MULT), chunk)
    eos = E.eos_id()
    idx = np.full((len(seqs), T), eos, dtype=np.int64)
    ok = np.zeros((len(seqs), T), dtype=bool)
    for r, s in enumerate(seqs):
        s = s[:T]
        idx[r, :len(s)] = s
        ok[r, :len(s)] = True
    return (torch.from_numpy(idx).to(device), torch.from_numpy(ok).to(device))


# ------------------------------------------------------------------------ training ---
def optimiser(model, a):
    tids = P.trunk_param_ids(model)
    trunk = [p for p in model.parameters() if p.requires_grad and id(p) in tids]
    head = [p for p in model.parameters() if p.requires_grad and id(p) not in tids]
    groups = [g for g in ({"params": trunk, "lr": a.lr},
                          {"params": head, "lr": a.lr_head}) if g["params"]]
    return torch.optim.AdamW(groups, weight_decay=a.wd, betas=(0.9, 0.98))


def lr_at(step, a):
    if step < a.warmup:
        return (step + 1) / max(a.warmup, 1)
    t = (step - a.warmup) / max(a.steps - a.warmup, 1)
    return 0.5 * (1 + np.cos(np.pi * min(t, 1.0))) * (1 - a.lr_floor) + a.lr_floor


def run_training(model, ts, a, device, teacher=None):
    model.to(device).train()
    if teacher is not None:
        teacher.to(device).eval()
    opt = optimiser(model, a)
    base = [g["lr"] for g in opt.param_groups]
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(ts.items))
    cur, t0, hist = 0, time.time(), []
    amp = dict(device_type=device.type, dtype=torch.bfloat16,
               enabled=(device.type == "cuda"))
    for step in range(a.steps):
        if cur + a.batch > len(order):
            order, cur = rng.permutation(len(ts.items)), 0
        sel = order[cur:cur + a.batch]
        cur += a.batch
        (qi, qo), (di, do), lab = ts.batch(sel, a.chunk, a.n_neg, rng, device)
        s = lr_at(step, a)
        for g, b in zip(opt.param_groups, base):
            g["lr"] = b * s
        with torch.autocast(**amp):
            qv, dv = model(qi, qo), model(di, do)
            S = qv @ dv.T / a.temp
            loss = F.cross_entropy(S, lab)
            if teacher is not None:
                with torch.no_grad():
                    tq, td = teacher(qi, qo), teacher(di, do)
                    St = (tq @ td.T / a.temp).float()
                kl = F.kl_div(F.log_softmax(S.float(), -1),
                              F.log_softmax(St, -1), log_target=True,
                              reduction="batchmean")
                loss = (1 - a.kd_alpha) * loss + a.kd_alpha * kl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], a.clip)
        opt.step()
        hist.append(float(loss.detach()))
        if step % a.log_every == 0 or step == a.steps - 1:
            with torch.no_grad():
                acc = float((S.argmax(-1) == lab).float().mean())
            log(f"    step {step + 1}/{a.steps} loss {np.mean(hist[-a.log_every:]):.4f} "
                f"acc {acc:.3f} lr x{s:.3f} gnorm {float(gn):.2f} "
                f"({time.time() - t0:.0f}s)")
    return dict(final_loss=round(float(np.mean(hist[-20:])), 5),
                steps=a.steps, train_seconds=round(time.time() - t0, 1))


# -------------------------------------------------------------- index with it, save --
class EmbArgs:
    def __init__(self, a):
        self.chunk, self.max_chunks = a.chunk, a.max_chunks
        self.batch, self.pool, self.bidir = a.embed_batch, a.pool, a.bidir


def embed_and_meta(model, kind, cfg, a, device, extra):
    """Write the same cache phase7_enc.embed writes, so `dense` ranks it unchanged.

    Weights stay fp32 and the matmuls run under autocast instead: NF4Linear holds
    quantised buffers that a .to(bfloat16) would corrupt, and fp32 for a 35M trunk is
    140MB, which is free here."""
    ea = EmbArgs(a)
    model.trunk.eval()
    peak0 = (torch.cuda.max_memory_allocated() if device.type == "cuda" else 0)
    build_s = {}
    for split in a.splits.split(","):
        for side in ("q", "c"):
            p = E.emb_path(a.tag, split, side)
            if os.path.exists(p) and not a.force:
                log(f"  {p} exists")
                continue
            tk = E.Tok(split, side)
            t0 = time.time()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=(device.type == "cuda")):
                flat, off = E.embed_side(model.trunk, kind, cfg, tk, ea, device)
            tmp = p + ".tmp.npz"
            np.savez(tmp, flat=flat, off=off, ids=np.array(tk.ids))
            os.replace(tmp, p)
            build_s[f"{split}_{side}"] = round(time.time() - t0, 1)
            log(f"  wrote {p}: {len(tk)} docs, {flat.shape[0]} chunks "
                f"({build_s[f'{split}_{side}']}s)")
    meta = dict(tag=a.tag, kind="model", model=a.model, encoder=kind, pool=a.pool,
                bidir=bool(a.bidir), chunk=a.chunk, max_chunks=a.max_chunks,
                d_model=cfg.d_model, index_seconds=build_s,
                n_params=sum(p.numel() for p in model.trunk.parameters()),
                peak_mem_mb=round((torch.cuda.max_memory_allocated() - peak0) / 2**20, 1)
                if device.type == "cuda" else None, **extra)
    R.save_run(a.tag, "meta", {}, None, meta)
    return meta


def save_ckpt(model, cfg, a, info, keep_every=1):
    """cfg + state_dict + the method and its hyperparameters, so load_trained can
    rebuild the same patched module graph before loading the weights."""
    os.makedirs(MODELS, exist_ok=True)
    p = f"{MODELS}/{a.tag}.pt"
    torch.save(dict(cfg=dataclasses.asdict(cfg) if dataclasses.is_dataclass(cfg) else cfg,
                    model=model.state_dict(), method=a.method, keep_every=keep_every,
                    peft=dict(targets=a.targets, rank=a.rank, alpha=a.alpha,
                              lora_dropout=a.lora_dropout, bottleneck=a.bottleneck,
                              train_top=a.train_top, ada_target=a.ada_target),
                    base=dict(model=a.model, ckpt=a.ckpt, chunk=a.chunk, pool=a.pool,
                              bidir=bool(a.bidir)),
                    info=info), p)
    log(f"  saved {p}")
    return p


def load_trained(tag, device):
    """Rebuild a trained retriever: base trunk, same adapters, then the weights."""
    p = f"{MODELS}/{tag}.pt"
    if not os.path.exists(p):
        raise SystemExit(f"no checkpoint for {tag} at {p}")
    ck = torch.load(p, map_location="cpu", weights_only=False)
    b = ck["base"]
    ns = argparse.Namespace(model=b["model"], ckpt=b["ckpt"], chunk=b["chunk"],
                            pool=b["pool"], bidir=b["bidir"])
    model, kind, cfg = build(ns, ck.get("keep_every", 1))
    ns2 = argparse.Namespace(method=ck["method"], **ck["peft"])
    P.apply_method(model, ns2)
    miss, unexp = model.load_state_dict(ck["model"], strict=False)
    miss = [k for k in miss if not k.endswith("head.weight")]
    if miss or unexp:
        log(f"  !! {tag}: {len(miss)} missing, {len(unexp)} unexpected weights")
    return model.to(device), kind, cfg, ck


# ------------------------------------------------------------------------ commands ---
def cmd_train(a):
    device = gpu_or_die(a.cpu)
    model, kind, cfg = build(a)
    n, ctrl, info = P.apply_method(model, a)
    total = sum(p.numel() for p in model.parameters())
    log(f"  method {a.method}: {n:,} trainable of {total:,} ({100.0 * n / total:.2f}%) "
        f"{json.dumps(info)}")
    if ctrl is not None:
        ctrl.total = a.steps
    ts = TrainSet(a.neg_tag, a.neg_pool)
    stats = run_training(model, ts, a, device)
    log(f"  trained in {stats['train_seconds']}s, final loss {stats['final_loss']}")
    save_ckpt(model, cfg, a, dict(**stats, **info, trainable=n))
    meta = embed_and_meta(model, kind, cfg, a, device,
                          dict(method=a.method, trainable=n, total_params=total, **stats))
    log(f"  rank it with:  python phase7_enc.py dense --tag {a.tag} "
        f"--agg maxsim --splits {a.splits} --topn 200")
    return meta


def cmd_kd(a):
    """Compulsory distillation: teacher = a trained retriever, student = the same trunk
    with every `keep-every`-th layer kept.  The loss is KL between the student's and the
    teacher's softmax over the SAME in-batch candidate set, mixed with the student's own
    InfoNCE, so the student learns the teacher's ordering and not just its embedding."""
    device = gpu_or_die(a.cpu)
    teacher, tkind, tcfg, tck = load_trained(a.teacher, device)
    for p in teacher.parameters():
        p.requires_grad_(False)
    tn = sum(p.numel() for p in teacher.trunk.parameters())
    log(f"  teacher {a.teacher}: {tn:,} params, method {tck['method']}")
    model, kind, cfg = build(a, a.keep_every)
    n, ctrl, info = P.apply_method(model, a)
    sn = sum(p.numel() for p in model.trunk.parameters())
    log(f"  student: {sn:,} params ({100.0 * sn / tn:.1f}% of teacher), "
        f"{n:,} trainable, kd_alpha {a.kd_alpha}")
    ts = TrainSet(a.neg_tag, a.neg_pool)
    stats = run_training(model, ts, a, device, teacher=teacher)
    del teacher
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    save_ckpt(model, cfg, a, dict(**stats, **info, trainable=n), a.keep_every)
    meta = embed_and_meta(model, kind, cfg, a, device,
                          dict(method=f"kd/{a.method}", teacher=a.teacher,
                               teacher_params=tn, student_params=sn,
                               param_ratio=round(sn / tn, 4), kd_alpha=a.kd_alpha,
                               trainable=n, **stats))
    log(f"  rank it with:  python phase7_enc.py dense --tag {a.tag} "
        f"--agg maxsim --splits {a.splits} --topn 200")
    return meta


def cmd_bench(a):
    """The efficiency table s13.2.2 asks for: parameters, layers, peak indexing memory,
    index build time and single-query latency, measured not estimated."""
    device = gpu_or_die(a.cpu)
    rows = []
    for tag in [t for t in a.tags.split(",") if t]:
        model, kind, cfg, ck = load_trained(tag, device)
        model.trunk.eval()
        nl = (cfg.n_enc_layer + cfg.n_dec_layer if hasattr(cfg, "n_enc_layer")
              else cfg.n_layer)
        tk = E.Tok("test", "q")
        d = tk.doc(0)[:a.chunk]
        qi, qo = pad([d], a.chunk, device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        with torch.no_grad(), torch.autocast(device_type=device.type,
                                             dtype=torch.bfloat16,
                                             enabled=(device.type == "cuda")):
            for _ in range(3):
                model(qi, qo)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(a.reps):
                model(qi, qo)
            if device.type == "cuda":
                torch.cuda.synchronize()
            lat = (time.time() - t0) / a.reps * 1000
        meta = {}
        mp = f"{R.rank_dir(tag)}/meta.json"
        if os.path.exists(mp):
            meta = json.load(open(mp))
        rank, _ = R.load_run(tag, "devx")
        f1 = R.evaluate(rank, "devx")["f1"] if rank else None
        rows.append(dict(tag=tag, params=sum(p.numel() for p in model.trunk.parameters()),
                         layers=nl, latency_ms=round(lat, 2),
                         peak_mem_mb=(round(torch.cuda.max_memory_allocated() / 2**20, 1)
                                      if device.type == "cuda" else None),
                         index_seconds=meta.get("index_seconds"),
                         devx_f1=(round(f1, 4) if f1 is not None else None)))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    D.atomic_json(f"{P7}/results/kd_efficiency.json", rows)
    log(f"\n  {'tag':20s} {'params':>12s} {'layers':>7s} {'ms/query':>9s} "
        f"{'peak MB':>8s} {'devx F1':>8s}")
    for r in rows:
        log(f"  {r['tag']:20s} {r['params']:12,d} {r['layers']:7d} "
            f"{r['latency_ms']:9.2f} {str(r['peak_mem_mb']):>8s} "
            f"{str(r['devx_f1']):>8s}")
    log(f"  wrote {P7}/results/kd_efficiency.json")


def common(p, tag_default=None):
    p.add_argument("--tag", default=tag_default, required=tag_default is None)
    p.add_argument("--model", default="M2_LexArcED")
    p.add_argument("--ckpt", default="best.pt")
    p.add_argument("--pool", default="mean", choices=["mean", "max", "last", "first"])
    p.add_argument("--bidir", action="store_true")
    p.add_argument("--chunk", type=int, default=1024)
    p.add_argument("--max-chunks", type=int, default=8)
    p.add_argument("--embed-batch", type=int, default=16)
    p.add_argument("--splits", default="devx,test")
    p.add_argument("--force", action="store_true")
    p.add_argument("--cpu", action="store_true", help="smoke tests only")
    # data
    p.add_argument("--neg-tag", default="tfidf_sqrt")
    p.add_argument("--neg-pool", type=int, default=100)
    p.add_argument("--n-neg", type=int, default=3)
    # optimisation
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--lr-floor", type=float, default=0.05)
    p.add_argument("--warmup", type=int, default=40)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--temp", type=float, default=0.05)
    p.add_argument("--log-every", type=int, default=25)
    # peft
    p.add_argument("--method", default="lorma", choices=list(METHODS))
    p.add_argument("--targets", default="all")
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=float, default=16.0)
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument("--bottleneck", type=int, default=32)
    p.add_argument("--train-top", type=int, default=4)
    p.add_argument("--ada-target", type=int, default=4)
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    common(sp.add_parser("train")).set_defaults(fn=cmd_train)
    p = common(sp.add_parser("kd"))
    p.add_argument("--teacher", required=True)
    p.add_argument("--keep-every", type=int, default=2)
    p.add_argument("--kd-alpha", type=float, default=0.7)
    p.set_defaults(fn=cmd_kd)
    p = sp.add_parser("bench")
    p.add_argument("--tags", required=True)
    p.add_argument("--chunk", type=int, default=1024)
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--cpu", action="store_true")
    p.set_defaults(fn=cmd_bench)
    a = ap.parse_args()
    if getattr(a, "cmd", None) == "kd" and not hasattr(a, "kd_alpha"):
        a.kd_alpha = 0.7
    a.fn(a)


if __name__ == "__main__":
    main()
