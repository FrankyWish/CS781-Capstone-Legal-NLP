#!/usr/bin/env python3
"""CS-781 Phase 1 -- LexArc-ED (encoder-decoder): gates, train, submission.zip.

    python train_ed.py --model M2_LexArcED --hours 6.0

Shares the loader, metrics, phase mix and dev cleaning with train_phase1.py so
M1 and M2 are compared under identical data, identical tokenizer and identical
wall-clock. Only the architecture and the objective differ -- that is the whole
point of the comparison.

Gates are the same three in spirit, re-derived for this architecture:
  G1  forward()'s decoder loss equals CE(logits_full, window[:, split:])
  G2  overfit one window to < 0.1, then argmax == target on >95% of positions
  G3  incremental decode (self-attn cache + fixed cross-attn K/V) matches a
      full teacher-forced forward
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import zipfile

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.environ.get("CS781_ROOT", "/content/drive/MyDrive/cs781_capstone")
sys.path.insert(0, f"{ROOT}/shared/lib")
from cs781_bpe import BPETokenizer                              # noqa: E402
from cs781_model import Muon                                    # noqa: E402
from cs781_model_ed import EDConfig, LexArcED                   # noqa: E402
from train_phase1 import (ShardLoader, mix_at, fluency, score_dev,             # noqa: E402
                          bertscore_maybe, left_pad_batch, load_clean_dev, log)


def safe_save(obj, path):
    """Write a checkpoint without ever killing the run.

    A failed checkpoint costs 20 minutes of progress. An unhandled exception
    from a full Drive costs the whole run, and Drive filling up mid-run is a
    thing that actually happens. Writes to .tmp then renames, so a partial
    write never replaces a good checkpoint.
    """
    try:
        torch.save(obj, path + ".tmp")
        os.replace(path + ".tmp", path)
        return True
    except Exception as e:
        log(f"   CHECKPOINT WRITE FAILED ({type(e).__name__}: {e}) -- continuing")
        try:
            os.remove(path + ".tmp")
        except OSError:
            pass
        return False


def build_optimizers_ed(model, muon_lr=0.025, adam_lr=1e-3, weight_decay=0.1):
    muon_p, adam_d, adam_p = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2 and "emb" not in n and "head" not in n:
            muon_p.append(p)
        elif p.ndim >= 2:
            adam_d.append(p)
        else:
            adam_p.append(p)
    return [Muon(muon_p, lr=muon_lr),
            torch.optim.AdamW([{"params": adam_d, "weight_decay": weight_decay},
                               {"params": adam_p, "weight_decay": 0.0}],
                              lr=adam_lr, betas=(0.9, 0.95), fused=True)], \
           dict(muon=len(muon_p), adam=len(adam_d) + len(adam_p))


# ------------------------------------------------------------------- gates --
def gate_shift(model, loader):
    log("G1 shift: decoder loss must equal CE(logits_full, window[:, split:])")
    w = loader.batch({"train_supreme_court": 1.0})[:4]
    split = model.cfg.split_points[1]
    cfg = model.cfg
    z, lam = cfg.z_loss, cfg.mlm_lambda
    cfg.z_loss, cfg.mlm_lambda = 0.0, 0.0
    model.eval()
    with torch.no_grad():
        a = model(w, split)[0].item()
        lg = model.logits_full(w, split)
        b = F.cross_entropy(lg.reshape(-1, cfg.vocab_size).float(),
                            w[:, split:].reshape(-1)).item()
    cfg.z_loss, cfg.mlm_lambda = z, lam
    log(f"   forward={a:.6f}  manual={b:.6f}  delta={abs(a-b):.2e}")
    assert abs(a - b) < 1e-3, "G1 FAILED: decoder targets are misaligned."
    log("G1 PASS")


def gate_overfit(model, loader, steps=200, target=0.1):
    log(f"G2 overfit: {steps} steps on one fixed window, loss must go < {target}")
    w = loader.batch({"train_supreme_court": 1.0})[:4].contiguous()
    split = model.cfg.split_points[1]
    lam, model.cfg.mlm_lambda = model.cfg.mlm_lambda, 0.0   # isolate the decoder
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=True)
    model.train()
    last = None
    for i in range(steps):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = model(w, split)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        last = loss.item()
        if i % 50 == 0:
            log(f"   step {i:3d} loss {last:.4f}")
    log(f"   final loss {last:.4f}")
    assert last < target, f"G2 FAILED: cannot memorise 4 windows (loss {last:.3f})"
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        lg = model.logits_full(w, split)
    acc = (lg.argmax(-1) == w[:, split:]).float().mean().item()
    log(f"   argmax(logits[i]) == target[i] on {acc:.1%} of positions")
    assert acc > 0.95, f"G2 FAILED: memorised but misaligned ({acc:.1%})"
    model.cfg.mlm_lambda = lam
    for p in model.parameters():
        p.grad = None
    log("G2 PASS")


def gate_kv_cache(model, loader, n_new=8):
    log("G3 kv-cache: incremental decode must match a teacher-forced forward")
    w = loader.batch({"train_supreme_court": 1.0})[:4]
    split = model.cfg.split_points[1]
    enc = w[:, :split]
    mask = torch.ones_like(enc, dtype=torch.bool)
    model.eval()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        gen = model.generate(enc, mask, n_new, greedy=True)
        full = torch.cat([enc, gen], 1)
        lg = model.logits_full(full, split)
    ref = lg.argmax(-1)
    first = (ref[:, 0] == gen[:, 0]).float().mean().item()
    agree = (ref == gen).float().mean().item()
    log(f"   first token {first:.0%} | all {n_new} tokens {agree:.0%}")
    assert first == 1.0, "G3 FAILED: encoder pass or cross-attention K/V is wrong."
    assert agree > 0.6, f"G3 FAILED: decoder self-attn cache diverges ({agree:.0%})"
    log("G3 PASS")


# -------------------------------------------------------------- generation --
@torch.no_grad()
def generate_for_ed(model, tok, prompts, device, max_ctx=832, batch=16,
                    n_new=210, **dec):
    model.eval()
    keys = list(prompts)
    enc = {k: tok.encode(prompts[k]["text"])[-max_ctx:] for k in keys}
    order = sorted(keys, key=lambda k: len(enc[k]))
    out, t0 = {}, time.time()
    for i in range(0, len(order), batch):
        grp = order[i:i + batch]
        idx, msk = left_pad_batch([enc[k] for k in grp], device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            gen = model.generate(idx, msk, n_new, **dec)
        for k, row in zip(grp, gen.tolist()):
            out[k] = {"text": " ".join(tok.decode(row).split()[:128])}
        if i % (batch * 10) == 0:
            log(f"   generated {i+len(grp)}/{len(order)} "
                f"({(i+len(grp))/max(time.time()-t0,1):.1f}/s)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="M2_LexArcED")
    ap.add_argument("--hours", type=float, default=6.0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--muon-lr", type=float, default=0.025)
    ap.add_argument("--adam-lr", type=float, default=1e-3)
    ap.add_argument("--mlm-lambda", type=float, default=0.3)   # 0.0 = ablation
    ap.add_argument("--warmup", type=float, default=0.02)
    ap.add_argument("--ckpt-min", type=float, default=20.0)
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--skip-gates", action="store_true")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore an existing resume.pt and start over")
    a = ap.parse_args()

    OUT = f"{ROOT}/{a.model}"
    for d in ("ckpt", "logs", "gen"):
        os.makedirs(f"{OUT}/{d}", exist_ok=True)
    dev = "cuda"
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(1337)

    p = torch.cuda.get_device_properties(0)
    sm = p.major * 10 + p.minor
    assert sm >= 80, f"{p.name} is SM{sm}; bf16 is emulated below SM80."
    log(f"GPU {p.name} SM{sm} {p.total_memory/1e9:.1f} GB")

    tok = BPETokenizer.load(f"{ROOT}/shared/tokenizer/bpe_legal_32k.json")
    cfg = EDConfig(vocab_size=tok.vocab_size, mlm_lambda=a.mlm_lambda,
                   mask_id=tok.specials["<|mask|>"],
                   bos_id=tok.specials["<|endoftext|>"])
    model = LexArcED(cfg).to(dev)
    n_all = model.n_params()
    log(f"params {n_all/1e6:.1f}M total, {model.n_params(True)/1e6:.1f}M non-embedding "
        f"({cfg.n_enc_layer} enc + {cfg.n_dec_layer} dec layers)")
    assert 50e6 < n_all < 300e6, f"PARAM BUDGET VIOLATED: {n_all/1e6:.1f}M"

    loader = ShardLoader(a.batch, cfg.window, dev)
    log(f"corpus {loader.tok_total/1e9:.3f}B tokens, window {cfg.window}, "
        f"splits {cfg.split_points}")
    dev_prompts, dev_truth = load_clean_dev(tok)

    resuming = os.path.exists(f"{OUT}/ckpt/resume.pt") and not a.fresh
    if resuming:
        log("resume.pt found -- skipping gates (they reinitialise the model)")
    if not a.skip_gates and not resuming:
        gate_shift(model, loader)
        gate_overfit(model, loader)
        gate_kv_cache(model, loader)
        model = LexArcED(cfg).to(dev)
        log("gates passed, model reinitialised")

    opts, counts = build_optimizers_ed(model, a.muon_lr, a.adam_lr)
    log(f"optimizers: Muon on {counts['muon']} matrices, AdamW on {counts['adam']}")

    # --- resume ------------------------------------------------------------
    # Only the model weights, in fp32, plus the wall-clock already spent. The
    # optimiser state is deliberately NOT saved: Muon's momentum buffers alone
    # are ~400 MB, a Drive write every 20 min would cost real throughput, and
    # momentum rebuilds in ~100 steps out of ~25,000. Losing it is cheaper than
    # saving it.
    resume_path = f"{OUT}/ckpt/resume.pt"
    start_elapsed, step, best_val = 0.0, 0, float("inf")
    if os.path.exists(resume_path) and not a.fresh:
        ck = torch.load(resume_path, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        step, start_elapsed, best_val = ck["step"], ck["elapsed"], ck["val"]
        log(f"RESUMED from step {step:,}, {start_elapsed/3600:.2f} h of "
            f"{a.hours:.1f} h already spent, best val {best_val:.4f}")

    net = model if a.no_compile else torch.compile(model)
    rng = np.random.default_rng(0)

    budget = a.hours * 3600
    for _ in range(5):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = net(loader.batch(mix_at(0.0)), int(rng.choice(cfg.split_points)))
        loss.backward()
        for o in opts:
            o.step()
            o.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(20):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = net(loader.batch(mix_at(0.0)), int(rng.choice(cfg.split_points)))
        loss.backward()
        for o in opts:
            o.step()
            o.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    tps = 20 * a.batch * cfg.window / (time.time() - t0)
    log(f"post-compile throughput: {tps:,.0f} tok/s -> "
        f"~{tps*budget/1e6:,.0f}M tokens in {a.hours:.1f} h")

    base = [[g["lr"] for g in o.param_groups] for o in opts]

    def set_lr(frac):
        if frac < a.warmup:
            sc = (frac + 1e-9) / a.warmup
        else:
            pr = (frac - a.warmup) / (1.0 - a.warmup)
            sc = 0.5 * (1 + math.cos(math.pi * min(pr, 1.0)))
        for o, b in zip(opts, base):
            for g, bb in zip(o.param_groups, b):
                g["lr"] = bb * sc

    start = time.time() - start_elapsed
    ckpt_n = 0
    last_ckpt, checked, hist = time.time(), set(), []
    model.train()

    while True:
        elapsed = time.time() - start
        if elapsed > budget:
            break
        frac = elapsed / budget
        set_lr(frac)
        sp = int(rng.choice(cfg.split_points))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, mlm = net(loader.batch(mix_at(frac)), sp)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for o in opts:
            o.step()
            o.zero_grad(set_to_none=True)
        step += 1

        if step % 100 == 0:
            hist.append(dict(step=step, loss=loss.item(), mlm=float(mlm),
                             grad_norm=float(gn), frac=round(frac, 4),
                             lr=opts[0].param_groups[0]["lr"], elapsed=round(elapsed)))
            if step % 500 == 0:
                log(f"step {step:6d} ({frac:5.1%}) loss {loss.item():.4f} "
                    f"mlm {float(mlm):.3f} gn {float(gn):.2f} "
                    f"{step*a.batch*cfg.window/1e6:,.0f}M tok "
                    f"{elapsed/60:.0f}m/{budget/60:.0f}m")
                with open(f"{OUT}/logs/train.jsonl", "a") as fh:
                    for r in hist:
                        fh.write(json.dumps(r) + "\n")
                hist = []

        for mark in (0.10, 0.25):
            if frac >= mark and mark not in checked:
                checked.add(mark)
                sub = dict(list(dev_prompts.items())[:8])
                g = generate_for_ed(model, tok, sub, dev, batch=8, greedy=True)
                fl = sum(fluency(v["text"]) for v in g.values()) / len(g)
                log(f"FLUENCY @{mark:.0%}: {fl:.1f} bigrams/128w (prose 15-25, salad 1-2)")
                log(f"   sample: {list(g.values())[0]['text'][:240]!r}")
                model.train()
                if fl < 4.0 and mark == 0.25:
                    raise SystemExit(
                        f"ABORT: fluency {fl:.1f} at 25% of budget is word salad. "
                        "Stopping rather than burning the remaining GPU hours.")

        if (time.time() - last_ckpt) > a.ckpt_min * 60:
            last_ckpt = time.time()
            model.eval()
            vl = []
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for s in loader.val:
                    vl.append(model(loader.val_batch(s, 4, 0),
                                    cfg.split_points[1])[0].item())
            model.train()
            v = sum(vl) / len(vl)
            ckpt_n += 1
            blob = {"model": {k: x.to(torch.bfloat16) for k, x in
                              model.state_dict().items()},
                    "cfg": vars(cfg), "step": step, "val": v}
            safe_save(blob, f"{OUT}/ckpt/slot{ckpt_n%3}.pt")
            safe_save({"model": model.state_dict(), "cfg": vars(cfg), "step": step,
                       "elapsed": time.time() - start, "val": min(v, best_val)},
                      f"{OUT}/ckpt/resume.pt")
            if v < best_val:
                best_val = v
                safe_save(blob, f"{OUT}/ckpt/best.pt")
            log(f"ckpt step {step} val_loss {v:.4f} bits/tok {v/math.log(2):.3f}")

    log(f"training done: {step:,} steps, {step*a.batch*cfg.window/1e6:,.0f}M tokens, "
        f"{(time.time()-start)/3600:.2f} h")
    safe_save({"model": model.state_dict(), "cfg": vars(cfg), "step": step},
              f"{OUT}/ckpt/final.pt")

    sub = dict(list(dev_prompts.items())[:200])
    sub_truth = {k: dev_truth[k] for k in sub}
    results = {}
    for name, kw in [("greedy", dict(greedy=True)),
                     ("t0.8_p0.9", dict(temperature=0.8, top_p=0.9,
                                        repetition_penalty=1.05))]:
        g = generate_for_ed(model, tok, sub, dev, batch=16, **kw)
        s = score_dev(g, sub_truth)
        s["bertscore"] = bertscore_maybe(g, sub_truth)
        results[name] = s
        log(f"   {name:12s} ROUGE-L {s['rouge_l']:.4f} BLEU-4 {s['bleu4']:.4f} "
            f"fluency {s['fluency']:.1f} BERTScore {s['bertscore']}")
        json.dump(g, open(f"{OUT}/gen/dev_{name}.json", "w"), ensure_ascii=False)
    best = max(results, key=lambda k: results[k]["bertscore"] or results[k]["rouge_l"])
    log(f"best dev setting: {best}")
    json.dump(results, open(f"{OUT}/logs/dev_scores.json", "w"), indent=2)

    log("generating the 1000 test continuations")
    test = json.load(open(f"{ROOT}/shared/test/test.json", encoding="utf-8"))
    kw = dict(greedy=True) if best == "greedy" else dict(
        temperature=0.8, top_p=0.9, repetition_penalty=1.05)
    pred = generate_for_ed(model, tok, test, dev, batch=16, **kw)
    assert set(pred) == set(test), "ID MISMATCH against the official test file"
    wc = [len(v["text"].split()) for v in pred.values()]
    log(f"continuation words: min {min(wc)} median {sorted(wc)[len(wc)//2]} max {max(wc)}")

    pj = f"{OUT}/gen/prediction.json"
    json.dump(pred, open(pj, "w", encoding="utf-8"), ensure_ascii=False)
    zp = f"{OUT}/gen/submission.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(pj, "prediction.json")
    log(f"WROTE {zp} ({os.path.getsize(zp)/1e6:.1f} MB)")
    log("   " + repr(pred[list(pred)[0]]["text"][:300]))
    log("ALL DONE")


if __name__ == "__main__":
    main()
