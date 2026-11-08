#!/usr/bin/env python3
"""CS-781 Phase 1 -- task-format SFT on top of a pretrained LexArc-D.

    python sft_phase1.py --from M1_LexArcD_full --model M1_SFT --hours 1.5

WHY THIS AND NOT MORE PRETRAINING

Measured, not assumed: going from 55M to 586M pretraining tokens -- 10.6x --
moved the leaderboard from 0.8308 to 0.8359, i.e. +0.005 for a 10x compute
increase. Another 6-hour pretraining run buys maybe +0.002. Meanwhile one
decoding knob on M2 was worth 0.025. Pretraining tokens are not where the
remaining score is.

What the pretrained model actually does is next-token prediction over arbitrary
1024-token windows. What the competition asks for is: given exactly 512 words of
judgment, produce exactly 128 words of continuation. Those are different tasks
and the second is never trained for. This script trains for it directly:

    window        800 tokens
    [0 : 630]     prompt context -- read, never graded
    [630 : 800]   continuation   -- the only region with loss

630 tokens is the measured median of a real 512-word test prompt (624), and 170
tokens is ~128 words at the test set's 1.207 tokens/word. So the model is
trained on the exact conditional the scorer evaluates, including how much
context it gets and how far it has to run.

MODEL SELECTION IS BY DEV BERTSCORE, NOT BY LOSS

SFT on a narrow distribution overfits, and SFT loss is not the objective. The
best checkpoint here is the one that generates best, measured on the held-out
dev split with the real metric. Loss is logged and ignored.

The data mix is ~90% Supreme Court, matching the classifier's estimate of the
official test set, because unlike pretraining there is no reason to want
breadth at this stage.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

ROOT = os.environ.get("CS781_ROOT", "/content/drive/MyDrive/cs781_capstone")
sys.path.insert(0, f"{ROOT}/shared/lib")
from cs781_bpe import BPETokenizer                                   # noqa: E402
from cs781_model import Config, LexArcD, build_optimizers            # noqa: E402
from train_phase1 import ShardLoader, load_clean_dev, log, safe_save  # noqa: E402
from score_and_submit import Scorer, gen_guaranteed, report, admissible  # noqa: E402

# matches the measured source composition of the official test set (87-95% SC)
SFT_MIX = {"train_supreme_court": 0.90, "train_high_courts": 0.05,
           "train_acts": 0.03, "train_tribunals": 0.02}
EVAL_DEC = dict(temperature=0.8, top_p=0.9, repetition_penalty=1.05)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", required=True,
                    help="model dir to start from, e.g. M4_LexArcP")
    ap.add_argument("--src-ckpt", default="best.pt")
    ap.add_argument("--model", default="M1_SFT")
    ap.add_argument("--hours", type=float, default=1.5)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--target-tokens", type=int, default=200)
    ap.add_argument("--muon-lr", type=float, default=0.0035)   # ~14% of pretrain
    ap.add_argument("--adam-lr", type=float, default=1.5e-4)
    ap.add_argument("--eval-min", type=float, default=18.0)
    ap.add_argument("--n-dev", type=int, default=150)
    ap.add_argument("--no-compile", action="store_true")
    a = ap.parse_args()

    OUT = f"{ROOT}/{a.model}"
    for d in ("ckpt", "logs", "gen"):
        os.makedirs(f"{OUT}/{d}", exist_ok=True)
    dev = "cuda"
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(1337)

    p = torch.cuda.get_device_properties(0)
    assert p.major * 10 + p.minor >= 80, f"{p.name}: bf16 is emulated below SM80"
    log(f"GPU {p.name} {p.total_memory/1e9:.1f} GB")

    tok = BPETokenizer.load(f"{ROOT}/shared/tokenizer/bpe_legal_32k.json")
    src = f"{ROOT}/{a.src}/ckpt/{a.src_ckpt}"
    ck = torch.load(src, map_location="cpu", weights_only=False)
    cfg = Config(**{k: v for k, v in ck["cfg"].items() if k in Config.__annotations__})
    if getattr(cfg, "use_pointer", False):
        from cs781_model_ptr import LexArcP
        model = LexArcP(cfg).to(dev)
    else:
        model = LexArcD(cfg).to(dev)
    model.load_state_dict({k: v.float() for k, v in ck["model"].items()})
    log(f"loaded {src} (pretrain step {ck.get('step')}, val {ck.get('val')})")
    n = model.n_params()
    log(f"params {n/1e6:.1f}M")
    assert 50e6 < n < 300e6, f"PARAM BUDGET VIOLATED: {n/1e6:.1f}M"

    # Prompt length is SAMPLED, not fixed. Real test prompts tokenise to a
    # median of 624 with p95 at 744, and training on one fixed split risks
    # teaching the model "the continuation begins at position 630" rather than
    # "continue this judgment". Three shapes -> three compile variants, then
    # stable. Target stays 200 tokens, comfortably above the ~165 needed for
    # 128 words at the test set's 1.207 tokens/word.
    PROMPTS = (560, 630, 700)
    windows = [q + a.target_tokens for q in PROMPTS]
    assert max(windows) <= cfg.ctx, (
        f"window {max(windows)} exceeds the trained context {cfg.ctx}; RoPE "
        "would extrapolate into positions the model has never seen")
    loaders = {w: ShardLoader(a.batch, w, dev) for w in windows}
    rng = np.random.default_rng(0)
    dp, dt = load_clean_dev(tok)
    keys = list(dp)[:a.n_dev]
    dev_p, dev_t = {k: dp[k] for k in keys}, {k: dt[k] for k in keys}
    sc = Scorer(dev)

    log(f"SFT: prompt tokens sampled from {PROMPTS} (ungraded) + "
        f"{a.target_tokens} target (graded) | mix {SFT_MIX}")

    # --- baseline: what the pretrained model scores before any SFT -----------
    g = gen_guaranteed(model, "d", tok, dev_p, dev, a.batch, **EVAL_DEC)
    base = sc(g, dev_t)
    report("pretrained", base)
    best_bs, best_tag = base["bertscore"], "pretrained(no SFT)"
    safe_save({"model": {k: v.to(torch.bfloat16) for k, v in model.state_dict().items()},
               "cfg": vars(cfg), "step": 0, "val": None, "dev_bertscore": best_bs},
              f"{OUT}/ckpt/best.pt")

    opts, counts = build_optimizers(model, a.muon_lr, a.adam_lr)
    log(f"optimizers: Muon on {counts['muon']}, AdamW on {counts['adam']} "
        f"(lr {a.muon_lr} / {a.adam_lr}, ~14% of pretraining)")
    net = model if a.no_compile else torch.compile(model)
    base_lrs = [[g_["lr"] for g_ in o.param_groups] for o in opts]

    budget = a.hours * 3600
    warm = 0.03

    def set_lr(frac):
        s = (frac + 1e-9) / warm if frac < warm else \
            0.5 * (1 + math.cos(math.pi * min((frac - warm) / (1 - warm), 1.0)))
        for o, bl in zip(opts, base_lrs):
            for g_, b in zip(o.param_groups, bl):
                g_["lr"] = b * s

    start, last_eval, step, hist = time.time(), time.time(), 0, []
    history = [dict(step=0, dev_bertscore=best_bs, tag="pretrained")]

    while True:
        elapsed = time.time() - start
        if elapsed > budget:
            break
        set_lr(elapsed / budget)
        q = int(rng.choice(PROMPTS))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = net(loaders[q + a.target_tokens].batch(SFT_MIX), q)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for o in opts:
            o.step()
            o.zero_grad(set_to_none=True)
        step += 1

        if step % 200 == 0:
            hist.append(dict(step=step, loss=loss.item(), grad_norm=float(gn),
                             elapsed=round(elapsed)))
            if step % 600 == 0:
                log(f"step {step:5d} ({elapsed/budget:5.1%}) loss {loss.item():.4f} "
                    f"gn {float(gn):.2f} {elapsed/60:.0f}m/{budget/60:.0f}m")

        # --- select on generation quality, not on loss -----------------------
        if (time.time() - last_eval) > a.eval_min * 60:
            last_eval = time.time()
            model.eval()
            g = gen_guaranteed(model, "d", tok, dev_p, dev, a.batch, **EVAL_DEC)
            s = sc(g, dev_t)
            model.train()
            report(f"sft@{step}", s)
            history.append(dict(step=step, dev_bertscore=s["bertscore"],
                                rouge_l=s["rouge_l"], bleu4=s["bleu4"],
                                fluency=s["fluency"], repeat=s["repeat"]))
            if admissible(s) and s["bertscore"] > best_bs:
                best_bs, best_tag = s["bertscore"], f"sft@{step}"
                safe_save({"model": {k: v.to(torch.bfloat16)
                                     for k, v in model.state_dict().items()},
                           "cfg": vars(cfg), "step": step, "val": None,
                           "dev_bertscore": best_bs}, f"{OUT}/ckpt/best.pt")
                log(f"   new best: {best_bs:.4f} (+{best_bs-base['bertscore']:+.4f} "
                    f"over pretrained)")
            with open(f"{OUT}/logs/sft.jsonl", "w") as fh:
                for r in history:
                    fh.write(json.dumps(r) + "\n")

    mean_w = sum(windows) / len(windows)
    log(f"SFT done: {step:,} steps, {step*a.batch*mean_w/1e6:,.0f}M tokens seen "
        f"({step*a.batch*a.target_tokens/1e6:,.0f}M graded), "
        f"{(time.time()-start)/3600:.2f} h")
    log(f"best checkpoint: {best_tag} at dev BERTScore {best_bs:.4f} "
        f"(pretrained baseline {base['bertscore']:.4f}, "
        f"delta {best_bs-base['bertscore']:+.4f})")
    if best_tag.startswith("pretrained"):
        log("   SFT did not beat the starting point. That is a result: report it, "
            "do not submit it.")
    json.dump(history, open(f"{OUT}/logs/sft_history.json", "w"), indent=2)
    log("now run: score_and_submit.py --model " + a.model)


if __name__ == "__main__":
    main()
