#!/usr/bin/env python3
"""CS-781 Phase 1 -- checkpoint selection, decoding sweep, final submission.

    python score_and_submit.py --model M1_LexArcD_full
    python score_and_submit.py --model M2_LexArcED --quick

This is where the last 0.005-0.010 of BERTScore lives. The leaderboard band
between rank 1 and rank 10 is 0.0147, so decoding is not a detail.

Three things it does that the training script does not:

1. CHECKPOINT SELECTION, measured. best.pt (lowest val loss) vs final.pt vs a
   soup of the rotating slots. Validation loss and generation quality are not
   the same objective and §7.4 asks for exactly this comparison.

2. DECODING SWEEP against real BERTScore, staged rather than a full grid, with
   a bootstrap confidence interval on every number. On 250 dev examples a
   BERTScore gap under ~0.003 is noise, and chasing noise is how you convince
   yourself of a result that does not transfer.
   Every setting is also gated on the function-word fluency count, because
   BERTScore cannot see broken syntax and will reward content-word soup -- a
   decoding sweep on the previous attempt actively preferred the more broken
   setting for exactly this reason.

3. LENGTH GUARANTEE. The legal-aware tokenizer splits digit runs, so a
   citation-dense continuation can yield 36 words from 210 tokens. The 1024
   context cannot simply be given a larger generation budget, so short outputs
   are topped up by sliding the window: prompt+generated, left-truncated,
   generate more, repeat. BERTScore F1 is symmetric and the references are
   exactly 128 words, so being short costs recall directly.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zipfile

import numpy as np
import torch

ROOT = os.environ.get("CS781_ROOT", "/content/drive/MyDrive/cs781_capstone")
sys.path.insert(0, f"{ROOT}/shared/lib")
from cs781_bpe import BPETokenizer                                  # noqa: E402
from cs781_model import Config, LexArcD                             # noqa: E402
from cs781_model_ed import EDConfig, LexArcED                       # noqa: E402
from train_phase1 import (generate_for, load_clean_dev, score_dev,  # noqa: E402
                          fluency, log)


def repeat_rate(text, n=4):
    """Fraction of 4-grams that are not unique.

    The function-word bigram count alone is NOT a sufficient fluency gate, and
    the 10%-of-training sample proved it: a model looping
    "the words used in the clause must be given a restricted meaning and that"
    scored 9.3 bigrams/128w -- comfortably "fluent" -- because a repetition
    loop manufactures exactly the bigrams the metric counts. Degenerate
    repetition inflates the thing meant to detect degeneracy.

    Real legal prose repeats some boilerplate, so this is not zero; a loop
    drives it above ~0.4.
    """
    w = text.lower().split()
    if len(w) < n + 1:
        return 0.0
    g = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    return 1.0 - len(set(g)) / len(g)
from train_ed import generate_for_ed                                # noqa: E402


# ------------------------------------------------------------- checkpoints --
def load_ckpt(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfgd = ck["cfg"]
    if "n_enc_layer" in cfgd:
        cfg = EDConfig(**{k: v for k, v in cfgd.items() if k in EDConfig.__annotations__})
        cfg.split_points = tuple(cfg.split_points)
        model = LexArcED(cfg)
        kind = "ed"
    else:
        cfg = Config(**{k: v for k, v in cfgd.items() if k in Config.__annotations__})
        if getattr(cfg, "use_pointer", False):
            from cs781_model_ptr import LexArcP
            model = LexArcP(cfg)
        else:
            model = LexArcD(cfg)
        kind = "d"
    model.load_state_dict({k: v.float() for k, v in ck["model"].items()})
    return model.to(device).eval(), cfg, kind, ck.get("step"), ck.get("val")


def soup(paths, device):
    """Uniform weight average. Free, and it usually buys a little."""
    acc, n = None, 0
    for p in paths:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        sd = {k: v.float() for k, v in ck["model"].items()}
        acc = sd if acc is None else {k: acc[k] + sd[k] for k in acc}
        n += 1
    return {k: v / n for k, v in acc.items()}, n


# --------------------------------------------------------------- generation --
def gen(model, kind, tok, prompts, device, max_ctx, n_new, batch, **kw):
    f = generate_for if kind == "d" else generate_for_ed
    return f(model, tok, prompts, device, max_ctx=max_ctx, n_new=n_new,
             batch=batch, **kw)


def gen_guaranteed(model, kind, tok, prompts, device, batch=16, rounds=3, **kw):
    """Generate, then slide the window to top up anything under 128 words."""
    max_ctx = 790 if kind == "d" else 832
    out = gen(model, kind, tok, prompts, device, max_ctx, 210, batch, **kw)
    for r in range(rounds):
        short = {k: v for k, v in out.items() if len(v["text"].split()) < 128}
        if not short:
            break
        log(f"   top-up round {r+1}: {len(short)} continuations under 128 words")
        nxt = {}
        for k in short:
            # continue from prompt + what we already produced
            nxt[k] = {"text": prompts[k]["text"] + " " + out[k]["text"]}
        more = gen(model, kind, tok, nxt, device, max_ctx - 120, 330, batch, **kw)
        for k in short:
            merged = (out[k]["text"] + " " + more[k]["text"]).split()[:128]
            out[k] = {"text": " ".join(merged)}
    still = sum(1 for v in out.values() if len(v["text"].split()) < 128)
    if still:
        log(f"   WARNING: {still} continuations still short after {rounds} rounds")
    return out


# ------------------------------------------------------------------ scoring --
class Scorer:
    """One BERTScorer, reused. Loading roberta-large per call would dominate."""

    def __init__(self, device):
        import bert_score
        self.s = bert_score.BERTScorer(lang="en", rescale_with_baseline=False,
                                       batch_size=32, device=device)

    def __call__(self, pred, truth, n_boot=1000, seed=0):
        ks = list(pred)
        _, _, f1 = self.s.score([pred[k]["text"] for k in ks],
                                [truth[k]["text"] for k in ks])
        f1 = f1.numpy()
        rng = np.random.default_rng(seed)
        boot = np.array([f1[rng.integers(0, len(f1), len(f1))].mean()
                         for _ in range(n_boot)])
        base = score_dev(pred, truth)
        base.update(bertscore=float(f1.mean()),
                    bs_lo=float(np.percentile(boot, 2.5)),
                    bs_hi=float(np.percentile(boot, 97.5)),
                    repeat=float(np.mean([repeat_rate(v["text"])
                                          for v in pred.values()])),
                    words=float(np.mean([len(v["text"].split())
                                         for v in pred.values()])))
        return base


FLUENCY_FLOOR = 8.0     # below this the text is degenerating, whatever BERTScore says
REPEAT_CEIL = 0.35      # above this it is looping, which inflates fluency


def admissible(s):
    return s["fluency"] >= FLUENCY_FLOOR and s["repeat"] <= REPEAT_CEIL


def report(tag, s):
    why = []
    if s["fluency"] < FLUENCY_FLOOR:
        why.append("fluency")
    if s["repeat"] > REPEAT_CEIL:
        why.append("repetition")
    flag = "" if not why else "   <-- REJECTED: " + "+".join(why)
    log(f"   {tag:16s} BERTScore {s['bertscore']:.4f} "
        f"[{s['bs_lo']:.4f},{s['bs_hi']:.4f}]  ROUGE-L {s['rouge_l']:.4f}  "
        f"BLEU-4 {s['bleu4']:.4f}  fl {s['fluency']:4.1f}  "
        f"rep4 {s['repeat']:.2f}  words {s['words']:5.1f}{flag}")


def pick(results):
    """BERTScore decides, but only among settings that are actually prose.

    BERTScore cannot see broken syntax and a sweep on the previous attempt
    actively preferred the more broken setting. Both guards are needed: the
    fluency floor catches content-word soup, the repetition ceiling catches
    loops -- and loops are what pass the fluency floor.
    """
    ok = {k: v for k, v in results.items() if admissible(v)}
    if not ok:
        log("   every setting was rejected; falling back to the least repetitive")
        return min(results, key=lambda k: results[k]["repeat"])
    top = max(ok.values(), key=lambda v: v["bertscore"])["bertscore"]
    # Ties go to the LEAST aggressive setting. cb 6.0 once beat cb 3.0 by
    # 0.0001 -- inside the bootstrap noise -- while visibly degrading grammar
    # that repeat4 cannot see. Prefer the milder knob when the score cannot
    # tell them apart.
    # 0.0005, not 0.0015. The wider window was meant to stop cb 6.0 beating
    # cb 3.0 by 0.0001 while degrading grammar; it instead discarded a real
    # 0.0012 gain on M3 (cb 2.0 chosen over cb 3.0). Measured run-to-run noise
    # is ~0.001, so only differences well inside that count as ties.
    near = {k: v for k, v in ok.items() if top - v["bertscore"] <= 0.0005}
    # .get("kw", {}): checkpoint-selection entries carry no decoding kwargs,
    # only sweep entries do, and pick() is called on both.
    # Among statistical ties, prefer the setting that DISTORTS the distribution
    # least (low copy_bias, low repetition_penalty) but PROTECTS most
    # (no_repeat_ngram on). nrn is not an aggressive knob -- it cannot degrade
    # grammar the way a high copy bias can, it only forbids self-repetition.
    # M4's sweep picked plain sampling over rp1.05+nrn4 on a 0.0001 difference
    # and shipped repeat4 0.09 instead of 0.00 as a result.
    return min(near, key=lambda k: (near[k].get("kw", {}).get("copy_bias", 0.0),
                                    near[k].get("kw", {}).get("repetition_penalty", 1.0),
                                    -near[k].get("kw", {}).get("no_repeat_ngram", 0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-dev", type=int, default=250)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--quick", action="store_true", help="skip stages 2 and 3")
    ap.add_argument("--copy-only", action="store_true",
                    help="skip stages 1-2 and checkpoint selection, start from "
                         "the known winner and sweep copy_bias widely")
    ap.add_argument("--ckpt", default=None, help="force one checkpoint file")
    a = ap.parse_args()

    OUT = f"{ROOT}/{a.model}"
    device = "cuda"
    torch.set_float32_matmul_precision("high")
    tok = BPETokenizer.load(f"{ROOT}/shared/tokenizer/bpe_legal_32k.json")
    dp, dt = load_clean_dev(tok)
    keys = list(dp)[:a.n_dev]
    dev_p = {k: dp[k] for k in keys}
    dev_t = {k: dt[k] for k in keys}
    log(f"dev subset: {len(dev_p)} examples")
    sc = Scorer(device)
    results, t0 = {}, time.time()

    # --- 1. which checkpoint? val loss and generation quality differ ----------
    cands = []
    if a.ckpt:
        cands = [("forced", a.ckpt)]
    elif a.copy_only and os.path.exists(f"{OUT}/ckpt/soup.pt"):
        cands = [("soup3", f"{OUT}/ckpt/soup.pt")]   # already measured as the winner
    else:
        for n in ("best.pt", "final.pt"):
            if os.path.exists(f"{OUT}/ckpt/{n}"):
                cands.append((n[:-3], f"{OUT}/ckpt/{n}"))
        slots = [f"{OUT}/ckpt/slot{i}.pt" for i in range(3)
                 if os.path.exists(f"{OUT}/ckpt/slot{i}.pt")]
        if len(slots) >= 2:
            sd, n = soup(slots, device)
            torch.save({"model": sd,
                        "cfg": torch.load(slots[0], map_location="cpu",
                                          weights_only=False)["cfg"],
                        "step": -1}, f"{OUT}/ckpt/soup.pt")
            cands.append((f"soup{n}", f"{OUT}/ckpt/soup.pt"))
    assert cands, f"no checkpoints under {OUT}/ckpt"

    # Checkpoint selection uses mild sampling, NOT greedy: both architectures
    # loop under greedy at this scale, and comparing three checkpoints on
    # degenerate text compares the loops, not the models.
    CKPT_DEC = dict(temperature=0.8, top_p=0.9, repetition_penalty=1.05)
    if len(cands) == 1:
        log(f"--- checkpoint: {cands[0][0]} (selection skipped) ---")
    else:
        log(f"--- checkpoint selection ({CKPT_DEC}) ---")
    ck_scores = {}
    if len(cands) == 1:
        chosen = cands[0][0]
    else:
        for name, path in cands:
            model, cfg, kind, step, val = load_ckpt(path, device)
            g = gen_guaranteed(model, kind, tok, dev_p, device, a.batch, **CKPT_DEC)
            s = sc(g, dev_t)
            s["step"], s["val_loss"] = step, val
            ck_scores[name] = s
            report(f"ckpt:{name}", s)
            del model
            torch.cuda.empty_cache()
        chosen = pick(ck_scores)
    log(f"chosen checkpoint: {chosen}")
    model, cfg, kind, _, _ = load_ckpt(dict(cands)[chosen], device)

    # --- 2. decoding sweep, staged ------------------------------------------
    BASE_WINNER = dict(temperature=0.8, top_p=0.95, repetition_penalty=1.05,
                       no_repeat_ngram=4)
    if a.copy_only:
        log(f"--- copy-only mode, base {BASE_WINNER} ---")
        g = gen_guaranteed(model, kind, tok, dev_p, device, a.batch, **BASE_WINNER)
        results["base"] = sc(g, dev_t)
        results["base"]["kw"] = BASE_WINNER
        report("base", results["base"])
        best = "base"
    else:
        log("--- decoding sweep, stage 1: sampling shape ---")
    stage1 = [("greedy", dict(greedy=True)),
              ("t0.6_p0.90", dict(temperature=0.6, top_p=0.90)),
              ("t0.8_p0.90", dict(temperature=0.8, top_p=0.90)),
              ("t0.8_p0.95", dict(temperature=0.8, top_p=0.95)),
              ("t1.0_p0.95", dict(temperature=1.0, top_p=0.95))]
    if not a.copy_only:
        for tag, kw in stage1:
            g = gen_guaranteed(model, kind, tok, dev_p, device, a.batch, **kw)
            results[tag] = sc(g, dev_t)
            results[tag]["kw"] = kw
            report(tag, results[tag])
        best = pick(results)
        log(f"stage 1 winner: {best}")

    if not a.quick and not a.copy_only:
        # Looping is the dominant failure mode at this scale, so repetition
        # control gets a real sweep rather than one knob.
        log("--- stage 2: repetition control ---")
        for label, extra in (("rp1.05", dict(repetition_penalty=1.05)),
                             ("rp1.15", dict(repetition_penalty=1.15)),
                             ("nrn4", dict(no_repeat_ngram=4)),
                             ("rp1.05+nrn4", dict(repetition_penalty=1.05,
                                                  no_repeat_ngram=4))):
            kw = dict(results[best]["kw"], **extra)
            tag = f"{best}_{label}"
            g = gen_guaranteed(model, kind, tok, dev_p, device, a.batch, **kw)
            results[tag] = sc(g, dev_t)
            results[tag]["kw"] = kw
            report(tag, results[tag])
        best = pick(results)
        log(f"stage 2 winner: {best}")

    # Prompt-copy bias. The leaderboard's BLEU-4 of ~0.10 implied the leaders
    # lift terminology straight out of the prompt; measured at +0.0028 on M1
    # and +0.0025 on M2, so this runs in --copy-only mode too -- it is the
    # whole point of that mode.
    if not a.quick:
        log("--- stage 3: prompt-copy logit bias ---")
        # cb 2.0 was the edge of the first grid and still climbing, so the
        # range runs well past it until the fluency/repetition gates bite.
        for cb in (1.0, 2.0, 3.0, 4.0, 6.0):
            kw = dict(results[best]["kw"], copy_bias=cb)
            tag = f"{best}_cb{cb}"
            g = gen_guaranteed(model, kind, tok, dev_p, device, a.batch, **kw)
            results[tag] = sc(g, dev_t)
            results[tag]["kw"] = kw
            report(tag, results[tag])
        best = pick(results)
        log(f"stage 3 winner: {best}")

        # Does biasing toward generated tokens as well as prompt tokens help or
        # hurt? The original implementation did both without distinguishing.
        if results[best]["kw"].get("copy_bias"):
            kw = dict(results[best]["kw"], copy_scope="all")
            tag = f"{best}_scopeall"
            g = gen_guaranteed(model, kind, tok, dev_p, device, a.batch, **kw)
            results[tag] = sc(g, dev_t)
            results[tag]["kw"] = kw
            report(tag, results[tag])
            best = pick(results)
            log(f"stage 3b winner: {best}")

    # is the winner actually distinguishable from the runner-up?
    ok = sorted([k for k in results if admissible(results[k])],
                key=lambda k: -results[k]["bertscore"])
    if len(ok) > 1:
        d = results[ok[0]]["bertscore"] - results[ok[1]]["bertscore"]
        halfwidth = (results[ok[0]]["bs_hi"] - results[ok[0]]["bs_lo"]) / 2
        log(f"top by score: {ok[0]} beats {ok[1]} by {d:.4f}; bootstrap "
            f"half-width {halfwidth:.4f} -> "
            f"{'SIGNIFICANT' if d > halfwidth else 'WITHIN NOISE'}")
        if best != ok[0]:
            log(f"SELECTED {best} ({results[best]['bertscore']:.4f}) over "
                f"{ok[0]} ({results[ok[0]]['bertscore']:.4f}) -- tie within "
                f"0.0005, preferring the milder setting")

    json.dump({k: {kk: vv for kk, vv in v.items() if kk != "kw"}
               for k, v in {**ck_scores, **results}.items()},
              open(f"{OUT}/logs/sweep.json", "w"), indent=2)

    # --- 3. final submission -------------------------------------------------
    log(f"--- generating 1000 test continuations with {best} ---")
    test = json.load(open(f"{ROOT}/shared/test/test.json", encoding="utf-8"))
    kw = {k: v for k, v in results[best]["kw"].items()}
    pred = gen_guaranteed(model, kind, tok, test, device, a.batch, **kw)

    assert set(pred) == set(test), "ID MISMATCH against the official test file"
    wc = sorted(len(v["text"].split()) for v in pred.values())
    log(f"continuation words: min {wc[0]} p5 {wc[len(wc)//20]} "
        f"median {wc[len(wc)//2]} max {wc[-1]}")
    fl = np.mean([fluency(v["text"]) for v in pred.values()])
    rp = np.mean([repeat_rate(v["text"]) for v in pred.values()])
    log(f"test fluency: {fl:.1f} bigrams/128w | repeat4 {rp:.2f}")

    pj = f"{OUT}/gen/prediction.json"
    json.dump(pred, open(pj, "w", encoding="utf-8"), ensure_ascii=False)
    zp = f"{OUT}/gen/submission_{chosen}_{best}.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(pj, "prediction.json")
    log(f"WROTE {zp}")
    log(f"   expected dev BERTScore {results[best]['bertscore']:.4f} "
        f"[{results[best]['bs_lo']:.4f},{results[best]['bs_hi']:.4f}]")
    for k in list(pred)[:2]:
        log(f"   sample: {pred[k]['text'][:260]!r}")
    log(f"ALL DONE in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
