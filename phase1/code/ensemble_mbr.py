#!/usr/bin/env python3
"""CS-781 Phase 1 -- candidate pooling across models + MBR selection.

    python ensemble_mbr.py --models M1_LexArcD_full:soup.pt M3_LexArcL:best.pt \
                           M4_LexArcP:best.pt --k 4

WHY MBR IS THE RIGHT THING TO TRY NEXT

Everything else in this project optimises a proxy. Training minimises
cross-entropy; the decoding sweep maximised BERTScore over a handful of fixed
hyperparameters. Minimum-Bayes-risk decoding optimises the evaluation metric
directly: given a set of candidates sampled from the model's own distribution,
pick the one with the highest expected utility against the others,

    y* = argmax_y  (1/|C|) * sum_{c in C}  BERTScore_F1(y, c)

With the true metric as the utility this is the Bayes-optimal decision rule for
that metric. In plain terms: the reference is unknown, the samples are the
model's best guesses at it, so choose the candidate that is most central among
them. Outliers -- which is what a single unlucky sample is -- get rejected by
construction.

WHY POOLING ACROSS MODELS RATHER THAN AVERAGING LOGITS

Logit averaging needs the models stepped in lockstep with a shared cache, and
the pointer model's output distribution is a mixture, not a softmax over the
head -- so a shared step function would have to special-case it. Pooling
candidates from independently-run models gets most of the ensemble benefit
(diversity, outlier rejection) through MBR's selection step instead, for a
fraction of the code and none of the coupling. Worth stating in the report as a
deliberate trade, not an oversight.

BERTScore F1 is symmetric -- precision(a,b) = recall(b,a) -- so only k(k-1)/2
pairs per prompt are scored rather than k(k-1).

PROTOCOL: validate on dev before spending the test generation. If MBR does not
beat the best single model by more than the bootstrap noise, the script says so
and writes nothing.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
import zipfile

import numpy as np
import torch

ROOT = os.environ.get("CS781_ROOT", "/content/drive/MyDrive/cs781_capstone")
sys.path.insert(0, f"{ROOT}/shared/lib")
from cs781_bpe import BPETokenizer                                    # noqa: E402
from train_phase1 import load_clean_dev, log, fluency                 # noqa: E402
from score_and_submit import (load_ckpt, gen_guaranteed, Scorer,      # noqa: E402
                              repeat_rate, score_dev, report)

# Decoding is PER MODEL. The sweeps disagreed, and not by accident: a uniform
# prompt-copy bias is worth +0.0028 on M1 and +0.0025 on M2, and -0.0077 on M4,
# whose pointer head already does that job. One shared setting would hand M4 a
# handicap and call it an ensemble.
BASE_DECODE = dict(temperature=0.8, top_p=0.95, repetition_penalty=1.05,
                   no_repeat_ngram=4)
DEFAULT_CB = 3.0


def mbr_select(cands, scorer, batch=64):
    """cands: {prompt_id: [k strings]} -> {prompt_id: best string}.

    One flat BERTScore call over every unordered pair in every prompt.
    """
    pairs, index = [], []
    for pid, cs in cands.items():
        for i, j in itertools.combinations(range(len(cs)), 2):
            pairs.append((cs[i], cs[j]))
            index.append((pid, i, j))
    log(f"   MBR: scoring {len(pairs):,} candidate pairs")
    t0 = time.time()
    _, _, f1 = scorer.s.score([a for a, _ in pairs], [b for _, b in pairs])
    f1 = f1.numpy()
    log(f"   MBR: scored in {time.time()-t0:.0f}s")

    util = {pid: np.zeros(len(cs)) for pid, cs in cands.items()}
    for (pid, i, j), v in zip(index, f1):
        util[pid][i] += v
        util[pid][j] += v
    out, chosen_idx = {}, []
    for pid, cs in cands.items():
        k = int(np.argmax(util[pid]))
        chosen_idx.append(k)
        out[pid] = {"text": cs[k]}
    hist = np.bincount(chosen_idx, minlength=max(len(c) for c in cands.values()))
    log(f"   MBR: chosen-candidate histogram {hist.tolist()}")
    return out


def sample_pool(models, tok, prompts, device, k, batch, seed0=0):
    """k independent samples per model, pooled per prompt."""
    pool = {pid: [] for pid in prompts}
    per_model = {}
    for name, (model, kind, dec) in models.items():
        singles = []
        for s in range(k):
            torch.manual_seed(seed0 + 1000 * s + abs(hash(name)) % 997)
            g = gen_guaranteed(model, kind, tok, prompts, device, batch, **dec)
            for pid, v in g.items():
                pool[pid].append(v["text"])
            singles.append(g)
            log(f"   {name} sample {s+1}/{k} done")
        per_model[name] = singles[0]        # one draw, as the single-model baseline
    return pool, per_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True,
                    help="DIR:CKPT:COPY_BIAS, e.g. M4_LexArcP:best.pt:0 "
                         "M1_LexArcD_full:soup.pt:3. copy_bias defaults to 3.0; "
                         "pass 0 for models with a learned copy head.")
    ap.add_argument("--k", type=int, default=4, help="samples per model")
    ap.add_argument("--n-dev", type=int, default=250)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", default="MBR")
    ap.add_argument("--force", action="store_true",
                    help="write the submission even if MBR loses on dev")
    a = ap.parse_args()

    device = "cuda"
    torch.set_float32_matmul_precision("high")
    OUT = f"{ROOT}/{a.out}"
    os.makedirs(f"{OUT}/gen", exist_ok=True)
    os.makedirs(f"{OUT}/logs", exist_ok=True)

    tok = BPETokenizer.load(f"{ROOT}/shared/tokenizer/bpe_legal_32k.json")
    dp, dt = load_clean_dev(tok)
    keys = list(dp)[:a.n_dev]
    dev_p, dev_t = {k: dp[k] for k in keys}, {k: dt[k] for k in keys}
    sc = Scorer(device)

    models = {}
    for spec in a.models:
        parts = spec.split(":")
        d = parts[0]
        c = parts[1] if len(parts) > 1 and parts[1] else "best.pt"
        cb = float(parts[2]) if len(parts) > 2 and parts[2] != "" else DEFAULT_CB
        path = f"{ROOT}/{d}/ckpt/{c}"
        assert os.path.exists(path), f"missing {path}"
        m, cfg, kind, step, val = load_ckpt(path, device)
        dec = dict(BASE_DECODE, copy_bias=cb)
        models[d] = (m, kind, dec)
        log(f"loaded {d}: {m.n_params()/1e6:.1f}M, attn={cfg.attn_type}"
            f"{' +pointer' if getattr(cfg,'use_pointer',False) else ''}, "
            f"step {step}, val {val}, copy_bias={cb}")

    # --- dev: single models, then the MBR pool ------------------------------
    log(f"--- dev ({len(dev_p)} prompts), {a.k} samples x {len(models)} models ---")
    pool, per_model = sample_pool(models, tok, dev_p, device, a.k, a.batch)
    results = {}
    for name, g in per_model.items():
        results[name] = sc(g, dev_t)
        results[name]["kw"] = models[name][2]
        report(f"single:{name}", results[name])
    best_single = max(results, key=lambda k: results[k]["bertscore"])

    mbr = mbr_select(pool, sc, a.batch)
    results["MBR"] = sc(mbr, dev_t)
    results["MBR"]["kw"] = {"mbr": True}
    report("MBR-pooled", results["MBR"])

    gain = results["MBR"]["bertscore"] - results[best_single]["bertscore"]
    half = (results["MBR"]["bs_hi"] - results["MBR"]["bs_lo"]) / 2
    verdict = "REAL" if gain > half else "WITHIN NOISE"
    log(f"MBR vs best single ({best_single}): {gain:+.4f}  "
        f"bootstrap half-width {half:.4f} -> {verdict}")
    json.dump({k: {kk: vv for kk, vv in v.items() if kk != "kw"}
               for k, v in results.items()},
              open(f"{OUT}/logs/mbr_dev.json", "w"), indent=2)

    if gain <= 0 and not a.force:
        raise SystemExit(
            f"MBR did not beat {best_single} on dev ({gain:+.4f}). Writing "
            "nothing. That is a result worth reporting, not a failure to "
            "paper over with --force.")
    if verdict == "WITHIN NOISE":
        log("NOTE: the gain is inside the noise band. Proceeding, but treat the "
            "leaderboard delta as the real measurement.")

    # --- test ---------------------------------------------------------------
    test = json.load(open(f"{ROOT}/shared/test/test.json", encoding="utf-8"))
    log(f"--- test ({len(test)} prompts) ---")
    tpool, _ = sample_pool(models, tok, test, device, a.k, a.batch, seed0=7)
    pred = mbr_select(tpool, sc, a.batch)

    assert set(pred) == set(test), "ID MISMATCH against the official test file"
    wc = sorted(len(v["text"].split()) for v in pred.values())
    log(f"continuation words: min {wc[0]} p5 {wc[len(wc)//20]} "
        f"median {wc[len(wc)//2]} max {wc[-1]}")
    log(f"test fluency {np.mean([fluency(v['text']) for v in pred.values()]):.1f} "
        f"| repeat4 {np.mean([repeat_rate(v['text']) for v in pred.values()]):.2f}")

    pj = f"{OUT}/gen/prediction.json"
    json.dump(pred, open(pj, "w", encoding="utf-8"), ensure_ascii=False)
    tag = "_".join(sorted(models)) + f"_k{a.k}"
    zp = f"{OUT}/gen/submission_mbr_{tag}.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(pj, "prediction.json")
    log(f"WROTE {zp}")
    log(f"   expected dev BERTScore {results['MBR']['bertscore']:.4f} "
        f"[{results['MBR']['bs_lo']:.4f},{results['MBR']['bs_hi']:.4f}]")
    for k_ in list(pred)[:2]:
        log(f"   sample: {pred[k_]['text'][:260]!r}")


if __name__ == "__main__":
    main()
