#!/usr/bin/env python3
"""CS-781 Phase 2 -- emission ensembling across the four Phase-1 architectures.

    python phase2_ensemble.py --specs M1_SC:full M2_LexArcED:full M3_LexArcL:full M4_LexArcP:full
    python phase2_ensemble.py --specs ... --crf --submit

WHY

Every run so far decodes from one model. The boundary diagnostic says the strict
score is capped by two separate things:

    overlap-F1 ~0.58   what the detector FINDS at all
    strict  -0.29      how much of that it loses on span boundaries

A CRF attacks the second. This attacks the first, which is the one that actually
bounds everything else -- strict F1 can never exceed overlap F1.

Four architectures were trained in Phase 1 for exactly this reason, and they fail
differently: M2's bidirectional encoder is strongest on JUDGE, M1_SC on COURT and
STAT, and their false positives are largely disjoint. Averaging the per-token tag
emissions before decoding is the NER analogue of the candidate pooling that was
worth +0.0063 in Phase 1, except it operates on logits rather than finished text,
so it needs no utility function and no extra generation pass.

Averaging happens at the EMISSION level, not over decoded spans. Decoded spans
cannot be averaged -- two models that disagree by one token produce two
incompatible spans and a vote between them is arbitrary. Logits over a shared
token sequence are directly comparable, because every model here uses the same
Phase-1 tokenizer and therefore the same tokenisation of every document.

Models are trained fresh inside this script rather than loaded, because the
training runs never persisted their fine-tuned weights, and at ~4 minutes each
that is cheaper than adding checkpointing everywhere.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch
import torch.nn.functional as F

BASE = os.environ.get("CS781_BASE", "/content/drive/MyDrive/cs781_capstone")
P1 = os.environ.get("CS781_P1", f"{BASE}/phase1")
P2 = os.environ.get("CS781_P2", f"{BASE}/phase2")
sys.path.insert(0, f"{P1}/shared/lib")
sys.path.insert(0, f"{P2}/code")
from cs781_bpe import BPETokenizer                                    # noqa: E402
from cs781_model import Config, LexArcD                               # noqa: E402
from phase2_data import (parse_gold, fetch, load_json, log,           # noqa: E402
                         n_tags_for, decode_tags)
from phase2_ner import (NERModel, build, batches, apply_method,       # noqa: E402
                        strict_macro_f1, validate, boundary_report)


def make(spec, labels, n_tags, use_crf, device):
    """'DIR:method' -> an untrained NERModel on the given Phase-1 trunk."""
    d, _, method = spec.partition(":")
    method = method or "full"
    ck = torch.load(f"{P1}/{d}/ckpt/best.pt", map_location="cpu", weights_only=False)
    cfg_d = ck["cfg"]
    if "n_enc_layer" in cfg_d:
        from cs781_model_ed import EDConfig, LexArcED
        cfg = EDConfig(**cfg_d)
        t = LexArcED(cfg)
        t.load_state_dict(ck["model"], strict=False)
        m = NERModel(t, cfg.d_model, n_tags, "ed", 0, True, use_crf)
    else:
        cfg = Config(**cfg_d)
        if getattr(cfg, "use_pointer", False):
            from cs781_model_ptr import LexArcP
            t = LexArcP(cfg)
        else:
            t = LexArcD(cfg)
        t.load_state_dict(ck["model"], strict=False)
        m = NERModel(t, cfg.d_model, n_tags, "dec", cfg.n_register, False, use_crf)
    apply_method(m, method, 8)
    return m.to(device), d, method


def class_weights(tr, n_tags, pow_, cap, device):
    freq = torch.zeros(n_tags)
    for c in tr:
        for t in c["tags"]:
            if t >= 0:
                freq[t] += 1
    w = (freq.sum() / freq.clamp(min=1)).pow(pow_)
    return (w / w.mean()).clamp(max=cap).to(device)


def train(model, tr, w, n_tags, a, device, name):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95), weight_decay=0.01)
    steps = a.epochs * max(1, math.ceil(len(tr) / a.batch))
    warm, step = max(1, int(0.1 * steps)), 0
    for ep in range(a.epochs):
        for idx, tag, pad, _ in batches(tr, a.batch, device, True, n_tags):
            lr = a.lr * (step / warm if step < warm else
                         0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, steps - warm))))
            for g in opt.param_groups:
                g["lr"] = lr
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(idx, pad)
                if model.crf is not None:
                    loss = model.crf.nll(logits.float(), tag, pad & (tag != -100))
                else:
                    loss = F.cross_entropy(logits.reshape(-1, n_tags).float(),
                                           tag.reshape(-1), weight=w, ignore_index=-100)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
        log(f"    {name} epoch {ep+1}/{a.epochs}  loss {loss.item():.4f}")
    return model


@torch.no_grad()
def emissions(model, data, device, bs, n_tags):
    """{doc: (n_tokens, n_tags)} log-probs, averaged over overlapping windows.

    LOG-PROBS, not raw logits. Models calibrate differently and a model with a
    larger logit scale would otherwise dominate the average regardless of how
    confident it actually is. log_softmax puts every model on the same scale.
    """
    model.eval()
    acc = {d: torch.zeros(next(c["n"] for c in data if c["doc"] == d), n_tags)
           for d in {c["doc"] for c in data}}
    cnt = {d: torch.zeros(acc[d].size(0), 1) for d in acc}
    for idx, _, pad, chunk in batches(data, bs, device, False, n_tags):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = model(idx, pad).float()
        lg = F.log_softmax(lg, -1).cpu()
        for i, c in enumerate(chunk):
            k = c.get("skip", 0)
            n = len(c["ids"]) - k
            acc[c["doc"]][c["start"]:c["start"] + n] += lg[i, k:k + n]
            cnt[c["doc"]][c["start"]:c["start"] + n] += 1
    model.train()
    return {d: acc[d] / cnt[d].clamp(min=1) for d in acc}


def decode_all(pool, data, labels, text, scheme, crf_model=None, device="cuda"):
    out = {}
    for d, a in pool.items():
        spans = next(c["spans"] for c in data if c["doc"] == d)
        if a.size(0) == 0:
            out[d] = []
            continue
        if crf_model is not None:
            tags = crf_model.crf.viterbi(a.unsqueeze(0).to(device))
        else:
            tags = a.argmax(-1).tolist()
        out[d] = [list(x) for x in decode_tags(tags, spans, labels, text[d], scheme)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", nargs="+", required=True,
                    help="DIR:method, e.g. M1_SC:full M2_LexArcED:full")
    ap.add_argument("--scheme", default="bio", choices=["bio", "bioes"])
    ap.add_argument("--crf", action="store_true")
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-pow", type=float, default=0.5)
    ap.add_argument("--weight-cap", type=float, default=20.0)
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--tag", default="ensemble")
    a = ap.parse_args()

    device = "cuda"
    torch.set_float32_matmul_precision("high")
    p = torch.cuda.get_device_properties(0)
    assert p.major * 10 + p.minor >= 80, f"{p.name}: bf16 emulated below SM80"
    log(f"GPU {p.name} | ensemble of {len(a.specs)}: {a.specs} | scheme={a.scheme}"
        f"{' +crf' if a.crf else ''}")

    with open(fetch("Train/labels.txt"), encoding="utf-8") as f:
        labels = [x.strip() for x in f if x.strip()]
    text, gold = {}, {}
    for fold in ("fold1", "fold2"):
        text.update({str(k): v for k, v in load_json(f"Train/{fold}-text.json").items()})
        gold.update(parse_gold(load_json(f"Train/{fold}-gold.json")))
    split = json.load(open(f"{P2}/split.json"))
    tok = BPETokenizer.load(f"{P1}/shared/tokenizer/bpe_legal_32k.json")
    label_to_id = {l: i for i, l in enumerate(labels)}
    n_tags = n_tags_for(labels, a.scheme)

    singles, ens = {s: [] for s in a.specs}, []
    # Held-out ENSEMBLE predictions, dumped so the post-processing and
    # rule stages can be tuned on CPU against this system rather than a
    # single-model proxy. Tuning against the proxy is what sent a
    # --replace submission that lost 0.068 on the leaderboard.
    cv_pred = {}
    for tr_fold, ev_fold in (("fold1", "fold2"), ("fold2", "fold1")):
        log(f"\n--- train {tr_fold} -> score {ev_fold} ---")
        tr = build(tok, split[tr_fold], text, gold, label_to_id, a.ctx, a.ctx // 2,
                   scheme=a.scheme)
        ev = build(tok, split[ev_fold], text, gold, label_to_id, a.ctx, a.ctx // 2,
                   scheme=a.scheme)
        truth = {d: [list(x) for x in gold.get(d, [])] for d in split[ev_fold]}
        w = class_weights(tr, n_tags, a.weight_pow, a.weight_cap, device)

        pool, last = None, None
        for spec in a.specs:
            m, d, meth = make(spec, labels, n_tags, a.crf, device)
            log(f"  {d} ({meth}):")
            train(m, tr, w, n_tags, a, device, d)
            em = emissions(m, ev, device, a.batch, n_tags)
            f1, _ = strict_macro_f1(
                decode_all(em, ev, labels, text, a.scheme, m if a.crf else None),
                truth, labels)
            singles[spec].append(f1)
            log(f"    single {d}: strict {f1:.4f}")
            pool = em if pool is None else {k: pool[k] + em[k] for k in pool}
            last = m if a.crf else None
            if not a.crf:
                del m
                torch.cuda.empty_cache()
        pool = {k: v / len(a.specs) for k, v in pool.items()}
        pred = decode_all(pool, ev, labels, text, a.scheme, last)
        f1, per = strict_macro_f1(pred, truth, labels)
        ov = sum(v[6] for v in per.values()) / len(labels)
        best_single = max(singles[s][-1] for s in a.specs)
        log(f"\n  ENSEMBLE strict {f1:.4f}  overlap {ov:.4f}   "
            f"(best single {best_single:.4f}, gain {f1-best_single:+.4f})")
        for l in labels:
            pr, rc, ff, tp, fp, fn, rf = per[l]
            log(f"    {l:11s} P={pr:.3f} R={rc:.3f} F1={ff:.3f} | overlap={rf:.3f}")
        boundary_report(pred, truth, text, labels, n_examples=4)
        cv_pred.update({d: [list(x) for x in v] for d, v in pred.items()})
        ens.append(f1)
        del last
        torch.cuda.empty_cache()

    cv = sum(ens) / len(ens)
    log(f"\n2-fold CV  ENSEMBLE {cv:.4f}")
    os.makedirs(f"{P2}/results", exist_ok=True)
    cvp = f"{P2}/results/cvpred_{a.tag}.json"
    json.dump(cv_pred, open(cvp, "w", encoding="utf-8"), ensure_ascii=False)
    log(f"wrote {cvp}  ({sum(len(v) for v in cv_pred.values())} spans)")
    for s in a.specs:
        log(f"           {s:22s} {sum(singles[s])/len(singles[s]):.4f}")

    if a.submit:
        log(f"\n--- final: every model refit on all 70 documents ---")
        all_docs = split["fold1"] + split["fold2"]
        tr = build(tok, all_docs, text, gold, label_to_id, a.ctx, a.ctx // 2,
                   scheme=a.scheme)
        w = class_weights(tr, n_tags, a.weight_pow, a.weight_cap, device)
        test_text = {str(k): v for k, v in load_json("Test/fold3-text.json").items()}
        te = build(tok, sorted(test_text), test_text, {}, label_to_id, a.ctx,
                   a.ctx // 2, with_tags=False, scheme=a.scheme)
        pool, last = None, None
        for spec in a.specs:
            m, d, meth = make(spec, labels, n_tags, a.crf, device)
            log(f"  {d} ({meth}):")
            train(m, tr, w, n_tags, a, device, d)
            em = emissions(m, te, device, a.batch, n_tags)
            pool = em if pool is None else {k: pool[k] + em[k] for k in pool}
            last = m if a.crf else None
            if not a.crf:
                del m
                torch.cuda.empty_cache()
        pool = {k: v / len(a.specs) for k, v in pool.items()}
        pred = decode_all(pool, te, labels, test_text, a.scheme, last)
        pred = {d: pred.get(d, []) for d in test_text}
        validate(pred, labels)
        n_ent = sum(len(v) for v in pred.values())
        got = {l for v in pred.values() for _, _, l in v}
        miss = [l for l in labels if l not in got]
        log(f"  {n_ent} entities ({n_ent/len(pred):.1f}/doc; training 119.9)")
        if miss:
            log(f"  WARNING: never predicted {miss} -- each costs 1/12")
        os.makedirs(f"{P2}/submissions", exist_ok=True)
        pj = f"{P2}/submissions/prediction_{a.tag}.json"
        json.dump(pred, open(pj, "w", encoding="utf-8"), ensure_ascii=False)
        import zipfile
        zp = f"{P2}/submissions/submission_{a.tag}.zip"
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(pj, "prediction.json")
        log(f"  WROTE {zp}")

    os.makedirs(f"{P2}/results", exist_ok=True)
    json.dump({"tag": a.tag, "specs": a.specs, "scheme": a.scheme, "crf": a.crf,
               "ensemble_cv": cv, "folds": ens,
               "singles": {s: singles[s] for s in a.specs}},
              open(f"{P2}/results/{a.tag}.json", "w"), indent=2)
    log(f"wrote {P2}/results/{a.tag}.json")


if __name__ == "__main__":
    main()
