#!/usr/bin/env python3
"""CS-781 Phase 3 -- run saved fold checkpoints on new documents (teacher labels).

    python phase3_predict.py --runs lora_M4 lora_M4_aux rslora_M1 lorma_M3 \
        --docs $CS781_P3/external/marro.jsonl $CS781_P3/external/kalamkar_mask.jsonl \
        --merge $CS781_P3/teacher_push.pt --out $CS781_P3/teacher_ext.pt

Every run trained with checkpoints saved (all adapter runs; the full fine-tunes
used --no-ckpt) has its 5 fold models in ckpt/<tag>/fold<k>.pt. This loads each,
predicts the given documents, and averages log-probabilities over folds and
runs. With --merge, the result is added to an existing teacher file (out-of-fold
+ test logits from phase3_ens.py --dump-teacher), so one file feeds
`phase3_rr.py --kd-logits` for gold, test AND external documents.

Why: the external corpora carry coarse labels ("Argument", "Precedent").
phase3_rr.py masks the teacher's distribution to the roles each coarse label
allows, turning them into fine-grained soft labels in this task's 13-role scheme.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from phase3_data import P3, LABELS, log  # noqa: E402
import phase3_rr as rr                   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--docs", nargs="+", required=True, help="jsonl files (phase3_external format)")
    ap.add_argument("--merge", default=None, help="existing teacher .pt to extend")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    docs = {}
    for p in a.docs:
        docs.update(rr.load_jsonl_docs(p, "ext"))     # same "ext:<id>" keys as training
    ids = sorted(docs)
    log(f"  {len(ids)} documents, {sum(len(docs[d]['text']) for d in ids)} sentences")

    acc = {d: torch.zeros(len(docs[d]["text"]), len(LABELS)) for d in ids}
    n_models = 0
    toks = None
    for tag in a.runs:
        meta_p = f"{P3}/results/{tag}/meta.json"
        cks = sorted(f for f in os.listdir(f"{P3}/ckpt/{tag}")) if os.path.isdir(f"{P3}/ckpt/{tag}") else []
        if not os.path.exists(meta_p) or not cks:
            log(f"  skip {tag}: no meta.json or no saved checkpoints")
            continue
        args = SimpleNamespace(**json.load(open(meta_p))["args"])
        # runs from before a flag existed have no entry for it
        for k, v in dict(grad_ckpt=False, trunk_path=None, shift_aux=0.0, keep_every=1).items():
            if not hasattr(args, k):
                setattr(args, k, v)
        if toks is None or toks_key != (args.sent_max,):
            toks = rr.tokenize_docs(docs, args)
            toks_key = (args.sent_max,)
        for f in cks:
            t0 = time.time()
            model, _, _, _, _ = rr.build_model(args, device)
            state = torch.load(f"{P3}/ckpt/{tag}/{f}", map_location="cpu", weights_only=False)["state"]
            missing, unexpected = model.load_state_dict(
                {k: v.float() if v.is_floating_point() else v for k, v in state.items()}, strict=False)
            bad = [k for k in unexpected]
            assert not bad, f"{tag}/{f}: unexpected keys {bad[:3]}"
            em = rr.predict_docs(model, ids, toks, docs, args, device)
            for d in ids:
                acc[d] += F.log_softmax(em[d].float(), -1)
            n_models += 1
            log(f"  {tag}/{f}: done ({time.time()-t0:.0f}s)")
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    assert n_models, "no checkpoints were usable"
    teacher = {d: (v / n_models) for d, v in acc.items()}
    if a.merge:
        base = torch.load(a.merge, weights_only=False)
        clash = set(base) & set(teacher)
        assert not clash, f"{len(clash)} documents already in {a.merge}"
        base.update(teacher)
        teacher = base
    torch.save(teacher, a.out)
    log(f"  averaged {n_models} fold models; teacher for {len(teacher)} documents -> {a.out}")


if __name__ == "__main__":
    main()
