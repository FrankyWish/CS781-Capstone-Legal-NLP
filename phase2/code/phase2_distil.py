#!/usr/bin/env python3
"""CS-781 Phase 2 -- knowledge distillation into a half-depth student. COMPULSORY.

    python phase2_distil.py --model M1_SC --scheme bioes --epochs 10
    python phase2_distil.py --model M1_SC --scheme bioes --epochs 10 --submit

WHAT SS8.2.2 ASKS FOR

"Use a larger Phase-1/Phase-2 model as the teacher and train a smaller student
model to transfer useful task-specific knowledge. Investigate whether the student
can retain useful entity-level information while reducing the number of
parameters, memory requirements, and computational cost."

So the deliverable is not just a number -- it is a trade-off curve. This script
reports, for teacher and student on identical data: strict macro-F1, overlap
macro-F1, parameter count, and measured inference throughput.

WHERE THE STUDENT COMES FROM

The phase forbids any pretrained weights other than our own Phase-1 models, so
the student cannot be an off-the-shelf small model and training one from scratch
on 70 documents would be hopeless. It is therefore built the way DistilBERT's
was: KEEP EVERY SECOND LAYER of the fine-tuned teacher. Half the depth, the
teacher's embeddings and final norm, and layer i of the student initialised from
layer 2i of the teacher. That is a smaller model derived entirely from our own
Phase-1 trunk, and it starts from a function that already partly solves the task
rather than from noise.

THE OBJECTIVE

    L = alpha * T^2 * KL(student/T || teacher/T)  +  (1 - alpha) * CE(student, gold)

The T^2 factor restores the gradient magnitude that dividing logits by T removes
(Hinton et al. 2015, SS2.1). The soft targets carry what the hard labels cannot:
that a token the teacher scores 0.6 B-APP / 0.3 B-RESP is a genuinely ambiguous
party mention, which on 70 documents is most of the signal the student can get.

Masked positions (-100, from header prepending) are excluded from BOTH terms;
including them in the KL would train the student to imitate the teacher on
padding, which is free loss reduction and no learning.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import replace

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
                         n_tags_for)
from phase2_ner import (NERModel, build, batches, predict,            # noqa: E402
                        strict_macro_f1, validate, boundary_report)


def make_student(teacher, cfg, n_tags, keep_every=2):
    """Half-depth copy of the teacher: student layer i <- teacher layer i*keep."""
    cfg_s = replace(cfg, n_layer=max(1, cfg.n_layer // keep_every))
    trunk = LexArcD(cfg_s)
    t = teacher.trunk
    trunk.emb.load_state_dict(t.emb.state_dict())
    trunk.norm.load_state_dict(t.norm.state_dict())
    if getattr(t, "reg", None) is not None and trunk.reg is not None:
        trunk.reg.data.copy_(t.reg.data)
    src = list(range(0, cfg.n_layer, keep_every))[:cfg_s.n_layer]
    for i, j in enumerate(src):
        trunk.blocks[i].load_state_dict(t.blocks[j].state_dict())
    log(f"  student keeps layers {src} of {cfg.n_layer}")
    s = NERModel(trunk, cfg.d_model, n_tags, "dec", cfg_s.n_register, False)
    s.head.load_state_dict(teacher.head.state_dict())
    return s


@torch.no_grad()
def throughput(model, data, device, bs, n_tags, seconds=8.0):
    """Tokens per second at inference -- the efficiency half of the deliverable."""
    model.eval()
    n, t0 = 0, time.time()
    while time.time() - t0 < seconds:
        for idx, _, pad, _ in batches(data, bs, device, False, n_tags):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                model(idx, pad)
            n += int(pad.sum())
            if time.time() - t0 >= seconds:
                break
    torch.cuda.synchronize()
    model.train()
    return n / (time.time() - t0)


def fit(model, tr, w, n_tags, epochs, lr, batch, device, teacher=None,
        alpha=0.7, temp=2.0, label=""):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    steps = epochs * max(1, math.ceil(len(tr) / batch))
    warm, step = max(1, int(0.1 * steps)), 0
    for ep in range(epochs):
        for idx, tag, pad, _ in batches(tr, batch, device, True, n_tags):
            cur = lr * (step / warm if step < warm else
                        0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, steps - warm))))
            for g in opt.param_groups:
                g["lr"] = cur
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(idx, pad)
                loss = F.cross_entropy(logits.reshape(-1, n_tags).float(),
                                       tag.reshape(-1), weight=w, ignore_index=-100)
                if teacher is not None:
                    with torch.no_grad():
                        tl = teacher(idx, pad)
                    keep = (tag != -100).reshape(-1)        # never imitate on masks
                    sl = logits.reshape(-1, n_tags)[keep].float()
                    tt = tl.reshape(-1, n_tags)[keep].float()
                    kd = F.kl_div(F.log_softmax(sl / temp, -1),
                                  F.log_softmax(tt / temp, -1),
                                  reduction="batchmean", log_target=True)
                    loss = alpha * (temp ** 2) * kd + (1 - alpha) * loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
        log(f"    {label} epoch {ep+1}/{epochs}  loss {loss.item():.4f}")
    return model


def class_weights(tr, n_tags, pow_, cap, device):
    freq = torch.zeros(n_tags)
    for c in tr:
        for t in c["tags"]:
            if t >= 0:
                freq[t] += 1
    w = (freq.sum() / freq.clamp(min=1)).pow(pow_)
    return (w / w.mean()).clamp(max=cap).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="M1_SC")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--scheme", default="bioes", choices=["bio", "bioes"])
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--alpha", type=float, default=0.7, help="weight on the KD term")
    ap.add_argument("--temp", type=float, default=2.0)
    ap.add_argument("--keep", type=int, default=2, help="keep every Nth layer")
    ap.add_argument("--header", type=int, default=0)
    ap.add_argument("--weight-pow", type=float, default=0.5)
    ap.add_argument("--weight-cap", type=float, default=20.0)
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--tag", default="distil")
    a = ap.parse_args()

    device = "cuda"
    torch.set_float32_matmul_precision("high")
    p = torch.cuda.get_device_properties(0)
    assert p.major * 10 + p.minor >= 80, f"{p.name}: bf16 emulated below SM80"
    log(f"GPU {p.name} | teacher {a.model}/{a.ckpt} | scheme={a.scheme} "
        f"| keep every {a.keep} layers")

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

    ck = torch.load(f"{P1}/{a.model}/ckpt/{a.ckpt}", map_location="cpu",
                    weights_only=False)
    cfg = Config(**ck["cfg"])

    def fresh_teacher():
        t = LexArcD(cfg)
        t.load_state_dict(ck["model"], strict=False)
        return NERModel(t, cfg.d_model, n_tags, "dec", cfg.n_register, False)

    rows, t_scores, s_scores = [], [], []
    for tr_fold, ev_fold in (("fold1", "fold2"), ("fold2", "fold1")):
        log(f"\n--- train {tr_fold} -> score {ev_fold} ---")
        tr = build(tok, split[tr_fold], text, gold, label_to_id, a.ctx,
                   a.ctx // 2, header=a.header, scheme=a.scheme)
        ev = build(tok, split[ev_fold], text, gold, label_to_id, a.ctx,
                   a.ctx // 2, header=a.header, scheme=a.scheme)
        truth = {d: [list(x) for x in gold.get(d, [])] for d in split[ev_fold]}
        w = class_weights(tr, n_tags, a.weight_pow, a.weight_cap, device)

        teacher = fresh_teacher().to(device)
        for q in teacher.parameters():
            q.requires_grad_(True)
        log("  teacher:")
        fit(teacher, tr, w, n_tags, a.epochs, a.lr, a.batch, device, label="teacher")
        teacher.eval()
        tf1, tper = strict_macro_f1(
            predict(teacher, ev, labels, text, device, a.batch, n_tags, a.scheme),
            truth, labels)

        student = make_student(teacher, cfg, n_tags, a.keep).to(device)
        for q in student.parameters():
            q.requires_grad_(True)
        log("  student (distilled):")
        fit(student, tr, w, n_tags, a.epochs, a.lr, a.batch, device,
            teacher=teacher, alpha=a.alpha, temp=a.temp, label="student")
        spred = predict(student, ev, labels, text, device, a.batch, n_tags, a.scheme)
        sf1, sper = strict_macro_f1(spred, truth, labels)

        tp_n = sum(q.numel() for q in teacher.parameters())
        st_n = sum(q.numel() for q in student.parameters())
        t_tps = throughput(teacher, ev, device, a.batch, n_tags)
        s_tps = throughput(student, ev, device, a.batch, n_tags)
        t_ov = sum(v[6] for v in tper.values()) / len(labels)
        s_ov = sum(v[6] for v in sper.values()) / len(labels)
        log(f"\n  teacher  strict {tf1:.4f}  overlap {t_ov:.4f}  "
            f"{tp_n/1e6:.1f}M params  {t_tps:,.0f} tok/s")
        log(f"  student  strict {sf1:.4f}  overlap {s_ov:.4f}  "
            f"{st_n/1e6:.1f}M params  {s_tps:,.0f} tok/s")
        log(f"  retained {100*sf1/max(tf1,1e-9):.1f}% of strict F1 at "
            f"{100*st_n/tp_n:.0f}% of the parameters and "
            f"{s_tps/max(t_tps,1e-9):.2f}x the throughput")
        boundary_report(spred, truth, text, labels, n_examples=3)
        rows.append(dict(fold=ev_fold, teacher_f1=tf1, student_f1=sf1,
                         teacher_overlap=t_ov, student_overlap=s_ov,
                         teacher_params=tp_n, student_params=st_n,
                         teacher_tok_s=t_tps, student_tok_s=s_tps))
        t_scores.append(tf1)
        s_scores.append(sf1)
        del teacher, student
        torch.cuda.empty_cache()

    tcv, scv = sum(t_scores) / 2, sum(s_scores) / 2
    log(f"\n2-fold CV  teacher {tcv:.4f}  student {scv:.4f}  "
        f"({100*scv/max(tcv,1e-9):.1f}% retained)")

    if a.submit:
        log("\n--- final: teacher then student on all 70 documents ---")
        all_docs = split["fold1"] + split["fold2"]
        tr = build(tok, all_docs, text, gold, label_to_id, a.ctx, a.ctx // 2,
                   header=a.header, scheme=a.scheme)
        w = class_weights(tr, n_tags, a.weight_pow, a.weight_cap, device)
        teacher = fresh_teacher().to(device)
        fit(teacher, tr, w, n_tags, a.epochs, a.lr, a.batch, device, label="teacher")
        teacher.eval()
        student = make_student(teacher, cfg, n_tags, a.keep).to(device)
        fit(student, tr, w, n_tags, a.epochs, a.lr, a.batch, device,
            teacher=teacher, alpha=a.alpha, temp=a.temp, label="student")
        test_text = {str(k): v for k, v in load_json("Test/fold3-text.json").items()}
        te = build(tok, sorted(test_text), test_text, {}, label_to_id, a.ctx,
                   a.ctx // 2, with_tags=False, header=a.header, scheme=a.scheme)
        pred = predict(student, te, labels, test_text, device, a.batch, n_tags, a.scheme)
        pred = {d: pred.get(d, []) for d in test_text}
        validate(pred, labels)
        os.makedirs(f"{P2}/submissions", exist_ok=True)
        pj = f"{P2}/submissions/prediction_{a.tag}.json"
        json.dump(pred, open(pj, "w", encoding="utf-8"), ensure_ascii=False)
        import zipfile
        zp = f"{P2}/submissions/submission_{a.tag}.zip"
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(pj, "prediction.json")
        log(f"  {sum(len(v) for v in pred.values())} entities; WROTE {zp}")

    os.makedirs(f"{P2}/results", exist_ok=True)
    json.dump({"tag": a.tag, "teacher": a.model, "scheme": a.scheme,
               "keep_every": a.keep, "alpha": a.alpha, "temp": a.temp,
               "teacher_cv": tcv, "student_cv": scv, "folds": rows},
              open(f"{P2}/results/{a.tag}.json", "w"), indent=2)
    log(f"wrote {P2}/results/{a.tag}.json")


if __name__ == "__main__":
    main()
