#!/usr/bin/env python3
"""CS-781 Phase 6 -- a Phase-1 trunk fine-tuned to choose sentences for BERTScore.

    python phase6_nsel.py --procs 6 labels                 # cache/nsel_labels_{dev,fit}.pkl
    python phase6_nsel.py --procs 6 gb                     # results/bsel3_f_model.pkl
    python phase6_nsel.py --procs 6 train --tag nsel_m1 --model M1_SC
    python phase6_nsel.py --procs 6 sweep --members nsel_m1,nsel_m2

`bsel` moved the score because its labels are the metric's own.  What limits it now is
what it can see: 27 hand features.  The greedy oracle over the same sentences reaches
0.885 on training documents against 0.855 for the scorer, so the room is in choosing,
and the one thing still unused for choosing is a model that has read the legal corpus.

  labels  y_f and y_g from phase6_bsel, cached on Drive this time, per document.
  gb      HistGB on every fit document, no word-level score.  bsel2 changed the data and
          the features together; this separates them.
  train   the top blocks of a Phase-1 trunk plus a small head.  Sentences are read in
          document order in chunks of the trunk's trained length, each sentence is the
          mean of its tokens, and the head also sees the chunk and the 27 hand features.
          M2's encoder is bidirectional, M1 is causal, so the two members disagree usefully.
  sweep   blends and a cross-fitted stack, chosen on the half of dev the sweep may read
          and reported on the other half with a paired standard error, because
          bsel2_f's +0.0025 on dev came back as -0.0007 on test.

Nothing neural except the Phase-1 trunks runs at test time; roberta-large only ever sees
training documents.  Submissions are kind='extractive': the text is the document's own.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import re
import shutil
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as TF

import phase6_bsel as B
import phase6_data as D
import phase6_sel as S
import phase6_sum as SM
import phase6_win as W

P6 = D.P6
RES = f"{P6}/results"
CACHE = f"{P6}/cache"
log = S.log

CITE = re.compile(r"[(\[]?\b(?:A\.?\s?I\.?\s?R\.?\s*\(?\d{4}\)?|\d{4}\s*[)\]]?\s*\d*\s*"
                  r"(?:S\.?\s?C\.?\s?R\.?|S\.?\s?C\.?\s?C\.?)|I\.?\s?L\.?\s?R\.?)"
                  r"\s*[\w.\s]{0,20}?\d+[)\]]?")


def clean(s):
    """Law-report citations: tokens no headnote aligns with, in a 510-token budget."""
    t = " ".join(CITE.sub(" ", s).split())
    return t if D.nwords(t) >= 4 else s


# -------------------------------------------------------------------- labels --
def labs_path(split):
    return f"{CACHE}/nsel_labels_{split}.pkl"


def load_labs(split):
    p = labs_path(split)
    if not os.path.exists(p):
        return {}
    with open(p, "rb") as f:
        return pickle.load(f)


def save_labs(split, lab):
    tmp = labs_path(split) + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(lab, f)
    os.replace(tmp, labs_path(split))


def doc_labels(ss, summ):
    """phase6_bsel.cmd_fit's two labels for one document: y_f in document z-score units,
    y_g in [0, 1].  None when the reference has no tokens."""
    E, seg, n = B.embed(B.enc(ss))
    R = B.embed(B.enc([summ]))[0]
    if not R.shape[0]:
        return None
    p, C = B.tables(E, seg, len(ss), R)
    pre, rec = p / n.clamp(min=1).float(), C.mean(1)
    f = (2 * pre * rec / (pre + rec).clamp(min=1e-6)).cpu().numpy()
    order, best = B.greedy([p], [C], n, n, min(B.WINDOW, R.shape[0]))
    g = np.zeros(len(ss), dtype=np.float32)
    for j, i in enumerate(order):
        g[i] = 1 - 0.5 * j / len(order)
    return ((f - f.mean()) / max(float(f.std()), 1e-6)).astype(np.float32), g, float(best)


def cmd_labels(a):
    os.makedirs(CACHE, exist_ok=True)
    S.load_model()
    rng = np.random.default_rng(D.SEED)
    work = []
    for split in ("dev", "fit"):            # dev first: the sweep needs all of it
        lab = load_labs(split)
        recs = [r for r in D.load(split) if r["id"] not in lab]
        recs = [recs[i] for i in rng.permutation(len(recs))]
        rows = S._map(S._pred_rows, recs, a.procs)      # workers first: no fork once on the GPU
        work.append((split, lab, rows, recs))
        log(f"  {split}: {len(lab)} labelled already, {len(recs)} to do")
    t0 = last = time.time()
    for split, lab, rows, recs in work:
        for k, ((rid, ss, _, _), r) in enumerate(zip(rows, recs)):
            if time.time() - t0 > a.budget_min * 60:
                log(f"  time budget reached at {k}/{len(rows)} {split} docs")
                break
            lab[rid] = None if len(ss) > a.max_sents else doc_labels(ss, r["summ"])
            if time.time() - last > 600:
                save_labs(split, lab)
                last = time.time()
                log(f"  {split} {k}/{len(rows)}, {(time.time() - t0) / 60:.1f} min")
        save_labs(split, lab)
        got = [v for v in lab.values() if v is not None]
        log(f"  {split}: {len(got)} docs labelled, surrogate oracle F "
            f"{np.mean([v[2] for v in got]):.4f}")
    with open(f"{CACHE}/nsel_labels.done", "w") as f:
        f.write("done")


def cmd_gb(a):
    from sklearn.ensemble import HistGradientBoostingRegressor
    S.load_model()
    lab = load_labs("fit")
    recs = [r for r in D.load("fit") if lab.get(r["id"]) is not None]
    rows = S._map(S._pred_rows, recs, a.procs)
    X = np.concatenate([r[2] for r in rows])
    y = 0.05 * np.concatenate([lab[r[0]][0] for r in rows])
    m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, min_samples_leaf=40,
                                      l2_regularization=1.0, early_stopping=True,
                                      validation_fraction=0.1, random_state=D.SEED)
    m.fit(X, y)
    with open(f"{RES}/bsel3_f_model.pkl", "wb") as f:
        pickle.dump(dict(model=m, docs=len(rows)), f)
    log(f"  wrote {RES}/bsel3_f_model.pkl ({X.shape[0]} sentences, {len(rows)} docs, "
        f"{m.n_iter_} trees)")


# -------------------------------------------------------------------- chunks --
def _enc_doc(ss):
    return [SM._enc(" " + s) for s in ss]


def tokenise(rows, procs):
    docs = [r[1] for r in rows]
    if procs > 1:
        from multiprocessing import Pool
        with Pool(procs, initializer=SM._enc_init, initargs=(SM.TOKJSON,)) as pool:
            return pool.map(_enc_doc, docs, chunksize=16)
    SM._enc_init(SM.TOKJSON)
    return [_enc_doc(d) for d in docs]


def chunks(doc, tk, T):
    """[(doc index, tokens, [(sentence, start, end)])]: consecutive sentences, at most T
    tokens each.  A sentence longer than T keeps its first T tokens."""
    out, cur, sp = [], [], []
    for i, t in enumerate(tk):
        t = t[:T]
        if not len(t):
            continue
        if cur and sum(map(len, cur)) + len(t) > T:
            out.append((doc, np.concatenate(cur), sp))
            cur, sp = [], []
        at = sum(map(len, cur))
        sp.append((i, at, at + len(t)))
        cur.append(t)
    if cur:
        out.append((doc, np.concatenate(cur), sp))
    return out


def collate(items, T, eos, Fs, Ys=None):
    """Left-padded to exactly T, as the summariser's prompts were."""
    idx = np.full((len(items), T), eos, dtype=np.int64)
    seg = np.full((len(items), T), -1, dtype=np.int64)
    row, who = [], []
    for r, (d, tk, sp) in enumerate(items):
        off = T - len(tk)
        idx[r, off:] = tk
        for i, s, e in sp:
            seg[r, off + s:off + e] = len(who)
            row.append(r)
            who.append((d, i))
    b = dict(idx=torch.from_numpy(idx), seg=torch.from_numpy(seg), row=torch.tensor(row),
             F=torch.from_numpy(np.stack([Fs[d][i] for d, i in who])), who=who)
    if Ys is not None:
        b["y"] = torch.from_numpy(np.stack([Ys[d][i] for d, i in who]))
    return b


# --------------------------------------------------------------------- model --
class Head(nn.Module):
    def __init__(self, d, mean, std, drop=0.1):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32))
        self.net = nn.Sequential(nn.Linear(2 * d + len(mean), 512), nn.GELU(), nn.Dropout(drop),
                                 nn.Linear(512, 2))

    def forward(self, Z, C, F):
        return self.net(torch.cat([Z, C, (F - self.mean) / self.std], 1))


def hidden(lm, kind, cfg, idx):
    if kind == "ed":
        return lm.encode(idx)
    from cs781_model import build_rope
    R = cfg.n_register
    cos, sin = build_rope(idx.size(1) + R, cfg.head_dim, idx.device, lm.emb.weight.dtype,
                          cfg.rope_theta)
    return lm._trunk(idx, cos, sin)[0][:, R:]


def outputs(lm, kind, cfg, head, b, device):
    """(B sentences, 2): z-scored own-F1 estimate, greedy-membership logit."""
    h = hidden(lm, kind, cfg, b["idx"].to(device)).float()
    seg, row = b["seg"].to(device), b["row"].to(device)
    m = seg >= 0
    n = len(row)
    Z = torch.zeros(n, h.size(-1), device=device).index_add_(0, seg[m], h[m])
    c = torch.bincount(seg[m], minlength=n).clamp(min=1)
    return head(Z / c[:, None].float(), h.mean(1)[row], b["F"].to(device))


def load_trunk(a, device):
    lm, kind, cfg = SM.load_lm(a.model, a.ckpt)
    T = min(a.tokens, SM.trained_len(kind, cfg)[0])
    if kind == "ed":
        lm.dec = nn.ModuleList()            # only the encoder reads documents here
    for p in lm.parameters():
        p.requires_grad_(False)
    blocks = lm.enc if kind == "ed" else lm.blocks
    for b in list(blocks)[-a.train_top:]:
        for p in b.parameters():
            p.requires_grad_(True)
    if device.type == "cuda":               # frozen weights in bf16, trainable in fp32
        for p in lm.parameters():
            if not p.requires_grad and p.is_floating_point():
                p.data = p.data.to(torch.bfloat16)
    return lm.to(device), kind, cfg, T


def scores(lm, kind, cfg, head, rows, tk, T, eos, device, ac, batch):
    """{id: (sentences, 2)}.  A sentence with no tokens stays at zero."""
    lm.eval(), head.eval()
    out = {r[0]: np.zeros((len(r[1]), 2), dtype=np.float32) for r in rows}
    Fs = [r[2] for r in rows]
    items = [c for d, t in enumerate(tk) for c in chunks(d, t, T)]
    with torch.no_grad():
        for i in range(0, len(items), batch):
            b = collate(items[i:i + batch], T, eos, Fs)
            with ac():
                p = outputs(lm, kind, cfg, head, b, device)
            p = p.float().cpu().numpy()
            for (d, s), v in zip(b["who"], p):
                out[rows[d][0]][s] = v
    return out


def doc_corr(a, b):
    r = [np.corrcoef(x, y)[0, 1] for x, y in zip(a, b) if len(x) > 2 and x.std() > 0 and y.std() > 0]
    return float(np.mean(r)) if r else 0.0


# --------------------------------------------------------------------- train --
def cmd_train(a):
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and "T4" in torch.cuda.get_device_name(0):
        raise SystemExit("T4: bf16 is emulated on SM75 at ~30x slowdown. Use an L4.")
    d = f"{RES}/{a.tag}"
    os.makedirs(d, exist_ok=True)
    os.makedirs(SM.CKPT, exist_ok=True)
    S.load_model()
    lab = load_labs("fit")
    recs = [r for r in D.load("fit") if lab.get(r["id"]) is not None]
    if a.limit:
        recs = recs[:a.limit]
    rows = S._map(S._pred_rows, recs, a.procs)      # workers first: no fork once on the GPU
    dev_rows = S._map(S._pred_rows, D.load("dev"), a.procs)
    test_rows = S._map(S._pred_rows, D.load("test"), a.procs)
    tk, tk_dev, tk_test = (tokenise(x, a.procs) for x in (rows, dev_rows, test_rows))
    Fs = [r[2] for r in rows]
    Ys = [np.c_[lab[r[0]][0], lab[r[0]][1]] for r in rows]
    eos = SM.specials()[0]

    lm, kind, cfg, T = load_trunk(a, device)
    items = [c for i, t in enumerate(tk) for c in chunks(i, t, T)]
    all_F = np.concatenate(Fs[:2000])
    head = Head(cfg.d_model, all_F.mean(0), all_F.std(0) + 1e-3).to(device)
    trunk_p = [p for p in lm.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([dict(params=trunk_p, lr=a.lr), dict(params=head.parameters(), lr=a.head_lr)],
                            weight_decay=a.wd, betas=(0.9, 0.95))
    log(f"{a.tag}: {a.model} {kind} | top {a.train_top} blocks, "
        f"{sum(p.numel() for p in trunk_p) / 1e6:.1f}M trunk + "
        f"{sum(p.numel() for p in head.parameters()) / 1e6:.2f}M head trainable | "
        f"{len(recs)} docs, {len(items)} chunks of up to {T} tokens")
    ac = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if device.type == "cuda" \
        else (lambda: torch.autocast("cpu", enabled=False))

    final, resume = f"{SM.CKPT}/{a.tag}_final.pt", f"{SM.CKPT}/{a.tag}_resume.pt"
    keys = {n for n, p in lm.named_parameters() if p.requires_grad}
    step, t_train, total, probed = 0, 0.0, 10 ** 9, False
    if os.path.exists(final):
        r = torch.load(final, map_location=device, weights_only=False)
        lm.load_state_dict(r["trunk"], strict=False)
        head.load_state_dict(r["head"])
        total, probed, step = 0, True, r["step"]
        log(f"  {final} exists, scoring only")
    elif os.path.exists(resume):
        r = torch.load(resume, map_location=device, weights_only=False)
        lm.load_state_dict(r["trunk"], strict=False)
        head.load_state_dict(r["head"])
        step, total, t_train, probed = r["step"], r["total"], r["t_train"], True
        log(f"  resumed at step {step}/{total} ({t_train / 60:.1f} train-min done)")

    budget, losses, last = a.budget_min * 60, [], time.time()
    rng = np.random.default_rng(a.seed + step)
    lm.train(), head.train()
    while step < total:
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        b = collate([items[i] for i in rng.integers(len(items), size=a.batch)], T, eos, Fs, Ys)
        y = b["y"].to(device)
        with ac():
            p = outputs(lm, kind, cfg, head, b, device).float()
            loss = TF.mse_loss(p[:, 0], y[:, 0]) + TF.binary_cross_entropy_with_logits(p[:, 1], y[:, 1])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trunk_p + list(head.parameters()), 1.0)
        warm = max(1, int(0.05 * min(total, 20000)))
        f = (step + 1) / warm if step < warm else 0.1 + 0.9 * 0.5 * (1 + math.cos(
            math.pi * min(1.0, (step - warm) / max(1, total - warm))))
        opt.param_groups[0]["lr"], opt.param_groups[1]["lr"] = a.lr * f, a.head_lr * f
        opt.step()
        step += 1
        t_train += time.time() - t0
        losses.append(float(loss.detach()))
        if not probed and step == a.probe:          # fix the schedule to the time budget
            per = t_train / step
            total = max(step + 1, int(budget / per))
            probed = True
            log(f"  probe: {per:.3f}s/step -> {total} steps, "
                f"{total * a.batch / len(items):.2f} passes over the chunks in {a.budget_min:.0f} min")
        if step % 200 == 0 or step == total:
            log(f"  step {step}/{total} loss {np.mean(losses[-200:]):.4f} {t_train / 60:.1f} min")
        if t_train > budget * 1.05:
            log("  time budget reached")
            break
        if time.time() - last > a.save_every_min * 60:
            tmp = resume + ".tmp"
            torch.save(dict(trunk={k: v for k, v in lm.state_dict().items() if k in keys},
                            head=head.state_dict(), step=step, total=total, t_train=t_train), tmp)
            os.replace(tmp, resume)
            last = time.time()
    if not os.path.exists(final):
        torch.save(dict(trunk={k: v for k, v in lm.state_dict().items() if k in keys},
                        head=head.state_dict(), step=step), final)

    sd = scores(lm, kind, cfg, head, dev_rows, tk_dev, T, eos, device, ac, a.batch * 2)
    st = scores(lm, kind, cfg, head, test_rows, tk_test, T, eos, device, ac, a.batch * 2)
    with open(f"{d}/scores.pkl", "wb") as f:
        pickle.dump(dict(dev=sd, test=st), f)
    dl = load_labs("dev")
    have = [r for r in dev_rows if dl.get(r[0]) is not None]
    rel = W.load_head(a.compare)
    y = [dl[r[0]][0] for r in have]
    nn_r, gb_r = doc_corr([sd[r[0]][:, 0] for r in have], y), doc_corr([rel.predict(r[2]) for r in have], y)
    meta = dict(tag=a.tag, kind="extractive", model=a.model, method="partial", train_top=a.train_top,
                steps=step, train_min=round(t_train / 60, 2), T=T, docs=len(recs),
                final_loss=float(np.mean(losses[-200:])) if losses else None,
                dev_corr=nn_r, dev_corr_gb=gb_r, args=vars(a))
    with open(f"{d}/meta.json", "w") as f:
        json.dump(meta, f, indent=1, default=str)
    if os.path.exists(resume):
        os.remove(resume)
    log(f"DONE {a.tag}: per-document correlation with y_f on dev, {nn_r:.4f} vs {a.compare} "
        f"{gb_r:.4f} | {meta['train_min']} train-min")


# --------------------------------------------------------------------- sweep --
class Fixed:
    """A scorer whose answer was computed already: S.select only ever calls predict(F)."""

    def __init__(self, y):
        self.y = y

    def predict(self, F):
        return self.y


def texts(recs, feats, rel, mult, win, m_all, lam):
    tg = S.len_targets(recs, "model", mult)
    return {f[0]: W.build(f[1:], tg[f[0]], win, Fixed(rel[f[0]]), m_all, lam) for f in feats}


def stack(fd, ft, base_d, base_t, dl, pick, hold):
    """Depth-3 HistGB over everything, trained on one half of dev and applied to the other,
    so neither half is scored by a model that saw it.  Test takes the mean of the two."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    X = lambda f, base: np.c_[(f[2], *[v[f[0]] for v in base])]
    by = {f[0]: f for f in fd}

    def fit(ids):
        ids = [k for k in ids if dl.get(k) is not None]
        m = HistGradientBoostingRegressor(max_depth=3, max_iter=200, learning_rate=0.05,
                                          min_samples_leaf=200, l2_regularization=1.0,
                                          early_stopping=False, random_state=D.SEED)
        return m.fit(np.concatenate([X(by[k], base_d) for k in ids]),
                     0.05 * np.concatenate([dl[k][0] for k in ids]))
    mp, mh = fit(pick), fit(hold)
    out = {k: mh.predict(X(by[k], base_d)) for k in pick}
    out.update({k: mp.predict(X(by[k], base_d)) for k in hold})
    return out, {f[0]: (mp.predict(X(f, base_t)) + mh.predict(X(f, base_t))) / 2 for f in ft}


def cmd_sweep(a):
    m_all = S.load_model()
    with open(f"{RES}/bsel_best/meta.json") as f:
        bm = json.load(f)
    bsel, gb3 = W.load_head(bm["head"]), W.load_head("bsel3_f")
    dev, test = D.load("dev"), D.load("test")
    fd, ft = S._map(S._pred_rows, dev, a.procs), S._map(S._pred_rows, test, a.procs)
    refs, dl = D.refs_of("dev"), load_labs("dev")
    perm = np.random.default_rng(D.SEED).permutation(len(dev))       # phase6_win's two halves
    half = len(dev) // 2
    pick, hold = [dev[i]["id"] for i in perm[:half]], [dev[i]["id"] for i in perm[half:]]

    mem = []
    for t in a.members.split(","):
        if os.path.exists(f"{RES}/{t}/scores.pkl"):
            with open(f"{RES}/{t}/scores.pkl", "rb") as f:
                mem.append(pickle.load(f))
        else:
            log(f"  !! {t} has no scores, left out")
    if not mem:
        raise SystemExit("no neural member finished")

    def table(feats, split):
        b = {f[0]: bsel.predict(f[2]) for f in feats}
        g = {f[0]: gb3.predict(f[2]) for f in feats}
        n = {f[0]: 0.05 * np.mean([m[split][f[0]][:, 0] for m in mem], 0) for f in feats}
        lg = {f[0]: np.mean([m[split][f[0]][:, 1] for m in mem], 0) for f in feats}
        each = [{f[0]: m[split][f[0]][:, 0] for f in feats} for m in mem] if len(mem) > 2 else []
        return (b, g, n, lg, *each)      # with 3+ members the stack can weigh each one

    TD, TT = table(fd, "dev"), table(ft, "test")
    (bd, gd, nd, ld), (bt, gt, nt, lt) = TD[:4], TT[:4]
    mix = lambda x, y, w: {k: (1 - w) * x[k] + w * y[k] for k in x}
    sd, st = stack(fd, ft, TD, TT, dl, pick, hold)
    heads = dict(bsel=(bd, bt), bsel3=(gd, gt), nn=(nd, nt), stack=(sd, st))
    heads.update({f"mix{int(w * 100)}": (mix(bd, nd, w), mix(bt, nt, w)) for w in (0.25, 0.5, 0.75)})

    win0, mult = bm["win"], bm["mult"]
    sum_dev = lambda h, win, lam, feats=fd: texts(dev, feats, heads[h][0], mult, win, m_all, lam)
    sum_test = lambda h, win, lam, feats=ft: texts(test, feats, heads[h][1], mult, win, m_all, lam)
    part = lambda sm, ids: D.score_summaries({k: sm[k] for k in ids}, {k: refs[k] for k in ids},
                                             per_doc=True)

    ctrl = sum_dev("bsel", win0, bm["lam"])
    with open(f"{RES}/bsel_best/summaries_dev.json", encoding="utf-8") as f:
        old = json.load(f)
    same = sum(ctrl[k] == old.get(k) for k in ctrl)
    log(f"  CONTROL: rebuilt bsel_best word for word on {same}/{len(ctrl)} dev docs")
    base = part(old, hold)
    vs = None
    if a.versus and os.path.exists(f"{RES}/{a.versus}/summaries_dev.json"):
        with open(f"{RES}/{a.versus}/summaries_dev.json", encoding="utf-8") as f:
            vs = part(json.load(f), hold)
        log(f"  VERSUS {a.versus}: held-out half {vs['bertscore']:.4f}, the score to beat")

    rows = []
    for h in heads:
        for lam in (0.15, 0.45):
            s = part(sum_dev(h, win0, lam), pick)
            rows.append(dict(head=h, win=win0, lam=lam, clean=False, pick=s["bertscore"]))
            log(f"  {h} lam={lam}: pick half {s['bertscore']:.4f} rouge1 {s['rouge1']:.4f}")

    # the best head per family goes to the held-out half, once
    fam = {h: max((r for r in rows if r["head"] == h), key=lambda r: r["pick"]) for h in heads}
    top = max(fam.values(), key=lambda r: r["pick"])
    fc = [(f[0], [clean(s) for s in f[1]], f[2], f[3]) for f in fd]
    tc = [(f[0], [clean(s) for s in f[1]], f[2], f[3]) for f in ft]
    extra = [dict(top, win=w) for w in a.wins] + [dict(top, clean=True)]
    for r in extra:
        s = part(sum_dev(r["head"], r["win"], r["lam"], fc if r["clean"] else fd), pick)
        r["pick"] = s["bertscore"]
        log(f"  {r['head']} win={r['win']} lam={r['lam']} clean={r['clean']}: pick half "
            f"{s['bertscore']:.4f}")
    best = max([top] + extra, key=lambda r: r["pick"])

    cand = {f"{a.prefix}_{h}": r for h, r in fam.items() if h != "bsel"}
    if best is not top:
        cand[f"{a.prefix}_best"] = best
    held = []
    for tag, r in cand.items():
        sm = sum_dev(r["head"], r["win"], r["lam"], fc if r["clean"] else fd)
        s = part(sm, hold)
        d = np.array([s["per_doc_bert"][k] - base["per_doc_bert"][k] for k in hold])
        se = float(d.std(ddof=1) / max(len(d), 1) ** 0.5) if len(d) > 1 else 0.0
        held.append(dict(r, tag=tag, score=s["bertscore"], gain=float(d.mean()), se=se, rouge1=s["rouge1"]))
        log(f"  HELD-OUT HALF: {tag} {s['bertscore']:.4f} vs bsel_best {base['bertscore']:.4f} gain "
            f"{d.mean():+.4f} +- {se:.4f} | ROUGE-1 {s['rouge1']:.4f} vs {base['rouge1']:.4f} | n={len(hold)}")
        if vs is not None:
            e = np.array([s["per_doc_bert"][k] - vs["per_doc_bert"][k] for k in hold])
            held[-1].update(vs_gain=float(e.mean()), vs_se=float(e.std(ddof=1) / len(e) ** 0.5))
            log(f"  VS {a.versus}: {tag} {e.mean():+.4f} +- {held[-1]['vs_se']:.4f}")
        o = f"{RES}/{tag}"
        os.makedirs(o, exist_ok=True)
        with open(f"{o}/summaries_dev.json", "w", encoding="utf-8") as f:
            json.dump(sm, f, ensure_ascii=False)
    held.sort(key=lambda r: -r["gain"])

    fin = f"{P6}/submissions/{a.out}"
    os.makedirs(fin, exist_ok=True)
    for i, r in enumerate(held, 1):
        o = f"{RES}/{r['tag']}"
        tm = sum_test(r["head"], r["win"], r["lam"], tc if r["clean"] else ft)
        with open(f"{o}/summaries_test.json", "w", encoding="utf-8") as f:
            json.dump(tm, f, ensure_ascii=False)
        with open(f"{o}/meta.json", "w") as f:
            json.dump(dict(r, kind="extractive", len="model", mult=mult, members=a.members), f, indent=1)
        z = D.write_submission(r["tag"], tm, kind="extractive", dev_scores=dict(bertscore=r["score"]),
                               note=f"{r['head']} scorer, win={r['win']} lam={r['lam']} clean={r['clean']}")
        shutil.copyfile(z, f"{fin}/{i:02d}_{r['tag']}.zip")
    with open(f"{RES}/{a.prefix}_sweep.json", "w") as f:
        json.dump(dict(rows=rows, extra=extra, held=held, control=same, base=base["bertscore"]), f, indent=1)
    log(f"SWEEP DONE: {len(held)} zips in submissions/{a.out}, best held-out gain "
        f"{held[0]['gain']:+.4f} ({held[0]['tag']})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--procs", type=int, default=3)
    sp = ap.add_subparsers(dest="cmd", required=True)
    f = sp.add_parser("labels")
    f.add_argument("--budget-min", type=float, default=85)
    f.add_argument("--max-sents", type=int, default=1500)
    f.set_defaults(fn=cmd_labels)
    g = sp.add_parser("gb")
    g.set_defaults(fn=cmd_gb)
    t = sp.add_parser("train")
    t.add_argument("--tag", required=True)
    t.add_argument("--model", required=True, help="M1_SC M2_LexArcED M3_LexArcL M4_LexArcP")
    t.add_argument("--ckpt", default="best.pt")
    t.add_argument("--tokens", type=int, default=1024)
    t.add_argument("--train-top", type=int, default=4)
    t.add_argument("--batch", type=int, default=8)
    t.add_argument("--lr", type=float, default=5e-5)
    t.add_argument("--head-lr", type=float, default=1e-3)
    t.add_argument("--wd", type=float, default=0.01)
    t.add_argument("--budget-min", type=float, default=60)
    t.add_argument("--probe", type=int, default=20)
    t.add_argument("--save-every-min", type=float, default=10)
    t.add_argument("--compare", default="bsel_f", help="scorer the dev correlation is set against")
    t.add_argument("--seed", type=int, default=1)
    t.add_argument("--limit", type=int, default=0)
    t.set_defaults(fn=cmd_train)
    s = sp.add_parser("sweep")
    s.add_argument("--members", default="nsel_m1,nsel_m2")
    s.add_argument("--wins", type=lambda x: [int(v) for v in x.split(",")], default=[360, 390])
    s.add_argument("--prefix", default="nsel", help="tag prefix; also names results/<prefix>_sweep.json")
    s.add_argument("--out", default="_final2", help="folder under submissions/ for the ranked zips")
    s.add_argument("--versus", default="", help="earlier tag to report a paired held-out gain against")
    s.set_defaults(fn=cmd_sweep)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
