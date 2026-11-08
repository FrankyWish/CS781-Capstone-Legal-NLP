#!/usr/bin/env python
"""Phase 7 -- a cross-encoder reranker over the first-stage pool.  The last real lever.

Why this and not more retrieval.  27 paired leaderboard scores put test at devx - 0.0095
(sd 0.015), and the devx oracle@100 -- the best F1@7 a PERFECT rerank of the top 100
could reach -- is 0.5481.  The stacker reaches 0.4158.  So about 100 reachable hits are
sitting inside the pool we already retrieve, in the wrong order.  Nothing in this phase
has yet looked at the TEXT of a (query, candidate) pair jointly: every system so far
scores the two sides independently (BM25, bi-encoder cosine) and the stacker only
reweights those independent scores.  A cross-encoder is the one family that can use the
interaction, and it is what moved Phase 6 off its plateau.

What it is.  M2_LexArcED's bidirectional encoder over one sequence

    [ query head (q_tok tokens) ] [ EOS ] [ a window of the candidate ]

pooled and passed through a scalar head, trained listwise: softmax over one gold prior
case and n_neg hard negatives mined from the first-stage ranking, cross-entropy against
the gold.  A document is scored by the maximum over its windows, the same MaxP logic the
lexical index uses: a case is cited for one passage of itself.

The part that is easy to get wrong.  To be a stacker member a system needs scores on
TRAIN, and a reranker trained on train queries scores them in-sample -- it would look
near-perfect there, the stacker would learn to trust it, and the weight would not
transfer.  That is very likely why `stack_ft` (0.4069) came in BELOW `stack_d100`
(0.4158) despite having strictly more features: the six trained bi-encoder members all
fed the stacker in-sample train scores.  So this reranker is CROSS-FITTED: `--folds k`
models, each trained on k-1 folds of the train queries and used only to score the fold it
never saw, then one final model trained on everything to score devx and test.  Costs k+1
trainings and buys a feature the stacker can actually weight.

    python phase7_rerank.py fit  --tag ce_lorma --folds 3 --base tfidf_sqrt
    python phase7_rerank.py score --tag ce_lorma --splits devx,test   # reuse a ckpt

Outputs results/<tag>/{train,devx,test}_rank.json with exactly --depth reranked ids per
query (the reranker reorders a pool, it does not retrieve), plus results/<tag>_mix with
the reranker z-blended against the first stage, which is the safety net if the reranker
is strong on some queries and erratic on others.
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
import phase7_emb as M

PEFT_DIR = os.environ.get(
    "CS781_PEFT", os.path.normpath(os.path.join(HERE, "..", "..", "phase6", "code")))
if PEFT_DIR not in sys.path:
    sys.path.insert(0, PEFT_DIR)
import phase6_peft as P

P7 = D.P7
MODELS = f"{P7}/models"
SEED = D.SEED


def log(*a):
    print(" ", *a, flush=True)


# ------------------------------------------------------------------- the model ------
class CrossEncoder(nn.Module):
    """Phase-1 trunk + a scalar relevance head.

    `trunk` is the attribute name phase6_peft.apply_method patches and freezes, and the
    head lives outside it, so it stays trainable and picks up `--lr-head` through
    phase7_emb.optimiser without a special case."""

    def __init__(self, lm, kind, cfg, pool="mean", bidir=False):
        super().__init__()
        self.trunk = lm
        self.kind, self.cfg = kind, cfg
        self.pool_how, self.bidir = pool, bidir
        self.score = nn.Linear(cfg.d_model, 1)
        nn.init.zeros_(self.score.bias)
        nn.init.normal_(self.score.weight, std=0.02)

    def forward(self, idx, ok):
        st = (E.enc_forward(self.trunk, self.cfg, idx, ok) if self.kind == "ed"
              else E.dec_forward(self.trunk, self.cfg, idx, ok, bidir=self.bidir))
        v = E.pool(st.float(), ok, self.pool_how)
        return self.score(v).squeeze(-1)


def build(a):
    lm, kind, cfg = E.load_lm(a.model, a.ckpt)
    factor = max(a.chunk / E.trained_len(kind, cfg), 1.0)
    theta = E.ntk(cfg, factor)
    log(f"{a.model} kind={kind} d={cfg.d_model} chunk={a.chunk} "
        f"rope x{factor:.2f} theta {theta:.0f}")
    return CrossEncoder(lm, kind, cfg, a.pool, a.bidir), kind, cfg


# -------------------------------------------------------------------- the pairs -----
def pair_rows(q_ids, c_ids, a, rng=None):
    """[query head ; EOS ; candidate window] -> one token array per window.

    The query contributes its head: a judgment's opening carries the facts, and a fixed
    deterministic window keeps a candidate's score comparable across queries.  Training
    takes ONE random candidate window per step (so the whole document is covered over
    training without paying for it every step); scoring takes `--windows` deterministic
    windows and keeps the maximum."""
    eos = np.asarray([E.eos_id()], dtype=np.int64)
    qh = np.asarray(q_ids[:a.q_tok], dtype=np.int64)
    cw = max(a.chunk - len(qh) - 1, 64)
    if rng is not None:
        if len(c_ids) <= cw:
            wins = [c_ids]
        else:
            s = int(rng.integers(0, len(c_ids) - cw + 1))
            wins = [c_ids[s:s + cw]]
    else:
        wins = E.chunks_of(np.asarray(c_ids, dtype=np.int64), cw, a.windows)
    return [np.concatenate([qh, eos, np.asarray(w, dtype=np.int64)]) for w in wins]


def fold_of(qid, folds):
    """Deterministic fold assignment -- stable across runs and machines, unlike hash()."""
    import hashlib
    return int(hashlib.md5(str(qid).encode()).hexdigest(), 16) % folds


def train_batch(ts, idxs, a, rng, device):
    rows, groups = [], []
    for j in idxs:
        q, pos, negs = ts.items[j]
        qd = ts.qtok.doc(ts.qtok.pos[q])
        p = pos[int(rng.integers(len(pos)))]
        cs = [p] + [str(c) for c in
                    rng.choice(negs, size=min(a.n_neg, len(negs)), replace=False)]
        g = []
        for c in cs:
            g.append(len(rows))
            rows.extend(pair_rows(qd, ts.ctok.doc(ts.ctok.pos[str(c)]), a, rng))
        groups.append(g)                      # groups[*][0] is the gold
    idx, ok = M.pad(rows, a.chunk, device)
    return idx, ok, groups


def run_training(model, ts, items, a, device):
    """items: indices into ts.items this model is allowed to see."""
    model.to(device).train()
    opt = M.optimiser(model, a)
    base = [g["lr"] for g in opt.param_groups]
    rng = np.random.default_rng(SEED)
    order = rng.permutation(items)
    cur, t0, hist, accs = 0, time.time(), [], []
    amp = dict(device_type=device.type, dtype=torch.bfloat16,
               enabled=(device.type == "cuda"))
    for step in range(a.steps):
        if cur + a.batch > len(order):
            order, cur = rng.permutation(items), 0
        sel = order[cur:cur + a.batch]
        cur += a.batch
        idx, ok, groups = train_batch(ts, sel, a, rng, device)
        s = M.lr_at(step, a)
        for g, b in zip(opt.param_groups, base):
            g["lr"] = b * s
        with torch.autocast(**amp):
            out = model(idx, ok)
            zero = torch.zeros(1, dtype=torch.long, device=device)
            loss = torch.stack([F.cross_entropy((out[g] / a.temp).unsqueeze(0), zero)
                                for g in groups]).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], a.clip)
        opt.step()
        hist.append(float(loss.detach()))
        with torch.no_grad():
            accs.append(float(np.mean([float(out[g].argmax() == 0) for g in groups])))
        if step % a.log_every == 0 or step == a.steps - 1:
            log(f"    step {step + 1}/{a.steps} loss {np.mean(hist[-a.log_every:]):.4f} "
                f"top1 {np.mean(accs[-a.log_every:]):.3f} lr x{s:.3f} "
                f"gnorm {float(gn):.2f} ({time.time() - t0:.0f}s)")
    return dict(final_loss=round(float(np.mean(hist[-20:])), 5),
                final_top1=round(float(np.mean(accs[-50:])), 4),
                steps=a.steps, train_seconds=round(time.time() - t0, 1))


# ------------------------------------------------------------------- scoring --------
def score_one(model, qd, cands, ctok, a, device):
    """-> scores aligned with `cands`, each the max over that candidate's windows."""
    rows, owner = [], []
    for k, c in enumerate(cands):
        for r in pair_rows(qd, ctok.doc(ctok.pos[c]), a, None):
            rows.append(r)
            owner.append(k)
    out = np.full(len(rows), -1e9, dtype=np.float32)
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]))
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                         enabled=(device.type == "cuda")):
        for s in range(0, len(order), a.score_batch):
            sel = order[s:s + a.score_batch]
            idx, ok = M.pad([rows[i] for i in sel], a.chunk, device)
            out[sel] = model(idx, ok).float().cpu().numpy()
    agg = np.full(len(cands), -1e9, dtype=np.float32)
    np.maximum.at(agg, np.asarray(owner, dtype=np.int64), out)
    return agg


def score_split(model, split, a, device, only=None):
    """-> {qid: (cand ids, reranker scores)} over the first stage's top --depth."""
    base, _ = R.load_run(a.base, split)
    if base is None:
        raise SystemExit(f"no {split} ranking for the first stage {a.base}")
    qtok, ctok = E.Tok(split, "q"), E.Tok(split, "c")
    qs = [r["id"] for r in D.queries(split)
          if r["id"] in qtok.pos and (only is None or r["id"] in only)]
    out, t0 = {}, time.time()
    for n, q in enumerate(qs):
        cands = [c for c in base.get(q, [])[:a.depth] if c in ctok.pos]
        if not cands:
            continue
        out[q] = (cands, score_one(model, qtok.doc(qtok.pos[q]), cands, ctok, a, device))
        if n % 25 == 0 or n == len(qs) - 1:
            log(f"    scored {n + 1}/{len(qs)} {split} queries "
                f"({time.time() - t0:.0f}s)")
    return out


def to_runs(scored, split, a):
    """-> (pure reranker ranking, z-blend with the first stage), both id lists."""
    base, bsc = R.load_run(a.base, split)
    ce_rank, ce_score, mx_rank, mx_score = {}, {}, {}, {}
    for q, (cands, sc) in scored.items():
        o = np.argsort(-sc, kind="stable")
        ce_rank[q] = [cands[i] for i in o]
        ce_score[q] = sc[o].astype(np.float32)
        bm = {}
        ids, bs = base.get(q, []), (bsc or {}).get(q)
        if bs is not None:
            for i, c in enumerate(ids[:len(bs)]):
                bm[c] = float(bs[i])
        bv = np.array([bm.get(c, np.nan) for c in cands], dtype=np.float64)
        if np.isfinite(bv).sum() > 1:
            bv = np.where(np.isfinite(bv), bv, np.nanmin(bv))
            mix = a.mix * R.zscore(sc.astype(np.float64)) + (1 - a.mix) * R.zscore(bv)
        else:
            mix = sc.astype(np.float64)
        o2 = np.argsort(-mix, kind="stable")
        mx_rank[q] = [cands[i] for i in o2]
        mx_score[q] = mix[o2].astype(np.float32)
    return (ce_rank, ce_score), (mx_rank, mx_score)


# ------------------------------------------------------------------- bookkeeping ----
def save_ckpt(model, cfg, a, info, suffix=""):
    os.makedirs(MODELS, exist_ok=True)
    p = f"{MODELS}/{a.tag}{suffix}.pt"
    torch.save(dict(cfg=dataclasses.asdict(cfg) if dataclasses.is_dataclass(cfg) else cfg,
                    model=model.state_dict(), method=a.method,
                    peft=dict(targets=a.targets, rank=a.rank, alpha=a.alpha,
                              lora_dropout=a.lora_dropout, bottleneck=a.bottleneck,
                              train_top=a.train_top, ada_target=a.ada_target),
                    base=dict(model=a.model, ckpt=a.ckpt, chunk=a.chunk, pool=a.pool,
                              bidir=bool(a.bidir), q_tok=a.q_tok, windows=a.windows,
                              first_stage=a.base, depth=a.depth),
                    info=info), p)
    log(f"saved {p}")
    return p


def load_ckpt(tag, device, a=None):
    p = f"{MODELS}/{tag}.pt"
    if not os.path.exists(p):
        raise SystemExit(f"no checkpoint for {tag} at {p}")
    ck = torch.load(p, map_location="cpu", weights_only=False)
    b = ck["base"]
    ns = argparse.Namespace(model=b["model"], ckpt=b["ckpt"], chunk=b["chunk"],
                            pool=b["pool"], bidir=b["bidir"])
    model, kind, cfg = build(ns)
    P.apply_method(model, argparse.Namespace(method=ck["method"], **ck["peft"]))
    miss, unexp = model.load_state_dict(ck["model"], strict=False)
    miss = [k for k in miss if not k.endswith("head.weight")]
    if miss or unexp:
        log(f"!! {tag}: {len(miss)} missing, {len(unexp)} unexpected weights")
    if a is not None:                       # the checkpoint's own geometry wins
        a.q_tok, a.windows, a.chunk = b["q_tok"], b["windows"], b["chunk"]
    return model.to(device), kind, cfg, ck


def evaluate_covered(rk, split):
    """R.evaluate restricted to the queries this system actually scored.

    A cross-fitted train ranking CANNOT cover every gold query. `TrainSet` drops the
    contaminated query, every gold-less one, and any query whose golds fall outside the
    tokenised pool, so those queries belong to no fold and are never scored. The official
    scorer is right to refuse a partial prediction -- but refusing it HERE killed an
    82-minute L4 run at its last step, after the model was trained and devx and test were
    fully scored. So the covered subset is scored and its size is reported, loudly,
    because an F1 over 816 of 827 queries has a different denominator and is not
    comparable to a full-split number.
    """
    g, gp = D.golds(split), D.golds_in_pool(split)
    cov = {q: v for q, v in g.items() if q in rk}
    sub = {q: rk[q] for q in cov}
    reach = {q: gp.get(q, []) for q in cov}
    s = D.score_ranking(sub, cov, reachable=reach)
    s["recall_curve"] = D.recall_at(sub, reach)
    s.pop("per_query", None)
    s["covered"], s["split_queries"] = len(cov), len(g)
    return s


def write(tag, kind, scored_by_split, a, extra):
    meta = dict(tag=tag, kind=kind, first_stage=a.base, depth=a.depth,
                config=dict(method=a.method, q_tok=a.q_tok, windows=a.windows,
                            chunk=a.chunk, n_neg=a.n_neg, steps=a.steps,
                            folds=a.folds, mix=a.mix,
                            topn=getattr(a, "topn", None)), **extra)
    mmeta = dict(meta, tag=f"{tag}_mix", kind=kind)
    # SAVE EVERY SPLIT FIRST, then evaluate.  The previous order interleaved them, so
    # when the train evaluation raised, the devx and test rankings that had already been
    # computed -- 17 minutes of L4 forward passes -- were never written and the whole run
    # was lost.  Scoring is expensive and evaluation is free: nothing that can fail
    # belongs between the GPU work and the file it produces.
    done = {}
    for split, scored in scored_by_split.items():
        if not scored:
            continue
        (cr, cs), (mr, ms) = to_runs(scored, split, a)
        R.save_run(tag, split, cr, cs)
        R.save_run(f"{tag}_mix", split, mr, ms)
        done[split] = (cr, mr)
    for split, (cr, mr) in done.items():
        if split == "test":
            continue
        for t, rk, mt in ((tag, cr, meta), (f"{tag}_mix", mr, mmeta)):
            st = evaluate_covered(rk, split)
            mt[f"{split}_scores"] = {k: v for k, v in st.items() if k != "per_query"}
            extra_note = ("" if st["covered"] == st["split_queries"]
                          else f"[{st['covered']}/{st['split_queries']} queries covered]")
            if extra_note:
                log(f"  {t}/{split}: {extra_note} -- a cross-fit cannot score a query "
                    f"that was never trainable, so this F1 has its own denominator")
            R.report(f"{t}/{split}", st, extra_note)
    R.save_run(tag, "meta", {}, None, meta)
    R.save_run(f"{tag}_mix", "meta", {}, None, mmeta)
    return meta


# ---------------------------------------------------------------------- commands ----
def fit_one(a, items, suffix, device):
    model, kind, cfg = build(a)
    n, ctrl, info = P.apply_method(model, a)
    n += sum(p.numel() for p in model.score.parameters())
    total = sum(p.numel() for p in model.parameters())
    log(f"method {a.method}: {n:,} trainable of {total:,} ({100.0 * n / total:.2f}%) "
        f"{json.dumps(info)}")
    if ctrl is not None:
        ctrl.total = a.steps
    stats = run_training(model, a._ts, items, a, device)
    log(f"trained{suffix or ' (final)'} in {stats['train_seconds']}s, "
        f"loss {stats['final_loss']} top1 {stats['final_top1']}")
    save_ckpt(model, cfg, a, dict(**stats, **info, trainable=n), suffix)
    return model, stats, n, total


def cmd_fit(a):
    device = M.gpu_or_die(a.cpu)
    a._ts = M.TrainSet(a.neg_tag, a.neg_pool)
    splits = [s for s in a.splits.split(",") if s]
    scored, folds_stats = {}, []

    # ---- cross-fitted train scores: a fold is only ever scored by a model that did
    # ---- not train on it, so the stacker's train feature is out of sample.
    if a.folds > 1 and "train" in splits:
        qid = [a._ts.items[i][0] for i in range(len(a._ts.items))]
        assign = np.array([fold_of(q, a.folds) for q in qid])
        got = {}
        for k in range(a.folds):
            tr = np.flatnonzero(assign != k)
            ho = {qid[i] for i in np.flatnonzero(assign == k)}
            log(f"==== fold {k + 1}/{a.folds}: train on {len(tr)}, score {len(ho)}")
            model, st, _, _ = fit_one(a, tr, f"_f{k}", device)
            got.update(score_split(model, "train", a, device, only=ho))
            folds_stats.append(dict(fold=k, n_train=int(len(tr)), n_held=len(ho), **st))
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        scored["train"] = got
        log(f"cross-fitted train scores for {len(got)} queries")

    # ---- the reported model: trained on every usable train query
    model, stats, n, total = fit_one(a, np.arange(len(a._ts.items)), "", device)
    for s in splits:
        if s != "train":
            scored[s] = score_split(model, s, a, device)
    return write(a.tag, a.kind, scored, a,
                 dict(trainable=n, total_params=total, folds_stats=folds_stats, **stats))


def cmd_score(a):
    """Re-score splits from a saved final checkpoint, without retraining."""
    device = M.gpu_or_die(a.cpu)
    model, kind, cfg, ck = load_ckpt(a.tag, device, a)
    scored = {s: score_split(model, s, a, device) for s in a.splits.split(",") if s}
    return write(a.tag, a.kind, scored, a, dict(ck.get("info", {})))


def cmd_oracle(a):
    """Oracle@N on devx for a system's pool: the ceiling a perfect rerank would reach.

    Measured, not assumed -- it is the number that says whether a better reranker can
    still pay for itself, and it is the number that said 0.53 is out of reach by ranking
    alone."""
    gold, gp = D.golds("devx"), D.golds_in_pool("devx")
    rows = []
    for tag in a.tags.split(","):
        rk, _ = R.load_run(tag, "devx")
        if rk is None:
            log(f"skip {tag}: no devx run")
            continue
        have = max((len(v) for v in rk.values()), default=0)
        for n in [int(x) for x in a.depths.split(",") if x]:
            if n > have:
                log(f"skip {tag} oracle@{n}: the saved run is only {have} deep "
                    f"(rebuild it with a larger --topn)")
                continue
            best = {}
            for q, g in gp.items():
                pool = set(rk.get(q, [])[:n])
                hit = [c for c in g if c in pool]
                best[q] = hit[:7] + [f"_pad{i}" for i in range(max(0, 7 - len(hit)))]
            s = D.score_ranking(best, gold, reachable=gp)
            rows.append(dict(tag=tag, depth=n, oracle_f1=round(s["f1"], 4),
                             hits=s["hits"]))
            log(f"{tag:22s} oracle@{n:<5d} f1 {s['f1']:.4f} hits {s['hits']}/"
                f"{sum(len(v) for v in gp.values())}")
    D.atomic_json(f"{P7}/results/oracle_depth.json", rows)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("fit", cmd_fit), ("score", cmd_score)):
        p = M.common(sp.add_parser(name))
        p.add_argument("--kind", default="hybrid",
                       help="lexical first stage + neural reranker is a hybrid")
        p.add_argument("--base", default="tfidf_sqrt", help="first-stage system")
        p.add_argument("--depth", type=int, default=50, help="candidates reranked")
        p.add_argument("--q-tok", type=int, default=384)
        p.add_argument("--windows", type=int, default=3)
        p.add_argument("--score-batch", type=int, default=24)
        p.add_argument("--folds", type=int, default=3,
                       help="cross-fit folds for the train-side scores; 0/1 disables")
        p.add_argument("--mix", type=float, default=0.5,
                       help="weight on the reranker in the <tag>_mix z-blend")
        # A reranker reorders a pool, so its output length is --depth, not --topn.  The
        # flag exists anyway because every other system in the phase takes it and the
        # drivers pass it uniformly: without it argparse rejects the whole command, which
        # is exactly how the first ce run died after the GPU stage had already warmed up.
        p.add_argument("--topn", type=int, default=R.TOPN,
                       help="recorded in meta only; the reranker emits --depth ids")
        p.set_defaults(fn=fn)
    p = sp.add_parser("oracle")
    p.add_argument("--tags", default="tfidf_sqrt,stack_d100")
    p.add_argument("--depths", default="20,50,100,200,500,1000")
    p.set_defaults(fn=cmd_oracle)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
