#!/usr/bin/env python3
"""CS-781 Phase 2 -- weak supervision from the Supreme Court corpus.

    python phase2_weaksup.py --n-docs 2000                 # 2-fold CV
    python phase2_weaksup.py --n-docs 2000 --submit --tag ws2000

GPU. Roughly 20 min of labelling on CPU plus 12 min per 1000 weak documents
of training, on top of the usual fold time.

WHY

70 labelled documents is the binding constraint of this phase, and the
evidence is direct rather than inferred: the identical configuration scores
0.2941 trained on 35 documents and 0.3854 trained on 70. Doubling the data
bought +0.09, while doubling the ensemble buys +0.009 and raising the
parameter count buys nothing. Every intervention so far has worked on the
model or on its output. This one works on the data.

The Phase-1 Supreme Court corpus is permitted, in-domain and effectively
unlimited -- the Phase-2 documents are Supreme Court judgments and so is it.
What was missing until now was any way to label it. `phase2_rules.py` supplies
that: 26 patterns, each with exact precision measured against all 8,393 gold
spans, gated at 0.40, plus a caption parser that resolves all six observed
layouts and case-insensitive propagation across each document.

The point is NOT that the model learns the rules -- it could just apply them.
The point is that the rules give it tens of thousands of in-context examples
of what a party, a witness, an authority or a counsel LOOKS like, so it
generalises to the ones the rules miss. Rule recall is 0.04-0.46; the ceiling
that matters is the detector's, and APP, RESP, WIT and AUTH currently cap at
0.35-0.49 overlap because the detector never sees enough of them.

PARTIAL SUPERVISION, WHICH IS THE WHOLE DIFFICULTY

A weakly labelled document is not a labelled document. The rules cover no
COURT and no PREC at the 0.40 gate, so training on "everything the rules did
not mark is O" would actively teach the model to stop predicting two labels it
currently handles well (COURT 0.745, PREC 0.259). That would be worse than not
training at all.

So weak documents are supervised only where the rules have an opinion: every
token is -100 (ignored) except those inside a rule span, and those within
`--radius` tokens of one, which are supervised as O. The model learns entity
shapes and their boundaries from local positive and negative evidence, and
learns nothing at all about the regions the rules cannot see.

Gold fine-tuning always runs afterwards, at full supervision, so the final
model is still fitted to the real annotation.
"""

import argparse
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn.functional as F

BASE = os.environ.get("CS781_BASE", "/content/drive/MyDrive/cs781_capstone")
P1 = os.environ.get("CS781_P1", "%s/phase1" % BASE)
P2 = os.environ.get("CS781_P2", "%s/phase2" % BASE)
sys.path.insert(0, "%s/shared/lib" % P1)
sys.path.insert(0, "%s/code" % P2)

from phase2_data import (P2 as _P2, align, fetch, load_json, log,  # noqa: E402
                         n_tags_for, parse_gold)
from phase2_ner import (batches, build, predict, strict_macro_f1,  # noqa: E402
                        windows, boundary_report)
from phase2_ensemble import (class_weights, decode_all, emissions,  # noqa: E402
                             make)
from phase2_rules import proposals                                 # noqa: E402
from cs781_bpe import BPETokenizer                                 # noqa: E402

REPO = "Exploration-Lab/CS-781-Capstone"


def stream_records(path, bufsize=1 << 22):
    """Yield each record of a large JSON array without loading the file.

    Lifted from phase1/code/build_corpus.py: the corpus is hundreds of MB and
    json.load would not fit in a Colab runtime.
    """
    dec = json.JSONDecoder()
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        buf = f.read(bufsize)
        pos = buf.find("{")
        if pos < 0:
            return
        while True:
            try:
                obj, end = dec.raw_decode(buf, pos)
            except ValueError:
                more = f.read(bufsize)
                if not more:
                    return
                buf = buf[pos:] + more
                pos = buf.find("{")
                if pos < 0:
                    return
                continue
            yield obj
            k = end
            while k < len(buf) and buf[k] in ", \n\r\t":
                k += 1
            if k >= len(buf) or buf[k] == "]":
                more = f.read(bufsize)
                if not more:
                    return
                buf = buf[k:] + more
                pos = buf.find("{")
                if pos < 0:
                    return
            elif len(buf) - k < (1 << 20):
                more = f.read(bufsize)
                buf = buf[k:] + (more or "")
                pos = buf.find("{")
                if pos < 0:
                    return
            else:
                pos = k


def encode_weak(spans, weak, label_to_id, radius):
    """Partial BIO: -100 everywhere the rules have no opinion.

    0 = O, 2k+1 = B-label_k, 2k+2 = I-label_k, matching phase2_data.encode_bio.
    """
    tags = [-100] * len(spans)
    for a, b, lb in weak:
        k = label_to_id.get(lb)
        if k is None:
            continue
        inside = [i for i, (ts, te) in enumerate(spans) if ts < b and te > a]
        for j, i in enumerate(inside):
            tags[i] = (2 * k + 1) if j == 0 else (2 * k + 2)
    marked = [i for i, t in enumerate(tags) if t >= 0]
    for i in marked:
        for j in range(max(0, i - radius), min(len(tags), i + radius + 1)):
            if tags[j] < 0:
                tags[j] = 0
    return tags


def collect_weak(tok, labels, label_to_id, a):
    """Label `--n-docs` Supreme Court judgments with the validated rules.

    Cached: labelling is pure CPU and takes about 45 minutes per thousand
    documents, so doing it again inside a GPU session would burn compute units
    producing a file we already have.
    """
    if a.weak_cache:
        # Pre-built caches: rule-labelled corpus windows, InLegalNER windows,
        # or both concatenated. Mixing them is the point -- InLegalNER supplies
        # real annotation for ten labels and is silent on A.COUNSEL and
        # R.COUNSEL, which is exactly where the rules are strongest.
        out = []
        for f in a.weak_cache:
            d = torch.load(f, weights_only=False)
            log("  %s -> %d windows" % (f, len(d)))
            out += d
        log("  %d windows total" % len(out))
        return out
    cache = "%s/results/weak_%d_r%d_p%02d_%s.pt" % (
        P2, a.n_docs, a.radius, int(a.min_prec * 100),
        "".join(x[6] for x in a.sources))
    if os.path.exists(cache):
        d = torch.load(cache, weights_only=False)
        log("  reusing %s (%d windows)" % (cache, len(d)))
        return d
    from huggingface_hub import hf_hub_download
    # The official corpus has four sources and only one has ever been read.
    # train_high_courts is the LARGEST of them (450M tokens against the
    # Supreme Court's 400M) and Phase-2 documents are not all Supreme Court --
    # one of the three inspected was 'HIGH COURT OF MADHYA PRADESH BENCH AT
    # INDORE, Writ Petition No.15591/2020'. High Court and tribunal judgments
    # carry the same caption, BENCH and counsel structure the rules key on, so
    # they are in-distribution weak-supervision material that costs nothing.
    # Documents are taken round-robin so one source cannot dominate.
    paths = [hf_hub_download(REPO, "Phase_1_Pretraining/Train/%s.json" % src,
                             repo_type="dataset") for src in a.sources]
    for src, pth in zip(a.sources, paths):
        log("  %-22s %s" % (src, pth))
    streams = [stream_records(pth) for pth in paths]
    names = list(a.sources)   # deleted in step with streams, or the labels drift
    data, seen, kept = [], 0, 0
    n_spans = 0
    by_label = {l: 0 for l in labels}
    by_source = {src: 0 for src in a.sources}
    t0 = time.time()
    si = -1
    while streams:
        si = (si + 1) % len(streams)
        try:
            rec = next(streams[si])
        except StopIteration:
            log("  %s exhausted" % names[si])
            del streams[si]
            del names[si]
            si -= 1
            continue
        src = names[si]
        seen += 1
        t = rec.get("text")
        if not isinstance(t, str):
            continue
        # Match the Phase-2 distribution: median gold document is 26,640
        # characters. Very short records are orders and notices, not judgments.
        if not (a.min_chars <= len(t) <= a.max_chars):
            continue
        p = proposals(t, None, a.min_prec, True)
        weak = [(x, y, l) for l, v in p.items() for x, y in v]
        if len(weak) < a.min_spans:
            continue
        ids, spans, _ = align(tok, t)
        if not ids:
            continue
        tags = encode_weak(spans, weak, label_to_id, a.radius)
        if sum(1 for x in tags if x > 0) < a.min_spans:
            continue
        for st, wi, wt, sk in windows(ids, tags, a.ctx, a.ctx // 2):
            # Only windows that actually contain a rule span are kept: a
            # window of pure -100 contributes no gradient and would just slow
            # the pass down. Character spans are dropped -- weak windows are
            # never decoded, only trained on, and keeping them would make the
            # cache many times larger.
            if wt is not None and any(x > 0 for x in wt):
                data.append(dict(ids=wi, tags=wt))
        for _, _, l in weak:
            by_label[l] = by_label.get(l, 0) + 1
        n_spans += len(weak)
        by_source[src] += 1
        kept += 1
        if kept % 250 == 0:
            log("    %d kept of %d read, %d windows, %.0fs"
                % (kept, seen, len(data), time.time() - t0))
        if kept >= a.n_docs:
            break
    log("  %d documents kept of %d read; %d weak spans, %d windows"
        % (kept, seen, n_spans, len(data)))
    log("  documents per source: " + "  ".join(
        "%s=%d" % (k, v) for k, v in by_source.items()))
    log("  weak spans per label: " + "  ".join(
        "%s=%d" % (l, by_label.get(l, 0))
        for l in sorted(labels, key=lambda x: -by_label.get(x, 0))))
    sup = sum(sum(1 for x in c["tags"] if x >= 0) for c in data)
    tot = sum(len(c["tags"]) for c in data)
    log("  supervised positions: %d of %d (%.1f%%) -- the rest is -100, "
        "because the rules cover no COURT and no PREC and claiming O there "
        "would unteach two working labels"
        % (sup, tot, 100 * sup / max(tot, 1)))
    os.makedirs("%s/results" % P2, exist_ok=True)
    torch.save(data, cache)
    log("  cached to %s" % cache)
    return data


@torch.no_grad()
def emissions2(model, data, device, bs, n_tags):
    """Per-document log-probs under two window-combination rules.

    avg  every window containing a token votes equally -- what predict() and
         phase2_ensemble.emissions() have always done
    ctx  each token takes the prediction from the ONE window where it had the
         most context: the largest left context for a causal trunk, the most
         central position for a bidirectional one

    The reason for `ctx`: with 50% stride every token sits in the second half
    of one window and the first half of the next. For a causal trunk the
    first-half copy is predicted from as little as zero tokens of left context,
    and averaging it in dilutes the well-informed copy. Inference-only, so both
    are saved and the choice is made offline in phase2_decode.py.
    """
    model.eval()
    causal = not (model.kind == "ed" or model.bi)
    size = {c["doc"]: c["n"] for c in data}
    acc = {d: torch.zeros(n, n_tags) for d, n in size.items()}
    cnt = {d: torch.zeros(n, 1) for d, n in size.items()}
    best = {d: torch.zeros(n, n_tags) for d, n in size.items()}
    bsc = {d: torch.full((n,), -1.0) for d, n in size.items()}
    for idx, _, pad, chunk in batches(data, bs, device, False, n_tags):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = model(idx, pad).float()
        lg = F.log_softmax(lg, -1).cpu()
        for i, c in enumerate(chunk):
            d, k = c["doc"], c.get("skip", 0)
            n = len(c["ids"]) - k
            st = c["start"]
            x = lg[i, k:k + n]
            acc[d][st:st + n] += x
            cnt[d][st:st + n] += 1
            pos = torch.arange(n, dtype=torch.float32)
            sc = pos if causal else torch.minimum(pos, n - 1 - pos)
            upd = sc > bsc[d][st:st + n]
            blk = best[d][st:st + n]
            blk[upd] = x[upd]
            bsc[d][st:st + n] = torch.maximum(bsc[d][st:st + n], sc)
    model.train()
    return {"avg": {d: (acc[d] / cnt[d].clamp(min=1)).half() for d in acc},
            "ctx": {d: best[d].half() for d in best}}


def save_spans(path, data):
    if not os.path.exists(path):
        torch.save({c["doc"]: c["spans"] for c in data}, path)


def fit(model, weak, gold_tr, n_tags, a, device, tag,
        ev=None, truth=None, labels=None, text=None, gold_epochs=None):
    """Weak pretraining, then gold fine-tuning, with separate schedules.

    The gold phase keeps its BEST epoch, not its last. `phase2_ner.py` has
    always done this -- on fold2 its best was epoch 4 of 8 -- but the first
    version of this script ran all eight and took whatever came out. Gold loss
    reaches 0.0003 on 70 documents, which is memorisation, and every epoch
    past the optimum erodes the weak-pretrained representation that the whole
    method exists to build. When `ev` is given the model is rewound to its
    best epoch; the submit path has no held-out fold, so it is given the epoch
    count the CV folds chose.
    """
    best_f1, best_state, best_ep = -1.0, None, a.epochs

    def run(data, epochs, lr, w, name):
        nonlocal best_f1, best_state, best_ep
        if not data or epochs <= 0:
            return
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95),
                                weight_decay=0.01)
        steps = epochs * max(1, math.ceil(len(data) / a.batch))
        warm, step = max(1, int(0.1 * steps)), 0
        for ep in range(epochs):
            last = 0.0
            for idx, tg, pad, _ in batches(data, a.batch, device, True, n_tags):
                cur = lr * (step / warm if step < warm else
                            0.5 * (1 + math.cos(math.pi * (step - warm)
                                                / max(1, steps - warm))))
                for g in opt.param_groups:
                    g["lr"] = cur
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = F.cross_entropy(
                        model(idx, pad).reshape(-1, n_tags).float(),
                        tg.reshape(-1), weight=w, ignore_index=-100)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                last = loss.item()
            msg = "    %s %s epoch %d/%d  loss %.4f" % (tag, name, ep + 1,
                                                        epochs, last)
            if name == "gold" and ev is not None:
                p = predict(model, ev, labels, text, device, a.batch, n_tags,
                            a.scheme, a.merge_gap)
                f1, _ = strict_macro_f1(p, truth, labels)
                msg += "  macro-F1 %.4f" % f1
                if f1 > best_f1:
                    best_f1, best_ep = f1, ep + 1
                    best_state = {k: v.detach().cpu().clone()
                                  for k, v in model.state_dict().items()}
            log(msg)

    # The weak pass sees far more data and should not overwrite the trunk, so
    # it runs at a lower learning rate than the gold pass that follows.
    wgt = class_weights(weak, n_tags, a.weight_pow, a.weight_cap, device) \
        if weak else None
    run(weak, a.weak_epochs, a.weak_lr, wgt, "weak")
    ggt = class_weights(gold_tr, n_tags, a.weight_pow, a.weight_cap, device)
    run(gold_tr, gold_epochs or a.epochs, a.lr, ggt, "gold")
    if ev is not None and best_state is not None:
        model.load_state_dict(best_state)
        log("    %s gold best epoch %d  (F1 %.4f)" % (tag, best_ep, best_f1))
    return model, best_ep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="M4_LexArcP")
    ap.add_argument("--specs", nargs="*", default=None, metavar="SPEC",
                    help="ensemble members, e.g. M4_LexArcP:full x3. Each is "
                         "weak-pretrained then gold fine-tuned, and their "
                         "log-probabilities are averaged before decoding. "
                         "Overrides --model.")
    ap.add_argument("--weak-cache", nargs="*", default=None, metavar="PT",
                    help="use these prebuilt caches instead of labelling")
    ap.add_argument("--n-docs", type=int, default=2000)
    ap.add_argument("--sources", nargs="*",
                    default=["train_supreme_court", "train_high_courts",
                             "train_tribunals"],
                    help="official corpus sources to label, read round-robin")
    ap.add_argument("--min-chars", type=int, default=4000)
    ap.add_argument("--max-chars", type=int, default=120000)
    ap.add_argument("--min-spans", type=int, default=12)
    ap.add_argument("--radius", type=int, default=24,
                    help="tokens either side of a rule span supervised as O")
    ap.add_argument("--min-prec", type=float, default=0.40)
    ap.add_argument("--weak-epochs", type=int, default=2)
    ap.add_argument("--weak-lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--weight-pow", type=float, default=0.5)
    ap.add_argument("--weight-cap", type=float, default=20.0)
    ap.add_argument("--scheme", default="bio")
    ap.add_argument("--merge-gap", type=int, default=0)
    ap.add_argument("--header", type=int, default=0)
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--skip-cv", action="store_true",
                    help="submit path only; needs --gold-epochs. Lets CV and "
                         "the final fit run as two parallel jobs")
    ap.add_argument("--gold-epochs", type=int, default=None,
                    help="final-fit gold epochs, overriding the CV mean")
    ap.add_argument("--gold-extra", nargs="*", default=None, metavar="PT",
                    help="partially-labelled windows (phase2_pseudo.py) "
                         "trained IN the gold phase of the final fit, at the "
                         "gold learning rate -- not in the weak phase, where "
                         "external data stopped transferring (+0.003 on test)")
    ap.add_argument("--cv-folds", nargs="*", default=["fold2", "fold1"],
                    metavar="EVAL_FOLD",
                    help="which held-out folds to run; one per runtime lets "
                         "two GPUs split the CV half")
    ap.add_argument("--infer-div", type=int, default=4,
                    help="saved emissions use windows at stride ctx/N, so "
                         "every token is scored from N window positions and "
                         "the ctx rule can pick one with >= (N-1)/N of a "
                         "window of left context")
    ap.add_argument("--extra-in-cv", action="store_true",
                    help="also add --gold-extra windows to each CV fold's "
                         "gold training")
    ap.add_argument("--emis-dir", default=None,
                    help="per-member emissions are saved here and REUSED on a "
                         "rerun, so a crash costs one member, not the run "
                         "(default results/emis_<tag>)")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    a.tag = a.tag or ("ws%d" % a.n_docs)
    assert not (a.skip_cv and not (a.submit and a.gold_epochs)), \
        "--skip-cv needs --submit and --gold-epochs"
    emis = a.emis_dir or "%s/results/emis_%s" % (P2, a.tag)
    os.makedirs(emis, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log("GPU %s | %s | weak %d docs | radius %d"
        % (torch.cuda.get_device_name(0) if device == "cuda" else "none",
           a.model, a.n_docs, a.radius))

    with open(fetch("Train/labels.txt"), encoding="utf-8") as f:
        labels = [x.strip() for x in f if x.strip()]
    label_to_id = {l: i for i, l in enumerate(labels)}
    n_tags = n_tags_for(labels, a.scheme)
    text, gold = {}, {}
    for fold in ("fold1", "fold2"):
        text.update({str(k): v
                     for k, v in load_json("Train/%s-text.json" % fold).items()})
        gold.update(parse_gold(load_json("Train/%s-gold.json" % fold)))
    split = json.load(open("%s/split.json" % P2))
    tok = BPETokenizer.load("%s/shared/tokenizer/bpe_legal_32k.json" % P1)

    log("")
    log("--- labelling the Supreme Court corpus with the validated rules ---")
    weak = collect_weak(tok, labels, label_to_id, a)

    specs = a.specs or ["%s:full" % a.model]
    json.dump({"specs": specs, "labels": labels, "scheme": a.scheme,
               "tag": a.tag}, open("%s/meta.json" % emis, "w"), indent=1)
    scores, cv_pred, best_eps = [], {}, []
    for tr_fold, ev_fold in (() if a.skip_cv else
                             [(t, e) for t, e in (("fold1", "fold2"),
                                                  ("fold2", "fold1"))
                              if e in a.cv_folds]):
        log("")
        log("--- train %s -> score %s ---" % (tr_fold, ev_fold))
        tr = build(tok, split[tr_fold], text, gold, label_to_id, a.ctx,
                   a.ctx // 2, scheme=a.scheme)
        ev = build(tok, split[ev_fold], text, gold, label_to_id, a.ctx,
                   a.ctx // 2, scheme=a.scheme)
        if a.extra_in_cv:
            # The CV twin of a --gold-extra submit run: the same fold3
            # pseudo-labels join the 35 gold documents, so the stacker sees
            # this system's held-out behaviour, not a stand-in's.
            for f in a.gold_extra or []:
                tr = tr + torch.load(f, weights_only=False)
        ev_inf = build(tok, split[ev_fold], text, gold, label_to_id, a.ctx,
                       a.ctx // a.infer_div, scheme=a.scheme)
        save_spans("%s/spans_%s.pt" % (emis, ev_fold), ev)
        truth = {d: [list(x) for x in gold.get(d, [])] for d in split[ev_fold]}
        pool = None
        for i, spec in enumerate(specs):
            ef = "%s/cv_%s_%d.pt" % (emis, ev_fold, i)
            if os.path.exists(ef):
                saved = torch.load(ef, weights_only=False)
                bep, dname = saved["best_ep"], spec
                log("    %s: reusing %s" % (spec, ef))
            else:
                model, dname, _ = make(spec, labels, n_tags, False, device)
                _, bep = fit(model, weak, tr, n_tags, a, device, dname,
                             ev=ev, truth=truth, labels=labels, text=text)
                saved = emissions2(model, ev_inf, device, a.batch, n_tags)
                saved.update(best_ep=bep, spec=spec)
                torch.save(saved, ef)
                del model
                torch.cuda.empty_cache()
            best_eps.append(bep)
            em = {d: v.float() for d, v in saved["avg"].items()}
            s1, _ = strict_macro_f1(decode_all(em, ev, labels, text, a.scheme),
                                    truth, labels)
            log("    single %s: strict %.4f" % (dname, s1))
            pool = em if pool is None else {k: pool[k] + em[k] for k in pool}
        pool = {k: v / len(specs) for k, v in pool.items()}
        pred = decode_all(pool, ev, labels, text, a.scheme)
        f1, per = strict_macro_f1(pred, truth, labels)
        ov = sum(v[6] for v in per.values()) / len(labels)
        log("  strict %.4f  overlap %.4f" % (f1, ov))
        for l in labels:
            pr, rc, ff, tp, fp, fn, rf = per[l]
            log("    %-11s P=%.3f R=%.3f F1=%.3f | overlap=%.3f  "
                "tp=%4d fp=%4d fn=%4d" % (l, pr, rc, ff, rf, tp, fp, fn))
        boundary_report(pred, truth, text, labels, n_examples=4)
        scores.append(f1)
        cv_pred.update({d: [list(x) for x in v] for d, v in pred.items()})
        torch.cuda.empty_cache()

    cv = sum(scores) / len(scores) if scores else None
    if scores:
        log("")
        log("CV %.4f  (folds %s)" % (cv, " / ".join("%.4f" % s for s in scores)))
        log("  baseline M4 full fine-tuning, no weak data: 0.3258")
        os.makedirs("%s/results" % P2, exist_ok=True)
        cvp = "%s/results/cvpred_%s%s.json" % (
            P2, a.tag, "" if len(scores) == 2 else "_" + a.cv_folds[0])
        json.dump(cv_pred, open(cvp, "w", encoding="utf-8"), ensure_ascii=False)
        log("wrote %s" % cvp)
        log("CV gold best epochs %s -> final fit would use %d"
            % (best_eps, max(1, round(sum(best_eps) / len(best_eps)))))

    if a.submit:
        log("")
        log("--- final: weak pretraining, then all 70 gold documents ---")
        all_docs = split["fold1"] + split["fold2"]
        tr = build(tok, all_docs, text, gold, label_to_id, a.ctx, a.ctx // 2,
                   scheme=a.scheme)
        for f in a.gold_extra or []:
            extra = torch.load(f, weights_only=False)
            log("  gold phase += %s (%d windows, alongside %d gold)"
                % (f, len(extra), len(tr)))
            tr = tr + extra
        test_text = {str(k): v
                     for k, v in load_json("Test/fold3-text.json").items()}
        te = build(tok, sorted(test_text), test_text, {}, label_to_id, a.ctx,
                   a.ctx // 2, with_tags=False, scheme=a.scheme)
        pool = None
        ge = a.gold_epochs or (max(1, round(sum(best_eps) / len(best_eps)))
                               if best_eps else a.epochs)
        log("  gold epochs: %d  (CV best epochs %s%s)"
            % (ge, best_eps, ", overridden" if a.gold_epochs else ""))
        te_inf = build(tok, sorted(test_text), test_text, {}, label_to_id,
                       a.ctx, a.ctx // a.infer_div, with_tags=False,
                       scheme=a.scheme)
        save_spans("%s/spans_test.pt" % emis, te)
        for i, spec in enumerate(specs):
            ef = "%s/test_%d_e%d.pt" % (emis, i, ge)
            if os.path.exists(ef):
                saved = torch.load(ef, weights_only=False)
                log("    %s: reusing %s" % (spec, ef))
            else:
                model, dname, _ = make(spec, labels, n_tags, False, device)
                fit(model, weak, tr, n_tags, a, device, dname, gold_epochs=ge)
                saved = emissions2(model, te_inf, device, a.batch, n_tags)
                saved.update(gold_epochs=ge, spec=spec)
                torch.save(saved, ef)
                log("    saved %s" % ef)
                del model
                torch.cuda.empty_cache()
            em = {d: v.float() for d, v in saved["avg"].items()}
            pool = em if pool is None else {k: pool[k] + em[k] for k in pool}
        pool = {k: v / len(specs) for k, v in pool.items()}
        pred = decode_all(pool, te, labels, test_text, a.scheme)
        n = sum(len(v) for v in pred.values())
        log("  %d entities (%.1f/doc; training 119.9)" % (n, n / len(pred)))
        os.makedirs("%s/submissions" % P2, exist_ok=True)
        pj = "%s/submissions/prediction_%s.json" % (P2, a.tag)
        json.dump(pred, open(pj, "w", encoding="utf-8"), ensure_ascii=False)
        import zipfile
        zp = "%s/submissions/submission_%s.zip" % (P2, a.tag)
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(pj, "prediction.json")
        log("  WROTE %s" % zp)

    json.dump({"tag": a.tag, "model": a.model, "specs": specs,
               "n_docs": a.n_docs,
               "radius": a.radius, "sources": a.sources,
               "weak_epochs": a.weak_epochs,
               "weak_lr": a.weak_lr, "cv": cv, "folds": scores,
               "gold_best_epochs": best_eps, "gold_epochs": a.gold_epochs,
               "emis_dir": emis},
              open("%s/results/%s%s.json" % (
                  P2, a.tag, "_submit" if a.skip_cv else
                  ("" if len(a.cv_folds) == 2 else "_cv" + a.cv_folds[0])),
                  "w"), indent=2)


if __name__ == "__main__":
    main()
