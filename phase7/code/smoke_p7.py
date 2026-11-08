#!/usr/bin/env python3
"""Local smoke test for CS-781 Phase 7, on fake data and tiny trunks.

Builds a fake Phase_7_PCR tree that carries every trap the real data might:
  * a dev query that is a verbatim copy of a test query  (Phases 4 and 5 shipped this)
  * a query that is also a candidate in its own pool     (self-match)
  * gold labels that are not in the split's own pool      (unreachable, ceiling < 1)
  * masked citation tokens in the queries only
  * train/test candidate-pool overlap and a popular, often-cited candidate
Then runs the whole stage-1 pipeline and asserts:
  * our Micro-F1@7 equals spec/evaluation_scripts/evaluate_phase7.py to 1e-12
  * F1 = 2*hits/(7Q + sum|g|)                       (the constant-denominator claim)
  * enc_forward reproduces LexArcED.encode() when nothing is padded
  * a submission zip is flat, exactly 7 unique in-pool IDs per test query, and the
    official validator accepts it
  * every step is skip-if-done on a re-run

    python smoke_p7.py [--keep]          # --keep reuses the previous work dir

Needs phase1/code (the Phase-1 model + tokenizer source) and
spec/evaluation_scripts/evaluate_phase7.py, both inside this repo.  Set CS781_LIB_SRC
if phase1/code lives elsewhere, P7_SMOKE_DIR to choose the scratch directory.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import random
import shutil
import subprocess
import sys
import zipfile

import tempfile

CODE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(CODE, "..", ".."))
LIB = os.environ.get("CS781_LIB_SRC", os.path.join(REPO, "phase1", "code"))
WORK = os.environ.get("P7_SMOKE_DIR",
                      os.path.join(tempfile.gettempdir(), "cs781_p7_smoke"))
DATA = os.path.join(WORK, "Phase_7_PCR")
P7 = os.path.join(WORK, "phase7")
P1 = os.path.join(WORK, "phase1")
V = 900
FAIL = []

# the phase7 modules read these at import time, so they are set before any import
os.environ.update(CS781_P7_DATA=DATA, CS781_P7=P7, CS781_P1=P1, CS781_LIB=LIB,
                  CS781_TOKENIZER=os.path.join(P1, "tok.json"), P7_TOPN="25")


def ok(name, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"   {detail}" if detail else ""),
          flush=True)
    if not cond:
        FAIL.append(name)


def run(args, env=None, expect=0):
    e = dict(os.environ)
    e.update(dict(CS781_P7_DATA=DATA, CS781_P7=P7, CS781_P1=P1, CS781_LIB=LIB,
                  CS781_TOKENIZER=os.path.join(P1, "tok.json"),
                  PYTHONPATH=LIB + os.pathsep + CODE, P7_TOPN="25"))
    e.update(env or {})
    r = subprocess.run([sys.executable] + args, cwd=CODE, env=e, capture_output=True,
                       text=True)
    if r.returncode != expect:
        print(r.stdout[-6000:])
        print(r.stderr[-6000:])
        raise SystemExit(f"FAILED: {' '.join(args)} -> {r.returncode}")
    return r.stdout + r.stderr


# ----------------------------------------------------------------- fake data --
LEGAL = ("appellant respondent magistrate tribunal section article petition appeal "
         "judgment order decree plaintiff defendant evidence witness testimony statute "
         "limitation jurisdiction arbitration conviction acquittal sentence bail "
         "compensation negligence contract possession tenancy eviction injunction "
         "mandamus certiorari writ constitution amendment schedule proviso clause").split()


def sentence(rng, n=14, topic=None):
    w = [rng.choice(LEGAL) for _ in range(n)]
    if topic:
        w[2:2 + len(topic)] = topic
    return " ".join(w).capitalize() + "."


def doc(rng, n_sent=30, topic=None, mask=False):
    s = [sentence(rng, topic=topic if rng.random() < 0.4 else None) for _ in range(n_sent)]
    if mask:
        s.insert(len(s) // 2, "The court relied on <CITATION> and on <CITATION> .")
    return s


def build_data():
    rng = random.Random(781)
    os.makedirs(DATA, exist_ok=True)
    topics = [[f"topic{i}a", f"topic{i}b", f"topic{i}c"] for i in range(14)]

    def pool(prefix, n, shared=()):
        """Candidate pool; `shared` IDs are reused verbatim across splits."""
        out = {}
        for i in range(n):
            t = topics[i % len(topics)]
            out[f"{prefix}{i:04d}"] = {"text": doc(rng, rng.randint(14, 40), t)}
        for cid, rec in shared:
            out[cid] = rec
        return out

    shared = [(f"SH{i:04d}", {"text": doc(rng, 25, topics[i % len(topics)])}) for i in range(6)]
    pools = {"train": pool("TR", 40, shared), "dev": pool("DV", 34, shared),
             "test": pool("TE", 34, shared)}
    popular = "SH0000"                       # cited by many train queries -> the prior

    def qset(prefix, n, pool_ids, labelled=True, start=0):
        qs = {}
        for i in range(start, start + n):
            t = topics[i % len(topics)]
            same = [c for c in pool_ids if c.startswith(prefix[:2]) or c.startswith("SH")]
            gold = [c for c in same if True][i % max(len(same) - 9, 1):][:rng.randint(1, 9)]
            if labelled and rng.random() < 0.5:
                gold = list(dict.fromkeys(gold + [popular]))
            rec = {"text": doc(rng, rng.randint(20, 46), t, mask=True)}
            if labelled:
                rec["relevant_candidates"] = gold
            qs[f"{prefix}{i:04d}"] = rec
        return qs

    train_q = qset("QTR", 26, list(pools["train"]))
    dev_q = qset("QDV", 20, list(pools["dev"]))
    test_q = qset("QTE", 16, list(pools["test"]), labelled=False)

    # trap 1: a dev query that is a verbatim copy of a test query
    first_test = sorted(test_q)[0]
    dev_q["QDV9001"] = {"text": list(test_q[first_test]["text"]),
                        "relevant_candidates": [sorted(pools["dev"])[0]]}
    # trap 2: a dev query that is also a candidate in its own pool, by ID and by text
    selfq = sorted(dev_q)[0]
    pools["dev"][selfq] = {"text": list(dev_q[selfq]["text"])}
    dev_q[selfq]["relevant_candidates"] = list(dict.fromkeys(
        dev_q[selfq]["relevant_candidates"] + [sorted(pools["dev"])[1]]))
    # trap 3: an unreachable gold label (not in the dev pool at all)
    dev_q[sorted(dev_q)[2]]["relevant_candidates"].append("NOT_IN_POOL_0001")
    # trap 4: a dev query with no golds at all
    dev_q[sorted(dev_q)[3]]["relevant_candidates"] = []

    for split, qs, cs in (("Train", train_q, pools["train"]), ("Dev", dev_q, pools["dev"]),
                          ("Test", test_q, pools["test"])):
        d = os.path.join(DATA, split)
        os.makedirs(d, exist_ok=True)
        low = split.lower()
        with open(os.path.join(d, f"{low}.json"), "w", encoding="utf-8") as f:
            json.dump(qs, f)
        with open(os.path.join(d, f"{low}_candidates.json"), "w", encoding="utf-8") as f:
            json.dump(cs, f)
    print(f"  fake data: {len(train_q)}/{len(dev_q)}/{len(test_q)} queries, pools "
          f"{len(pools['train'])}/{len(pools['dev'])}/{len(pools['test'])}")
    return pools, dev_q, test_q


# --------------------------------------------------------- tokenizer + trunks --
def build_models():
    sys.path.insert(0, LIB)
    import torch
    from cs781_bpe import BPETokenizer
    from cs781_model import Config, LexArcD
    from cs781_model_ed import EDConfig, LexArcED
    os.makedirs(P1, exist_ok=True)
    tp = os.path.join(P1, "tok.json")
    if not os.path.exists(tp):
        texts = []
        for split in ("Train", "Dev", "Test"):
            for nm in os.listdir(os.path.join(DATA, split)):
                with open(os.path.join(DATA, split, nm), encoding="utf-8") as f:
                    for v in json.load(f).values():
                        texts.append(" ".join(v["text"]))
        t = BPETokenizer(vocab_size=V)
        t.train_from_counts(t.count_chunks(texts), min_freq=1, verbose=False)
        t.save(tp)
        print(f"  tokenizer: {t.vocab_size} types, specials {t.specials}")
    tok = BPETokenizer.load(tp)
    vs = tok.vocab_size

    def save(name, cfg, model):
        d = os.path.join(P1, name, "ckpt")
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, "best.pt")
        if not os.path.exists(p):
            torch.save(dict(cfg=dataclasses.asdict(cfg), model=model.state_dict()), p)
        return p

    ed = EDConfig(vocab_size=vs, d_model=64, n_enc_layer=2, n_dec_layer=2, n_head=4,
                  n_kv_head=2, ffn_hidden=128, window=256, split_points=(96, 128, 160))
    save("M2_LexArcED", ed, LexArcED(ed))
    dec = Config(vocab_size=vs, d_model=64, n_layer=2, n_head=4, n_kv_head=2,
                 ffn_hidden=128, ctx=256, n_register=2)
    save("M1_SC", dec, LexArcD(dec))
    print(f"  tiny trunks saved under {P1}")
    return vs


def check_enc_forward():
    """enc_forward must equal LexArcED.encode() when nothing is padded."""
    sys.path.insert(0, LIB)
    sys.path.insert(0, CODE)
    import torch
    import phase7_enc as E
    lm, kind, cfg = E.load_lm("M2_LexArcED")
    lm.eval()
    idx = torch.randint(0, cfg.vocab_size, (3, 96))
    okm = torch.ones(3, 96, dtype=torch.bool)
    with torch.no_grad():
        a = lm.encode(idx)
        b = E.enc_forward(lm, cfg, idx, okm)
    d = (a - b).abs().max().item()
    ok("enc_forward == LexArcED.encode (unpadded)", d < 1e-4, f"max |diff| {d:.2e}")
    # a padded row must give the same vector for its real part as the row alone
    short = idx[:1, :40]
    okm2 = torch.zeros(1, 96, dtype=torch.bool)
    okm2[0, :40] = True
    padded = idx[:1].clone()
    with torch.no_grad():
        v_pad = E.pool(E.enc_forward(lm, cfg, padded, okm2).float(), okm2, "mean")
        v_raw = E.pool(E.enc_forward(lm, cfg, short, torch.ones(1, 40, dtype=torch.bool)).float(),
                       torch.ones(1, 40, dtype=torch.bool), "mean")
    d2 = (v_pad - v_raw).abs().max().item()
    ok("padding does not leak into the pooled vector", d2 < 1e-4, f"max |diff| {d2:.2e}")


def check_metric_algebra():
    sys.path.insert(0, CODE)
    import phase7_data as D
    gold = {"a": ["1", "2", "3"], "b": ["4"], "c": ["5", "6", "7", "8", "9", "1", "2", "3"]}
    pred = {"a": ["1", "2", "x", "y", "z", "w", "v"], "b": ["4", "a", "b", "c", "d", "e", "f"],
            "c": ["5", "6", "7", "8", "9", "1", "q"]}
    s = D.score_ranking(pred, gold)
    hits = 2 + 1 + 6
    den = 7 * 3 + (3 + 1 + 8)
    ok("F1@7 == 2*hits/(7Q + sum|g|)", abs(s["f1"] - 2 * hits / den) < 1e-12,
       f"{s['f1']:.6f} vs {2 * hits / den:.6f}, hits {s['hits']}")
    ok("denominator is prediction-independent", s["denominator"] == den, str(s["denominator"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()
    if os.path.exists(WORK) and not a.keep:
        shutil.rmtree(WORK)
    os.makedirs(P7, exist_ok=True)
    print("== fake data ==")
    build_data()
    print("== tiny tokenizer + trunks ==")
    build_models()
    print("== unit checks ==")
    check_metric_algebra()
    check_enc_forward()

    print("== gate ==")
    out = run(["phase7_data.py", "gate", "--sample", "10"])
    ok("gate found the dev/test query copy", "dev queries copying a test query 1" in out
       or "1 dev queries are copies" in out, "")
    ok("gate found the self-match", "test queries are also test candidates" in out
       and "self-match" in out)
    ok("gate found unreachable gold labels", "NOT in the split" in out)
    ok("gate found the citation mask", "angle_citation" in out and "masked" in out.lower())
    ok("gate printed a ceiling", "CEILING Micro-F1@7" in out)
    ok("gate printed the prior", "most-cited train candidates score" in out)
    g = json.load(open(os.path.join(P7, "results", "gate.json")))
    ok("gate ceiling < 1 with >7-gold queries", 0 < g["dev_gold"]["ceiling_f1"] < 1.0,
       f"{g['dev_gold']['ceiling_f1']:.4f}")
    ok("devx dropped the contaminated query",
       json.load(open(os.path.join(P7, "results", "dev_contaminated.json")))["n"] == 1)

    print("== lexical ==")
    out = run(["phase7_rank.py", "bm25", "--sweep"])
    ok("bm25 sweep chose a config", "BEST bm25" in out)
    out = run(["phase7_rank.py", "bm25", "--tag", "bm25", "--use-sweep",
               "--splits", "devx,test", "--topn", "25"])
    ok("bm25 wrote dev + test rankings",
       os.path.exists(f"{P7}/results/bm25/devx_rank.json")
       and os.path.exists(f"{P7}/results/bm25/test_rank.json"))
    out = run(["phase7_rank.py", "tfidf", "--tag", "tfidf", "--splits", "devx,test",
               "--topn", "25"])
    ok("tfidf ran", os.path.exists(f"{P7}/results/tfidf/devx_rank.json"))
    # a query must never retrieve itself
    rank = json.load(open(f"{P7}/results/bm25/devx_rank.json"))
    ok("no query retrieves itself", all(q not in v for q, v in rank.items()))

    print("== official scorer agreement ==")
    sys.path.insert(0, CODE)
    for m in list(sys.modules):
        if m.startswith("phase7"):
            del sys.modules[m]
    import phase7_data as D
    import phase7_rank as R
    pred = {q: v[:7] for q, v in rank.items()}
    s = R.evaluate(rank, "devx")
    pj = os.path.join(P7, "results", "bm25", "dev_top7.json")
    D.atomic_json(pj, pred)
    off = D.official_score(pj, "devx")
    # the evaluator prints 10 decimals, so 1e-9 is the tightest meaningful tolerance
    ok("our F1@7 == the official evaluator", off is not None and abs(off - s["f1"]) < 1e-9,
       f"ours {s['f1']:.10f} official {off}")
    ok("bm25 beats random on fake data", s["hits"] > 0, D.brief(s))

    print("== dense (tiny trunks) ==")
    run(["phase7_enc.py", "tok", "--splits", "devx,test", "--procs", "1"])
    run(["phase7_enc.py", "embed", "--tag", "dense_m2", "--model", "M2_LexArcED",
         "--split", "devx,test", "--chunk", "128", "--max-chunks", "2", "--batch", "4"])
    out = run(["phase7_enc.py", "dense", "--tag", "dense_m2", "--agg", "mean",
               "--splits", "devx,test", "--topn", "25"])
    ok("dense/mean ran", "dense_m2/devx" in out)
    out = run(["phase7_enc.py", "dense", "--tag", "dense_m2x", "--agg", "maxsim",
               "--splits", "devx", "--topn", "25"], expect=1) if False else None
    # maxsim reads dense_m2's embeddings under its own tag, so copy them
    for sp in ("devx",):
        for side in ("q", "c"):
            shutil.copyfile(f"{P7}/cache/emb_dense_m2_{sp}_{side}.npz",
                            f"{P7}/cache/emb_dense_mx_{sp}_{side}.npz")
    out = run(["phase7_enc.py", "dense", "--tag", "dense_mx", "--agg", "maxsim",
               "--splits", "devx", "--topn", "25"])
    ok("dense/maxsim ran", "dense_mx/devx" in out)
    run(["phase7_enc.py", "embed", "--tag", "dense_m1", "--model", "M1_SC",
         "--split", "devx", "--chunk", "128", "--max-chunks", "2", "--batch", "4"])
    out = run(["phase7_enc.py", "dense", "--tag", "dense_m1", "--agg", "mean",
               "--splits", "devx", "--topn", "25"])
    ok("decoder trunk (M1) embeds and ranks", "dense_m1/devx" in out)
    run(["phase7_enc.py", "embed", "--tag", "dense_m1b", "--model", "M1_SC", "--bidir",
         "--split", "devx", "--chunk", "128", "--max-chunks", "2", "--batch", "4"])
    out = run(["phase7_enc.py", "dense", "--tag", "dense_m1b", "--splits", "devx",
               "--topn", "25"])
    ok("--bidir path runs on a decoder trunk", "dense_m1b/devx" in out)

    print("== fusion + submission ==")
    out = run(["phase7_rank.py", "fuse", "--tag", "fuse", "--members", "bm25,tfidf,dense_m2",
               "--splits", "devx,test", "--topn", "25"])
    ok("fusion picked a mode and weights", "BEST fusion" in out)
    out = run(["phase7_rank.py", "zip", "--tag", "fuse", "--kind", "hybrid"])
    ok("submission validated", "validated=True" in out, out.strip().splitlines()[-1][:120])
    zp = f"{P7}/submissions/fuse/submission.zip"
    with zipfile.ZipFile(zp) as z:
        ok("zip is flat with one file", z.namelist() == ["prediction.json"], str(z.namelist()))
        p = json.loads(z.read("prediction.json"))
    pool = set(json.load(open(os.path.join(DATA, "Test", "test_candidates.json"))))
    tq = set(json.load(open(os.path.join(DATA, "Test", "test.json"))))
    ok("every test query is predicted", set(p) == tq, f"{len(p)} vs {len(tq)}")
    ok("exactly 7 unique in-pool IDs per query",
       all(len(v) == 7 and len(set(v)) == 7 and set(v) <= pool for v in p.values()))
    ok("_upload copy exists", os.path.exists(f"{P7}/submissions/_upload/submission_fuse.zip"))

    print("== re-run is skip-if-done ==")
    out = run(["phase7_enc.py", "tok", "--splits", "devx", "--procs", "1"])
    ok("tok skips existing caches", "exists" in out)
    out = run(["phase7_enc.py", "embed", "--tag", "dense_m2", "--model", "M2_LexArcED",
               "--split", "devx", "--chunk", "128", "--max-chunks", "2"])
    ok("embed skips existing caches", "exists" in out)
    out = run(["phase7_rank.py", "show"])
    ok("show lists the systems", "bm25" in out and "fuse" in out)
    print(out.strip())

    print()
    if FAIL:
        print(f"SMOKE FAILED: {len(FAIL)} check(s): {FAIL}")
        raise SystemExit(1)
    print("SMOKE OK")


if __name__ == "__main__":
    main()
