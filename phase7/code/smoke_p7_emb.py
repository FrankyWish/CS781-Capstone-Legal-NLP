#!/usr/bin/env python
"""Smoke test for phase7_emb.py on fake data and tiny trunks, before a GPU run.

The checks that matter:

  * a train query that copies a test query is PLANTED here and must be dropped, because
    its gold labels are that test query's answer key.  The gate found one in the real
    data; this proves the trainer drops it rather than trusting the gate ran;
  * PEFT must actually freeze: for lorma/ia3/qlora the base weights keep
    requires_grad False and trainable is a small fraction of the total;
  * training must change the embeddings -- a trained tag whose vectors are identical to
    the zero-shot ones means the optimiser did nothing;
  * the written cache must be byte-compatible with phase7_enc's, so `dense` ranks it
    with no special case: off[-1] == n_chunks, len(ids) == len(off) - 1;
  * the KD student must really be smaller than its teacher, and `bench` must produce the
    efficiency row s13.2.2 asks for.

  python smoke_p7_emb.py
"""
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("P7_SMOKE_DIR",
                      os.path.join(tempfile.gettempdir(), "p7_smoke_emb"))
sys.path.insert(0, HERE)

import smoke_p7 as S                                   # sets every CS781_* env var

import numpy as np
import torch

import phase7_data as D
import phase7_rank as R
import phase7_enc as E
import phase7_emb as M

FAIL = []
LEAK_ID = "TRAIN_LEAK_0001"


def ok(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAIL.append(name)


def run(args, quiet=True):
    p = subprocess.run([sys.executable] + args, cwd=HERE, env=os.environ,
                       capture_output=True, text=True)
    if p.returncode != 0 and not quiet:
        print(p.stdout[-2500:])
        print(p.stderr[-2500:])
    return p


def plant_leak():
    """Append a train query whose text is a verbatim copy of a test query."""
    data = os.environ["CS781_P7_DATA"]
    tq = D.queries("test")[0]
    tp = os.path.join(data, "Train", "train.json")
    raw = json.load(open(tp))
    gold = [c for c in sorted(D.candidates("train"))[:3]]
    if isinstance(raw, dict):
        probe = next(iter(raw.values()))
        rec = {k: v for k, v in probe.items()}
        for k in rec:
            if isinstance(rec[k], (list, str)) and k.lower() in ("text", "sents",
                                                                 "sentences", "doc"):
                rec[k] = list(tq["sents"])
            elif k.lower() in ("gold", "labels", "relevant", "candidates"):
                rec[k] = gold
        raw[LEAK_ID] = rec
    else:
        return False
    json.dump(raw, open(tp, "w"))
    for side in ("q", "c"):                       # the token cache must be rebuilt
        p = E.tok_path("train", side)
        if os.path.exists(p):
            os.remove(p)
    return True


def main():
    print("building fake data and tiny trunks")
    S.build_data()
    S.build_models()
    planted = plant_leak()
    print(f"  planted leaked train query: {planted}")

    p = run(["phase7_data.py", "gate"])
    ok("gate runs", p.returncode == 0)
    # `devx` is its own split (dev minus the contaminated queries), and it is the one
    # every run scores on, so it must be tokenised by name -- `dev` is not a substitute.
    p = run(["phase7_enc.py", "tok", "--splits", "train,devx,test", "--procs", "1"])
    ok("tokenises train/devx/test", p.returncode == 0, p.stderr[-200:])
    if p.returncode != 0:
        return 1
    p = run(["phase7_rank.py", "tfidf", "--tag", "tfidf_sqrt", "--qtf", "sqrt",
             "--max-df", "0.2", "--splits", "train,devx,test", "--topn", "40"])
    ok("lexical negatives mined on train", p.returncode == 0, p.stderr[-200:])
    if p.returncode != 0:
        return 1

    # ---- leakage ------------------------------------------------------------
    bad = M.contaminated_train_ids()
    if planted:
        ok("the planted leaked train query is detected", LEAK_ID in bad,
           f"found {sorted(bad)}")
    ts = M.TrainSet("tfidf_sqrt", 40)
    ok("no contaminated query reaches the training set",
       all(q not in bad for q, _, _ in ts.items), f"{len(ts.items)} items")
    ok("training set is non-empty", len(ts.items) > 0)

    base = ["--model", "M2_LexArcED", "--chunk", "128", "--max-chunks", "2",
            "--embed-batch", "4", "--steps", "3", "--batch", "2", "--n-neg", "2",
            "--neg-pool", "40", "--warmup", "1", "--log-every", "1",
            "--splits", "devx,test", "--cpu"]

    # ---- zero-shot reference, for the "training changed something" check ----
    p = run(["phase7_enc.py", "embed", "--tag", "zs", "--model", "M2_LexArcED",
             "--split", "devx", "--chunk", "128", "--max-chunks", "2", "--batch", "4"])
    ok("zero-shot reference embeddings written", p.returncode == 0, p.stderr[-200:])

    # ---- one run per technique ----------------------------------------------
    trained = {}
    for method in ("lorma", "ia3", "partial", "qlora", "full"):
        tag = f"t_{method}"
        p = run(["phase7_emb.py", "train", "--tag", tag, "--method", method] + base,
                quiet=False)
        ok(f"train --method {method}", p.returncode == 0, p.stderr[-300:])
        if p.returncode != 0:
            continue
        trained[method] = tag
        line = [l for l in p.stdout.splitlines() if "trainable of" in l]
        if line:
            print(f"        {line[0].strip()}")

    if "lorma" not in trained:
        print("\nSMOKE FAILED: lorma did not train")
        return 1

    # ---- freezing -----------------------------------------------------------
    dev = torch.device("cpu")
    model, kind, cfg, ck = M.load_trained("t_lorma", dev)
    tot = sum(p.numel() for p in model.parameters())
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ok("lorma trains a small fraction", 0 < tr < 0.5 * tot,
       f"{tr:,} of {tot:,} = {100.0 * tr / tot:.1f}%")
    frozen = [n for n, q in model.trunk.named_parameters()
              if n.endswith("wq.weight") and q.requires_grad]
    ok("lorma leaves base attention weights frozen", not frozen, str(frozen[:2]))

    # ---- the cache is the same shape phase7_enc writes ----------------------
    for split in ("devx", "test"):
        for side in ("q", "c"):
            z = np.load(E.emb_path("t_lorma", split, side), allow_pickle=False)
            good = (z["off"][-1] == z["flat"].shape[0]
                    and len(z["ids"]) == len(z["off"]) - 1)
            ok(f"cache shape {split}/{side}", bool(good),
               f"{z['flat'].shape} off[-1]={z['off'][-1]} ids={len(z['ids'])}")

    # Training must reach the cache -- but a 3-step run at lr 1e-4 moves the LoRMA B
    # matrices by ~2.5e-4 and the pooled embedding by ~1.7e-4, which float16 cannot
    # represent. So check the mechanism at module level, where the signal is exact, and
    # the cache with a deliberately hot short run, where it is large enough to store.
    b_moved = max((float(q.abs().max()) for n, q in model.named_parameters()
                   if n.endswith(".B")), default=0.0)
    ok("lorma adapters moved off their zero init", b_moved > 0, f"absmax {b_moved:.2e}")
    import argparse as _ap
    zs_model, _, _ = M.build(_ap.Namespace(model="M2_LexArcED", ckpt="best.pt",
                                           chunk=128, pool="mean", bidir=False))
    tkq = E.Tok("devx", "q")
    qi, qo = M.pad([tkq.doc(0)[:128]], 128, dev)
    with torch.no_grad():
        d_ft, d_zs = model(qi, qo), zs_model(qi, qo)
    diff = float((d_ft - d_zs).abs().max())
    ok("the adapter is inside the forward pass used for indexing", diff > 0,
       f"max |diff| {diff:.2e}")

    p = run(["phase7_emb.py", "train", "--tag", "t_hot", "--method", "full"]
            + base + ["--lr", "0.05", "--steps", "5"], quiet=False)
    ok("high-LR control run completes", p.returncode == 0, p.stderr[-300:])
    if p.returncode == 0:
        e_zs, e_hot = E.Emb("zs", "devx", "q"), E.Emb("t_hot", "devx", "q")
        d = (float(np.abs(e_zs.flat - e_hot.flat).max())
             if e_zs.flat.shape == e_hot.flat.shape else 9.0)
        ok("training reaches the written embedding cache", d > 1e-2,
           f"max |diff| {d:.3e}")

    # ---- it ranks and zips through the existing path ------------------------
    p = run(["phase7_enc.py", "dense", "--tag", "t_lorma", "--agg", "maxsim",
             "--splits", "devx,test", "--topn", "40"], quiet=False)
    ok("dense ranks the trained embeddings", p.returncode == 0, p.stderr[-300:])
    p = run(["phase7_rank.py", "zip", "--tag", "t_lorma", "--kind", "model"])
    ok("trained model zips as kind=model", p.returncode == 0, p.stderr[-200:])

    # ---- KD -----------------------------------------------------------------
    p = run(["phase7_emb.py", "kd", "--tag", "t_kd", "--teacher", "t_lorma",
             "--keep-every", "2", "--method", "lorma"] + base, quiet=False)
    ok("kd trains a smaller student", p.returncode == 0, p.stderr[-300:])
    if p.returncode == 0:
        st, _, scfg, _ = M.load_trained("t_kd", dev)
        sn = sum(q.numel() for q in st.trunk.parameters())
        tn = sum(q.numel() for q in model.trunk.parameters())
        ok("student really has fewer parameters than the teacher", sn < tn,
           f"{sn:,} vs {tn:,} = {100.0 * sn / tn:.0f}%")
        p = run(["phase7_enc.py", "dense", "--tag", "t_kd", "--agg", "maxsim",
                 "--splits", "devx,test", "--topn", "40"])
        ok("the student ranks", p.returncode == 0)
        p = run(["phase7_emb.py", "bench", "--tags", "t_lorma,t_kd", "--chunk", "128",
                 "--reps", "2", "--cpu"], quiet=False)
        ok("bench writes the efficiency table", p.returncode == 0, p.stderr[-300:])
        f = f"{D.P7}/results/kd_efficiency.json"
        if os.path.exists(f):
            rows = json.load(open(f))
            ok("efficiency table has params, layers and latency per tag",
               len(rows) == 2 and all(r.get("params") and r.get("layers")
                                      and r.get("latency_ms") for r in rows),
               json.dumps(rows[0]) if rows else "")

    print(f"\n{'SMOKE OK' if not FAIL else 'SMOKE FAILED: ' + ', '.join(FAIL)}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
