#!/usr/bin/env python3
"""CS-781 Phase 1 -- build the shared substrate. Resumable, unattended.

Produces, under MyDrive/cs781_capstone/shared/:
    tokenizer/bpe_legal_32k.json   trained BPE (skipped if present)
    corpus/<source>_NNN.bin        uint16 token shards, per source
    corpus/manifest.json           shard inventory + token counts
    corpus/val.bin                 held-out tokens for validation loss
    dev/dev_prompts.json           500 x 512-word prompts, test-shaped
    dev/dev_truth.json             their 128-word continuations
    dev/holdout_hashes.json        excluded from every shard
    corpus/leakage_report.json     are the test prompts in the training files?

Per-source shards rather than one mixed corpus: the training mix is a loader
weight, so the three-phase schedule can change without re-encoding 1B tokens.

Safe to kill and restart. Each shard is written to .tmp and renamed, and the
manifest is updated after every shard, so a dead session loses one shard.

Run:
    python build_corpus.py 2>&1 | tee .../shared/corpus/build.log
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import time

ROOT = os.environ.get("CS781_ROOT", "/content/drive/MyDrive/cs781_capstone")
LIB = f"{ROOT}/shared/lib"
sys.path.insert(0, LIB)

import numpy as np
from cs781_bpe import BPETokenizer

# ----------------------------------------------------------------- config ---
VOCAB = 32768
MIN_WORDS = 900           # below this a document cannot fill a 1024-token
                          # window and is usually a case header, not prose
SHARD_TOKENS = 50_000_000
N_DEV = 500
DEV_MIX = {"train_supreme_court": 0.90, "train_high_courts": 0.05,
           "train_acts": 0.03, "train_tribunals": 0.02}   # mirrors the test set
TOKEN_CAP = {                                # how much of each source to encode
    "train_supreme_court": 400_000_000,      # effectively "all of it"
    "train_high_courts":   450_000_000,
    "train_tribunals":     400_000_000,
    "train_acts":          150_000_000,
}
VAL_TOKENS_PER_SOURCE = 1_500_000
CACHE_CAP = 3_000_000     # chunk cache entries before a flush, keeps RAM sane
SEED = 1337

SOURCES = list(TOKEN_CAP)
CORPUS = f"{ROOT}/shared/corpus"
DEVDIR = f"{ROOT}/shared/dev"
TOKJSON = f"{ROOT}/shared/tokenizer/bpe_legal_32k.json"
for d in (CORPUS, DEVDIR, f"{ROOT}/shared/tokenizer"):
    os.makedirs(d, exist_ok=True)


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def doc_hash(text):
    return hashlib.sha1(text[:2000].encode("utf-8", "ignore")).hexdigest()[:16]


# ------------------------------------------------------- streaming reader ---
def stream_records(path, bufsize=1 << 22):
    """Yield every record of a large JSON array without loading the file.

    Text mode so multi-byte characters are never split at a buffer boundary;
    raw_decode is the C decoder, so this runs far faster than a pure-Python
    incremental parser.
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


def find_paths():
    from huggingface_hub import hf_hub_download
    return {s: hf_hub_download("Exploration-Lab/CS-781-Capstone",
                               f"Phase_1_Pretraining/Train/{s}.json",
                               repo_type="dataset") for s in SOURCES}


# ------------------------------------------------------- step 1: tokenizer ---
def step_tokenizer():
    if os.path.exists(TOKJSON):
        log("tokenizer present, loading")
        return BPETokenizer.load(TOKJSON)
    sample = "/content/bpe_sample.txt"
    if not os.path.exists(sample):
        raise SystemExit(f"{sample} missing -- run cell 3 first")
    texts = open(sample, encoding="utf-8").read().split("\n")
    log(f"training BPE on {len(texts):,} docs / {sum(map(len, texts))/1e6:.0f} MB")
    tok = BPETokenizer(vocab_size=VOCAB, pattern="legal")
    t0 = time.time()
    freq = tok.count_chunks(texts, log_every=200_000)
    log(f"counted {len(freq):,} distinct chunks in {time.time()-t0:.0f}s")
    t0 = time.time()
    tok.train_from_counts(freq, min_freq=2, log_every=2000)
    log(f"merges learned in {(time.time()-t0)/60:.1f} min")
    tmp = TOKJSON + ".tmp"
    tok.save(tmp)
    os.replace(tmp, TOKJSON)
    bad = [d for d in texts[:100] if tok.decode(tok.encode(d)) != d]
    log(f"ROUND-TRIP FAILURES: {len(bad)}  (must be 0)")
    if bad:
        raise SystemExit("tokenizer round-trip failed -- stopping")
    return tok


# --------------------------------------------------------- step 2: leakage ---
def step_leakage(paths):
    out = f"{CORPUS}/leakage_report.json"
    if os.path.exists(out):
        log("leakage report present, skipping")
        return json.load(open(out))
    test = json.load(open(f"{ROOT}/shared/test/test.json", encoding="utf-8"))
    probes = []
    for k in list(test)[:60]:
        w = test[k]["text"].split()
        probes.append((k, " ".join(w[120:160])))      # 40 words from mid-prompt
    rep = {}
    for s, p in paths.items():
        hits, t0 = [], time.time()
        needles = [(k, n.encode("utf-8")) for k, n in probes]
        with open(p, "rb") as f:
            tail = b""
            while needles:
                blk = f.read(1 << 24)
                if not blk:
                    break
                hay = tail + blk
                for k, n in list(needles):
                    if hay.find(n) >= 0:
                        hits.append(k)
                        needles = [(a, b) for a, b in needles if a != k]
                tail = hay[-4096:]
        rep[s] = dict(probed=len(probes), found=len(hits),
                      ids=hits[:20], seconds=round(time.time() - t0, 1))
        log(f"leakage {s:22s} {len(hits)}/{len(probes)} test prompts found "
            f"({time.time()-t0:.0f}s)")
    json.dump(rep, open(out, "w"), indent=2)
    if any(v["found"] for v in rep.values()):
        log("!! TEST PROMPTS APPEAR IN THE TRAINING CORPUS -- see leakage_report.json")
    return rep


# --------------------------------------------------------- step 3: dev set ---
def step_dev(paths, tok):
    if os.path.exists(f"{DEVDIR}/dev_truth.json"):
        log("dev split present, loading holdout hashes")
        return set(json.load(open(f"{DEVDIR}/holdout_hashes.json")))
    rng = random.Random(SEED)
    want = {s: int(round(N_DEV * f)) for s, f in DEV_MIX.items()}
    # Reservoir sampling, so the dev set is spread uniformly over each file
    # rather than over whatever prefix a fixed accept-probability happens to
    # reach. The files are ordered by year and court, so a prefix is biased.
    # HC and Tribunals supply ~5% of dev between them, so capping their scan
    # costs nothing worth the extra minutes of streaming 17 GB.
    SCAN_CAP = {"train_high_courts": 400_000, "train_tribunals": 400_000}
    prompts, truth, hashes = {}, {}, []
    idx = 0
    for s in SOURCES:
        need = want.get(s, 0)
        if need <= 0:
            continue
        cap, seen, res = SCAN_CAP.get(s), 0, []
        for rec in stream_records(paths[s]):
            t = rec.get("text")
            if not isinstance(t, str):
                continue
            wc = rec.get("word_count") or (t.count(" ") + 1)
            if wc < MIN_WORDS:
                continue
            w = t.split()
            if len(w) < 640:
                continue
            seen += 1
            if len(res) < need:
                res.append((w, doc_hash(t)))
            else:
                j = rng.randrange(seen)
                if j < need:
                    res[j] = (w, doc_hash(t))
            if cap and seen >= cap:
                break
        for w, h in res:
            start = rng.randrange(0, len(w) - 640 + 1)
            idx += 1
            k = str(idx)
            prompts[k] = {"text": " ".join(w[start:start + 512])}
            truth[k] = {"text": " ".join(w[start + 512:start + 640])}
            hashes.append(h)                 # hash of the RAW text, matching step_encode
        log(f"dev from {s:22s} {len(res):4d} of {need} wanted "
            f"(reservoir over {seen:,} eligible docs)")
    json.dump(prompts, open(f"{DEVDIR}/dev_prompts.json", "w"), ensure_ascii=False)
    json.dump(truth, open(f"{DEVDIR}/dev_truth.json", "w"), ensure_ascii=False)
    json.dump(hashes, open(f"{DEVDIR}/holdout_hashes.json", "w"))
    log(f"dev split: {len(prompts)} examples, {len(hashes)} documents excluded from training")
    return set(hashes)


# ---------------------------------------------------------- step 4: encode ---
def load_manifest():
    p = f"{CORPUS}/manifest.json"
    return json.load(open(p)) if os.path.exists(p) else {"sources": {}}


def save_manifest(m):
    tmp = f"{CORPUS}/manifest.json.tmp"
    json.dump(m, open(tmp, "w"), indent=2)
    os.replace(tmp, f"{CORPUS}/manifest.json")


def write_shard(name, arr):
    tmp = f"{CORPUS}/{name}.tmp"
    np.asarray(arr, dtype=np.uint16).tofile(tmp)
    os.replace(tmp, f"{CORPUS}/{name}")


def step_encode(paths, tok, holdout):
    man = load_manifest()
    eot = tok.specials["<|endoftext|>"]
    assert tok.vocab_size <= 65536, "uint16 shards need vocab <= 65536"

    for s in SOURCES:
        st = man["sources"].setdefault(s, {"shards": [], "tokens": 0, "done": False,
                                           "docs": 0, "skipped_short": 0, "skipped_holdout": 0})
        if st["done"]:
            log(f"{s}: done ({st['tokens']:,} tokens), skipping")
            continue
        cap = TOKEN_CAP[s]
        log(f"{s}: encoding up to {cap:,} tokens (have {st['tokens']:,})")

        buf, t0, last = [], time.time(), time.time()
        val_left = VAL_TOKENS_PER_SOURCE if not os.path.exists(f"{CORPUS}/val_{s}.bin") else 0
        val = []
        skip_tokens = st["tokens"]          # resume: re-stream and skip what we have
        produced = 0

        for rec in stream_records(paths[s]):
            t = rec.get("text")
            if not isinstance(t, str):
                continue
            wc = rec.get("word_count")
            if wc is None:
                wc = t.count(" ") + 1
            if wc < MIN_WORDS:
                st["skipped_short"] += 1
                continue
            if doc_hash(t) in holdout:
                st["skipped_holdout"] += 1
                continue

            ids = tok.encode(t)
            ids.append(eot)
            produced += len(ids)
            if produced <= skip_tokens:
                continue                      # already in an earlier shard

            if val_left > 0:
                val.extend(ids)
                val_left -= len(ids)
                if val_left <= 0:
                    write_shard(f"val_{s}.bin", val)
                    log(f"{s}: wrote val_{s}.bin ({len(val):,} tokens)")
                    val = []
                continue

            buf.extend(ids)
            st["docs"] += 1

            if len(tok._cache) > CACHE_CAP:
                tok._cache.clear()

            if len(buf) >= SHARD_TOKENS:
                name = f"{s}_{len(st['shards']):03d}.bin"
                write_shard(name, buf[:SHARD_TOKENS])
                st["shards"].append(name)
                st["tokens"] += SHARD_TOKENS
                buf = buf[SHARD_TOKENS:]
                save_manifest(man)
                rate = st["tokens"] / max(time.time() - t0, 1)
                log(f"{s}: {name} | {st['tokens']:,}/{cap:,} tokens | "
                    f"{rate:,.0f} tok/s | {st['docs']:,} docs")

            if st["tokens"] >= cap:
                break

            if time.time() - last > 300:
                last = time.time()
                log(f"{s}: ... {st['tokens'] + len(buf):,} tokens, {st['docs']:,} docs")

        if buf:
            name = f"{s}_{len(st['shards']):03d}.bin"
            write_shard(name, buf)
            st["shards"].append(name)
            st["tokens"] += len(buf)
        st["done"] = True
        save_manifest(man)
        log(f"{s}: DONE {st['tokens']:,} tokens, {st['docs']:,} docs, "
            f"{st['skipped_short']:,} short, {st['skipped_holdout']:,} held out")

    total = sum(v["tokens"] for v in man["sources"].values())
    log(f"corpus total: {total:,} tokens ({total*2/1e9:.2f} GB on disk)")
    return man


# ---------------------------------------------------------- step 5: verify ---
def step_verify(tok, man):
    log("--- verification ---")
    for s, st in man["sources"].items():
        if not st["shards"]:
            continue
        a = np.fromfile(f"{CORPUS}/{st['shards'][0]}", dtype=np.uint16, count=400)
        txt = tok.decode(a.tolist())
        log(f"{s}: {st['tokens']:,} tokens in {len(st['shards'])} shards")
        print(f"    {txt[:220]!r}")
        assert a.max() < tok.vocab_size, f"{s}: token id >= vocab size"
    dp = json.load(open(f"{DEVDIR}/dev_prompts.json", encoding="utf-8"))
    dt = json.load(open(f"{DEVDIR}/dev_truth.json", encoding="utf-8"))
    pw = [len(v["text"].split()) for v in dp.values()]
    tw = [len(v["text"].split()) for v in dt.values()]
    log(f"dev: {len(dp)} examples | prompt words {min(pw)}-{max(pw)} | "
        f"continuation words {min(tw)}-{max(tw)}  (want 512 and 128)")
    tp = [len(tok.encode(v["text"])) for v in list(dp.values())[:200]]
    tp.sort()
    log(f"512 words -> tokens: median {tp[len(tp)//2]}, max {tp[-1]}  "
        f"(ctx 1024 is safe below ~850)")


def main():
    t0 = time.time()
    log("ROOT =", ROOT)
    paths = find_paths()
    tok = step_tokenizer()
    log(f"tokenizer vocab {tok.vocab_size}, specials {tok.specials}")
    step_leakage(paths)
    holdout = step_dev(paths, tok)
    man = step_encode(paths, tok, holdout)
    step_verify(tok, man)
    log(f"ALL DONE in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
