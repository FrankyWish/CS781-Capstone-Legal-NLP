#!/usr/bin/env python3
"""Carve a local Phase-1 development split from the training corpus.

Phase 1 ships no public dev set, so model selection has to happen against a
held-out split you build yourself. This writes two files in exactly the shapes
the official Phase-1 test input and evaluate_phase1.py expect:

  dev_prompts.json  {"dev_0001": {"text": "<512-word prompt>"}, ...}
  dev_truth.json    {"dev_0001": {"text": "<128-word reference>"}, ...}

Generate continuations for dev_prompts.json the same way you generate for the
real test set, then:

  python cs781_submit.py eval --phase 1 \
      --prediction dev_prediction.json --truth dev_truth.json

Leakage matters more than usual here. The held-out documents must be excluded
from training or the local BERTScore is measuring memorization, so this also
writes a manifest of exactly which records were held out, keyed the same way
the loader indexes them.

Standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys

PROMPT_WORDS = 512
REFERENCE_WORDS = 128


# --------------------------------------------------------------------------
# Streaming JSON reader.
#
# The Phase-1 training files are multi-gigabyte arrays. json.load() on one
# builds the whole structure in memory and the kernel kills the process --
# a bare SIGKILL with no traceback. This yields one record at a time over a
# sliding buffer, so memory stays flat whatever the file size.
#
# Deliberately duplicated from code/phase1/jsonstream.py: this script has to
# run standalone, without the phase1 package on sys.path.
# --------------------------------------------------------------------------

CHUNK = 1 << 22
TRIM_AT = 1 << 20


def _first_token(handle):
    while True:
        char = handle.read(1)
        if not char:
            return ""
        if char in "[{":
            return char
        if not char.isspace() and char != "﻿":
            raise SystemExit(f"expected '[' or '{{' at start of JSON, got {char!r}")


def iter_json_records(path, progress_every=0):
    """Yield (key, record) from a JSON array or object without loading it."""
    decoder = json.JSONDecoder()
    size = os.path.getsize(path)

    with open(path, encoding="utf-8") as handle:
        opener = _first_token(handle)
        if not opener:
            return
        is_array = opener == "["
        buffer, index, count = "", 0, 0

        def refill():
            nonlocal buffer
            more = handle.read(CHUNK)
            if not more:
                return False
            buffer += more
            return True

        while True:
            while True:
                if index >= len(buffer):
                    buffer, index = "", 0
                    if not refill():
                        return
                    continue
                char = buffer[index]
                if char in " \t\r\n,":
                    index += 1
                elif char in "]}":
                    return
                else:
                    break

            key = count
            if not is_array:
                while True:
                    try:
                        key, index = decoder.raw_decode(buffer, index)
                        break
                    except ValueError:
                        if not refill():
                            return
                while True:
                    if index >= len(buffer):
                        if not refill():
                            return
                        continue
                    if buffer[index] in " \t\r\n:":
                        index += 1
                        if buffer[index - 1] == ":":
                            break
                    else:
                        break

            while True:
                try:
                    record, index = decoder.raw_decode(buffer, index)
                    break
                except ValueError:
                    if not refill():
                        return

            yield str(key), record
            count += 1
            if progress_every and count % progress_every == 0:
                print(f"      {count:,} records, "
                      f"{handle.tell()/1e9:.2f}/{size/1e9:.2f} GB", flush=True)
            if index > TRIM_AT:
                buffer = buffer[index:]
                index = 0


def record_text(record):
    if isinstance(record, str):
        return record
    if isinstance(record, dict):
        for field in ("text", "document", "content", "body"):
            value = record.get(field)
            if isinstance(value, str) and value.strip():
                return value
    return None


def stable_key(text):
    """Content hash, so the manifest survives files being reordered."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def carve(text, rng, prompt_words, reference_words, stride_jitter=True):
    """Cut a (prompt, reference) pair out of one document.

    The window starts at a random offset rather than at word 0. Documents in
    this corpus open with boilerplate headers that are near-identical across
    cases; prompts drawn only from the start would make the dev set far easier
    than the real test set and flatter every model equally.
    """
    words = text.split()
    needed = prompt_words + reference_words
    if len(words) < needed:
        return None

    slack = len(words) - needed
    start = rng.randint(0, slack) if (stride_jitter and slack > 0) else 0
    prompt = words[start:start + prompt_words]
    reference = words[start + prompt_words:start + prompt_words + reference_words]
    return " ".join(prompt), " ".join(reference)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Build a held-out Phase-1 dev split (prompts + references)."
    )
    parser.add_argument("--train", nargs="+", required=True,
                        help="One or more Phase-1 training JSON files.")
    parser.add_argument("--out-dir", default="dev_split")
    parser.add_argument("--n", type=int, default=200,
                        help="Number of dev examples (default 200).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt-words", type=int, default=PROMPT_WORDS)
    parser.add_argument("--reference-words", type=int, default=REFERENCE_WORDS)
    parser.add_argument("--per-source", action="store_true",
                        help="Draw an equal number from each --train file, so the "
                             "dev set is balanced across Acts/HC/SC/Tribunals.")
    parser.add_argument("--no-jitter", action="store_true",
                        help="Always cut from the start of the document.")
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    needed = args.prompt_words + args.reference_words

    # How many documents to draw from each source.
    if args.per_source and len(args.train) > 1:
        per = args.n // len(args.train)
        quotas = [per + (1 if i < args.n - per * len(args.train) else 0)
                  for i in range(len(args.train))]
    else:
        quotas = [args.n] * len(args.train)   # pooled, trimmed after

    # Reservoir sampling, one pass per file ------------------------------
    #
    # A uniform sample of k items from a stream of unknown length, holding only
    # k items at a time. The alternative -- collecting every eligible document
    # and then sampling -- means the whole corpus in memory, which is what
    # killed the earlier version of this script.
    chosen = []
    for path, quota in zip(args.train, quotas):
        reservoir = []
        seen = 0
        scanned = 0
        for key, record in iter_json_records(path, progress_every=50000):
            scanned += 1
            text = record_text(record)
            if not text or len(text.split()) < needed:
                continue
            seen += 1
            if len(reservoir) < quota:
                reservoir.append((key, text))
            else:
                # Item `seen` replaces a random slot with probability quota/seen.
                slot = rng.randrange(seen)
                if slot < quota:
                    reservoir[slot] = (key, text)

        print(f"{os.path.basename(path)}: {seen:,} of {scanned:,} documents are "
              f"long enough ({needed}+ words); kept {len(reservoir)}")
        if len(reservoir) < quota:
            raise SystemExit(
                f"{path}: needs {quota} documents of {needed}+ words, found {seen}."
            )
        chosen += [(path, key, text) for key, text in reservoir]

    if len(chosen) > args.n:
        chosen = rng.sample(chosen, args.n)
    if len(chosen) < args.n:
        raise SystemExit(f"Only {len(chosen)} eligible documents, asked for {args.n}.")

    # Carve ---------------------------------------------------------------
    prompts, truth, manifest = {}, {}, []
    for number, (path, index, text) in enumerate(sorted(chosen, key=lambda x: (x[0], x[1])), 1):
        cut = carve(text, rng, args.prompt_words, args.reference_words,
                    stride_jitter=not args.no_jitter)
        if cut is None:
            continue
        prompt, reference = cut
        example_id = f"dev_{number:04d}"
        prompts[example_id] = {"text": prompt}
        truth[example_id] = {"text": reference}
        manifest.append({
            "id": example_id,
            "source_file": os.path.basename(path),
            "record_index": index,
            "content_sha1": stable_key(text),
        })

    os.makedirs(args.out_dir, exist_ok=True)

    def write(name, payload):
        path = os.path.join(args.out_dir, name)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=1)
            handle.write("\n")
        return path

    p_path = write("dev_prompts.json", prompts)
    t_path = write("dev_truth.json", truth)
    m_path = write("holdout_manifest.json", {
        "seed": args.seed,
        "prompt_words": args.prompt_words,
        "reference_words": args.reference_words,
        "held_out": manifest,
    })

    print(f"\n{len(prompts)} dev examples written.")
    print(f"  prompts  : {p_path}")
    print(f"  truth    : {t_path}")
    print(f"  manifest : {m_path}")
    print("\nExclude these from training. In your data loader:\n")
    print("    import json")
    print(f"    manifest = json.load(open({os.path.basename(m_path)!r}))")
    print("    blocked = {(h['source_file'], h['record_index'])"
          " for h in manifest['held_out']}")
    print("    # skip a record when (source_file, index) is in blocked")
    print("\nA content_sha1 is stored per record too, so the holdout survives "
          "the files being reordered or re-exported.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
