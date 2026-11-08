"""Byte-level BPE for CS-781 Phase 1.

Own implementation (extends the A-1/A-3 BPETokenizer). No sentencepiece, no
HF tokenizers. Two changes over A-1, both ablatable:

  1. Legal-aware pre-tokenization. A-1 split on ``\\s*\\S+|\\s+``, so
     ``AIR 1973 SC 1461``, ``S.138/1881`` and ``Act,`` each became single
     chunks. That burns vocabulary on citation strings that never recur and
     makes ``Act``, ``Act,`` and ``Act.`` three unrelated types. The
     replacement splits digit runs into groups of <=3 and separates
     punctuation from word characters.
  2. ``min_freq`` pruning of the chunk table before the merge loop, which cuts
     the type count by ~3-5x on legal text at no measurable cost.

The merge loop is A-1's: a heap over pair counts with incremental updates and
a reverse index from pair -> containing chunks, so each merge only touches the
chunks that actually contain it.

Encoding the corpus is the part that has to scale, and it does not scale by
making this file faster -- it scales by never calling it in a training loop.
Chunks are Zipfian, so ``build_chunk_table`` encodes each unique chunk once and
``encode_to_ids`` is then dictionary lookups.
"""

from __future__ import annotations

import heapq
import json
import re
from collections import Counter, defaultdict
from functools import lru_cache

SPECIALS = ["<|endoftext|>", "<|pad|>", "<|reg|>", "<|mask|>"]

# words | <=3-digit groups | punctuation runs | whitespace, each with an
# optional leading space so the space is carried by the token that follows it.
LEGAL_PAT = re.compile(
    r"""'(?:s|t|re|ve|m|ll|d)|"""     # contractions
    r""" ?[^\W\d_]+|"""               # letter runs (unicode-aware)
    r""" ?\d{1,3}|"""                 # digits, groups of at most three
    r""" ?[^\s\w]+|"""                # punctuation / symbol runs
    r"""\s+(?!\S)|\s+""",             # trailing whitespace, then any whitespace
    re.UNICODE,
)

# A-1's pattern, kept so the ablation can be run.
LEGACY_PAT = re.compile(r"\s*\S+|\s+")


@lru_cache(maxsize=1)
def bytes_to_unicode():
    """GPT-2's reversible byte <-> printable-unicode map."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


class BPETokenizer:
    def __init__(self, vocab_size=32768, pattern="legal"):
        self.target_vocab_size = vocab_size
        self.pattern_name = pattern
        self._pat = LEGAL_PAT if pattern == "legal" else LEGACY_PAT
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.merges, self.ranks, self.itos, self.stoi = [], {}, [], {}
        self.vocab_size = 0
        self.specials = {}
        self._cache = {}

    # ---------------------------------------------------------------- helpers
    def _chunks(self, text):
        return self._pat.findall(text)

    def _to_symbols(self, chunk):
        return [self.byte_encoder[b] for b in chunk.encode("utf-8")]

    # ------------------------------------------------------------------ train
    def count_chunks(self, texts, log_every=0):
        freq = Counter()
        for i, t in enumerate(texts):
            freq.update(self._chunks(t))
            if log_every and (i + 1) % log_every == 0:
                print(f"  counted {i+1:,} docs, {len(freq):,} distinct chunks", flush=True)
        return freq

    def train_from_counts(self, chunk_freq, min_freq=2, verbose=True, log_every=2000):
        kept = {c: f for c, f in chunk_freq.items() if f >= min_freq}
        if verbose:
            print(f"chunk types: {len(chunk_freq):,} -> {len(kept):,} after min_freq={min_freq} "
                  f"({sum(kept.values()):,} occurrences)", flush=True)

        words, freqs = [], []
        for c, f in kept.items():
            words.append(self._to_symbols(c))
            freqs.append(f)

        pair_counts = Counter()
        pair_words = defaultdict(set)
        for i, syms in enumerate(words):
            f = freqs[i]
            for a, b in zip(syms, syms[1:]):
                pair_counts[(a, b)] += f
                pair_words[(a, b)].add(i)

        heap = [(-c, p) for p, c in pair_counts.items()]
        heapq.heapify(heap)

        itos = sorted(set(self.byte_encoder.values()))
        n_target = max(self.target_vocab_size - len(itos) - len(SPECIALS), 0)
        self.merges = []

        while len(self.merges) < n_target:
            best = None
            while heap:
                negc, p = heapq.heappop(heap)
                if pair_counts.get(p, 0) == -negc and -negc > 1:
                    best = p
                    break
            if best is None:
                if verbose:
                    print(f"exhausted pairs at {len(self.merges):,} merges", flush=True)
                break

            new_sym = best[0] + best[1]
            self.merges.append([best[0], best[1]])
            itos.append(new_sym)
            touched = set()

            for i in list(pair_words.get(best, ())):
                syms = words[i]
                if len(syms) < 2:
                    continue
                f = freqs[i]
                for a, b in zip(syms, syms[1:]):
                    pair_counts[(a, b)] -= f
                    pair_words[(a, b)].discard(i)
                out, j = [], 0
                while j < len(syms):
                    if j + 1 < len(syms) and syms[j] == best[0] and syms[j + 1] == best[1]:
                        out.append(new_sym)
                        j += 2
                    else:
                        out.append(syms[j])
                        j += 1
                words[i] = out
                for a, b in zip(out, out[1:]):
                    pair_counts[(a, b)] += f
                    pair_words[(a, b)].add(i)
                    touched.add((a, b))

            pair_counts.pop(best, None)
            pair_words.pop(best, None)
            for p in touched:
                heapq.heappush(heap, (-pair_counts[p], p))

            if verbose and log_every and len(self.merges) % log_every == 0:
                print(f"  {len(self.merges):,}/{n_target:,} merges | "
                      f"last={new_sym!r} count={-negc:,}", flush=True)

        self._finalize(itos)
        if verbose:
            print(f"done: {len(self.merges):,} merges, vocab_size={self.vocab_size:,}", flush=True)
        return self

    def _finalize(self, itos):
        for s in SPECIALS:
            itos.append(s)
        self.itos = itos
        self.stoi = {t: i for i, t in enumerate(itos)}
        self.vocab_size = len(itos)
        self.ranks = {(a, b): i for i, (a, b) in enumerate(self.merges)}
        self.specials = {s: self.stoi[s] for s in SPECIALS}
        self._cache = {}

    # ----------------------------------------------------------------- encode
    def _bpe_chunk(self, chunk):
        hit = self._cache.get(chunk)
        if hit is not None:
            return hit
        word = self._to_symbols(chunk)
        while len(word) > 1:
            best_rank, best_i = None, None
            for i in range(len(word) - 1):
                r = self.ranks.get((word[i], word[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_i = r, i
            if best_rank is None:
                break
            a, b = word[best_i], word[best_i + 1]
            merged, out, i = a + b, [], 0
            while i < len(word):
                if i + 1 < len(word) and word[i] == a and word[i + 1] == b:
                    out.append(merged)
                    i += 2
                else:
                    out.append(word[i])
                    i += 1
            word = out
        self._cache[chunk] = word
        return word

    def build_chunk_table(self, chunk_freq, verbose=True):
        """chunk -> list[int]. Build once, then encoding is dict lookups."""
        table = {}
        for n, c in enumerate(chunk_freq):
            table[c] = [self.stoi[t] for t in self._bpe_chunk(c) if t in self.stoi]
            if verbose and (n + 1) % 200_000 == 0:
                print(f"  chunk table {n+1:,}/{len(chunk_freq):,}", flush=True)
        return table

    def tokenize(self, text):
        out = []
        for c in self._chunks(text):
            out.extend(self._bpe_chunk(c))
        return out

    def encode(self, text):
        return [self.stoi[t] for t in self.tokenize(text) if t in self.stoi]

    def encode_with_table(self, text, table):
        out = []
        for c in self._chunks(text):
            ids = table.get(c)
            out.extend(ids if ids is not None else self.encode(c))
        return out

    def decode(self, ids):
        rev = {v: k for k, v in self.specials.items()}
        parts = [self.itos[i] for i in ids if i not in rev]
        s = "".join(parts)
        return bytearray(self.byte_decoder[c] for c in s if c in self.byte_decoder).decode(
            "utf-8", errors="replace"
        )

    # ------------------------------------------------------------- save / load
    def to_dict(self):
        return {
            "merges": self.merges,
            "itos": self.itos,
            "vocab_size": self.vocab_size,
            "pattern": self.pattern_name,
            "specials": self.specials,
        }

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            obj = json.load(f)
        t = cls(vocab_size=obj["vocab_size"], pattern=obj.get("pattern", "legal"))
        t.merges = [list(m) for m in obj["merges"]]
        t.itos = list(obj["itos"])
        t.stoi = {s: i for i, s in enumerate(t.itos)}
        t.vocab_size = len(t.itos)
        t.ranks = {(a, b): i for i, (a, b) in enumerate(t.merges)}
        t.specials = obj.get("specials", {s: t.stoi[s] for s in SPECIALS if s in t.stoi})
        return t
