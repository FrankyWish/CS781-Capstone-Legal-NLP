"""Reject a prediction file whose text is not fluent English.

Run this on every prediction.json before packing a submission. It exists
because BERTScore does not catch broken syntax: R1 and R2 both scored ~0.75
on dev with output that was word salad, and the sweep actively *preferred*
the broken setting.

    python tools/check_fluency.py runs/R2_.../outputs/prediction.json

The test is a count of common English function-word bigrams per 128 words.
Those bigrams are the cheapest available proxy for syntax: they need the
right words in the right order, they are frequent enough that real prose
cannot avoid them, and they survive any tokenizer. Content-word overlap --
which is all BERTScore really measures here -- leaves them untouched.

Measured on this project's own files:

    R1 prediction.json   median 1   broken (unwindowed repetition penalty)
    R2 prediction.json   median 1   broken (same)
    fluent legal prose   15-25      expected

A median below 6 means the continuations are not sentences. Do not submit.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys

# Bigrams that fluent English cannot avoid and scrambled text cannot produce.
FUNCTION_BIGRAMS = {
    "of the", "in the", "to the", "that the", "and the", "by the", "on the",
    "for the", "is not", "it is", "has been", "as the", "with the",
    "from the", "of this", "in this", "shall be", "may be", "would be",
    "does not", "did not", "there is", "to be", "in a", "of a", "such as",
    "the court", "the appellant", "the respondent", "the learned",
    "the said", "the high", "the supreme", "the trial", "the accused",
    "the petitioner", "the order", "the judgment", "the act", "the section",
}

FLOOR = 6.0


def load(path: str) -> dict[str, str]:
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        return {str(r["id"]): r.get("text", "") for r in data}
    return {str(k): (v["text"] if isinstance(v, dict) else v)
            for k, v in data.items()}


def bigram_count(text: str) -> int:
    words = [w.lower() for w in text.split()]
    pairs = (" ".join(words[i:i + 2]) for i in range(len(words) - 1))
    return sum(1 for p in pairs if p in FUNCTION_BIGRAMS)


def orphan_rate(text: str) -> float:
    """Fraction of words that look like a glued-on BPE continuation piece.

    'theistrate', 'informantacc', 'ofable' -- a known function word with a
    word tail welded to it. The signature of a dropped word-initial token.
    """
    heads = ("the", "of", "and", "in", "to", "that", "is", "for", "with", "as")
    words = [w.strip(".,;:()[]\"'").lower() for w in text.split()]
    hits = sum(1 for w in words
               if len(w) > 6 and any(w.startswith(h) and len(h) < len(w) - 2
                                     for h in heads)
               and not w.isalpha() is False)
    return hits / max(len(words), 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prediction")
    parser.add_argument("--floor", type=float, default=FLOOR,
                        help=f"Minimum median bigram count (default {FLOOR}).")
    parser.add_argument("--show", type=int, default=2,
                        help="Print this many sample continuations.")
    args = parser.parse_args()

    predictions = load(args.prediction)
    if not predictions:
        print("No predictions loaded.")
        return 2

    counts = [bigram_count(t) for t in predictions.values()]
    words = [len(t.split()) for t in predictions.values()]
    median = statistics.median(counts)

    print(f"Checking {args.prediction}")
    print(f"  {len(predictions):,} continuations, mean {statistics.mean(words):.1f} words")
    print(f"  function-word bigrams: median {median:.1f}  "
          f"mean {statistics.mean(counts):.2f}  max {max(counts)}")

    for key in list(predictions)[:args.show]:
        print(f"\n  [{key}] {predictions[key][:200]}")

    print()
    if median < args.floor:
        print(f"  FAIL  median {median:.1f} is below {args.floor}. These are "
              f"not sentences.")
        print("        Do not submit. Read the text above -- if it is word "
              "salad with")
        print("        glued fragments, check --rep-window in generate.py.")
        return 1

    print(f"  PASS  median {median:.1f} >= {args.floor}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
