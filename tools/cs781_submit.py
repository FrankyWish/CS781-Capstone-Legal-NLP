#!/usr/bin/env python3
"""CS-781 Capstone submission manager.

Validates a prediction.json against the rules the Codabench scorer enforces,
packages it as submission.zip, and keeps a ledger of every submission so the
per-phase minimums, the daily cap, and the "no identical resubmissions" rule
are all tracked in one place.

Standard library only, so it runs unchanged on Windows, Kaggle and Colab.

Commands
--------
  validate   check a prediction.json without packaging it
  pack       validate, write submission.zip, record it in the ledger
  status     submission counts, caps, distinct methods, missing requirements
  score      attach a local or leaderboard score to a recorded submission
  report     dump the ledger as a markdown table for the report / README

Run `python cs781_submit.py <command> --help` for per-command options.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import zipfile
from collections import Counter

# --------------------------------------------------------------------------
# Competition constants (Capstone Document, sections 6.5, 8.2, 9.2, ...)
# --------------------------------------------------------------------------

MIN_SUBMISSIONS = {1: 4, 2: 6, 3: 6, 4: 6, 5: 6, 6: 6, 7: 6}
MAX_PER_PHASE = 40
MAX_PER_DAY = 10

PHASE_NAMES = {
    1: "Legal Language Model Pretraining",
    2: "Legal Named Entity Recognition (L-NER)",
    3: "Rhetorical Role Prediction (RR)",
    4: "Court Judgment Prediction (CJPE)",
    5: "Legal Statute Identification (LSI)",
    6: "Abstractive Summarization",
    7: "Prior Case Retrieval (PCR)",
}

PHASE2_LABEL_COUNT = 12
PHASE3_LABEL_COUNT = 13
PHASE7_K = 7

# Phases 2-7 must include a LORMA run and a knowledge-distillation run among
# the six. `status` flags these if the method tags never mention them.
REQUIRED_METHOD_KEYWORDS = {"lorma": "LORMA", "distil": "knowledge distillation"}


# --------------------------------------------------------------------------
# Strict JSON loading -- mirrors the scorer's own loader
# --------------------------------------------------------------------------

class ValidationError(Exception):
    """Raised for a problem that would make the Codabench scorer reject the file."""


def _reject_constant(value):
    raise ValidationError(
        f"Non-standard JSON numeric value {value!r} is not allowed. "
        "Check for NaN/Infinity leaking out of your model outputs."
    )


def _pairs_no_duplicates(pairs):
    seen = {}
    dupes = []
    for key, value in pairs:
        if key in seen:
            dupes.append(key)
        seen[key] = value
    if dupes:
        raise ValidationError(
            f"Duplicate JSON object keys: {list(dict.fromkeys(dupes))[:10]}"
        )
    return seen


def load_strict(path):
    """Load JSON exactly as the scorer does: no duplicate keys, no NaN/Infinity."""
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(
                handle,
                object_pairs_hook=_pairs_no_duplicates,
                parse_constant=_reject_constant,
            )
    except FileNotFoundError:
        raise ValidationError(f"File not found: {path}")
    except json.JSONDecodeError as exc:
        raise ValidationError(
            f"Invalid JSON in {path} at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc


def load_plain(path):
    """Load a reference/test file (not held to the scorer's strictness)."""
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


# --------------------------------------------------------------------------
# Helpers for reading the official test files
#
# The released test files differ in shape between phases (object keyed by ID,
# or list of {"id": ...} records), so both are accepted and the ID/field
# extraction is deliberately tolerant.
# --------------------------------------------------------------------------

def test_ids(data, path):
    """Return the list of official IDs in a test file."""
    if isinstance(data, dict):
        return [str(k) for k in data]
    if isinstance(data, list):
        out = []
        for i, record in enumerate(data):
            if not isinstance(record, dict) or "id" not in record:
                raise ValidationError(f"{path}: record {i} has no 'id' field.")
            out.append(str(record["id"]))
        return out
    raise ValidationError(f"{path}: expected a JSON object or list.")


def test_field(data, path, *candidates):
    """Map ID -> the first present field among `candidates` (e.g. 'text', 'document')."""
    out = {}
    if isinstance(data, dict):
        items = [(str(k), v) for k, v in data.items()]
    elif isinstance(data, list):
        items = [(str(r["id"]), r) for r in data]
    else:
        raise ValidationError(f"{path}: expected a JSON object or list.")

    for key, value in items:
        if isinstance(value, dict):
            for name in candidates:
                if name in value:
                    out[key] = value[name]
                    break
        else:
            # Phase 2 test files map ID straight to the document string.
            out[key] = value
    return out


def strip_txt(doc_id):
    """Phase 3 treats 'doc.txt' and 'doc' as the same document."""
    return doc_id[:-4] if doc_id.endswith(".txt") else doc_id


def compare_ids(pred_ids, official_ids, report, normalize=lambda x: x):
    """Record missing/unknown IDs. This is the single most common silent failure."""
    pred_set = {normalize(i) for i in pred_ids}
    official_set = {normalize(i) for i in official_ids}

    missing = sorted(official_set - pred_set)
    extra = sorted(pred_set - official_set)
    if missing:
        report.error(
            f"{len(missing)} official test ID(s) missing from prediction.json. "
            f"First few: {missing[:5]}"
        )
    if extra:
        report.error(
            f"{len(extra)} ID(s) in prediction.json are not official test IDs. "
            f"First few: {extra[:5]}"
        )
    if not missing and not extra:
        report.ok(f"All {len(official_set)} official test IDs present, none extra.")


# --------------------------------------------------------------------------
# Report object -- collects errors (blocking) and warnings (advisory)
# --------------------------------------------------------------------------

class Report:
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.notes = []

    def error(self, message):
        self.errors.append(message)

    def warn(self, message):
        self.warnings.append(message)

    def ok(self, message):
        self.notes.append(message)

    @property
    def passed(self):
        return not self.errors

    def render(self):
        for note in self.notes:
            print(f"  [ok]   {note}")
        for warning in self.warnings:
            print(f"  [warn] {warning}")
        for err in self.errors:
            print(f"  [FAIL] {err}")
        print()
        if self.errors:
            print(f"VALIDATION FAILED: {len(self.errors)} error(s), "
                  f"{len(self.warnings)} warning(s).")
        else:
            print(f"Validation passed with {len(self.warnings)} warning(s).")


# --------------------------------------------------------------------------
# Per-phase validators
#
# Each validator reproduces the checks in the matching evaluate_phaseN.py so a
# file that passes here will not be rejected by the platform, and adds a few
# advisory checks for mistakes the scorer silently scores as zero.
# --------------------------------------------------------------------------

def validate_phase1(pred, args, report):
    if not isinstance(pred, dict) or not pred:
        report.error("Phase 1 prediction must be a non-empty JSON object mapping ID -> record.")
        return

    prompts = {}
    if args.test:
        data = load_plain(args.test)
        compare_ids(pred.keys(), test_ids(data, args.test), report)
        prompts = test_field(data, args.test, "text")
    else:
        report.warn("No --test given; official test IDs were not checked.")

    word_counts = []
    for key, record in pred.items():
        where = f"prediction[{key!r}]"
        if not isinstance(key, str) or not key.strip():
            report.error(f"{where}: ID must be a non-empty string.")
            continue
        if not isinstance(record, dict) or "text" not in record:
            report.error(f"{where}: must be an object containing a 'text' field.")
            continue
        text = record["text"]
        if not isinstance(text, str) or not text.strip():
            report.error(f"{where}: 'text' must be a non-empty string.")
            continue

        words = text.split()
        word_counts.append(len(words))

        # The brief asks for a 128-word continuation. Far off that hurts BERTScore.
        if len(words) < 80:
            report.warn(f"{where}: only {len(words)} words (target 128).")
        elif len(words) > 200:
            report.warn(f"{where}: {len(words)} words (target 128); overlong.")

        # "must not include the original prompt" -- catch an echoed prompt.
        prompt = prompts.get(key)
        if isinstance(prompt, str) and len(words) >= 12:
            head = " ".join(words[:12])
            if head in prompt:
                report.warn(f"{where}: begins with text copied from the prompt.")

        # A small LM at low temperature loops; repetition destroys BERTScore.
        if len(words) >= 20:
            grams = [tuple(words[i:i + 4]) for i in range(len(words) - 3)]
            counts = Counter(grams)
            repeated = sum(c - 1 for c in counts.values() if c > 1)
            if repeated / max(len(grams), 1) > 0.5:
                report.warn(f"{where}: heavily repetitive output (degenerate decoding?).")

    if word_counts:
        mean = sum(word_counts) / len(word_counts)
        report.ok(f"{len(pred)} continuations, mean length {mean:.1f} words.")


def validate_phase2(pred, args, report):
    if not isinstance(pred, dict) or not pred:
        report.error("Phase 2 prediction must be a non-empty JSON object.")
        return

    if not args.labels:
        report.error("Phase 2 needs --labels pointing at the official labels.txt.")
        return
    with open(args.labels, encoding="utf-8") as handle:
        labels = [line.strip() for line in handle if line.strip()]
    if len(labels) != PHASE2_LABEL_COUNT:
        report.error(f"labels.txt must contain {PHASE2_LABEL_COUNT} labels, found {len(labels)}.")
        return
    allowed = set(labels)

    documents = {}
    if args.test:
        data = load_plain(args.test)
        compare_ids(pred.keys(), test_ids(data, args.test), report)
        documents = test_field(data, args.test, "text")
    else:
        report.warn("No --test given; official test IDs were not checked.")

    total_spans = 0
    per_label = Counter()
    docs_without_entities = 0

    for doc_id, spans in pred.items():
        where = f"prediction[{doc_id!r}]"
        if spans is None:
            docs_without_entities += 1
            continue
        if not isinstance(spans, list):
            report.error(f"{where}: expected a list of entities.")
            continue
        if not spans:
            docs_without_entities += 1

        document = documents.get(doc_id)
        seen = set()
        for index, span in enumerate(spans):
            at = f"{where}, entity {index}"
            if not isinstance(span, list) or len(span) not in (3, 4):
                report.error(f"{at}: expected [start, end, label] or [start, end, label, text].")
                continue
            start, end, label = span[:3]
            if isinstance(start, bool) or not isinstance(start, int) \
                    or isinstance(end, bool) or not isinstance(end, int):
                report.error(f"{at}: start and end must be integers (not floats or strings).")
                continue
            if start < 0 or end <= start:
                report.error(f"{at}: require 0 <= start < end, got [{start}, {end}].")
                continue
            if not isinstance(label, str) or label not in allowed:
                report.error(f"{at}: invalid label {label!r}.")
                continue
            key = (start, end, label)
            if key in seen:
                report.error(f"{at}: duplicate span {key}.")
                continue
            seen.add(key)
            total_spans += 1
            per_label[label] += 1

            # Advisory: offsets past the end of the document can never match.
            if isinstance(document, str):
                if end > len(document):
                    report.error(f"{at}: end offset {end} exceeds document length {len(document)}.")
                elif len(span) == 4 and isinstance(span[3], str):
                    actual = document[start:end]
                    if actual != span[3]:
                        report.warn(
                            f"{at}: entity_text does not match document[{start}:{end}] "
                            f"({span[3]!r} vs {actual!r}) -- your offsets are shifted."
                        )

    missing = sorted(set(labels) - set(per_label))
    if missing:
        report.warn(
            f"No entity predicted for {len(missing)} label(s): {missing}. "
            "Macro-F1 averages over all 12 labels, so each empty label scores 0."
        )
    if docs_without_entities:
        report.warn(f"{docs_without_entities} document(s) have no predicted entities.")
    report.ok(f"{total_spans} spans across {len(pred)} documents.")


def validate_phase3(pred, args, report):
    if not isinstance(pred, dict) or not pred:
        report.error("Phase 3 prediction must be a non-empty JSON object.")
        return

    if not args.labels:
        report.error("Phase 3 needs --labels pointing at label_vocab.json.")
        return
    vocab = load_plain(args.labels)
    if isinstance(vocab, list):
        labels = vocab
    elif isinstance(vocab, dict):
        values = list(vocab.values())
        if all(isinstance(v, int) and not isinstance(v, bool) for v in values):
            labels = list(vocab.keys())
        elif all(isinstance(v, str) for v in values):
            labels = values
        else:
            report.error("label_vocab.json has an unsupported shape.")
            return
    else:
        report.error("label_vocab.json must be a list or object.")
        return
    if len(labels) != PHASE3_LABEL_COUNT:
        report.error(f"Expected {PHASE3_LABEL_COUNT} labels, found {len(labels)}.")
        return
    allowed = set(labels)

    # Normalized IDs: the scorer strips a trailing '.txt'.
    normalized = {}
    for raw in pred:
        norm = strip_txt(str(raw))
        if norm in normalized:
            report.error(f"Duplicate document ID after .txt normalization: {norm}")
        normalized[norm] = raw

    sentence_counts = {}
    if args.test_cl or args.test_it:
        official = []
        for path in filter(None, (args.test_cl, args.test_it)):
            data = load_plain(path)
            official.extend(test_ids(data, path))
            for doc_id, sentences in test_field(data, path, "text", "sentences").items():
                if isinstance(sentences, list):
                    sentence_counts[strip_txt(doc_id)] = len(sentences)
        compare_ids(pred.keys(), official, report, normalize=strip_txt)
    else:
        report.warn("No --test-cl/--test-it given; IDs and sentence counts were not checked.")

    predicted_labels = Counter()
    for raw, sequence in pred.items():
        where = f"prediction[{raw!r}]"
        if not isinstance(sequence, list):
            report.error(f"{where}: labels must be a JSON list.")
            continue
        if not sequence:
            report.error(f"{where}: empty label list.")
            continue
        for label in sequence:
            if label not in allowed:
                report.error(f"{where}: unknown label {label!r}.")
                break
            predicted_labels[label] += 1

        expected = sentence_counts.get(strip_txt(str(raw)))
        if expected is not None and len(sequence) != expected:
            report.error(
                f"{where}: sentence-count mismatch -- predicted {len(sequence)}, "
                f"document has {expected}."
            )

    unused = sorted(allowed - set(predicted_labels))
    if unused:
        report.warn(
            f"{len(unused)} label(s) never predicted: {unused}. "
            "Macro-F1 averages over all 13, so each unused label scores 0."
        )
    report.ok(f"{len(pred)} documents, {sum(predicted_labels.values())} sentence labels.")


def validate_phase4(pred, args, report):
    if not isinstance(pred, list) or not pred:
        report.error("Phase 4 prediction must be a non-empty JSON list.")
        return

    seen = set()
    counts = Counter()
    for index, record in enumerate(pred):
        where = f"prediction[{index}]"
        if not isinstance(record, dict) or "id" not in record or "pred" not in record:
            report.error(f"{where}: must be an object with 'id' and 'pred'.")
            continue
        case_id = record["id"]
        value = record["pred"]
        if not isinstance(case_id, str):
            report.error(f"{where}: 'id' must be a string (got {type(case_id).__name__}).")
            continue
        if isinstance(value, bool) or value not in (0, 1):
            report.error(f"{where}: 'pred' must be the integer 0 or 1, got {value!r}.")
            continue
        if case_id in seen:
            report.error(f"{where}: duplicate ID {case_id!r}.")
            continue
        seen.add(case_id)
        counts[value] += 1

    if args.test:
        data = load_plain(args.test)
        compare_ids(seen, test_ids(data, args.test), report)
    else:
        report.warn("No --test given; official test IDs were not checked.")

    if counts and min(counts.values(), default=0) == 0:
        report.warn("Only one class predicted; Macro-F1 caps around 0.33 in that case.")
    total = sum(counts.values())
    if total:
        report.ok(f"{total} predictions -- class 0: {counts[0]}, class 1: {counts[1]}.")


def validate_phase5(pred, args, report):
    if not isinstance(pred, list) or not pred:
        report.error("Phase 5 prediction must be a non-empty JSON list.")
        return

    seen = set()
    label_uses = Counter()
    empty = 0
    highest = -1

    for index, record in enumerate(pred):
        where = f"prediction[{index}]"
        if not isinstance(record, dict) or "id" not in record or "pred" not in record:
            report.error(f"{where}: must be an object with 'id' and 'pred'.")
            continue
        case_id = record["id"]
        values = record["pred"]
        if not isinstance(case_id, str):
            report.error(f"{where}: 'id' must be a string.")
            continue
        if case_id in seen:
            report.error(f"{where}: duplicate ID {case_id!r}.")
            continue
        seen.add(case_id)
        if not isinstance(values, list):
            report.error(f"{where}: 'pred' must be a list of integer statute IDs.")
            continue
        if not values:
            empty += 1
        inner = set()
        for value in values:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                report.error(f"{where}: statute IDs must be non-negative integers, got {value!r}.")
                continue
            if value in inner:
                report.error(f"{where}: duplicate statute ID {value}.")
                continue
            inner.add(value)
            label_uses[value] += 1
            highest = max(highest, value)

    if args.test:
        data = load_plain(args.test)
        compare_ids(seen, test_ids(data, args.test), report)
    else:
        report.warn("No --test given; official test IDs were not checked.")

    if args.max_label is not None and highest > args.max_label:
        report.error(
            f"Predicted statute ID {highest} exceeds --max-label {args.max_label}; "
            "the scorer rejects IDs outside the reference label space."
        )
    elif args.max_label is None:
        report.warn(
            "No --max-label given. The scorer derives the label count from the private "
            "references and rejects any ID above it; pass the maximum ID seen in train/dev."
        )

    if empty:
        report.warn(f"{empty} document(s) predicted with no statutes at all.")
    if seen:
        per_doc = sum(label_uses.values()) / len(seen)
        report.ok(
            f"{len(seen)} documents, {len(label_uses)} distinct statutes, "
            f"{per_doc:.2f} statutes/document."
        )


def validate_phase6(pred, args, report):
    # The scorer accepts a list of {"id","summary"} or an object {id: summary}.
    if isinstance(pred, dict):
        mapping = pred
        if not mapping:
            report.error("Phase 6 prediction object is empty.")
            return
        report.warn("Object format accepted, but the brief recommends the list format.")
    elif isinstance(pred, list):
        if not pred:
            report.error("Phase 6 prediction list is empty.")
            return
        mapping = {}
        for index, record in enumerate(pred):
            where = f"prediction[{index}]"
            if not isinstance(record, dict) or "id" not in record or "summary" not in record:
                report.error(f"{where}: must be an object with 'id' and 'summary'.")
                continue
            if record["id"] in mapping:
                report.error(f"{where}: duplicate ID {record['id']!r}.")
                continue
            mapping[record["id"]] = record["summary"]
    else:
        report.error("Phase 6 prediction must be a JSON list or object.")
        return

    lengths = []
    for doc_id, summary in mapping.items():
        where = f"prediction[{doc_id!r}]"
        if not isinstance(doc_id, str):
            report.error(f"{where}: ID must be a string.")
            continue
        if not isinstance(summary, str) or not summary.strip():
            report.error(f"{where}: summary must be a non-empty string.")
            continue
        words = summary.split()
        lengths.append(len(words))
        if len(words) < 20:
            report.warn(f"{where}: very short summary ({len(words)} words).")
        if len(words) >= 20:
            grams = [tuple(words[i:i + 4]) for i in range(len(words) - 3)]
            counts = Counter(grams)
            repeated = sum(c - 1 for c in counts.values() if c > 1)
            if repeated / max(len(grams), 1) > 0.5:
                report.warn(f"{where}: heavily repetitive summary.")

    if args.test:
        data = load_plain(args.test)
        compare_ids(mapping.keys(), test_ids(data, args.test), report)
    else:
        report.warn("No --test given; official test IDs were not checked.")

    if lengths:
        report.ok(
            f"{len(mapping)} summaries, mean {sum(lengths) / len(lengths):.1f} words "
            f"(min {min(lengths)}, max {max(lengths)})."
        )


def validate_phase7(pred, args, report):
    if not isinstance(pred, dict) or not pred:
        report.error("Phase 7 prediction must be a non-empty JSON object.")
        return

    pool = set()
    if args.candidates:
        data = load_plain(args.candidates)
        if isinstance(data, dict):
            pool = {str(k) for k in data}
        elif isinstance(data, list):
            pool = {str(v) if not isinstance(v, dict) else str(v["id"]) for v in data}
        else:
            report.error(f"{args.candidates}: expected a JSON object or list.")
    else:
        report.error(
            "Phase 7 needs --candidates pointing at test_candidates.json. "
            "Candidate IDs outside that pool are rejected outright."
        )

    if args.test:
        data = load_plain(args.test)
        compare_ids(pred.keys(), test_ids(data, args.test), report)
    else:
        report.warn("No --test given; official test query IDs were not checked.")

    usage = Counter()
    for query_id, ranked in pred.items():
        where = f"prediction[{query_id!r}]"
        if not isinstance(ranked, list):
            report.error(f"{where}: must be a list of candidate IDs.")
            continue
        if len(ranked) != PHASE7_K:
            report.error(f"{where}: expected exactly {PHASE7_K} candidates, got {len(ranked)}.")
        seen = set()
        for position, candidate in enumerate(ranked):
            if not isinstance(candidate, str) or not candidate.strip():
                report.error(f"{where}: position {position} is not a non-empty string.")
                continue
            if candidate in seen:
                report.error(f"{where}: duplicate candidate {candidate!r}.")
                continue
            seen.add(candidate)
            usage[candidate] += 1
            if pool and candidate not in pool:
                report.error(f"{where}: candidate {candidate!r} is outside the candidate pool.")

    if usage:
        most_common, count = usage.most_common(1)[0]
        if count > 0.5 * len(pred):
            report.warn(
                f"Candidate {most_common!r} is returned for {count}/{len(pred)} queries -- "
                "the ranker may be collapsing to a few generic cases."
            )
        report.ok(f"{len(pred)} queries x {PHASE7_K} candidates, "
                  f"{len(usage)} distinct candidates used.")


VALIDATORS = {
    1: validate_phase1,
    2: validate_phase2,
    3: validate_phase3,
    4: validate_phase4,
    5: validate_phase5,
    6: validate_phase6,
    7: validate_phase7,
}


def run_validation(args):
    """Validate args.prediction for args.phase. Returns (Report, parsed prediction)."""
    report = Report()
    print(f"Phase {args.phase} -- {PHASE_NAMES[args.phase]}")
    print(f"Checking {args.prediction}\n")
    try:
        pred = load_strict(args.prediction)
    except ValidationError as exc:
        report.error(str(exc))
        return report, None
    VALIDATORS[args.phase](pred, args, report)
    return report, pred


# --------------------------------------------------------------------------
# Ledger
# --------------------------------------------------------------------------

def canonical_hash(pred):
    """Hash the parsed content, so reformatting alone is still a duplicate.

    Codabench does not count identical resubmissions toward the minimum, and
    the brief warns that repeated submissions are discounted. Sorting keys and
    stripping whitespace means a re-dumped copy of the same predictions hashes
    the same as the original.
    """
    blob = json.dumps(pred, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def ledger_path(root):
    return os.path.join(root, "ledger.json")


def load_ledger(root):
    path = ledger_path(root)
    if not os.path.exists(path):
        return {"version": 1, "submissions": []}
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def save_ledger(root, ledger):
    os.makedirs(root, exist_ok=True)
    path = ledger_path(root)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(ledger, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(tmp, path)


def phase_entries(ledger, phase):
    return [e for e in ledger["submissions"] if e["phase"] == phase]


def record_count(pred):
    return len(pred) if isinstance(pred, (dict, list)) else 0


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------

def cmd_validate(args):
    report, _ = run_validation(args)
    report.render()
    return 0 if report.passed else 1


def cmd_pack(args):
    report, pred = run_validation(args)
    report.render()
    if not report.passed and not args.force:
        print("\nNothing was packaged. Fix the errors, or re-run with --force.")
        return 1
    if not report.passed:
        print("\n--force given: packaging despite validation errors.")

    ledger = load_ledger(args.ledger)
    existing = phase_entries(ledger, args.phase)
    digest = canonical_hash(pred)

    # Cap checks --------------------------------------------------------
    if len(existing) >= MAX_PER_PHASE:
        print(f"\nPhase cap reached: {len(existing)}/{MAX_PER_PHASE} submissions already logged.")
        if not args.force:
            return 1

    now = dt.datetime.now(dt.timezone.utc)
    today_utc = now.date().isoformat()
    today_local = dt.datetime.now().date().isoformat()
    utc_today = sum(1 for e in existing if e["created_utc"][:10] == today_utc)
    local_today = sum(1 for e in existing if e.get("created_local", "")[:10] == today_local)
    if max(utc_today, local_today) >= MAX_PER_DAY:
        print(f"\nDaily cap reached for this phase "
              f"({utc_today} today UTC / {local_today} today local, limit {MAX_PER_DAY}).")
        if not args.force:
            return 1

    # Duplicate check ---------------------------------------------------
    duplicate_of = next((e["seq"] for e in existing if e["sha256"] == digest), None)
    if duplicate_of is not None:
        print(f"\nThese predictions are identical in content to submission "
              f"#{duplicate_of} in this phase (formatting aside).")
        print("Identical resubmissions do not count toward the minimum.")
        if not args.force:
            print("Change something real, or re-run with --force to log it anyway.")
            return 1

    if args.dry_run:
        print("\n--dry-run: validation and cap checks passed, nothing written.")
        return 0

    # Write the archive -------------------------------------------------
    seq = max((e["seq"] for e in ledger["submissions"]), default=0) + 1
    tag = args.method.lower().replace(" ", "-").replace("/", "-")
    folder = os.path.join(args.ledger, f"phase{args.phase}", f"{seq:03d}_{tag}")
    os.makedirs(folder, exist_ok=True)
    zip_path = os.path.join(folder, "submission.zip")

    # prediction.json must sit at the archive root with no wrapping directory.
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(args.prediction, arcname="prediction.json")

    # Read the archive back and re-parse it, so what ships is what was checked.
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        if names != ["prediction.json"]:
            print(f"\nArchive layout is wrong: {names}")
            return 1
        try:
            json.loads(
                archive.read("prediction.json").decode("utf-8"),
                object_pairs_hook=_pairs_no_duplicates,
                parse_constant=_reject_constant,
            )
        except (ValidationError, json.JSONDecodeError) as exc:
            print(f"\nRepacked archive failed re-parsing: {exc}")
            return 1

    entry = {
        "seq": seq,
        "phase": args.phase,
        "method": args.method,
        "base_model": args.base_model,
        "notes": args.notes,
        "sha256": digest,
        "n_records": record_count(pred),
        "created_utc": now.isoformat(timespec="seconds"),
        "created_local": dt.datetime.now().isoformat(timespec="seconds"),
        "zip": os.path.relpath(zip_path, args.ledger).replace("\\", "/"),
        "source_prediction": os.path.abspath(args.prediction),
        "local_score": args.score,
        "leaderboard": None,
        "counts_toward_minimum": duplicate_of is None,
        "warnings": len(report.warnings),
    }
    ledger["submissions"].append(entry)
    save_ledger(args.ledger, ledger)

    size_kb = os.path.getsize(zip_path) / 1024
    print(f"\nSubmission #{seq} packaged.")
    print(f"  archive : {zip_path}  ({size_kb:,.0f} KB)")
    print(f"  method  : {args.method}")
    print(f"  records : {entry['n_records']}")
    print(f"  sha256  : {digest[:16]}...")
    print("\nUpload that submission.zip to Codabench, then record the score with:")
    print(f"  python cs781_submit.py score --seq {seq} --leaderboard <value>")
    return 0


def cmd_eval(args):
    """Shell out to the official evaluate_phaseN.py with the right flags.

    Each phase's evaluator takes a different flag set; getting them wrong
    wastes a round trip. Needs the packages in requirements.txt (phases 1 and 6
    pull bert-score and torch).
    """
    import subprocess

    script = os.path.join(args.scripts, f"evaluate_phase{args.phase}.py")
    if not os.path.exists(script):
        print(f"Evaluator not found: {script}\nPass --scripts <dir>.")
        return 1

    command = [sys.executable, script, "--prediction", args.prediction]
    if args.phase == 3:
        if not (args.truth_cl and args.truth_it and args.labels):
            print("Phase 3 needs --truth-cl, --truth-it and --labels.")
            return 1
        command += ["--truth-cl", args.truth_cl, "--truth-it", args.truth_it,
                    "--labels", args.labels]
    else:
        if not args.truth:
            print(f"Phase {args.phase} needs --truth (your held-out reference split).")
            return 1
        command += ["--truth", args.truth]
        if args.phase == 2:
            if not args.labels:
                print("Phase 2 needs --labels labels.txt.")
                return 1
            command += ["--labels", args.labels]
        elif args.phase == 7:
            if not args.candidate_ids:
                print("Phase 7 needs --candidate-ids (held-out pool as a JSON list).")
                return 1
            command += ["--candidate-ids", args.candidate_ids]

    print("Running:", " ".join(command), "\n")
    sys.stdout.flush()  # keep our line ahead of the child's output
    return subprocess.call(command)


def cmd_status(args):
    ledger = load_ledger(args.ledger)
    phases = [args.phase] if args.phase else sorted(MIN_SUBMISSIONS)

    for phase in phases:
        entries = phase_entries(ledger, phase)
        counting = [e for e in entries if e["counts_toward_minimum"]]
        minimum = MIN_SUBMISSIONS[phase]
        methods = sorted({e["method"] for e in counting})
        models = sorted({e["base_model"] for e in counting if e.get("base_model")})

        shortfall = minimum - len(counting)
        verdict = "OK" if shortfall <= 0 else f"SHORT BY {shortfall}"

        print(f"Phase {phase} -- {PHASE_NAMES[phase]}")
        print(f"  logged          : {len(entries)} (max {MAX_PER_PHASE})")
        print(f"  counting        : {len(counting)} / {minimum} required -- {verdict}")
        print(f"  distinct methods: {len(methods)}"
              + (f" -- {', '.join(methods)}" if methods else ""))
        if models:
            print(f"  base models     : {', '.join(models)}")

        if phase >= 2:
            # Phases 2-7 require a LORMA run and a distillation run among the six,
            # and at least four distinct Phase-1 architectures across the phase.
            blob = " ".join(methods).lower()
            for needle, label in REQUIRED_METHOD_KEYWORDS.items():
                if needle not in blob:
                    print(f"  MISSING         : no submission tagged as {label}")
            if len(models) < 4:
                print(f"  NOTE            : {len(models)}/4 distinct Phase-1 models used "
                      "(the brief asks for all four).")

        today_utc = dt.datetime.now(dt.timezone.utc).date().isoformat()
        used_today = sum(1 for e in entries if e["created_utc"][:10] == today_utc)
        if used_today:
            print(f"  today (UTC)     : {used_today} / {MAX_PER_DAY}")

        scored = [e for e in entries if e.get("leaderboard") is not None]
        if scored:
            best = max(scored, key=lambda e: e["leaderboard"])
            print(f"  best leaderboard: {best['leaderboard']} "
                  f"(#{best['seq']}, {best['method']})")
        print()
    return 0


def cmd_score(args):
    ledger = load_ledger(args.ledger)
    entry = next((e for e in ledger["submissions"] if e["seq"] == args.seq), None)
    if entry is None:
        print(f"No submission #{args.seq} in the ledger.")
        return 1
    if args.local is not None:
        entry["local_score"] = args.local
    if args.leaderboard is not None:
        entry["leaderboard"] = args.leaderboard
    if args.notes:
        entry["notes"] = args.notes
    save_ledger(args.ledger, ledger)
    print(f"#{entry['seq']} (phase {entry['phase']}, {entry['method']}): "
          f"local={entry['local_score']} leaderboard={entry['leaderboard']}")
    return 0


def cmd_report(args):
    ledger = load_ledger(args.ledger)
    entries = ledger["submissions"]
    if args.phase:
        entries = [e for e in entries if e["phase"] == args.phase]
    if not entries:
        print("No submissions logged yet.")
        return 0

    print("| # | Phase | Method | Base model | Local | Leaderboard | Date (UTC) | Notes |")
    print("|---|-------|--------|------------|-------|-------------|------------|-------|")
    for entry in sorted(entries, key=lambda e: e["seq"]):
        marker = "" if entry["counts_toward_minimum"] else " (dup)"
        print("| {seq}{marker} | {phase} | {method} | {model} | {local} | {board} | {date} | {notes} |".format(
            seq=entry["seq"],
            marker=marker,
            phase=entry["phase"],
            method=entry["method"],
            model=entry.get("base_model") or "-",
            local=entry.get("local_score") if entry.get("local_score") is not None else "-",
            board=entry.get("leaderboard") if entry.get("leaderboard") is not None else "-",
            date=entry["created_utc"][:10],
            notes=(entry.get("notes") or "").replace("|", "/"),
        ))
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def add_data_arguments(parser):
    """Paths to the official released files, used to check IDs and shapes."""
    parser.add_argument("--test", help="Official test file for the phase (checks IDs).")
    parser.add_argument("--labels", help="Phase 2: labels.txt. Phase 3: label_vocab.json.")
    parser.add_argument("--test-cl", help="Phase 3: CL_test.json.")
    parser.add_argument("--test-it", help="Phase 3: IT_test.json.")
    parser.add_argument("--candidates", help="Phase 7: test_candidates.json.")
    parser.add_argument("--max-label", type=int,
                        help="Phase 5: highest valid statute ID (from train/dev).")


def build_parser():
    parser = argparse.ArgumentParser(
        prog="cs781_submit.py",
        description="Validate, package and track CS-781 capstone submissions.",
    )
    parser.add_argument(
        "--ledger", default=os.environ.get("CS781_LEDGER", "submissions"),
        help="Directory holding ledger.json and the packaged archives "
             "(default: ./submissions, or $CS781_LEDGER).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_validate = sub.add_parser("validate", help="Check a prediction.json.")
    p_validate.add_argument("--phase", type=int, required=True, choices=sorted(PHASE_NAMES))
    p_validate.add_argument("--prediction", required=True)
    add_data_arguments(p_validate)
    p_validate.set_defaults(func=cmd_validate)

    p_pack = sub.add_parser("pack", help="Validate, zip and log a submission.")
    p_pack.add_argument("--phase", type=int, required=True, choices=sorted(PHASE_NAMES))
    p_pack.add_argument("--prediction", required=True)
    p_pack.add_argument("--method", required=True,
                        help="Adaptation technique, e.g. 'LORMA' or 'knowledge-distillation'. "
                             "Phases 2-7 need six methodologically distinct ones.")
    p_pack.add_argument("--base-model", default="",
                        help="Which Phase-1 architecture this started from.")
    p_pack.add_argument("--notes", default="", help="One line for the report table.")
    p_pack.add_argument("--score", type=float, default=None,
                        help="Local held-out score for the primary metric.")
    p_pack.add_argument("--dry-run", action="store_true",
                        help="Run every check but write nothing.")
    p_pack.add_argument("--force", action="store_true",
                        help="Package despite errors, duplicates or caps.")
    add_data_arguments(p_pack)
    p_pack.set_defaults(func=cmd_pack)

    p_eval = sub.add_parser("eval", help="Run the official local evaluator for a phase.")
    p_eval.add_argument("--phase", type=int, required=True, choices=sorted(PHASE_NAMES))
    p_eval.add_argument("--prediction", required=True)
    p_eval.add_argument("--truth", help="Your held-out reference file.")
    p_eval.add_argument("--truth-cl", help="Phase 3 only.")
    p_eval.add_argument("--truth-it", help="Phase 3 only.")
    p_eval.add_argument("--labels", help="Phase 2 labels.txt / Phase 3 label_vocab.json.")
    p_eval.add_argument("--candidate-ids", help="Phase 7: held-out pool as a JSON list.")
    p_eval.add_argument("--scripts", default=os.path.join(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))), "evaluation_scripts"),
        help="Directory holding evaluate_phaseN.py.")
    p_eval.set_defaults(func=cmd_eval)

    p_status = sub.add_parser("status", help="Counts, caps and missing requirements.")
    p_status.add_argument("--phase", type=int, choices=sorted(PHASE_NAMES))
    p_status.set_defaults(func=cmd_status)

    p_score = sub.add_parser("score", help="Attach scores to a logged submission.")
    p_score.add_argument("--seq", type=int, required=True)
    p_score.add_argument("--local", type=float)
    p_score.add_argument("--leaderboard", type=float)
    p_score.add_argument("--notes")
    p_score.set_defaults(func=cmd_score)

    p_report = sub.add_parser("report", help="Markdown experiment table.")
    p_report.add_argument("--phase", type=int, choices=sorted(PHASE_NAMES))
    p_report.set_defaults(func=cmd_report)

    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ValidationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
