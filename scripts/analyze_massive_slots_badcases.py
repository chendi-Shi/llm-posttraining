"""Aggregate dev-only failure modes for four-slot MASSIVE predictions.

The report contains counts and rates, never row IDs, utterances, gold slots,
or raw model output. Categories are row-level flags and can overlap. They are
diagnostic evidence, not an exclusive assignment of errors to root causes.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Sequence

from compare_massive_slots import aligned_raw_outputs, parse_system
from evaluate_massive_slots import _validate_study_split, read_jsonl, sha256_file
from massive_slots import parse_prediction


def _rate(count: int, denominator: int) -> float | None:
    return count / denominator if denominator else None


def _has_type_confusion(gold: Counter, predicted: Counter) -> bool:
    """An unmatched literal value appears under a different slot type."""
    return any(
        gold_value == predicted_value and gold_type != predicted_type
        for gold_type, gold_value in gold
        for predicted_type, predicted_value in predicted
    )


def _has_same_type_different_value(gold: Counter, predicted: Counter) -> bool:
    return any(
        gold_type == predicted_type and gold_value != predicted_value
        for gold_type, gold_value in gold
        for predicted_type, predicted_value in predicted
    )


def _has_boundary_candidate(gold: Counter, predicted: Counter) -> bool:
    """Conservative boundary cue: one same-type value contains the other."""
    return any(
        gold_type == predicted_type
        and gold_value != predicted_value
        and (gold_value in predicted_value or predicted_value in gold_value)
        for gold_type, gold_value in gold
        for predicted_type, predicted_value in predicted
    )


def analyze_rows(rows: Sequence[dict], raw_predictions: Sequence[str]) -> dict:
    """Summarize valid/invalid outputs and overlapping error flags.

    Exact matches use a multiset of (type, literal value). A valid output may
    satisfy several flags. Invalid outputs are never interpreted as semantic
    predictions; they still count as missed-all when gold has slots.
    """
    if not rows or len(rows) != len(raw_predictions):
        raise ValueError("Rows and raw predictions must have equal nonzero length")
    if len({str(row["id"]) for row in rows}) != len(rows):
        raise ValueError("Duplicate evaluation row ID")

    counts: Counter[str] = Counter()
    validity_errors: Counter[str] = Counter()
    denominators: Counter[str] = Counter()
    for row, raw in zip(rows, raw_predictions, strict=True):
        gold_slots = row["slots"]
        if not isinstance(gold_slots, list) or not isinstance(row["utt"], str):
            raise ValueError("Every row requires an utterance and slot list")
        gold = Counter((slot["type"], slot["value"]) for slot in gold_slots)
        parsed, validity = parse_prediction(raw, row["utt"])
        predicted = Counter((slot["type"], slot["value"]) for slot in (parsed or []))
        valid = parsed is not None

        denominators["all_rows"] += 1
        denominators["gold_nonempty_rows"] += int(bool(gold))
        denominators["gold_empty_rows"] += int(not gold)
        denominators["valid_output_rows"] += int(valid)
        counts["invalid_json"] += int(not validity["json_valid"])
        counts["invalid_schema"] += int(validity["json_valid"] and not validity["schema_valid"])
        counts["invalid_copy"] += int(validity["schema_valid"] and not validity["copy_valid"])
        if validity["error"]:
            validity_errors[validity["error"]] += 1

        exact_matches = gold & predicted
        true_positive = sum(exact_matches.values())
        false_positive = sum((predicted - gold).values())
        false_negative = sum((gold - predicted).values())
        exact = valid and predicted == gold
        counts["fully_correct"] += int(exact)
        counts["incorrect"] += int(not exact)
        counts["missed_all_gold_slots"] += int(bool(gold) and true_positive == 0)
        counts["missed_all_gold_slots_valid_output"] += int(
            bool(gold) and true_positive == 0 and valid
        )
        counts["extra_slots_on_gold_empty"] += int(not gold and bool(predicted))
        counts["partial_correct"] += int(
            valid and true_positive > 0 and (false_positive > 0 or false_negative > 0)
        )

        if not valid:
            continue
        unmatched_gold = gold - predicted
        unmatched_predicted = predicted - gold
        counts["type_confusion_same_value"] += int(
            _has_type_confusion(unmatched_gold, unmatched_predicted)
        )
        counts["same_type_different_value"] += int(
            _has_same_type_different_value(unmatched_gold, unmatched_predicted)
        )
        counts["possible_span_boundary_mismatch"] += int(
            _has_boundary_candidate(unmatched_gold, unmatched_predicted)
        )
        counts["duplicate_slot_output"] += int(any(value > 1 for value in predicted.values()))
        counts["excess_duplicate_slot"] += int(
            any(value > max(1, gold.get(key, 0)) for key, value in predicted.items())
        )

    definitions = {
        "invalid_json": "JSON parsing failed.",
        "invalid_schema": "JSON parsed but did not meet the strict slot schema, including duplicate object keys.",
        "invalid_copy": "Schema passed but at least one value was not a literal source substring.",
        "missed_all_gold_slots": "Gold has slots and zero exact (type, value) multiset matches; includes invalid outputs.",
        "missed_all_gold_slots_valid_output": "Same as missed_all_gold_slots, restricted to valid outputs.",
        "extra_slots_on_gold_empty": "Valid output contains slots when gold contains none.",
        "type_confusion_same_value": "After exact matching, the same literal value appears under different gold and predicted types.",
        "same_type_different_value": "After exact matching, an unmatched gold and predicted slot share a type but have different literal values; this is a possible value mismatch, not a proven entity alignment.",
        "possible_span_boundary_mismatch": "A subset of same_type_different_value where one literal value contains the other; source occurrence offsets are not aligned.",
        "partial_correct": "A valid output matches at least one gold slot exactly and still has false positives or false negatives.",
        "duplicate_slot_output": "A valid output repeats a (type, value) pair; this can be correct when gold repeats it too.",
        "excess_duplicate_slot": "A valid output repeats a (type, value) pair more times than gold, with at least two copies.",
        "fully_correct": "A valid output exactly equals the gold slot multiset.",
        "incorrect": "Not fully correct; includes invalid outputs.",
    }
    n = denominators["all_rows"]
    categories = {}
    for name, definition in definitions.items():
        count = counts[name]
        category = {
            "examples": count,
            "rate_all_rows": _rate(count, n),
            "definition": definition,
        }
        if name.startswith("missed_all"):
            category["rate_gold_nonempty_rows"] = _rate(
                count, denominators["gold_nonempty_rows"]
            )
        if name == "extra_slots_on_gold_empty":
            category["rate_gold_empty_rows"] = _rate(
                count, denominators["gold_empty_rows"]
            )
        categories[name] = category
    return {
        "examples": n,
        "denominators": dict(denominators),
        "categories_overlap": True,
        "counts_are_row_level": True,
        "entity_match": "multiset of (type, literal value) per utterance",
        "categories": categories,
        "validity_error_counts": dict(sorted(validity_errors.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-file", type=Path, default=Path("data/massive-zh/slots/dev.jsonl"))
    parser.add_argument("--system", type=parse_system, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.eval_file.name != "dev.jsonl":
        parser.error("This analysis accepts only a locked dev.jsonl, never confirmation")
    if not args.eval_file.with_name("manifest.json").is_file():
        parser.error("A study manifest is required to verify the locked development split")
    if len({name for name, _ in args.system}) != len(args.system):
        parser.error("System names must be unique")

    rows = read_jsonl(args.eval_file)
    _validate_study_split(rows, args.eval_file, "dev")
    eval_sha256 = sha256_file(args.eval_file)
    systems = {}
    for name, path in args.system:
        raw = aligned_raw_outputs(rows, read_jsonl(path), eval_sha256)
        systems[name] = {
            "prediction_file_sha256": sha256_file(path),
            "analysis": analyze_rows(rows, raw),
        }
    report = {
        "task": "MASSIVE 1.0 zh-CN four-slot copy-only JSON extraction",
        "role": "dev",
        "source_partition": "train",
        "eval_file_sha256": eval_sha256,
        "examples": len(rows),
        "row_level_data_in_report": False,
        "systems": systems,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "examples": len(rows),
        "systems": {
            name: {
                "incorrect": value["analysis"]["categories"]["incorrect"]["examples"],
                "invalid_json": value["analysis"]["categories"]["invalid_json"]["examples"],
                "invalid_schema": value["analysis"]["categories"]["invalid_schema"]["examples"],
                "invalid_copy": value["analysis"]["categories"]["invalid_copy"]["examples"],
            }
            for name, value in systems.items()
        },
    }, ensure_ascii=False, indent=2))
    print(f"Aggregate report: {args.output}")


if __name__ == "__main__":
    main()
