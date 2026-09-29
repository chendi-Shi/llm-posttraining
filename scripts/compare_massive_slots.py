"""Compare slot systems with paired, group-resampled confidence intervals.

Each prediction JSONL must contain id, group_sha256, eval_file_sha256 and the
unrepaired raw_output. Give systems in a meaningful order; every later system
is compared against each earlier system. Dev is the default; frozen confirmation
requires an explicit unlock. Aggregate output contains no utterances.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from itertools import combinations
from pathlib import Path

from evaluate_massive_slots import (
    _validate_study_split,
    load_locked_split_manifest,
    paired_bootstrap_f1_delta,
    read_jsonl,
    score_predictions,
    sha256_file,
)


def aligned_raw_outputs(rows: list[dict], prediction_rows: list[dict], eval_sha256: str) -> list[str]:
    """Require the same source IDs, phrase groups, and immutable eval-file hash."""
    expected = {str(row["id"]): row for row in rows}
    if len(expected) != len(rows):
        raise ValueError("Evaluation has duplicate IDs")
    found: dict[str, dict] = {}
    for record in prediction_rows:
        identifier = str(record.get("id", ""))
        if identifier in found:
            raise ValueError(f"Prediction file has duplicate ID {identifier}")
        found[identifier] = record
    if set(found) != set(expected):
        raise ValueError("Prediction IDs do not exactly match evaluation IDs")
    aligned = []
    for row in rows:
        record = found[str(row["id"])]
        if record.get("group_sha256") != row.get("group_sha256"):
            raise ValueError(f"Phrase-group hash mismatch at ID {row['id']}")
        if record.get("eval_file_sha256") != eval_sha256:
            raise ValueError(f"Evaluation-file hash mismatch at ID {row['id']}")
        raw = record.get("raw_output")
        if not isinstance(raw, str):
            raise ValueError(f"Missing raw_output string at ID {row['id']}")
        aligned.append(raw)
    return aligned


def parse_system(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,39}", name) or not path:
        raise argparse.ArgumentTypeError("Use --system NAME=prediction.jsonl")
    return name, Path(path)


def validate_locked_confirmation(rows: list[dict], manifest: dict, eval_sha256: str) -> None:
    """Verify the frozen split against the study manifest before predictions load."""
    if manifest.get("format_version") != 1 or manifest.get("source_partition_used") != "train":
        raise ValueError("Unexpected locked study manifest format or source partition")
    splits = manifest.get("splits", {})
    expected = splits.get("confirmation")
    if not isinstance(expected, dict):
        raise ValueError("Locked confirmation entry missing from study manifest")
    if expected.get("jsonl_sha256") != eval_sha256 or expected.get("examples") != len(rows):
        raise ValueError("Confirmation file hash or row count differs from locked manifest")
    if [str(row["id"]) for row in rows] != [str(identifier) for identifier in expected.get("source_ids", [])]:
        raise ValueError("Confirmation IDs differ from locked manifest")
    expected_groups = [item["group_sha256"] for item in expected.get("groups", [])]
    actual_groups = [row.get("group_sha256") for row in rows]
    if actual_groups != expected_groups or len(set(actual_groups)) != len(actual_groups):
        raise ValueError("Confirmation phrase-group hashes differ from locked manifest")
    for other_role in ("train", "dev"):
        other = splits.get(other_role)
        if not isinstance(other, dict):
            raise ValueError(f"Locked {other_role} entry missing from study manifest")
        other_groups = {item["group_sha256"] for item in other.get("groups", [])}
        if set(actual_groups) & other_groups:
            raise ValueError(f"Confirmation phrase groups overlap locked {other_role}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-file", type=Path, default=Path("data/massive-zh/slots/dev.jsonl"))
    parser.add_argument("--role", choices=("dev", "confirmation"), default="dev")
    parser.add_argument("--unlock-confirmation", action="store_true")
    parser.add_argument("--system", type=parse_system, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260928)
    args = parser.parse_args()
    if args.eval_file.name != f"{args.role}.jsonl":
        parser.error(f"--role {args.role} requires {args.role}.jsonl")
    if args.role == "confirmation" and not args.unlock_confirmation:
        parser.error("Frozen confirmation comparison requires --unlock-confirmation")
    if len(args.system) < 2:
        parser.error("Provide at least two --system NAME=FILE entries")
    if len({name for name, _ in args.system}) != len(args.system):
        parser.error("System names must be unique")
    if args.bootstrap_samples < 1:
        parser.error("--bootstrap-samples must be positive")

    manifest = load_locked_split_manifest(args.eval_file, args.role)
    rows = read_jsonl(args.eval_file)
    _validate_study_split(rows, args.eval_file, args.role, manifest)
    eval_sha256 = sha256_file(args.eval_file)
    locked_manifest_sha256 = sha256_file(args.eval_file.with_name("manifest.json"))
    if args.role == "confirmation":
        manifest_path = args.eval_file.with_name("manifest.json")
        if not manifest_path.is_file():
            raise ValueError("Frozen confirmation requires the locked study manifest")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_locked_confirmation(rows, manifest, eval_sha256)
    systems: dict[str, dict] = {}
    raw_outputs: dict[str, list[str]] = {}
    for name, path in args.system:
        raw = aligned_raw_outputs(rows, read_jsonl(path), eval_sha256)
        raw_outputs[name] = raw
        systems[name] = {
            "prediction_file": str(path),
            "prediction_file_sha256": sha256_file(path),
            "metrics": score_predictions(
                rows, raw, bootstrap_samples=args.bootstrap_samples, seed=args.seed
            ),
        }
    pairs = {}
    for (reference, _), (candidate, _) in combinations(args.system, 2):
        pairs[f"{candidate}_minus_{reference}"] = {
            "reference": reference,
            "candidate": candidate,
            **paired_bootstrap_f1_delta(
                rows,
                raw_outputs[reference],
                raw_outputs[candidate],
                samples=args.bootstrap_samples,
                seed=args.seed,
            ),
        }
    ids = [str(row["id"]) for row in rows]
    aggregate = {
        "task": "MASSIVE 1.0 zh-CN four-slot copy-only JSON extraction",
        "role": args.role,
        "locked_manifest_sha256": locked_manifest_sha256,
        "source_partition": "train",
        "eval_file": str(args.eval_file),
        "eval_file_sha256": eval_sha256,
        "eval_ids_sha256": hashlib.sha256(
            json.dumps(ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "examples": len(rows),
        "systems": systems,
        "paired_comparisons": pairs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "examples": len(rows),
        "micro_f1": {name: data["metrics"]["micro_f1"] for name, data in systems.items()},
        "pairwise_delta_ci95": {
            name: value["delta_ci95"] for name, value in pairs.items()
        },
    }, ensure_ascii=False, indent=2))
    print(f"Aggregate comparison: {args.output}")


if __name__ == "__main__":
    main()
