"""Audit overlap with prior MASSIVE partitions and dev-only score sensitivity.

Only aggregate counts and metrics are written. The selected study splits stay
unchanged. Confirmation predictions require an explicit unlock flag.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

from compare_massive_slots import aligned_raw_outputs, parse_system
from evaluate_massive_slots import (
    paired_bootstrap_f1_delta,
    read_jsonl,
    score_predictions,
    sha256_file,
)
from massive_slots import group_key
from prepare_massive_slots import SOURCE_SHA256, read_archive


SPLITS = ("train", "dev", "confirmation")
PRIOR_PARTITIONS = ("dev", "test")


def prior_group_sets(source_rows: list[dict]) -> dict[str, set[str]]:
    """Build text groups from the official partitions used in the old study."""
    result = {partition: set() for partition in PRIOR_PARTITIONS}
    for row in source_rows:
        partition = row.get("partition")
        if partition in result:
            result[partition].add(group_key(row["utt"]))
    return result


def overlap_flags(rows: list[dict], prior_groups: dict[str, set[str]]) -> list[dict[str, bool]]:
    """Flags remain internal; published audit contains only their totals."""
    return [
        {
            partition: group_key(row["utt"]) in prior_groups[partition]
            for partition in PRIOR_PARTITIONS
        }
        for row in rows
    ]


def overlap_summary(flags: list[dict[str, bool]]) -> dict:
    if not flags:
        raise ValueError("Cannot audit an empty split")
    old_dev = sum(flag["dev"] for flag in flags)
    old_test = sum(flag["test"] for flag in flags)
    both = sum(flag["dev"] and flag["test"] for flag in flags)
    either = old_dev + old_test - both
    return {
        "examples": len(flags),
        "overlap_with_prior_dev": old_dev,
        "overlap_with_prior_test": old_test,
        "overlap_with_either": either,
        "overlap_with_both": both,
        "prior_partition_disjoint": len(flags) - either,
        "prior_partition_disjoint_fraction": (len(flags) - either) / len(flags),
    }


def sensitivity_metrics(rows: list[dict], raw_outputs: list[str], flags: list[dict[str, bool]]) -> dict:
    """Compare unchanged split scores with scores after excluding old text."""
    if len(rows) != len(raw_outputs) or len(rows) != len(flags):
        raise ValueError("Rows, outputs, and overlap flags must align")
    keep = [index for index, flag in enumerate(flags) if not (flag["dev"] or flag["test"])]
    if not keep:
        raise ValueError("No partition-disjoint rows remain")
    full = score_predictions(rows, raw_outputs, bootstrap_samples=0)
    clean = score_predictions(
        [rows[index] for index in keep],
        [raw_outputs[index] for index in keep],
        bootstrap_samples=0,
    )
    return {
        "full_examples": len(rows),
        "prior_partition_disjoint_examples": len(keep),
        "full_micro_f1": full["micro_f1"],
        "prior_partition_disjoint_micro_f1": clean["micro_f1"],
        "micro_f1_delta_disjoint_minus_full": clean["micro_f1"] - full["micro_f1"],
        "full_sentence_exact_accuracy": full["sentence_exact_accuracy"],
        "prior_partition_disjoint_sentence_exact_accuracy": clean["sentence_exact_accuracy"],
        "full_json_valid_rate": full["json_valid_rate"],
        "prior_partition_disjoint_json_valid_rate": clean["json_valid_rate"],
    }


def read_locked_split(split_dir: Path, manifest: dict, split: str) -> tuple[list[dict], Path]:
    path = split_dir / f"{split}.jsonl"
    if sha256_file(path) != manifest["splits"][split]["jsonl_sha256"]:
        raise ValueError(f"{split} differs from locked manifest")
    rows = read_jsonl(path)
    if len(rows) != manifest["splits"][split]["examples"]:
        raise ValueError(f"{split} count differs from locked manifest")
    if [str(row["id"]) for row in rows] != manifest["splits"][split]["source_ids"]:
        raise ValueError(f"{split} IDs differ from locked manifest")
    return rows, path


def _score_systems(
    systems: list[tuple[str, Path]], rows: list[dict], flags: list[dict[str, bool]],
    eval_sha256: str, *, role: str,
) -> dict:
    if len({name for name, _ in systems}) != len(systems):
        raise ValueError(f"Duplicate {role} system name")
    raw_by_name: dict[str, list[str]] = {}
    result = {}
    for name, path in systems:
        raw = aligned_raw_outputs(rows, read_jsonl(path), eval_sha256)
        raw_by_name[name] = raw
        result[name] = {
            "prediction_file_sha256": sha256_file(path),
            **sensitivity_metrics(rows, raw, flags),
        }
    if len(systems) > 1:
        keep = [index for index, flag in enumerate(flags) if not (flag["dev"] or flag["test"])]
        clean_rows = [rows[index] for index in keep]
        pairs = {}
        for (reference, _), (candidate, _) in combinations(systems, 2):
            pairs[f"{candidate}_minus_{reference}"] = paired_bootstrap_f1_delta(
                clean_rows,
                [raw_by_name[reference][index] for index in keep],
                [raw_by_name[candidate][index] for index in keep],
                samples=1000,
                seed=20260928,
            )
        result["_paired_disjoint_comparisons"] = pairs
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("data/raw/amazon-massive-dataset-1.0.tar.gz"))
    parser.add_argument("--split-dir", type=Path, default=Path("data/massive-zh/slots"))
    parser.add_argument("--dev-system", action="append", type=parse_system, default=[])
    parser.add_argument("--confirmation-system", action="append", type=parse_system, default=[])
    parser.add_argument("--unlock-confirmation", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("reports/massive-slots-overlap-audit.json"))
    args = parser.parse_args()

    # This guard precedes every data read, including reading a prediction file.
    if args.confirmation_system and not args.unlock_confirmation:
        parser.error("Confirmation predictions require --unlock-confirmation")
    if any("confirmation" in path.name.casefold() for _, path in args.dev_system):
        parser.error("Use --confirmation-system for a confirmation prediction file")

    manifest_path = args.split_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_sha256 = sha256_file(args.archive)
    if source_sha256 != SOURCE_SHA256 or source_sha256 != manifest["source_sha256"]:
        raise SystemExit("Official MASSIVE archive differs from the locked source")
    source_rows, _ = read_archive(args.archive)
    if {partition: sum(row.get("partition") == partition for row in source_rows)
            for partition in ("train", "dev", "test")} != manifest["source_partition_sizes"]:
        raise SystemExit("Official MASSIVE partition counts differ from the locked manifest")
    prior_groups = prior_group_sets(source_rows)
    split_rows = {}
    split_flags = {}
    split_files = {}
    for split in SPLITS:
        rows, path = read_locked_split(args.split_dir, manifest, split)
        split_rows[split] = rows
        split_flags[split] = overlap_flags(rows, prior_groups)
        split_files[split] = path
    report = {
        "task": "MASSIVE 1.0 zh-CN four-slot prior-partition text-overlap audit",
        "source_sha256": source_sha256,
        "normalization": "NFKC + casefold + remove all Unicode whitespace",
        "comparison_partitions": list(PRIOR_PARTITIONS),
        "data_license": "CC-BY-4.0",
        "split_audit": {
            split: {"split_file_sha256": sha256_file(split_files[split]),
                    **overlap_summary(split_flags[split])}
            for split in SPLITS
        },
        "confirmation_predictions_scored": bool(args.confirmation_system),
    }
    if args.dev_system:
        report["dev_sensitivity"] = _score_systems(
            args.dev_system,
            split_rows["dev"], split_flags["dev"],
            manifest["splits"]["dev"]["jsonl_sha256"], role="dev",
        )
    if args.confirmation_system:
        report["confirmation_sensitivity_posthoc"] = _score_systems(
            args.confirmation_system,
            split_rows["confirmation"], split_flags["confirmation"],
            manifest["splits"]["confirmation"]["jsonl_sha256"], role="confirmation",
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "split_audit": report["split_audit"],
        "dev_systems_scored": len(args.dev_system),
        "confirmation_systems_scored": len(args.confirmation_system),
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
