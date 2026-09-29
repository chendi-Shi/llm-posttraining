"""Paired v2 slot analysis including no-slot failures and positive-only F1.

Every system must provide local row-level predictions tied to the same locked
v2 split. Public output contains aggregate statistics and file hashes only.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from compare_massive_slots import aligned_raw_outputs, parse_system
from evaluate_massive_slots import (
    _validate_study_split,
    load_locked_split_manifest,
    paired_bootstrap_f1_delta,
    read_jsonl,
    score_predictions,
    sha256_file,
)
from massive_slots import parse_prediction


EXPECTED_COUNTS = {"train": 640, "dev": 250, "confirmation": 400}
CONFIRMATION_OUTPUT = Path("reports/massive-slots-v2-confirmation-comparison.json")
CONFIRMATION_SELECTION_LOCK = Path("reports/massive-slots-v2-selection-lock.json")
ROOT = Path(__file__).resolve().parents[1]
CONFIRMATION_CODE_FILES = (
    "scripts/compare_massive_slots_v2.py",
    "scripts/compare_massive_slots.py",
    "scripts/evaluate_massive_slots.py",
    "scripts/prepare_massive_slots.py",
    "scripts/massive_slots.py",
    "scripts/v2_model_assets.py",
)


def check_confirmation_code_hashes(lock: dict) -> None:
    code_hashes = lock.get("code_sha256")
    if not isinstance(code_hashes, dict):
        raise ValueError("Confirmation comparison lock lacks code fingerprints")
    for relative in CONFIRMATION_CODE_FILES:
        expected = code_hashes.get(relative)
        if (not isinstance(expected, str) or len(expected) != 64
                or any(char not in "0123456789abcdef" for char in expected.lower())
                or sha256_file(ROOT / relative) != expected.lower()):
            raise ValueError(f"Confirmation comparison code changed after selection: {relative}")


def validate_confirmation_reports(
    lock: dict, eval_file: Path, systems: list[tuple[str, Path]],
    metric_files: list[tuple[str, Path]], expected_lock_sha256: str,
) -> tuple[dict[str, str], dict[str, dict]]:
    """Bind each raw prediction file to a preregistered model run."""
    names = [name for name, _ in systems]
    if names != lock.get("reference_systems", {}).get("comparison_order"):
        raise ValueError("Confirmation system names/order differ from the frozen plan")
    if [name for name, _ in metric_files] != names:
        raise ValueError("A matching aggregate metrics file is required for each system")
    eval_hash = sha256_file(eval_file)
    manifest_hash = sha256_file(eval_file.with_name("manifest.json"))
    references = lock["reference_systems"]
    if (lock.get("candidate_dev_results", {}).get(str(lock["selected_step"]), {}).get("eligible") is not True
            or lock.get("selected_adapter_sha256") is None):
        raise ValueError("Frozen v2 candidate did not pass development gates")
    report_hashes = {}
    reported_metrics = {}
    for (name, predictions_path), (_, report_path) in zip(systems, metric_files, strict=True):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        predictions_hash = sha256_file(predictions_path)
        if report.get("role") != "confirmation" or report.get("prediction_file_sha256") != predictions_hash:
            raise ValueError(f"{name} confirmation report does not bind its predictions")
        if name == "bio":
            bio = references["bio"]
            if (report.get("selection_lock_sha256") != expected_lock_sha256
                    or report.get("manifest_sha256") != manifest_hash
                    or report.get("eval_file_sha256") != eval_hash
                    or report.get("train_file_sha256") != bio["train_file_sha256"]
                    or report.get("model_sha256") != bio["model_sha256"]):
                raise ValueError("BIO confirmation report differs from frozen reference")
            scored = report.get("metrics") or {}
        else:
            expected_system = "v1" if name == "v1" else f"v2_step{lock['selected_step']}"
            expected_adapter = (references["v1"]["adapter_sha256"] if name == "v1"
                                else lock["selected_adapter_sha256"])
            expected_adapter_path = (references["v1"]["adapter_path"] if name == "v1"
                                     else lock["selected_sft_checkpoint"])
            base_files = lock.get("base_model_files_sha256")
            adapter_files = (references["v1"].get("adapter_files_sha256") if name == "v1"
                             else lock.get("selected_adapter_files_sha256"))
            if (not isinstance(base_files, dict) or not base_files
                    or not isinstance(adapter_files, dict) or not adapter_files
                    or set(base_files) & set(adapter_files)):
                raise ValueError(f"{name} confirmation lock lacks complete model asset fingerprints")
            frozen = report.get("frozen_model_files_sha256") or {}
            generation = report.get("generation") or {}
            if (report.get("system") != expected_system
                    or not isinstance(report.get("adapter"), str)
                    or Path(report["adapter"]).resolve() != Path(expected_adapter_path).resolve()
                    or report.get("selection_lock_sha256") != expected_lock_sha256
                    or report.get("source_sha256") != lock["data_sha256"]["source_archive_sha256"]
                    or report.get("study_manifest_sha256") != manifest_hash
                    or report.get("eval_file_sha256") != eval_hash
                    or base_files.get("model.safetensors") != lock["base_model_sha256"]
                    or base_files.get("tokenizer.json") != lock["tokenizer_sha256"]
                    or adapter_files.get("adapter_model.safetensors") != expected_adapter
                    or frozen != {**base_files, **adapter_files}
                    or any(generation.get(key) != value for key, value in lock["generation"].items())):
                raise ValueError(f"{name} confirmation report differs from frozen model/generation")
            scored = report
        if not isinstance(scored, dict) or "micro_f1" not in scored or "no_slot_failure_rate" not in scored:
            raise ValueError(f"{name} confirmation report lacks scoring aggregates")
        reported_metrics[name] = scored
        report_hashes[name] = sha256_file(report_path)
    return report_hashes, reported_metrics


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    point = fraction * (len(ordered) - 1)
    low = int(point)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (point - low) * (ordered[high] - ordered[low])


def no_slot_failure(row: dict, raw: str) -> int:
    if row["slots"]:
        raise ValueError("No-slot failure is only defined for gold-empty rows")
    parsed, _ = parse_prediction(raw, row["utt"])
    return int(parsed is None or bool(parsed))


def paired_no_slot_reduction(
    rows: list[dict], reference: list[str], candidate: list[str],
    *, samples: int = 1000, seed: int = 20260929,
) -> dict:
    """Reference minus candidate failure, resampling normalized phrase groups."""
    if len(rows) != len(reference) or len(rows) != len(candidate) or samples < 1:
        raise ValueError("Aligned rows and a positive bootstrap count are required")
    empty = [index for index, row in enumerate(rows) if not row["slots"]]
    if not empty:
        raise ValueError("No gold-empty rows")
    ref = {index: no_slot_failure(rows[index], reference[index]) for index in empty}
    cand = {index: no_slot_failure(rows[index], candidate[index]) for index in empty}
    groups: dict[str, list[int]] = defaultdict(list)
    for index in empty:
        groups[rows[index]["group_sha256"]].append(index)
    names = sorted(groups)
    observed = (sum(ref.values()) - sum(cand.values())) / len(empty)
    rng = random.Random(seed)
    values = []
    for _ in range(samples):
        sampled = [index for _ in names for index in groups[rng.choice(names)]]
        values.append((sum(ref[index] for index in sampled)
                       - sum(cand[index] for index in sampled)) / len(sampled))
    return {
        "definition": "reference no-slot failure rate minus candidate rate; positive favors candidate",
        "no_slot_examples": len(empty),
        "groups": len(names),
        "samples": samples,
        "seed": seed,
        "observed_reduction": observed,
        "reduction_ci95": [_percentile(values, 0.025), _percentile(values, 0.975)],
    }


def positive_only_metrics(rows: list[dict], raw: list[str]) -> dict:
    selected = [(row, prediction) for row, prediction in zip(rows, raw, strict=True) if row["slots"]]
    if not selected:
        raise ValueError("No positive rows")
    metrics = score_predictions([row for row, _ in selected],
                                [prediction for _, prediction in selected],
                                bootstrap_samples=0)
    return {"examples": len(selected), "micro_f1": metrics["micro_f1"],
            "sentence_exact_accuracy": metrics["sentence_exact_accuracy"]}


def decide_v2_improvement(pairs: dict, reports: dict) -> dict:
    primary = pairs["v2_minus_v1"]
    f1_lower = primary["f1"]["delta_ci95"][0]
    no_slot_lower = primary["no_slot_failure_reduction"]["reduction_ci95"][0]
    failure_rate = reports["v2"]["no_slot_failure_rate"]
    return {
        "rule": "v2-v1 paired F1 CI lower > 0; paired no-slot failure reduction CI lower > 0; v2 absolute no-slot failure <= 0.30",
        "f1_delta_ci95_lower": f1_lower,
        "no_slot_failure_reduction_ci95_lower": no_slot_lower,
        "v2_no_slot_failure_rate": failure_rate,
        "passed": f1_lower > 0 and no_slot_lower > 0 and failure_rate <= 0.30,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("dev", "confirmation"), default="dev")
    parser.add_argument("--unlock-confirmation", action="store_true")
    parser.add_argument("--selection-lock", type=Path,
                        default=Path("reports/massive-slots-v2-selection-lock.json"))
    parser.add_argument("--expected-lock-sha256")
    parser.add_argument("--eval-file", type=Path, default=Path("data/massive-zh/slots-v2/dev.jsonl"))
    parser.add_argument("--system", action="append", type=parse_system, required=True)
    parser.add_argument("--metrics", action="append", type=parse_system,
                        help="NAME=aggregate-metrics.json; required for each confirmation system")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260929)
    args = parser.parse_args(argv)
    if args.role == "confirmation" and not args.unlock_confirmation:
        parser.error("Confirmation comparison requires --unlock-confirmation")
    if args.eval_file.name != f"{args.role}.jsonl":
        parser.error("Evaluation filename must match --role")
    if len(args.system) < 2 or len({name for name, _ in args.system}) != len(args.system):
        parser.error("Provide at least two systems with distinct names")
    if args.bootstrap_samples < 1:
        parser.error("--bootstrap-samples must be positive")

    confirmation_report_hashes = None
    confirmation_report_metrics = None
    if args.role == "confirmation":
        if args.selection_lock.resolve() != CONFIRMATION_SELECTION_LOCK.resolve():
            parser.error("Confirmation comparison requires the canonical v2 selection lock path")
        if args.output.resolve() != CONFIRMATION_OUTPUT.resolve():
            parser.error(f"Confirmation output must be {CONFIRMATION_OUTPUT}")
        if args.output.exists():
            raise ValueError("Frozen confirmation comparison already exists; refusing to overwrite")
        if args.bootstrap_samples != 1000 or args.seed != 20260929:
            parser.error("Confirmation bootstrap must use the frozen 1000 samples and seed 20260929")
        if (not isinstance(args.expected_lock_sha256, str)
                or len(args.expected_lock_sha256) != 64
                or any(c not in "0123456789abcdef" for c in args.expected_lock_sha256.lower())):
            parser.error("Confirmation requires a frozen --expected-lock-sha256")
        if sha256_file(args.selection_lock) != args.expected_lock_sha256.lower():
            raise ValueError("Selection lock SHA-256 mismatch")
        lock = json.loads(args.selection_lock.read_text(encoding="utf-8"))
        if (lock.get("selected_step") not in (80, 160)
                or lock.get("status") != "selected"
                or not lock.get("selected_adapter_sha256")):
            raise ValueError("Confirmation data differs from the frozen v2 selection lock")
        check_confirmation_code_hashes(lock)
        if (sha256_file(args.eval_file.with_name("manifest.json")) != lock.get("manifest_sha256")
                or sha256_file(args.eval_file) != lock.get("confirmation_file_sha256")):
            raise ValueError("Confirmation data differs from the frozen v2 selection lock")
        confirmation_report_hashes, confirmation_report_metrics = validate_confirmation_reports(
            lock, args.eval_file, args.system, args.metrics or [],
            args.expected_lock_sha256.lower(),
        )

    manifest = load_locked_split_manifest(args.eval_file, args.role)
    if {role: manifest["splits"][role]["examples"] for role in EXPECTED_COUNTS} != EXPECTED_COUNTS:
        raise ValueError("Not the frozen v2 split sizes")
    rows = read_jsonl(args.eval_file)
    _validate_study_split(rows, args.eval_file, args.role, manifest)
    eval_hash = sha256_file(args.eval_file)
    raw: dict[str, list[str]] = {}
    reports: dict[str, dict] = {}
    for name, path in args.system:
        predictions = aligned_raw_outputs(rows, read_jsonl(path), eval_hash)
        raw[name] = predictions
        full = score_predictions(rows, predictions,
                                 bootstrap_samples=args.bootstrap_samples, seed=args.seed)
        reports[name] = {
            "prediction_file_sha256": sha256_file(path),
            "micro_f1": full["micro_f1"],
            "micro_f1_ci95": full["bootstrap"]["micro_f1_ci95"],
            "per_type": full["per_type"],
            "sentence_exact_accuracy": full["sentence_exact_accuracy"],
            "json_valid_rate": full["json_valid_rate"],
            "schema_valid_rate": full["schema_valid_rate"],
            "copy_valid_rate": full["copy_valid_rate"],
            "no_slot_examples": full["no_slot_examples"],
            "no_slot_false_positive_count": full["no_slot_false_positive_count"],
            "no_slot_invalid_output_count": full["no_slot_invalid_output_count"],
            "no_slot_failure_rate": full["no_slot_failure_rate"],
            "positive_only": positive_only_metrics(rows, predictions),
        }
        if confirmation_report_metrics is not None:
            source = confirmation_report_metrics[name]
            for key in ("micro_f1", "no_slot_failure_rate", "copy_valid_rate"):
                if abs(reports[name][key] - source[key]) > 1e-12:
                    raise ValueError(f"{name} raw predictions disagree with their confirmation report: {key}")
    names = [name for name, _ in args.system]
    pairs = {}
    for reference, candidate in combinations(names, 2):
        key = f"{candidate}_minus_{reference}"
        pairs[key] = {
            "f1": paired_bootstrap_f1_delta(
                rows, raw[reference], raw[candidate],
                samples=args.bootstrap_samples, seed=args.seed,
            ),
            "no_slot_failure_reduction": paired_no_slot_reduction(
                rows, raw[reference], raw[candidate],
                samples=args.bootstrap_samples, seed=args.seed,
            ),
        }
    decision = None
    if args.role == "confirmation":
        decision = decide_v2_improvement(pairs, reports)
    aggregate = {
        "task": "MASSIVE 1.0 zh-CN four-slot v2 natural-frequency evaluation",
        "role": args.role,
        "selection_lock_sha256": args.expected_lock_sha256.lower() if args.role == "confirmation" else None,
        "manifest_sha256": sha256_file(args.eval_file.with_name("manifest.json")),
        "eval_file_sha256": eval_hash,
        "examples": len(rows),
        "systems": reports,
        "pairs": pairs,
        "preregistered_decision": decision,
        "confirmation_metrics_file_sha256": confirmation_report_hashes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.role == "confirmation":
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n")
    else:
        args.output.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"role": args.role, "systems": reports,
                      "pair_f1_delta_ci95": {key: value["f1"]["delta_ci95"] for key, value in pairs.items()}},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
