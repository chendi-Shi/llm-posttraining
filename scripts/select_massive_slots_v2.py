"""Freeze a v2 SFT checkpoint from development aggregates only.

This selector never parses confirmation rows or model predictions. It hashes the
confirmation file as opaque bytes, and refuses to create a lock unless a
checkpoint passes every predefined validity and no-slot gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Sequence

from prepare_massive_slots import SOURCE_SHA256
from v2_model_assets import hash_adapter_files, hash_base_files


ROOT = Path(__file__).resolve().parents[1]
LOCK_OUTPUT = Path("reports/massive-slots-v2-selection-lock.json")
DEFAULT_CODE_FILES = (
    Path("scripts/prepare_massive.py"),
    Path("scripts/prepare_massive_slots.py"),
    Path("scripts/prepare_massive_slots_v2.py"),
    Path("scripts/massive_slots.py"),
    Path("scripts/common.py"),
    Path("scripts/train_sft.py"),
    Path("scripts/evaluate_massive_slots.py"),
    Path("scripts/evaluate_massive_slots_v2.py"),
    Path("scripts/evaluate_massive_slots_bio.py"),
    Path("scripts/evaluate_massive_slots_v2_bio.py"),
    Path("scripts/compare_massive_slots.py"),
    Path("scripts/compare_massive_slots_v2.py"),
    Path("scripts/select_massive_slots_v2.py"),
    Path("scripts/v2_model_assets.py"),
)
EXPECTED_GENERATION = {
    "decoding": "greedy_unconstrained",
    "do_sample": False,
    "batch_size": 8,
    "max_input_tokens": 256,
    "max_new_tokens": 64,
}
GATES = {
    "json_valid_rate_min": 0.95,
    "schema_valid_rate_min": 0.95,
    "copy_valid_rate_min": 0.95,
    "no_slot_failure_rate_max": 0.30,
}
EXPECTED_TRAINING = {
    "max_steps": 160,
    "global_step": 160,
    "max_length": 160,
    "learning_rate": 0.0003,
    "seed": 20260928,
    "gradient_accumulation_steps": 4,
    "save_steps": 80,
}
DEV_PREDICTIONS = {
    80: Path("_tmp/massive-slots-v2-v2_step80-dev-predictions.jsonl"),
    160: Path("_tmp/massive-slots-v2-v2_step160-dev-predictions.jsonl"),
}
EXPECTED_SPLIT_COUNTS = {"train": 640, "dev": 250, "confirmation": 400}
EXPECTED_DATA_SHA256 = {
    "manifest": "96d8f6501931a29569421de7edbc82912025150a379aa93796ed27d64ee5dd2c",
    "train": "08916652746cc880aaa7fe17cd999f4bfbeb3834e55e5c43fbbe8148233372cd",
    "dev": "354c8cb707813f2d148d06e905e9a169ba8606a3d8cba51f857e928dc17ce912",
    "confirmation": "0975c57f2738041629d4842e8545fafef9806e99b06d2ea58306aaba31380e8d",
}
ENTITY_MATCH = "multiset of (type, literal value) per utterance"
NO_SLOT_FAILURE = "invalid output or nonempty slots on a gold-empty utterance"


class SelectionError(ValueError):
    """The input evidence cannot justify opening the confirmation set."""


def sha256_file(path: Path) -> str:
    if not path.is_file():
        raise SelectionError(f"Missing frozen file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_object(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SelectionError(f"Cannot read JSON object {path}: {error}") from error
    if not isinstance(value, dict):
        raise SelectionError(f"Expected JSON object: {path}")
    return value


def same_path(left: str | Path, right: Path) -> bool:
    return Path(left).resolve() == right.resolve()


def require_hash(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value.lower()):
        raise SelectionError(f"Invalid SHA-256 for {name}")
    return value.lower()


def require_count(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SelectionError(f"Invalid count for {name}")
    return value


def require_rate(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise SelectionError(f"Invalid rate for {name}")
    return float(value)


def validate_manifest(
    manifest: dict, manifest_path: Path, source_path: Path,
    *, expected_source_sha256: str = SOURCE_SHA256,
    expected_counts: dict[str, int] = EXPECTED_SPLIT_COUNTS,
    expected_data_sha256: dict[str, str] = EXPECTED_DATA_SHA256,
) -> dict[str, str | int]:
    if manifest.get("format_version") != 1 or manifest.get("source_partition_used") != "train":
        raise SelectionError("Unexpected v2 manifest format or source partition")
    if "v2" not in str(manifest.get("study", "")).casefold():
        raise SelectionError("Manifest is not labelled as the v2 study")
    source_sha256 = sha256_file(source_path)
    if (source_sha256 != expected_source_sha256
            or require_hash(manifest.get("source_sha256"), "source") != source_sha256):
        raise SelectionError("Official source archive differs from v2 manifest")
    manifest_sha256 = sha256_file(manifest_path)
    if manifest_sha256 != expected_data_sha256["manifest"]:
        raise SelectionError("v2 manifest differs from the preregistered study")
    hashes: dict[str, str | int] = {
        "source_archive_sha256": source_sha256,
        "manifest_sha256": manifest_sha256,
    }
    splits = manifest.get("splits")
    if not isinstance(splits, dict):
        raise SelectionError("v2 split manifest is missing")
    groups_by_split = {}
    for split in ("train", "dev", "confirmation"):
        details = splits.get(split)
        if not isinstance(details, dict):
            raise SelectionError(f"Missing {split} split in v2 manifest")
        path = manifest_path.with_name(f"{split}.jsonl")
        observed_sha256 = sha256_file(path)  # Opaque bytes: never parse confirmation rows.
        if (require_hash(details.get("jsonl_sha256"), split) != observed_sha256
                or observed_sha256 != expected_data_sha256[split]):
            raise SelectionError(f"{split} file differs from v2 manifest")
        count = require_count(details.get("examples"), f"{split} examples", minimum=1)
        if count != expected_counts[split]:
            raise SelectionError(f"{split} size differs from preregistered v2 plan")
        identifiers = details.get("source_ids")
        groups = details.get("groups")
        if not isinstance(identifiers, list) or len(identifiers) != count or len(set(map(str, identifiers))) != count:
            raise SelectionError(f"Invalid {split} source IDs in v2 manifest")
        if not isinstance(groups, list) or len(groups) != count:
            raise SelectionError(f"Invalid {split} groups in v2 manifest")
        group_hashes = [require_hash(item.get("group_sha256") if isinstance(item, dict) else None, f"{split} group") for item in groups]
        if len(set(group_hashes)) != count:
            raise SelectionError(f"Duplicate {split} phrase group")
        groups_by_split[split] = set(group_hashes)
        hashes[f"{split}_file_sha256"] = observed_sha256
        hashes[f"{split}_examples"] = count
    for left, right in (("train", "dev"), ("train", "confirmation"), ("dev", "confirmation")):
        if groups_by_split[left] & groups_by_split[right]:
            raise SelectionError(f"v2 phrase-group leakage between {left} and {right}")
    return hashes


def _validate_frozen_hashes(report: dict, expected: dict[str, str], name: str) -> None:
    frozen = report.get("frozen_model_files_sha256")
    if frozen is None:
        if name.startswith("step"):
            raise SelectionError(f"{name} dev report lacks inference-time model hashes")
        return  # Optional historical baseline may use the older evaluator.
    if not isinstance(frozen, dict):
        raise SelectionError(f"Invalid {name} frozen_model_files_sha256")
    if set(frozen) != set(expected):
        raise SelectionError(f"{name} inference asset file set differs from current files")
    for filename, digest in expected.items():
        if require_hash(frozen.get(filename), f"{name}/{filename}") != digest:
            raise SelectionError(f"{name} frozen {filename} differs from current file")


def validate_dev_report(
    report: dict,
    *,
    name: str,
    manifest_hashes: dict[str, str | int],
    dev_path: Path,
    model_dir: Path,
    adapter_dir: Path | None,
    model_hashes: dict[str, str],
    prediction_path: Path | None = None,
) -> dict:
    if report.get("role") != "dev" or report.get("source_partition") != "train":
        raise SelectionError(f"{name} is not a train-derived dev report")
    if report.get("task") != "MASSIVE 1.0 zh-CN four-slot copy-only JSON extraction":
        raise SelectionError(f"{name} has another task/metric definition")
    definition = report.get("metric_definition")
    if (
        not isinstance(definition, dict)
        or definition.get("entity_match") != ENTITY_MATCH
        or definition.get("no_slot_failure") != NO_SLOT_FAILURE
    ):
        raise SelectionError(f"{name} does not use strict entity matching")
    if not isinstance(report.get("eval_file"), str) or not same_path(report["eval_file"], dev_path):
        raise SelectionError(f"{name} refers to another development file")
    if require_hash(report.get("eval_file_sha256"), f"{name} dev") != manifest_hashes["dev_file_sha256"]:
        raise SelectionError(f"{name} dev file hash differs")
    if require_hash(report.get("study_manifest_sha256"), f"{name} manifest") != manifest_hashes["manifest_sha256"]:
        raise SelectionError(f"{name} study manifest hash differs")
    if require_count(report.get("examples"), f"{name} examples", minimum=1) != manifest_hashes["dev_examples"]:
        raise SelectionError(f"{name} development row count differs")
    if not isinstance(report.get("model"), str) or not same_path(report["model"], model_dir):
        raise SelectionError(f"{name} base model path differs")
    actual_adapter = report.get("adapter")
    if adapter_dir is None:
        if actual_adapter is not None:
            raise SelectionError(f"{name} baseline unexpectedly uses an adapter")
    elif not isinstance(actual_adapter, str) or not same_path(actual_adapter, adapter_dir):
        raise SelectionError(f"{name} adapter path differs")
    frozen_expected = dict(model_hashes)
    if adapter_dir is not None:
        frozen_expected.update(hash_adapter_files(adapter_dir))
    _validate_frozen_hashes(report, frozen_expected, name)
    if prediction_path is not None:
        if (require_hash(report.get("prediction_file_sha256"), f"{name} predictions")
                != sha256_file(prediction_path)):
            raise SelectionError(f"{name} dev predictions differ from the scored report")

    generation = report.get("generation")
    if not isinstance(generation, dict) or any(generation.get(key) != value for key, value in EXPECTED_GENERATION.items()):
        raise SelectionError(f"{name} generation differs from frozen v2 plan")
    max_observed = require_count(generation.get("max_observed_input_tokens"), f"{name} max observed input")
    if max_observed > EXPECTED_GENERATION["max_input_tokens"]:
        raise SelectionError(f"{name} input exceeded the frozen generation cap")

    micro = report.get("micro")
    if not isinstance(micro, dict):
        raise SelectionError(f"{name} micro counts missing")
    tp = require_count(micro.get("tp"), f"{name} TP")
    fp = require_count(micro.get("fp"), f"{name} FP")
    fn = require_count(micro.get("fn"), f"{name} FN")
    f1 = Fraction(2 * tp, 2 * tp + fp + fn) if 2 * tp + fp + fn else Fraction(0)
    if abs(require_rate(report.get("micro_f1"), f"{name} micro-F1") - float(f1)) > 1e-12:
        raise SelectionError(f"{name} micro-F1 differs from strict counts")
    if abs(require_rate(micro.get("f1"), f"{name} nested micro-F1") - float(f1)) > 1e-12:
        raise SelectionError(f"{name} nested micro-F1 differs from strict counts")
    if require_count(report.get("gold_slot_count"), f"{name} gold slots") != tp + fn:
        raise SelectionError(f"{name} gold slot count differs")
    if require_count(report.get("predicted_slot_count"), f"{name} predicted slots") != tp + fp:
        raise SelectionError(f"{name} predicted slot count differs")
    no_slot_examples = require_count(report.get("no_slot_examples"), f"{name} no-slot examples", minimum=1)
    if no_slot_examples > report["examples"]:
        raise SelectionError(f"{name} no-slot examples exceed development rows")
    no_slot_fp = require_count(report.get("no_slot_false_positive_count"), f"{name} no-slot false positives")
    no_slot_invalid = require_count(report.get("no_slot_invalid_output_count"), f"{name} no-slot invalid outputs")
    if no_slot_fp + no_slot_invalid > no_slot_examples:
        raise SelectionError(f"{name} no-slot failures exceed no-slot examples")
    failure_fraction = Fraction(no_slot_fp + no_slot_invalid, no_slot_examples)
    failure_rate = require_rate(report.get("no_slot_failure_rate"), f"{name} no-slot failure")
    if abs(failure_rate - float(failure_fraction)) > 1e-12:
        raise SelectionError(f"{name} no-slot failure rate differs from counts")
    rates = {key: require_rate(report.get(key), f"{name} {key}") for key in ("json_valid_rate", "schema_valid_rate", "copy_valid_rate")}
    if not rates["copy_valid_rate"] <= rates["schema_valid_rate"] <= rates["json_valid_rate"]:
        raise SelectionError(f"{name} validity rates violate their hierarchy")
    failures = [
        key for key, minimum in (("json_valid_rate", 0.95), ("schema_valid_rate", 0.95), ("copy_valid_rate", 0.95))
        if rates[key] < minimum - 1e-12
    ]
    if failure_fraction > Fraction(3, 10):
        failures.append("no_slot_failure_rate")
    return {
        "strict_micro_f1": float(f1),
        "no_slot_failure_rate": float(failure_fraction),
        "json_valid_rate": rates["json_valid_rate"],
        "schema_valid_rate": rates["schema_valid_rate"],
        "copy_valid_rate": rates["copy_valid_rate"],
        "gold_slot_count": tp + fn,
        "no_slot_examples": no_slot_examples,
        "eligible": not failures,
        "gate_failures": failures,
        "_ranking": (-f1, failure_fraction),
    }


def choose_checkpoint(results: dict[int, dict]) -> int:
    eligible = [step for step, result in results.items() if result["eligible"]]
    if not eligible:
        raise SelectionError("No v2 checkpoint passes all gates; confirmation remains locked")
    return min(eligible, key=lambda step: (*results[step]["_ranking"], step))


def validate_reference_systems(
    *, v1_lock_path: Path, bio_dev_report_path: Path, bio_model_path: Path,
    data_hashes: dict[str, str | int], model_hashes: dict[str, str],
) -> dict:
    """Freeze the predeclared v1 SFT and matched-data BIO comparators."""
    v1_lock = load_object(v1_lock_path)
    if (require_hash(v1_lock.get("base_model_sha256"), "v1 base") != model_hashes["model.safetensors"]
            or require_hash(v1_lock.get("tokenizer_sha256"), "v1 tokenizer") != model_hashes["tokenizer.json"]):
        raise SelectionError("v1 SFT reference uses another base or tokenizer")
    v1_adapter_path = Path(v1_lock.get("selected_sft_checkpoint", "")) / "adapter_model.safetensors"
    v1_adapter_hash = sha256_file(v1_adapter_path)
    if require_hash(v1_lock.get("selected_adapter_sha256"), "v1 adapter") != v1_adapter_hash:
        raise SelectionError("v1 SFT adapter differs from its prior selection lock")
    v1_adapter_hashes = hash_adapter_files(v1_adapter_path.parent)

    bio_report = load_object(bio_dev_report_path)
    bio_model_hash = sha256_file(bio_model_path)
    if (bio_report.get("role") != "dev"
            or bio_report.get("confirmation_used") is not False
            or require_hash(bio_report.get("model_sha256"), "BIO model") != bio_model_hash
            or require_hash(bio_report.get("manifest_sha256"), "BIO manifest") != data_hashes["manifest_sha256"]
            or require_hash(bio_report.get("train_file_sha256"), "BIO train") != data_hashes["train_file_sha256"]
            or require_hash(bio_report.get("eval_file_sha256"), "BIO dev") != data_hashes["dev_file_sha256"]):
        raise SelectionError("Matched BIO reference differs from its v2 development report")
    return {
        "comparison_order": ["bio", "v1", "v2"],
        "primary_pair": "v2_minus_v1",
        "v1": {
            "prior_selection_lock_sha256": sha256_file(v1_lock_path),
            "adapter_path": str(v1_adapter_path.parent),
            "adapter_sha256": v1_adapter_hash,
            "adapter_files_sha256": v1_adapter_hashes,
        },
        "bio": {
            "model_path": str(bio_model_path),
            "model_sha256": bio_model_hash,
            "dev_report_sha256": sha256_file(bio_dev_report_path),
            "train_file_sha256": data_hashes["train_file_sha256"],
        },
    }


def validate_training_manifest(
    path: Path, *, train_path: Path, model_dir: Path,
    train_sha256: str, model_hashes: dict[str, str],
) -> str:
    """Bind the completed 160-step SFT run to the preregistered train bytes."""
    run = load_object(path)
    if (not isinstance(run.get("train_file"), str)
            or not same_path(run["train_file"], train_path)
            or require_hash(run.get("train_file_sha256"), "training data") != train_sha256
            or not isinstance(run.get("model"), str)
            or not same_path(run["model"], model_dir)):
        raise SelectionError("SFT run manifest uses another training data/model")
    for key, expected in EXPECTED_TRAINING.items():
        if run.get(key) != expected:
            raise SelectionError(f"SFT run manifest has unexpected {key}")
    resume = run.get("resume_from_checkpoint")
    checkpoint80 = path.parent / "checkpoint-80"
    if not isinstance(resume, str) or not same_path(resume, checkpoint80):
        raise SelectionError("SFT run must resume from the recorded step-80 checkpoint")
    checkpoint_state = load_object(checkpoint80 / "trainer_state.json")
    if checkpoint_state.get("global_step") != 80 or checkpoint_state.get("max_steps") != 160:
        raise SelectionError("SFT resume checkpoint has an unexpected training step")
    assets = run.get("model_asset_sha256")
    if (not isinstance(assets, dict)
            or assets.get("model.safetensors") != model_hashes["model.safetensors"]
            or assets.get("tokenizer.json") != model_hashes["tokenizer.json"]):
        raise SelectionError("SFT run manifest records different base model bytes")
    versions = run.get("package_versions")
    if (not isinstance(versions, dict)
            or any(not isinstance(versions.get(name), str) for name in
                   ("torch", "transformers", "trl", "peft", "bitsandbytes"))):
        raise SelectionError("SFT run manifest lacks training package versions")
    return sha256_file(path)


def select_and_lock(
    *,
    manifest_path: Path,
    source_path: Path,
    model_dir: Path,
    sft_dir: Path,
    step80_metrics_path: Path,
    step160_metrics_path: Path,
    output_path: Path,
    code_files: Sequence[Path],
    base_metrics_path: Path | None = None,
    expected_source_sha256: str = SOURCE_SHA256,
    expected_counts: dict[str, int] = EXPECTED_SPLIT_COUNTS,
    expected_data_sha256: dict[str, str] = EXPECTED_DATA_SHA256,
    dev_prediction_paths: dict[int, Path] = DEV_PREDICTIONS,
    reference_v1_lock_path: Path | None = Path("reports/massive-slots-selection-lock.json"),
    reference_bio_dev_report_path: Path | None = Path("reports/massive-slots-v2-bio-dev.json"),
    reference_bio_model_path: Path | None = Path("outputs/massive-slots-v2-bio.joblib"),
) -> dict:
    if output_path.exists():
        raise SelectionError(f"Existing v2 lock must not be overwritten: {output_path}")
    manifest = load_object(manifest_path)
    data_hashes = validate_manifest(
        manifest, manifest_path, source_path,
        expected_source_sha256=expected_source_sha256,
        expected_counts=expected_counts,
        expected_data_sha256=expected_data_sha256,
    )
    model_hashes = hash_base_files(model_dir)
    training_manifest_sha256 = validate_training_manifest(
        sft_dir / "run_manifest.json",
        train_path=manifest_path.with_name("train.jsonl"),
        model_dir=model_dir,
        train_sha256=data_hashes["train_file_sha256"],
        model_hashes=model_hashes,
    )
    adapter_dirs = {80: sft_dir / "checkpoint-80", 160: sft_dir}
    metric_paths = {80: step80_metrics_path, 160: step160_metrics_path}
    results = {}
    report_hashes = {}
    for step in (80, 160):
        name = f"step{step}"
        report = load_object(metric_paths[step])
        results[step] = validate_dev_report(
            report,
            name=name,
            manifest_hashes=data_hashes,
            dev_path=manifest_path.with_name("dev.jsonl"),
            model_dir=model_dir,
            adapter_dir=adapter_dirs[step],
            model_hashes=model_hashes,
            prediction_path=dev_prediction_paths[step],
        )
        report_hashes[name] = sha256_file(metric_paths[step])
    if results[80]["gold_slot_count"] != results[160]["gold_slot_count"] or results[80]["no_slot_examples"] != results[160]["no_slot_examples"]:
        raise SelectionError("v2 development gold/no-slot counts differ between checkpoints")
    if base_metrics_path is not None:
        baseline = validate_dev_report(
            load_object(base_metrics_path),
            name="base",
            manifest_hashes=data_hashes,
            dev_path=manifest_path.with_name("dev.jsonl"),
            model_dir=model_dir,
            adapter_dir=None,
            model_hashes=model_hashes,
        )
        if baseline["gold_slot_count"] != results[80]["gold_slot_count"] or baseline["no_slot_examples"] != results[80]["no_slot_examples"]:
            raise SelectionError("v2 base report uses different development labels")
        report_hashes["base"] = sha256_file(base_metrics_path)
    selected_step = choose_checkpoint(results)
    if (reference_v1_lock_path is None or reference_bio_dev_report_path is None
            or reference_bio_model_path is None):
        reference_systems = None  # Only for synthetic unit fixtures.
    else:
        reference_systems = validate_reference_systems(
            v1_lock_path=reference_v1_lock_path,
            bio_dev_report_path=reference_bio_dev_report_path,
            bio_model_path=reference_bio_model_path,
            data_hashes=data_hashes,
            model_hashes=model_hashes,
        )
    code_hashes = {}
    for path in code_files:
        key = path.as_posix()
        if key in code_hashes:
            raise SelectionError(f"Duplicate code path: {key}")
        code_hashes[key] = sha256_file(path)
    generation_sha256 = hashlib.sha256(
        json.dumps(EXPECTED_GENERATION, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    lock = {
        "format_version": 1,
        "study": "MASSIVE 1.0 zh-CN four-slot JSON SFT v2",
        "status": "selected",
        "locked_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "selection_rule": "Among steps 80 and 160 passing every gate, maximize strict entity micro-F1; then minimize no-slot failure; then prefer the earlier step",
        "gates": GATES,
        "candidate_dev_results": {
            str(step): {key: value for key, value in results[step].items() if not key.startswith("_")}
            for step in (80, 160)
        },
        "selected_step": selected_step,
        "selected_sft_checkpoint": str(adapter_dirs[selected_step]),
        "selected_adapter_sha256": sha256_file(adapter_dirs[selected_step] / "adapter_model.safetensors"),
        "selected_adapter_files_sha256": hash_adapter_files(adapter_dirs[selected_step]),
        "reference_systems": reference_systems,
        "base_model_sha256": model_hashes["model.safetensors"],
        "tokenizer_sha256": model_hashes["tokenizer.json"],
        "base_model_files_sha256": model_hashes,
        "training_manifest_sha256": training_manifest_sha256,
        "manifest_sha256": data_hashes["manifest_sha256"],
        "dev_file_sha256": data_hashes["dev_file_sha256"],
        "confirmation_file_sha256": data_hashes["confirmation_file_sha256"],
        "data_sha256": data_hashes,
        "dev_metrics_sha256": report_hashes,
        "generation": EXPECTED_GENERATION,
        "generation_sha256": generation_sha256,
        "code_sha256": code_hashes,
        "dev_model_hash_policy": "v2 candidate dev reports must contain the full direct-file fingerprints of base and adapter directories at inference time",
        "confirmation_policy": "No confirmation prediction before this lock; evaluate the selected checkpoint once without retuning",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(lock, ensure_ascii=False, indent=2) + "\n")
    except FileExistsError as error:
        raise SelectionError(f"Existing v2 lock must not be overwritten: {output_path}") from error
    return lock


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/massive-zh/slots-v2/manifest.json"))
    parser.add_argument("--source", type=Path, default=Path("data/raw/amazon-massive-dataset-1.0.tar.gz"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/Qwen2.5-0.5B-Instruct"))
    parser.add_argument("--sft-dir", type=Path, default=Path("outputs/massive-slots-v2-sft-lr3e4-160"))
    parser.add_argument("--step80-metrics", type=Path, default=Path("reports/massive-slots-v2-sft-step80-dev.json"))
    parser.add_argument("--step160-metrics", type=Path, default=Path("reports/massive-slots-v2-sft-step160-dev.json"))
    parser.add_argument("--base-metrics", type=Path, default=None, help="Optional unadapted Qwen report on the same v2 dev")
    parser.add_argument("--code-file", type=Path, action="append", default=None, help="Repeat to override the default v2 code fingerprint list")
    parser.add_argument("--output", type=Path, default=LOCK_OUTPUT)
    args = parser.parse_args()
    if args.output.resolve() != LOCK_OUTPUT.resolve():
        parser.error(f"v2 selection lock must use {LOCK_OUTPUT}")
    try:
        lock = select_and_lock(
            manifest_path=args.manifest,
            source_path=args.source,
            model_dir=args.model_dir,
            sft_dir=args.sft_dir,
            step80_metrics_path=args.step80_metrics,
            step160_metrics_path=args.step160_metrics,
            output_path=args.output,
            code_files=args.code_file if args.code_file is not None else DEFAULT_CODE_FILES,
            base_metrics_path=args.base_metrics,
        )
    except SelectionError as error:
        parser.exit(2, f"Selection refused; confirmation remains locked: {error}\n")
    print(json.dumps({"selected_step": lock["selected_step"], "lock": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
