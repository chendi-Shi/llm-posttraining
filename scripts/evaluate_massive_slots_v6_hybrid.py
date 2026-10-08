"""Evaluate the frozen v6 BIO plus v4 DPO hybrid on one locked split.

Development never opens test.jsonl. Independent test needs a separately
published selection lock and its SHA-256; this script never creates that lock.
Per-row predictions and confidence scores remain in ignored local _tmp files.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import re
import time
from itertools import combinations
from pathlib import Path

import joblib

import evaluate_massive_slots_v5_test as v5
from compare_massive_slots_v2 import positive_only_metrics
from evaluate_massive_slots import (
    generate_raw_predictions, paired_bootstrap_f1_delta, read_jsonl,
    score_predictions, sha256_file,
)
from evaluate_massive_slots_bio import MODEL_CONFIG, predict_slots
from evaluate_massive_slots_v2 import prediction_payload, write_predictions
from massive_slots import canonical_output
from prepare_massive_slots import SOURCE_SHA256
from prepare_massive_slots_v2 import group_sha256, strong_group_key
from slot_confidence import MEAN_LOGPROB_THRESHOLD, choose_hybrid_output
from v2_model_assets import hash_inference_files


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/massive-zh/slots-v6"
BASE = ROOT / "models/Qwen2.5-0.5B-Instruct"
DPO = ROOT / "outputs/massive-slots-v4-balanced-dpo-64"
OLD_BIO = ROOT / "outputs/massive-slots-v2-bio.joblib"
NEW_BIO = ROOT / "outputs/massive-slots-v6-bio.joblib"
LOCK = ROOT / "reports/massive-slots-v6-selection-lock.json"
DEV_REPORT = ROOT / "reports/massive-slots-v6-dev.json"
TEST_REPORT = ROOT / "reports/massive-slots-v6-test.json"

EXPECTED_COUNTS = {"train": 1600, "dev": 300, "test": 600}
EXPECTED_MANIFEST_SHA256 = "7982b3760333a364ebf17c91556f9978b32811fdacc0321db3129b6abee41a70"
EXPECTED_FILE_SHA256 = {
    "train": "ea1cd537dcfd924b24eec7997b692a42ae9df757c52f89c904358665ec06a3d3",
    "dev": "6ee1812c5ac9a12e44047cd57e5973be606b9dcdf546d1d30ccf96f4ff1a157c",
    "test": "482efe79ef69a7a43fd7feb9128df8985db663a08b38d17a057237d4bf96293b",
}
SYSTEMS = ("old_bio", "new_bio", "v4_dpo", "old_v5_hybrid", "new_hybrid")
OLD_BIO_SHA256 = "0d7a7fad835eea383223a4f9c173e42488a02416addd660ef8171141d7c361fa"
BASE_SHA256 = "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe"
TOKENIZER_SHA256 = "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"
DPO_SHA256 = "ed76990eef5bf7b89e2ce01ba516bbb5455de4c91e1242dc74c164d6603f8a6f"
GENERATION = {"decoding": "greedy_unconstrained", "batch_size": 8,
              "max_input_tokens": 256, "max_new_tokens": 64}
BOOTSTRAP = {"dev": {"samples": 1000, "seed": 20261003},
             "test": {"samples": 2000, "seed": 20261003}}
V5_MANIFEST_SHA256 = "e4531ab27746a0cfe21dc0219331f5cc8237e614bb4c7090b355000fdce1d453"
PACKAGE_NAMES = ("torch", "transformers", "peft", "trl", "bitsandbytes",
                 "scikit-learn", "joblib", "numpy")
CODE_FILES = (
    "scripts/prepare_massive_slots_v6.py",
    "scripts/prepare_massive_slots_v5_test.py",
    "scripts/prepare_massive_slots_v4.py",
    "scripts/prepare_massive_slots_v3_preferences.py",
    "scripts/prepare_massive_slots_v2.py",
    "scripts/prepare_massive_slots.py",
    "scripts/evaluate_massive_slots_v6_bio.py",
    "scripts/evaluate_massive_slots_v6_hybrid.py",
    "scripts/evaluate_massive_slots_v5_test.py",
    "scripts/slot_confidence.py",
    "scripts/evaluate_massive_slots.py",
    "scripts/evaluate_massive_slots_bio.py",
    "scripts/evaluate_massive_slots_v2.py",
    "scripts/compare_massive_slots_v2.py",
    "scripts/compare_massive_slots.py",
    "scripts/massive_slots.py",
    "scripts/common.py",
    "scripts/v2_model_assets.py",
)


def require_hex(value: object, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", value):
        raise ValueError(f"A 64-hex SHA-256 is required for {label}")
    return value.lower()


def package_versions() -> dict[str, str]:
    return {name: importlib.metadata.version(name) for name in PACKAGE_NAMES}


def output_paths(role: str) -> tuple[Path, dict[str, Path], Path]:
    report = DEV_REPORT if role == "dev" else TEST_REPORT
    predictions = {
        system: ROOT / f"_tmp/massive-slots-v6-{role}-{system}-predictions.jsonl"
        for system in SYSTEMS
    }
    confidence = ROOT / f"_tmp/massive-slots-v6-{role}-confidence.jsonl"
    return report, predictions, confidence


def require_test_arguments(args: argparse.Namespace) -> None:
    """Refuse accidental test use before opening any local data or model file."""
    if args.role != "test":
        if args.unlock_test or args.expected_selection_lock_sha256 is not None:
            raise ValueError("Test unlock arguments may only be used with --role test")
        return
    if not args.unlock_test:
        raise ValueError("Independent test requires --unlock-test")
    require_hex(args.expected_selection_lock_sha256,
                "--expected-selection-lock-sha256")


def check_output_paths(role: str) -> tuple[Path, dict[str, Path], Path]:
    report, predictions, confidence = output_paths(role)
    ignored = (ROOT / "_tmp").resolve()
    if any(not path.resolve().is_relative_to(ignored)
           for path in (*predictions.values(), confidence)):
        raise ValueError("Per-row outputs must remain in ignored _tmp")
    if any(path.exists() for path in (report, *predictions.values(), confidence)):
        raise ValueError(f"v6 {role} outputs already exist; refusing to overwrite")
    return report, predictions, confidence


def check_confidence_assets() -> None:
    """The reused v5 scorer must point to exactly the frozen v4 DPO assets."""
    if (Path.cwd().resolve() != ROOT
            or (ROOT / v5.BASE).resolve() != BASE
            or (ROOT / v5.ADAPTER).resolve() != DPO):
        raise ValueError("Run from repository root with the frozen v5 DPO confidence assets")


def validate_manifest(manifest: dict) -> None:
    if (manifest.get("format_version") != 1
            or "v6" not in str(manifest.get("study", "")).lower()
            or manifest.get("source_sha256") != SOURCE_SHA256
            or manifest.get("source_partition_used") != "train"
            or manifest.get("seed") != BOOTSTRAP["dev"]["seed"]
            or manifest.get("v5_manifest_sha256") != V5_MANIFEST_SHA256):
        raise ValueError("Expected the frozen MASSIVE v6 train-partition manifest")
    splits = manifest.get("splits")
    if not isinstance(splits, dict) or set(splits) != set(EXPECTED_COUNTS):
        raise ValueError("v6 manifest must have only train/dev/test splits")
    all_groups: set[str] = set()
    for role, count in EXPECTED_COUNTS.items():
        split = splits[role]
        if not isinstance(split, dict) or split.get("examples") != count:
            raise ValueError(f"Unexpected v6 {role} count")
        if require_hex(split.get("jsonl_sha256"), f"v6 {role} file") != EXPECTED_FILE_SHA256[role]:
            raise ValueError(f"v6 {role} file differs from the preregistered SHA-256")
        groups = split.get("group_sha256")
        if (not isinstance(groups, list) or len(groups) != count
                or any(not isinstance(item, str)
                       or not re.fullmatch(r"[0-9a-f]{64}", item) for item in groups)
                or len(set(groups)) != count):
            raise ValueError(f"Unexpected v6 {role} phrase groups")
        if all_groups.intersection(groups):
            raise ValueError("v6 train/dev/test phrase groups overlap")
        all_groups.update(groups)


def check_split_hashes(manifest: dict, role: str) -> None:
    # The dev route deliberately does not hash or open test.jsonl.
    names = ("train", "dev") if role == "dev" else ("train", "dev", "test")
    for name in names:
        if sha256_file(DATA / f"{name}.jsonl") != manifest["splits"][name]["jsonl_sha256"]:
            raise ValueError(f"v6 {name} file differs from its manifest")


def validate_rows(rows: list[dict], manifest: dict, role: str) -> None:
    expected = manifest["splits"][role]
    groups = [row.get("group_sha256") for row in rows]
    identifiers = [str(row.get("id")) for row in rows]
    if (len(rows) != expected["examples"]
            or groups != expected["group_sha256"]
            or len(set(identifiers)) != len(rows)
            or any(row.get("source_partition") != "train" for row in rows)):
        raise ValueError(f"v6 {role} rows differ from the frozen manifest")
    if any(not isinstance(row.get("utt"), str)
           or group_sha256(strong_group_key(row["utt"])) != row["group_sha256"]
           for row in rows):
        raise ValueError(f"v6 {role} phrase group differs from its utterance")
    if "source_ids" in expected and identifiers != [str(item) for item in expected["source_ids"]]:
        raise ValueError(f"v6 {role} IDs differ from the frozen manifest")


def model_hashes() -> dict:
    old_bio_sha = sha256_file(OLD_BIO)
    new_bio_sha = sha256_file(NEW_BIO)
    dpo_files = hash_inference_files(BASE, DPO)
    if (old_bio_sha != OLD_BIO_SHA256
            or dpo_files.get("model.safetensors") != BASE_SHA256
            or dpo_files.get("tokenizer.json") != TOKENIZER_SHA256
            or dpo_files.get("adapter_model.safetensors") != DPO_SHA256):
        raise ValueError("A frozen v2 BIO or v4 DPO model differs")
    return {"old_bio": old_bio_sha, "new_bio": new_bio_sha,
            "dpo_inference_files": dpo_files}


def require_test_lock(expected_lock_sha256: str) -> dict:
    """Validate published selection and all code/model assets before test text."""
    expected = require_hex(expected_lock_sha256, "selection lock")
    if sha256_file(LOCK) != expected:
        raise ValueError("v6 selection lock SHA-256 mismatch")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if (lock.get("status") != "selected_for_one_test"
            or lock.get("primary_candidate") != "new_hybrid"
            or lock.get("primary_reference") != "old_v5_hybrid"
            or lock.get("confidence_threshold") != MEAN_LOGPROB_THRESHOLD
            or lock.get("generation") != GENERATION
            or lock.get("bootstrap") != BOOTSTRAP["test"]):
        raise ValueError("v6 selection lock differs from the prespecified plan")
    for name in ("manifest_sha256", "train_file_sha256", "dev_file_sha256",
                 "test_file_sha256", "old_bio_model_sha256", "new_bio_model_sha256",
                 "v4_dpo_adapter_sha256", "dev_report_sha256"):
        require_hex(lock.get(name), name)
    if (lock["old_bio_model_sha256"] != OLD_BIO_SHA256
            or lock["v4_dpo_adapter_sha256"] != DPO_SHA256):
        raise ValueError("Frozen reference model differs from the v6 plan")
    if lock.get("package_versions") != package_versions():
        raise ValueError("v6 evaluation dependencies differ from the selection lock")
    code = lock.get("code_sha256")
    if not isinstance(code, dict) or set(code) != set(CODE_FILES):
        raise ValueError("v6 selection lock lacks complete inference code fingerprints")
    for relative in CODE_FILES:
        if sha256_file(ROOT / relative) != require_hex(code[relative], relative):
            raise ValueError(f"v6 inference code changed after selection: {relative}")
    observed_models = model_hashes()
    if (observed_models["new_bio"] != lock["new_bio_model_sha256"]
            or observed_models["dpo_inference_files"]
            != lock.get("dpo_inference_files_sha256")):
        raise ValueError("v6 model files differ from the selection lock")
    if sha256_file(DATA / "manifest.json") != lock["manifest_sha256"]:
        raise ValueError("v6 manifest differs from the selection lock")
    if sha256_file(DEV_REPORT) != lock["dev_report_sha256"]:
        raise ValueError("v6 development report differs from the selection lock")
    dev_report = json.loads(DEV_REPORT.read_text(encoding="utf-8"))
    candidate = dev_report.get("systems", {}).get("new_hybrid", {})
    metrics = candidate.get("metrics", {}) if isinstance(candidate, dict) else {}
    positive = candidate.get("positive_only", {}) if isinstance(candidate, dict) else {}
    required_metrics = {"micro_f1", "no_slot_failure_rate", "json_valid_rate",
                        "schema_valid_rate", "copy_valid_rate"}
    if (dev_report.get("role") != "dev"
            or dev_report.get("manifest_sha256") != lock["manifest_sha256"]
            or dev_report.get("train_file_sha256") != lock["train_file_sha256"]
            or dev_report.get("eval_file_sha256") != lock["dev_file_sha256"]
            or dev_report.get("model_sha256") != observed_models
            or dev_report.get("package_versions") != lock["package_versions"]
            or dev_report.get("code_sha256") != code
            or dev_report.get("confidence_threshold") != MEAN_LOGPROB_THRESHOLD
            or any(dev_report.get("generation", {}).get(key) != value
                   for key, value in GENERATION.items())
            or dev_report.get("bootstrap") != BOOTSTRAP["dev"]
            or not isinstance(metrics, dict) or not required_metrics <= metrics.keys()
            or not isinstance(positive, dict) or "micro_f1" not in positive
            or dev_report.get("development_gate") != development_gate(metrics, positive)
            or not dev_report["development_gate"]["passed"]):
        raise ValueError("v6 development did not identify and pass the frozen candidate")
    return lock


def check_test_preflight(lock: dict, manifest: dict) -> None:
    if (lock["train_file_sha256"] != manifest["splits"]["train"]["jsonl_sha256"]
            or lock["dev_file_sha256"] != manifest["splits"]["dev"]["jsonl_sha256"]
            or lock["test_file_sha256"] != manifest["splits"]["test"]["jsonl_sha256"]):
        raise ValueError("v6 split hashes differ from the selection lock")
    check_split_hashes(manifest, "test")


def load_bio(path: Path, *, expected_train_sha256: str | None = None,
             expected_manifest_sha256: str | None = None) -> dict:
    bundle = joblib.load(path)
    if bundle.get("configuration") != MODEL_CONFIG:
        raise ValueError(f"BIO configuration differs: {path.name}")
    if expected_train_sha256 is not None and (
            bundle.get("train_file_sha256") != expected_train_sha256
            or bundle.get("manifest_sha256") != expected_manifest_sha256):
        raise ValueError("New BIO model was not trained on the frozen v6 training split")
    if not all(key in bundle for key in ("vectorizer", "model")):
        raise ValueError(f"BIO model bundle is incomplete: {path.name}")
    return bundle


def bio_outputs(rows: list[dict], bundle: dict) -> list[str]:
    return [canonical_output(predict_slots(row["utt"], bundle["vectorizer"],
                                           bundle["model"])) for row in rows]


def choose_outputs(rows: list[dict], old_bio: list[str], new_bio: list[str],
                   dpo: list[str], confidence: list[float | None]) -> tuple[dict, dict]:
    if not (len(rows) == len(old_bio) == len(new_bio) == len(dpo) == len(confidence)):
        raise ValueError("v6 prediction arrays have different lengths")
    old_choice = [choose_hybrid_output(bio, raw, row["utt"], score)
                  for row, bio, raw, score in zip(rows, old_bio, dpo, confidence, strict=True)]
    new_choice = [choose_hybrid_output(bio, raw, row["utt"], score)
                  for row, bio, raw, score in zip(rows, new_bio, dpo, confidence, strict=True)]
    predictions = {"old_bio": old_bio, "new_bio": new_bio, "v4_dpo": dpo,
                   "old_v5_hybrid": [item[0] for item in old_choice],
                   "new_hybrid": [item[0] for item in new_choice]}
    replacements = {"old_v5_hybrid": sum(item[1] for item in old_choice),
                    "new_hybrid": sum(item[1] for item in new_choice)}
    return predictions, replacements


def development_gate(metrics: dict, positive: dict) -> dict:
    checks = {
        "strict_entity_f1_at_least_60pct": metrics["micro_f1"] is not None
        and metrics["micro_f1"] >= .60,
        "positive_subset_f1_at_least_60pct": positive["micro_f1"] is not None
        and positive["micro_f1"] >= .60,
        "no_slot_failure_at_most_20pct": metrics["no_slot_failure_rate"] <= .20,
        "json_valid_at_least_95pct": metrics["json_valid_rate"] >= .95,
        "schema_valid_at_least_95pct": metrics["schema_valid_rate"] >= .95,
        "copy_valid_at_least_95pct": metrics["copy_valid_rate"] >= .95,
    }
    return {"checks": checks, "passed": all(checks.values())}


def independent_test_gate(metrics: dict, positive: dict, paired: dict) -> dict:
    checks = development_gate(metrics, positive)["checks"]
    lower = paired["delta_ci95"][0]
    checks["new_hybrid_minus_old_v5_hybrid_ci95_lower_positive"] = (
        lower is not None and lower > 0)
    return {"checks": checks, "passed": all(checks.values())}


def write_confidence(path: Path, rows: list[dict],
                     values: list[float | None]) -> str:
    if not path.resolve().is_relative_to((ROOT / "_tmp").resolve()):
        raise ValueError("Confidence scores must remain in ignored _tmp")
    if len(rows) != len(values) or any(
            value is not None and not math.isfinite(value) for value in values):
        raise ValueError("Invalid v6 confidence values")
    content = "".join(json.dumps({"id": str(row["id"]),
                                   "group_sha256": row["group_sha256"],
                                   "mean_logprob": value},
                                  ensure_ascii=False, separators=(",", ":")) + "\n"
                      for row, value in zip(rows, values, strict=True)).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(content)
    return hashlib.sha256(content).hexdigest()


def run(role: str, lock: dict | None, expected_lock_sha256: str | None,
        manifest: dict, manifest_sha256: str,
        report_path: Path, prediction_paths: dict[str, Path],
        confidence_path: Path) -> dict:
    if role not in ("dev", "test") or (role == "test" and (
            lock is None or expected_lock_sha256 is None)):
        raise ValueError("Independent v6 test requires a verified selection lock")
    if role == "test":
        if lock != require_test_lock(expected_lock_sha256):
            raise ValueError("v6 test lock changed after preflight")
        if (manifest_sha256 != EXPECTED_MANIFEST_SHA256
                or sha256_file(DATA / "manifest.json") != manifest_sha256):
            raise ValueError("v6 test manifest changed after preflight")
        validate_manifest(manifest)
        check_test_preflight(lock, manifest)
    eval_file = DATA / f"{role}.jsonl"
    rows = read_jsonl(eval_file)
    validate_rows(rows, manifest, role)
    before = model_hashes()
    if lock is not None and (before["new_bio"] != lock["new_bio_model_sha256"]
                             or before["dpo_inference_files"]
                             != lock["dpo_inference_files_sha256"]):
        raise ValueError("v6 model changed before inference")
    old_bundle = load_bio(OLD_BIO)
    new_bundle = load_bio(
        NEW_BIO, expected_train_sha256=manifest["splits"]["train"]["jsonl_sha256"],
        expected_manifest_sha256=manifest_sha256)
    started = time.perf_counter()
    old_raw = bio_outputs(rows, old_bundle)
    new_raw = bio_outputs(rows, new_bundle)
    dpo_raw, inference = generate_raw_predictions(
        rows, model_name=str(BASE), adapter=str(DPO),
        batch_size=GENERATION["batch_size"],
        max_input_tokens=GENERATION["max_input_tokens"],
        max_new_tokens=GENERATION["max_new_tokens"])
    confidence, confidence_audit = v5.generate_confidence(rows, dpo_raw)
    if model_hashes() != before or sha256_file(eval_file) != manifest["splits"][role]["jsonl_sha256"]:
        raise ValueError("v6 model or evaluation data changed during inference")
    code_hashes = {name: sha256_file(ROOT / name) for name in CODE_FILES}
    if lock is not None and (
            code_hashes != lock["code_sha256"]
            or sha256_file(LOCK) != expected_lock_sha256):
        raise ValueError("v6 code or selection lock changed during independent test")
    predictions, replacements = choose_outputs(rows, old_raw, new_raw, dpo_raw, confidence)
    settings = BOOTSTRAP[role]
    systems = {
        name: {"metrics": score_predictions(rows, raw, bootstrap_samples=settings["samples"],
                                            seed=settings["seed"]),
               "positive_only": positive_only_metrics(rows, raw)}
        for name, raw in predictions.items()
    }
    paired = {
        f"{candidate}_minus_{baseline}": paired_bootstrap_f1_delta(
            rows, predictions[baseline], predictions[candidate],
            samples=settings["samples"], seed=settings["seed"])
        for baseline, candidate in combinations(SYSTEMS, 2)
    }
    primary = paired["new_hybrid_minus_old_v5_hybrid"]
    candidate = systems["new_hybrid"]
    gate = (development_gate(candidate["metrics"], candidate["positive_only"])
            if role == "dev" else independent_test_gate(
                candidate["metrics"], candidate["positive_only"], primary))
    eval_hash = sha256_file(eval_file)
    prediction_hashes = {
        name: write_predictions(path, prediction_payload(rows, predictions[name], eval_hash),
                                exclusive=True)
        for name, path in prediction_paths.items()
    }
    prediction_hashes["confidence"] = write_confidence(confidence_path, rows, confidence)
    report = {
        "study": "MASSIVE zh-CN four-slot v6 expanded BIO plus frozen DPO hybrid",
        "role": role,
        "source_partition": "train",
        "manifest_sha256": manifest_sha256,
        "train_file_sha256": manifest["splits"]["train"]["jsonl_sha256"],
        "eval_file_sha256": eval_hash,
        "selection_lock_sha256": expected_lock_sha256,
        "model_sha256": before,
        "package_versions": package_versions(),
        "code_sha256": code_hashes,
        "prediction_file_sha256": prediction_hashes,
        "generation": {**GENERATION, **inference},
        "confidence_threshold": MEAN_LOGPROB_THRESHOLD,
        "confidence_scored_nonempty": confidence_audit["eligible_nonempty"],
        "hybrid_replaced_examples": replacements,
        "bootstrap": settings,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "systems": systems,
        "paired_f1": paired,
        "development_gate" if role == "dev" else "independent_success_gate": gate,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("dev", "test"), default="dev")
    parser.add_argument("--unlock-test", action="store_true")
    parser.add_argument("--expected-selection-lock-sha256")
    args = parser.parse_args(argv)
    require_test_arguments(args)
    report_path, prediction_paths, confidence_path = check_output_paths(args.role)
    check_confidence_assets()
    lock = require_test_lock(args.expected_selection_lock_sha256) if args.role == "test" else None
    manifest_path = DATA / "manifest.json"
    manifest_sha256 = sha256_file(manifest_path)
    if manifest_sha256 != EXPECTED_MANIFEST_SHA256:
        raise ValueError("v6 manifest differs from the preregistered SHA-256")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    if lock is not None:
        if manifest_sha256 != lock["manifest_sha256"]:
            raise ValueError("v6 manifest changed after selection")
        check_test_preflight(lock, manifest)
    else:
        check_split_hashes(manifest, "dev")
    report = run(args.role, lock, args.expected_selection_lock_sha256.lower()
                 if lock is not None else None, manifest, manifest_sha256,
                 report_path, prediction_paths, confidence_path)
    print(json.dumps({"role": args.role,
                      "new_hybrid_f1": report["systems"]["new_hybrid"]["metrics"]["micro_f1"],
                      "old_v5_hybrid_f1": report["systems"]["old_v5_hybrid"]["metrics"]["micro_f1"],
                      "gate_passed": report["development_gate" if args.role == "dev"
                                            else "independent_success_gate"]["passed"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
