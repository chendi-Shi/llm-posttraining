"""Hash-locked Qwen evaluation for the independent MASSIVE four-slot v2 study.

Development records the exact model bytes used for each run. Confirmation
requires the previously selected SFT lock before any utterance is parsed.
Per-row predictions are permitted only under the ignored local _tmp directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

from evaluate_massive_slots import (
    _validate_study_split,
    generate_raw_predictions,
    load_locked_split_manifest,
    read_jsonl,
    score_predictions,
    sha256_file,
)
from massive_slots import parse_prediction
from prepare_massive_slots import SOURCE_SHA256
from v2_model_assets import hash_inference_files


ROOT = Path(__file__).resolve().parents[1]
LOCK_OUTPUT = Path("reports/massive-slots-v2-selection-lock.json")
EXPECTED_COUNTS = {"train": 640, "dev": 250, "confirmation": 400}
GENERATION = {
    "decoding": "greedy_unconstrained",
    "do_sample": False,
    "batch_size": 8,
    "max_input_tokens": 256,
    "max_new_tokens": 64,
}
SYSTEM_ADAPTER = {
    "v2_step80": Path("outputs/massive-slots-v2-sft-lr3e4-160/checkpoint-80"),
    "v2_step160": Path("outputs/massive-slots-v2-sft-lr3e4-160"),
    "v1": Path("outputs/massive-slots-sft-lr3e4-96"),
}
DEV_REPORT = {
    "v2_step80": Path("reports/massive-slots-v2-sft-step80-dev.json"),
    "v2_step160": Path("reports/massive-slots-v2-sft-step160-dev.json"),
    "v1": Path("reports/massive-slots-v2-v1sft-dev.json"),
}
CONFIRMATION_REPORT = {
    "v2_step80": Path("reports/massive-slots-v2-sft-confirmation.json"),
    "v2_step160": Path("reports/massive-slots-v2-sft-confirmation.json"),
    "v1": Path("reports/massive-slots-v2-v1sft-confirmation.json"),
}
CODE_FILES = (
    "scripts/evaluate_massive_slots_v2.py",
    "scripts/evaluate_massive_slots.py",
    "scripts/prepare_massive_slots.py",
    "scripts/massive_slots.py",
    "scripts/common.py",
    "scripts/v2_model_assets.py",
)


def require_hex(value: object, name: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", value):
        raise ValueError(f"A 64-character SHA-256 is required for {name}")
    return value.lower()


def require_confirmation_arguments(args: argparse.Namespace) -> None:
    """Check all switches before the first filesystem read."""
    if args.role != "confirmation":
        return
    if not args.unlock_confirmation:
        raise ValueError("Confirmation requires --unlock-confirmation")
    if args.selection_lock.resolve() != LOCK_OUTPUT.resolve():
        raise ValueError(f"Confirmation requires the canonical selection lock {LOCK_OUTPUT}")
    require_hex(args.expected_selection_lock_sha256, "--expected-selection-lock-sha256")


def model_file_hashes(model_dir: Path, adapter_dir: Path) -> dict[str, str]:
    return hash_inference_files(model_dir, adapter_dir)


def confirm_unchanged(before: dict[str, str], after: dict[str, str]) -> None:
    if before != after:
        raise ValueError("Model, tokenizer, or adapter changed during v2 generation")


def check_manifest_preflight(manifest: dict, data_dir: Path, *, role: str) -> None:
    if (
        manifest.get("format_version") != 1
        or manifest.get("source_partition_used") != "train"
        or "v2" not in str(manifest.get("study", "")).lower()
        or manifest.get("source_sha256") != SOURCE_SHA256
    ):
        raise ValueError("Expected the locked MASSIVE v2 manifest")
    splits = manifest.get("splits", {})
    if {name: splits[name]["examples"] for name in EXPECTED_COUNTS} != EXPECTED_COUNTS:
        raise ValueError("Unexpected v2 split sizes")
    for name in (("train", "dev", "confirmation") if role == "confirmation" else ("train", "dev")):
        if sha256_file(data_dir / f"{name}.jsonl") != splits[name]["jsonl_sha256"]:
            raise ValueError(f"v2 {name} differs from its manifest")


def verify_selection_lock(
    args: argparse.Namespace, manifest: dict, manifest_path: Path,
    model_hashes: dict[str, str], adapter_dir: Path,
) -> dict:
    """Read only hashes/metadata; refuse before confirmation JSONL is parsed."""
    expected_lock_hash = require_hex(
        args.expected_selection_lock_sha256, "--expected-selection-lock-sha256"
    )
    if sha256_file(args.selection_lock) != expected_lock_hash:
        raise ValueError("v2 selection lock SHA-256 mismatch")
    lock = json.loads(args.selection_lock.read_text(encoding="utf-8"))
    if lock.get("status") != "selected" or lock.get("selected_step") not in (80, 160):
        raise ValueError("The v2 SFT checkpoint has not been selected")
    if lock.get("manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("v2 manifest differs from the selection lock")
    if lock.get("confirmation_file_sha256") != manifest["splits"]["confirmation"]["jsonl_sha256"]:
        raise ValueError("v2 confirmation split differs from the selection lock")
    if lock.get("dev_file_sha256") != manifest["splits"]["dev"]["jsonl_sha256"]:
        raise ValueError("v2 development split differs from the selection lock")
    base_hashes = {key: value for key, value in model_hashes.items()
                   if key != "adapter_model.safetensors" and not key.startswith("adapter/")}
    adapter_hashes = {key: value for key, value in model_hashes.items()
                      if key == "adapter_model.safetensors" or key.startswith("adapter/")}
    if (lock.get("base_model_files_sha256") != base_hashes
            or lock.get("base_model_sha256") != model_hashes["model.safetensors"]
            or lock.get("tokenizer_sha256") != model_hashes["tokenizer.json"]):
        raise ValueError("Qwen base or tokenizer differs from the v2 selection lock")

    if args.system == "v1":
        v1_reference = lock.get("reference_systems", {}).get("v1", {})
        expected_adapter = require_hex(v1_reference.get("adapter_sha256"), "v1 adapter in lock")
        expected_adapter_hashes = v1_reference.get("adapter_files_sha256")
        locked_v1_path = v1_reference.get("adapter_path")
        if not isinstance(locked_v1_path, str) or Path(locked_v1_path).resolve() != adapter_dir.resolve():
            raise ValueError("v1 adapter path differs from the selection lock")
    else:
        selected_system = f"v2_step{lock['selected_step']}"
        if args.system != selected_system:
            raise ValueError("Confirmation may use only the selected v2 SFT checkpoint")
        expected_adapter = require_hex(lock.get("selected_adapter_sha256"), "selected_adapter_sha256 in lock")
        expected_adapter_hashes = lock.get("selected_adapter_files_sha256")
        if Path(lock.get("selected_sft_checkpoint", "")).resolve() != adapter_dir.resolve():
            raise ValueError("Selected v2 adapter path differs from the selection lock")
    if (model_hashes["adapter_model.safetensors"] != expected_adapter
            or adapter_hashes != expected_adapter_hashes):
        raise ValueError("SFT adapter differs from the v2 selection lock")

    code_hashes = lock.get("code_sha256")
    if not isinstance(code_hashes, dict):
        raise ValueError("v2 selection lock has no code fingerprints")
    for relative in CODE_FILES:
        expected = require_hex(code_hashes.get(relative), f"{relative} in lock")
        if sha256_file(ROOT / relative) != expected:
            raise ValueError(f"v2 evaluation code changed after selection: {relative}")
    generation = lock.get("generation")
    if not isinstance(generation, dict) or any(generation.get(key) != value for key, value in GENERATION.items()):
        raise ValueError("Generation settings differ from the v2 selection lock")
    return lock


def prediction_payload(rows: list[dict], raw: list[str], eval_sha256: str) -> bytes:
    lines = []
    for row, output in zip(rows, raw, strict=True):
        parsed, validity = parse_prediction(output, row["utt"])
        lines.append(json.dumps({
            "id": str(row["id"]),
            "group_sha256": row["group_sha256"],
            "eval_file_sha256": eval_sha256,
            "raw_output": output,
            "prediction": parsed,
            "validity": validity,
        }, ensure_ascii=False, separators=(",", ":")) + "\n")
    return "".join(lines).encode("utf-8")


def write_predictions(path: Path, content: bytes, *, exclusive: bool = False) -> str:
    if not path.resolve().is_relative_to((ROOT / "_tmp").resolve()):
        raise ValueError("Per-row v2 predictions must stay under the ignored _tmp directory")
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        with path.open("xb") as stream:
            stream.write(content)
    else:
        path.write_bytes(content)
    observed = sha256_file(path)
    if observed != hashlib.sha256(content).hexdigest():
        raise ValueError("Written v2 predictions differ from generated bytes")
    return observed


def default_report(system: str, role: str) -> Path:
    return DEV_REPORT[system] if role == "dev" else CONFIRMATION_REPORT[system]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=tuple(SYSTEM_ADAPTER), required=True)
    parser.add_argument("--role", choices=("dev", "confirmation"), default="dev")
    parser.add_argument("--unlock-confirmation", action="store_true")
    parser.add_argument("--selection-lock", type=Path, default=LOCK_OUTPUT)
    parser.add_argument("--expected-selection-lock-sha256")
    parser.add_argument("--data-dir", type=Path, default=Path("data/massive-zh/slots-v2"))
    parser.add_argument("--archive", type=Path, default=Path("data/raw/amazon-massive-dataset-1.0.tar.gz"))
    parser.add_argument("--model", type=Path, default=Path("models/Qwen2.5-0.5B-Instruct"))
    parser.add_argument("--adapter", type=Path, default=None, help="Override this system's default adapter path")
    parser.add_argument("--metrics", type=Path, default=None)
    parser.add_argument("--predictions", type=Path, default=None)
    args = parser.parse_args(argv)
    require_confirmation_arguments(args)
    if args.role == "confirmation" and (args.metrics is not None or args.predictions is not None):
        parser.error("Confirmation uses fixed metrics and prediction paths once")

    adapter = args.adapter or SYSTEM_ADAPTER[args.system]
    eval_path = args.data_dir / f"{args.role}.jsonl"
    manifest_path = args.data_dir / "manifest.json"
    metrics_path = args.metrics or default_report(args.system, args.role)
    predictions_path = args.predictions or Path(
        f"_tmp/massive-slots-v2-{args.system}-{args.role}-predictions.jsonl"
    )
    if not predictions_path.resolve().is_relative_to((ROOT / "_tmp").resolve()):
        parser.error("--predictions must be inside the repository's ignored _tmp directory")
    if metrics_path.resolve() == predictions_path.resolve():
        parser.error("Aggregate metrics and per-row predictions must use different paths")
    if args.role == "confirmation" and (metrics_path.exists() or predictions_path.exists()):
        raise ValueError("Confirmation outputs already exist; refusing a second run")

    started = time.perf_counter()
    # Preflight uses opaque hashes and manifest metadata before reading utterances.
    manifest = load_locked_split_manifest(eval_path, args.role)
    check_manifest_preflight(manifest, args.data_dir, role=args.role)
    if sha256_file(args.archive) != SOURCE_SHA256:
        raise ValueError("Official MASSIVE 1.0 archive SHA-256 mismatch")
    before = model_file_hashes(args.model, adapter)
    lock_sha256 = None
    if args.role == "confirmation":
        verify_selection_lock(args, manifest, manifest_path, before, adapter)
        lock_sha256 = args.expected_selection_lock_sha256.lower()

    rows = read_jsonl(eval_path)
    _validate_study_split(rows, eval_path, args.role, manifest)
    raw, inference = generate_raw_predictions(
        rows,
        model_name=str(args.model),
        adapter=str(adapter),
        batch_size=GENERATION["batch_size"],
        max_input_tokens=GENERATION["max_input_tokens"],
        max_new_tokens=GENERATION["max_new_tokens"],
    )
    after = model_file_hashes(args.model, adapter)
    confirm_unchanged(before, after)
    metrics = score_predictions(rows, raw, bootstrap_samples=1000, seed=20260929)
    eval_sha256 = sha256_file(eval_path)
    payload = prediction_payload(rows, raw, eval_sha256)
    prediction_sha256 = write_predictions(
        predictions_path, payload, exclusive=args.role == "confirmation"
    )
    metrics.update({
        "task": "MASSIVE 1.0 zh-CN four-slot copy-only JSON extraction",
        "role": args.role,
        "system": args.system,
        "source_partition": "train",
        "eval_file": str(eval_path),
        "eval_file_sha256": eval_sha256,
        "study_manifest_sha256": sha256_file(manifest_path),
        "source_sha256": SOURCE_SHA256,
        "model": str(args.model),
        "adapter": str(adapter),
        "frozen_model_files_sha256": before,
        "selection_lock_sha256": lock_sha256,
        "prediction_file_sha256": prediction_sha256,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "generation": {**GENERATION, **inference},
    })
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    if args.role == "confirmation":
        with metrics_path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")
    else:
        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "system": args.system,
        "role": args.role,
        "micro_f1": metrics["micro_f1"],
        "no_slot_failure_rate": metrics["no_slot_failure_rate"],
        "prediction_file_sha256": prediction_sha256,
        "metrics": str(metrics_path),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
