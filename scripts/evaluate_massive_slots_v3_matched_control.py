"""One exploratory dev comparison for same-example continued SFT versus DPO."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from compare_massive_slots import aligned_raw_outputs
from compare_massive_slots_v2 import paired_no_slot_reduction, positive_only_metrics
from evaluate_massive_slots import (
    _validate_study_split,
    generate_raw_predictions,
    load_locked_split_manifest,
    paired_bootstrap_f1_delta,
    read_jsonl,
    score_predictions,
    sha256_file,
)
from evaluate_massive_slots_v2 import (
    GENERATION, check_manifest_preflight, prediction_payload,
    write_predictions,
)
from evaluate_massive_slots_v3_dev import adapter_dtypes
from prepare_massive_slots import SOURCE_SHA256
from v2_model_assets import hash_inference_files


BASE = Path("models/Qwen2.5-0.5B-Instruct")
ADAPTER = Path("outputs/massive-slots-v3-matched-sft-64")
TRAIN = Path("data/massive-zh/slots-v3/matched-sft.jsonl")
DEV = Path("data/massive-zh/slots-v2/dev.jsonl")
V2_REPORT = Path("reports/massive-slots-v2-sft-step160-dev.json")
V2_PREDICTIONS = Path("_tmp/massive-slots-v2-v2_step160-dev-predictions.jsonl")
DPO_REPORT = Path("reports/massive-slots-v3-dpo-dev.json")
DPO_PREDICTIONS = Path("_tmp/massive-slots-v3-dpo-dev-predictions.jsonl")
REPORT = Path("reports/massive-slots-v3-matched-sft-dev.json")
PREDICTIONS = Path("_tmp/massive-slots-v3-matched-sft-dev-predictions.jsonl")
EXPECTED_TRAIN_SHA256 = "8de93fdabe1b580def8bb10bef6e6cdb05942ff621bc29c0e7a34db1f0eed25c"
EXPECTED_START_SHA256 = "206bc1eee5115e20844e22fa146893c9f651058709c6aff5f15e77ec63f61ce6"
EXPECTED_BASE_SHA256 = "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe"
EXPECTED_TOKENIZER_SHA256 = "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"


def check_run(run: dict, hashes: dict[str, str], expected_adapter: str,
              expected_dtype: str) -> None:
    expected = {
        "train_file_sha256": EXPECTED_TRAIN_SHA256,
        "initial_adapter_sha256": EXPECTED_START_SHA256,
        "max_steps": 64,
        "global_step": 64,
        "max_length": 256,
        "learning_rate": 1e-5,
        "seed": 20260930,
        "gradient_accumulation_steps": 4,
        "save_steps": 32,
    }
    if any(run.get(key) != value for key, value in expected.items()):
        raise ValueError("Continued SFT run differs from the frozen control plan")
    if sha256_file(TRAIN) != EXPECTED_TRAIN_SHA256:
        raise ValueError("Matched SFT training data differs")
    if (hashes.get("model.safetensors") != EXPECTED_BASE_SHA256
            or hashes.get("tokenizer.json") != EXPECTED_TOKENIZER_SHA256
            or hashes.get("adapter_model.safetensors") != expected_adapter):
        raise ValueError("Control inference model differs from the preflight fingerprint")
    if (run.get("model_asset_sha256", {}).get("model.safetensors") != EXPECTED_BASE_SHA256
            or run.get("model_asset_sha256", {}).get("tokenizer.json") != EXPECTED_TOKENIZER_SHA256):
        raise ValueError("Control training model differs from inference")
    if adapter_dtypes(ADAPTER / "adapter_model.safetensors") != [expected_dtype]:
        raise ValueError("Saved control adapter precision differs from preflight")


def checked_reference(rows: list[dict], dev_sha256: str,
                      report_path: Path, predictions_path: Path,
                      *, role: str) -> tuple[list[str], dict]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (report.get("prediction_file_sha256") != sha256_file(predictions_path)
            or report.get("eval_file_sha256", report.get("dev_file_sha256")) != dev_sha256):
        raise ValueError(f"Frozen {role} reference report differs from predictions or dev split")
    if role == "v2" and report.get("frozen_model_files_sha256", {}).get(
            "adapter_model.safetensors") != EXPECTED_START_SHA256:
        raise ValueError("Frozen v2 SFT reference differs")
    if role == "dpo" and report.get("system") != "v3_dpo_step64":
        raise ValueError("Frozen v3 DPO reference differs")
    return aligned_raw_outputs(rows, read_jsonl(predictions_path), dev_sha256), report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-adapter-sha256", required=True)
    parser.add_argument("--expected-adapter-dtype", choices=("torch.float32", "torch.bfloat16"),
                        required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-fA-F]{64}", args.expected_adapter_sha256):
        parser.error("--expected-adapter-sha256 must be a 64-character SHA-256")
    if REPORT.exists() or PREDICTIONS.exists():
        raise ValueError("Matched SFT final development evaluation already exists")
    started = time.perf_counter()
    hashes = hash_inference_files(BASE, ADAPTER)
    run = json.loads((ADAPTER / "run_manifest.json").read_text(encoding="utf-8"))
    check_run(run, hashes, args.expected_adapter_sha256.lower(), args.expected_adapter_dtype)
    manifest = load_locked_split_manifest(DEV, "dev")
    check_manifest_preflight(manifest, DEV.parent, role="dev")
    if sha256_file(Path("data/raw/amazon-massive-dataset-1.0.tar.gz")) != SOURCE_SHA256:
        raise ValueError("Official MASSIVE archive differs")
    rows = read_jsonl(DEV)
    _validate_study_split(rows, DEV, "dev", manifest)
    dev_sha256 = sha256_file(DEV)
    v2_raw, v2_report = checked_reference(rows, dev_sha256, V2_REPORT, V2_PREDICTIONS, role="v2")
    dpo_raw, dpo_report = checked_reference(rows, dev_sha256, DPO_REPORT, DPO_PREDICTIONS,
                                             role="dpo")
    raw, inference = generate_raw_predictions(
        rows, model_name=str(BASE), adapter=str(ADAPTER),
        batch_size=GENERATION["batch_size"],
        max_input_tokens=GENERATION["max_input_tokens"],
        max_new_tokens=GENERATION["max_new_tokens"],
    )
    if hash_inference_files(BASE, ADAPTER) != hashes:
        raise ValueError("Control model changed during development inference")
    metrics = score_predictions(rows, raw, bootstrap_samples=1000, seed=20260929)
    positive = positive_only_metrics(rows, raw)
    comparison = {
        "control_minus_v2_sft_f1": paired_bootstrap_f1_delta(
            rows, v2_raw, raw, samples=1000, seed=20260929),
        "v2_sft_minus_control_no_slot_failure": paired_no_slot_reduction(
            rows, v2_raw, raw, samples=1000, seed=20260929),
        "dpo_minus_control_f1": paired_bootstrap_f1_delta(
            rows, raw, dpo_raw, samples=1000, seed=20260929),
        "control_minus_dpo_no_slot_failure": paired_no_slot_reduction(
            rows, raw, dpo_raw, samples=1000, seed=20260929),
    }
    payload = prediction_payload(rows, raw, dev_sha256)
    prediction_sha256 = write_predictions(PREDICTIONS, payload, exclusive=True)
    report = {
        "study": "MASSIVE zh-CN four-slot same-example continued-SFT exploratory control",
        "role": "dev",
        "source_partition": "train",
        "source_sha256": SOURCE_SHA256,
        "system": "v3_matched_sft_step64",
        "dev_file_sha256": dev_sha256,
        "dev_manifest_sha256": sha256_file(DEV.with_name("manifest.json")),
        "train_file_sha256": EXPECTED_TRAIN_SHA256,
        "training_run_manifest_sha256": sha256_file(ADAPTER / "run_manifest.json"),
        "inference_files_sha256": hashes,
        "saved_adapter_dtypes": [args.expected_adapter_dtype],
        "prediction_file_sha256": prediction_sha256,
        "reference_prediction_file_sha256": {
            "v2_sft": v2_report["prediction_file_sha256"],
            "v3_dpo": dpo_report["prediction_file_sha256"],
        },
        "generation": {**GENERATION, **inference},
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "metrics": metrics,
        "positive_only": positive,
        "exploratory_paired_comparison": comparison,
        "confirmation_used": False,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with REPORT.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"micro_f1": metrics["micro_f1"],
                      "no_slot_failure_rate": metrics["no_slot_failure_rate"],
                      "positive_micro_f1": positive["micro_f1"],
                      "report": str(REPORT)}))


if __name__ == "__main__":
    main()
