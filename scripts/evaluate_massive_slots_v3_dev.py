"""One fixed development evaluation for the preregistered v3 DPO adapter.

This module has no confirmation mode and never reads the sealed split.
"""

from __future__ import annotations

import json
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
from prepare_massive_slots import SOURCE_SHA256
from v2_model_assets import hash_inference_files


BASE = Path("models/Qwen2.5-0.5B-Instruct")
ADAPTER = Path("outputs/massive-slots-v3-dpo-64")
PREFERENCES = Path("data/massive-zh/slots-v3/preferences.jsonl")
PREFERENCE_MANIFEST = PREFERENCES.with_name("preferences-manifest.json")
DEV = Path("data/massive-zh/slots-v2/dev.jsonl")
V2_REPORT = Path("reports/massive-slots-v2-sft-step160-dev.json")
V2_PREDICTIONS = Path("_tmp/massive-slots-v2-v2_step160-dev-predictions.jsonl")
REPORT = Path("reports/massive-slots-v3-dpo-dev.json")
PREDICTIONS = Path("_tmp/massive-slots-v3-dpo-dev-predictions.jsonl")
EXPECTED_PREFERENCE_SHA256 = "310c26b37fd7eee7cbfb4876def2d398fcd5f0d188a24da20b24aa474471c450"
EXPECTED_SOURCE_ADAPTER_SHA256 = "206bc1eee5115e20844e22fa146893c9f651058709c6aff5f15e77ec63f61ce6"
EXPECTED_BASE_SHA256 = "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe"
EXPECTED_TOKENIZER_SHA256 = "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"
EXPECTED_DPO_ADAPTER_SHA256 = "634c83b63b840708e5d8ad63e710fe3a2f219924b3d4cad00ee6f6eb9d721c8a"


def adapter_dtypes(path: Path) -> list[str]:
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as archive:
        return sorted({str(archive.get_tensor(key).dtype) for key in archive.keys()})


def check_training_run(run: dict, preference_manifest: dict, hashes: dict[str, str]) -> None:
    expected = {
        "train_file_sha256": EXPECTED_PREFERENCE_SHA256,
        "source_adapter_sha256": EXPECTED_SOURCE_ADAPTER_SHA256,
        "reference_adapter_sha256": EXPECTED_SOURCE_ADAPTER_SHA256,
        "max_steps": 64,
        "global_step": 64,
        "max_length": 256,
        "learning_rate": 1e-5,
        "beta": 0.1,
        "seed": 20260930,
        "gradient_accumulation_steps": 4,
    }
    if any(run.get(key) != value for key, value in expected.items()):
        raise ValueError("DPO run differs from the frozen v3 training plan")
    if (preference_manifest.get("pair_count") != 256
            or preference_manifest.get("preference_file_sha256") != EXPECTED_PREFERENCE_SHA256
            or sha256_file(PREFERENCES) != EXPECTED_PREFERENCE_SHA256):
        raise ValueError("v3 preference pairs differ from the frozen run")
    if (hashes.get("model.safetensors") != EXPECTED_BASE_SHA256
            or hashes.get("tokenizer.json") != EXPECTED_TOKENIZER_SHA256
            or hashes.get("adapter_model.safetensors") != EXPECTED_DPO_ADAPTER_SHA256):
        raise ValueError("v3 inference assets differ from the frozen plan")
    if (sha256_file(ADAPTER / "ref" / "adapter_model.safetensors")
            != EXPECTED_SOURCE_ADAPTER_SHA256):
        raise ValueError("DPO frozen reference adapter differs from its SFT source")
    if adapter_dtypes(ADAPTER / "adapter_model.safetensors") != ["torch.bfloat16"]:
        raise ValueError("Saved DPO adapter precision differs from the recorded run")
    if (run.get("model_asset_sha256", {}).get("model.safetensors") != EXPECTED_BASE_SHA256
            or run.get("model_asset_sha256", {}).get("tokenizer.json") != EXPECTED_TOKENIZER_SHA256):
        raise ValueError("DPO training base or tokenizer differs from evaluation")


def development_gate(metrics: dict, positive_f1: float,
                     f1_comparison: dict, no_slot_comparison: dict) -> dict:
    checks = {
        "json_valid_at_least_95pct": metrics["json_valid_rate"] >= 0.95,
        "schema_valid_at_least_95pct": metrics["schema_valid_rate"] >= 0.95,
        "copy_valid_at_least_95pct": metrics["copy_valid_rate"] >= 0.95,
        "no_slot_failure_at_most_30pct": metrics["no_slot_failure_rate"] <= 0.30,
        "strict_entity_f1_at_least_55pct": metrics["micro_f1"] >= 0.55,
        "positive_subset_f1_at_least_60pct": positive_f1 >= 0.60,
        "paired_f1_ci95_lower_positive": f1_comparison["delta_ci95"][0] > 0,
        "paired_no_slot_reduction_ci95_lower_positive": no_slot_comparison["reduction_ci95"][0] > 0,
    }
    return {"checks": checks, "passed": all(checks.values())}


def main() -> None:
    if REPORT.exists() or PREDICTIONS.exists():
        raise ValueError("v3 final development evaluation already exists")
    started = time.perf_counter()
    run = json.loads((ADAPTER / "run_manifest.json").read_text(encoding="utf-8"))
    preference_manifest = json.loads(PREFERENCE_MANIFEST.read_text(encoding="utf-8"))
    hashes = hash_inference_files(BASE, ADAPTER)
    check_training_run(run, preference_manifest, hashes)
    dev_manifest = load_locked_split_manifest(DEV, "dev")
    check_manifest_preflight(dev_manifest, DEV.parent, role="dev")
    if sha256_file(Path("data/raw/amazon-massive-dataset-1.0.tar.gz")) != SOURCE_SHA256:
        raise ValueError("Official MASSIVE archive differs")
    rows = read_jsonl(DEV)
    _validate_study_split(rows, DEV, "dev", dev_manifest)
    baseline_report = json.loads(V2_REPORT.read_text(encoding="utf-8"))
    dev_sha256 = sha256_file(DEV)
    if (baseline_report.get("eval_file_sha256") != dev_sha256
            or baseline_report.get("prediction_file_sha256") != sha256_file(V2_PREDICTIONS)
            or baseline_report.get("frozen_model_files_sha256", {}).get("adapter_model.safetensors")
            != EXPECTED_SOURCE_ADAPTER_SHA256):
        raise ValueError("Frozen v2 SFT baseline differs")
    baseline_raw = aligned_raw_outputs(rows, read_jsonl(V2_PREDICTIONS), dev_sha256)
    raw, inference = generate_raw_predictions(
        rows, model_name=str(BASE), adapter=str(ADAPTER),
        batch_size=GENERATION["batch_size"],
        max_input_tokens=GENERATION["max_input_tokens"],
        max_new_tokens=GENERATION["max_new_tokens"],
    )
    if hash_inference_files(BASE, ADAPTER) != hashes:
        raise ValueError("v3 model changed during development inference")
    metrics = score_predictions(rows, raw, bootstrap_samples=1000, seed=20260929)
    positive = positive_only_metrics(rows, raw)
    f1 = paired_bootstrap_f1_delta(rows, baseline_raw, raw, samples=1000, seed=20260929)
    no_slot = paired_no_slot_reduction(rows, baseline_raw, raw, samples=1000, seed=20260929)
    gate = development_gate(metrics, positive["micro_f1"], f1, no_slot)
    payload = prediction_payload(rows, raw, dev_sha256)
    prediction_sha256 = write_predictions(PREDICTIONS, payload, exclusive=True)
    report = {
        "study": "MASSIVE zh-CN four-slot v3 DPO development only",
        "system": "v3_dpo_step64",
        "role": "dev",
        "source_partition": "train",
        "source_sha256": SOURCE_SHA256,
        "dev_file_sha256": dev_sha256,
        "dev_manifest_sha256": sha256_file(DEV.with_name("manifest.json")),
        "preference_file_sha256": EXPECTED_PREFERENCE_SHA256,
        "preference_manifest_sha256": sha256_file(PREFERENCE_MANIFEST),
        "training_run_manifest_sha256": sha256_file(ADAPTER / "run_manifest.json"),
        "inference_files_sha256": hashes,
        "saved_adapter_dtypes": adapter_dtypes(ADAPTER / "adapter_model.safetensors"),
        "frozen_reference_adapter_sha256": EXPECTED_SOURCE_ADAPTER_SHA256,
        "prediction_file_sha256": prediction_sha256,
        "baseline_prediction_file_sha256": baseline_report["prediction_file_sha256"],
        "generation": {**GENERATION, **inference},
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "metrics": metrics,
        "positive_only": positive,
        "v3_minus_v2_sft_paired_f1": f1,
        "v2_sft_minus_v3_no_slot_failure": no_slot,
        "development_gate": gate,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with REPORT.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "micro_f1": metrics["micro_f1"],
        "no_slot_failure_rate": metrics["no_slot_failure_rate"],
        "development_gate_passed": gate["passed"],
        "report": str(REPORT),
    }))


if __name__ == "__main__":
    main()
