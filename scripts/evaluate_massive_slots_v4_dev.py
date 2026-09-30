"""Evaluate the single v4 DPO candidate against frozen v3 DPO on new dev."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from build_massive_slots_v3_preferences import ADAPTER_SHA256, BASE_SHA256, TOKENIZER_SHA256
from compare_massive_slots_v2 import paired_no_slot_reduction, positive_only_metrics
from evaluate_massive_slots import (
    generate_raw_predictions, paired_bootstrap_f1_delta, read_jsonl,
    score_predictions, sha256_file,
)
from evaluate_massive_slots_v2 import prediction_payload, write_predictions
from prepare_massive_slots_v4 import V3_MANIFEST_SHA256
from v2_model_assets import hash_inference_files


DATA = Path("data/massive-zh/slots-v4")
BASE = Path("models/Qwen2.5-0.5B-Instruct")
V3 = Path("outputs/massive-slots-v3-dpo-64")
V4 = Path("outputs/massive-slots-v4-balanced-dpo-64")
EXPECTED_MANIFEST_SHA256 = "c3bc00cadb70fc6c7a7487e19a2e23f6e57726b4b97cbfaa3582c9835395c206"
EXPECTED_V3_ADAPTER_SHA256 = "634c83b63b840708e5d8ad63e710fe3a2f219924b3d4cad00ee6f6eb9d721c8a"
REPORT = Path("reports/massive-slots-v4-dev.json")
V3_PREDICTIONS = Path("_tmp/massive-slots-v4-v3-dpo-dev-predictions.jsonl")
V4_PREDICTIONS = Path("_tmp/massive-slots-v4-balanced-dpo-dev-predictions.jsonl")


def gate(metrics: dict, positive: dict, paired_f1: dict) -> dict:
    checks = {
        "strict_entity_f1_at_least_55pct": metrics["micro_f1"] >= .55,
        "positive_subset_f1_at_least_60pct": positive["micro_f1"] >= .60,
        "no_slot_failure_at_most_30pct": metrics["no_slot_failure_rate"] <= .30,
        "json_valid_at_least_95pct": metrics["json_valid_rate"] >= .95,
        "schema_valid_at_least_95pct": metrics["schema_valid_rate"] >= .95,
        "copy_valid_at_least_95pct": metrics["copy_valid_rate"] >= .95,
        "paired_f1_ci95_lower_positive": paired_f1["delta_ci95"][0] > 0,
    }
    return {"checks": checks, "passed": all(checks.values())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-adapter-sha256", required=True)
    args = parser.parse_args()
    if REPORT.exists() or V3_PREDICTIONS.exists() or V4_PREDICTIONS.exists():
        raise ValueError("v4 development evaluation is single-use")
    if sha256_file(DATA / "manifest.json") != EXPECTED_MANIFEST_SHA256:
        raise ValueError("v4 split manifest differs")
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    if manifest["v3_manifest_sha256"] != V3_MANIFEST_SHA256:
        raise ValueError("v4 reservation differs")
    dev = DATA / "dev.jsonl"
    expected = manifest["splits"]["dev"]
    if sha256_file(dev) != expected["jsonl_sha256"] or expected["examples"] != 250:
        raise ValueError("v4 development split differs")
    rows = read_jsonl(dev)
    if [row["group_sha256"] for row in rows] != expected["group_sha256"]:
        raise ValueError("v4 development group order differs")
    for split in ("train", "confirmation"):
        if sha256_file(DATA / f"{split}.jsonl") != manifest["splits"][split]["jsonl_sha256"]:
            raise ValueError(f"v4 {split} file hash differs")
    preference_file = DATA / "preferences.jsonl"
    preference_manifest = json.loads((DATA / "preferences-manifest.json").read_text(encoding="utf-8"))
    if (preference_manifest["v4_manifest_sha256"] != EXPECTED_MANIFEST_SHA256
            or sha256_file(preference_file) != preference_manifest["preference_file_sha256"]
            or preference_manifest["pair_count"] != 256):
        raise ValueError("v4 preference pairs differ")
    run = json.loads((V4 / "run_manifest.json").read_text(encoding="utf-8"))
    if (run["train_file_sha256"] != preference_manifest["preference_file_sha256"]
            or run["source_adapter_sha256"] != ADAPTER_SHA256
            or run["reference_adapter_sha256"] != ADAPTER_SHA256
            or run["max_steps"] != 64 or run["global_step"] != 64
            or run["beta"] != .1 or run["learning_rate"] != 1e-5
            or run["seed"] != 20261001):
        raise ValueError("v4 training run differs from the frozen plan")
    before_v3 = hash_inference_files(BASE, V3)
    before_v4 = hash_inference_files(BASE, V4)
    for hashes, adapter_hash in ((before_v3, EXPECTED_V3_ADAPTER_SHA256),
                                 (before_v4, args.expected_adapter_sha256)):
        if (hashes.get("model.safetensors") != BASE_SHA256
                or hashes.get("tokenizer.json") != TOKENIZER_SHA256
                or hashes.get("adapter_model.safetensors") != adapter_hash):
            raise ValueError("A frozen model or adapter differs")
    started = time.perf_counter()
    generation = {"batch_size": 8, "max_input_tokens": 256, "max_new_tokens": 64}
    v3_raw, v3_inference = generate_raw_predictions(
        rows, model_name=str(BASE), adapter=str(V3), **generation)
    v4_raw, v4_inference = generate_raw_predictions(
        rows, model_name=str(BASE), adapter=str(V4), **generation)
    if (hash_inference_files(BASE, V3) != before_v3
            or hash_inference_files(BASE, V4) != before_v4):
        raise ValueError("Models changed during v4 development inference")
    v3_metrics = score_predictions(rows, v3_raw, bootstrap_samples=1000, seed=20261001)
    v4_metrics = score_predictions(rows, v4_raw, bootstrap_samples=1000, seed=20261001)
    v3_positive = positive_only_metrics(rows, v3_raw)
    v4_positive = positive_only_metrics(rows, v4_raw)
    paired_f1 = paired_bootstrap_f1_delta(rows, v3_raw, v4_raw,
                                          samples=1000, seed=20261001)
    no_slot = paired_no_slot_reduction(rows, v3_raw, v4_raw,
                                       samples=1000, seed=20261001)
    dev_hash = sha256_file(dev)
    v3_prediction_hash = write_predictions(
        V3_PREDICTIONS, prediction_payload(rows, v3_raw, dev_hash), exclusive=True)
    v4_prediction_hash = write_predictions(
        V4_PREDICTIONS, prediction_payload(rows, v4_raw, dev_hash), exclusive=True)
    decision = gate(v4_metrics, v4_positive, paired_f1)
    report = {
        "study": "MASSIVE zh-CN four-slot v4 balanced-preference DPO development",
        "role": "dev",
        "v4_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "dev_file_sha256": dev_hash,
        "preference_file_sha256": preference_manifest["preference_file_sha256"],
        "v4_training_run_manifest_sha256": sha256_file(V4 / "run_manifest.json"),
        "inference_files_sha256": {"v3_dpo": before_v3, "v4_balanced_dpo": before_v4},
        "prediction_file_sha256": {"v3_dpo": v3_prediction_hash,
                                   "v4_balanced_dpo": v4_prediction_hash},
        "generation": {"decoding": "greedy_unconstrained", **generation,
                       "v3": v3_inference, "v4": v4_inference},
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "v3_dpo": {"metrics": v3_metrics, "positive_only": v3_positive},
        "v4_balanced_dpo": {"metrics": v4_metrics, "positive_only": v4_positive},
        "v4_minus_v3_paired_f1": paired_f1,
        "v3_minus_v4_no_slot_failure": no_slot,
        "development_gate": decision,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with REPORT.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"v3_f1": v3_metrics["micro_f1"],
                      "v4_f1": v4_metrics["micro_f1"],
                      "v4_positive_f1": v4_positive["micro_f1"],
                      "v4_no_slot_failure": v4_metrics["no_slot_failure_rate"],
                      "gate_passed": decision["passed"]}))


if __name__ == "__main__":
    main()
