"""Create the single-use v6 test selection lock after a passing dev report.

The test JSONL is never opened here. Its SHA-256 comes from the preregistered
manifest and the public constant in the v6 evaluator. This helper is deliberately
outside the evaluator's frozen inference-code fingerprint list.
"""

from __future__ import annotations

import json
from pathlib import Path

import evaluate_massive_slots_v6_hybrid as v6


def build_lock(
    *, manifest: dict, manifest_sha256: str, dev_report: dict,
    dev_report_sha256: str, models: dict, code_sha256: dict[str, str],
    versions: dict[str, str],
) -> dict:
    """Check dev evidence and compose only fields accepted by require_test_lock."""
    if manifest_sha256 != v6.EXPECTED_MANIFEST_SHA256:
        raise ValueError("v6 manifest differs from its preregistered SHA-256")
    splits = manifest["splits"]
    if any(splits[role]["jsonl_sha256"] != v6.EXPECTED_FILE_SHA256[role]
           for role in v6.EXPECTED_COUNTS):
        raise ValueError("v6 split SHA-256 differs from preregistration")
    if (models.get("old_bio") != v6.OLD_BIO_SHA256
            or models.get("dpo_inference_files", {}).get("model.safetensors") != v6.BASE_SHA256
            or models.get("dpo_inference_files", {}).get("tokenizer.json") != v6.TOKENIZER_SHA256
            or models.get("dpo_inference_files", {}).get("adapter_model.safetensors") != v6.DPO_SHA256):
        raise ValueError("Frozen BIO or DPO assets differ from the v6 plan")
    if set(code_sha256) != set(v6.CODE_FILES):
        raise ValueError("v6 code fingerprint list is incomplete")
    candidate = dev_report.get("systems", {}).get("new_hybrid", {})
    metrics = candidate.get("metrics", {}) if isinstance(candidate, dict) else {}
    positive = candidate.get("positive_only", {}) if isinstance(candidate, dict) else {}
    required_metrics = {"micro_f1", "no_slot_failure_rate", "json_valid_rate",
                        "schema_valid_rate", "copy_valid_rate"}
    if (not isinstance(metrics, dict) or not required_metrics <= metrics.keys()
            or not isinstance(positive, dict) or "micro_f1" not in positive):
        raise ValueError("v6 dev report lacks the prespecified candidate metrics")
    decision = v6.development_gate(metrics, positive)
    if (dev_report.get("role") != "dev"
            or dev_report.get("manifest_sha256") != manifest_sha256
            or dev_report.get("train_file_sha256") != splits["train"]["jsonl_sha256"]
            or dev_report.get("eval_file_sha256") != splits["dev"]["jsonl_sha256"]
            or dev_report.get("model_sha256") != models
            or dev_report.get("code_sha256") != code_sha256
            or dev_report.get("package_versions") != versions
            or dev_report.get("confidence_threshold") != v6.MEAN_LOGPROB_THRESHOLD
            or any(dev_report.get("generation", {}).get(key) != value
                   for key, value in v6.GENERATION.items())
            or dev_report.get("bootstrap") != v6.BOOTSTRAP["dev"]
            or dev_report.get("development_gate") != decision
            or not decision["passed"]):
        raise ValueError("v6 development candidate did not pass its frozen gate")
    return {
        "status": "selected_for_one_test",
        "primary_candidate": "new_hybrid",
        "primary_reference": "old_v5_hybrid",
        "manifest_sha256": manifest_sha256,
        "train_file_sha256": splits["train"]["jsonl_sha256"],
        "dev_file_sha256": splits["dev"]["jsonl_sha256"],
        "test_file_sha256": splits["test"]["jsonl_sha256"],
        "old_bio_model_sha256": models["old_bio"],
        "new_bio_model_sha256": models["new_bio"],
        "v4_dpo_adapter_sha256": models["dpo_inference_files"]["adapter_model.safetensors"],
        "dpo_inference_files_sha256": models["dpo_inference_files"],
        "dev_report_sha256": dev_report_sha256,
        "confidence_threshold": v6.MEAN_LOGPROB_THRESHOLD,
        "generation": v6.GENERATION,
        "bootstrap": v6.BOOTSTRAP["test"],
        "package_versions": versions,
        "code_sha256": code_sha256,
    }


def verify_dev_prediction_files(dev_report: dict) -> None:
    """Bind the aggregate development decision to retained local outputs."""
    _report, paths, confidence = v6.output_paths("dev")
    expected = dev_report.get("prediction_file_sha256")
    if not isinstance(expected, dict) or set(expected) != {*paths, "confidence"}:
        raise ValueError("v6 dev report lacks complete local prediction fingerprints")
    for name, path in {**paths, "confidence": confidence}.items():
        if v6.sha256_file(path) != v6.require_hex(expected[name], f"dev {name} predictions"):
            raise ValueError(f"v6 dev {name} predictions changed after evaluation")


def create_lock() -> tuple[dict, str]:
    if v6.LOCK.exists():
        raise ValueError("v6 selection lock already exists; refusing to overwrite")
    if not v6.DEV_REPORT.is_file():
        raise ValueError("A completed v6 development report is required")
    # The only test-related paths inspected here are output paths. Never hash,
    # parse, or open data/massive-zh/slots-v6/test.jsonl in this helper.
    v6.check_output_paths("test")
    v6.check_confidence_assets()
    manifest_path = v6.DATA / "manifest.json"
    manifest_sha256 = v6.sha256_file(manifest_path)
    if manifest_sha256 != v6.EXPECTED_MANIFEST_SHA256:
        raise ValueError("v6 manifest differs from its preregistered SHA-256")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    v6.validate_manifest(manifest)
    v6.check_split_hashes(manifest, "dev")
    if manifest.get("test_status") != "sealed; no model inference or score":
        raise ValueError("v6 manifest does not identify a sealed test")
    dev_report_sha256 = v6.sha256_file(v6.DEV_REPORT)
    dev_report = json.loads(v6.DEV_REPORT.read_text(encoding="utf-8"))
    verify_dev_prediction_files(dev_report)
    models = v6.model_hashes()
    v6.load_bio(v6.OLD_BIO)
    v6.load_bio(
        v6.NEW_BIO,
        expected_train_sha256=manifest["splits"]["train"]["jsonl_sha256"],
        expected_manifest_sha256=manifest_sha256,
    )
    code_sha256 = {name: v6.sha256_file(v6.ROOT / name) for name in v6.CODE_FILES}
    versions = v6.package_versions()
    lock = build_lock(
        manifest=manifest, manifest_sha256=manifest_sha256,
        dev_report=dev_report, dev_report_sha256=dev_report_sha256,
        models=models, code_sha256=code_sha256, versions=versions,
    )
    v6.LOCK.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation remains the final race-safe guard against replacement.
    with v6.LOCK.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(lock, ensure_ascii=False, indent=2) + "\n")
    lock_sha256 = v6.sha256_file(v6.LOCK)
    if v6.require_test_lock(lock_sha256) != lock:
        raise ValueError("Written v6 selection lock failed evaluator verification")
    return lock, lock_sha256


def main() -> None:
    _lock, digest = create_lock()
    print(json.dumps({"status": "selected_for_one_test",
                      "selection_lock_sha256": digest,
                      "test_status": "sealed; no test rows opened"}))


if __name__ == "__main__":
    main()
