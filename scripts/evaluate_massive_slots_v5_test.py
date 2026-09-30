"""Single-use independent test of frozen BIO plus confidence-gated v4 DPO."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import joblib

from compare_massive_slots_v2 import positive_only_metrics
from evaluate_massive_slots import (
    generate_raw_predictions, paired_bootstrap_f1_delta, read_jsonl,
    score_predictions, sha256_file,
)
from evaluate_massive_slots_bio import MODEL_CONFIG, predict_slots
from evaluate_massive_slots_v2 import prediction_payload, write_predictions
from massive_slots import canonical_output, parse_prediction
from slot_confidence import (
    MEAN_LOGPROB_THRESHOLD, choose_hybrid_output, completion_mean_logprob,
)
from v2_model_assets import hash_inference_files


ROOT = Path(__file__).resolve().parents[1]
DATA = Path("data/massive-zh/slots-v5")
BASE = Path("models/Qwen2.5-0.5B-Instruct")
ADAPTER = Path("outputs/massive-slots-v4-balanced-dpo-64")
BIO = Path("outputs/massive-slots-v2-bio.joblib")
LOCK = Path("reports/massive-slots-v5-selection-lock.json")
REPORT = Path("reports/massive-slots-v5-test.json")
BIO_PRED = Path("_tmp/massive-slots-v5-bio-test-predictions.jsonl")
DPO_PRED = Path("_tmp/massive-slots-v5-v4-dpo-test-predictions.jsonl")
HYBRID_PRED = Path("_tmp/massive-slots-v5-hybrid-test-predictions.jsonl")
SCORES = Path("_tmp/massive-slots-v5-confidence-test.jsonl")
TEST_MANIFEST_SHA256 = "e4531ab27746a0cfe21dc0219331f5cc8237e614bb4c7090b355000fdce1d453"
TEST_FILE_SHA256 = "afd5655d9d7e7092ae3c7798179578fc93c3e5d4a88939e9fbfb1d5978459b3c"
BIO_SHA256 = "0d7a7fad835eea383223a4f9c173e42488a02416addd660ef8171141d7c361fa"
BASE_SHA256 = "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe"
TOKENIZER_SHA256 = "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"
ADAPTER_SHA256 = "ed76990eef5bf7b89e2ce01ba516bbb5455de4c91e1242dc74c164d6603f8a6f"
GENERATION = {"batch_size": 8, "max_input_tokens": 256, "max_new_tokens": 64}
CODE_FILES = (
    "scripts/prepare_massive_slots_v5_test.py",
    "scripts/evaluate_massive_slots_v5_test.py",
    "scripts/slot_confidence.py",
    "scripts/evaluate_massive_slots.py",
    "scripts/evaluate_massive_slots_bio.py",
    "scripts/evaluate_massive_slots_v2.py",
    "scripts/compare_massive_slots_v2.py",
    "scripts/massive_slots.py",
    "scripts/common.py",
)


def require_lock(expected_sha256: str) -> tuple[dict, dict[str, str]]:
    if not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256):
        raise ValueError("A 64-character selection lock SHA-256 is required")
    if sha256_file(LOCK) != expected_sha256.lower():
        raise ValueError("Selection lock changed")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if (lock.get("status") != "selected_for_one_test"
            or lock.get("test_manifest_sha256") != TEST_MANIFEST_SHA256
            or lock.get("test_file_sha256") != TEST_FILE_SHA256
            or lock.get("bio_model_sha256") != BIO_SHA256
            or lock.get("dpo_adapter_sha256") != ADAPTER_SHA256
            or lock.get("confidence_threshold") != MEAN_LOGPROB_THRESHOLD
            or lock.get("generation") != {"decoding": "greedy_unconstrained", **GENERATION}
            or lock.get("bootstrap") != {"samples": 2000, "seed": 20261002}):
        raise ValueError("Selection lock does not match the v5 plan")
    if lock.get("code_sha256") != {
            name: sha256_file(ROOT / name) for name in CODE_FILES}:
        raise ValueError("Scoring or inference code changed after selection")
    if sha256_file(DATA / "manifest.json") != TEST_MANIFEST_SHA256:
        raise ValueError("v5 test manifest changed")
    if sha256_file(DATA / "test.jsonl") != TEST_FILE_SHA256:
        raise ValueError("v5 test data changed")
    if sha256_file(BIO) != BIO_SHA256:
        raise ValueError("BIO model changed")
    model_hashes = hash_inference_files(BASE, ADAPTER)
    if (model_hashes.get("model.safetensors") != BASE_SHA256
            or model_hashes.get("tokenizer.json") != TOKENIZER_SHA256
            or model_hashes.get("adapter_model.safetensors") != ADAPTER_SHA256
            or lock.get("dpo_inference_files_sha256") != model_hashes):
        raise ValueError("DPO model differs from the selected files")
    return lock, model_hashes


def success_gate(metrics: dict, positive: dict, paired: dict) -> dict:
    checks = {
        "strict_entity_f1_at_least_60pct": metrics["micro_f1"] >= .60,
        "positive_subset_f1_at_least_60pct": positive["micro_f1"] >= .60,
        "no_slot_failure_at_most_20pct": metrics["no_slot_failure_rate"] <= .20,
        "json_valid_at_least_95pct": metrics["json_valid_rate"] >= .95,
        "schema_valid_at_least_95pct": metrics["schema_valid_rate"] >= .95,
        "copy_valid_at_least_95pct": metrics["copy_valid_rate"] >= .95,
        "paired_f1_ci95_lower_positive": paired["delta_ci95"][0] > 0,
    }
    return {"checks": checks, "passed": all(checks.values())}


def generate_confidence(rows: list[dict], dpo_raw: list[str]) -> tuple[list[float | None], dict]:
    from peft import PeftModel
    from common import configure_cpu, load_quantized_base, load_tokenizer

    configure_cpu()
    tokenizer = load_tokenizer(str(BASE))
    model = PeftModel.from_pretrained(
        load_quantized_base(str(BASE), training=False), str(ADAPTER), is_trainable=False)
    model.eval()
    scores: list[float | None] = []
    eligible = 0
    for index, (row, raw) in enumerate(zip(rows, dpo_raw, strict=True), 1):
        parsed, validity = parse_prediction(raw, row["utt"])
        if parsed and all(validity[key] for key in ("json_valid", "schema_valid", "copy_valid")):
            score = completion_mean_logprob(model, tokenizer, row["utt"], raw)
            eligible += 1
        else:
            score = None
        scores.append(score)
        if index % 32 == 0:
            print(f"Confidence scored {index}/{len(rows)}", flush=True)
    return scores, {"eligible_nonempty": eligible}


def write_scores(rows: list[dict], values: list[float | None]) -> str:
    if not SCORES.resolve().is_relative_to((ROOT / "_tmp").resolve()):
        raise ValueError("Confidence scores must remain in ignored local storage")
    content = "".join(json.dumps({"id": str(row["id"]),
                                   "group_sha256": row["group_sha256"],
                                   "mean_logprob": value},
                                  ensure_ascii=False, separators=(",", ":")) + "\n"
                      for row, value in zip(rows, values, strict=True)).encode("utf-8")
    SCORES.parent.mkdir(parents=True, exist_ok=True)
    with SCORES.open("xb") as stream:
        stream.write(content)
    return hashlib.sha256(content).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unlock-test", action="store_true")
    parser.add_argument("--expected-selection-lock-sha256")
    args = parser.parse_args()
    if not args.unlock_test or not args.expected_selection_lock_sha256:
        parser.error("Independent test requires --unlock-test and the published selection lock SHA-256")
    if any(path.exists() for path in (REPORT, BIO_PRED, DPO_PRED, HYBRID_PRED, SCORES)):
        raise ValueError("v5 independent test outputs already exist; refusing to overwrite")
    lock, model_hashes = require_lock(args.expected_selection_lock_sha256)
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("test_examples") != 600
            or manifest.get("test_jsonl_sha256") != TEST_FILE_SHA256
            or len(manifest.get("test_group_sha256", [])) != 600):
        raise ValueError("Unexpected test manifest content")
    rows = read_jsonl(DATA / "test.jsonl")
    if (len(rows) != 600
            or [row["group_sha256"] for row in rows] != manifest["test_group_sha256"]
            or len({str(row["id"]) for row in rows}) != 600
            or any(row.get("source_partition") != "train" for row in rows)):
        raise ValueError("Test rows differ from the frozen manifest")
    bundle = joblib.load(BIO)
    if bundle.get("configuration") != MODEL_CONFIG:
        raise ValueError("Frozen BIO configuration differs")
    started = time.perf_counter()
    bio_raw = [canonical_output(predict_slots(row["utt"],
                                              bundle["vectorizer"], bundle["model"]))
               for row in rows]
    dpo_raw, inference = generate_raw_predictions(
        rows, model_name=str(BASE), adapter=str(ADAPTER), **GENERATION)
    confidence, confidence_audit = generate_confidence(rows, dpo_raw)
    if hash_inference_files(BASE, ADAPTER) != model_hashes or sha256_file(BIO) != BIO_SHA256:
        raise ValueError("Models changed during independent test")
    chosen = [choose_hybrid_output(bio, dpo, row["utt"], score)
              for row, bio, dpo, score in zip(rows, bio_raw, dpo_raw, confidence, strict=True)]
    hybrid_raw = [item[0] for item in chosen]
    replaced = sum(item[1] for item in chosen)
    bio_metrics = score_predictions(rows, bio_raw, bootstrap_samples=1000, seed=20261002)
    dpo_metrics = score_predictions(rows, dpo_raw, bootstrap_samples=1000, seed=20261002)
    hybrid_metrics = score_predictions(rows, hybrid_raw, bootstrap_samples=1000, seed=20261002)
    bio_positive = positive_only_metrics(rows, bio_raw)
    dpo_positive = positive_only_metrics(rows, dpo_raw)
    hybrid_positive = positive_only_metrics(rows, hybrid_raw)
    paired = paired_bootstrap_f1_delta(rows, bio_raw, hybrid_raw,
                                       samples=2000, seed=20261002)
    decision = success_gate(hybrid_metrics, hybrid_positive, paired)
    bio_prediction_hash = write_predictions(
        BIO_PRED, prediction_payload(rows, bio_raw, TEST_FILE_SHA256), exclusive=True)
    dpo_prediction_hash = write_predictions(
        DPO_PRED, prediction_payload(rows, dpo_raw, TEST_FILE_SHA256), exclusive=True)
    hybrid_prediction_hash = write_predictions(
        HYBRID_PRED, prediction_payload(rows, hybrid_raw, TEST_FILE_SHA256), exclusive=True)
    score_hash = write_scores(rows, confidence)
    report = {
        "study": "MASSIVE zh-CN four-slot v5 independent test of BIO plus confidence-gated DPO",
        "role": "independent_test",
        "selection_lock_sha256": sha256_file(LOCK),
        "test_manifest_sha256": TEST_MANIFEST_SHA256,
        "test_file_sha256": TEST_FILE_SHA256,
        "bio_model_sha256": BIO_SHA256,
        "dpo_inference_files_sha256": model_hashes,
        "generation": {"decoding": "greedy_unconstrained", **GENERATION, **inference},
        "confidence_threshold": MEAN_LOGPROB_THRESHOLD,
        "confidence_scored_nonempty": confidence_audit["eligible_nonempty"],
        "hybrid_replaced_examples": replaced,
        "prediction_file_sha256": {"bio": bio_prediction_hash,
                                   "v4_dpo": dpo_prediction_hash,
                                   "hybrid": hybrid_prediction_hash,
                                   "confidence": score_hash},
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "bio": {"metrics": bio_metrics, "positive_only": bio_positive},
        "v4_dpo": {"metrics": dpo_metrics, "positive_only": dpo_positive},
        "hybrid": {"metrics": hybrid_metrics, "positive_only": hybrid_positive},
        "hybrid_minus_bio_paired_f1": paired,
        "independent_success_gate": decision,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with REPORT.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"bio_f1": bio_metrics["micro_f1"],
                      "v4_dpo_f1": dpo_metrics["micro_f1"],
                      "hybrid_f1": hybrid_metrics["micro_f1"],
                      "hybrid_minus_bio_ci95": paired["delta_ci95"],
                      "success_gate_passed": decision["passed"]}))


if __name__ == "__main__":
    main()
