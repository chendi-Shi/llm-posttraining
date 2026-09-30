"""Check cross-report arithmetic and provenance using only public JSON files.

This is not a rerun of inference or bootstrap: raw data, weights, and per-row
predictions are intentionally absent from the public repository.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


class EvidenceError(ValueError):
    pass


class Audit:
    def __init__(self) -> None:
        self.checks = 0

    def require(self, condition: bool, message: str) -> None:
        self.checks += 1
        if not condition:
            raise EvidenceError(message)

    def close(self, actual: float, expected: float, message: str) -> None:
        self.require(math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12), message)


def read(root: Path, name: str) -> dict:
    return json.loads((root / "reports" / name).read_text(encoding="utf-8"))


def f1(tp: int, fp: int, fn: int) -> float:
    return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0


def check_metrics(a: Audit, label: str, m: dict) -> None:
    n = m["examples"]
    tp, fp, fn = (m["micro"][key] for key in ("tp", "fp", "fn"))
    a.require(n > 0 and min(tp, fp, fn) >= 0, f"{label}: invalid counts")
    a.require(tp + fn == m["gold_slot_count"], f"{label}: gold count")
    a.require(tp + fp == m["predicted_slot_count"], f"{label}: predicted count")
    a.close(m["micro_f1"], f1(tp, fp, fn), f"{label}: micro F1")
    a.close(m["micro"]["f1"], m["micro_f1"], f"{label}: duplicate F1")
    a.close(m["micro"]["precision"], tp / (tp + fp) if tp + fp else 0.0,
            f"{label}: precision")
    a.close(m["micro"]["recall"], tp / (tp + fn) if tp + fn else 0.0,
            f"{label}: recall")
    if "per_type" in m:
        for key in ("tp", "fp", "fn"):
            a.require(sum(row[key] for row in m["per_type"].values()) == m["micro"][key],
                      f"{label}: per-type {key}")
    a.close(m["sentence_exact_accuracy"], m["sentence_exact_count"] / n,
            f"{label}: sentence exact")
    no_slot = m["no_slot_examples"]
    a.require(0 < no_slot <= n, f"{label}: no-slot denominator")
    false_positive = m["no_slot_false_positive_count"]
    invalid = m["no_slot_invalid_output_count"]
    a.require(0 <= false_positive + invalid <= no_slot, f"{label}: no-slot counts")
    a.close(m["no_slot_failure_rate"], (false_positive + invalid) / no_slot,
            f"{label}: no-slot failure")
    a.close(m["no_slot_false_positive_rate"], false_positive / no_slot,
            f"{label}: no-slot false-positive rate")
    a.require(sum(m["error_counts"].values()) == m["invalid_output_count"],
              f"{label}: invalid output count")
    for key in ("json_valid_rate", "schema_valid_rate", "copy_valid_rate"):
        a.require(0 <= m[key] <= 1, f"{label}: {key} out of range")


def check_pair(a: Audit, label: str, pair: dict, baseline: dict, candidate: dict) -> None:
    a.close(pair["baseline_micro_f1"], baseline["micro_f1"], f"{label}: baseline F1")
    a.close(pair["candidate_micro_f1"], candidate["micro_f1"], f"{label}: candidate F1")
    a.close(pair["candidate_minus_baseline"],
            candidate["micro_f1"] - baseline["micro_f1"], f"{label}: F1 delta")
    a.close(pair["baseline_sentence_exact_accuracy"], baseline["sentence_exact_accuracy"],
            f"{label}: baseline exact")
    a.close(pair["candidate_sentence_exact_accuracy"], candidate["sentence_exact_accuracy"],
            f"{label}: candidate exact")
    a.close(pair["sentence_exact_candidate_minus_baseline"],
            candidate["sentence_exact_accuracy"] - baseline["sentence_exact_accuracy"],
            f"{label}: sentence exact delta")
    a.require(pair["clusters"] == baseline["examples"] == candidate["examples"],
              f"{label}: cluster count")
    a.require(pair["samples"] == pair["valid_bootstrap_draws"] > 0,
              f"{label}: bootstrap draw count")
    for key in ("delta_ci95", "sentence_exact_delta_ci95"):
        lo, hi = pair[key]
        a.require(-1 <= lo <= hi <= 1, f"{label}: {key} bounds")


def check_no_slot_pair(a: Audit, label: str, pair: dict,
                       reference: dict, candidate: dict) -> None:
    a.require(pair["no_slot_examples"] == reference["no_slot_examples"]
              == candidate["no_slot_examples"], f"{label}: no-slot denominator")
    a.close(pair["observed_reduction"],
            reference["no_slot_failure_rate"] - candidate["no_slot_failure_rate"],
            f"{label}: no-slot delta")
    lo, hi = pair["reduction_ci95"]
    a.require(-1 <= lo <= hi <= 1 and pair["samples"] > 0,
              f"{label}: interval bounds")


def audit(root: Path, reports: dict[str, dict] | None = None) -> dict:
    a = Audit()
    load = (lambda name: read(root, name)) if reports is None else reports.__getitem__
    lock = load("massive-slots-selection-lock.json")
    v1 = load("massive-slots-confirmation-comparison.json")
    v2 = load("massive-slots-v2-sft-step160-dev.json")
    dpo = load("massive-slots-v3-dpo-dev.json")
    control = load("massive-slots-v3-matched-sft-dev.json")

    a.require(v1["role"] == "confirmation" and v1["examples"] == 200,
              "v1: confirmation role or size")
    a.require(v1["eval_file_sha256"] == lock["confirmation_file_sha256"],
              "v1: confirmation split differs from selection lock")
    a.require(v1["locked_manifest_sha256"] == lock["locked_data_manifest_sha256"],
              "v1: locked data manifest differs")
    for label, row in v1["systems"].items():
        check_metrics(a, f"v1/{label}", row["metrics"])
        a.require(row["metrics"]["examples"] == v1["examples"], f"v1/{label}: size")
    for label, pair in v1["paired_comparisons"].items():
        check_pair(a, f"v1/{label}", pair,
                   v1["systems"][pair["reference"]]["metrics"],
                   v1["systems"][pair["candidate"]]["metrics"])

    metrics = {"v2": v2, "dpo": dpo["metrics"], "control": control["metrics"]}
    for label, row in metrics.items():
        check_metrics(a, label, row)
        a.require(row["examples"] == 250, f"{label}: dev size")
    a.require(v2["role"] == dpo["role"] == control["role"] == "dev",
              "v2/v3: split role")
    a.require(control["confirmation_used"] is False, "control: confirmation was used")
    a.require(v2["eval_file_sha256"] == dpo["dev_file_sha256"]
              == control["dev_file_sha256"], "v2/v3: development split differs")
    a.require(dpo["dev_manifest_sha256"] == control["dev_manifest_sha256"],
              "v3: development manifest differs")
    a.require(v2["source_sha256"] == dpo["source_sha256"]
              == control["source_sha256"], "v2/v3: source archive differs")
    a.require(v2["generation"] == dpo["generation"] == control["generation"],
              "v2/v3: generation protocol differs")
    for key in ("model.safetensors", "tokenizer.json"):
        a.require(v2["frozen_model_files_sha256"][key]
                  == dpo["inference_files_sha256"][key]
                  == control["inference_files_sha256"][key], f"v2/v3: {key} differs")
    a.require(dpo["frozen_reference_adapter_sha256"]
              == v2["frozen_model_files_sha256"]["adapter_model.safetensors"],
              "v3: DPO reference differs from v2 SFT")
    a.require(dpo["baseline_prediction_file_sha256"] == v2["prediction_file_sha256"],
              "v3: DPO baseline predictions differ from v2 SFT")
    a.require(control["reference_prediction_file_sha256"] == {
        "v2_sft": v2["prediction_file_sha256"],
        "v3_dpo": dpo["prediction_file_sha256"],
    }, "control: reference predictions differ")

    for label, report in (("dpo", dpo), ("control", control)):
        positive = report["positive_only"]
        a.require(positive["examples"] == report["metrics"]["examples"]
                  - report["metrics"]["no_slot_examples"],
                  f"{label}: positive denominator")
        a.require(0 <= positive["micro_f1"] <= 1, f"{label}: positive F1 range")
    check_pair(a, "dpo-v2", dpo["v3_minus_v2_sft_paired_f1"], v2, dpo["metrics"])
    check_no_slot_pair(a, "dpo-v2", dpo["v2_sft_minus_v3_no_slot_failure"],
                       v2, dpo["metrics"])
    pairs = control["exploratory_paired_comparison"]
    check_pair(a, "control-v2", pairs["control_minus_v2_sft_f1"], v2,
               control["metrics"])
    check_no_slot_pair(a, "control-v2", pairs["v2_sft_minus_control_no_slot_failure"],
                       v2, control["metrics"])
    check_pair(a, "dpo-control", pairs["dpo_minus_control_f1"],
               control["metrics"], dpo["metrics"])
    check_no_slot_pair(a, "dpo-control", pairs["control_minus_dpo_no_slot_failure"],
                       control["metrics"], dpo["metrics"])

    checks = dpo["development_gate"]["checks"]
    expected = {
        "json_valid_at_least_95pct": dpo["metrics"]["json_valid_rate"] >= .95,
        "schema_valid_at_least_95pct": dpo["metrics"]["schema_valid_rate"] >= .95,
        "copy_valid_at_least_95pct": dpo["metrics"]["copy_valid_rate"] >= .95,
        "no_slot_failure_at_most_30pct": dpo["metrics"]["no_slot_failure_rate"] <= .30,
        "strict_entity_f1_at_least_55pct": dpo["metrics"]["micro_f1"] >= .55,
        "positive_subset_f1_at_least_60pct": dpo["positive_only"]["micro_f1"] >= .60,
        "paired_f1_ci95_lower_positive": dpo["v3_minus_v2_sft_paired_f1"]["delta_ci95"][0] > 0,
        "paired_no_slot_reduction_ci95_lower_positive":
            dpo["v2_sft_minus_v3_no_slot_failure"]["reduction_ci95"][0] > 0,
    }
    a.require(checks == expected, "v3: development gate does not match reported metrics")
    a.require(dpo["development_gate"]["passed"] == all(expected.values()),
              "v3: development gate decision")
    a.require(dpo["development_gate"]["passed"] is False,
              "v3: failed model was marked as selected")
    return {"checks": a.checks, "v1_confirmation_sft_f1": v1["systems"]["sft96"]["metrics"]["micro_f1"],
            "v3_dpo_dev_f1": dpo["metrics"]["micro_f1"],
            "v3_control_dev_f1": control["metrics"]["micro_f1"],
            "v3_confirmation_used": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    try:
        result = audit(args.root)
    except (EvidenceError, KeyError, OSError, TypeError, ValueError) as exc:
        parser.exit(1, f"Public evidence audit failed: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
