"""Check paired no-slot statistics and positive-only reporting."""

from __future__ import annotations

import sys
import json
import shutil
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from compare_massive_slots_v2 import (  # noqa: E402
    CONFIRMATION_CODE_FILES, decide_v2_improvement, main,
    paired_no_slot_reduction, positive_only_metrics,
    validate_confirmation_reports,
)
from evaluate_massive_slots import sha256_file  # noqa: E402


@contextmanager
def local_fixture_dir():
    root = Path(__file__).resolve().parents[1] / "_tmp"
    root.mkdir(exist_ok=True)
    work = root / f"compare-v2-{uuid.uuid4().hex}"
    work.mkdir()
    try:
        yield work
    finally:
        if work.resolve().parent != root.resolve():
            raise RuntimeError("Refusing to remove a fixture outside the workspace _tmp")
        shutil.rmtree(work)


class V2ComparisonTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = [
            {"id": "1", "utt": "你好", "slots": [], "group_sha256": "a"},
            {"id": "2", "utt": "谢谢", "slots": [], "group_sha256": "b"},
            {"id": "3", "utt": "找王明", "slots": [{"type": "person", "value": "王明"}], "group_sha256": "c"},
        ]

    def test_no_slot_reduction_counts_invalid_as_failure(self) -> None:
        reference = ["not JSON", '{"slots":[{"type":"person","value":"谢谢"}]}',
                     '{"slots":[{"type":"person","value":"王明"}]}']
        candidate = ['{"slots":[]}', '{"slots":[]}',
                     '{"slots":[{"type":"person","value":"王明"}]}']
        result = paired_no_slot_reduction(self.rows, reference, candidate, samples=100, seed=1)
        self.assertEqual(result["observed_reduction"], 1.0)
        self.assertEqual(result["reduction_ci95"], [1.0, 1.0])
        self.assertEqual(positive_only_metrics(self.rows, candidate)["micro_f1"], 1.0)

    def test_confirmation_requires_explicit_unlock_before_file_access(self) -> None:
        with patch.object(Path, "read_text", side_effect=AssertionError("file opened")), \
             patch.object(Path, "open", side_effect=AssertionError("file opened")):
            with self.assertRaises(SystemExit) as caught:
                main(["--role", "confirmation", "--eval-file", "confirmation.jsonl",
                      "--system", "a=a.jsonl", "--system", "b=b.jsonl", "--output", "out.json"])
            self.assertEqual(caught.exception.code, 2)

    def test_confirmation_refuses_alternate_selection_lock_before_heldout_read(self) -> None:
        with patch.object(Path, "read_text", side_effect=AssertionError("held-out read")), \
             patch("compare_massive_slots_v2.sha256_file", side_effect=AssertionError("held-out hash")):
            with self.assertRaises(SystemExit) as caught:
                main(["--role", "confirmation", "--unlock-confirmation",
                      "--selection-lock", "reports/alternate-v2-selection-lock.json",
                      "--expected-lock-sha256", "a" * 64,
                      "--eval-file", "data/massive-zh/slots-v2/confirmation.jsonl",
                      "--system", "bio=_tmp/bio.jsonl", "--system", "v1=_tmp/v1.jsonl",
                      "--output", "reports/massive-slots-v2-confirmation-comparison.json"])
            self.assertEqual(caught.exception.code, 2)

    def test_changed_scoring_dependency_rejected_before_confirmation_rows_are_read(self) -> None:
        lock = {
            "status": "selected", "selected_step": 80,
            "selected_adapter_sha256": "b" * 64,
            "code_sha256": {relative: "a" * 64 for relative in CONFIRMATION_CODE_FILES},
        }
        lock["code_sha256"]["scripts/evaluate_massive_slots.py"] = "c" * 64
        with patch.object(Path, "exists", return_value=False), \
             patch.object(Path, "read_text", return_value=json.dumps(lock)), \
             patch("compare_massive_slots_v2.sha256_file", return_value="a" * 64), \
             patch("compare_massive_slots_v2.read_jsonl", side_effect=AssertionError("held-out read")), \
             patch("compare_massive_slots_v2.load_locked_split_manifest", side_effect=AssertionError("held-out preflight")):
            with self.assertRaisesRegex(ValueError, "evaluate_massive_slots.py"):
                main(["--role", "confirmation", "--unlock-confirmation",
                      "--expected-lock-sha256", "a" * 64,
                      "--eval-file", "data/massive-zh/slots-v2/confirmation.jsonl",
                      "--system", "bio=_tmp/bio.jsonl", "--system", "v1=_tmp/v1.jsonl",
                      "--output", "reports/massive-slots-v2-confirmation-comparison.json"])

    def test_preregistered_decision_requires_both_ci_bounds_and_absolute_gate(self) -> None:
        pairs = {"v2_minus_v1": {
            "f1": {"delta_ci95": [0.01, 0.1]},
            "no_slot_failure_reduction": {"reduction_ci95": [0.02, 0.2]},
        }}
        reports = {"v2": {"no_slot_failure_rate": 0.3}}
        self.assertTrue(decide_v2_improvement(pairs, reports)["passed"])
        pairs["v2_minus_v1"]["f1"]["delta_ci95"][0] = 0.0
        self.assertFalse(decide_v2_improvement(pairs, reports)["passed"])
        pairs["v2_minus_v1"]["f1"]["delta_ci95"][0] = 0.01
        pairs["v2_minus_v1"]["no_slot_failure_reduction"]["reduction_ci95"][0] = 0.0
        self.assertFalse(decide_v2_improvement(pairs, reports)["passed"])
        pairs["v2_minus_v1"]["no_slot_failure_reduction"]["reduction_ci95"][0] = 0.02
        reports["v2"]["no_slot_failure_rate"] = 0.31
        self.assertFalse(decide_v2_improvement(pairs, reports)["passed"])

    def test_confirmation_reports_bind_models_and_prediction_bytes(self) -> None:
        with local_fixture_dir() as work:
            eval_file = work / "confirmation.jsonl"
            manifest = work / "manifest.json"
            eval_file.write_text("opaque held-out bytes\n", encoding="utf-8")
            manifest.write_text("opaque manifest bytes\n", encoding="utf-8")
            systems = []
            metric_files = []
            lock = {
                "selected_step": 80,
                "selected_adapter_sha256": "a" * 64,
                "selected_sft_checkpoint": str(work / "v2-adapter"),
                "base_model_sha256": "b" * 64,
                "tokenizer_sha256": "c" * 64,
                "base_model_files_sha256": {
                    "model.safetensors": "b" * 64,
                    "tokenizer.json": "c" * 64,
                    "base/config.json": "2" * 64,
                },
                "selected_adapter_files_sha256": {
                    "adapter_model.safetensors": "a" * 64,
                    "adapter/adapter_config.json": "3" * 64,
                },
                "data_sha256": {"source_archive_sha256": "d" * 64},
                "generation": {"batch_size": 8, "max_new_tokens": 64},
                "candidate_dev_results": {"80": {"eligible": True}},
                "reference_systems": {
                    "comparison_order": ["bio", "v1", "v2"],
                    "v1": {"adapter_sha256": "e" * 64,
                           "adapter_path": str(work / "v1-adapter"),
                           "adapter_files_sha256": {
                               "adapter_model.safetensors": "e" * 64,
                               "adapter/adapter_config.json": "4" * 64,
                           }},
                    "bio": {"model_sha256": "f" * 64,
                            "train_file_sha256": "1" * 64},
                },
            }
            for name in ("bio", "v1", "v2"):
                prediction = work / f"{name}.jsonl"
                prediction.write_text(f"{name} predictions\n", encoding="utf-8")
                report_path = work / f"{name}.json"
                common = {
                    "role": "confirmation",
                    "prediction_file_sha256": sha256_file(prediction),
                    "eval_file_sha256": sha256_file(eval_file),
                    "micro_f1": 0.5,
                    "no_slot_failure_rate": 0.2,
                    "copy_valid_rate": 1.0,
                }
                if name == "bio":
                    report = {**common,
                              "selection_lock_sha256": "9" * 64,
                              "manifest_sha256": sha256_file(manifest),
                              "train_file_sha256": "1" * 64,
                              "model_sha256": "f" * 64,
                              "metrics": {key: common[key] for key in
                                          ("micro_f1", "no_slot_failure_rate", "copy_valid_rate")}}
                else:
                    report = {**common,
                              "system": "v1" if name == "v1" else "v2_step80",
                              "adapter": str(work / ("v1-adapter" if name == "v1" else "v2-adapter")),
                              "selection_lock_sha256": "9" * 64,
                              "source_sha256": "d" * 64,
                              "study_manifest_sha256": sha256_file(manifest),
                              "frozen_model_files_sha256": {
                                  **lock["base_model_files_sha256"],
                                  **(lock["reference_systems"]["v1"]["adapter_files_sha256"]
                                     if name == "v1" else lock["selected_adapter_files_sha256"]),
                              },
                              "generation": dict(lock["generation"])}
                report_path.write_text(json.dumps(report), encoding="utf-8")
                systems.append((name, prediction))
                metric_files.append((name, report_path))
            hashes, scored = validate_confirmation_reports(
                lock, eval_file, systems, metric_files, "9" * 64,
            )
            self.assertEqual(set(hashes), {"bio", "v1", "v2"})
            self.assertEqual(scored["v2"]["micro_f1"], 0.5)
            v2_report = metric_files[-1][1]
            original = json.loads(v2_report.read_text(encoding="utf-8"))
            altered = json.loads(v2_report.read_text(encoding="utf-8"))
            altered["frozen_model_files_sha256"]["adapter/adapter_config.json"] = "5" * 64
            v2_report.write_text(json.dumps(altered), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "model/generation"):
                validate_confirmation_reports(lock, eval_file, systems, metric_files, "9" * 64)
            v2_report.write_text(json.dumps(original), encoding="utf-8")
            systems[-1][1].write_text("changed prediction bytes\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not bind its predictions"):
                validate_confirmation_reports(lock, eval_file, systems, metric_files, "9" * 64)


if __name__ == "__main__":
    unittest.main()
