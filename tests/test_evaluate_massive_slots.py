"""Small synthetic checks for strict slot extraction metrics and split guards."""

from __future__ import annotations

import sys
import json
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate_massive_slots import (  # noqa: E402
    _validate_study_split,
    load_locked_split_manifest,
    paired_bootstrap_f1_delta,
    score_predictions,
    validate_frozen_model_files,
)
from massive_slots import canonical_output  # noqa: E402
from prepare_massive_slots import SOURCE_SHA256  # noqa: E402


def row(identifier: str, utterance: str, slots: list[dict], group: str | None = None) -> dict:
    return {
        "id": identifier,
        "utt": utterance,
        "slots": slots,
        "source_partition": "train",
        "group_sha256": group or identifier,
    }


class SlotMetricTests(unittest.TestCase):
    def test_multiset_exactness_and_invalid_outputs(self) -> None:
        rows = [
            row("1", "今天和今天", [
                {"type": "date", "value": "今天"},
                {"type": "date", "value": "今天"},
            ]),
            row("2", "北京找小明", [
                {"type": "place_name", "value": "北京"},
                {"type": "person", "value": "小明"},
            ]),
            row("3", "请放音乐", []),
            row("4", "明天提醒", [{"type": "date", "value": "明天"}]),
        ]
        predictions = [
            canonical_output([{"type": "date", "value": "今天"}]),
            canonical_output([
                {"type": "place_name", "value": "北京"},
                {"type": "person", "value": "小明"},
            ]),
            "garbage",
            canonical_output([{"type": "date", "value": "后天"}]),
        ]
        scored = score_predictions(rows, predictions, bootstrap_samples=30)
        self.assertEqual(scored["micro"]["tp"], 3)
        self.assertEqual(scored["micro"]["fp"], 0)
        self.assertEqual(scored["micro"]["fn"], 2)
        self.assertAlmostEqual(scored["micro_f1"], 0.75)
        self.assertEqual(scored["sentence_exact_count"], 1)
        self.assertEqual(scored["json_valid_rate"], 0.75)
        self.assertEqual(scored["schema_valid_rate"], 0.75)
        self.assertEqual(scored["copy_valid_rate"], 0.5)
        self.assertEqual(scored["no_slot_false_positive_rate"], 0.0)
        self.assertEqual(scored["no_slot_failure_rate"], 1.0)
        self.assertEqual(scored["per_type"]["date"]["fn"], 2)
        self.assertEqual(scored["bootstrap"]["clusters"], 4)
        self.assertEqual(scored["bootstrap"]["samples"], 30)

    def test_overprediction_on_empty_gold_and_group_bootstrap(self) -> None:
        rows = [
            row("1", "明天", [{"type": "date", "value": "明天"}], group="same"),
            row("2", "明天", [{"type": "date", "value": "明天"}], group="same"),
            row("3", "今天", [], group="other"),
        ]
        predictions = [
            canonical_output([{"type": "date", "value": "明天"}]),
            canonical_output([{"type": "date", "value": "明天"}]),
            canonical_output([{"type": "date", "value": "今天"}]),
        ]
        one = score_predictions(rows, predictions, bootstrap_samples=50, seed=7)
        two = score_predictions(rows, predictions, bootstrap_samples=50, seed=7)
        self.assertEqual(one, two)
        self.assertEqual(one["bootstrap"]["clusters"], 2)
        self.assertEqual(one["no_slot_false_positive_rate"], 1.0)
        self.assertEqual(one["sentence_exact_count"], 2)

    def test_rejects_duplicate_ids_and_invalid_gold(self) -> None:
        duplicated = [row("1", "今天", []), row("1", "明天", [])]
        with self.assertRaisesRegex(ValueError, "Duplicate row ID"):
            score_predictions(duplicated, ["{}", "{}"])
        with self.assertRaisesRegex(ValueError, "Invalid gold slot"):
            score_predictions(
                [row("3", "今天", [{"type": "date", "value": "明天"}])],
                ["{}"],
            )

    def test_paired_bootstrap_uses_same_groups(self) -> None:
        rows = [
            row("1", "明天", [{"type": "date", "value": "明天"}], group="g1"),
            row("2", "今天", [{"type": "date", "value": "今天"}], group="g2"),
            row("3", "北京", [{"type": "place_name", "value": "北京"}], group="g3"),
        ]
        baseline = [canonical_output([])] * 3
        candidate = [canonical_output(row["slots"]) for row in rows]
        comparison = paired_bootstrap_f1_delta(
            rows, baseline, candidate, samples=60, seed=11
        )
        self.assertEqual(comparison["baseline_micro_f1"], 0.0)
        self.assertEqual(comparison["candidate_micro_f1"], 1.0)
        self.assertEqual(comparison["delta_ci95"], [1.0, 1.0])
        self.assertEqual(comparison["sentence_exact_candidate_minus_baseline"], 1.0)
        self.assertEqual(comparison["sentence_exact_delta_ci95"], [1.0, 1.0])
        self.assertEqual(comparison["clusters"], 3)

    def test_rejects_official_test_and_overlapping_train_group(self) -> None:
        train = row("1", "今天", [], group="same")
        dev = row("2", "明天", [], group="same")
        manifest = {
            "splits": {
                "train": {"examples": 1, "jsonl_sha256": "trainhash", "source_ids": ["1"],
                          "groups": [{"group_sha256": "same"}]},
                "dev": {"examples": 1, "jsonl_sha256": "devhash", "source_ids": ["2"],
                        "groups": [{"group_sha256": "same"}]},
            }
        }
        with (
            patch("evaluate_massive_slots.Path.is_file", lambda path: path.name == "train.jsonl"),
            patch("evaluate_massive_slots.read_jsonl", return_value=[train]),
            patch("evaluate_massive_slots.sha256_file", side_effect=lambda path:
                  "trainhash" if path.name == "train.jsonl" else "devhash"),
        ):
            with self.assertRaisesRegex(ValueError, "shares a normalized phrase group"):
                _validate_study_split([dev], Path("study/dev.jsonl"), "dev", manifest)
            dev["source_partition"] = "test"
            with self.assertRaisesRegex(ValueError, "Only official MASSIVE train"):
                _validate_study_split([dev], Path("study/dev.jsonl"), "dev", manifest)

    def test_dev_requires_exact_manifest_sha_before_reading_rows(self) -> None:
        path = Path("study/dev.jsonl")
        manifest = {"format_version": 1, "source_partition_used": "train",
                    "source_sha256": SOURCE_SHA256,
                    "splits": {"dev": {"jsonl_sha256": "expected"}}}
        with patch("evaluate_massive_slots.Path.is_file", return_value=False):
            with self.assertRaisesRegex(ValueError, "manifest is required"):
                load_locked_split_manifest(path, "dev")
        with (
            patch("evaluate_massive_slots.Path.is_file", return_value=True),
            patch("evaluate_massive_slots.Path.read_text", return_value=json.dumps(manifest)),
            patch("evaluate_massive_slots.sha256_file", return_value="wrong"),
        ):
            with self.assertRaisesRegex(ValueError, "differs from the locked"):
                load_locked_split_manifest(path, "dev")
            with self.assertRaisesRegex(ValueError, "requires dev.jsonl"):
                load_locked_split_manifest(Path("study/confirmation.jsonl"), "dev")
        with (
            patch("evaluate_massive_slots.Path.is_file", return_value=True),
            patch("evaluate_massive_slots.Path.read_text", return_value=json.dumps(manifest)),
            patch("evaluate_massive_slots.sha256_file", return_value="expected"),
        ):
            self.assertEqual(load_locked_split_manifest(path, "dev"), manifest)

    def test_confirmation_pins_base_tokenizer_and_adapter_bytes(self) -> None:
        hashes = {
            "model.safetensors": "a" * 64,
            "tokenizer.json": "b" * 64,
            "adapter_model.safetensors": "c" * 64,
        }
        with (
            patch("evaluate_massive_slots.Path.is_file", return_value=True),
            patch("evaluate_massive_slots.sha256_file",
                  side_effect=lambda path: hashes[path.name]),
        ):
            self.assertEqual(
                validate_frozen_model_files("model", "adapter", "a" * 64,
                                            "b" * 64, "c" * 64),
                hashes,
            )
            with self.assertRaisesRegex(ValueError, "adapter_model.safetensors SHA-256 mismatch"):
                validate_frozen_model_files("model", "adapter", "a" * 64,
                                            "b" * 64, "d" * 64)
            with self.assertRaisesRegex(ValueError, "requires --expected-adapter-sha256"):
                validate_frozen_model_files("model", "adapter", "a" * 64,
                                            "b" * 64, None)


if __name__ == "__main__":
    unittest.main()
