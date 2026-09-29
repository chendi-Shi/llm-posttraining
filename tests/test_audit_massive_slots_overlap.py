"""Focused tests for the prior-partition overlap and score sensitivity audit."""

from __future__ import annotations

import io
import sys
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from audit_massive_slots_overlap import (  # noqa: E402
    main,
    overlap_flags,
    overlap_summary,
    prior_group_sets,
    sensitivity_metrics,
)
from massive_slots import canonical_output  # noqa: E402


class PriorPartitionOverlapTests(unittest.TestCase):
    def test_normalized_overlap_counts_union_without_double_counting(self) -> None:
        source = [
            {"partition": "dev", "utt": "Ａ B 早安"},
            {"partition": "test", "utt": "ab早安"},
            {"partition": "test", "utt": "明 天 提 醒"},
            {"partition": "train", "utt": "只在训练集中"},
        ]
        selected = [
            {"utt": "ab 早安"},
            {"utt": "明天提醒"},
            {"utt": "只在训练集中"},
        ]
        flags = overlap_flags(selected, prior_group_sets(source))
        self.assertEqual(flags, [
            {"dev": True, "test": True},
            {"dev": False, "test": True},
            {"dev": False, "test": False},
        ])
        summary = overlap_summary(flags)
        self.assertEqual(summary["overlap_with_prior_dev"], 1)
        self.assertEqual(summary["overlap_with_prior_test"], 2)
        self.assertEqual(summary["overlap_with_both"], 1)
        self.assertEqual(summary["overlap_with_either"], 2)
        self.assertEqual(summary["prior_partition_disjoint"], 1)
        self.assertAlmostEqual(summary["prior_partition_disjoint_fraction"], 1 / 3)
        self.assertNotIn("utt", summary)
        self.assertNotIn("id", summary)

    def test_sensitivity_uses_only_rows_disjoint_from_old_dev_and_test(self) -> None:
        rows = [
            {"id": "1", "utt": "明天提醒", "slots": [{"type": "date", "value": "明天"}]},
            {"id": "2", "utt": "今天提醒", "slots": [{"type": "date", "value": "今天"}]},
        ]
        predictions = [
            canonical_output([]),
            canonical_output([{"type": "date", "value": "今天"}]),
        ]
        flags = [{"dev": False, "test": True}, {"dev": False, "test": False}]
        result = sensitivity_metrics(rows, predictions, flags)
        self.assertEqual(result["full_examples"], 2)
        self.assertEqual(result["prior_partition_disjoint_examples"], 1)
        self.assertAlmostEqual(result["full_micro_f1"], 2 / 3)
        self.assertEqual(result["prior_partition_disjoint_micro_f1"], 1.0)
        self.assertAlmostEqual(result["micro_f1_delta_disjoint_minus_full"], 1 / 3)
        with self.assertRaisesRegex(ValueError, "must align"):
            sensitivity_metrics(rows, predictions[:1], flags)

    def test_confirmation_prediction_gate_fails_before_file_reads(self) -> None:
        with (
            patch.object(sys, "argv", [
                "audit_massive_slots_overlap.py",
                "--confirmation-system", "qwen=confirmation-predictions.jsonl",
            ]),
            patch.object(Path, "read_text", side_effect=AssertionError("Read a file before unlock")),
            redirect_stderr(io.StringIO()),
        ):
            with self.assertRaises(SystemExit) as context:
                main()
        self.assertEqual(context.exception.code, 2)

    def test_dev_option_rejects_named_confirmation_prediction_file(self) -> None:
        with (
            patch.object(sys, "argv", [
                "audit_massive_slots_overlap.py",
                "--dev-system", "qwen=confirmation-predictions.jsonl",
            ]),
            patch.object(Path, "read_text", side_effect=AssertionError("Read a confirmation file")),
            redirect_stderr(io.StringIO()),
        ):
            with self.assertRaises(SystemExit) as context:
                main()
        self.assertEqual(context.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
