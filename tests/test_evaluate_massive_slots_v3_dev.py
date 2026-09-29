"""The v3 gate must require every preregistered threshold."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_massive_slots_v3_dev import development_gate  # noqa: E402


class DevelopmentGateTests(unittest.TestCase):
    def test_all_thresholds_and_both_paired_intervals_are_required(self) -> None:
        metrics = {
            "json_valid_rate": 0.95,
            "schema_valid_rate": 0.95,
            "copy_valid_rate": 0.95,
            "no_slot_failure_rate": 0.30,
            "micro_f1": 0.55,
        }
        paired_f1 = {"delta_ci95": [0.001, 0.2]}
        paired_no_slot = {"reduction_ci95": [0.001, 0.2]}
        self.assertTrue(development_gate(metrics, 0.60, paired_f1, paired_no_slot)["passed"])
        self.assertFalse(development_gate(metrics, 0.599, paired_f1, paired_no_slot)["passed"])
        self.assertFalse(development_gate(metrics, 0.60,
                                          {"delta_ci95": [0.0, 0.2]}, paired_no_slot)["passed"])
        self.assertFalse(development_gate(metrics, 0.60, paired_f1,
                                          {"reduction_ci95": [-0.01, 0.2]})["passed"])


if __name__ == "__main__":
    unittest.main()
