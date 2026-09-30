from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate_massive_slots_v4_dev import gate


class DevelopmentGateTests(unittest.TestCase):
    def test_requires_every_preset_condition(self) -> None:
        metrics = {
            "micro_f1": .55,
            "no_slot_failure_rate": .30,
            "json_valid_rate": .95,
            "schema_valid_rate": .95,
            "copy_valid_rate": .95,
        }
        positive = {"micro_f1": .60}
        paired = {"delta_ci95": [.001, .1]}
        self.assertTrue(gate(metrics, positive, paired)["passed"])
        for field, failing in (("micro_f1", .549), ("no_slot_failure_rate", .301),
                               ("json_valid_rate", .949), ("schema_valid_rate", .949),
                               ("copy_valid_rate", .949)):
            with self.subTest(field=field):
                self.assertFalse(gate({**metrics, field: failing}, positive, paired)["passed"])
        self.assertFalse(gate(metrics, {"micro_f1": .599}, paired)["passed"])
        self.assertFalse(gate(metrics, positive, {"delta_ci95": [0.0, .1]})["passed"])


if __name__ == "__main__":
    unittest.main()
