"""The public evidence gate must catch changed results and provenance links."""

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from audit_public_results import EvidenceError, audit  # noqa: E402


FILES = (
    "massive-slots-selection-lock.json",
    "massive-slots-confirmation-comparison.json",
    "massive-slots-v2-sft-step160-dev.json",
    "massive-slots-v3-dpo-dev.json",
    "massive-slots-v3-matched-sft-dev.json",
)


class PublicEvidenceAuditTests(unittest.TestCase):
    def setUp(self):
        self.reports = {
            name: json.loads((ROOT / "reports" / name).read_text(encoding="utf-8"))
            for name in FILES
        }

    def mutate(self, name, change):
        change(self.reports[name])

    def test_published_reports_are_consistent(self):
        self.assertGreaterEqual(audit(ROOT)["checks"], 200)

    def test_changed_f1_is_rejected(self):
        self.mutate("massive-slots-v3-dpo-dev.json",
                    lambda report: report["metrics"].__setitem__(
                        "micro_f1", report["metrics"]["micro_f1"] + 0.01))
        with self.assertRaisesRegex(EvidenceError, "dpo: micro F1"):
            audit(ROOT, self.reports)

    def test_changed_reference_prediction_hash_is_rejected(self):
        self.mutate("massive-slots-v3-matched-sft-dev.json",
                    lambda report: report["reference_prediction_file_sha256"].__setitem__(
                        "v2_sft", "0" * 64))
        with self.assertRaisesRegex(EvidenceError, "control: reference predictions differ"):
            audit(ROOT, self.reports)

    def test_changed_generation_protocol_is_rejected(self):
        self.mutate("massive-slots-v2-sft-step160-dev.json",
                    lambda report: report["generation"].__setitem__("max_new_tokens", 32))
        with self.assertRaisesRegex(EvidenceError, "generation protocol differs"):
            audit(ROOT, self.reports)

    def test_unearned_gate_pass_is_rejected(self):
        self.mutate("massive-slots-v3-dpo-dev.json",
                    lambda report: report["development_gate"].__setitem__("passed", True))
        with self.assertRaisesRegex(EvidenceError, "development gate decision"):
            audit(ROOT, self.reports)


if __name__ == "__main__":
    unittest.main()
