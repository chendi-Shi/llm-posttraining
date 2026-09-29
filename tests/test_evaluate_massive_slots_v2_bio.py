"""Safety checks for the independent v2 matched-data BIO control."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate_massive_slots_v2_bio import (  # noqa: E402
    CONFIRMATION_CODE_FILES, check_manifest_counts, main, write_bio_predictions,
)


class V2BioSafetyTests(unittest.TestCase):
    def test_confirmation_stays_unopened_until_all_hashes_are_supplied(self) -> None:
        with patch.object(Path, "read_text", side_effect=AssertionError("read")), \
             patch.object(Path, "open", side_effect=AssertionError("open")):
            with self.assertRaisesRegex(ValueError, "--unlock-confirmation"):
                main(["--role", "confirmation"])
            with self.assertRaisesRegex(ValueError, "expected-selection-lock-sha256"):
                main(["--role", "confirmation", "--unlock-confirmation"])
            with self.assertRaisesRegex(ValueError, "expected-manifest-sha256"):
                main(["--role", "confirmation", "--unlock-confirmation",
                      "--expected-selection-lock-sha256", "a" * 64])

    def test_v1_manifest_cannot_be_used_as_v2(self) -> None:
        manifest = {"splits": {
            "train": {"examples": 640},
            "dev": {"examples": 250},
            "confirmation": {"examples": 400},
        }}
        check_manifest_counts(manifest)
        manifest["splits"]["train"]["examples"] = 384
        with self.assertRaisesRegex(ValueError, "v2 train/dev/confirmation"):
            check_manifest_counts(manifest)

    def test_confirmation_refuses_existing_outputs_before_heldout_read(self) -> None:
        with patch.object(Path, "exists", return_value=True), \
             patch("evaluate_massive_slots_v2_bio.sha256_file", side_effect=AssertionError("hashed")):
            with self.assertRaisesRegex(ValueError, "outputs already exist"):
                main(["--role", "confirmation", "--unlock-confirmation",
                      "--expected-selection-lock-sha256", "a" * 64,
                      "--expected-manifest-sha256", "b" * 64,
                       "--expected-model-sha256", "c" * 64,
                       "--expected-dev-report-sha256", "d" * 64])

    def test_confirmation_refuses_alternate_output_paths_before_heldout_read(self) -> None:
        with patch.object(Path, "read_text", side_effect=AssertionError("held-out read")), \
             patch("evaluate_massive_slots_v2_bio.sha256_file", side_effect=AssertionError("held-out hash")):
            for flag, path in (("--predictions", "_tmp/alternate-confirmation.jsonl"),
                               ("--report", "reports/alternate-confirmation.json")):
                with self.subTest(flag=flag), self.assertRaisesRegex(ValueError, "fixed paths"):
                    main(["--role", "confirmation", "--unlock-confirmation",
                          "--expected-selection-lock-sha256", "a" * 64,
                          "--expected-manifest-sha256", "b" * 64,
                          "--expected-model-sha256", "c" * 64,
                          "--expected-dev-report-sha256", "d" * 64,
                          flag, path])

    def test_confirmation_refuses_alternate_selection_lock_before_heldout_read(self) -> None:
        with patch.object(Path, "read_text", side_effect=AssertionError("held-out read")), \
             patch("evaluate_massive_slots_v2_bio.sha256_file", side_effect=AssertionError("held-out hash")):
            with self.assertRaisesRegex(ValueError, "canonical v2 selection lock path"):
                main(["--role", "confirmation", "--unlock-confirmation",
                      "--expected-selection-lock-sha256", "a" * 64,
                      "--expected-manifest-sha256", "b" * 64,
                      "--expected-model-sha256", "c" * 64,
                      "--expected-dev-report-sha256", "d" * 64,
                      "--selection-lock", "reports/alternate-v2-selection-lock.json"])

    def test_changed_dependency_rejected_before_confirmation_rows_are_read(self) -> None:
        lock = {
            "status": "selected", "selected_step": 80,
            "manifest_sha256": "b" * 64,
            "reference_systems": {"bio": {
                "model_sha256": "c" * 64,
                "dev_report_sha256": "d" * 64,
            }},
            "code_sha256": {relative: "a" * 64 for relative in CONFIRMATION_CODE_FILES},
        }
        lock["code_sha256"]["scripts/evaluate_massive_slots_bio.py"] = "e" * 64
        with patch.object(Path, "exists", return_value=False), \
             patch.object(Path, "read_text", return_value=json.dumps(lock)), \
             patch("evaluate_massive_slots_v2_bio.check_file_hash"), \
             patch("evaluate_massive_slots_v2_bio.sha256_file", return_value="a" * 64), \
             patch("evaluate_massive_slots_v2_bio.read_jsonl", side_effect=AssertionError("held-out read")), \
             patch("evaluate_massive_slots_v2_bio.load_locked_split_manifest", side_effect=AssertionError("held-out preflight")):
            with self.assertRaisesRegex(ValueError, "evaluate_massive_slots_bio.py"):
                main(["--role", "confirmation", "--unlock-confirmation",
                      "--expected-selection-lock-sha256", "a" * 64,
                      "--expected-manifest-sha256", "b" * 64,
                      "--expected-model-sha256", "c" * 64,
                      "--expected-dev-report-sha256", "d" * 64])

    def test_predictions_stay_ignored_and_confirmation_uses_exclusive_creation(self) -> None:
        row = {"id": "1", "group_sha256": "g"}
        with self.assertRaisesRegex(ValueError, "ignored _tmp"):
            write_bio_predictions(Path("reports/leak.jsonl"), [row], ['{"slots":[]}'],
                                  "a" * 64, exclusive=False)
        with patch.object(Path, "mkdir"), patch.object(Path, "open") as opener:
            write_bio_predictions(Path("_tmp/v2-bio-exclusive.jsonl"), [row],
                                  ['{"slots":[]}'], "a" * 64, exclusive=True)
        opener.assert_called_once_with("x", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    unittest.main()
