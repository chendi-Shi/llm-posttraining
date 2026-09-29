"""Prediction provenance and alignment checks for paired slot comparison."""

from __future__ import annotations

import sys
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from compare_massive_slots import (  # noqa: E402
    aligned_raw_outputs,
    main,
    parse_system,
    validate_locked_confirmation,
)


class ComparisonInputTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = [
            {"id": "1", "group_sha256": "phrase-1"},
            {"id": "2", "group_sha256": "phrase-2"},
        ]
        self.predictions = [
            {"id": "2", "group_sha256": "phrase-2", "eval_file_sha256": "file-hash", "raw_output": "{}"},
            {"id": "1", "group_sha256": "phrase-1", "eval_file_sha256": "file-hash", "raw_output": "{\"slots\":[]}"},
        ]

    def test_aligns_by_id_even_when_prediction_file_order_differs(self) -> None:
        self.assertEqual(
            aligned_raw_outputs(self.rows, self.predictions, "file-hash"),
            ['{"slots":[]}', "{}"],
        )

    def test_rejects_wrong_phrase_or_file_hash(self) -> None:
        wrong_phrase = [dict(record) for record in self.predictions]
        wrong_phrase[0]["group_sha256"] = "other"
        with self.assertRaisesRegex(ValueError, "Phrase-group hash mismatch"):
            aligned_raw_outputs(self.rows, wrong_phrase, "file-hash")
        with self.assertRaisesRegex(ValueError, "Evaluation-file hash mismatch"):
            aligned_raw_outputs(self.rows, self.predictions, "other-file")

    def test_rejects_missing_or_duplicate_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "do not exactly match"):
            aligned_raw_outputs(self.rows, self.predictions[:1], "file-hash")
        with self.assertRaisesRegex(ValueError, "duplicate ID"):
            aligned_raw_outputs(self.rows, self.predictions + self.predictions[:1], "file-hash")

    def test_system_argument(self) -> None:
        self.assertEqual(parse_system("base=predictions.jsonl"), ("base", Path("predictions.jsonl")))
        with self.assertRaises(Exception):
            parse_system("bad input")

    def test_confirmation_requires_unlock_before_reading_data(self) -> None:
        arguments = [
            "compare_massive_slots.py",
            "--role", "confirmation",
            "--eval-file", "data/massive-zh/slots/confirmation.jsonl",
            "--system", "base=base.jsonl",
            "--system", "sft=sft.jsonl",
            "--output", "report.json",
        ]
        with (
            patch.object(sys, "argv", arguments),
            patch("compare_massive_slots.read_jsonl") as reader,
            redirect_stderr(StringIO()),
            self.assertRaises(SystemExit) as stopped,
        ):
            main()
        self.assertEqual(stopped.exception.code, 2)
        reader.assert_not_called()

    def test_locked_manifest_checks_ids_groups_and_hash(self) -> None:
        rows = [
            {"id": "5", "group_sha256": "confirmation-group-1"},
            {"id": "6", "group_sha256": "confirmation-group-2"},
        ]
        manifest = {
            "format_version": 1,
            "source_partition_used": "train",
            "splits": {
                "train": {"groups": [{"group_sha256": "training-group"}]},
                "dev": {"groups": [{"group_sha256": "development-group"}]},
                "confirmation": {
                    "jsonl_sha256": "frozen-file-hash",
                    "examples": 2,
                    "source_ids": ["5", "6"],
                    "groups": [
                        {"group_sha256": "confirmation-group-1"},
                        {"group_sha256": "confirmation-group-2"},
                    ],
                },
            },
        }
        validate_locked_confirmation(rows, manifest, "frozen-file-hash")
        with self.assertRaisesRegex(ValueError, "file hash"):
            validate_locked_confirmation(rows, manifest, "different-hash")
        with self.assertRaisesRegex(ValueError, "IDs differ"):
            validate_locked_confirmation(list(reversed(rows)), manifest, "frozen-file-hash")
        altered = [dict(row) for row in rows]
        altered[1]["group_sha256"] = "unexpected-group"
        with self.assertRaisesRegex(ValueError, "group hashes differ"):
            validate_locked_confirmation(altered, manifest, "frozen-file-hash")
        manifest["splits"]["dev"]["groups"].append(
            {"group_sha256": "confirmation-group-2"}
        )
        with self.assertRaisesRegex(ValueError, "overlap locked dev"):
            validate_locked_confirmation(rows, manifest, "frozen-file-hash")


if __name__ == "__main__":
    unittest.main()
