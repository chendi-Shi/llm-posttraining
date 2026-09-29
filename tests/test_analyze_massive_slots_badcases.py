"""Diagnostic error flags stay aggregate, dev-only, and multiset-aware."""

from __future__ import annotations

import contextlib
import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analyze_massive_slots_badcases import analyze_rows, main  # noqa: E402
from massive_slots import canonical_output  # noqa: E402


def slot(kind: str, value: str) -> dict[str, str]:
    return {"type": kind, "value": value}


def row(identifier: str, utterance: str, slots: list[dict]) -> dict:
    return {
        "id": identifier,
        "utt": utterance,
        "slots": slots,
        "source_partition": "train",
        "group_sha256": "group-" + identifier,
    }


class BadCaseAnalysisTests(unittest.TestCase):
    def test_overlapping_flags_and_repeated_slot_multisets(self) -> None:
        rows = [
            row("1", "今天和今天", [slot("date", "今天"), slot("date", "今天")]),
            row("2", "今天", [slot("date", "今天")]),
            row("3", "明天", [slot("date", "明天")]),
            row("4", "北京市", [slot("place_name", "北京市")]),
            row("5", "小明和小红", [slot("person", "小明")]),
            row("6", "今天听歌", []),
            row("7", "今天", [slot("date", "今天")]),
            row("8", "今天", [slot("date", "今天")]),
            row("9", "今天", [slot("date", "今天")]),
            row("10", "请放音乐", []),
            row("11", "今天和今天", [slot("date", "今天"), slot("date", "今天")]),
            row("12", "明天从北京到上海", [slot("date", "明天"), slot("place_name", "北京")]),
        ]
        raw = [
            canonical_output([slot("date", "今天")]),
            canonical_output([slot("date", "今天"), slot("date", "今天")]),
            canonical_output([slot("time", "明天")]),
            canonical_output([slot("place_name", "北京")]),
            canonical_output([slot("person", "小红")]),
            canonical_output([slot("date", "今天")]),
            "not json",
            '{"slots":"bad"}',
            canonical_output([slot("date", "明天")]),
            canonical_output([]),
            canonical_output([slot("date", "今天"), slot("date", "今天")]),
            canonical_output([slot("date", "明天"), slot("place_name", "上海")]),
        ]
        report = analyze_rows(rows, raw)
        category = lambda name: report["categories"][name]["examples"]
        self.assertEqual(report["denominators"]["gold_nonempty_rows"], 10)
        self.assertEqual(report["denominators"]["gold_empty_rows"], 2)
        self.assertEqual(category("invalid_json"), 1)
        self.assertEqual(category("invalid_schema"), 1)
        self.assertEqual(category("invalid_copy"), 1)
        self.assertEqual(category("missed_all_gold_slots"), 6)
        self.assertEqual(category("missed_all_gold_slots_valid_output"), 3)
        self.assertEqual(category("extra_slots_on_gold_empty"), 1)
        self.assertEqual(category("type_confusion_same_value"), 1)
        self.assertEqual(category("same_type_different_value"), 3)
        self.assertEqual(category("possible_span_boundary_mismatch"), 1)
        self.assertEqual(category("partial_correct"), 3)
        self.assertEqual(category("duplicate_slot_output"), 2)
        self.assertEqual(category("excess_duplicate_slot"), 1)
        self.assertEqual(category("fully_correct"), 2)
        self.assertEqual(category("incorrect"), 10)
        self.assertAlmostEqual(
            report["categories"]["extra_slots_on_gold_empty"]["rate_gold_empty_rows"],
            0.5,
        )
        self.assertTrue(report["categories_overlap"])

    def test_empty_relevant_denominator_is_null(self) -> None:
        report = analyze_rows([row("1", "今天", [slot("date", "今天")])], [canonical_output([])])
        self.assertIsNone(
            report["categories"]["extra_slots_on_gold_empty"]["rate_gold_empty_rows"]
        )

    def test_cli_output_is_aggregate_and_requires_locked_dev_alignment(self) -> None:
        example = row("private-id-123", "隐私原文甲乙", [slot("person", "甲乙")])
        prediction = {
            "id": example["id"],
            "group_sha256": example["group_sha256"],
            "eval_file_sha256": "file-hash",
            "raw_output": "private-raw-output-invalid-json",
        }
        argv = [
            "analyze_massive_slots_badcases.py", "--eval-file", "study/dev.jsonl",
            "--system", "candidate=study/predictions.jsonl", "--output", "study/aggregate.json",
        ]
        with (
            patch.object(sys, "argv", argv),
            patch("analyze_massive_slots_badcases.read_jsonl", side_effect=lambda path: [example] if path.name == "dev.jsonl" else [prediction]),
            patch("analyze_massive_slots_badcases._validate_study_split") as validate,
            patch("analyze_massive_slots_badcases.sha256_file", return_value="file-hash"),
            patch.object(Path, "is_file", return_value=True),
            patch.object(Path, "mkdir"),
            patch.object(Path, "write_text") as write_report,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            main()
        validate.assert_called_once()
        published = write_report.call_args.args[0]
        self.assertNotIn("private-id-123", published)
        self.assertNotIn("隐私原文甲乙", published)
        self.assertNotIn("private-raw-output", published)
        self.assertEqual(json.loads(published)["systems"]["candidate"]["analysis"]["categories"]["invalid_json"]["examples"], 1)

        argv[2] = "study/confirmation.jsonl"
        with patch.object(sys, "argv", argv), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                main()

        prediction["eval_file_sha256"] = "wrong-hash"
        argv[2] = "study/dev.jsonl"
        with (
            patch.object(sys, "argv", argv),
            patch("analyze_massive_slots_badcases.read_jsonl", side_effect=lambda path: [example] if path.name == "dev.jsonl" else [prediction]),
            patch("analyze_massive_slots_badcases._validate_study_split"),
            patch("analyze_massive_slots_badcases.sha256_file", return_value="file-hash"),
            patch.object(Path, "is_file", return_value=True),
        ):
            with self.assertRaisesRegex(ValueError, "Evaluation-file hash mismatch"):
                main()


if __name__ == "__main__":
    unittest.main()
