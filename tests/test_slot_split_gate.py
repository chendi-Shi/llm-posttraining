"""Synthetic CLI checks that dev scoring cannot spend confirmation data."""

from __future__ import annotations

import json
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import compare_massive_slots  # noqa: E402
import evaluate_massive_slots  # noqa: E402
from prepare_massive_slots import SOURCE_SHA256  # noqa: E402


class SplitGateCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.train = {
            "id": "1", "utt": "今天", "slots": [{"type": "date", "value": "今天"}],
            "source_partition": "train", "group_sha256": "train-group",
        }
        self.dev = {
            "id": "2", "utt": "明天", "slots": [{"type": "date", "value": "明天"}],
            "source_partition": "train", "group_sha256": "dev-group",
        }
        self.manifest = {
            "format_version": 1,
            "source_partition_used": "train",
            "source_sha256": SOURCE_SHA256,
            "splits": {
                "train": {"examples": 1, "jsonl_sha256": "trainhash", "source_ids": ["1"],
                          "groups": [{"group_sha256": "train-group"}]},
                "dev": {"examples": 1, "jsonl_sha256": "devhash", "source_ids": ["2"],
                        "groups": [{"group_sha256": "dev-group"}]},
            },
        }

    @staticmethod
    def fake_hash(path: Path) -> str:
        return {"dev.jsonl": "devhash", "train.jsonl": "trainhash"}.get(path.name, "predictionhash")

    def test_qwen_dev_main_runs_with_locked_synthetic_split(self) -> None:
        def read_rows(path: Path) -> list[dict]:
            return [self.train] if path.name == "train.jsonl" else [self.dev]

        with (
            patch.object(sys, "argv", ["evaluate_massive_slots.py", "--eval-file", "study/dev.jsonl",
                                       "--metrics", "aggregate.json"]),
            patch.object(Path, "is_file", return_value=True),
            patch.object(Path, "read_text", return_value=json.dumps(self.manifest)),
            patch.object(Path, "write_text"),
            patch.object(Path, "mkdir"),
            patch.object(evaluate_massive_slots, "sha256_file", side_effect=self.fake_hash),
            patch.object(evaluate_massive_slots, "read_jsonl", side_effect=read_rows),
            patch.object(evaluate_massive_slots, "generate_raw_predictions",
                         return_value=([json.dumps({"slots": self.dev["slots"]}, ensure_ascii=False)],
                                       {"max_observed_input_tokens": 8})) as generate,
            redirect_stdout(StringIO()),
        ):
            evaluate_massive_slots.main()
        generate.assert_called_once()

    def test_compare_dev_main_runs_with_locked_synthetic_split(self) -> None:
        empty = {"id": "2", "group_sha256": "dev-group", "eval_file_sha256": "devhash",
                 "raw_output": '{"slots":[]}'}
        correct = dict(empty, raw_output=json.dumps({"slots": self.dev["slots"]}, ensure_ascii=False))

        def read_rows(path: Path) -> list[dict]:
            return {"dev.jsonl": [self.dev], "base.jsonl": [empty],
                    "sft.jsonl": [correct]}[path.name]

        with (
            patch.object(sys, "argv", ["compare_massive_slots.py", "--eval-file", "study/dev.jsonl",
                                       "--system", "base=base.jsonl", "--system", "sft=sft.jsonl",
                                       "--output", "aggregate.json"]),
            patch.object(Path, "is_file", return_value=True),
            patch.object(Path, "read_text", return_value=json.dumps(self.manifest)),
            patch.object(Path, "write_text"),
            patch.object(Path, "mkdir"),
            patch.object(evaluate_massive_slots, "sha256_file", side_effect=self.fake_hash),
            patch.object(evaluate_massive_slots, "read_jsonl", return_value=[self.train]),
            patch.object(compare_massive_slots, "sha256_file", side_effect=self.fake_hash),
            patch.object(compare_massive_slots, "read_jsonl", side_effect=read_rows),
            redirect_stdout(StringIO()),
        ):
            compare_massive_slots.main()

    def test_wrong_dev_sha_stops_compare_before_prediction_files_are_opened(self) -> None:
        with (
            patch.object(sys, "argv", ["compare_massive_slots.py", "--eval-file", "study/dev.jsonl",
                                       "--system", "base=base.jsonl", "--system", "sft=sft.jsonl",
                                       "--output", "aggregate.json"]),
            patch.object(Path, "is_file", return_value=True),
            patch.object(Path, "read_text", return_value=json.dumps(self.manifest)),
            patch.object(evaluate_massive_slots, "sha256_file", return_value="wrong"),
            patch.object(compare_massive_slots, "read_jsonl") as reader,
        ):
            with self.assertRaisesRegex(ValueError, "differs from the locked"):
                compare_massive_slots.main()
        reader.assert_not_called()

    def test_qwen_confirmation_requires_unlock_before_any_file_read(self) -> None:
        with (
            patch.object(sys, "argv", ["evaluate_massive_slots.py", "--role", "confirmation",
                                       "--eval-file", "study/confirmation.jsonl", "--metrics", "aggregate.json"]),
            patch.object(Path, "read_text") as reader,
            patch.object(evaluate_massive_slots, "read_jsonl") as rows_reader,
            redirect_stderr(StringIO()),
        ):
            with self.assertRaises(SystemExit):
                evaluate_massive_slots.main()
        reader.assert_not_called()
        rows_reader.assert_not_called()

    def test_qwen_confirmation_requires_frozen_file_hashes_before_data_read(self) -> None:
        with (
            patch.object(sys, "argv", ["evaluate_massive_slots.py", "--role", "confirmation",
                                       "--unlock-confirmation", "--eval-file", "study/confirmation.jsonl",
                                       "--metrics", "aggregate.json"]),
            patch.object(Path, "read_text") as reader,
            patch.object(evaluate_massive_slots, "read_jsonl") as rows_reader,
            redirect_stderr(StringIO()),
        ):
            with self.assertRaises(SystemExit):
                evaluate_massive_slots.main()
        reader.assert_not_called()
        rows_reader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
