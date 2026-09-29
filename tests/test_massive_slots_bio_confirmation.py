"""Confirmation gate and frozen provenance checks without using real gold."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate_massive_slots_bio_confirmation import (  # noqa: E402
    check_frozen_hashes,
    main,
    validate_selected_rows,
)


class FrozenConfirmationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.hashes = {key: character * 64 for key, character in
                       (("source", "a"), ("train", "b"),
                        ("confirmation", "c"), ("model", "d"))}
        self.manifest = {
            "source_sha256": self.hashes["source"],
            "splits": {
                "train": {"examples": 384},
                "confirmation": {"examples": 200},
            },
        }
        self.dev_report = {
            "model_sha256": self.hashes["model"],
            "train_file_sha256": self.hashes["train"],
            "source_sha256": self.hashes["source"],
            "confirmation_used": False,
        }
        self.manifest["splits"]["train"]["jsonl_sha256"] = self.hashes["train"]
        self.manifest["splits"]["confirmation"]["jsonl_sha256"] = self.hashes["confirmation"]

    def test_unlock_is_checked_before_any_confirmation_file_access(self) -> None:
        with patch.object(Path, "read_text", side_effect=AssertionError("file opened")), \
             patch.object(Path, "open", side_effect=AssertionError("file opened")):
            with self.assertRaisesRegex(SystemExit, "Confirmation remains locked"):
                main(["--expected-model-sha256", self.hashes["model"]])

    def test_model_and_data_hashes_must_match_frozen_dev_run(self) -> None:
        check_frozen_hashes(self.manifest, self.dev_report, self.hashes,
                            self.hashes["model"])
        changed = dict(self.hashes, model="e" * 64)
        with self.assertRaisesRegex(ValueError, "model SHA-256 mismatch"):
            check_frozen_hashes(self.manifest, self.dev_report, changed,
                                self.hashes["model"])
        changed = dict(self.hashes, confirmation="e" * 64)
        with self.assertRaisesRegex(ValueError, "confirmation SHA-256 mismatch"):
            check_frozen_hashes(self.manifest, self.dev_report, changed,
                                self.hashes["model"])

    def test_selected_ids_and_group_hashes_are_verified(self) -> None:
        train = [{"id": str(index), "group_sha256": f"train-{index}"}
                 for index in range(384)]
        confirmation = [{"id": str(index), "group_sha256": f"confirm-{index}"}
                        for index in range(384, 584)]
        for split, rows in (("train", train), ("confirmation", confirmation)):
            self.manifest["splits"][split]["source_ids"] = [row["id"] for row in rows]
            self.manifest["splits"][split]["groups"] = [
                {"group_sha256": row["group_sha256"]} for row in rows
            ]
        validate_selected_rows(train, confirmation, self.manifest)
        confirmation[0] = dict(confirmation[0], group_sha256="train-0")
        with self.assertRaisesRegex(ValueError, "phrase hashes differ"):
            validate_selected_rows(train, confirmation, self.manifest)


if __name__ == "__main__":
    unittest.main()
