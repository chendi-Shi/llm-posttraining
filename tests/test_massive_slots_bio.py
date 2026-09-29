"""Check exact-span labels and legal decoding for the BIO baseline."""

from __future__ import annotations

import sys
import json
import unittest
from pathlib import Path
from unittest.mock import mock_open, patch

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate_massive_slots_bio import (  # noqa: E402
    constrained_bio_decode,
    gold_bio_tags,
    slots_from_bio,
    write_predictions,
)


class BioDecodingTests(unittest.TestCase):
    def test_gold_offsets_keep_repeated_values_as_distinct_spans(self) -> None:
        utterance = "Tom和Tom"
        spans = [
            {"type": "person", "start": 0, "end": 3},
            {"type": "person", "start": 4, "end": 7},
        ]
        tags = gold_bio_tags(utterance, spans)
        self.assertEqual(tags, ["B-person", "I-person", "I-person", "O",
                                "B-person", "I-person", "I-person"])
        self.assertEqual(
            slots_from_bio(utterance, tags),
            [{"type": "person", "value": "Tom"}, {"type": "person", "value": "Tom"}],
        )

    def test_viterbi_rejects_initial_and_cross_type_inside_tags(self) -> None:
        classes = ["O", "B-date", "I-date", "B-time", "I-time"]
        scores = np.array([
            [0.0, 2.0, 10.0, -10.0, 0.0],
            [0.0, 0.0, 4.0, 1.0, 9.0],
            [2.0, 0.0, 0.0, 0.0, 0.0],
        ])
        tags = constrained_bio_decode(scores, classes)
        self.assertEqual(tags[0], "B-date")
        self.assertEqual(tags[1], "I-date")
        self.assertEqual(tags[2], "O")

    def test_invalid_gold_overlap_and_length_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            gold_bio_tags("明天", [
                {"type": "date", "start": 0, "end": 2},
                {"type": "time", "start": 1, "end": 2},
            ])
        with self.assertRaisesRegex(ValueError, "One BIO tag"):
            slots_from_bio("明天", ["O"])

    def test_pairwise_prediction_file_has_hashes_and_no_utterances(self) -> None:
        rows = [
            {"id": "7", "utt": "明天", "group_sha256": "abc"},
            {"id": "8", "utt": "后天", "group_sha256": "def"},
        ]
        mocked_open = mock_open()
        with patch.object(Path, "mkdir"), patch.object(Path, "open", mocked_open):
            write_predictions(Path("ignored.jsonl"), rows,
                              ['{"slots":[]}', '{"slots":[]}'], "filehash")
        written = "".join(call.args[0] for call in mocked_open().write.call_args_list)
        outputs = [json.loads(line) for line in written.splitlines()]
        self.assertEqual(len(outputs), 2)
        self.assertEqual(set(outputs[0]),
                         {"id", "group_sha256", "eval_file_sha256", "raw_output"})
        self.assertEqual(outputs[0]["eval_file_sha256"], "filehash")
        self.assertNotIn("utt", outputs[0])
        with self.assertRaisesRegex(ValueError, "Prediction count"):
            write_predictions(Path("unused"), rows, ['{"slots":[]}'], "filehash")


if __name__ == "__main__":
    unittest.main()
