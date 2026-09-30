from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from slot_confidence import choose_hybrid_output, token_ids


class ConfidencePolicyTests(unittest.TestCase):
    def test_token_id_container_is_flattened(self) -> None:
        self.assertEqual(token_ids({"input_ids": [[1, 2, 3]]}), [1, 2, 3])

    def test_only_valid_high_confidence_nonempty_dpo_replaces_bio(self) -> None:
        utterance = "明天去北京"
        bio = '{"slots":[]}'
        dpo = '{"slots":[{"type":"place_name","value":"北京"}]}'
        self.assertEqual(choose_hybrid_output(bio, dpo, utterance, -0.08), (dpo, True))
        self.assertEqual(choose_hybrid_output(bio, dpo, utterance, -0.08001), (bio, False))
        self.assertEqual(choose_hybrid_output(bio, dpo, utterance, None), (bio, False))
        self.assertEqual(choose_hybrid_output(bio, '{"slots":[]}', utterance, 0.0), (bio, False))
        invalid = '{"slots":[{"type":"place_name","value":"上海"}]}'
        self.assertEqual(choose_hybrid_output(bio, invalid, utterance, 0.0), (bio, False))


if __name__ == "__main__":
    unittest.main()
