"""The v3 candidate builder excludes sealed v2 groups without reading labels."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_massive_slots_v2 import group_sha256, strong_group_key  # noqa: E402
from prepare_massive_slots_v3_preferences import add_reserved_keys  # noqa: E402


class ReservedGroupsTests(unittest.TestCase):
    def test_reserved_group_is_excluded_by_utterance_only(self) -> None:
        row = {"utt": "北京，明天", "annot_utt": object()}
        key = strong_group_key("北京明天")
        prior = add_reserved_keys(set(), [row], {group_sha256(key)})
        self.assertEqual(prior, {key})

    def test_missing_reserved_group_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "every reserved v2 group"):
            add_reserved_keys(set(), [{"utt": "另一句话"}], {group_sha256("未找到")})


if __name__ == "__main__":
    unittest.main()
