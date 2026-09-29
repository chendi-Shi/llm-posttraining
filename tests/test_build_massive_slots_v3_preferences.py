"""Offline checks for frozen preference quotas and chosen/rejected semantics."""

from __future__ import annotations

import sys
import unittest
from collections import UserDict
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_massive_slots_v3_preferences import (  # noqa: E402
    QUOTAS, fallback_rejected, make_pair, select_candidates,
)
from massive_slots import parse_prediction  # noqa: E402


class FakeTokenizer:
    def apply_chat_template(self, messages, **_kwargs):
        values = list("".join(item["content"] for item in messages))
        return UserDict({"input_ids": values, "attention_mask": [1] * len(values)})


class PreferenceTests(unittest.TestCase):
    def test_selection_is_fixed_and_group_unique(self) -> None:
        rows = []
        for bucket, count in (("date", 48), ("time", 48), ("place_name", 48),
                              ("person", 48), ("hard_negative", 192),
                              ("pure_negative", 192)):
            for number in range(count):
                rows.append({"bucket": bucket, "group_sha256": f"{bucket}-{number}"})
        first = select_candidates(rows)
        second = select_candidates(list(reversed(rows)))
        self.assertEqual(first, second)
        self.assertEqual(len(first), 256)
        self.assertEqual({bucket: sum(row["bucket"] == bucket for row in first)
                          for bucket in QUOTAS}, QUOTAS)

    def test_negative_fallback_is_copied_but_wrong(self) -> None:
        row = {"utt": "一欧元值多少钱", "slots": [], "group_sha256": "0" * 64}
        rejected = fallback_rejected(row)
        parsed, validity = parse_prediction(rejected, row["utt"])
        self.assertTrue(all(validity[name] for name in ("json_valid", "schema_valid", "copy_valid")))
        self.assertTrue(parsed)

    def test_correct_model_reply_uses_opposite_fallback(self) -> None:
        row = {"utt": "明天去北京", "slots": [{"type": "date", "value": "明天"}],
               "group_sha256": "a" * 64,
               "prompt": [{"role": "user", "content": "抽取四槽：明天去北京"}],
               "completion": [{"role": "assistant", "content": '{"slots":[{"type":"date","value":"明天"}]}'}]}
        pair, source, length = make_pair(row, row["completion"][0]["content"], FakeTokenizer())
        self.assertEqual(source, "deterministic_fallback")
        self.assertEqual(pair["rejected"][0]["content"], '{"slots":[]}')
        self.assertLessEqual(length, 256)


if __name__ == "__main__":
    unittest.main()
