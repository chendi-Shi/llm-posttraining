"""Focused checks for the new structured slot dataset."""

from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from massive_slots import (  # noqa: E402
    TARGET_TYPES,
    canonical_json,
    canonical_output,
    group_key,
    parse_annotated,
    parse_annotated_spans,
    parse_prediction,
    slots_from_annotation,
)
from prepare_massive_slots import select_splits  # noqa: E402


class SlotAnnotationTests(unittest.TestCase):
    def test_chinese_annotation_spaces_map_to_exact_source_values(self) -> None:
        slots = slots_from_annotation(
            "这周五点叫我起床", "[date : 这周] [time : 五点] 叫我起床"
        )
        self.assertEqual(
            slots,
            [
                {"type": "date", "value": "这周"},
                {"type": "time", "value": "五点"},
            ],
        )
        self.assertEqual(
            canonical_json(slots),
            '{"slots":[{"type":"date","value":"这周"},{"type":"time","value":"五点"}]}',
        )

    def test_repeated_type_and_internal_spaces_are_preserved(self) -> None:
        self.assertEqual(
            slots_from_annotation(
                "给 Tom 和 Tom 发消息", "给 [person : Tom] 和 [person : Tom] 发消息"
            ),
            [
                {"type": "person", "value": "Tom"},
                {"type": "person", "value": "Tom"},
            ],
        )
        self.assertEqual(
            slots_from_annotation("New York 的天气", "[place_name : New York] 的天气"),
            [{"type": "place_name", "value": "New York"}],
        )
        self.assertEqual(
            parse_annotated_spans("New York 的天气", "[place_name : New York] 的天气"),
            [{"type": "place_name", "value": "New York", "start": 0, "end": 8}],
        )

    def test_other_slot_types_are_valid_negatives(self) -> None:
        self.assertEqual(slots_from_annotation("把灯调成红色", "把灯调成 [color_type : 红色]"), [])
        self.assertEqual(canonical_json([]), '{"slots":[]}')

    def test_misaligned_or_broken_annotations_fail(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be aligned"):
            slots_from_annotation("五点叫我", "[time : 六点] 叫我")
        with self.assertRaisesRegex(ValueError, "Unparsed annotation"):
            slots_from_annotation("五点叫我", "[time 五点] 叫我")

    def test_group_key_collapses_case_width_and_whitespace(self) -> None:
        self.assertEqual(group_key("Ａ B 早安"), group_key("ab早安"))

    def test_public_api_and_strict_prediction_validation(self) -> None:
        self.assertEqual(parse_annotated("明天提醒我", "[date : 明天] 提醒我"),
                         [{"type": "date", "value": "明天"}])
        self.assertEqual(canonical_output([]), '{"slots":[]}')
        parsed, validity = parse_prediction(
            '{"slots":[{"type":"date","value":"明天"},{"type":"date","value":"明天"}]}',
            "明天提醒我",
        )
        self.assertEqual(len(parsed or []), 2)
        self.assertEqual(validity, {"json_valid": True, "schema_valid": True,
                                    "copy_valid": True, "error": None})
        invalid, validity = parse_prediction('{"slots":[{"type":"date","value":"昨天"}]}',
                                             "明天提醒我")
        self.assertIsNone(invalid)
        self.assertEqual(validity["error"], "value_not_copied_from_utterance")
        self.assertTrue(validity["schema_valid"])
        invalid, validity = parse_prediction('```json', "明天提醒我")
        self.assertIsNone(invalid)
        self.assertFalse(validity["json_valid"])
        invalid, validity = parse_prediction('{"slots":[],"slots":[]}', "明天提醒我")
        self.assertIsNone(invalid)
        self.assertEqual(validity["error"], "duplicate_json_key")


class SplitSelectionTests(unittest.TestCase):
    @staticmethod
    def example(identifier: int, utterance: str, annotated: str) -> dict:
        return {
            "id": str(identifier),
            "locale": "zh-CN",
            "partition": "train",
            "utt": utterance,
            "annot_utt": annotated,
        }

    def make_source(self) -> tuple[list[dict], set[str], str]:
        rows: list[dict] = []
        old_keys: set[str] = set()
        identifier = 1
        for slot_type in TARGET_TYPES:
            for index in range(300):
                value = f"{slot_type}_{index}"
                rows.append(self.example(identifier, value, f"[{slot_type} : {value}]"))
                if index < 15:
                    old_keys.add(group_key(value))
                identifier += 1
        for index in range(250):
            value = f"negative_{index}"
            rows.append(self.example(identifier, value, value))
            identifier += 1
        conflict_key = group_key("conflict")
        rows.append(self.example(identifier, "conflict", "[date : conflict]"))
        rows.append(self.example(identifier + 1, "conflict", "conflict"))
        return rows, old_keys, conflict_key

    def test_fixed_quotas_group_isolation_and_old_sft_exclusion(self) -> None:
        rows, old_keys, conflict_key = self.make_source()
        selected, audit = select_splits(rows, old_keys, seed=42)
        self.assertEqual({split: len(items) for split, items in selected.items()},
                         {"train": 384, "dev": 100, "confirmation": 200})
        self.assertEqual(audit["conflict_groups_excluded"], 1)
        self.assertEqual(audit["conflict_rows_excluded"], 2)
        keys = {split: {item["group_key"] for item in items} for split, items in selected.items()}
        self.assertNotIn(conflict_key, set().union(*keys.values()))
        self.assertFalse(keys["train"] & keys["dev"])
        self.assertFalse(keys["train"] & keys["confirmation"])
        self.assertFalse(keys["dev"] & keys["confirmation"])
        self.assertFalse(keys["dev"] & old_keys)
        self.assertFalse(keys["confirmation"] & old_keys)
        for split, quotas in {
            "train": {"date": 80, "time": 80, "place_name": 80, "person": 80, "none": 64},
            "dev": {"date": 20, "time": 20, "place_name": 20, "person": 20, "none": 20},
            "confirmation": {"date": 40, "time": 40, "place_name": 40, "person": 40, "none": 40},
        }.items():
            self.assertEqual(Counter(item["bucket"] for item in selected[split]), quotas)

    def test_selection_is_repeatable(self) -> None:
        rows, old_keys, _ = self.make_source()
        first, _ = select_splits(rows, old_keys, seed=42)
        second, _ = select_splits(rows, old_keys, seed=42)
        self.assertEqual(
            {name: [item["row"]["id"] for item in items] for name, items in first.items()},
            {name: [item["row"]["id"] for item in items] for name, items in second.items()},
        )


if __name__ == "__main__":
    unittest.main()
