"""V2 source-natural splitting, leakage exclusion, and immutable output checks."""

from __future__ import annotations

import hashlib
import json
import sys
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import prepare_massive_slots_v2 as v2  # noqa: E402
import evaluate_massive_slots as evaluator  # noqa: E402
from evaluate_massive_slots import _validate_study_split, load_locked_split_manifest  # noqa: E402


def example(identifier: int, utterance: str, annotation: str, partition: str = "train") -> dict:
    return {
        "id": str(identifier), "locale": "zh-CN", "partition": partition,
        "utt": utterance, "annot_utt": annotation,
    }


class V2SelectionTests(unittest.TestCase):
    def make_source(self) -> list[dict]:
        rows: list[dict] = []
        identifier = 1
        for slot_type in v2.TARGET_TYPES:
            for index in range(80):
                value = f"{slot_type}{index}"
                rows.append(example(identifier, value, f"[{slot_type} : {value}]"))
                identifier += 1
        for prefix, annotated in (("hard", True), ("pure", False)):
            for index in range(200):
                value = f"{prefix}{index}"
                label = f"[song_name : {value}]" if annotated else value
                rows.append(example(identifier, value, label))
                identifier += 1
        rows.extend((
            example(identifier, "同一句，", "[date : 同一句，]"),
            example(identifier + 1, "同一句!", "同一句!"),
            example(identifier + 2, "旧样本？", "旧样本？"),
        ))
        return rows

    @staticmethod
    def small_plan() -> dict[str, int]:
        return {
            "dev": 35, "confirmation": 45, "positive_per_type": 10,
            "hard_negative": 15, "pure_negative": 15,
        }

    def test_strong_key_folds_width_case_space_punctuation_but_not_symbols(self) -> None:
        self.assertEqual(v2.strong_group_key("Ａ b，早安!"), v2.strong_group_key("ab早安"))
        self.assertNotEqual(v2.strong_group_key("¥5"), v2.strong_group_key("5"))

    def test_natural_holdouts_precede_quotas_and_all_groups_are_disjoint(self) -> None:
        rows = self.make_source()
        prior = {v2.strong_group_key("旧样本")}
        plan = self.small_plan()
        selected, audit = v2.select_splits(rows, prior, seed=17, plan=plan)
        self.assertEqual({name: len(items) for name, items in selected.items()},
                         {"train": 70, "dev": 35, "confirmation": 45})
        self.assertEqual(audit["conflict_groups_excluded"], 1)
        self.assertEqual(audit["prior_phrase_groups_excluded_from_train_source"], 1)
        self.assertEqual(audit["eligible_groups"], 720)
        keys = {name: {item["key"] for item in items} for name, items in selected.items()}
        self.assertEqual(len(set.union(*keys.values())), 150)
        self.assertFalse(set.union(*keys.values()) & prior)
        self.assertNotIn(v2.strong_group_key("同一句"), set.union(*keys.values()))
        self.assertEqual(Counter(item["bucket"] for item in selected["train"]), {
            "date": 10, "time": 10, "place_name": 10, "person": 10,
            "hard_negative": 15, "pure_negative": 15,
        })
        for item in selected["train"]:
            if item["bucket"] in ("hard_negative", "pure_negative"):
                self.assertFalse(item["slots"])
                self.assertEqual(item["negative_kind"], item["bucket"])

        natural = sorted(
            (v2.strong_group_key(row["utt"]) for row in rows[:-3]),
            key=lambda key: (v2._rank(17, "natural", key), key),
        )
        self.assertEqual(keys["confirmation"], set(natural[:45]))
        self.assertEqual(keys["dev"], set(natural[45:80]))
        self.assertTrue(all(item["bucket"] == "natural" for item in selected["dev"]))

        again, _ = v2.select_splits(list(reversed(rows)), prior, seed=17, plan=plan)
        self.assertEqual(
            {split: [item["row"]["id"] for item in items] for split, items in selected.items()},
            {split: [item["row"]["id"] for item in items] for split, items in again.items()},
        )

    def test_holdout_selection_does_not_use_labels(self) -> None:
        rows = self.make_source()
        altered = [dict(row) for row in rows]
        for row in altered[:8]:
            row["annot_utt"] = row["utt"]
        plan = self.small_plan()
        before, _ = v2.select_splits(rows, set(), seed=17, plan=plan)
        after, _ = v2.select_splits(altered, set(), seed=17, plan=plan)
        for split in ("dev", "confirmation"):
            self.assertEqual(
                [item["row"]["id"] for item in before[split]],
                [item["row"]["id"] for item in after[split]],
            )


class V2PriorAndManifestTests(unittest.TestCase):
    def test_prior_files_are_verified_and_every_old_split_is_excluded(self) -> None:
        old_intent = Path("old-intent.jsonl")
        v1_dir = Path("slots")
        rows = [
            example(1, "旧意图", "旧意图"),
            example(2, "旧槽训练，", "旧槽训练，"),
            example(3, "旧槽开发", "旧槽开发"),
            example(4, "旧槽确认", "旧槽确认"),
            example(5, "剩余", "剩余"),
        ]
        rows += [example(6, "官方开发!", "官方开发!", "dev"),
                 example(7, "官方测试", "官方测试", "test")]
        file_rows = {"old-intent.jsonl": [{"id": "1", "utt": "旧意图"}]}
        manifest = {
            "format_version": 1, "source_sha256": v2.SOURCE_SHA256,
            "source_partition_used": "train", "splits": {},
            "old_sft_train_file_sha256": "hash-intent",
        }
        for split, identifier in (("train", 2), ("dev", 3), ("confirmation", 4)):
            row = rows[identifier - 1]
            file_rows[f"{split}.jsonl"] = [{"id": row["id"], "utt": row["utt"]}]
            manifest["splits"][split] = {
                "examples": 1, "source_ids": [row["id"]],
                "jsonl_sha256": f"hash-{split}",
            }

        observed_hashes = {
            "old-intent.jsonl": "hash-intent", "manifest.json": "hash-manifest",
            **{f"{split}.jsonl": f"hash-{split}" for split in ("train", "dev", "confirmation")},
        }

        def run_loader() -> tuple[set[str], dict]:
            with (
                patch.object(v2, "SOURCE_PARTITION_SIZES", {"train": 5, "dev": 1, "test": 1}),
                patch.object(Path, "is_file", return_value=True),
                patch.object(Path, "read_text", return_value=json.dumps(manifest)),
                patch.object(v2, "read_jsonl", side_effect=lambda path: file_rows[path.name]),
                patch.object(v2, "file_sha256", side_effect=lambda path: observed_hashes[path.name]),
            ):
                return v2.load_prior_keys(
                    rows, old_intent, v1_dir,
                    expected_old_sizes={"train": 1, "dev": 1, "confirmation": 1},
                    expected_intent_size=1,
                    expected_v1_manifest_sha256="hash-manifest",
                )

        keys, audit = run_loader()
        self.assertEqual(len(keys), 6)
        self.assertEqual(audit["union_excluded_phrase_groups"], 6)
        for phrase in ("旧槽训练", "旧槽开发", "旧槽确认", "官方开发", "官方测试"):
            self.assertIn(v2.strong_group_key(phrase), keys)
        observed_hashes["confirmation.jsonl"] = "tampered"
        with self.assertRaisesRegex(ValueError, "locked manifest"):
            run_loader()

    def test_locked_files_refuse_changes_and_eval_manifest_accepts_v2_shape(self) -> None:
        root = Path("locked-study")
        selected = {
            "train": [{"key": "train", "row": example(1, "今天", "[date : 今天]"),
                       "slots": [{"type": "date", "value": "今天"}], "bucket": "date",
                       "source_ids": ["1"]}],
            "dev": [{"key": "dev", "row": example(2, "明天", "[date : 明天]"),
                     "slots": [{"type": "date", "value": "明天"}], "bucket": "natural",
                     "source_ids": ["2"]}],
        }
        contents = {}
        manifest = {"format_version": 1, "source_partition_used": "train",
                    "source_sha256": v2.SOURCE_SHA256, "splits": {}}
        records_by_split = {}
        for split, items in selected.items():
            records = [v2.make_record(item) for item in items]
            records_by_split[split] = records
            payload = v2._jsonl_bytes(records)
            contents[f"{split}.jsonl"] = payload
            manifest["splits"][split] = {
                "examples": 1, "jsonl_sha256": hashlib.sha256(payload).hexdigest(),
                "source_ids": [records[0]["id"]],
                "groups": [{"group_sha256": records[0]["group_sha256"],
                            "source_ids": items[0]["source_ids"]}],
            }
        contents["manifest.json"] = json.dumps(manifest).encode()
        disk: dict[str, bytes] = {}
        with (
            patch.object(Path, "exists", new=lambda path: path.name in disk),
            patch.object(Path, "read_bytes", new=lambda path: disk[path.name]),
            patch.object(Path, "write_bytes", new=lambda path, data: disk.setdefault(path.name, data)),
            patch.object(Path, "mkdir"),
        ):
            v2.write_locked_files(root, contents)
            v2.write_locked_files(root, contents)
            with self.assertRaisesRegex(ValueError, "Refusing to overwrite"):
                v2.write_locked_files(root, {"dev.jsonl": b"different\n"})
        self.assertEqual(disk["dev.jsonl"], contents["dev.jsonl"])

        path = root / "dev.jsonl"
        with (
            patch.object(Path, "is_file", return_value=True),
            patch.object(Path, "read_text", return_value=json.dumps(manifest)),
            patch.object(evaluator, "sha256_file", side_effect=lambda path: manifest["splits"][path.stem]["jsonl_sha256"]),
            patch.object(evaluator, "read_jsonl", side_effect=lambda path: records_by_split[path.stem]),
        ):
            loaded = load_locked_split_manifest(path, "dev")
            _validate_study_split(records_by_split["dev"], path, "dev", loaded)


if __name__ == "__main__":
    unittest.main()
