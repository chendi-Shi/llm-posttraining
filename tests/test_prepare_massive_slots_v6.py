"""V6 provenance, unseen-group isolation, and sealed-file checks."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_massive_slots_v6 as v6  # noqa: E402
from prepare_massive_slots_v2 import group_sha256, strong_group_key  # noqa: E402

LOCAL_TMP = Path(__file__).resolve().parents[1] / "_tmp"


@contextmanager
def local_test_dir():
    """Avoid Windows tempfile ACLs that deny nested sandbox writes."""
    root = LOCAL_TMP / f"test-v6-{uuid.uuid4().hex}"
    root.mkdir()
    try:
        yield root
    finally:
        if not root.resolve().is_relative_to(LOCAL_TMP.resolve()):
            raise ValueError("Test cleanup target escaped the workspace")
        shutil.rmtree(root)


def source_row(identifier: int, utterance: str, annotation: str) -> dict:
    return {
        "id": str(identifier), "locale": "zh-CN", "partition": "train",
        "utt": utterance, "annot_utt": annotation,
    }


class V6PreparationTests(unittest.TestCase):
    def test_fresh_checkout_rejects_a_different_preregistered_split(self) -> None:
        contents = {
            "train.jsonl": b"train\n", "dev.jsonl": b"dev\n",
            "test.jsonl": b"sealed-test\n", "manifest.json": b"manifest\n",
        }
        expected = {name: hashlib.sha256(payload).hexdigest()
                    for name, payload in contents.items()}
        with patch.object(v6, "FROZEN_OUTPUT_SHA256", expected):
            v6.assert_frozen_output_hashes(contents)
            with self.assertRaisesRegex(ValueError, "test.jsonl differs"):
                v6.assert_frozen_output_hashes({**contents, "test.jsonl": b"changed\n"})
            with self.assertRaisesRegex(ValueError, "Missing a preregistered"):
                v6.assert_frozen_output_hashes({name: payload for name, payload in contents.items()
                                                if name != "manifest.json"})

    def test_v5_manifest_and_payload_must_match_frozen_hashes(self) -> None:
        with local_test_dir() as root:
            test_path = root / "test.jsonl"
            manifest_path = root / "manifest.json"
            test_path.write_bytes(b"sealed test bytes\n")
            test_sha = hashlib.sha256(test_path.read_bytes()).hexdigest()
            groups = [hashlib.sha256(str(index).encode()).hexdigest() for index in range(600)]
            manifest = {
                "source_sha256": v6.SOURCE_SHA256,
                "v2_manifest_sha256": v6.V2_MANIFEST_SHA256,
                "v3_manifest_sha256": v6.V3_MANIFEST_SHA256,
                "v4_manifest_sha256": v6.V4_MANIFEST_SHA256,
                "test_examples": 600,
                "test_jsonl_sha256": test_sha,
                "test_group_sha256": groups,
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            manifest_sha = v6.file_sha256(manifest_path)
            with (
                patch.object(v6, "V5_MANIFEST_SHA256", manifest_sha),
                patch.object(v6, "V5_TEST_SHA256", test_sha),
            ):
                self.assertEqual(v6.reserved_v5_group_hashes(manifest_path, test_path), set(groups))
                test_path.write_bytes(b"changed\n")
                with self.assertRaisesRegex(ValueError, "test file differs"):
                    v6.reserved_v5_group_hashes(manifest_path, test_path)
                test_path.write_bytes(b"sealed test bytes\n")
                manifest["test_group_sha256"][1] = manifest["test_group_sha256"][0]
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "manifest differs"):
                    v6.reserved_v5_group_hashes(manifest_path, test_path)

    def test_all_prior_groups_excluded_and_outputs_immutable(self) -> None:
        rows: list[dict] = []
        identifier = 1
        for slot_type in ("date", "time", "place_name", "person"):
            for index in range(20):
                value = f"{slot_type}{index}"
                rows.append(source_row(identifier, value, f"[{slot_type} : {value}]"))
                identifier += 1
        for index in range(30):
            value = f"hard{index}"
            rows.append(source_row(identifier, value, f"[song_name : {value}]"))
            identifier += 1
        for index in range(30):
            value = f"pure{index}"
            rows.append(source_row(identifier, value, value))
            identifier += 1
        prior = {strong_group_key(rows[0]["utt"])}
        reserved = {
            "v2": {group_sha256(strong_group_key(rows[1]["utt"]))},
            "v3": {group_sha256(strong_group_key(rows[2]["utt"]))},
            "v4": {group_sha256(strong_group_key(rows[3]["utt"]))},
            "v5": {group_sha256(strong_group_key(rows[4]["utt"]))},
        }
        plan = {"confirmation": 10, "dev": 10, "positive_per_type": 3,
                "hard_negative": 4, "pure_negative": 4}
        train_buckets = {name: 3 for name in ("date", "time", "place_name", "person")}
        train_buckets.update({"hard_negative": 4, "pure_negative": 4})
        source = Path("synthetic-source.tar.gz")

        with local_test_dir() as root:
            output = root / "v6"
            with (
                patch.object(v6, "file_sha256", side_effect=lambda path: v6.SOURCE_SHA256 if path == source else "unused"),
                patch.object(v6, "read_archive", return_value=(rows, b"license")),
                patch.object(v6, "SOURCE_PARTITION_SIZES", {"train": len(rows)}),
                patch.object(v6, "load_prior_keys", return_value=(prior.copy(), {"synthetic": True})),
                patch.object(v6, "reserved_v2_group_hashes", return_value=reserved["v2"]),
                patch.object(v6, "reserved_v3_group_hashes", return_value=reserved["v3"]),
                patch.object(v6, "reserved_v4_group_hashes", return_value=reserved["v4"]),
                patch.object(v6, "reserved_v5_group_hashes", return_value=reserved["v5"]),
                patch.object(v6, "PLAN", plan),
                patch.object(v6, "EXPECTED_TRAIN_BUCKETS", train_buckets),
                patch.object(v6, "EXPECTED_SPLIT_SIZES", {"train": 20, "dev": 10, "test": 10}),
                patch.object(v6, "assert_frozen_output_hashes") as frozen_check,
            ):
                def run() -> dict:
                    return v6.prepare(
                        source, Path("old-intent"), Path("v1"), Path("v2"), Path("v3"),
                        Path("v4"), Path("v5"), Path("v5-test"), output,
                    )

                manifest = run()
                self.assertEqual({role: manifest["splits"][role]["examples"]
                                  for role in ("train", "dev", "test")},
                                 {"train": 20, "dev": 10, "test": 10})
                self.assertNotIn("positive_examples", manifest["splits"]["test"])
                seen = set()
                for role in ("train", "dev", "test"):
                    entry = manifest["splits"][role]
                    self.assertEqual(len(entry["group_sha256"]), entry["examples"])
                    self.assertEqual(
                        hashlib.sha256((output / f"{role}.jsonl").read_bytes()).hexdigest(),
                        entry["jsonl_sha256"],
                    )
                    self.assertNotIn("source_ids", entry)
                    self.assertNotIn("groups", entry)
                    current = set(entry["group_sha256"])
                    self.assertFalse(seen & current)
                    seen.update(current)
                self.assertFalse(seen & set.union(*reserved.values()))
                self.assertNotIn(group_sha256(next(iter(prior))), seen)
                self.assertEqual(run(), manifest)
                (output / "dev.jsonl").write_bytes(b"tampered\n")
                with self.assertRaisesRegex(ValueError, "Refusing to overwrite"):
                    run()
                self.assertEqual(frozen_check.call_count, 3)


if __name__ == "__main__":
    unittest.main()
