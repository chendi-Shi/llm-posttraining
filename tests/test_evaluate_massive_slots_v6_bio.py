"""Guard the v6 BIO development control against test access and split drift."""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import sys
import unittest
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import joblib


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import evaluate_massive_slots_v6_bio as v6  # noqa: E402


def jsonl_bytes(rows: list[dict]) -> bytes:
    return b"".join(
        (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
        for row in rows
    )


@contextmanager
def workspace_scratch():
    local_tmp = Path(__file__).resolve().parents[1] / "_tmp"
    local_tmp.mkdir(exist_ok=True)
    root = local_tmp / f"v6-bio-test-{uuid4().hex}"
    root.mkdir()
    try:
        yield root
    finally:
        if not root.resolve().is_relative_to(local_tmp.resolve()):
            raise ValueError("Test scratch path escaped the workspace")
        shutil.rmtree(root)


class V6BioDevTests(unittest.TestCase):
    def test_manifest_rejects_overlap_with_sealed_test_using_metadata_only(self) -> None:
        source = "a" * 64
        manifest = {
            "format_version": 1, "source_partition_used": "train",
            "source_sha256": source, "seed": v6.SEED,
            "splits": {
                role: {"examples": 1, "jsonl_sha256": "b" * 64,
                       "group_sha256": [group]}
                for role, group in (("train", "one"), ("dev", "two"), ("test", "three"))
            },
        }
        with patch.object(v6, "SOURCE_SHA256", source), \
             patch.object(v6, "EXPECTED_COUNTS", {"train": 1, "dev": 1, "test": 1}), \
             patch.object(v6, "LOCKED_SPLIT_SHA256", {role: "b" * 64 for role in ("train", "dev", "test")}):
            v6.validate_manifest(manifest)
            manifest["splits"]["test"]["jsonl_sha256"] = "d" * 64
            with self.assertRaisesRegex(ValueError, "preregistered"):
                v6.validate_manifest(manifest)
            manifest["splits"]["test"]["jsonl_sha256"] = "b" * 64
            manifest["splits"]["test"]["group_sha256"] = ["two"]
            with self.assertRaisesRegex(ValueError, "repeat"):
                v6.validate_manifest(manifest)

    def test_changed_manifest_rejected_before_split_rows_are_read(self) -> None:
        with workspace_scratch() as root:
            data = root / "data"
            data.mkdir()
            (data / "manifest.json").write_text("{}", encoding="utf-8")
            with patch.object(v6, "ROOT", root), \
                 patch.object(v6, "read_jsonl", side_effect=AssertionError("split read")):
                with self.assertRaisesRegex(ValueError, "preregistered"):
                    v6.main(["--data-dir", str(data),
                             "--model", str(root / "model.joblib"),
                             "--report", str(root / "report.json"),
                             "--predictions", str(root / "_tmp" / "predictions.jsonl")])

    def test_test_role_is_rejected_before_any_file_read(self) -> None:
        with patch.object(Path, "open", side_effect=AssertionError("opened")), \
             redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                v6.main(["--role", "test"])

    def test_development_run_never_opens_test_file_and_records_provenance(self) -> None:
        with workspace_scratch() as root:
            data = root / "data"
            data.mkdir()
            archive = root / "source.tar.gz"
            archive.write_bytes(b"fixture archive")
            source_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
            model = root / "outputs" / "bio.joblib"
            report = root / "reports" / "bio-dev.json"
            predictions = root / "_tmp" / "bio-dev-predictions.jsonl"
            test_path = data / "test.jsonl"
            test_path.write_text("sealed test data", encoding="utf-8")

            train_specs = [
                ("1", "明天", "[date:明天]", [{"type": "date", "value": "明天"}]),
                ("2", "九点", "[time:九点]", [{"type": "time", "value": "九点"}]),
                ("3", "北京", "[place_name:北京]", [{"type": "place_name", "value": "北京"}]),
                ("4", "小王", "[person:小王]", [{"type": "person", "value": "小王"}]),
            ]
            dev_specs = [
                ("5", "后天", "[date:后天]", [{"type": "date", "value": "后天"}]),
                ("6", "天气", "天气", []),
            ]
            source_rows = []
            splits: dict[str, dict] = {}
            for role, specs in (("train", train_specs), ("dev", dev_specs)):
                rows = []
                for identifier, utterance, annotated, slots in specs:
                    source_rows.append({"id": identifier, "partition": "train",
                                        "utt": utterance, "annot_utt": annotated})
                    rows.append({"id": identifier, "source_partition": "train",
                                 "utt": utterance, "slots": slots,
                                 "group_sha256": f"group-{identifier}"})
                content = jsonl_bytes(rows)
                (data / f"{role}.jsonl").write_bytes(content)
                splits[role] = {
                    "examples": len(rows),
                    "jsonl_sha256": hashlib.sha256(content).hexdigest(),
                    "group_sha256": [row["group_sha256"] for row in rows],
                }
            splits["test"] = {"examples": 3, "jsonl_sha256": "c" * 64,
                              "group_sha256": ["test-1", "test-2", "test-3"]}
            manifest = {
                "format_version": 1, "source_partition_used": "train",
                "source_sha256": source_sha, "seed": v6.SEED, "splits": splits,
            }
            manifest_path = data / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            original_open = Path.open

            def guard_test_open(path: Path, *args: object, **kwargs: object):
                if path.resolve() == test_path.resolve():
                    raise AssertionError("v6 test file was opened")
                return original_open(path, *args, **kwargs)

            with patch.object(v6, "ROOT", root), \
                 patch.object(v6, "SOURCE_SHA256", source_sha), \
                 patch.object(v6, "EXPECTED_COUNTS", {"train": 4, "dev": 2, "test": 3}), \
                 patch.object(v6, "LOCKED_MANIFEST_SHA256", hashlib.sha256(manifest_path.read_bytes()).hexdigest()), \
                 patch.object(v6, "LOCKED_SPLIT_SHA256", {role: split["jsonl_sha256"] for role, split in splits.items()}), \
                 patch.object(v6, "read_archive", return_value=(source_rows, b"license")), \
                 patch.object(v6, "train_model", return_value=("vectorizer", "model")), \
                 patch.object(v6, "predict_slots", return_value=[]), \
                 patch.object(Path, "open", guard_test_open), \
                 redirect_stdout(io.StringIO()):
                v6.main(["--data-dir", str(data), "--archive", str(archive),
                         "--model", str(model), "--report", str(report),
                         "--predictions", str(predictions)])

            bundle = joblib.load(model)
            saved_report = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(bundle["configuration"], v6.MODEL_CONFIG)
            self.assertEqual(bundle["train_file_sha256"], splits["train"]["jsonl_sha256"])
            self.assertEqual(bundle["manifest_sha256"], hashlib.sha256(manifest_path.read_bytes()).hexdigest())
            self.assertEqual(bundle["source_sha256"], source_sha)
            self.assertEqual(saved_report["role"], "dev")
            self.assertFalse(saved_report["test_used"])
            self.assertEqual(saved_report["metrics"]["examples"], 2)
            self.assertEqual(len(predictions.read_text(encoding="utf-8").splitlines()), 2)


if __name__ == "__main__":
    unittest.main()
