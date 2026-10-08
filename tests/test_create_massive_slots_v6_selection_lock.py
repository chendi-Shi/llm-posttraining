"""The v6 lock maker may use dev evidence, never test examples."""

from __future__ import annotations

import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import create_massive_slots_v6_selection_lock as creator  # noqa: E402


def evidence() -> tuple[dict, dict, dict, dict, dict]:
    v6 = creator.v6
    manifest = {"splits": {
        role: {"jsonl_sha256": digest}
        for role, digest in v6.EXPECTED_FILE_SHA256.items()
    }}
    models = {
        "old_bio": v6.OLD_BIO_SHA256,
        "new_bio": "a" * 64,
        "dpo_inference_files": {
            "model.safetensors": v6.BASE_SHA256,
            "tokenizer.json": v6.TOKENIZER_SHA256,
            "adapter_model.safetensors": v6.DPO_SHA256,
        },
    }
    code = {name: "b" * 64 for name in v6.CODE_FILES}
    versions = {name: "1.0" for name in v6.PACKAGE_NAMES}
    metrics = {"micro_f1": .60, "no_slot_failure_rate": .20,
               "json_valid_rate": .95, "schema_valid_rate": .95,
               "copy_valid_rate": .95}
    positive = {"micro_f1": .60}
    dev_report = {
        "role": "dev",
        "manifest_sha256": v6.EXPECTED_MANIFEST_SHA256,
        "train_file_sha256": v6.EXPECTED_FILE_SHA256["train"],
        "eval_file_sha256": v6.EXPECTED_FILE_SHA256["dev"],
        "model_sha256": models,
        "code_sha256": code,
        "package_versions": versions,
        "confidence_threshold": v6.MEAN_LOGPROB_THRESHOLD,
        "generation": {**v6.GENERATION, "max_observed_input_tokens": 100},
        "bootstrap": v6.BOOTSTRAP["dev"],
        "systems": {"new_hybrid": {"metrics": metrics, "positive_only": positive}},
        "development_gate": v6.development_gate(metrics, positive),
    }
    return manifest, dev_report, models, code, versions


class V6SelectionLockTests(unittest.TestCase):
    def test_passing_dev_creates_exact_frozen_selection_fields(self) -> None:
        manifest, report, models, code, versions = evidence()
        lock = creator.build_lock(
            manifest=manifest, manifest_sha256=creator.v6.EXPECTED_MANIFEST_SHA256,
            dev_report=report, dev_report_sha256="c" * 64,
            models=models, code_sha256=code, versions=versions,
        )
        self.assertEqual(lock["status"], "selected_for_one_test")
        self.assertEqual(lock["primary_candidate"], "new_hybrid")
        self.assertEqual(lock["primary_reference"], "old_v5_hybrid")
        self.assertEqual(lock["test_file_sha256"], creator.v6.EXPECTED_FILE_SHA256["test"])
        self.assertEqual(lock["code_sha256"], code)
        self.assertEqual(lock["package_versions"], versions)
        self.assertEqual(lock["bootstrap"], creator.v6.BOOTSTRAP["test"])

    def test_composed_lock_passes_evaluator_verifier_without_test_data(self) -> None:
        manifest, report, models, code, versions = evidence()
        v6 = creator.v6
        lock = creator.build_lock(
            manifest=manifest, manifest_sha256=v6.EXPECTED_MANIFEST_SHA256,
            dev_report=report, dev_report_sha256="c" * 64,
            models=models, code_sha256=code, versions=versions,
        )
        lock_path = Path("synthetic-v6-selection-lock.json")
        dev_path = Path("synthetic-v6-dev-report.json")
        data_dir = Path("synthetic-v6-data")
        hashed: list[Path] = []

        def fake_sha(path: Path) -> str:
            hashed.append(path)
            if path.name == "test.jsonl":
                raise AssertionError("test JSONL hashed")
            if path == lock_path:
                return "f" * 64
            if path == dev_path:
                return "c" * 64
            if path == data_dir / "manifest.json":
                return v6.EXPECTED_MANIFEST_SHA256
            if path.is_relative_to(v6.ROOT):
                return code[path.relative_to(v6.ROOT).as_posix()]
            raise AssertionError(f"Unexpected hashed path: {path}")

        def fake_read(path: Path, *args: object, **kwargs: object) -> str:
            if path == lock_path:
                return json.dumps(lock)
            if path == dev_path:
                return json.dumps(report)
            raise AssertionError(f"Unexpected text read: {path}")

        with patch.object(v6, "LOCK", lock_path), \
             patch.object(v6, "DEV_REPORT", dev_path), \
             patch.object(v6, "DATA", data_dir), \
             patch.object(v6, "sha256_file", side_effect=fake_sha), \
             patch.object(v6, "model_hashes", return_value=models), \
             patch.object(v6, "package_versions", return_value=versions), \
             patch.object(Path, "read_text", fake_read):
            self.assertEqual(v6.require_test_lock("f" * 64), lock)
        self.assertNotIn(data_dir / "test.jsonl", hashed)

    def test_failed_or_forged_dev_gate_cannot_make_lock(self) -> None:
        manifest, report, models, code, versions = evidence()
        report["systems"]["new_hybrid"]["metrics"]["micro_f1"] = .599
        for claimed in (True, False):
            report["development_gate"]["passed"] = claimed
            with self.subTest(claimed=claimed), self.assertRaisesRegex(
                    ValueError, "did not pass"):
                creator.build_lock(
                    manifest=manifest, manifest_sha256=creator.v6.EXPECTED_MANIFEST_SHA256,
                    dev_report=report, dev_report_sha256="c" * 64,
                    models=models, code_sha256=code, versions=versions,
                )

    def test_changed_model_code_or_manifest_rejected(self) -> None:
        manifest, report, models, code, versions = evidence()
        for field, bad in (("manifest_sha256", "f" * 64),
                           ("models", {**models, "new_bio": "f" * 64}),
                           ("code_sha256", {**code, "scripts/other.py": "f" * 64})):
            with self.subTest(field=field), self.assertRaises(ValueError):
                creator.build_lock(
                    manifest=manifest,
                    manifest_sha256=bad if field == "manifest_sha256"
                    else creator.v6.EXPECTED_MANIFEST_SHA256,
                    dev_report=report, dev_report_sha256="c" * 64,
                    models=bad if field == "models" else models,
                    code_sha256=bad if field == "code_sha256" else code,
                    versions=versions,
                )

    def test_existing_lock_is_rejected_before_any_dev_or_test_read(self) -> None:
        with patch.object(Path, "exists", return_value=True), \
             patch.object(Path, "open", side_effect=AssertionError("file opened")):
            with self.assertRaisesRegex(ValueError, "already exists"):
                creator.create_lock()

    def test_lock_creation_uses_only_preregistered_test_hash(self) -> None:
        manifest, report, models, code, versions = evidence()
        report["prediction_file_sha256"] = {
            **{name: "d" * 64 for name in creator.v6.SYSTEMS},
            "confidence": "d" * 64,
        }
        folder = Path("ignored-v6-lock-test")
        lock_path = folder / "selection-lock.json"
        dev_path = folder / "dev-report.json"
        manifest_path = folder / "manifest.json"
        hashed: list[Path] = []

        class MemoryStream(io.StringIO):
            def close(self) -> None:
                pass

        written = MemoryStream()

        def fake_sha(path: Path) -> str:
            hashed.append(path)
            if path.name == "test.jsonl":
                raise AssertionError("v6 test data was opened")
            if path == manifest_path:
                return creator.v6.EXPECTED_MANIFEST_SHA256
            if path == dev_path:
                return "c" * 64
            if path in (folder / "train.jsonl", folder / "dev.jsonl"):
                return creator.v6.EXPECTED_FILE_SHA256[path.stem]
            if path == lock_path:
                return "e" * 64
            if path.name.endswith("predictions.jsonl") or path.name.endswith("confidence.jsonl"):
                return "d" * 64
            if path.is_relative_to(creator.v6.ROOT):
                return code[path.relative_to(creator.v6.ROOT).as_posix()]
            raise AssertionError(f"Unexpected hashed path: {path}")

        def fake_read(path: Path, *args: object, **kwargs: object) -> str:
            if path == manifest_path:
                return json.dumps({**manifest,
                    "test_status": "sealed; no model inference or score"})
            if path == dev_path:
                return json.dumps(report)
            raise AssertionError(f"Unexpected text read: {path}")

        def fake_open(path: Path, mode: str, **kwargs: object) -> MemoryStream:
            if path != lock_path or mode != "x":
                raise AssertionError(f"Unexpected file opened: {path}")
            return written

        with patch.object(creator.v6, "LOCK", lock_path), \
             patch.object(creator.v6, "DEV_REPORT", dev_path), \
             patch.object(creator.v6, "DATA", folder), \
             patch.object(creator.v6, "sha256_file", side_effect=fake_sha), \
             patch.object(creator.v6, "validate_manifest"), \
             patch.object(creator.v6, "check_output_paths"), \
             patch.object(creator.v6, "check_confidence_assets"), \
             patch.object(creator.v6, "model_hashes", return_value=models), \
             patch.object(creator.v6, "load_bio"), \
             patch.object(creator.v6, "package_versions", return_value=versions), \
             patch.object(creator.v6, "require_test_lock",
                          side_effect=lambda _digest: json.loads(written.getvalue())), \
             patch.object(Path, "exists", return_value=False), \
             patch.object(Path, "is_file", return_value=True), \
             patch.object(Path, "mkdir"), \
             patch.object(Path, "read_text", fake_read), \
             patch.object(Path, "open", autospec=True, side_effect=fake_open):
            lock, digest = creator.create_lock()
        self.assertEqual(lock["test_file_sha256"], creator.v6.EXPECTED_FILE_SHA256["test"])
        self.assertEqual(digest, "e" * 64)
        self.assertNotIn(folder / "test.jsonl", hashed)


if __name__ == "__main__":
    unittest.main()
