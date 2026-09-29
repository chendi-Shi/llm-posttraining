"""Safety and provenance checks for the v2 Qwen evaluation entry point."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import evaluate_massive_slots_v2 as v2  # noqa: E402


MODEL_HASHES = {
    "model.safetensors": "b" * 64,
    "tokenizer.json": "c" * 64,
    "adapter_model.safetensors": "a" * 64,
}


class V2QwenGateTests(unittest.TestCase):
    def test_confirmation_without_unlock_does_not_read_any_file(self) -> None:
        with (
            patch.object(Path, "open", side_effect=AssertionError("file opened")) as opener,
            patch.object(Path, "read_text", side_effect=AssertionError("file read")) as reader,
            patch.object(v2, "sha256_file", side_effect=AssertionError("file hashed")) as hasher,
        ):
            with self.assertRaisesRegex(ValueError, "--unlock-confirmation"):
                v2.main(["--system", "v2_step80", "--role", "confirmation"])
        opener.assert_not_called()
        reader.assert_not_called()
        hasher.assert_not_called()

    def test_confirmation_without_lock_sha_does_not_read_any_file(self) -> None:
        with (
            patch.object(Path, "open", side_effect=AssertionError("file opened")) as opener,
            patch.object(Path, "read_text", side_effect=AssertionError("file read")) as reader,
            patch.object(v2, "sha256_file", side_effect=AssertionError("file hashed")) as hasher,
        ):
            with self.assertRaisesRegex(ValueError, "--expected-selection-lock-sha256"):
                v2.main(["--system", "v2_step80", "--role", "confirmation",
                         "--unlock-confirmation"])
        opener.assert_not_called()
        reader.assert_not_called()
        hasher.assert_not_called()

    def test_confirmation_requires_frozen_lock_before_utterance_read(self) -> None:
        manifest = {"splits": {"dev": {"jsonl_sha256": "d" * 64},
                               "confirmation": {"jsonl_sha256": "c" * 64}}}
        with (
            patch.object(v2, "load_locked_split_manifest", return_value=manifest),
            patch.object(v2, "check_manifest_preflight"),
            patch.object(v2, "sha256_file", return_value=v2.SOURCE_SHA256),
            patch.object(v2, "model_file_hashes", return_value=MODEL_HASHES),
            patch.object(v2, "verify_selection_lock", side_effect=ValueError("wrong lock")),
            patch.object(v2, "read_jsonl") as row_reader,
            patch.object(Path, "exists", return_value=False),
        ):
            with self.assertRaisesRegex(ValueError, "wrong lock"):
                v2.main(["--system", "v2_step80", "--role", "confirmation",
                         "--unlock-confirmation", "--expected-selection-lock-sha256", "e" * 64])
        row_reader.assert_not_called()

    def test_selected_v2_and_frozen_v1_are_the_only_confirmable_adapters(self) -> None:
        adapter80 = v2.SYSTEM_ADAPTER["v2_step80"]
        adapter160 = v2.SYSTEM_ADAPTER["v2_step160"]
        old_adapter = v2.SYSTEM_ADAPTER["v1"]
        lock = {
            "status": "selected", "selected_step": 80,
            "manifest_sha256": "m" * 64, "confirmation_file_sha256": "c" * 64,
            "dev_file_sha256": "d" * 64,
            "base_model_sha256": "b" * 64, "tokenizer_sha256": "c" * 64,
            "base_model_files_sha256": {
                "model.safetensors": "b" * 64, "tokenizer.json": "c" * 64,
            },
            "selected_adapter_sha256": "a" * 64,
            "selected_adapter_files_sha256": {"adapter_model.safetensors": "a" * 64},
            "selected_sft_checkpoint": str(adapter80),
            "reference_systems": {"v1": {"adapter_sha256": "f" * 64,
                                         "adapter_files_sha256": {"adapter_model.safetensors": "f" * 64},
                                         "adapter_path": str(old_adapter)}},
            "code_sha256": {name: "e" * 64 for name in v2.CODE_FILES},
            "generation": dict(v2.GENERATION),
        }
        manifest = {"splits": {"dev": {"jsonl_sha256": "d" * 64},
                               "confirmation": {"jsonl_sha256": "c" * 64}}}
        args = argparse.Namespace(
            expected_selection_lock_sha256="1" * 64,
            selection_lock=Path("lock.json"), system="v2_step80",
        )

        def file_hash(path: Path) -> str:
            return "1" * 64 if path == args.selection_lock else "m" * 64 if path.name == "manifest.json" else "e" * 64

        with (
            patch.object(v2, "sha256_file", side_effect=file_hash),
            patch.object(Path, "read_text", return_value=json.dumps(lock)),
        ):
            self.assertEqual(
                v2.verify_selection_lock(args, manifest, Path("manifest.json"), MODEL_HASHES, adapter80),
                lock,
            )
            args.system = "v2_step160"
            with self.assertRaisesRegex(ValueError, "selected v2"):
                v2.verify_selection_lock(args, manifest, Path("manifest.json"), MODEL_HASHES, adapter160)
            args.system = "v1"
            old_hashes = {**MODEL_HASHES, "adapter_model.safetensors": "f" * 64}
            self.assertEqual(
                v2.verify_selection_lock(args, manifest, Path("manifest.json"), old_hashes, old_adapter),
                lock,
            )
            with self.assertRaisesRegex(ValueError, "adapter differs"):
                v2.verify_selection_lock(args, manifest, Path("manifest.json"), MODEL_HASHES, old_adapter)

    def test_model_replacement_aborts_before_scoring_or_writing(self) -> None:
        row = {"id": "1", "utt": "你好", "slots": [], "group_sha256": "g"}
        with (
            patch.object(v2, "load_locked_split_manifest", return_value={"splits": {}}),
            patch.object(v2, "check_manifest_preflight"),
            patch.object(v2, "sha256_file", return_value=v2.SOURCE_SHA256),
            patch.object(v2, "model_file_hashes", side_effect=[
                MODEL_HASHES, {**MODEL_HASHES, "adapter_model.safetensors": "z" * 64},
            ]),
            patch.object(v2, "read_jsonl", return_value=[row]),
            patch.object(v2, "_validate_study_split"),
            patch.object(v2, "generate_raw_predictions", return_value=(["{\"slots\":[]}"], {"max_observed_input_tokens": 90})),
            patch.object(v2, "score_predictions") as scorer,
            patch.object(v2, "write_predictions") as writer,
        ):
            with self.assertRaisesRegex(ValueError, "changed during"):
                v2.main(["--system", "v2_step80", "--role", "dev"])
        scorer.assert_not_called()
        writer.assert_not_called()

    def test_confirmation_cannot_change_output_paths(self) -> None:
        with patch.object(v2, "load_locked_split_manifest") as loader:
            with self.assertRaises(SystemExit):
                v2.main(["--system", "v2_step80", "--role", "confirmation",
                         "--unlock-confirmation", "--expected-selection-lock-sha256", "e" * 64,
                         "--metrics", "reports/second-confirmation.json"])
        loader.assert_not_called()

    def test_dev_report_records_exact_prediction_file_hash_and_generation(self) -> None:
        row = {"id": "1", "utt": "你好", "slots": [], "group_sha256": "g"}
        disk: dict[str, bytes] = {}
        report_text: dict[str, str] = {}

        def file_hash(path: Path) -> str:
            if path.name in disk:
                return hashlib.sha256(disk[path.name]).hexdigest()
            if path.name == "dev.jsonl":
                return "e" * 64
            if path.name == "manifest.json":
                return "m" * 64
            return v2.SOURCE_SHA256

        def write_bytes(path: Path, content: bytes) -> int:
            disk[path.name] = content
            return len(content)

        def write_text(path: Path, content: str, encoding: str = "utf-8") -> int:
            report_text[path.name] = content
            return len(content)

        with (
            patch.object(v2, "load_locked_split_manifest", return_value={"splits": {}}),
            patch.object(v2, "check_manifest_preflight"),
            patch.object(v2, "sha256_file", side_effect=file_hash),
            patch.object(v2, "model_file_hashes", return_value=MODEL_HASHES),
            patch.object(v2, "read_jsonl", return_value=[row]),
            patch.object(v2, "_validate_study_split"),
            patch.object(v2, "generate_raw_predictions", return_value=(["{\"slots\":[]}"], {"max_observed_input_tokens": 90})),
            patch.object(v2, "score_predictions", return_value={"micro_f1": 0.0, "no_slot_failure_rate": 0.0}),
            patch.object(Path, "mkdir"),
            patch.object(Path, "write_bytes", new=write_bytes),
            patch.object(Path, "write_text", new=write_text),
        ):
            v2.main(["--system", "v2_step80", "--role", "dev"])
        report = json.loads(next(iter(report_text.values())))
        prediction_bytes = next(iter(disk.values()))
        self.assertEqual(report["prediction_file_sha256"], hashlib.sha256(prediction_bytes).hexdigest())
        self.assertEqual(report["frozen_model_files_sha256"], MODEL_HASHES)
        self.assertEqual(report["generation"]["max_new_tokens"], 64)
        self.assertEqual(report["generation"]["max_observed_input_tokens"], 90)
        self.assertEqual(json.loads(prediction_bytes)["eval_file_sha256"], "e" * 64)

    def test_confirmation_prediction_file_uses_exclusive_creation(self) -> None:
        content = b'{"raw_output":"ok"}\n'
        digest = hashlib.sha256(content).hexdigest()
        stream = MagicMock()
        with (
            patch.object(Path, "mkdir"),
            patch.object(Path, "open") as opener,
            patch.object(v2, "sha256_file", return_value=digest),
        ):
            opener.return_value.__enter__.return_value = stream
            observed = v2.write_predictions(Path("_tmp/v2-exclusive.jsonl"), content, exclusive=True)
        self.assertEqual(observed, digest)
        opener.assert_called_once_with("xb")
        stream.write.assert_called_once_with(content)


if __name__ == "__main__":
    unittest.main()
