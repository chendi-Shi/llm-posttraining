"""Safety and fixed-rule checks for the v6 hybrid evaluation entry point."""

from __future__ import annotations

import argparse
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import evaluate_massive_slots_v6_hybrid as v6  # noqa: E402
from massive_slots import canonical_output  # noqa: E402


def manifest() -> dict:
    sizes = v6.EXPECTED_COUNTS
    offset = 0
    splits = {}
    for name, count in sizes.items():
        splits[name] = {
            "examples": count,
            "jsonl_sha256": v6.EXPECTED_FILE_SHA256[name],
            "group_sha256": [format(index, "064x")
                             for index in range(offset + 1, offset + count + 1)],
        }
        offset += count
    return {"study": "MASSIVE v6 four-slot experiment", "format_version": 1,
            "source_sha256": v6.SOURCE_SHA256,
            "source_partition_used": "train", "seed": v6.BOOTSTRAP["dev"]["seed"],
            "v5_manifest_sha256": v6.V5_MANIFEST_SHA256, "splits": splits}


class V6HybridSafetyTests(unittest.TestCase):
    def test_test_requires_explicit_unlock_and_published_lock_hash_before_files(self) -> None:
        with patch.object(Path, "open", side_effect=AssertionError("file opened")), \
             patch.object(Path, "read_text", side_effect=AssertionError("file read")):
            with self.assertRaisesRegex(ValueError, "--unlock-test"):
                v6.main(["--role", "test"])
            with self.assertRaisesRegex(ValueError, "--expected-selection-lock-sha256"):
                v6.main(["--role", "test", "--unlock-test"])

    def test_existing_test_output_blocks_lock_and_test_data_access(self) -> None:
        with patch.object(Path, "exists", return_value=True), \
             patch.object(Path, "open", side_effect=AssertionError("file opened")), \
             patch("evaluate_massive_slots_v6_hybrid.require_test_lock",
                   side_effect=AssertionError("lock inspected")):
            with self.assertRaisesRegex(ValueError, "outputs already exist"):
                v6.main(["--role", "test", "--unlock-test",
                         "--expected-selection-lock-sha256", "a" * 64])

    def test_stale_inference_code_rejected_before_test_data_access(self) -> None:
        frozen = {name: "a" * 64 for name in v6.CODE_FILES}
        lock = {
            "status": "selected_for_one_test",
            "primary_candidate": "new_hybrid",
            "primary_reference": "old_v5_hybrid",
            "confidence_threshold": v6.MEAN_LOGPROB_THRESHOLD,
            "generation": v6.GENERATION,
            "bootstrap": v6.BOOTSTRAP["test"],
            "package_versions": {"torch": "synthetic-test-version"},
            **{key: "a" * 64 for key in (
                "manifest_sha256", "train_file_sha256", "dev_file_sha256",
                "test_file_sha256", "new_bio_model_sha256", "dev_report_sha256")},
            "old_bio_model_sha256": v6.OLD_BIO_SHA256,
            "v4_dpo_adapter_sha256": v6.DPO_SHA256,
            "code_sha256": frozen,
        }
        hashed = []

        def fake_sha(path: Path) -> str:
            hashed.append(path)
            if path == v6.LOCK:
                return "b" * 64
            if path == v6.ROOT / v6.CODE_FILES[0]:
                return "c" * 64
            raise AssertionError(f"Unexpected file hashed: {path}")

        with patch("evaluate_massive_slots_v6_hybrid.sha256_file", side_effect=fake_sha), \
             patch.object(Path, "read_text", return_value=json.dumps(lock)), \
             patch("evaluate_massive_slots_v6_hybrid.package_versions",
                   return_value=lock["package_versions"]), \
             patch("evaluate_massive_slots_v6_hybrid.model_hashes",
                   side_effect=AssertionError("model loaded")):
            with self.assertRaisesRegex(ValueError, "inference code changed"):
                v6.require_test_lock("b" * 64)
        self.assertNotIn(v6.DATA / "test.jsonl", hashed)

    def test_dev_file_preflight_never_hashes_test_file(self) -> None:
        frozen = manifest()
        hashed = []

        def fake_sha(path: Path) -> str:
            hashed.append(path)
            if path.name == "test.jsonl":
                raise AssertionError("dev opened the test file")
            return frozen["splits"][path.stem]["jsonl_sha256"]

        with patch("evaluate_massive_slots_v6_hybrid.sha256_file", side_effect=fake_sha):
            v6.check_split_hashes(frozen, "dev")
        self.assertEqual([item.name for item in hashed], ["train.jsonl", "dev.jsonl"])

    def test_manifest_rejects_overlap_and_wrong_split_size(self) -> None:
        frozen = manifest()
        v6.validate_manifest(frozen)
        frozen["splits"]["test"]["group_sha256"][0] = frozen["splits"]["dev"]["group_sha256"][0]
        with self.assertRaisesRegex(ValueError, "overlap"):
            v6.validate_manifest(frozen)
        frozen = manifest()
        frozen["splits"]["train"]["examples"] = 1599
        with self.assertRaisesRegex(ValueError, "train count"):
            v6.validate_manifest(frozen)
        frozen = manifest()
        frozen["splits"]["test"]["jsonl_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "preregistered SHA-256"):
            v6.validate_manifest(frozen)

    def test_changed_manifest_rejected_before_any_split_is_read(self) -> None:
        with patch.object(Path, "exists", return_value=False), \
             patch.object(Path, "read_text", side_effect=AssertionError("split opened")), \
             patch("evaluate_massive_slots_v6_hybrid.sha256_file",
                   return_value="f" * 64):
            with self.assertRaisesRegex(ValueError, "preregistered SHA-256"):
                v6.main(["--role", "dev"])

    def test_direct_test_runner_refuses_missing_verified_lock_before_rows(self) -> None:
        with patch("evaluate_massive_slots_v6_hybrid.read_jsonl",
                   side_effect=AssertionError("test rows read")):
            with self.assertRaisesRegex(ValueError, "verified selection lock"):
                v6.run("test", None, None, {}, "", Path("unused"), {}, Path("unused"))

    def test_row_group_must_match_normalized_utterance(self) -> None:
        row = {"id": "1", "utt": "明天在北京", "slots": [], "source_partition": "train",
               "group_sha256": v6.group_sha256(v6.strong_group_key("明天在北京"))}
        frozen = {"splits": {"dev": {"examples": 1,
                                      "group_sha256": [row["group_sha256"]]}}}
        v6.validate_rows([row], frozen, "dev")
        row["utt"] = "今天在北京"
        with self.assertRaisesRegex(ValueError, "differs from its utterance"):
            v6.validate_rows([row], frozen, "dev")

    def test_reused_v5_confidence_scorer_must_use_frozen_adapter(self) -> None:
        v6.check_confidence_assets()
        with patch.object(v6.v5, "ADAPTER", Path("outputs/other-adapter")):
            with self.assertRaisesRegex(ValueError, "frozen v5 DPO confidence assets"):
                v6.check_confidence_assets()

    def test_new_bio_bundle_must_identify_frozen_training_split(self) -> None:
        bundle = {"configuration": v6.MODEL_CONFIG, "vectorizer": object(),
                  "model": object(), "train_file_sha256": "a" * 64,
                  "manifest_sha256": "b" * 64}
        with patch("evaluate_massive_slots_v6_hybrid.joblib.load", return_value=bundle):
            v6.load_bio(Path("model.joblib"), expected_train_sha256="a" * 64,
                        expected_manifest_sha256="b" * 64)
            with self.assertRaisesRegex(ValueError, "not trained on the frozen v6"):
                v6.load_bio(Path("model.joblib"), expected_train_sha256="c" * 64,
                            expected_manifest_sha256="b" * 64)

    def test_both_hybrids_use_same_frozen_dpo_gate(self) -> None:
        rows = [{"utt": "明天见小王"}, {"utt": "今天见小李"}]
        old = [canonical_output([]), canonical_output([])]
        new = [canonical_output([{"type": "person", "value": "小王"}]),
               canonical_output([{"type": "person", "value": "小李"}])]
        dpo = [canonical_output([{"type": "date", "value": "明天"}]),
               canonical_output([{"type": "date", "value": "今天"}])]
        chosen, replaced = v6.choose_outputs(rows, old, new, dpo, [-.08, -.081])
        self.assertEqual(chosen["old_v5_hybrid"], [dpo[0], old[1]])
        self.assertEqual(chosen["new_hybrid"], [dpo[0], new[1]])
        self.assertEqual(replaced, {"old_v5_hybrid": 1, "new_hybrid": 1})

    def test_development_and_test_gates_keep_the_preset_thresholds(self) -> None:
        metrics = {"micro_f1": .60, "no_slot_failure_rate": .20,
                   "json_valid_rate": .95, "schema_valid_rate": .95,
                   "copy_valid_rate": .95}
        positive = {"micro_f1": .60}
        self.assertTrue(v6.development_gate(metrics, positive)["passed"])
        self.assertTrue(v6.independent_test_gate(
            metrics, positive, {"delta_ci95": [.0001, .1]})["passed"])
        self.assertFalse(v6.independent_test_gate(
            metrics, positive, {"delta_ci95": [0, .1]})["passed"])
        for field, value in (("micro_f1", .599), ("no_slot_failure_rate", .201),
                             ("json_valid_rate", .949), ("schema_valid_rate", .949),
                             ("copy_valid_rate", .949)):
            with self.subTest(field=field):
                self.assertFalse(v6.development_gate({**metrics, field: value},
                                                     positive)["passed"])
        self.assertFalse(v6.development_gate(metrics, {"micro_f1": .599})["passed"])


if __name__ == "__main__":
    unittest.main()
