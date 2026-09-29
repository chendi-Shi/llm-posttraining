"""Offline checks for v2 development-only checkpoint selection and freezing."""

from __future__ import annotations

import json
import shutil
import sys
import unittest
import uuid
from fractions import Fraction
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from select_massive_slots_v2 import (  # noqa: E402
    EXPECTED_GENERATION,
    EXPECTED_TRAINING,
    SelectionError,
    choose_checkpoint,
    select_and_lock,
    sha256_file,
)
from v2_model_assets import hash_adapter_files, hash_base_files  # noqa: E402


class V2Fixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.source = root / "source.tar.gz"
        self.source.write_bytes(b"synthetic official source fixture")
        self.study = root / "slots-v2"
        self.study.mkdir()
        self.model = root / "base"
        self.model.mkdir()
        (self.model / "model.safetensors").write_bytes(b"base weights fixture")
        (self.model / "tokenizer.json").write_bytes(b"tokenizer fixture")
        (self.model / "config.json").write_text('{"fixture":1}', encoding="utf-8")
        self.sft = root / "sft"
        self.sft.mkdir()
        (self.sft / "checkpoint-80").mkdir()
        (self.sft / "checkpoint-80" / "adapter_model.safetensors").write_bytes(b"step 80 fixture")
        (self.sft / "checkpoint-80" / "adapter_config.json").write_text('{"step":80}', encoding="utf-8")
        (self.sft / "adapter_model.safetensors").write_bytes(b"step 160 fixture")
        (self.sft / "adapter_config.json").write_text('{"step":160}', encoding="utf-8")
        self.code = root / "code.py"
        self.code.write_text("# synthetic code fixture\n", encoding="utf-8")
        splits = {}
        for split in ("train", "dev", "confirmation"):
            path = self.study / f"{split}.jsonl"
            path.write_text(f"SENSITIVE_{split.upper()}_ROW\n" * 20, encoding="utf-8")
            splits[split] = {
                "examples": 20,
                "jsonl_sha256": sha256_file(path),
                "source_ids": [f"{split}-{number}" for number in range(20)],
                "groups": [
                    {"group_sha256": __import__("hashlib").sha256(f"{split}-{number}".encode()).hexdigest()}
                    for number in range(20)
                ],
            }
        self.manifest = self.study / "manifest.json"
        self.manifest.write_text(json.dumps({
            "study": "MASSIVE zh-CN four-slot v2",
            "format_version": 1,
            "source_partition_used": "train",
            "source_sha256": sha256_file(self.source),
            "splits": splits,
        }), encoding="utf-8")
        self.run_manifest = self.sft / "run_manifest.json"
        self.run_manifest.write_text(json.dumps({
            "train_file": str(self.study / "train.jsonl"),
            "train_file_sha256": sha256_file(self.study / "train.jsonl"),
            "model": str(self.model),
            "model_asset_sha256": {
                "model.safetensors": sha256_file(self.model / "model.safetensors"),
                "tokenizer.json": sha256_file(self.model / "tokenizer.json"),
            },
            **EXPECTED_TRAINING,
            "package_versions": {name: "fixture" for name in
                                 ("torch", "transformers", "trl", "peft", "bitsandbytes")},
        }), encoding="utf-8")
        self.step80 = root / "step80.json"
        self.step160 = root / "step160.json"
        self.base = root / "base.json"
        self.predictions = {
            80: root / "step80-predictions.jsonl",
            160: root / "step160-predictions.jsonl",
        }
        for step, path in self.predictions.items():
            path.write_text(f"synthetic-step-{step}\n", encoding="utf-8")
        self.lock = root / "selection-lock.json"
        self.write_report(self.step80, 80, tp=4, fp=4, fn=6, failure_count=2)
        self.write_report(self.step160, 160, tp=6, fp=4, fn=4, failure_count=3)
        self.write_report(self.base, None, tp=1, fp=1, fn=9, failure_count=10)

    def write_report(
        self,
        path: Path,
        step: int | None,
        *,
        tp: int,
        fp: int,
        fn: int,
        failure_count: int,
        copy_rate: float = 0.96,
    ) -> None:
        adapter = None if step is None else self.sft / ("checkpoint-80" if step == 80 else "")
        f1 = 2 * tp / (2 * tp + fp + fn)
        report = {
            "examples": 20,
            "metric_definition": {
                "entity_match": "multiset of (type, literal value) per utterance",
                "no_slot_failure": "invalid output or nonempty slots on a gold-empty utterance",
            },
            "gold_slot_count": tp + fn,
            "predicted_slot_count": tp + fp,
            "micro": {"tp": tp, "fp": fp, "fn": fn, "f1": f1},
            "micro_f1": f1,
            "json_valid_rate": 1.0,
            "schema_valid_rate": 0.98,
            "copy_valid_rate": copy_rate,
            "no_slot_examples": 10,
            "no_slot_false_positive_count": failure_count - 1,
            "no_slot_invalid_output_count": 1,
            "no_slot_failure_rate": failure_count / 10,
            "task": "MASSIVE 1.0 zh-CN four-slot copy-only JSON extraction",
            "role": "dev",
            "source_partition": "train",
            "eval_file": str(self.study / "dev.jsonl"),
            "eval_file_sha256": sha256_file(self.study / "dev.jsonl"),
            "study_manifest_sha256": sha256_file(self.manifest),
            "model": str(self.model),
            "adapter": str(adapter) if adapter is not None else None,
            "frozen_model_files_sha256": {
                **hash_base_files(self.model),
                **(hash_adapter_files(adapter) if adapter is not None else {}),
            },
            "prediction_file_sha256": (
                sha256_file(self.predictions[step]) if step is not None else None
            ),
            "generation": {**EXPECTED_GENERATION, "max_observed_input_tokens": 100},
        }
        path.write_text(json.dumps(report), encoding="utf-8")

    def run_selector(self, *, base: bool = False) -> dict:
        return select_and_lock(
            manifest_path=self.manifest,
            source_path=self.source,
            model_dir=self.model,
            sft_dir=self.sft,
            step80_metrics_path=self.step80,
            step160_metrics_path=self.step160,
            output_path=self.lock,
            code_files=[self.code],
            base_metrics_path=self.base if base else None,
            expected_source_sha256=sha256_file(self.source),
            expected_counts={"train": 20, "dev": 20, "confirmation": 20},
            expected_data_sha256={
                "manifest": sha256_file(self.manifest),
                **{split: sha256_file(self.study / f"{split}.jsonl")
                   for split in ("train", "dev", "confirmation")},
            },
            dev_prediction_paths=self.predictions,
            reference_v1_lock_path=None,
            reference_bio_dev_report_path=None,
            reference_bio_model_path=None,
        )


class SelectMassiveSlotsV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        local_tmp = ROOT / "_tmp"
        local_tmp.mkdir(exist_ok=True)
        self.fixture_root = local_tmp / f"select-massive-slots-v2-{uuid.uuid4().hex}"
        self.fixture_root.mkdir()
        self.addCleanup(self.cleanup_fixture)
        self.fixture = V2Fixture(self.fixture_root)

    def cleanup_fixture(self) -> None:
        if self.fixture_root.resolve().parent != (ROOT / "_tmp").resolve():
            raise RuntimeError("Refusing to remove a fixture outside the workspace _tmp directory")
        shutil.rmtree(self.fixture_root)

    def test_selects_higher_f1_and_freezes_only_aggregate_hashes(self) -> None:
        lock = self.fixture.run_selector(base=True)
        self.assertEqual(lock["selected_step"], 160)
        self.assertEqual(lock["selected_adapter_sha256"], sha256_file(self.fixture.sft / "adapter_model.safetensors"))
        self.assertEqual(lock["manifest_sha256"], sha256_file(self.fixture.manifest))
        self.assertEqual(lock["dev_file_sha256"], sha256_file(self.fixture.study / "dev.jsonl"))
        self.assertEqual(lock["confirmation_file_sha256"], sha256_file(self.fixture.study / "confirmation.jsonl"))
        self.assertIn("base", lock["dev_metrics_sha256"])
        serialized = self.fixture.lock.read_text(encoding="utf-8")
        self.assertNotIn("SENSITIVE_", serialized)
        self.assertNotIn("raw_output", serialized)
        self.assertNotIn("source_ids", serialized)

    def test_ties_use_lower_no_slot_failure_then_earlier_step(self) -> None:
        results = {
            80: {"eligible": True, "_ranking": (-Fraction(3, 5), Fraction(1, 5))},
            160: {"eligible": True, "_ranking": (-Fraction(3, 5), Fraction(1, 10))},
        }
        self.assertEqual(choose_checkpoint(results), 160)
        results[160]["_ranking"] = (-Fraction(3, 5), Fraction(1, 5))
        self.assertEqual(choose_checkpoint(results), 80)

    def test_all_gate_failures_create_no_lock(self) -> None:
        self.fixture.write_report(self.fixture.step80, 80, tp=4, fp=4, fn=6, failure_count=4, copy_rate=0.94)
        self.fixture.write_report(self.fixture.step160, 160, tp=6, fp=4, fn=4, failure_count=4)
        with self.assertRaisesRegex(SelectionError, "No v2 checkpoint passes"):
            self.fixture.run_selector()
        self.assertFalse(self.fixture.lock.exists())

    def test_invalid_outputs_count_as_no_slot_failures(self) -> None:
        report = json.loads(self.fixture.step160.read_text(encoding="utf-8"))
        report["no_slot_false_positive_count"] = 0
        report["no_slot_invalid_output_count"] = 4
        report["no_slot_failure_rate"] = 0.4
        self.fixture.step160.write_text(json.dumps(report), encoding="utf-8")
        lock = self.fixture.run_selector()
        self.assertEqual(lock["selected_step"], 80)
        self.assertEqual(lock["candidate_dev_results"]["160"]["gate_failures"], ["no_slot_failure_rate"])

    def test_rejects_different_dev_manifest_model_and_generation(self) -> None:
        for changed_key, changed_value in (
            ("eval_file_sha256", "0" * 64),
            ("study_manifest_sha256", "0" * 64),
            ("model", str(self.fixture.root / "other-model")),
        ):
            with self.subTest(changed_key=changed_key):
                original = json.loads(self.fixture.step80.read_text(encoding="utf-8"))
                modified = {**original, changed_key: changed_value}
                self.fixture.step80.write_text(json.dumps(modified), encoding="utf-8")
                with self.assertRaises(SelectionError):
                    self.fixture.run_selector()
                self.assertFalse(self.fixture.lock.exists())
                self.fixture.step80.write_text(json.dumps(original), encoding="utf-8")
        report = json.loads(self.fixture.step80.read_text(encoding="utf-8"))
        report["generation"]["max_new_tokens"] = 96
        self.fixture.step80.write_text(json.dumps(report), encoding="utf-8")
        with self.assertRaisesRegex(SelectionError, "generation differs"):
            self.fixture.run_selector()

    def test_rejects_changed_confirmation_bytes_without_reading_rows(self) -> None:
        (self.fixture.study / "confirmation.jsonl").write_text("changed private rows\n", encoding="utf-8")
        with self.assertRaisesRegex(SelectionError, "confirmation file differs"):
            self.fixture.run_selector()
        self.assertFalse(self.fixture.lock.exists())

    def test_candidate_must_record_inference_time_weight_hashes(self) -> None:
        report = json.loads(self.fixture.step80.read_text(encoding="utf-8"))
        report["frozen_model_files_sha256"] = None
        self.fixture.step80.write_text(json.dumps(report), encoding="utf-8")
        with self.assertRaisesRegex(SelectionError, "inference-time model hashes"):
            self.fixture.run_selector()
        self.assertFalse(self.fixture.lock.exists())

    def test_candidate_report_must_bind_its_prediction_file(self) -> None:
        self.fixture.predictions[160].write_text("different predictions\n", encoding="utf-8")
        with self.assertRaisesRegex(SelectionError, "dev predictions differ"):
            self.fixture.run_selector()
        self.assertFalse(self.fixture.lock.exists())

    def test_inference_config_change_rejects_dev_report(self) -> None:
        (self.fixture.model / "config.json").write_text('{"fixture":2}', encoding="utf-8")
        with self.assertRaisesRegex(SelectionError, "frozen base/config.json differs"):
            self.fixture.run_selector()
        self.assertFalse(self.fixture.lock.exists())

    def test_training_manifest_must_match_preregistered_run(self) -> None:
        run = json.loads(self.fixture.run_manifest.read_text(encoding="utf-8"))
        run["seed"] = 1
        self.fixture.run_manifest.write_text(json.dumps(run), encoding="utf-8")
        with self.assertRaisesRegex(SelectionError, "unexpected seed"):
            self.fixture.run_selector()
        self.assertFalse(self.fixture.lock.exists())

    def test_existing_lock_cannot_be_overwritten(self) -> None:
        self.fixture.lock.write_text("previous frozen choice", encoding="utf-8")
        with self.assertRaisesRegex(SelectionError, "must not be overwritten"):
            self.fixture.run_selector()
        self.assertEqual(self.fixture.lock.read_text(encoding="utf-8"), "previous frozen choice")

    def test_production_selector_refuses_synthetic_source(self) -> None:
        with self.assertRaisesRegex(SelectionError, "Official source archive differs"):
            select_and_lock(
                manifest_path=self.fixture.manifest,
                source_path=self.fixture.source,
                model_dir=self.fixture.model,
                sft_dir=self.fixture.sft,
                step80_metrics_path=self.fixture.step80,
                step160_metrics_path=self.fixture.step160,
                output_path=self.fixture.lock,
                code_files=[self.fixture.code],
            )
        self.assertFalse(self.fixture.lock.exists())


if __name__ == "__main__":
    unittest.main()
