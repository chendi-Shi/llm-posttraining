"""Matched-data BIO control for the independent MASSIVE four-slot v2 study.

Development fits one model on v2 train. Confirmation only loads that frozen
model and requires explicit file hashes before reading confirmation examples.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib

from evaluate_massive_slots import (
    _validate_study_split,
    load_locked_split_manifest,
    read_jsonl,
    score_predictions,
    sha256_file,
)
from evaluate_massive_slots_bio import (
    MODEL_CONFIG,
    align_gold,
    predict_slots,
    train_model,
)
from massive_slots import canonical_output
from prepare_massive_slots import read_archive


EXPECTED_COUNTS = {"train": 640, "dev": 250, "confirmation": 400}
ROOT = Path(__file__).resolve().parents[1]
CONFIRMATION_CODE_FILES = (
    "scripts/evaluate_massive_slots_v2_bio.py",
    "scripts/evaluate_massive_slots_bio.py",
    "scripts/evaluate_massive_slots.py",
    "scripts/prepare_massive_slots.py",
    "scripts/massive_slots.py",
)
CONFIRMATION_PREDICTIONS = Path("_tmp/massive-slots-v2-bio-confirmation-predictions.jsonl")
CONFIRMATION_REPORT = Path("reports/massive-slots-v2-bio-confirmation.json")
CONFIRMATION_SELECTION_LOCK = Path("reports/massive-slots-v2-selection-lock.json")


def require_confirmation_lock(args: argparse.Namespace) -> None:
    """Reject an accidental confirmation call before any file is opened."""
    if args.role != "confirmation":
        return
    if not args.unlock_confirmation:
        raise ValueError("Confirmation requires --unlock-confirmation after model selection")
    for name in ("expected_selection_lock_sha256", "expected_manifest_sha256", "expected_model_sha256", "expected_dev_report_sha256"):
        value = getattr(args, name)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value.lower()):
            raise ValueError(f"Confirmation requires a 64-hex --{name.replace('_', '-')}")


def check_file_hash(path: Path, expected: str, label: str) -> None:
    if sha256_file(path) != expected.lower():
        raise ValueError(f"Frozen {label} SHA-256 mismatch")


def check_confirmation_code_hashes(lock: dict) -> None:
    code_hashes = lock.get("code_sha256")
    if not isinstance(code_hashes, dict):
        raise ValueError("BIO confirmation lock lacks code fingerprints")
    for relative in CONFIRMATION_CODE_FILES:
        expected = code_hashes.get(relative)
        if (not isinstance(expected, str) or len(expected) != 64
                or any(char not in "0123456789abcdef" for char in expected.lower())
                or sha256_file(ROOT / relative) != expected.lower()):
            raise ValueError(f"BIO confirmation code changed after selection: {relative}")


def check_manifest_counts(manifest: dict) -> None:
    if {role: manifest["splits"][role]["examples"] for role in EXPECTED_COUNTS} != EXPECTED_COUNTS:
        raise ValueError("Unexpected v2 train/dev/confirmation split sizes")


def make_predictions(rows: list[dict], vectorizer: object, model: object) -> list[str]:
    return [canonical_output(predict_slots(row["utt"], vectorizer, model)) for row in rows]


def write_bio_predictions(
    path: Path, rows: list[dict], raw: list[str], eval_sha256: str,
    *, exclusive: bool,
) -> None:
    if not path.resolve().is_relative_to((ROOT / "_tmp").resolve()):
        raise ValueError("Per-row BIO predictions must stay in the ignored _tmp directory")
    if len(rows) != len(raw):
        raise ValueError("BIO prediction count differs from evaluation rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf-8", newline="\n") as stream:
        for row, output in zip(rows, raw, strict=True):
            stream.write(json.dumps({
                "id": str(row["id"]),
                "group_sha256": row["group_sha256"],
                "eval_file_sha256": eval_sha256,
                "raw_output": output,
            }, ensure_ascii=False, separators=(",", ":")) + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("dev", "confirmation"), default="dev")
    parser.add_argument("--unlock-confirmation", action="store_true")
    parser.add_argument("--selection-lock", type=Path,
                        default=Path("reports/massive-slots-v2-selection-lock.json"))
    parser.add_argument("--expected-selection-lock-sha256")
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument("--expected-model-sha256")
    parser.add_argument("--expected-dev-report-sha256")
    parser.add_argument("--archive", type=Path, default=Path("data/raw/amazon-massive-dataset-1.0.tar.gz"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/massive-zh/slots-v2"))
    parser.add_argument("--model", type=Path, default=Path("outputs/massive-slots-v2-bio.joblib"))
    parser.add_argument("--dev-report", type=Path, default=Path("reports/massive-slots-v2-bio-dev.json"))
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--predictions", type=Path, default=None)
    args = parser.parse_args(argv)

    # This gate must run before inspecting any local data or model file.
    require_confirmation_lock(args)
    if (args.role == "confirmation"
            and args.selection_lock.resolve() != CONFIRMATION_SELECTION_LOCK.resolve()):
        raise ValueError("BIO confirmation requires the canonical v2 selection lock path")
    predictions = args.predictions or Path(f"_tmp/massive-slots-v2-bio-{args.role}-predictions.jsonl")
    report_path = args.report or Path(f"reports/massive-slots-v2-bio-{args.role}.json")
    if args.role == "confirmation" and (
        predictions.resolve() != CONFIRMATION_PREDICTIONS.resolve()
        or report_path.resolve() != CONFIRMATION_REPORT.resolve()
    ):
        raise ValueError("BIO confirmation outputs must use their fixed paths")
    if not predictions.resolve().is_relative_to((ROOT / "_tmp").resolve()):
        raise ValueError("Per-row BIO predictions must stay in the ignored _tmp directory")
    if args.role == "confirmation" and (predictions.exists() or report_path.exists()):
        raise ValueError("BIO confirmation outputs already exist; refusing to overwrite")
    manifest_path = args.data_dir / "manifest.json"
    if args.role == "confirmation":
        check_file_hash(args.selection_lock, args.expected_selection_lock_sha256, "SFT selection lock")
        lock = json.loads(args.selection_lock.read_text(encoding="utf-8"))
        if (lock.get("status") != "selected"
                or lock.get("selected_step") not in (80, 160)
                or lock.get("manifest_sha256") != args.expected_manifest_sha256.lower()
                or lock.get("reference_systems", {}).get("bio", {}).get("model_sha256")
                != args.expected_model_sha256.lower()
                or lock.get("reference_systems", {}).get("bio", {}).get("dev_report_sha256")
                != args.expected_dev_report_sha256.lower()):
            raise ValueError("BIO confirmation requires the frozen v2 SFT selection lock")
        check_confirmation_code_hashes(lock)
        check_file_hash(manifest_path, args.expected_manifest_sha256, "manifest")
        check_file_hash(args.data_dir / "confirmation.jsonl",
                        lock["confirmation_file_sha256"], "confirmation data")
        check_file_hash(args.model, args.expected_model_sha256, "BIO model")
        check_file_hash(args.dev_report, args.expected_dev_report_sha256, "BIO dev report")

    eval_path = args.data_dir / f"{args.role}.jsonl"
    manifest = load_locked_split_manifest(eval_path, args.role)
    check_manifest_counts(manifest)
    if sha256_file(args.archive) != manifest["source_sha256"]:
        raise ValueError("Official MASSIVE archive differs from v2 manifest")
    train_path = args.data_dir / "train.jsonl"
    if sha256_file(train_path) != manifest["splits"]["train"]["jsonl_sha256"]:
        raise ValueError("v2 training file differs from manifest")
    bundle = None
    if args.role == "confirmation":
        # Reject a mismatched dev run or saved model before parsing held-out text.
        dev_report = json.loads(args.dev_report.read_text(encoding="utf-8"))
        if (dev_report.get("role") != "dev"
                or dev_report.get("model_sha256") != args.expected_model_sha256.lower()
                or dev_report.get("manifest_sha256") != args.expected_manifest_sha256.lower()
                or dev_report.get("train_file_sha256") != manifest["splits"]["train"]["jsonl_sha256"]):
            raise ValueError("BIO development report does not identify frozen v2 model/data")
        bundle = joblib.load(args.model)
        if (bundle.get("configuration") != MODEL_CONFIG
                or bundle.get("source_sha256") != manifest["source_sha256"]
                or bundle.get("train_file_sha256") != manifest["splits"]["train"]["jsonl_sha256"]
                or bundle.get("manifest_sha256") != args.expected_manifest_sha256.lower()):
            raise ValueError("BIO model bundle does not identify frozen v2 training data")
    rows = read_jsonl(eval_path)
    _validate_study_split(rows, eval_path, args.role, manifest)
    train_rows = read_jsonl(train_path)
    _validate_study_split(train_rows, train_path, "train", manifest)

    started = time.perf_counter()
    training_seconds = None
    if args.role == "dev":
        archive_rows, _ = read_archive(args.archive)
        source_by_id = {str(row["id"]): row for row in archive_rows if row["partition"] == "train"}
        train_spans = align_gold(train_rows, source_by_id)
        align_gold(rows, source_by_id)
        vectorizer, model = train_model(train_rows, train_spans)
        training_seconds = round(time.perf_counter() - started, 2)
        args.model.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "vectorizer": vectorizer,
            "model": model,
            "configuration": MODEL_CONFIG,
            "source_sha256": manifest["source_sha256"],
            "train_file_sha256": manifest["splits"]["train"]["jsonl_sha256"],
            "manifest_sha256": sha256_file(manifest_path),
        }, args.model)
    else:
        assert bundle is not None
        vectorizer, model = bundle["vectorizer"], bundle["model"]

    raw = make_predictions(rows, vectorizer, model)
    metrics = score_predictions(rows, raw, bootstrap_samples=1000, seed=20260929)
    write_bio_predictions(
        predictions, rows, raw, sha256_file(eval_path),
        exclusive=args.role == "confirmation",
    )
    report = {
        "task": "MASSIVE 1.0 zh-CN four-slot JSON extraction, v2 matched BIO control",
        "role": args.role,
        "source_sha256": manifest["source_sha256"],
        "manifest_sha256": sha256_file(manifest_path),
        "train_file_sha256": sha256_file(train_path),
        "eval_file_sha256": sha256_file(eval_path),
        "training_configuration": MODEL_CONFIG,
        "model_sha256": sha256_file(args.model),
        "prediction_file_sha256": sha256_file(predictions),
        "training_seconds": training_seconds,
        "confirmation_used": args.role == "confirmation",
        "selection_lock_sha256": args.expected_selection_lock_sha256.lower() if args.role == "confirmation" else None,
        "data_license": "CC-BY-4.0",
        "metrics": metrics,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("x" if args.role == "confirmation" else "w", encoding="utf-8") as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"role": args.role, "micro_f1": metrics["micro_f1"],
                      "no_slot_failure_rate": metrics["no_slot_failure_rate"],
                      "model_sha256": report["model_sha256"], "report": str(report_path)},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
