"""One-time locked confirmation for the already-trained four-slot BIO model.

This script never fits a model. It refuses to read the confirmation data until
``--unlock-confirmation`` is explicitly supplied and verifies the frozen
model, source archive, training file, and confirmation file before inference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from evaluate_massive_slots import score_predictions
from evaluate_massive_slots_bio import predict_slots, read_jsonl, write_predictions
from massive_slots import canonical_output
from prepare_massive_slots import file_sha256


EXPECTED_EXAMPLES = {"train": 384, "confirmation": 200}


def require_unlock(unlocked: bool) -> None:
    if not unlocked:
        raise SystemExit(
            "Confirmation remains locked. Pass --unlock-confirmation only after "
            "the model, decoding, and comparison plan have been frozen."
        )


def check_frozen_hashes(
    manifest: dict,
    dev_report: dict,
    actual: dict[str, str],
    expected_model_sha256: str,
) -> None:
    """Verify the previously selected model and exact study files."""
    for name in ("source", "train", "confirmation", "model"):
        if name not in actual or not isinstance(actual[name], str):
            raise ValueError(f"Missing actual {name} SHA-256")
    if len(expected_model_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in expected_model_sha256.lower()
    ):
        raise ValueError("Expected model SHA-256 must be 64 hexadecimal characters")
    expected = {
        "source": manifest["source_sha256"],
        "train": manifest["splits"]["train"]["jsonl_sha256"],
        "confirmation": manifest["splits"]["confirmation"]["jsonl_sha256"],
        "model": expected_model_sha256.lower(),
    }
    for name, expected_hash in expected.items():
        if actual[name].lower() != expected_hash.lower():
            raise ValueError(f"Frozen {name} SHA-256 mismatch")
    if dev_report.get("model_sha256", "").lower() != actual["model"].lower():
        raise ValueError("Saved model differs from the development report")
    if dev_report.get("train_file_sha256") != actual["train"]:
        raise ValueError("Development report used another training file")
    if dev_report.get("source_sha256") != actual["source"]:
        raise ValueError("Development report used another source archive")
    if dev_report.get("confirmation_used") is not False:
        raise ValueError("Development report is not an untouched confirmation candidate")
    for split, count in EXPECTED_EXAMPLES.items():
        if manifest["splits"][split]["examples"] != count:
            raise ValueError(f"Unexpected {split} size in the frozen manifest")


def validate_selected_rows(train_rows: list[dict], confirmation_rows: list[dict], manifest: dict) -> None:
    """Check IDs and phrase hashes before the model sees confirmation text."""
    for split, rows in (("train", train_rows), ("confirmation", confirmation_rows)):
        if len(rows) != EXPECTED_EXAMPLES[split]:
            raise ValueError(f"Unexpected {split} row count")
        if [str(row["id"]) for row in rows] != manifest["splits"][split]["source_ids"]:
            raise ValueError(f"{split} IDs differ from the frozen manifest")
        if len({str(row["id"]) for row in rows}) != len(rows):
            raise ValueError(f"Duplicate {split} IDs")
        expected_groups = [item["group_sha256"] for item in manifest["splits"][split]["groups"]]
        if [row["group_sha256"] for row in rows] != expected_groups:
            raise ValueError(f"{split} phrase hashes differ from the frozen manifest")
    if {row["group_sha256"] for row in train_rows} & {
        row["group_sha256"] for row in confirmation_rows
    }:
        raise ValueError("Training and confirmation phrases overlap")
    dev_groups = {
        item["group_sha256"] for item in manifest["splits"].get("dev", {}).get("groups", [])
    }
    if dev_groups & {row["group_sha256"] for row in confirmation_rows}:
        raise ValueError("Development and confirmation phrases overlap")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unlock-confirmation", action="store_true")
    parser.add_argument("--expected-model-sha256", default=None)
    parser.add_argument("--archive", type=Path, default=Path("data/raw/amazon-massive-dataset-1.0.tar.gz"))
    parser.add_argument("--train-file", type=Path, default=Path("data/massive-zh/slots/train.jsonl"))
    parser.add_argument("--confirmation-file", type=Path, default=Path("data/massive-zh/slots/confirmation.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("data/massive-zh/slots/manifest.json"))
    parser.add_argument("--dev-report", type=Path, default=Path("reports/massive-slots-bio-dev.json"))
    parser.add_argument("--model", type=Path, default=Path("outputs/massive-slots-bio.joblib"))
    parser.add_argument("--report", type=Path, default=Path("reports/massive-slots-bio-confirmation.json"))
    parser.add_argument(
        "--predictions-out", type=Path,
        default=Path("_tmp/massive-slots-bio-confirmation-predictions.jsonl"),
    )
    args = parser.parse_args(argv)

    # This must precede even hashing, opening, or existence-checking the
    # confirmation file. No accidental command can spend the locked set.
    require_unlock(args.unlock_confirmation)
    if args.expected_model_sha256 is None:
        raise SystemExit("A reviewed --expected-model-sha256 is required")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    dev_report = json.loads(args.dev_report.read_text(encoding="utf-8"))
    actual = {
        "source": file_sha256(args.archive),
        "train": file_sha256(args.train_file),
        "confirmation": file_sha256(args.confirmation_file),
        "model": file_sha256(args.model),
    }
    check_frozen_hashes(manifest, dev_report, actual, args.expected_model_sha256)
    train_rows = read_jsonl(args.train_file)
    confirmation_rows = read_jsonl(args.confirmation_file)
    validate_selected_rows(train_rows, confirmation_rows, manifest)

    bundle = joblib.load(args.model)
    if bundle.get("train_file_sha256") != actual["train"]:
        raise ValueError("Model bundle carries another training-file SHA-256")
    if bundle.get("source_sha256") != actual["source"]:
        raise ValueError("Model bundle carries another source-archive SHA-256")
    if bundle.get("configuration") != dev_report.get("training_configuration"):
        raise ValueError("Model configuration differs from development report")
    vectorizer, model = bundle["vectorizer"], bundle["model"]
    raw_predictions = [
        canonical_output(predict_slots(row["utt"], vectorizer, model))
        for row in confirmation_rows
    ]
    write_predictions(
        args.predictions_out,
        confirmation_rows,
        raw_predictions,
        actual["confirmation"],
    )
    metrics = score_predictions(confirmation_rows, raw_predictions, bootstrap_samples=1000, seed=20260928)
    report = {
        "task": "MASSIVE 1.0 zh-CN four-slot JSON extraction",
        "model": "frozen character BIO contextual features + LinearSVC + constrained BIO decoding",
        "training_configuration": bundle["configuration"],
        "source_sha256": actual["source"],
        "train_file_sha256": actual["train"],
        "confirmation_file_sha256": actual["confirmation"],
        "model_sha256": actual["model"],
        "data_license": "CC-BY-4.0",
        "prediction_file_sha256": file_sha256(args.predictions_out),
        "train_examples": len(train_rows),
        "confirmation_examples": len(confirmation_rows),
        "confirmation_used": True,
        "metrics": metrics,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "confirmation_examples": len(confirmation_rows),
                "model_sha256": actual["model"],
                "micro_f1": metrics["micro_f1"],
                "report": str(args.report),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
