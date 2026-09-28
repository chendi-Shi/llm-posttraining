"""One-time official test evaluation of the frozen MASSIVE linear baseline.

This script only loads the saved classifier and runs inference. It never fits,
chooses a candidate, or tunes a threshold. The official test was previously
inspected in the earlier SFT study, so this is a confirmation, not a pristine
blind evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import joblib

from evaluate_massive_linear import compact_metrics, model_text, text_key
from massive_metrics import score
from prepare_massive import read_partitions


LOCKED_MODEL_SHA256 = "fa9f1cb3c72496060492703087eb1a5fb558e35d276e4049cd81fefbe07fe13e"
EXPECTED_SPLIT_SIZES = {"train": 11514, "dev": 2033, "test": 2974}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="data/raw/amazon-massive-dataset-1.0.tar.gz")
    parser.add_argument("--model", default="outputs/massive-linear-baseline.joblib")
    parser.add_argument("--dev-report", default="reports/massive-linear-dev.json")
    parser.add_argument("--report", default="reports/massive-linear-test.json")
    parser.add_argument("--predictions", default="reports/massive-linear-test.jsonl")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    source, model_path, dev_report_path = map(Path, (args.source, args.model, args.dev_report))
    report_path, predictions_path = map(Path, (args.report, args.predictions))
    for path in (report_path, predictions_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite an existing test result: {path}")
    for path in (source, model_path, dev_report_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    model_sha256 = file_sha256(model_path)
    if model_sha256 != LOCKED_MODEL_SHA256:
        raise ValueError(f"Model SHA-256 differs from frozen dev candidate: {model_sha256}")
    dev_report = json.loads(dev_report_path.read_text(encoding="utf-8"))
    if dev_report["selected_candidate"] != "char_1_3":
        raise ValueError("Dev report does not select the locked char_1_3 model")
    source_sha256 = file_sha256(source)
    if source_sha256 != dev_report["source_sha256"]:
        raise ValueError("Official source archive differs from dev run")

    started = time.perf_counter()
    partitions = read_partitions(source)
    split_sizes = {name: len(partitions.get(name, [])) for name in EXPECTED_SPLIT_SIZES}
    if split_sizes != EXPECTED_SPLIT_SIZES:
        raise ValueError(f"Unexpected official zh-CN split sizes: {split_sizes}")
    train_rows, dev_rows, test_rows = (
        partitions[name] for name in ("train", "dev", "test")
    )
    ids = {name: {row["id"] for row in rows} for name, rows in partitions.items()}
    if ids["test"] & (ids["train"] | ids["dev"]):
        raise ValueError("Test IDs overlap train/dev IDs")
    if len(ids["test"]) != len(test_rows):
        raise ValueError("Duplicate test IDs")

    train_texts = {row["utt"].strip() for row in train_rows}
    dev_texts = {row["utt"].strip() for row in dev_rows}
    train_keys = {text_key(row["utt"]) for row in train_rows}
    dev_keys = {text_key(row["utt"]) for row in dev_rows}
    exact_seen = train_texts | dev_texts
    normalized_seen = train_keys | dev_keys
    clean_test = [row for row in test_rows if text_key(row["utt"]) not in normalized_seen]
    if not clean_test:
        raise ValueError("No phrase-disjoint test rows")
    duplicate_audit = {
        "test_rows_with_exact_train_text": sum(row["utt"].strip() in train_texts for row in test_rows),
        "test_rows_with_exact_dev_text": sum(row["utt"].strip() in dev_texts for row in test_rows),
        "test_rows_with_exact_train_or_dev_text": sum(row["utt"].strip() in exact_seen for row in test_rows),
        "test_rows_with_normalized_train_text": sum(text_key(row["utt"]) in train_keys for row in test_rows),
        "test_rows_with_normalized_dev_text": sum(text_key(row["utt"]) in dev_keys for row in test_rows),
        "test_rows_with_normalized_train_or_dev_text": len(test_rows) - len(clean_test),
        "phrase_disjoint_test_rows": len(clean_test),
        "phrase_disjoint_test_intent_classes": len({row["intent"] for row in clean_test}),
    }

    before_load = time.perf_counter()
    bundle = joblib.load(model_path)
    loaded_at = time.perf_counter()
    if bundle.get("source_sha256") != source_sha256 or len(bundle.get("labels", [])) != 60:
        raise ValueError("Saved model metadata does not match official source and labels")
    classifier = bundle["classifier"]
    vectorizer = bundle["vectorizer"]
    if len(classifier.classes_) != 60:
        raise ValueError("Saved classifier does not support all 60 train intents")
    predicted = classifier.predict(vectorizer.transform([model_text(row["utt"]) for row in test_rows]))
    predicted_at = time.perf_counter()
    predictions = {row["id"]: str(label) for row, label in zip(test_rows, predicted, strict=True)}
    clean_predictions = {row["id"]: predictions[row["id"]] for row in clean_test}
    full_metrics = score(test_rows, predictions, bundle["labels"])
    clean_metrics = score(clean_test, clean_predictions, bundle["labels"])

    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    with predictions_path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in test_rows:
            stream.write(
                json.dumps({"id": row["id"], "prediction": predictions[row["id"]]}, ensure_ascii=False)
                + "\n"
            )
    report = {
        "task": "MASSIVE 1.0 zh-CN intent classification",
        "evaluation_status": "locked-model test confirmation; not a pristine blind test",
        "prior_test_disclosure": (
            "The same official test split was previously viewed in the SFT study. "
            "This evaluation confirms the frozen linear baseline; it does not constitute "
            "a fully unseen final test. No training, candidate changes, or threshold tuning "
            "were performed here."
        ),
        "source_archive": str(source),
        "source_sha256": source_sha256,
        "split_sizes": split_sizes,
        "model_file": str(model_path),
        "locked_model_sha256": model_sha256,
        "model_bytes": model_path.stat().st_size,
        "dev_selection_report": str(dev_report_path),
        "duplicate_audit": duplicate_audit,
        "full_test": compact_metrics(full_metrics),
        "phrase_disjoint_test": compact_metrics(clean_metrics),
        "timing_seconds": {
            "model_load": round(loaded_at - before_load, 4),
            "vectorize_and_predict_2974_rows": round(predicted_at - loaded_at, 4),
            "total_including_archive_read_and_scoring": round(time.perf_counter() - started, 4),
        },
        "predictions_file": str(predictions_path),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"Full test: n={len(test_rows)}, accuracy={full_metrics['accuracy']:.4f}, "
        f"macro-F1={full_metrics['macro_f1']:.4f}; phrase-disjoint: "
        f"n={len(clean_test)}, accuracy={clean_metrics['accuracy']:.4f}, "
        f"macro-F1={clean_metrics['macro_f1']:.4f}. Report: {report_path}"
    )


if __name__ == "__main__":
    main()
