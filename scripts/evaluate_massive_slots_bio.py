"""Matched-data character BIO baseline for four MASSIVE Chinese slots.

This script trains only on the locked 384-row training file and evaluates only
the 100-row development file. The 200-row confirmation file is never opened.
"""

from __future__ import annotations

import argparse
import json
import time
import unicodedata
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import LinearSVC

from massive_slots import canonical_output, parse_annotated_spans
from prepare_massive_slots import file_sha256, read_archive


SEED = 20260928
MODEL_CONFIG = {"estimator": "LinearSVC", "C": 1.0, "class_weight": "balanced", "max_iter": 5000}


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_predictions(
    path: Path, rows: list[dict], raw_predictions: list[str], eval_file_sha256: str
) -> None:
    """Save row-level outputs only in the caller's ignored local path."""
    if len(rows) != len(raw_predictions):
        raise ValueError("Prediction count differs from evaluation rows")
    if len({str(row["id"]) for row in rows}) != len(rows):
        raise ValueError("Duplicate evaluation ID")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row, raw_output in zip(rows, raw_predictions, strict=True):
            if not isinstance(raw_output, str) or not row.get("group_sha256"):
                raise ValueError("Prediction or group hash is missing")
            record = {
                "id": str(row["id"]),
                "group_sha256": row["group_sha256"],
                "eval_file_sha256": eval_file_sha256,
                "raw_output": raw_output,
            }
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def contextual_features(utterance: str, index: int) -> dict[str, str | bool]:
    char = utterance[index]
    features: dict[str, str | bool] = {
        "bias": True,
        "category": unicodedata.category(char),
        "digit": char.isdigit(),
        "space": char.isspace(),
        "latin": char.isascii() and char.isalpha(),
        "first": index == 0,
        "last": index == len(utterance) - 1,
    }
    for offset in (-2, -1, 0, 1, 2):
        position = index + offset
        features[f"c{offset:+d}"] = (
            utterance[position] if 0 <= position < len(utterance) else "<BOUNDARY>"
        )
    for left, right in ((-2, 0), (-1, 0), (0, 1), (0, 2), (-1, 1)):
        start, stop = index + left, index + right + 1
        if 0 <= start and stop <= len(utterance):
            features[f"gram{left:+d}:{right:+d}"] = utterance[start:stop]
    return features


def gold_bio_tags(utterance: str, spans: list[dict]) -> list[str]:
    tags = ["O"] * len(utterance)
    for span in spans:
        start, end, slot_type = span["start"], span["end"], span["type"]
        if not (0 <= start < end <= len(utterance)):
            raise ValueError("Invalid gold slot offsets")
        if any(tag != "O" for tag in tags[start:end]):
            raise ValueError("Overlapping gold slots")
        tags[start] = f"B-{slot_type}"
        for index in range(start + 1, end):
            tags[index] = f"I-{slot_type}"
    return tags


def _legal(previous: str | None, current: str) -> bool:
    if not current.startswith("I-"):
        return True
    slot_type = current[2:]
    return previous in {f"B-{slot_type}", f"I-{slot_type}"}


def constrained_bio_decode(scores: np.ndarray, classes: list[str]) -> list[str]:
    """Choose the best sequence under hard BIO transition constraints."""
    if scores.ndim != 2 or scores.shape[1] != len(classes):
        raise ValueError("Expected one score per character and BIO class")
    if scores.shape[0] == 0:
        return []
    width = len(classes)
    best = np.full((scores.shape[0], width), -np.inf)
    back = np.full((scores.shape[0], width), -1, dtype=np.int32)
    for current, label in enumerate(classes):
        if _legal(None, label):
            best[0, current] = scores[0, current]
    for position in range(1, scores.shape[0]):
        for current, label in enumerate(classes):
            candidates = [
                previous
                for previous, previous_label in enumerate(classes)
                if _legal(previous_label, label)
            ]
            if not candidates:
                continue
            previous = max(candidates, key=lambda candidate: best[position - 1, candidate])
            best[position, current] = best[position - 1, previous] + scores[position, current]
            back[position, current] = previous
    current = int(np.argmax(best[-1]))
    if not np.isfinite(best[-1, current]):
        raise ValueError("No legal BIO sequence")
    path = [current]
    for position in range(scores.shape[0] - 1, 0, -1):
        current = int(back[position, current])
        path.append(current)
    return [classes[index] for index in reversed(path)]


def slots_from_bio(utterance: str, tags: list[str]) -> list[dict[str, str]]:
    if len(utterance) != len(tags):
        raise ValueError("One BIO tag is required for each character")
    slots: list[dict[str, str]] = []
    active_type: str | None = None
    start = 0

    def close(end: int) -> None:
        if active_type is not None:
            slots.append({"type": active_type, "value": utterance[start:end]})

    for index, tag in enumerate(tags):
        if tag == "O":
            close(index)
            active_type = None
            continue
        if len(tag) < 3 or tag[1] != "-" or tag[0] not in {"B", "I"}:
            raise ValueError(f"Invalid BIO tag {tag!r}")
        slot_type = tag[2:]
        if tag[0] == "I" and active_type == slot_type:
            continue
        # Also repair an isolated I-type deterministically if called without
        # the constrained decoder.
        close(index)
        active_type, start = slot_type, index
    close(len(utterance))
    return slots


def align_gold(rows: list[dict], source_by_id: dict[str, dict]) -> list[list[dict]]:
    aligned = []
    for row in rows:
        identifier = str(row["id"])
        source = source_by_id.get(identifier)
        if source is None or source["partition"] != "train" or source["utt"] != row["utt"]:
            raise ValueError(f"Selected row does not match official train ID {identifier}")
        spans = parse_annotated_spans(row["utt"], source["annot_utt"])
        stripped = [{"type": span["type"], "value": span["value"]} for span in spans]
        if stripped != row["slots"]:
            raise ValueError(f"Selected slot labels differ from official annotation: {identifier}")
        aligned.append(spans)
    return aligned


def train_model(rows: list[dict], gold_spans: list[list[dict]]) -> tuple[DictVectorizer, LinearSVC]:
    examples = [
        contextual_features(row["utt"], index)
        for row in rows
        for index in range(len(row["utt"]))
    ]
    labels = [
        tag
        for row, spans in zip(rows, gold_spans)
        for tag in gold_bio_tags(row["utt"], spans)
    ]
    if len(examples) != len(labels) or not examples:
        raise ValueError("Empty or misaligned training characters")
    vectorizer = DictVectorizer(sparse=True)
    matrix = vectorizer.fit_transform(examples)
    model = LinearSVC(
        C=MODEL_CONFIG["C"],
        class_weight=MODEL_CONFIG["class_weight"],
        max_iter=MODEL_CONFIG["max_iter"],
        random_state=SEED,
    )
    model.fit(matrix, labels)
    return vectorizer, model


def predict_slots(
    utterance: str, vectorizer: DictVectorizer, model: LinearSVC
) -> list[dict[str, str]]:
    if not utterance:
        return []
    matrix = vectorizer.transform(
        [contextual_features(utterance, index) for index in range(len(utterance))]
    )
    scores = model.decision_function(matrix)
    if scores.ndim == 1:
        raise ValueError("Training must include more than two BIO classes")
    tags = constrained_bio_decode(scores, list(model.classes_))
    return slots_from_bio(utterance, tags)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path, default=Path("data/massive-zh/slots/train.jsonl"))
    parser.add_argument("--dev-file", type=Path, default=Path("data/massive-zh/slots/dev.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("data/massive-zh/slots/manifest.json"))
    parser.add_argument("--archive", type=Path, default=Path("data/raw/amazon-massive-dataset-1.0.tar.gz"))
    parser.add_argument("--model-out", type=Path, default=Path("outputs/massive-slots-bio.joblib"))
    parser.add_argument("--report", type=Path, default=Path("reports/massive-slots-bio-dev.json"))
    parser.add_argument(
        "--predictions-out", type=Path,
        default=Path("_tmp/massive-slots-bio-dev-predictions.jsonl"),
        help="Ignored local JSONL for paired model comparison",
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    for split, path, expected in (
        ("train", args.train_file, 384),
        ("dev", args.dev_file, 100),
    ):
        if file_sha256(path) != manifest["splits"][split]["jsonl_sha256"]:
            raise SystemExit(f"{split} file differs from the locked manifest")
        if manifest["splits"][split]["examples"] != expected:
            raise SystemExit(f"Unexpected {split} size in manifest")
    if file_sha256(args.archive) != manifest["source_sha256"]:
        raise SystemExit("Official source archive differs from the locked manifest")
    train_rows = read_jsonl(args.train_file)
    dev_rows = read_jsonl(args.dev_file)
    if len(train_rows) != 384 or len(dev_rows) != 100:
        raise SystemExit("Expected 384 training and 100 development rows")
    if {row["group_sha256"] for row in train_rows} & {row["group_sha256"] for row in dev_rows}:
        raise SystemExit("Train/dev phrase-group leakage")
    archive_rows, _ = read_archive(args.archive)
    source_by_id = {str(row["id"]): row for row in archive_rows if row["partition"] == "train"}
    train_spans = align_gold(train_rows, source_by_id)
    align_gold(dev_rows, source_by_id)

    start = time.perf_counter()
    vectorizer, model = train_model(train_rows, train_spans)
    train_seconds = time.perf_counter() - start
    raw_predictions = [
        canonical_output(predict_slots(row["utt"], vectorizer, model))
        for row in dev_rows
    ]
    write_predictions(
        args.predictions_out,
        dev_rows,
        raw_predictions,
        manifest["splits"]["dev"]["jsonl_sha256"],
    )
    from evaluate_massive_slots import score_predictions

    metrics = score_predictions(dev_rows, raw_predictions, bootstrap_samples=1000, seed=SEED)
    report = {
        "task": "MASSIVE 1.0 zh-CN four-slot JSON extraction",
        "model": "character BIO contextual features + LinearSVC + constrained BIO decoding",
        "training_configuration": MODEL_CONFIG,
        "data_license": "CC-BY-4.0",
        "source_sha256": manifest["source_sha256"],
        "train_file_sha256": manifest["splits"]["train"]["jsonl_sha256"],
        "dev_file_sha256": manifest["splits"]["dev"]["jsonl_sha256"],
        "train_examples": len(train_rows),
        "dev_examples": len(dev_rows),
        "train_characters": sum(len(row["utt"]) for row in train_rows),
        "train_bio_classes": dict(Counter(tag for row, spans in zip(train_rows, train_spans) for tag in gold_bio_tags(row["utt"], spans))),
        "train_seconds": train_seconds,
        "confirmation_used": False,
        "prediction_file_sha256": file_sha256(args.predictions_out),
        "metrics": metrics,
    }
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "vectorizer": vectorizer,
            "model": model,
            "configuration": MODEL_CONFIG,
            "train_file_sha256": manifest["splits"]["train"]["jsonl_sha256"],
            "source_sha256": manifest["source_sha256"],
        },
        args.model_out,
    )
    report["model_sha256"] = file_sha256(args.model_out)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"train_examples": len(train_rows), "dev_examples": len(dev_rows), "train_seconds": round(train_seconds, 2), "model_sha256": report["model_sha256"], "report": str(args.report)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
