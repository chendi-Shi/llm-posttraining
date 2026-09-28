"""Train a CPU-only character TF-IDF baseline on MASSIVE zh-CN train.

This script reads the official archive, uses only its train/dev partitions, and
never scores or writes predictions for test. Candidate selection uses the dev
rows whose normalized utterance does not occur in train.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from massive_linear_text import model_text, text_key
from massive_metrics import score


EXPECTED = {"train": 11514, "dev": 2033}
CANDIDATES = (
    ("char_1_3", (1, 3), None),
    ("char_2_4_balanced", (2, 4), "balanced"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default="data/raw/amazon-massive-dataset-1.0.tar.gz",
        help="Official MASSIVE 1.0 archive",
    )
    parser.add_argument("--report", default="reports/massive-linear-dev.json")
    parser.add_argument("--predictions", default="reports/massive-linear-dev.jsonl")
    parser.add_argument("--model-out", default="outputs/massive-linear-baseline.joblib")
    parser.add_argument("--seed", type=int, default=20260924)
    return parser.parse_args()


def source_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_train_dev(path: Path) -> dict[str, list[dict]]:
    rows: dict[str, list[dict]] = {"train": [], "dev": []}
    with tarfile.open(path, "r:*") as archive:
        members = [member for member in archive.getmembers() if member.name.endswith("zh-CN.jsonl")]
        if len(members) != 1:
            raise ValueError(f"Expected one zh-CN.jsonl in {path}, found {len(members)}")
        member_stream = archive.extractfile(members[0])
        if member_stream is None:
            raise ValueError("Could not read zh-CN.jsonl from archive")
        with member_stream, io.TextIOWrapper(member_stream, encoding="utf-8") as stream:
            for line in stream:
                item = json.loads(line)
                split = item.get("partition")
                if split not in rows:
                    continue
                if item.get("locale") != "zh-CN":
                    raise ValueError(f"Unexpected locale in {split}: {item.get('locale')}")
                rows[split].append(
                    {
                        "id": str(item["id"]),
                        "scenario": str(item["scenario"]),
                        "utt": str(item["utt"]),
                        "intent": str(item["intent"]),
                    }
                )
    observed = {split: len(values) for split, values in rows.items()}
    if observed != EXPECTED:
        raise ValueError(f"Unexpected MASSIVE zh-CN train/dev sizes: {observed}")
    train_ids = {item["id"] for item in rows["train"]}
    dev_ids = {item["id"] for item in rows["dev"]}
    if len(train_ids) != EXPECTED["train"] or len(dev_ids) != EXPECTED["dev"]:
        raise ValueError("Duplicate IDs within train or dev")
    if train_ids & dev_ids:
        raise ValueError("Train and dev IDs overlap")
    return rows


def compact_metrics(metrics: dict) -> dict:
    return {
        key: metrics[key]
        for key in (
            "examples",
            "scored_intent_classes",
            "accuracy",
            "macro_f1",
            "per_intent",
            "per_scenario",
        )
    }


def main() -> None:
    args = parse_args()
    source = Path(args.source)
    if not source.is_file():
        raise SystemExit(f"Source archive not found: {source}")
    splits = read_train_dev(source)
    train_rows, dev_rows = splits["train"], splits["dev"]
    labels = sorted({row["intent"] for row in train_rows})
    if len(labels) != 60:
        raise ValueError(f"Expected 60 train intents, found {len(labels)}")
    if not {row["intent"] for row in dev_rows}.issubset(labels):
        raise ValueError("Dev contains an intent unseen in train")

    train_raw = {row["utt"].strip() for row in train_rows}
    train_keys = {text_key(row["utt"]) for row in train_rows}
    dev_raw = {row["utt"].strip() for row in dev_rows}
    dev_keys = {text_key(row["utt"]) for row in dev_rows}
    if "" in train_keys or "" in dev_keys:
        raise ValueError("Empty normalized utterance")
    clean_dev = [row for row in dev_rows if text_key(row["utt"]) not in train_keys]
    if not clean_dev:
        raise ValueError("No phrase-disjoint dev rows")
    duplicate_audit = {
        "train_unique_exact_texts": len(train_raw),
        "dev_unique_exact_texts": len(dev_raw),
        "train_dev_shared_exact_texts": len(train_raw & dev_raw),
        "dev_rows_with_exact_train_text": sum(row["utt"].strip() in train_raw for row in dev_rows),
        "train_dev_shared_normalized_texts": len(train_keys & dev_keys),
        "dev_rows_with_normalized_train_text": len(dev_rows) - len(clean_dev),
        "clean_dev_rows": len(clean_dev),
        "clean_dev_intent_classes": len({row["intent"] for row in clean_dev}),
    }

    texts_train = [model_text(row["utt"]) for row in train_rows]
    texts_dev = [model_text(row["utt"]) for row in dev_rows]
    targets = [row["intent"] for row in train_rows]
    best = None
    candidates = []
    for name, ngram_range, class_weight in CANDIDATES:
        print(f"Fitting {name} on {len(train_rows)} train rows", flush=True)
        began = time.perf_counter()
        vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=ngram_range,
            min_df=2,
            sublinear_tf=True,
            dtype=np.float32,
        )
        x_train = vectorizer.fit_transform(texts_train)
        x_dev = vectorizer.transform(texts_dev)
        classifier = LinearSVC(
            C=1.0,
            class_weight=class_weight,
            dual="auto",
            max_iter=3000,
            tol=1e-3,
            random_state=args.seed,
        )
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always", ConvergenceWarning)
            classifier.fit(x_train, targets)
        predicted = classifier.predict(x_dev)
        elapsed = round(time.perf_counter() - began, 2)
        predictions = {row["id"]: str(label) for row, label in zip(dev_rows, predicted, strict=True)}
        clean_predictions = {row["id"]: predictions[row["id"]] for row in clean_dev}
        full_metrics = score(dev_rows, predictions, labels)
        clean_metrics = score(clean_dev, clean_predictions, labels)
        candidate = {
            "name": name,
            "ngram_range": list(ngram_range),
            "class_weight": class_weight,
            "C": 1.0,
            "vocabulary_size": len(vectorizer.vocabulary_),
            "train_nonzero_features": int(x_train.nnz),
            "fit_and_dev_predict_seconds": elapsed,
            "convergence_warnings": [str(item.message) for item in captured if issubclass(item.category, ConvergenceWarning)],
            "full_dev": compact_metrics(full_metrics),
            "phrase_disjoint_dev": compact_metrics(clean_metrics),
        }
        candidates.append(candidate)
        print(
            f"{name}: clean dev macro-F1={clean_metrics['macro_f1']:.4f}, "
            f"accuracy={clean_metrics['accuracy']:.4f}; "
            f"full dev macro-F1={full_metrics['macro_f1']:.4f}",
            flush=True,
        )
        ranking = (clean_metrics["macro_f1"], clean_metrics["accuracy"])
        if best is None or ranking > best["ranking"]:
            best = {
                "ranking": ranking,
                "name": name,
                "vectorizer": vectorizer,
                "classifier": classifier,
                "predictions": predictions,
            }
    assert best is not None

    report_path = Path(args.report)
    predictions_path = Path(args.predictions)
    model_path = Path(args.model_out)
    for path in (report_path, predictions_path, model_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite an existing result: {path}")
    joblib.dump(
        {
            "vectorizer": best["vectorizer"],
            "classifier": best["classifier"],
            "labels": labels,
            "normalizer": "NFKC-casefold-strip",
            "source_sha256": source_sha256(source),
        },
        model_path,
    )
    with predictions_path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in dev_rows:
            stream.write(
                json.dumps(
                    {"id": row["id"], "prediction": best["predictions"][row["id"]]},
                    ensure_ascii=False,
                )
                + "\n"
            )
    report = {
        "task": "MASSIVE 1.0 zh-CN intent classification",
        "evaluated_partitions": ["train", "dev"],
        "test_used": False,
        "source_archive": str(source),
        "source_sha256": source_sha256(source),
        "train_rows": len(train_rows),
        "dev_rows": len(dev_rows),
        "train_intent_classes": len(labels),
        "duplicate_audit": duplicate_audit,
        "model_selection_metric": "phrase-disjoint dev macro-F1, accuracy tiebreaker",
        "candidate_selection_is_exploratory": True,
        "candidates": candidates,
        "selected_candidate": best["name"],
        "predictions_file": str(predictions_path),
        "saved_model": str(model_path),
        "model_bytes": model_path.stat().st_size,
        "versions": {"numpy": np.__version__, "scikit_learn": __import__("sklearn").__version__},
        "seed": args.seed,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Selected {best['name']}; report: {report_path}; model: {model_path}")


if __name__ == "__main__":
    main()
