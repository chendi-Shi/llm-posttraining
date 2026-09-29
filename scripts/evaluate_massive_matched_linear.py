"""Compare the existing 594-row SFT with a linear model using the same rows.

This is an exploratory development-set diagnostic. It never reads test data or
selects hyperparameters from the resulting metric.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from evaluate_massive import stratified_subset
from massive_linear_text import model_text, text_key
from massive_metrics import score


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def compact(metrics: dict) -> dict:
    return {key: metrics[key] for key in ("examples", "scored_intent_classes", "accuracy", "macro_f1", "per_intent")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/massive-zh/train.jsonl"))
    parser.add_argument("--dev", type=Path, default=Path("data/massive-zh/dev.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("reports/massive-matched-594-linear-dev.json"))
    args = parser.parse_args()
    train = read_jsonl(args.train)
    dev = read_jsonl(args.dev)
    if len(train) != 594 or len(dev) != 2033:
        raise SystemExit(f"Unexpected MASSIVE split sizes: {len(train)} train, {len(dev)} dev")
    selected_dev = stratified_subset(dev, 300, 20260924)
    train_keys = {text_key(row["utt"]) for row in train}
    clean_dev = [row for row in selected_dev if text_key(row["utt"]) not in train_keys]
    labels = sorted({row["intent"] for row in train})
    if len(labels) != 60:
        raise SystemExit(f"Expected 60 intent labels, got {len(labels)}")

    vectorizer = TfidfVectorizer(
        analyzer="char", ngram_range=(1, 3), min_df=2,
        sublinear_tf=True, dtype=np.float32,
    )
    x_train = vectorizer.fit_transform([model_text(row["utt"]) for row in train])
    classifier = LinearSVC(C=1.0, dual="auto", max_iter=3000, tol=1e-3, random_state=20260924)
    classifier.fit(x_train, [row["intent"] for row in train])
    predicted = classifier.predict(vectorizer.transform([model_text(row["utt"]) for row in selected_dev]))
    predictions = {str(row["id"]): str(label) for row, label in zip(selected_dev, predicted, strict=True)}
    full = score(selected_dev, predictions, labels)
    clean = score(clean_dev, {str(row["id"]): predictions[str(row["id"])] for row in clean_dev}, labels)

    report = {
        "task": "MASSIVE 1.0 zh-CN intent classification, matched 594-row training comparison",
        "development_only": True,
        "test_used": False,
        "source_train_sha256": sha256(args.train),
        "source_dev_sha256": sha256(args.dev),
        "train_rows": len(train),
        "selected_dev_rows": len(selected_dev),
        "phrase_disjoint_dev_rows": len(clean_dev),
        "dev_selection": "same 300-row stratified seed 20260924 as the prior SFT experiment",
        "fixed_algorithm": "char 1-3 gram TF-IDF (min_df=2, sublinear_tf) + LinearSVC(C=1)",
        "algorithm_was_used_in_prior_full_train_baseline": True,
        "full_dev_300": compact(full),
        "phrase_disjoint_dev_300": compact(clean),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    if args.report.exists():
        raise FileExistsError(f"Refusing to overwrite {args.report}")
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"train": len(train), "dev": len(selected_dev), "clean_dev": len(clean_dev),
                      "full_accuracy": full["accuracy"], "full_macro_f1": full["macro_f1"],
                      "clean_accuracy": clean["accuracy"], "clean_macro_f1": clean["macro_f1"]}, indent=2))


if __name__ == "__main__":
    main()
