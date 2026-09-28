from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from massive_task import parse_prediction


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSONL at {path}:{line_number}: {error}") from error
    return rows


def prediction_map(path: str | Path, intent_labels: list[str] | None = None) -> dict[str, str | None]:
    predictions = {}
    for row in read_jsonl(path):
        item_id = str(row["id"])
        if item_id in predictions:
            raise ValueError(f"Duplicate prediction ID {item_id} in {path}")
        prediction = row.get("prediction")
        if prediction is None and "raw_output" in row:
            prediction = parse_prediction(row["raw_output"], intent_labels or [])
        elif isinstance(prediction, str) and intent_labels is not None:
            prediction = prediction if prediction in intent_labels else None
        elif not isinstance(prediction, str):
            prediction = None
        predictions[item_id] = prediction
    return predictions


def score(
    gold_rows: list[dict],
    predictions: dict[str, str | None],
    known_intents: list[str] | None = None,
) -> dict:
    gold_ids = [str(row["id"]) for row in gold_rows]
    if len(gold_ids) != len(set(gold_ids)):
        raise ValueError("Duplicate IDs in gold evaluation data")
    missing = set(gold_ids) - set(predictions)
    extra = set(predictions) - set(gold_ids)
    if missing or extra:
        raise ValueError(f"Prediction IDs mismatch: {len(missing)} missing, {len(extra)} extra")
    if not gold_rows:
        raise ValueError("Gold evaluation data is empty")

    classes = sorted({row["intent"] for row in gold_rows})
    known = set(known_intents or classes)
    true_counts = Counter()
    predicted_counts = Counter()
    true_positive = Counter()
    scenario_stats = defaultdict(lambda: {"n": 0, "correct": 0})
    correct = 0
    valid = 0
    unknown = 0
    correct_by_id = {}
    per_class_confusion = Counter()

    for row in gold_rows:
        item_id = str(row["id"])
        gold = row["intent"]
        pred = predictions[item_id]
        is_valid = pred in known if pred is not None else False
        valid += int(is_valid)
        unknown += int(pred is not None and pred not in known)
        is_correct = pred == gold
        correct += int(is_correct)
        correct_by_id[item_id] = int(is_correct)
        true_counts[gold] += 1
        if pred in classes:
            predicted_counts[pred] += 1
        if is_correct:
            true_positive[gold] += 1
        per_class_confusion[(gold, pred if pred is not None else "<INVALID>")] += 1
        scenario = scenario_stats[row["scenario"]]
        scenario["n"] += 1
        scenario["correct"] += int(is_correct)

    per_intent = {}
    f1_values = []
    for intent in classes:
        tp = true_positive[intent]
        precision_denominator = predicted_counts[intent]
        recall_denominator = true_counts[intent]
        precision = tp / precision_denominator if precision_denominator else 0.0
        recall = tp / recall_denominator if recall_denominator else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_intent[intent] = {
            "support": recall_denominator,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    macro_f1 = sum(f1_values) / len(f1_values) if f1_values else 0.0

    return {
        "examples": len(gold_rows),
        "scored_intent_classes": len(classes),
        "valid_label_rate": valid / len(gold_rows),
        "unknown_intent_rate": unknown / len(gold_rows),
        "accuracy": correct / len(gold_rows),
        "macro_f1": macro_f1,
        "per_intent": per_intent,
        "per_scenario": {
            name: {"examples": stats["n"], "accuracy": stats["correct"] / stats["n"]}
            for name, stats in sorted(scenario_stats.items())
        },
        "confusion": {
            f"{actual} -> {predicted}": count
            for (actual, predicted), count in sorted(per_class_confusion.items())
            if actual != predicted
        },
        "correct_by_id": correct_by_id,
    }


def _macro_f1_from_confusion(
    confusion: Counter[tuple[str, str]], classes: list[str], supports: Counter[str]
) -> float:
    predicted_counts = Counter()
    for (_, predicted), count in confusion.items():
        predicted_counts[predicted] += count
    values = []
    for intent in classes:
        tp = confusion[(intent, intent)]
        predicted = predicted_counts[intent]
        denominator = supports[intent] + predicted
        values.append(2 * tp / denominator if denominator else 0.0)
    return sum(values) / len(values) if values else 0.0


def paired_bootstrap_delta(
    gold_rows: list[dict],
    base: dict[str, str | None],
    candidate: dict[str, str | None],
    *,
    iterations: int = 2000,
    seed: int = 20260924,
) -> dict:
    if iterations < 1:
        raise ValueError("Bootstrap iteration count must be positive")
    classes = sorted({row["intent"] for row in gold_rows})
    grouped_indices: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(gold_rows):
        grouped_indices[row["intent"]].append(index)
    supports = Counter({label: len(indices) for label, indices in grouped_indices.items()})
    rng = random.Random(seed)
    macro_deltas = []
    accuracy_deltas = []

    for _ in range(iterations):
        base_confusion: Counter[tuple[str, str]] = Counter()
        candidate_confusion: Counter[tuple[str, str]] = Counter()
        base_correct = candidate_correct = sampled_count = 0
        for indices in grouped_indices.values():
            for _ in indices:
                index = rng.choice(indices)
                row = gold_rows[index]
                item_id = str(row["id"])
                actual = row["intent"]
                base_label = base[item_id] or "<INVALID>"
                candidate_label = candidate[item_id] or "<INVALID>"
                base_confusion[(actual, base_label)] += 1
                candidate_confusion[(actual, candidate_label)] += 1
                base_correct += int(base_label == actual)
                candidate_correct += int(candidate_label == actual)
                sampled_count += 1
        macro_deltas.append(
            _macro_f1_from_confusion(candidate_confusion, classes, supports)
            - _macro_f1_from_confusion(base_confusion, classes, supports)
        )
        accuracy_deltas.append((candidate_correct - base_correct) / sampled_count)

    def interval(values: list[float]) -> list[float]:
        ordered = sorted(values)
        low = ordered[int(0.025 * iterations)]
        high = ordered[min(iterations - 1, int(0.975 * iterations))]
        return [low, high]

    base_score = score(gold_rows, base)
    candidate_score = score(gold_rows, candidate)
    return {
        "iterations": iterations,
        "seed": seed,
        "macro_f1_delta": candidate_score["macro_f1"] - base_score["macro_f1"],
        "macro_f1_delta_ci95_stratified": interval(macro_deltas),
        "accuracy_delta": candidate_score["accuracy"] - base_score["accuracy"],
        "accuracy_delta_ci95_stratified": interval(accuracy_deltas),
    }
