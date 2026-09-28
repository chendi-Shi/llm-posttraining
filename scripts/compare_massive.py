from __future__ import annotations

import argparse
import json
from pathlib import Path

from massive_metrics import paired_bootstrap_delta, prediction_map, read_jsonl, score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired comparison of MASSIVE intent-classification predictions")
    parser.add_argument("--gold", required=True)
    parser.add_argument("--intent-labels", default=None, help="Defaults to intents.json beside GOLD")
    parser.add_argument("--base", required=True, help="Base model prediction JSONL")
    parser.add_argument("--candidate", required=True, help="SFT or DPO prediction JSONL")
    parser.add_argument("--base-name", default="base")
    parser.add_argument("--candidate-name", default="candidate")
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--allow-subset", action="store_true", help="Compare matching prediction IDs that are a subset of GOLD")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def compact(metrics: dict) -> dict:
    return {key: metrics[key] for key in (
        "examples", "scored_intent_classes", "valid_label_rate", "unknown_intent_rate",
        "accuracy", "macro_f1", "per_intent", "per_scenario"
    )}


def main() -> None:
    args = parse_args()
    gold_rows = read_jsonl(args.gold)
    labels_path = Path(args.intent_labels) if args.intent_labels else Path(args.gold).with_name("intents.json")
    if not labels_path.is_file():
        raise SystemExit(f"Intent label list not found: {labels_path}")
    intent_labels = json.loads(labels_path.read_text(encoding="utf-8"))
    base_predictions = prediction_map(args.base, intent_labels)
    candidate_predictions = prediction_map(args.candidate, intent_labels)
    if args.allow_subset:
        if set(base_predictions) != set(candidate_predictions):
            raise SystemExit("Base and candidate prediction IDs must match when comparing a subset")
        gold_by_id = {str(row["id"]): row for row in gold_rows}
        if not set(base_predictions).issubset(gold_by_id):
            raise SystemExit("Prediction IDs include examples not present in GOLD")
        gold_rows = [gold_by_id[item_id] for item_id in base_predictions]
    base_metrics = score(gold_rows, base_predictions, intent_labels)
    candidate_metrics = score(gold_rows, candidate_predictions, intent_labels)
    delta = paired_bootstrap_delta(
        gold_rows,
        base_predictions,
        candidate_predictions,
        iterations=args.bootstrap_iterations,
    )
    improvement = (
        delta["macro_f1_delta"] >= 0.02
        and delta["macro_f1_delta_ci95_stratified"][0] > 0
    )
    result = {
        "gold_file": args.gold,
        "intent_labels_file": str(labels_path),
        "base_predictions": args.base,
        "candidate_predictions": args.candidate,
        "allow_subset": args.allow_subset,
        "base_name": args.base_name,
        "candidate_name": args.candidate_name,
        "base": compact(base_metrics),
        "candidate": compact(candidate_metrics),
        "paired_delta": delta,
        "predefined_improvement_gate_met": improvement,
        "gate": "macro-F1 delta >= 0.02 and stratified paired 95% CI lower bound > 0",
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
