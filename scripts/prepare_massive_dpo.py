from __future__ import annotations

import argparse
import json
from pathlib import Path

from massive_metrics import prediction_map, read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build intent-label DPO pairs from SFT errors on training data only")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--predictions", required=True, help="SFT generations for the same training rows")
    parser.add_argument("--output", default="data/massive-zh/dpo.jsonl")
    parser.add_argument("--minimum-pairs", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_rows = read_jsonl(args.train_file)
    prediction_rows = read_jsonl(args.predictions)
    labels_path = Path(args.train_file).with_name("intents.json")
    intent_labels = json.loads(labels_path.read_text(encoding="utf-8"))
    predictions = prediction_map(args.predictions, intent_labels)
    raw_outputs = {str(row["id"]): row.get("raw_output", "") for row in prediction_rows}
    train_ids = {str(row["id"]) for row in train_rows}
    if set(predictions) != train_ids:
        raise SystemExit(
            f"Prediction IDs must match training IDs exactly: "
            f"{len(train_ids - set(predictions))} missing, {len(set(predictions) - train_ids)} extra"
        )
    pairs = []
    for row in train_rows:
        item_id = str(row["id"])
        prediction = predictions[item_id]
        rejected = raw_outputs[item_id]
        if prediction == row["intent"] or not rejected.strip():
            continue
        pairs.append(
            {
                "prompt": row["prompt"],
                "chosen": [{"role": "assistant", "content": row["intent"]}],
                "rejected": [{"role": "assistant", "content": rejected.strip()}],
                "source_id": item_id,
            }
        )
    if len(pairs) < args.minimum_pairs:
        raise SystemExit(
            f"Only {len(pairs)} distinct SFT errors are available; at least {args.minimum_pairs} are required. "
            "No DPO file was written."
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for pair in pairs:
            stream.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"Wrote {len(pairs)} train-only preference pairs to {output}")


if __name__ == "__main__":
    main()
