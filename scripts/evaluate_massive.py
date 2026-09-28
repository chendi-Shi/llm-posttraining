from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
from peft import PeftModel

from common import DEFAULT_MODEL, configure_cpu, load_quantized_base, load_tokenizer
from massive_metrics import read_jsonl, score
from massive_task import parse_prediction, user_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run constrained MASSIVE Chinese intent classification")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--eval-file", required=True)
    parser.add_argument("--intent-labels", default=None, help="Defaults to intents.json beside EVAL-FILE when present")
    parser.add_argument("--output", required=True, help="Per-example prediction JSONL")
    parser.add_argument("--metrics", default=None, help="Defaults to OUTPUT.metrics.json")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--max-examples", type=int, default=None, help="Optional smoke-check limit")
    parser.add_argument("--seed", type=int, default=20260924, help="Seed for a stratified smoke subset")
    return parser.parse_args()


def stratified_subset(rows: list[dict], limit: int, seed: int) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["intent"]].append(row)
    rng = random.Random(seed)
    for values in groups.values():
        rng.shuffle(values)
    selected = []
    labels = sorted(groups)
    while len(selected) < limit:
        added = False
        for label in labels:
            if groups[label] and len(selected) < limit:
                selected.append(groups[label].pop())
                added = True
        if not added:
            break
    return sorted(selected, key=lambda row: row["id"])


def main() -> None:
    args = parse_args()
    configure_cpu()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    rows = read_jsonl(args.eval_file)
    if args.max_examples is not None:
        if args.max_examples < 1:
            raise SystemExit("--max-examples must be positive")
        if args.max_examples < len(rows):
            rows = stratified_subset(rows, args.max_examples, args.seed)
    if not rows:
        raise SystemExit("Evaluation file is empty")

    labels_path = Path(args.intent_labels) if args.intent_labels else Path(args.eval_file).with_name("intents.json")
    if not labels_path.is_file():
        raise SystemExit(f"Intent label list not found: {labels_path}")
    intent_labels = json.loads(labels_path.read_text(encoding="utf-8"))
    if not isinstance(intent_labels, list) or not intent_labels:
        raise SystemExit(f"Intent label file must contain a nonempty JSON list: {labels_path}")

    tokenizer = load_tokenizer(args.model)
    tokenizer.padding_side = "left"
    label_token_ids = {
        label: tokenizer.encode(label, add_special_tokens=False)
        for label in intent_labels
    }
    if any(not token_ids for token_ids in label_token_ids.values()):
        raise SystemExit("One or more intent labels produced an empty token sequence")
    required_tokens = max(map(len, label_token_ids.values())) + 1
    if args.max_new_tokens < required_tokens:
        raise SystemExit(
            f"--max-new-tokens must be at least {required_tokens} to fit the longest label and EOS"
        )

    trie: dict = {}
    for token_ids in label_token_ids.values():
        node = trie
        for token_id in token_ids:
            node = node.setdefault(token_id, {})
        node[None] = True

    model = load_quantized_base(args.model, training=False)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
    model.eval()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    predictions = {}
    truncated_inputs = 0
    with output_path.open("w", encoding="utf-8", newline="\n") as stream:
        for offset in range(0, len(rows), args.batch_size):
            batch = rows[offset : offset + args.batch_size]
            conversations = [
                [{"role": "user", "content": user_prompt(row["utt"])}]
                for row in batch
            ]
            encoded = tokenizer.apply_chat_template(
                conversations,
                tokenize=True,
                add_generation_prompt=True,
                padding=True,
                return_tensors="pt",
                return_dict=True,
            )
            input_lengths = encoded["attention_mask"].sum(dim=1).tolist()
            oversized = [length > args.max_length for length in input_lengths]
            truncated_inputs += sum(oversized)
            if any(oversized):
                encoded = tokenizer.apply_chat_template(
                    conversations,
                    tokenize=True,
                    add_generation_prompt=True,
                    padding=True,
                    truncation=True,
                    max_length=args.max_length,
                    return_tensors="pt",
                    return_dict=True,
                )
            prompt_width = encoded["input_ids"].shape[1]

            def allowed_next_tokens(_batch_id, input_ids):
                node = trie
                for token_id in input_ids[prompt_width:].tolist():
                    if token_id not in node:
                        return [tokenizer.eos_token_id]
                    node = node[token_id]
                allowed = [token_id for token_id in node if token_id is not None]
                if None in node:
                    allowed.append(tokenizer.eos_token_id)
                return allowed

            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    prefix_allowed_tokens_fn=allowed_next_tokens,
                )
            continuation = generated[:, prompt_width:]
            for row, token_ids in zip(batch, continuation, strict=True):
                raw_output = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
                parsed = parse_prediction(raw_output, intent_labels)
                predictions[str(row["id"])] = parsed
                stream.write(
                    json.dumps(
                        {"id": str(row["id"]), "raw_output": raw_output, "prediction": parsed},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            stream.flush()
            done = min(offset + len(batch), len(rows))
            if done % 100 == 0 or done == len(rows):
                print(f"Generated {done}/{len(rows)} examples", flush=True)

    metrics = score(rows, predictions, intent_labels)
    metrics.update(
        {
            "model": args.model,
            "adapter": args.adapter,
            "eval_file": args.eval_file,
            "intent_labels_file": str(labels_path) if labels_path.is_file() else None,
            "predictions_file": str(output_path),
            "elapsed_seconds": round(time.time() - start, 2),
            "truncated_input_count": truncated_inputs,
            "truncated_input_rate": truncated_inputs / len(rows),
            "generation": {
                "do_sample": False,
                "constrained_to_intent_labels": True,
                "batch_size": args.batch_size,
                "max_input_tokens": args.max_length,
                "max_new_tokens": args.max_new_tokens,
                "smoke_sample_seed": args.seed if args.max_examples is not None else None,
            },
        }
    )
    metrics_path = Path(args.metrics) if args.metrics else output_path.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: metrics[key] for key in (
        "examples", "valid_label_rate", "unknown_intent_rate", "accuracy", "macro_f1",
        "truncated_input_count", "elapsed_seconds"
    )}, ensure_ascii=False, indent=2))
    print(f"Predictions: {output_path}")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
