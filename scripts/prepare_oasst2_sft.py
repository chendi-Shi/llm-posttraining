from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path


DATASET_ID = "OpenAssistant/oasst2"
DATASET_URL = "https://huggingface.co/datasets/OpenAssistant/oasst2"
DATASET_REVISION = "179dd21fc55192153d94adb0e0ce8f69e222bf75"
DEFAULT_TOKENIZER = "models/Qwen2.5-0.5B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a small Chinese SFT subset from licensed OASST2 messages"
    )
    parser.add_argument("--output", default="data/sft.licensed.jsonl")
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--cache-dir", default="_tmp/hf-datasets")
    return parser.parse_args()


def approved_human_message(row: dict) -> bool:
    return (
        row.get("lang") == "zh"
        and row.get("review_result") is True
        and row.get("deleted") is False
        and row.get("synthetic") is False
        and isinstance(row.get("text"), str)
        and bool(row["text"].strip())
    )


def main() -> None:
    args = parse_args()
    os.environ.setdefault("HF_HOME", str(Path("_tmp/hf-home").resolve()))
    os.environ.setdefault("HF_DATASETS_CACHE", str(Path(args.cache_dir).resolve()))

    from datasets import load_dataset
    from transformers import AutoTokenizer

    revision = DATASET_REVISION
    print(f"Loading {DATASET_ID} at revision {revision}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    messages = load_dataset(
        DATASET_ID,
        split="train",
        revision=revision,
        cache_dir=args.cache_dir,
    )
    rows = {row["message_id"]: row for row in messages}
    eligible_ids = {
        row["message_id"]
        for row in messages
        if row.get("role") == "assistant"
        and approved_human_message(row)
        and row.get("rank") == 0
    }

    examples: list[dict] = []
    for message_id in sorted(eligible_ids):
        target = rows[message_id]
        chain = [target]
        parent_id = target.get("parent_id")
        valid = True
        while parent_id:
            parent = rows.get(parent_id)
            if parent is None or not approved_human_message(parent):
                valid = False
                break
            chain.append(parent)
            parent_id = parent.get("parent_id")
            if len(chain) > 12:
                valid = False
                break
        if not valid:
            continue

        chain.reverse()
        if not chain or chain[0].get("role") != "prompter":
            continue
        turns = [
            {
                "role": "user" if row["role"] == "prompter" else "assistant",
                "content": row["text"].strip(),
            }
            for row in chain
        ]
        if not turns or turns[-1]["role"] != "assistant":
            continue
        completion = turns.pop()
        if not turns or turns[-1]["role"] != "user":
            continue
        if sum(len(turn["content"]) for turn in turns) > 1800 or len(completion["content"]) > 1800:
            continue
        tokenized = tokenizer.apply_chat_template(
            turns + [completion], tokenize=True, add_generation_prompt=False
        )
        if len(tokenized["input_ids"]) > args.max_length:
            continue
        examples.append(
            {
                "prompt": turns,
                "completion": [completion],
                "source_message_id": message_id,
                "source_parent_id": target.get("parent_id"),
            }
        )

    unique: dict[str, dict] = {}
    for example in examples:
        unique[example["source_message_id"]] = example
    examples = list(unique.values())
    random.Random(args.seed).shuffle(examples)
    if len(examples) < args.count:
        raise SystemExit(
            f"Only {len(examples)} Chinese human-reviewed examples passed filters; "
            f"requested {args.count}. Reduce --count or review the filtering criteria."
        )
    selected = examples[: args.count]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as stream:
        for example in selected:
            stream.write(json.dumps(example, ensure_ascii=False) + "\n")

    output_sha256 = hashlib.sha256(output_path.read_bytes()).hexdigest()
    manifest = {
        "dataset": DATASET_ID,
        "dataset_url": DATASET_URL,
        "license": "Apache-2.0 (dataset card declaration)",
        "revision": revision,
        "split": "train",
        "filters": [
            "lang == zh",
            "review_result == true",
            "deleted == false",
            "synthetic == false",
            "assistant response rank == 0 (top-ranked reply)",
            "all messages in the prompt/answer chain pass the same filters",
            "prompt/answer chain is at most 12 messages",
            "prompt and target are each at most 1800 Unicode characters",
            f"full chat-template sequence is at most {args.max_length} tokens with tokenizer {args.tokenizer}",
        ],
        "eligible_examples": len(examples),
        "selected_examples": len(selected),
        "seed": args.seed,
        "tokenizer": args.tokenizer,
        "max_length": args.max_length,
        "output_sha256": output_sha256,
        "output": str(output_path),
        "source_message_ids": [example["source_message_id"] for example in selected],
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Eligible examples: {len(examples)}")
    print(f"Wrote {len(selected)} examples to {output_path}")
    print(f"Wrote provenance manifest to {manifest_path}")


if __name__ == "__main__":
    main()
