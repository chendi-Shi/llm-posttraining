from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import defaultdict
from pathlib import Path


DATASET_ID = "OpenAssistant/oasst2"
DATASET_URL = "https://huggingface.co/datasets/OpenAssistant/oasst2"
DATASET_REVISION = "179dd21fc55192153d94adb0e0ce8f69e222bf75"
DEFAULT_TOKENIZER = "models/Qwen2.5-0.5B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Chinese OASST2 rank-based preference pairs for DPO"
    )
    parser.add_argument("--output", default="data/dpo.licensed.jsonl")
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--split", choices=("train", "validation"), default="train")
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

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    revision = DATASET_REVISION
    print(f"Loading {DATASET_ID} at revision {revision}")
    messages = load_dataset(
        DATASET_ID,
        split=args.split,
        revision=revision,
        cache_dir=args.cache_dir,
    )
    rows = {row["message_id"]: row for row in messages}
    by_parent: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in messages:
        if (
            row.get("role") == "assistant"
            and approved_human_message(row)
            and row.get("rank") in (0, 1)
            and row.get("parent_id")
        ):
            by_parent[row["parent_id"]][row["rank"]].append(row)

    examples: list[dict] = []
    for parent_id in sorted(by_parent):
        parent = rows.get(parent_id)
        if (
            parent is None
            or parent.get("role") != "prompter"
            or not approved_human_message(parent)
        ):
            continue
        ranked = by_parent[parent_id]
        if not ranked.get(0) or not ranked.get(1):
            continue

        chain = [parent]
        ancestor_id = parent.get("parent_id")
        valid = True
        while ancestor_id:
            ancestor = rows.get(ancestor_id)
            if ancestor is None or not approved_human_message(ancestor):
                valid = False
                break
            chain.append(ancestor)
            ancestor_id = ancestor.get("parent_id")
            if len(chain) > 12:
                valid = False
                break
        if not valid:
            continue
        chain.reverse()
        if not chain or chain[0].get("role") != "prompter":
            continue
        prompt = [
            {
                "role": "user" if row["role"] == "prompter" else "assistant",
                "content": row["text"].strip(),
            }
            for row in chain
        ]
        if not prompt or prompt[-1]["role"] != "user":
            continue
        if sum(len(turn["content"]) for turn in prompt) > 1800:
            continue

        chosen = sorted(ranked[0], key=lambda row: row["message_id"])[0]
        rejected = sorted(ranked[1], key=lambda row: row["message_id"])[0]
        if len(chosen["text"]) > 1800 or len(rejected["text"]) > 1800:
            continue
        if chosen["text"].strip() == rejected["text"].strip():
            continue
        chosen_turn = {"role": "assistant", "content": chosen["text"].strip()}
        rejected_turn = {"role": "assistant", "content": rejected["text"].strip()}
        chosen_tokens = tokenizer.apply_chat_template(
            prompt + [chosen_turn], tokenize=True, add_generation_prompt=False
        )
        rejected_tokens = tokenizer.apply_chat_template(
            prompt + [rejected_turn], tokenize=True, add_generation_prompt=False
        )
        if max(len(chosen_tokens["input_ids"]), len(rejected_tokens["input_ids"])) > args.max_length:
            continue
        examples.append(
            {
                "prompt": prompt,
                "chosen": [chosen_turn],
                "rejected": [rejected_turn],
                "source_parent_id": parent_id,
                "source_chosen_message_id": chosen["message_id"],
                "source_rejected_message_id": rejected["message_id"],
                "source_chosen_rank": 0,
                "source_rejected_rank": 1,
            }
        )

    unique = {example["source_parent_id"]: example for example in examples}
    examples = list(unique.values())
    random.Random(args.seed).shuffle(examples)
    if len(examples) < args.count:
        raise SystemExit(
            f"Only {len(examples)} Chinese rank-0/rank-1 pairs passed filters; "
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
        "split": args.split,
        "filters": [
            "lang == zh",
            "review_result == true",
            "deleted == false",
            "synthetic == false",
            "same approved user message has both rank-0 and rank-1 assistant replies",
            "chosen and rejected response texts are different",
            "all messages in the prompt/answer chain pass the same filters",
            "prompt/answer chain is at most 12 messages",
            "prompt and replies are each at most 1800 Unicode characters",
            f"chosen and rejected full sequences are each at most {args.max_length} tokens with tokenizer {args.tokenizer}",
        ],
        "eligible_examples": len(examples),
        "selected_examples": len(selected),
        "seed": args.seed,
        "tokenizer": args.tokenizer,
        "max_length": args.max_length,
        "output_sha256": output_sha256,
        "output": str(output_path),
        "source_ids": [
            {
                "parent": example["source_parent_id"],
                "chosen": example["source_chosen_message_id"],
                "rejected": example["source_rejected_message_id"],
            }
            for example in selected
        ],
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Eligible preference pairs: {len(examples)}")
    print(f"Wrote {len(selected)} preference pairs to {output_path}")
    print(f"Wrote provenance manifest to {manifest_path}")


if __name__ == "__main__":
    main()
