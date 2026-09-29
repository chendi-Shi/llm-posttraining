"""Check the real TRL SFT preprocessing without loading model weights.

This catches silent prompt-loss and chat-template mismatches before a long CPU
training run. It reads a few local JSONL examples and prints counts only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from datasets import Dataset
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", default="data/massive-zh/train.jsonl")
    parser.add_argument("--tokenizer", default="models/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--all", action="store_true", help="Check every row, including the longest examples")
    return parser.parse_args()


def selected_rows(path: Path, check_all: bool = False) -> list[dict]:
    if not path.is_file():
        raise SystemExit(f"Training file not found: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) < 3:
        raise SystemExit("Need at least three training examples")
    return rows if check_all else [rows[index] for index in (0, len(rows) // 2, len(rows) - 1)]


def main() -> None:
    args = parse_args()
    rows = selected_rows(Path(args.train_file), args.all)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = SFTConfig(
        output_dir="_tmp/sft-mask-diagnostic",
        use_cpu=True,
        max_length=args.max_length,
        report_to="none",
    )
    # SFTTrainer's prompt/completion default, resolved as in its constructor.
    completion_only_loss = (
        "prompt" in rows[0] and "completion" in rows[0]
        if config.completion_only_loss is None
        else config.completion_only_loss
    )
    assert completion_only_loss, "SFT would optimize prompt tokens"
    assert not config.assistant_only_loss, "Unexpected assistant mask setting"

    dataset = Dataset.from_list(rows).select_columns(["prompt", "completion"])
    # Invoke TRL's actual preparation path with the same defaults as train_sft.py.
    # Only the tokenizer and a few records are needed; no Qwen weights are loaded.
    preprocessing_context = SimpleNamespace(
        _tokenizer=tokenizer,
        chat_template=None,
        completion_only_loss=completion_only_loss,
    )
    prepared = SFTTrainer._prepare_dataset(
        preprocessing_context,
        dataset,
        processing_class=tokenizer,
        args=config,
        packing=False,
        formatting_func=None,
        dataset_name="mask diagnostic",
    )
    assert len(prepared) == len(rows), "TRL dropped a checked example"
    collator = DataCollatorForLanguageModeling(pad_token_id=tokenizer.pad_token_id)
    batch = collator([prepared[index] for index in range(len(prepared))])

    counts = []
    for index, (row, tokenized) in enumerate(zip(rows, prepared, strict=True)):
        assert len(row["prompt"]) == 1 and row["prompt"][0]["role"] == "user"
        assert len(row["completion"]) == 1 and row["completion"][0]["role"] == "assistant"
        assert isinstance(row["completion"][0]["content"], str)

        inference_prefix = tokenizer.apply_chat_template(
            row["prompt"],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        )["input_ids"]
        ids = tokenized["input_ids"]
        labels = tokenized["labels"]
        prefix_length = len(inference_prefix)
        assert prefix_length < len(ids) <= args.max_length
        assert ids[:prefix_length] == inference_prefix, "Train/inference chat prefixes differ"
        assert labels[:prefix_length] == [-100] * prefix_length, "Prompt tokens are not masked"
        assert labels[prefix_length:] == ids[prefix_length:], "Completion tokens are masked"
        assert tokenizer.decode(ids[prefix_length:], skip_special_tokens=False).startswith(
            row["completion"][0]["content"]
        )

        seq_length = len(ids)
        assert batch["input_ids"][index, :seq_length].tolist() == ids
        assert batch["labels"][index, :seq_length].tolist() == labels
        assert all(value == -100 for value in batch["labels"][index, seq_length:].tolist())
        counts.append({"prompt_tokens_masked": prefix_length, "completion_tokens_active": len(ids) - prefix_length})

    print(json.dumps({"checked_examples": len(rows), "max_length": args.max_length, "tokens": counts if not args.all else {"min_prompt_tokens_masked": min(item["prompt_tokens_masked"] for item in counts), "min_completion_tokens_active": min(item["completion_tokens_active"] for item in counts), "max_total_tokens": max(item["prompt_tokens_masked"] + item["completion_tokens_active"] for item in counts)}}, indent=2))


if __name__ == "__main__":
    main()
