from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

from common import DEFAULT_MODEL, configure_cpu, load_quantized_base, load_tokenizer, lora_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CPU QLoRA supervised fine-tuning")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--train-file", default="data/sft.licensed.jsonl")
    parser.add_argument("--output-dir", default="outputs/sft")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--cache-dir", default="_tmp/hf-datasets")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_cpu()
    train_file = Path(args.train_file)
    if not train_file.is_file():
        raise SystemExit(f"SFT file not found: {train_file}")

    tokenizer = load_tokenizer(args.model)
    model = load_quantized_base(args.model)
    dataset = load_dataset(
        "json",
        data_files=str(train_file),
        split="train",
        cache_dir=args.cache_dir,
    )
    dataset = dataset.select_columns(["prompt", "completion"])
    training_args = SFTConfig(
        output_dir=args.output_dir,
        use_cpu=True,
        max_steps=args.max_steps,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        logging_steps=1,
        save_strategy="steps",
        save_steps=max(1, args.max_steps // 2),
        save_total_limit=2,
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config(),
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"SFT adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
