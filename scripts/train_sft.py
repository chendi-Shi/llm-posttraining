from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

from datasets import load_dataset
from peft import PeftModel
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
    parser.add_argument("--init-adapter", default=None,
                        help="Continue training an existing LoRA adapter instead of initializing a new one")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--save-steps", type=int, default=None)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--cache-dir", default="_tmp/hf-datasets")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.max_steps < 1 or args.max_length < 1 or args.gradient_accumulation_steps < 1:
        raise SystemExit("Steps, sequence length, and gradient accumulation must be positive")
    if args.save_steps is not None and args.save_steps < 1:
        raise SystemExit("--save-steps must be positive")
    initial_adapter = Path(args.init_adapter) if args.init_adapter else None
    if initial_adapter is not None and not (initial_adapter / "adapter_model.safetensors").is_file():
        raise SystemExit(f"Initial adapter weights not found: {initial_adapter}")
    configure_cpu()
    train_file = Path(args.train_file)
    if not train_file.is_file():
        raise SystemExit(f"SFT file not found: {train_file}")

    source_sha256 = sha256_file(train_file)
    initial_adapter_sha256 = (sha256_file(initial_adapter / "adapter_model.safetensors")
                              if initial_adapter else None)
    model_dir = Path(args.model)
    model_asset_sha256 = (
        {
            path.name: sha256_file(path)
            for path in sorted((*model_dir.glob("*.safetensors"), model_dir / "tokenizer.json"))
            if path.is_file()
        }
        if model_dir.is_dir()
        else {}
    )
    tokenizer = load_tokenizer(args.model)
    model = load_quantized_base(args.model)
    if initial_adapter is not None:
        model = PeftModel.from_pretrained(model, str(initial_adapter), is_trainable=True)
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
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        logging_steps=1,
        save_strategy="steps",
        save_steps=args.save_steps or max(1, args.max_steps // 2),
        save_total_limit=2,
        seed=args.seed,
        data_seed=args.seed,
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=None if initial_adapter is not None else lora_config(),
    )
    started = time.perf_counter()
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    train_seconds = time.perf_counter() - started
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    if initial_adapter is not None and sha256_file(initial_adapter / "adapter_model.safetensors") != initial_adapter_sha256:
        raise ValueError("Initial adapter changed during continued SFT")
    output_dir = Path(args.output_dir)
    manifest = {
        "train_file": str(train_file),
        "train_file_sha256": source_sha256,
        "model": args.model,
        "model_asset_sha256": model_asset_sha256,
        "initial_adapter": str(initial_adapter) if initial_adapter else None,
        "initial_adapter_sha256": initial_adapter_sha256,
        "max_steps": args.max_steps,
        "max_length": args.max_length,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "save_steps": args.save_steps or max(1, args.max_steps // 2),
        "resume_from_checkpoint": args.resume_from_checkpoint,
        "training_seconds": round(train_seconds, 2),
        "global_step": trainer.state.global_step,
        "package_versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "trl", "peft", "bitsandbytes")
        },
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "trainer_log_history.json").write_text(
        json.dumps(trainer.state.log_history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"SFT adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
