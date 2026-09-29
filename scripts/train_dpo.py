from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

from datasets import load_dataset
from peft import PeftModel
from trl import DPOConfig, DPOTrainer

from common import DEFAULT_MODEL, configure_cpu, load_quantized_base, load_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CPU QLoRA direct preference optimization")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--sft-adapter", default="outputs/sft-oasst2-pilot-clean")
    parser.add_argument("--train-file", default="data/dpo.licensed.jsonl")
    parser.add_argument("--output-dir", default="outputs/dpo-oasst2-pilot")
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--beta", type=float, default=0.1)
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
    if args.learning_rate <= 0 or args.beta <= 0:
        raise SystemExit("Learning rate and beta must be positive")
    if args.save_steps is not None and args.save_steps < 1:
        raise SystemExit("--save-steps must be positive")
    configure_cpu()
    train_file = Path(args.train_file)
    adapter_dir = Path(args.sft_adapter)
    if not train_file.is_file():
        raise SystemExit(f"DPO file not found: {train_file}")
    if not adapter_dir.is_dir():
        raise SystemExit(f"SFT adapter directory not found: {adapter_dir}")
    adapter_file = adapter_dir / "adapter_model.safetensors"
    if not adapter_file.is_file():
        raise SystemExit(f"SFT adapter weights not found: {adapter_file}")

    train_sha256 = sha256_file(train_file)
    source_adapter_sha256 = sha256_file(adapter_file)
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
    base_model = load_quantized_base(args.model)
    model = PeftModel.from_pretrained(base_model, str(adapter_dir), is_trainable=True)
    dataset = load_dataset(
        "json",
        data_files=str(train_file),
        split="train",
        cache_dir=args.cache_dir,
    )
    dataset = dataset.select_columns(["prompt", "chosen", "rejected"])
    training_args = DPOConfig(
        output_dir=args.output_dir,
        use_cpu=True,
        max_steps=args.max_steps,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        beta=args.beta,
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
        precompute_ref_log_probs=True,
    )
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    started = time.perf_counter()
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    train_seconds = time.perf_counter() - started
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    output_dir = Path(args.output_dir)
    manifest = {
        "train_file": str(train_file),
        "train_file_sha256": train_sha256,
        "source_adapter": str(adapter_dir),
        "source_adapter_sha256": source_adapter_sha256,
        "model": args.model,
        "model_asset_sha256": model_asset_sha256,
        "max_steps": args.max_steps,
        "max_length": args.max_length,
        "learning_rate": args.learning_rate,
        "beta": args.beta,
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
    print(f"DPO adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
