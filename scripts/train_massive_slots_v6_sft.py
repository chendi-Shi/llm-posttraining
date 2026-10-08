"""Run the predeclared v6 secondary Qwen 4-bit QLoRA SFT on the 1,600 groups.

The v6 design fixes this recipe: 4-bit QLoRA from the same frozen
Qwen2.5-0.5B-Instruct base, the same 1,600 training groups used by the v6 BIO
model, learning rate 3e-4, batch 1, gradient accumulation 4, max_length 256,
seed 20261003, and 400 optimizer steps (about one pass). Only the final weights
are evaluated; checkpoints are not selected.

The design also requires a real-tokenizer template and completion-mask check
before training, and refuses to silently truncate an overlong example. This
script enforces both: it runs the TRL preprocessing path on every row and stops
if any row has no active completion tokens or exceeds max_length.

This experiment is secondary and never enters main system selection. It writes
only to ignored local paths; the dev report is written separately.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path
from types import SimpleNamespace

from datasets import load_dataset
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

from common import DEFAULT_MODEL, configure_cpu, load_quantized_base, lora_config


ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = ROOT / "data/massive-zh/slots-v6/train.jsonl"
MANIFEST = ROOT / "data/massive-zh/slots-v6/manifest.json"
EXPECTED_TRAIN_SHA256 = "ea1cd537dcfd924b24eec7997b692a42ae9df757c52f89c904358665ec06a3d3"
EXPECTED_MANIFEST_SHA256 = "7982b3760333a364ebf17c91556f9978b32811fdacc0321db3129b6abee41a70"
EXPECTED_EXAMPLES = 1600
FIXED_RECIPE = {
    "max_steps": 400,
    "max_length": 256,
    "learning_rate": 3e-4,
    "seed": 20261003,
    "gradient_accumulation_steps": 4,
}
PACKAGE_NAMES = ("torch", "transformers", "trl", "peft", "bitsandbytes")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--train-file", default=str(TRAIN_FILE))
    parser.add_argument("--output-dir", default="outputs/massive-slots-v6-sft-1600-400")
    parser.add_argument("--cache-dir", default="_tmp/hf-datasets")
    parser.add_argument("--save-steps", type=int, default=100,
                        help=("Checkpoint interval. The predeclared recipe fixes the step "
                              "count, learning rate and seed, not the checkpoint cadence; "
                              "frequent checkpoints only bound the loss from an interruption."))
    parser.add_argument("--smoke-steps", type=int, default=0,
                        help="Run only this many steps as a pipeline check; the run manifest records it")
    return parser.parse_args()


def require_frozen_inputs(train_file: Path) -> str:
    if not train_file.is_file():
        raise SystemExit(f"v6 training file not found: {train_file}")
    actual = sha256_file(train_file)
    if actual != EXPECTED_TRAIN_SHA256:
        raise SystemExit(f"v6 training file changed: {actual}")
    if sha256_file(MANIFEST) != EXPECTED_MANIFEST_SHA256:
        raise SystemExit("v6 manifest changed")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest["splits"]["train"]["jsonl_sha256"] != EXPECTED_TRAIN_SHA256:
        raise SystemExit("v6 manifest does not lock the expected training file")
    if manifest["splits"]["train"]["examples"] != EXPECTED_EXAMPLES:
        raise SystemExit("v6 manifest train split size differs")
    return actual


def check_template_and_mask(rows: list[dict], tokenizer, max_length: int) -> dict:
    """Run TRL's real preprocessing path and refuse silent truncation."""
    config = SFTConfig(
        output_dir="_tmp/v6-sft-mask-check",
        use_cpu=True,
        max_length=max_length,
        report_to="none",
    )
    completion_only_loss = (
        "prompt" in rows[0] and "completion" in rows[0]
        if config.completion_only_loss is None
        else config.completion_only_loss
    )
    if not completion_only_loss:
        raise SystemExit("SFT would optimize prompt tokens")
    from datasets import Dataset

    dataset = Dataset.from_list(rows).select_columns(["prompt", "completion"])
    context = SimpleNamespace(
        _tokenizer=tokenizer,
        chat_template=None,
        completion_only_loss=completion_only_loss,
    )
    prepared = SFTTrainer._prepare_dataset(
        context,
        dataset,
        processing_class=tokenizer,
        args=config,
        packing=False,
        formatting_func=None,
        dataset_name="v6 SFT mask check",
    )
    if len(prepared) != len(rows):
        raise SystemExit("TRL dropped a training example during preprocessing")
    collator = DataCollatorForLanguageModeling(pad_token_id=tokenizer.pad_token_id)
    batch = collator([prepared[index] for index in range(len(prepared))])
    prompt_tokens = []
    completion_tokens = []
    for index, (row, tokenized) in enumerate(zip(rows, prepared, strict=True)):
        if (len(row["prompt"]) != 1 or row["prompt"][0]["role"] != "user"
                or len(row["completion"]) != 1
                or row["completion"][0]["role"] != "assistant"
                or not isinstance(row["completion"][0]["content"], str)):
            raise SystemExit(f"Unexpected v6 SFT conversation shape in row {row.get('id')}")
        prefix = tokenizer.apply_chat_template(
            row["prompt"], tokenize=True, add_generation_prompt=True, return_dict=True
        )["input_ids"]
        ids = tokenized["input_ids"]
        labels = tokenized["labels"]
        if not prefix:
            raise SystemExit("Empty chat template prefix")
        if len(ids) > max_length:
            raise SystemExit(
                f"Row {row['id']} uses {len(ids)} tokens, above max_length {max_length}; "
                "the design forbids silent truncation"
            )
        if ids[:len(prefix)] != prefix:
            raise SystemExit("Train and inference chat prefixes differ")
        if labels[:len(prefix)] != [-100] * len(prefix):
            raise SystemExit("Prompt tokens are not masked")
        if labels[len(prefix):] != ids[len(prefix):]:
            raise SystemExit("Completion tokens are masked")
        if len(ids) - len(prefix) < 1:
            raise SystemExit(f"Row {row['id']} has no active completion tokens")
        if batch["input_ids"][index, :len(ids)].tolist() != ids:
            raise SystemExit("Collated input ids differ from the prepared row")
        if batch["labels"][index, :len(ids)].tolist() != labels:
            raise SystemExit("Collated labels differ from the prepared row")
        prompt_tokens.append(len(prefix))
        completion_tokens.append(len(ids) - len(prefix))
    return {
        "checked_examples": len(rows),
        "max_length": max_length,
        "min_prompt_tokens_masked": min(prompt_tokens),
        "min_completion_tokens_active": min(completion_tokens),
        "max_total_tokens": max(p + c for p, c in zip(prompt_tokens, completion_tokens, strict=True)),
    }


def main() -> None:
    args = parse_args()
    train_file = Path(args.train_file)
    train_sha256 = require_frozen_inputs(train_file)
    configure_cpu()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = load_dataset("json", data_files=str(train_file), split="train",
                           cache_dir=args.cache_dir).select_columns(["prompt", "completion"])
    if len(dataset) != EXPECTED_EXAMPLES:
        raise SystemExit(f"Expected {EXPECTED_EXAMPLES} v6 training rows, found {len(dataset)}")
    mask_audit = check_template_and_mask(list(dataset), tokenizer, FIXED_RECIPE["max_length"])
    print(json.dumps({"mask_audit": mask_audit}, ensure_ascii=False), flush=True)

    max_steps = args.smoke_steps or FIXED_RECIPE["max_steps"]
    model = load_quantized_base(args.model)
    training_args = SFTConfig(
        output_dir=args.output_dir,
        use_cpu=True,
        max_steps=max_steps,
        max_length=FIXED_RECIPE["max_length"],
        learning_rate=FIXED_RECIPE["learning_rate"],
        per_device_train_batch_size=1,
        gradient_accumulation_steps=FIXED_RECIPE["gradient_accumulation_steps"],
        gradient_checkpointing=True,
        logging_steps=1,
        save_strategy="steps",
        save_steps=min(args.save_steps, max(1, max_steps // 2)),
        save_total_limit=4,
        seed=FIXED_RECIPE["seed"],
        data_seed=FIXED_RECIPE["seed"],
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config(),
    )
    # Resume from the latest checkpoint when an earlier run was interrupted, so
    # an external kill does not discard the whole 400-step run. The final
    # weights are still the only ones evaluated, and the optimizer state is
    # restored by the trainer rather than re-created.
    resumable = sorted(
        (path for path in Path(args.output_dir).glob("checkpoint-*")
         if (path / "trainer_state.json").is_file()),
        key=lambda path: int(path.name.rsplit("-", 1)[1]),
    )
    resume_from = str(resumable[-1]) if resumable else None
    started = time.perf_counter()
    trainer.train(resume_from_checkpoint=resume_from)
    train_seconds = time.perf_counter() - started
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    output_dir = Path(args.output_dir)
    manifest = {
        "study": "MASSIVE zh-CN four-slot v6 secondary QLoRA SFT on the 1,600 training groups",
        "status": "smoke_check_only" if args.smoke_steps else "fixed_final_recipe",
        "train_file": str(train_file),
        "train_file_sha256": train_sha256,
        "v6_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "model": args.model,
        "model_asset_sha256": {
            path.name: sha256_file(path)
            for path in sorted((*Path(args.model).glob("*.safetensors"),
                                Path(args.model) / "tokenizer.json"))
            if path.is_file()
        },
        "recipe": {**FIXED_RECIPE, "per_device_train_batch_size": 1,
                   "save_steps": min(args.save_steps, max(1, max_steps // 2))},
        "executed_max_steps": max_steps,
        "resumed_from_checkpoint": resume_from,
        "mask_audit": mask_audit,
        "training_seconds": round(train_seconds, 2),
        "global_step": trainer.state.global_step,
        "package_versions": {
            name: importlib.metadata.version(name) for name in PACKAGE_NAMES
        },
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "trainer_log_history.json").write_text(
        json.dumps(trainer.state.log_history, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"v6 secondary SFT adapter saved to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
