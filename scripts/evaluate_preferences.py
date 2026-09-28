from __future__ import annotations

import argparse
import hashlib
import json
from contextlib import nullcontext
from pathlib import Path

import torch
from peft import PeftModel

from common import DEFAULT_MODEL, configure_cpu, load_quantized_base, load_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare base, SFT, and DPO models on held-out preference pairs"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--sft-adapter", default="outputs/sft-oasst2-pilot-clean")
    parser.add_argument("--dpo-adapter", default="outputs/dpo-oasst2-pilot")
    parser.add_argument("--eval-file", default="data/dpo.eval.jsonl")
    parser.add_argument("--output", default="reports/oasst2_preference_eval.json")
    return parser.parse_args()


def completion_logps(model, tokenizer, prompt: list[dict], completion: list[dict]) -> tuple[float, float]:
    prefix = tokenizer.apply_chat_template(
        prompt,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )["input_ids"]
    input_ids = tokenizer.apply_chat_template(
        prompt + completion,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt",
        return_dict=True,
    )["input_ids"]
    prefix_len = prefix.shape[1]
    if input_ids.shape[1] > 256:
        raise ValueError(f"Evaluation sequence exceeds configured window: {input_ids.shape[1]} tokens")
    if not torch.equal(input_ids[:, :prefix_len], prefix):
        raise ValueError("The chat template prefix does not match the full conversation")

    with torch.inference_mode():
        logits = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            use_cache=False,
        ).logits[:, :-1, :].float()
    targets = input_ids[:, 1:]
    token_logps = torch.log_softmax(logits, dim=-1).gather(
        -1, targets.unsqueeze(-1)
    ).squeeze(-1)
    target_positions = torch.arange(1, input_ids.shape[1], device=input_ids.device)
    completion_mask = target_positions >= prefix_len
    selected = token_logps[0, completion_mask]
    if selected.numel() == 0:
        raise ValueError("No completion tokens remain after the prompt")
    total_logp = float(selected.sum().item())
    mean_logp = total_logp / selected.numel()
    return total_logp, mean_logp


def score_model(model, tokenizer, rows: list[dict], adapter_name: str | None) -> dict:
    if adapter_name is None:
        adapter_context = model.disable_adapter()
    else:
        model.set_adapter(adapter_name)
        adapter_context = nullcontext()

    margins_sum: list[float] = []
    margins_mean: list[float] = []
    wins_sum = 0
    wins_mean = 0
    with adapter_context:
        for row in rows:
            chosen_sum, chosen_mean = completion_logps(
                model, tokenizer, row["prompt"], row["chosen"]
            )
            rejected_sum, rejected_mean = completion_logps(
                model, tokenizer, row["prompt"], row["rejected"]
            )
            margins_sum.append(chosen_sum - rejected_sum)
            margins_mean.append(chosen_mean - rejected_mean)
            wins_sum += chosen_sum > rejected_sum
            wins_mean += chosen_mean > rejected_mean

    count = len(rows)
    return {
        "pairs": count,
        "chosen_win_rate_sum_logp": wins_sum / count,
        "chosen_win_rate_mean_token_logp": wins_mean / count,
        "mean_margin_sum_logp": sum(margins_sum) / count,
        "mean_margin_per_token_logp": sum(margins_mean) / count,
    }


def main() -> None:
    args = parse_args()
    configure_cpu()
    eval_path = Path(args.eval_file)
    if not eval_path.is_file():
        raise SystemExit(f"Evaluation file not found: {eval_path}")
    if not Path(args.sft_adapter).is_dir() or not Path(args.dpo_adapter).is_dir():
        raise SystemExit("Both SFT and DPO adapter directories must exist")

    rows = [json.loads(line) for line in eval_path.read_text(encoding="utf-8").splitlines()]
    if not rows:
        raise SystemExit("Evaluation file is empty")
    eval_manifest_path = eval_path.with_suffix(".manifest.json")
    if not eval_manifest_path.is_file():
        raise SystemExit(f"Evaluation provenance manifest not found: {eval_manifest_path}")
    eval_manifest = json.loads(eval_manifest_path.read_text(encoding="utf-8"))
    if eval_manifest.get("split") != "validation":
        raise SystemExit("Preference evaluation must use the held-out validation split")
    if len(rows) != eval_manifest.get("selected_examples"):
        raise SystemExit("Evaluation JSONL row count does not match its manifest")
    eval_sha256 = hashlib.sha256(eval_path.read_bytes()).hexdigest()
    if eval_sha256 != eval_manifest.get("output_sha256"):
        raise SystemExit("Evaluation JSONL checksum does not match its manifest")

    eval_source_ids = {
        source_id
        for item in eval_manifest.get("source_ids", [])
        for source_id in item.values()
    }
    for train_manifest_path in (
        Path("data/dpo.licensed.manifest.json"),
        Path("data/sft.licensed.manifest.json"),
    ):
        if not train_manifest_path.is_file():
            continue
        train_manifest = json.loads(train_manifest_path.read_text(encoding="utf-8"))
        if train_manifest.get("split") != "train":
            raise SystemExit(f"Expected training split in {train_manifest_path}")
        train_source_ids = set(train_manifest.get("source_message_ids", []))
        for item in train_manifest.get("source_ids", []):
            train_source_ids.update(item.values())
        overlap = eval_source_ids & train_source_ids
        if overlap:
            raise SystemExit(
                f"Found {len(overlap)} source message IDs shared with {train_manifest_path}"
            )

    tokenizer = load_tokenizer(args.model)
    model = load_quantized_base(args.model, training=False)
    model = PeftModel.from_pretrained(
        model, args.sft_adapter, adapter_name="sft", is_trainable=False
    )
    model.load_adapter(args.dpo_adapter, adapter_name="dpo", is_trainable=False)
    model.eval()

    results = {}
    for name, adapter_name in (("base", None), ("sft", "sft"), ("dpo", "dpo")):
        print(f"Scoring {name} on {len(rows)} held-out pairs...", flush=True)
        results[name] = score_model(model, tokenizer, rows, adapter_name)
    report = {
        "dataset": "OpenAssistant/oasst2",
        "split": "validation",
        "revision": eval_manifest["revision"],
        "evaluation_pairs": len(rows),
        "max_sequence_tokens": eval_manifest["max_length"],
        "evaluation_seed": eval_manifest["seed"],
        "evaluation_sha256": eval_sha256,
        "model": args.model,
        "adapters": {"sft": args.sft_adapter, "dpo": args.dpo_adapter},
        "metric_note": "Raw sequence log-probability ranking is a small held-out preference proxy, not a general capability benchmark.",
        "eval_file": str(eval_path),
        "results": results,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote evaluation report to {output_path}")


if __name__ == "__main__":
    main()
