from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from contextlib import nullcontext
from pathlib import Path

import torch
from peft import PeftModel

from common import DEFAULT_MODEL, configure_cpu, load_quantized_base, load_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate anonymized base/SFT/DPO answers for human review"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--sft-adapter", default="outputs/sft-oasst2-1epoch")
    parser.add_argument("--dpo-adapter", default="outputs/dpo-oasst2-1epoch")
    parser.add_argument("--eval-file", default="data/dpo.eval.jsonl")
    parser.add_argument("--output", default="reports/oasst2_blind_generation_eval.md")
    parser.add_argument("--key-output", default="reports/oasst2_blind_generation_key.json")
    parser.add_argument("--ratings-output", default="reports/oasst2_blind_generation_ratings.csv")
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    return parser.parse_args()


def render_turns(messages: list[dict]) -> str:
    lines = []
    for message in messages:
        role = message.get("role", "unknown")
        content = message.get("content", "")
        lines.append(f"**{role}:** {content}")
    return "\n\n".join(lines)


def generate(model, tokenizer, prompt: list[dict], adapter_name: str | None, limit: int) -> str:
    if adapter_name is None:
        adapter_context = model.disable_adapter()
    else:
        model.set_adapter(adapter_name)
        adapter_context = nullcontext()

    inputs = tokenizer.apply_chat_template(
        prompt,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    with adapter_context, torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=limit,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = generated[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main() -> None:
    args = parse_args()
    configure_cpu()
    eval_path = Path(args.eval_file)
    sft_path = Path(args.sft_adapter)
    dpo_path = Path(args.dpo_adapter)
    if not eval_path.is_file() or not sft_path.is_dir() or not dpo_path.is_dir():
        raise SystemExit("Evaluation data and both adapter directories must exist")

    manifest_path = eval_path.with_suffix(".manifest.json")
    if not manifest_path.is_file():
        raise SystemExit(f"Evaluation provenance manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("split") != "validation":
        raise SystemExit("Blind generation review must use the held-out validation split")
    checksum = hashlib.sha256(eval_path.read_bytes()).hexdigest()
    if checksum != manifest.get("output_sha256"):
        raise SystemExit("Evaluation JSONL checksum does not match its manifest")

    rows = [json.loads(line) for line in eval_path.read_text(encoding="utf-8").splitlines()]
    if len(rows) != manifest.get("selected_examples") or not rows:
        raise SystemExit("Evaluation row count does not match its manifest")
    eval_source_ids = {
        value for item in manifest.get("source_ids", []) for value in item.values()
    }
    for train_manifest_path in (
        Path("data/sft.licensed.manifest.json"),
        Path("data/dpo.licensed.manifest.json"),
    ):
        if not train_manifest_path.is_file():
            continue
        train_manifest = json.loads(train_manifest_path.read_text(encoding="utf-8"))
        train_ids = set(train_manifest.get("source_message_ids", []))
        train_ids.update(
            value
            for item in train_manifest.get("source_ids", [])
            for value in item.values()
        )
        if eval_source_ids & train_ids:
            raise SystemExit(f"Evaluation source IDs overlap with {train_manifest_path}")

    tokenizer = load_tokenizer(args.model)
    model = load_quantized_base(args.model, training=False)
    model = PeftModel.from_pretrained(
        model, str(sft_path), adapter_name="sft", is_trainable=False
    )
    model.load_adapter(str(dpo_path), adapter_name="dpo", is_trainable=False)
    model.eval()

    rng = random.Random(args.seed)
    answer_rows: list[dict] = []
    key_rows: list[dict] = []
    model_specs = [("base", None), ("sft", "sft"), ("dpo", "dpo")]
    for case_number, row in enumerate(rows, start=1):
        generated = {
            name: generate(model, tokenizer, row["prompt"], adapter, args.max_new_tokens)
            for name, adapter in model_specs
        }
        labels = ["A", "B", "C"]
        rng.shuffle(labels)
        label_to_model = dict(zip(labels, [name for name, _ in model_specs]))
        model_to_label = {name: label for label, name in label_to_model.items()}
        answer_rows.append(
            {
                "case": case_number,
                "prompt": row["prompt"],
                "answers": {
                    model_to_label[name]: generated[name]
                    for name, _ in model_specs
                },
            }
        )
        key_rows.append({"case": case_number, "label_to_model": label_to_model})
        print(f"Generated anonymized answers for case {case_number}/{len(rows)}", flush=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# OASST2 匿名生成对比",
        "",
        f"验证样本：{len(rows)}；每个模型最多生成 {args.max_new_tokens} 个 token。",
        "请按是否切题、是否遵循要求、表达是否清楚，逐题选择 A/B/C 或平局。模型标签映射保存在单独的 key 文件中；评分前不要查看该文件。",
        "",
        "评分记录模板：`oasst2_blind_generation_ratings.csv`。",
        "",
    ]
    for item in answer_rows:
        lines.extend(
            [
                f"## 案例 {item['case']}",
                "",
                "### 对话",
                "",
                render_turns(item["prompt"]),
                "",
            ]
        )
        for label in ("A", "B", "C"):
            answer = item["answers"][label].replace("\r\n", "\n").replace("\r", "\n")
            quoted = "\n".join(f"> {line}" if line else ">" for line in answer.split("\n"))
            lines.extend([f"### 回答 {label}", "", quoted or "> （空回答）", ""])

    output_path.write_text("\n".join(lines), encoding="utf-8")
    key_path = Path(args.key_output)
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.write_text(
        json.dumps(
            {
                "dataset": "OpenAssistant/oasst2",
                "split": "validation",
                "revision": manifest["revision"],
                "evaluation_sha256": checksum,
                "seed": args.seed,
                "max_new_tokens": args.max_new_tokens,
                "adapters": {"sft": str(sft_path), "dpo": str(dpo_path)},
                "cases": key_rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    ratings_path = Path(args.ratings_output)
    ratings_path.parent.mkdir(parents=True, exist_ok=True)
    with ratings_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["case", "winner (A/B/C/tie)", "reason"])
        writer.writerows([[number, "", ""] for number in range(1, len(rows) + 1)])

    print(f"Wrote anonymized comparison to {output_path}")
    print(f"Wrote label key to {key_path}")
    print(f"Wrote rating template to {ratings_path}")


if __name__ == "__main__":
    main()
