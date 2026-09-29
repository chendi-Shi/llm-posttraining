"""Classify MASSIVE intents by scoring every complete label continuation.

The existing trie evaluator greedily chooses one token at a time. That is not
equivalent to choosing the label with the highest full-sequence likelihood when
labels contain multiple tokens. This evaluator provides a separate comparison.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from massive_metrics import read_jsonl, score
from massive_task import user_prompt


@dataclass(frozen=True)
class Candidate:
    label: str
    input_ids: tuple[int, ...]
    prompt_length: int


def stratified_subset(rows: list[dict], limit: int, seed: int) -> list[dict]:
    """Match the existing MASSIVE evaluator's fixed stratified dev subset."""
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


def tokenize_candidates(
    tokenizer, utterance: str, labels: list[str], *, include_turn_end: bool = False
) -> list[Candidate]:
    """Preserve the exact chat prefix and tokenize each candidate in context."""
    def input_ids(encoding):
        # Recent Transformers versions return BatchEncoding even without
        # return_dict=True; older versions return a list of token IDs.
        return list(encoding["input_ids"] if isinstance(encoding, Mapping) else encoding)

    conversation = [{"role": "user", "content": user_prompt(utterance)}]
    prefix = input_ids(
        tokenizer.apply_chat_template(
            conversation, tokenize=True, add_generation_prompt=True
        )
    )
    if not prefix:
        raise ValueError("The chat template returned an empty generation prefix")

    candidates = []
    for label in labels:
        full = input_ids(
            tokenizer.apply_chat_template(
                conversation + [{"role": "assistant", "content": label}],
                tokenize=True,
                add_generation_prompt=False,
            )
        )
        label_ids = list(tokenizer.encode(label, add_special_tokens=False))
        if not label_ids:
            raise ValueError(f"Intent label has no tokens: {label!r}")
        if full[: len(prefix)] != prefix:
            raise ValueError(f"Chat prefix changes when adding label {label!r}")
        suffix = full[len(prefix) :]
        if suffix[: len(label_ids)] != label_ids:
            raise ValueError(f"Label tokenization changes inside the chat template: {label!r}")
        target = suffix if include_turn_end else label_ids
        candidates.append(Candidate(label, tuple(prefix + target), len(prefix)))
    return candidates


def batched_candidate_logprobs(
    model,
    candidates: list[Candidate],
    *,
    pad_token_id: int,
    batch_size: int,
    device: str = "cpu",
) -> dict[str, list[float]]:
    """Return per-token log P(label token | prompt and preceding label tokens)."""
    import torch

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if len({candidate.label for candidate in candidates}) != len(candidates):
        raise ValueError("Candidate labels must be unique")
    result = {}
    with torch.inference_mode():
        for offset in range(0, len(candidates), batch_size):
            batch = candidates[offset : offset + batch_size]
            width = max(len(candidate.input_ids) for candidate in batch)
            input_ids = torch.full(
                (len(batch), width), pad_token_id, dtype=torch.long, device=device
            )
            attention_mask = torch.zeros_like(input_ids)
            for index, candidate in enumerate(batch):
                if candidate.prompt_length < 1 or candidate.prompt_length >= len(candidate.input_ids):
                    raise ValueError(f"Invalid prompt/completion boundary for {candidate.label!r}")
                length = len(candidate.input_ids)
                input_ids[index, :length] = torch.tensor(candidate.input_ids, device=device)
                attention_mask[index, :length] = 1

            logits = model(
                input_ids=input_ids, attention_mask=attention_mask, use_cache=False
            ).logits
            for index, candidate in enumerate(batch):
                begin = candidate.prompt_length
                end = len(candidate.input_ids)
                targets = input_ids[index, begin:end]
                # A causal LM predicts token k from the logits at position k - 1.
                relevant_logits = logits[index, begin - 1 : end - 1].float()
                token_logprobs = torch.log_softmax(relevant_logits, dim=-1)
                selected = token_logprobs.gather(1, targets.unsqueeze(1)).squeeze(1)
                result[candidate.label] = selected.cpu().tolist()
    return result


def rank_candidates(
    token_logprobs: dict[str, list[float]], normalization: str = "sum"
) -> list[dict]:
    if normalization not in {"sum", "mean"}:
        raise ValueError("normalization must be 'sum' or 'mean'")
    ranked = []
    for label, values in token_logprobs.items():
        if not values:
            raise ValueError(f"No target tokens were scored for {label!r}")
        total = sum(values)
        mean = total / len(values)
        ranked.append(
            {
                "label": label,
                "sum_logprob": total,
                "mean_logprob": mean,
                "scored_tokens": len(values),
                "selection_score": total if normalization == "sum" else mean,
            }
        )
    return sorted(ranked, key=lambda item: (-item["selection_score"], item["label"]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--eval-file", required=True)
    parser.add_argument("--intent-labels", default=None)
    parser.add_argument("--output", required=True, help="Per-example prediction JSONL")
    parser.add_argument("--metrics", default=None, help="Defaults to OUTPUT.metrics.json")
    parser.add_argument("--candidate-batch-size", type=int, default=4)
    parser.add_argument("--max-input-tokens", type=int, default=512)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--normalization", choices=("sum", "mean"), default="sum")
    parser.add_argument(
        "--include-turn-end",
        action="store_true",
        help="Also score the chat template's assistant end token(s)",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.candidate_batch_size < 1 or args.max_input_tokens < 1:
        raise SystemExit("Candidate batch size and input token limit must be positive")
    if args.max_examples is not None and args.max_examples < 1:
        raise SystemExit("--max-examples must be positive")

    rows = read_jsonl(args.eval_file)
    if args.max_examples is not None and args.max_examples < len(rows):
        rows = stratified_subset(rows, args.max_examples, args.seed)
    if not rows:
        raise SystemExit("Evaluation file is empty")
    labels_path = Path(args.intent_labels) if args.intent_labels else Path(args.eval_file).with_name("intents.json")
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    if not isinstance(labels, list) or not labels or len(labels) != len(set(labels)):
        raise SystemExit("Intent labels must be a nonempty list of unique strings")
    if any(not isinstance(label, str) or not label for label in labels):
        raise SystemExit("Intent labels must be nonempty strings")

    output_path = Path(args.output)
    metrics_path = Path(args.metrics) if args.metrics else output_path.with_suffix(".metrics.json")
    if not args.overwrite and (output_path.exists() or metrics_path.exists()):
        raise SystemExit("Output or metrics file exists; use --overwrite to replace")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    from peft import PeftModel
    from common import configure_cpu, load_quantized_base, load_tokenizer

    configure_cpu()
    tokenizer = load_tokenizer(args.model)
    model = load_quantized_base(args.model, training=False)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
    model.eval()

    predictions = {}
    start = time.time()
    longest_input = 0
    with output_path.open("w", encoding="utf-8", newline="\n") as stream:
        for index, row in enumerate(rows, start=1):
            candidates = tokenize_candidates(
                tokenizer, row["utt"], labels, include_turn_end=args.include_turn_end
            )
            longest_input = max(longest_input, *(len(item.input_ids) for item in candidates))
            if longest_input > args.max_input_tokens:
                raise ValueError(
                    f"Input for id {row['id']} exceeds {args.max_input_tokens} tokens; "
                    "the evaluator does not silently truncate prompts"
                )
            token_logprobs = batched_candidate_logprobs(
                model,
                candidates,
                pad_token_id=tokenizer.pad_token_id,
                batch_size=args.candidate_batch_size,
            )
            ranked = rank_candidates(token_logprobs, args.normalization)
            prediction = ranked[0]["label"]
            predictions[str(row["id"])] = prediction
            stream.write(
                json.dumps(
                    {
                        "id": str(row["id"]),
                        "prediction": prediction,
                        "normalization": args.normalization,
                        "ranked_candidates": ranked,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            stream.flush()
            if index % 10 == 0 or index == len(rows):
                print(f"Scored {index}/{len(rows)} examples", flush=True)

    metrics = score(rows, predictions, labels)
    metrics.update(
        {
            "model": args.model,
            "adapter": args.adapter,
            "eval_file": args.eval_file,
            "intent_labels_file": str(labels_path),
            "predictions_file": str(output_path),
            "elapsed_seconds": round(time.time() - start, 2),
            "longest_input_tokens": longest_input,
            "scoring": {
                "method": "complete_candidate_log_likelihood",
                "candidate_count": len(labels),
                "candidate_batch_size": args.candidate_batch_size,
                "normalization": args.normalization,
                "include_turn_end": args.include_turn_end,
                "max_input_tokens": args.max_input_tokens,
                "subset_seed": args.seed if args.max_examples is not None else None,
            },
        }
    )
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {key: metrics[key] for key in ("examples", "accuracy", "macro_f1", "elapsed_seconds")},
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"Predictions: {output_path}\nMetrics: {metrics_path}")


if __name__ == "__main__":
    main()
