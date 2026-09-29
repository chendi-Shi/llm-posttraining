"""Strict, reproducible evaluation for MASSIVE zh-CN four-slot extraction.

The development split is for model selection. The confirmation split requires an
explicit flag and should be run only after the model and decoding settings are
frozen. All records originate in the official MASSIVE train partition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

from massive_slots import TARGET_TYPES, group_key, parse_prediction, user_prompt
from prepare_massive_slots import SOURCE_SHA256


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _prf(tp: int, fp: int, fn: int) -> dict[str, float | int | None]:
    denominator = 2 * tp + fp + fn
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "f1": 2 * tp / denominator if denominator else None,
    }


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    point = fraction * (len(ordered) - 1)
    lower = int(point)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (point - lower) * (ordered[upper] - ordered[lower])


def _bootstrap(
    outcomes: list[dict], cluster_keys: list[str], samples: int, seed: int
) -> dict:
    cluster_rows: dict[str, list[int]] = defaultdict(list)
    for index, cluster in enumerate(cluster_keys):
        cluster_rows[cluster].append(index)
    clusters = sorted(cluster_rows)
    rng = random.Random(seed)
    micro_values: list[float] = []
    exact_values: list[float] = []
    for _ in range(samples):
        sampled_indices = [
            index
            for _ in clusters
            for index in cluster_rows[rng.choice(clusters)]
        ]
        tp = sum(outcomes[index]["tp"] for index in sampled_indices)
        fp = sum(outcomes[index]["fp"] for index in sampled_indices)
        fn = sum(outcomes[index]["fn"] for index in sampled_indices)
        micro = _prf(tp, fp, fn)["f1"]
        if micro is not None:
            micro_values.append(micro)
        exact_values.append(
            sum(outcomes[index]["sentence_exact"] for index in sampled_indices)
            / len(sampled_indices)
        )
    return {
        "unit": "normalized_utterance_group",
        "clusters": len(clusters),
        "samples": samples,
        "seed": seed,
        "confidence": 0.95,
        "micro_f1_ci95": [_percentile(micro_values, 0.025), _percentile(micro_values, 0.975)],
        "sentence_exact_ci95": [_percentile(exact_values, 0.025), _percentile(exact_values, 0.975)],
    }


def score_predictions(
    rows: Sequence[dict],
    raw_predictions: Sequence[str],
    *,
    bootstrap_samples: int = 1000,
    seed: int = 20260928,
) -> dict:
    """Score raw JSON strings in row order, without output repair.

    A malformed output earns no true positives and fails sentence exactness.
    Its syntax/schema/copy failure is reported separately. Exact entities are
    counted as a multiset of (slot type, literal source substring), so repeated
    entities cannot be accidentally deduplicated.
    """
    if not rows or len(rows) != len(raw_predictions):
        raise ValueError("rows and raw_predictions must be nonempty and have the same length")
    if bootstrap_samples < 0:
        raise ValueError("bootstrap_samples must be nonnegative")
    identifiers = [str(row["id"]) for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Duplicate row ID")

    total_tp = total_fp = total_fn = exact = 0
    json_valid = schema_valid = copy_valid = 0
    empty_gold = empty_gold_fp = empty_gold_invalid = 0
    counts_by_type = {slot_type: {"tp": 0, "fp": 0, "fn": 0} for slot_type in TARGET_TYPES}
    errors: Counter[str] = Counter()
    cluster_keys: list[str] = []
    outcomes: list[dict] = []

    for row, raw in zip(rows, raw_predictions, strict=True):
        utterance = row["utt"]
        gold = row["slots"]
        if not isinstance(utterance, str) or not isinstance(gold, list):
            raise ValueError("Every row needs an utterance and a slot list")
        for slot in gold:
            if (
                not isinstance(slot, dict)
                or set(slot) != {"type", "value"}
                or slot["type"] not in TARGET_TYPES
                or not isinstance(slot["value"], str)
                or not slot["value"]
                or slot["value"] not in utterance
            ):
                raise ValueError(f"Invalid gold slot in row {row['id']}")
        predicted, validity = parse_prediction(raw, utterance)
        json_valid += int(validity["json_valid"])
        schema_valid += int(validity["schema_valid"])
        copy_valid += int(validity["copy_valid"])
        if validity["error"]:
            errors[validity["error"]] += 1
        gold_counts = Counter((slot["type"], slot["value"]) for slot in gold)
        predicted_counts = Counter((slot["type"], slot["value"]) for slot in (predicted or []))
        true_counts = gold_counts & predicted_counts
        false_counts = predicted_counts - gold_counts
        missed_counts = gold_counts - predicted_counts
        tp, fp, fn = map(lambda x: sum(x.values()), (true_counts, false_counts, missed_counts))
        total_tp += tp
        total_fp += fp
        total_fn += fn
        row_exact = int(predicted is not None and predicted_counts == gold_counts)
        exact += row_exact
        if not gold:
            empty_gold += 1
            empty_gold_fp += int(bool(predicted_counts))
            empty_gold_invalid += int(predicted is None)
        for slot_type in TARGET_TYPES:
            counts_by_type[slot_type]["tp"] += sum(
                count for (kind, _), count in true_counts.items() if kind == slot_type
            )
            counts_by_type[slot_type]["fp"] += sum(
                count for (kind, _), count in false_counts.items() if kind == slot_type
            )
            counts_by_type[slot_type]["fn"] += sum(
                count for (kind, _), count in missed_counts.items() if kind == slot_type
            )
        cluster_keys.append(row.get("group_sha256") or group_key(utterance))
        outcomes.append({"tp": tp, "fp": fp, "fn": fn, "sentence_exact": row_exact})

    n = len(rows)
    per_type = {slot_type: _prf(**counts_by_type[slot_type]) for slot_type in TARGET_TYPES}
    result = {
        "examples": n,
        "metric_definition": {
            "entity_match": "multiset of (type, literal value) per utterance",
            "invalid_output": "zero matched/predicted entities; sentence exact is false; validity reported separately",
            "no_slot_false_positive": "valid output with one or more target slots on a gold-empty utterance",
            "no_slot_failure": "invalid output or nonempty slots on a gold-empty utterance",
        },
        "gold_slot_count": total_tp + total_fn,
        "predicted_slot_count": total_tp + total_fp,
        "micro": _prf(total_tp, total_fp, total_fn),
        "micro_f1": _prf(total_tp, total_fp, total_fn)["f1"],
        "per_type": per_type,
        "macro_f1": sum((per_type[slot_type]["f1"] or 0.0) for slot_type in TARGET_TYPES)
        / len(TARGET_TYPES),
        "sentence_exact_accuracy": exact / n,
        "sentence_exact_count": exact,
        "json_valid_rate": json_valid / n,
        "schema_valid_rate": schema_valid / n,
        "copy_valid_rate": copy_valid / n,
        "invalid_output_count": n - copy_valid,
        "error_counts": dict(sorted(errors.items())),
        "no_slot_examples": empty_gold,
        "no_slot_false_positive_count": empty_gold_fp,
        "no_slot_false_positive_rate": _ratio(empty_gold_fp, empty_gold),
        "no_slot_invalid_output_count": empty_gold_invalid,
        "no_slot_failure_rate": _ratio(empty_gold_fp + empty_gold_invalid, empty_gold),
        "bootstrap": _bootstrap(outcomes, cluster_keys, bootstrap_samples, seed),
    }
    return result


def paired_bootstrap_f1_delta(
    rows: Sequence[dict],
    baseline_raw_predictions: Sequence[str],
    candidate_raw_predictions: Sequence[str],
    *,
    samples: int = 1000,
    seed: int = 20260928,
) -> dict:
    """Estimate candidate-minus-baseline micro-F1 on paired phrase groups.

    The same sampled group IDs are applied to both systems at each bootstrap
    draw. Raw malformed outputs receive no matched entities, as in
    ``score_predictions``. This is a development-set comparison, not a test.
    """
    if not rows or len(rows) != len(baseline_raw_predictions) or len(rows) != len(candidate_raw_predictions):
        raise ValueError("rows and both prediction lists must have equal nonzero length")
    if samples < 1:
        raise ValueError("samples must be positive")
    if len({str(row["id"]) for row in rows}) != len(rows):
        raise ValueError("Duplicate row ID")

    def row_counts(row: dict, raw: str) -> tuple[int, int, int, int]:
        gold = Counter((slot["type"], slot["value"]) for slot in row["slots"])
        parsed, _ = parse_prediction(raw, row["utt"])
        predicted = Counter((slot["type"], slot["value"]) for slot in (parsed or []))
        return (
            sum((gold & predicted).values()),
            sum((predicted - gold).values()),
            sum((gold - predicted).values()),
            int(parsed is not None and predicted == gold),
        )

    baseline = [row_counts(row, raw) for row, raw in zip(rows, baseline_raw_predictions, strict=True)]
    candidate = [row_counts(row, raw) for row, raw in zip(rows, candidate_raw_predictions, strict=True)]
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row.get("group_sha256") or group_key(row["utt"])].append(index)
    group_names = sorted(groups)

    def f1_for(counts: list[tuple[int, int, int, int]], indices: list[int]) -> float | None:
        tp = sum(counts[index][0] for index in indices)
        fp = sum(counts[index][1] for index in indices)
        fn = sum(counts[index][2] for index in indices)
        return _prf(tp, fp, fn)["f1"]

    all_indices = list(range(len(rows)))
    observed_baseline = f1_for(baseline, all_indices)
    observed_candidate = f1_for(candidate, all_indices)
    observed_baseline_exact = sum(value[3] for value in baseline) / len(rows)
    observed_candidate_exact = sum(value[3] for value in candidate) / len(rows)
    if observed_baseline is None or observed_candidate is None:
        raise ValueError("Paired F1 is undefined without any gold or predicted slots")
    rng = random.Random(seed)
    differences: list[float] = []
    exact_differences: list[float] = []
    for _ in range(samples):
        indices = [
            index
            for _ in group_names
            for index in groups[rng.choice(group_names)]
        ]
        base_f1 = f1_for(baseline, indices)
        candidate_f1 = f1_for(candidate, indices)
        if base_f1 is not None and candidate_f1 is not None:
            differences.append(candidate_f1 - base_f1)
        exact_differences.append(
            sum(candidate[index][3] - baseline[index][3] for index in indices) / len(indices)
        )
    return {
        "unit": "normalized_utterance_group",
        "clusters": len(group_names),
        "samples": samples,
        "seed": seed,
        "baseline_micro_f1": observed_baseline,
        "candidate_micro_f1": observed_candidate,
        "candidate_minus_baseline": observed_candidate - observed_baseline,
        "delta_ci95": [_percentile(differences, 0.025), _percentile(differences, 0.975)],
        "bootstrap_probability_delta_positive": _ratio(sum(value > 0 for value in differences), len(differences)),
        "valid_bootstrap_draws": len(differences),
        "baseline_sentence_exact_accuracy": observed_baseline_exact,
        "candidate_sentence_exact_accuracy": observed_candidate_exact,
        "sentence_exact_candidate_minus_baseline": observed_candidate_exact - observed_baseline_exact,
        "sentence_exact_delta_ci95": [
            _percentile(exact_differences, 0.025), _percentile(exact_differences, 0.975)
        ],
    }


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_locked_split_manifest(path: Path, role: str) -> dict:
    """Check the named study file against its manifest before reading rows.

    This preflight prevents a copied confirmation file renamed to ``dev.jsonl``
    from being parsed or scored through the development route.
    """
    if role not in ("train", "dev", "confirmation") or path.name != f"{role}.jsonl":
        raise ValueError(f"{role} evaluation requires {role}.jsonl")
    manifest_path = path.with_name("manifest.json")
    if not manifest_path.is_file():
        raise ValueError("A locked study manifest is required")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != 1 or manifest.get("source_partition_used") != "train":
        raise ValueError("Unexpected locked study manifest format or source partition")
    if manifest.get("source_sha256") != SOURCE_SHA256:
        raise ValueError("Locked study manifest cites another MASSIVE source archive")
    expected = manifest.get("splits", {}).get(role)
    if not isinstance(expected, dict) or expected.get("jsonl_sha256") != sha256_file(path):
        raise ValueError(f"{role} file differs from the locked study manifest")
    return manifest


def validate_frozen_model_files(
    model_name: str,
    adapter: str | None,
    expected_model_sha256: str,
    expected_tokenizer_sha256: str,
    expected_adapter_sha256: str | None,
) -> dict[str, str]:
    """Pin exact local weights before confirmation text enters the model."""
    expected = {
        "model.safetensors": expected_model_sha256,
        "tokenizer.json": expected_tokenizer_sha256,
    }
    paths = {
        "model.safetensors": Path(model_name) / "model.safetensors",
        "tokenizer.json": Path(model_name) / "tokenizer.json",
    }
    if adapter is not None:
        if expected_adapter_sha256 is None:
            raise ValueError("An adapter requires --expected-adapter-sha256")
        expected["adapter_model.safetensors"] = expected_adapter_sha256
        paths["adapter_model.safetensors"] = Path(adapter) / "adapter_model.safetensors"
    elif expected_adapter_sha256 is not None:
        raise ValueError("--expected-adapter-sha256 requires --adapter")
    actual = {}
    for name, wanted in expected.items():
        if not re.fullmatch(r"[0-9a-fA-F]{64}", wanted):
            raise ValueError(f"Expected {name} SHA-256 must be 64 hexadecimal characters")
        if not paths[name].is_file():
            raise ValueError(f"Frozen {name} file is missing: {paths[name]}")
        observed = sha256_file(paths[name])
        if observed.lower() != wanted.lower():
            raise ValueError(f"Frozen {name} SHA-256 mismatch")
        actual[name] = observed
    return actual


def _validate_study_split(
    rows: list[dict], path: Path, role: str, manifest: dict | None = None
) -> None:
    if not rows:
        raise ValueError("Empty evaluation file")
    for row in rows:
        if row.get("source_partition") != "train":
            raise ValueError("Only official MASSIVE train-partition rows are allowed in this study")
    manifest = manifest or load_locked_split_manifest(path, role)
    expected = manifest["splits"][role]
    if sha256_file(path) != expected["jsonl_sha256"] or len(rows) != expected["examples"]:
        raise ValueError(f"{role} file hash or row count differs from the study manifest")
    ids = [str(row["id"]) for row in rows]
    groups = [row.get("group_sha256") for row in rows]
    if len(set(ids)) != len(ids) or ids != [str(value) for value in expected["source_ids"]]:
        raise ValueError(f"{role} IDs differ from the study manifest")
    if (
        len(set(groups)) != len(groups)
        or groups != [item["group_sha256"] for item in expected["groups"]]
    ):
        raise ValueError(f"{role} phrase groups differ from the study manifest")
    for other_role in (("train",) if role == "dev" else ("train", "dev") if role == "confirmation" else ()):
        other_path = path.with_name(f"{other_role}.jsonl")
        other_expected = manifest["splits"].get(other_role)
        if (
            not other_path.is_file()
            or not isinstance(other_expected, dict)
            or sha256_file(other_path) != other_expected["jsonl_sha256"]
        ):
            raise ValueError(f"Locked {other_role} file is missing or differs from the manifest")
        other_rows = read_jsonl(other_path)
        other_ids = [str(row["id"]) for row in other_rows]
        other_groups = [row.get("group_sha256") for row in other_rows]
        if (
            len(other_rows) != other_expected["examples"]
            or other_ids != [str(value) for value in other_expected["source_ids"]]
            or other_groups != [item["group_sha256"] for item in other_expected["groups"]]
        ):
            raise ValueError(f"Locked {other_role} rows differ from the manifest")
        if set(groups) & set(other_groups):
            raise ValueError(f"Evaluation shares a normalized phrase group with {other_role}")


def generate_raw_predictions(
    rows: list[dict], *, model_name: str, adapter: str | None, batch_size: int,
    max_input_tokens: int, max_new_tokens: int,
) -> tuple[list[str], dict]:
    """Greedy generation; model imports remain lazy for lightweight metric tests."""
    import torch
    from peft import PeftModel

    from common import configure_cpu, load_quantized_base, load_tokenizer

    configure_cpu()
    tokenizer = load_tokenizer(model_name)
    tokenizer.padding_side = "left"
    model = load_quantized_base(model_name, training=False)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    model.eval()
    raw_outputs: list[str] = []
    max_observed_input_tokens = 0
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        conversations = [
            [{"role": "user", "content": user_prompt(row["utt"])}] for row in batch
        ]
        encoded = tokenizer.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            padding=True,
            return_tensors="pt",
            return_dict=True,
        )
        length = max(encoded["attention_mask"].sum(dim=1).tolist())
        max_observed_input_tokens = max(max_observed_input_tokens, length)
        if length > max_input_tokens:
            raise ValueError(
                f"Prompt uses {length} tokens, above max input {max_input_tokens}; "
                "increase the bound instead of truncating an evaluation prompt"
            )
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=False,
                num_beams=1,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        completion = generated[:, encoded["input_ids"].shape[1]:]
        raw_outputs.extend(
            tokenizer.decode(tokens, skip_special_tokens=True).strip()
            for tokens in completion
        )
        print(f"Generated {len(raw_outputs)}/{len(rows)} examples", flush=True)
    return raw_outputs, {"max_observed_input_tokens": max_observed_input_tokens}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--eval-file", type=Path, default=Path("data/massive-zh/slots/dev.jsonl"))
    parser.add_argument("--role", choices=("train", "dev", "confirmation"), default="dev")
    parser.add_argument("--unlock-confirmation", action="store_true")
    parser.add_argument("--expected-model-sha256", default=None)
    parser.add_argument("--expected-tokenizer-sha256", default=None)
    parser.add_argument("--expected-adapter-sha256", default=None)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, default=None, help="Optional per-row JSONL")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-input-tokens", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260928)
    args = parser.parse_args()
    if min(args.batch_size, args.max_input_tokens, args.max_new_tokens) < 1:
        parser.error("Batch size and token limits must be positive")
    if args.role == "confirmation" and not args.unlock_confirmation:
        parser.error("Confirmation evaluation requires --unlock-confirmation after freezing settings")
    frozen_model_files = None
    if args.role == "confirmation":
        if args.expected_model_sha256 is None or args.expected_tokenizer_sha256 is None:
            parser.error(
                "Confirmation requires --expected-model-sha256 and --expected-tokenizer-sha256"
            )
        frozen_model_files = validate_frozen_model_files(
            args.model, args.adapter,
            args.expected_model_sha256, args.expected_tokenizer_sha256,
            args.expected_adapter_sha256,
        )
    manifest = load_locked_split_manifest(args.eval_file, args.role)
    rows = read_jsonl(args.eval_file)
    _validate_study_split(rows, args.eval_file, args.role, manifest)

    start = time.time()
    raw_outputs, inference = generate_raw_predictions(
        rows,
        model_name=args.model,
        adapter=args.adapter,
        batch_size=args.batch_size,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
    )
    metrics = score_predictions(
        rows, raw_outputs, bootstrap_samples=args.bootstrap_samples, seed=args.seed
    )
    metrics.update({
        "task": "MASSIVE 1.0 zh-CN four-slot copy-only JSON extraction",
        "role": args.role,
        "source_partition": "train",
        "eval_file": str(args.eval_file),
        "eval_file_sha256": sha256_file(args.eval_file),
        "study_manifest_sha256": sha256_file(args.eval_file.with_name("manifest.json")),
        "model": args.model,
        "adapter": args.adapter,
        "frozen_model_files_sha256": frozen_model_files,
        "elapsed_seconds": round(time.time() - start, 2),
        "generation": {
            "decoding": "greedy_unconstrained",
            "do_sample": False,
            "batch_size": args.batch_size,
            "max_input_tokens": args.max_input_tokens,
            "max_new_tokens": args.max_new_tokens,
            **inference,
        },
    })
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.predictions is not None:
        args.predictions.parent.mkdir(parents=True, exist_ok=True)
        with args.predictions.open("w", encoding="utf-8", newline="\n") as stream:
            for row, raw in zip(rows, raw_outputs, strict=True):
                parsed, validity = parse_prediction(raw, row["utt"])
                stream.write(json.dumps({
                    "id": str(row["id"]),
                    "group_sha256": row["group_sha256"],
                    "eval_file_sha256": metrics["eval_file_sha256"],
                    "raw_output": raw,
                    "prediction": parsed, "validity": validity,
                }, ensure_ascii=False) + "\n")
    print(json.dumps({
        "examples": metrics["examples"],
        "micro_f1": metrics["micro_f1"],
        "sentence_exact_accuracy": metrics["sentence_exact_accuracy"],
        "json_valid_rate": metrics["json_valid_rate"],
        "copy_valid_rate": metrics["copy_valid_rate"],
        "elapsed_seconds": metrics["elapsed_seconds"],
    }, ensure_ascii=False, indent=2))
    print(f"Metrics: {args.metrics}")


if __name__ == "__main__":
    main()
