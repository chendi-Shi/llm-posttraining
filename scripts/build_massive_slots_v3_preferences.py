"""Build train-only DPO pairs from frozen v2 SFT mistakes on fresh MASSIVE rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

from evaluate_massive_slots import generate_raw_predictions, read_jsonl
from massive_slots import TARGET_TYPES, parse_prediction
from prepare_massive_slots import SOURCE_SHA256, file_sha256
from prepare_massive_slots_v2 import _jsonl_bytes, write_locked_files
from v2_model_assets import hash_inference_files


ROOT = Path(__file__).resolve().parents[1]
SEED = 20260930
CANDIDATE_SHA256 = "2fc7e03e67417b4817bfcc3a557d0f1e950015e141e8cfdf26664214bd2e0c0f"
CANDIDATE_MANIFEST_SHA256 = "493a4743b974cad98fc42e9535839b6d5e4c7fa60e1d0bb980eb08bcb0db26d0"
BASE_SHA256 = "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe"
TOKENIZER_SHA256 = "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"
ADAPTER_SHA256 = "206bc1eee5115e20844e22fa146893c9f651058709c6aff5f15e77ec63f61ce6"
QUOTAS = {**{slot_type: 16 for slot_type in TARGET_TYPES},
          "hard_negative": 96, "pure_negative": 96}
GENERATION = {"batch_size": 8, "max_input_tokens": 256, "max_new_tokens": 64,
              "decoding": "greedy_unconstrained"}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def select_candidates(rows: list[dict]) -> list[dict]:
    if len(rows) != 576 or len({row["group_sha256"] for row in rows}) != 576:
        raise ValueError("Not the frozen 576 unique v3 candidate groups")
    buckets: dict[str, list[dict]] = {bucket: [] for bucket in QUOTAS}
    for row in rows:
        bucket = row.get("bucket")
        if bucket not in buckets:
            raise ValueError(f"Unexpected candidate bucket: {bucket}")
        buckets[bucket].append(row)
    selected = []
    for bucket, quota in QUOTAS.items():
        candidates = sorted(buckets[bucket], key=lambda row: (
            sha256_bytes(f"{SEED}|dpo|{bucket}|{row['group_sha256']}".encode()),
            row["group_sha256"],
        ))
        if len(candidates) < quota:
            raise ValueError(f"Insufficient v3 preference candidates in {bucket}")
        selected.extend(candidates[:quota])
    selected.sort(key=lambda row: (sha256_bytes(f"{SEED}|dpo|order|{row['group_sha256']}".encode()),
                                   row["group_sha256"]))
    if len(selected) != 256 or len({row["group_sha256"] for row in selected}) != 256:
        raise ValueError("DPO selection is not 256 disjoint groups")
    return selected


def wrong_literal(utterance: str) -> str:
    """Take a deterministic copied span for a syntactically valid bad reply."""
    for start, char in enumerate(utterance):
        if char.isalnum():
            end = start + 1
            while end < len(utterance) and end - start < 3 and utterance[end].isalnum():
                end += 1
            return utterance[start:end]
    raise ValueError("A no-target candidate has no literal alphanumeric span")


def fallback_rejected(row: dict) -> str:
    if row["slots"]:
        return '{"slots":[]}'
    slot_index = int(row["group_sha256"][:8], 16) % len(TARGET_TYPES)
    literal = wrong_literal(row["utt"])
    return json.dumps({"slots": [{"type": TARGET_TYPES[slot_index], "value": literal}]},
                      ensure_ascii=False, separators=(",", ":"))


def semantically_wrong(raw: str, row: dict) -> bool:
    parsed, _ = parse_prediction(raw, row["utt"])
    if parsed is None:
        return True
    return Counter((item["type"], item["value"]) for item in parsed) != Counter(
        (item["type"], item["value"]) for item in row["slots"]
    )


def token_length(tokenizer, prompt: list[dict], answer: str) -> int:
    tokens = tokenizer.apply_chat_template(
        prompt + [{"role": "assistant", "content": answer}],
        tokenize=True, add_generation_prompt=False,
    )
    if isinstance(tokens, Mapping):
        tokens = tokens["input_ids"]
    if hasattr(tokens, "shape"):
        return int(tokens.shape[-1])
    if tokens and isinstance(tokens[0], list):
        return max(len(sequence) for sequence in tokens)
    return len(tokens)


def make_pair(row: dict, raw: str, tokenizer) -> tuple[dict, str, int]:
    chosen = row["completion"][0]["content"]
    chosen_slots, chosen_validity = parse_prediction(chosen, row["utt"])
    if chosen_slots != row["slots"] or not all(chosen_validity[name] for name in
                                               ("json_valid", "schema_valid", "copy_valid")):
        raise ValueError("Frozen source row has an invalid chosen completion")
    if token_length(tokenizer, row["prompt"], chosen) > 256:
        raise ValueError("Chosen answer exceeds the fixed DPO length")
    use_model = bool(raw.strip()) and semantically_wrong(raw, row)
    rejected = raw.strip() if use_model else fallback_rejected(row)
    if token_length(tokenizer, row["prompt"], rejected) > 256:
        rejected = fallback_rejected(row)
        use_model = False
    if rejected == chosen or token_length(tokenizer, row["prompt"], rejected) > 256:
        raise ValueError("A chosen/rejected pair is identical or too long")
    pair = {
        "prompt": row["prompt"],
        "chosen": row["completion"],
        "rejected": [{"role": "assistant", "content": rejected}],
    }
    return pair, "model_error" if use_model else "deterministic_fallback", max(
        token_length(tokenizer, row["prompt"], chosen),
        token_length(tokenizer, row["prompt"], rejected),
    )


def build(candidate_file: Path, candidate_manifest: Path, model: Path,
          adapter: Path, output_dir: Path) -> dict:
    if file_sha256(candidate_file) != CANDIDATE_SHA256 or file_sha256(candidate_manifest) != CANDIDATE_MANIFEST_SHA256:
        raise ValueError("v3 candidate data differs from the public frozen plan")
    source_manifest = json.loads(candidate_manifest.read_text(encoding="utf-8"))
    if (source_manifest.get("source_sha256") != SOURCE_SHA256
            or source_manifest.get("candidate_file_sha256") != CANDIDATE_SHA256):
        raise ValueError("v3 candidate manifest differs from source data")
    rows = read_jsonl(candidate_file)
    selected = select_candidates(rows)
    from common import load_tokenizer
    tokenizer = load_tokenizer(str(model))
    for row in selected:
        make_pair(row, "", tokenizer)  # Check every deterministic fallback before inference.
    before = hash_inference_files(model, adapter)
    if (before.get("model.safetensors") != BASE_SHA256
            or before.get("tokenizer.json") != TOKENIZER_SHA256
            or before.get("adapter_model.safetensors") != ADAPTER_SHA256):
        raise ValueError("Frozen v2 SFT source model differs")
    raw, inference = generate_raw_predictions(
        selected, model_name=str(model), adapter=str(adapter),
        batch_size=GENERATION["batch_size"],
        max_input_tokens=GENERATION["max_input_tokens"],
        max_new_tokens=GENERATION["max_new_tokens"],
    )
    after = hash_inference_files(model, adapter)
    if before != after:
        raise ValueError("Source model changed while generating v3 preferences")
    pairs = []
    sources = Counter()
    maximum_length = 0
    for row, reply in zip(selected, raw, strict=True):
        pair, source, length = make_pair(row, reply, tokenizer)
        pairs.append(pair)
        sources[source] += 1
        maximum_length = max(maximum_length, length)
    content = _jsonl_bytes(pairs)
    manifest = {
        "study": "MASSIVE zh-CN four-slot v3 train-only DPO preferences",
        "format_version": 1,
        "source_sha256": SOURCE_SHA256,
        "candidate_file_sha256": CANDIDATE_SHA256,
        "candidate_manifest_sha256": CANDIDATE_MANIFEST_SHA256,
        "seed": SEED,
        "pair_count": len(pairs),
        "pair_quotas": QUOTAS,
        "pair_group_sha256": [row["group_sha256"] for row in selected],
        "generation": {**GENERATION, **inference},
        "source_model_files_sha256": before,
        "rejected_sources": dict(sources),
        "max_observed_pair_tokens": maximum_length,
        "preference_file_sha256": sha256_bytes(content),
        "code_sha256": {name: file_sha256(ROOT / name) for name in (
            "scripts/prepare_massive_slots_v3_preferences.py",
            "scripts/build_massive_slots_v3_preferences.py",
            "scripts/massive_slots.py",
            "scripts/evaluate_massive_slots.py",
        )},
    }
    write_locked_files(output_dir, {
        "preferences.jsonl": content,
        "preferences-manifest.json": (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    })
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-file", type=Path, default=Path("data/massive-zh/slots-v3/candidates.jsonl"))
    parser.add_argument("--candidate-manifest", type=Path, default=Path("data/massive-zh/slots-v3/manifest.json"))
    parser.add_argument("--model", type=Path, default=Path("models/Qwen2.5-0.5B-Instruct"))
    parser.add_argument("--adapter", type=Path, default=Path("outputs/massive-slots-v2-sft-lr3e4-160"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/massive-zh/slots-v3"))
    args = parser.parse_args()
    result = build(args.candidate_file, args.candidate_manifest, args.model,
                   args.adapter, args.output_dir)
    print(json.dumps({"pairs": result["pair_count"],
                      "preferences_sha256": result["preference_file_sha256"],
                      "sources": result["rejected_sources"],
                      "max_pair_tokens": result["max_observed_pair_tokens"]}))


if __name__ == "__main__":
    main()
