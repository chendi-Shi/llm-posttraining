"""Build the one frozen v4 DPO training set from 256 fresh train groups."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from build_massive_slots_v3_preferences import (
    ADAPTER_SHA256, BASE_SHA256, TOKENIZER_SHA256, make_pair,
)
from common import load_tokenizer
from evaluate_massive_slots import generate_raw_predictions, read_jsonl, sha256_file
from prepare_massive_slots_v2 import _jsonl_bytes, write_locked_files
from v2_model_assets import hash_inference_files


DATA = Path("data/massive-zh/slots-v4")
MODEL = Path("models/Qwen2.5-0.5B-Instruct")
ADAPTER = Path("outputs/massive-slots-v2-sft-lr3e4-160")
EXPECTED_MANIFEST_SHA256 = "c3bc00cadb70fc6c7a7487e19a2e23f6e57726b4b97cbfaa3582c9835395c206"


def main() -> None:
    if sha256_file(DATA / "manifest.json") != EXPECTED_MANIFEST_SHA256:
        raise ValueError("v4 split manifest differs from the predeclared plan")
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    train_file = DATA / "train.jsonl"
    split = manifest["splits"]["train"]
    if sha256_file(train_file) != split["jsonl_sha256"] or split["examples"] != 256:
        raise ValueError("v4 training rows differ from the locked manifest")
    rows = read_jsonl(train_file)
    if [row["group_sha256"] for row in rows] != split["group_sha256"]:
        raise ValueError("v4 training group order differs")
    tokenizer = load_tokenizer(str(MODEL))
    for row in rows:
        make_pair(row, "", tokenizer)
    before = hash_inference_files(MODEL, ADAPTER)
    if (before.get("model.safetensors") != BASE_SHA256
            or before.get("tokenizer.json") != TOKENIZER_SHA256
            or before.get("adapter_model.safetensors") != ADAPTER_SHA256):
        raise ValueError("Frozen v2 starting model differs")
    raw, inference = generate_raw_predictions(
        rows, model_name=str(MODEL), adapter=str(ADAPTER),
        batch_size=8, max_input_tokens=256, max_new_tokens=64,
    )
    if hash_inference_files(MODEL, ADAPTER) != before:
        raise ValueError("Starting model changed during preference generation")
    pairs = []
    sources: Counter[str] = Counter()
    longest = 0
    for row, reply in zip(rows, raw, strict=True):
        pair, source, length = make_pair(row, reply, tokenizer)
        pairs.append(pair)
        sources[source] += 1
        longest = max(longest, length)
    content = _jsonl_bytes(pairs)
    preference_hash = hashlib.sha256(content).hexdigest()
    pair_manifest = {
        "study": "MASSIVE zh-CN four-slot v4 balanced DPO training preferences",
        "format_version": 1,
        "v4_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "train_file_sha256": split["jsonl_sha256"],
        "pair_count": len(pairs),
        "bucket_counts": split["bucket_counts"],
        "pair_group_sha256": split["group_sha256"],
        "generation": {"batch_size": 8, "max_input_tokens": 256,
                       "max_new_tokens": 64, "decoding": "greedy_unconstrained", **inference},
        "source_model_files_sha256": before,
        "rejected_sources": dict(sources),
        "max_observed_pair_tokens": longest,
        "preference_file_sha256": preference_hash,
    }
    write_locked_files(DATA, {
        "preferences.jsonl": content,
        "preferences-manifest.json": (json.dumps(pair_manifest, ensure_ascii=False, indent=2) + "\n").encode(),
    })
    print(json.dumps({"pairs": len(pairs), "sha256": preference_hash,
                      "rejected_sources": dict(sources), "max_pair_tokens": longest}))


if __name__ == "__main__":
    main()
