"""Derive a same-example continued-SFT control from frozen v3 DPO chosen replies."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from evaluate_massive_slots import read_jsonl, sha256_file
from prepare_massive_slots_v2 import _jsonl_bytes, write_locked_files


ROOT = Path(__file__).resolve().parents[1]
PREFERENCES = Path("data/massive-zh/slots-v3/preferences.jsonl")
PREFERENCE_MANIFEST = PREFERENCES.with_name("preferences-manifest.json")
OUTPUT_DIR = PREFERENCES.parent
PREFERENCE_SHA256 = "310c26b37fd7eee7cbfb4876def2d398fcd5f0d188a24da20b24aa474471c450"
PREFERENCE_MANIFEST_SHA256 = "5f20af854896b61e38ad2eda3b73f49c195f58afda0a01a8bb1dff6b6007f33a"


def derive_pairs(pairs: list[dict]) -> list[dict]:
    if len(pairs) != 256:
        raise ValueError("Expected the frozen 256 v3 DPO pairs")
    rows = []
    for pair in pairs:
        if set(pair) != {"prompt", "chosen", "rejected"}:
            raise ValueError("Unexpected v3 pair structure")
        if (not isinstance(pair["prompt"], list) or not pair["prompt"]
                or not isinstance(pair["chosen"], list) or len(pair["chosen"]) != 1
                or pair["chosen"][0].get("role") != "assistant"
                or not isinstance(pair["chosen"][0].get("content"), str)):
            raise ValueError("Invalid chosen conversation")
        rows.append({"prompt": pair["prompt"], "completion": pair["chosen"]})
    return rows


def main() -> None:
    if (sha256_file(PREFERENCES) != PREFERENCE_SHA256
            or sha256_file(PREFERENCE_MANIFEST) != PREFERENCE_MANIFEST_SHA256):
        raise ValueError("Frozen v3 DPO preferences differ")
    source_manifest = json.loads(PREFERENCE_MANIFEST.read_text(encoding="utf-8"))
    if (source_manifest.get("preference_file_sha256") != PREFERENCE_SHA256
            or source_manifest.get("pair_count") != 256
            or len(source_manifest.get("pair_group_sha256", [])) != 256):
        raise ValueError("Frozen v3 preference manifest differs")
    rows = derive_pairs(read_jsonl(PREFERENCES))
    content = _jsonl_bytes(rows)
    manifest = {
        "study": "MASSIVE zh-CN four-slot v3 same-example continued-SFT control",
        "format_version": 1,
        "examples": 256,
        "preference_file_sha256": PREFERENCE_SHA256,
        "preference_manifest_sha256": PREFERENCE_MANIFEST_SHA256,
        "group_sha256": source_manifest["pair_group_sha256"],
        "sft_file_sha256": hashlib.sha256(content).hexdigest(),
        "code_sha256": sha256_file(ROOT / "scripts/prepare_massive_slots_v3_matched_sft.py"),
    }
    write_locked_files(OUTPUT_DIR, {
        "matched-sft.jsonl": content,
        "matched-sft-manifest.json": (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    })
    print(json.dumps({"examples": 256, "sft_sha256": manifest["sft_file_sha256"]}))


if __name__ == "__main__":
    main()
