"""Freeze a new untouched MASSIVE train-partition test for the v5 hybrid."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from prepare_massive_slots import SOURCE_SHA256, file_sha256, read_archive
from prepare_massive_slots_v2 import (
    SOURCE_PARTITION_SIZES, _jsonl_bytes, load_prior_keys, make_record,
    select_splits, write_locked_files,
)
from prepare_massive_slots_v3_preferences import (
    V2_MANIFEST_SHA256, add_reserved_keys, reserved_v2_group_hashes,
)
from prepare_massive_slots_v4 import V3_MANIFEST_SHA256, reserved_v3_group_hashes


SEED = 20261002
TEST_SIZE = 600
V4_MANIFEST_SHA256 = "c3bc00cadb70fc6c7a7487e19a2e23f6e57726b4b97cbfaa3582c9835395c206"
PLAN = {"confirmation": TEST_SIZE, "dev": 0, "positive_per_type": 0,
        "hard_negative": 0, "pure_negative": 0}


def reserved_v4_group_hashes(path: Path) -> set[str]:
    if file_sha256(path) != V4_MANIFEST_SHA256:
        raise ValueError("v4 split manifest differs from the frozen study")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (manifest.get("source_sha256") != SOURCE_SHA256
            or manifest.get("v2_manifest_sha256") != V2_MANIFEST_SHA256
            or manifest.get("v3_manifest_sha256") != V3_MANIFEST_SHA256):
        raise ValueError("Unexpected v4 source or exclusions")
    groups = [value for split in ("train", "dev", "confirmation")
              for value in manifest["splits"][split]["group_sha256"]]
    if len(groups) != 906 or len(set(groups)) != 906:
        raise ValueError("v4 reserved groups missing or duplicated")
    return set(groups)


def prepare(source: Path, old_intent_train: Path, v1_dir: Path,
            v2_manifest: Path, v3_manifest: Path, v4_manifest: Path,
            output_dir: Path) -> dict:
    if file_sha256(source) != SOURCE_SHA256:
        raise ValueError("Official MASSIVE archive differs")
    source_rows, license_bytes = read_archive(source)
    partitions = Counter(row.get("partition") for row in source_rows)
    if dict(partitions) != SOURCE_PARTITION_SIZES or any(
            row.get("locale") != "zh-CN" for row in source_rows):
        raise ValueError("Unexpected source partitions or locale")
    official_train = [row for row in source_rows if row["partition"] == "train"]
    prior_keys, prior_audit = load_prior_keys(source_rows, old_intent_train, v1_dir)
    v2 = reserved_v2_group_hashes(v2_manifest)
    v3 = reserved_v3_group_hashes(v3_manifest)
    v4 = reserved_v4_group_hashes(v4_manifest)
    if v2 & v3 or v2 & v4 or v3 & v4:
        raise ValueError("An earlier study has overlapping reserved groups")
    add_reserved_keys(prior_keys, official_train, v2 | v3 | v4)
    selected, audit = select_splits(official_train, prior_keys,
                                    seed=SEED, plan=PLAN)
    if selected["train"] or selected["dev"] or len(selected["confirmation"]) != TEST_SIZE:
        raise ValueError("Unexpected v5 test selection")
    rows = [make_record(item) for item in selected["confirmation"]]
    groups = [row["group_sha256"] for row in rows]
    if len(set(groups)) != TEST_SIZE or set(groups) & (v2 | v3 | v4):
        raise ValueError("v5 test shares prior text groups")
    content = _jsonl_bytes(rows)
    manifest = {
        "study": "MASSIVE zh-CN four-slot v5 BIO plus confidence-gated DPO independent test",
        "format_version": 1,
        "source_sha256": SOURCE_SHA256,
        "source_partition_used": "train",
        "data_license": "CC-BY-4.0",
        "seed": SEED,
        "selection_rule": "Hash-ranked source-natural groups after excluding every prior exposed strong phrase group",
        "test_examples": TEST_SIZE,
        "test_jsonl_sha256": hashlib.sha256(content).hexdigest(),
        "test_group_sha256": groups,
        "test_positive_examples": sum(bool(row["slots"]) for row in rows),
        "v2_manifest_sha256": V2_MANIFEST_SHA256,
        "v3_manifest_sha256": V3_MANIFEST_SHA256,
        "v4_manifest_sha256": V4_MANIFEST_SHA256,
        "reserved_group_counts": {"v2": len(v2), "v3": len(v3), "v4": len(v4)},
        "prior_audit": prior_audit,
        "selection_audit": audit,
    }
    write_locked_files(output_dir, {
        "test.jsonl": content,
        "manifest.json": (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode(),
        "MASSIVE-LICENSE.txt": license_bytes,
    })
    return manifest


def main() -> None:
    source = Path("data/raw/amazon-massive-dataset-1.0.tar.gz")
    old_intent_train = Path("data/massive-zh/train.jsonl")
    v1_dir = Path("data/massive-zh/slots")
    v2_manifest = Path("data/massive-zh/slots-v2/manifest.json")
    v3_manifest = Path("data/massive-zh/slots-v3/manifest.json")
    v4_manifest = Path("data/massive-zh/slots-v4/manifest.json")
    output_dir = Path("data/massive-zh/slots-v5")
    manifest = prepare(source, old_intent_train, v1_dir,
                       v2_manifest, v3_manifest, v4_manifest, output_dir)
    print(json.dumps({"test_examples": manifest["test_examples"],
                      "test_sha256": manifest["test_jsonl_sha256"],
                      "positive_examples": manifest["test_positive_examples"]}))


if __name__ == "__main__":
    main()
