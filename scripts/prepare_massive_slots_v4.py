"""Freeze fresh MASSIVE train groups for a balanced-preference v4 study.

The old v2 confirmation and all v3 candidate groups are excluded by hash.
The new confirmation JSONL is written once and must remain unopened until the
development gate is met and the model and inference code are locked.
"""

from __future__ import annotations

import argparse
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


SEED = 20261001
PLAN = {"confirmation": 400, "dev": 250, "positive_per_type": 32,
        "hard_negative": 64, "pure_negative": 64}
V3_MANIFEST_SHA256 = "493a4743b974cad98fc42e9535839b6d5e4c7fa60e1d0bb980eb08bcb0db26d0"


def reserved_v3_group_hashes(path: Path) -> set[str]:
    if file_sha256(path) != V3_MANIFEST_SHA256:
        raise ValueError("v3 candidate manifest differs from the frozen study")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    groups = manifest.get("candidate_group_sha256", [])
    if (manifest.get("source_sha256") != SOURCE_SHA256
            or manifest.get("v2_manifest_sha256") != V2_MANIFEST_SHA256
            or manifest.get("candidate_examples") != 576
            or len(groups) != 576 or len(set(groups)) != 576):
        raise ValueError("Unexpected v3 candidate pool")
    return set(groups)


def prepare(source: Path, old_intent_train: Path, v1_dir: Path,
            v2_manifest: Path, v3_manifest: Path, output_dir: Path) -> dict:
    if file_sha256(source) != SOURCE_SHA256:
        raise ValueError("Official MASSIVE archive SHA-256 mismatch")
    source_rows, license_bytes = read_archive(source)
    partitions = Counter(row.get("partition") for row in source_rows)
    if dict(partitions) != SOURCE_PARTITION_SIZES or any(
            row.get("locale") != "zh-CN" for row in source_rows):
        raise ValueError("Unexpected MASSIVE source partitions or locale")
    official_train = [row for row in source_rows if row["partition"] == "train"]
    prior_keys, prior_audit = load_prior_keys(source_rows, old_intent_train, v1_dir)
    v2_groups = reserved_v2_group_hashes(v2_manifest)
    v3_groups = reserved_v3_group_hashes(v3_manifest)
    if v2_groups & v3_groups:
        raise ValueError("v2 and v3 reserved group hashes overlap")
    add_reserved_keys(prior_keys, official_train, v2_groups | v3_groups)
    selected, audit = select_splits(official_train, prior_keys, seed=SEED, plan=PLAN)
    contents: dict[str, bytes] = {}
    splits: dict[str, dict] = {}
    all_groups: set[str] = set()
    for split in ("train", "dev", "confirmation"):
        records = [make_record(item) for item in selected[split]]
        groups = [record["group_sha256"] for record in records]
        if len(groups) != len(set(groups)) or all_groups.intersection(groups):
            raise ValueError("A group repeats across v4 splits")
        all_groups.update(groups)
        content = _jsonl_bytes(records)
        contents[f"{split}.jsonl"] = content
        splits[split] = {
            "examples": len(records),
            "positive_examples": sum(bool(record["slots"]) for record in records),
            "bucket_counts": dict(Counter(record["bucket"] for record in records)),
            "group_sha256": groups,
            "jsonl_sha256": hashlib.sha256(content).hexdigest(),
        }
    expected_train = {**{name: 32 for name in ("date", "time", "place_name", "person")},
                      "hard_negative": 64, "pure_negative": 64}
    if (len(all_groups) != 906 or splits["train"]["bucket_counts"] != expected_train
            or splits["dev"]["examples"] != 250
            or splits["confirmation"]["examples"] != 400):
        raise ValueError("Unexpected v4 split counts")
    manifest = {
        "study": "MASSIVE zh-CN four-slot v4 balanced-preference study",
        "format_version": 1,
        "source_sha256": SOURCE_SHA256,
        "source_partition_used": "train",
        "data_license": "CC-BY-4.0",
        "seed": SEED,
        "plan": PLAN,
        "selection_rule": "Strong phrase groups; hash-ranked natural confirmation then dev, followed by fixed train label quotas",
        "v2_manifest_sha256": V2_MANIFEST_SHA256,
        "v3_manifest_sha256": V3_MANIFEST_SHA256,
        "reserved_v2_groups": len(v2_groups),
        "reserved_v3_groups": len(v3_groups),
        "prior_audit": prior_audit,
        "selection_audit": audit,
        "splits": splits,
    }
    contents["manifest.json"] = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode()
    contents["MASSIVE-LICENSE.txt"] = license_bytes
    write_locked_files(output_dir, contents)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/raw/amazon-massive-dataset-1.0.tar.gz"))
    parser.add_argument("--old-intent-train", type=Path, default=Path("data/massive-zh/train.jsonl"))
    parser.add_argument("--v1-dir", type=Path, default=Path("data/massive-zh/slots"))
    parser.add_argument("--v2-manifest", type=Path, default=Path("data/massive-zh/slots-v2/manifest.json"))
    parser.add_argument("--v3-manifest", type=Path, default=Path("data/massive-zh/slots-v3/manifest.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/massive-zh/slots-v4"))
    args = parser.parse_args()
    manifest = prepare(args.source, args.old_intent_train, args.v1_dir,
                       args.v2_manifest, args.v3_manifest, args.output_dir)
    print(json.dumps({"splits": {name: info["examples"] for name, info in manifest["splits"].items()},
                      "positive_train": manifest["splits"]["train"]["positive_examples"],
                      "dev_sha256": manifest["splits"]["dev"]["jsonl_sha256"],
                      "confirmation_sha256": manifest["splits"]["confirmation"]["jsonl_sha256"]}))


if __name__ == "__main__":
    main()
