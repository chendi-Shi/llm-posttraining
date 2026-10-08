"""Freeze unseen MASSIVE train groups for the v6 four-slot study.

The source-natural test and dev groups are selected before the label-balanced
training groups. Every group reserved by v1-v5, including unopened holdouts,
is excluded. Preparing test records does not run inference or score them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from prepare_massive_slots import SOURCE_SHA256, file_sha256, read_archive
from prepare_massive_slots_v2 import (
    SOURCE_PARTITION_SIZES,
    _jsonl_bytes,
    load_prior_keys,
    make_record,
    select_splits,
    write_locked_files,
)
from prepare_massive_slots_v3_preferences import (
    V2_MANIFEST_SHA256,
    add_reserved_keys,
    reserved_v2_group_hashes,
)
from prepare_massive_slots_v4 import V3_MANIFEST_SHA256, reserved_v3_group_hashes
from prepare_massive_slots_v5_test import V4_MANIFEST_SHA256, reserved_v4_group_hashes


SEED = 20261003
PLAN = {
    "confirmation": 600,  # The v2 selector's first natural holdout is v6 test.
    "dev": 300,
    "positive_per_type": 128,
    "hard_negative": 544,
    "pure_negative": 544,
}
V5_MANIFEST_SHA256 = "e4531ab27746a0cfe21dc0219331f5cc8237e614bb4c7090b355000fdce1d453"
V5_TEST_SHA256 = "afd5655d9d7e7092ae3c7798179578fc93c3e5d4a88939e9fbfb1d5978459b3c"
EXPECTED_TRAIN_BUCKETS = {
    "date": 128, "time": 128, "place_name": 128, "person": 128,
    "hard_negative": 544, "pure_negative": 544,
}
EXPECTED_SPLIT_SIZES = {"train": 1600, "dev": 300, "test": 600}
FROZEN_OUTPUT_SHA256 = {
    "train.jsonl": "ea1cd537dcfd924b24eec7997b692a42ae9df757c52f89c904358665ec06a3d3",
    "dev.jsonl": "6ee1812c5ac9a12e44047cd57e5973be606b9dcdf546d1d30ccf96f4ff1a157c",
    "test.jsonl": "482efe79ef69a7a43fd7feb9128df8985db663a08b38d17a057237d4bf96293b",
    "manifest.json": "7982b3760333a364ebf17c91556f9978b32811fdacc0321db3129b6abee41a70",
}


def assert_frozen_output_hashes(contents: dict[str, bytes]) -> None:
    """Reject split drift even in a fresh checkout with no existing output."""
    if not FROZEN_OUTPUT_SHA256.keys() <= contents.keys():
        raise ValueError("Missing a preregistered v6 output file")
    for name, expected in FROZEN_OUTPUT_SHA256.items():
        if hashlib.sha256(contents[name]).hexdigest() != expected:
            raise ValueError(f"v6 {name} differs from its preregistered SHA-256")


def reserved_v5_group_hashes(manifest_path: Path, test_path: Path) -> set[str]:
    """Verify v5's sealed source and return only its hashed phrase groups."""
    if file_sha256(manifest_path) != V5_MANIFEST_SHA256:
        raise ValueError("v5 manifest differs from the frozen study")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    groups = manifest.get("test_group_sha256")
    if (
        manifest.get("source_sha256") != SOURCE_SHA256
        or manifest.get("v2_manifest_sha256") != V2_MANIFEST_SHA256
        or manifest.get("v3_manifest_sha256") != V3_MANIFEST_SHA256
        or manifest.get("v4_manifest_sha256") != V4_MANIFEST_SHA256
        or manifest.get("test_examples") != 600
        or manifest.get("test_jsonl_sha256") != V5_TEST_SHA256
        or not isinstance(groups, list)
        or len(groups) != 600
        or len(set(groups)) != 600
        or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)
            for value in groups
        )
    ):
        raise ValueError("Unexpected v5 source, test, or group hashes")
    # Hashing the old file verifies provenance without reading its examples.
    if file_sha256(test_path) != V5_TEST_SHA256:
        raise ValueError("v5 test file differs from its frozen manifest")
    return set(groups)


def prepare(
    source: Path,
    old_intent_train: Path,
    v1_dir: Path,
    v2_manifest: Path,
    v3_manifest: Path,
    v4_manifest: Path,
    v5_manifest: Path,
    v5_test: Path,
    output_dir: Path,
) -> dict:
    if file_sha256(source) != SOURCE_SHA256:
        raise ValueError("Official MASSIVE archive SHA-256 mismatch")
    source_rows, license_bytes = read_archive(source)
    partitions = Counter(row.get("partition") for row in source_rows)
    if dict(partitions) != SOURCE_PARTITION_SIZES or any(
        row.get("locale") != "zh-CN" for row in source_rows
    ):
        raise ValueError("Unexpected source partitions or locale")
    official_train = [row for row in source_rows if row["partition"] == "train"]
    prior_keys, prior_audit = load_prior_keys(source_rows, old_intent_train, v1_dir)
    reserved = {
        "v2": reserved_v2_group_hashes(v2_manifest),
        "v3": reserved_v3_group_hashes(v3_manifest),
        "v4": reserved_v4_group_hashes(v4_manifest),
        "v5": reserved_v5_group_hashes(v5_manifest, v5_test),
    }
    all_reserved: set[str] = set()
    for name, groups in reserved.items():
        if all_reserved & groups:
            raise ValueError(f"{name} shares reserved phrase groups with an earlier study")
        all_reserved.update(groups)
    add_reserved_keys(prior_keys, official_train, all_reserved)
    selected, audit = select_splits(official_train, prior_keys, seed=SEED, plan=PLAN)
    selected_by_role = {
        "train": selected["train"],
        "dev": selected["dev"],
        "test": selected["confirmation"],
    }
    if (
        any(len(selected_by_role[role]) != size for role, size in EXPECTED_SPLIT_SIZES.items())
        or Counter(item["bucket"] for item in selected_by_role["train"]) != EXPECTED_TRAIN_BUCKETS
    ):
        raise ValueError("Unexpected v6 split or training bucket counts")

    contents: dict[str, bytes] = {}
    splits: dict[str, dict] = {}
    all_selected: set[str] = set()
    for role, items in selected_by_role.items():
        records = [make_record(item) for item in items]
        groups = [record["group_sha256"] for record in records]
        if len(groups) != len(set(groups)) or all_selected.intersection(groups):
            raise ValueError("A v6 phrase group repeats across splits")
        if set(groups) & all_reserved:
            raise ValueError("A v6 phrase group appeared in an earlier study")
        all_selected.update(groups)
        payload = _jsonl_bytes(records)
        contents[f"{role}.jsonl"] = payload
        info = {
            "examples": len(records),
            "jsonl_sha256": hashlib.sha256(payload).hexdigest(),
            "group_sha256": groups,
        }
        # Test labels are sealed: do not put their distribution in public output.
        if role != "test":
            info["positive_examples"] = sum(bool(record["slots"]) for record in records)
            info["bucket_counts"] = dict(Counter(record["bucket"] for record in records))
        splits[role] = info
    if len(all_selected) != sum(EXPECTED_SPLIT_SIZES.values()):
        raise ValueError("Unexpected number of unique v6 phrase groups")

    manifest = {
        "study": "MASSIVE zh-CN four-slot v6 expanded-data BIO and SFT study",
        "format_version": 1,
        "source_sha256": SOURCE_SHA256,
        "source_partition_sizes": dict(partitions),
        "source_partition_used": "train",
        "data_license": "CC-BY-4.0",
        "seed": SEED,
        "plan": PLAN,
        "selection_rule": "Strong phrase groups; hash-ranked natural test then dev; fixed training label quotas",
        "v2_manifest_sha256": V2_MANIFEST_SHA256,
        "v3_manifest_sha256": V3_MANIFEST_SHA256,
        "v4_manifest_sha256": V4_MANIFEST_SHA256,
        "v5_manifest_sha256": V5_MANIFEST_SHA256,
        "reserved_group_counts": {name: len(groups) for name, groups in reserved.items()},
        "prior_audit": prior_audit,
        "selection_audit": audit,
        "splits": splits,
        "test_status": "sealed; no model inference or score",
    }
    contents["manifest.json"] = (
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")
    contents["MASSIVE-LICENSE.txt"] = license_bytes
    assert_frozen_output_hashes(contents)
    write_locked_files(output_dir, contents)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/raw/amazon-massive-dataset-1.0.tar.gz"))
    parser.add_argument("--old-intent-train", type=Path, default=Path("data/massive-zh/train.jsonl"))
    parser.add_argument("--v1-dir", type=Path, default=Path("data/massive-zh/slots"))
    parser.add_argument("--v2-manifest", type=Path, default=Path("data/massive-zh/slots-v2/manifest.json"))
    parser.add_argument("--v3-manifest", type=Path, default=Path("data/massive-zh/slots-v3/manifest.json"))
    parser.add_argument("--v4-manifest", type=Path, default=Path("data/massive-zh/slots-v4/manifest.json"))
    parser.add_argument("--v5-manifest", type=Path, default=Path("data/massive-zh/slots-v5/manifest.json"))
    parser.add_argument("--v5-test", type=Path, default=Path("data/massive-zh/slots-v5/test.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/massive-zh/slots-v6"))
    args = parser.parse_args()
    manifest = prepare(
        args.source,
        args.old_intent_train,
        args.v1_dir,
        args.v2_manifest,
        args.v3_manifest,
        args.v4_manifest,
        args.v5_manifest,
        args.v5_test,
        args.output_dir,
    )
    print(json.dumps({
        "split_examples": {role: manifest["splits"][role]["examples"] for role in ("train", "dev", "test")},
        "split_sha256": {role: manifest["splits"][role]["jsonl_sha256"] for role in ("train", "dev", "test")},
        "manifest_sha256": file_sha256(args.output_dir / "manifest.json"),
        "test_status": manifest["test_status"],
    }))


if __name__ == "__main__":
    main()
