"""Prepare fresh train-only MASSIVE rows for a four-slot DPO correction study.

The v2 confirmation file is never opened. Its strong phrase groups are read
only as hashes from the v2 manifest and excluded before any label parsing.
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
    group_sha256,
    load_prior_keys,
    make_record,
    select_splits,
    strong_group_key,
    write_locked_files,
)


SEED = 20260930
PLAN = {"confirmation": 0, "dev": 0, "positive_per_type": 48,
        "hard_negative": 192, "pure_negative": 192}
V2_MANIFEST_SHA256 = "96d8f6501931a29569421de7edbc82912025150a379aa93796ed27d64ee5dd2c"
EXPECTED_V2_GROUPS = 640 + 250 + 400
EXPECTED_CANDIDATES = 4 * PLAN["positive_per_type"] + PLAN["hard_negative"] + PLAN["pure_negative"]


def reserved_v2_group_hashes(manifest_path: Path) -> set[str]:
    if file_sha256(manifest_path) != V2_MANIFEST_SHA256:
        raise ValueError("v2 manifest differs from the frozen public design")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("source_sha256") != SOURCE_SHA256:
        raise ValueError("v2 manifest records another source archive")
    groups: list[str] = []
    for split, count in (("train", 640), ("dev", 250), ("confirmation", 400)):
        info = manifest["splits"][split]
        if info.get("examples") != count or len(info.get("groups", [])) != count:
            raise ValueError(f"v2 {split} size differs")
        groups.extend(item["group_sha256"] for item in info["groups"])
    if len(groups) != EXPECTED_V2_GROUPS or len(set(groups)) != EXPECTED_V2_GROUPS:
        raise ValueError("v2 reserved groups are duplicated or missing")
    if any(len(value) != 64 or any(char not in "0123456789abcdef" for char in value) for value in groups):
        raise ValueError("Invalid v2 group hash")
    return set(groups)


def add_reserved_keys(prior_keys: set[str], official_train: list[dict],
                      reserved_hashes: set[str]) -> set[str]:
    """Use utterance keys only; do not inspect labels of reserved groups."""
    matched: set[str] = set()
    for row in official_train:
        key = strong_group_key(row["utt"])
        digest = group_sha256(key)
        if digest in reserved_hashes:
            prior_keys.add(key)
            matched.add(digest)
    if matched != reserved_hashes:
        raise ValueError("Official train source does not contain every reserved v2 group")
    return prior_keys


def prepare(source: Path, old_intent_train: Path, v1_dir: Path,
            v2_manifest: Path, output_dir: Path) -> dict:
    if file_sha256(source) != SOURCE_SHA256:
        raise ValueError("Official MASSIVE archive SHA-256 mismatch")
    source_rows, _ = read_archive(source)
    partitions = Counter(row.get("partition") for row in source_rows)
    if dict(partitions) != SOURCE_PARTITION_SIZES or any(row.get("locale") != "zh-CN" for row in source_rows):
        raise ValueError("Unexpected MASSIVE source partitions or locale")
    prior_keys, prior_audit = load_prior_keys(source_rows, old_intent_train, v1_dir)
    official_train = [row for row in source_rows if row["partition"] == "train"]
    reserved = reserved_v2_group_hashes(v2_manifest)
    add_reserved_keys(prior_keys, official_train, reserved)
    selected, audit = select_splits(official_train, prior_keys, seed=SEED, plan=PLAN)
    if selected["confirmation"] or selected["dev"] or len(selected["train"]) != EXPECTED_CANDIDATES:
        raise ValueError("Unexpected v3 candidate selection")
    records = [make_record(item) for item in selected["train"]]
    if {record["group_sha256"] for record in records} & reserved:
        raise ValueError("v3 candidates overlap frozen v2 groups")
    counts = Counter(record["bucket"] for record in records)
    if counts != Counter({**{kind: 48 for kind in ("date", "time", "place_name", "person")},
                          "hard_negative": 192, "pure_negative": 192}):
        raise ValueError("Unexpected v3 candidate buckets")
    content = _jsonl_bytes(records)
    manifest = {
        "study": "MASSIVE zh-CN four-slot v3 preference candidate pool",
        "format_version": 1,
        "source_sha256": SOURCE_SHA256,
        "v2_manifest_sha256": V2_MANIFEST_SHA256,
        "v2_reserved_group_count": len(reserved),
        "seed": SEED,
        "candidate_plan": PLAN,
        "candidate_examples": len(records),
        "bucket_counts": dict(counts),
        "candidate_file_sha256": hashlib.sha256(content).hexdigest(),
        "candidate_group_sha256": [record["group_sha256"] for record in records],
        "prior_audit": prior_audit,
        "selection_audit": audit,
    }
    write_locked_files(output_dir, {
        "candidates.jsonl": content,
        "manifest.json": (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    })
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/raw/amazon-massive-dataset-1.0.tar.gz"))
    parser.add_argument("--old-intent-train", type=Path, default=Path("data/massive-zh/train.jsonl"))
    parser.add_argument("--v1-dir", type=Path, default=Path("data/massive-zh/slots"))
    parser.add_argument("--v2-manifest", type=Path, default=Path("data/massive-zh/slots-v2/manifest.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/massive-zh/slots-v3"))
    args = parser.parse_args()
    try:
        manifest = prepare(args.source, args.old_intent_train, args.v1_dir,
                           args.v2_manifest, args.output_dir)
    except (ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"v3 preference preparation failed: {error}\n")
    print(json.dumps({"candidates": manifest["candidate_examples"],
                      "sha256": manifest["candidate_file_sha256"],
                      "reserved_v2_groups": manifest["v2_reserved_group_count"]}))


if __name__ == "__main__":
    main()
