"""Build a locked, phrase-disjoint MASSIVE zh-CN slot extraction study."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

from massive_slots import TARGET_TYPES, canonical_output, group_key, parse_annotated, user_prompt


SOURCE_URL = "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.0.tar.gz"
SOURCE_SHA256 = "7df623fd2d300a4d235d6ee5bd396c9a28258d3a0ccb29abdb054506eba153f8"
SEED = 20260928
SPLIT_PLAN = {
    "train": {"time": 80, "person": 80, "place_name": 80, "date": 80, "none": 64},
    "dev": {"time": 20, "person": 20, "place_name": 20, "date": 20, "none": 20},
    "confirmation": {"time": 40, "person": 40, "place_name": 40, "date": 40, "none": 40},
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_archive(path: Path) -> tuple[list[dict], bytes]:
    with tarfile.open(path, "r:gz") as archive:
        data_members = [member for member in archive if member.name.endswith("/data/zh-CN.jsonl")]
        license_members = [member for member in archive if member.name.endswith("/LICENSE")]
        if len(data_members) != 1 or len(license_members) != 1:
            raise ValueError("Expected one MASSIVE zh-CN JSONL and one LICENSE in the archive")
        data_stream = archive.extractfile(data_members[0])
        license_stream = archive.extractfile(license_members[0])
        if data_stream is None or license_stream is None:
            raise ValueError("Cannot read MASSIVE archive members")
        with io.TextIOWrapper(data_stream, encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        license_bytes = license_stream.read()
    return rows, license_bytes


def _id_sort_key(row: dict) -> tuple[int, int | str]:
    identifier = str(row["id"])
    return (0, int(identifier)) if identifier.isdecimal() else (1, identifier)


def _rank(seed: int, split: str, bucket: str, key: str) -> str:
    return hashlib.sha256(f"{seed}|{split}|{bucket}|{key}".encode("utf-8")).hexdigest()


def _group_hash(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def select_splits(
    official_train: list[dict], old_sft_group_keys: set[str], seed: int = SEED
) -> tuple[dict[str, list[dict]], dict]:
    """Select one representative per phrase group with fixed bucket quotas."""
    groups: dict[str, list[dict]] = defaultdict(list)
    observed_ids: set[str] = set()
    for row in official_train:
        identifier = str(row["id"])
        if identifier in observed_ids:
            raise ValueError(f"Duplicate official train ID: {identifier}")
        observed_ids.add(identifier)
        if row.get("partition") != "train" or row.get("locale") != "zh-CN":
            raise ValueError(f"Unexpected source row {identifier}: not zh-CN train")
        enriched = dict(row)
        enriched["slots"] = parse_annotated(row["utt"], row["annot_utt"])
        groups[group_key(row["utt"])].append(enriched)

    eligible: dict[str, dict] = {}
    conflicts = []
    for key, members in groups.items():
        signatures = {
            tuple(sorted((slot["type"], slot["value"]) for slot in row["slots"]))
            for row in members
        }
        if len(signatures) > 1:
            conflicts.append(
                {
                    "group_sha256": _group_hash(key),
                    "source_ids": [str(row["id"]) for row in sorted(members, key=_id_sort_key)],
                }
            )
            continue
        representative = min(members, key=_id_sort_key)
        eligible[key] = {
            "row": representative,
            "source_ids": [str(row["id"]) for row in sorted(members, key=_id_sort_key)],
            "types": {slot["type"] for slot in representative["slots"]},
        }

    selected: dict[str, list[dict]] = {name: [] for name in SPLIT_PLAN}
    used_keys: set[str] = set()
    # Lock confirmation first, then dev. Both exclude phrases seen by the old SFT.
    for split in ("confirmation", "dev", "train"):
        for bucket, quota in SPLIT_PLAN[split].items():
            candidates = [
                key
                for key, item in eligible.items()
                if key not in used_keys
                and (split == "train" or key not in old_sft_group_keys)
                and (not item["types"] if bucket == "none" else bucket in item["types"])
            ]
            candidates.sort(key=lambda key: (_rank(seed, split, bucket, key), key))
            if len(candidates) < quota:
                raise ValueError(
                    f"Insufficient distinct groups for {split}/{bucket}: "
                    f"need {quota}, found {len(candidates)}"
                )
            for key in candidates[:quota]:
                used_keys.add(key)
                item = eligible[key]
                selected[split].append(
                    {"group_key": key, "bucket": bucket, **item}
                )
        selected[split].sort(key=lambda item: (_rank(seed, split, "order", item["group_key"]), item["group_key"]))

    audit = {
        "official_train_rows": len(official_train),
        "unique_phrase_groups": len(groups),
        "conflict_groups_excluded": len(conflicts),
        "conflict_rows_excluded": sum(len(item["source_ids"]) for item in conflicts),
        "conflicts": sorted(conflicts, key=lambda item: item["group_sha256"]),
        "old_sft_phrase_groups_excluded_from_evaluation": len(old_sft_group_keys),
        "eligible_phrase_groups": len(eligible),
    }
    return selected, audit


def read_old_sft_groups(path: Path, official_train: list[dict]) -> tuple[set[str], list[str]]:
    source_by_id = {str(row["id"]): row for row in official_train}
    old_ids: list[str] = []
    keys: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            old_row = json.loads(line)
            identifier = str(old_row["id"])
            if identifier not in source_by_id:
                raise ValueError(f"Old SFT ID is absent from official train: {identifier}")
            if group_key(old_row["utt"]) != group_key(source_by_id[identifier]["utt"]):
                raise ValueError(f"Old SFT utterance differs from source: {identifier}")
            old_ids.append(identifier)
            keys.add(group_key(old_row["utt"]))
    if len(old_ids) != 594 or len(old_ids) != len(set(old_ids)):
        raise ValueError(f"Expected 594 distinct old SFT train IDs; found {len(old_ids)}")
    return keys, old_ids


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def make_record(item: dict) -> dict:
    row = item["row"]
    slots = row["slots"]
    return {
        "id": str(row["id"]),
        "source_partition": "train",
        "utt": row["utt"],
        "slots": slots,
        "group_sha256": _group_hash(item["group_key"]),
        "bucket": item["bucket"],
        "prompt": [{"role": "user", "content": user_prompt(row["utt"])}],
        "completion": [{"role": "assistant", "content": canonical_output(slots)}],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/raw/amazon-massive-dataset-1.0.tar.gz"))
    parser.add_argument("--old-sft-train", type=Path, default=Path("data/massive-zh/train.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/massive-zh/slots"))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if not args.source.is_file() or not args.old_sft_train.is_file():
        raise SystemExit("Official MASSIVE archive and old SFT train file must exist")
    source_hash = file_sha256(args.source)
    if source_hash != SOURCE_SHA256:
        raise SystemExit(f"Unexpected MASSIVE 1.0 archive SHA-256: {source_hash}")
    rows, license_bytes = read_archive(args.source)
    partition_counts = Counter(row.get("partition") for row in rows)
    if partition_counts != {"train": 11514, "dev": 2033, "test": 2974}:
        raise SystemExit(f"Unexpected MASSIVE zh-CN partition sizes: {partition_counts}")
    if any(row.get("locale") != "zh-CN" for row in rows):
        raise SystemExit("Archive member contains another locale")
    official_train = [row for row in rows if row["partition"] == "train"]
    old_keys, old_ids = read_old_sft_groups(args.old_sft_train, official_train)
    selected, audit = select_splits(official_train, old_keys, args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "MASSIVE-LICENSE.txt").write_bytes(license_bytes)
    manifest = {
        "study": "MASSIVE 1.0 zh-CN four-slot JSON SFT",
        "format_version": 1,
        "source_url": SOURCE_URL,
        "source_sha256": source_hash,
        "source_partition_sizes": dict(partition_counts),
        "source_partition_used": "train",
        "data_license": "CC-BY-4.0",
        "target_types": list(TARGET_TYPES),
        "seed": args.seed,
        "group_key_rule": "NFKC + casefold + remove all Unicode whitespace",
        "split_plan": SPLIT_PLAN,
        "old_sft_train_file_sha256": file_sha256(args.old_sft_train),
        "old_sft_train_ids": old_ids,
        "audit": audit,
        "splits": {},
    }
    for split, items in selected.items():
        records = [make_record(item) for item in items]
        output_path = args.output_dir / f"{split}.jsonl"
        write_jsonl(output_path, records)
        types = Counter(slot["type"] for record in records for slot in record["slots"])
        manifest["splits"][split] = {
            "examples": len(records),
            "positive_examples": sum(bool(record["slots"]) for record in records),
            "slot_spans_by_type": {name: types[name] for name in TARGET_TYPES},
            "bucket_counts": dict(Counter(record["bucket"] for record in records)),
            "source_ids": [record["id"] for record in records],
            "groups": [
                {"group_sha256": record["group_sha256"], "source_ids": item["source_ids"]}
                for record, item in zip(records, items)
            ],
            "jsonl_sha256": file_sha256(output_path),
        }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Wrote train/dev/confirmation: "
        f"{len(selected['train'])}/{len(selected['dev'])}/{len(selected['confirmation'])}; "
        f"excluded {audit['conflict_groups_excluded']} conflicting phrase groups"
    )


if __name__ == "__main__":
    main()
