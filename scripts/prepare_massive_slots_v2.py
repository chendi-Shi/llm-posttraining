"""Prepare a new, source-natural MASSIVE zh-CN four-slot study.

The v2 holdouts are sampled before label-balanced training examples. All text
previously used by the intent study, the v1 slot study, or official dev/test
is excluded under a stronger punctuation-insensitive phrase key.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from massive_slots import ANNOTATION, TARGET_TYPES, canonical_output, parse_annotated, user_prompt
from prepare_massive_slots import SOURCE_SHA256, SOURCE_URL, file_sha256, read_archive


SEED = 20260929
PLAN = {
    "dev": 250,
    "confirmation": 400,
    "positive_per_type": 64,
    "hard_negative": 192,
    "pure_negative": 192,
}
OLD_SPLIT_SIZES = {"train": 384, "dev": 100, "confirmation": 200}
SOURCE_PARTITION_SIZES = {"train": 11514, "dev": 2033, "test": 2974}
V1_MANIFEST_SHA256 = "d6736c2779f00e898dd623911bed08f7c93ab053c287b5ffaaf211a6eee36e32"


def strong_group_key(utterance: str) -> str:
    """Fold width, case, whitespace, and Unicode punctuation for leakage checks."""
    normalized = unicodedata.normalize("NFKC", utterance).casefold()
    return "".join(
        char
        for char in normalized
        if not char.isspace() and not unicodedata.category(char).startswith("P")
    )


def group_sha256(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _rank(seed: int, purpose: str, key: str) -> str:
    return hashlib.sha256(f"v2|{seed}|{purpose}|{key}".encode("utf-8")).hexdigest()


def _id_sort_key(row: dict) -> tuple[int, int | str]:
    identifier = str(row["id"])
    return (0, int(identifier)) if identifier.isdecimal() else (1, identifier)


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_prior_keys(
    source_rows: list[dict], old_intent_train: Path, v1_dir: Path,
    expected_old_sizes: dict[str, int] = OLD_SPLIT_SIZES,
    expected_intent_size: int = 594,
    expected_v1_manifest_sha256: str = V1_MANIFEST_SHA256,
) -> tuple[set[str], dict]:
    """Verify each earlier asset and return every previously exposed phrase."""
    official_train = {str(row["id"]): row for row in source_rows if row["partition"] == "train"}
    if len(official_train) != SOURCE_PARTITION_SIZES["train"]:
        raise ValueError("Official train IDs are missing or duplicated")
    official_eval_keys = {
        strong_group_key(row["utt"])
        for row in source_rows
        if row["partition"] in ("dev", "test")
    }
    previous_keys = set(official_eval_keys)
    if not old_intent_train.is_file():
        raise ValueError("The old intent SFT training file is required for leakage exclusion")
    intent_rows = read_jsonl(old_intent_train)
    if len(intent_rows) != expected_intent_size:
        raise ValueError("Unexpected number of old intent SFT examples")
    old_ids: set[str] = set()
    intent_keys: set[str] = set()
    for row in intent_rows:
        identifier = str(row["id"])
        if identifier in old_ids or identifier not in official_train:
            raise ValueError("Old intent SFT ID is duplicated or absent from official train")
        if row["utt"] != official_train[identifier]["utt"]:
            raise ValueError("Old intent SFT utterance differs from official source")
        old_ids.add(identifier)
        intent_keys.add(strong_group_key(row["utt"]))
    previous_keys.update(intent_keys)

    manifest_path = v1_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("The v1 slot manifest is required for leakage exclusion")
    if file_sha256(manifest_path) != expected_v1_manifest_sha256:
        raise ValueError("The v1 slot manifest differs from the previously frozen study")
    v1_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        v1_manifest.get("format_version") != 1
        or v1_manifest.get("source_sha256") != SOURCE_SHA256
        or v1_manifest.get("source_partition_used") != "train"
        or v1_manifest.get("old_sft_train_file_sha256") != file_sha256(old_intent_train)
    ):
        raise ValueError("Unexpected v1 slot study manifest")
    v1_keys: dict[str, set[str]] = {}
    v1_hashes: dict[str, str] = {}
    for split, expected_size in expected_old_sizes.items():
        path = v1_dir / f"{split}.jsonl"
        expected = v1_manifest.get("splits", {}).get(split)
        if (
            not path.is_file()
            or not isinstance(expected, dict)
            or expected.get("examples") != expected_size
            or expected.get("jsonl_sha256") != file_sha256(path)
        ):
            raise ValueError(f"The v1 {split} file differs from its locked manifest")
        rows = read_jsonl(path)
        if len(rows) != expected_size or [str(row["id"]) for row in rows] != [
            str(identifier) for identifier in expected.get("source_ids", [])
        ]:
            raise ValueError(f"The v1 {split} IDs differ from its locked manifest")
        keys = set()
        for row in rows:
            identifier = str(row["id"])
            if identifier not in official_train or row["utt"] != official_train[identifier]["utt"]:
                raise ValueError(f"The v1 {split} utterance differs from official source")
            keys.add(strong_group_key(row["utt"]))
        if len(keys) != expected_size:
            raise ValueError(f"The v1 {split} contains duplicate strong phrase groups")
        v1_keys[split] = keys
        v1_hashes[split] = file_sha256(path)
        previous_keys.update(keys)

    return previous_keys, {
        "official_dev_test_groups": len(official_eval_keys),
        "old_intent_train_examples": len(intent_rows),
        "old_intent_train_groups": len(intent_keys),
        "old_intent_train_sha256": file_sha256(old_intent_train),
        "v1_manifest_sha256": file_sha256(manifest_path),
        "v1_split_examples": dict(expected_old_sizes),
        "v1_split_group_counts": {name: len(keys) for name, keys in v1_keys.items()},
        "v1_split_sha256": v1_hashes,
        "union_excluded_phrase_groups": len(previous_keys),
    }


def select_splits(
    official_train: list[dict], prior_keys: set[str], seed: int = SEED,
    plan: dict[str, int] = PLAN,
) -> tuple[dict[str, list[dict]], dict]:
    """Sample natural holdouts first, then a fixed positive/negative training mix."""
    groups: dict[str, list[dict]] = defaultdict(list)
    seen_ids: set[str] = set()
    for row in official_train:
        identifier = str(row["id"])
        if identifier in seen_ids or row.get("partition") != "train" or row.get("locale") != "zh-CN":
            raise ValueError("Source train IDs, partitions, or locales are invalid")
        seen_ids.add(identifier)
        groups[strong_group_key(row["utt"])].append(row)

    eligible: dict[str, dict] = {}
    conflicts = 0
    conflict_rows = 0
    excluded_prior = 0
    excluded_empty = 0
    for key, members in groups.items():
        if not key:
            excluded_empty += 1
            continue
        if key in prior_keys:
            excluded_prior += 1
            continue
        parsed = [(row, parse_annotated(row["utt"], row["annot_utt"])) for row in members]
        signatures = {
            tuple(sorted((slot["type"], slot["value"]) for slot in slots))
            for _, slots in parsed
        }
        if len(signatures) > 1:
            conflicts += 1
            conflict_rows += len(members)
            continue
        row, slots = min(parsed, key=lambda pair: _id_sort_key(pair[0]))
        eligible[key] = {
            "key": key,
            "row": row,
            "slots": slots,
            "types": {slot["type"] for slot in slots},
            "source_ids": [str(item["id"]) for item in sorted(members, key=_id_sort_key)],
            "negative_kind": (
                "hard_negative" if ANNOTATION.search(row["annot_utt"]) else "pure_negative"
            ) if not slots else None,
        }

    natural = sorted(eligible, key=lambda key: (_rank(seed, "natural", key), key))
    confirmation_n = plan["confirmation"]
    dev_n = plan["dev"]
    if len(natural) < confirmation_n + dev_n:
        raise ValueError("Insufficient groups for source-natural holdouts")
    selected: dict[str, list[dict]] = {
        "confirmation": [{**eligible[key], "bucket": "natural"} for key in natural[:confirmation_n]],
        "dev": [{**eligible[key], "bucket": "natural"} for key in natural[confirmation_n:confirmation_n + dev_n]],
        "train": [],
    }
    used = {item["key"] for split in ("confirmation", "dev") for item in selected[split]}
    for slot_type in TARGET_TYPES:
        candidates = [
            key for key, item in eligible.items()
            if key not in used and slot_type in item["types"]
        ]
        candidates.sort(key=lambda key: (_rank(seed, f"train|{slot_type}", key), key))
        quota = plan["positive_per_type"]
        if len(candidates) < quota:
            raise ValueError(f"Insufficient train positives for {slot_type}: {len(candidates)} < {quota}")
        for key in candidates[:quota]:
            used.add(key)
            selected["train"].append({**eligible[key], "bucket": slot_type})
    for bucket in ("hard_negative", "pure_negative"):
        candidates = [
            key for key, item in eligible.items()
            if key not in used and item["negative_kind"] == bucket
        ]
        candidates.sort(key=lambda key: (_rank(seed, f"train|{bucket}", key), key))
        quota = plan[bucket]
        if len(candidates) < quota:
            raise ValueError(f"Insufficient {bucket} groups: {len(candidates)} < {quota}")
        for key in candidates[:quota]:
            used.add(key)
            selected["train"].append({**eligible[key], "bucket": bucket})
    for split, items in selected.items():
        items.sort(key=lambda item: (_rank(seed, f"{split}|order", item["key"]), item["key"]))
    if len(used) != sum(map(len, selected.values())):
        raise AssertionError("A phrase group was selected more than once")
    eligible_positive = sum(bool(item["slots"]) for item in eligible.values())
    audit = {
        "official_train_rows": len(official_train),
        "unique_strong_phrase_groups": len(groups),
        "prior_phrase_groups_excluded_from_train_source": excluded_prior,
        "empty_key_groups_excluded": excluded_empty,
        "conflict_groups_excluded": conflicts,
        "conflict_rows_excluded": conflict_rows,
        "eligible_groups": len(eligible),
        "eligible_positive_groups": eligible_positive,
        "eligible_no_target_groups": len(eligible) - eligible_positive,
        "eligible_positive_rate": eligible_positive / len(eligible),
        "eligible_groups_by_target_type": {
            slot_type: sum(slot_type in item["types"] for item in eligible.values())
            for slot_type in TARGET_TYPES
        },
        "eligible_hard_negative_groups": sum(
            item["negative_kind"] == "hard_negative" for item in eligible.values()
        ),
        "eligible_pure_negative_groups": sum(
            item["negative_kind"] == "pure_negative" for item in eligible.values()
        ),
    }
    return selected, audit


def make_record(item: dict) -> dict:
    row = item["row"]
    return {
        "id": str(row["id"]),
        "source_partition": "train",
        "utt": row["utt"],
        "slots": item["slots"],
        "group_sha256": group_sha256(item["key"]),
        "bucket": item["bucket"],
        "prompt": [{"role": "user", "content": user_prompt(row["utt"])}],
        "completion": [{"role": "assistant", "content": canonical_output(item["slots"])}],
    }


def _jsonl_bytes(records: list[dict]) -> bytes:
    return "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        for record in records
    ).encode("utf-8")


def write_locked_files(output_dir: Path, contents: dict[str, bytes]) -> None:
    """Idempotent reruns are allowed; changing an existing split is forbidden."""
    for name, content in contents.items():
        path = output_dir / name
        if path.exists() and path.read_bytes() != content:
            raise ValueError(f"Refusing to overwrite a different locked v2 file: {path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in contents.items():
        path = output_dir / name
        if not path.exists():
            path.write_bytes(content)


def prepare(
    source: Path, old_intent_train: Path, v1_dir: Path, output_dir: Path,
    seed: int = SEED,
) -> dict:
    if seed != SEED:
        raise ValueError(f"The v2 protocol fixes the seed at {SEED}")
    if not source.is_file():
        raise ValueError("Official MASSIVE 1.0 archive is required")
    source_sha256 = file_sha256(source)
    if source_sha256 != SOURCE_SHA256:
        raise ValueError("Official MASSIVE 1.0 archive SHA-256 differs from the fixed source")
    source_rows, license_bytes = read_archive(source)
    partitions = Counter(row.get("partition") for row in source_rows)
    if dict(partitions) != SOURCE_PARTITION_SIZES or any(
        row.get("locale") != "zh-CN" for row in source_rows
    ):
        raise ValueError("Unexpected source partition sizes or locale")
    prior_keys, prior_audit = load_prior_keys(source_rows, old_intent_train, v1_dir)
    official_train = [row for row in source_rows if row["partition"] == "train"]
    selected, selection_audit = select_splits(official_train, prior_keys, seed)
    split_contents: dict[str, bytes] = {}
    split_manifest: dict[str, dict] = {}
    for split in ("train", "dev", "confirmation"):
        items = selected[split]
        records = [make_record(item) for item in items]
        content = _jsonl_bytes(records)
        split_contents[f"{split}.jsonl"] = content
        slot_counts = Counter(slot["type"] for record in records for slot in record["slots"])
        positive = sum(bool(record["slots"]) for record in records)
        split_manifest[split] = {
            "examples": len(records),
            "positive_examples": positive,
            "no_target_examples": len(records) - positive,
            "positive_rate": positive / len(records),
            "slot_spans_by_type": {name: slot_counts[name] for name in TARGET_TYPES},
            "bucket_counts": dict(Counter(record["bucket"] for record in records)),
            "source_ids": [record["id"] for record in records],
            "groups": [
                {"group_sha256": record["group_sha256"], "source_ids": item["source_ids"]}
                for record, item in zip(records, items)
            ],
            "jsonl_sha256": hashlib.sha256(content).hexdigest(),
        }
    manifest = {
        "study": "MASSIVE 1.0 zh-CN four-slot JSON SFT v2 source-natural evaluation",
        "format_version": 1,
        "source_url": SOURCE_URL,
        "source_sha256": source_sha256,
        "source_partition_sizes": dict(partitions),
        "source_partition_used": "train",
        "data_license": "CC-BY-4.0",
        "target_types": list(TARGET_TYPES),
        "seed": seed,
        "group_key_rule": "NFKC + casefold + remove Unicode whitespace and punctuation",
        "split_plan": dict(PLAN),
        "selection_rule": "Hash-rank eligible groups; confirmation 400 then dev 250 without labels; train from remaining fixed label quotas",
        "prior_exclusion": prior_audit,
        "audit": selection_audit,
        "splits": split_manifest,
    }
    contents = {
        **split_contents,
        "MASSIVE-LICENSE.txt": license_bytes,
        "manifest.json": (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    }
    write_locked_files(output_dir, contents)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/raw/amazon-massive-dataset-1.0.tar.gz"))
    parser.add_argument("--old-intent-train", type=Path, default=Path("data/massive-zh/train.jsonl"))
    parser.add_argument("--v1-dir", type=Path, default=Path("data/massive-zh/slots"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/massive-zh/slots-v2"))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    try:
        manifest = prepare(args.source, args.old_intent_train, args.v1_dir, args.output_dir, args.seed)
    except ValueError as exc:
        parser.exit(2, f"v2 preparation failed: {exc}\n")
    print(
        "Wrote locked v2 train/dev/confirmation: "
        + "/".join(str(manifest["splits"][name]["examples"]) for name in ("train", "dev", "confirmation"))
        + f"; eligible groups {manifest['audit']['eligible_groups']}; "
        + f"natural dev positives {manifest['splits']['dev']['positive_examples']}; "
        + f"natural confirmation positives {manifest['splits']['confirmation']['positive_examples']}"
    )


if __name__ == "__main__":
    main()
