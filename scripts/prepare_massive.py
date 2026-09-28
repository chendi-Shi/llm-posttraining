from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import tarfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import TextIO

from massive_task import user_prompt


DATASET_URL = "https://github.com/alexa/massive"
ARCHIVE_URL = "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.0.tar.gz"
EXPECTED_SPLIT_SIZES = {"train": 11514, "dev": 2033, "test": 2974}
DEFAULT_TOKENIZER = "models/Qwen2.5-0.5B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a balanced Chinese MASSIVE intent-classification SFT set")
    parser.add_argument("--source", required=True, help="MASSIVE 1.0 archive or its zh-CN.jsonl file")
    parser.add_argument("--output-dir", default="data/massive-zh")
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--per-intent", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260924)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def open_jsonl(path: Path) -> TextIO:
    if path.name.lower().endswith(".jsonl"):
        return path.open("r", encoding="utf-8")
    archive = tarfile.open(path, mode="r:*")
    members = [member for member in archive.getmembers() if member.name.endswith("zh-CN.jsonl")]
    if len(members) != 1:
        archive.close()
        raise SystemExit(f"Expected exactly one zh-CN.jsonl in archive, found {len(members)}")
    extracted = archive.extractfile(members[0])
    if extracted is None:
        archive.close()
        raise SystemExit("Could not read zh-CN.jsonl from archive")

    class Utf8Stream:
        def __init__(self, binary, owner):
            self._text = __import__("io").TextIOWrapper(binary, encoding="utf-8")
            self._owner = owner

        def __iter__(self):
            return iter(self._text)

        def close(self):
            self._text.close()
            self._owner.close()

    return Utf8Stream(extracted, archive)  # type: ignore[return-value]


def read_partitions(source: Path) -> dict[str, list[dict]]:
    partitions: dict[str, list[dict]] = defaultdict(list)
    stream = open_jsonl(source)
    try:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("locale") != "zh-CN":
                raise ValueError(f"line {line_number}: expected locale zh-CN, got {row.get('locale')!r}")
            partition = row.get("partition")
            if partition not in {"train", "dev", "test"}:
                raise ValueError(f"line {line_number}: unexpected partition {partition!r}")
            utterance = row.get("utt")
            intent = row.get("intent")
            if not all(isinstance(value, str) and value.strip() for value in (utterance, intent)):
                raise ValueError(f"line {line_number}: missing utterance or intent")
            partitions[partition].append(
                {
                    "id": str(row["id"]),
                    "scenario": str(row["scenario"]),
                    "utt": utterance,
                    "intent": intent,
                }
            )
    finally:
        stream.close()
    return dict(partitions)


def chat_length(tokenizer, utterance: str, target: str) -> int:
    conversation = [
        {"role": "user", "content": user_prompt(utterance)},
        {"role": "assistant", "content": target},
    ]
    encoded = tokenizer.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
    )
    input_ids = encoded["input_ids"]
    if hasattr(input_ids, "shape"):
        return int(input_ids.shape[-1])
    return len(input_ids)


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def copy_embedded_license(source: Path, output_dir: Path) -> str | None:
    if source.name.lower().endswith(".jsonl"):
        return None
    with tarfile.open(source, mode="r:*") as archive:
        candidates = [
            member
            for member in archive.getmembers()
            if member.isfile() and Path(member.name).name.lower() in {"license", "license.txt"}
        ]
        if not candidates:
            return None
        member = min(candidates, key=lambda item: len(Path(item.name).parts))
        content = archive.extractfile(member)
        if content is None:
            return None
        destination = output_dir / "MASSIVE-LICENSE.txt"
        destination.write_bytes(content.read())
        return str(destination)


def main() -> None:
    args = parse_args()
    source = Path(args.source)
    if not source.is_file():
        raise SystemExit(f"Source file not found: {source}")
    from transformers import AutoTokenizer

    print(f"Reading MASSIVE 1.0 Chinese records from {source}", flush=True)
    partitions = read_partitions(source)
    counts = {name: len(partitions.get(name, [])) for name in ("train", "dev", "test")}
    print(f"Observed partition sizes: {counts}", flush=True)
    if counts != EXPECTED_SPLIT_SIZES:
        raise SystemExit(
            f"Split sizes do not match MASSIVE 1.0 zh-CN reference {EXPECTED_SPLIT_SIZES}; "
            "check the downloaded archive/version before using this dataset."
        )

    for name, rows in partitions.items():
        ids = [row["id"] for row in rows]
        if len(ids) != len(set(ids)):
            raise SystemExit(f"Duplicate IDs within {name} split")
    split_ids = {
        name: {row["id"] for row in rows}
        for name, rows in partitions.items()
    }
    split_names = ("train", "dev", "test")
    for left_index, left in enumerate(split_names):
        for right in split_names[left_index + 1 :]:
            overlap = split_ids[left].intersection(split_ids[right])
            if overlap:
                raise SystemExit(f"ID leakage between {left} and {right}: {len(overlap)} IDs")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    intents = sorted({row["intent"] for row in partitions["train"]})
    if len(intents) != 60:
        raise SystemExit(f"Expected 60 train intents, found {len(intents)}")
    eligible: dict[str, list[dict]] = defaultdict(list)
    for row in sorted(partitions["train"], key=lambda item: item["id"]):
        target = row["intent"]
        row["full_tokens"] = chat_length(tokenizer, row["utt"], target)
        if row["full_tokens"] <= args.max_length:
            eligible[row["intent"]].append(row)

    rng = random.Random(args.seed)
    selected = []
    for intent in intents:
        candidates = eligible[intent]
        selected.extend(rng.sample(candidates, min(args.per_intent, len(candidates))))
    selected.sort(key=lambda row: (row["intent"], row["id"]))

    train_records = []
    for row in selected:
        train_records.append(
            {
                "id": row["id"],
                "scenario": row["scenario"],
                "utt": row["utt"],
                "intent": row["intent"],
                "full_sequence_tokens": row["full_tokens"],
                "prompt": [{"role": "user", "content": user_prompt(row["utt"])}],
                "completion": [{"role": "assistant", "content": row["intent"]}],
            }
        )

    eval_records = {}
    for split in ("dev", "test"):
        eval_records[split] = [
            {key: row[key] for key in ("id", "scenario", "utt", "intent")}
            for row in sorted(partitions[split], key=lambda item: item["id"])
        ]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "train.jsonl", train_records)
    write_jsonl(output_dir / "dev.jsonl", eval_records["dev"])
    write_jsonl(output_dir / "test.jsonl", eval_records["test"])
    (output_dir / "intents.json").write_text(
        json.dumps(intents, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    copied_license = copy_embedded_license(source, output_dir)
    selected_lengths = sorted(row["full_tokens"] for row in selected)

    manifest = {
        "dataset": "AmazonScience/MASSIVE",
        "dataset_version": "1.0",
        "dataset_url": DATASET_URL,
        "archive_url": ARCHIVE_URL,
        "data_license": "CC-BY-4.0",
        "attribution": "MASSIVE 1.0 zh-CN, Amazon Science, licensed under CC BY 4.0",
        "citations": [
            "Fitzgerald et al. (2023), MASSIVE, ACL Anthology 2023.acl-long.235",
            "Bastianelli et al. (2020), SLURP, ACL Anthology 2020.emnlp-main.588",
        ],
        "repository_code_license": "Apache-2.0 (not the data license)",
        "source_file": str(source),
        "source_sha256": sha256(source),
        "copied_dataset_license_file": copied_license,
        "locale": "zh-CN",
        "observed_split_sizes": counts,
        "reference_split_sizes": EXPECTED_SPLIT_SIZES,
        "intent_count": len(intents),
        "intents": intents,
        "intent_labels_repeated_in_prompt": False,
        "train_per_intent_limit": args.per_intent,
        "train_count": len(train_records),
        "selected_train_token_lengths": {
            "min": selected_lengths[0],
            "median": statistics.median(selected_lengths),
            "p90": selected_lengths[round(0.9 * (len(selected_lengths) - 1))],
            "max": selected_lengths[-1],
        },
        "intent_shortfalls": {
            intent: len(eligible[intent])
            for intent in intents
            if len(eligible[intent]) < args.per_intent
        },
        "length_filtered_train_count": sum(map(len, eligible.values())),
        "eval_counts": {name: len(rows) for name, rows in eval_records.items()},
        "seed": args.seed,
        "tokenizer": args.tokenizer,
        "max_train_tokens": args.max_length,
        "selected_train_ids": [row["id"] for row in selected],
        "split_intent_counts": {
            split: dict(sorted(Counter(row["intent"] for row in rows).items()))
            for split, rows in {"train": train_records, **eval_records}.items()
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(train_records)} balanced SFT examples and full dev/test sets to {output_dir}")
    print(f"Wrote provenance manifest to {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
