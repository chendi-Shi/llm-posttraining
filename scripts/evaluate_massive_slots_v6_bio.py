"""Train the v6 matched-data BIO control and score the development split only.

The v6 test file is deliberately absent from this entry point. Its hash and
contents must remain unopened until a separate, frozen test protocol exists.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import joblib

from evaluate_massive_slots import read_jsonl, score_predictions, sha256_file
from evaluate_massive_slots_bio import (
    MODEL_CONFIG, SEED as BIO_TRAIN_SEED, align_gold, predict_slots, train_model,
)
from massive_slots import canonical_output
from prepare_massive_slots import SOURCE_SHA256, read_archive


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_COUNTS = {"train": 1600, "dev": 300, "test": 600}
SEED = 20261003
LOCKED_MANIFEST_SHA256 = "7982b3760333a364ebf17c91556f9978b32811fdacc0321db3129b6abee41a70"
LOCKED_SPLIT_SHA256 = {
    "train": "ea1cd537dcfd924b24eec7997b692a42ae9df757c52f89c904358665ec06a3d3",
    "dev": "6ee1812c5ac9a12e44047cd57e5973be606b9dcdf546d1d30ccf96f4ff1a157c",
    "test": "482efe79ef69a7a43fd7feb9128df8985db663a08b38d17a057237d4bf96293b",
}


def _split_groups(split: dict) -> list[str]:
    """Accept the compact v4-style or expanded v2-style manifest layout."""
    groups = split.get("group_sha256")
    if groups is None and isinstance(split.get("groups"), list):
        groups = [item.get("group_sha256") for item in split["groups"]]
    if not isinstance(groups, list) or not all(isinstance(group, str) and group for group in groups):
        raise ValueError("Manifest has invalid phrase-group hashes")
    return groups


def validate_manifest(manifest: dict) -> None:
    """Validate split metadata without opening any split, especially test."""
    if (manifest.get("format_version") != 1
            or manifest.get("source_partition_used") != "train"
            or manifest.get("source_sha256") != SOURCE_SHA256
            or manifest.get("seed") != SEED):
        raise ValueError("Unexpected v6 manifest source, format, or seed")
    splits = manifest.get("splits")
    if not isinstance(splits, dict) or set(splits) != set(EXPECTED_COUNTS):
        raise ValueError("Expected only v6 train, dev, and test split metadata")
    seen_groups: set[str] = set()
    for role, count in EXPECTED_COUNTS.items():
        split = splits[role]
        if not isinstance(split, dict) or split.get("examples") != count:
            raise ValueError(f"Unexpected v6 {role} count")
        digest = split.get("jsonl_sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"Invalid v6 {role} file hash")
        if digest != LOCKED_SPLIT_SHA256[role]:
            raise ValueError(f"v6 {role} hash differs from the preregistered study")
        groups = _split_groups(split)
        if len(groups) != count or len(set(groups)) != count or seen_groups.intersection(groups):
            raise ValueError("v6 phrase groups repeat within or across splits")
        seen_groups.update(groups)


def validate_rows(rows: list[dict], role: str, manifest: dict) -> None:
    """Check the two readable splits against the locked manifest."""
    if role not in ("train", "dev"):
        raise ValueError("The v6 BIO development entry point cannot inspect test rows")
    split = manifest["splits"][role]
    if len(rows) != EXPECTED_COUNTS[role]:
        raise ValueError(f"Unexpected v6 {role} row count")
    ids = [str(row["id"]) for row in rows]
    groups = [row.get("group_sha256") for row in rows]
    if len(set(ids)) != len(ids) or groups != _split_groups(split):
        raise ValueError(f"v6 {role} IDs or groups differ from the manifest")
    if "source_ids" in split and ids != [str(identifier) for identifier in split["source_ids"]]:
        raise ValueError(f"v6 {role} IDs differ from the manifest")
    if any(row.get("source_partition") != "train" for row in rows):
        raise ValueError("v6 BIO only accepts official train-partition rows")


def write_predictions(path: Path, rows: list[dict], raw: list[str], dev_sha256: str) -> None:
    if not path.resolve().is_relative_to((ROOT / "_tmp").resolve()):
        raise ValueError("Per-row BIO predictions must stay in the ignored _tmp directory")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row, answer in zip(rows, raw, strict=True):
            stream.write(json.dumps({
                "id": str(row["id"]),
                "group_sha256": row["group_sha256"],
                "eval_file_sha256": dev_sha256,
                "raw_output": answer,
            }, ensure_ascii=False, separators=(",", ":")) + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("dev",), default="dev")
    parser.add_argument("--data-dir", type=Path, default=Path("data/massive-zh/slots-v6"))
    parser.add_argument("--archive", type=Path, default=Path("data/raw/amazon-massive-dataset-1.0.tar.gz"))
    parser.add_argument("--model", type=Path, default=Path("outputs/massive-slots-v6-bio.joblib"))
    parser.add_argument("--report", type=Path, default=Path("reports/massive-slots-v6-bio-dev.json"))
    parser.add_argument("--predictions", type=Path,
                        default=Path("_tmp/massive-slots-v6-bio-dev-predictions.jsonl"))
    args = parser.parse_args(argv)

    if not args.predictions.resolve().is_relative_to((ROOT / "_tmp").resolve()):
        raise ValueError("Per-row BIO predictions must stay in the ignored _tmp directory")
    if any(path.exists() for path in (args.model, args.report, args.predictions)):
        raise ValueError("v6 BIO development outputs already exist; refusing to overwrite")

    manifest_path = args.data_dir / "manifest.json"
    manifest_sha256 = sha256_file(manifest_path)
    if manifest_sha256 != LOCKED_MANIFEST_SHA256:
        raise ValueError("v6 manifest differs from the preregistered study")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    train_path = args.data_dir / "train.jsonl"
    dev_path = args.data_dir / "dev.jsonl"
    # No reference to the v6 test path occurs anywhere in this entry point.
    for role, path in (("train", train_path), ("dev", dev_path)):
        if sha256_file(path) != manifest["splits"][role]["jsonl_sha256"]:
            raise ValueError(f"v6 {role} file differs from the locked manifest")
    if sha256_file(args.archive) != manifest["source_sha256"]:
        raise ValueError("Official MASSIVE archive differs from the v6 manifest")

    train_rows = read_jsonl(train_path)
    dev_rows = read_jsonl(dev_path)
    validate_rows(train_rows, "train", manifest)
    validate_rows(dev_rows, "dev", manifest)
    archive_rows, _ = read_archive(args.archive)
    source_by_id = {str(row["id"]): row for row in archive_rows if row["partition"] == "train"}
    train_spans = align_gold(train_rows, source_by_id)
    align_gold(dev_rows, source_by_id)

    started = time.perf_counter()
    vectorizer, model = train_model(train_rows, train_spans)
    train_seconds = round(time.perf_counter() - started, 2)
    raw = [canonical_output(predict_slots(row["utt"], vectorizer, model)) for row in dev_rows]
    metrics = score_predictions(dev_rows, raw, bootstrap_samples=1000, seed=SEED)

    args.model.parent.mkdir(parents=True, exist_ok=True)
    with args.model.open("xb") as stream:
        joblib.dump({
            "vectorizer": vectorizer,
            "model": model,
            "configuration": MODEL_CONFIG,
            "training_seed": BIO_TRAIN_SEED,
            "source_sha256": manifest["source_sha256"],
            "train_file_sha256": manifest["splits"]["train"]["jsonl_sha256"],
            "manifest_sha256": manifest_sha256,
        }, stream)
    write_predictions(args.predictions, dev_rows, raw, manifest["splits"]["dev"]["jsonl_sha256"])
    report = {
        "task": "MASSIVE 1.0 zh-CN four-slot v6 matched BIO control",
        "role": "dev",
        "model": "character BIO contextual features + LinearSVC + constrained BIO decoding",
        "training_configuration": MODEL_CONFIG,
        "training_seed": BIO_TRAIN_SEED,
        "bootstrap_seed": SEED,
        "data_license": "CC-BY-4.0",
        "source_sha256": manifest["source_sha256"],
        "manifest_sha256": manifest_sha256,
        "train_file_sha256": manifest["splits"]["train"]["jsonl_sha256"],
        "dev_file_sha256": manifest["splits"]["dev"]["jsonl_sha256"],
        "train_examples": len(train_rows),
        "dev_examples": len(dev_rows),
        "train_seconds": train_seconds,
        "test_used": False,
        "model_sha256": sha256_file(args.model),
        "prediction_file_sha256": sha256_file(args.predictions),
        "metrics": metrics,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"role": "dev", "micro_f1": metrics["micro_f1"],
                      "no_slot_failure_rate": metrics["no_slot_failure_rate"],
                      "model_sha256": report["model_sha256"], "report": str(args.report)},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
