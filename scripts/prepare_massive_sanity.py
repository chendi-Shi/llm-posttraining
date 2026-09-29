"""Make a fixed 64-row train-only set for an SFT memorization check.

This is a diagnostic, not a development or test set. The source file must be
the 594-row MASSIVE training sample created by prepare_massive.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from massive_linear_text import text_key


LABELS = (
    "alarm_query",
    "alarm_remove",
    "alarm_set",
    "calendar_query",
    "calendar_remove",
    "calendar_set",
    "lists_query",
    "lists_remove",
)
PER_LABEL = 8
SEED = 20260928


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/massive-zh/train.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/massive-zh/sanity-64.jsonl"))
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.source.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != 594:
        raise SystemExit(f"Expected the fixed 594-row MASSIVE train sample, got {len(rows)} rows")
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["intent"]].append(row)

    rng = random.Random(SEED)
    selected = []
    seen_texts = set()
    for label in LABELS:
        candidates = sorted(groups[label], key=lambda row: int(row["id"]))
        rng.shuffle(candidates)
        accepted = 0
        for row in candidates:
            key = text_key(row["utt"])
            if not key or key in seen_texts:
                continue
            seen_texts.add(key)
            selected.append(row)
            accepted += 1
            if accepted == PER_LABEL:
                break
        if accepted != PER_LABEL:
            raise SystemExit(f"Could select only {accepted}/{PER_LABEL} unique rows for {label}")

    selected.sort(key=lambda row: int(row["id"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as stream:
        for row in selected:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "purpose": "train-only memorization diagnostic; never use as validation or test",
        "source": str(args.source),
        "source_sha256": sha256(args.source),
        "output_sha256": sha256(args.output),
        "seed": SEED,
        "labels": list(LABELS),
        "per_label": PER_LABEL,
        "row_count": len(selected),
        "source_ids": [str(row["id"]) for row in selected],
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(selected), "source_sha256": manifest["source_sha256"], "output_sha256": manifest["output_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
