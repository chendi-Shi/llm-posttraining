"""Reject private, large, or row-level artifacts from the public repository."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BLOCKED_PREFIXES = (
    ".venv/",
    "_tmp/",
    "cache/",
    "models/",
    "outputs/",
    "data/raw/",
    "data/massive-zh/",
    "data/private/",
)
BLOCKED_SUFFIXES = (".safetensors", ".gguf", ".joblib", ".onnx", ".pt", ".ckpt")
MAX_FILE_BYTES = 1024 * 1024
ROW_LEVEL_KEYS = frozenset(
    {
        "id",
        "ids",
        "source_id",
        "source_ids",
        "group_sha256",
        "utt",
        "utterance",
        "raw_output",
        "prediction",
        "correct_by_id",
        "per_row",
        "rows",
    }
)


def row_level_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        found = set(value) & ROW_LEVEL_KEYS
        for item in value.values():
            found.update(row_level_keys(item))
        return found
    if isinstance(value, list):
        found: set[str] = set()
        for item in value:
            found.update(row_level_keys(item))
        return found
    return set()


def main() -> None:
    # Include new, non-ignored files so the check works before they are staged.
    candidates = set()
    for arguments in (("ls-files", "-z"), ("ls-files", "--others", "--exclude-standard", "-z")):
        candidates.update(
            item
            for item in subprocess.check_output(["git", *arguments], cwd=ROOT).split(b"\0")
            if item
        )
    violations = []
    for raw_path in sorted(candidates):
        relative = raw_path.decode("utf-8", errors="surrogateescape").replace("\\", "/")
        path = ROOT / relative
        if (
            relative.startswith(BLOCKED_PREFIXES)
            or relative.lower().endswith(BLOCKED_SUFFIXES)
            or relative.startswith("reports/") and relative.endswith(".jsonl")
            or relative.startswith("reports/oasst2_blind_generation_eval")
            or relative.startswith("reports/oasst2_blind_generation_key")
            or relative.startswith("reports/oasst2_blind_generation_ratings")
            or relative.startswith("data/sft.licensed.")
            or relative.startswith("data/dpo.licensed.")
            or relative.startswith("data/dpo.eval.")
            or relative == ".env" or relative.startswith(".env.")
        ):
            violations.append(f"forbidden path: {relative}")
        if path.is_file() and path.stat().st_size > MAX_FILE_BYTES:
            violations.append(f"file over 1 MiB: {relative}")
        if path.is_file() and relative.startswith("reports/massive-slots-") and relative.endswith(".json"):
            try:
                report = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                violations.append(f"invalid public JSON report: {relative}: {exc}")
            else:
                keys = row_level_keys(report)
                if keys:
                    violations.append(f"row-level keys in {relative}: {', '.join(sorted(keys))}")
    if violations:
        raise SystemExit("\n".join(violations))
    print(f"Release tree check passed: {len(candidates)} tracked or new non-ignored files")


if __name__ == "__main__":
    main()
