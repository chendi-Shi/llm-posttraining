"""Reject private, large, or row-level artifacts from the public repository."""

from __future__ import annotations

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


def main() -> None:
    tracked = subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=ROOT
    ).split(b"\0")
    violations = []
    for raw_path in tracked:
        if not raw_path:
            continue
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
    if violations:
        raise SystemExit("\n".join(violations))
    print(f"Release tree check passed: {sum(bool(item) for item in tracked)} tracked files")


if __name__ == "__main__":
    main()
