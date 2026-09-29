"""Fingerprint every direct file in the model directories used by v2 inference.

This includes weights, model/adapter configuration, tokenizer assets and chat
templates.  A newly added file also changes the fingerprint's set of keys.
Checkpoint subdirectories are excluded: inference loads the selected directory
itself, not its nested training checkpoints.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_direct_files(directory: Path, *, scope: str, primary: set[str]) -> dict[str, str]:
    if not directory.is_dir():
        raise ValueError(f"Missing {scope} model directory: {directory}")
    paths = sorted(path for path in directory.iterdir() if path.is_file())
    names = {path.name for path in paths}
    if not primary <= names:
        raise ValueError(f"Missing {scope} model files: {sorted(primary - names)}")
    if any(path.is_symlink() for path in paths):
        raise ValueError(f"Symlink in {scope} model directory: {directory}")
    return {
        path.name if path.name in primary else f"{scope}/{path.name}": sha256_file(path)
        for path in paths
    }


def hash_base_files(directory: Path) -> dict[str, str]:
    return _hash_direct_files(
        directory, scope="base", primary={"model.safetensors", "tokenizer.json"}
    )


def hash_adapter_files(directory: Path) -> dict[str, str]:
    return _hash_direct_files(
        directory, scope="adapter", primary={"adapter_model.safetensors"}
    )


def hash_inference_files(model_dir: Path, adapter_dir: Path) -> dict[str, str]:
    return {**hash_base_files(model_dir), **hash_adapter_files(adapter_dir)}
