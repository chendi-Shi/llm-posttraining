from __future__ import annotations

import os
from pathlib import Path

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_DIR = Path("models/Qwen2.5-0.5B-Instruct")
MODEL_LICENSE_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "licenses"
    / "Qwen2.5-0.5B-Instruct-LICENSE"
)


def ensure_model_license() -> None:
    """Keep the upstream license beside any locally downloaded model weights."""
    license_path = MODEL_DIR / "LICENSE"
    license_text = (
        license_path.read_bytes()
        if license_path.is_file()
        else MODEL_LICENSE_SOURCE.read_bytes()
    )

    if b"Apache License" not in license_text or b"Copyright 2024 Alibaba Cloud" not in license_text:
        raise RuntimeError(f"Unexpected Qwen license content in {license_path if license_path.is_file() else MODEL_LICENSE_SOURCE}")
    if not license_path.is_file():
        license_path.write_bytes(license_text)


def main() -> None:
    from modelscope_hub import HubApi

    os.environ.setdefault("MODELSCOPE_DOWNLOAD_PARALLEL_WORKERS", "4")
    api = HubApi()
    snapshot = api.download_repo(
        MODEL_ID,
        "model",
        local_dir=MODEL_DIR,
        allow_patterns=[
            "*.safetensors",
            "*.json",
            "*.txt",
            "*.model",
            "*.jinja",
            "LICENSE",
            "NOTICE*",
        ],
        max_workers=4,
    )
    ensure_model_license()
    print(f"Model downloaded to {snapshot}")


if __name__ == "__main__":
    main()
