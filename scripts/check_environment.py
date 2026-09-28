from __future__ import annotations

import importlib.metadata
import sys

import torch


PACKAGES = ["transformers", "trl", "peft", "bitsandbytes", "datasets", "accelerate"]


def main() -> None:
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CPU threads: {torch.get_num_threads()}")
    for package in PACKAGES:
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            print(f"{package}: MISSING")
        else:
            print(f"{package}: {version}")

    try:
        import bitsandbytes

        version = importlib.metadata.version("bitsandbytes")
        print(f"bitsandbytes import: OK ({version}; {bitsandbytes.__file__})")
    except Exception as exc:
        print(f"bitsandbytes import: FAILED ({type(exc).__name__}: {exc})")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
