"""Score the v6 secondary QLoRA SFT adapter on the v6 development split only.

The v6 design keeps the secondary SFT experiment out of main system selection
and requires a separately fixed protocol for any test evaluation. This entry
point therefore reads only train/dev metadata and dev.jsonl: the string
"test.jsonl" is never constructed here, and any request for another role is
rejected before touching local files.

It also re-scores the frozen v6 baseline systems from their stored dev
predictions on the identical 300 rows, so the reported comparisons are
same-example paired bootstrap deltas rather than cross-report arithmetic.
Per-row outputs stay in the ignored local _tmp directory.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
from itertools import combinations
from pathlib import Path

from compare_massive_slots_v2 import positive_only_metrics
from evaluate_massive_slots import (
    generate_raw_predictions, paired_bootstrap_f1_delta, read_jsonl,
    score_predictions, sha256_file,
)
from evaluate_massive_slots_v2 import prediction_payload, write_predictions


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/massive-zh/slots-v6"
MANIFEST = DATA / "manifest.json"
BASE = ROOT / "models/Qwen2.5-0.5B-Instruct"
DEFAULT_ADAPTER = ROOT / "outputs/massive-slots-v6-sft-1600-400"
REPORT_PATH = ROOT / "reports/massive-slots-v6-sft-dev.json"
PREDICTION_PATH = ROOT / "_tmp/massive-slots-v6-sft-dev-predictions.jsonl"

EXPECTED_MANIFEST_SHA256 = "7982b3760333a364ebf17c91556f9978b32811fdacc0321db3129b6abee41a70"
EXPECTED_DEV_SHA256 = "6ee1812c5ac9a12e44047cd57e5973be606b9dcdf546d1d30ccf96f4ff1a157c"
EXPECTED_ROWS = 300
GENERATION = {"decoding": "greedy_unconstrained", "batch_size": 8,
              "max_input_tokens": 256, "max_new_tokens": 64}
BOOTSTRAP = {"samples": 1000, "seed": 20261003}
PACKAGE_NAMES = ("torch", "transformers", "peft", "trl", "bitsandbytes",
                 "scikit-learn", "joblib", "numpy")
BASELINES = ("old_bio", "new_bio", "v4_dpo", "old_v5_hybrid", "new_hybrid")
SFT_SYSTEM = "v6_sft_1600_400"
CODE_FILES = (
    "scripts/train_massive_slots_v6_sft.py",
    "scripts/evaluate_massive_slots_v6_sft.py",
    "scripts/evaluate_massive_slots.py",
    "scripts/massive_slots.py",
    "scripts/common.py",
)


def package_versions() -> dict[str, str]:
    return {name: importlib.metadata.version(name) for name in PACKAGE_NAMES}


def check_output_paths() -> None:
    ignored = (ROOT / "_tmp").resolve()
    if not PREDICTION_PATH.resolve().is_relative_to(ignored):
        raise ValueError("Per-row v6 SFT predictions must stay under ignored _tmp")
    for path in (REPORT_PATH, PREDICTION_PATH):
        if path.exists():
            raise ValueError(f"v6 SFT dev output already exists: {path.name}; refusing to overwrite")


def require_dev_manifest() -> dict:
    actual = sha256_file(MANIFEST)
    if actual != EXPECTED_MANIFEST_SHA256:
        raise ValueError("v6 manifest differs from the preregistered SHA-256")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    splits = manifest.get("splits")
    if not isinstance(splits, dict) or "dev" not in splits:
        raise ValueError("v6 manifest has no development split")
    if splits["dev"].get("jsonl_sha256") != EXPECTED_DEV_SHA256:
        raise ValueError("v6 development split is not the frozen file")
    if splits["dev"].get("examples") != EXPECTED_ROWS:
        raise ValueError("Unexpected v6 development split size")
    return manifest


def require_adapter(adapter: Path) -> dict:
    manifest_path = adapter / "run_manifest.json"
    weights = adapter / "adapter_model.safetensors"
    if not manifest_path.is_file() or not weights.is_file():
        raise SystemExit(f"v6 SFT adapter is incomplete: {adapter}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "fixed_final_recipe":
        raise SystemExit(
            "Adapter was not trained with the fixed final v6 SFT recipe; "
            "a smoke-check adapter must not be scored as a result"
        )
    recipe = manifest.get("recipe") or {}
    if (manifest.get("executed_max_steps") != recipe.get("max_steps")
            or recipe.get("max_steps") != 400
            or recipe.get("learning_rate") != 3e-4
            or recipe.get("seed") != 20261003
            or recipe.get("gradient_accumulation_steps") != 4
            or recipe.get("max_length") != 256):
        raise SystemExit("Adapter recipe differs from the predeclared v6 SFT recipe")
    if (manifest.get("train_file_sha256")
            != "ea1cd537dcfd924b24eec7997b692a42ae9df757c52f89c904358665ec06a3d3"
            or manifest.get("v6_manifest_sha256") != EXPECTED_MANIFEST_SHA256):
        raise SystemExit("Adapter was not trained on the frozen v6 training split")
    return manifest


def load_baseline_raw(rows: list[dict]) -> dict[str, list[str]]:
    raw: dict[str, list[str]] = {}
    for system in BASELINES:
        path = ROOT / f"_tmp/massive-slots-v6-dev-{system}-predictions.jsonl"
        if not path.is_file():
            raise SystemExit(f"Frozen v6 baseline predictions are missing: {path.name}")
        stored = read_jsonl(path)
        if len(stored) != len(rows):
            raise ValueError(f"{system} predictions have the wrong row count")
        outputs = []
        for row, item in zip(rows, stored, strict=True):
            if str(item.get("id")) != str(row["id"]):
                raise ValueError(f"{system} predictions are not in dev row order")
            if item.get("group_sha256") != row["group_sha256"]:
                raise ValueError(f"{system} predictions belong to different phrase groups")
            if item.get("eval_file_sha256") != EXPECTED_DEV_SHA256:
                raise ValueError(f"{system} predictions were made against another split")
            outputs.append(item["raw_output"])
        raw[system] = outputs
    return raw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", default=str(DEFAULT_ADAPTER))
    parser.add_argument("--role", default="dev",
                        help="Only 'dev' is accepted; the sealed test split is out of scope")
    args = parser.parse_args()
    if args.role != "dev":
        raise SystemExit(
            "The v6 secondary SFT experiment is scored on the development split only; "
            "a test run would need a separately published protocol and model fingerprints"
        )
    if Path.cwd().resolve() != ROOT:
        raise SystemExit("Run from the repository root")
    check_output_paths()
    require_dev_manifest()

    eval_file = DATA / "dev.jsonl"
    if sha256_file(eval_file) != EXPECTED_DEV_SHA256:
        raise ValueError("v6 dev.jsonl changed")
    rows = read_jsonl(eval_file)
    if len(rows) != EXPECTED_ROWS:
        raise ValueError("Unexpected v6 development row count")

    adapter = Path(args.adapter).resolve()
    adapter_manifest = require_adapter(adapter)
    baselines = load_baseline_raw(rows)

    sft_raw, inference = generate_raw_predictions(
        rows, model_name=str(BASE), adapter=str(adapter),
        batch_size=GENERATION["batch_size"],
        max_input_tokens=GENERATION["max_input_tokens"],
        max_new_tokens=GENERATION["max_new_tokens"])
    if sha256_file(eval_file) != EXPECTED_DEV_SHA256:
        raise ValueError("v6 dev.jsonl changed during inference")

    predictions = {**baselines, SFT_SYSTEM: sft_raw}
    settings = BOOTSTRAP
    systems = {
        name: {
            "metrics": score_predictions(rows, raw, bootstrap_samples=settings["samples"],
                                         seed=settings["seed"]),
            "positive_only": positive_only_metrics(rows, raw),
        }
        for name, raw in predictions.items()
    }
    paired = {
        f"{candidate}_minus_{baseline}": paired_bootstrap_f1_delta(
            rows, predictions[baseline], predictions[candidate],
            samples=settings["samples"], seed=settings["seed"])
        for baseline, candidate in combinations(predictions, 2)
    }

    prediction_sha = write_predictions(
        PREDICTION_PATH,
        prediction_payload(rows, sft_raw, EXPECTED_DEV_SHA256),
        exclusive=True)

    report = {
        "study": "MASSIVE zh-CN four-slot v6 secondary QLoRA SFT on the 1,600 training groups",
        "role": "dev",
        "scope": "development split only; the sealed v6 test split is not used",
        "source_partition": "train",
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "eval_file_sha256": EXPECTED_DEV_SHA256,
        "adapter": str(adapter),
        "adapter_sha256": sha256_file(adapter / "adapter_model.safetensors"),
        "adapter_run_manifest": adapter_manifest,
        "base_model_sha256": {
            path.name: sha256_file(path)
            for path in sorted((*BASE.glob("*.safetensors"), BASE / "tokenizer.json"))
            if path.is_file()
        },
        "package_versions": package_versions(),
        "code_sha256": {name: sha256_file(ROOT / name) for name in CODE_FILES},
        "prediction_file_sha256": prediction_sha,
        "generation": {**GENERATION, **inference},
        "bootstrap": settings,
        "systems": systems,
        "paired_f1": paired,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "sft_f1": systems[SFT_SYSTEM]["metrics"]["micro_f1"],
        "new_hybrid_f1": systems["new_hybrid"]["metrics"]["micro_f1"],
        "new_bio_f1": systems["new_bio"]["metrics"]["micro_f1"],
        "sft_minus_new_bio_ci95": paired[f"{SFT_SYSTEM}_minus_new_bio"]["delta_ci95"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
