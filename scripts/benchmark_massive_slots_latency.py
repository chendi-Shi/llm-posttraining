"""Characterise serving latency and throughput for the frozen four-slot system.

The project has reported entity quality but never measured the cost of answering
one request. This harness reports that cost honestly:

- cold start: model loading time, measured separately from request latency;
- warm single-request latency: p50/p95/p99 over a fixed, local, predeclared
  sample of development utterances, with the stage breakdown (BIO, Qwen
  generation, confidence scoring) kept visible;
- throughput under limited concurrency, including the queueing effect of the
  service's single-inference-slot design;
- memory footprint.

It uses only the development split. It is a local characterisation on one CPU
machine and is not a capacity plan or an acceptance result for real traffic.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import statistics
import threading
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/massive-zh/slots-v6"
DEV_FILE = DATA / "dev.jsonl"
EXPECTED_DEV_SHA256 = "6ee1812c5ac9a12e44047cd57e5973be606b9dcdf546d1d30ccf96f4ff1a157c"
BASE = ROOT / "models/Qwen2.5-0.5B-Instruct"
DPO = ROOT / "outputs/massive-slots-v4-balanced-dpo-64"
NEW_BIO = ROOT / "outputs/massive-slots-v6-bio.joblib"
OLD_BIO = ROOT / "outputs/massive-slots-v2-bio.joblib"
REPORT_PATH = ROOT / "reports/massive-slots-v6-serving-latency.json"
PACKAGE_NAMES = ("torch", "transformers", "peft", "bitsandbytes", "scikit-learn", "joblib", "numpy")
GENERATION = {"decoding": "greedy_unconstrained", "max_input_tokens": 256, "max_new_tokens": 64}


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("Cannot take a percentile of no samples")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarise(values: list[float]) -> dict:
    return {
        "samples": len(values),
        "min_ms": round(min(values), 2),
        "p50_ms": round(percentile(values, 0.50), 2),
        "p95_ms": round(percentile(values, 0.95), 2),
        "p99_ms": round(percentile(values, 0.99), 2),
        "max_ms": round(max(values), 2),
        "mean_ms": round(statistics.fmean(values), 2),
    }


def current_rss_mb() -> float | None:
    try:
        import resource  # POSIX only
    except ImportError:
        pass
    try:
        import psutil  # optional
    except ImportError:
        return None
    return round(psutil.Process().memory_info().rss / (1024 * 1024), 1)


def build_pipeline():
    """Load the frozen v6 hybrid once and return a per-utterance callable."""
    import joblib
    import torch

    from common import configure_cpu, load_quantized_base, load_tokenizer
    from evaluate_massive_slots import sha256_file
    from evaluate_massive_slots_bio import MODEL_CONFIG, predict_slots
    from massive_slots import canonical_output, user_prompt
    from slot_confidence import completion_mean_logprob, choose_hybrid_output

    if sha256_file(DEV_FILE) != EXPECTED_DEV_SHA256:
        raise SystemExit("v6 dev.jsonl changed; refusing to benchmark against a different split")

    configure_cpu()
    timings: dict[str, float] = {}

    started = time.perf_counter()
    bundle = joblib.load(NEW_BIO)
    if bundle.get("configuration") != MODEL_CONFIG:
        raise SystemExit("New BIO bundle configuration differs")
    timings["bio_load_seconds"] = round(time.perf_counter() - started, 2)

    started = time.perf_counter()
    tokenizer = load_tokenizer(str(BASE))
    model = load_quantized_base(str(BASE), training=False)
    from peft import PeftModel

    model = PeftModel.from_pretrained(model, str(DPO), is_trainable=False)
    model.eval()
    timings["qwen_load_seconds"] = round(time.perf_counter() - started, 2)

    def run(utterance: str) -> dict:
        stage: dict[str, float] = {}

        mark = time.perf_counter()
        bio_raw = canonical_output(predict_slots(utterance, bundle["vectorizer"], bundle["model"]))
        stage["bio_ms"] = (time.perf_counter() - mark) * 1000

        prompt = [[{"role": "user", "content": user_prompt(utterance)}]]
        encoded = tokenizer.apply_chat_template(
            prompt, tokenize=True, add_generation_prompt=True,
            return_tensors="pt", return_dict=True)
        length = int(encoded["attention_mask"].sum().item())
        if length > GENERATION["max_input_tokens"]:
            raise ValueError(f"Prompt uses {length} tokens, above the fixed bound")

        mark = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **encoded, do_sample=False, num_beams=1,
                max_new_tokens=GENERATION["max_new_tokens"],
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id)
        prompt_width = encoded["input_ids"].shape[1]
        dpo_raw = tokenizer.decode(generated[0, prompt_width:], skip_special_tokens=True).strip()
        stage["qwen_generate_ms"] = (time.perf_counter() - mark) * 1000

        mark = time.perf_counter()
        try:
            score = completion_mean_logprob(model, tokenizer, utterance, dpo_raw)
        except ValueError:
            score = None
        stage["confidence_ms"] = (time.perf_counter() - mark) * 1000

        chosen, replaced = choose_hybrid_output(bio_raw, dpo_raw, utterance, score)
        return {"output": chosen, "replaced": replaced, "stages": stage}

    return run, timings, {"tokenizer": tokenizer, "model": model, "bundle": bundle}


def read_utterances(limit: int | None) -> list[str]:
    rows = [json.loads(line) for line in DEV_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
    if limit is not None:
        rows = rows[:limit]
    return [row["utt"] for row in rows]


def measure_sequential(run, utterances: list[str], warmup: int, repeats: int) -> dict:
    for utterance in utterances[:warmup]:
        run(utterance)
    latencies: list[float] = []
    stages = {"bio_ms": [], "qwen_generate_ms": [], "confidence_ms": []}
    replacements = 0
    calls = 0
    for _ in range(repeats):
        for utterance in utterances:
            mark = time.perf_counter()
            result = run(utterance)
            latencies.append((time.perf_counter() - mark) * 1000)
            for key in stages:
                stages[key].append(result["stages"][key])
            replacements += int(result["replaced"])
            calls += 1
    wall = statistics.fsum(latencies) / 1000
    return {
        "requests": calls,
        "wall_seconds": round(wall, 2),
        "throughput_requests_per_second": round(calls / wall, 4) if wall else None,
        "latency": summarise(latencies),
        "stage_latency": {key: summarise(values) for key, values in stages.items()},
        "hybrid_replacement_rate": round(replacements / calls, 4) if calls else None,
    }


def measure_concurrency(run, utterance: str, concurrency: int, per_worker: int) -> dict:
    """Model the service's single-slot queueing with an explicit lock."""
    lock = threading.Lock()
    latencies: list[float] = []
    errors: list[str] = []

    def worker():
        for _ in range(per_worker):
            mark = time.perf_counter()
            try:
                with lock:
                    run(utterance)
                latencies.append((time.perf_counter() - mark) * 1000)
            except Exception as exc:  # keep the sweep running, report the failure
                errors.append(f"{type(exc).__name__}: {exc}")

    threads = [threading.Thread(target=worker) for _ in range(concurrency)]
    started = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    wall = time.perf_counter() - started
    return {
        "concurrency": concurrency,
        "requests": len(latencies),
        "errors": len(errors),
        "wall_seconds": round(wall, 2),
        "throughput_requests_per_second": round(len(latencies) / wall, 4) if wall else None,
        "latency": summarise(latencies) if latencies else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=30,
                        help="Utterances taken from the start of the frozen dev split")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--concurrency", default="1,2,4")
    parser.add_argument("--requests-per-worker", type=int, default=3)
    parser.add_argument("--output", default=str(REPORT_PATH))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if Path.cwd().resolve() != ROOT:
        raise SystemExit("Run from the repository root")
    if args.limit < 1 or args.repeats < 1 or args.warmup < 0:
        raise SystemExit("Limit and repeats must be positive")

    run, load_timings, _assets = build_pipeline()
    utterances = read_utterances(args.limit)
    rss_after_load = current_rss_mb()

    sequential = measure_sequential(run, utterances, args.warmup, args.repeats)
    rss_after_load_run = current_rss_mb()

    sweeps = []
    for raw in args.concurrency.split(","):
        level = int(raw)
        if level < 1:
            raise SystemExit("Concurrency must be at least 1")
        sweeps.append(measure_concurrency(run, utterances[0], level, args.requests_per_worker))

    report = {
        "study": "MASSIVE zh-CN four-slot v6 hybrid local serving characterisation",
        "scope": "local single-machine CPU measurement; not a capacity plan or traffic acceptance",
        "role": "dev",
        "eval_file_sha256": EXPECTED_DEV_SHA256,
        "sample_utterances": len(utterances),
        "generation": GENERATION,
        "cold_start": load_timings,
        "memory_mb": {"after_load": rss_after_load, "after_sequential_run": rss_after_load_run},
        "sequential": sequential,
        "concurrency_sweep": sweeps,
        "package_versions": {name: importlib.metadata.version(name) for name in PACKAGE_NAMES},
        "notes": [
            "Latency is warm after the stated warm-up requests; cold start is reported separately.",
            "Concurrency uses an explicit lock to model the service's single-inference-slot design.",
            "The confidence stage skips scoring when the generated answer is unparsable.",
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise SystemExit(f"Refusing to overwrite existing report: {output}")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "p50_ms": sequential["latency"]["p50_ms"],
        "p95_ms": sequential["latency"]["p95_ms"],
        "p99_ms": sequential["latency"]["p99_ms"],
        "sequential_rps": sequential["throughput_requests_per_second"],
        "cold_start": load_timings,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
