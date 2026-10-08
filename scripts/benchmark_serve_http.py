"""Load-test a local review-only HTTP endpoint and report latency percentiles.

This complements `benchmark_massive_slots_latency.py`: that script measures the
model pipeline in-process, while this one measures the whole served path —
HTTP parsing, request validation, the single inference slot, queueing and
response encoding. Both matter, and they answer different questions.

Only `/health/*` and the review-only inference route are exercised. Non-200
responses are classified separately (in particular 429, which is how the
single-slot design rejects concurrent work) and never counted as latency
samples, so a queueing collapse cannot masquerade as fast responses.

Because the service admits one inference at a time and rejects the rest with
429 rather than queueing, **concurrency 1 is the only setting that measures
service throughput**. Sweeping higher concurrency measures how the endpoint
sheds load, which is useful for a different question (an upstream caller must
retry, so effective throughput per caller is lower than the single-slot rate).

Measurements must be taken on an otherwise idle machine.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import threading
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEXT = "明天天气如何"
DEFAULT_SLOT_TEXT = "设置一个星期二和莫娜会议的提醒"


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


def summarise(values: list[float]) -> dict | None:
    if not values:
        return None
    return {
        "samples": len(values),
        "min_ms": round(min(values), 2),
        "p50_ms": round(percentile(values, 0.50), 2),
        "p95_ms": round(percentile(values, 0.95), 2),
        "p99_ms": round(percentile(values, 0.99), 2),
        "max_ms": round(max(values), 2),
        "mean_ms": round(statistics.fmean(values), 2),
    }


def get_json(url: str, timeout: float) -> tuple[int, dict]:
    try:
        with urlopen(url, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        return error.code, json.loads(error.read().decode("utf-8") or "{}")


def post_json(url: str, payload: dict, timeout: float) -> tuple[int, dict]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(url, data=body,
                      headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        try:
            return error.code, json.loads(error.read().decode("utf-8") or "{}")
        except (UnicodeDecodeError, json.JSONDecodeError):
            return error.code, {}


def warm_up(route_url: str, text: str, count: int, timeout: float) -> dict:
    """Pay one-time costs before measuring, and report them separately."""
    latencies: list[float] = []
    for _ in range(count):
        started = time.perf_counter()
        status, _ = post_json(route_url, {"text": text}, timeout)
        latencies.append((time.perf_counter() - started) * 1000)
        if status != 200:
            raise SystemExit(f"Warm-up request failed with status {status}")
    return summarise(latencies) or {}


def measure(route_url: str, text: str, concurrency: int, per_worker: int,
            timeout: float) -> dict:
    latencies: list[float] = []
    statuses: dict[str, int] = {}
    lock = threading.Lock()
    elapsed_ms: list[float] = []

    def worker():
        local_latencies: list[float] = []
        local_statuses: dict[str, int] = {}
        local_server_ms: list[float] = []
        for _ in range(per_worker):
            started = time.perf_counter()
            try:
                status, payload = post_json(route_url, {"text": text}, timeout)
            except (URLError, TimeoutError, OSError) as exc:
                local_statuses[f"transport:{type(exc).__name__}"] = (
                    local_statuses.get(f"transport:{type(exc).__name__}", 0) + 1)
                continue
            key = str(status)
            local_statuses[key] = local_statuses.get(key, 0) + 1
            if status == 200:
                local_latencies.append((time.perf_counter() - started) * 1000)
                reported = payload.get("elapsed_ms")
                if isinstance(reported, (int, float)):
                    local_server_ms.append(float(reported))
        with lock:
            latencies.extend(local_latencies)
            elapsed_ms.extend(local_server_ms)
            for key, value in local_statuses.items():
                statuses[key] = statuses.get(key, 0) + value

    threads = [threading.Thread(target=worker) for _ in range(concurrency)]
    started = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    wall = time.perf_counter() - started
    return {
        "concurrency": concurrency,
        "requests_attempted": concurrency * per_worker,
        "http_status_counts": dict(sorted(statuses.items())),
        "successful_samples": len(latencies),
        "wall_seconds": round(wall, 3),
        "successful_throughput_requests_per_second": (
            round(len(latencies) / wall, 4) if wall else None),
        "client_latency": summarise(latencies),
        "server_reported_elapsed_ms": summarise(elapsed_ms),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True,
                        help="Port of an already running local service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--route", choices=("/v1/intents", "/v1/slots"), default="/v1/intents")
    parser.add_argument("--text", default=None,
                        help="Request text; defaults per route")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--concurrency", default="1",
                        help=("Comma-separated client concurrency. Use 1 to measure the "
                              "service's real throughput: the single-slot design answers "
                              "concurrent work with 429 instead of queueing it, so higher "
                              "values characterise rejection, not capacity."))
    parser.add_argument("--requests-per-worker", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--output", default=None)
    parser.add_argument("--label", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.warmup < 0 or args.requests_per_worker < 1:
        raise SystemExit("Warm-up must be nonnegative and requests per worker positive")
    base = f"http://{args.host}:{args.port}"
    text = args.text or (DEFAULT_SLOT_TEXT if args.route == "/v1/slots" else DEFAULT_TEXT)

    live_status, _ = get_json(base + "/health/live", args.timeout_seconds)
    if live_status != 200:
        raise SystemExit(f"/health/live returned {live_status}; is the service running?")
    ready_status, ready = get_json(base + "/health/ready", args.timeout_seconds)
    if ready_status != 200:
        raise SystemExit(f"/health/ready returned {ready_status}: {ready}")

    route_url = base + args.route
    warmup = warm_up(route_url, text, args.warmup, args.timeout_seconds) if args.warmup else None
    sweeps = []
    for raw in args.concurrency.split(","):
        level = int(raw)
        if level < 1:
            raise SystemExit("Concurrency must be at least 1")
        sweeps.append(measure(route_url, text, level, args.requests_per_worker,
                              args.timeout_seconds))

    report = {
        "study": "local review-only HTTP endpoint load characterisation",
        "route": args.route,
        "label": args.label,
        "base_url": base,
        "request_text_chars": len(text),
        "model_version": ready.get("model_version"),
        "warmup": warmup,
        "sweeps": sweeps,
        "notes": [
            "Only HTTP 200 responses contribute latency samples; 429/503/504 are counted "
            "separately so queueing rejection cannot look like fast service.",
            "The single-inference-slot design rejects concurrent work with 429 instead of "
            "queueing it, so concurrency 1 measures throughput and higher concurrency "
            "measures load shedding.",
            "client_latency includes HTTP overhead; server_reported_elapsed_ms is the "
            "elapsed_ms field the service reports for the same requests.",
            "Single-machine local loopback measurement; not a network or capacity result.",
        ],
    }
    text_out = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        target = Path(args.output)
        if target.exists():
            raise SystemExit(f"Refusing to overwrite existing report: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text_out, encoding="utf-8")
    print(text_out)


if __name__ == "__main__":
    main()
