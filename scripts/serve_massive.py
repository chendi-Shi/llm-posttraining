"""Local, review-only HTTP service for MASSIVE intent classifiers.

The model has no calibrated confidence or out-of-domain detector. Every
prediction is therefore a candidate for human review, never an action.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Protocol

from massive_linear_text import model_text


LOCKED_LINEAR_SHA256 = "fa9f1cb3c72496060492703087eb1a5fb558e35d276e4049cd81fefbe07fe13e"


class InputTooLong(ValueError):
    pass


class Busy(RuntimeError):
    pass


class InferenceTimeout(RuntimeError):
    pass


class ServiceUnavailable(RuntimeError):
    pass


class Classifier(Protocol):
    model_version: str

    def predict(self, text: str, max_input_tokens: int) -> str: ...


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class LinearIntentModel:
    """Loads the frozen local TF-IDF/LinearSVC bundle after hash verification."""

    def __init__(self, model_path: Path, expected_sha256: str = LOCKED_LINEAR_SHA256):
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        if len(expected_sha256) != 64 or any(char not in "0123456789abcdef" for char in expected_sha256.lower()):
            raise ValueError("Expected linear model SHA-256 must be 64 hexadecimal characters")
        digest = sha256(model_path)
        if digest != expected_sha256.lower():
            raise ValueError(f"Linear model SHA-256 mismatch: {digest}")

        # joblib.load can execute code. The hash above must refer to a locally
        # trained, reviewed artifact; never accept an arbitrary uploaded file.
        import joblib

        bundle = joblib.load(model_path)
        if not isinstance(bundle, dict) or bundle.get("normalizer") != "NFKC-casefold-strip":
            raise ValueError("Linear model has an unexpected bundle format or normalizer")
        labels = bundle.get("labels")
        if not isinstance(labels, list) or len(labels) != 60 or any(
            not isinstance(label, str) or not label for label in labels
        ) or len(set(labels)) != 60:
            raise ValueError("Linear model must contain 60 unique intent labels")
        vectorizer = bundle.get("vectorizer")
        classifier = bundle.get("classifier")
        if not callable(getattr(vectorizer, "transform", None)) or not callable(getattr(classifier, "predict", None)):
            raise ValueError("Linear model lacks a vectorizer or classifier")
        classes = {str(label) for label in getattr(classifier, "classes_", [])}
        if classes != set(labels):
            raise ValueError("Linear classifier classes differ from the bundle labels")

        self._vectorizer = vectorizer
        self._classifier = classifier
        self._labels = set(labels)
        self.artifact_hashes = {"linear_model": digest}
        self.model_version = f"linear-{digest[:12]}"

    def predict(self, text: str, max_input_tokens: int) -> str:
        # The HTTP character limit applies to both backends. This model has no
        # tokenizer; max_input_tokens applies only to the SFT backend.
        features = self._vectorizer.transform([model_text(text)])
        result = self._classifier.predict(features)
        candidate = str(result[0])
        if candidate not in self._labels:
            raise RuntimeError("Linear model emitted an invalid intent label")
        return candidate


class IntentModel:
    """Loads the same base, adapter, prompt and trie decoder as offline eval."""

    def __init__(self, model_path: Path, adapter_path: Path, labels_path: Path):
        # These imports are deferred so the HTTP contract can be tested without
        # loading the large model or importing the training dependencies.
        import torch
        from peft import PeftModel

        from common import configure_cpu, load_quantized_base, load_tokenizer
        from massive_task import parse_prediction, user_prompt

        base_file = model_path / "model.safetensors"
        adapter_file = adapter_path / "adapter_model.safetensors"
        for required in (base_file, adapter_file, labels_path):
            if not required.is_file():
                raise FileNotFoundError(required)

        labels = json.loads(labels_path.read_text(encoding="utf-8"))
        if not isinstance(labels, list) or not labels or any(
            not isinstance(label, str) or not label for label in labels
        ) or len(set(labels)) != len(labels):
            raise ValueError("Intent labels must be a nonempty list of unique strings")

        configure_cpu()
        tokenizer = load_tokenizer(str(model_path))
        encoded_labels = {
            label: tokenizer.encode(label, add_special_tokens=False)
            for label in labels
        }
        if any(not tokens for tokens in encoded_labels.values()):
            raise ValueError("An intent label produced no tokens")
        if len({tuple(tokens) for tokens in encoded_labels.values()}) != len(labels):
            raise ValueError("Two intent labels have the same token sequence")

        trie: dict = {}
        for tokens in encoded_labels.values():
            node = trie
            for token in tokens:
                node = node.setdefault(token, {})
            node[None] = True

        model = load_quantized_base(str(model_path), training=False)
        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False)
        model.eval()

        self._torch = torch
        self._model = model
        self._tokenizer = tokenizer
        self._labels = labels
        self._trie = trie
        self._max_new_tokens = max(len(tokens) for tokens in encoded_labels.values()) + 1
        self._parse_prediction = parse_prediction
        self._user_prompt = user_prompt
        self.artifact_hashes = {
            "base": sha256(base_file),
            "adapter": sha256(adapter_file),
            "labels": sha256(labels_path),
        }
        self.model_version = "-".join(value[:12] for value in self.artifact_hashes.values())

    def predict(self, text: str, max_input_tokens: int) -> str:
        tokenizer = self._tokenizer
        encoded = tokenizer.apply_chat_template(
            [[{"role": "user", "content": self._user_prompt(text)}]],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        input_length = int(encoded["attention_mask"].sum().item())
        if input_length > max_input_tokens:
            raise InputTooLong(f"Prompt is {input_length} tokens; limit is {max_input_tokens}")
        prompt_width = encoded["input_ids"].shape[1]

        def allowed_next_tokens(_batch_id, input_ids):
            node = self._trie
            for token_id in input_ids[prompt_width:].tolist():
                if token_id not in node:
                    return [tokenizer.eos_token_id]
                node = node[token_id]
            allowed = [token_id for token_id in node if token_id is not None]
            if None in node:
                allowed.append(tokenizer.eos_token_id)
            return allowed

        with self._torch.inference_mode():
            generated = self._model.generate(
                **encoded,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                prefix_allowed_tokens_fn=allowed_next_tokens,
            )
        raw = tokenizer.decode(generated[0, prompt_width:], skip_special_tokens=True).strip()
        candidate = self._parse_prediction(raw, self._labels)
        if candidate is None:
            raise RuntimeError("Model emitted an invalid intent label")
        return candidate


class IntentHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 8

    def __init__(
        self,
        address: tuple[str, int],
        classifier: Classifier,
        *,
        max_chars: int = 500,
        max_body_bytes: int = 4096,
        max_input_tokens: int = 512,
        timeout_seconds: float = 30.0,
    ):
        if address[0] not in {"127.0.0.1", "0.0.0.0"}:
            raise ValueError("Host must be 127.0.0.1 or explicitly 0.0.0.0")
        if max_chars < 1 or max_body_bytes < 1 or max_input_tokens < 1 or timeout_seconds <= 0:
            raise ValueError("All service limits must be positive")
        self.classifier = classifier
        self.max_chars = max_chars
        self.max_body_bytes = max_body_bytes
        self.max_input_tokens = max_input_tokens
        self.timeout_seconds = timeout_seconds
        self._inference_slot = threading.BoundedSemaphore(1)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="intent-inference")
        self._state_lock = threading.Lock()
        self._timed_out_worker = False
        try:
            super().__init__(address, IntentHandler)
        except Exception:
            self._executor.shutdown(wait=False, cancel_futures=True)
            raise

    @property
    def ready(self) -> bool:
        with self._state_lock:
            return not self._timed_out_worker

    def classify(self, text: str) -> str:
        if not self.ready:
            raise ServiceUnavailable("A timed-out inference is still running")
        if not self._inference_slot.acquire(blocking=False):
            raise Busy("The inference worker is occupied")
        try:
            future = self._executor.submit(self.classifier.predict, text, self.max_input_tokens)
        except Exception:
            self._inference_slot.release()
            raise

        def finished(_future):
            with self._state_lock:
                self._timed_out_worker = False
            self._inference_slot.release()

        future.add_done_callback(finished)
        try:
            return future.result(timeout=self.timeout_seconds)
        except FutureTimeout as exc:
            with self._state_lock:
                if not future.done():
                    self._timed_out_worker = True
            raise InferenceTimeout("Inference exceeded the response deadline") from exc

    def server_close(self):
        super().server_close()
        # A Python thread cannot forcibly stop a hung model call. The HTTP
        # deadline returns 504; the process must be restarted if it stays hung.
        self._executor.shutdown(wait=False, cancel_futures=True)


class IntentHandler(BaseHTTPRequestHandler):
    server: IntentHTTPServer
    protocol_version = "HTTP/1.1"

    def setup(self):
        super().setup()
        self.connection.settimeout(10)

    def _send_json(self, status: int, payload: dict):
        body = (json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True

    def _error(self, status: int, code: str, message: str):
        self._send_json(status, {"error": {"code": code, "message": message}})

    def do_GET(self):
        if self.path == "/health/live":
            self._send_json(200, {"status": "alive"})
        elif self.path == "/health/ready":
            if self.server.ready:
                self._send_json(200, {
                    "status": "ready",
                    "model_version": self.server.classifier.model_version,
                })
            else:
                self._send_json(503, {"status": "unavailable"})
        else:
            self._error(404, "not_found", "Unknown route")

    def do_POST(self):
        if self.path != "/v1/intents":
            self._error(404, "not_found", "Unknown route")
            return
        if self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower() != "application/json":
            self._error(415, "unsupported_media_type", "Content-Type must be application/json")
            return
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            self._error(411, "length_required", "Content-Length is required")
            return
        try:
            length = int(raw_length)
        except ValueError:
            self._error(400, "invalid_length", "Content-Length must be an integer")
            return
        if length < 1:
            self._error(400, "empty_body", "Request body is empty")
            return
        if length > self.server.max_body_bytes:
            self._error(413, "body_too_large", "Request body exceeds the byte limit")
            return
        try:
            body = self.rfile.read(length)
        except (socket.timeout, TimeoutError):
            self._error(408, "read_timeout", "Timed out reading the request")
            return
        if len(body) != length:
            self._error(400, "incomplete_body", "Request body is incomplete")
            return
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._error(400, "invalid_json", "Request body must be UTF-8 JSON")
            return
        if not isinstance(payload, dict) or set(payload) != {"text"} or not isinstance(payload["text"], str):
            self._error(400, "invalid_request", "Expected exactly one string field: text")
            return
        text = payload["text"].strip()
        if not text:
            self._error(400, "empty_text", "text must not be blank")
            return
        if len(text) > self.server.max_chars:
            self._error(413, "text_too_long", "text exceeds the character limit")
            return

        start = time.perf_counter()
        try:
            candidate = self.server.classify(text)
        except InputTooLong as exc:
            self._error(413, "prompt_too_long", str(exc))
            return
        except Busy as exc:
            self._error(429, "busy", str(exc))
            return
        except ServiceUnavailable as exc:
            self._error(503, "unavailable", str(exc))
            return
        except InferenceTimeout as exc:
            self._error(504, "inference_timeout", str(exc))
            return
        except Exception:
            self._error(500, "inference_failed", "Inference failed; inspect the local service log")
            logging.exception("Inference raised an exception")
            return
        self._send_json(200, {
            "intent": None,
            "candidate_intent": candidate,
            "abstained": True,
            "reason": "confidence_not_calibrated",
            "review_required": True,
            "auto_execute": False,
            "model_version": self.server.classifier.model_version,
            "elapsed_ms": round((time.perf_counter() - start) * 1000, 3),
        })

    def log_message(self, format, *args):
        # BaseHTTPRequestHandler logs the route and status, not request text.
        super().log_message(format, *args)


def parse_args():
    parser = argparse.ArgumentParser(description="Review-only local MASSIVE intent service")
    parser.add_argument("--host", choices=("127.0.0.1", "0.0.0.0"), default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--backend", choices=("linear", "sft"), default="linear")
    parser.add_argument("--linear-model", type=Path, default=Path("outputs/massive-linear-baseline.joblib"))
    parser.add_argument("--expected-linear-sha256", default=LOCKED_LINEAR_SHA256)
    parser.add_argument("--model", type=Path, default=Path("models/Qwen2.5-0.5B-Instruct"))
    parser.add_argument("--adapter", type=Path, default=Path("outputs/massive-sft"))
    parser.add_argument("--labels", type=Path, default=Path("data/massive-zh/intents.json"))
    parser.add_argument("--max-chars", type=int, default=500)
    parser.add_argument("--max-body-bytes", type=int, default=4096)
    parser.add_argument("--max-input-tokens", type=int, default=512)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.backend == "linear":
        classifier = LinearIntentModel(args.linear_model, args.expected_linear_sha256)
    else:
        classifier = IntentModel(args.model, args.adapter, args.labels)
    with IntentHTTPServer(
        (args.host, args.port),
        classifier,
        max_chars=args.max_chars,
        max_body_bytes=args.max_body_bytes,
        max_input_tokens=args.max_input_tokens,
        timeout_seconds=args.timeout_seconds,
    ) as server:
        print(json.dumps({
            "listening": f"http://{args.host}:{server.server_port}",
            "model_version": classifier.model_version,
            "artifact_sha256": classifier.artifact_hashes,
            "backend": args.backend,
            "mode": "review_only",
        }), flush=True)
        try:
            server.serve_forever(poll_interval=0.2)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
