"""HTTP contract tests run without torch, model weights or network downloads."""

from __future__ import annotations

import json
import sys
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from serve_massive import InputTooLong, IntentHTTPServer, LinearIntentModel, parse_args


class MockClassifier:
    model_version = "mock-v1"

    def predict(self, text: str, max_input_tokens: int) -> str:
        if text == "token overflow":
            raise InputTooLong("Prompt exceeds token limit")
        return "weather_query"


class BlockingClassifier(MockClassifier):
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()

    def predict(self, text: str, max_input_tokens: int) -> str:
        self.started.set()
        self.release.wait(timeout=5)
        return "weather_query"


class ServiceContractTest(unittest.TestCase):
    def start_server(self, classifier=None, **kwargs):
        classifier = classifier or MockClassifier()
        server = IntentHTTPServer(("127.0.0.1", 0), classifier, **kwargs)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 2)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return f"http://127.0.0.1:{server.server_port}"

    def request(self, base_url: str, route: str, payload=None, *, headers=None):
        if payload is None:
            request = Request(base_url + route)
        else:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            request = Request(
                base_url + route,
                data=body,
                headers=headers or {"Content-Type": "application/json"},
                method="POST",
            )
        try:
            response = urlopen(request, timeout=3)
        except HTTPError as error:
            response = error
        with response:
            return response.status, json.loads(response.read().decode("utf-8"))

    def test_health_and_candidate_are_review_only(self):
        base = self.start_server()
        self.assertEqual(self.request(base, "/health/live")[0], 200)
        ready_status, ready = self.request(base, "/health/ready")
        self.assertEqual(ready_status, 200)
        self.assertEqual(ready["model_version"], "mock-v1")
        status, result = self.request(base, "/v1/intents", {"text": "明天天气如何"})
        self.assertEqual(status, 200)
        self.assertIsNone(result["intent"])
        self.assertEqual(result["candidate_intent"], "weather_query")
        self.assertTrue(result["abstained"])
        self.assertTrue(result["review_required"])
        self.assertFalse(result["auto_execute"])
        self.assertEqual(result["reason"], "confidence_not_calibrated")

    def test_default_backend_and_linear_bundle_normalization(self):
        with patch.object(sys, "argv", ["serve_massive.py"]):
            args = parse_args()
            self.assertEqual(args.backend, "linear")
            self.assertEqual(args.host, "127.0.0.1")
        with patch.object(sys, "argv", ["serve_massive.py", "--host", "0.0.0.0"]):
            self.assertEqual(parse_args().host, "0.0.0.0")

        labels = [f"intent_{index}" for index in range(60)]
        seen = []

        class FakeVectorizer:
            def transform(self, values):
                seen.extend(values)
                return values

        class FakeClassifier:
            classes_ = labels

            def predict(self, values):
                return [labels[0]]

        bundle = {
            "normalizer": "NFKC-casefold-strip",
            "labels": labels,
            "vectorizer": FakeVectorizer(),
            "classifier": FakeClassifier(),
        }
        fake_joblib = SimpleNamespace(load=lambda _path: bundle)
        path = Path("mocked-local-model.joblib")
        digest = "a" * 64
        with patch.object(Path, "is_file", return_value=True), patch(
            "serve_massive.sha256", return_value=digest
        ), patch.dict(sys.modules, {"joblib": fake_joblib}):
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                LinearIntentModel(path, "0" * 64)
            model = LinearIntentModel(path, digest)
        self.assertEqual(model.predict("  ＡＢＣ  ", 1), labels[0])
        self.assertEqual(seen, ["abc"])
        self.assertEqual(model.artifact_hashes["linear_model"], digest)
        self.assertEqual(model.model_version, f"linear-{digest[:12]}")

    def test_validation_and_token_limit(self):
        base = self.start_server(max_chars=20, max_body_bytes=100)
        cases = [
            ({"text": "  "}, 400, "empty_text"),
            ({"text": 10}, 400, "invalid_request"),
            ({"text": "okay", "extra": 1}, 400, "invalid_request"),
            ({"text": "x" * 21}, 413, "text_too_long"),
            ({"text": "token overflow"}, 413, "prompt_too_long"),
            ({"text": "x" * 101}, 413, "body_too_large"),
        ]
        for payload, expected_status, expected_code in cases:
            with self.subTest(payload=payload):
                status, result = self.request(base, "/v1/intents", payload)
                self.assertEqual(status, expected_status)
                self.assertEqual(result["error"]["code"], expected_code)

    def test_busy_timeout_and_readiness_recovery(self):
        classifier = BlockingClassifier()
        base = self.start_server(classifier, timeout_seconds=0.2)
        first_result = {}

        def first_request():
            first_result["value"] = self.request(base, "/v1/intents", {"text": "first"})

        worker = threading.Thread(target=first_request)
        worker.start()
        self.addCleanup(classifier.release.set)
        self.assertTrue(classifier.started.wait(timeout=2))
        busy_status, busy = self.request(base, "/v1/intents", {"text": "second"})
        self.assertEqual(busy_status, 429)
        self.assertEqual(busy["error"]["code"], "busy")
        worker.join(timeout=2)
        self.assertFalse(worker.is_alive())
        self.assertEqual(first_result["value"][0], 504)
        self.assertEqual(self.request(base, "/health/ready")[0], 503)
        classifier.release.set()
        for _ in range(20):
            if self.request(base, "/health/ready")[0] == 200:
                break
            time.sleep(0.05)
        else:
            self.fail("readiness did not recover after inference finished")


if __name__ == "__main__":
    unittest.main()
