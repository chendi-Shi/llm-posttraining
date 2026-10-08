"""HTTP contract tests for the review-only four-slot route.

These run without torch, weights or network access: the slot backend is a stub,
so only the routing, response contract and failure mapping are under test.
"""

from __future__ import annotations

import json
import sys
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from serve_massive import InputTooLong, IntentHTTPServer, slot_report

UTTERANCE = "设置一个星期二和莫娜会议的提醒"
VALID_OUTPUT = '{"slots":[{"type":"date","value":"星期二"},{"type":"person","value":"莫娜"}]}'


class MockSlotExtractor:
    model_version = "slots-mock-v1"
    artifact_hashes = {"base": "a" * 64, "bio": "b" * 64, "dpo_adapter": "c" * 64}

    def extract(self, utterance: str, max_input_tokens: int) -> dict:
        if utterance == "token overflow":
            raise InputTooLong("Prompt exceeds token limit")
        if utterance == "boom":
            raise RuntimeError("backend exploded")
        if utterance == "empty":
            return {"output": '{"slots":[]}', "replaced": False}
        if utterance == "unparsable":
            return {"output": "not json at all", "replaced": False}
        return {"output": VALID_OUTPUT, "replaced": True}


class SlotContractTest(unittest.TestCase):
    def start_server(self, extractor=None, **kwargs):
        extractor = extractor or MockSlotExtractor()
        server = IntentHTTPServer(("127.0.0.1", 0), extractor, **kwargs)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 2)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return f"http://127.0.0.1:{server.server_port}"

    def post(self, base_url: str, route: str, payload, *, raw_body: bytes | None = None):
        body = raw_body if raw_body is not None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = Request(base_url + route, data=body,
                          headers={"Content-Type": "application/json"}, method="POST")
        try:
            response = urlopen(request, timeout=5)
        except HTTPError as error:
            response = error
        with response:
            return response.status, json.loads(response.read().decode("utf-8"))

    def test_slot_route_is_review_only(self):
        base = self.start_server()
        status, body = self.post(base, "/v1/slots", {"text": UTTERANCE})
        self.assertEqual(status, 200)
        self.assertEqual(body["slots"], [{"type": "date", "value": "星期二"},
                                         {"type": "person", "value": "莫娜"}])
        self.assertTrue(body["usable_for_review"])
        self.assertTrue(body["replaced_by_dpo"])
        self.assertTrue(body["review_required"])
        self.assertFalse(body["auto_execute"])
        self.assertTrue(body["abstained"])
        self.assertEqual(body["reason"], "rejection_not_calibrated")
        self.assertEqual(body["model_version"], "slots-mock-v1")
        self.assertGreaterEqual(body["elapsed_ms"], 0)

    def test_health_reports_slot_model_version(self):
        base = self.start_server()
        request = Request(base + "/health/ready")
        with urlopen(request, timeout=5) as response:
            self.assertEqual(response.status, 200)
            self.assertEqual(json.loads(response.read().decode("utf-8"))["model_version"],
                             "slots-mock-v1")

    def test_empty_and_invalid_outputs_are_not_usable(self):
        base = self.start_server()
        for utterance in ("empty", "unparsable"):
            status, body = self.post(base, "/v1/slots", {"text": utterance})
            self.assertEqual(status, 200)
            self.assertFalse(body["usable_for_review"])
            self.assertEqual(body["slots"], [])
            self.assertFalse(body["auto_execute"])

    def test_input_and_backend_failures_are_mapped(self):
        base = self.start_server()
        self.assertEqual(self.post(base, "/v1/slots", {"text": "token overflow"})[0], 413)
        self.assertEqual(self.post(base, "/v1/slots", {"text": "boom"})[0], 500)

    def test_request_validation_is_shared_with_the_intent_route(self):
        base = self.start_server()
        self.assertEqual(self.post(base, "/v1/slots", {"text": "  "})[0], 400)
        self.assertEqual(self.post(base, "/v1/slots", {"text": "a", "other": 1})[0], 400)
        self.assertEqual(self.post(base, "/v1/slots", {"text": "ab"}, raw_body=b"{not json")[0], 400)

    def test_busy_slot_worker_returns_429(self):
        class BlockingSlot(MockSlotExtractor):
            def __init__(self):
                self.started = threading.Event()
                self.release = threading.Event()

            def extract(self, utterance, max_input_tokens):
                self.started.set()
                self.release.wait(timeout=5)
                return {"output": VALID_OUTPUT, "replaced": False}

        blocker = BlockingSlot()
        base = self.start_server(blocker)

        results: list[int] = []

        def first():
            results.append(self.post(base, "/v1/slots", {"text": UTTERANCE})[0])

        thread = threading.Thread(target=first, daemon=True)
        thread.start()
        self.assertTrue(blocker.started.wait(timeout=5))
        status, body = self.post(base, "/v1/slots", {"text": UTTERANCE})
        self.assertEqual(status, 429)
        self.assertEqual(body["error"]["code"], "busy")
        blocker.release.set()
        thread.join(5)
        self.assertEqual(results, [200])

    def test_unknown_route_is_404(self):
        base = self.start_server()
        self.assertEqual(self.post(base, "/v1/nothing", {"text": "hi"})[0], 404)

    def test_slot_report_marks_uncalibrated_rejection(self):
        report = slot_report(UTTERANCE, {"output": VALID_OUTPUT, "replaced": False},
                             "v1", 12.5)
        self.assertTrue(report["abstained"])
        self.assertEqual(report["reason"], "rejection_not_calibrated")
        self.assertFalse(report["auto_execute"])
        self.assertEqual(report["elapsed_ms"], 12.5)


if __name__ == "__main__":
    unittest.main()
