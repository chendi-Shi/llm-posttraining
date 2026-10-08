"""One end-to-end smoke check of the real /v1/slots endpoint.

Loads the frozen v6 BIO bundle and the frozen Qwen base plus v4 DPO adapter
through the HTTP service, issues a single real request, prints the response and
shuts the server down. This proves the route works with real weights; it is not
a latency or quality measurement.
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from serve_massive import IntentHTTPServer  # noqa: E402
from serve_massive_slots import SlotHybridModel  # noqa: E402

UTTERANCE = "设置一个星期二和莫娜会议的提醒"


def main() -> None:
    extractor = SlotHybridModel()
    server = IntentHTTPServer(("127.0.0.1", 0), extractor,
                              max_input_tokens=256, timeout_seconds=120.0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(base + "/health/ready", timeout=10) as response:
            ready = json.loads(response.read().decode("utf-8"))
        payload = json.dumps({"text": UTTERANCE}, ensure_ascii=False).encode("utf-8")
        request = Request(base + "/v1/slots", data=payload,
                          headers={"Content-Type": "application/json"}, method="POST")
        with urlopen(request, timeout=300) as response:
            result = json.loads(response.read().decode("utf-8"))
        print(json.dumps({"health": ready, "response": result}, ensure_ascii=False,
                         indent=2))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)


if __name__ == "__main__":
    main()
