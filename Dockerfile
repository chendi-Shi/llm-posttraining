# syntax=docker/dockerfile:1.7

FROM python:3.10-slim-bookworm AS runtime-base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-serve.txt /app/requirements-serve.txt
RUN python -m pip install --no-cache-dir --disable-pip-version-check -r requirements-serve.txt


# This stage is intentionally inspectable. It contains the source archive and
# full dev report, and must never be deployed as the serving image.
FROM runtime-base AS trained

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY scripts/evaluate_massive_linear.py scripts/massive_linear_text.py scripts/massive_metrics.py scripts/massive_task.py /app/scripts/

RUN mkdir -p data/raw outputs reports attribution \
    && curl --fail --location --show-error --silent --retry 3 --retry-delay 2 \
       https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.0.tar.gz \
       --output data/raw/amazon-massive-dataset-1.0.tar.gz \
    && printf '%s\n' '7df623fd2d300a4d235d6ee5bd396c9a28258d3a0ccb29abdb054506eba153f8  data/raw/amazon-massive-dataset-1.0.tar.gz' \
       | sha256sum --check --status \
    && tar -xOf data/raw/amazon-massive-dataset-1.0.tar.gz 1.0/LICENSE > attribution/MASSIVE-LICENSE.txt \
    && tar -xOf data/raw/amazon-massive-dataset-1.0.tar.gz 1.0/NOTICE.md > attribution/MASSIVE-NOTICE.md \
    && printf '%s\n' \
       'MASSIVE 1.0 zh-CN, Copyright Amazon.com Inc. or its affiliates.' \
       'Source: https://github.com/alexa/massive' \
       'License: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/' \
       'This service uses a character TF-IDF / LinearSVC model trained on the official MASSIVE 1.0 zh-CN train partition.' \
       'The archive, example text, dev predictions, and test data are not included in the serving image.' \
       'MASSIVE was localized from the SLURP English text dataset; see the official NOTICE.' \
       > attribution/ATTRIBUTION.txt

RUN python scripts/evaluate_massive_linear.py \
    --source data/raw/amazon-massive-dataset-1.0.tar.gz \
    --report reports/massive-linear-dev.json \
    --predictions reports/massive-linear-dev.jsonl \
    --model-out outputs/massive-linear-baseline.joblib \
    && python -c 'import json; r=json.load(open("reports/massive-linear-dev.json", encoding="utf-8")); assert r["source_sha256"] == "7df623fd2d300a4d235d6ee5bd396c9a28258d3a0ccb29abdb054506eba153f8"; assert r["selected_candidate"] == "char_1_3", "Unexpected selected candidate: " + str(r["selected_candidate"]); assert r["test_used"] is False; c=next(c for c in r["candidates"] if c["name"] == "char_1_3"); print("Selected char_1_3; phrase-disjoint dev macro-F1:", c["phrase_disjoint_dev"]["macro_f1"], "accuracy:", c["phrase_disjoint_dev"]["accuracy"])' \
    && sha256sum outputs/massive-linear-baseline.joblib


# A different OS may serialize an equivalent joblib model to different bytes.
# This gate requires the reviewed hash to be supplied explicitly at build time.
FROM trained AS verified

ARG EXPECTED_MODEL_SHA256=fa9f1cb3c72496060492703087eb1a5fb558e35d276e4049cd81fefbe07fe13e
RUN set -eu; \
    actual="$(sha256sum outputs/massive-linear-baseline.joblib | cut -d ' ' -f 1)"; \
    printf 'Built model SHA-256: %s\nExpected model SHA-256: %s\n' "$actual" "$EXPECTED_MODEL_SHA256"; \
    if [ "$actual" != "$EXPECTED_MODEL_SHA256" ]; then \
      echo 'Model hash differs. Inspect the trained stage dev report and explicitly pass the reviewed hash.' >&2; \
      exit 1; \
    fi; \
    printf '%s\n' "$actual" > MODEL_SHA256


FROM runtime-base AS runtime

COPY LICENSE NOTICE /app/
COPY scripts/serve_massive.py scripts/massive_linear_text.py /app/scripts/
COPY --from=verified /app/outputs/massive-linear-baseline.joblib /app/outputs/massive-linear-baseline.joblib
COPY --from=verified /app/MODEL_SHA256 /app/MODEL_SHA256
COPY --from=verified /app/attribution/ /app/attribution/

USER 10001:10001
EXPOSE 8765

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c 'import urllib.request; urllib.request.urlopen("http://127.0.0.1:8765/health/ready", timeout=3).close()'

CMD ["sh", "-c", "exec python scripts/serve_massive.py --backend linear --host 0.0.0.0 --port 8765 --linear-model outputs/massive-linear-baseline.joblib --expected-linear-sha256 \"$(cat /app/MODEL_SHA256)\""]
