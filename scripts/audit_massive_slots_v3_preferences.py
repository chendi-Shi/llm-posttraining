"""Verify v3 train-only preferences and publish aggregate facts without utterances."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from build_massive_slots_v3_preferences import (
    CANDIDATE_SHA256, QUOTAS, select_candidates, semantically_wrong,
    token_length,
)
from common import load_tokenizer
from evaluate_massive_slots import read_jsonl, sha256_file
from massive_slots import parse_prediction


CANDIDATES = Path("data/massive-zh/slots-v3/candidates.jsonl")
PREFERENCES = Path("data/massive-zh/slots-v3/preferences.jsonl")
MANIFEST = Path("data/massive-zh/slots-v3/preferences-manifest.json")
V2_MANIFEST = Path("data/massive-zh/slots-v2/manifest.json")
REPORT = Path("reports/massive-slots-v3-preference-audit.json")


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if sha256_file(CANDIDATES) != CANDIDATE_SHA256:
        raise ValueError("v3 candidate pool differs")
    if sha256_file(PREFERENCES) != manifest.get("preference_file_sha256"):
        raise ValueError("v3 preference pairs differ")
    rows = select_candidates(read_jsonl(CANDIDATES))
    pairs = read_jsonl(PREFERENCES)
    if (len(rows) != len(pairs) or len(pairs) != 256
            or [row["group_sha256"] for row in rows] != manifest["pair_group_sha256"]):
        raise ValueError("Preference pairs do not match the fixed selection")
    old = json.loads(V2_MANIFEST.read_text(encoding="utf-8"))
    old_groups = {item["group_sha256"] for split in old["splits"].values()
                  for item in split["groups"]}
    if old_groups & set(manifest["pair_group_sha256"]):
        raise ValueError("v3 preference rows overlap the v2 locked study")
    tokenizer = load_tokenizer("models/Qwen2.5-0.5B-Instruct")
    bucket_counts: Counter[str] = Counter()
    rejected_validity: Counter[str] = Counter()
    longest = 0
    for row, pair in zip(rows, pairs, strict=True):
        if pair["prompt"] != row["prompt"] or pair["chosen"] != row["completion"]:
            raise ValueError("Preference prompt/chosen differs from gold source")
        if len(pair["rejected"]) != 1 or pair["rejected"][0]["role"] != "assistant":
            raise ValueError("Preference rejected reply has wrong format")
        chosen = pair["chosen"][0]["content"]
        rejected = pair["rejected"][0]["content"]
        gold, validity = parse_prediction(chosen, row["utt"])
        if gold != row["slots"] or not all(validity[name] for name in
                                             ("json_valid", "schema_valid", "copy_valid")):
            raise ValueError("Invalid chosen reply")
        if rejected == chosen or not semantically_wrong(rejected, row):
            raise ValueError("Rejected reply is not a different wrong answer")
        length = max(token_length(tokenizer, row["prompt"], chosen),
                     token_length(tokenizer, row["prompt"], rejected))
        if length > 256:
            raise ValueError("Preference reply exceeds the frozen length")
        longest = max(longest, length)
        bucket_counts[row["bucket"]] += 1
        parsed, _ = parse_prediction(rejected, row["utt"])
        rejected_validity["valid" if parsed is not None else "invalid"] += 1
    if dict(bucket_counts) != QUOTAS or longest != manifest["max_observed_pair_tokens"]:
        raise ValueError("Preference quotas or token length differs")
    report = {
        "study": "MASSIVE zh-CN four-slot v3 train-only preference audit",
        "pair_count": len(pairs),
        "pair_bucket_counts": dict(bucket_counts),
        "rejected_sources": manifest["rejected_sources"],
        "rejected_parse_validity": dict(rejected_validity),
        "max_observed_pair_tokens": longest,
        "v2_group_overlap_count": 0,
        "candidate_file_sha256": sha256_file(CANDIDATES),
        "preference_file_sha256": sha256_file(PREFERENCES),
        "preference_manifest_sha256": sha256_file(MANIFEST),
        "v2_manifest_sha256": sha256_file(V2_MANIFEST),
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
