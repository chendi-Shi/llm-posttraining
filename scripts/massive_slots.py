"""Exact, copy-only slot targets for MASSIVE 1.0 Chinese utterances."""

from __future__ import annotations

import json
import re
import unicodedata


TARGET_TYPES = ("date", "time", "place_name", "person")
ANNOTATION = re.compile(r"\[\s*([A-Za-z_]+)\s*:\s*([^\[\]]+?)\]")


def compact(text: str) -> str:
    """Remove whitespace only; annotation and source text must then be identical."""
    return "".join(character for character in text if not character.isspace())


def group_key(utterance: str) -> str:
    """Conservative phrase key used to prevent near-identical split leakage."""
    return compact(unicodedata.normalize("NFKC", utterance).casefold())


def user_prompt(utterance: str) -> str:
    return (
        "从原句抽取 date、time、place_name、person 槽位。只输出 JSON 对象，"
        "键为 slots，元素包含 type 和 value；没有目标槽位输出 {\"slots\":[]}。"
        "value 必须逐字复制原句。\n原句："
        + utterance
    )


def canonical_json(slots: list[dict[str, str]]) -> str:
    return json.dumps({"slots": slots}, ensure_ascii=False, separators=(",", ":"))


def canonical_output(slots: list[dict[str, str]]) -> str:
    """Public name shared by the builder and evaluator."""
    return canonical_json(slots)


def parse_prediction(raw: str, utterance: str) -> tuple[list[dict[str, str]] | None, dict]:
    """Validate one generation without repairing it or dropping duplicate slots.

    Invalid outputs return ``None`` so an evaluator can score them as wrong
    while reporting JSON syntax, schema, and source-copy rates separately.
    """
    validity = {
        "json_valid": False,
        "schema_valid": False,
        "copy_valid": False,
        "error": None,
    }
    if not isinstance(raw, str) or not isinstance(utterance, str):
        validity["error"] = "non_string_input"
        return None, validity
    duplicate_keys: list[str] = []

    def unique_object(pairs: list[tuple[str, object]]) -> dict:
        obj = {}
        for key, value in pairs:
            if key in obj:
                duplicate_keys.append(key)
            obj[key] = value
        return obj

    try:
        parsed = json.loads(raw, object_pairs_hook=unique_object)
    except json.JSONDecodeError:
        validity["error"] = "invalid_json"
        return None, validity
    validity["json_valid"] = True
    if duplicate_keys:
        validity["error"] = "duplicate_json_key"
        return None, validity
    if not isinstance(parsed, dict) or set(parsed) != {"slots"} or not isinstance(parsed["slots"], list):
        validity["error"] = "invalid_top_level_schema"
        return None, validity
    slots = parsed["slots"]
    for slot in slots:
        if not isinstance(slot, dict) or set(slot) != {"type", "value"}:
            validity["error"] = "invalid_slot_schema"
            return None, validity
        if (
            not isinstance(slot["type"], str)
            or slot["type"] not in TARGET_TYPES
            or not isinstance(slot["value"], str)
            or not slot["value"].strip()
        ):
            validity["error"] = "invalid_slot_type_or_value"
            return None, validity
    validity["schema_valid"] = True
    if any(slot["value"] not in utterance for slot in slots):
        validity["error"] = "value_not_copied_from_utterance"
        return None, validity
    validity["copy_valid"] = True
    return slots, validity


def parse_annotated_spans(utterance: str, annotated: str) -> list[dict]:
    """Map marked spans to literal substrings and character offsets in ``utt``.

    MASSIVE inserts spaces around Chinese annotation markers. Its unmarked
    annotation therefore differs from ``utt`` in whitespace, while the
    sequence of non-whitespace characters is identical. Mapping by that
    sequence also disambiguates repeated slot values in one utterance.
    """
    if not isinstance(utterance, str) or not isinstance(annotated, str):
        raise ValueError("utt and annot_utt must be strings")
    if not utterance or not annotated:
        raise ValueError("utt and annot_utt must be nonempty")

    pieces: list[tuple[str, str | None]] = []
    previous_end = 0
    for match in ANNOTATION.finditer(annotated):
        plain = annotated[previous_end : match.start()]
        if "[" in plain or "]" in plain:
            raise ValueError("Unparsed annotation marker")
        pieces.append((plain, None))
        pieces.append((match.group(2), match.group(1)))
        previous_end = match.end()
    tail = annotated[previous_end:]
    if "[" in tail or "]" in tail:
        raise ValueError("Unparsed annotation marker")
    pieces.append((tail, None))

    rendered = "".join(text for text, _ in pieces)
    if compact(rendered) != compact(utterance):
        raise ValueError("annot_utt cannot be aligned with utt")

    source_positions = [index for index, char in enumerate(utterance) if not char.isspace()]
    cursor = 0
    spans: list[dict] = []
    for text, slot_type in pieces:
        width = len(compact(text))
        if slot_type is not None and width == 0:
            raise ValueError("Empty annotated slot")
        if slot_type in TARGET_TYPES:
            start = source_positions[cursor]
            end = source_positions[cursor + width - 1] + 1
            value = utterance[start:end]
            if compact(value) != compact(text):
                raise ValueError("Annotated slot is not a source substring")
            spans.append({"type": slot_type, "value": value, "start": start, "end": end})
        cursor += width

    if cursor != len(source_positions):
        raise ValueError("Annotation alignment did not consume utt")
    return spans


def slots_from_annotation(utterance: str, annotated: str) -> list[dict[str, str]]:
    """Return canonical schema values while preserving duplicate slots."""
    return [
        {"type": span["type"], "value": span["value"]}
        for span in parse_annotated_spans(utterance, annotated)
    ]


def parse_annotated(utt: str, annot_utt: str) -> list[dict[str, str]]:
    """Public name shared by the builder and evaluator."""
    return slots_from_annotation(utt, annot_utt)
