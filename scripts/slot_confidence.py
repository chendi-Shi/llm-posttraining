"""Frozen confidence rule for the v5 BIO plus DPO hybrid."""

from __future__ import annotations

import math
from collections.abc import Mapping

from massive_slots import parse_prediction, user_prompt


MEAN_LOGPROB_THRESHOLD = -0.08


def token_ids(value: object) -> list[int]:
    """Flatten tokenizer chat-template output without changing token values."""
    if isinstance(value, Mapping):
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list) or not all(isinstance(item, int) for item in value):
        raise ValueError("Chat template did not return one token sequence")
    return value


def completion_mean_logprob(model: object, tokenizer: object,
                            utterance: str, answer: str) -> float:
    """Mean conditional log probability of the complete assistant turn.

    This includes the end-of-message token and matches the v5 development
    exploration. It does not compare against an empty answer or alter logits.
    """
    import torch

    prompt = [{"role": "user", "content": user_prompt(utterance)}]
    prefix = token_ids(tokenizer.apply_chat_template(
        prompt, tokenize=True, add_generation_prompt=True))
    full = token_ids(tokenizer.apply_chat_template(
        prompt + [{"role": "assistant", "content": answer}],
        tokenize=True, add_generation_prompt=False))
    if full[:len(prefix)] != prefix or len(full) <= len(prefix):
        raise ValueError("Assistant answer does not follow the generation prompt")
    tokens = torch.tensor(full, dtype=torch.long).unsqueeze(0)
    with torch.inference_mode():
        logits = model(input_ids=tokens, use_cache=False).logits[0]
        log_probs = torch.log_softmax(logits[len(prefix)-1:-1].float(), dim=-1)
        target = tokens[0, len(prefix):]
        values = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
        mean = values.mean().item()
    if not math.isfinite(mean):
        raise ValueError("Nonfinite assistant log probability")
    return mean


def choose_hybrid_output(bio_raw: str, dpo_raw: str,
                         utterance: str, mean_logprob: float | None) -> tuple[str, bool]:
    slots, validity = parse_prediction(dpo_raw, utterance)
    if (slots and all(validity[key] for key in ("json_valid", "schema_valid", "copy_valid"))
            and mean_logprob is not None and math.isfinite(mean_logprob)
            and mean_logprob >= MEAN_LOGPROB_THRESHOLD):
        return dpo_raw, True
    return bio_raw, False
