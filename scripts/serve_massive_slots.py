"""Frozen v6 BIO plus v4 DPO hybrid, exposed as one utterance-at-a-time inference.

This wraps exactly the system that `evaluate_massive_slots_v6_hybrid.py`
reports on: a frozen v6 character BIO default answer, replaced only when the
frozen v4 DPO adapter returns a valid, copyable, nonempty answer whose complete
assistant turn scores at or above the frozen threshold `-0.08`.

It is a characterisation backend, not a decision system. Slot values are
verbatim copies of the request text and rejection is not calibrated, so the
HTTP layer always marks the answer as a review candidate.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from massive_slots import canonical_output, user_prompt
from serve_massive import InputTooLong
from slot_confidence import MEAN_LOGPROB_THRESHOLD, choose_hybrid_output, completion_mean_logprob


DEFAULT_BASE = Path("models/Qwen2.5-0.5B-Instruct")
DEFAULT_BIO = Path("outputs/massive-slots-v6-bio.joblib")
DEFAULT_DPO = Path("outputs/massive-slots-v4-balanced-dpo-64")
GENERATION = {"decoding": "greedy_unconstrained", "max_new_tokens": 64}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class SlotHybridModel:
    """Loads the frozen BIO bundle and the frozen Qwen plus DPO adapter once."""

    def __init__(self, base: Path = DEFAULT_BASE, bio: Path = DEFAULT_BIO,
                 dpo: Path = DEFAULT_DPO):
        import joblib

        from common import configure_cpu, load_quantized_base, load_tokenizer
        from evaluate_massive_slots_bio import MODEL_CONFIG, predict_slots

        base_file = base / "model.safetensors"
        adapter_file = dpo / "adapter_model.safetensors"
        for path in (base_file, adapter_file, bio):
            if not path.is_file():
                raise FileNotFoundError(path)

        bundle = joblib.load(bio)
        if bundle.get("configuration") != MODEL_CONFIG:
            raise ValueError("BIO bundle configuration differs from the frozen setting")
        if not all(key in bundle for key in ("vectorizer", "model")):
            raise ValueError("BIO bundle is incomplete")

        configure_cpu()
        tokenizer = load_tokenizer(str(base))
        model = load_quantized_base(str(base), training=False)
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(dpo), is_trainable=False)
        model.eval()

        self._predict_slots = predict_slots
        self._bundle = bundle
        self._tokenizer = tokenizer
        self._model = model
        self.artifact_hashes = {
            "base": sha256(base_file),
            "bio": sha256(bio),
            "dpo_adapter": sha256(adapter_file),
        }
        self.model_version = "-".join(value[:12] for value in self.artifact_hashes.values())
        self.confidence_threshold = MEAN_LOGPROB_THRESHOLD

    def extract(self, utterance: str, max_input_tokens: int) -> dict:
        import torch

        bio_raw = canonical_output(
            self._predict_slots(utterance, self._bundle["vectorizer"], self._bundle["model"]))

        tokenizer = self._tokenizer
        encoded = tokenizer.apply_chat_template(
            [[{"role": "user", "content": user_prompt(utterance)}]],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        length = int(encoded["attention_mask"].sum().item())
        if length > max_input_tokens:
            raise InputTooLong(f"Prompt is {length} tokens; limit is {max_input_tokens}")
        prompt_width = encoded["input_ids"].shape[1]

        with torch.inference_mode():
            generated = self._model.generate(
                **encoded,
                do_sample=False,
                num_beams=1,
                max_new_tokens=GENERATION["max_new_tokens"],
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        dpo_raw = tokenizer.decode(
            generated[0, prompt_width:], skip_special_tokens=True).strip()

        try:
            score = completion_mean_logprob(self._model, tokenizer, utterance, dpo_raw)
        except ValueError:
            score = None

        chosen, replaced = choose_hybrid_output(bio_raw, dpo_raw, utterance, score)
        return {
            "output": chosen,
            "replaced": replaced,
            "raw_bio": bio_raw,
            "raw_dpo": dpo_raw,
            "mean_logprob": score,
        }
