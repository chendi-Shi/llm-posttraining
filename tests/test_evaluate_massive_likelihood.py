"""Small deterministic checks; no weights, downloads, or real Qwen inference."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate_massive_likelihood import (
    Candidate,
    batched_candidate_logprobs,
    rank_candidates,
    tokenize_candidates,
)


class FakeTokenizer:
    labels = {"A": [1, 3], "B": [2, 4]}

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize
        if add_generation_prompt:
            return [99]
        return [99, *self.labels[messages[-1]["content"]], 98]

    def encode(self, label, *, add_special_tokens):
        assert not add_special_tokens
        return self.labels[label]


class FakeMappingTokenizer(FakeTokenizer):
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        ids = super().apply_chat_template(
            messages, tokenize=tokenize, add_generation_prompt=add_generation_prompt
        )
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


class CandidateLikelihoodTest(unittest.TestCase):
    def test_tokenization_scores_label_without_turn_end_by_default(self):
        candidates = tokenize_candidates(FakeTokenizer(), "请求", ["A", "B"])
        self.assertEqual(candidates[0], Candidate("A", (99, 1, 3), 1))
        with_end = tokenize_candidates(
            FakeTokenizer(), "请求", ["A"], include_turn_end=True
        )
        self.assertEqual(with_end[0].input_ids, (99, 1, 3, 98))
        mapping = tokenize_candidates(FakeMappingTokenizer(), "请求", ["A"])
        self.assertEqual(mapping[0].input_ids, (99, 1, 3))

    def test_full_sequence_can_reverse_greedy_first_token(self):
        # Greedy decoding chooses A because its first token has probability .6.
        # Complete sequence probabilities are A=.6*.1 and B=.4*.9.
        ranked = rank_candidates(
            {"A": [math.log(0.6), math.log(0.1)], "B": [math.log(0.4), math.log(0.9)]}
        )
        self.assertEqual(ranked[0]["label"], "B")
        self.assertEqual(ranked[0]["scored_tokens"], 2)

    def test_length_normalization_is_explicit(self):
        values = {"short": [-0.4], "long": [-0.3, -0.3]}
        self.assertEqual(rank_candidates(values, "sum")[0]["label"], "short")
        self.assertEqual(rank_candidates(values, "mean")[0]["label"], "long")

    def test_model_logits_are_shifted_and_padding_does_not_change_scores(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is not installed in the offline CI environment")

        class FakeModel:
            def __call__(self, *, input_ids, attention_mask, use_cache):
                self_last = input_ids.shape
                logits = torch.full((*self_last, 100), -100.0)
                for row in range(self_last[0]):
                    for pos in range(self_last[1]):
                        previous = input_ids[row, pos].item()
                        if previous == 99:
                            logits[row, pos, 1] = math.log(0.6)
                            logits[row, pos, 2] = math.log(0.4)
                        elif previous == 1:
                            logits[row, pos, 3] = math.log(0.1)
                            logits[row, pos, 0] = math.log(0.9)
                        elif previous == 2:
                            logits[row, pos, 4] = math.log(0.9)
                            logits[row, pos, 0] = math.log(0.1)
                return SimpleNamespace(logits=logits)

        candidates = tokenize_candidates(FakeTokenizer(), "请求", ["A", "B"])
        # A longer third candidate forces right padding for A and B.
        candidates.append(Candidate("C", (99, 2, 4, 0), 1))
        single = batched_candidate_logprobs(
            FakeModel(), candidates, pad_token_id=98, batch_size=1
        )
        paired = batched_candidate_logprobs(
            FakeModel(), candidates, pad_token_id=98, batch_size=3
        )
        self.assertEqual(rank_candidates(paired)[0]["label"], "B")
        self.assertAlmostEqual(paired["A"][0], math.log(0.6), places=5)
        self.assertAlmostEqual(paired["A"][1], math.log(0.1), places=5)
        self.assertAlmostEqual(paired["B"][0], math.log(0.4), places=5)
        self.assertAlmostEqual(paired["B"][1], math.log(0.9), places=5)
        for label in ("A", "B", "C"):
            for left, right in zip(single[label], paired[label], strict=True):
                self.assertAlmostEqual(left, right, places=6)


if __name__ == "__main__":
    unittest.main()
