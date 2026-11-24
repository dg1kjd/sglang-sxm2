"""DFly verify applies the constrained-decoding mask before it accepts drafts."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.constrained.base_grammar_backend import GrammarMask
from sglang.srt.speculative.dfly_worker_v2 import DFlyWorkerV2
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

VOCAB = 8


class _BoolMaskGrammar:
    def apply_vocab_mask(self, logits, vocab_mask):
        logits.masked_fill_(~vocab_mask, float("-inf"))


def _worker(block_size):
    worker = DFlyWorkerV2.__new__(DFlyWorkerV2)
    worker.block_size = block_size
    worker._tp_sync = SimpleNamespace(sync=lambda site, tensor: None)
    return worker


class TestDFlyGrammarVerify(CustomTestCase):
    def test_masked_draft_is_rejected_and_bonus_is_allowed(self):
        # Chain: column 0 is the committed token, columns 1.. are drafts.
        candidates = torch.tensor([[0, 5, 6]])
        logits = torch.full((3, VOCAB), -10.0)
        logits[0, 5] = 3.0  # target agrees with draft 5 ...
        logits[0, 3] = 1.0  # ... but only token 3 is allowed by the grammar
        logits[1, 6] = 3.0
        logits[2, 1] = 3.0
        allowed = torch.zeros((3, VOCAB), dtype=torch.bool)
        allowed[0, 3] = True
        allowed[1:, :] = True
        mask = GrammarMask(_BoolMaskGrammar(), allowed)

        num_correct, bonus = _worker(3)._accept(
            SimpleNamespace(sampling_info=None),
            SimpleNamespace(),
            candidates,
            logits,
            mask,
        )

        self.assertEqual(num_correct.tolist(), [0])
        self.assertEqual(bonus.tolist(), [3])

    def test_no_mask_accepts_the_chain(self):
        candidates = torch.tensor([[0, 5, 6]])
        logits = torch.full((3, VOCAB), -10.0)
        logits[0, 5] = 3.0
        logits[1, 6] = 3.0
        logits[2, 1] = 3.0

        num_correct, bonus = _worker(3)._accept(
            SimpleNamespace(sampling_info=None),
            SimpleNamespace(),
            candidates,
            logits,
            None,
        )

        self.assertEqual(num_correct.tolist(), [2])
        self.assertEqual(bonus.tolist(), [1])


if __name__ == "__main__":
    unittest.main()
