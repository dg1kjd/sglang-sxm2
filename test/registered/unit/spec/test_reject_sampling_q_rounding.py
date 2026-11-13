"""Chain reject sampling accepts a draft whose q exceeds 1.0 by float rounding,
as FlashInfer's softmax returns for peaked rows, and still rejects broken q
(0, NaN, inf, or far above 1). Both the one-program-per-request kernel and the
vocabulary-tiled path used at small batch with a large vocabulary."""

import unittest

import torch

from sglang.kernels.ops.speculative.reject_sampling import (
    _should_split_final_sampling,
    chain_speculative_sampling_triton,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

GAMMA = 3
DRAFT = 7
# One vocabulary tile (one program per request) and three tiles (split path).
VOCABS = (64, 3 * 4096)


def _num_accept(draft_q: float, target_p: float = 1.0, vocab: int = 64) -> int:
    """Accepted drafts for one request whose three drafts are all token DRAFT."""
    device = "cuda"
    slots = GAMMA + 1
    candidates = torch.full((1, slots), DRAFT, dtype=torch.int64, device=device)
    target_probs = torch.zeros((1, slots, vocab), dtype=torch.float32, device=device)
    target_probs[:, :, DRAFT] = target_p
    target_probs[:, :, DRAFT + 1] = max(0.0, 1.0 - target_p)
    draft_probs = torch.zeros((1, GAMMA, vocab), dtype=torch.float32, device=device)
    draft_probs[:, :, DRAFT] = draft_q
    accept_token_num = torch.empty((1,), dtype=torch.int32, device=device)
    chain_speculative_sampling_triton(
        predicts=torch.empty((slots,), dtype=torch.int32, device=device),
        accept_index=torch.empty((1, slots), dtype=torch.int32, device=device),
        accept_token_num=accept_token_num,
        candidates=candidates,
        retrive_index=torch.arange(slots, dtype=torch.int64, device=device).view(1, -1),
        retrive_next_token=None,
        retrive_next_sibling=None,
        uniform_samples=torch.full((1, slots), 0.5, device=device),
        uniform_samples_for_final_sampling=torch.full((1,), 0.5, device=device),
        target_probs=target_probs,
        draft_probs=draft_probs,
        threshold_single=1.0,
        threshold_acc=1.0,
        deterministic=True,
    )
    return int(accept_token_num.item())


class TestRejectSamplingQRounding(CustomTestCase):
    def test_both_paths_are_covered(self):
        device = torch.device("cuda")
        self.assertFalse(_should_split_final_sampling(1, VOCABS[0] // 4096 + 1, device))
        self.assertTrue(_should_split_final_sampling(1, VOCABS[1] // 4096, device))

    def test_q_rounding_above_one_is_accepted(self):
        for vocab in VOCABS:
            for overshoot in (1.2e-7, 7.6e-6, 4e-5):
                with self.subTest(vocab=vocab, overshoot=overshoot):
                    q = 1.0 + overshoot
                    self.assertEqual(_num_accept(q, q, vocab), GAMMA)
                    self.assertEqual(_num_accept(q, 0.999, vocab), GAMMA)

    def test_broken_q_is_still_rejected(self):
        for vocab in VOCABS:
            for q in (0.0, float("nan"), float("inf"), float("-inf"), 2.0):
                with self.subTest(vocab=vocab, q=q):
                    self.assertEqual(_num_accept(q, vocab=vocab), 0)


if __name__ == "__main__":
    unittest.main()
