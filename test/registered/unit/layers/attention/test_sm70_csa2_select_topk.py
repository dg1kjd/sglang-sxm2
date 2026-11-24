"""SM70 CSA2 decode top-k: the same index list, ascending, on every call.

The sparse decode sums the selected keys in list order, so a selection whose
order or tie choice changes between calls changes the attention output.
"""

from __future__ import annotations

import unittest

import torch

from sglang.srt.layers.attention.dsv4.sm70_csa2 import _select_topk
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(
    est_time=30,
    stage="base-b-kernel-unit",
    runner_config="1-gpu-large",
    disabled="SM70 V100 worktree only; do not register GPU CI",
)


def _reference(scores: torch.Tensor, length: int, k: int) -> torch.Tensor:
    """Stable sort by descending score, cut at k, sorted by index; -inf and absent are -1."""
    row = scores[0, :length].cpu()
    order = torch.sort(-row, stable=True).indices[:k]
    picked = order.sort().values
    picked = torch.where(row[picked] > -torch.inf, picked, -1)
    out = torch.full((k,), -1, dtype=torch.int32)
    out[: picked.numel()] = picked.int()
    return out


class TestSm70Csa2SelectTopk(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 7:
            raise unittest.SkipTest("SM70 required")

    def test_ascending_and_repeatable(self):
        torch.manual_seed(3)
        # DSV4.1 decode shapes: index_topk 512 over dense and candidate logits,
        # candidate_topk_blocks 2048 over block maxima; ties and -inf included.
        for k, width, length in ((512, 16384, 16381), (512, 65536, 40000), (2048, 32768, 30001)):
            with self.subTest(k=k, width=width):
                scores = torch.randint(0, 32, (1, width), device="cuda").float()
                scores[0, :: 7] = -torch.inf
                lens = torch.tensor([length], dtype=torch.int32, device="cuda")
                expected = _reference(scores, length, k)
                for _ in range(5):
                    got = _select_topk(scores, lens, k)
                    self.assertTrue(torch.equal(got[0].cpu(), expected))

    def test_short_row(self):
        scores = torch.randn(1, 1024, device="cuda")
        lens = torch.tensor([300], dtype=torch.int32, device="cuda")
        got = _select_topk(scores, lens, 512)[0].cpu()
        expected = torch.cat([torch.arange(300, dtype=torch.int32), torch.full((212,), -1, dtype=torch.int32)])
        self.assertTrue(torch.equal(got, expected))


if __name__ == "__main__":
    unittest.main()
