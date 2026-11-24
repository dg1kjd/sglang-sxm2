"""SM70 NVFP4 MoE: gate and up halves with different per-expert global scales.

marlin_v100 takes one global scale per expert for the fused W13. When a
checkpoint stores different gate and up global scales, the block scales of one
half are rescaled so that block scale x global scale is unchanged per column.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

import torch

from sglang.srt.layers.quantization.marlin_utils import (
    sm70_nvfp4_marlin_process_scales,
)
from sglang.srt.layers.quantization.modelopt_quant import (
    _sm70_w13_column_multiplier,
)


def _decode_sm70_scales(encoded: torch.Tensor) -> torch.Tensor:
    """Inverse of the S0E5M3 encoding: logical [E, K/16, N] scales x 2^7."""
    num_experts, num_groups, size_n = encoded.shape
    bits = (encoded.view(torch.uint8).to(torch.int32) << 7).to(torch.int16)
    values = bits.view(torch.float16).float().reshape(-1, size_n)
    # The encoder reorders each group of four columns as [0, 2, 1, 3].
    values = values.view(-1, 4)[:, [0, 2, 1, 3]].reshape(-1, size_n)
    return values.reshape(num_experts, num_groups, size_n)


def _random_e4m3_scales(num_experts, num_groups, size_n, seed):
    generator = torch.Generator().manual_seed(seed)
    # All positive E4M3 codes (subnormals included), max 448.
    codes = torch.randint(1, 0x7F, (num_experts, num_groups, size_n), generator=generator)
    scales = codes.to(torch.uint8).view(torch.float8_e4m3fn)
    scales[0, 0, :] = 448.0
    return scales


class TestSm70GateUpScaleFold(unittest.TestCase):
    def _effective(self, scales, gate_scale2, up_scale2):
        size_n = scales.shape[-1]
        common = torch.maximum(gate_scale2, up_scale2)
        multiplier = None
        if not torch.equal(gate_scale2, up_scale2):
            multiplier = _sm70_w13_column_multiplier(
                gate_scale2, up_scale2, common, size_n=size_n
            )
        encoded, factor = sm70_nvfp4_marlin_process_scales(
            scales, torch.float16, column_multiplier=multiplier
        )
        self.assertEqual(factor, 1.0)
        return _decode_sm70_scales(encoded) * (common / 2**7)[:, None, None]

    def test_unequal_power_of_two_scales_are_exact(self):
        num_experts, num_groups, size_n = 4, 3, 16
        scales = _random_e4m3_scales(num_experts, num_groups, size_n, seed=0)
        gate_scale2 = torch.tensor([2.0**-3, 2.0**-4, 2.0**-2, 2.0**-5])
        up_scale2 = torch.tensor([2.0**-2, 2.0**-4, 2.0**-3, 2.0**-5])
        half = size_n // 2
        expected = scales.float().clone()
        expected[..., :half] *= gate_scale2[:, None, None]
        expected[..., half:] *= up_scale2[:, None, None]

        effective = self._effective(scales, gate_scale2, up_scale2)

        self.assertTrue(torch.isfinite(effective).all())
        torch.testing.assert_close(effective, expected, rtol=0, atol=0)

    def test_equal_scales_keep_the_unmodified_encoding(self):
        scales = _random_e4m3_scales(2, 2, 8, seed=1)
        reference, _ = sm70_nvfp4_marlin_process_scales(scales, torch.float16)
        encoded, _ = sm70_nvfp4_marlin_process_scales(
            scales, torch.float16, column_multiplier=None
        )
        self.assertTrue(torch.equal(encoded.view(torch.uint8), reference.view(torch.uint8)))


if __name__ == "__main__":
    unittest.main()
