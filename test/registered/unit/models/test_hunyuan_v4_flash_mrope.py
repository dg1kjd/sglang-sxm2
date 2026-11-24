"""Hy4-Flash attention rope with vision on: the MRoPE wrapper reads the plain
rope's fp32 table.

A text token's (t, h, w) are equal, so prefill (MRoPE, 2-D positions) must
rotate it exactly as decode (plain rope, 1-D positions) does, and the MRoPE
must not keep a second table (256 MiB a rope base at 1M positions).
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.model_executor.model_runner_components import load_model_utils
from sglang.srt.models.hunyuan_v4_flash import _build_mrope, _FlashMRotaryEmbedding
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="nightly", runner_config="1-gpu-small")

ROPE_DIM = 64
SECTION = (12, 10, 10)
MAX_POSITION = 1 << 20
TOKENS = 2048


def _build(theta, neox_style, max_position=MAX_POSITION):
    with torch.device("cuda"):
        rope = get_rope(
            ROPE_DIM,
            rotary_dim=ROPE_DIM,
            max_position=max_position,
            base=theta,
            is_neox_style=neox_style,
        )
        before = torch.cuda.memory_allocated()
        wrapper = _FlashMRotaryEmbedding(
            rope,
            _build_mrope(
                rope_dim=ROPE_DIM,
                theta=theta,
                section=SECTION,
                interleaved=True,
                neox_style=neox_style,
            ),
        )
        built = torch.cuda.memory_allocated() - before
    return rope, wrapper, built


def _qk(gen):
    q = torch.randn(TOKENS, 8, ROPE_DIM, device="cuda", generator=gen).half()
    k = torch.randn(TOKENS, 1, ROPE_DIM, device="cuda", generator=gen).half()
    return q, k


class TestHy4FlashMRope(CustomTestCase):
    def test_text_positions_match_plain_rope(self):
        gen = torch.Generator(device="cuda").manual_seed(0)
        for theta in (10000, 10000000):
            for neox_style in (False, True):
                with self.subTest(theta=theta, neox_style=neox_style):
                    rope, wrapper, built = _build(theta, neox_style)
                    self.assertLess(built, 1 << 20, "MRoPE built its own table")
                    pos = torch.randint(
                        0, 262144, (TOKENS,), device="cuda", generator=gen
                    )
                    q, k = _qk(gen)
                    q1, k1 = wrapper(pos, q.clone(), k.clone())
                    xyz = pos[None].expand(3, -1).contiguous()
                    q2, k2 = wrapper(xyz, q.clone(), k.clone())
                    self.assertEqual(rope.cos_sin_cache.dtype, torch.float32)
                    self._assert_same_rotation(q2, q1)
                    self._assert_same_rotation(k2, k1)

    def test_loader_reservation_grows_only_the_shared_table(self):
        # reserve_rope_cache_for_long_sequences grows rope tables to the served
        # context after loading; the MRoPE's one-row table must stay one row.
        rope, wrapper, _ = _build(10000, False, max_position=4096)
        model = nn.Module()
        model.layer = nn.Module()
        model.layer.rotary_emb = wrapper
        model_bag = SimpleNamespace(context_length=16384)
        spec_bag = SimpleNamespace(
            speculative_num_steps=None, speculative_num_draft_tokens=None
        )
        with patch.object(load_model_utils, "get_model", return_value=model_bag):
            with patch.object(load_model_utils, "get_spec", return_value=spec_bag):
                load_model_utils.reserve_rope_cache_for_long_sequences(model, None)
        self.assertGreaterEqual(rope.cos_sin_cache.shape[0], 16384)
        self.assertEqual(wrapper.mrope.cos_sin_cache.shape[0], 1)

    def _assert_same_rotation(self, got, want):
        # Same fp32 table and math; the two kernels may round a product
        # differently, which moves a rare element by one fp16 step (|x| < 8).
        diff = (got.float() - want.float()).abs()
        self.assertLessEqual(diff.max().item(), 2**-8)
        self.assertLess((diff > 0).float().mean().item(), 1e-3)

    def test_vision_positions_match_fp32_reference(self):
        gen = torch.Generator(device="cuda").manual_seed(1)
        for neox_style in (False, True):
            with self.subTest(neox_style=neox_style):
                _, wrapper, _ = _build(10000000, neox_style)
                with torch.device("cuda"):
                    reference = get_rope(
                        ROPE_DIM,
                        rotary_dim=ROPE_DIM,
                        max_position=4096,
                        base=10000000,
                        is_neox_style=neox_style,
                        rope_scaling={
                            "rope_type": "default",
                            "mrope_section": list(SECTION),
                            "mrope_interleaved": True,
                        },
                    )
                xyz = torch.randint(0, 4096, (3, TOKENS), device="cuda", generator=gen)
                q, k = _qk(gen)
                got_q, got_k = wrapper(xyz, q.clone(), k.clone())
                want_q, want_k = reference.forward_native(
                    xyz, q.float().flatten(1), k.float().flatten(1)
                )
                torch.testing.assert_close(
                    got_q.float(), want_q.view_as(q), atol=4e-3, rtol=0
                )
                torch.testing.assert_close(
                    got_k.float(), want_k.view_as(k), atol=4e-3, rtol=0
                )


if __name__ == "__main__":
    unittest.main()
