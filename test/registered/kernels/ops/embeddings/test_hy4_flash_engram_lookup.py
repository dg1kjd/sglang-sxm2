"""Hunyuan-4-Flash Engram lookup: an FP8 (E4M3) table gives the same rows as
the same table stored in bf16, on every GPU (below sm_89 the kernel decodes the
E4M3 bytes itself)."""

import pytest
import torch

from sglang.kernels.ops.embeddings.hy4_flash_engram import hy4_flash_engram_lookup
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")

_DIM = 128
_HEADS = 4


def _e4m3_table(rows: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(0)
    codes = torch.randint(0, 256, (rows, _DIM), dtype=torch.uint8, generator=gen)
    # Every code, subnormals and both signs included, except the NaN codes.
    every = torch.tensor([c for c in range(256) if c & 0x7F != 0x7F], dtype=torch.uint8)
    codes.view(-1)[: every.numel()] = every
    codes[(codes & 0x7F) == 0x7F] = 0x01
    return codes.view(torch.float8_e4m3fn).cuda()


@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("with_scale", [False, True])
def test_fp8_table_matches_bf16_table(with_scale, out_dtype):
    rows, tokens = 64, 37
    table = _e4m3_table(rows)
    gen = torch.Generator().manual_seed(1)
    scale = (torch.rand(rows, generator=gen) + 0.5).cuda() if with_scale else None
    gamma = (torch.rand(_HEADS, _DIM, generator=gen) + 0.5).to(torch.bfloat16).cuda()
    vocab_start, head_start = 5, 2
    hashes = torch.randint(
        vocab_start, vocab_start + rows, (tokens, head_start + _HEADS + 1), generator=gen
    ).cuda()
    hashes[3, head_start + 1] = -1

    got = hy4_flash_engram_lookup(
        hashes, head_start, _HEADS, vocab_start, table, scale, gamma, out_dtype
    )
    expected = hy4_flash_engram_lookup(
        hashes,
        head_start,
        _HEADS,
        vocab_start,
        table.to(torch.bfloat16),
        scale,
        gamma,
        out_dtype,
    )
    assert got.dtype == out_dtype

    assert torch.isfinite(got.float()).all()
    assert torch.equal(got, expected)
    assert not got[3, _DIM : 2 * _DIM].any()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
