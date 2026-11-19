"""SM70 split-KV grouped decode attention over fp16 KV.

One query token per request, head_dim 256. Reads the CSR ``kv_indptr`` /
``kv_indices`` of the Triton decode metadata. The chunk length is computed on
the device from the live lengths, so a call is CUDA-graph safe.
"""

from __future__ import annotations

import functools
import math

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

SUPPORTED_GROUPS = (2, 4, 8)
# kMaxChunks in sm70_grouped_decode.cuh; a launch needs batch + target chunks.
_MAX_CHUNKS = 1024
# Every request takes a chunk and the split target is at least 1.
MAX_BATCH = _MAX_CHUNKS - 1
_WARPS = 4


@cache_once
def _module(group: int, warps: int):
    args = make_cpp_args(group, warps)
    return load_jit(
        "sm70_grouped_decode",
        *args,
        cuda_files=["attention/sm70_grouped_decode.cuh"],
        cuda_wrappers=[("run", f"Sm70GroupedDecodeKernel<{args}>::run")],
        extra_cuda_cflags=["-O3"],
    )


# Partial CTAs resident per SM: 194 registers x 128 threads (ptxas -v on sm_70).
_CTAS_PER_SM = 2
# Measured on V100-SXM2, 4 q heads per kv head; re-tune with the kernel.
_MIN_CHUNK_TOKENS = 64


@functools.cache
def _sm_count(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def _work_split(batch: int, sm_count: int) -> tuple[int, int]:
    """Target chunk count and minimum chunk tokens; the grid is one wave of CTAs."""
    return max(_CTAS_PER_SM * sm_count - batch, 1), _MIN_CHUNK_TOKENS


def sm70_grouped_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    *,
    softmax_scale: float,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Decode attention for one query token per request.

    ``q`` is ``[batch, heads, 256]`` fp16. ``k_cache`` / ``v_cache`` are
    ``[pool_tokens, kv_heads, 256]`` fp16. ``kv_indptr`` is int32 with at least
    ``batch + 1`` entries into the int64 token slots ``kv_indices``.
    ``batch`` is at most ``MAX_BATCH``. Returns ``out`` (``[batch, heads, 256]`` fp16).
    """
    batch, heads, _ = q.shape
    group = heads // k_cache.shape[1]
    if group not in SUPPORTED_GROUPS or group * k_cache.shape[1] != heads:
        raise ValueError(f"unsupported heads {heads} / kv_heads {k_cache.shape[1]}")
    target, min_tokens = _work_split(batch, _sm_count(q.device.index))
    if out is None:
        out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    _module(group, _WARPS).run(
        q,
        k_cache,
        v_cache,
        kv_indptr,
        kv_indices,
        out,
        target,
        min_tokens,
        softmax_scale * k_scale * math.log2(math.e),
        v_scale,
    )
    return out
