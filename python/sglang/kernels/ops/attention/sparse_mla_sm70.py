"""SM70 sparse MLA over the 512-d latent.

Fp16 query and KV, fp32 softmax. The bf16 Triton and TileLang DSA kernels
do not run on Volta. The indexer has already chosen the keys; these kernels
only attend to those rows.

Without a rope tail, sinks or per-row lengths, the NoPE kernel runs. With any
of them, a tensor-core kernel scores the latent plus a 64-d rope tail, adds
per-head sinks to the softmax denominator and stops each row at its length.
"""

from __future__ import annotations

from typing import Optional

import torch

from sglang.kernels.jit.utils import cache_once, empty_sentinel, load_jit


@cache_once
def _sparse_mla_sm70_module():
    return load_jit(
        "sparse_mla_sm70",
        cuda_files=["attention/sparse_mla_sm70.cuh"],
        cuda_wrappers=[("run", "SparseMlaSm70Kernel::run")],
        extra_cuda_cflags=["-O3"],
    )


@cache_once
def _sparse_mla_sm70_rope_module():
    return load_jit(
        "sparse_mla_sm70_rope",
        cuda_files=["attention/sparse_mla_sm70_rope.cuh"],
        cuda_wrappers=[("run", "SparseMlaSm70RopeKernel::run")],
        extra_cuda_cflags=["-O3"],
    )


def sparse_mla_sm70(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    *,
    q_rope: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Sparse attention over indexer-selected latent rows.

    ``q`` is ``[S, H, 512]`` fp16. ``kv`` is ``[N, D]`` or ``[N, 1, D]`` fp16
    with ``D = 512`` plus the rope tail, and ``indices`` is ``[S, topk]`` (or
    ``[S, 1, topk]``). Negative indices are ignored. Returns
    ``[1, S, H, 512]`` fp16.

    ``q_rope`` (``[S, H, 64]``) scores the last 64 columns of each kv row;
    the output still averages only the first 512. ``attn_sink`` (``[H]``
    fp32) adds ``exp(attn_sink[h])`` to head h's softmax denominator.
    ``topk_length`` (``[S]``) limits row s to its first
    ``min(topk_length[s], topk)`` slots. A row with no usable slot is zero.
    """
    if q_rope is not None or attn_sink is not None or topk_length is not None:
        return _sparse_mla_sm70_rope(
            q, q_rope, kv, indices, sm_scale, attn_sink, topk_length
        )
    if q.device.type != "cuda" or torch.cuda.get_device_capability(q.device) != (7, 0):
        raise ValueError("sparse_mla_sm70 requires an SM70 CUDA device")
    if q.dtype != torch.float16 or kv.dtype != torch.float16:
        raise ValueError(
            f"SM70 sparse MLA requires fp16 q and kv, got {q.dtype} and {kv.dtype}"
        )
    if q.ndim != 3 or q.shape[-1] != 512:
        raise ValueError(f"q must be [S, H, 512], got {tuple(q.shape)}")
    if indices.ndim == 3:
        if indices.shape[1] != 1:
            raise ValueError(
                f"indices must be [S, topk] or [S, 1, topk], got {tuple(indices.shape)}"
            )
        indices = indices[:, 0, :]
    if indices.ndim != 2 or indices.shape[0] != q.shape[0]:
        raise ValueError(
            f"indices must be [S, topk] with S={q.shape[0]}, got {tuple(indices.shape)}"
        )
    if kv.ndim == 3:
        if kv.shape[1] != 1 or kv.shape[-1] != 512:
            raise ValueError(f"kv must be [N, 1, 512], got {tuple(kv.shape)}")
    elif kv.ndim != 2 or kv.shape[-1] != 512:
        raise ValueError(f"kv must be [N, 512] or [N, 1, 512], got {tuple(kv.shape)}")

    # The kernel takes q's token and head strides; it reads each row as fp16 pairs.
    if q.stride(-1) != 1 or q.stride(0) % 2 or q.stride(1) % 2 or q.data_ptr() % 4:
        q = q.contiguous()
    if indices.dtype != torch.int32:
        indices = indices.to(torch.int32)
    indices = indices.contiguous()
    if indices.shape[-1] == 0:
        return torch.zeros((1, *q.shape), dtype=torch.float16, device=q.device)
    out = torch.empty(q.shape, dtype=torch.float16, device=q.device)
    splits = _num_splits(q.shape[0], q.shape[1], indices.shape[-1])
    part_shape = (q.shape[0], q.shape[1], splits) if splits > 1 else (0,)
    part_acc = torch.empty((*part_shape, 512) if splits > 1 else (0,), dtype=torch.float32, device=q.device)
    part_ml = torch.empty((*part_shape, 2) if splits > 1 else (0,), dtype=torch.float32, device=q.device)
    _sparse_mla_sm70_module().run(q, kv, indices, out, part_acc, part_ml, float(sm_scale), splits)
    return out.unsqueeze(0)


# Two blocks per V100 SM (80 SMs); fewer (token, head-tile) blocks than that
# split the topk slots, with at least one 32-row chunk per split.
_TARGET_BLOCKS = 160
_ROWS_PER_CHUNK = 32


def _num_splits(tokens: int, heads: int, topk: int) -> int:
    # Decode and MTP verify (up to four tokens) share the one-token split, so a
    # verify row merges its softmax partials in the same order as at decode.
    blocks = (1 if tokens <= 4 else tokens) * ((heads + 7) // 8)
    if blocks >= _TARGET_BLOCKS // 2 or topk <= _ROWS_PER_CHUNK:
        return 1
    max_splits = (topk + _ROWS_PER_CHUNK - 1) // _ROWS_PER_CHUNK
    return max(1, min(max_splits, (_TARGET_BLOCKS + blocks - 1) // blocks))


def _rows_16b_aligned(x: torch.Tensor) -> bool:
    return (
        x.stride(-1) == 1
        and x.stride(0) % 8 == 0
        and x.stride(1) % 8 == 0
        and x.data_ptr() % 16 == 0
    )


def _sparse_mla_sm70_rope(
    q: torch.Tensor,
    q_rope: Optional[torch.Tensor],
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    attn_sink: Optional[torch.Tensor],
    topk_length: Optional[torch.Tensor],
) -> torch.Tensor:
    if q.device.type != "cuda" or torch.cuda.get_device_capability(q.device) != (7, 0):
        raise ValueError("sparse_mla_sm70 requires an SM70 CUDA device")
    if q.dtype != torch.float16 or kv.dtype != torch.float16:
        raise ValueError(
            f"SM70 sparse MLA requires fp16 q and kv, got {q.dtype} and {kv.dtype}"
        )
    if q.ndim != 3 or q.shape[-1] != 512:
        raise ValueError(f"q must be [S, H, 512], got {tuple(q.shape)}")
    tail = 0
    if q_rope is not None:
        if q_rope.dtype != torch.float16:
            raise ValueError(f"q_rope must be fp16, got {q_rope.dtype}")
        if q_rope.shape != (*q.shape[:2], 64):
            raise ValueError(
                f"q_rope must be [S, H, 64] with S, H = {tuple(q.shape[:2])}, "
                f"got {tuple(q_rope.shape)}"
            )
        tail = 64
    kv_dim = 512 + tail
    if kv.ndim == 3:
        if kv.shape[1] != 1 or kv.shape[-1] != kv_dim:
            raise ValueError(f"kv must be [N, 1, {kv_dim}], got {tuple(kv.shape)}")
    elif kv.ndim != 2 or kv.shape[-1] != kv_dim:
        raise ValueError(
            f"kv must be [N, {kv_dim}] or [N, 1, {kv_dim}], got {tuple(kv.shape)}"
        )
    if kv.stride(-1) != 1 or kv.stride(0) % 8 or kv.data_ptr() % 16:
        raise ValueError("kv rows must be contiguous and 16-byte aligned")
    if indices.ndim == 3:
        if indices.shape[1] != 1:
            raise ValueError(
                f"indices must be [S, topk] or [S, 1, topk], got {tuple(indices.shape)}"
            )
        indices = indices[:, 0, :]
    if indices.ndim != 2 or indices.shape[0] != q.shape[0]:
        raise ValueError(
            f"indices must be [S, topk] with S={q.shape[0]}, got {tuple(indices.shape)}"
        )

    num_tokens, num_heads = q.shape[:2]
    # The kernel reads q rows in 16-byte vectors at the given token and head
    # strides; a fresh contiguous copy is aligned.
    if not _rows_16b_aligned(q):
        q = q.clone(memory_format=torch.contiguous_format)
    if q_rope is not None and not _rows_16b_aligned(q_rope):
        q_rope = q_rope.clone(memory_format=torch.contiguous_format)
    indices = indices.to(torch.int32).contiguous()
    if topk_length is None:
        lengths = empty_sentinel(q.device, torch.int32)
    else:
        if topk_length.shape != (num_tokens,):
            raise ValueError(
                f"topk_length must be [{num_tokens}], got {tuple(topk_length.shape)}"
            )
        lengths = topk_length.to(torch.int32).contiguous()
    if attn_sink is None:
        sink = empty_sentinel(q.device, torch.float32)
    else:
        if attn_sink.shape != (num_heads,):
            raise ValueError(
                f"attn_sink must be [{num_heads}], got {tuple(attn_sink.shape)}"
            )
        sink = attn_sink.to(torch.float32).contiguous()

    out = torch.empty(q.shape, dtype=torch.float16, device=q.device)
    width = indices.shape[-1]
    if width == 0 or num_tokens == 0:
        out.zero_()
        return out.unsqueeze(0)
    splits = _rope_num_splits(num_tokens, num_heads, width, q.device)
    if splits > 1:
        part_acc = torch.empty(
            (num_tokens, num_heads, splits, 512), dtype=torch.float32, device=q.device
        )
        part_ml = torch.empty(
            (num_tokens, num_heads, splits, 2), dtype=torch.float32, device=q.device
        )
    else:
        part_acc = part_ml = empty_sentinel(q.device, torch.float32)
    _sparse_mla_sm70_rope_module().run(
        q,
        q_rope if q_rope is not None else empty_sentinel(q.device, torch.float16),
        kv,
        indices,
        lengths,
        sink,
        out,
        part_acc,
        part_ml,
        float(sm_scale),
        splits,
    )
    return out.unsqueeze(0)


# The rope kernel takes 32 slots per tile and up to 8 heads per block; at
# most two blocks share an SM. Fewer (token, head-tile) blocks than two per SM
# split the slots, keeping at least _ROPE_MIN_TILES_PER_SPLIT tiles per split.
_ROPE_KEYS_PER_TILE = 32
_ROPE_HEADS_PER_BLOCK = 8
_ROPE_MIN_TILES_PER_SPLIT = 1


@cache_once
def _sm_count(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def _rope_num_splits(tokens: int, heads: int, width: int, device: torch.device) -> int:
    target = 2 * _sm_count(
        device.index if device.index is not None else torch.cuda.current_device()
    )
    blocks = tokens * -(-heads // _ROPE_HEADS_PER_BLOCK)
    tiles = -(-width // _ROPE_KEYS_PER_TILE)
    if blocks >= target:
        return 1
    max_splits = max(1, tiles // _ROPE_MIN_TILES_PER_SPLIT)
    return max(1, min(max_splits, -(-target // blocks)))
