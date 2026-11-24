"""Hunyuan-4-Flash IndexPool indexer on SM70 (V100).

Same contract as ``hy4_flash_indexpool`` with fp16 in place of FP8: a completed
group of 4 tokens stores its pooled key as 128 fp16 values (256 bytes) at the
key bytes of the group's first token slot in the DSA index-key cache (the
group's first two token rows of the page); the scale bytes are unused. Queries
are fp16 after rope; scores are ``sum_h relu(q_h . k) * w_h`` with fp32
accumulation. Selection is exact: per row the ``k`` highest scores among the
groups before the row's own group, ties to the lower group index, emitted in
group order as token slots, then the row's own group up to the row, then -1.
Raw per-token keys are fp16.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

import torch

from sglang.kernels.arch import current_arch
from sglang.kernels.jit.utils import cache_once, empty_sentinel, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module

INDEX_DIM = 128
NUM_HEADS = 16
RATIO = 4
HIST_BINS = 2048
TILE_GROUPS = 64


@cache_once
def _module() -> Module:
    cls = "hy4_indexpool_sm70::Hy4IndexPoolSm70"
    return load_jit(
        "hy4_indexpool_sm70",
        cuda_files=["attention/hy4_indexpool_sm70.cuh"],
        cuda_wrappers=[
            (name, f"{cls}::{name}")
            for name in (
                "pool_write",
                "store_raw",
                "decode_update",
                "query",
                "logits",
                "select",
            )
        ],
    )


def _mrope(positions: torch.Tensor, mrope_section: Optional[Sequence[int]]):
    if mrope_section is None:
        return positions, False, 0, 0
    if positions.ndim == 1:
        positions = positions[:, None].expand(-1, 3)
    return positions, True, int(mrope_section[1]), int(mrope_section[2])


def _packed(x: torch.Tensor) -> torch.Tensor:
    # Row-major strides, size-1 dims included (`.contiguous()` keeps any stride
    # on a size-1 dim); the kernels index these tensors without strides.
    if x.stride() == torch.empty(x.shape, device="meta").stride():
        return x
    return x.clone(memory_format=torch.contiguous_format)


def indexpool_store_raw_sm70(key, norm_weight, eps, out_loc, raw_keys):
    """RMSNorm of the raw key projection ``key`` [R, >=128] (fp16/bf16, unit
    column stride) with the fp32 ``norm_weight``, stored fp16 at
    ``raw_keys[out_loc]`` (slots clamped into range)."""
    _module().store_raw(key, norm_weight, float(eps), _packed(out_loc), raw_keys)


def indexpool_pool_write_sm70(
    raw_keys,
    member_slots,
    first_positions,
    write_slots,
    wv_t,
    wg_t,
    ape,
    key_norm,
    eps,
    inv_freq,
    index_buf,
    page_size,
    mrope_section=None,
):
    """Pool each group of raw keys and store its fp16 key at ``write_slots``
    (< 0: skip). ``member_slots`` [G, 4] int64 (clamped into range);
    ``first_positions`` [G], or [G, 3] with MRoPE."""
    pos, mrope, s1, s2 = _mrope(first_positions, mrope_section)
    _module().pool_write(
        raw_keys,
        _packed(member_slots),
        pos,
        _packed(write_slots),
        wv_t,
        wg_t,
        ape,
        key_norm,
        float(eps),
        inv_freq,
        index_buf,
        page_size,
        mrope,
        s1,
        s2,
    )


def indexpool_decode_update_sm70(
    raw_key_proj,
    key_norm_weight,
    key_eps,
    out_loc,
    seq_lens,
    pages,
    raw_keys,
    wv_t,
    wg_t,
    ape,
    key_norm,
    eps,
    inv_freq,
    index_buf,
    page_size,
    rope_xyz=None,
    mrope_section=None,
):
    """Decode step: RMSNorm + store of each row's raw key (``key_norm_weight``
    fp32), and the pooled fp16 key of every group this step completes.
    ``seq_lens`` [R] int32 including this token; ``pages`` [R, NP] int32."""
    mrope = mrope_section is not None
    _module().decode_update(
        raw_key_proj,
        key_norm_weight,
        float(key_eps),
        _packed(out_loc),
        _packed(seq_lens),
        pages,
        raw_keys,
        wv_t,
        wg_t,
        ape,
        key_norm,
        float(eps),
        inv_freq,
        index_buf,
        page_size,
        _packed(rope_xyz) if mrope else empty_sentinel(raw_keys.device, torch.int32),
        mrope,
        int(mrope_section[1]) if mrope else 0,
        int(mrope_section[2]) if mrope else 0,
    )


def indexpool_query_sm70(
    query,
    positions,
    head_weights,
    inv_freq,
    num_heads,
    mrope_section=None,
    with_hist=False,
):
    """Rope of the indexer queries to fp16 and the folded head weights
    ``hw * H^-0.5 * 128^-0.5``. query: [R, H*128] fp16/bf16 (unit column
    stride); head_weights: [R, H]. With ``with_hist`` also returns a zeroed
    [R, 2048] int32 score histogram for ``indexpool_select_sm70``."""
    rows = query.shape[0]
    device = query.device
    q_out = torch.empty(
        (rows, num_heads, INDEX_DIM), dtype=torch.float16, device=device
    )
    w_out = torch.empty((rows, num_heads), dtype=torch.float32, device=device)
    hist = (
        torch.empty((rows, HIST_BINS), dtype=torch.int32, device=device)
        if with_hist
        else None
    )
    pos, mrope, s1, s2 = _mrope(positions, mrope_section)
    _module().query(
        query,
        pos,
        head_weights,
        inv_freq,
        q_out,
        w_out,
        hist if hist is not None else empty_sentinel(device, torch.int32),
        float(num_heads**-0.5 * INDEX_DIM**-0.5),
        mrope,
        s1,
        s2,
    )
    return q_out, w_out, hist


def _blocks_x(rows: int, rows_per_block: int, num_groups: int) -> int:
    tiles = max(1, -(-num_groups // TILE_GROUPS))
    row_blocks = -(-rows // rows_per_block)
    # Resident blocks: two per SM with one row per block, one with up to eight.
    slots = (2 if rows_per_block == 1 else 1) * current_arch().num_sms
    # Few row blocks (decode, verify): one wave, each block looping over its
    # tiles. Many (prefill): about sixteen waves, so the last one is short.
    target = slots if row_blocks <= slots else 16 * slots
    return max(1, min(tiles, -(-target // row_blocks), 65535))


def indexpool_logits_sm70(
    q,
    weights,
    pages,
    row_to_batch,
    positions,
    index_buf,
    page_size,
    num_groups,
    rows_per_block=8,
    hist=None,
    out=None,
):
    """Scores [R, num_groups] fp32 of each row against the groups before its own
    group (other columns are left unwritten). Rows ``row_to_batch`` equal and
    adjacent share key reads; ``rows_per_block`` is 1..8.
    With ``hist`` (zeroed [R, 2048] int32, e.g. from ``indexpool_query_sm70``)
    the score histogram of each row is added to it."""
    rows = q.shape[0]
    if out is None:
        out = torch.empty((rows, num_groups), dtype=torch.float32, device=q.device)
    _module().logits(
        q,
        weights,
        pages,
        _packed(row_to_batch),
        _packed(positions),
        index_buf,
        page_size,
        out,
        hist if hist is not None else empty_sentinel(q.device, torch.int32),
        rows_per_block,
        _blocks_x(rows, rows_per_block, num_groups),
    )
    return out


def indexpool_topk_expand_sm70(
    logits, positions, pages, row_to_batch, ragged_offsets, topk, k, hist=None, out=None
):
    """Exact top-``k`` groups per row from ``logits`` (fp32, rows 16-byte
    aligned), expanded to [R, topk] int32 token indices."""
    rows = logits.shape[0]
    device = logits.device
    if out is None:
        out = torch.empty((rows, topk), dtype=torch.int32, device=device)
    _module().select(
        logits,
        hist if hist is not None else empty_sentinel(device, torch.int32),
        _packed(positions),
        pages,
        _packed(row_to_batch),
        _packed(ragged_offsets)
        if ragged_offsets is not None
        else empty_sentinel(device, torch.int64),
        out,
        k,
    )
    return out


def indexpool_select_sm70(
    q,
    weights,
    positions,
    pages,
    row_to_batch,
    ragged_offsets,
    index_buf,
    page_size,
    num_groups,
    topk,
    k,
    rows_per_block,
    hist=None,
    logits_budget=512 << 20,
):
    """Scores and selection for all rows, in row chunks whose fp32 scores fit in
    ``logits_budget`` bytes. ``hist`` is a zeroed [R, 2048] int32 buffer (from
    ``indexpool_query_sm70``); without it each chunk gets its own. Returns
    [R, topk] int32 token indices."""
    rows = q.shape[0]
    device = q.device
    num_groups = max(TILE_GROUPS, -(-num_groups // TILE_GROUPS) * TILE_GROUPS)
    out = torch.empty((rows, topk), dtype=torch.int32, device=device)
    if rows == 0:
        return out
    chunk = max(8, (logits_budget // (4 * num_groups)) // 8 * 8)
    for begin in range(0, rows, chunk):
        end = min(rows, begin + chunk)
        chunk_hist = (
            hist[begin:end]
            if hist is not None
            else torch.zeros((end - begin, HIST_BINS), dtype=torch.int32, device=device)
        )
        logits = indexpool_logits_sm70(
            q[begin:end],
            weights[begin:end],
            pages,
            row_to_batch[begin:end],
            positions[begin:end],
            index_buf,
            page_size,
            num_groups,
            rows_per_block=rows_per_block,
            hist=chunk_hist,
        )
        indexpool_topk_expand_sm70(
            logits,
            positions[begin:end],
            pages,
            row_to_batch[begin:end],
            ragged_offsets[begin:end] if ragged_offsets is not None else None,
            topk,
            k,
            hist=chunk_hist,
            out=out[begin:end],
        )
    return out
