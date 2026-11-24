"""Triton kernels for the Hunyuan-4-Flash Engram block.

The block per token: look up one table row per local hash head, RMSNorm each
row (``lookup_gamma``), project the concatenation to 4 keys + 1 value (one GEMM
and all-reduce, outside these kernels), then gate each iHC stream by the
RMSNorm'd key/query dot product and add the normalized value to it.

Both kernels follow the reference's rounding points (bf16 results of
``F.rms_norm`` and of the bf16 elementwise ops), so they differ from the eager
module only in reduction order.
"""

import torch
import triton
import triton.language as tl

_FP32_EPS = torch.finfo(torch.float32).eps
_BF16_EPS = torch.finfo(torch.bfloat16).eps


def _needs_e4m3_byte_decode(table: torch.Tensor) -> bool:
    # Triton has no E4M3 (fp8e4nv) type below sm_89; read the bytes instead.
    return (
        table.dtype == torch.float8_e4m3fn
        and torch.cuda.get_device_capability(table.device) < (8, 9)
    )


@triton.jit
def _e4m3_bytes_to_f32(b):
    """E4M3FN bit pattern (uint8) to fp32, exact; NaN codes are not handled."""
    b = b.to(tl.int32)
    exp = (b >> 3) & 0xF
    mant = (b & 0x7).to(tl.float32)
    normal = (8.0 + mant) * tl.exp2((exp - 10).to(tl.float32))
    subnormal = mant * 0.001953125  # 2^-9
    value = tl.where(exp == 0, subnormal, normal)
    return tl.where((b & 0x80) != 0, -value, value)


@triton.jit
def _engram_lookup_kernel(
    hash_ptr,  # [T, n_hash] int
    hash_stride,
    table_ptr,  # [rows, D] bf16 or fp8
    scale_ptr,  # [rows] per-row scale (HAS_SCALE)
    gamma_ptr,  # [heads, D] bf16
    out_ptr,  # [T, heads * D] out_dtype
    head_start,
    vocab_start,
    rows,
    eps,
    HEADS: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    E4M3_BYTES: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1)
    d = tl.arange(0, BLOCK_D)
    mask = d < D
    idx = tl.load(hash_ptr + t * hash_stride + head_start + h).to(tl.int64)
    row = tl.minimum(tl.maximum(idx - vocab_start, 0), rows - 1)
    if E4M3_BYTES:
        x = _e4m3_bytes_to_f32(tl.load(table_ptr + row * D + d, mask=mask, other=0))
    else:
        x = tl.load(table_ptr + row * D + d, mask=mask, other=0.0).to(tl.float32)
    if HAS_SCALE:
        x = x * tl.load(scale_ptr + row).to(tl.float32)
    x = tl.where(idx < 0, 0.0, x)
    gamma = tl.load(gamma_ptr + h * D + d, mask=mask, other=0.0).to(tl.float32)
    y = x * tl.rsqrt(tl.sum(x * x, axis=0) / D + eps) * gamma
    tl.store(
        out_ptr + t * HEADS * D + h * D + d,
        y.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


def hy4_flash_engram_lookup(
    hashes: torch.Tensor,
    head_start: int,
    heads: int,
    vocab_start: int,
    table: torch.Tensor,
    table_scale: torch.Tensor | None,
    gamma: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """[T, heads * D] out_dtype: per-head RMSNorm'd table rows of this rank's
    hash heads; a negative hash gives a zero row."""
    tokens = hashes.shape[0]
    dim = table.shape[1]
    out = torch.empty((tokens, heads * dim), dtype=out_dtype, device=hashes.device)
    if tokens == 0 or heads == 0:
        return out
    assert hashes.stride(1) == 1 and table.is_contiguous() and gamma.is_contiguous()
    has_scale = table_scale is not None and table_scale.numel() > 0
    e4m3_bytes = _needs_e4m3_byte_decode(table)
    if e4m3_bytes:
        table = table.view(torch.uint8)
    _engram_lookup_kernel[(tokens, heads)](
        hashes,
        hashes.stride(0),
        table,
        table_scale if has_scale else table,
        gamma,
        out,
        head_start,
        vocab_start,
        table.shape[0],
        _FP32_EPS,
        HEADS=heads,
        D=dim,
        BLOCK_D=triton.next_power_of_2(dim),
        HAS_SCALE=has_scale,
        E4M3_BYTES=e4m3_bytes,
        num_warps=1,
    )
    return out


@triton.jit
def _engram_combine_kernel(
    kv_ptr,  # [T, (HC + 1) * H] bf16: HC keys then the value
    hidden_ptr,  # [T, HC, H] bf16
    out_ptr,  # [T, HC, H] bf16
    norm_key_ptr,  # [HC, H] bf16
    norm_query_ptr,  # [HC, H] bf16
    pre_gamma_ptr,  # [H] bf16
    hc_scale_ptr,  # [HC, H] bf16
    hash_ptr,  # [T, n_hash] int: a token whose hashes are all negative is skipped
    hash_stride,
    n_hash,
    H: tl.constexpr,
    HC: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_HASH: tl.constexpr,
    eps32: tl.constexpr,
    eps_bf16: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    d = tl.arange(0, BLOCK_H)
    mask = d < H
    kv_row = kv_ptr + t * (HC + 1) * H

    hs = tl.arange(0, BLOCK_HASH)
    hv = tl.load(hash_ptr + t * hash_stride + hs, mask=hs < n_hash, other=-1)
    live = tl.sum((hv >= 0).to(tl.int32), axis=0) > 0

    value = tl.load(kv_row + HC * H + d, mask=mask, other=0.0).to(tl.float32)
    pre_gamma = tl.load(pre_gamma_ptr + d, mask=mask, other=0.0).to(tl.float32)
    signal = pre_gamma * value * tl.rsqrt(tl.sum(value * value, axis=0) / H + eps32)

    for c in tl.static_range(HC):
        key = tl.load(kv_row + c * H + d, mask=mask, other=0.0).to(tl.float32)
        wk = tl.load(norm_key_ptr + c * H + d, mask=mask, other=0.0).to(tl.float32)
        key_n = key * tl.rsqrt(tl.sum(key * key, axis=0) / H + eps_bf16) * wk
        key_n = key_n.to(tl.bfloat16).to(tl.float32)
        h = tl.load(hidden_ptr + (t * HC + c) * H + d, mask=mask, other=0.0).to(
            tl.float32
        )
        wq = tl.load(norm_query_ptr + c * H + d, mask=mask, other=0.0).to(tl.float32)
        q_n = h * tl.rsqrt(tl.sum(h * h, axis=0) / H + eps_bf16) * wq
        q_n = q_n.to(tl.bfloat16).to(tl.float32)
        prod = (key_n * q_n).to(tl.bfloat16).to(tl.float32)
        s = tl.sum(prod, axis=0).to(tl.bfloat16).to(tl.float32)
        s = (s * (H**-0.5)).to(tl.bfloat16).to(tl.float32)
        mag = tl.maximum(tl.abs(s), 1e-6).to(tl.bfloat16).to(tl.float32)
        root = tl.sqrt(mag).to(tl.bfloat16).to(tl.float32)
        g = tl.where(s > 0, root, tl.where(s < 0, -root, 0.0))
        gate = tl.sigmoid(g).to(tl.bfloat16).to(tl.float32)
        scale = tl.load(hc_scale_ptr + c * H + d, mask=mask, other=0.0).to(tl.float32)
        result = ((gate + scale) * signal).to(tl.bfloat16).to(tl.float32)
        result = tl.where(live, result, 0.0)
        tl.store(
            out_ptr + (t * HC + c) * H + d,
            (h + result).to(out_ptr.dtype.element_ty),
            mask=mask,
        )


def hy4_flash_engram_combine(
    kv: torch.Tensor,
    hidden: torch.Tensor,
    hashes: torch.Tensor,
    norm_key: torch.Tensor,
    norm_query: torch.Tensor,
    pre_gamma: torch.Tensor,
    hc_scales: torch.Tensor,
) -> torch.Tensor:
    """hidden + per-stream gate x normalized value; kv: [T, (HC + 1) * H] (keys
    then value), hidden: [T, HC, H], hashes: [T, n_hash] of this layer."""
    tokens, hc, hidden_size = hidden.shape
    out = torch.empty_like(hidden)
    if tokens == 0:
        return out
    kv = kv.contiguous()
    hidden = hidden.contiguous()
    assert kv.shape == (tokens, (hc + 1) * hidden_size)
    assert hashes.stride(1) == 1
    _engram_combine_kernel[(tokens,)](
        kv,
        hidden,
        out,
        norm_key,
        norm_query,
        pre_gamma,
        hc_scales,
        hashes,
        hashes.stride(0),
        hashes.shape[1],
        H=hidden_size,
        HC=hc,
        BLOCK_H=triton.next_power_of_2(hidden_size),
        BLOCK_HASH=triton.next_power_of_2(max(hashes.shape[1], 2)),
        eps32=_FP32_EPS,
        eps_bf16=_BF16_EPS,
        num_warps=8,
    )
    return out
