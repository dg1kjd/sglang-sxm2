"""Batch-invariant fp16 / FP8-block GEMM on SM70 tensor cores.

y = x @ W^T (+ bias) with fp32 accumulation, W fp16 or FP8 e4m3 with 128 x 128
block scales. The summation order of every output element depends only on K
(see the header of ``sm70_fixed_order_gemm.cuh``), so a row of y is bitwise the
same at any M, through the decode and the prefill tiling alike. The tiling and
warp counts chosen here change the speed only.

The fp16 callers take this kernel for M <= MAX_ROWS rows only (``supported``)
and keep cuBLAS or another SM70 kernel above it, so their
rows are batch invariant for M <= MAX_ROWS only. The block-FP8 layer path takes
it at every M.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# Tiling choice (speed only). Decode time grows with N * ceil(M / 8) (each CTA
# owns 8 x 8 outputs over all of K), prefill time with the number of waves of
# 128 x 128 tiles (one CTA per SM). Measured on V100 per unit of K: decode
# ~1.75e-6 us * N * ceil(M / 8) for fp16 W and ~2.2e-6 us for FP8, prefill
# ~0.044 us per wave. Decode wins while N * ceil(M / 8) is below this many per wave.
_DECODE_ROWS_PER_WAVE = {False: 25000, True: 20000}

# The fp16 callers' cut-over (``supported``). Up to 8 rows (one decode m-tile)
# the kernel is about as fast as cuBLAS or faster on the Hunyuan-4-Flash TP8
# shapes, and it covers the decode step (M = 1), the DFly verify step and the
# drafter's layers (M = 6), and the drafter's fusion, correction and greedy
# head (M = 1). Above it cuBLAS is faster (the decode tiling re-reads W per 8
# rows and the prefill tiling has no split-K).
MAX_ROWS = 8

_EXPORTS = {
    # (kernel, fp8 weight, out dtype) -> export name
    ("decode", False, torch.float16): "decode_w16_o16",
    ("decode", False, torch.float32): "decode_w16_o32",
    ("decode", True, torch.float16): "decode_w8_o16",
    ("prefill", False, torch.float16): "prefill_w16_o16",
    ("prefill", False, torch.float32): "prefill_w16_o32",
    ("prefill", True, torch.float16): "prefill_w8_o16",
}

_CPP_OUT = {torch.float16: "fp16_t", torch.float32: "fp32_t"}


@cache_once
def _module() -> Module:
    wrappers = [
        (name, f"sm70_fixed_order_gemm::{kind}<{'true' if fp8 else 'false'}, {_CPP_OUT[out]}>")
        for (kind, fp8, out), name in _EXPORTS.items()
    ]
    return load_jit(
        "sm70_fixed_order_gemm",
        cuda_files=["gemm/sm70_fixed_order_gemm.cuh"],
        cuda_wrappers=wrappers,
    )


@cache_once
def _is_sm70(device_index) -> bool:
    return torch.cuda.get_device_capability(device_index) == (7, 0)


@cache_once
def _num_sms(device_index) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def rows_aligned(t: torch.Tensor) -> bool:
    """Unit last stride and 16-byte aligned start and row/batch strides."""
    return (
        t.stride(-1) == 1
        and t.data_ptr() % 16 == 0
        and all(s * t.element_size() % 16 == 0 for s, n in zip(t.stride()[:-1], t.shape[:-1]) if n > 1)
    )


def out_supported(out: torch.Tensor, shape, dtype: torch.dtype) -> bool:
    return (
        out.shape == shape
        and out.dtype == dtype
        and out.device.type == "cuda"
        and out.stride(-1) == 1
        and out.data_ptr() % (2 * out.element_size()) == 0
        and all(s % 2 == 0 for s, n in zip(out.stride()[:-1], out.shape[:-1]) if n > 1)
    )


def supported(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> bool:
    """Whether a caller should take the kernel: x [M, K] or [B, M, K] fp16 with
    M <= MAX_ROWS; weight [N, K] or [B, N, K], fp16 or float8_e4m3fn (2D only);
    N % 8 == 0, K % 32 == 0 (fp16) or K % 128 == 0 (FP8); bias [N] fp16 (2D
    only); 16-byte aligned rows; SM70 only. ``linear``, ``fp8_linear`` and
    ``bmm`` themselves take any M."""
    fp8 = weight.dtype == torch.float8_e4m3fn
    return (
        x.is_cuda
        and x.dtype == torch.float16
        and weight.dtype in (torch.float16, torch.float8_e4m3fn)
        and weight.device == x.device
        and x.ndim == weight.ndim
        and x.ndim in (2, 3)
        and x.shape[-2] <= MAX_ROWS
        and (x.ndim == 2 or x.shape[0] == weight.shape[0])
        and not (fp8 and x.ndim == 3)
        and x.shape[-1] == weight.shape[-1]
        and weight.shape[-2] % 8 == 0
        and weight.shape[-1] % (128 if fp8 else 32) == 0
        and (
            bias is None
            or (
                x.ndim == 2
                and bias.dtype == torch.float16
                and bias.shape == (weight.shape[0],)
                and bias.stride(0) == 1
                and bias.data_ptr() % 4 == 0
            )
        )
        and rows_aligned(x)
        and rows_aligned(weight)
        and _is_sm70(x.device.index)
    )


def use_decode(batch: int, m: int, n: int, fp8: bool, device_index) -> bool:
    """True when the decode tiling is the faster one for this shape. Speed only."""
    waves = -(-batch * -(-n // 128) * -(-m // 128) // _num_sms(device_index))
    return batch * n * -(-m // 8) < _DECODE_ROWS_PER_WAVE[fp8] * waves


# The decode tiling keeps every chain's partial sums of an 8 x 8 tile in shared
# memory (``kMaxSmemBytes`` in the .cuh).
_DECODE_MAX_SMEM_BYTES = 96 * 1024


def decode_fits(m: int, k: int, fp8: bool) -> bool:
    """Whether the decode tiling's shared memory holds K. Above it the prefill
    tiling, which takes any K and sums in the same order, runs instead."""
    return k // (64 if fp8 else 32) * min(m, 8) * 8 * 4 <= _DECODE_MAX_SMEM_BYTES


def decode_warps(k: int, fp8: bool) -> int:
    """Warps per decode CTA: one per task of 4 chains, at most 8. Speed only."""
    return max(1, min(8, (k // (64 if fp8 else 32) + 3) // 4))


def _run(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    out: torch.Tensor,
    force: Optional[str],
) -> None:
    if x.shape[-2] == 0 or weight.shape[-2] == 0 or x.ndim == 3 and x.shape[0] == 0:
        return
    x3 = x if x.ndim == 3 else x.unsqueeze(0)
    w3 = weight if weight.ndim == 3 else weight.unsqueeze(0)
    o3 = out if out.ndim == 3 else out.unsqueeze(0)
    fp8 = weight.dtype == torch.float8_e4m3fn
    decode = decode_fits(x3.shape[1], w3.shape[2], fp8) and use_decode(
        x3.shape[0], x3.shape[1], w3.shape[1], fp8, x.device.index
    )
    kind = force or ("decode" if decode else "prefill")
    fn = getattr(_module(), _EXPORTS[(kind, fp8, out.dtype)])
    if kind == "decode":
        fn(o3, x3, w3, scale, bias, decode_warps(w3.shape[2], fp8))
    else:
        fn(o3, x3, w3, scale, bias)


def linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    *,
    out: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.float16,
    force: Optional[str] = None,
) -> torch.Tensor:
    """x [M, K] @ weight[N, K]^T (+ bias [N] fp16) -> [M, N] in out_dtype (fp16 or fp32).
    A given ``out`` must have that shape and dtype, unit last stride, even row
    stride and a start aligned to two elements."""
    shape = (x.shape[0], weight.shape[0])
    if out is None:
        out = torch.empty(shape, dtype=out_dtype, device=x.device)
    elif not out_supported(out, shape, out_dtype):
        raise ValueError("sm70_fixed_order_gemm.linear: unsupported out tensor")
    _run(x, weight, None, bias, out, force)
    return out


def fp8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    *,
    out: Optional[torch.Tensor] = None,
    force: Optional[str] = None,
) -> torch.Tensor:
    """x [M, K] fp16 @ dequant(weight [N, K] e4m3, scale [ceil(N/128), K/128] fp32)^T -> [M, N] fp16."""
    shape = (x.shape[0], weight.shape[0])
    if out is None:
        out = torch.empty(shape, dtype=torch.float16, device=x.device)
    elif not out_supported(out, shape, torch.float16):
        raise ValueError("sm70_fixed_order_gemm.fp8_linear: unsupported out tensor")
    _run(x, weight, scale, bias, out, force)
    return out


def bmm(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    force: Optional[str] = None,
) -> torch.Tensor:
    """x [B, M, K] @ weight[B, N, K]^T -> [B, M, N] fp16; strides free except the last."""
    shape = (x.shape[0], x.shape[1], weight.shape[1])
    if out is None:
        out = torch.empty(shape, dtype=torch.float16, device=x.device)
    elif not out_supported(out, shape, torch.float16):
        raise ValueError("sm70_fixed_order_gemm.bmm: unsupported out tensor")
    _run(x, weight, None, None, out, force)
    return out
