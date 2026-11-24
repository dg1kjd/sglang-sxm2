"""Hunyuan-4-Flash routed-expert NVFP4 MoE on SM70 (V100) at TP8, batch invariant.

Coverage: fp16 routed latent x [M, 1536] for any M >= 1, top-16 int32 ids and
fp32 weights, per-rank intermediate 192, gated SiLU with the clamped SwiGLU
(``swiglu_limit``, no clamp when <= 0) and a routed scaling factor. Weights are
the SM70 Marlin NVFP4 packs, read in place: w13 int32 [E, 96, 768], w2 int32
[E, 12, 3072], fp8 e4m3 group scales [E, 96, 384] and [E, 12, 1536], fp32
global scales [E]. Ids outside [0, E) add zero. Everything else (other
shapes, bias, zero points, EP maps, the GPT-OSS SwiGLU, uncombined output)
stays on ``fused_marlin_moe``.

Batch invariance: a token's output bits do not depend on M, on which other
tokens are in the call, or on which of the two kernel sets runs it (decode for
M <= 8, prefill above). Both run the same per-element sequence of fp32
operations, fixed by the shapes (the spec is at the top of
``moe/sm70_hy4_nvfp4_moe.cuh``). The fp16 rounding points are those of
``fused_marlin_moe`` with ``swiglu_limit_func`` and ``moe_sum_reduce_triton``;
the fp32 sum order inside the dot products is this kernel's own, so outputs
differ from Marlin in the last fp16 bit now and then.

The decode workspace is one per device: two streams must not run this op at
the same time on one device, and the first decode call on a device must not
be inside CUDA graph capture (it raises).
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.arch import current_arch
from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module

logger = logging.getLogger(__name__)

HIDDEN = 1536
INTERMEDIATE = 192
TOPK = 16
MAX_DECODE_TOKENS = 8
# Tokens per prefill launch: bounds the gate_up and route-row workspaces
# (M * 16 * (384 + 1536) fp16, 126 MB at 2048).
PREFILL_CHUNK = 2048
_GATE_UP = 2 * INTERMEDIATE
_SPLIT = 4  # gate_up K splits, fixed by the reduction order
_DOWN_GROUPS = HIDDEN // 64
_ROWS = 8  # routes per HMMA block

# Decode kernel rows -> (rows, dedupe, gate_up column warps, down column
# warps). M=1 has no repeated experts to share, so it skips the dedupe ballot.
# M=5..8 (speculative verify) runs on 8. The column warps only set how many
# columns one CTA covers, never the arithmetic of an element; picked by timing
# on V100 (one layer, E=512, weights from HBM).
_DECODE_CONFIGS = {
    1: (1, False, 1, 6),
    2: (2, True, 2, 4),
    4: (4, True, 3, 4),
    8: (8, True, 3, 4),
}
# Prefill: (gate_up column warps, down column warps, route blocks of 8 per CTA).
_PREFILL_CONFIG = (1, 6, 8)

_WS: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
_LOGGED: set[str] = set()


@cache_once
def _decode_module(rows, dedupe, cw1, cw2) -> Module:
    args = make_cpp_args(rows, dedupe, cw1, cw2)
    return load_jit(
        "sm70_hy4_nvfp4_moe_decode",
        *args,
        cuda_files=["moe/sm70_hy4_nvfp4_moe.cuh"],
        cuda_wrappers=[("moe_decode", f"sm70_hy4_moe::moe_decode<{args}>")],
        # Same SwiGLU arithmetic as the SM70 glue Marlin calls, built the same way.
        extra_cuda_cflags=["--fmad=false"],
    )


@cache_once
def _prefill_module(cw1, cw2, blocks) -> Module:
    args = make_cpp_args(cw1, cw2, blocks)
    return load_jit(
        "sm70_hy4_nvfp4_moe_prefill",
        *args,
        cuda_files=["moe/sm70_hy4_nvfp4_moe.cuh"],
        cuda_wrappers=[("moe_prefill", f"sm70_hy4_moe::moe_prefill<{args}>")],
        extra_cuda_cflags=["--fmad=false"],
    )


def _rows(m: int) -> int:
    return 1 if m == 1 else 2 if m == 2 else 4 if m <= 4 else 8


def sm70_hy4_nvfp4_moe_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        arch = current_arch().cuda
    except (AssertionError, RuntimeError):
        return False
    return arch is not None and arch.is_sm(7, 0)


def sm70_hy4_nvfp4_moe_eligible(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scales: torch.Tensor,
    w2_scales: torch.Tensor,
    w13_global_scale: Optional[torch.Tensor],
    w2_global_scale: Optional[torch.Tensor],
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    swiglu_limit: Optional[float],
) -> bool:
    """True iff the tensors have the shapes and dtypes this op serves.

    The caller checks the rest of the Marlin call (weight bits, zero points,
    bias, expert map, activation, the GPT-OSS options, input/output scales).
    None of the checks depends on M beyond M >= 1, so a model either runs every
    batch here or none.
    """
    if hidden_states.dtype != torch.float16 or hidden_states.ndim != 2:
        return False
    m = int(hidden_states.shape[0])
    if m < 1 or int(hidden_states.shape[1]) != HIDDEN:
        return False
    if tuple(topk_ids.shape) != (m, TOPK) or topk_ids.dtype != torch.int32:
        return False
    if tuple(topk_weights.shape) != (m, TOPK) or topk_weights.dtype != torch.float32:
        return False
    if w13.ndim != 3 or w2.ndim != 3 or w13.dtype != torch.int32:
        return False
    e = int(w13.shape[0])
    if e < 1 or w2.dtype != torch.int32:
        return False
    if tuple(w13.shape) != (e, HIDDEN // 16, 2 * _GATE_UP):
        return False
    if tuple(w2.shape) != (e, INTERMEDIATE // 16, 2 * HIDDEN):
        return False
    if tuple(w13_scales.shape) != (e, HIDDEN // 16, _GATE_UP):
        return False
    if tuple(w2_scales.shape) != (e, INTERMEDIATE // 16, HIDDEN):
        return False
    if (
        w13_scales.dtype != torch.float8_e4m3fn
        or w2_scales.dtype != torch.float8_e4m3fn
    ):
        return False
    for g in (w13_global_scale, w2_global_scale):
        if g is None or g.dtype != torch.float32 or tuple(g.shape) != (e,):
            return False
    for t in (
        hidden_states,
        w13,
        w2,
        w13_scales,
        w2_scales,
        w13_global_scale,
        w2_global_scale,
    ):
        if t.device != hidden_states.device or not t.is_contiguous():
            return False
    # The kernels read x and the scales in 16- and 8-byte vectors.
    if (
        hidden_states.data_ptr() % 16
        or w13_scales.data_ptr() % 8
        or w2_scales.data_ptr() % 8
    ):
        return False
    # None selects silu_and_mul in fused_marlin_moe, a different rounding.
    if swiglu_limit is None:
        return False
    try:
        return math.isfinite(float(swiglu_limit))
    except (TypeError, ValueError):
        return False


def _workspace(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    key = device.index if device.index is not None else torch.cuda.current_device()
    ws = _WS.get(key)
    if ws is None:
        # Allocated inside a capture, the counters' zero fill would only be
        # recorded, not run: eager calls before the first replay would read
        # whatever the graph pool block held.
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "sm70_hy4_nvfp4_moe: the first decode call on a device must run "
                "outside CUDA graph capture (it allocates the workspace)"
            )
        routes = MAX_DECODE_TOKENS * TOPK
        partials = torch.empty(
            (routes * _SPLIT * _GATE_UP,), dtype=torch.float32, device=device
        )
        cache = torch.empty((routes, HIDDEN), dtype=torch.float16, device=device)
        # Each launch leaves the counters at zero again.
        counters = torch.zeros(
            (MAX_DECODE_TOKENS * _DOWN_GROUPS,), dtype=torch.int32, device=device
        )
        ws = (partials, cache, counters)
        _WS[key] = ws
    return ws


def _plan(
    topk_ids: torch.Tensor, num_experts: int, routes_per_item: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Routes grouped by expert, on the device, without a host sync.

    Returns ``order`` (route indices, valid ids first, by expert), ``expert_off``
    (first position of each expert in ``order``, E + 1 entries) and ``work_off``
    (first work item of each expert; expert e has ceil(count / routes_per_item)
    items).
    """
    flat = topk_ids.reshape(-1)
    valid = (flat >= 0) & (flat < num_experts)
    key = torch.where(valid, flat, torch.full_like(flat, num_experts))
    sorted_key, order = torch.sort(key, stable=True)
    bounds = torch.arange(num_experts + 1, dtype=torch.int32, device=flat.device)
    expert_off = torch.searchsorted(sorted_key, bounds, out_int32=True)
    counts = expert_off[1:] - expert_off[:-1]
    work = torch.div(
        counts + (routes_per_item - 1), routes_per_item, rounding_mode="floor"
    )
    work_off = torch.zeros(num_experts + 1, dtype=torch.int32, device=flat.device)
    work_off[1:] = torch.cumsum(work, 0, dtype=torch.int32)
    return order.to(torch.int32), expert_off, work_off


def _log_once(kind: str, m: int, e: int, swiglu_limit: float) -> None:
    if kind not in _LOGGED:
        logger.info(
            "Hy4 SM70 NVFP4 MoE %s live: M=%d E=%d swiglu_limit=%s",
            kind,
            m,
            e,
            swiglu_limit,
        )
        _LOGGED.add(kind)


def sm70_hy4_nvfp4_moe(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scales: torch.Tensor,
    w2_scales: torch.Tensor,
    w13_global_scale: torch.Tensor,
    w2_global_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    swiglu_limit: float,
    routed_scaling_factor: Optional[float] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Routed-expert sum for any number of tokens; the caller checked ``eligible``.

    ``out`` (fp16 [M, 1536], contiguous, 16-byte aligned) may alias
    ``hidden_states``: each token's input is read before its output is written.
    """
    m = int(hidden_states.shape[0])
    e = int(w13.shape[0])
    if (
        out is None
        or out.dtype != torch.float16
        or out.shape != hidden_states.shape
        or not out.is_contiguous()
        or out.data_ptr() % 16 != 0
    ):
        out = torch.empty_like(hidden_states)
    ids = topk_ids if topk_ids.is_contiguous() else topk_ids.contiguous()
    wts = topk_weights if topk_weights.is_contiguous() else topk_weights.contiguous()
    limit = float(swiglu_limit)
    scale = 1.0 if routed_scaling_factor is None else float(routed_scaling_factor)

    if m <= MAX_DECODE_TOKENS:
        _log_once("decode", m, e, swiglu_limit)
        partials, cache, counters = _workspace(hidden_states.device)
        _decode_module(*_DECODE_CONFIGS[_rows(m)]).moe_decode(
            out,
            hidden_states,
            w13,
            w2,
            w13_scales,
            w2_scales,
            w13_global_scale,
            w2_global_scale,
            ids,
            wts,
            partials,
            cache,
            counters,
            limit,
            scale,
        )
        return out

    _log_once("prefill", m, e, swiglu_limit)
    module = _prefill_module(*_PREFILL_CONFIG)
    routes_per_item = _PREFILL_CONFIG[2] * _ROWS
    chunk = min(m, PREFILL_CHUNK)
    gate_up = torch.empty(
        (chunk * TOPK, _GATE_UP), dtype=torch.float16, device=hidden_states.device
    )
    cache = torch.empty(
        (chunk * TOPK, HIDDEN), dtype=torch.float16, device=hidden_states.device
    )
    for start in range(0, m, PREFILL_CHUNK):
        end = min(m, start + PREFILL_CHUNK)
        chunk_ids = ids[start:end]
        order, expert_off, work_off = _plan(chunk_ids, e, routes_per_item)
        module.moe_prefill(
            out[start:end],
            hidden_states[start:end],
            w13,
            w2,
            w13_scales,
            w2_scales,
            w13_global_scale,
            w2_global_scale,
            chunk_ids,
            wts[start:end],
            order,
            expert_off,
            work_off,
            gate_up,
            cache,
            limit,
            scale,
        )
    return out
