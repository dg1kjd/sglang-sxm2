"""SM70 warp-per-token sigmoid top-8 / top-16 router; bitwise equal to _router_triton_kernel."""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module

TOPKS = (8, 16)
MIN_EXPERTS = 257
MAX_EXPERTS = 512


@cache_once
def _module(topk: int) -> Module:
    args = make_cpp_args(topk)
    return load_jit(
        "sm70_router_topk",
        *args,
        cuda_files=["moe/sm70_router_topk.cuh"],
        cuda_wrappers=[("route", f"sm70_router_topk::route<{args}>")],
        extra_cuda_cflags=["--fmad=false"],
    )


def sm70_router_covered(scores: torch.Tensor, bias: torch.Tensor, topk: int) -> bool:
    return (
        topk in TOPKS
        and scores.dim() == 2
        and MIN_EXPERTS <= scores.size(1) <= MAX_EXPERTS
        and scores.dtype == torch.float32
        and scores.stride(1) == 1
        # Triton vectorizes only 16-byte aligned rows and sums in another order otherwise.
        and scores.stride(0) % 16 == 0
        and scores.data_ptr() % 16 == 0
        and bias.dtype == torch.float32
        and bias.is_contiguous()
        and scores.is_cuda
        # A CPU-offloaded bias stays on the Triton router.
        and bias.device == scores.device
        and torch.cuda.get_device_capability(scores.device) == (7, 0)
    )


def sm70_router_topk(
    scores: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
    apply_scale: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    m = scores.size(0)
    weights = torch.empty((m, topk), dtype=torch.float32, device=scores.device)
    ids = torch.empty((m, topk), dtype=torch.int32, device=scores.device)
    _module(topk).route(
        weights,
        ids,
        scores,
        bias,
        float(routed_scaling_factor),
        bool(renormalize),
        bool(apply_scale),
    )
    return weights, ids
