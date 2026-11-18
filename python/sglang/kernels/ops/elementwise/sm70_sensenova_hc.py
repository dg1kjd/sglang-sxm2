"""SM70 mHC router and merge for SenseNova-Loop (2 streams, hidden 4096).

One router call is two kernels (fp64 partial dots per column split, then the
logits, sigmoids, Sinkhorn and the branch combine with an optional Gemma norm);
merge_route runs the previous stage's merge inside the first of them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module

HIDDEN = 4096
STREAMS = 2
SPLITS = 32
ACC = 9


@cache_once
def _module() -> Module:
    cap = torch.cuda.get_device_capability()
    if cap != (7, 0):
        raise RuntimeError(f"sm70_sensenova_hc requires SM70; got SM{cap[0]}{cap[1]}")
    return load_jit(
        "sm70_sensenova_hc",
        cuda_files=["elementwise/sm70_sensenova_hc.cuh"],
        cuda_wrappers=[
            ("route", "sm70_sensenova_hc::route"),
            ("route_norm", "sm70_sensenova_hc::route_norm"),
            ("merge_route", "sm70_sensenova_hc::merge_route"),
            ("merge_route_norm", "sm70_sensenova_hc::merge_route_norm"),
            ("merge", "sm70_sensenova_hc::merge"),
        ],
        extra_cuda_cflags=["--fmad=false"],
    )


def sensenova_hc_supported(
    *,
    streams: int,
    hidden_size: int,
    fixed_identity: bool,
    pre_scale: float,
    iterations: int,
) -> bool:
    return (
        streams == STREAMS
        and hidden_size == HIDDEN
        and not fixed_identity
        and pre_scale == 1.0
        and iterations >= 1
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability() == (7, 0)
    )


def _route_outputs(tokens: int, device: torch.device):
    return (
        torch.empty((tokens, HIDDEN), dtype=torch.float16, device=device),
        torch.empty((tokens, STREAMS), dtype=torch.float32, device=device),
        torch.empty((tokens, STREAMS), dtype=torch.float32, device=device),
        torch.empty((tokens, STREAMS, STREAMS), dtype=torch.float32, device=device),
        torch.empty((tokens, SPLITS, ACC), dtype=torch.float64, device=device),
    )


def sensenova_hc_route(
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    iterations: int,
    rms_eps: float,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """residual [T, 8192] fp16, weight [8, 8192] and bias [8] fp32.

    Returns branch [T, 4096] fp16, pre [T, 2], post [T, 2] and mix [T, 2, 2] fp32.
    With norm_weight (a GemmaRMSNorm weight, [4096] fp16) branch comes back as
    sgl_kernel gemma_rmsnorm(branch, norm_weight, norm_eps), bitwise.
    """
    residual = residual.contiguous()
    out, pre, post, mix, partials = _route_outputs(residual.shape[0], residual.device)
    if residual.shape[0]:
        args = (out, pre, post, mix, partials, residual, weight, bias)
        if norm_weight is None:
            _module().route(*args, int(iterations), float(rms_eps))
        else:
            _module().route_norm(
                *args, norm_weight, int(iterations), float(rms_eps), float(norm_eps)
            )
    return out, pre, post, mix


def sensenova_hc_merge_route(
    branch: torch.Tensor,
    residual: torch.Tensor,
    mix: torch.Tensor,
    post: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    iterations: int,
    rms_eps: float,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """sensenova_hc_route(sensenova_hc_merge(branch, residual, mix, post), ...).

    Returns the router's branch, the merged streams [T, 8192] fp16, and the
    router's pre, post and mix.
    """
    tokens = residual.shape[0]
    residual = residual.contiguous()
    merged = torch.empty_like(residual)
    out, pre, post_out, mix_out, partials = _route_outputs(tokens, residual.device)
    if tokens:
        args = (
            out,
            pre,
            post_out,
            mix_out,
            partials,
            merged,
            branch.contiguous(),
            residual,
            mix.contiguous(),
            post.contiguous(),
            weight,
            bias,
        )
        if norm_weight is None:
            _module().merge_route(*args, int(iterations), float(rms_eps))
        else:
            _module().merge_route_norm(
                *args, norm_weight, int(iterations), float(rms_eps), float(norm_eps)
            )
    return out, merged, pre, post_out, mix_out


def sensenova_hc_merge(
    branch: torch.Tensor,
    residual: torch.Tensor,
    mix: torch.Tensor,
    post: torch.Tensor,
) -> torch.Tensor:
    """fp16(mix16 @ streams) + fp16(post16 * branch), as the torch fp16 reference.

    branch [T, 4096] fp16, residual [T, 8192] fp16, mix [T, 2, 2], post [T, 2] fp32.
    """
    residual = residual.contiguous()
    out = torch.empty_like(residual)
    if residual.shape[0]:
        _module().merge(
            out, branch.contiguous(), residual, mix.contiguous(), post.contiguous()
        )
    return out
