from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_ordered_topk_module() -> Module:
    args = make_cpp_args(is_arch_support_pdl())
    return load_jit(
        "ordered_topk",
        *args,
        cuda_files=["attention/ordered_topk.cuh"],
        cuda_wrappers=[("ordered_topk", f"sglang::OrderedTopKKernel<{args}>::run")],
    )


def ordered_topk(
    score: torch.Tensor,
    lengths: torch.Tensor,
    topk: int,
    row_starts: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Exact per-row top-k over a fp32 score matrix, with a fixed output.

    Row b selects the `topk` largest values in
    ``score[b, row_starts[b] : row_starts[b] + lengths[b]]`` and returns their
    indices relative to ``row_starts[b]`` in ascending order. Among equal
    scores at the cut-off the smaller index is selected, so the result is the
    same as a stable sort by descending score, cut at `topk`, then sorted by
    index. It does not depend on the batch, the call or a CUDA graph replay.

    A row with ``lengths[b] <= topk`` gets ``0 .. lengths[b] - 1`` followed by
    -1, and its scores are not read. -0.0 equals +0.0. NaN is ordered by its
    bits: above +inf with the sign bit clear, below -inf with it set.

    Coverage: any CUDA architecture, fp32 scores, any `topk` >= 1 and row length.

    Parameters
    ----------
    score      : CUDA fp32 tensor [B, L]
    lengths    : CUDA int32 tensor [B]
    topk       : number of indices per row
    row_starts : optional CUDA int32 tensor [B]; defaults to zeros

    Returns
    -------
    CUDA int32 tensor [B, topk]
    """
    if topk < 1:
        # The kernel's cut-off search needs at least one element to find.
        raise ValueError(f"ordered_topk needs topk >= 1, got {topk}")
    batch = score.shape[0]
    if row_starts is None:
        row_starts = torch.zeros(batch, dtype=torch.int32, device=score.device)
    indices = score.new_empty((batch, topk), dtype=torch.int32)

    _jit_ordered_topk_module().ordered_topk(score, row_starts, indices, lengths)
    return indices
