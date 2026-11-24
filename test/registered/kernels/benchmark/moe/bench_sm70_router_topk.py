"""SM70 warp router vs _router_triton_kernel (V100 worktree only).

Also the only guard for the fast path: with it off the outputs stay exact
(the tie fallback serves every row), but top-16 runs about 2.3x slower.
"""

from unittest.mock import patch

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.moe import sm70_router_topk
from sglang.kernels.ops.moe.moe_fused_gate import moe_fused_gate
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=30,
    stage="base-b-kernel-benchmark",
    runner_config="1-gpu-large",
    disabled="SM70 V100 worktree only; do not register GPU CI",
)

# name -> (experts, topk, routed_scaling_factor)
CONFIGS = {
    "glm5_3_flash": (288, 8, 2.5),
    "hy4_flash": (512, 16, 3.998),
}


def _triton(*args, **kwargs):
    with patch.object(sm70_router_topk, "sm70_router_covered", lambda *args: False):
        return moe_fused_gate(*args, **kwargs)


@marker.parametrize("config", list(CONFIGS), list(CONFIGS))
# 6 tokens is Hy4-flash's speculative verify batch.
@marker.parametrize("num_tokens", [1, 6, 64, 4096], [1, 6])
@marker.benchmark("provider", ["triton", "sm70"])
def benchmark(num_tokens: int, config: str, provider: str):
    num_experts, topk, scale = CONFIGS[config]
    torch.manual_seed(0)
    scores = torch.randn(num_tokens, num_experts, device="cuda") * 2
    bias = torch.randn(num_experts, device="cuda") * 0.1
    if provider == "triton":
        fn = _triton
        kwargs = dict(
            topk=topk,
            scoring_func="sigmoid",
            renormalize=True,
            routed_scaling_factor=scale,
        )
    else:
        fn = sm70_router_topk.sm70_router_topk
        kwargs = dict(
            topk=topk,
            renormalize=True,
            routed_scaling_factor=scale,
            apply_scale=False,
        )
    return marker.do_bench(fn, input_args=(scores, bias), input_kwargs=kwargs)


if __name__ == "__main__":
    benchmark.run()
