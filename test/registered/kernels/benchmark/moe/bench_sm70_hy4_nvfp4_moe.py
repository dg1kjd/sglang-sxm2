"""Hunyuan-4-Flash routed experts at TP8 on SM70: Marlin against the HMMA kernels.

One MoE layer per call (E = 512, top-16, routed latent 1536, I = 192 per rank).
Every captured call routes to a different random expert set, so the weights
come from HBM as in a decode step, not from L2. At M = 1 the layer reads
7.96 MB of weights and scales. M <= 8 runs the decode kernels, larger M the
prefill kernels.
"""

import itertools

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=120,
    stage="base-b-kernel-benchmark",
    runner_config="1-gpu-large",
    disabled="SM70 V100 worktree only; do not register GPU CI",
)

E = 512
H = 1536
I = 192
TOPK = 16
LIMIT = 10.0
ROUTED = 3.998
ROUTE_SETS = 64

_LAYER = {}


def _layer():
    if not _LAYER:
        import sglang.srt.layers.quantization  # noqa: F401  (import order)
        from sglang.srt.hardware_backend.gpu.quantization.gptq_kernels import (
            gptq_marlin_moe_repack,
        )
        from sglang.srt.layers.quantization.marlin_utils import (
            sm70_nvfp4_marlin_process_global_scale,
            sm70_nvfp4_marlin_process_scales,
        )

        gen = torch.Generator(device="cuda")
        gen.manual_seed(0)
        for name, n, k in (("w13", 2 * I, H), ("w2", H, I)):
            raw = torch.randint(
                0, 256, (E, n, k // 2), dtype=torch.uint8, device="cuda", generator=gen
            )
            scales = (torch.rand(E, n, k // 16, device="cuda", generator=gen) + 0.5).to(
                torch.float8_e4m3fn
            )
            gscale = torch.rand(E, device="cuda", generator=gen) * 0.01 + 0.005
            packed = gptq_marlin_moe_repack(
                raw.view(torch.int32).transpose(1, 2).contiguous(),
                torch.empty((E, 0), dtype=torch.int32, device="cuda"),
                k,
                n,
                4,
            )
            enc, factor = sm70_nvfp4_marlin_process_scales(
                scales.transpose(1, 2).contiguous(), torch.float16
            )
            g = sm70_nvfp4_marlin_process_global_scale(gscale, torch.float16) / factor
            _LAYER[name] = (packed, enc, g)
    return _LAYER["w13"], _LAYER["w2"]


def _marlin(x, ids, wts):
    from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import (
        fused_marlin_moe,
    )
    from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace

    (w13, s13, g13), (w2, s2, g2) = _layer()
    return fused_marlin_moe(
        hidden_states=x,
        w1=w13,
        w2=w2,
        w1_scale=s13,
        w2_scale=s2,
        gating_output=torch.empty(
            (x.shape[0], E), dtype=torch.float32, device=x.device
        ),
        topk_weights=wts,
        topk_ids=ids,
        global_num_experts=-1,
        expert_map=None,
        is_expert_parallel=False,
        w1_global_scale=g13,
        w2_global_scale=g2,
        workspace=marlin_make_workspace(x.device, max_blocks_per_sm=4),
        num_bits=4,
        is_k_full=True,
        inplace=False,
        routed_scaling_factor=ROUTED,
        clamp_limit=LIMIT,
        activation="silu",
        is_gated=True,
    ).to(x.dtype)


def _hy4(x, ids, wts):
    from sglang.kernels.ops.moe.sm70_hy4_nvfp4_moe import sm70_hy4_nvfp4_moe

    (w13, s13, g13), (w2, s2, g2) = _layer()
    return sm70_hy4_nvfp4_moe(x, w13, w2, s13, s2, g13, g2, ids, wts, LIMIT, ROUTED)


@marker.parametrize("num_tokens", [1, 2, 3, 4, 6, 8, 64, 512, 2048], [1])
@marker.benchmark("provider", ["marlin", "hy4"])
def benchmark(num_tokens: int, provider: str):
    if torch.cuda.get_device_capability() != (7, 0):
        marker.skip("SM70 (V100) only")
    _layer()
    gen = torch.Generator(device="cuda")
    gen.manual_seed(num_tokens)
    routes = []
    for _ in range(ROUTE_SETS):
        perm = torch.argsort(
            torch.rand(num_tokens, E, device="cuda", generator=gen), dim=1
        )
        ids = perm[:, :TOPK].int().contiguous()
        wts = torch.rand(num_tokens, TOPK, device="cuda", generator=gen) * 0.1
        routes.append((ids, wts))
    x = torch.randn(num_tokens, H, device="cuda", generator=gen).half()
    cycle = itertools.cycle(routes)
    impl = _marlin if provider == "marlin" else _hy4

    def run(x):
        ids, wts = next(cycle)
        return impl(x, ids, wts)

    return marker.do_bench(run, input_args=(x,), disable_log_bandwidth=True)


if __name__ == "__main__":
    benchmark.run()
