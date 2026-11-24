"""Hunyuan-4-Flash linears at TP8 on SM70: the batch-invariant fixed-order GEMM
against what it replaces, cuBLAS for fp16 weights and TurboMind for block-FP8.

do_bench rotates copies of the operands past L2, so the weights come from HBM
as in a decode step. Floor per shape: weight bytes / 900 GB/s (lm_head 103 us,
fused_qkv_a 10.9 us at M = 1).
"""

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=180,
    stage="base-b-kernel-benchmark",
    runner_config="1-gpu-large",
    disabled="SM70 V100 worktree only; do not register GPU CI",
)

# name -> (N, K, kind): f16 (fp16 out), f16o32 (fp32 out), fp8 (block-FP8 W).
SHAPES = {
    "fused_qkv_a": (1600, 3072, "f16"),
    "routed_latent_down": (1536, 3072, "f16"),
    "shared_down": (3072, 192, "f16"),
    "router": (512, 3072, "f16o32"),
    "dense_gate_up": (3456, 3072, "f16"),
    "engram_kv": (15360, 192, "f16"),
    "lm_head": (15104, 3072, "f16o32"),
    "dfly_eh_proj": (3072, 6144, "f16"),
    "linear_gate_swa": (2048, 3072, "fp8"),
    "o_proj_swa": (3072, 2048, "fp8"),
}

_TM = {}


def _turbomind():
    if not _TM:
        from sglang.srt.layers.quantization import sm70_turbomind_fp8

        assert sm70_turbomind_fp8._load_sm70_turbomind_fp8_ops()
        _TM["ops"] = torch.ops.sglang_sm70_turbomind
    return _TM["ops"]


@marker.parametrize("shape", list(SHAPES), ["fused_qkv_a", "lm_head", "o_proj_swa"])
@marker.parametrize("num_tokens", [1, 2, 4, 6, 8, 64, 512, 2048], [1, 6])
@marker.benchmark("provider", ["cublas", "turbomind", "fixed_order"])
def benchmark(shape: str, num_tokens: int, provider: str):
    if torch.cuda.get_device_capability() != (7, 0):
        marker.skip("SM70 (V100) only")
    from sglang.kernels.ops.gemm import sm70_fixed_order_gemm as g

    n, k, kind = SHAPES[shape]
    fp8 = kind == "fp8"
    if provider == ("cublas" if fp8 else "turbomind"):
        marker.skip(f"{provider} does not serve {kind} weights")
    gen = torch.Generator(device="cuda")
    gen.manual_seed(0)
    x = torch.randn(num_tokens, k, device="cuda", generator=gen).half()
    out_dtype = torch.float32 if kind == "f16o32" else torch.float16
    out = torch.empty(num_tokens, n, device="cuda", dtype=out_dtype)
    if fp8:
        w = (torch.randn(n, k, device="cuda", generator=gen) * 0.5).to(
            torch.float8_e4m3fn
        )
        scale = torch.rand(n // 128, k // 128, device="cuda", generator=gen) * 1e-2
        if provider == "fixed_order":
            return marker.do_bench(
                lambda x, w, s: g.fp8_linear(x, w, s, out=out),
                input_args=(x, w, scale),
                disable_log_bandwidth=True,
            )
        tw, ts, meta = _turbomind().fp8_prepare(w, scale, 128, False)
        k_ld, q_ld = int(meta[0].item()), int(meta[1].item())
        return marker.do_bench(
            lambda x, w, s: _turbomind().fp8_gemm(out, x, w, s, 128, k_ld, q_ld, False),
            input_args=(x, tw, ts),
            disable_log_bandwidth=True,
        )
    w = (torch.randn(n, k, device="cuda", generator=gen) * 0.02).half()
    if provider == "fixed_order":
        fn = lambda x, w: g.linear(x, w, out=out, out_dtype=out_dtype)
    elif out_dtype == torch.float32:
        fn = lambda x, w: torch.mm(x, w.T, out_dtype=torch.float32, out=out)
    else:
        fn = lambda x, w: torch.mm(x, w.T, out=out)
    return marker.do_bench(fn, input_args=(x, w), disable_log_bandwidth=True)


if __name__ == "__main__":
    benchmark.run()
