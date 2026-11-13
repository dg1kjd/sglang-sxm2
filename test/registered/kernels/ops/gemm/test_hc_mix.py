import functools
import hashlib
import os
import pathlib
import sys
import tempfile

import pytest
import torch
import torch.nn.functional as F

from sglang.kernels.ops.gemm.hc_mix import (
    _FUSED_MIX_MAX_ROWS,
    fused_hc_mix,
    fused_hc_mix_supported,
)
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

HC_COUNT = 4
HIDDEN_SIZE = 2560
LOWRANK = 320


def _reference_mix(
    hyper_input_normed: torch.Tensor,
    w_down: torch.Tensor,
    w_up: torch.Tensor,
    hc: int,
    hs: int,
    compute_dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Mirrors GatedResidual._mix_compute in hyperconnection.py."""
    x = hyper_input_normed.to(compute_dtype)
    t = F.silu(F.linear(x, w_down.to(compute_dtype)) / hc)
    u = torch.sigmoid(F.linear(t, w_up.to(compute_dtype)))
    return (u.unflatten(-1, (hc, hs)) * x.unflatten(-1, (hc, hs))).mean(dim=-2)


def _make_inputs(num_tokens: int, dtype: torch.dtype):
    torch.manual_seed(0)
    x = torch.randn(num_tokens, HC_COUNT * HIDDEN_SIZE, dtype=dtype, device="cuda")
    w_down = (
        torch.randn(LOWRANK, HC_COUNT * HIDDEN_SIZE, dtype=dtype, device="cuda") * 0.02
    )
    w_up = (
        torch.randn(HC_COUNT * HIDDEN_SIZE, LOWRANK, dtype=dtype, device="cuda") * 0.02
    )
    return x, w_down, w_up


_TOLERANCES = {
    torch.bfloat16: dict(rtol=1e-2, atol=5e-3),
    torch.float16: dict(rtol=2e-3, atol=1e-3),
}


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("num_tokens", [1, 4, 7, _FUSED_MIX_MAX_ROWS])
def test_fused_hc_mix_matches_reference(dtype, num_tokens):
    x, w_down, w_up = _make_inputs(num_tokens, dtype)
    assert fused_hc_mix_supported(x, w_down, w_up)
    out = fused_hc_mix(x, w_down, w_up, HC_COUNT, HIDDEN_SIZE)
    ref = _reference_mix(x, w_down, w_up, HC_COUNT, HIDDEN_SIZE)
    torch.testing.assert_close(out.to(torch.float64), ref, **_TOLERANCES[dtype])


def test_fused_hc_mix_no_less_accurate_than_eager():
    """The fused kernel (fp32 accumulation throughout) must not be farther
    from the fp64 reference than the eager bf16 chain it replaces."""
    x, w_down, w_up = _make_inputs(8, torch.bfloat16)
    ref = _reference_mix(x, w_down, w_up, HC_COUNT, HIDDEN_SIZE)
    fused = fused_hc_mix(x, w_down, w_up, HC_COUNT, HIDDEN_SIZE)
    eager = _reference_mix(
        x, w_down, w_up, HC_COUNT, HIDDEN_SIZE, compute_dtype=torch.bfloat16
    )
    fused_err = (fused.to(torch.float64) - ref).abs().max()
    eager_err = (eager.to(torch.float64) - ref).abs().max()
    assert fused_err <= eager_err * 1.5 + 1e-6


def test_fused_hc_mix_gate_rejects_prefill_rows():
    x, w_down, w_up = _make_inputs(_FUSED_MIX_MAX_ROWS + 1, torch.bfloat16)
    assert not fused_hc_mix_supported(x, w_down, w_up)


# ---- SM70 native projections (sm70_hc_batch.cuh) ----
#
# The kernels below are the ones the native path shipped with first: one CTA
# per down output, one warp per up hidden coordinate. Greedy baselines were
# recorded with them, so the current kernels must return the same bits. The
# copy lives only here; namespaces are renamed so no symbol is shared.
_FROZEN_SM70_HC = r"""
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.cuh>
#include <tvm/ffi/container/tensor.h>
#include <cuda_fp16.h>
namespace sglang::sm70_hc_frozen {
template <int M, int NT, bool Inject = false>
__global__ void down(
    const half* __restrict__ x,
    const half* __restrict__ w,
    half* __restrict__ o,
    const half* __restrict__ inject = nullptr,
    float* __restrict__ partials = nullptr) {
  int row = blockIdx.x;
  if constexpr (Inject) {
    if (row >= 320) {
      int branch = row - 320, warp = threadIdx.x / 32, lane = threadIdx.x % 32;
      float sum[M] = {};
#pragma unroll
      for (int j = 0; j < 5; ++j) {
        int k = (threadIdx.x + j * 256) * 8;
        int4 wv = *reinterpret_cast<const int4*>(inject + branch * 10240 + k);
        const half* hw = reinterpret_cast<const half*>(&wv);
#pragma unroll
        for (int m = 0; m < M; ++m) {
          int4 xv = *reinterpret_cast<const int4*>(x + m * 10240 + k);
          const half* hx = reinterpret_cast<const half*>(&xv);
#pragma unroll
          for (int i = 0; i < 4; ++i)
            sum[m] += __half2float(hx[2 * i]) * __half2float(hw[2 * i]) +
                      __half2float(hx[2 * i + 1]) * __half2float(hw[2 * i + 1]);
        }
      }
#pragma unroll
      for (int m = 0; m < M; ++m) {
#pragma unroll
        for (int d = 16; d > 0; d /= 2)
          sum[m] += __shfl_xor_sync(0xffffffff, sum[m], d);
        if (lane == 0) partials[(m * 8 + warp) * 4 + branch] = sum[m];
      }
      return;
    }
  }
  float acc[M][8] = {};
#pragma unroll
  for (int j = 0; j < (10240 + NT * 8 - 1) / (NT * 8); ++j) {
    int k = (threadIdx.x + j * NT) * 8;
    if (k >= 10240) continue;
    int4 wv = *reinterpret_cast<const int4*>(w + row * 10240 + k);
    const half* hw = reinterpret_cast<const half*>(&wv);
#pragma unroll
    for (int m = 0; m < M; ++m) {
      int4 xv = *reinterpret_cast<const int4*>(x + m * 10240 + k);
      const half* hx = reinterpret_cast<const half*>(&xv);
#pragma unroll
      for (int i = 0; i < 8; ++i)
        acc[m][i] += __half2float(hx[i]) * __half2float(hw[i]);
    }
  }
  __shared__ float part[M][NT / 32];
#pragma unroll
  for (int m = 0; m < M; ++m) {
    float sum = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i)
      sum += acc[m][i];
#pragma unroll
    for (int d = 16; d > 0; d /= 2)
      sum += __shfl_down_sync(0xffffffff, sum, d);
    if (threadIdx.x % 32 == 0) part[m][threadIdx.x / 32] = sum;
  }
  __syncthreads();
  if (threadIdx.x < M) {
    int m = threadIdx.x;
    float sum = 0;
#pragma unroll
    for (int i = 0; i < NT / 32; ++i)
      sum += part[m][i];
    sum = __half2float(__float2half_rn(sum)) * .25f;
    o[m * 320 + row] = __float2half_rn(sum / (1.f + __expf(-sum)));
  }
}
template <int M, int NT>
__global__ void
up(const half* __restrict__ a, const half* __restrict__ x, const half* __restrict__ w, half* __restrict__ o) {
  int lane = threadIdx.x % 32, sub = lane % 8, branch = lane / 8, h = blockIdx.x * (NT / 32) + threadIdx.x / 32,
      row = branch * 2560 + h;
  float acc[M][8] = {};
#pragma unroll
  for (int j = 0; j < 5; ++j) {
    int k = (sub + j * 8) * 8;
    int4 wv = *reinterpret_cast<const int4*>(w + row * 320 + k);
    const half* hw = reinterpret_cast<const half*>(&wv);
#pragma unroll
    for (int m = 0; m < M; ++m) {
      int4 av = *reinterpret_cast<const int4*>(a + m * 320 + k);
      const half* ha = reinterpret_cast<const half*>(&av);
#pragma unroll
      for (int i = 0; i < 8; ++i)
        acc[m][i] += __half2float(ha[i]) * __half2float(hw[i]);
    }
  }
#pragma unroll
  for (int m = 0; m < M; ++m) {
    float v = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i)
      v += acc[m][i];
#pragma unroll
    for (int d = 4; d > 0; d /= 2)
      v += __shfl_down_sync(0xffffffff, v, d, 8);
    v = __half2float(__float2half_rn(v));
    v = __half2float(x[m * 10240 + row]) / (1.f + __expf(-v));
    float b1 = __shfl_sync(0xffffffff, v, 8), b2 = __shfl_sync(0xffffffff, v, 16), b3 = __shfl_sync(0xffffffff, v, 24);
    if (lane == 0) o[m * 2560 + h] = __float2half_rn((((v + b1) + b2) + b3) * .25f);
  }
}
// The single-row kernels were a separate source file.
__global__ __launch_bounds__(256) void down_single(
    const half* __restrict__ x, const half* __restrict__ weight,
    half* __restrict__ output) {
  const int row = blockIdx.x;
  float accum[8] = {};
#pragma unroll
  for (int j = 0; j < 5; ++j) {
    const int offset = (threadIdx.x + j * 256) * 8;
    const int4 xv = *reinterpret_cast<const int4*>(x + offset);
    const int4 wv =
        *reinterpret_cast<const int4*>(weight + row * 10240 + offset);
    const half* hx = reinterpret_cast<const half*>(&xv);
    const half* hw = reinterpret_cast<const half*>(&wv);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      accum[i] += __half2float(hx[i]) * __half2float(hw[i]);
    }
  }
  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < 8; ++i) sum += accum[i];
#pragma unroll
  for (int delta = 16; delta > 0; delta /= 2) {
    sum += __shfl_down_sync(0xffffffffu, sum, delta);
  }
  __shared__ float partials[8];
  if ((threadIdx.x & 31) == 0) partials[threadIdx.x / 32] = sum;
  __syncthreads();
  if (threadIdx.x == 0) {
    sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) sum += partials[i];
    sum = __half2float(__float2half_rn(sum)) * 0.25f;
    output[row] = __float2half_rn(sum / (1.0f + __expf(-sum)));
  }
}
__global__ __launch_bounds__(256) void up_single(
    const half* __restrict__ activated, const half* __restrict__ x,
    const half* __restrict__ weight, half* __restrict__ output) {
  const int lane = threadIdx.x & 31;
  const int sublane = lane & 7;
  const int branch = lane >> 3;
  const int hidden = blockIdx.x * 8 + threadIdx.x / 32;
  const int row = branch * 2560 + hidden;
  float accum[8] = {};
#pragma unroll
  for (int j = 0; j < 5; ++j) {
    const int k = (sublane + j * 8) * 8;
    const int4 av = *reinterpret_cast<const int4*>(activated + k);
    const int4 wv = *reinterpret_cast<const int4*>(weight + row * 320 + k);
    const half* ha = reinterpret_cast<const half*>(&av);
    const half* hw = reinterpret_cast<const half*>(&wv);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      accum[i] += __half2float(ha[i]) * __half2float(hw[i]);
    }
  }
  float value = 0.0f;
#pragma unroll
  for (int i = 0; i < 8; ++i) value += accum[i];
#pragma unroll
  for (int delta = 4; delta > 0; delta /= 2) {
    value += __shfl_down_sync(0xffffffffu, value, delta, 8);
  }
  value = __half2float(__float2half_rn(value));
  value = __half2float(x[row]) / (1.0f + __expf(-value));
  const float branch1 = __shfl_sync(0xffffffffu, value, 8);
  const float branch2 = __shfl_sync(0xffffffffu, value, 16);
  const float branch3 = __shfl_sync(0xffffffffu, value, 24);
  if (lane == 0) {
    output[hidden] =
        __float2half_rn((((value + branch1) + branch2) + branch3) * 0.25f);
  }
}
using TV = tvm::ffi::TensorView;
template <typename T>
T* ptr(TV t) { return static_cast<T*>(t.data_ptr()); }
template <int M>
void down_run(TV x, TV w, TV o) {
  if constexpr (M == 1) {
    host::LaunchKernel(320, 256, x.device())(down_single, ptr<const half>(x), ptr<const half>(w), ptr<half>(o));
  } else {
    host::LaunchKernel(320, 256, x.device())(
        down<M, 256>, ptr<const half>(x), ptr<const half>(w), ptr<half>(o),
        static_cast<const half*>(nullptr), static_cast<float*>(nullptr));
  }
}
template <int M>
void up_run(TV a, TV x, TV w, TV o) {
  if constexpr (M == 1) {
    host::LaunchKernel(320, 256, x.device())(
        up_single, ptr<const half>(a), ptr<const half>(x), ptr<const half>(w), ptr<half>(o));
  } else {
    host::LaunchKernel(640, 128, x.device())(
        up<M, 128>, ptr<const half>(a), ptr<const half>(x), ptr<const half>(w), ptr<half>(o));
  }
}
template <int M>
void gate_down_run(TV x, TV w, TV inject, TV o, TV partials) {
  host::LaunchKernel(324, 256, x.device())(
      down<M, 256, true>, ptr<const half>(x), ptr<const half>(w), ptr<half>(o),
      ptr<const half>(inject), ptr<float>(partials));
}
}  // namespace sglang::sm70_hc_frozen
"""

# A Qwen3.8-Flash-Next shard holding hyper-connection weights, e.g. its
# model-bf16-00001.safetensors; the model-weight case skips without it.
_REAL_WEIGHTS = envs.SGLANG_TEST_HC_MIX_WEIGHTS.get()

sm70_only = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="The native hyperconnection kernels require V100",
)


@functools.lru_cache(maxsize=None)
def _frozen_module(rows: int, gated: bool):
    from sglang.kernels.jit.utils import load_jit

    # A content-addressed path, so repeated runs reuse one JIT cache entry.
    digest = hashlib.sha256(_FROZEN_SM70_HC.encode()).hexdigest()[:16]
    source = pathlib.Path(tempfile.gettempdir()) / f"hc_first_kernels_{digest}.cuh"
    if not source.exists() or source.read_text() != _FROZEN_SM70_HC:
        staging = source.with_suffix(f".{os.getpid()}.tmp")
        staging.write_text(_FROZEN_SM70_HC)
        staging.replace(source)
    # Built as shipped: the plain module without FMA contraction, the gated
    # module with the compiler default.
    if gated:
        return load_jit(
            "sm70_hc_frozen_gate",
            rows,
            cuda_files=[str(source)],
            cuda_wrappers=[("down", f"sglang::sm70_hc_frozen::gate_down_run<{rows}>")],
        )
    return load_jit(
        "sm70_hc_frozen",
        rows,
        cuda_files=[str(source)],
        cuda_wrappers=[
            ("down", f"sglang::sm70_hc_frozen::down_run<{rows}>"),
            ("up", f"sglang::sm70_hc_frozen::up_run<{rows}>"),
        ],
        extra_cuda_cflags=["--fmad=false"],
    )


def _frozen_down(x, w_down):
    out = torch.empty((x.shape[0], LOWRANK), dtype=x.dtype, device=x.device)
    _frozen_module(x.shape[0], gated=False).down(x, w_down, out)
    return out


def _frozen_up(activated, x, w_up):
    out = torch.empty((x.shape[0], HIDDEN_SIZE), dtype=x.dtype, device=x.device)
    _frozen_module(x.shape[0], gated=False).up(activated, x, w_up, out)
    return out


def _frozen_gated_down(x, w_down, inject):
    out = torch.empty((x.shape[0], LOWRANK), dtype=x.dtype, device=x.device)
    partials = torch.empty((x.shape[0], 8, 4), dtype=torch.float32, device=x.device)
    _frozen_module(x.shape[0], gated=True).down(x, w_down, inject, out, partials)
    return out, partials


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int16 if t.dtype == torch.float16 else torch.int32)


def _assert_same_bits(actual: torch.Tensor, expected: torch.Tensor, what: str):
    differing = (_bits(actual) != _bits(expected)).sum().item()
    assert differing == 0, f"{what}: {differing} of {actual.numel()} values differ"


def _sm70_inputs(rows: int, kind: str, seed: int):
    """(x, activated, w_down, w_up, inject) in fp16 on the GPU."""
    g = torch.Generator().manual_seed(seed)
    wide = HC_COUNT * HIDDEN_SIZE
    x = torch.randn(rows, wide, generator=g)
    activated = torch.randn(rows, LOWRANK, generator=g)
    w_down = torch.randn(LOWRANK, wide, generator=g) * 0.02
    w_up = torch.randn(wide, LOWRANK, generator=g) * 0.02
    inject = torch.randn(HC_COUNT, wide, generator=g) * 0.02
    if kind == "wide_range":
        # Sums that cancel and span many binades, so every reordering of the
        # FP32 additions shows up in the rounded result.
        x = x * torch.exp2(torch.randint(-6, 7, x.shape, generator=g).float())
        activated = activated * torch.exp2(
            torch.randint(-6, 7, activated.shape, generator=g).float()
        )
        w_down = w_down * 8
        w_up = w_up * 8
    elif kind == "special":
        # Subnormals, signed zeros and overflow to infinity take the same path
        # through a fused multiply-add as through multiply-then-add. Only every
        # 16th down output overflows, and through it the chained up projection.
        x = x.half()
        x.view(torch.int16)[:, ::7] = 1
        x.view(torch.int16)[:, 1::7] = -32768
        x[:, 2::511] = 60000.0
        x[:, 3::1021] = -60000.0
        w_down = w_down.half()
        w_down[::16, 2::511] = 30000.0
        w_down.view(torch.int16)[:, 5::13] = 3
        activated = activated.half()
        activated.view(torch.int16)[:, ::37] = 1
        activated.view(torch.int16)[:, 1::37] = -32768
    return tuple(
        t.to(torch.float16).cuda().contiguous()
        for t in (x, activated, w_down, w_up, inject)
    )


def _native():
    from sglang.kernels.ops.elementwise import sm70_hc_mix

    return sm70_hc_mix


@sm70_only
@pytest.mark.parametrize("kind", ["normal", "wide_range", "special"])
@pytest.mark.parametrize("rows", [1, 2, 4])
def test_sm70_hc_same_bits_as_first_kernels(rows, kind):
    native = _native()
    for seed in range(3):
        x, activated, w_down, w_up, inject = _sm70_inputs(rows, kind, seed)
        down = native.hc_down(x, w_down)
        _assert_same_bits(down, _frozen_down(x, w_down), "down")
        # Both a free activation and the chained one: the chained values are
        # small, the free ones exercise the gate's full range.
        for act in (activated, down):
            _assert_same_bits(
                native.hc_up(act, x, w_up), _frozen_up(act, x, w_up), "up"
            )
        if rows == 1:
            continue
        gated, partials = native.hc_down_with_gate(x, w_down, inject)
        frozen_gated, frozen_partials = _frozen_gated_down(x, w_down, inject)
        _assert_same_bits(gated, frozen_gated, "gated down")
        _assert_same_bits(gated, down, "gated down vs plain down")
        _assert_same_bits(partials, frozen_partials, "gate partials")


@sm70_only
@pytest.mark.skipif(
    not _REAL_WEIGHTS or not pathlib.Path(_REAL_WEIGHTS).exists(),
    reason="set SGLANG_TEST_HC_MIX_WEIGHTS to a Qwen3.8 shard with HC weights",
)
@pytest.mark.parametrize("rows", [1, 2, 4])
def test_sm70_hc_same_bits_on_model_weights(rows):
    from safetensors import safe_open

    native = _native()
    with safe_open(_REAL_WEIGHTS, framework="pt", device="cpu") as f:
        prefixes = sorted(
            {
                key.rsplit(".input_mix_weight_down.weight", 1)[0]
                for key in f.keys()
                if key.endswith("hyper_connection.input_mix_weight_down.weight")
            }
        )
        assert prefixes, "no hyper-connection weights in the first shard"
        picked = [prefixes[0], prefixes[len(prefixes) // 2], prefixes[-1]]
        for index, prefix in enumerate(picked):
            w_down, w_up, inject = (
                f.get_tensor(f"{prefix}.{name}.weight").to(torch.float16).cuda()
                for name in (
                    "input_mix_weight_down",
                    "input_mix_weight_up",
                    "block_inject_weight",
                )
            )
            x = _sm70_inputs(rows, "normal", 100 + index)[0]
            down = native.hc_down(x, w_down)
            _assert_same_bits(down, _frozen_down(x, w_down), f"{prefix} down")
            _assert_same_bits(
                native.hc_up(down, x, w_up), _frozen_up(down, x, w_up), f"{prefix} up"
            )
            if rows > 1:
                gated, partials = native.hc_down_with_gate(x, w_down, inject)
                frozen_gated, frozen_partials = _frozen_gated_down(x, w_down, inject)
                _assert_same_bits(gated, frozen_gated, f"{prefix} gated down")
                _assert_same_bits(partials, frozen_partials, f"{prefix} partials")


@sm70_only
def test_sm70_hc_row_result_independent_of_batch_shape():
    """A token row gets the same bits alone, in a pair and in a batch of four;
    MTP verify and the single-row paths must agree on a shared token."""
    native = _native()
    x, activated, w_down, w_up, inject = _sm70_inputs(4, "wide_range", 7)
    down4 = native.hc_down(x, w_down)
    up4 = native.hc_up(activated, x, w_up)
    gated4, partials4 = native.hc_down_with_gate(x, w_down, inject)
    for rows in (1, 2):
        for start in range(0, 4, rows):
            sl = slice(start, start + rows)
            xs, acts = x[sl].contiguous(), activated[sl].contiguous()
            _assert_same_bits(native.hc_down(xs, w_down), down4[sl], "down")
            _assert_same_bits(native.hc_up(acts, xs, w_up), up4[sl], "up")
            if rows == 2:
                gated, partials = native.hc_down_with_gate(xs, w_down, inject)
                _assert_same_bits(gated, gated4[sl], "gated down")
                _assert_same_bits(partials, partials4[sl], "gate partials")


@sm70_only
@pytest.mark.parametrize("rows", [1, 2, 4])
def test_sm70_hc_repeatable_over_calls_and_graph_replays(rows):
    native = _native()
    x, _, w_down, w_up, inject = _sm70_inputs(rows, "wide_range", 11)

    def run():
        if rows > 1:
            down, partials = native.hc_down_with_gate(x, w_down, inject)
        else:
            down, partials = native.hc_down(x, w_down), None
        return down, native.hc_up(down, x, w_up), partials

    first = [t.clone() for t in run() if t is not None]
    for _ in range(50):
        for got, want in zip((t for t in run() if t is not None), first):
            _assert_same_bits(got, want, "repeat call")

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = [t for t in run() if t is not None]
    scales = [1.0, 0.5, -3.0]
    for step in range(6):
        # New input values in the captured buffer, then back to the first ones.
        scale = scales[step % 3]
        original = x.clone()
        x.mul_(scale)
        graph.replay()
        replayed = [t.clone() for t in captured]
        eager = [t for t in run() if t is not None]
        frozen_down = _frozen_down(x, w_down)
        x.copy_(original)
        for got, want in zip(replayed, eager):
            _assert_same_bits(got, want, f"replay {step}")
        _assert_same_bits(replayed[0], frozen_down, f"replay {step} vs first kernels")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
