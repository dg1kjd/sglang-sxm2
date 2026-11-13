// SPDX-License-Identifier: Apache-2.0
// Qwen3.8 Flash Next's FP16 hyper-connection projections on Volta, for 1, 2 or
// 4 token rows. The explicit FP16 rounding at each projection output is part of
// the contract.
#include <sgl_kernel/tensor.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cuda_fp16.h>
namespace sglang::sm70_hc_batch {

constexpr int kWide = 10240;   // 4 branches x 2560 hidden
constexpr int kLowRank = 320;  // down-projection outputs
constexpr int kHidden = 2560;
constexpr int kDownThreads = 256;
constexpr int kUpThreads = 128;
// A thread owns this many 8-half vectors of a kWide row (down) or of a
// kLowRank row within its 8-lane group (up).
constexpr int kVecs = 5;

__device__ __forceinline__ int4 load16(const half* p) {
  return *reinterpret_cast<const int4*>(p);
}

__device__ __forceinline__ void to_floats(const int4& raw, float* f) {
  const half* h = reinterpret_cast<const half*>(&raw);
#pragma unroll
  for (int i = 0; i < 8; ++i)
    f[i] = __half2float(h[i]);
}

// acc[t][i] += x[t][j][i] * w[j][i] over j, in order. An FP16 x FP16 product is
// exact in FP32, so the fused multiply-add rounds exactly like multiply-then-add
// and the result does not depend on the compiler's contraction setting.
template <int T>
__device__ __forceinline__ void
dot_chains(const float (&xf)[T][kVecs][8], const int4 (&raw)[kVecs], float (&acc)[T][8]) {
#pragma unroll
  for (int j = 0; j < kVecs; ++j) {
    float wf[8];
    to_floats(raw[j], wf);
#pragma unroll
    for (int t = 0; t < T; ++t)
#pragma unroll
      for (int i = 0; i < 8; ++i)
        acc[t][i] = fmaf(xf[t][j][i], wf[i], acc[t][i]);
  }
}

/**
 * \brief Down projection with SiLU: o[m, r] = silu(fp16(dot(x[m], w[r])) / 4).
 *
 * A CTA computes R consecutive outputs for T of the G * T token rows. Thread t
 * owns vectors t, t + 256, ... of the kWide axis; it converts its slice of the
 * CTA's token rows to FP32 once and keeps it in registers. With G > 1 a weight
 * row is read by G CTAs, and all but the first read are served by L2: this
 * keeps the registers per thread low enough for two CTAs per SM, which
 * measures faster than one CTA holding all token rows.
 *
 * A dot product is summed as: 8 chains per thread over its 5 vectors, the 8
 * chains in index order, a pairwise tree over the lanes of a warp, then the 8
 * warps in order. That order is fixed; T, G, R and the grid do not change any
 * result.
 *
 * With Inject the grid has 4 * G more CTAs, one per token row group and
 * branch c. Their warps w write the partial dot product of each token row with
 * inject[c] over the warp's slice to partials[m, w, c]; the combine kernel adds
 * the 8 warps in order.
 *
 * \tparam T      Token rows per CTA.
 * \tparam G      Token row groups; the kernel handles G * T token rows.
 * \tparam R      Outputs per CTA; the grid is G * kLowRank / R, plus 4 * G with Inject.
 * \tparam Inject Also compute the injection-gate partials.
 */
template <int T, int G, int R, bool Inject = false>
__global__ __launch_bounds__(kDownThreads) void down(
    const half* __restrict__ x,
    const half* __restrict__ w,
    half* __restrict__ o,
    const half* __restrict__ inject = nullptr,
    float* __restrict__ partials = nullptr) {
  static_assert(kWide == kDownThreads * kVecs * 8 && kLowRank % R == 0);
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const int m0 = blockIdx.x % G * T, chunk = blockIdx.x / G, row0 = chunk * R;
  if constexpr (Inject) {
    if (chunk >= kLowRank / R) {
      const int branch = chunk - kLowRank / R;
      float sum[T] = {};
#pragma unroll
      for (int j = 0; j < kVecs; ++j) {
        const int k = (threadIdx.x + j * kDownThreads) * 8;
        float wf[8];
        to_floats(load16(inject + branch * kWide + k), wf);
#pragma unroll
        for (int t = 0; t < T; ++t) {
          float xf[8];
          to_floats(load16(x + (m0 + t) * kWide + k), xf);
#pragma unroll
          for (int i = 0; i < 4; ++i)
            sum[t] += fmaf(xf[2 * i], wf[2 * i], xf[2 * i + 1] * wf[2 * i + 1]);
        }
      }
#pragma unroll
      for (int t = 0; t < T; ++t) {
#pragma unroll
        for (int d = 16; d > 0; d /= 2)
          sum[t] += __shfl_xor_sync(0xffffffff, sum[t], d);
        if (lane == 0) partials[((m0 + t) * 8 + warp) * 4 + branch] = sum[t];
      }
      return;
    }
  }
  const half* wp = w + row0 * kWide + threadIdx.x * 8;
  int4 raw[kVecs];
#pragma unroll
  for (int j = 0; j < kVecs; ++j)
    raw[j] = load16(wp + j * kDownThreads * 8);
  float xf[T][kVecs][8];
#pragma unroll
  for (int t = 0; t < T; ++t)
#pragma unroll
    for (int j = 0; j < kVecs; ++j)
      to_floats(load16(x + (m0 + t) * kWide + (threadIdx.x + j * kDownThreads) * 8), xf[t][j]);
  __shared__ float part[R][T][kDownThreads / 32];
#pragma unroll
  for (int r = 0; r < R; ++r) {
    float acc[T][8] = {};
    dot_chains<T>(xf, raw, acc);
    if (r + 1 < R) {
#pragma unroll
      for (int j = 0; j < kVecs; ++j)
        raw[j] = load16(wp + (r + 1) * kWide + j * kDownThreads * 8);
    }
#pragma unroll
    for (int t = 0; t < T; ++t) {
      float sum = 0;
#pragma unroll
      for (int i = 0; i < 8; ++i)
        sum += acc[t][i];
#pragma unroll
      for (int d = 16; d > 0; d /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, d);
      if (lane == 0) part[r][t][warp] = sum;
    }
  }
  __syncthreads();
  if (threadIdx.x < R * T) {
    const int r = threadIdx.x / T, t = threadIdx.x % T;
    float sum = 0;
#pragma unroll
    for (int i = 0; i < kDownThreads / 32; ++i)
      sum += part[r][t][i];
    sum = __half2float(__float2half_rn(sum)) * .25f;
    o[(m0 + t) * kLowRank + row0 + r] = __float2half_rn(sum / (1.f + __expf(-sum)));
  }
}

/**
 * \brief Up projection, gate and branch mean:
 *   o[m, h] = mean over 4 branches b of x[m, b * 2560 + h] * sigmoid(fp16(dot(a[m], w[b * 2560 + h]))).
 *
 * A warp computes H consecutive hidden coordinates for T of the G * T token
 * rows; its four 8-lane groups are the four branches. Lane group member s owns
 * vectors s, s + 8, ... of the kLowRank axis and keeps its slice of the CTA's
 * activation rows in FP32 registers. A dot product is summed as: 8 chains per
 * lane over its 5 vectors, the 8 chains in index order, then a pairwise tree
 * over the 8 lanes. The branches are added in index order.
 *
 * \tparam T Token rows per CTA.
 * \tparam G Token row groups; the kernel handles G * T token rows.
 * \tparam H Hidden coordinates per warp; the grid is G * kHidden / (4 * H).
 */
template <int T, int G, int H>
__global__ __launch_bounds__(kUpThreads) void up(
    const half* __restrict__ a, const half* __restrict__ x, const half* __restrict__ w, half* __restrict__ o) {
  static_assert(kLowRank == 8 * kVecs * 8 && kHidden % (kUpThreads / 32 * H) == 0);
  const int lane = threadIdx.x % 32, sub = lane % 8, branch = lane / 8;
  const int m0 = blockIdx.x % G * T;
  const int h0 = (blockIdx.x / G * (kUpThreads / 32) + threadIdx.x / 32) * H;
  const half* wrow = w + (branch * kHidden + h0) * kLowRank + sub * 8;
  int4 raw[kVecs];
#pragma unroll
  for (int j = 0; j < kVecs; ++j)
    raw[j] = load16(wrow + j * 64);
  float af[T][kVecs][8];
#pragma unroll
  for (int t = 0; t < T; ++t)
#pragma unroll
    for (int j = 0; j < kVecs; ++j)
      to_floats(load16(a + (m0 + t) * kLowRank + (sub + j * 8) * 8), af[t][j]);
#pragma unroll
  for (int q = 0; q < H; ++q) {
    const int h = h0 + q, row = branch * kHidden + h;
    half xh[T];
#pragma unroll
    for (int t = 0; t < T; ++t)
      xh[t] = x[(m0 + t) * kWide + row];
    float acc[T][8] = {};
    dot_chains<T>(af, raw, acc);
    if (q + 1 < H) {
#pragma unroll
      for (int j = 0; j < kVecs; ++j)
        raw[j] = load16(wrow + (q + 1) * kLowRank + j * 64);
    }
#pragma unroll
    for (int t = 0; t < T; ++t) {
      float v = 0;
#pragma unroll
      for (int i = 0; i < 8; ++i)
        v += acc[t][i];
#pragma unroll
      for (int d = 4; d > 0; d /= 2)
        v += __shfl_down_sync(0xffffffff, v, d, 8);
      v = __half2float(__float2half_rn(v));
      v = __half2float(xh[t]) / (1.f + __expf(-v));
      float b1 = __shfl_sync(0xffffffff, v, 8), b2 = __shfl_sync(0xffffffff, v, 16),
            b3 = __shfl_sync(0xffffffff, v, 24);
      if (lane == 0) o[(m0 + t) * kHidden + h] = __float2half_rn((((v + b1) + b2) + b3) * .25f);
    }
  }
}

// Launch shapes per token row count M, measured on V100 under CUDA-graph
// replay with different weights on every call; re-tune when the kernels
// change. The results do not depend on them.
template <int M>
constexpr int kTokensPerCta = M == 4 ? 2 : 1;
template <int M>
constexpr int kDownRows = M == 1 ? 2 : 4;
template <int M>
constexpr int kUpCoords = M == 1 ? 2 : 4;
template <int M>
constexpr int kDownGrid = M / kTokensPerCta<M> * (kLowRank / kDownRows<M>);
template <int M>
constexpr int kUpGrid = M / kTokensPerCta<M> * (kHidden / (kUpThreads / 32 * kUpCoords<M>));

template <int M>
void down_run(tvm::ffi::TensorView x, tvm::ffi::TensorView w, tvm::ffi::TensorView o) {
  static_assert(M == 1 || M == 2 || M == 4);
  using namespace host;
  auto dev = SymbolicDevice{};
  dev.set_options<kDLCUDA>();
  TensorMatcher({M, kWide}).with_dtype<half>().with_device(dev).verify(x);
  TensorMatcher({kLowRank, kWide}).with_dtype<half>().with_device(dev).verify(w);
  TensorMatcher({M, kLowRank}).with_dtype<half>().with_device(dev).verify(o);
  RuntimeCheck(
      reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0,
      "Input and weight must be 16-byte aligned");
  LaunchKernel(kDownGrid<M>, kDownThreads, dev.unwrap())(
      down<kTokensPerCta<M>, M / kTokensPerCta<M>, kDownRows<M>>,
      static_cast<const half*>(x.data_ptr()),
      static_cast<const half*>(w.data_ptr()),
      static_cast<half*>(o.data_ptr()),
      static_cast<const half*>(nullptr),
      static_cast<float*>(nullptr));
}
template <int M>
void up_run(tvm::ffi::TensorView a, tvm::ffi::TensorView x, tvm::ffi::TensorView w, tvm::ffi::TensorView o) {
  static_assert(M == 1 || M == 2 || M == 4);
  using namespace host;
  auto dev = SymbolicDevice{};
  dev.set_options<kDLCUDA>();
  TensorMatcher({M, kLowRank}).with_dtype<half>().with_device(dev).verify(a);
  TensorMatcher({M, kWide}).with_dtype<half>().with_device(dev).verify(x);
  TensorMatcher({kWide, kLowRank}).with_dtype<half>().with_device(dev).verify(w);
  TensorMatcher({M, kHidden}).with_dtype<half>().with_device(dev).verify(o);
  RuntimeCheck(
      reinterpret_cast<uintptr_t>(a.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0,
      "Activation and weight must be 16-byte aligned");
  LaunchKernel(kUpGrid<M>, kUpThreads, dev.unwrap())(
      up<kTokensPerCta<M>, M / kTokensPerCta<M>, kUpCoords<M>>,
      static_cast<const half*>(a.data_ptr()),
      static_cast<const half*>(x.data_ptr()),
      static_cast<const half*>(w.data_ptr()),
      static_cast<half*>(o.data_ptr()));
}
}  // namespace sglang::sm70_hc_batch
