// SPDX-License-Identifier: Apache-2.0
// SenseNova-Loop mHC router and merge on Volta: 2 residual streams of 4096, an
// 8-wide router (pre 2, post 2, 2x2 mix) over the flattened streams.
// The RMS scale and the 8 dots accumulate in fp64 and the Sinkhorn runs in fp64;
// pre/post/mix round once to fp32. The fp16 branch and merge round like the
// torch reference (fp32 sums of exact fp16 products, one fp16 rounding per op).
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <sgl_kernel/ptx/barrier.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang::sm70_sensenova_hc {

inline constexpr int kStreams = 2;
inline constexpr int kHidden = 4096;
inline constexpr int kFlat = kStreams * kHidden;
// pre[2], post[2], mix[2][2] row-major.
inline constexpr int kLogits = 2 * kStreams + kStreams * kStreams;
inline constexpr int kMix = kStreams * kStreams;
// The 8 dots with the router weight, then the sum of squares.
inline constexpr int kAcc = kLogits + 1;

// partial_kernel: one thread per hidden column (both streams), kSplits blocks
// across the columns. A pass reduces kTile tokens (kTile * kAcc values, one per
// lane); a block walks kTileLoops passes.
inline constexpr int kSplitThreads = 128;
inline constexpr int kSplitWarps = kSplitThreads / 32;
inline constexpr int kSplits = kHidden / kSplitThreads;
inline constexpr int kTile = 32 / kAcc;
inline constexpr int kTileLoops = 16;
inline constexpr int kTokensPerBlock = kTile * kTileLoops;

// finish_kernel: 16 warps combine one fp16 vector of 8 each (the launch shape of
// flashinfer's RMSNormKernel<8, half> at d=4096), one more warp runs the Sinkhorn.
inline constexpr int kVec = 8;
inline constexpr int kCombineThreads = kHidden / kVec;
inline constexpr int kCombineWarps = kCombineThreads / 32;
inline constexpr int kFinishThreads = kCombineThreads + 32;
// Named barrier of the combine warps; 0 is __syncthreads.
inline constexpr uint32_t kCombineBarrier = 1;

inline constexpr int kMergeThreads = 64;
inline constexpr int kMergeBlocks = kHidden / kVec / kMergeThreads;

static_assert(kHidden % kSplitThreads == 0);
static_assert(kTile >= 1 && kTile * kAcc <= 32, "a pass reduces one value per lane");
static_assert(kCombineWarps <= 32, "the cross-warp norm sum runs on one warp");
static_assert(kHidden % (kVec * kMergeThreads) == 0);

using hvec_t = device::AlignedVector<fp16_t, kVec>;

SGL_DEVICE float round_fp16(float x) {
  return static_cast<float>(DTypeTrait<fp16_t>::from(x));
}

// torch's fp16 bmm over 2 terms: the exact fp16 products summed into an fp32
// accumulator that starts at +0 (so an all -0 sum comes out +0).
SGL_DEVICE float dot2(float a0, float b0, float a1, float b1) {
  return __fadd_rn(__fadd_rn(0.0f, __fmul_rn(a0, b0)), __fmul_rn(a1, b1));
}

// mix16 @ streams + post16 * branch for one output stream: torch's fp16 bmm,
// fp16 product, fp16 add.
SGL_DEVICE fp16_t merge_value(float m_a, float m_b, float p, float r_a, float r_b, float x) {
  const float mixed = round_fp16(dot2(m_a, r_a, m_b, r_b));
  const float update = round_fp16(p * x);
  return DTypeTrait<fp16_t>::from(mixed + update);
}

struct MergeCoeffs {
  float m00, m01, m10, m11, p0, p1;
};

SGL_DEVICE MergeCoeffs load_merge_coeffs(const fp32_t* __restrict__ mix, const fp32_t* __restrict__ post, uint32_t t) {
  const fp32_t* m = mix + static_cast<int64_t>(t) * kMix;
  const fp32_t* p = post + static_cast<int64_t>(t) * kStreams;
  return {round_fp16(m[0]), round_fp16(m[1]), round_fp16(m[2]), round_fp16(m[3]), round_fp16(p[0]), round_fp16(p[1])};
}

/// One step of warp_sum_scatter: keep the half of v this lane owns, add the
/// partner's copy of it, and hand the partner the other half.
template <int kN>
SGL_DEVICE void sum_scatter_step(const double (&v)[kN], double (&out)[kN / 2], uint32_t lane) {
  constexpr int kHalf = kN / 2;
  const bool upper = (lane & kHalf) != 0;
#pragma unroll
  for (int i = 0; i < kHalf; ++i) {
    const double keep = upper ? v[i + kHalf] : v[i];
    const double send = upper ? v[i] : v[i + kHalf];
    out[i] = keep + __shfl_xor_sync(0xffffffffu, send, kHalf);
  }
}

/// Warp sum of each value, returned on lane j for value j. Every value is summed
/// over the same xor-butterfly tree, so a token's sums do not depend on its slot.
template <int kValues>
SGL_DEVICE double warp_sum_scatter(const double (&acc)[kValues], uint32_t lane) {
  static_assert(kValues <= 32);
  double v32[32], v16[16], v8[8], v4[4], v2[2], v1[1];
#pragma unroll
  for (int j = 0; j < 32; ++j) {
    v32[j] = j < kValues ? acc[j] : 0.0;
  }
  sum_scatter_step(v32, v16, lane);
  sum_scatter_step(v16, v8, lane);
  sum_scatter_step(v8, v4, lane);
  sum_scatter_step(v4, v2, lane);
  sum_scatter_step(v2, v1, lane);
  return v1[0];
}

/// Grid (kSplits, ceil(T / kTokensPerBlock)). partials[t, s, :] over hidden
/// columns [s * 128, (s + 1) * 128) of both streams: the 8 dots with the router
/// weight, then the sum of squares, in fp64 (fp16 x fp32 products are exact).
/// A token's partials come out the same whatever T or its tile slot.
/// With kMerge the streams are first merged (merge_kernel) and written to merged.
template <bool kMerge>
__global__ __launch_bounds__(kSplitThreads) void partial_kernel(
    double* __restrict__ partials,
    fp16_t* __restrict__ merged,
    const fp16_t* __restrict__ residual,
    const fp16_t* __restrict__ branch,
    const fp32_t* __restrict__ mix,
    const fp32_t* __restrict__ post,
    const fp32_t* __restrict__ weight,
    uint32_t num_tokens) {
  const uint32_t s = blockIdx.x;
  const uint32_t h = s * kSplitThreads + threadIdx.x;
  const uint32_t lane = threadIdx.x & 31;
  const uint32_t warp = threadIdx.x >> 5;
  double w0[kLogits], w1[kLogits];
#pragma unroll
  for (int i = 0; i < kLogits; ++i) {
    w0[i] = static_cast<double>(weight[static_cast<int64_t>(i) * kFlat + h]);
    w1[i] = static_cast<double>(weight[static_cast<int64_t>(i) * kFlat + kHidden + h]);
  }
  __shared__ double s_part[kSplitWarps][32];
  const uint32_t begin = blockIdx.y * kTokensPerBlock;
  const uint32_t end = min(begin + kTokensPerBlock, num_tokens);
  for (uint32_t t0 = begin; t0 < end; t0 += kTile) {
    double acc[kTile * kAcc];
#pragma unroll
    for (int u = 0; u < kTile; ++u) {
      const uint32_t t = t0 + u;
      double x0 = 0.0, x1 = 0.0;
      if (t < end) {
        const int64_t row = static_cast<int64_t>(t) * kFlat;
        float r0 = static_cast<float>(residual[row + h]);
        float r1 = static_cast<float>(residual[row + kHidden + h]);
        if constexpr (kMerge) {
          const MergeCoeffs c = load_merge_coeffs(mix, post, t);
          const float x = static_cast<float>(branch[static_cast<int64_t>(t) * kHidden + h]);
          const fp16_t y0 = merge_value(c.m00, c.m01, c.p0, r0, r1, x);
          const fp16_t y1 = merge_value(c.m10, c.m11, c.p1, r0, r1, x);
          merged[row + h] = y0;
          merged[row + kHidden + h] = y1;
          r0 = static_cast<float>(y0);
          r1 = static_cast<float>(y1);
        }
        x0 = static_cast<double>(r0);
        x1 = static_cast<double>(r1);
      }
#pragma unroll
      for (int i = 0; i < kLogits; ++i) {
        acc[u * kAcc + i] = fma(x1, w1[i], x0 * w0[i]);
      }
      acc[u * kAcc + kLogits] = fma(x1, x1, x0 * x0);
    }
    const double sum = warp_sum_scatter(acc, lane);
    if (lane < kTile * kAcc) {
      s_part[warp][lane] = sum;
    }
    __syncthreads();
    if (threadIdx.x < kTile * kAcc) {
      double total = s_part[0][threadIdx.x];
#pragma unroll
      for (int w = 1; w < kSplitWarps; ++w) {
        total += s_part[w][threadIdx.x];
      }
      const uint32_t t = t0 + threadIdx.x / kAcc;
      if (t < end) {
        partials[(static_cast<int64_t>(t) * kSplits + s) * kAcc + threadIdx.x % kAcc] = total;
      }
    }
    __syncthreads();
  }
}

// Row logit gap above which the closed form's intermediates could overflow fp64
// (they stay below exp(2 * gap + 1)); such tokens run the log-space iteration.
inline constexpr double kClosedFormMaxGap = 300.0;

SGL_DEVICE double logaddexp(double x, double y) {
  const double m = fmax(x, y);
  return m + log1p(exp(-fabs(x - y)));
}

/// exp of `iterations` log-space row-then-column normalizations of the 2x2 logits
/// l (row-major), in closed form.
///
/// Normalization commutes with exp. A row normalization leaves the rows (u, 1)
/// and (v, 1) up to scale, u = exp(l00 - l01) and v = exp(l10 - l11) at first.
/// Row and column scalings keep u / v = K = exp(dl), dl = (l00 - l01) - (l10 - l11).
/// With k = sqrt(K), u = k z and v = z / k, a column then row normalization maps
/// z to (c z + 2) / (2 z + c), c = k + 1/k: a Moebius map with matrix
/// [[c, 2], [2, c]], eigenvectors (1, 1) and (1, -1), eigenvalues c + 2, c - 2.
/// n maps give z_n = ((1 + rho) z + 1 - rho) / ((1 - rho) z + 1 + rho) with
/// rho = ((c - 2) / (c + 2))^n = tanh(|dl| / 4)^(2n). The result is the last
/// column normalization of the rows (u, 1) / (1 + u), (v, 1) / (1 + v) after
/// n = iterations - 1 maps:
///   [[u (1 + v), (1 + v)], [v (1 + u), (1 + u)]] / [u + v + 2uv, 2 + u + v] by column.
SGL_DEVICE void sinkhorn(const double* __restrict__ l, int32_t iterations, fp32_t* __restrict__ mix) {
  const double row0 = l[0] - l[1];
  const double row1 = l[2] - l[3];
  if (fmax(fabs(row0), fabs(row1)) > kClosedFormMaxGap) {
    double a = l[0], b = l[1], c = l[2], d = l[3];
    for (int32_t i = 0; i < iterations; ++i) {
      const double r0 = logaddexp(a, b);
      const double r1 = logaddexp(c, d);
      a -= r0;
      b -= r0;
      c -= r1;
      d -= r1;
      const double c0 = logaddexp(a, c);
      const double c1 = logaddexp(b, d);
      a -= c0;
      c -= c0;
      b -= c1;
      d -= c1;
    }
    mix[0] = static_cast<float>(exp(a));
    mix[1] = static_cast<float>(exp(b));
    mix[2] = static_cast<float>(exp(c));
    mix[3] = static_cast<float>(exp(d));
    return;
  }
  const double log_k = 0.5 * (row0 - row1);
  double rho = 1.0, one_minus_rho = 0.0;
  if (iterations > 1) {
    // log tanh(|dl| / 4) = log1p(-2 / (exp(|dl| / 2) + 1)); -inf at dl = 0.
    const double log_tanh = log1p(-2.0 / (exp(fabs(log_k)) + 1.0));
    const double log_rho = 2.0 * (iterations - 1) * log_tanh;
    rho = exp(log_rho);
    one_minus_rho = -expm1(log_rho);
  }
  const double z0 = exp(0.5 * (row0 + row1));
  const double z = ((1.0 + rho) * z0 + one_minus_rho) / (one_minus_rho * z0 + (1.0 + rho));
  const double u = z * exp(log_k);
  const double v = z * exp(-log_k);
  const double col0 = 1.0 / (u + v + 2.0 * u * v);
  const double col1 = 1.0 / (2.0 + u + v);
  mix[0] = static_cast<float>(u * (1.0 + v) * col0);
  mix[1] = static_cast<float>((1.0 + v) * col1);
  mix[2] = static_cast<float>(v * (1.0 + u) * col0);
  mix[3] = static_cast<float>((1.0 + u) * col1);
}

SGL_DEVICE float rsqrt_approx_ftz(float x) {
  float y;
  asm volatile("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

/// Grid (T). Reduces partial_kernel's [T, kSplits, kAcc] in split order to the
/// logits (fp64), writes pre/post/mix and branch = sum_s fp16(pre[s]) * streams[s].
/// With kNorm the branch comes out as sgl_kernel gemma_rmsnorm(branch, w, eps),
/// bitwise: per-thread fma sum of squares, xor butterfly within and across the 16
/// warps, IEEE divide by d, rsqrt.approx.ftz, then (x * rcp) * (1 + w).
template <bool kNorm>
__global__ __launch_bounds__(kFinishThreads) void finish_kernel(
    fp16_t* __restrict__ out,
    fp32_t* __restrict__ pre_out,
    fp32_t* __restrict__ post_out,
    fp32_t* __restrict__ mix_out,
    const double* __restrict__ partials,
    const fp16_t* __restrict__ residual,
    const fp32_t* __restrict__ bias,
    const fp16_t* __restrict__ norm_weight,
    int32_t iterations,
    double rms_eps,
    float norm_eps) {
  const uint32_t t = blockIdx.x;
  const uint32_t lane = threadIdx.x & 31;
  const uint32_t warp = threadIdx.x >> 5;
  __shared__ double s_logit[kLogits];
  __shared__ float s_pre[kStreams];
  __shared__ float s_sq[kCombineWarps];

  // The combine's loads do not depend on the logits; issue them first.
  const fp16_t* res_row = residual + static_cast<int64_t>(t) * kFlat;
  hvec_t r0, r1, w;
  if (warp < kCombineWarps) {
    r0.load(res_row, threadIdx.x);
    r1.load(res_row + kHidden, threadIdx.x);
    if constexpr (kNorm) {
      w.load(norm_weight, threadIdx.x);
    }
  }

  if (warp == 0) {
    double v = 0.0;
    if (lane < kAcc) {
      const double* p = partials + static_cast<int64_t>(t) * kSplits * kAcc + lane;
#pragma unroll
      for (int s = 0; s < kSplits; ++s) {
        v += p[s * kAcc];
      }
    }
    const double sum_sq = __shfl_sync(0xffffffffu, v, kLogits);
    if (lane < kLogits) {
      const double scale = 1.0 / sqrt(sum_sq / kFlat + rms_eps);
      const double logit = fma(v, scale, static_cast<double>(bias[lane]));
      s_logit[lane] = logit;
      if (lane < 2 * kStreams) {
        const double gate = 1.0 / (1.0 + exp(-logit));
        if (lane < kStreams) {
          const float p = static_cast<float>(gate);
          s_pre[lane] = p;
          pre_out[static_cast<int64_t>(t) * kStreams + lane] = p;
        } else {
          post_out[static_cast<int64_t>(t) * kStreams + lane - kStreams] = static_cast<float>(2.0 * gate);
        }
      }
    }
  }
  __syncthreads();

  if (warp == kCombineWarps) {
    if (lane == 0) {
      sinkhorn(s_logit + 2 * kStreams, iterations, mix_out + static_cast<int64_t>(t) * kMix);
    }
    return;
  }

  const float p0 = round_fp16(s_pre[0]);
  const float p1 = round_fp16(s_pre[1]);
  hvec_t x;
#pragma unroll
  for (int i = 0; i < kVec; ++i) {
    x[i] = DTypeTrait<fp16_t>::from(dot2(p0, static_cast<float>(r0[i]), p1, static_cast<float>(r1[i])));
  }
  fp16_t* out_row = out + static_cast<int64_t>(t) * kHidden;
  if constexpr (!kNorm) {
    x.store(out_row, threadIdx.x);
  } else {
    float sum_sq = 0.0f;
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
      const float v = static_cast<float>(x[i]);
      sum_sq = __fmaf_rn(v, v, sum_sq);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      sum_sq = __fadd_rn(sum_sq, __shfl_xor_sync(0xffffffffu, sum_sq, off));
    }
    s_sq[warp] = sum_sq;
    device::ptx::bar_sync(kCombineBarrier, kCombineThreads);
    if (warp == 0) {
      sum_sq = lane < kCombineWarps ? s_sq[lane] : 0.0f;
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) {
        sum_sq = __fadd_rn(sum_sq, __shfl_xor_sync(0xffffffffu, sum_sq, off));
      }
      if (lane == 0) {
        s_sq[0] = sum_sq;
      }
    }
    device::ptx::bar_sync(kCombineBarrier, kCombineThreads);
    const float rcp = rsqrt_approx_ftz(__fadd_rn(__fdiv_rn(s_sq[0], static_cast<float>(kHidden)), norm_eps));
    hvec_t y;
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
      const float scaled = __fmul_rn(static_cast<float>(x[i]), rcp);
      y[i] = DTypeTrait<fp16_t>::from(__fmul_rn(scaled, __fadd_rn(1.0f, static_cast<float>(w[i]))));
    }
    y.store(out_row, threadIdx.x);
  }
}

/// out[t, j, h] = fp16(mix16[t, j, :] @ residual[t, :, h]) + fp16(post16[t, j] * branch[t, h]).
/// Grid (kMergeBlocks, T); one fp16 vector of both streams per thread.
__global__ __launch_bounds__(kMergeThreads) void merge_kernel(
    fp16_t* __restrict__ out,
    const fp16_t* __restrict__ branch,
    const fp16_t* __restrict__ residual,
    const fp32_t* __restrict__ mix,
    const fp32_t* __restrict__ post) {
  const uint32_t t = blockIdx.y;
  const uint32_t vi = blockIdx.x * kMergeThreads + threadIdx.x;
  const int64_t row = static_cast<int64_t>(t) * kFlat;
  hvec_t xv, r0, r1, y0, y1;
  xv.load(branch + static_cast<int64_t>(t) * kHidden, vi);
  r0.load(residual + row, vi);
  r1.load(residual + row + kHidden, vi);
  const MergeCoeffs c = load_merge_coeffs(mix, post, t);
#pragma unroll
  for (int i = 0; i < kVec; ++i) {
    const float a = static_cast<float>(r0[i]);
    const float b = static_cast<float>(r1[i]);
    const float x = static_cast<float>(xv[i]);
    y0[i] = merge_value(c.m00, c.m01, c.p0, a, b, x);
    y1[i] = merge_value(c.m10, c.m11, c.p1, a, b, x);
  }
  y0.store(out + row, vi);
  y1.store(out + row + kHidden, vi);
}

struct RouteOut {
  tvm::ffi::TensorView out, pre, post, mix, partials;
};

/// Checks the router outputs and inputs shared by route and merge_route; returns T.
inline uint32_t check_route(
    const RouteOut& o,
    tvm::ffi::TensorView residual,
    tvm::ffi::TensorView weight,
    tvm::ffi::TensorView bias,
    host::SymbolicDevice& device_) {
  using namespace host;
  SymbolicSize n_tokens = {"num_tokens"};
  TensorMatcher({n_tokens, kFlat})  //
      .with_dtype<fp16_t>()
      .with_device<kDLCUDA>(device_)
      .ensure_alignment(16)
      .verify(residual);
  TensorMatcher({n_tokens, kHidden})  //
      .with_dtype<fp16_t>()
      .with_device<kDLCUDA>(device_)
      .ensure_alignment(16)
      .verify(o.out);
  TensorMatcher({n_tokens, kStreams})  //
      .with_dtype<fp32_t>()
      .with_device<kDLCUDA>(device_)
      .verify(o.pre)
      .verify(o.post);
  TensorMatcher({n_tokens, kStreams, kStreams})  //
      .with_dtype<fp32_t>()
      .with_device<kDLCUDA>(device_)
      .verify(o.mix);
  TensorMatcher({n_tokens, kSplits, kAcc})  //
      .with_dtype<double>()
      .with_device<kDLCUDA>(device_)
      .verify(o.partials);
  TensorMatcher({kLogits, kFlat})  //
      .with_dtype<fp32_t>()
      .with_device<kDLCUDA>(device_)
      .verify(weight);
  TensorMatcher({kLogits})  //
      .with_dtype<fp32_t>()
      .with_device<kDLCUDA>(device_)
      .verify(bias);
  const uint32_t n = static_cast<uint32_t>(n_tokens.unwrap());
  CHECK_HOST(n > 0);
  return n;
}

inline const fp16_t* check_norm_weight(tvm::ffi::TensorView norm_weight, host::SymbolicDevice& device_) {
  using namespace host;
  TensorMatcher({kHidden})  //
      .with_dtype<fp16_t>()
      .with_device<kDLCUDA>(device_)
      .ensure_alignment(16)
      .verify(norm_weight);
  return static_cast<const fp16_t*>(norm_weight.data_ptr());
}

inline void launch_finish(
    const RouteOut& o,
    const fp16_t* residual,
    tvm::ffi::TensorView bias,
    const fp16_t* norm_weight,
    uint32_t n,
    int64_t iterations,
    double rms_eps,
    double norm_eps,
    DLDevice device) {
  using namespace host;
  CHECK_HOST(iterations >= 1);
  const auto kernel = norm_weight == nullptr ? finish_kernel<false> : finish_kernel<true>;
  LaunchKernel(n, kFinishThreads, device)(
      kernel,
      static_cast<fp16_t*>(o.out.data_ptr()),
      static_cast<fp32_t*>(o.pre.data_ptr()),
      static_cast<fp32_t*>(o.post.data_ptr()),
      static_cast<fp32_t*>(o.mix.data_ptr()),
      static_cast<const double*>(o.partials.data_ptr()),
      residual,
      static_cast<const fp32_t*>(bias.data_ptr()),
      norm_weight,
      static_cast<int32_t>(iterations),
      rms_eps,
      static_cast<float>(norm_eps));
}

inline void launch_route(
    const RouteOut& o,
    tvm::ffi::TensorView residual,
    tvm::ffi::TensorView weight,
    tvm::ffi::TensorView bias,
    const tvm::ffi::TensorView* norm_weight,
    int64_t iterations,
    double rms_eps,
    double norm_eps) {
  using namespace host;
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();
  const uint32_t n = check_route(o, residual, weight, bias, device_);
  const fp16_t* w = norm_weight == nullptr ? nullptr : check_norm_weight(*norm_weight, device_);
  const DLDevice device = device_.unwrap();
  LaunchKernel(dim3(kSplits, div_ceil(n, static_cast<uint32_t>(kTokensPerBlock))), kSplitThreads, device)(
      partial_kernel<false>,
      static_cast<double*>(o.partials.data_ptr()),
      nullptr,
      static_cast<const fp16_t*>(residual.data_ptr()),
      nullptr,
      nullptr,
      nullptr,
      static_cast<const fp32_t*>(weight.data_ptr()),
      n);
  launch_finish(o, static_cast<const fp16_t*>(residual.data_ptr()), bias, w, n, iterations, rms_eps, norm_eps, device);
}

inline void launch_merge_route(
    const RouteOut& o,
    tvm::ffi::TensorView merged,
    tvm::ffi::TensorView branch,
    tvm::ffi::TensorView residual,
    tvm::ffi::TensorView mix,
    tvm::ffi::TensorView post,
    tvm::ffi::TensorView weight,
    tvm::ffi::TensorView bias,
    const tvm::ffi::TensorView* norm_weight,
    int64_t iterations,
    double rms_eps,
    double norm_eps) {
  using namespace host;
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();
  // The router reads the merged streams.
  const uint32_t n = check_route(o, merged, weight, bias, device_);
  TensorMatcher({n, kFlat})  //
      .with_dtype<fp16_t>()
      .with_device<kDLCUDA>(device_)
      .verify(residual);
  TensorMatcher({n, kHidden})  //
      .with_dtype<fp16_t>()
      .with_device<kDLCUDA>(device_)
      .verify(branch);
  TensorMatcher({n, kStreams, kStreams})  //
      .with_dtype<fp32_t>()
      .with_device<kDLCUDA>(device_)
      .verify(mix);
  TensorMatcher({n, kStreams})  //
      .with_dtype<fp32_t>()
      .with_device<kDLCUDA>(device_)
      .verify(post);
  CHECK_HOST(merged.data_ptr() != residual.data_ptr()) << "merge_route cannot merge in place";
  const fp16_t* w = norm_weight == nullptr ? nullptr : check_norm_weight(*norm_weight, device_);
  const DLDevice device = device_.unwrap();
  LaunchKernel(dim3(kSplits, div_ceil(n, static_cast<uint32_t>(kTokensPerBlock))), kSplitThreads, device)(
      partial_kernel<true>,
      static_cast<double*>(o.partials.data_ptr()),
      static_cast<fp16_t*>(merged.data_ptr()),
      static_cast<const fp16_t*>(residual.data_ptr()),
      static_cast<const fp16_t*>(branch.data_ptr()),
      static_cast<const fp32_t*>(mix.data_ptr()),
      static_cast<const fp32_t*>(post.data_ptr()),
      static_cast<const fp32_t*>(weight.data_ptr()),
      n);
  launch_finish(o, static_cast<const fp16_t*>(merged.data_ptr()), bias, w, n, iterations, rms_eps, norm_eps, device);
}

/// Router on residual [T, 8192] fp16 with weight [8, 8192] and bias [8] fp32.
/// Writes out [T, 4096] fp16 (the branch), pre/post [T, 2] and mix [T, 2, 2] fp32;
/// partials [T, 32, 9] fp64 is scratch.
void route(
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView pre,
    tvm::ffi::TensorView post,
    tvm::ffi::TensorView mix,
    tvm::ffi::TensorView partials,
    tvm::ffi::TensorView residual,
    tvm::ffi::TensorView weight,
    tvm::ffi::TensorView bias,
    int64_t iterations,
    double rms_eps) {
  launch_route({out, pre, post, mix, partials}, residual, weight, bias, nullptr, iterations, rms_eps, 0.0);
}

/// route with out = gemma_rmsnorm(branch, norm_weight [4096] fp16, norm_eps).
void route_norm(
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView pre,
    tvm::ffi::TensorView post,
    tvm::ffi::TensorView mix,
    tvm::ffi::TensorView partials,
    tvm::ffi::TensorView residual,
    tvm::ffi::TensorView weight,
    tvm::ffi::TensorView bias,
    tvm::ffi::TensorView norm_weight,
    int64_t iterations,
    double rms_eps,
    double norm_eps) {
  launch_route({out, pre, post, mix, partials}, residual, weight, bias, &norm_weight, iterations, rms_eps, norm_eps);
}

/// merge(branch_in, residual, mix_in, post_in) into merged [T, 8192], then route on it.
void merge_route(
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView pre,
    tvm::ffi::TensorView post_out,
    tvm::ffi::TensorView mix_out,
    tvm::ffi::TensorView partials,
    tvm::ffi::TensorView merged,
    tvm::ffi::TensorView branch_in,
    tvm::ffi::TensorView residual,
    tvm::ffi::TensorView mix_in,
    tvm::ffi::TensorView post_in,
    tvm::ffi::TensorView weight,
    tvm::ffi::TensorView bias,
    int64_t iterations,
    double rms_eps) {
  launch_merge_route(
      {out, pre, post_out, mix_out, partials},
      merged,
      branch_in,
      residual,
      mix_in,
      post_in,
      weight,
      bias,
      nullptr,
      iterations,
      rms_eps,
      0.0);
}

void merge_route_norm(
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView pre,
    tvm::ffi::TensorView post_out,
    tvm::ffi::TensorView mix_out,
    tvm::ffi::TensorView partials,
    tvm::ffi::TensorView merged,
    tvm::ffi::TensorView branch_in,
    tvm::ffi::TensorView residual,
    tvm::ffi::TensorView mix_in,
    tvm::ffi::TensorView post_in,
    tvm::ffi::TensorView weight,
    tvm::ffi::TensorView bias,
    tvm::ffi::TensorView norm_weight,
    int64_t iterations,
    double rms_eps,
    double norm_eps) {
  launch_merge_route(
      {out, pre, post_out, mix_out, partials},
      merged,
      branch_in,
      residual,
      mix_in,
      post_in,
      weight,
      bias,
      &norm_weight,
      iterations,
      rms_eps,
      norm_eps);
}

/// out [T, 8192] = merge(branch [T, 4096], residual [T, 8192], mix [T, 2, 2], post [T, 2]).
void merge(
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView branch,
    tvm::ffi::TensorView residual,
    tvm::ffi::TensorView mix,
    tvm::ffi::TensorView post) {
  using namespace host;
  SymbolicSize n_tokens = {"num_tokens"};
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();
  TensorMatcher({n_tokens, kFlat})  //
      .with_dtype<fp16_t>()
      .with_device<kDLCUDA>(device_)
      .ensure_alignment(16)
      .verify(out)
      .verify(residual);
  TensorMatcher({n_tokens, kHidden})  //
      .with_dtype<fp16_t>()
      .with_device<kDLCUDA>(device_)
      .ensure_alignment(16)
      .verify(branch);
  TensorMatcher({n_tokens, kStreams, kStreams})  //
      .with_dtype<fp32_t>()
      .with_device<kDLCUDA>(device_)
      .verify(mix);
  TensorMatcher({n_tokens, kStreams})  //
      .with_dtype<fp32_t>()
      .with_device<kDLCUDA>(device_)
      .verify(post);
  const uint32_t n = static_cast<uint32_t>(n_tokens.unwrap());
  CHECK_HOST(n > 0);
  LaunchKernel(dim3(kMergeBlocks, n), kMergeThreads, device_.unwrap())(
      merge_kernel,
      static_cast<fp16_t*>(out.data_ptr()),
      static_cast<const fp16_t*>(branch.data_ptr()),
      static_cast<const fp16_t*>(residual.data_ptr()),
      static_cast<const fp32_t*>(mix.data_ptr()),
      static_cast<const fp32_t*>(post.data_ptr()));
}

}  // namespace sglang::sm70_sensenova_hc
