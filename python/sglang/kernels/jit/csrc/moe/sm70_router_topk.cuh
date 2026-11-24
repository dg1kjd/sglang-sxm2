// SPDX-License-Identifier: Apache-2.0
// Ungrouped sigmoid top-8 / top-16 routing on Volta, one warp per token.
// Reproduces _router_triton_kernel's SM70 arithmetic (ex2.approx, div.full, its
// sum order), so ids and weights match it bitwise for 257..512 experts.
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/warp.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang::sm70_router_topk {

inline constexpr int kWarp = 32;
using device::warp::kFullMask;
inline constexpr int kPerLane = 16;
inline constexpr int kMinExperts = 257;
inline constexpr int kMaxExperts = kWarp * kPerLane;

SGL_DEVICE float ex2_approx(float x) {
  float y;
  asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

SGL_DEVICE float div_full(float a, float b) {
  float y;
  asm("div.full.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b));
  return y;
}

// tl.sigmoid as Triton lowers it for SM70; 0x3FB8AA3B is its fp32 log2(e).
SGL_DEVICE float triton_sigmoid(float x) {
  const float t = __fmul_rn(__fsub_rn(0.0f, x), __int_as_float(0x3FB8AA3B));
  return div_full(1.0f, __fadd_rn(ex2_approx(t), 1.0f));
}

// Order-preserving float bits.
SGL_DEVICE uint32_t order_bits(float v) {
  const uint32_t b = __float_as_uint(v);
  return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

// Bitonic sort, descending. Inputs are never NaN or -0.0, so fmaxf/fminf
// (one FMNMX each; Volta has no integer min/max) order them exactly.
template <int kLen>
SGL_DEVICE void sort_desc(float (&k)[kLen]) {
  static_assert(kLen > 0 && (kLen & (kLen - 1)) == 0, "bitonic network");
#pragma unroll
  for (int size = 2; size <= kLen; size *= 2) {
#pragma unroll
    for (int stride = size / 2; stride > 0; stride /= 2) {
#pragma unroll
      for (int i = 0; i < kLen; ++i) {
        const int p = i ^ stride;
        if (p > i) {
          const float hi = fmaxf(k[i], k[p]);
          const float lo = fminf(k[i], k[p]);
          k[i] = (i & size) == 0 ? hi : lo;
          k[p] = (i & size) == 0 ? lo : hi;
        }
      }
    }
  }
}

// a := the largest kLen of a and b with repeats, descending; both come in descending.
template <int kLen>
SGL_DEVICE void merge_top(float (&a)[kLen], const float (&b)[kLen]) {
  static_assert(kLen > 0 && (kLen & (kLen - 1)) == 0, "bitonic network");
#pragma unroll
  for (int i = 0; i < kLen; ++i) {
    a[i] = fmaxf(a[i], b[kLen - 1 - i]);
  }
#pragma unroll
  for (int stride = kLen / 2; stride > 0; stride /= 2) {
#pragma unroll
    for (int i = 0; i < kLen; ++i) {
      const int p = i ^ stride;
      if (p > i) {
        const float hi = fmaxf(a[i], a[p]);
        a[p] = fminf(a[i], a[p]);
        a[i] = hi;
      }
    }
  }
}

// Sum of the selected weights (lane k holds slot k) in _router_triton_kernel's
// order: its [1, BLOCK_K] slot tile gives each lane 4 consecutive slots
// (sizePerThread [1, 4], which needs 16-byte aligned scores), added in order,
// then a butterfly across the BLOCK_K / 4 lanes. Every lane gets the sum.
template <int kTopK>
SGL_DEVICE float triton_slot_sum(float w, int lane) {
  static_assert(kTopK == 8 || kTopK == 16, "sum order read from Triton's PTX for BLOCK_K 8 and 16 only");
  constexpr int kLanes = kTopK / 4;
  const int first = (lane % kLanes) * 4;
  float sum = __shfl_sync(kFullMask, w, first);
#pragma unroll
  for (int j = 1; j < 4; ++j) {
    sum = __fadd_rn(sum, __shfl_sync(kFullMask, w, first + j));
  }
#pragma unroll
  for (int off = kLanes / 2; off > 0; off >>= 1) {
    sum = __fadd_rn(sum, __shfl_xor_sync(kFullMask, sum, off));
  }
  return sum;
}

template <int kTopK, int kSlots>
__global__ __launch_bounds__(kWarp) void route_kernel(
    float* __restrict__ weights,
    int32_t* __restrict__ ids,
    const float* __restrict__ scores,
    int64_t score_stride,
    const float* __restrict__ bias,
    int32_t num_experts,
    float scale,
    bool renormalize,
    bool apply_scale) {
  static_assert(kSlots <= 32, "picked holds one bit per slot");
  __shared__ float s_act[kSlots * kWarp];
  __shared__ float s_pick_val[kTopK];
  __shared__ int32_t s_pick_id[kTopK];
  __shared__ int32_t s_ranked_id[kTopK];
  const int64_t t = blockIdx.x;
  const int lane = static_cast<int>(threadIdx.x);
  const float* row = scores + t * score_stride;
  // Biased values; padding is -inf. The sigmoid is never -0.0, so neither is
  // its sum with the bias. This and the exact match with Triton need a build
  // that keeps subnormals (no flush-to-zero): Triton keeps them.
  float val[kSlots];
#pragma unroll
  for (int j = 0; j < kSlots; ++j) {
    const int e = lane + kWarp * j;
    const bool valid = e < num_experts;
    const float a = triton_sigmoid(valid ? row[e] : 0.0f);
    const float b = __fadd_rn(a, valid ? bias[e] : 0.0f);
    s_act[e] = a;
    val[j] = valid ? (b == b ? b : -1e30f) : -INFINITY;
  }

  // The kTopK largest values with repeats, descending: each lane sorts its
  // own, then a butterfly merges lane lists until every lane holds the warp's.
  float top[kTopK];
#pragma unroll
  for (int j = 0; j < kTopK; ++j) {
    top[j] = j < kSlots ? val[j] : -INFINITY;
  }
  sort_desc(top);
#pragma unroll
  for (int c = kTopK; c < kSlots; c += kTopK) {
    float chunk[kTopK];
#pragma unroll
    for (int j = 0; j < kTopK; ++j) {
      chunk[j] = c + j < kSlots ? val[c + j] : -INFINITY;
    }
    sort_desc(chunk);
    merge_top(top, chunk);
  }
  // Rolled: one decode launch runs this once, so an unrolled body is fetched
  // cold from memory on every layer.
#pragma unroll 1
  for (int off = kWarp / 2; off > 0; off >>= 1) {
    float other[kTopK];
#pragma unroll
    for (int j = 0; j < kTopK; ++j) {
      other[j] = __shfl_xor_sync(kFullMask, top[j], off);
    }
    merge_top(top, other);
  }

  // A cut of -inf takes padding and all live experts, so it never passes the
  // count check below.
  const float cut = top[kTopK - 1];
  bool distinct = true;
#pragma unroll
  for (int i = 0; i + 1 < kTopK; ++i) {
    distinct = distinct && top[i] > top[i + 1];
  }
  uint32_t picked = 0;
#pragma unroll
  for (int j = 0; j < kSlots; ++j) {
    picked |= (val[j] >= cut ? 1u : 0u) << j;
  }
  const int count = __popc(picked);
  const int upto = device::warp::inclusive_sum(count, static_cast<uint32_t>(lane));
  const int total = device::warp::broadcast(upto, kWarp - 1);

  int32_t my_id = 0;
  if (distinct && total == kTopK) {
    // No two candidates share a value, so sorting by value alone gives
    // Triton's ids and order.
    int slot = upto - count;
#pragma unroll
    for (int j = 0; j < kSlots; ++j) {
      if ((picked >> j) & 1u) {
        s_pick_val[slot] = val[j];
        s_pick_id[slot] = lane + kWarp * j;
        ++slot;
      }
    }
    __syncwarp();
    if (lane < kTopK) {
      const float mine = s_pick_val[lane];
      int rank = 0;
#pragma unroll
      for (int i = 0; i < kTopK; ++i) {
        rank += top[i] > mine ? 1 : 0;
      }
      s_ranked_id[rank] = s_pick_id[lane];
    }
    __syncwarp();
    my_id = s_ranked_id[lane % kTopK];
  } else {
    // Equal values among the candidates: pick one at a time; the 64-bit key
    // puts the lowest expert id first among equals, as Triton's argmax does.
    uint64_t key[kSlots];
#pragma unroll
    for (int j = 0; j < kSlots; ++j) {
      // Padding (0) and taken experts (cleared to 0) rank below every live key.
      const int e = lane + kWarp * j;
      key[j] = e < num_experts ? (static_cast<uint64_t>(order_bits(val[j])) << 32) | static_cast<uint32_t>(~e) : 0;
    }
#pragma unroll 1
    for (int k = 0; k < kTopK; ++k) {
      uint64_t best = key[0];
#pragma unroll
      for (int j = 1; j < kSlots; ++j) {
        best = key[j] > best ? key[j] : best;
      }
#pragma unroll
      for (int off = kWarp / 2; off > 0; off >>= 1) {
        const uint64_t other = __shfl_xor_sync(kFullMask, best, off);
        best = other > best ? other : best;
      }
#pragma unroll
      for (int j = 0; j < kSlots; ++j) {
        key[j] = key[j] == best ? 0 : key[j];
      }
      my_id = lane == k ? static_cast<int32_t>(~static_cast<uint32_t>(best)) : my_id;
    }
    __syncwarp();
  }
  const float my_w = s_act[my_id];

  float out = my_w;
  if (renormalize) {
    const float sum = triton_slot_sum<kTopK>(my_w, lane);
    out = div_full(my_w, sum > 0.0f ? sum : 1.0f);
  }
  if (apply_scale) {
    out = __fmul_rn(out, scale);
  }
  if (lane < kTopK) {
    weights[t * kTopK + lane] = out;
    ids[t * kTopK + lane] = my_id;
  }
}

template <int kTopK, int kSlots>
void launch(
    uint32_t num_tokens,
    DLDevice device,
    float* weights,
    int32_t* ids,
    const float* scores,
    int64_t score_stride,
    const float* bias,
    int32_t num_experts,
    float scale,
    bool renormalize,
    bool apply_scale) {
  if constexpr (kSlots < kPerLane) {
    if (num_experts <= kSlots * kWarp) {
      host::LaunchKernel(num_tokens, kWarp, device)(
          route_kernel<kTopK, kSlots>,
          weights,
          ids,
          scores,
          score_stride,
          bias,
          num_experts,
          scale,
          renormalize,
          apply_scale);
      return;
    }
    launch<kTopK, kSlots + 1>(
        num_tokens, device, weights, ids, scores, score_stride, bias, num_experts, scale, renormalize, apply_scale);
  } else {
    host::LaunchKernel(num_tokens, kWarp, device)(
        route_kernel<kTopK, kSlots>,
        weights,
        ids,
        scores,
        score_stride,
        bias,
        num_experts,
        scale,
        renormalize,
        apply_scale);
  }
}

/// \brief weights/ids [M, kTopK] from raw logits scores [M, N] (row stride free),
///        fp32 bias [N]; 257 <= N <= 512.
/// \tparam kTopK 8 or 16.
template <int kTopK>
void route(
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView ids,
    tvm::ffi::TensorView scores,
    tvm::ffi::TensorView bias,
    double scale,
    bool renormalize,
    bool apply_scale) {
  using namespace host;
  SymbolicSize m = {"num_tokens"};
  SymbolicSize n = {"num_experts"};
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();
  TensorMatcher({m, n}).with_dtype<fp32_t>().with_strides({-1, 1}).with_device<kDLCUDA>(device_).verify(scores);
  TensorMatcher({n}).with_dtype<fp32_t>().with_device<kDLCUDA>(device_).verify(bias);
  TensorMatcher({m, kTopK}).with_dtype<fp32_t>().with_device<kDLCUDA>(device_).verify(weights);
  TensorMatcher({m, kTopK}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(ids);
  const int64_t num_tokens = m.unwrap();
  const int64_t num_experts = n.unwrap();
  CHECK_HOST(num_experts >= kMinExperts && num_experts <= kMaxExperts)
      << "sm70_router_topk: num_experts " << num_experts << " outside [" << kMinExperts << ", " << kMaxExperts << "]";
  if (num_tokens == 0) {
    return;
  }
  launch<kTopK, (kMinExperts + kWarp - 1) / kWarp>(
      static_cast<uint32_t>(num_tokens),
      device_.unwrap(),
      static_cast<float*>(weights.data_ptr()),
      static_cast<int32_t*>(ids.data_ptr()),
      static_cast<const float*>(scores.data_ptr()),
      scores.stride(0),
      static_cast<const float*>(bias.data_ptr()),
      static_cast<int32_t>(num_experts),
      static_cast<float>(scale),
      renormalize,
      apply_scale);
}

}  // namespace sglang::sm70_router_topk
