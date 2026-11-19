// SPDX-License-Identifier: Apache-2.0
// Split-KV grouped decode attention on Volta: one query token per request, fp16 KV
// in a [slots, kv_heads, 256] pool, tokens addressed through the CSR kv_indptr /
// kv_indices of the Triton decode metadata.
//
// Work split: the summed context T is cut into chunks of
// C = round_up(max(min_chunk_tokens, ceil(T / target_chunks)), 16) tokens that never
// cross a request, so there are at most target_chunks + batch chunks. The grid is
// therefore fixed per batch size (CUDA-graph safe) while C follows the live lengths.
// Each CTA writes the normalized partial output and log2-sum-exp of its chunk per q
// head; the combine kernel merges the chunks of each request.

#include <sgl_kernel/ffi.h>
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/warp.cuh>

#include <cub/block/block_reduce.cuh>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <cuda_fp16.h>

namespace sglang {

namespace sm70_grouped_decode {

constexpr int kDim = 256;
constexpr int kLaneElems = kDim / 32;  // one 16-byte load per lane covers a 512-byte row
constexpr int kChunkAlign = 16;        // arbitrary; a multiple of every tile length
constexpr int kMaxChunks = 1024;       // combine weights live in shared memory
constexpr int kCombineThreads = 128;
constexpr int kCombineDims = 32;
constexpr int kCombineGroups = kCombineThreads / (kCombineDims / 4);
constexpr unsigned kFull = 0xffffffffu;

struct Chunk {
  int request;  // -1 when the chunk id is past the last chunk
  int first;    // first chunk id of the request
};

SGL_DEVICE int chunk_tokens(const int32_t* indptr, int batch, int target, int min_tokens) {
  const int total = indptr[batch] - indptr[0];
  const int tokens = max(min_tokens, (total + target - 1) / target);
  return (tokens + kChunkAlign - 1) / kChunkAlign * kChunkAlign;
}

SGL_DEVICE int chunk_count(const int32_t* indptr, int request, int tokens) {
  return (indptr[request + 1] - indptr[request] + tokens - 1) / tokens;
}

// Warp-wide; every lane returns the same value.
SGL_DEVICE Chunk locate_chunk(const int32_t* indptr, int batch, int tokens, int cid) {
  const int lane = threadIdx.x % 32;
  int base = 0;
  for (int b0 = 0; b0 < batch; b0 += 32) {
    const int b = b0 + lane;
    const int n = b < batch ? chunk_count(indptr, b, tokens) : 0;
    const int incl = device::warp::inclusive_sum(n, static_cast<uint32_t>(lane));
    const unsigned hit = __ballot_sync(kFull, b < batch && cid < base + incl);
    if (hit != 0) {
      const int src = __ffs(hit) - 1;
      const int end = __shfl_sync(kFull, incl, src);
      const int count = __shfl_sync(kFull, n, src);
      return {b0 + src, base + end - count};
    }
    base += __shfl_sync(kFull, incl, 31);
  }
  return {-1, 0};
}

SGL_DEVICE void load_row(float (&dst)[kLaneElems], const uint4& raw) {
  const half2* h = reinterpret_cast<const half2*>(&raw);
#pragma unroll
  for (int i = 0; i < kLaneElems / 2; ++i) {
    const float2 f = __half22float2(h[i]);
    dst[2 * i] = f.x;
    dst[2 * i + 1] = f.y;
  }
}

// Slot of token t0 + lane % kTile, or 0 past the end.
template <int kTile>
SGL_DEVICE int load_slots(const int64_t* indices, int t0, int hi, int lane) {
  const int t = t0 + lane % kTile;
  return t < hi ? static_cast<int>(indices[t]) : 0;
}

// Row i of the tile comes from the slot held by lane i; rows past `valid` are zero.
template <int kTile>
SGL_DEVICE void load_tile(uint4 (&dst)[kTile], const half* base, int64_t stride, int slots, int valid) {
#pragma unroll
  for (int i = 0; i < kTile; ++i) {
    const int slot = __shfl_sync(kFull, slots, i);
    dst[i] = make_uint4(0, 0, 0, 0);
    if (i < valid) dst[i] = *reinterpret_cast<const uint4*>(base + slot * stride);
  }
}

template <int kWidth>
SGL_DEVICE void reduce_scatter_step(float (&v)[32], int lane) {
  const bool upper = (lane & kWidth) != 0;
#pragma unroll
  for (int k = 0; k < kWidth; ++k) {
    const float send = upper ? v[k] : v[k + kWidth];
    const float keep = upper ? v[k + kWidth] : v[k];
    v[k] = keep + __shfl_xor_sync(kFull, send, kWidth);
  }
}

// In: 32 per-lane partial sums. Out: v[0] on lane l is the warp sum of index l.
SGL_DEVICE void reduce_scatter_32(float (&v)[32], int lane) {
  reduce_scatter_step<16>(v, lane);
  reduce_scatter_step<8>(v, lane);
  reduce_scatter_step<4>(v, lane);
  reduce_scatter_step<2>(v, lane);
  reduce_scatter_step<1>(v, lane);
}

// Grid (max_chunks, kv_heads), one warp walks every kWarps-th tile of the chunk. A
// tile is kTile tokens x kGroup heads, one score per lane.
template <int kGroup, int kWarps>
__global__ void __launch_bounds__(kWarps * 32) partial_kernel(
    const half* __restrict__ q,
    int64_t q_stride_b,
    int64_t q_stride_h,
    const half* __restrict__ k,
    int64_t k_stride_t,
    int64_t k_stride_h,
    const half* __restrict__ v,
    int64_t v_stride_t,
    int64_t v_stride_h,
    const int32_t* __restrict__ indptr,
    const int64_t* __restrict__ indices,
    float* __restrict__ part_o,
    float* __restrict__ part_lse,
    int batch,
    int heads,
    int target,
    int min_tokens,
    float scale_log2,
    float v_scale) {
  constexpr int kTile = 32 / kGroup;
  static_assert(kGroup * kTile == 32, "one score per lane per tile");
  static_assert(kGroup >= 2 && kGroup <= 8, "register budget");
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  const int cid = blockIdx.x;
  const int kvh = blockIdx.y;

  __shared__ float s_p[kWarps][32];
  __shared__ float s_alpha[kWarps][kGroup];
  __shared__ float s_m[kWarps][kGroup];
  __shared__ float s_l[kWarps][kGroup];
  __shared__ __align__(16) float s_acc[kWarps][kGroup][kDim];

  const int tokens = chunk_tokens(indptr, batch, target, min_tokens);
  const Chunk chunk = locate_chunk(indptr, batch, tokens, cid);
  if (chunk.request < 0) return;  // same for every warp of the CTA
  const int start = indptr[chunk.request];
  const int lo = (cid - chunk.first) * tokens;
  const int hi = min(indptr[chunk.request + 1] - start, lo + tokens);

  float qf[kGroup][kLaneElems];
#pragma unroll
  for (int h = 0; h < kGroup; ++h) {
    const half* src = q + chunk.request * q_stride_b + (kvh * kGroup + h) * q_stride_h + lane * kLaneElems;
    load_row(qf[h], *reinterpret_cast<const uint4*>(src));
#pragma unroll
    for (int e = 0; e < kLaneElems; ++e)
      qf[h][e] *= scale_log2;
  }

  float acc[kGroup][kLaneElems];
#pragma unroll
  for (int h = 0; h < kGroup; ++h)
#pragma unroll
    for (int e = 0; e < kLaneElems; ++e)
      acc[h][e] = 0.f;
  // Running max and partial sum of head lane % kGroup.
  float m_run = -INFINITY;
  float l_run = 0.f;

  const half* k_base = k + kvh * k_stride_h + lane * kLaneElems;
  const half* v_base = v + kvh * v_stride_h + lane * kLaneElems;
  constexpr int kStep = kWarps * kTile;
  // Software pipeline: the next tile's K is in flight during this tile's softmax
  // and PV, its V during the next QK.
  int t0 = lo + warp * kTile;
  uint4 kr[kTile], vr[kTile];
  const int64_t* req_indices = indices + start;
  int slots = load_slots<kTile>(req_indices, t0, hi, lane);
  load_tile(kr, k_base, k_stride_t, slots, hi - t0);
  load_tile(vr, v_base, v_stride_t, slots, hi - t0);
  slots = load_slots<kTile>(req_indices, t0 + kStep, hi, lane);
  for (; t0 < hi; t0 += kStep) {
    const int next = t0 + kStep;
    float s[32];
#pragma unroll
    for (int i = 0; i < kTile; ++i) {
      float kf[kLaneElems];
      load_row(kf, kr[i]);
#pragma unroll
      for (int h = 0; h < kGroup; ++h) {
        float dot = 0.f;
#pragma unroll
        for (int e = 0; e < kLaneElems; ++e)
          dot = fmaf(qf[h][e], kf[e], dot);
        s[i * kGroup + h] = dot;
      }
    }
    load_tile(kr, k_base, k_stride_t, slots, hi - next);
    reduce_scatter_32(s, lane);
    const float score = t0 + lane / kGroup < hi ? s[0] : -INFINITY;
    // Token t0 is valid, so every head has a finite tile max.
    const float m_new = fmaxf(m_run, device::warp::reduce_max<kGroup, 32>(score));
    const float p = exp2f(score - m_new);
    const float alpha = exp2f(m_run - m_new);
    l_run = l_run * alpha + p;
    m_run = m_new;

    __syncwarp();
    s_p[warp][lane] = p;
    if (lane < kGroup) s_alpha[warp][lane] = alpha;
    __syncwarp();
    float pa[32];
#pragma unroll
    for (int i = 0; i < 32; i += 4) {
      const float4 t = *reinterpret_cast<const float4*>(&s_p[warp][i]);
      pa[i] = t.x;
      pa[i + 1] = t.y;
      pa[i + 2] = t.z;
      pa[i + 3] = t.w;
    }
#pragma unroll
    for (int h = 0; h < kGroup; ++h) {
      const float a = s_alpha[warp][h];
#pragma unroll
      for (int e = 0; e < kLaneElems; ++e)
        acc[h][e] *= a;
    }
#pragma unroll
    for (int i = 0; i < kTile; ++i) {
      float vf[kLaneElems];
      load_row(vf, vr[i]);
#pragma unroll
      for (int h = 0; h < kGroup; ++h) {
        const float pv = pa[i * kGroup + h];
#pragma unroll
        for (int e = 0; e < kLaneElems; ++e)
          acc[h][e] = fmaf(pv, vf[e], acc[h][e]);
      }
    }
    load_tile(vr, v_base, v_stride_t, slots, hi - next);
    slots = load_slots<kTile>(req_indices, next + kStep, hi, lane);
  }

  const float l_warp = device::warp::reduce_sum<kGroup, 32>(l_run);
  if (lane < kGroup) {
    s_m[warp][lane] = m_run;
    s_l[warp][lane] = l_warp;
  }
#pragma unroll
  for (int h = 0; h < kGroup; ++h)
#pragma unroll
    for (int e = 0; e < kLaneElems; e += 4)
      *reinterpret_cast<float4*>(&s_acc[warp][h][lane * kLaneElems + e]) =
          make_float4(acc[h][e], acc[h][e + 1], acc[h][e + 2], acc[h][e + 3]);
  __syncthreads();

  // Warp 0 always owns token lo, so the merged max is finite.
  for (int idx = threadIdx.x; idx < kGroup * kDim; idx += kWarps * 32) {
    const int h = idx / kDim;
    const int e = idx % kDim;
    float mx = -INFINITY;
#pragma unroll
    for (int w = 0; w < kWarps; ++w)
      mx = fmaxf(mx, s_m[w][h]);
    float den = 0.f, num = 0.f;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) {
      const float wgt = exp2f(s_m[w][h] - mx);
      den = fmaf(s_l[w][h], wgt, den);
      num = fmaf(s_acc[w][h][e], wgt, num);
    }
    const int64_t row = static_cast<int64_t>(cid) * heads + kvh * kGroup + h;
    part_o[row * kDim + e] = num / den * v_scale;
    if (e == 0) part_lse[row] = mx + log2f(den);
  }
}

// Grid (kDim / kCombineDims, heads, batch).
__global__ void __launch_bounds__(kCombineThreads) combine_kernel(
    const float* __restrict__ part_o,
    const float* __restrict__ part_lse,
    const int32_t* __restrict__ indptr,
    half* __restrict__ out,
    int64_t out_stride_b,
    int64_t out_stride_h,
    int batch,
    int heads,
    int target,
    int min_tokens) {
  using Reduce = cub::BlockReduce<float, kCombineThreads>;
  __shared__ typename Reduce::TempStorage temp;
  __shared__ float s_w[kMaxChunks];
  __shared__ __align__(16) float s_sum[kCombineGroups][kCombineDims];
  __shared__ float s_max, s_den;
  __shared__ int s_first;

  const int b = blockIdx.z;
  const int h = blockIdx.y;
  const int d0 = blockIdx.x * kCombineDims;
  half* dst = out + b * out_stride_b + h * out_stride_h + d0;
  const int tokens = chunk_tokens(indptr, batch, target, min_tokens);
  const int count = chunk_count(indptr, b, tokens);
  if (count == 0) {
    if (threadIdx.x < kCombineDims) dst[threadIdx.x] = __float2half_rn(0.f);
    return;
  }
  if (threadIdx.x < 32) {
    int before = 0;
    for (int r = threadIdx.x; r < b; r += 32)
      before += chunk_count(indptr, r, tokens);
    before = device::warp::reduce_sum(before);
    if (threadIdx.x == 0) s_first = before;
  }
  __syncthreads();
  const int first = s_first;

  float mx = -INFINITY;
  for (int c = threadIdx.x; c < count; c += kCombineThreads) {
    const float lse = part_lse[static_cast<int64_t>(first + c) * heads + h];
    s_w[c] = lse;
    mx = fmaxf(mx, lse);
  }
  mx = Reduce(temp).Reduce(mx, cub::Max());
  if (threadIdx.x == 0) s_max = mx;
  __syncthreads();
  float den = 0.f;
  for (int c = threadIdx.x; c < count; c += kCombineThreads) {
    const float w = exp2f(s_w[c] - s_max);
    s_w[c] = w;
    den += w;
  }
  den = Reduce(temp).Sum(den);
  if (threadIdx.x == 0) s_den = den;
  __syncthreads();

  const int g = threadIdx.x / (kCombineDims / 4);
  const int d = threadIdx.x % (kCombineDims / 4) * 4;
  float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll 4
  for (int c = g; c < count; c += kCombineGroups) {
    const float4 o =
        *reinterpret_cast<const float4*>(part_o + (static_cast<int64_t>(first + c) * heads + h) * kDim + d0 + d);
    const float w = s_w[c];
    acc.x = fmaf(w, o.x, acc.x);
    acc.y = fmaf(w, o.y, acc.y);
    acc.z = fmaf(w, o.z, acc.z);
    acc.w = fmaf(w, o.w, acc.w);
  }
  *reinterpret_cast<float4*>(&s_sum[g][d]) = acc;
  __syncthreads();
  if (threadIdx.x < kCombineDims) {
    float total = 0.f;
#pragma unroll
    for (int i = 0; i < kCombineGroups; ++i)
      total += s_sum[i][threadIdx.x];
    dst[threadIdx.x] = __float2half_rn(total / s_den);
  }
}

}  // namespace sm70_grouped_decode

/// \brief Grouped decode attention over fp16 KV, Volta.
/// \tparam kGroup q heads per kv head: 2, 4 or 8.
/// \tparam kWarps warps per partial CTA.
template <int kGroup, int kWarps>
struct Sm70GroupedDecodeKernel {
  /// \param q [batch, heads, 256] fp16, last dim contiguous.
  /// \param k_cache, v_cache [pool_tokens, kv_heads, 256] fp16, last dim contiguous, under 2^31 tokens.
  /// \param kv_indptr int32 [>= batch + 1], token offsets into kv_indices.
  /// \param kv_indices int64 token slots.
  /// \param out [batch, heads, 256] fp16, last dim contiguous.
  /// \param target_chunks chunks to split the batch's tokens into; batch + target_chunks <= 1024.
  /// \param scale_log2 softmax scale times k_scale times log2(e).
  static void
  run(tvm::ffi::TensorView q,
      tvm::ffi::TensorView k_cache,
      tvm::ffi::TensorView v_cache,
      tvm::ffi::TensorView kv_indptr,
      tvm::ffi::TensorView kv_indices,
      tvm::ffi::TensorView out,
      int64_t target_chunks,
      int64_t min_chunk_tokens,
      double scale_log2,
      double v_scale) {
    using namespace host;
    using sm70_grouped_decode::kDim;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    auto batch = SymbolicSize{"batch"}, heads = SymbolicSize{"heads"}, kv_heads = SymbolicSize{"kv_heads"};
    auto pool = SymbolicSize{"pool_tokens"}, indptr_len = SymbolicSize{};
    auto q_sb = SymbolicSize{}, q_sh = SymbolicSize{}, k_st = SymbolicSize{}, k_sh = SymbolicSize{};
    auto v_st = SymbolicSize{}, v_sh = SymbolicSize{}, o_sb = SymbolicSize{}, o_sh = SymbolicSize{};
    TensorMatcher({batch, heads, kDim})  //
        .with_strides({q_sb, q_sh, 1})
        .with_dtype<fp16_t>()
        .with_device(device)
        .ensure_alignment(16)
        .verify(q);
    TensorMatcher({pool, kv_heads, kDim})  //
        .with_strides({k_st, k_sh, 1})
        .with_dtype<fp16_t>()
        .with_device(device)
        .ensure_alignment(16)
        .verify(k_cache);
    TensorMatcher({pool, kv_heads, kDim})  //
        .with_strides({v_st, v_sh, 1})
        .with_dtype<fp16_t>()
        .with_device(device)
        .ensure_alignment(16)
        .verify(v_cache);
    TensorMatcher({indptr_len}).with_dtype<int32_t>().with_device(device).verify(kv_indptr);
    TensorMatcher({-1}).with_dtype<int64_t>().with_device(device).verify(kv_indices);
    TensorMatcher({batch, heads, kDim})  //
        .with_strides({o_sb, o_sh, 1})
        .with_dtype<fp16_t>()
        .with_device(device)
        .ensure_alignment(16)
        .verify(out);
    const int64_t b = batch.unwrap(), h = heads.unwrap(), hkv = kv_heads.unwrap(), c = target_chunks + b;
    CHECK_HOST(h == hkv * kGroup) << "heads " << h << " != kv_heads " << hkv << " * group " << kGroup;
    CHECK_HOST(indptr_len.unwrap() >= b + 1) << "kv_indptr has " << indptr_len.unwrap() << " entries, batch " << b;
    CHECK_HOST(target_chunks >= 1 && c <= sm70_grouped_decode::kMaxChunks)
        << "batch " << b << " plus target " << target_chunks << " chunks exceeds " << sm70_grouped_decode::kMaxChunks;
    CHECK_HOST(min_chunk_tokens >= 1) << "min_chunk_tokens " << min_chunk_tokens;
    CHECK_HOST(pool.unwrap() <= INT32_MAX) << "slots are 32-bit, pool has " << pool.unwrap() << " tokens";
    for (const int64_t stride :
         {q_sb.unwrap(),
          q_sh.unwrap(),
          k_st.unwrap(),
          k_sh.unwrap(),
          v_st.unwrap(),
          v_sh.unwrap(),
          o_sb.unwrap(),
          o_sh.unwrap()}) {
      CHECK_HOST(stride % 8 == 0) << "rows are read as 16-byte vectors, stride " << stride << " is not a multiple of 8";
    }
    if (b == 0) return;
    const DLDevice dev = device.unwrap();
    // Per chunk: the fp32 partial output of every head, then its log2-sum-exp.
    const int64_t part_o_elems = c * h * kDim;
    const auto workspace = host::ffi::alloc_workspace_tensor((part_o_elems + c * h) * sizeof(float), dev);
    float* const part_o = static_cast<float*>(workspace.data_ptr());
    float* const part_lse = part_o + part_o_elems;
    LaunchKernel(dim3(static_cast<uint32_t>(c), static_cast<uint32_t>(hkv)), kWarps * 32, dev)(
        sm70_grouped_decode::partial_kernel<kGroup, kWarps>,
        static_cast<const half*>(q.data_ptr()),
        q_sb.unwrap(),
        q_sh.unwrap(),
        static_cast<const half*>(k_cache.data_ptr()),
        k_st.unwrap(),
        k_sh.unwrap(),
        static_cast<const half*>(v_cache.data_ptr()),
        v_st.unwrap(),
        v_sh.unwrap(),
        static_cast<const int32_t*>(kv_indptr.data_ptr()),
        static_cast<const int64_t*>(kv_indices.data_ptr()),
        part_o,
        part_lse,
        static_cast<int>(b),
        static_cast<int>(h),
        static_cast<int>(target_chunks),
        static_cast<int>(min_chunk_tokens),
        static_cast<float>(scale_log2),
        static_cast<float>(v_scale));
    using sm70_grouped_decode::kCombineDims;
    LaunchKernel(
        dim3(kDim / kCombineDims, static_cast<uint32_t>(h), static_cast<uint32_t>(b)),
        sm70_grouped_decode::kCombineThreads,
        dev)(
        sm70_grouped_decode::combine_kernel,
        part_o,
        part_lse,
        static_cast<const int32_t*>(kv_indptr.data_ptr()),
        static_cast<half*>(out.data_ptr()),
        o_sb.unwrap(),
        o_sh.unwrap(),
        static_cast<int>(b),
        static_cast<int>(h),
        static_cast<int>(target_chunks),
        static_cast<int>(min_chunk_tokens));
  }
};

}  // namespace sglang
