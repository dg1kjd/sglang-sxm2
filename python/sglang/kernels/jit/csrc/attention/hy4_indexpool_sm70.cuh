// SPDX-License-Identifier: Apache-2.0
// Hunyuan-4-Flash IndexPool indexer on Volta (SM70): fp16 pooled keys and
// queries, fp32 math, HMMA scores, exact top-k.
//
// Key layout. The DSA index-key cache keeps 64-token pages of 64 * 128 key
// bytes followed by 64 * 4 scale bytes. A completed group of 4 tokens stores
// its pooled key as 128 fp16 values (256 bytes) at the byte offset of the
// group's first token row, i.e. in the key bytes of the group's first two
// token rows. The scale bytes are not used.
//
// Score of query row r against group g (groups before the row's own group):
//   sum_h relu(q[r, h] . k[g]) * w[r, h]
// with q after rope, w = head weight * H^-0.5 * 128^-0.5. Rope is GPT-J style
// on dims 64..127 with the rotated half de-interleaved, the angle computed in
// fp32 as position * inv_freq.
//
// Selection keeps, per row, the k highest scores among its groups; on equal
// scores the lower group index wins. Selected groups are emitted in
// increasing group order, each as its 4 token slots, then the row's own group
// up to the row, then -1.
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>

namespace sglang::hy4_indexpool_sm70 {

inline constexpr int kDim = 128;
inline constexpr int kPairs = 32;
inline constexpr int kHeads = 16;
inline constexpr int kRatio = 4;
inline constexpr int kPageTokens = 64;
inline constexpr int kGroupsPerPage = kPageTokens / kRatio;
inline constexpr int kRowBytes = 128;  // key bytes per token row of a page
inline constexpr int kGroupStride = kRatio * kRowBytes;
inline constexpr int kWarp = 32;
inline constexpr uint32_t kFull = 0xffffffffu;

// Top-level score histogram: the 11 high bits of the order-preserving key.
inline constexpr int kBins = 2048;
inline constexpr int kBinShift = 21;

SGL_DEVICE int64_t load_index(const void* p, bool is64, int64_t i) {
  return is64 ? static_cast<const int64_t*>(p)[i] : static_cast<int64_t>(static_cast<const int32_t*>(p)[i]);
}

template <typename T>
SGL_DEVICE float to_float(T x);
template <>
SGL_DEVICE float to_float<fp16_t>(fp16_t x) {
  return __half2float(x);
}
template <>
SGL_DEVICE float to_float<bf16_t>(bf16_t x) {
  return __bfloat162float(x);
}
template <>
SGL_DEVICE float to_float<float>(float x) {
  return x;
}

// Rope position of one pair. Plain rope reads ptr[r * stride_r]; MRoPE reads
// (t, h, w) at ptr[r * stride_r + a * stride_a] and picks h or w for the
// interleaved pairs of their sections.
struct RopePos {
  const void* ptr;
  int64_t stride_r;
  int64_t stride_a;
  bool is64;
  bool mrope;
  int32_t s1;
  int32_t s2;
};

SGL_DEVICE int64_t rope_position(const RopePos& rp, int64_t r, int pair) {
  const int64_t base = r * rp.stride_r;
  if (!rp.mrope) {
    return load_index(rp.ptr, rp.is64, base);
  }
  int axis = 0;
  if (pair % 3 == 1 && pair < 3 * rp.s1) {
    axis = 1;
  } else if (pair % 3 == 2 && pair < 3 * rp.s2) {
    axis = 2;
  }
  return load_index(rp.ptr, rp.is64, base + axis * rp.stride_a);
}

// cos/sin of the fp32 angle position * inv_freq, evaluated in fp64.
SGL_DEVICE void rope_cos_sin(int64_t pos, float inv_freq, float& c, float& s) {
  const float angle = static_cast<float>(pos) * inv_freq;
  double sd, cd;
  sincos(static_cast<double>(angle), &sd, &cd);
  c = static_cast<float>(cd);
  s = static_cast<float>(sd);
}

// Output dim j (0..127) of rope applied to x (128 fp32 values in shared memory).
SGL_DEVICE float rope_output(const float* x, int j, const float* cos_s, const float* sin_s) {
  if (j < 64) {
    return x[j];
  }
  const int o = j - 64;
  const int pair = o < kPairs ? o : o - kPairs;
  const float xe = x[64 + 2 * pair];
  const float xo = x[65 + 2 * pair];
  const float c = cos_s[pair];
  const float s = sin_s[pair];
  return o < kPairs ? xe * c - xo * s : xo * c + xe * s;
}

SGL_DEVICE __half* key_ptr(uint8_t* buf, int64_t page_stride, int64_t slot) {
  return reinterpret_cast<__half*>(buf + (slot / kPageTokens) * page_stride + (slot % kPageTokens) * kRowBytes);
}

// Block-wide sum over the 128 threads of a pooling block, fixed order.
SGL_DEVICE float block128_sum(float v, float* red, int slot) {
  for (int o = 16; o > 0; o >>= 1) {
    v += __shfl_xor_sync(kFull, v, o);
  }
  const int warp = threadIdx.x / kWarp;
  if ((threadIdx.x & (kWarp - 1)) == 0) {
    red[slot * 4 + warp] = v;
  }
  __syncthreads();
  return (red[slot * 4 + 0] + red[slot * 4 + 1]) + (red[slot * 4 + 2] + red[slot * 4 + 3]);
}

// ---------------------------------------------------------------------------
// Pooling: per channel softmax over the group's 4 rows of gates + ape, the
// softmax-weighted sum of values, RMSNorm with key_norm, rope, fp16 store.
// ---------------------------------------------------------------------------

struct PoolWeights {
  const float* __restrict__ wv_t;  // [128 in, 128 out]
  const float* __restrict__ wg_t;  // [128 in, 128 out]
  const float* __restrict__ ape;   // [4, 128]
  const float* __restrict__ key_norm;
  const float* __restrict__ inv_freq;  // [32]
  float eps;
};

template <int kG>
struct PoolSmem {
  float x[kG * kRatio][kDim];
  float pooled[kG][kDim];
  float cos_v[kG][kPairs];
  float sin_v[kG][kPairs];
  float red[kG * 4];
};

// 128 threads; thread c owns output channel c. x rows of skipped groups may be
// anything finite; their results are not stored.
template <int kG>
SGL_DEVICE void
pool_store(PoolSmem<kG>& s, const PoolWeights& w, const int64_t (&dst)[kG], uint8_t* buf, int64_t page_stride) {
  const int c = threadIdx.x;
  float v[kG * kRatio];
  float g[kG * kRatio];
#pragma unroll
  for (int i = 0; i < kG * kRatio; ++i) {
    v[i] = 0.f;
    g[i] = 0.f;
  }
  for (int k = 0; k < kDim; k += 4) {
    float wv[4], wg[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      wv[j] = __ldg(&w.wv_t[(k + j) * kDim + c]);
      wg[j] = __ldg(&w.wg_t[(k + j) * kDim + c]);
    }
#pragma unroll
    for (int i = 0; i < kG * kRatio; ++i) {
      const float4 x = *reinterpret_cast<const float4*>(&s.x[i][k]);
      v[i] = fmaf(x.x, wv[0], v[i]);
      v[i] = fmaf(x.y, wv[1], v[i]);
      v[i] = fmaf(x.z, wv[2], v[i]);
      v[i] = fmaf(x.w, wv[3], v[i]);
      g[i] = fmaf(x.x, wg[0], g[i]);
      g[i] = fmaf(x.y, wg[1], g[i]);
      g[i] = fmaf(x.z, wg[2], g[i]);
      g[i] = fmaf(x.w, wg[3], g[i]);
    }
  }
  float pooled[kG];
#pragma unroll
  for (int gi = 0; gi < kG; ++gi) {
    float logit[kRatio];
    float m = -INFINITY;
#pragma unroll
    for (int j = 0; j < kRatio; ++j) {
      logit[j] = g[gi * kRatio + j] + __ldg(&w.ape[j * kDim + c]);
      m = fmaxf(m, logit[j]);
    }
    float e[kRatio];
    float sum = 0.f;
#pragma unroll
    for (int j = 0; j < kRatio; ++j) {
      e[j] = expf(logit[j] - m);
      sum += e[j];
    }
    float acc = 0.f;
#pragma unroll
    for (int j = 0; j < kRatio; ++j) {
      acc += v[gi * kRatio + j] * (e[j] / sum);
    }
    pooled[gi] = acc;
  }
#pragma unroll
  for (int gi = 0; gi < kG; ++gi) {
    const float ss = block128_sum(pooled[gi] * pooled[gi], s.red, gi);
    s.pooled[gi][c] = pooled[gi] * rsqrtf(ss / kDim + w.eps) * __ldg(&w.key_norm[c]);
  }
  __syncthreads();
#pragma unroll
  for (int gi = 0; gi < kG; ++gi) {
    if (dst[gi] >= 0) {
      const float out = rope_output(s.pooled[gi], c, s.cos_v[gi], s.sin_v[gi]);
      key_ptr(buf, page_stride, dst[gi])[c] = __float2half_rn(out);
    }
  }
}

struct PoolParams {
  const fp16_t* __restrict__ raw;  // [raw_slots, 128]
  int64_t raw_slots;
  const int64_t* __restrict__ members;      // [G, 4] member slots
  const int64_t* __restrict__ write_slots;  // [G], < 0: skip
  RopePos pos;                              // rope position of the group's first token, per group
  PoolWeights w;
  uint8_t* __restrict__ buf;
  int64_t page_stride;
  int64_t groups;
};

inline constexpr int kPoolGroups = 8;

__global__ void __launch_bounds__(kDim) pool_kernel(const PoolParams p) {
  __shared__ __align__(16) PoolSmem<kPoolGroups> s;
  const int c = threadIdx.x;
  const int64_t g0 = static_cast<int64_t>(blockIdx.x) * kPoolGroups;
  int64_t dst[kPoolGroups];
#pragma unroll
  for (int gi = 0; gi < kPoolGroups; ++gi) {
    const int64_t g = g0 + gi;
    dst[gi] = g < p.groups ? p.write_slots[g] : -1;
#pragma unroll
    for (int j = 0; j < kRatio; ++j) {
      float x = 0.f;
      if (dst[gi] >= 0) {
        int64_t slot = p.members[g * kRatio + j];
        slot = slot < 0 ? 0 : (slot >= p.raw_slots ? p.raw_slots - 1 : slot);
        x = __half2float(p.raw[slot * kDim + c]);
      }
      s.x[gi * kRatio + j][c] = x;
    }
  }
  for (int i = c; i < kPoolGroups * kPairs; i += kDim) {
    const int gi = i / kPairs;
    const int pair = i % kPairs;
    const int64_t g = g0 + gi;
    float cs = 1.f, sn = 0.f;
    if (g < p.groups && p.write_slots[g] >= 0) {
      rope_cos_sin(rope_position(p.pos, g, pair), p.w.inv_freq[pair], cs, sn);
    }
    s.cos_v[gi][pair] = cs;
    s.sin_v[gi][pair] = sn;
  }
  __syncthreads();
  pool_store<kPoolGroups>(s, p.w, dst, p.buf, p.page_stride);
}

// ---------------------------------------------------------------------------
// Raw keys: RMSNorm of the token key projection, stored fp16 at the token's
// slot. One warp per row.
// ---------------------------------------------------------------------------

struct StoreRawParams {
  const void* __restrict__ key;  // [R, >=128] fp16/bf16
  int64_t key_stride;
  const float* __restrict__ norm;  // [128]
  float eps;
  const void* __restrict__ out_loc;
  bool loc64;
  fp16_t* __restrict__ raw;
  int64_t raw_slots;
  int64_t rows;
};

template <typename T>
__global__ void store_raw_kernel(const StoreRawParams p) {
  const int lane = threadIdx.x & (kWarp - 1);
  const int64_t r = static_cast<int64_t>(blockIdx.x) * (blockDim.x / kWarp) + threadIdx.x / kWarp;
  if (r >= p.rows) {
    return;
  }
  const T* key = static_cast<const T*>(p.key) + r * p.key_stride;
  float x[4];
  float ss = 0.f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    x[i] = to_float(key[lane * 4 + i]);
    ss += x[i] * x[i];
  }
  for (int o = 16; o > 0; o >>= 1) {
    ss += __shfl_xor_sync(kFull, ss, o);
  }
  const float inv = rsqrtf(ss / kDim + p.eps);
  int64_t slot = load_index(p.out_loc, p.loc64, r);
  slot = slot < 0 ? 0 : (slot >= p.raw_slots ? p.raw_slots - 1 : slot);
  __align__(8) __half out[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    out[i] = __float2half_rn(x[i] * inv * p.norm[lane * 4 + i]);
  }
  *reinterpret_cast<uint2*>(p.raw + slot * kDim + lane * 4) = *reinterpret_cast<const uint2*>(out);
}

// ---------------------------------------------------------------------------
// Decode: store each row's raw key; pool the row's group when this token
// completes it. One 128-thread block per row.
// ---------------------------------------------------------------------------

struct DecodeParams {
  const void* __restrict__ key;  // [R, >=128] fp16/bf16 raw key projection
  int64_t key_stride;
  const float* __restrict__ key_norm_w;  // [128] token key RMSNorm weight
  float key_eps;
  const void* __restrict__ out_loc;
  bool loc64;
  const int32_t* __restrict__ seq_lens;  // [R] including this token
  const int32_t* __restrict__ pages;     // [R, NP]
  int64_t pages_stride;
  int32_t num_pages;
  fp16_t* __restrict__ raw;
  int64_t raw_slots;
  RopePos xyz;  // MRoPE only: per-slot (t, h, w) positions
  PoolWeights w;
  uint8_t* __restrict__ buf;
  int64_t page_stride;
};

template <typename T>
__global__ void __launch_bounds__(kDim) decode_update_kernel(const DecodeParams p) {
  __shared__ __align__(16) PoolSmem<1> s;
  const int c = threadIdx.x;
  const int64_t r = blockIdx.x;
  const float x = to_float(static_cast<const T*>(p.key)[r * p.key_stride + c]);
  const float ss = block128_sum(x * x, s.red, 0);
  const float key = x * rsqrtf(ss / kDim + p.key_eps) * p.key_norm_w[c];
  int64_t slot = load_index(p.out_loc, p.loc64, r);
  slot = slot < 0 ? 0 : (slot >= p.raw_slots ? p.raw_slots - 1 : slot);
  p.raw[slot * kDim + c] = __float2half_rn(key);
  const int64_t pos = static_cast<int64_t>(p.seq_lens[r]) - 1;
  if (pos < 0 || pos % kRatio != kRatio - 1) {
    return;
  }
  // The group's other rows were stored by earlier steps; this block's own
  // store is visible to the block after the barrier.
  __syncthreads();
  const int64_t first = pos - (kRatio - 1);
  int64_t dst[1];
#pragma unroll
  for (int j = 0; j < kRatio; ++j) {
    const int64_t tok = first + j;
    int64_t col = tok / kPageTokens;
    col = col < p.num_pages - 1 ? col : p.num_pages - 1;
    int64_t ms = static_cast<int64_t>(p.pages[r * p.pages_stride + col]) * kPageTokens + tok % kPageTokens;
    ms = ms < 0 ? 0 : (ms >= p.raw_slots ? p.raw_slots - 1 : ms);
    if (j == 0) {
      dst[0] = ms;
    }
    s.x[j][c] = __half2float(p.raw[ms * kDim + c]);
  }
  if (c < kPairs) {
    int64_t rope_pos = first;
    if (p.xyz.mrope) {
      rope_pos = rope_position(p.xyz, dst[0], c);
    }
    rope_cos_sin(rope_pos, p.w.inv_freq[c], s.cos_v[0][c], s.sin_v[0][c]);
  }
  __syncthreads();
  pool_store<1>(s, p.w, dst, p.buf, p.page_stride);
}

// ---------------------------------------------------------------------------
// Query: rope to fp16, head weights folded with the score scale. Also clears
// the row's score histogram for the score kernel. One 128-thread block per row.
// ---------------------------------------------------------------------------

struct QueryParams {
  const void* __restrict__ q;  // [R, H * 128] fp16/bf16 (row stride q_stride)
  int64_t q_stride;
  RopePos pos;
  const void* __restrict__ hw;  // [R, H] (row stride hw_stride)
  int64_t hw_stride;
  const float* __restrict__ inv_freq;
  fp16_t* __restrict__ q_out;  // [R, H, 128]
  float* __restrict__ w_out;   // [R, H]
  int32_t* __restrict__ hist;  // [R, kBins] or null
  float weight_scale;
};

template <typename T, typename TW>
__global__ void __launch_bounds__(kDim) query_kernel(const QueryParams p) {
  __shared__ float cos_s[kPairs];
  __shared__ float sin_s[kPairs];
  const int t = threadIdx.x;
  const int64_t r = blockIdx.x;
  if (t < kPairs) {
    rope_cos_sin(rope_position(p.pos, r, t), p.inv_freq[t], cos_s[t], sin_s[t]);
  }
  if (t < kHeads) {
    const float hw = to_float(static_cast<const TW*>(p.hw)[r * p.hw_stride + t]);
    p.w_out[r * kHeads + t] = hw * p.weight_scale;
  }
  if (p.hist != nullptr) {
    int4* h = reinterpret_cast<int4*>(p.hist + r * kBins);
    for (int i = t; i < kBins / 4; i += kDim) {
      h[i] = make_int4(0, 0, 0, 0);
    }
  }
  __syncthreads();
  // Thread t: head t / 8, output dims [16 * (t % 8), +16).
  const int h = t / 8;
  const int seg = t % 8;
  const T* src = static_cast<const T*>(p.q) + r * p.q_stride + h * kDim;
  fp16_t* dst = p.q_out + (r * kHeads + h) * kDim + seg * 16;
  __align__(16) __half out[16];
  if (seg < 4) {
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      out[i] = __float2half_rn(to_float(src[seg * 16 + i]));
    }
  } else {
    const int o0 = (seg - 4) * 16;  // de-interleaved rope output offset
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const int o = o0 + i;
      const int pair = o < kPairs ? o : o - kPairs;
      const float xe = to_float(src[64 + 2 * pair]);
      const float xo = to_float(src[65 + 2 * pair]);
      const float c = cos_s[pair];
      const float s = sin_s[pair];
      out[i] = __float2half_rn(o < kPairs ? xe * c - xo * s : xo * c + xe * s);
    }
  }
  reinterpret_cast<uint4*>(dst)[0] = reinterpret_cast<const uint4*>(out)[0];
  reinterpret_cast<uint4*>(dst)[1] = reinterpret_cast<const uint4*>(out)[1];
}

// ---------------------------------------------------------------------------
// Scores. A block scores a run of up to kRows query rows that read the same
// keys (rows of one request) against 64-group tiles. q.k runs on HMMA with
// fp32 accumulation; the relu-weighted head sum is fp32 in a fixed order, so
// a score does not depend on the block shape.
// ---------------------------------------------------------------------------

SGL_DEVICE uint32_t order_key(float v) {
  // -0 and +0 compare equal. A NaN sorts by its sign bit: above +inf when it is
  // clear, below -inf when it is set.
  v = v == 0.f ? 0.f : v;
  const uint32_t u = __float_as_uint(v);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

struct LogitsParams {
  const fp16_t* __restrict__ q;       // [R, 16, 128]
  const float* __restrict__ w;        // [R, 16]
  const int32_t* __restrict__ pages;  // [B, NP]
  int64_t pages_stride;
  int32_t num_pages;
  const void* __restrict__ row_batch;  // [R] page-table row of each query row
  bool rb64;
  const void* __restrict__ pos;
  bool pos64;
  const uint8_t* __restrict__ buf;
  int64_t page_stride;
  float* __restrict__ out;  // [R, num_groups]
  int64_t out_stride;
  int32_t num_groups;
  int32_t* __restrict__ hist;  // [R, kBins] or null
  int32_t rows;
  int32_t rows_per_block;
};

SGL_DEVICE int32_t row_upper(const void* pos, bool pos64, int64_t r, int32_t num_groups) {
  const int64_t p = load_index(pos, pos64, r);
  int64_t u = p >= 0 ? p / kRatio : 0;
  return static_cast<int32_t>(u < num_groups ? u : num_groups);
}

inline constexpr int kTileGroups = 64;
inline constexpr int kTileLd = kDim + 8;  // halves; pads rows across banks

template <int kRows>
struct LogitsCfg {
  static constexpr int kWarps = kRows == 1 ? 4 : 8;
  static constexpr int kThreads = kWarps * kWarp;
  static constexpr int kWarpsPerRow = kWarps / kRows;
  static constexpr int kFrags = kTileGroups / 16 / kWarpsPerRow;           // 16-group column tiles per warp
  static constexpr int kLoads = kTileGroups * (kDim * 2 / 16) / kThreads;  // uint4 per thread per tile
  static constexpr int kHistBytes = kRows * kBins * 4;                     // dynamic shared memory
};

template <int kRows>
struct LogitsSmem {
  __half k[kTileGroups][kTileLd];
  float scratch[LogitsCfg<kRows>::kWarps][16 * 16];
  float w[kRows][kHeads];
  int64_t batch[kRows];
  int32_t upper[kRows];
};

// Grid: (tile stride, row blocks). Block y scores rows [y * rows_per_block, +rows_per_block);
// consecutive rows with the same page-table row share each key tile. Block x
// takes tiles x, x + gridDim.x, ... Scores are written only for groups below
// each row's own group. With p.hist, each row's score histogram is counted in
// dynamic shared memory (Cfg::kHistBytes) and added to p.hist at the end.
template <int kRows>
__global__ void __launch_bounds__(LogitsCfg<kRows>::kThreads) logits_kernel(const LogitsParams p) {
  using Cfg = LogitsCfg<kRows>;
  using namespace nvcuda;
  __shared__ __align__(32) LogitsSmem<kRows> s;
  extern __shared__ __align__(16) int32_t hist_smem[];  // [kRows][kBins] with p.hist
  const int tid = threadIdx.x;
  const int warp = tid / kWarp;
  const int lane = tid & (kWarp - 1);
  const int64_t row0 = static_cast<int64_t>(blockIdx.y) * p.rows_per_block;
  const int nrows = static_cast<int>(min(static_cast<int64_t>(p.rows_per_block), p.rows - row0));
  const bool with_hist = p.hist != nullptr;

  if (tid < kRows) {
    s.upper[tid] = tid < nrows ? row_upper(p.pos, p.pos64, row0 + tid, p.num_groups) : 0;
    s.batch[tid] = tid < nrows ? load_index(p.row_batch, p.rb64, row0 + tid) : -1;
  }
  for (int i = tid; i < kRows * kHeads; i += Cfg::kThreads) {
    const int rr = i / kHeads;
    s.w[rr][i % kHeads] = rr < nrows ? p.w[(row0 + rr) * kHeads + i % kHeads] : 0.f;
  }
  if (with_hist) {
    for (int i = tid; i < kRows * kBins; i += Cfg::kThreads) {
      hist_smem[i] = 0;
    }
  }
  __syncthreads();

  const int my_row = warp / Cfg::kWarpsPerRow;
  const int frag0 = (warp % Cfg::kWarpsPerRow) * Cfg::kFrags;
  const bool row_live = my_row < nrows;
  const int my_upper = s.upper[my_row];

  wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a[kDim / 16];
  if (row_live) {
    const __half* qrow = reinterpret_cast<const __half*>(p.q) + (row0 + my_row) * kHeads * kDim;
#pragma unroll
    for (int d = 0; d < kDim / 16; ++d) {
      wmma::load_matrix_sync(a[d], qrow + d * 16, kDim);
    }
  }

  uint4 pre[Cfg::kLoads];
  for (int seg = 0; seg < nrows;) {
    const int64_t b = s.batch[seg];
    int seg_end = seg + 1;
    while (seg_end < nrows && s.batch[seg_end] == b) {
      ++seg_end;
    }
    int umax = 0;
    for (int i = seg; i < seg_end; ++i) {
      umax = max(umax, s.upper[i]);
    }
    const int tiles = (umax + kTileGroups - 1) / kTileGroups;
    const int32_t* page_row = p.pages + b * p.pages_stride;
    const bool mine = row_live && my_row >= seg && my_row < seg_end;

    auto load_tile = [&](int t) {
#pragma unroll
      for (int i = 0; i < Cfg::kLoads; ++i) {
        const int idx = tid + i * Cfg::kThreads;
        const int j = idx >> 4;
        const int part = idx & 15;
        const int g = t * kTileGroups + j;
        uint4 v = make_uint4(0, 0, 0, 0);
        if (g < umax) {
          int col = g / kGroupsPerPage;
          col = col < p.num_pages - 1 ? col : p.num_pages - 1;
          const int64_t page = page_row[col];
          const uint8_t* src = p.buf + page * p.page_stride + (g % kGroupsPerPage) * kGroupStride + part * 16;
          v = __ldg(reinterpret_cast<const uint4*>(src));
        }
        pre[i] = v;
      }
    };

    int t = blockIdx.x;
    if (t < tiles) {
      load_tile(t);
    }
    for (; t < tiles; t += gridDim.x) {
#pragma unroll
      for (int i = 0; i < Cfg::kLoads; ++i) {
        const int idx = tid + i * Cfg::kThreads;
        *reinterpret_cast<uint4*>(&s.k[idx >> 4][(idx & 15) * 8]) = pre[i];
      }
      __syncthreads();
      if (t + static_cast<int>(gridDim.x) < tiles) {
        load_tile(t + gridDim.x);
      }
      const int g_tile = t * kTileGroups;
      if (mine && g_tile < my_upper) {
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[Cfg::kFrags];
#pragma unroll
        for (int f = 0; f < Cfg::kFrags; ++f) {
          wmma::fill_fragment(acc[f], 0.f);
        }
#pragma unroll
        for (int d = 0; d < kDim / 16; ++d) {
#pragma unroll
          for (int f = 0; f < Cfg::kFrags; ++f) {
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> bf;
            wmma::load_matrix_sync(bf, &s.k[(frag0 + f) * 16][d * 16], kTileLd);
            wmma::mma_sync(acc[f], a[d], bf, acc[f]);
          }
        }
        float* scratch = s.scratch[warp];
        const int col = lane & 15;
        const int parity = lane >> 4;
        float* out_row = p.out + (row0 + my_row) * p.out_stride;
#pragma unroll
        for (int f = 0; f < Cfg::kFrags; ++f) {
          wmma::store_matrix_sync(scratch, acc[f], 16, wmma::mem_row_major);
          __syncwarp();
          float part = 0.f;
#pragma unroll
          for (int h2 = 0; h2 < kHeads / 2; ++h2) {
            const int h = 2 * h2 + parity;
            part += fmaxf(scratch[h * 16 + col], 0.f) * s.w[my_row][h];
          }
          // Even heads plus odd heads.
          const float score = part + __shfl_xor_sync(kFull, part, 16);
          __syncwarp();
          const int g = g_tile + (frag0 + f) * 16 + col;
          if (lane < 16 && g < my_upper) {
            out_row[g] = score;
            if (with_hist) {
              atomicAdd(&hist_smem[my_row * kBins + (order_key(score) >> kBinShift)], 1);
            }
          }
        }
      }
      __syncthreads();
    }
    seg = seg_end;
  }
  if (with_hist) {
    __syncthreads();
    int32_t* h = p.hist + row0 * kBins;
    for (int i = tid; i < nrows * kBins; i += Cfg::kThreads) {
      const int v = hist_smem[i];
      if (v != 0) {
        atomicAdd(&h[i], v);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Selection and expansion. One 512-thread block per row.
//
// The threshold key (the k-th largest) is found by radix select: a histogram of
// the top 11 key bits (from the score kernel, or one pass over the row), then
// finer bins until one key value is left. Counts that do not match the row send
// the search back to a histogram counted from the row. Once the keys at or
// above the current bin fit in shared memory they are gathered with their
// groups, and the rest of the search and the output run there: the selected
// groups are marked in a bitmap and written in group order. Rows of at most k
// groups, rows past kFastGroups groups, and rows whose candidates never fit in
// shared memory (many ties at the threshold) take an ordered pass over the row
// instead.
// ---------------------------------------------------------------------------

inline constexpr int kSelThreads = 512;
inline constexpr int kSelWarps = kSelThreads / kWarp;
inline constexpr int kSelCap = 6144;                     // candidates gathered in shared memory
inline constexpr int kFastGroups = kBins * 32;           // groups the bitmap over SelectSmem::hist covers
inline constexpr int kSelWords = kBins / kSelThreads;    // bitmap words per thread
inline constexpr int kSelVec = 4;                        // float4 loads in flight per thread
inline constexpr int kSelPer = 4 * kSelVec;              // consecutive scores per thread
inline constexpr int kSelChunk = kSelThreads * kSelPer;  // scores per pass step

struct SelectParams {
  const float* __restrict__ logits;  // [R, >= upper], row stride logits_stride
  int64_t logits_stride;
  int32_t num_groups;
  const int32_t* __restrict__ hist;  // [R, kBins] or null
  const void* __restrict__ pos;
  bool pos64;
  const int32_t* __restrict__ pages;
  int64_t pages_stride;
  int32_t num_pages;
  const void* __restrict__ row_batch;
  bool rb64;
  const int64_t* __restrict__ ragged;  // [R] or null
  int32_t* __restrict__ out;           // [R, topk]
  int64_t out_stride;
  int32_t topk;
  int32_t k;
};

struct SelectSmem {
  int32_t hist[kBins];         // bin counts; at the end the selected groups as a bitmap
  uint32_t cand[kSelCap];      // candidate keys
  uint16_t cand_idx[kSelCap];  // their groups (rows of up to kFastGroups groups)
  int32_t warp_sum[kSelWarps];
  int32_t n_cand;
  int32_t found_bin;
  int32_t found_above;
};

// Inclusive prefix sum over the block in thread order.
SGL_DEVICE int32_t block_scan(int32_t v, int32_t* warp_sum, int32_t& total) {
  const int lane = threadIdx.x & (kWarp - 1);
  const int warp = threadIdx.x / kWarp;
#pragma unroll
  for (int o = 1; o < kWarp; o <<= 1) {
    const int32_t n = __shfl_up_sync(kFull, v, o);
    if (lane >= o) {
      v += n;
    }
  }
  if (lane == kWarp - 1) {
    warp_sum[warp] = v;
  }
  __syncthreads();
  int32_t before = 0;
  total = 0;
#pragma unroll
  for (int i = 0; i < kSelWarps; ++i) {
    const int32_t x = warp_sum[i];
    before += i < warp ? x : 0;
    total += x;
  }
  __syncthreads();
  return v + before;
}

// Bin b of a histogram (bins ordered by score) such that the bins above b hold
// fewer than `need` elements and bins >= b at least `need`, in s.found_bin, or
// -1 when all bins together hold fewer than `need`. Returns the histogram total.
template <int kN>
SGL_DEVICE int32_t find_bin(SelectSmem& s, int need) {
  constexpr int kPer = kN / kSelThreads;
  const int t = threadIdx.x;
  if (t == 0) {
    s.found_bin = -1;  // the barriers in block_scan order this before the search
  }
  const int hi = kN - kPer * t;  // this thread owns bins [hi - kPer, hi), visited top-down
  int local = 0;
#pragma unroll
  for (int i = 1; i <= kPer; ++i) {
    local += s.hist[hi - i];
  }
  int32_t total;
  const int32_t incl = block_scan(local, s.warp_sum, total);
  int above = incl - local;
#pragma unroll
  for (int i = 1; i <= kPer; ++i) {
    const int b = hi - i;
    const int n = s.hist[b];
    if (above < need && need <= above + n) {
      s.found_bin = b;
      s.found_above = above;
    }
    above += n;
  }
  __syncthreads();
  return total;
}

SGL_DEVICE void hist_add(int32_t* hist, uint32_t bin, bool active) {
  const uint32_t live = __ballot_sync(kFull, active);
  if (active) {
    const uint32_t peers = __match_any_sync(live, bin);
    if ((__ffs(peers) - 1) == static_cast<int>(threadIdx.x & (kWarp - 1))) {
      atomicAdd(&hist[bin], __popc(peers));
    }
  }
}

SGL_DEVICE float4 load4(const float* row, int i, int n) {
  if (i + 3 < n) {
    return __ldg(reinterpret_cast<const float4*>(row + i));
  }
  float4 v;
  v.x = i < n ? __ldg(row + i) : 0.f;
  v.y = i + 1 < n ? __ldg(row + i + 1) : 0.f;
  v.z = i + 2 < n ? __ldg(row + i + 2) : 0.f;
  v.w = 0.f;
  return v;
}

SGL_DEVICE float elem(const float4& v, int e) {
  return e == 0 ? v.x : (e == 1 ? v.y : (e == 2 ? v.z : v.w));
}

// Scores [i, i + kSelPer) of a row, zero at and past n. All loads are issued
// before any is used, so a pass keeps kSelVec loads per thread in flight.
SGL_DEVICE void load_scores(const float* row, int i, int n, float4 (&v)[kSelVec]) {
  if (i + kSelPer <= n) {
#pragma unroll
    for (int j = 0; j < kSelVec; ++j) {
      v[j] = __ldg(reinterpret_cast<const float4*>(row + i + 4 * j));
    }
  } else {
#pragma unroll
    for (int j = 0; j < kSelVec; ++j) {
      v[j] = load4(row, i + 4 * j, n);
    }
  }
}

__global__ void __launch_bounds__(kSelThreads) select_kernel(const SelectParams p) {
  __shared__ __align__(16) SelectSmem s;
  const int t = threadIdx.x;
  const int64_t r = blockIdx.x;
  const int64_t pos = load_index(p.pos, p.pos64, r);
  const int64_t upper = pos >= 0 ? pos / kRatio : 0;
  const int u = static_cast<int>(upper < p.num_groups ? upper : p.num_groups);
  const float* row = p.logits + r * p.logits_stride;
  const bool take_all = u <= p.k;
  const int n_sel = take_all ? u : p.k;
  // Gather the keys above the threshold bin too, with their groups, so the
  // output can be written from shared memory.
  const bool fast = u <= kFastGroups;

  uint32_t thresh = 0;  // selected: key > thresh, or key == thresh among the first need_eq
  int need_eq = 0;
  int n_eq = 0;  // keys equal to thresh
  bool in_smem = false;
  if (!take_all) {
    // Every count the search uses is checked where the scores are read again:
    // the top histogram's total, the keys above the current bin in each pass
    // over the row, the number of keys gathered and each finer histogram's
    // total. A p.hist that does not match the row fails one of them, and the
    // search starts over from a histogram counted here.
    bool from_row = p.hist == nullptr;
    bool ok = false;
    uint32_t prefix = 0;
    int need = 0;
    int count = 0;
    for (;;) {
      if (!from_row) {
        for (int i = t; i < kBins; i += kSelThreads) {
          s.hist[i] = p.hist[r * kBins + i];
        }
      } else {
        for (int i = t; i < kBins; i += kSelThreads) {
          s.hist[i] = 0;
        }
        __syncthreads();
        for (int base = 0; base < u; base += kSelChunk) {
          const int i = base + kSelPer * t;
          float4 v[kSelVec];
          load_scores(row, i, u, v);
#pragma unroll
          for (int e = 0; e < kSelPer; ++e) {
            hist_add(s.hist, order_key(elem(v[e / 4], e % 4)) >> kBinShift, i + e < u);
          }
        }
      }
      __syncthreads();
      need = p.k;
      ok = find_bin<kBins>(s, need) == u && s.found_bin >= 0;
      if (ok) {
        prefix = static_cast<uint32_t>(s.found_bin);
        need -= s.found_above;
        count = s.hist[s.found_bin];
      }
      int shift = kBinShift;
      in_smem = false;
      __syncthreads();
      // Narrow the threshold key bin by bin: from shared-memory candidates once
      // they fit, otherwise by another pass over the row. need counts the keys
      // still to take from the current bin; p.k - need are above it.
      while (ok && shift > 0) {
        int above = 0;  // this thread's keys above the current bin, in a pass over the row
        if (!in_smem && count + (fast ? p.k - need : 0) <= kSelCap) {
          if (t == 0) {
            s.n_cand = 0;
          }
          __syncthreads();
          for (int base = 0; base < u; base += kSelChunk) {
            const int i = base + kSelPer * t;
            float4 v[kSelVec];
            load_scores(row, i, u, v);
#pragma unroll
            for (int e = 0; e < kSelPer; ++e) {
              const uint32_t key = order_key(elem(v[e / 4], e % 4));
              above += i + e < u && (key >> shift) > prefix;
              const bool take = fast ? (key >> shift) >= prefix : (key >> shift) == prefix;
              if (i + e < u && take) {
                const int j = atomicAdd(&s.n_cand, 1);
                if (j < kSelCap) {
                  s.cand[j] = key;
                  s.cand_idx[j] = static_cast<uint16_t>(i + e);
                }
              }
            }
          }
          in_smem = true;
          int32_t total_above;
          block_scan(above, s.warp_sum, total_above);
          ok = total_above == p.k - need && s.n_cand == count + (fast ? p.k - need : 0);
          if (!ok) {
            break;
          }
        }
        const int nb = shift > 11 ? 11 : shift;
        const int sub = shift - nb;
        const uint32_t mask = (1u << nb) - 1u;
        for (int i = t; i < kBins; i += kSelThreads) {
          s.hist[i] = 0;
        }
        __syncthreads();
        if (in_smem) {
          const int n = s.n_cand;
          for (int base = 0; base < n; base += kSelThreads) {
            const int i = base + t;
            const uint32_t key = i < n ? s.cand[i] : 0u;
            hist_add(s.hist, (key >> sub) & mask, i < n && (key >> shift) == prefix);
          }
        } else {
          for (int base = 0; base < u; base += kSelChunk) {
            const int i = base + kSelPer * t;
            float4 v[kSelVec];
            load_scores(row, i, u, v);
#pragma unroll
            for (int e = 0; e < kSelPer; ++e) {
              const uint32_t key = order_key(elem(v[e / 4], e % 4));
              above += i + e < u && (key >> shift) > prefix;
              hist_add(s.hist, (key >> sub) & mask, i + e < u && (key >> shift) == prefix);
            }
          }
        }
        __syncthreads();
        int32_t total_above = p.k - need;
        if (!in_smem) {
          block_scan(above, s.warp_sum, total_above);
        }
        // The last level has 10 bits; bins 1024.. are empty.
        const int32_t total = nb == 11 ? find_bin<2048>(s, need) : find_bin<1024>(s, need);
        ok = total == count && total_above == p.k - need && s.found_bin >= 0;
        if (ok) {
          prefix = (prefix << nb) | static_cast<uint32_t>(s.found_bin);
          need -= s.found_above;
          count = s.hist[s.found_bin];
        }
        shift = sub;
        __syncthreads();
      }
      if (ok || from_row) {
        break;
      }
      from_row = true;
    }
    if (ok) {
      thresh = prefix;
      need_eq = need;
      n_eq = count;
    } else {
      // Not reached: a histogram counted here always matches the row. The
      // ordered pass then writes the row's first n_sel groups.
      thresh = 0;
      need_eq = n_sel;
      in_smem = false;
    }
  }

  // Selected groups' tokens, packed, in group order.
  const int64_t b = load_index(p.row_batch, p.rb64, r);
  const int32_t* __restrict__ page_row = p.pages + b * p.pages_stride;
  const int64_t off = p.ragged != nullptr ? p.ragged[r] : 0;
  int32_t* out = p.out + r * p.out_stride;
  // A group's kRatio tokens are in one page.
  static_assert(kPageTokens % kRatio == 0);
  auto slot_of = [&](int64_t tok) -> int64_t {
    if (p.ragged != nullptr) {
      return off + tok;
    }
    int64_t col = tok / kPageTokens;
    col = col < p.num_pages - 1 ? col : p.num_pages - 1;
    return static_cast<int64_t>(__ldg(page_row + col)) * kPageTokens + tok % kPageTokens;
  };
  auto write_group = [&](int dst, int g) {
    const int32_t v0 = static_cast<int32_t>(slot_of(static_cast<int64_t>(g) * kRatio));
    *reinterpret_cast<int4*>(out + 4 * dst) = make_int4(v0, v0 + 1, v0 + 2, v0 + 3);
  };

  if (!take_all && fast && in_smem) {
    // From the candidates: thread t owns bitmap words [kSelWords * t, +kSelWords).
    uint32_t* bits = reinterpret_cast<uint32_t*>(s.hist);
    const int n = s.n_cand;
    for (int i = t; i < kBins; i += kSelThreads) {
      bits[i] = 0;
    }
    __syncthreads();
    for (int i = t; i < n; i += kSelThreads) {
      if (s.cand[i] == thresh) {
        const int g = s.cand_idx[i];
        atomicOr(&bits[g >> 5], 1u << (g & 31));
      }
    }
    __syncthreads();
    if (need_eq < n_eq) {
      // Keep the need_eq tied groups with the lowest indices.
      uint32_t w[kSelWords];
      int local = 0;
#pragma unroll
      for (int j = 0; j < kSelWords; ++j) {
        w[j] = bits[kSelWords * t + j];
        local += __popc(w[j]);
      }
      int32_t total;
      int before = block_scan(local, s.warp_sum, total) - local;
#pragma unroll
      for (int j = 0; j < kSelWords; ++j) {
        const int keep = need_eq - before;
        before += __popc(w[j]);
        if (keep <= 0) {
          w[j] = 0;
        } else if (keep < __popc(w[j])) {
          uint32_t high = w[j];
          for (int q = 0; q < keep; ++q) {
            high &= high - 1;  // drop the lowest set bit
          }
          w[j] ^= high;
        }
        bits[kSelWords * t + j] = w[j];
      }
      __syncthreads();
    }
    for (int i = t; i < n; i += kSelThreads) {
      if (s.cand[i] > thresh) {
        const int g = s.cand_idx[i];
        atomicOr(&bits[g >> 5], 1u << (g & 31));
      }
    }
    __syncthreads();
    uint32_t w[kSelWords];
    int local = 0;
#pragma unroll
    for (int j = 0; j < kSelWords; ++j) {
      w[j] = bits[kSelWords * t + j];
      local += __popc(w[j]);
    }
    int32_t total;
    int dst = block_scan(local, s.warp_sum, total) - local;
    // The picked groups in order into s.cand (free now), then written with
    // consecutive threads on consecutive outputs: a dense bitmap would
    // otherwise leave the writes to a few threads.
#pragma unroll
    for (int j = 0; j < kSelWords; ++j) {
      for (uint32_t x = w[j]; x != 0; x &= x - 1) {
        s.cand[dst++] = 32 * (kSelWords * t + j) + __ffs(x) - 1;
      }
    }
    __syncthreads();
    for (int i = t; i < n_sel; i += kSelThreads) {
      write_group(i, static_cast<int>(s.cand[i]));
    }
  } else {
    // Ordered pass over the row. A thread's kSelPer groups are the tokens of
    // one page: one page-table read per step, issued with the score loads.
    static_assert(kSelPer * kRatio == kPageTokens);
    int gt_run = 0;
    int eq_run = 0;
    for (int base = 0; base < u; base += kSelChunk) {
      const int i = base + kSelPer * t;
      float4 v[kSelVec];
      if (take_all) {
#pragma unroll
        for (int j = 0; j < kSelVec; ++j) {
          v[j] = make_float4(0.f, 0.f, 0.f, 0.f);
        }
      } else {
        load_scores(row, i, u, v);
      }
      const int64_t slot0 = i < u ? slot_of(static_cast<int64_t>(i) * kRatio) : 0;
      uint32_t gt_bits = 0, eq_bits = 0;
#pragma unroll
      for (int e = 0; e < kSelPer; ++e) {
        if (i + e < u) {
          const uint32_t key = order_key(elem(v[e / 4], e % 4));
          if (take_all || key > thresh) {
            gt_bits |= 1u << e;
          } else if (key == thresh) {
            eq_bits |= 1u << e;
          }
        }
      }
      const int32_t local = __popc(gt_bits) | (__popc(eq_bits) << 16);
      int32_t total;
      const int32_t incl = block_scan(local, s.warp_sum, total);
      if ((gt_bits | eq_bits) != 0) {
        int gt_before = gt_run + ((incl - local) & 0xffff);
        int eq_before = eq_run + ((incl - local) >> 16);
#pragma unroll
        for (int e = 0; e < kSelPer; ++e) {
          const bool gt = (gt_bits >> e) & 1u;
          const bool eq = (eq_bits >> e) & 1u;
          const int dst = gt_before + min(eq_before, need_eq);
          if ((gt || (eq && eq_before < need_eq)) && dst < n_sel) {
            const int32_t v0 = static_cast<int32_t>(slot0 + e * kRatio);
            *reinterpret_cast<int4*>(out + 4 * dst) = make_int4(v0, v0 + 1, v0 + 2, v0 + 3);
          }
          gt_before += gt;
          eq_before += eq;
        }
      }
      gt_run += total & 0xffff;
      eq_run += total >> 16;
    }
  }
  // The row's own group up to the row, then -1. This is the group of the row's
  // position even when the scores stop before it (num_groups below the row's
  // group): num_groups bounds the groups scored, not the row.
  const int tail0 = kRatio * n_sel;
  for (int i = tail0 + t; i < p.topk; i += kSelThreads) {
    int32_t v = -1;
    if (i < tail0 + kRatio) {
      const int64_t tok = upper * kRatio + (i - tail0);
      if (pos >= 0 && tok <= pos) {
        v = static_cast<int32_t>(slot_of(tok));
      }
    }
    out[i] = v;
  }
}

// ---------------------------------------------------------------------------
// Host side.
// ---------------------------------------------------------------------------

inline bool is_int64(const tvm::ffi::TensorView& t) {
  return t.dtype().code == kDLInt && t.dtype().bits == 64;
}

inline void check_index(const tvm::ffi::TensorView& t, const char* what) {
  CHECK_HOST(t.dtype().code == kDLInt && (t.dtype().bits == 32 || t.dtype().bits == 64))
      << what << " must be int32 or int64";
}

// An index vector the kernels read without a stride.
inline void check_vector(const tvm::ffi::TensorView& t, const char* what) {
  check_index(t, what);
  CHECK_HOST(t.dim() == 1 && (t.size(0) <= 1 || t.stride(0) == 1)) << what << " must be a packed [N] vector";
}

inline RopePos make_rope(const tvm::ffi::TensorView& pos, int64_t s1, int64_t s2, bool mrope) {
  check_index(pos, "positions");
  if (mrope) {
    CHECK_HOST(pos.dim() == 2 && pos.size(1) >= 3) << "MRoPE positions must be [N, 3]";
  } else {
    CHECK_HOST(pos.dim() == 1) << "positions must be [N]";
  }
  RopePos rp{};
  rp.ptr = pos.data_ptr();
  rp.is64 = is_int64(pos);
  rp.mrope = mrope;
  rp.s1 = static_cast<int32_t>(s1);
  rp.s2 = static_cast<int32_t>(s2);
  rp.stride_r = pos.stride(0);
  rp.stride_a = mrope ? pos.stride(1) : 0;
  return rp;
}

inline PoolWeights make_weights(
    const tvm::ffi::TensorView& wv_t,
    const tvm::ffi::TensorView& wg_t,
    const tvm::ffi::TensorView& ape,
    const tvm::ffi::TensorView& key_norm,
    const tvm::ffi::TensorView& inv_freq,
    double eps) {
  using namespace host;
  for (const auto* t : {&wv_t, &wg_t}) {
    TensorMatcher({kDim, kDim}).with_dtype<fp32_t>().with_strides({kDim, 1}).verify(*t);
  }
  TensorMatcher({kRatio, kDim}).with_dtype<fp32_t>().with_strides({kDim, 1}).verify(ape);
  TensorMatcher({kDim}).with_dtype<fp32_t>().with_strides({1}).verify(key_norm);
  TensorMatcher({kPairs}).with_dtype<fp32_t>().with_strides({1}).verify(inv_freq);
  return PoolWeights{
      .wv_t = static_cast<const float*>(wv_t.data_ptr()),
      .wg_t = static_cast<const float*>(wg_t.data_ptr()),
      .ape = static_cast<const float*>(ape.data_ptr()),
      .key_norm = static_cast<const float*>(key_norm.data_ptr()),
      .inv_freq = static_cast<const float*>(inv_freq.data_ptr()),
      .eps = static_cast<float>(eps),
  };
}

inline void check_buf(const tvm::ffi::TensorView& buf, int64_t page_size) {
  using namespace host;
  CHECK_HOST(page_size == kPageTokens) << "Hy4 SM70 index cache needs 64-token pages";
  // Key tiles are read 16 bytes at a time.
  TensorMatcher({-1, -1})  //
      .with_dtype<uint8_t>()
      .with_strides({-1, 1})
      .with_device<kDLCUDA>()
      .ensure_alignment(16)
      .verify(buf);
  CHECK_HOST(buf.size(1) >= kPageTokens * kRowBytes) << "index cache page too small";
}

template <typename T>
inline void launch_query(const QueryParams& p, int64_t rows, const host::SymbolicDType& hw_dtype, DLDevice device) {
  const auto blocks = static_cast<uint32_t>(rows);
  if (hw_dtype.is_type<fp32_t>()) {
    host::LaunchKernel(blocks, kDim, device)(query_kernel<T, fp32_t>, p);
  } else if (hw_dtype.is_type<fp16_t>()) {
    host::LaunchKernel(blocks, kDim, device)(query_kernel<T, fp16_t>, p);
  } else {
    host::LaunchKernel(blocks, kDim, device)(query_kernel<T, bf16_t>, p);
  }
}

struct Hy4IndexPoolSm70 {
  static void pool_write(
      tvm::ffi::TensorView raw,
      tvm::ffi::TensorView members,
      tvm::ffi::TensorView first_pos,
      tvm::ffi::TensorView write_slots,
      tvm::ffi::TensorView wv_t,
      tvm::ffi::TensorView wg_t,
      tvm::ffi::TensorView ape,
      tvm::ffi::TensorView key_norm,
      double eps,
      tvm::ffi::TensorView inv_freq,
      tvm::ffi::TensorView buf,
      int64_t page_size,
      bool mrope,
      int64_t s1,
      int64_t s2) {
    using namespace host;
    auto G = SymbolicSize{"groups"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({-1, kDim}).with_dtype<fp16_t>().with_strides({kDim, 1}).with_device(device).verify(raw);
    TensorMatcher({G, kRatio}).with_dtype<int64_t>().with_strides({kRatio, 1}).with_device(device).verify(members);
    TensorMatcher({G}).with_dtype<int64_t>().with_strides({1}).with_device(device).verify(write_slots);
    CHECK_HOST(first_pos.size(0) == G.unwrap()) << "first positions must have one entry per group";
    check_buf(buf, page_size);
    const PoolParams params{
        .raw = static_cast<const fp16_t*>(raw.data_ptr()),
        .raw_slots = raw.size(0),
        .members = static_cast<const int64_t*>(members.data_ptr()),
        .write_slots = static_cast<const int64_t*>(write_slots.data_ptr()),
        .pos = make_rope(first_pos, s1, s2, mrope),
        .w = make_weights(wv_t, wg_t, ape, key_norm, inv_freq, eps),
        .buf = static_cast<uint8_t*>(buf.data_ptr()),
        .page_stride = buf.stride(0),
        .groups = G.unwrap(),
    };
    if (params.groups == 0) {
      return;
    }
    const auto blocks = static_cast<uint32_t>((params.groups + kPoolGroups - 1) / kPoolGroups);
    LaunchKernel(blocks, kDim, device.unwrap())(pool_kernel, params);
  }

  static void store_raw(
      tvm::ffi::TensorView key,
      tvm::ffi::TensorView norm,
      double eps,
      tvm::ffi::TensorView out_loc,
      tvm::ffi::TensorView raw) {
    using namespace host;
    auto R = SymbolicSize{"rows"};
    auto dtype = SymbolicDType{};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({R, -1}).with_dtype<fp16_t, bf16_t>(dtype).with_strides({-1, 1}).with_device(device).verify(key);
    CHECK_HOST(key.size(1) >= kDim) << "key rows must hold 128 values";
    TensorMatcher({kDim}).with_dtype<fp32_t>().with_strides({1}).with_device(device).verify(norm);
    TensorMatcher({-1, kDim}).with_dtype<fp16_t>().with_strides({kDim, 1}).with_device(device).verify(raw);
    check_vector(out_loc, "out_loc");
    CHECK_HOST(out_loc.size(0) >= R.unwrap()) << "out_loc shorter than the rows";
    const StoreRawParams params{
        .key = key.data_ptr(),
        .key_stride = key.stride(0),
        .norm = static_cast<const float*>(norm.data_ptr()),
        .eps = static_cast<float>(eps),
        .out_loc = out_loc.data_ptr(),
        .loc64 = is_int64(out_loc),
        .raw = static_cast<fp16_t*>(raw.data_ptr()),
        .raw_slots = raw.size(0),
        .rows = R.unwrap(),
    };
    if (params.rows == 0) {
      return;
    }
    constexpr int kRowsPerBlock = 4;
    const auto blocks = static_cast<uint32_t>((params.rows + kRowsPerBlock - 1) / kRowsPerBlock);
    if (dtype.is_type<fp16_t>()) {
      LaunchKernel(blocks, kRowsPerBlock * kWarp, device.unwrap())(store_raw_kernel<fp16_t>, params);
    } else {
      LaunchKernel(blocks, kRowsPerBlock * kWarp, device.unwrap())(store_raw_kernel<bf16_t>, params);
    }
  }

  static void decode_update(
      tvm::ffi::TensorView key,
      tvm::ffi::TensorView key_norm_w,
      double key_eps,
      tvm::ffi::TensorView out_loc,
      tvm::ffi::TensorView seq_lens,
      tvm::ffi::TensorView pages,
      tvm::ffi::TensorView raw,
      tvm::ffi::TensorView wv_t,
      tvm::ffi::TensorView wg_t,
      tvm::ffi::TensorView ape,
      tvm::ffi::TensorView key_norm,
      double eps,
      tvm::ffi::TensorView inv_freq,
      tvm::ffi::TensorView buf,
      int64_t page_size,
      tvm::ffi::TensorView xyz,
      bool mrope,
      int64_t s1,
      int64_t s2) {
    using namespace host;
    auto R = SymbolicSize{"rows"};
    auto dtype = SymbolicDType{};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({R, -1}).with_dtype<fp16_t, bf16_t>(dtype).with_strides({-1, 1}).with_device(device).verify(key);
    CHECK_HOST(key.size(1) >= kDim) << "key rows must hold 128 values";
    TensorMatcher({kDim}).with_dtype<fp32_t>().with_strides({1}).with_device(device).verify(key_norm_w);
    TensorMatcher({-1}).with_dtype<int32_t>().with_strides({1}).with_device(device).verify(seq_lens);
    TensorMatcher({-1, -1}).with_dtype<int32_t>().with_strides({-1, 1}).with_device(device).verify(pages);
    TensorMatcher({-1, kDim}).with_dtype<fp16_t>().with_strides({kDim, 1}).with_device(device).verify(raw);
    check_vector(out_loc, "out_loc");
    check_buf(buf, page_size);
    const int64_t rows = R.unwrap();
    CHECK_HOST(out_loc.size(0) >= rows && seq_lens.size(0) >= rows && pages.size(0) >= rows)
        << "decode tensors shorter than the rows";
    RopePos xyz_pos{};
    if (mrope) {
      TensorMatcher({-1, 3}).with_dtype<int32_t>().with_strides({3, 1}).with_device(device).verify(xyz);
      xyz_pos = make_rope(xyz, s1, s2, true);
    }
    const DecodeParams params{
        .key = key.data_ptr(),
        .key_stride = key.stride(0),
        .key_norm_w = static_cast<const float*>(key_norm_w.data_ptr()),
        .key_eps = static_cast<float>(key_eps),
        .out_loc = out_loc.data_ptr(),
        .loc64 = is_int64(out_loc),
        .seq_lens = static_cast<const int32_t*>(seq_lens.data_ptr()),
        .pages = static_cast<const int32_t*>(pages.data_ptr()),
        .pages_stride = pages.stride(0),
        .num_pages = static_cast<int32_t>(pages.size(1)),
        .raw = static_cast<fp16_t*>(raw.data_ptr()),
        .raw_slots = raw.size(0),
        .xyz = xyz_pos,
        .w = make_weights(wv_t, wg_t, ape, key_norm, inv_freq, eps),
        .buf = static_cast<uint8_t*>(buf.data_ptr()),
        .page_stride = buf.stride(0),
    };
    if (rows == 0) {
      return;
    }
    CHECK_HOST(pages.size(1) > 0) << "page table has no columns";
    const auto blocks = static_cast<uint32_t>(rows);
    if (dtype.is_type<fp16_t>()) {
      LaunchKernel(blocks, kDim, device.unwrap())(decode_update_kernel<fp16_t>, params);
    } else {
      LaunchKernel(blocks, kDim, device.unwrap())(decode_update_kernel<bf16_t>, params);
    }
  }

  static void query(
      tvm::ffi::TensorView q,
      tvm::ffi::TensorView pos,
      tvm::ffi::TensorView hw,
      tvm::ffi::TensorView inv_freq,
      tvm::ffi::TensorView q_out,
      tvm::ffi::TensorView w_out,
      tvm::ffi::TensorView hist,
      double weight_scale,
      bool mrope,
      int64_t s1,
      int64_t s2) {
    using namespace host;
    auto R = SymbolicSize{"rows"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    auto q_dtype = SymbolicDType{};
    auto hw_dtype = SymbolicDType{};
    TensorMatcher({R, kHeads * kDim})
        .with_dtype<fp16_t, bf16_t>(q_dtype)
        .with_strides({-1, 1})
        .with_device(device)
        .verify(q);
    TensorMatcher({R, kHeads})
        .with_dtype<fp32_t, fp16_t, bf16_t>(hw_dtype)
        .with_strides({-1, 1})
        .with_device(device)
        .verify(hw);
    TensorMatcher({kPairs}).with_dtype<fp32_t>().with_strides({1}).with_device(device).verify(inv_freq);
    TensorMatcher({R, kHeads, kDim})
        .with_dtype<fp16_t>()
        .with_strides({kHeads * kDim, kDim, 1})
        .with_device(device)
        .verify(q_out);
    TensorMatcher({R, kHeads}).with_dtype<fp32_t>().with_strides({kHeads, 1}).with_device(device).verify(w_out);
    int32_t* hist_ptr = nullptr;
    if (hist.numel() != 0) {
      TensorMatcher({R, kBins}).with_dtype<int32_t>().with_strides({kBins, 1}).with_device(device).verify(hist);
      hist_ptr = static_cast<int32_t*>(hist.data_ptr());
    }
    CHECK_HOST(pos.size(0) >= R.unwrap()) << "positions shorter than the rows";
    const QueryParams params{
        .q = q.data_ptr(),
        .q_stride = q.stride(0),
        .pos = make_rope(pos, s1, s2, mrope),
        .hw = hw.data_ptr(),
        .hw_stride = hw.stride(0),
        .inv_freq = static_cast<const float*>(inv_freq.data_ptr()),
        .q_out = static_cast<fp16_t*>(q_out.data_ptr()),
        .w_out = static_cast<float*>(w_out.data_ptr()),
        .hist = hist_ptr,
        .weight_scale = static_cast<float>(weight_scale),
    };
    if (R.unwrap() == 0) {
      return;
    }
    if (q_dtype.is_type<fp16_t>()) {
      launch_query<fp16_t>(params, R.unwrap(), hw_dtype, device.unwrap());
    } else {
      launch_query<bf16_t>(params, R.unwrap(), hw_dtype, device.unwrap());
    }
  }

  template <int kRows>
  static void launch_logits(dim3 grid, DLDevice device, const LogitsParams& params) {
    using Cfg = LogitsCfg<kRows>;
    const std::size_t smem = params.hist != nullptr ? Cfg::kHistBytes : 0;
    if (sizeof(LogitsSmem<kRows>) + smem > 48 * 1024) {
      // Past 48 KB a kernel needs a per-device opt-in; the call is cheap.
      host::RuntimeDeviceCheck(
          ::cudaFuncSetAttribute(
              logits_kernel<kRows>, ::cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)));
    }
    host::LaunchKernel(grid, Cfg::kThreads, device, smem)(logits_kernel<kRows>, params);
  }

  static void logits(
      tvm::ffi::TensorView q,
      tvm::ffi::TensorView w,
      tvm::ffi::TensorView pages,
      tvm::ffi::TensorView row_batch,
      tvm::ffi::TensorView pos,
      tvm::ffi::TensorView buf,
      int64_t page_size,
      tvm::ffi::TensorView out,
      tvm::ffi::TensorView hist,
      int64_t rows_per_block,
      int64_t blocks_x) {
    using namespace host;
    auto R = SymbolicSize{"rows"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({R, kHeads, kDim})
        .with_dtype<fp16_t>()
        .with_strides({kHeads * kDim, kDim, 1})
        .with_device(device)
        .verify(q);
    TensorMatcher({R, kHeads}).with_dtype<fp32_t>().with_strides({kHeads, 1}).with_device(device).verify(w);
    TensorMatcher({-1, -1}).with_dtype<int32_t>().with_strides({-1, 1}).with_device(device).verify(pages);
    TensorMatcher({R, -1}).with_dtype<fp32_t>().with_strides({-1, 1}).with_device(device).verify(out);
    TensorMatcher({-1}).with_device(device).verify(row_batch);
    TensorMatcher({-1}).with_device(device).verify(pos);
    check_vector(row_batch, "row_to_batch");
    check_vector(pos, "positions");
    check_buf(buf, page_size);
    const int64_t rows = R.unwrap();
    CHECK_HOST(row_batch.size(0) >= rows && pos.size(0) >= rows) << "row tensors shorter than the rows";
    CHECK_HOST(rows_per_block >= 1 && rows_per_block <= 8) << "rows_per_block must be 1..8";
    CHECK_HOST(blocks_x >= 1 && blocks_x < 65536) << "blocks_x out of range";
    int32_t* hist_ptr = nullptr;
    if (hist.numel() != 0) {
      TensorMatcher({R, kBins}).with_dtype<int32_t>().with_strides({kBins, 1}).with_device(device).verify(hist);
      hist_ptr = static_cast<int32_t*>(hist.data_ptr());
    }
    const int64_t num_groups = out.size(1);
    CHECK_HOST(num_groups < (int64_t{1} << 30)) << "too many groups";
    CHECK_HOST(rows < (int64_t{1} << 31)) << "too many rows";
    const LogitsParams params{
        .q = static_cast<const fp16_t*>(q.data_ptr()),
        .w = static_cast<const float*>(w.data_ptr()),
        .pages = static_cast<const int32_t*>(pages.data_ptr()),
        .pages_stride = pages.stride(0),
        .num_pages = static_cast<int32_t>(pages.size(1)),
        .row_batch = row_batch.data_ptr(),
        .rb64 = is_int64(row_batch),
        .pos = pos.data_ptr(),
        .pos64 = is_int64(pos),
        .buf = static_cast<const uint8_t*>(buf.data_ptr()),
        .page_stride = buf.stride(0),
        .out = static_cast<float*>(out.data_ptr()),
        .out_stride = out.stride(0),
        .num_groups = static_cast<int32_t>(num_groups),
        .hist = hist_ptr,
        .rows = static_cast<int32_t>(rows),
        .rows_per_block = static_cast<int32_t>(rows_per_block),
    };
    if (rows == 0 || num_groups == 0) {
      return;
    }
    CHECK_HOST(pages.size(1) > 0) << "page table has no columns";
    const dim3 grid(
        static_cast<uint32_t>(blocks_x), static_cast<uint32_t>((rows + rows_per_block - 1) / rows_per_block));
    CHECK_HOST(grid.y < 65536) << "too many row blocks";
    if (rows_per_block == 1) {
      launch_logits<1>(grid, device.unwrap(), params);
    } else {
      launch_logits<8>(grid, device.unwrap(), params);
    }
  }

  static void select(
      tvm::ffi::TensorView logits,
      tvm::ffi::TensorView hist,
      tvm::ffi::TensorView pos,
      tvm::ffi::TensorView pages,
      tvm::ffi::TensorView row_batch,
      tvm::ffi::TensorView ragged,
      tvm::ffi::TensorView out,
      int64_t k) {
    using namespace host;
    auto R = SymbolicSize{"rows"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({R, -1}).with_dtype<fp32_t>().with_strides({-1, 1}).with_device(device).verify(logits);
    TensorMatcher({R, -1}).with_dtype<int32_t>().with_strides({-1, 1}).with_device(device).verify(out);
    TensorMatcher({-1, -1}).with_dtype<int32_t>().with_strides({-1, 1}).with_device(device).verify(pages);
    TensorMatcher({-1}).with_device(device).verify(row_batch);
    TensorMatcher({-1}).with_device(device).verify(pos);
    check_vector(row_batch, "row_to_batch");
    check_vector(pos, "positions");
    const int64_t rows = R.unwrap();
    CHECK_HOST(row_batch.size(0) >= rows && pos.size(0) >= rows) << "row tensors shorter than the rows";
    CHECK_HOST(logits.stride(0) % 4 == 0 && reinterpret_cast<uintptr_t>(logits.data_ptr()) % 16 == 0)
        << "score rows must be 16-byte aligned";
    CHECK_HOST(out.stride(0) % 4 == 0 && reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0)
        << "index rows must be 16-byte aligned";
    CHECK_HOST(k >= 1 && kRatio * (k + 1) <= out.size(1)) << "topk too small for k groups plus the own group";
    const int32_t* hist_ptr = nullptr;
    if (hist.numel() != 0) {
      TensorMatcher({R, kBins}).with_dtype<int32_t>().with_strides({kBins, 1}).with_device(device).verify(hist);
      hist_ptr = static_cast<const int32_t*>(hist.data_ptr());
    }
    const int64_t* ragged_ptr = nullptr;
    if (ragged.numel() != 0) {
      TensorMatcher({-1}).with_dtype<int64_t>().with_strides({1}).with_device(device).verify(ragged);
      CHECK_HOST(ragged.size(0) >= rows) << "ragged offsets shorter than the rows";
      ragged_ptr = static_cast<const int64_t*>(ragged.data_ptr());
    }
    const SelectParams params{
        .logits = static_cast<const float*>(logits.data_ptr()),
        .logits_stride = logits.stride(0),
        .num_groups = static_cast<int32_t>(logits.size(1)),
        .hist = hist_ptr,
        .pos = pos.data_ptr(),
        .pos64 = is_int64(pos),
        .pages = static_cast<const int32_t*>(pages.data_ptr()),
        .pages_stride = pages.stride(0),
        .num_pages = static_cast<int32_t>(pages.size(1)),
        .row_batch = row_batch.data_ptr(),
        .rb64 = is_int64(row_batch),
        .ragged = ragged_ptr,
        .out = static_cast<int32_t*>(out.data_ptr()),
        .out_stride = out.stride(0),
        .topk = static_cast<int32_t>(out.size(1)),
        .k = static_cast<int32_t>(k),
    };
    if (rows == 0) {
      return;
    }
    CHECK_HOST(pages.size(1) > 0) << "page table has no columns";
    LaunchKernel(static_cast<uint32_t>(rows), kSelThreads, device.unwrap())(select_kernel, params);
  }
};

}  // namespace sglang::hy4_indexpool_sm70
