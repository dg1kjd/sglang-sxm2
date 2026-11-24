// SPDX-License-Identifier: Apache-2.0
// Hy4-flash routed experts on Volta (SM70) at TP8: NVFP4 weights in the SM70
// Marlin layout, read in place, for any number of tokens.
//
// Shapes: hidden (routed latent) 1536, per-rank intermediate 192, top-16.
//
// Batch invariance: a token's output bits depend only on its own row, ids and
// weights, never on how many tokens share the launch, which tokens share an
// expert, or which kernel (decode or prefill) runs it. Every value goes
// through the same fixed sequence of fp32 operations, set by the shapes alone:
//
//   gate_up[c] for c < 384 (fp32, then one fp16 rounding):
//     P[s][w] = HMMA chain from 0 over k-tiles 24s + 6w .. 24s + 6w + 5,
//               4 k per mma.m8n8k4 step, k ascending (s < 4 splits, w < 4)
//     S[s]    = ((P[s][0] + P[s][1]) + P[s][2]) + P[s][3]
//     G       = ((S[0] + S[1]) + S[2]) + S[3]
//     gu[c]   = fp16(G * g13[e])
//   act[j] = swiglu(gu[j], gu[192 + j]) for j < 192 (swiglu_limit_func on fp16)
//   row[c] for c < 1536:
//     Q[w]    = HMMA chain from 0 over k-tiles 6w .. 6w + 5 (w < 2)
//     row[c]  = fp16((float(Q[0] + Q[1]) * topk_weight) * g2[e])
//   out[c] = fp16(((((0 + row_0) + row_1) + ...) + row_15) * routed_scale),
//            slots in top-k order, an invalid id adding +0.
//
// The decode kernels (1..8 tokens) put the K splits in separate CTAs for
// bandwidth; the prefill kernels run the splits in one CTA over blocks of 8
// routes of one expert. Both call the same inner functions in the same order.
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <algorithm>
#include <cstdint>
#include <cuda_fp16.h>

namespace sglang {

namespace device::sm70_hy4_moe::ptx {

// D = A * B + D with A 8x4 row-major, B 4x8 row-major, f32 accumulators.
// Per quad pair q = (lane % 16) / 4, j = lane % 4, h = lane / 16:
//   A: the lane holds A[j + 4h][0..3]; B: the lane holds B[j][4h..4h+3];
//   d[i] = D[(lane & 1) + (i & 2) + 4h][(i & 4) + (lane & 2) + (i & 1)].
SGL_DEVICE void
mma_sync_m8n8k4_row_row_f32_f16_f16_f32(float (&d)[8], uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m8n8k4.row.row.f32.f16.f16.f32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
      : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

}  // namespace device::sm70_hy4_moe::ptx

namespace sm70_hy4_moe {

inline constexpr int kWarp = 32;
inline constexpr int kHidden = 1536;
inline constexpr int kInter = 192;
inline constexpr int kGateUp = 2 * kInter;
inline constexpr int kTopK = 16;
inline constexpr int kMaxTokens = 8;  // decode kernels
inline constexpr int kMaxRoutes = kMaxTokens * kTopK;
inline constexpr int kMaskWords = kMaxRoutes / kWarp;
inline constexpr int kMmaRows = 8;     // A rows of one mma.m8n8k4
inline constexpr int kKTile = 16;      // Marlin k-tile and NVFP4 scale group
inline constexpr int kWordCols = 8;    // one int32 word = 8 fp4 columns at one k
inline constexpr int kChunkWords = 8;  // words per warp column chunk
inline constexpr int kChunkCols = kChunkWords * kWordCols;
inline constexpr int kW13KTiles = kHidden / kKTile;
inline constexpr int kW2KTiles = kInter / kKTile;
inline constexpr int kW13Chunks = kGateUp / kChunkCols;
inline constexpr int kW2Chunks = kHidden / kChunkCols;
// The reduction order (see the top of the file). Fixed: it must not depend on
// the token count or the launch shape.
inline constexpr int kSplit = 4;    // gate_up K splits
inline constexpr int kKWarps1 = 4;  // gate_up K slices per split
inline constexpr int kKWarps2 = 2;  // down K slices
inline constexpr int kSplitTiles = kW13KTiles / kSplit;
inline constexpr int kTiles1 = kSplitTiles / kKWarps1;
inline constexpr int kTiles2 = kW2KTiles / kKWarps2;
inline constexpr int kSplitK = kSplitTiles * kKTile;
// Marlin's column-tile grouping: 4 for N % 256 == 0, else 2 (N = 384 here).
inline constexpr int kW13GroupTiles = 2;
inline constexpr int kW2GroupTiles = 4;
inline constexpr int kW13WordsPerTile = 2 * kGateUp;
inline constexpr int kW2WordsPerTile = 2 * kHidden;
inline constexpr int64_t kW13WordsPerExpert = int64_t{kW13KTiles} * kW13WordsPerTile;
inline constexpr int64_t kW2WordsPerExpert = int64_t{kW2KTiles} * kW2WordsPerTile;
// Shared-memory row strides: +4 halves put the 8 A rows of one k step on
// distinct banks and keep 8-byte alignment.
inline constexpr int kXStride = kSplitK + 4;
inline constexpr int kAStride = kInter + 4;

static_assert(kGateUp % (64 * kW13GroupTiles) == 0 && kHidden % (64 * kW2GroupTiles) == 0);
static_assert(kChunkWords * kW13Chunks == kGateUp / kWordCols && kChunkWords * kW2Chunks == kHidden / kWordCols);
static_assert(kSplit * kSplitTiles == kW13KTiles && kKWarps1 * kTiles1 == kSplitTiles);
static_assert(kKWarps2 * kTiles2 == kW2KTiles);

// Word offset of (k-tile kt, k row lk in the tile, chunk word wl) inside one
// expert. Marlin stores, per k-tile, column groups of 64 * kGroupTiles columns;
// inside a group, row lk holds 8 * kGroupTiles words, word wi = v * kGroupTiles + s
// covering columns s * 64 + v * 8 + 0..7. Chunk c of the group is words 8c..8c+7:
// one 32-byte sector per k row.
template <int kN, int kGroupTiles>
SGL_DEVICE int word_offset(int kt, int chunk, int lk, int wl) {
  const int g = chunk / kGroupTiles;
  const int c = chunk % kGroupTiles;
  return kt * (2 * kN) + g * (128 * kGroupTiles) + lk * (8 * kGroupTiles) + 8 * c + wl;
}

// First of the 8 columns that chunk word wl holds.
template <int kGroupTiles>
SGL_DEVICE int word_col(int chunk, int wl) {
  const int g = chunk / kGroupTiles;
  const int wi = 8 * (chunk % kGroupTiles) + wl;
  return g * 64 * kGroupTiles + (wi % kGroupTiles) * 64 + (wi / kGroupTiles) * 8;
}

// Same bit operations as dequant_fp4x8 in sm70_nvfp4_moe_decode.cu: values[p]
// holds columns (2p, 2p + 1) as fp16 code * 2^-14, exact.
SGL_DEVICE void dequant_fp4x8(uint32_t packed, uint32_t (&values)[4]) {
  const uint32_t even = ((packed << 1) & 0x0e0e0e0eu) | ((packed << 4) & 0x80808080u);
  packed >>= 4;
  const uint32_t odd = ((packed << 1) & 0x0e0e0e0eu) | ((packed << 4) & 0x80808080u);
  values[0] = __byte_perm(even, 0, 0x2404);
  values[1] = __byte_perm(odd, 0, 0x2404);
  values[2] = __byte_perm(even, 0, 0x3414);
  values[3] = __byte_perm(odd, 0, 0x3414);
}

// SM70 Marlin scale bytes for 8 columns, stored in [0, 2, 1, 3] order per 4;
// byte << 7 is the fp16 of scale * 128. Pair p matches dequant_fp4x8's values[p]
// (same as expand_scales8 in sm70_nvfp4_moe_decode.cu).
SGL_DEVICE void expand_scales8(uint2 raw, uint32_t (&out)[4]) {
  out[0] = __byte_perm(raw.x, 0, 0x4240) << 7;
  out[1] = __byte_perm(raw.x, 0, 0x4341) << 7;
  out[2] = __byte_perm(raw.y, 0, 0x4240) << 7;
  out[3] = __byte_perm(raw.y, 0, 0x4341) << 7;
}

// fp4 value (2 significant bits) times scale (4 significant bits) is exact in fp16.
SGL_DEVICE uint32_t hmul2_bits(uint32_t a, uint32_t b) {
  const __half2 r = __hmul2(*reinterpret_cast<const __half2*>(&a), *reinterpret_cast<const __half2*>(&b));
  return *reinterpret_cast<const uint32_t*>(&r);
}

SGL_DEVICE __half clamp_max_nan(__half v, __half hi) {
  return __hisnan(v) ? v : (__hlt(hi, v) ? hi : v);
}

SGL_DEVICE __half clamp_nan(__half v, __half lo, __half hi) {
  return __hisnan(v) ? v : clamp_max_nan(__hlt(v, lo) ? lo : v, hi);
}

// swiglu_limit_func on fp16: clamps, torch's silu (x / (1 + expf(-x)) in fp32,
// rounded to fp16), then the fp16 product. Same code as swiglu_clamp_kernel in
// moe/sm70_moe_glue.cuh; both are built with --fmad=false.
SGL_DEVICE __half swiglu(__half g, __half u, __half limit, bool clamp) {
  if (clamp) {
    g = clamp_max_nan(g, limit);
    u = clamp_nan(u, __hneg(limit), limit);
  }
  const float gf = __half2float(g);
  const float s = __fdiv_rn(gf, __fadd_rn(1.0f, expf(-gf)));
  return __float2half_rn(__fmul_rn(__half2float(__float2half_rn(s)), __half2float(u)));
}

// The gate_up store: fp16(G * global).
SGL_DEVICE __half gate_up_round(float g, float g13e) {
  return __float2half_rn(__fmul_rn(g, g13e));
}

// The down store: fp16((sum * topk weight) * global).
SGL_DEVICE __half down_round(float v, float w, float g2e) {
  return __float2half_rn(__fmul_rn(__fmul_rn(v, w), g2e));
}

// Loads one warp's weights for kTiles k-tiles from k-tile kt0 of a 64-column
// chunk: lane (j, wl) takes k rows 4a + j of each tile and the 8 columns of
// chunk word wl, plus the 8 scale bytes of those columns per tile.
template <int kN, int kGroupTiles, int kTiles>
SGL_DEVICE void load_weights(
    uint32_t (&wv)[kTiles][4],
    uint2 (&sv)[kTiles],
    const uint32_t* w,
    const uint8_t* s,
    int expert,
    int kt0,
    int chunk,
    int lane) {
  constexpr int kKTiles = kN == kGateUp ? kW13KTiles : kW2KTiles;
  const int q = (lane >> 2) & 3;
  const int j = lane & 3;
  const int h = lane >> 4;
  const int wl = 2 * q + h;
  const uint32_t* wp =
      w + static_cast<int64_t>(expert) * kKTiles * (2 * kN) + word_offset<kN, kGroupTiles>(kt0, chunk, j, wl);
  const uint8_t* sp = s + static_cast<int64_t>(expert) * kKTiles * kN + kt0 * kN + word_col<kGroupTiles>(chunk, wl);
#pragma unroll
  for (int t = 0; t < kTiles; ++t) {
#pragma unroll
    for (int a = 0; a < 4; ++a) {
      wv[t][a] = __ldg(wp + t * (2 * kN) + a * 4 * (8 * kGroupTiles));
    }
    sv[t] = __ldg(reinterpret_cast<const uint2*>(sp + t * kN));
  }
}

// One warp's HMMA chain over kTiles k-tiles of a 64-column chunk, k ascending,
// from zero. A rows come from shared memory (rows past kRows are zero without
// a read); B is the dequantized, scaled weight. acc[m] covers word columns
// 4m..4m+3.
template <int kRows, int kTiles>
SGL_DEVICE void chunk_mma(
    float (&acc)[2][8], const uint32_t (&wv)[kTiles][4], const uint2 (&sv)[kTiles], const __half* arow, bool arow_ok) {
#pragma unroll
  for (int i = 0; i < 8; ++i)
    acc[0][i] = 0.0f, acc[1][i] = 0.0f;
#pragma unroll
  for (int t = 0; t < kTiles; ++t) {
    uint32_t sc[4];
    expand_scales8(sv[t], sc);
#pragma unroll
    for (int a = 0; a < 4; ++a) {
      uint2 av = make_uint2(0u, 0u);
      if (arow_ok) av = *reinterpret_cast<const uint2*>(arow + t * kKTile + 4 * a);
      uint32_t b[4];
      dequant_fp4x8(wv[t][a], b);
#pragma unroll
      for (int p = 0; p < 4; ++p)
        b[p] = hmul2_bits(b[p], sc[p]);
      device::sm70_hy4_moe::ptx::mma_sync_m8n8k4_row_row_f32_f16_f16_f32(acc[0], av.x, av.y, b[0], b[1]);
      device::sm70_hy4_moe::ptx::mma_sync_m8n8k4_row_row_f32_f16_f16_f32(acc[1], av.x, av.y, b[2], b[3]);
    }
  }
}

// Scatter one warp's accumulators into red[row][kRedCols] at its chunk's 64
// local columns (word wl of the chunk at 8 * wl).
template <int kRows, int kRedCols>
SGL_DEVICE void store_fragments(float (*red)[kRedCols], const float (&acc)[2][8], int col0, int lane) {
  const int q = (lane >> 2) & 3;
  const int h = lane >> 4;
#pragma unroll
  for (int m = 0; m < 2; ++m) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int row = (lane & 1) + (i & 2) + 4 * h;
      const int n = (i & 4) + (lane & 2) + (i & 1);
      const int lc = (2 * q + (n >> 2)) * kWordCols + 4 * m + (n & 3);
      if (row < kRows) red[row][col0 + lc] = acc[m][i];
    }
  }
}

// Sum of the kKWarps K slices of 8 consecutive local columns, slice 0 first.
template <int kKWarps, int kRows, int kRedCols>
SGL_DEVICE void sum_slices(float (&v)[8], const float (*red)[kRows][kRedCols], int row, int lc) {
  const float4 lo = *reinterpret_cast<const float4*>(&red[0][row][lc]);
  const float4 hi = *reinterpret_cast<const float4*>(&red[0][row][lc + 4]);
  v[0] = lo.x, v[1] = lo.y, v[2] = lo.z, v[3] = lo.w;
  v[4] = hi.x, v[5] = hi.y, v[6] = hi.z, v[7] = hi.w;
#pragma unroll
  for (int s = 1; s < kKWarps; ++s) {
    const float4 l = *reinterpret_cast<const float4*>(&red[s][row][lc]);
    const float4 u = *reinterpret_cast<const float4*>(&red[s][row][lc + 4]);
    v[0] = __fadd_rn(v[0], l.x), v[1] = __fadd_rn(v[1], l.y), v[2] = __fadd_rn(v[2], l.z), v[3] = __fadd_rn(v[3], l.w);
    v[4] = __fadd_rn(v[4], u.x), v[5] = __fadd_rn(v[5], u.y), v[6] = __fadd_rn(v[6], u.z), v[7] = __fadd_rn(v[7], u.w);
  }
}

// Token t's output columns col..col+7: its 16 route rows summed in slot order
// in fp32 from zero (an invalid id adds +0), times the routed scale, to fp16.
SGL_DEVICE uint4
route_sum8(const __half* cache, const int32_t* ids, int t, int col, int num_experts, float routed_scale) {
  float sum[8] = {};
  for (int k = 0; k < kTopK; ++k) {
    const int route = t * kTopK + k;
    const int e = ids[route];
    uint4 r = make_uint4(0u, 0u, 0u, 0u);
    if (e >= 0 && e < num_experts)
      r = __ldcg(reinterpret_cast<const uint4*>(cache + static_cast<int64_t>(route) * kHidden + col));
    const __half* hr = reinterpret_cast<const __half*>(&r);
#pragma unroll
    for (int c = 0; c < 8; ++c)
      sum[c] = __fadd_rn(sum[c], __half2float(hr[c]));
  }
  uint4 o;
  __half* ho = reinterpret_cast<__half*>(&o);
#pragma unroll
  for (int c = 0; c < 8; ++c)
    ho[c] = __float2half_rn(__fmul_rn(sum[c], routed_scale));
  return o;
}

// ---------------------------------------------------------------------------
// Decode: 1..8 tokens.
// ---------------------------------------------------------------------------

// Routes this CTA serves, as a bit set over route index (token * 16 + slot).
// Without dedupe: only its own route. With dedupe: every route whose id equals
// `expert`, and only the lowest such route's CTA does the work.
template <bool kDedupe, int kThreads>
SGL_DEVICE bool serve_routes(uint32_t* masks, const int32_t* ids, int route, int expert, int num_routes) {
  const int tid = static_cast<int>(threadIdx.x);
  if constexpr (kDedupe) {
    static_assert(kThreads >= kMaxRoutes, "one ballot lane per route");
    if (tid < kMaxRoutes) {
      const bool hit = tid < num_routes && ids[tid] == expert;
      const uint32_t bits = __ballot_sync(0xffffffffu, hit);
      if (tid % kWarp == 0) masks[tid / kWarp] = bits;
    }
    __syncthreads();
    int first = kMaxRoutes;
#pragma unroll
    for (int w = kMaskWords - 1; w >= 0; --w) {
      if (masks[w] != 0u) first = w * kWarp + __ffs(static_cast<int>(masks[w])) - 1;
    }
    return first == route;
  } else {
    if (tid < kMaskWords) masks[tid] = tid == route / kWarp ? 1u << (route % kWarp) : 0u;
    __syncthreads();
    return true;
  }
}

// Bits of token t's 16 routes within the served set.
SGL_DEVICE uint32_t token_routes(const uint32_t* masks, int t) {
  return (masks[t / 2] >> ((t & 1) * kTopK)) & 0xffffu;
}

// gate_up, one CTA per (route, K split, column group): S[split] for every
// token that uses the expert, into partials[route][split][384].
template <int kRows, bool kDedupe, int kColWarps>
__global__ __launch_bounds__(kColWarps* kKWarps1* kWarp) void gate_up_kernel(
    float* __restrict__ partials,
    const __half* __restrict__ x,
    const uint32_t* __restrict__ w13,
    const uint8_t* __restrict__ s13,
    const int32_t* __restrict__ ids,
    int num_tokens,
    int num_experts) {
  constexpr int kThreads = kColWarps * kKWarps1 * kWarp;
  constexpr int kGroups = kW13Chunks / kColWarps;
  constexpr int kRedCols = kColWarps * kChunkCols;
  static_assert(kW13Chunks % kColWarps == 0, "column warps must tile the 6 chunks");
  static_assert(kRows >= 1 && kRows <= kMaxTokens);

  __shared__ uint32_t masks[kMaskWords];
  __shared__ __align__(16) __half xs[kRows][kXStride];
  __shared__ __align__(16) float red[kKWarps1][kRows][kRedCols];

  const int route = static_cast<int>(blockIdx.y);
  const int group = static_cast<int>(blockIdx.x) % kGroups;
  const int split = static_cast<int>(blockIdx.x) / kGroups;
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid / kWarp;
  const int lane = tid % kWarp;
  const int cw = warp % kColWarps;
  const int kw = warp / kColWarps;

  const int expert = ids[route];
  if (expert < 0 || expert >= num_experts) {
    return;  // the route adds zero
  }
  if (!serve_routes<kDedupe, kThreads>(masks, ids, route, expert, num_tokens * kTopK)) {
    return;
  }

  // Issue every weight and scale load of this warp first.
  uint32_t wv[kTiles1][4];
  uint2 sv[kTiles1];
  load_weights<kGateUp, kW13GroupTiles, kTiles1>(
      wv, sv, w13, s13, expert, split * kSplitTiles + kw * kTiles1, group * kColWarps + cw, lane);

  // Stage this split's slice of every token row; rows past num_tokens are zero.
  constexpr int kXVecs = kSplitK / 4;
  for (int e = tid; e < kRows * kXVecs; e += kThreads) {
    const int row = e / kXVecs;
    const int v = e % kXVecs;
    uint2 val = make_uint2(0u, 0u);
    if (row < num_tokens) {
      val = __ldg(reinterpret_cast<const uint2*>(x + static_cast<int64_t>(row) * kHidden + split * kSplitK) + v);
    }
    *reinterpret_cast<uint2*>(&xs[row][4 * v]) = val;
  }
  __syncthreads();

  float acc[2][8];
  const int arow = (lane & 3) + 4 * (lane >> 4);
  const bool arow_ok = arow < kRows;
  chunk_mma<kRows, kTiles1>(acc, wv, sv, &xs[arow_ok ? arow : 0][kw * kTiles1 * kKTile], arow_ok);
  store_fragments<kRows, kRedCols>(red[kw], acc, cw * kChunkCols, lane);
  __syncthreads();

  // One thread per (token, 8-column word): partials[route][split][col..col+7].
  constexpr int kWords = kColWarps * kChunkWords;
  for (int e = tid; e < kRows * kWords; e += kThreads) {
    const int row = e / kWords;
    const int word = e % kWords;
    if (row >= num_tokens) continue;
    uint32_t bits = token_routes(masks, row);
    if (bits == 0u) continue;
    float v[8];
    sum_slices<kKWarps1, kRows, kRedCols>(v, red, row, word * kWordCols);
    const int col = word_col<kW13GroupTiles>(group * kColWarps + word / kChunkWords, word % kChunkWords);
    while (bits != 0u) {
      const int i = row * kTopK + __ffs(static_cast<int>(bits)) - 1;
      bits &= bits - 1u;
      float4* dst = reinterpret_cast<float4*>(partials + (static_cast<int64_t>(i) * kSplit + split) * kGateUp + col);
      dst[0] = make_float4(v[0], v[1], v[2], v[3]);
      dst[1] = make_float4(v[4], v[5], v[6], v[7]);
    }
  }
}

// down, one CTA per (route, column group): G from the splits, the gate_up
// store, SwiGLU, W2, the route rows into cache; the last CTA of each
// (token, column group) writes the route sum.
template <int kRows, bool kDedupe, int kColWarps>
__global__ __launch_bounds__(kColWarps* kKWarps2* kWarp) void down_kernel(
    __half* __restrict__ out,
    __half* __restrict__ cache,
    int32_t* __restrict__ counters,
    const float* __restrict__ partials,
    const uint32_t* __restrict__ w2,
    const uint8_t* __restrict__ s2,
    const float* __restrict__ g13,
    const float* __restrict__ g2,
    const int32_t* __restrict__ ids,
    const float* __restrict__ topk_weights,
    int num_tokens,
    int num_experts,
    __half limit,
    bool clamp,
    float routed_scale) {
  constexpr int kThreads = kColWarps * kKWarps2 * kWarp;
  constexpr int kRedCols = kColWarps * kChunkCols;
  constexpr int kWords = kColWarps * kChunkWords;
  static_assert(kW2Chunks % kColWarps == 0, "column warps must tile the 24 chunks");
  static_assert(kRows >= 1 && kRows <= kMaxTokens);

  __shared__ uint32_t masks[kMaskWords];
  __shared__ int finish[kRows];
  __shared__ __align__(16) __half act[kRows][kAStride];
  __shared__ __align__(16) float red[kKWarps2][kRows][kRedCols];

  const int route = static_cast<int>(blockIdx.y);
  const int group = static_cast<int>(blockIdx.x);
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid / kWarp;
  const int lane = tid % kWarp;
  const int cw = warp % kColWarps;
  const int kw = warp / kColWarps;

  const int expert = ids[route];
  const bool valid = expert >= 0 && expert < num_experts;
  if (valid) {
    if (!serve_routes<kDedupe, kThreads>(masks, ids, route, expert, num_tokens * kTopK)) {
      return;
    }
    uint32_t wv[kTiles2][4];
    uint2 sv[kTiles2];
    load_weights<kHidden, kW2GroupTiles, kTiles2>(wv, sv, w2, s2, expert, kw * kTiles2, group * kColWarps + cw, lane);

    // Activation rows: G = ((S0 + S1) + S2) + S3, the gate_up store, then the
    // clamped SwiGLU. Tokens this CTA does not serve stay zero.
    const float g13e = g13[expert];
    for (int e = tid; e < kRows * kInter; e += kThreads) {
      const int row = e / kInter;
      const int c = e % kInter;
      __half v = __ushort_as_half(0);
      if (row < num_tokens) {
        const uint32_t bits = token_routes(masks, row);
        if (bits != 0u) {
          const int i = row * kTopK + __ffs(static_cast<int>(bits)) - 1;
          const float* p = partials + static_cast<int64_t>(i) * kSplit * kGateUp;
          float gate = __ldcg(p + c);
          float up = __ldcg(p + kInter + c);
#pragma unroll
          for (int s = 1; s < kSplit; ++s) {
            gate = __fadd_rn(gate, __ldcg(p + s * kGateUp + c));
            up = __fadd_rn(up, __ldcg(p + s * kGateUp + kInter + c));
          }
          v = swiglu(gate_up_round(gate, g13e), gate_up_round(up, g13e), limit, clamp);
        }
      }
      act[row][c] = v;
    }
    __syncthreads();

    float acc[2][8];
    const int arow = (lane & 3) + 4 * (lane >> 4);
    const bool arow_ok = arow < kRows;
    chunk_mma<kRows, kTiles2>(acc, wv, sv, &act[arow_ok ? arow : 0][kw * kTiles2 * kKTile], arow_ok);
    store_fragments<kRows, kRedCols>(red[kw], acc, cw * kChunkCols, lane);
    __syncthreads();

    const float g2e = g2[expert];
    for (int e = tid; e < kRows * kWords; e += kThreads) {
      const int row = e / kWords;
      const int word = e % kWords;
      if (row >= num_tokens) continue;
      uint32_t bits = token_routes(masks, row);
      if (bits == 0u) continue;
      float v[8];
      sum_slices<kKWarps2, kRows, kRedCols>(v, red, row, word * kWordCols);
      const int col = word_col<kW2GroupTiles>(group * kColWarps + word / kChunkWords, word % kChunkWords);
      while (bits != 0u) {
        const int i = row * kTopK + __ffs(static_cast<int>(bits)) - 1;
        bits &= bits - 1u;
        const float w = topk_weights[i];
        uint4 o;
        __half* ho = reinterpret_cast<__half*>(&o);
#pragma unroll
        for (int c = 0; c < 8; ++c)
          ho[c] = down_round(v[c], w, g2e);
        *reinterpret_cast<uint4*>(cache + static_cast<int64_t>(i) * kHidden + col) = o;
      }
    }
  } else {
    // No expert: the route adds zero (route_sum8 skips it; the row is zeroed
    // so the workspace holds no stale values).
    serve_routes<false, kThreads>(masks, ids, route, expert, num_tokens * kTopK);
    for (int word = tid; word < kWords; word += kThreads) {
      const int col = word_col<kW2GroupTiles>(group * kColWarps + word / kChunkWords, word % kChunkWords);
      *reinterpret_cast<uint4*>(cache + static_cast<int64_t>(route) * kHidden + col) = make_uint4(0u, 0u, 0u, 0u);
    }
  }

  // The last CTA to finish a (token, column group) sums its 16 route rows.
  __threadfence();
  __syncthreads();
  if (tid == 0) {
    for (int t = 0; t < kRows; ++t) {
      finish[t] = 0;
      if (t >= num_tokens) continue;
      const int n = __popc(token_routes(masks, t));
      if (n == 0) continue;
      const int done = atomicAdd(&counters[t * kW2Chunks + group], n) + n;
      finish[t] = done == kTopK;
    }
    __threadfence();
  }
  __syncthreads();
  for (int t = 0; t < num_tokens && t < kRows; ++t) {
    if (!finish[t]) continue;
    for (int word = tid; word < kWords; word += kThreads) {
      const int col = word_col<kW2GroupTiles>(group * kColWarps + word / kChunkWords, word % kChunkWords);
      *reinterpret_cast<uint4*>(out + static_cast<int64_t>(t) * kHidden + col) =
          route_sum8(cache, ids, t, col, num_experts, routed_scale);
    }
    if (tid == 0) counters[t * kW2Chunks + group] = 0;
  }
}

// ---------------------------------------------------------------------------
// Prefill: any number of tokens. Routes are grouped by expert (order,
// expert_off); a work item is up to kBlocks * 8 routes of one expert, and
// work_off[e] is the first work item of expert e.
// ---------------------------------------------------------------------------

template <int kRoutesPerItem>
SGL_DEVICE bool find_work(
    const int32_t* work_off,
    const int32_t* expert_off,
    int num_experts,
    int item,
    int& expert,
    int& first,
    int& count) {
  if (item >= work_off[num_experts]) return false;
  int lo = 0, hi = num_experts;  // work_off[lo] <= item < work_off[hi]
  while (hi - lo > 1) {
    const int mid = (lo + hi) / 2;
    if (work_off[mid] <= item) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  expert = lo;
  first = expert_off[lo] + (item - work_off[lo]) * kRoutesPerItem;
  count = min(kRoutesPerItem, expert_off[lo + 1] - first);
  return true;
}

// gate_up for one work item and column group: the K splits one after the other
// (the warps' registers hold one split's weights), G kept in shared memory per
// route block, then the gate_up store into gate_up[route][384].
template <int kColWarps, int kBlocks>
__global__ __launch_bounds__(kColWarps* kKWarps1* kWarp) void gate_up_prefill_kernel(
    __half* __restrict__ gate_up,
    const __half* __restrict__ x,
    const uint32_t* __restrict__ w13,
    const uint8_t* __restrict__ s13,
    const float* __restrict__ g13,
    const int32_t* __restrict__ order,
    const int32_t* __restrict__ expert_off,
    const int32_t* __restrict__ work_off,
    int num_experts) {
  constexpr int kThreads = kColWarps * kKWarps1 * kWarp;
  constexpr int kRedCols = kColWarps * kChunkCols;
  constexpr int kWords = kColWarps * kChunkWords;
  constexpr int kRoutes = kBlocks * kMmaRows;
  static_assert(kW13Chunks % kColWarps == 0, "column warps must tile the 6 chunks");

  __shared__ int32_t rows[kRoutes];
  __shared__ __align__(16) __half xs[kMmaRows][kXStride];
  __shared__ __align__(16) float red[kKWarps1][kMmaRows][kRedCols];
  __shared__ __align__(16) float gsum[kBlocks][kMmaRows][kRedCols];

  int expert, first, count;
  if (!find_work<kRoutes>(work_off, expert_off, num_experts, static_cast<int>(blockIdx.x), expert, first, count)) {
    return;
  }
  const int group = static_cast<int>(blockIdx.y);
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid / kWarp;
  const int lane = tid % kWarp;
  const int cw = warp % kColWarps;
  const int kw = warp / kColWarps;
  const int chunk = group * kColWarps + cw;
  const int blocks = (count + kMmaRows - 1) / kMmaRows;
  for (int i = tid; i < kRoutes; i += kThreads)
    rows[i] = i < count ? order[first + i] : -1;
  const int arow = (lane & 3) + 4 * (lane >> 4);
  constexpr int kXVecs = kSplitK / 4;

  for (int split = 0; split < kSplit; ++split) {
    uint32_t wv[kTiles1][4];
    uint2 sv[kTiles1];
    load_weights<kGateUp, kW13GroupTiles, kTiles1>(
        wv, sv, w13, s13, expert, split * kSplitTiles + kw * kTiles1, chunk, lane);
    for (int b = 0; b < blocks; ++b) {
      __syncthreads();  // rows[] written; xs and red free
      for (int e = tid; e < kMmaRows * kXVecs; e += kThreads) {
        const int row = e / kXVecs;
        const int v = e % kXVecs;
        const int route = rows[b * kMmaRows + row];
        uint2 val = make_uint2(0u, 0u);
        if (route >= 0) {
          val = __ldg(
              reinterpret_cast<const uint2*>(x + static_cast<int64_t>(route / kTopK) * kHidden + split * kSplitK) + v);
        }
        *reinterpret_cast<uint2*>(&xs[row][4 * v]) = val;
      }
      __syncthreads();
      float acc[2][8];
      chunk_mma<kMmaRows, kTiles1>(acc, wv, sv, &xs[arow][kw * kTiles1 * kKTile], true);
      store_fragments<kMmaRows, kRedCols>(red[kw], acc, cw * kChunkCols, lane);
      __syncthreads();
      for (int e = tid; e < kMmaRows * kWords; e += kThreads) {
        const int row = e / kWords;
        const int lc = (e % kWords) * kWordCols;
        float v[8];
        sum_slices<kKWarps1, kMmaRows, kRedCols>(v, red, row, lc);
        float* g = &gsum[b][row][lc];
        if (split == 0) {
#pragma unroll
          for (int c = 0; c < 8; ++c)
            g[c] = v[c];
        } else {
#pragma unroll
          for (int c = 0; c < 8; ++c)
            g[c] = __fadd_rn(g[c], v[c]);
        }
      }
    }
  }
  __syncthreads();

  const float g13e = g13[expert];
  for (int e = tid; e < blocks * kMmaRows * kWords; e += kThreads) {
    const int r = e / kWords;
    const int word = e % kWords;
    const int route = rows[r];
    if (route < 0) continue;
    const float* g = &gsum[r / kMmaRows][r % kMmaRows][word * kWordCols];
    uint4 o;
    __half* ho = reinterpret_cast<__half*>(&o);
#pragma unroll
    for (int c = 0; c < 8; ++c)
      ho[c] = gate_up_round(g[c], g13e);
    const int col = word_col<kW13GroupTiles>(group * kColWarps + word / kChunkWords, word % kChunkWords);
    *reinterpret_cast<uint4*>(gate_up + static_cast<int64_t>(route) * kGateUp + col) = o;
  }
}

// down for one work item and column group: SwiGLU of the stored gate_up, W2,
// the route rows into cache[route][1536].
template <int kColWarps, int kBlocks>
__global__ __launch_bounds__(kColWarps* kKWarps2* kWarp) void down_prefill_kernel(
    __half* __restrict__ cache,
    const __half* __restrict__ gate_up,
    const uint32_t* __restrict__ w2,
    const uint8_t* __restrict__ s2,
    const float* __restrict__ g2,
    const float* __restrict__ topk_weights,
    const int32_t* __restrict__ order,
    const int32_t* __restrict__ expert_off,
    const int32_t* __restrict__ work_off,
    int num_experts,
    __half limit,
    bool clamp) {
  constexpr int kThreads = kColWarps * kKWarps2 * kWarp;
  constexpr int kRedCols = kColWarps * kChunkCols;
  constexpr int kWords = kColWarps * kChunkWords;
  constexpr int kRoutes = kBlocks * kMmaRows;
  static_assert(kW2Chunks % kColWarps == 0, "column warps must tile the 24 chunks");

  __shared__ int32_t rows[kRoutes];
  __shared__ __align__(16) __half act[kMmaRows][kAStride];
  __shared__ __align__(16) float red[kKWarps2][kMmaRows][kRedCols];

  int expert, first, count;
  if (!find_work<kRoutes>(work_off, expert_off, num_experts, static_cast<int>(blockIdx.x), expert, first, count)) {
    return;
  }
  const int group = static_cast<int>(blockIdx.y);
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid / kWarp;
  const int lane = tid % kWarp;
  const int cw = warp % kColWarps;
  const int kw = warp / kColWarps;
  const int blocks = (count + kMmaRows - 1) / kMmaRows;

  uint32_t wv[kTiles2][4];
  uint2 sv[kTiles2];
  load_weights<kHidden, kW2GroupTiles, kTiles2>(wv, sv, w2, s2, expert, kw * kTiles2, group * kColWarps + cw, lane);
  for (int i = tid; i < kRoutes; i += kThreads)
    rows[i] = i < count ? order[first + i] : -1;
  const float g2e = g2[expert];
  const int arow = (lane & 3) + 4 * (lane >> 4);
  constexpr int kAVecs = kInter / 4;

  for (int b = 0; b < blocks; ++b) {
    __syncthreads();  // rows[] written; act and red free
    for (int e = tid; e < kMmaRows * kAVecs; e += kThreads) {
      const int row = e / kAVecs;
      const int c = (e % kAVecs) * 4;
      const int route = rows[b * kMmaRows + row];
      uint2 val = make_uint2(0u, 0u);
      if (route >= 0) {
        const __half* gu = gate_up + static_cast<int64_t>(route) * kGateUp;
        const uint2 g = *reinterpret_cast<const uint2*>(gu + c);
        const uint2 u = *reinterpret_cast<const uint2*>(gu + kInter + c);
        const __half* gh = reinterpret_cast<const __half*>(&g);
        const __half* uh = reinterpret_cast<const __half*>(&u);
        __half* vh = reinterpret_cast<__half*>(&val);
#pragma unroll
        for (int i = 0; i < 4; ++i)
          vh[i] = swiglu(gh[i], uh[i], limit, clamp);
      }
      *reinterpret_cast<uint2*>(&act[row][c]) = val;
    }
    __syncthreads();
    float acc[2][8];
    chunk_mma<kMmaRows, kTiles2>(acc, wv, sv, &act[arow][kw * kTiles2 * kKTile], true);
    store_fragments<kMmaRows, kRedCols>(red[kw], acc, cw * kChunkCols, lane);
    __syncthreads();
    for (int e = tid; e < kMmaRows * kWords; e += kThreads) {
      const int row = e / kWords;
      const int word = e % kWords;
      const int route = rows[b * kMmaRows + row];
      if (route < 0) continue;
      float v[8];
      sum_slices<kKWarps2, kMmaRows, kRedCols>(v, red, row, word * kWordCols);
      const float w = topk_weights[route];
      uint4 o;
      __half* ho = reinterpret_cast<__half*>(&o);
#pragma unroll
      for (int c = 0; c < 8; ++c)
        ho[c] = down_round(v[c], w, g2e);
      const int col = word_col<kW2GroupTiles>(group * kColWarps + word / kChunkWords, word % kChunkWords);
      *reinterpret_cast<uint4*>(cache + static_cast<int64_t>(route) * kHidden + col) = o;
    }
  }
}

// One thread per (token, 8 output columns).
__global__ void route_sum_kernel(
    __half* __restrict__ out,
    const __half* __restrict__ cache,
    const int32_t* __restrict__ ids,
    int num_tokens,
    int num_experts,
    float routed_scale) {
  constexpr int kWordsPerRow = kHidden / kWordCols;
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= static_cast<int64_t>(num_tokens) * kWordsPerRow) return;
  const int t = static_cast<int>(idx / kWordsPerRow);
  const int col = static_cast<int>(idx % kWordsPerRow) * kWordCols;
  *reinterpret_cast<uint4*>(out + static_cast<int64_t>(t) * kHidden + col) =
      route_sum8(cache, ids, t, col, num_experts, routed_scale);
}

// ---------------------------------------------------------------------------
// Host entry points.
// ---------------------------------------------------------------------------

struct Weights {
  const uint32_t* w13;
  const uint32_t* w2;
  const uint8_t* s13;
  const uint8_t* s2;
  const float* g13;
  const float* g2;
};

inline Weights verify_common(
    host::SymbolicSize& m,
    host::SymbolicSize& e,
    host::SymbolicDevice& device,
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView x,
    tvm::ffi::TensorView w13,
    tvm::ffi::TensorView w2,
    tvm::ffi::TensorView s13,
    tvm::ffi::TensorView s2,
    tvm::ffi::TensorView g13,
    tvm::ffi::TensorView g2,
    tvm::ffi::TensorView ids,
    tvm::ffi::TensorView topk_weights) {
  using namespace host;
  device.set_options<kDLCUDA>();
  TensorMatcher({m, kHidden})  //
      .with_dtype<fp16_t>()
      .with_device(device)
      .ensure_alignment(16)
      .verify(x)
      .verify(out);
  TensorMatcher({e, kW13KTiles, kW13WordsPerTile})  //
      .with_dtype<int32_t>()
      .with_device(device)
      .verify(w13);
  TensorMatcher({e, kW2KTiles, kW2WordsPerTile})  //
      .with_dtype<int32_t>()
      .with_device(device)
      .verify(w2);
  TensorMatcher({e, kW13KTiles, kGateUp})  //
      .with_dtype<uint8_t, fp8_e4m3_t>()
      .with_device(device)
      .ensure_alignment(8)
      .verify(s13);
  TensorMatcher({e, kW2KTiles, kHidden})  //
      .with_dtype<uint8_t, fp8_e4m3_t>()
      .with_device(device)
      .ensure_alignment(8)
      .verify(s2);
  TensorMatcher({e})  //
      .with_dtype<fp32_t>()
      .with_device(device)
      .verify(g13)
      .verify(g2);
  TensorMatcher({m, kTopK})  //
      .with_dtype<int32_t>()
      .with_device(device)
      .verify(ids);
  TensorMatcher({m, kTopK})  //
      .with_dtype<fp32_t>()
      .with_device(device)
      .verify(topk_weights);
  const int64_t experts = e.unwrap();
  RuntimeCheck(experts >= 1 && experts <= (int64_t{1} << 30), "bad expert count ", experts);
  return Weights{
      static_cast<const uint32_t*>(w13.data_ptr()),
      static_cast<const uint32_t*>(w2.data_ptr()),
      static_cast<const uint8_t*>(s13.data_ptr()),
      static_cast<const uint8_t*>(s2.data_ptr()),
      static_cast<const float*>(g13.data_ptr()),
      static_cast<const float*>(g2.data_ptr()),
  };
}

/// \brief Hy4-flash routed experts for 1..kRows decode tokens on SM70.
/// \tparam kRows token rows the kernels hold (num_tokens <= kRows <= 8)
/// \tparam kDedupe one CTA per distinct expert (tokens share its weight stream)
/// \tparam kColWarps1, kColWarps2 column warps of the gate_up and down CTAs
/// \param out fp16 [M, 1536]; x fp16 [M, 1536]
/// \param w13 int32 [E, 96, 768], w2 int32 [E, 12, 3072]: SM70 Marlin NVFP4 packs
/// \param s13 uint8 [E, 96, 384], s2 uint8 [E, 12, 1536]: SM70 Marlin scale bytes
/// \param g13, g2 fp32 [E]: processed global scales
/// \param ids int32 [M, 16]; topk_weights fp32 [M, 16]
/// \param partials fp32 workspace, >= M * 16 * 4 * 384
/// \param cache fp16 workspace [>= M * 16, 1536]
/// \param counters int32 [8 * 24], zero before the first call; left zero
/// \param swiglu_limit clamp bound, no clamp when <= 0
/// \param routed_scale multiplies the route sum (moe_sum_reduce's factor)
template <int kRows, bool kDedupe, int kColWarps1, int kColWarps2>
void moe_decode(
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView x,
    tvm::ffi::TensorView w13,
    tvm::ffi::TensorView w2,
    tvm::ffi::TensorView s13,
    tvm::ffi::TensorView s2,
    tvm::ffi::TensorView g13,
    tvm::ffi::TensorView g2,
    tvm::ffi::TensorView ids,
    tvm::ffi::TensorView topk_weights,
    tvm::ffi::TensorView partials,
    tvm::ffi::TensorView cache,
    tvm::ffi::TensorView counters,
    double swiglu_limit,
    double routed_scale) {
  using namespace host;
  SymbolicSize m{"num_tokens"};
  SymbolicSize e{"num_experts"};
  SymbolicDevice device_;
  const Weights w = verify_common(m, e, device_, out, x, w13, w2, s13, s2, g13, g2, ids, topk_weights);
  TensorMatcher({-1})  //
      .with_dtype<fp32_t>()
      .with_device(device_)
      .ensure_alignment(16)
      .verify(partials);
  TensorMatcher({-1, kHidden})  //
      .with_dtype<fp16_t>()
      .with_device(device_)
      .ensure_alignment(16)
      .verify(cache);
  TensorMatcher({kMaxTokens * kW2Chunks})  //
      .with_dtype<int32_t>()
      .with_device(device_)
      .verify(counters);

  const int64_t tokens = m.unwrap();
  RuntimeCheck(tokens >= 1 && tokens <= kRows, "hy4 moe decode built for 1..", kRows, " tokens, got ", tokens);
  RuntimeCheck(partials.size(0) >= tokens * kTopK * kSplit * kGateUp, "partials workspace too small");
  RuntimeCheck(cache.size(0) >= tokens * kTopK, "cache workspace too small");

  const int num_tokens = static_cast<int>(tokens);
  const int num_experts = static_cast<int>(e.unwrap());
  const DLDevice device = device_.unwrap();
  const auto routes = static_cast<uint32_t>(tokens * kTopK);
  const auto* idp = static_cast<const int32_t*>(ids.data_ptr());
  auto* partp = static_cast<float*>(partials.data_ptr());

  LaunchKernel(
      dim3(static_cast<uint32_t>(kW13Chunks / kColWarps1 * kSplit), routes), kColWarps1 * kKWarps1 * kWarp, device)(
      gate_up_kernel<kRows, kDedupe, kColWarps1>,
      partp,
      static_cast<const __half*>(x.data_ptr()),
      w.w13,
      w.s13,
      idp,
      num_tokens,
      num_experts);

  // torch.clamp casts the bound to the tensor dtype first.
  const __half lim = __float2half_rn(static_cast<float>(swiglu_limit));
  LaunchKernel(dim3(static_cast<uint32_t>(kW2Chunks / kColWarps2), routes), kColWarps2 * kKWarps2 * kWarp, device)(
      down_kernel<kRows, kDedupe, kColWarps2>,
      static_cast<__half*>(out.data_ptr()),
      static_cast<__half*>(cache.data_ptr()),
      static_cast<int32_t*>(counters.data_ptr()),
      static_cast<const float*>(partp),
      w.w2,
      w.s2,
      w.g13,
      w.g2,
      idp,
      static_cast<const float*>(topk_weights.data_ptr()),
      num_tokens,
      num_experts,
      lim,
      swiglu_limit > 0.0,
      static_cast<float>(routed_scale));
}

/// \brief Hy4-flash routed experts for any number of tokens on SM70, with
/// the same per-element arithmetic as moe_decode.
/// \tparam kColWarps1, kColWarps2 column warps of the gate_up and down CTAs
/// \tparam kBlocks route blocks of 8 per work item
/// \param order int32 [M * 16]: route indices grouped by expert, valid ids first
/// \param expert_off int32 [E + 1]: first position in order of each expert
/// \param work_off int32 [E + 1]: first work item of each expert, where expert
///        e has ceil(count_e / (8 * kBlocks)) items
/// \param gate_up fp16 workspace [>= M * 16, 384]
/// \param cache fp16 workspace [>= M * 16, 1536]
/// Other parameters as moe_decode.
template <int kColWarps1, int kColWarps2, int kBlocks>
void moe_prefill(
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView x,
    tvm::ffi::TensorView w13,
    tvm::ffi::TensorView w2,
    tvm::ffi::TensorView s13,
    tvm::ffi::TensorView s2,
    tvm::ffi::TensorView g13,
    tvm::ffi::TensorView g2,
    tvm::ffi::TensorView ids,
    tvm::ffi::TensorView topk_weights,
    tvm::ffi::TensorView order,
    tvm::ffi::TensorView expert_off,
    tvm::ffi::TensorView work_off,
    tvm::ffi::TensorView gate_up,
    tvm::ffi::TensorView cache,
    double swiglu_limit,
    double routed_scale) {
  using namespace host;
  SymbolicSize m{"num_tokens"};
  SymbolicSize e{"num_experts"};
  SymbolicSize r{"num_routes"};
  SymbolicSize e1{"num_experts_plus_one"};
  SymbolicDevice device_;
  const Weights w = verify_common(m, e, device_, out, x, w13, w2, s13, s2, g13, g2, ids, topk_weights);
  TensorMatcher({r})  //
      .with_dtype<int32_t>()
      .with_device(device_)
      .verify(order);
  TensorMatcher({e1})  //
      .with_dtype<int32_t>()
      .with_device(device_)
      .verify(expert_off)
      .verify(work_off);
  TensorMatcher({-1, kGateUp})  //
      .with_dtype<fp16_t>()
      .with_device(device_)
      .ensure_alignment(16)
      .verify(gate_up);
  TensorMatcher({-1, kHidden})  //
      .with_dtype<fp16_t>()
      .with_device(device_)
      .ensure_alignment(16)
      .verify(cache);

  const int64_t tokens = m.unwrap();
  const int64_t experts = e.unwrap();
  const int64_t routes = tokens * kTopK;
  RuntimeCheck(tokens >= 1 && routes <= (int64_t{1} << 30), "bad token count ", tokens);
  RuntimeCheck(r.unwrap() == routes, "order must hold M * 16 routes");
  RuntimeCheck(e1.unwrap() == experts + 1, "expert_off and work_off must hold E + 1 entries");
  RuntimeCheck(gate_up.size(0) >= routes && cache.size(0) >= routes, "prefill workspace too small");

  constexpr int64_t kRoutesPerItem = int64_t{kBlocks} * kMmaRows;
  // Each expert with routes adds at most one partial item.
  const int64_t items = (routes + kRoutesPerItem - 1) / kRoutesPerItem + std::min(experts, routes);
  RuntimeCheck(items < (int64_t{1} << 31), "too many work items");
  const int num_tokens = static_cast<int>(tokens);
  const int num_experts = static_cast<int>(experts);
  const DLDevice device = device_.unwrap();
  const auto* orderp = static_cast<const int32_t*>(order.data_ptr());
  const auto* eoffp = static_cast<const int32_t*>(expert_off.data_ptr());
  const auto* woffp = static_cast<const int32_t*>(work_off.data_ptr());
  auto* gup = static_cast<__half*>(gate_up.data_ptr());
  auto* cachep = static_cast<__half*>(cache.data_ptr());

  LaunchKernel(
      dim3(static_cast<uint32_t>(items), static_cast<uint32_t>(kW13Chunks / kColWarps1)),
      kColWarps1 * kKWarps1 * kWarp,
      device)(
      gate_up_prefill_kernel<kColWarps1, kBlocks>,
      gup,
      static_cast<const __half*>(x.data_ptr()),
      w.w13,
      w.s13,
      w.g13,
      orderp,
      eoffp,
      woffp,
      num_experts);

  const __half lim = __float2half_rn(static_cast<float>(swiglu_limit));
  LaunchKernel(
      dim3(static_cast<uint32_t>(items), static_cast<uint32_t>(kW2Chunks / kColWarps2)),
      kColWarps2 * kKWarps2 * kWarp,
      device)(
      down_prefill_kernel<kColWarps2, kBlocks>,
      cachep,
      static_cast<const __half*>(gup),
      w.w2,
      w.s2,
      w.g2,
      static_cast<const float*>(topk_weights.data_ptr()),
      orderp,
      eoffp,
      woffp,
      num_experts,
      lim,
      swiglu_limit > 0.0);

  constexpr int kSumThreads = 256;
  const int64_t sum_items = tokens * (kHidden / kWordCols);
  LaunchKernel(static_cast<uint32_t>((sum_items + kSumThreads - 1) / kSumThreads), kSumThreads, device)(
      route_sum_kernel,
      static_cast<__half*>(out.data_ptr()),
      static_cast<const __half*>(cachep),
      static_cast<const int32_t*>(ids.data_ptr()),
      num_tokens,
      num_experts,
      static_cast<float>(routed_scale));
}

}  // namespace sm70_hy4_moe

}  // namespace sglang
