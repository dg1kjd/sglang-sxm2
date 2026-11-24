// SM70 sparse MLA with a rope tail; sparse_mla_sm70.cuh is the NoPE-only
// kernel. Scores over the 512-d latent plus a 64-d rope tail (or no tail),
// values over the latent, optional per-head attention sinks and an optional
// per-row count of valid selection slots. Volta WMMA m8n32k16, fp16
// operands, fp32 accumulators.
//
// One block owns one query token, up to 8 heads (the WMMA M dimension; heads
// past H are zero rows) and one split of the selection slots. Key rows go
// through shared memory 32 at a time; the next tile's rows are loaded into
// registers while the current tile is computed. For QK, warp w holds the
// query fragments of the 16-d steps w, w + 8, ... in registers and the warps'
// partial scores are summed in warp order. For the softmax, warp h owns head
// h and lane n key n of the tile; for PV each warp owns 64 output dims. The
// softmax is fp32 and online, P is rounded to fp16 for the PV product.
//
// A slot is used when its index is >= 0 and its position is below the row's
// length (lengths[row] clamped to [0, width]; the full width without
// lengths). A sink adds exp(sink[h]) to the softmax denominator and has no
// value row. A row without a usable slot writes zeros. With several splits
// each block writes its per-head (max, sum, unnormalized acc) and
// sparse_mla_sm70_rope_combine_kernel merges the splits in split order.

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <mma.h>

namespace sglang {

constexpr int kSparseMlaSm70RopeLatent = 512;

SGL_DEVICE float sparse_mla_sm70_rope_warp_sum(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    v += __shfl_xor_sync(0xffffffff, v, off);
  }
  return v;
}

constexpr int kSparseMlaSm70RopeKeys = 32;  // key rows per tile (WMMA N for QK, two K steps for PV)
constexpr int kSparseMlaSm70RopeHeads = 8;  // heads per block (WMMA M)
constexpr int kSparseMlaSm70RopeWarps = 8;
constexpr int kSparseMlaSm70RopeThreads = kSparseMlaSm70RopeWarps * 32;
constexpr int kSparseMlaSm70RopeIdxSlots = 3;
constexpr int kSparseMlaSm70RopeDimPerWarp = kSparseMlaSm70RopeLatent / kSparseMlaSm70RopeWarps;  // 64
constexpr int kSparseMlaSm70RopeLdP = kSparseMlaSm70RopeKeys + 8;  // padded P row, in halves
constexpr int kSparseMlaSm70CombineDims = 128;                     // output dims per combine block

static_assert(kSparseMlaSm70RopeHeads == kSparseMlaSm70RopeWarps, "the softmax step maps warp h to head h");
static_assert(kSparseMlaSm70RopeKeys == 32, "the softmax step maps lane n to key n");

struct SparseMlaSm70RopeParams {
  const fp16_t* __restrict__ q_nope;    // [S, H, 512]
  const fp16_t* __restrict__ q_rope;    // [S, H, tail]; unused when tail is 0
  const fp16_t* __restrict__ kv;        // rows of 512 + tail, row stride = stride_kv
  const int32_t* __restrict__ indices;  // [S, width]
  const int32_t* __restrict__ lengths;  // [S] or nullptr
  const fp32_t* __restrict__ sink;      // [H] or nullptr
  fp16_t* __restrict__ out;             // [S, H, 512]
  fp32_t* __restrict__ part_acc;        // [S, H, splits, 512], splits > 1 only
  fp32_t* __restrict__ part_ml;         // [S, H, splits, 2], splits > 1 only
  int64_t stride_kv;
  int64_t stride_qn_s;
  int64_t stride_qn_h;
  int64_t stride_qr_s;
  int64_t stride_qr_h;
  uint32_t H;
  uint32_t width;
  uint32_t splits;
  uint32_t keys_per_split;  // a multiple of kSparseMlaSm70RopeKeys
  fp32_t scale;
};

template <int kTail>
struct SparseMlaSm70RopeSmem {
  static constexpr int kDim = kSparseMlaSm70RopeLatent + kTail;
  static constexpr int kLd = kDim + 8;  // padded row, in halves
  // Key rows of the current tile; before the first tile, the block's query rows.
  alignas(32) fp16_t k[kSparseMlaSm70RopeKeys * kLd];
  alignas(32) fp32_t s_part[kSparseMlaSm70RopeWarps * kSparseMlaSm70RopeHeads * kSparseMlaSm70RopeKeys];
  alignas(32) fp16_t p[kSparseMlaSm70RopeHeads * kSparseMlaSm70RopeLdP];
  fp32_t alpha[kSparseMlaSm70RopeHeads];  // per-tile rescale, then the output factor
  int32_t idx[kSparseMlaSm70RopeIdxSlots][kSparseMlaSm70RopeKeys];
  int32_t any[kSparseMlaSm70RopeIdxSlots];
};

// Volta WMMA m8n32k16 fp32 accumulator: element i of a lane holds (row, col)
// below, as laid out by store_matrix_sync(..., mem_row_major). Elements i and
// i + 1 (i even) are adjacent columns of one row; elements with (i & 2) == 0
// share row sparse_mla_sm70_acc_row(lane, 0), the others sit two rows below it.
SGL_DEVICE int sparse_mla_sm70_acc_row(int lane, int i) {
  return (lane & 1) + (i & 2) + ((lane & 16) >> 2);
}

SGL_DEVICE int sparse_mla_sm70_acc_col(int lane, int i) {
  return (i & 1) + (lane & 2) + (i & 4) + ((lane & 4) << 1) + ((lane & 8) << 1);
}

template <int kTail>
__global__ void __launch_bounds__(kSparseMlaSm70RopeThreads, 2)
    sparse_mla_sm70_rope_kernel(const SparseMlaSm70RopeParams p) {
  using namespace nvcuda;
  using Smem = SparseMlaSm70RopeSmem<kTail>;
  constexpr int kDim = Smem::kDim;
  constexpr int kLd = Smem::kLd;
  constexpr int kKeys = kSparseMlaSm70RopeKeys;
  constexpr int kHeads = kSparseMlaSm70RopeHeads;
  constexpr int kThreads = kSparseMlaSm70RopeThreads;
  constexpr int kWarps = kSparseMlaSm70RopeWarps;
  constexpr int kSlots = kSparseMlaSm70RopeIdxSlots;
  constexpr int kLdP = kSparseMlaSm70RopeLdP;
  constexpr int kVecPerRow = kDim / 8;  // 16-byte vectors per key row
  constexpr int kVecPerTile = kKeys * kVecPerRow;
  constexpr int kLoads = (kVecPerTile + kThreads - 1) / kThreads;
  constexpr int kQVec = kHeads * kVecPerRow;
  constexpr int kQLoads = (kQVec + kThreads - 1) / kThreads;
  constexpr int kQkSteps = kDim / 16;
  constexpr int kQFrags = (kQkSteps + kWarps - 1) / kWarps;  // QK steps of one warp, at most
  constexpr int kDimPerWarp = kSparseMlaSm70RopeDimPerWarp;
  constexpr int kOutFrags = kDimPerWarp / 32;
  static_assert(kDim % 16 == 0 && kLd % 8 == 0 && kLdP % 8 == 0, "WMMA needs 16-d steps and 16-byte rows");
  static_assert(kHeads <= kKeys, "the query rows are staged in the key buffer");

  using FragQ = wmma::fragment<wmma::matrix_a, 8, 32, 16, half, wmma::row_major>;
  using FragK = wmma::fragment<wmma::matrix_b, 8, 32, 16, half, wmma::col_major>;
  using FragV = wmma::fragment<wmma::matrix_b, 8, 32, 16, half, wmma::row_major>;
  using FragAcc = wmma::fragment<wmma::accumulator, 8, 32, 16, float>;

  __shared__ Smem sm;

  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const uint32_t row = blockIdx.x;
  const uint32_t h0 = blockIdx.y * kHeads;
  const int nh = static_cast<int>(min(p.H - h0, static_cast<uint32_t>(kHeads)));
  const uint32_t split = blockIdx.z;

  uint32_t len = p.width;
  if (p.lengths != nullptr) {
    const int32_t n = p.lengths[row];
    len = n <= 0 ? 0u : min(static_cast<uint32_t>(n), p.width);
  }
  const uint32_t k_begin = split * p.keys_per_split;
  const uint32_t k_end = min(len, k_begin + p.keys_per_split);
  const int n_tiles = k_end > k_begin ? static_cast<int>((k_end - k_begin + kKeys - 1) / kKeys) : 0;
  const int32_t* idx_row = p.indices + static_cast<int64_t>(row) * p.width;

  // Query rows (latent, then rope tail) into the key buffer; heads past H are zero.
  {
    uint4 qv[kQLoads];
#pragma unroll
    for (int j = 0; j < kQLoads; ++j) {
      const int e = tid + j * kThreads;
      const int hh = e / kVecPerRow;
      const int c = (e - hh * kVecPerRow) * 8;
      qv[j] = make_uint4(0u, 0u, 0u, 0u);
      if ((kQVec % kThreads == 0 || e < kQVec) && hh < nh) {
        const int64_t h = static_cast<int64_t>(h0) + hh;
        const fp16_t* src = c < kSparseMlaSm70RopeLatent
                                ? p.q_nope + static_cast<int64_t>(row) * p.stride_qn_s + h * p.stride_qn_h + c
                                : p.q_rope + static_cast<int64_t>(row) * p.stride_qr_s + h * p.stride_qr_h +
                                      (c - kSparseMlaSm70RopeLatent);
        qv[j] = *reinterpret_cast<const uint4*>(src);
      }
    }
#pragma unroll
    for (int j = 0; j < kQLoads; ++j) {
      const int e = tid + j * kThreads;
      if (kQVec % kThreads == 0 || e < kQVec) {
        const int hh = e / kVecPerRow;
        const int c = (e - hh * kVecPerRow) * 8;
        *reinterpret_cast<uint4*>(&sm.k[hh * kLd + c]) = qv[j];
      }
    }
  }

  // Warp 0 fetches the indices of tile t (-1 for slots that are invalid or
  // outside this split) one tile before it stages them into slot t % kSlots
  // together with whether any is usable.
  auto fetch_indices = [&](int t) -> int32_t {
    int32_t ix = -1;
    if (t < n_tiles) {
      const uint32_t k = k_begin + static_cast<uint32_t>(t) * kKeys + lane;
      if (k < k_end) {
        ix = idx_row[k];
      }
    }
    return ix;
  };
  auto stage_indices = [&](int t, int32_t ix) {
    const int slot = t % kSlots;
    sm.idx[slot][lane] = ix < 0 ? -1 : ix;
    const unsigned usable = __ballot_sync(0xffffffffu, ix >= 0);
    if (lane == 0) {
      sm.any[slot] = usable != 0u;
    }
  };
  int32_t ix_ahead = -1;  // warp 0: indices of tile t + 2 while tile t runs
  if (warp == 0) {
    stage_indices(0, fetch_indices(0));
    stage_indices(1, fetch_indices(1));
    ix_ahead = fetch_indices(2);
  }
  __syncthreads();

  FragQ q_frag[kQFrags];
#pragma unroll
  for (int j = 0; j < kQFrags; ++j) {
    const int ks = warp + j * kWarps;
    if (ks < kQkSteps) {
      wmma::load_matrix_sync(q_frag[j], sm.k + ks * 16, kLd);
    }
  }
  FragAcc o_frag[kOutFrags];
#pragma unroll
  for (int dd = 0; dd < kOutFrags; ++dd) {
    wmma::fill_fragment(o_frag[dd], 0.f);
  }
  // Running max and sum of head `warp`, the same in every lane.
  float m_run = -INFINITY;
  float l_run = 0.f;

  // Row data of one tile, kLoads 16-byte vectors per thread; unused slots are zero.
  uint4 rows[kLoads];
  auto load_rows = [&](int slot) {
#pragma unroll
    for (int j = 0; j < kLoads; ++j) {
      const int e = tid + j * kThreads;
      rows[j] = make_uint4(0u, 0u, 0u, 0u);
      if (kVecPerTile % kThreads == 0 || e < kVecPerTile) {
        const int r = e / kVecPerRow;
        const int c = e - r * kVecPerRow;
        const int32_t ix = sm.idx[slot][r];
        if (ix >= 0) {
          rows[j] = __ldg(reinterpret_cast<const uint4*>(p.kv + static_cast<int64_t>(ix) * p.stride_kv) + c);
        }
      }
    }
  };
  if (n_tiles > 0) {
    load_rows(0);
  }
  __syncthreads();  // every warp holds its query fragments before tile 0 lands

  const int acc_r0 = sparse_mla_sm70_acc_row(lane, 0);
  for (int t = 0; t < n_tiles; ++t) {
    const int slot = t % kSlots;
#pragma unroll
    for (int j = 0; j < kLoads; ++j) {
      const int e = tid + j * kThreads;
      if (kVecPerTile % kThreads == 0 || e < kVecPerTile) {
        const int r = e / kVecPerRow;
        const int c = e - r * kVecPerRow;
        *reinterpret_cast<uint4*>(&sm.k[r * kLd + c * 8]) = rows[j];
      }
    }
    __syncthreads();
    if (t + 1 < n_tiles) {
      load_rows((t + 1) % kSlots);
    }
    if (warp == 0) {
      stage_indices(t + 2, ix_ahead);
      ix_ahead = fetch_indices(t + 3);
    }

    if (sm.any[slot]) {
      {
        FragK k_frag;
        FragAcc s_frag;
        wmma::fill_fragment(s_frag, 0.f);
#pragma unroll
        for (int j = 0; j < kQFrags; ++j) {
          const int ks = warp + j * kWarps;
          if (ks < kQkSteps) {
            wmma::load_matrix_sync(k_frag, sm.k + ks * 16, kLd);
            wmma::mma_sync(s_frag, q_frag[j], k_frag, s_frag);
          }
        }
        wmma::store_matrix_sync(sm.s_part + warp * (kHeads * kKeys), s_frag, kKeys, wmma::mem_row_major);
      }
      __syncthreads();

      // Warp h, lane n: score of head h and key n; the reductions stay in the warp.
      {
        float sc = 0.f;
#pragma unroll
        for (int w = 0; w < kWarps; ++w) {
          sc += sm.s_part[w * (kHeads * kKeys) + warp * kKeys + lane];
        }
        sc = sm.idx[slot][lane] >= 0 ? sc * p.scale : -INFINITY;
        float mt = sc;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
          mt = fmaxf(mt, __shfl_xor_sync(0xffffffffu, mt, off));
        }
        const float m_new = fmaxf(m_run, mt);
        const float m_use = m_new == -INFINITY ? 0.f : m_new;
        const float pe = expf(sc - m_use);
        sm.p[warp * kLdP + lane] = __float2half_rn(pe);
        const float ps = sparse_mla_sm70_rope_warp_sum(pe);
        const float alpha = expf(m_run - m_use);
        l_run = l_run * alpha + ps;
        m_run = m_new;
        if (lane == 0) {
          sm.alpha[warp] = alpha;
        }
      }
      __syncthreads();

      {
        const float a0 = sm.alpha[acc_r0];
        const float a1 = sm.alpha[acc_r0 + 2];
#pragma unroll
        for (int dd = 0; dd < kOutFrags; ++dd) {
#pragma unroll
          for (int i = 0; i < o_frag[dd].num_elements; ++i) {
            o_frag[dd].x[i] *= (i & 2) ? a1 : a0;
          }
        }
        FragQ p_frag;
        FragV v_frag;
#pragma unroll
        for (int kk = 0; kk < kKeys / 16; ++kk) {
          wmma::load_matrix_sync(p_frag, sm.p + kk * 16, kLdP);
#pragma unroll
          for (int dd = 0; dd < kOutFrags; ++dd) {
            wmma::load_matrix_sync(v_frag, sm.k + kk * 16 * kLd + warp * kDimPerWarp + dd * 32, kLd);
            wmma::mma_sync(o_frag[dd], p_frag, v_frag, o_frag[dd]);
          }
        }
      }
    }
    __syncthreads();
  }

  const int d_warp = warp * kDimPerWarp;
  if (p.splits == 1) {
    if (lane == 0) {
      float f = 0.f;
      if (warp < nh && l_run > 0.f) {
        float mf = m_run;
        float sink_term = 0.f;
        if (p.sink != nullptr) {
          const float sv = p.sink[h0 + warp];
          mf = fmaxf(m_run, sv);
          sink_term = expf(sv - mf);
        }
        const float sc = expf(m_run - mf);
        f = sc / (l_run * sc + sink_term);
      }
      sm.alpha[warp] = f;
    }
    __syncthreads();
    const float f0 = sm.alpha[acc_r0];
    const float f1 = sm.alpha[acc_r0 + 2];
#pragma unroll
    for (int dd = 0; dd < kOutFrags; ++dd) {
#pragma unroll
      for (int i = 0; i < o_frag[dd].num_elements; i += 2) {
        const int hh = sparse_mla_sm70_acc_row(lane, i);
        if (hh < nh) {
          const float f = (i & 2) ? f1 : f0;
          const int d = d_warp + dd * 32 + sparse_mla_sm70_acc_col(lane, i);
          *reinterpret_cast<fp16x2_t*>(
              p.out + (static_cast<int64_t>(row) * p.H + h0 + hh) * kSparseMlaSm70RopeLatent + d) =
              __floats2half2_rn(o_frag[dd].x[i] * f, o_frag[dd].x[i + 1] * f);
        }
      }
    }
    return;
  }

  if (lane == 0 && warp < nh) {
    const int64_t slot = (static_cast<int64_t>(row) * p.H + h0 + warp) * p.splits + split;
    p.part_ml[slot * 2] = m_run;
    p.part_ml[slot * 2 + 1] = l_run;
  }
#pragma unroll
  for (int dd = 0; dd < kOutFrags; ++dd) {
#pragma unroll
    for (int i = 0; i < o_frag[dd].num_elements; i += 2) {
      const int hh = sparse_mla_sm70_acc_row(lane, i);
      if (hh < nh) {
        const int64_t slot = (static_cast<int64_t>(row) * p.H + h0 + hh) * p.splits + split;
        const int d = d_warp + dd * 32 + sparse_mla_sm70_acc_col(lane, i);
        *reinterpret_cast<float2*>(p.part_acc + slot * kSparseMlaSm70RopeLatent + d) =
            make_float2(o_frag[dd].x[i], o_frag[dd].x[i + 1]);
      }
    }
  }
}

// Grid (S, H, 512 / 128). Warp w merges splits w, w + 8, ... for 128 dims
// (4 per lane); the warps' sums are added in warp order. The sink and the
// final normalization match the one-split path of the kernel above.
__global__ void __launch_bounds__(kSparseMlaSm70RopeThreads)
    sparse_mla_sm70_rope_combine_kernel(const SparseMlaSm70RopeParams p) {
  constexpr int kWarps = kSparseMlaSm70RopeWarps;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const uint32_t h = blockIdx.y;
  const int64_t sh = static_cast<int64_t>(blockIdx.x) * p.H + h;
  const fp32_t* ml = p.part_ml + sh * p.splits * 2;
  const fp32_t* acc = p.part_acc + sh * p.splits * kSparseMlaSm70RopeLatent;

  float m = -INFINITY;
  for (uint32_t s = lane; s < p.splits; s += 32) {
    m = fmaxf(m, ml[2 * s]);
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
  }
  const float sv = p.sink != nullptr ? p.sink[h] : -INFINITY;
  const float mf = fmaxf(m, sv);
  float l = 0.f;
  for (uint32_t s = lane; s < p.splits; s += 32) {
    const float ms = ml[2 * s];
    if (ms != -INFINITY) {
      l += expf(ms - mf) * ml[2 * s + 1];
    }
  }
  l = sparse_mla_sm70_rope_warp_sum(l);

  const int d = static_cast<int>(blockIdx.z) * kSparseMlaSm70CombineDims + lane * 4;
  float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll 4
  for (uint32_t s = warp; s < p.splits; s += kWarps) {
    const float ms = ml[2 * s];
    if (ms == -INFINITY) {
      continue;
    }
    const float w = expf(ms - mf);
    const float4 v = *reinterpret_cast<const float4*>(acc + static_cast<int64_t>(s) * kSparseMlaSm70RopeLatent + d);
    a.x += w * v.x;
    a.y += w * v.y;
    a.z += w * v.z;
    a.w += w * v.w;
  }
  __shared__ float4 red[kWarps][32];
  red[warp][lane] = a;
  __syncthreads();
  if (warp != 0) {
    return;
  }
  float4 t = red[0][lane];
#pragma unroll
  for (int w = 1; w < kWarps; ++w) {
    const float4 v = red[w][lane];
    t.x += v.x;
    t.y += v.y;
    t.z += v.z;
    t.w += v.w;
  }
  float f = 0.f;
  if (l > 0.f) {
    f = 1.f / (l + (p.sink != nullptr ? expf(sv - mf) : 0.f));
  }
  fp16_t* orow = p.out + sh * kSparseMlaSm70RopeLatent + d;
  *reinterpret_cast<fp16x2_t*>(orow) = __floats2half2_rn(t.x * f, t.y * f);
  *reinterpret_cast<fp16x2_t*>(orow + 2) = __floats2half2_rn(t.z * f, t.w * f);
}

struct SparseMlaSm70RopeKernel {
  /// \brief Sparse MLA with a rope tail, attention sinks and per-row lengths.
  /// \param q_nope `[S, H, 512]` fp16, last dim contiguous.
  /// \param q_rope `[S, H, 64]` fp16, last dim contiguous; empty for no rope tail.
  /// \param kv `[N, 512 + tail]` or `[N, 1, 512 + tail]` fp16, 16-byte aligned rows.
  /// \param indices `[S, width]` int32. Negative entries are masked out.
  /// \param lengths `[S]` int32 count of leading slots to use, or empty for all.
  /// \param sink `[H]` fp32 per-head sink logits, or empty for none.
  /// \param out `[S, H, 512]` fp16, contiguous.
  /// \param part_acc fp32 workspace, at least `S * H * splits * 512` when splits > 1.
  /// \param part_ml fp32 workspace, at least `S * H * splits * 2` when splits > 1.
  /// \param sm_scale Softmax scale applied to the query-key dot.
  /// \param splits Blocks per (token, head tile) over the selection slots.
  static void
  run(const tvm::ffi::TensorView q_nope,
      const tvm::ffi::TensorView q_rope,
      const tvm::ffi::TensorView kv,
      const tvm::ffi::TensorView indices,
      const tvm::ffi::TensorView lengths,
      const tvm::ffi::TensorView sink,
      const tvm::ffi::TensorView out,
      const tvm::ffi::TensorView part_acc,
      const tvm::ffi::TensorView part_ml,
      double sm_scale,
      int64_t splits) {
    using namespace host;

    auto S_ = SymbolicSize{"tokens"};
    auto H_ = SymbolicSize{"heads"};
    auto W_ = SymbolicSize{"width"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({S_, H_, kSparseMlaSm70RopeLatent})
        .with_dtype<fp16_t>()
        .with_device(device)
        .with_strides({-1, -1, 1})
        .verify(q_nope);
    TensorMatcher({S_, H_, kSparseMlaSm70RopeLatent})
        .with_dtype<fp16_t>()
        .with_device(device)
        .with_strides({-1, -1, 1})
        .verify(out);
    TensorMatcher({S_, W_}).with_dtype<int32_t>().with_device(device).with_strides({-1, 1}).verify(indices);

    const auto S = S_.unwrap();
    const auto H = static_cast<uint32_t>(H_.unwrap());
    const auto width = static_cast<uint32_t>(W_.unwrap());

    int64_t tail = 0;
    if (q_rope.numel() > 0) {
      auto T_ = SymbolicSize{"rope_dim"};
      TensorMatcher({S_, H_, T_}).with_dtype<fp16_t>().with_device(device).with_strides({-1, -1, 1}).verify(q_rope);
      tail = T_.unwrap();
      RuntimeCheck(tail == 64, "SM70 sparse MLA rope tail must be 64, got ", tail);
      RuntimeCheck(
          reinterpret_cast<uintptr_t>(q_rope.data_ptr()) % 16 == 0 && q_rope.stride(0) % 8 == 0 &&
              q_rope.stride(1) % 8 == 0,
          "q_rope rows must be 16-byte aligned");
    }
    RuntimeCheck(
        reinterpret_cast<uintptr_t>(q_nope.data_ptr()) % 16 == 0 && q_nope.stride(0) % 8 == 0 &&
            q_nope.stride(1) % 8 == 0,
        "q_nope rows must be 16-byte aligned");
    RuntimeCheck(
        out.stride(0) == static_cast<int64_t>(H) * kSparseMlaSm70RopeLatent &&
            out.stride(1) == kSparseMlaSm70RopeLatent,
        "out must be contiguous [S, H, 512]");
    RuntimeCheck(indices.stride(0) == static_cast<int64_t>(width), "indices must be contiguous [S, width]");

    const int64_t kv_dim = kSparseMlaSm70RopeLatent + tail;
    int64_t stride_kv = 0;
    if (kv.ndim() == 2) {
      auto N_ = SymbolicSize{"kv_rows"};
      TensorMatcher({N_, kv_dim}).with_dtype<fp16_t>().with_device(device).with_strides({-1, 1}).verify(kv);
      stride_kv = kv.stride(0);
    } else {
      auto N_ = SymbolicSize{"kv_rows"};
      TensorMatcher({N_, 1, kv_dim}).with_dtype<fp16_t>().with_device(device).with_strides({-1, -1, 1}).verify(kv);
      stride_kv = kv.stride(0);
    }
    RuntimeCheck(
        stride_kv > 0 && stride_kv % 8 == 0 && reinterpret_cast<uintptr_t>(kv.data_ptr()) % 16 == 0,
        "kv rows must be 16-byte aligned");

    const int32_t* lengths_ptr = nullptr;
    if (lengths.numel() > 0) {
      TensorMatcher({S_}).with_dtype<int32_t>().with_device(device).with_strides({1}).verify(lengths);
      lengths_ptr = static_cast<const int32_t*>(lengths.data_ptr());
    }
    const fp32_t* sink_ptr = nullptr;
    if (sink.numel() > 0) {
      TensorMatcher({H_}).with_dtype<fp32_t>().with_device(device).with_strides({1}).verify(sink);
      sink_ptr = static_cast<const fp32_t*>(sink.data_ptr());
    }

    if (S == 0 || H == 0) {
      return;
    }
    const uint32_t tiles = (width + kSparseMlaSm70RopeKeys - 1) / kSparseMlaSm70RopeKeys;
    RuntimeCheck(
        splits >= 1 && static_cast<uint64_t>(splits) <= std::max<uint32_t>(tiles, 1),
        "splits must be in [1, ceil(width / 32)]");
    const auto n_splits = static_cast<uint32_t>(splits);
    const uint32_t keys_per_split = ((tiles + n_splits - 1) / n_splits) * kSparseMlaSm70RopeKeys;
    if (n_splits > 1) {
      const int64_t slots = S * static_cast<int64_t>(H) * n_splits;
      RuntimeCheck(
          part_acc.is_contiguous() && part_acc.dtype().code == kDLFloat && part_acc.dtype().bits == 32 &&
              part_acc.numel() >= slots * kSparseMlaSm70RopeLatent,
          "part_acc must be contiguous fp32 [S, H, splits, 512]");
      RuntimeCheck(
          part_ml.is_contiguous() && part_ml.dtype().code == kDLFloat && part_ml.dtype().bits == 32 &&
              part_ml.numel() >= slots * 2,
          "part_ml must be contiguous fp32 [S, H, splits, 2]");
    }

    const SparseMlaSm70RopeParams params{
        .q_nope = static_cast<const fp16_t*>(q_nope.data_ptr()),
        .q_rope = tail > 0 ? static_cast<const fp16_t*>(q_rope.data_ptr()) : nullptr,
        .kv = static_cast<const fp16_t*>(kv.data_ptr()),
        .indices = static_cast<const int32_t*>(indices.data_ptr()),
        .lengths = lengths_ptr,
        .sink = sink_ptr,
        .out = static_cast<fp16_t*>(out.data_ptr()),
        .part_acc = n_splits > 1 ? static_cast<fp32_t*>(part_acc.data_ptr()) : nullptr,
        .part_ml = n_splits > 1 ? static_cast<fp32_t*>(part_ml.data_ptr()) : nullptr,
        .stride_kv = stride_kv,
        .stride_qn_s = q_nope.stride(0),
        .stride_qn_h = q_nope.stride(1),
        .stride_qr_s = tail > 0 ? q_rope.stride(0) : 0,
        .stride_qr_h = tail > 0 ? q_rope.stride(1) : 0,
        .H = H,
        .width = width,
        .splits = n_splits,
        .keys_per_split = keys_per_split,
        .scale = static_cast<fp32_t>(sm_scale),
    };

    const uint32_t head_tiles = (H + kSparseMlaSm70RopeHeads - 1) / kSparseMlaSm70RopeHeads;
    const dim3 grid(static_cast<uint32_t>(S), head_tiles, n_splits);
    if (tail == 0) {
      LaunchKernel(grid, kSparseMlaSm70RopeThreads, device.unwrap())(sparse_mla_sm70_rope_kernel<0>, params);
    } else {
      LaunchKernel(grid, kSparseMlaSm70RopeThreads, device.unwrap())(sparse_mla_sm70_rope_kernel<64>, params);
    }
    if (n_splits > 1) {
      LaunchKernel(
          dim3(static_cast<uint32_t>(S), H, kSparseMlaSm70RopeLatent / kSparseMlaSm70CombineDims),
          kSparseMlaSm70RopeThreads,
          device.unwrap())(sparse_mla_sm70_rope_combine_kernel, params);
    }
  }
};

}  // namespace sglang
