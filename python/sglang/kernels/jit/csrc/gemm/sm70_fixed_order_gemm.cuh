// SPDX-License-Identifier: Apache-2.0
// Batch-invariant fp16 GEMM for Volta tensor cores (mma.m8n8k4, fp32 accumulation):
// out[b, m, n] = sum_k x[b, m, k] * W[b, n, k] (+ bias[n]), W fp16 or FP8 e4m3 with
// one fp32 scale per 128 x 128 block.
//
// Every output element is computed by the same sequence of operations, whatever
// M, the kernel (decode or prefill) or the launch shape:
//   * K is cut into chains of kChainK consecutive k (32 for fp16 W, 64 for FP8 W).
//     A chain starts from 0.0f and runs its mma k-steps of 4 in increasing k.
//   * The chains are folded in index order from 0.0f:
//       fp16: total = total + chain_c
//       FP8:  total = fma(chain_c, scale[n / 128][c / 2] * 256, total)
//     (FP8 is widened to fp16 as value * 2^-8, which is exact; the 256 undoes it.)
//   * Then total + float(bias[n]), then the rounding to the output type.
// The decode kernel (a CTA per 8 output columns, warps split the chains) and the
// prefill kernel (128 x 128 tiles, one k-tile = 64 k) both follow it, so a row
// of out depends only on its row of x. The fp16 callers use the kernel up to
// MAX_ROWS rows only (sm70_fixed_order_gemm.py), so batch invariance holds there.
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <type_traits>

namespace sglang::sm70_fixed_order_gemm {

inline constexpr int kWarp = 32;
inline constexpr int kMaxDecodeWarps = 8;
inline constexpr int kDecodeRows = 8;
inline constexpr int kPrefillBM = 128;
inline constexpr int kPrefillBN = 128;
inline constexpr int kPrefillBK = 64;
inline constexpr int kPrefillThreads = 256;
inline constexpr int kPrefillStageBytes = (kPrefillBM + kPrefillBN) * kPrefillBK * 2;
inline constexpr int kPrefillSmemBytes = 2 * kPrefillStageBytes;
inline constexpr std::size_t kDefaultSmemBytes = 48 * 1024;
inline constexpr std::size_t kMaxSmemBytes = 96 * 1024;

struct Params {
  void* out;
  const fp16_t* x;
  const void* w;
  const float* scale;
  const fp16_t* bias;
  int64_t x_batch;
  int64_t x_row;
  int64_t w_batch;
  int64_t w_row;
  int64_t o_batch;
  int64_t o_row;
  int64_t s_row;
  int32_t m;
  int32_t n;
  int32_t k;
};

template <bool kFp8>
inline constexpr int kChainK = kFp8 ? 64 : 32;

SGL_DEVICE void mma884(float (&c)[8], uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]), "+f"(c[5]), "+f"(c[6]), "+f"(c[7])
      : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

// Two e4m3 bytes, placed in the high byte of each half, as fp16 value * 2^-8.
SGL_DEVICE uint32_t e4m3_hi_bytes_to_f16x2(uint32_t t) {
  return (t & 0x80008000u) | ((t >> 1) & 0x3F803F80u);
}

// Four e4m3 (k order in byte order) as two fp16x2 registers.
SGL_DEVICE void e4m3x4_to_f16x4(uint32_t v, uint32_t& lo, uint32_t& hi) {
  lo = e4m3_hi_bytes_to_f16x2(__byte_perm(v, 0u, 0x1404));
  hi = e4m3_hi_bytes_to_f16x2(__byte_perm(v, 0u, 0x3424));
}

SGL_DEVICE uint4 ld_stream(const void* p) {
  uint4 v;
  asm("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
      : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
      : "l"(p));
  return v;
}

SGL_DEVICE uint4 ld_cached(const void* p) {
  return __ldg(reinterpret_cast<const uint4*>(p));
}

SGL_DEVICE float chain_scale(const Params& p, int n, int chain) {
  return __fmul_rn(__ldg(p.scale + static_cast<int64_t>(n >> 7) * p.s_row + (chain >> 1)), 256.0f);
}

template <typename OutT>
SGL_DEVICE void store_pair(const Params& p, int b, int m, int n, float y0, float y1) {
  if (p.bias != nullptr) {
    const __half2 bb = *reinterpret_cast<const __half2*>(p.bias + n);
    y0 = __fadd_rn(y0, __low2float(bb));
    y1 = __fadd_rn(y1, __high2float(bb));
  }
  OutT* o = static_cast<OutT*>(p.out) + b * p.o_batch + static_cast<int64_t>(m) * p.o_row + n;
  if constexpr (std::is_same_v<OutT, float>) {
    *reinterpret_cast<float2*>(o) = make_float2(y0, y1);
  } else {
    *reinterpret_cast<__half2*>(o) = __floats2half2_rn(y0, y1);
  }
}

// Decode: one CTA per 8 columns n0..n0+7 and up to 8 rows m0..m0+7 of one batch.
// A task is 4 consecutive chains, one per quad pair (QP) of a warp; warps take
// tasks round robin. Lane l holds A row / B column (l & 3) + 4 * (l >> 4) of its
// QP and loads 64 contiguous bytes of that W row: a warp reads 256 bytes of each
// of 8 rows. Chain results go to shared memory and are folded in chain order.
// The m tiles of one n tile are adjacent in the grid, so W comes from DRAM once.
template <bool kFp8, typename OutT>
__global__ void __launch_bounds__(kMaxDecodeWarps* kWarp) decode_kernel(const Params p) {
  extern __shared__ float s_chain[];
  constexpr int kCK = kChainK<kFp8>;
  constexpr int kXVecs = kCK / 8;
  const int lane = threadIdx.x & (kWarp - 1);
  const int warp = threadIdx.x / kWarp;
  const int num_warps = blockDim.x / kWarp;
  const int qp = (lane >> 2) & 3;
  const int r = (lane & 3) | ((lane >> 4) << 2);
  const int m0 = blockIdx.x * kDecodeRows;
  const int n0 = blockIdx.y * kDecodeRows;
  const int b = blockIdx.z;
  const int rows = min(kDecodeRows, p.m - m0);
  const int chains = p.k / kCK;
  const int tasks = (chains + 3) >> 2;
  const bool x_live = r < rows;

  const char* w_row = static_cast<const char*>(p.w) +
                      (b * p.w_batch + static_cast<int64_t>(n0 + r) * p.w_row) * (kFp8 ? 1 : 2) + qp * 64;
  const fp16_t* x_row = p.x + b * p.x_batch + static_cast<int64_t>(m0 + (x_live ? r : 0)) * p.x_row + qp * kCK;

  const int acc_row = (lane & 1) | ((lane >> 4) << 2);
  const int acc_col = lane & 2;

  // The loads of task t + num_warps are issued right after the mma of task t,
  // so they are in flight while the chain is stored and the next task's mma runs.
  uint4 wv[4];
  uint4 xv[kXVecs];
  auto load = [&](int t) {
    const bool live = t * 4 + qp < chains;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      wv[i] = live ? ld_cached(w_row + t * 256 + i * 16) : make_uint4(0, 0, 0, 0);
    }
#pragma unroll
    for (int i = 0; i < kXVecs; ++i) {
      xv[i] = (live && x_live) ? ld_cached(x_row + t * 4 * kCK + i * 8) : make_uint4(0, 0, 0, 0);
    }
  };
  if (warp < tasks) {
    load(warp);
  }
  for (int t = warp; t < tasks; t += num_warps) {
    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    const uint32_t* wr = reinterpret_cast<const uint32_t*>(wv);
    const uint32_t* xr = reinterpret_cast<const uint32_t*>(xv);
    if constexpr (kFp8) {
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        uint32_t b0, b1;
        e4m3x4_to_f16x4(wr[j], b0, b1);
        mma884(acc, xr[2 * j], xr[2 * j + 1], b0, b1);
      }
    } else {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        mma884(acc, xr[2 * j], xr[2 * j + 1], wr[2 * j], wr[2 * j + 1]);
      }
    }
    if (t + num_warps < tasks) {
      load(t + num_warps);
    }
    const int c = t * 4 + qp;
    if (c < chains) {
      float* dst = s_chain + c * rows * kDecodeRows;
#pragma unroll
      for (int i = 0; i < 8; i += 2) {
        const int row = acc_row | (i & 2);
        if (row < rows) {
          *reinterpret_cast<float2*>(dst + row * kDecodeRows + (acc_col | (i & 4))) = make_float2(acc[i], acc[i + 1]);
        }
      }
    }
  }
  __syncthreads();

  const int elems = rows * kDecodeRows;
  for (int e = threadIdx.x * 2; e < elems; e += blockDim.x * 2) {
    float t0 = 0.f;
    float t1 = 0.f;
    const float* src = s_chain + e;
#pragma unroll 4
    for (int c = 0; c < chains; ++c) {
      const float2 v = *reinterpret_cast<const float2*>(src + c * elems);
      if constexpr (kFp8) {
        const float s = chain_scale(p, n0, c);
        t0 = __fmaf_rn(v.x, s, t0);
        t1 = __fmaf_rn(v.y, s, t1);
      } else {
        t0 = __fadd_rn(t0, v.x);
        t1 = __fadd_rn(t1, v.y);
      }
    }
    store_pair<OutT>(p, b, m0 + e / kDecodeRows, n0 + (e % kDecodeRows), t0, t1);
  }
}

// 128-byte shared rows of 8 16-byte units; this swizzle keeps the quarter-warp
// 16-byte loads below conflict free (rows r, r + 32 of a warp tile differ in it).
SGL_DEVICE int swizzle(int row, int unit) {
  return row * kPrefillBK + ((unit ^ ((row & 3) | ((row >> 3) & 4))) << 3);
}

SGL_DEVICE uint4 lds128(const fp16_t* p) {
  return *reinterpret_cast<const uint4*>(p);
}

// Prefill: 128 x 128 CTA tile, 8 warps as 2 (m) x 4 (n), warp tile 64 x 32. QP q
// owns rows 32 * (q & 1) .. +32 and columns 16 * (q >> 1) .. +16 of the warp tile:
// 4 x 2 blocks of 8 x 8. One k-tile is 64 k (two fp16 chains, one FP8 chain);
// x and the fp16-widened W are staged through registers into two shared stages.
template <bool kFp8, typename OutT>
__global__ void __launch_bounds__(kPrefillThreads, 1) prefill_kernel(const Params p) {
  extern __shared__ __align__(16) fp16_t s_tiles[];
  constexpr int kCK = kChainK<kFp8>;
  const int tid = threadIdx.x;
  const int lane = tid & (kWarp - 1);
  const int warp = tid / kWarp;
  const int qp = (lane >> 2) & 3;
  const int r = (lane & 3) | ((lane >> 4) << 2);
  const int m0 = blockIdx.y * kPrefillBM;
  const int n0 = blockIdx.x * kPrefillBN;
  const int b = blockIdx.z;
  const int chains = p.k / kCK;
  const int k_tiles = (p.k + kPrefillBK - 1) / kPrefillBK;

  const int a_row0 = (warp >> 2) * 64 + (qp & 1) * 32 + r;
  const int b_row0 = (warp & 3) * 32 + (qp >> 1) * 16 + r;

  // Staging: x tile 128 rows x 8 units (4 per thread); W tile 128 rows x 8 fp16
  // units (fp16: 4 per thread; FP8: 2 16-byte fp8 units per thread, 4 fp16 units).
  constexpr int kWLoads = kFp8 ? 2 : 4;
  uint4 xs[4];
  uint4 ws[kWLoads];
  const fp16_t* x_base = p.x + b * p.x_batch;
  const char* w_base = static_cast<const char*>(p.w) + b * p.w_batch * (kFp8 ? 1 : 2);

  auto load_tile = [&](int kt) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int idx = tid + j * kPrefillThreads;
      const int row = idx >> 3;
      const int k = kt * kPrefillBK + (idx & 7) * 8;
      const bool ok = m0 + row < p.m && k < p.k;
      xs[j] = ok ? ld_cached(x_base + static_cast<int64_t>(m0 + row) * p.x_row + k) : make_uint4(0, 0, 0, 0);
    }
#pragma unroll
    for (int j = 0; j < kWLoads; ++j) {
      const int idx = tid + j * kPrefillThreads;
      if constexpr (kFp8) {
        const int row = idx >> 2;
        const int k = kt * kPrefillBK + (idx & 3) * 16;
        const bool ok = n0 + row < p.n && k < p.k;
        ws[j] = ok ? ld_stream(w_base + static_cast<int64_t>(n0 + row) * p.w_row + k) : make_uint4(0, 0, 0, 0);
      } else {
        const int row = idx >> 3;
        const int k = kt * kPrefillBK + (idx & 7) * 8;
        const bool ok = n0 + row < p.n && k < p.k;
        ws[j] = ok ? ld_stream(w_base + (static_cast<int64_t>(n0 + row) * p.w_row + k) * 2) : make_uint4(0, 0, 0, 0);
      }
    }
  };

  auto store_tile = [&](int stage) {
    fp16_t* sa = s_tiles + stage * (kPrefillStageBytes / 2);
    fp16_t* sb = sa + kPrefillBM * kPrefillBK;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int idx = tid + j * kPrefillThreads;
      *reinterpret_cast<uint4*>(sa + swizzle(idx >> 3, idx & 7)) = xs[j];
    }
#pragma unroll
    for (int j = 0; j < kWLoads; ++j) {
      const int idx = tid + j * kPrefillThreads;
      if constexpr (kFp8) {
        const int row = idx >> 2;
        const int unit = (idx & 3) * 2;
        uint4 lo, hi;
        e4m3x4_to_f16x4(ws[j].x, lo.x, lo.y);
        e4m3x4_to_f16x4(ws[j].y, lo.z, lo.w);
        e4m3x4_to_f16x4(ws[j].z, hi.x, hi.y);
        e4m3x4_to_f16x4(ws[j].w, hi.z, hi.w);
        *reinterpret_cast<uint4*>(sb + swizzle(row, unit)) = lo;
        *reinterpret_cast<uint4*>(sb + swizzle(row, unit + 1)) = hi;
      } else {
        *reinterpret_cast<uint4*>(sb + swizzle(idx >> 3, idx & 7)) = ws[j];
      }
    }
  };

  float chain[4][2][8];
  float total[4][2][8];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
#pragma unroll
    for (int j = 0; j < 2; ++j) {
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        chain[i][j][e] = 0.f;
        total[i][j][e] = 0.f;
      }
    }
  }

  auto fold = [&](int c) {
    if (c >= chains) {
      return;
    }
    float s = 0.f;
    if constexpr (kFp8) {
      s = chain_scale(p, n0, c);
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
      for (int j = 0; j < 2; ++j) {
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          if constexpr (kFp8) {
            total[i][j][e] = __fmaf_rn(chain[i][j][e], s, total[i][j][e]);
          } else {
            total[i][j][e] = __fadd_rn(total[i][j][e], chain[i][j][e]);
          }
          chain[i][j][e] = 0.f;
        }
      }
    }
  };

  load_tile(0);
  store_tile(0);
  __syncthreads();

  for (int kt = 0; kt < k_tiles; ++kt) {
    if (kt + 1 < k_tiles) {
      load_tile(kt + 1);
    }
    const fp16_t* sa = s_tiles + (kt & 1) * (kPrefillStageBytes / 2);
    const fp16_t* sb = sa + kPrefillBM * kPrefillBK;
#pragma unroll
    for (int u = 0; u < kPrefillBK / 8; ++u) {
      uint4 a[4];
      uint4 w[2];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        a[i] = lds128(sa + swizzle(a_row0 + i * 8, u));
      }
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        w[j] = lds128(sb + swizzle(b_row0 + j * 8, u));
      }
#pragma unroll
      for (int i = 0; i < 4; ++i) {
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          mma884(chain[i][j], a[i].x, a[i].y, w[j].x, w[j].y);
        }
      }
#pragma unroll
      for (int i = 0; i < 4; ++i) {
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          mma884(chain[i][j], a[i].z, a[i].w, w[j].z, w[j].w);
        }
      }
      if constexpr (kFp8) {
        if (u == kPrefillBK / 8 - 1) {
          fold(kt);
        }
      } else {
        if (u % (kCK / 8) == kCK / 8 - 1) {
          fold(kt * (kPrefillBK / kCK) + u / (kCK / 8));
        }
      }
    }
    if (kt + 1 < k_tiles) {
      store_tile((kt + 1) & 1);
    }
    __syncthreads();
  }

  const int acc_row = (lane & 1) | ((lane >> 4) << 2);
  const int acc_col = lane & 2;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
#pragma unroll
    for (int j = 0; j < 2; ++j) {
#pragma unroll
      for (int e = 0; e < 8; e += 2) {
        const int m = m0 + a_row0 - r + i * 8 + (acc_row | (e & 2));
        const int n = n0 + b_row0 - r + j * 8 + (acc_col | (e & 4));
        if (m < p.m && n < p.n) {
          store_pair<OutT>(p, b, m, n, total[i][j][e], total[i][j][e + 1]);
        }
      }
    }
  }
}

template <bool kFp8, typename OutT>
Params make_params(
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView x,
    tvm::ffi::TensorView w,
    tvm::ffi::Optional<tvm::ffi::TensorView> scale,
    tvm::ffi::Optional<tvm::ffi::TensorView> bias,
    DLDevice& device) {
  using namespace host;
  using WT = std::conditional_t<kFp8, fp8_e4m3_t, fp16_t>;
  SymbolicSize nb = {"batch"};
  SymbolicSize m = {"num_tokens"};
  SymbolicSize n = {"out_features"};
  SymbolicSize k = {"in_features"};
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();
  TensorMatcher({nb, m, k})
      .with_strides({-1, -1, 1})
      .with_dtype<fp16_t>()
      .with_device(device_)
      .ensure_alignment(16)
      .verify(x);
  TensorMatcher({nb, n, k})
      .with_strides({-1, -1, 1})
      .with_dtype<WT>()
      .with_device(device_)
      .ensure_alignment(16)
      .verify(w);
  TensorMatcher({nb, m, n})
      .with_strides({-1, -1, 1})
      .with_dtype<OutT>()
      .with_device(device_)
      .ensure_alignment(2 * sizeof(OutT))
      .verify(out);
  const int64_t num_n = n.unwrap();
  const int64_t num_k = k.unwrap();
  CHECK_HOST(num_n % 8 == 0) << "out_features must be a multiple of 8, got " << num_n;
  CHECK_HOST(num_k % kChainK<kFp8> == 0) << "in_features must be a multiple of " << kChainK<kFp8>;
  Params p{};
  if constexpr (kFp8) {
    CHECK_HOST(nb.unwrap() == 1) << "FP8 weights take one batch";
    CHECK_HOST(scale.has_value()) << "FP8 weights need block scales";
    SymbolicSize s_rows = {"scale_rows"};
    TensorMatcher({s_rows, (num_k + 127) / 128})
        .with_strides({-1, 1})
        .with_dtype<fp32_t>()
        .with_device(device_)
        .verify(scale.value());
    CHECK_HOST(s_rows.unwrap() * 128 >= num_n) << "too few scale rows";
    p.scale = static_cast<const float*>(scale.value().data_ptr());
    p.s_row = scale.value().stride(0);
  } else {
    CHECK_HOST(!scale.has_value()) << "fp16 weights take no scales";
  }
  if (bias.has_value()) {
    TensorMatcher({n}).with_dtype<fp16_t>().with_device(device_).ensure_alignment(4).verify(bias.value());
    p.bias = static_cast<const fp16_t*>(bias.value().data_ptr());
  }
  // 16-byte loads of x and W rows, pair stores of out rows.
  const int64_t num_m = m.unwrap();
  const int64_t num_b = nb.unwrap();
  const auto aligned = [](int64_t stride, int64_t extent, int64_t bytes, int64_t align) {
    return extent <= 1 || stride * bytes % align == 0;
  };
  CHECK_HOST(aligned(x.stride(0), num_b, 2, 16) && aligned(x.stride(1), num_m, 2, 16))
      << "x strides must be multiples of 16 bytes";
  CHECK_HOST(aligned(w.stride(0), num_b, sizeof(WT), 16) && aligned(w.stride(1), num_n, sizeof(WT), 16))
      << "weight strides must be multiples of 16 bytes";
  CHECK_HOST(aligned(out.stride(0), num_b, 1, 2) && aligned(out.stride(1), num_m, 1, 2))
      << "out strides must be even";
  CHECK_HOST(num_n / kDecodeRows <= 65535) << "out_features too large";
  p.out = out.data_ptr();
  p.x = static_cast<const fp16_t*>(x.data_ptr());
  p.w = w.data_ptr();
  p.x_batch = x.stride(0);
  p.x_row = x.stride(1);
  p.w_batch = w.stride(0);
  p.w_row = w.stride(1);
  p.o_batch = out.stride(0);
  p.o_row = out.stride(1);
  p.m = static_cast<int32_t>(num_m);
  p.n = static_cast<int32_t>(num_n);
  p.k = static_cast<int32_t>(num_k);
  device = device_.unwrap();
  return p;
}

/// \brief out[B, M, N] = x[B, M, K] @ w[B, N, K]^T (+ bias), decode tiling (8 x 8 per CTA).
/// \param num_warps warps per CTA (1..8); it changes the speed only.
template <bool kFp8, typename OutT>
void decode(
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView x,
    tvm::ffi::TensorView w,
    tvm::ffi::Optional<tvm::ffi::TensorView> scale,
    tvm::ffi::Optional<tvm::ffi::TensorView> bias,
    int64_t num_warps) {
  using namespace host;
  DLDevice device;
  const Params p = make_params<kFp8, OutT>(out, x, w, scale, bias, device);
  CHECK_HOST(num_warps >= 1 && num_warps <= kMaxDecodeWarps) << "num_warps must be in 1..8";
  if (p.m == 0 || out.size(0) == 0) {
    return;
  }
  const int64_t chains = p.k / kChainK<kFp8>;
  const int64_t rows = p.m < kDecodeRows ? p.m : kDecodeRows;
  const std::size_t smem = static_cast<std::size_t>(chains * rows * kDecodeRows * sizeof(float));
  constexpr auto kernel = decode_kernel<kFp8, OutT>;
  CHECK_HOST(smem <= kMaxSmemBytes) << "in_features too large for the decode tiling";
  if (smem > kDefaultSmemBytes) {
    runtime::set_max_dynamic_smem_per_device<kernel>(device, kMaxSmemBytes);
  }
  const dim3 grid((p.m + kDecodeRows - 1) / kDecodeRows, p.n / kDecodeRows, static_cast<uint32_t>(out.size(0)));
  LaunchKernel(grid, static_cast<uint32_t>(num_warps * kWarp), device, smem)(kernel, p);
}

/// \brief Same contract as decode, 128 x 128 tiles; for large M.
template <bool kFp8, typename OutT>
void prefill(
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView x,
    tvm::ffi::TensorView w,
    tvm::ffi::Optional<tvm::ffi::TensorView> scale,
    tvm::ffi::Optional<tvm::ffi::TensorView> bias) {
  using namespace host;
  DLDevice device;
  const Params p = make_params<kFp8, OutT>(out, x, w, scale, bias, device);
  if (p.m == 0 || out.size(0) == 0) {
    return;
  }
  constexpr auto kernel = prefill_kernel<kFp8, OutT>;
  runtime::set_max_dynamic_smem_per_device<kernel>(device, kPrefillSmemBytes);
  const dim3 grid(
      (p.n + kPrefillBN - 1) / kPrefillBN, (p.m + kPrefillBM - 1) / kPrefillBM, static_cast<uint32_t>(out.size(0)));
  LaunchKernel(grid, kPrefillThreads, device, kPrefillSmemBytes)(kernel, p);
}

}  // namespace sglang::sm70_fixed_order_gemm
