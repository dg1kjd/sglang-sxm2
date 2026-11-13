// Exact per-row top-k whose output depends only on the scores: ties at the
// cut-off go to the smaller index and the indices come out ascending.
//
// One block of 1024 threads handles one row:
//   1. Find the k-th largest score exactly. Scores map to 32-bit keys with the
//      same order; up to three histogram passes over the row resolve the key
//      12 + 12 + 8 bits at a time. Each pass only narrows the key range, so no
//      candidate list is kept and no row can overflow one.
//   2. Mark every element above the cut-off, and every element equal to it, in
//      two bit masks laid out in index order.
//   3. Count the marks in front of each element. That count is the element's
//      output slot, so the output is in index order whatever order the threads
//      run in, and "the first m equal elements" needs no sorting.
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

namespace ordered_topk_detail {

constexpr uint32_t kThreads = 1024;
constexpr uint32_t kWarps = kThreads / device::kWarpThreads;
// Scores are read as 16-byte vectors of 4.
constexpr uint32_t kVecElems = 4;
constexpr uint32_t kHistBins = 4096;
constexpr uint32_t kBinsPerThread = kHistBins / kThreads;
// A group is the 32 vectors (128 scores) one warp reads at once. The marks of
// 16 groups per warp (65536 scores) are held in shared memory at a time; a
// longer row is marked and written in several windows of that size.
constexpr uint32_t kGroupVecs = device::kWarpThreads;
constexpr uint32_t kWindowGroupsPerWarp = 16;
constexpr uint32_t kWindowGroups = kWarps * kWindowGroupsPerWarp;
constexpr uint32_t kWindowVecs = kWindowGroups * kGroupVecs;
// Stands for "take every element equal to the cut-off".
constexpr uint32_t kTakeAllEqual = 0xFFFFFFFFu;

static_assert(device::kWarpThreads == 32, "the masks hold one bit per lane of a 32-thread warp");
static_assert(kHistBins % kThreads == 0);
// A group's two mark counts are packed into 16 bits each.
static_assert(kGroupVecs * kVecElems < (1u << 16));

struct OrderedTopKParams {
  const float* __restrict__ input;         // [B, input_stride]
  const int32_t* __restrict__ row_starts;  // [B]
  int32_t* __restrict__ indices;           // [B, topk]
  const int32_t* __restrict__ lengths;     // [B]
  int64_t input_stride;
  int32_t topk;
};

struct ThresholdBin {
  uint32_t bin;
  uint32_t need;   // how many elements of this bin are still to be selected
  uint32_t count;  // how many elements the bin holds
};

// Maps a score to a key with the same order: a larger score gives a larger
// key. -0.0 gets the key of +0.0, so the two tie. NaN is ordered by its bits:
// above +inf with the sign bit clear, below -inf with it set.
SGL_DEVICE uint32_t score_key(float score) {
  const uint32_t bits = __float_as_uint(score);
  // Flip the sign bit of a positive score and every bit of a negative one.
  const uint32_t key = bits ^ (static_cast<uint32_t>(static_cast<int32_t>(bits) >> 31) | 0x80000000u);
  // -0.0 lands one below +0.0 (0x80000000); no other score maps to that key.
  return key + (key == 0x7FFFFFFFu);
}

// One row, addressed in 16-byte vectors. The row may start anywhere inside a
// vector, so the first and the last vector can contain up to three floats
// that are not part of the row. `load` reads those two vectors one float at a
// time and only the row's own floats, so nothing outside the tensor is read.
struct RowView {
  const float* __restrict__ base;  // row start rounded down to 16 bytes
  uint32_t head;                   // floats between `base` and the row start
  uint32_t length;
  uint32_t num_vecs;

  SGL_DEVICE RowView(const float* __restrict__ row, uint32_t length) : length(length) {
    head = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(row) / sizeof(float)) % kVecElems;
    base = row - head;
    num_vecs = (head + length + kVecElems - 1) / kVecElems;
  }

  // Row-relative index of float 0 of vector `vec`; wraps below zero for the
  // floats in front of the row.
  SGL_DEVICE uint32_t first_index(uint32_t vec) const {
    return vec * kVecElems - head;
  }

  // Loads the keys of vector `vec`; bit j of the result is set when float j
  // belongs to the row, and only those keys are meaningful. Only the first and
  // the last vector can hold floats from outside the row.
  SGL_DEVICE uint32_t load(uint32_t vec, uint32_t key[kVecElems]) const {
    if (vec != 0 && vec != num_vecs - 1) {
      device::AlignedVector<float, kVecElems> scores;
      scores.load(base, vec);
#pragma unroll
      for (uint32_t j = 0; j < kVecElems; ++j) {
        key[j] = score_key(scores[j]);
      }
      return (1u << kVecElems) - 1;
    }
    const uint32_t first = first_index(vec);
    uint32_t valid = 0;
#pragma unroll
    for (uint32_t j = 0; j < kVecElems; ++j) {
      const bool in_row = first + j < length;
      key[j] = in_row ? score_key(base[vec * kVecElems + j]) : 0;
      valid |= static_cast<uint32_t>(in_row) << j;
    }
    return valid;
  }
};

SGL_DEVICE void clear_histogram(uint32_t* s_hist) {
#pragma unroll
  for (uint32_t i = 0; i < kBinsPerThread; ++i) {
    s_hist[threadIdx.x + i * kThreads] = 0;
  }
}

// Counts, per value of the digit `(key >> kDigitShift) & kDigitMask`, the row
// elements whose key matches `prefix` above bit kPrefixShift (all elements
// when kPrefixShift is 32). With kTrackRange it also records the smallest and
// largest matching key.
template <uint32_t kPrefixShift, uint32_t kDigitShift, uint32_t kDigitMask, bool kTrackRange>
SGL_DEVICE void histogram_pass(const RowView& row, uint32_t prefix, uint32_t* s_hist, uint32_t* s_key_range) {
  static_assert(kDigitMask < kHistBins);
  uint32_t key_min = 0xFFFFFFFFu;
  uint32_t key_max = 0;
  // A thread adds a run of equal digits to the histogram once. Rows with many
  // equal scores would otherwise send every thread to the same counter.
  uint32_t run_digit = 0;
  uint32_t run_count = 0;
  for (uint32_t vec = threadIdx.x; vec < row.num_vecs; vec += kThreads) {
    uint32_t key[kVecElems];
    const uint32_t valid = row.load(vec, key);
#pragma unroll
    for (uint32_t j = 0; j < kVecElems; ++j) {
      bool counted = (valid >> j) & 1;
      if constexpr (kPrefixShift < 32) {
        counted = counted && (key[j] >> kPrefixShift) == prefix;
      }
      if (counted) {
        const uint32_t digit = (key[j] >> kDigitShift) & kDigitMask;
        if (digit != run_digit) {
          ::atomicAdd(&s_hist[run_digit], run_count);
          run_digit = digit;
          run_count = 0;
        }
        run_count += 1;
        if constexpr (kTrackRange) {
          key_min = min(key_min, key[j]);
          key_max = max(key_max, key[j]);
        }
      }
    }
  }
  ::atomicAdd(&s_hist[run_digit], run_count);
  if constexpr (kTrackRange) {
    if (key_min <= key_max) {
      ::atomicMin(&s_key_range[0], key_min);
      ::atomicMax(&s_key_range[1], key_max);
    }
  }
}

// Finds the bin that holds the `need`-th largest counted element: fewer than
// `need` elements lie in higher bins, at least `need` with this bin included.
// The histogram must hold at least `need` elements. Ends with a block barrier.
SGL_DEVICE ThresholdBin
find_threshold_bin(const uint32_t* s_hist, uint32_t* s_warp_total, uint32_t* s_warp_base, uint32_t* s_found, uint32_t need) {
  const uint32_t lane = device::get_lane_id();
  const uint32_t warp_id = threadIdx.x / device::kWarpThreads;
  // Thread t owns bins kHistBins-1-4t down to kHistBins-4-4t, so a forward
  // sum over threads counts the elements in higher bins.
  const uint32_t top_bin = kHistBins - 1 - threadIdx.x * kBinsPerThread;
  uint32_t count[kBinsPerThread];
  uint32_t local = 0;
#pragma unroll
  for (uint32_t i = 0; i < kBinsPerThread; ++i) {
    count[i] = s_hist[top_bin - i];
    local += count[i];
  }
  const uint32_t inclusive = device::warp::inclusive_sum(local);
  if (lane == device::kWarpThreads - 1) s_warp_total[warp_id] = inclusive;
  __syncthreads();
  if (warp_id == 0) {
    const uint32_t total = s_warp_total[lane];
    s_warp_base[lane] = device::warp::inclusive_sum(total) - total;
  }
  __syncthreads();
  uint32_t above = s_warp_base[warp_id] + inclusive - local;
#pragma unroll
  for (uint32_t i = 0; i < kBinsPerThread; ++i) {
    if (above < need && need <= above + count[i]) {
      s_found[0] = top_bin - i;
      s_found[1] = need - above;
      s_found[2] = count[i];
    }
    above += count[i];
  }
  __syncthreads();
  return ThresholdBin{.bin = s_found[0], .need = s_found[1], .count = s_found[2]};
}

// Rows longer than topk. Selects every element whose key is above the cut-off
// key, plus the lowest-index elements equal to it, and writes the indices in
// ascending order.
SGL_DEVICE void select_row(const float* __restrict__ row_scores, int32_t* __restrict__ out, uint32_t length, uint32_t topk) {
  __shared__ uint32_t s_hist[kHistBins];
  // Per group: 4 words of "above" marks, then 4 words of "equal" marks. Word j
  // has bit `lane` set for float j of that lane's vector.
  __shared__ uint32_t s_mask[kWindowGroups][2 * kVecElems];
  __shared__ uint32_t s_group_count[kWindowGroups];  // above | equal << 16
  __shared__ uint32_t s_warp_above[kWarps];
  __shared__ uint32_t s_warp_equal[kWarps];
  __shared__ uint32_t s_base_above[kWarps];
  __shared__ uint32_t s_base_equal[kWarps];
  __shared__ uint32_t s_window_total[2];
  __shared__ uint32_t s_found[3];
  __shared__ uint32_t s_key_range[2];

  const uint32_t lane = device::get_lane_id();
  const uint32_t warp_id = threadIdx.x / device::kWarpThreads;
  const RowView row(row_scores, length);

  // Step 1: the cut-off key, and how many elements equal to it are selected.
  clear_histogram(s_hist);
  if (threadIdx.x == 0) {
    s_key_range[0] = 0xFFFFFFFFu;
    s_key_range[1] = 0;
  }
  __syncthreads();
  histogram_pass<32, 20, 0xFFF, false>(row, 0, s_hist, s_key_range);
  __syncthreads();
  const ThresholdBin top = find_threshold_bin(s_hist, s_warp_above, s_base_above, s_found, topk);

  uint32_t cutoff = top.bin << 20;
  uint32_t need_equal = kTakeAllEqual;
  // When the bin is selected as a whole, "at or above its lowest key" is the
  // answer. The same holds for the narrower bins below.
  if (top.need != top.count) {
    clear_histogram(s_hist);
    __syncthreads();
    histogram_pass<20, 8, 0xFFF, true>(row, top.bin, s_hist, s_key_range);
    __syncthreads();
    if (s_key_range[0] == s_key_range[1]) {
      // The whole bin is one value: the usual case when scores tie.
      cutoff = s_key_range[0];
      need_equal = top.need;
    } else {
      const ThresholdBin mid = find_threshold_bin(s_hist, s_warp_above, s_base_above, s_found, top.need);
      const uint32_t prefix = (top.bin << 12) | mid.bin;
      cutoff = prefix << 8;
      if (mid.need != mid.count) {
        clear_histogram(s_hist);
        __syncthreads();
        histogram_pass<8, 0, 0xFF, false>(row, prefix, s_hist, s_key_range);
        __syncthreads();
        const ThresholdBin low = find_threshold_bin(s_hist, s_warp_above, s_base_above, s_found, mid.need);
        cutoff |= low.bin;
        need_equal = low.need;
      }
    }
  }

  // Steps 2 and 3, one window of the row at a time. Inside a window each warp
  // owns a run of consecutive groups, so it can count its own marks as it goes
  // and only the per-warp totals need a sum across warps.
  uint32_t above_before_window = 0;
  uint32_t equal_before_window = 0;
  for (uint32_t window_vec = 0; window_vec < row.num_vecs; window_vec += kWindowVecs) {
    const uint32_t window_vecs = min(row.num_vecs - window_vec, kWindowVecs);
    const uint32_t window_groups = (window_vecs + kGroupVecs - 1) / kGroupVecs;
    const uint32_t groups_per_warp = (window_groups + kWarps - 1) / kWarps;
    const uint32_t first_group = warp_id * groups_per_warp;
    const uint32_t end_group = min(first_group + groups_per_warp, window_groups);

    uint32_t warp_above = 0;
    uint32_t warp_equal = 0;
    for (uint32_t group = first_group; group < end_group; ++group) {
      const uint32_t vec = window_vec + group * kGroupVecs + lane;
      uint32_t above = 0;
      uint32_t equal = 0;
      if (vec < row.num_vecs) {
        uint32_t key[kVecElems];
        const uint32_t valid = row.load(vec, key);
#pragma unroll
        for (uint32_t j = 0; j < kVecElems; ++j) {
          above |= static_cast<uint32_t>(key[j] > cutoff) << j;
          equal |= static_cast<uint32_t>(key[j] == cutoff) << j;
        }
        above &= valid;
        equal &= valid;
      }
      // Most groups of a long row have no "equal" mark, many have none at all.
      const bool any_above = __any_sync(device::warp::kFullMask, above != 0);
      const bool any_equal = __any_sync(device::warp::kFullMask, equal != 0);
      uint32_t group_above = 0;
      uint32_t group_equal = 0;
#pragma unroll
      for (uint32_t j = 0; j < kVecElems; ++j) {
        uint32_t above_word = 0;
        uint32_t equal_word = 0;
        if (any_above) above_word = __ballot_sync(device::warp::kFullMask, (above >> j) & 1);
        if (any_equal) equal_word = __ballot_sync(device::warp::kFullMask, (equal >> j) & 1);
        if (lane == 0) {
          s_mask[group][j] = above_word;
          s_mask[group][kVecElems + j] = equal_word;
        }
        group_above += __popc(above_word);
        group_equal += __popc(equal_word);
      }
      if (lane == 0) s_group_count[group] = group_above | (group_equal << 16);
      warp_above += group_above;
      warp_equal += group_equal;
    }
    if (lane == 0) {
      s_warp_above[warp_id] = warp_above;
      s_warp_equal[warp_id] = warp_equal;
    }
    __syncthreads();
    if (warp_id == 0) {
      const uint32_t above = s_warp_above[lane];
      const uint32_t equal = s_warp_equal[lane];
      const uint32_t above_sum = above_before_window + device::warp::inclusive_sum(above);
      const uint32_t equal_sum = equal_before_window + device::warp::inclusive_sum(equal);
      s_base_above[lane] = above_sum - above;
      s_base_equal[lane] = equal_sum - equal;
      if (lane == device::kWarpThreads - 1) {
        s_window_total[0] = above_sum;
        s_window_total[1] = equal_sum;
      }
    }
    __syncthreads();
    above_before_window = s_window_total[0];
    equal_before_window = s_window_total[1];

    // Marks in front of the current group, over the whole row.
    uint32_t above_before = s_base_above[warp_id];
    uint32_t equal_before = s_base_equal[warp_id];
    const uint32_t lower_lanes = (1u << lane) - 1;
    for (uint32_t group = first_group; group < end_group; ++group) {
      const uint32_t group_count = s_group_count[group];
      if (group_count == 0) continue;
      uint32_t my_above = above_before;
      uint32_t my_equal = equal_before;
      uint32_t above = 0;
      uint32_t equal = 0;
#pragma unroll
      for (uint32_t j = 0; j < kVecElems; ++j) {
        const uint32_t above_word = s_mask[group][j];
        const uint32_t equal_word = s_mask[group][kVecElems + j];
        my_above += __popc(above_word & lower_lanes);
        my_equal += __popc(equal_word & lower_lanes);
        above |= ((above_word >> lane) & 1) << j;
        equal |= ((equal_word >> lane) & 1) << j;
      }
      if ((above | equal) != 0) {
        const uint32_t first = row.first_index(window_vec + group * kGroupVecs + lane);
#pragma unroll
        for (uint32_t j = 0; j < kVecElems; ++j) {
          const bool is_above = (above >> j) & 1;
          const bool is_equal = (equal >> j) & 1;
          // The slot is the number of selected elements in front: all marked
          // "above", and at most need_equal of those marked "equal".
          if (is_above || (is_equal && my_equal < need_equal)) {
            out[my_above + min(my_equal, need_equal)] = static_cast<int32_t>(first + j);
          }
          my_above += is_above;
          my_equal += is_equal;
        }
      }
      above_before += group_count & 0xFFFF;
      equal_before += group_count >> 16;
    }
    // The next window reuses the mark buffers.
    __syncthreads();
  }
}

template <bool kUsePDL>
__global__ __launch_bounds__(ordered_topk_detail::kThreads) void ordered_topk_kernel(
    const ordered_topk_detail::OrderedTopKParams __grid_constant__ params) {
  using namespace ordered_topk_detail;
  device::PDLWaitPrimary<kUsePDL>();

  const auto row = static_cast<uint64_t>(blockIdx.x);
  const int32_t length = params.lengths[row];
  const int32_t topk = params.topk;
  int32_t* __restrict__ out = params.indices + row * topk;
  if (length <= topk) {
    // Every element is selected; the scores are not read.
    for (int32_t i = threadIdx.x; i < topk; i += kThreads) {
      out[i] = (i < length) ? i : -1;
    }
  } else {
    const float* __restrict__ scores = params.input + row * params.input_stride + params.row_starts[row];
    select_row(scores, out, static_cast<uint32_t>(length), static_cast<uint32_t>(topk));
  }

  device::PDLTriggerSecondary<kUsePDL>();
}

}  // namespace ordered_topk_detail

/**
 * \brief Exact, ordered per-row top-k over ragged rows of a fp32 score matrix.
 *
 * Row b selects the topk largest values in
 * score[b, row_starts[b] : row_starts[b] + lengths[b]) and writes their
 * indices, relative to row_starts[b], in ascending order. Among equal scores
 * at the cut-off the smaller index is selected. -0.0 equals +0.0. NaN is
 * ordered by its bits: above +inf with the sign bit clear, below -inf with it
 * set. A row with lengths[b] <= topk gets 0..lengths[b]-1 followed by -1, and
 * its scores are not read.
 *
 * \tparam kUsePDL Whether to launch with PDL enabled
 * \param score      [B, L] fp32, rows `input_stride` apart
 * \param row_starts [B] int32
 * \param indices    [B, topk] int32 output
 * \param lengths    [B] int32
 */
template <bool kUsePDL>
struct OrderedTopKKernel {
  static constexpr auto kernel = ordered_topk_detail::ordered_topk_kernel<kUsePDL>;

  static void
  run(const tvm::ffi::TensorView score,
      const tvm::ffi::TensorView row_starts,
      const tvm::ffi::TensorView indices,
      const tvm::ffi::TensorView lengths) {
    using namespace host;
    auto B = SymbolicSize{"batch"};
    auto K = SymbolicSize{"topk"};
    auto S = SymbolicSize{"input_stride"};
    auto device = SymbolicDevice{};
    device.set_options<kDLGPU>();

    TensorMatcher({B, -1})  // score
        .with_strides({S, 1})
        .with_dtype<fp32_t>()
        .with_device(device)
        .verify(score);
    TensorMatcher({B})  // row_starts, lengths
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(row_starts)
        .verify(lengths);
    TensorMatcher({B, K})  // indices
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(indices);
    // The kernel reads whole 16-byte vectors inside a row and single floats at
    // its two ends.
    CHECK_HOST(reinterpret_cast<uintptr_t>(score.data_ptr()) % sizeof(float) == 0)
        << "ordered_topk: score is not aligned to 4 bytes";

    const auto params = ordered_topk_detail::OrderedTopKParams{
        .input = static_cast<const float*>(score.data_ptr()),
        .row_starts = static_cast<const int32_t*>(row_starts.data_ptr()),
        .indices = static_cast<int32_t*>(indices.data_ptr()),
        .lengths = static_cast<const int32_t*>(lengths.data_ptr()),
        .input_stride = S.unwrap(),
        .topk = static_cast<int32_t>(K.unwrap()),
    };

    const auto num_rows = static_cast<uint32_t>(B.unwrap());
    LaunchKernel(num_rows, ordered_topk_detail::kThreads, device.unwrap()).enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace sglang
