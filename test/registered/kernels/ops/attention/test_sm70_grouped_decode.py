"""SM70 split-KV grouped decode (sm70_grouped_decode) over fp16 KV, head_dim 256.

Not registered for GPU CI (V100 worktree only).
"""

from __future__ import annotations

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(
    est_time=60,
    stage="base-b-kernel-unit",
    runner_config="1-gpu-large",
    disabled="SM70 V100 worktree only; do not register GPU CI",
)

DIM = 256
PAGE = 16
SCALE = DIM**-0.5
# fp16 output of an fp32-accumulated softmax average; measured max error is ~5e-4.
ATOL = 2e-3


def _sm70() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (7, 0)


def _paged_slots(lens: list[int], gen: torch.Generator) -> tuple[torch.Tensor, int]:
    """Token slots of shuffled 16-token pages, as the paged allocator hands out."""
    pages = [-(-length // PAGE) for length in lens]
    perm = torch.randperm(sum(pages) + 2, generator=gen)
    slots, offset = [], 0
    for length, count in zip(lens, pages):
        t = torch.arange(length)
        slots.append(perm[offset : offset + count][t // PAGE] * PAGE + t % PAGE)
        offset += count
    return torch.cat(slots), (sum(pages) + 2) * PAGE


def _case(lens, *, heads, kv_heads, seed, paged=True):
    gen = torch.Generator().manual_seed(seed)
    if paged:
        slots, pool = _paged_slots(lens, gen)
    else:
        pool = sum(lens) + 7
        slots = torch.randperm(pool, generator=gen)[: sum(lens)]
    indptr = torch.zeros(len(lens) + 1, dtype=torch.int32)
    indptr[1:] = torch.cumsum(torch.tensor(lens), 0)
    return dict(
        q=torch.randn(len(lens), heads, DIM, generator=gen).half().cuda(),
        k=torch.randn(pool, kv_heads, DIM, generator=gen).half().cuda(),
        v=torch.randn(pool, kv_heads, DIM, generator=gen).half().cuda(),
        kv_indptr=indptr.cuda(),
        kv_indices=slots.to(torch.int64).cuda(),
    )


def _reference(case, *, k_scale=1.0, v_scale=1.0) -> torch.Tensor:
    q, indptr = case["q"].float(), case["kv_indptr"].tolist()
    group = q.shape[1] // case["k"].shape[1]
    out = []
    for b in range(q.shape[0]):
        slots = case["kv_indices"][indptr[b] : indptr[b + 1]]
        k = case["k"][slots].float().repeat_interleave(group, 1) * k_scale
        v = case["v"][slots].float().repeat_interleave(group, 1) * v_scale
        scores = torch.einsum("hd,lhd->hl", q[b], k) * SCALE
        out.append(torch.einsum("hl,lhd->hd", scores.softmax(-1), v))
    return torch.stack(out)


def _leave_stale_nan() -> None:
    # Volta's L1 shares SRAM with shared memory, so this also seeds shared memory.
    junk = torch.full((64 << 20,), float("nan"), device="cuda")
    junk.sum()
    del junk


def _run(case, **kwargs) -> torch.Tensor:
    from sglang.kernels.ops.attention.sm70_grouped_decode import sm70_grouped_decode

    return sm70_grouped_decode(
        case["q"],
        case["k"],
        case["v"],
        case["kv_indptr"],
        case["kv_indices"],
        softmax_scale=SCALE,
        **kwargs,
    )


@unittest.skipUnless(_sm70(), "SM70 grouped decode requires an NVIDIA V100")
class TestSm70GroupedDecode(CustomTestCase):
    def _assert_close(self, actual, expected, msg):
        err = (actual.float() - expected).abs().max().item()
        self.assertLessEqual(err, ATOL, f"{msg}: max abs error {err:.3e}")

    def test_matches_fp32_reference(self):
        # Lengths 1 and 17 leave a partial tile and a partial chunk; 4097 and the
        # 16-request batch split one request over several CTAs with a ragged tail.
        # Each call follows stale NaN, which a read of unwritten memory picks up.
        lens_cases = [[1], [17], [257], [1, 17, 4096, 4097, 12000]]
        lens_cases.append([4096 + 3000 * i + (i % 3) for i in range(16)])
        for group, kv_heads in ((4, 1), (2, 2), (8, 1)):
            for paged in (True, False):
                for lens in lens_cases:
                    case = _case(
                        lens,
                        heads=group * kv_heads,
                        kv_heads=kv_heads,
                        seed=sum(lens) + group,
                        paged=paged,
                    )
                    _leave_stale_nan()
                    self._assert_close(
                        _run(case),
                        _reference(case),
                        f"group {group} kv_heads {kv_heads} paged {paged} lens {lens[:4]}",
                    )

    def test_kv_scales(self):
        case = _case([300, 5000], heads=4, kv_heads=1, seed=3)
        self._assert_close(
            _run(case, k_scale=0.75, v_scale=1.25),
            _reference(case, k_scale=0.75, v_scale=1.25),
            "k_scale 0.75 v_scale 1.25",
        )

    def test_batch_limit(self):
        # The backend sends batches up to MAX_BATCH here; the C++ chunk cap must agree.
        from sglang.kernels.ops.attention.sm70_grouped_decode import MAX_BATCH

        lens = [1 + (i * 7) % 40 for i in range(MAX_BATCH + 1)]
        case = _case(lens[:-1], heads=4, kv_heads=1, seed=13)
        self._assert_close(_run(case), _reference(case), f"batch {MAX_BATCH}")
        with self.assertRaisesRegex(RuntimeError, "exceeds 1024"):
            _run(_case(lens, heads=4, kv_heads=1, seed=13))

    def test_full_context(self):
        case = _case([262144], heads=4, kv_heads=1, seed=11)
        self._assert_close(_run(case), _reference(case), "one request of 262144")

    def test_matches_triton_decode(self):
        from sglang.kernels.ops.attention.decode_attention import decode_attention_fwd

        lens = [1000, 3001, 8999, 17000, 33333, 65536]
        case = _case(lens, heads=4, kv_heads=1, seed=5)
        splits = 128
        logits = torch.empty(
            len(lens), 4, splits, DIM, dtype=torch.float32, device="cuda"
        )
        lse = torch.empty(len(lens), 4, splits, dtype=torch.float32, device="cuda")
        num_splits = torch.full((len(lens),), splits, dtype=torch.int32, device="cuda")
        triton_out = torch.empty_like(case["q"])
        decode_attention_fwd(
            case["q"],
            case["k"],
            case["v"],
            triton_out,
            case["kv_indptr"],
            case["kv_indices"],
            logits,
            lse,
            num_splits,
            splits,
            SCALE,
            1.0,
            1.0,
            page_size=PAGE,
        )
        self._assert_close(_run(case), triton_out.float(), "vs Triton split-K")

    def test_graph_replay_follows_live_lengths(self):
        # The chunk split is derived on the device, so a graph captured at one set
        # of lengths must give the eager result for any other set it is replayed with.
        captured = _case([4096, 1, 300, 20000], heads=4, kv_heads=1, seed=7)
        replayed = _case([1, 70000, 17, 4097], heads=4, kv_heads=1, seed=8)
        pool = max(captured["k"].shape[0], replayed["k"].shape[0])
        static = dict(
            q=captured["q"].clone(),
            k=torch.zeros(pool, 1, DIM, dtype=torch.float16, device="cuda"),
            v=torch.zeros(pool, 1, DIM, dtype=torch.float16, device="cuda"),
            kv_indptr=captured["kv_indptr"].clone(),
            kv_indices=torch.zeros(80000, dtype=torch.int64, device="cuda"),
        )

        def load(case):
            static["q"].copy_(case["q"])
            static["k"][: case["k"].shape[0]].copy_(case["k"])
            static["v"][: case["v"].shape[0]].copy_(case["v"])
            static["kv_indptr"].copy_(case["kv_indptr"])
            static["kv_indices"][: case["kv_indices"].numel()].copy_(case["kv_indices"])

        load(captured)
        out = torch.empty_like(static["q"])
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            _run(static, out=out)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _run(static, out=out)

        load(replayed)
        graph.replay()
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(out, _run(static)), "replay differs from eager")
        self._assert_close(out, _reference(replayed), "replayed lengths")


if __name__ == "__main__":
    unittest.main()
