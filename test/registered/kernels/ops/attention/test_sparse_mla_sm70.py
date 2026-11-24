"""SM70 sparse MLA (sparse_mla_sm70) over fp16 latent rows.

The NoPE path (no rope tail, sinks or lengths) must give the same bytes as the
digests below. The rope-tail path (512-d latent plus a 64-d rope tail, per-head
sinks, per-row lengths) is checked against an fp32 reference.

Not registered for GPU CI (V100 worktree only).
"""

from __future__ import annotations

import hashlib
import unittest
from unittest import mock

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(
    est_time=180,
    stage="base-b-kernel-unit",
    runner_config="1-gpu-large",
    disabled="SM70 V100 worktree only; do not register GPU CI",
)

LATENT = 512
ROPE = 64
SCALE = 0.0625
# fp16 output of an fp32 softmax over fp16 P; measured max error is ~1e-3 at |out| ~ 3.
ATOL = 2e-3
RTOL = 2e-3


def _sm70() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (7, 0)


def _leave_stale_nan() -> None:
    # Volta's L1 shares SRAM with shared memory, so this also seeds shared memory.
    junk = torch.full((64 << 20,), float("nan"), device="cuda")
    junk.sum()
    del junk


def _op():
    from sglang.kernels.ops.attention import sparse_mla_sm70 as mod

    return mod


def _run(case, **overrides) -> torch.Tensor:
    args = dict(
        q_rope=case["q_rope"],
        attn_sink=case["sink"],
        topk_length=case["lengths"],
    )
    args.update(overrides)
    return _op().sparse_mla_sm70(case["q"], case["kv"], case["idx"], SCALE, **args)[0]


def _reference(case, *, use_rope=True, use_sink=True, lengths="case") -> torch.Tensor:
    """fp32: gather the rows, score latent + rope tail, add exp(sink) to the denominator."""
    q, kv, idx = case["q"], case["kv"], case["idx"]
    q_rope = case["q_rope"] if use_rope else None
    sink = case["sink"] if use_sink else None
    if lengths == "case":
        lengths = case["lengths"]
    S, H, _ = q.shape
    W = idx.shape[-1]
    kv2 = kv.reshape(kv.shape[0], -1)[
        :, : LATENT + (ROPE if case["q_rope"] is not None else 0)
    ].float()
    out = torch.empty(S, H, LATENT, dtype=torch.float32, device=q.device)
    for s0 in range(0, S, 32):
        s1 = min(S, s0 + 32)
        ix = idx[s0:s1].long()
        valid = ix >= 0
        if lengths is not None:
            pos = torch.arange(W, device=ix.device)[None, :]
            valid &= pos < lengths[s0:s1, None].long()
        rows = kv2[ix.clamp_min(0)]
        sc = torch.einsum("shd,swd->shw", q[s0:s1].float(), rows[..., :LATENT])
        if q_rope is not None:
            sc = sc + torch.einsum(
                "shd,swd->shw", q_rope[s0:s1].float(), rows[..., LATENT:]
            )
        sc = (sc * SCALE).masked_fill(~valid[:, None, :], float("-inf"))
        m = sc.amax(-1, keepdim=True)
        if sink is not None:
            m = torch.maximum(m, sink.view(1, H, 1))
        m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
        p = torch.exp(sc - m)
        den = p.sum(-1, keepdim=True)
        if sink is not None:
            den = den + torch.exp(sink.view(1, H, 1) - m)
        o = torch.einsum("shw,swd->shd", p, rows[..., :LATENT]) / den.clamp_min(1e-30)
        out[s0:s1] = torch.where(valid.any(-1)[:, None, None], o, torch.zeros_like(o))
    return out


def _case(
    S, H, W, *, seed, tail=ROPE, pool=50_000, invalid=0.0, lengths=None, sink=True
):
    """Random rows; `invalid` of the slots are -1, `lengths` is a list, "random" or None."""
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, pool, (S, W), generator=gen, dtype=torch.int32)
    if invalid:
        idx[torch.rand(S, W, generator=gen) < invalid] = -1
    if lengths == "random":
        lengths = torch.randint(-2, W + 3, (S,), generator=gen).tolist()
    sink_t = None
    if sink:
        sink_t = torch.linspace(-4.0, 12.0, H)[torch.randperm(H, generator=gen)]
    return dict(
        q=torch.randn(S, H, LATENT, generator=gen).half().cuda(),
        q_rope=torch.randn(S, H, tail, generator=gen).half().cuda() if tail else None,
        kv=torch.randn(pool, 1, LATENT + tail, generator=gen).half().cuda(),
        idx=idx.cuda(),
        lengths=None
        if lengths is None
        else torch.tensor(lengths, dtype=torch.int32).cuda(),
        sink=None if sink_t is None else sink_t.float().cuda(),
    )


@unittest.skipUnless(_sm70(), "SM70 sparse MLA requires an NVIDIA V100")
class TestSparseMlaSm70Rope(CustomTestCase):
    def _assert_close(self, actual, expected, msg):
        diff = (actual.float() - expected).abs()
        bound = ATOL + RTOL * expected.abs()
        worst = (diff - bound).max().item()
        self.assertLessEqual(
            worst,
            0.0,
            f"{msg}: max abs error {diff.max().item():.3e} (ref max {expected.abs().max().item():.2f})",
        )

    def _assert_differs(self, a, b, msg):
        # Guards the sensitivity of a check: the variants must differ well past the tolerance.
        self.assertGreater((a - b).abs().max().item(), 20 * ATOL, msg)

    def test_matches_fp32_reference(self):
        # Heads 4/8/16 are the Hy4 DSA and window layers at TP8 and TP4; 20
        # leaves a partial head tile. Widths 100 and 2047 leave a partial key
        # tile. One row is enough for the split path, 96 rows run one split.
        # Each call follows stale NaN, which a read of unwritten memory picks up.
        for sink in (True, False):
            for tail in (ROPE, 0):
                for H in (4, 8, 16, 20):
                    for W in (512, 2048, 100, 2047):
                        for S in (1, 6, 96):
                            case = _case(
                                S,
                                H,
                                W,
                                seed=S * 7 + H * 3 + W + tail,
                                tail=tail,
                                invalid=0.1,
                                lengths="random",
                                sink=sink,
                            )
                            _leave_stale_nan()
                            self._assert_close(
                                _run(case),
                                _reference(case),
                                f"sink {sink} tail {tail} H {H} W {W} S {S}",
                            )

    def test_prefill_rows(self):
        # Causal DSA rows: position p keeps its first min(p + 1, W) slots. Window
        # rows: the same layout over a 512-slot window.
        for S, H, W, start in (
            (3000, 4, 2048, 500),
            (3000, 8, 512, 0),
            (1000, 16, 512, 300),
        ):
            case = _case(
                S, H, W, seed=S + H, lengths=[min(start + i + 1, W) for i in range(S)]
            )
            pos = torch.arange(W, device="cuda")[None, :]
            case["idx"] = torch.where(
                pos < case["lengths"][:, None],
                case["idx"],
                torch.full_like(case["idx"], -1),
            )
            _leave_stale_nan()
            self._assert_close(
                _run(case), _reference(case), f"prefill S {S} H {H} W {W}"
            )

    def test_sinks(self):
        # Sinks from far below to far above the scores; short rows make the sink
        # a large share of the denominator. One split applies the sink in the
        # main kernel, several in the combine kernel.
        H = 8
        for W, lengths in ((2048, None), (2048, [1, 2, 3, 40]), (512, [0, 1, 7, 512])):
            S = 4
            case = _case(S, H, W, seed=W + S, lengths=lengths)
            case["sink"] = torch.tensor(
                [-30.0, -2.0, 0.0, 3.0, 6.0, 9.0, 12.0, 30.0], device="cuda"
            )
            ref = _reference(case)
            self._assert_differs(
                ref, _reference(case, use_sink=False), f"sinks move the output, W {W}"
            )
            for splits in (1, 4):
                with mock.patch.object(_op(), "_rope_num_splits", return_value=splits):
                    _leave_stale_nan()
                    self._assert_close(
                        _run(case),
                        ref,
                        f"sinks W {W} lengths {lengths} splits {splits}",
                    )

    def test_lengths(self):
        # Slots past the length hold valid indices, so a scan that ignores the
        # length, or stops one slot early or late, changes the output. Lengths
        # sit on both sides of the 32-slot tiles; negative and oversized ones clamp.
        W = 256
        lengths = [0, 1, 2, 3, 31, 32, 33, 63, 64, 65, 255, 256, 257, -1]
        case = _case(len(lengths), 4, W, seed=11, lengths=lengths)
        ref = _reference(case)
        shorter = torch.clamp(case["lengths"] - 1, min=0)
        self._assert_differs(
            ref[1:4],
            _reference(case, lengths=shorter)[1:4],
            "one slot moves short rows",
        )
        for splits in (1, 3, 8):
            with mock.patch.object(_op(), "_rope_num_splits", return_value=splits):
                _leave_stale_nan()
                self._assert_close(_run(case), ref, f"lengths, {splits} splits")

    def test_rope_tail(self):
        case = _case(6, 8, 2048, seed=5)
        ref = _reference(case)
        self._assert_differs(
            ref, _reference(case, use_rope=False), "the rope tail moves the output"
        )
        self._assert_close(_run(case), ref, "rope tail")

    def test_split_counts(self):
        # The split count follows the grid size; every count must give the same result.
        for W in (2048, 100):
            case = _case(2, 8, W, seed=W, invalid=0.2, lengths=[W - 5, W])
            ref = _reference(case)
            tiles = -(-W // 32)
            for splits in sorted({1, 2, 3, 5, tiles} & set(range(1, tiles + 1))):
                with mock.patch.object(_op(), "_rope_num_splits", return_value=splits):
                    _leave_stale_nan()
                    out = _run(case)
                self._assert_close(out, ref, f"W {W} splits {splits}")

    def test_empty_rows(self):
        case = _case(4, 8, 512, seed=3, lengths=[512, 0, 512, 10])
        case["idx"][0] = -1
        case["idx"][3, :10] = -1
        _leave_stale_nan()
        out = _run(case)
        self.assertTrue(
            torch.equal(out[[0, 1, 3]], torch.zeros_like(out[[0, 1, 3]])),
            "rows without a slot",
        )
        self._assert_close(out, _reference(case), "empty rows")
        mod = _op()
        empty = mod.sparse_mla_sm70(
            case["q"][:0], case["kv"], case["idx"][:0], SCALE, q_rope=case["q_rope"][:0]
        )
        self.assertEqual(tuple(empty.shape), (1, 0, 8, LATENT))
        none = mod.sparse_mla_sm70(
            case["q"], case["kv"], case["idx"][:, :0], SCALE, q_rope=case["q_rope"]
        )
        self.assertTrue(torch.equal(none, torch.zeros_like(none)))

    def test_empty_rows_without_sinks(self):
        # Without a sink a row with no usable slot has nothing in its softmax
        # denominator; it must still come out as zeros, in the one-split
        # epilogue and in the combine kernel (every split of the row empty).
        # Row 4 has slots and checks that the empty rows leave it alone.
        for tail in (ROPE, 0):
            case = _case(
                5,
                8,
                512,
                seed=13,
                tail=tail,
                lengths=[512, 0, 512, -3, 200],
                sink=False,
            )
            case["idx"][0] = -1
            case["idx"][2, :] = -1
            ref = _reference(case)
            for splits in (1, 4, 16):
                with mock.patch.object(_op(), "_rope_num_splits", return_value=splits):
                    _leave_stale_nan()
                    out = _run(case)
                self.assertTrue(
                    torch.equal(out[:4], torch.zeros_like(out[:4])),
                    f"rows without a slot, no sinks, tail {tail}, {splits} splits",
                )
                self._assert_close(
                    out, ref, f"empty rows, no sinks, tail {tail}, {splits} splits"
                )

    def test_deterministic(self):
        case = _case(2, 8, 2048, seed=9, invalid=0.1)
        first = _run(case)
        _leave_stale_nan()
        self.assertTrue(torch.equal(first, _run(case)), "two runs differ")

    def test_input_layouts(self):
        # q as views of one [S, H, 576] tensor, kv rows with a padded stride, kv
        # as [N, 576], indices as [S, 1, W] int64.
        case = _case(5, 8, 512, seed=21, lengths="random")
        ref = _reference(case)
        q_all = torch.cat([case["q"], case["q_rope"]], dim=-1)
        padded = torch.zeros(
            case["kv"].shape[0], 1, 640, dtype=torch.float16, device="cuda"
        )
        padded[..., : LATENT + ROPE] = case["kv"]
        variants = {
            "q views": dict(q=q_all[..., :LATENT], q_rope=q_all[..., LATENT:]),
            "padded kv": dict(kv=padded[..., : LATENT + ROPE]),
            "2-d kv": dict(kv=case["kv"][:, 0]),
            "3-d int64 indices": dict(idx=case["idx"].long().unsqueeze(1)),
        }
        for name, swap in variants.items():
            c = dict(case, **swap)
            self._assert_close(_run(c), ref, name)

    def test_graph_replay_follows_live_inputs(self):
        # Splits depend on the shape only, so a graph captured with one set of
        # indices, lengths and sinks must give the eager result for another.
        captured = _case(3, 4, 2048, seed=31, invalid=0.1, lengths=[2048, 100, 7])
        replayed = _case(3, 4, 2048, seed=32, invalid=0.3, lengths=[5, 2048, 1000])
        static = {
            k: (v.clone() if v is not None else None) for k, v in captured.items()
        }

        def load(case):
            for k, v in case.items():
                static[k].copy_(v)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            _run(static)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = _run(static)
        load(replayed)
        graph.replay()
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(out, _run(static)), "replay differs from eager")
        self._assert_close(out, _reference(replayed), "replayed inputs")

    def test_rejects_bad_inputs(self):
        case = _case(2, 4, 64, seed=1)
        with self.assertRaises(ValueError):
            _run(case, q_rope=case["q_rope"][..., :32])
        with self.assertRaises(ValueError):
            _run(case, attn_sink=case["sink"][:3])
        rows = case["kv"].shape[0] - 1
        misaligned = (
            case["kv"]
            .view(-1)[1 : 1 + rows * (LATENT + ROPE)]
            .view(rows, 1, LATENT + ROPE)
        )
        with self.assertRaises(ValueError):
            _run(dict(case, kv=misaligned))


def _mul32(x: torch.Tensor, m: int) -> torch.Tensor:
    # x * m mod 2^32 in int64 without overflow.
    return ((x & 0xFFFF) * m + (((x >> 16) * m) & 0xFFFF) * 65536) & 0xFFFFFFFF


def _hash32(i: torch.Tensor, salt: int) -> torch.Tensor:
    x = _mul32((i + salt * 0x632BE5AB) & 0xFFFFFFFF, 0x9E3779B1)
    x = x ^ (x >> 16)
    x = _mul32(x, 0x85EBCA6B)
    x = x ^ (x >> 13)
    x = _mul32(x, 0xC2B2AE35)
    return x ^ (x >> 16)


def _grid_values(shape, salt: int) -> torch.Tensor:
    """Values k / 256 for k in [-256, 256), exact in fp16, from an integer hash."""
    n = 1
    for s in shape:
        n *= s
    x = _hash32(torch.arange(n, dtype=torch.int64), salt)
    return ((x & 0x1FF) - 256).to(torch.float32).div(256).view(shape)


def _nope_case(S, H, W, pool, salt):
    i = torch.arange(S * W, dtype=torch.int64)
    idx = (_hash32(i, salt + 1) % pool).to(torch.int32).view(S, W)
    idx[(_hash32(i, salt + 2) % 11 == 0).view(S, W)] = -1
    return dict(
        q=_grid_values((S, H, LATENT), salt + 3).half().cuda(),
        kv=_grid_values((pool, 1, LATENT), salt + 4).half().cuda(),
        idx=idx.cuda(),
    )


# SHA-256 of the fp16 output bytes of the NoPE path for _nope_case(S, H, W,
# pool, salt). The NoPE kernel merges its splits in a fixed order, so the
# output depends only on its arithmetic.
_NOPE_CASES = (
    (1, 8, 2048, 40_000, 1),
    (4, 16, 2048, 40_000, 2),
    (64, 8, 2048, 40_000, 3),
    (600, 16, 2048, 40_000, 4),
    (7, 8, 100, 1_000, 5),
)
_NOPE_DIGESTS = {
    (
        1,
        8,
        2048,
        40_000,
        1,
    ): "c51adb88a83b0588c55cc441a301d87944c06afd7b2e829eec484759ddf6dd56",
    (
        4,
        16,
        2048,
        40_000,
        2,
    ): "585ad0a5130df91850fb2548741745ffe0edaf0f24cb0664bb6496a179658a64",
    (
        64,
        8,
        2048,
        40_000,
        3,
    ): "b79d01c392221d6e7705a002402d1c27d9c6507e49dc6d71bed9d7d142c6b339",
    (
        600,
        16,
        2048,
        40_000,
        4,
    ): "266b8fb4426e653014583e6ed4e6d7abe2c4bce47c49a6df64b444838fa2fce4",
    (
        7,
        8,
        100,
        1_000,
        5,
    ): "527f7092715cb851f120245c62197ce8448956ee1d2cf54eeb44e2d9ac52555d",
}


@unittest.skipUnless(_sm70(), "SM70 sparse MLA requires an NVIDIA V100")
class TestSparseMlaSm70NoPE(CustomTestCase):
    def test_frozen_digests(self):
        for key in _NOPE_CASES:
            case = _nope_case(*key)
            _leave_stale_nan()
            out = _op().sparse_mla_sm70(case["q"], case["kv"], case["idx"], SCALE)
            digest = hashlib.sha256(out.cpu().numpy().tobytes()).hexdigest()
            self.assertEqual(
                digest, _NOPE_DIGESTS[key], f"NoPE output bytes changed for {key}"
            )

    def test_matches_fp32_reference(self):
        for key in _NOPE_CASES:
            case = _nope_case(*key)
            case.update(q_rope=None, sink=None, lengths=None)
            out = _op().sparse_mla_sm70(case["q"], case["kv"], case["idx"], SCALE)[0]
            err = (out.float() - _reference(case)).abs().max().item()
            self.assertLessEqual(err, ATOL, f"NoPE {key}: max abs error {err:.3e}")

    def test_backend_dispatch(self):
        # The DSA backend keeps the NoPE kernel for a NoPE latent without sinks
        # (its lengths only bound -1 slots) and sends everything else to the
        # rope-tail kernel.
        from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend

        forward = DeepseekSparseAttnBackend._forward_sm70_sparse_mla
        case = _nope_case(4, 8, 2048, 40_000, 6)
        lengths = torch.full((4,), 2048, dtype=torch.int32, device="cuda")
        no_tail = case["q"][..., :0]
        nope = _op().sparse_mla_sm70(case["q"], case["kv"], case["idx"], SCALE)
        out = forward(
            None,
            case["q"],
            no_tail,
            case["kv"],
            case["idx"],
            SCALE,
            topk_length=lengths,
        )
        self.assertTrue(
            torch.equal(out, nope), "NoPE without sinks must use the NoPE kernel"
        )

        rope = _case(4, 8, 512, seed=7, lengths=[512, 3, 0, 100])
        out = forward(
            None,
            rope["q"],
            rope["q_rope"],
            rope["kv"],
            rope["idx"],
            SCALE,
            attn_sink=rope["sink"],
            topk_length=rope["lengths"],
        )
        self.assertTrue(
            torch.equal(out[0], _run(rope)), "rope tail with sinks and lengths"
        )
        out = forward(
            None,
            rope["q"],
            rope["q_rope"],
            rope["kv"],
            rope["idx"],
            SCALE,
            topk_length=rope["lengths"],
        )
        self.assertTrue(
            torch.equal(out[0], _run(rope, attn_sink=None)),
            "rope tail without sinks",
        )
        out = forward(
            None,
            rope["q"],
            rope["q_rope"],
            rope["kv"],
            rope["idx"],
            SCALE,
            attn_sink=rope["sink"],
            topk_length=rope["lengths"][:2],
        )
        self.assertTrue(
            torch.equal(out[0], _run(rope, topk_length=None)),
            "lengths for other rows are dropped",
        )


if __name__ == "__main__":
    unittest.main()
