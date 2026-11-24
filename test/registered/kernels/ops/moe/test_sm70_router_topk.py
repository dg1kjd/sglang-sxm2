"""The SM70 warp router must match _router_triton_kernel bitwise.

sm70_router_topk serves moe_fused_gate's ungrouped sigmoid routing on Volta
(GLM-5.3: top-8 of 288 experts, Hy4-flash: top-16 of 512). It ranks by value
alone unless two candidates are equal, and then by value and lowest expert id
like Triton's argmax; the renormalizing sum follows Triton's lane layout.
Every case compares ids, their order and the weight bits with the Triton
kernel on the same inputs.

Not registered for GPU CI (V100 worktree only).
"""

from __future__ import annotations

import unittest
from unittest import mock

import torch

from sglang.kernels.ops.moe import sm70_router_topk
from sglang.kernels.ops.moe.moe_fused_gate import moe_fused_gate
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(
    est_time=120,
    stage="base-b-kernel-unit",
    runner_config="1-gpu-large",
    disabled="SM70 V100 worktree only; do not register GPU CI",
)

# (topk, experts): the two models, plus expert counts that leave lanes padded.
SHAPES = ((8, 288), (16, 512), (8, 304), (16, 400))
# Hy4-flash's routed_scaling_factor; GLM-5.3 uses 2.5.
SCALE = 3.998


def _sm70() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (7, 0)


def _gate(scores, bias, topk, renormalize=True, scale=SCALE, apply_scale=False):
    return moe_fused_gate(
        scores,
        bias,
        topk,
        scoring_func="sigmoid",
        renormalize=renormalize,
        routed_scaling_factor=scale,
        apply_routed_scaling_factor_on_output=apply_scale,
    )


def _triton(*args, **kwargs):
    with mock.patch.object(sm70_router_topk, "sm70_router_covered", return_value=False):
        return _gate(*args, **kwargs)


@unittest.skipUnless(_sm70(), "SM70 required")
class TestSm70RouterTopk(CustomTestCase):
    def setUp(self):
        self.gen = torch.Generator(device="cuda").manual_seed(0)

    def _randn(self, *shape):
        return torch.randn(*shape, device="cuda", generator=self.gen)

    def _assert_same(
        self, scores, bias, topk, renormalize=True, scale=SCALE, apply_scale=False
    ):
        self.assertTrue(sm70_router_topk.sm70_router_covered(scores, bias, topk))
        ref_w, ref_i = _triton(scores, bias, topk, renormalize, scale, apply_scale)
        w, i = sm70_router_topk.sm70_router_topk(
            scores, bias, topk, renormalize, scale, apply_scale
        )
        bad = (i != ref_i).any(-1) | (
            w.view(torch.int32) != ref_w.view(torch.int32)
        ).any(-1)
        if bad.any():
            r = int(bad.nonzero()[0])
            self.fail(
                f"{int(bad.sum())} of {bad.numel()} rows differ, first row {r}:\n"
                f"  triton ids {ref_i[r].tolist()} weights {ref_w[r].tolist()}\n"
                f"  sm70   ids {i[r].tolist()} weights {w[r].tolist()}"
            )

    def test_random_scores(self):
        # Triton compiles M == 1 and M divisible by 16 separately; 6 is
        # Hy4-flash's speculative verify batch.
        for topk, n in SHAPES:
            for m in (1, 3, 6, 64, 4096):
                with self.subTest(topk=topk, n=n, m=m):
                    self._assert_same(self._randn(m, n) * 2, self._randn(n) * 0.1, topk)

    def test_offset_bias_near_ties(self):
        # GLM-5.3's correction bias: a large offset with a spread of a few
        # ulps, so biased values collide now and then.
        for topk, n in SHAPES:
            with self.subTest(topk=topk, n=n):
                steps = torch.randint(-4, 5, (n,), device="cuda", generator=self.gen)
                self._assert_same(
                    self._randn(4096, n), 8.0 + steps.float() * 2**-20, topk
                )

    def test_exact_ties_pick_lowest_id_first(self):
        for topk, n in SHAPES:
            with self.subTest(topk=topk, n=n):
                scores = torch.randint(
                    -3, 4, (4096, n), device="cuda", generator=self.gen
                ).float()
                scores[0] = 1.0
                self._assert_same(scores, torch.zeros(n, device="cuda"), topk)

    def test_ties_at_and_above_the_cut(self):
        # Distinct leaders with three equal values competing for the last
        # place, and two equal values among the leaders: both need the
        # id-ordered pick, not the value-only ranking.
        for topk, n in SHAPES:
            with self.subTest(topk=topk, n=n):
                scores = self._randn(2, n) * 0.01 - 5.0
                leaders = torch.randperm(n, device="cuda", generator=self.gen)[
                    : topk + 2
                ]
                scores[:, leaders[: topk - 1]] = 3.0 + 0.1 * torch.arange(
                    topk - 1, device="cuda"
                )
                scores[0, leaders[topk - 1 :]] = 1.0
                scores[1, leaders[topk - 1 :]] = -6.0
                scores[1, leaders[1]] = scores[1, leaders[topk - 2]]
                self._assert_same(scores, torch.zeros(n, device="cuda"), topk)

    def test_nan_scores(self):
        for topk, n in SHAPES:
            with self.subTest(topk=topk, n=n):
                scores = self._randn(64, n)
                scores[::3, ::7] = float("nan")
                scores[1] = float("nan")
                scores[2, 5:] = float("nan")  # fewer finite values than topk
                self._assert_same(scores, self._randn(n), topk)
                # A NaN becomes -1e30: above a -inf biased value, above a
                # bias of -3e38, equal to a bias of -1e30.
                scores = self._randn(64, n)
                scores[:, ::2] = float("nan")
                bias = self._randn(n)
                bias[1::2] = -float("inf")
                bias[1:6:2] = self._randn(3)
                self._assert_same(scores, bias, topk)
                scores = self._randn(64, n)
                scores[:, 100:110] = float("nan")
                bias = torch.full((n,), -3e38, device="cuda")
                bias[:4] = 0.0
                self._assert_same(scores, bias, topk)
                scores = self._randn(64, n)
                scores[:, 50:60] = float("nan")
                self._assert_same(scores, torch.full((n,), -1e30, device="cuda"), topk)

    def test_infinite_and_extreme_values(self):
        # |scores| up to 150 reach ex2's subnormal range and div.full's
        # scaled range. A +inf bias ties at the top; a -inf bias on every
        # expert ties them all.
        for topk, n in SHAPES:
            with self.subTest(topk=topk, n=n):
                scores = (
                    torch.rand(256, n, device="cuda", generator=self.gen) - 0.5
                ) * 300
                scores[0, :7] = float("inf")
                scores[1, :7] = -float("inf")
                bias = self._randn(n)
                self._assert_same(scores, bias, topk)
                bias[[3, 40, n - 1]] = float("inf")
                self._assert_same(scores, bias, topk)
                self._assert_same(
                    scores, torch.full((n,), -float("inf"), device="cuda"), topk
                )

    def test_subnormal_sigmoids(self):
        # Logits in [-104, -86] give subnormal sigmoids, distinct in Triton
        # and all +0 in a flush-to-zero build, where the lowest ids would win.
        for topk, n in SHAPES:
            with self.subTest(topk=topk, n=n):
                scores = -86.0 - 18.0 * torch.rand(
                    2048, n, device="cuda", generator=self.gen
                )
                self._assert_same(scores, torch.zeros(n, device="cuda"), topk)
                self._assert_same(scores, torch.full((n,), -0.0, device="cuda"), topk)
                scores = -86.0 - 30.0 * torch.rand(
                    2048, n, device="cuda", generator=self.gen
                )
                scores[:, :5] = self._randn(2048, 5)
                self._assert_same(scores, torch.zeros(n, device="cuda"), topk)

    def test_renormalize_and_scale_flags(self):
        for topk, n in SHAPES:
            for renormalize, apply_scale in (
                (True, False),
                (True, True),
                (False, True),
                (False, False),
            ):
                for scale in (SCALE, 2.5):
                    with self.subTest(
                        topk=topk,
                        n=n,
                        renormalize=renormalize,
                        apply_scale=apply_scale,
                        scale=scale,
                    ):
                        self._assert_same(
                            self._randn(256, n) * 4,
                            self._randn(n),
                            topk,
                            renormalize,
                            scale,
                            apply_scale,
                        )

    def test_strided_rows(self):
        for topk, n in SHAPES:
            with self.subTest(topk=topk, n=n):
                wide = self._randn(64, 576)
                self._assert_same(wide[:, 32 : 32 + n], self._randn(n), topk)
                # Triton's sum order follows the row alignment, not the
                # expert count.
                self._assert_same(wide[:, : n - 12], self._randn(n - 12), topk)

    def test_moe_fused_gate_dispatches_both_models(self):
        for topk, n, scale in ((8, 288, 2.5), (16, 512, SCALE)):
            for m in (1, 6, 4096):
                with self.subTest(topk=topk, n=n, m=m):
                    scores, bias = self._randn(m, n), self._randn(n) * 0.1
                    with mock.patch.object(
                        sm70_router_topk,
                        "sm70_router_topk",
                        wraps=sm70_router_topk.sm70_router_topk,
                    ) as spy:
                        w, i = _gate(scores, bias, topk, scale=scale)
                    spy.assert_called_once()
                    ref_w, ref_i = _triton(scores, bias, topk, scale=scale)
                    self.assertTrue(torch.equal(i, ref_i))
                    self.assertTrue(
                        torch.equal(w.view(torch.int32), ref_w.view(torch.int32))
                    )

    def test_cuda_graph_replay(self):
        # Decode and speculative verify replay the router from a captured
        # graph; the tied rows take the fallback inside the graph.
        for topk, n in ((8, 288), (16, 512)):
            with self.subTest(topk=topk, n=n):
                scores = self._randn(6, n)
                bias = torch.zeros(n, device="cuda")
                _gate(scores, bias, topk)  # build the module outside the capture
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    w, i = _gate(scores, bias, topk)
                for fill in (
                    self._randn(6, n),
                    torch.randint(-3, 4, (6, n), device="cuda", generator=self.gen),
                ):
                    scores.copy_(fill)
                    graph.replay()
                    ref_w, ref_i = _triton(scores, bias, topk)
                    self.assertTrue(torch.equal(i, ref_i))
                    self.assertTrue(
                        torch.equal(w.view(torch.int32), ref_w.view(torch.int32))
                    )

    def test_layouts_that_change_triton_sum_order_stay_on_triton(self):
        # Triton sums the picked weights in another order when a row is not
        # 16-byte aligned (base or row stride); the warp router only
        # reproduces the aligned order. Called directly on such rows it picks
        # the same ids with other weight bits: if that stops, a Triton change
        # made the gate stale.
        for topk, n in ((8, 288), (16, 512)):
            flat = self._randn(4096 * n + 1)
            wide = self._randn(4096, n + 8)
            cases = {
                "misaligned base": (flat[1:].view(4096, n), self._randn(n)),
                "row stride not a multiple of 16": (wide[:, :n], self._randn(n)),
            }
            for name, (scores, bias) in cases.items():
                with self.subTest(topk=topk, n=n, layout=name):
                    self.assertFalse(
                        sm70_router_topk.sm70_router_covered(scores, bias, topk)
                    )
                    ref_w, ref_i = _triton(scores, bias, topk)
                    w, i = sm70_router_topk.sm70_router_topk(
                        scores, bias, topk, True, SCALE, False
                    )
                    self.assertTrue(torch.equal(i, ref_i))
                    self.assertFalse(
                        torch.equal(w.view(torch.int32), ref_w.view(torch.int32))
                    )


if __name__ == "__main__":
    unittest.main()
