"""Batch-invariant fp16 / block-FP8 GEMM on SM70 (sm70_fixed_order_gemm).

For every Hunyuan-4-Flash linear at TP8 (target, DFly drafter, lm_head, MLA
absorption bmm, block-FP8 W8A16):

- every row of y bit for bit equal to the same row run alone (M = 1), through
  the tiling the dispatcher picks and through each tiling forced, at M = 1..8
  for fp16 weights (callers take the kernel up to MAX_ROWS = 8 rows) and at M in
  {1..8, 16, 33, 100, 101, 512, 2048, 4096} for block-FP8 (taken at every M);
- cuBLAS fails that check at M <= 8 (it picks kernels by M) on every fp16
  shape and on the router's fp32 output (its fp32-output lm_head happens to
  keep its rows at M <= 8);
- above MAX_ROWS rows, supported() is False and a caller gives the bits of its
  cuBLAS path;
- the documented chain fold order bit for bit, on weights with one nonzero per
  chain (every chain sum one exact product), decode and prefill tiling;
- strided x and out (column slices, the MLA transposed views, an fp16 out
  starting two elements in with a row stride of 2 mod 4), writes only inside
  out, bias, weight row slices (columns independent of N), CUDA graph replay;
- K above the decode tiling's shared memory: the prefill tiling runs instead,
  rows still equal to single-row runs;
- error against an fp64 reference next to cuBLAS's;
- the block-FP8 W8A16 layer path (prepare keeps the checkpoint layout, apply
  runs the kernel);
- Hunyuan-4's hyper-connection head (fp32 gates, then the stream sum): rows
  equal to single-row runs at M <= 8 with the env on, not with cuBLAS.

Not registered for GPU CI (V100 worktree only).
"""

from __future__ import annotations

import unittest
from typing import NamedTuple, Optional

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(
    est_time=300,
    stage="base-b-kernel-unit",
    runner_config="1-gpu-large",
    disabled="SM70 V100 worktree only; do not register GPU CI",
)

MS = (1, 2, 3, 4, 5, 6, 7, 8, 16, 33, 100, 101, 512, 2048, 4096)
FORCES = (None, "decode", "prefill")


class Shape(NamedTuple):
    name: str
    n: int
    k: int
    kind: str  # "f16", "f16o32", "fp8" or "bmm"
    batch: int = 1
    bias: bool = False


# Hunyuan-4-Flash at TP8, per rank. DSA layers have 4 local heads, sliding
# window layers 8.
SHAPES = (
    Shape("fused_qkv_a", 1600, 3072, "f16"),
    Shape("kv_b_dsa", 1792, 512, "f16"),
    Shape("kv_b_swa", 3584, 512, "f16"),
    Shape("routed_latent_down", 1536, 3072, "f16"),
    Shape("routed_latent_up", 3072, 1536, "f16"),
    Shape("shared_gate_up", 384, 3072, "f16"),
    Shape("shared_down", 3072, 192, "f16"),
    Shape("router", 512, 3072, "f16o32"),
    Shape("dense_gate_up", 3456, 3072, "f16"),
    Shape("dense_down", 3072, 1728, "f16"),
    Shape("indexer_query", 2048, 1024, "f16"),
    Shape("indexer_key_head", 144, 3072, "f16"),
    Shape("engram_kv", 15360, 192, "f16", bias=True),
    Shape("lm_head", 15104, 3072, "f16o32"),
    Shape("dfly_linear_gate", 1024, 3072, "f16"),
    Shape("dfly_q_b", 1024, 1024, "f16"),
    Shape("dfly_o_proj", 3072, 1024, "f16"),
    Shape("dfly_gate_up", 2304, 3072, "f16"),
    Shape("dfly_down", 3072, 1152, "f16"),
    Shape("dfly_eh_proj", 3072, 6144, "f16"),
    Shape("dfly_correction_gate_up", 768, 6144, "f16"),
    Shape("dfly_correction_down", 3072, 384, "f16"),
    Shape("w_kc_dsa", 512, 192, "bmm", batch=4),
    Shape("w_kc_swa", 512, 192, "bmm", batch=8),
    Shape("w_vc_dsa", 256, 512, "bmm", batch=4),
    Shape("w_vc_swa", 256, 512, "bmm", batch=8),
    Shape("q_b_dsa", 1024, 1024, "fp8"),
    Shape("q_b_swa", 2048, 1024, "fp8"),
    Shape("linear_gate_dsa", 1024, 3072, "fp8"),
    Shape("linear_gate_swa", 2048, 3072, "fp8"),
    Shape("o_proj_dsa", 3072, 1024, "fp8"),
    Shape("o_proj_swa", 3072, 2048, "fp8"),
    # Not a Hunyuan shape: an odd number of fp16 chains (K % 64 == 32), the
    # prefill tiling's unpaired last chain.
    Shape("odd_chains", 1024, 3104, "f16"),
)


def _sm70() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (7, 0)


def _ops():
    from sglang.kernels.ops.gemm import sm70_fixed_order_gemm

    return sm70_fixed_order_gemm


class Operands(NamedTuple):
    shape: Shape
    x: torch.Tensor  # [rows, K] or [B, rows, K]
    w: torch.Tensor
    scale: Optional[torch.Tensor]
    bias: Optional[torch.Tensor]


def _operands(shape: Shape, rows: int, seed: int = 0) -> Operands:
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    n, k = shape.n, shape.k
    lead = (shape.batch,) if shape.kind == "bmm" else ()
    x = torch.randn(*lead, rows, k, device="cuda", generator=gen).half()
    if rows > 1:
        x[..., 1, :] = 0
    scale = bias = None
    if shape.kind == "fp8":
        w = (torch.randn(n, k, device="cuda", generator=gen) * 0.5).to(
            torch.float8_e4m3fn
        )
        scale = torch.rand(-(-n // 128), k // 128, device="cuda", generator=gen)
        scale = scale * 1e-2 + 1e-3
    else:
        w = (torch.randn(*lead, n, k, device="cuda", generator=gen) * 0.02).half()
    if shape.bias:
        bias = torch.randn(n, device="cuda", generator=gen).half()
    return Operands(shape, x, w, scale, bias)


def fixed_order(op: Operands, x: torch.Tensor, force: Optional[str] = None):
    ops = _ops()
    kind = op.shape.kind
    if kind == "fp8":
        return ops.fp8_linear(x, op.w, op.scale, op.bias, force=force)
    if kind == "bmm":
        return ops.bmm(x, op.w, force=force)
    out_dtype = torch.float32 if kind == "f16o32" else torch.float16
    return ops.linear(x, op.w, op.bias, out_dtype=out_dtype, force=force)


def cublas(op: Operands, x: torch.Tensor, force: Optional[str] = None):
    """The paths the kernel replaces (fp8 here as a dequantized fp16 weight)."""
    kind = op.shape.kind
    if kind == "bmm":
        return torch.bmm(x, op.w.transpose(1, 2))
    if kind == "f16o32":
        return torch.mm(x, op.w.T, out_dtype=torch.float32)
    w = op.w
    if kind == "fp8":
        w = _dequant(op).half()
    return torch.nn.functional.linear(x, w, op.bias)


def _dequant(op: Operands) -> torch.Tensor:
    s = op.scale.repeat_interleave(128, 0)[: op.shape.n]
    return op.w.double() * s.repeat_interleave(128, 1)[:, : op.shape.k].double()


def _bits(y: torch.Tensor) -> torch.Tensor:
    return y.view(torch.int16 if y.dtype == torch.float16 else torch.int32)


def _rows(x: torch.Tensor, m: int) -> torch.Tensor:
    return x[..., :m, :]


def row_mismatches(fn, op: Operands, ms=MS, forces=(None,)) -> list:
    """(M, force, mismatching rows) wherever a row of fn at M differs from the
    same row run alone."""
    rows = max(ms)
    ref = torch.cat(
        [fn(op, _rows(op.x[..., i:, :], 1)) for i in range(rows)], dim=-2
    )
    bad = []
    for m in ms:
        for force in forces:
            y = fn(op, _rows(op.x, m), force)
            diff = _bits(y) != _bits(ref[..., :m, :])
            if diff.ndim == 3:
                diff = diff.any(0)
            n_bad = int(diff.any(-1).sum())
            if n_bad:
                bad.append((m, force, n_bad))
    return bad


@unittest.skipUnless(_sm70(), "SM70 (V100) only")
class TestSm70FixedOrderGemm(CustomTestCase):
    def test_rows_equal_single_row_runs(self):
        # The fp16 callers take the kernel up to MAX_ROWS rows, the block-FP8
        # layer path at every M.
        small = tuple(m for m in MS if m <= _ops().MAX_ROWS)
        for shape in SHAPES:
            with self.subTest(shape=shape.name):
                ms = MS if shape.kind == "fp8" else small
                op = _operands(shape, max(ms))
                self.assertEqual(row_mismatches(fixed_order, op, ms=ms, forces=FORCES), [])

    def test_check_detects_cublas(self):
        # The check above, on the cuBLAS paths it replaces: cuBLAS picks its
        # kernel (and split-K) by M, so rows change with the batch. Its
        # fp32-output lm_head keeps its rows up to 8 rows, the router does not.
        failing = []
        for shape in SHAPES:
            if shape.kind in ("fp8", "bmm"):
                continue
            op = _operands(shape, 8)
            if row_mismatches(cublas, op, ms=(1, 2, 6, 8)):
                failing.append(shape.name)
        self.assertIn("fused_qkv_a", failing)
        self.assertIn("router", failing)

    def test_fold_order_bit_exact(self):
        # One nonzero per chain and row: every chain sum is one exact product,
        # so the output is the documented fold of those products, in fp32
        # round-to-nearest, chain order, then the bias, then the output cast.
        cases = (
            Shape("f16", 64, 3072, "f16", bias=True),
            Shape("f16o32", 64, 3072, "f16o32", bias=True),
            Shape("fp8", 256, 3072, "fp8"),
        )
        for shape in cases:
            chain = 64 if shape.kind == "fp8" else 32
            chains = shape.k // chain
            gen = torch.Generator(device="cuda")
            gen.manual_seed(1)
            op = _operands(shape, 300, seed=2)
            pos = torch.randint(0, chain, (shape.n, chains), device="cuda", generator=gen)
            cols = pos + torch.arange(chains, device="cuda") * chain
            w = torch.zeros(shape.n, shape.k, device="cuda")
            vals = torch.randn(shape.n, chains, device="cuda", generator=gen)
            if shape.kind == "fp8":
                w.scatter_(1, cols, vals.to(torch.float8_e4m3fn).float())
                w = w.to(torch.float8_e4m3fn)
            else:
                w.scatter_(1, cols, vals)
                w = w.half()
            op = op._replace(w=w)
            # Exact in fp32: fp16 x fp16 or fp16 x e4m3.
            prods = (op.x.double()[:, cols] * w.double().gather(1, cols)[None]).float()
            total = torch.zeros(op.x.shape[0], shape.n, device="cuda")
            for c in range(chains):
                if shape.kind == "fp8":
                    s = op.scale[torch.arange(shape.n, device="cuda") // 128, c // 2]
                    # fma(chain, scale * 256, total) with chain = sum * 2^-8,
                    # rounded once (the fp64 product is exact).
                    t = prods[..., c].double() / 256.0 * (s * 256.0).double()
                    total = (t + total.double()).float()
                else:
                    total = total + prods[..., c]
            if op.bias is not None:
                total = total + op.bias.float()
            out_dtype = torch.float32 if shape.kind == "f16o32" else torch.float16
            want = total.to(out_dtype)
            for force in ("decode", "prefill"):
                with self.subTest(shape=shape.name, force=force):
                    got = fixed_order(op, op.x, force)
                    self.assertTrue(torch.equal(_bits(got), _bits(want)))

    def test_strided_operands(self):
        ops = _ops()
        gen = torch.Generator(device="cuda")
        gen.manual_seed(3)
        w = (torch.randn(1600, 3072, device="cuda", generator=gen) * 0.02).half()
        wide = torch.randn(64, 3072 + 256, device="cuda", generator=gen).half()
        x = wide[:, 128 : 128 + 3072]
        ref = ops.linear(x.contiguous(), w)
        # Row strides 1616 and 1602 (2 mod 4), starts 16 and 4 bytes in.
        for width, start in ((1600 + 16, 8), (1600 + 2, 2)):
            for force in ("decode", "prefill"):
                with self.subTest(width=width, force=force):
                    out_wide = torch.full(
                        (64, width), float("nan"), device="cuda", dtype=torch.float16
                    )
                    out = out_wide[:, start : start + 1600]
                    ops.linear(x, w, out=out, force=force)
                    self.assertTrue(torch.equal(_bits(out), _bits(ref)))
                    self.assertTrue(out_wide[:, :start].isnan().all())
                    self.assertTrue(out_wide[:, start + 1600 :].isnan().all())
        # The MLA absorption views: q_nope [M, H, 192] transposed to [H, M, 192],
        # into a [M, H * 256] buffer viewed as [H, M, 256].
        heads, m = 8, 37
        q = torch.randn(m, heads, 192, device="cuda", generator=gen).half()
        w_kc = (torch.randn(heads, 512, 192, device="cuda", generator=gen) * 0.02).half()
        ref = ops.bmm(q.transpose(0, 1).contiguous(), w_kc)
        for force in ("decode", "prefill"):
            self.assertTrue(
                torch.equal(_bits(ops.bmm(q.transpose(0, 1), w_kc, force=force)), _bits(ref))
            )
        a = torch.randn(m, heads, 512, device="cuda", generator=gen).half()
        w_vc = (torch.randn(heads, 256, 512, device="cuda", generator=gen) * 0.02).half()
        ref = ops.bmm(a.transpose(0, 1).contiguous(), w_vc)
        for force in ("decode", "prefill"):
            flat = torch.empty(m, heads * 256, device="cuda", dtype=torch.float16)
            out = flat.view(m, heads, 256).transpose(0, 1)
            ops.bmm(a.transpose(0, 1), w_vc, out=out, force=force)
            self.assertTrue(torch.equal(_bits(out), _bits(ref)))

    def test_k_above_decode_smem(self):
        # 385 fp16 chains (386 FP8) need more than 96 KiB at 8 decode rows but
        # fit at 1; the dispatcher then takes the prefill tiling, same order.
        ops = _ops()
        for shape, ms in (
            (Shape("large_k", 64, 12320, "f16"), (1, 2, 8)),
            (Shape("large_k_fp8", 256, 24704, "fp8"), (1, 2, 8, 16)),
        ):
            with self.subTest(shape=shape.name):
                fp8 = shape.kind == "fp8"
                self.assertTrue(ops.decode_fits(1, shape.k, fp8))
                self.assertFalse(ops.decode_fits(8, shape.k, fp8))
                op = _operands(shape, max(ms))
                self.assertEqual(row_mismatches(fixed_order, op, ms=ms), [])

    def test_hy4_hc_head_rows(self):
        from types import SimpleNamespace

        from sglang.srt.environ import envs
        from sglang.srt.models.hunyuan_v4 import HYV4HCHeadLayer

        cfg = SimpleNamespace(hc_mult=4, hidden_size=3072, rms_norm_eps=1e-6, hc_eps=1e-6)
        gen = torch.Generator(device="cuda")
        gen.manual_seed(5)
        head = HYV4HCHeadLayer(cfg, "hc_head").cuda()
        head.hc_head_fn.weight.data = (
            torch.randn(4, 4 * 3072, device="cuda", generator=gen) * 1e-3
        )
        head.hc_head_scale.data = torch.rand(1, device="cuda", generator=gen)
        head.hc_head_base.data = torch.randn(4, device="cuda", generator=gen)
        x = (torch.randn(8, 4, 3072, device="cuda", generator=gen) * 3).half()

        def bad_rows(on):
            with envs.SGLANG_USE_SM70_FIXED_ORDER_GEMM.override(on):
                single = torch.cat([head(x[i : i + 1]) for i in range(8)])
                return {
                    m: int((_bits(head(x[:m])) != _bits(single[:m])).any(-1).sum())
                    for m in (2, 6, 8)
                }

        self.assertEqual(bad_rows(True), {2: 0, 6: 0, 8: 0})
        self.assertNotEqual(bad_rows(False), {2: 0, 6: 0, 8: 0})

    def test_columns_do_not_depend_on_n(self):
        # The drafter's greedy head reads the lm_head's unpadded rows only.
        op = _operands(Shape("lm_head", 15104, 3072, "f16o32"), 6)
        full = fixed_order(op, op.x)
        part = _ops().linear(op.x, op.w[:15000], out_dtype=torch.float32)
        self.assertTrue(torch.equal(_bits(part), _bits(full[:, :15000])))
        op = _operands(Shape("q_b", 2048, 1024, "fp8"), 6)
        full = fixed_order(op, op.x)
        part = _ops().fp8_linear(op.x, op.w[:1024], op.scale[:8])
        self.assertTrue(torch.equal(_bits(part), _bits(full[:, :1024])))

    def test_cuda_graph_replay(self):
        names = ("fused_qkv_a", "router", "engram_kv", "lm_head", "w_kc_swa", "o_proj_swa")
        for shape in (s for s in SHAPES if s.name in names):
            for m in (1, 6, 300):
                with self.subTest(shape=shape.name, m=m):
                    op = _operands(shape, m, seed=4)
                    static_x = op.x.clone()
                    fixed_order(op, static_x)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        static_y = fixed_order(op, static_x)
                    for seed in (5, 6):
                        static_x.copy_(_operands(shape, m, seed=seed).x)
                        graph.replay()
                        torch.cuda.synchronize()
                        self.assertTrue(
                            torch.equal(_bits(static_y), _bits(fixed_order(op, static_x)))
                        )

    def test_error_next_to_cublas(self):
        for shape in SHAPES:
            with self.subTest(shape=shape.name):
                op = _operands(shape, 64, seed=7)
                w = _dequant(op) if shape.kind == "fp8" else op.w.double()
                ref = op.x.double() @ w.transpose(-1, -2)
                if op.bias is not None:
                    ref = ref + op.bias.double()
                ours = (fixed_order(op, op.x).double() - ref).abs()
                theirs = (cublas(op, op.x).double() - ref).abs()
                # Both round the same fp32 sum to the output type; the fp32 sum
                # order is the only difference.
                self.assertLessEqual(ours.max().item(), 2 * theirs.max().item() + 1e-6)
                self.assertLessEqual(ours.mean().item(), 1.5 * theirs.mean().item() + 1e-7)

    def test_large_m_keeps_cublas(self):
        from sglang.srt.environ import envs
        from sglang.srt.models.hunyuan_v4_flash import _linear

        ops = _ops()
        t = ops.MAX_ROWS
        op = _operands(Shape("fused_qkv_a", 1600, 3072, "f16"), t + 1, seed=9)
        self.assertTrue(ops.supported(op.x[:t], op.w))
        self.assertFalse(ops.supported(op.x, op.w))
        bmm_op = _operands(Shape("w_kc_swa", 512, 192, "bmm", batch=8), t + 1, seed=9)
        self.assertTrue(ops.supported(bmm_op.x[:, :t], bmm_op.w))
        self.assertFalse(ops.supported(bmm_op.x, bmm_op.w))
        with envs.SGLANG_USE_SM70_FIXED_ORDER_GEMM.override(True):
            small = _linear(op.x[:t], op.w)
            large = _linear(op.x, op.w)
        self.assertTrue(torch.equal(_bits(small), _bits(ops.linear(op.x[:t], op.w))))
        self.assertTrue(
            torch.equal(_bits(large), _bits(torch.nn.functional.linear(op.x, op.w)))
        )
        self.assertFalse(torch.equal(_bits(large), _bits(ops.linear(op.x, op.w))))

    def test_eligibility(self):
        ops = _ops()
        x = torch.randn(4, 3072, device="cuda").half()
        w = torch.randn(1600, 3072, device="cuda").half()
        self.assertTrue(ops.supported(x, w))
        self.assertTrue(ops.supported(x, w, torch.zeros(1600, device="cuda").half()))
        self.assertFalse(ops.supported(x, w, torch.zeros(1600, device="cuda")))
        self.assertFalse(ops.supported(x.float(), w))
        self.assertFalse(ops.supported(x[:, 8:], w[:, 8:]))  # 16-byte rows only
        self.assertFalse(ops.supported(x, torch.randn(1604, 3072, device="cuda").half()))
        self.assertFalse(ops.supported(x[:, :3056], w[:, :3056]))
        f8 = w.to(torch.float8_e4m3fn)
        self.assertTrue(ops.supported(x, f8))
        self.assertFalse(ops.supported(x[:, :3008], f8[:, :3008]))  # K % 128
        self.assertEqual(ops.linear(x[:0], w).shape, (0, 1600))
        odd = torch.empty(4, 1601, device="cuda").half()[:, :1600]
        with self.assertRaises(ValueError):
            ops.linear(x, w, out=odd)

    def test_fp8_layer_wiring(self):
        # The block-FP8 W8A16 layer with the env on: prepare keeps the
        # checkpoint layout (no TurboMind repack), apply runs the kernel for
        # 3D activations, strided rows, an fp16 and an fp32 bias, also above
        # MAX_ROWS rows (12 here).
        from sglang.srt.environ import envs
        from sglang.srt.layers.quantization.sm70_turbomind_fp8 import (
            apply_sm70_turbomind_fp8_linear,
            prepare_sm70_turbomind_fp8_linear,
        )

        ops = _ops()
        op = _operands(Shape("o_proj_swa", 3072, 2048, "fp8"), 12, seed=8)
        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(op.w.clone(), requires_grad=False)
        layer.weight_scale_inv = torch.nn.Parameter(
            op.scale.t().contiguous().t(), requires_grad=False
        )
        layer.orig_dtype = torch.float16
        layer.weight_block_size = [128, 128]
        layer.output_size_per_partition = op.shape.n
        layer.prefix = "model.layers.0.self_attn.o_proj"
        with envs.SGLANG_USE_SM70_FIXED_ORDER_GEMM.override(True):
            prepare_sm70_turbomind_fp8_linear(layer)
        self.assertTrue(layer.sm70_fp8_fixed_order)
        self.assertFalse(layer.sm70_fp8_turbomind)
        self.assertTrue(torch.equal(layer.weight.view(torch.uint8), op.w.view(torch.uint8)))
        self.assertTrue(layer.weight_scale_inv.is_contiguous())
        self.assertTrue(torch.equal(layer.weight_scale_inv, op.scale))

        want = ops.fp8_linear(op.x, op.w, op.scale)
        got = apply_sm70_turbomind_fp8_linear(layer, op.x.view(3, 4, -1), None)
        self.assertEqual(got.shape, (3, 4, op.shape.n))
        self.assertTrue(torch.equal(_bits(got.reshape(12, -1)), _bits(want)))
        for pad in (8, 4):  # 16-byte rows as a view; 8-byte rows are copied first
            wide = torch.zeros(12, op.shape.k + pad, device="cuda").half()
            wide[:, pad:] = op.x
            got = apply_sm70_turbomind_fp8_linear(layer, wide[:, pad:], None)
            self.assertTrue(torch.equal(_bits(got), _bits(want)))
        bias = torch.randn(op.shape.n, device="cuda").half()
        got = apply_sm70_turbomind_fp8_linear(layer, op.x, bias)
        self.assertTrue(torch.equal(_bits(got), _bits(ops.fp8_linear(op.x, op.w, op.scale, bias))))
        got = apply_sm70_turbomind_fp8_linear(layer, op.x, bias.float())
        self.assertTrue(torch.equal(_bits(got), _bits(want.clone().add_(bias.float()))))


if __name__ == "__main__":
    unittest.main()
