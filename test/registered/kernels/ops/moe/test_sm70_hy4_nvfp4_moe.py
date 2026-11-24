"""Hunyuan-4-Flash routed-expert NVFP4 MoE on SM70 (sm70_hy4_nvfp4_moe).

Against ``fused_marlin_moe`` (the path it replaces) and an fp64 reference built
from the checkpoint-format codes and scales:

- decode (1..8 tokens): each fp16 rounding point on the kernel's own
  intermediates (the workspace): the gate_up sums, each route's down row, and
  the route sum, which must be bit for bit the fp32 sum Marlin's
  ``moe_sum_reduce_triton`` computes;
- decode and prefill outputs against Marlin and the fp64 reference, within the
  bound those rounding points allow;
- batch invariance: every token's output bit for bit equal to the same token
  run alone, at M = 2..8 (each decode config) and at prefill sizes up to two
  chunks (each prefill config), with experts shared across tokens, repeated
  tokens and ids outside [0, E);
- the documented sum order bit for bit, on weights with one nonzero per output
  row (every dot product one exact product), decode and prefill; the gate_up
  K-split / K-warp order on weights with one nonzero per (split, warp) cell
  and exact cancellations between cells;
- a first decode call inside CUDA graph capture raises;
- the clamp triggered, routed weights, ids outside [0, E), CUDA graph replay.

Not registered for GPU CI (V100 worktree only).
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(
    est_time=300,
    stage="base-b-kernel-unit",
    runner_config="1-gpu-large",
    disabled="SM70 V100 worktree only; do not register GPU CI",
)

E = 512
H = 1536
I = 192
TOPK = 16
LIMIT = 10.0
ROUTED = 3.998
LEVELS = [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]
# fp32 accumulation error bound, in units of 2^-23 times the sum of |terms|:
# no term passes through more than ~40 roundings (24 HMMA k-steps per warp,
# then the K-warp and K-split sums).
F32_DEPTH = 64


def _sm70() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (7, 0)


def _ops():
    from sglang.kernels.ops.moe import sm70_hy4_nvfp4_moe

    return sm70_hy4_nvfp4_moe


def _leave_stale_nan() -> None:
    # Volta's L1 shares SRAM with shared memory, so this also seeds shared memory.
    junk = torch.full((64 << 20,), float("nan"), device="cuda")
    junk.sum()
    del junk


def _pack(raw: torch.Tensor, scales: torch.Tensor, gscale: torch.Tensor):
    """Checkpoint-format NVFP4 (``raw`` uint8 [e, n, k/2], low nibble first;
    e4m3 ``scales`` [e, n, k/16]; fp32 ``gscale`` [e]) packed like the SM70
    loader does: Marlin pack, encoded scales, processed global scale."""
    import sglang.srt.layers.quantization  # noqa: F401  (import order)
    from sglang.srt.hardware_backend.gpu.quantization.gptq_kernels import (
        gptq_marlin_moe_repack,
    )
    from sglang.srt.layers.quantization.marlin_utils import (
        sm70_nvfp4_marlin_process_global_scale,
        sm70_nvfp4_marlin_process_scales,
    )

    e, n, k = raw.shape[0], raw.shape[1], raw.shape[2] * 2
    packed = gptq_marlin_moe_repack(
        raw.view(torch.int32).transpose(1, 2).contiguous(),
        torch.empty((e, 0), dtype=torch.int32, device=raw.device),
        k,
        n,
        4,
    )
    enc, factor = sm70_nvfp4_marlin_process_scales(
        scales.transpose(1, 2).contiguous(), torch.float16
    )
    g = sm70_nvfp4_marlin_process_global_scale(gscale, torch.float16) / factor
    return packed, enc, g


def _make_weights(
    n: int, k: int, gen: torch.Generator, sparse: bool = False
) -> SimpleNamespace:
    """Random checkpoint-format NVFP4 weights, packed like the SM70 loader does.

    ``sparse``: one nonzero code per output row, at a random k, the rest +0.
    Each dot product is then one exact product, so every fp32 value in the
    kernels is known exactly; ``kpos`` and ``val`` (fp64, before the processed
    global scale, as ``_weight``) give that nonzero.
    """
    if sparse:
        kpos = torch.randint(0, k, (E, n), device="cuda", generator=gen)
        code = torch.randint(1, 15, (E, n), device="cuda", generator=gen)
        code = code + (code >= 8).long()  # no -0
        raw = torch.zeros((E, n, k // 2), dtype=torch.uint8, device="cuda")
        nibble = (code << (4 * (kpos % 2))).to(torch.uint8)
        raw.scatter_(2, (kpos // 2).unsqueeze(-1), nibble.unsqueeze(-1))
    else:
        raw = torch.randint(
            0, 256, (E, n, k // 2), dtype=torch.uint8, device="cuda", generator=gen
        )
    scales = (torch.rand(E, n, k // 16, device="cuda", generator=gen) + 0.5).to(
        torch.float8_e4m3fn
    )
    gscale = torch.rand(E, device="cuda", generator=gen) * 0.01 + 0.005
    packed, enc, g = _pack(raw, scales, gscale)
    w = SimpleNamespace(packed=packed, scales=enc, gscale=g, raw=raw, raw_scales=scales)
    if sparse:
        levels = torch.tensor(LEVELS, dtype=torch.float64, device=raw.device)
        s = scales.double().gather(2, (kpos // 16).unsqueeze(-1)).squeeze(-1)
        w.kpos = kpos
        w.val = levels[code] * s * 2.0**-7
    return w


def _weight(w: SimpleNamespace, e: int) -> torch.Tensor:
    """fp64 [n, k] weight of expert e before the processed global scale.

    fp4 value * scale * 2^-7: the SM70 dequantizer yields value * 2^-14 and
    the scale byte decodes to scale * 2^7; the processed global scale carries
    the 2^7 back.
    """
    raw = w.raw[e]
    n = raw.shape[0]
    codes = torch.stack((raw & 15, raw >> 4), dim=-1).reshape(n, -1).long()
    levels = torch.tensor(LEVELS, dtype=torch.float64, device=raw.device)
    s = w.raw_scales[e].double().repeat_interleave(16, dim=1)
    return levels[codes] * s * 2.0**-7


def _ulp16(a: torch.Tensor) -> torch.Tensor:
    a = a.abs().double()
    exp = torch.floor(torch.log2(a.clamp_min(2.0**-14)))
    return torch.exp2(exp - 10)


def _swiglu(gate_up: torch.Tensor) -> torch.Tensor:
    from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import (
        swiglu_limit_func,
    )

    out = torch.empty((gate_up.shape[0], I), dtype=torch.float16, device=gate_up.device)
    swiglu_limit_func(out, gate_up, LIMIT)
    return out


def _route_sum(rows: torch.Tensor) -> torch.Tensor:
    """moe_sum_reduce_triton for one token: fp32 sum in route order, then scale."""
    acc = torch.zeros(rows.shape[-1], dtype=torch.float32, device=rows.device)
    for k in range(rows.shape[0]):
        acc = acc + rows[k].float()
    return (acc * ROUTED).half()


def _marlin(x, w13, w2, ids, wts) -> torch.Tensor:
    from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import (
        fused_marlin_moe,
    )
    from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace

    return fused_marlin_moe(
        hidden_states=x,
        w1=w13.packed,
        w2=w2.packed,
        w1_scale=w13.scales,
        w2_scale=w2.scales,
        gating_output=torch.empty(
            (x.shape[0], E), dtype=torch.float32, device=x.device
        ),
        topk_weights=wts,
        topk_ids=ids,
        global_num_experts=-1,
        expert_map=None,
        is_expert_parallel=False,
        w1_global_scale=w13.gscale,
        w2_global_scale=w2.gscale,
        workspace=marlin_make_workspace(x.device, max_blocks_per_sm=4),
        num_bits=4,
        is_k_full=True,
        inplace=False,
        routed_scaling_factor=ROUTED,
        clamp_limit=LIMIT,
        activation="silu",
        is_gated=True,
    ).to(x.dtype)


# -- K-order design ------------------------------------------------------------
# The gate_up K range of each (split s, K-warp w) cell is k-tiles 24s+6w ..
# 24s+6w+5 (header spec of the .cuh). Every gate_up row of every expert gets
# one nonzero weight per cell, so each cell's HMMA chain holds one exact
# product P[s][w], and x cancels exactly between paired cells at large
# magnitude in 3 of the 8 pairs per row: the rounded sums then depend on the
# order the 16 cell products are added in.
KO_E = 32
KO_CELLS = 16
KO_CELL_K = 96


def _korder_design(seed: int) -> SimpleNamespace:
    gen = torch.Generator().manual_seed(seed)
    rows = KO_E * 2 * I
    base = torch.tensor([(24 * (c // 4) + 6 * (c % 4)) * 16 for c in range(KO_CELLS)])
    perm = torch.randperm(KO_CELLS, generator=gen)
    primary, partner = perm[0::2], perm[1::2]

    def pow2_scale(lo, hi, shape):
        e = torch.randint(lo, hi, shape, generator=gen).float()
        return 2.0**e * (1 + torch.randint(0, 8, shape, generator=gen).float() / 8)

    offs = torch.randint(KO_CELL_K // 2, KO_CELL_K, (rows, KO_CELLS), generator=gen)
    codes = torch.randint(1, 15, (rows, KO_CELLS), generator=gen)
    codes = codes + (codes >= 8).long()  # no -0
    scl = pow2_scale(-3, 3, (rows, KO_CELLS))
    large = torch.argsort(torch.rand(rows, 8, generator=gen), dim=1) < 3
    big_off = torch.randint(0, KO_CELL_K // 2, (rows, 8), generator=gen)
    big_code = torch.randint(1, 8, (rows, 8), generator=gen)
    big_scale = pow2_scale(0, 7, (rows, 8))
    for p in range(8):
        m = large[:, p]
        for c in (int(primary[p]), int(partner[p])):
            offs[m, c] = big_off[m, p]
            codes[m, c] = big_code[m, p]
            scl[m, c] = big_scale[m, p]
    kpos = base[None, :] + offs
    full = torch.zeros(rows, H, dtype=torch.long)
    full.scatter_(1, kpos, codes)
    scales = torch.ones(rows, H // 16)
    scales.scatter_(1, kpos // 16, scl)
    raw = (full[:, 0::2] | (full[:, 1::2] << 4)).to(torch.uint8)
    s8 = scales.to(torch.float8_e4m3fn)
    assert torch.equal(s8.float(), scales), "scales must be e4m3 values"
    levels = torch.tensor(LEVELS, dtype=torch.float64)
    val = levels[codes] * scl.double() * 2.0**-7
    gscale = torch.full((KO_E,), 2.0**-4, device="cuda")
    packed, enc, g = _pack(
        raw.view(KO_E, 2 * I, H // 2).cuda(),
        s8.view(KO_E, 2 * I, H // 16).cuda(),
        gscale,
    )
    return SimpleNamespace(
        base=base,
        primary=primary,
        partner=partner,
        kpos=kpos.view(KO_E, 2 * I, KO_CELLS).cuda(),
        val=val.view(KO_E, 2 * I, KO_CELLS).cuda(),
        packed=packed,
        scales=enc,
        gscale=g,
    )


def _korder_x(d: SimpleNamespace, m: int, gen: torch.Generator) -> torch.Tensor:
    """x with x[partner cell] = -x[primary cell]; large magnitudes in the
    first half of each cell (where the large weights sit), moderate after."""
    x = torch.empty(m, H, dtype=torch.float64)
    o = torch.arange(KO_CELL_K)
    for p in range(8):
        a, b = int(d.primary[p]), int(d.partner[p])
        sign = torch.where(torch.rand(m, KO_CELL_K, generator=gen) < 0.5, -1.0, 1.0)
        mant = 1 + torch.randint(0, 1024, (m, KO_CELL_K), generator=gen).double() / 1024
        ex_big = torch.randint(6, 15, (m, KO_CELL_K), generator=gen).double()
        ex_small = torch.randint(-4, 3, (m, KO_CELL_K), generator=gen).double()
        v = sign * mant * torch.exp2(torch.where(o < KO_CELL_K // 2, ex_big, ex_small))
        x[:, d.base[a] + o] = v
        x[:, d.base[b] + o] = -v
    xh = x.half()
    assert torch.equal(xh.double(), x), "x must be exact in fp16"
    return xh.cuda()


def _korder_sums(d: SimpleNamespace, x: torch.Tensor, ids: torch.Tensor) -> dict:
    """fp32 sums of the 16 exact cell products (cell c = 4s + w) per route
    and gate_up column: the documented order and alternatives to it."""
    e = ids.long()
    m = x.shape[0]
    kpos = d.kpos[e]  # [M, TOPK, 2I, 16]
    xg = torch.gather(x.double()[:, None, None, :].expand(m, TOPK, 2 * I, H), 3, kpos)
    p64 = xg * d.val[e]
    p32 = p64.float()
    assert torch.equal(p32.double(), p64), "cell products must be exact in fp32"
    c = [p32[..., i] + 0.0 for i in range(KO_CELLS)]

    def seq(terms):
        acc = terms[0]
        for t in terms[1:]:
            acc = acc + t
        return acc

    split = [seq([c[4 * s + w] for w in range(4)]) for s in range(4)]
    split_rev_warps = [seq([c[4 * s + w] for w in (3, 2, 1, 0)]) for s in range(4)]
    return {
        "S": torch.stack(split, -1),
        "S warp_rev": torch.stack(split_rev_warps, -1),
        "G": seq(split),
        "G split_rev": seq(split[::-1]),
        "G split_pair": (split[0] + split[1]) + (split[2] + split[3]),
        "G warp_rev": seq(split_rev_warps),
        "G flat_k": seq(c),
    }


@unittest.skipUnless(_sm70(), "SM70 (V100) only")
class TestSm70Hy4Nvfp4Moe(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        from sglang.srt.layers.quantization.marlin_utils import (
            _sm70_marlin_v100_repack_ops,
        )

        # Without marlin_v100's SM70 repack the stock one is a stub that writes
        # nothing: every weight would be stale memory and the bitwise tests
        # would compare garbage with garbage.
        if _sm70_marlin_v100_repack_ops()[0] is None:
            raise AssertionError("marlin_v100's SM70 repack is not available")
        gen = torch.Generator(device="cuda")
        gen.manual_seed(1234)
        cls.gen = gen
        cls.w13 = _make_weights(2 * I, H, gen)
        cls.w2 = _make_weights(H, I, gen)
        cls.ops = _ops()
        cls._cache13 = {}
        cls._cache2 = {}

    # -- inputs ----------------------------------------------------------------

    def _inputs(self, m: int, pool: int = E):
        """Tokens alternate loud (x * 8, gate/up far past the clamp) and quiet.
        Each token's 16 ids are distinct, drawn from the first ``pool`` experts,
        so a small pool repeats experts across tokens."""
        gen = self.gen
        x = torch.randn(m, H, dtype=torch.float32, device="cuda", generator=gen)
        loud = torch.tensor(
            [8.0 if t % 2 == 0 else 1.0 for t in range(m)], device="cuda"
        )
        x = (x * loud[:, None]).half()
        perm = torch.argsort(torch.rand(m, pool, device="cuda", generator=gen), dim=1)
        ids = perm[:, :TOPK].int().contiguous()
        wts = torch.rand(m, TOPK, device="cuda", generator=gen) * 0.2 + 0.01
        return x, ids, wts

    def _w(self, which: str, e: int) -> torch.Tensor:
        cache = self._cache13 if which == "w13" else self._cache2
        if e not in cache:
            cache[e] = _weight(self.w13 if which == "w13" else self.w2, e)
        return cache[e]

    def _run(self, x, ids, wts, config=None, out=None, poison=True):
        """The op, or one decode kernel config directly. With ``poison`` the
        output and the decode workspace start as NaN, so a value the kernels
        do not write in this call (or read before writing it) cannot pass as
        a stale copy from an earlier call."""
        ops = self.ops
        if poison:
            partials, cache, _ = ops._workspace(x.device)
            partials.fill_(float("nan"))
            cache.fill_(float("nan"))
            if out is None:
                out = torch.full_like(x, float("nan"))
        if config is None:
            return ops.sm70_hy4_nvfp4_moe(
                x,
                self.w13.packed,
                self.w2.packed,
                self.w13.scales,
                self.w2.scales,
                self.w13.gscale,
                self.w2.gscale,
                ids,
                wts,
                LIMIT,
                ROUTED,
                out=out,
            )
        partials, cache, counters = ops._workspace(x.device)
        y = torch.full_like(x, float("nan")) if out is None else out
        ops._decode_module(*config).moe_decode(
            y,
            x,
            self.w13.packed,
            self.w2.packed,
            self.w13.scales,
            self.w2.scales,
            self.w13.gscale,
            self.w2.gscale,
            ids,
            wts,
            partials,
            cache,
            counters,
            LIMIT,
            ROUTED,
        )
        return y

    def _prefill(self, x, ids, wts, config):
        """One prefill kernel config directly, for any M (one chunk). Output
        and workspaces start as NaN."""
        ops = self.ops
        cw1, cw2, blocks = config
        order, expert_off, work_off = ops._plan(ids, E, blocks * ops._ROWS)
        routes = x.shape[0] * TOPK
        nan = float("nan")
        gate_up = torch.full((routes, 2 * I), nan, dtype=torch.float16, device="cuda")
        cache = torch.full((routes, H), nan, dtype=torch.float16, device="cuda")
        y = torch.full_like(x, nan)
        ops._prefill_module(cw1, cw2, blocks).moe_prefill(
            y,
            x,
            self.w13.packed,
            self.w2.packed,
            self.w13.scales,
            self.w2.scales,
            self.w13.gscale,
            self.w2.gscale,
            ids,
            wts,
            order,
            expert_off,
            work_off,
            gate_up,
            cache,
            LIMIT,
            ROUTED,
        )
        return y

    def _token_pool(self, n: int):
        """n tokens: a third route within 24 experts (heavy sharing), the rest
        across all E; a few ids outside [0, E), one token with no valid id, one
        with an id twice. Returns the tokens and each token's output alone."""
        x, ids, wts = self._inputs(n, E)
        _, few, _ = self._inputs(n, 24)
        shared = (torch.arange(n, device="cuda") % 3 == 0)[:, None]
        ids = torch.where(shared, few, ids).contiguous()
        ids[5, 3] = -1
        ids[6, 0] = E
        ids[6, 15] = E + 7
        ids[7, :] = -1
        ids[8, 9] = ids[8, 2]
        _leave_stale_nan()
        alone = torch.cat(
            [self._run(x[t : t + 1], ids[t : t + 1], wts[t : t + 1]) for t in range(n)]
        )
        torch.cuda.synchronize()
        self.assertFalse(bool(torch.isnan(alone).any()))
        self.assertEqual(int(alone[7].abs().amax()), 0)
        return x, ids, wts, alone

    def _assert_rows_equal(self, y, want, label: str):
        bad = (y.view(torch.int16) != want.view(torch.int16)).any(dim=1)
        n_bad = int(bad.sum())
        first = int(bad.nonzero()[0]) if n_bad else -1
        self.assertEqual(n_bad, 0, f"{label}: {n_bad} tokens differ, first row {first}")

    # -- reference -------------------------------------------------------------

    def _reference(self, x, ids, wts):
        """fp64 dots with Marlin's fp16 rounding points. Returns the output,
        per-route gate_up (fp16) and down rows (fp16), and their error scales."""
        m = x.shape[0]
        xd = x.double()
        g13 = self.w13.gscale.double()
        g2 = self.w2.gscale.double()
        gu = torch.zeros(m, TOPK, 2 * I, dtype=torch.float16, device="cuda")
        gu_exact = torch.zeros(m, TOPK, 2 * I, dtype=torch.float64, device="cuda")
        gu_abs = torch.zeros_like(gu_exact)
        rows = torch.zeros(m, TOPK, H, dtype=torch.float16, device="cuda")
        for t in range(m):
            for s in range(TOPK):
                e = int(ids[t, s])
                if not 0 <= e < E:
                    continue
                w = self._w("w13", e)
                gu_exact[t, s] = (w @ xd[t]) * g13[e]
                gu_abs[t, s] = (w.abs() @ xd[t].abs()) * g13[e]
                gu[t, s] = gu_exact[t, s].half()
                act = _swiglu(gu[t, s][None])[0].double()
                d = (self._w("w2", e) @ act) * float(wts[t, s]) * g2[e]
                rows[t, s] = d.half()
        out = torch.stack([_route_sum(rows[t]) for t in range(m)])
        return SimpleNamespace(
            out=out, gu=gu, gu_exact=gu_exact, gu_abs=gu_abs, rows=rows
        )

    def _check_stages(self, x, ids, wts, y, label: str):
        """The decode kernel's own intermediates against the rounding points."""
        m = x.shape[0]
        split = self.ops._SPLIT
        # fp64 weights are 7 MB per expert; keep only this call's experts.
        self._cache13.clear()
        self._cache2.clear()
        partials, cache, _ = self.ops._workspace(x.device)
        ref = self._reference(x, ids, wts)
        g13 = self.w13.gscale
        g2 = self.w2.gscale
        exact_gu = exact_rows = total_gu = total_rows = 0
        u32 = 2.0**-23
        for t in range(m):
            for s in range(TOPK):
                i = t * TOPK + s
                e = int(ids[t, s])
                row = cache[i]
                if not 0 <= e < E:
                    self.assertEqual(int(row.abs().amax()), 0, f"{label} route {i}")
                    continue
                # gate_up: fp32 split sum, then Marlin's store fp16(sum * global).
                p = partials[i * split * 2 * I : (i + 1) * split * 2 * I].view(
                    split, 2 * I
                )
                acc = p[0].clone()
                for k in range(1, split):
                    acc = acc + p[k]
                gu = (acc * g13[e]).half()
                tol = _ulp16(torch.maximum(gu.double().abs(), ref.gu_exact[t, s].abs()))
                tol = tol + F32_DEPTH * u32 * ref.gu_abs[t, s]
                err = (gu.double() - ref.gu_exact[t, s]).abs()
                self.assertTrue(
                    bool((err <= tol).all()),
                    f"{label} gate_up route {i}: {float((err / tol).max())}",
                )
                exact_gu += int((gu == ref.gu[t, s]).sum())
                total_gu += gu.numel()
                # down row from the kernel's own gate_up: fp16((sum * w) * global).
                act = _swiglu(gu[None])[0].double()
                w2e = self._w("w2", e)
                scale = float(wts[t, s]) * float(g2[e])
                exact = (w2e @ act) * scale
                tol = _ulp16(torch.maximum(row.double().abs(), exact.abs()))
                tol = tol + F32_DEPTH * u32 * (w2e.abs() @ act.abs()) * abs(scale)
                err = (row.double() - exact).abs()
                self.assertTrue(
                    bool((err <= tol).all()),
                    f"{label} down route {i}: {float((err / tol).max())}",
                )
                exact_rows += int((row == exact.half()).sum())
                total_rows += row.numel()
            # route sum: bit for bit Marlin's reduce of the kernel's rows.
            want = _route_sum(cache[t * TOPK : (t + 1) * TOPK])
            self.assertTrue(torch.equal(y[t], want), f"{label} route sum token {t}")
        return ref, exact_gu / max(total_gu, 1), exact_rows / max(total_rows, 1)

    def _check_against(self, y, want, ref, label: str):
        """|y - want| within 8 fp16 ulps of the route rows' magnitude sum per
        column (each row may move one ulp at each rounding point) plus one ulp
        of the output. Returns the fraction of bitwise-equal outputs."""
        mag = ROUTED * ref.rows.double().abs().sum(dim=1)
        tol = 8 * _ulp16(mag) + _ulp16(
            torch.maximum(y.double().abs(), want.double().abs())
        )
        err = (y.double() - want.double()).abs()
        self.assertFalse(bool(torch.isnan(y).any()), label)
        self.assertTrue(
            bool((err <= tol).all()),
            f"{label}: worst {float((err / tol).max()):.3f} of the bound",
        )
        return float((y == want).float().mean())

    # -- tests -----------------------------------------------------------------

    def test_against_marlin_and_reference(self):
        stats = []
        for m in range(1, 9):
            for pool in (E, 24):
                if m == 1 and pool != E:
                    continue
                x, ids, wts = self._inputs(m, pool)
                _leave_stale_nan()
                y = self._run(x, ids, wts)
                torch.cuda.synchronize()
                label = f"M={m} pool={pool}"
                ref, gu_eq, rows_eq = self._check_stages(x, ids, wts, y, label)
                # The clamp is live: loud tokens push gate above and up past +-LIMIT.
                gate, up = ref.gu[..., :I].float(), ref.gu[..., I:].float()
                self.assertGreater(int((gate > LIMIT).sum()), 0, label)
                self.assertGreater(int((up.abs() > LIMIT).sum()), 0, label)
                eq_ref = self._check_against(y, ref.out, ref, label + " vs fp64")
                want = _marlin(x, self.w13, self.w2, ids, wts)
                eq_marlin = self._check_against(y, want, ref, label + " vs marlin")
                # Marlin sits within the same bound of the reference.
                self._check_against(want, ref.out, ref, label + " marlin vs fp64")
                stats.append((label, gu_eq, rows_eq, eq_ref, eq_marlin))
        for label, gu_eq, rows_eq, eq_ref, eq_marlin in stats:
            print(
                f"{label}: bitwise equal gate_up {gu_eq:.4f} down rows {rows_eq:.4f} "
                f"output vs fp64 {eq_ref:.4f} vs marlin {eq_marlin:.4f}"
            )

    def test_prefill_against_marlin_and_reference(self):
        for m, pool in ((9, E), (64, 48), (300, 40)):
            x, ids, wts = self._inputs(m, pool)
            _leave_stale_nan()
            y = self._run(x, ids, wts)
            torch.cuda.synchronize()
            label = f"prefill M={m} pool={pool}"
            self._cache13.clear()
            self._cache2.clear()
            ref = self._reference(x, ids, wts)
            gate, up = ref.gu[..., :I].float(), ref.gu[..., I:].float()
            self.assertGreater(int((gate > LIMIT).sum()), 0, label)
            self.assertGreater(int((up.abs() > LIMIT).sum()), 0, label)
            eq_ref = self._check_against(y, ref.out, ref, label + " vs fp64")
            want = _marlin(x, self.w13, self.w2, ids, wts)
            eq_marlin = self._check_against(y, want, ref, label + " vs marlin")
            self._check_against(want, ref.out, ref, label + " marlin vs fp64")
            print(
                f"{label}: bitwise equal output vs fp64 {eq_ref:.4f} vs marlin {eq_marlin:.4f}"
            )
        self._cache13.clear()
        self._cache2.clear()

    def test_batch_invariance(self):
        """Every token's output is bit for bit its output alone (M=1), for every
        batch size, position, batch mate, and kernel config."""
        ops = self.ops
        n = 2048
        x, ids, wts, alone = self._token_pool(n)
        pick = torch.Generator(device="cpu")
        pick.manual_seed(99)

        def batch(m, trial):
            # Random tokens, repeats allowed. On even trials some of the special
            # tokens 5..8 (ids outside [0, E), a repeated id) join, at most half
            # the batch.
            idx = torch.randint(0, n, (m,), generator=pick)
            if trial % 2 == 0:
                k = min(4, max(1, m // 2))
                pos = torch.randperm(m, generator=pick)[:k]
                idx[pos] = 5 + torch.randperm(4, generator=pick)[:k]
            return idx.cuda()

        decode = sorted(set(ops._DECODE_CONFIGS.values()))
        decode += [(4, False, 2, 4), (8, False, 2, 4), (8, True, 1, 2), (8, True, 3, 8)]
        for m in range(2, 9):
            for trial in range(8):
                idx = batch(m, trial)
                xb, ib, wb = x[idx], ids[idx].contiguous(), wts[idx].contiguous()
                want = alone[idx]
                self._assert_rows_equal(self._run(xb, ib, wb), want, f"M={m} op")
                for config in decode:
                    if config[0] >= m:
                        y = self._run(xb, ib, wb, config=config)
                        self._assert_rows_equal(y, want, f"M={m} decode {config}")
                # The prefill kernels on a decode-sized batch.
                y = self._prefill(xb, ib, wb, ops._PREFILL_CONFIG)
                self._assert_rows_equal(y, want, f"M={m} prefill kernels")

        prefill = [ops._PREFILL_CONFIG, (1, 2, 4), (2, 4, 1), (2, 4, 4), (3, 2, 2)]
        for m in (9, 16, 64, 512, 2048, 2048 + 300):
            for trial in range(2):
                idx = batch(m, trial)
                xb, ib, wb = x[idx], ids[idx].contiguous(), wts[idx].contiguous()
                want = alone[idx]
                _leave_stale_nan()
                self._assert_rows_equal(self._run(xb, ib, wb), want, f"M={m} op")
                if m <= ops.PREFILL_CHUNK and trial == 0:
                    for config in prefill:
                        y = self._prefill(xb, ib, wb, config)
                        self._assert_rows_equal(y, want, f"M={m} prefill {config}")
        # The whole pool in one call, every token once.
        self._assert_rows_equal(self._run(x, ids, wts), alone, f"M={n} pool")

    def test_sum_order_bit_exact(self):
        """The documented per-element order (header comment of the .cuh), bit
        for bit. With one nonzero weight per output row every dot product is a
        single exact product, so the steps the order names can be computed here
        exactly: the gate_up store fp16(G * g13), SwiGLU, the row store
        fp16((v * w) * g2) and the slot-order fp32 route sum times the routed
        scale. Route weights span 2^0 .. 2^-15, so the route sum rounds and a
        different slot order shows."""
        ops = self.ops
        gen = torch.Generator(device="cuda")
        gen.manual_seed(4321)
        w13 = _make_weights(2 * I, H, gen, sparse=True)
        w2 = _make_weights(H, I, gen, sparse=True)
        args = (w13.packed, w2.packed, w13.scales, w2.scales, w13.gscale, w2.gscale)
        # x such that |gate_up| reaches ~4x the clamp on loud rows.
        loud_scale = 4 * LIMIT / float((w13.val.abs() * w13.gscale[:, None]).median())
        slot_scale = torch.exp2(-torch.arange(TOPK, device="cuda", dtype=torch.float32))

        def inputs(m):
            x = torch.randn(m, H, device="cuda", generator=gen)
            # No fp16 subnormals in x.
            x = torch.where(x.abs() < 0.01, torch.full_like(x, 0.01), x)
            rowscale = torch.tensor(
                [loud_scale if t % 2 == 0 else loud_scale / 16 for t in range(m)],
                device="cuda",
            )
            x = (x * rowscale[:, None]).half()
            perm = torch.argsort(torch.rand(m, E, device="cuda", generator=gen), dim=1)
            ids = perm[:, :TOPK].int().contiguous()
            wts = (torch.rand(m, TOPK, device="cuda", generator=gen) + 0.5) * slot_scale
            return x, ids, wts.contiguous()

        def emulate(x, ids, wts):
            m = x.shape[0]
            e = ids.long()
            xg = torch.gather(
                x.double()[:, None, :].expand(-1, TOPK, -1), 2, w13.kpos[e]
            )
            g = (xg * w13.val[e]).float() + 0.0  # one exact product; -0 sums to +0
            gu = (g * w13.gscale[e][..., None]).half()
            act = _swiglu(gu.view(-1, 2 * I)).view(m, TOPK, I)
            v = (torch.gather(act.double(), 2, w2.kpos[e]) * w2.val[e]).float() + 0.0
            rows = ((v * wts[..., None]) * w2.gscale[e][..., None]).half()
            acc = torch.zeros(m, H, dtype=torch.float32, device="cuda")
            for k in range(TOPK):
                acc = acc + rows[:, k].float()
            return gu, rows, (acc * ROUTED).half()

        def n_diff(a, b):
            return int((a.view(torch.int16) != b.view(torch.int16)).sum())

        gate_over = up_over = 0
        for m in range(1, 9):
            for trial in range(6):
                x, ids, wts = inputs(m)
                partials, cache, _ = ops._workspace(x.device)
                partials.fill_(float("nan"))
                cache.fill_(float("nan"))
                y = torch.full_like(x, float("nan"))
                ops.sm70_hy4_nvfp4_moe(x, *args, ids, wts, LIMIT, ROUTED, out=y)
                cache = cache[: m * TOPK].view(m, TOPK, H)
                gu, rows, want = emulate(x, ids, wts)
                gate_over += int((gu[..., :I] > LIMIT).sum())
                up_over += int((gu[..., I:].abs() > LIMIT).sum())
                label = f"decode M={m} trial {trial}"
                self.assertEqual(n_diff(cache, rows), 0, f"{label}: down rows")
                self.assertEqual(n_diff(y, want), 0, f"{label}: output")
        self.assertGreater(gate_over, 0)
        self.assertGreater(up_over, 0)
        for m in (9, 64, 512):
            x, ids, wts = inputs(m)
            _, _, want = emulate(x, ids, wts)
            _leave_stale_nan()
            y = ops.sm70_hy4_nvfp4_moe(x, *args, ids, wts, LIMIT, ROUTED)
            self.assertEqual(n_diff(y, want), 0, f"prefill M={m}: output")

    def test_gate_up_k_order_bit_exact(self):
        """The documented gate_up K order bit for bit: S[s] = ((P0+P1)+P2)+P3
        over the K-warps of split s (decode partials, fp32), then
        G = ((S0+S1)+S2)+S3 (prefill gate_up, fp16(G * g13)). On data where
        a different order changes a sizeable share of the values (checked)."""
        ops = self.ops
        d = _korder_design(7)
        gen = torch.Generator().manual_seed(11)
        w2 = _make_weights(H, I, self.gen)
        w2_packed = w2.packed[:KO_E].contiguous()
        w2_scales = w2.scales[:KO_E].contiguous()
        w2_g = w2.gscale[:KO_E].contiguous()
        args = (d.packed, w2_packed, d.scales, w2_scales, d.gscale, w2_g)
        nan = float("nan")
        prefill_cfgs = [ops._PREFILL_CONFIG, (1, 2, 4), (2, 4, 1), (3, 2, 2)]
        changed = {}

        def n_diff(a, b):
            view = torch.int32 if a.dtype == torch.float32 else torch.int16
            return int((a.view(view) != b.view(view)).sum())

        def inputs(m):
            x = _korder_x(d, m, gen)
            ids = torch.argsort(torch.rand(m, KO_E, generator=gen), dim=1)[:, :TOPK]
            wts = torch.rand(m, TOPK, generator=gen) * 0.2 + 0.01
            return x, ids.int().cuda().contiguous(), wts.cuda().contiguous()

        def prefill(x, ids, wts, cfg):
            cw1, cw2, blocks = cfg
            order, eoff, woff = ops._plan(ids, KO_E, blocks * ops._ROWS)
            routes = x.shape[0] * TOPK
            gu = torch.full((routes, 2 * I), nan, dtype=torch.float16, device="cuda")
            cache = torch.full((routes, H), nan, dtype=torch.float16, device="cuda")
            y = torch.full_like(x, nan)
            ops._prefill_module(cw1, cw2, blocks).moe_prefill(
                y, x, *args, ids, wts, order, eoff, woff, gu, cache, LIMIT, ROUTED
            )
            return y, gu.view(x.shape[0], TOPK, 2 * I)

        def alone(x, ids, wts):
            return torch.cat(
                [
                    ops.sm70_hy4_nvfp4_moe(
                        x[t : t + 1],
                        *args,
                        ids[t : t + 1],
                        wts[t : t + 1],
                        LIMIT,
                        ROUTED,
                    )
                    for t in range(x.shape[0])
                ]
            )

        for m in (1, 2, 5, 8, 9, 64):
            for trial in range(2 if m <= 8 else 1):
                label = f"M={m} trial {trial}"
                x, ids, wts = inputs(m)
                sums = _korder_sums(d, x, ids)
                g13 = d.gscale[ids.long()][..., None]
                gu_doc = (sums["G"] * g13).half()
                for k, v in sums.items():
                    ref = sums["S"] if k.startswith("S") else sums["G"]
                    if k not in ("S", "G"):
                        n, tot = changed.get(k, (0, 0))
                        changed[k] = (n + n_diff(v, ref), tot + v.numel())
                if m <= 8:
                    partials, cache, _ = ops._workspace(x.device)
                    partials.fill_(nan)
                    cache.fill_(nan)
                y = torch.full_like(x, nan)
                ops.sm70_hy4_nvfp4_moe(x, *args, ids, wts, LIMIT, ROUTED, out=y)
                if m <= 8:
                    got = partials[: m * TOPK * 4 * 2 * I].view(m, TOPK, 4, 2 * I)
                    self.assertEqual(
                        n_diff(got.transpose(2, 3), sums["S"]),
                        0,
                        f"{label}: decode S[s]",
                    )
                self.assertEqual(n_diff(y, alone(x, ids, wts)), 0, f"{label}: vs alone")
                for cfg in prefill_cfgs:
                    yp, gu = prefill(x, ids, wts, cfg)
                    self.assertEqual(n_diff(gu, gu_doc), 0, f"{label} {cfg}: gate_up")
                    self.assertEqual(n_diff(yp, y), 0, f"{label} {cfg}: output")
        for k, (n, tot) in changed.items():
            self.assertGreater(n / tot, 0.05, f"order {k} would change too few values")

    def test_first_call_inside_capture_raises(self):
        """The workspace's zero counters cannot be allocated inside a capture
        (the fill would only be recorded), so a first call there raises."""
        ops = self.ops
        x, ids, wts = self._inputs(1, 40)
        self._run(x, ids, wts)  # builds the module
        torch.cuda.synchronize()
        saved = dict(ops._WS)
        ops._WS.clear()
        try:
            graph = torch.cuda.CUDAGraph()
            with self.assertRaisesRegex(RuntimeError, "outside CUDA graph capture"):
                with torch.cuda.graph(graph):
                    self._run(x, ids, wts, poison=False)
            self.assertFalse(ops._WS)
            del graph
        finally:
            ops._WS.clear()
            ops._WS.update(saved)

    def test_counters_return_to_zero_and_runs_repeat(self):
        for m in (1, 3, 6, 8):
            x, ids, wts = self._inputs(m, 40)
            first = self._run(x, ids, wts)
            second = self._run(x, ids, wts)
            torch.cuda.synchronize()
            self.assertTrue(torch.equal(first, second), f"M={m}")
            counters = self.ops._workspace(x.device)[2]
            self.assertEqual(int(counters.abs().sum()), 0, f"M={m}")

    def test_configs_agree_bitwise(self):
        """Padding rows and sharing an expert's weights change no bit."""
        rows_cfg = {r: self.ops._DECODE_CONFIGS[r] for r in (1, 2, 4, 8)}
        no_dedupe = (4, False) + rows_cfg[4][2:]
        for m in (1, 2, 3, 4):
            x, ids, wts = self._inputs(m, 20)
            outs = [
                self._run(x, ids, wts, config=c) for r, c in rows_cfg.items() if r >= m
            ]
            outs.append(self._run(x, ids, wts, config=no_dedupe))
            torch.cuda.synchronize()
            for o in outs[1:]:
                self.assertTrue(torch.equal(outs[0], o), f"M={m}")

    def test_invalid_ids_add_zero(self):
        m = 3
        x, ids, wts = self._inputs(m, 60)
        ids[0, 3] = -1
        ids[1, 0] = E
        ids[1, 15] = -1
        ids[2, :] = -1
        y = self._run(x, ids, wts)
        torch.cuda.synchronize()
        ref, _, _ = self._check_stages(x, ids, wts, y, "invalid ids")
        self._check_against(y, ref.out, ref, "invalid ids vs fp64")
        self.assertEqual(int(y[2].abs().amax()), 0)

    def test_output_may_alias_input(self):
        for m in (1, 5, 64, 2048 + 100):
            x, ids, wts = self._inputs(m)
            want = self._run(x, ids, wts)
            xa = x.clone()
            got = self._run(xa, ids, wts, out=xa)
            torch.cuda.synchronize()
            self.assertEqual(got.data_ptr(), xa.data_ptr())
            self.assertTrue(torch.equal(got, want), f"M={m}")

    def test_cuda_graph_replay(self):
        for m in (1, 6, 64):
            x, ids, wts = self._inputs(m, 100)
            sx, sids, swts = x.clone(), ids.clone(), wts.clone()
            self._run(sx, sids, swts)  # build and allocate before capture
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                sy = self._run(sx, sids, swts)
            for _ in range(3):
                x, ids, wts = self._inputs(m, 100)
                sx.copy_(x)
                sids.copy_(ids)
                swts.copy_(wts)
                graph.replay()
                want = self._run(x, ids, wts)
                torch.cuda.synchronize()
                self.assertTrue(torch.equal(sy, want), f"M={m}")
            del graph

    def test_eligibility(self):
        ops = self.ops
        x, ids, wts = self._inputs(2)
        args = [
            self.w13.packed,
            self.w2.packed,
            self.w13.scales,
            self.w2.scales,
            self.w13.gscale,
            self.w2.gscale,
        ]

        def ok(x_=x, ids_=ids, wts_=wts, limit=LIMIT, w=args):
            return ops.sm70_hy4_nvfp4_moe_eligible(x_, *w, ids_, wts_, limit)

        self.assertTrue(ops.sm70_hy4_nvfp4_moe_available())
        self.assertTrue(ok())
        self.assertTrue(ok(limit=0.0))
        self.assertFalse(ok(limit=None))  # fused_marlin_moe would use silu_and_mul
        self.assertFalse(ok(limit=float("inf")))
        # Every batch size runs here, so prefill and decode agree.
        for m in (9, 4096):
            xm = torch.zeros(m, H, dtype=torch.float16, device="cuda")
            self.assertTrue(
                ok(x_=xm, ids_=ids.new_zeros(m, TOPK), wts_=wts.new_zeros(m, TOPK))
            )
        x0 = torch.zeros(0, H, dtype=torch.float16, device="cuda")
        self.assertFalse(
            ok(x_=x0, ids_=ids.new_zeros(0, TOPK), wts_=wts.new_zeros(0, TOPK))
        )
        self.assertFalse(ok(x_=x.float()))
        self.assertFalse(
            ok(x_=torch.zeros(2, H + 8, dtype=torch.float16, device="cuda")[:, 8:])
        )
        self.assertFalse(ok(ids_=ids.long()))
        self.assertFalse(ok(wts_=wts.half()))
        self.assertFalse(ok(ids_=ids[:, :8].contiguous(), wts_=wts[:, :8].contiguous()))
        self.assertFalse(ok(w=args[:4] + [None, args[5]]))
        self.assertFalse(
            ok(
                w=[
                    args[0],
                    args[1],
                    args[2].view(torch.uint8),
                    args[3],
                    args[4],
                    args[5],
                ]
            )
        )
        self.assertFalse(ok(w=[args[0][:256], args[1]] + args[2:]))


if __name__ == "__main__":
    unittest.main()
