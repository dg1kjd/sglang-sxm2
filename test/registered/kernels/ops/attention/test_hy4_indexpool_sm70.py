"""Hunyuan-4-Flash IndexPool indexer on SM70 (hy4_flash_indexpool_sm70).

Each stage against an fp64 reference; the selected index rows against the tie
rule (higher score first, then the lower group index); the window layers'
index fill. Not registered for GPU CI (V100 worktree only).
"""

from __future__ import annotations

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(
    est_time=180,
    stage="base-b-kernel-unit",
    runner_config="1-gpu-large",
    disabled="SM70 V100 worktree only; do not register GPU CI",
)

HEADS = 16
DIM = 128
RATIO = 4
PAGE = 64
GROUPS_PER_PAGE = PAGE // RATIO
PAGE_BYTES = PAGE * (DIM + 4)
K_GROUPS = 511
TOPK = 2048
EPS = 1e-6
SECTION = (12, 10, 10)
WINDOW = 512
# Score error bound relative to sum_h |w_h| * sum_d |q_hd * k_d|: fp16 products
# are exact on HMMA, the fp32 sums of 128 terms stay well inside this.
SCORE_RTOL = 1e-4


def _sm70() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (7, 0)


def _ops():
    from sglang.kernels.ops.attention.dsa import hy4_flash_indexpool_sm70

    return hy4_flash_indexpool_sm70


def _leave_stale_nan() -> None:
    # Volta's L1 shares SRAM with shared memory, so this also seeds shared memory.
    junk = torch.full((64 << 20,), float("nan"), device="cuda")
    junk.sum()
    del junk


# ---------------------------------------------------------------------------
# fp64 reference
# ---------------------------------------------------------------------------


def _inv_freq() -> torch.Tensor:
    return (1e7 ** (-torch.arange(0, 64, 2, dtype=torch.float32) / 64)).cuda()


def _pair_positions(pos: torch.Tensor, section=None) -> torch.Tensor:
    """Position of each of the 32 rotary pairs: [N] or [N, 3] -> [N, 32]. With
    MRoPE, pair p takes h if p % 3 == 1 and p < 3 * section[1], w if p % 3 == 2
    and p < 3 * section[2], else t."""
    pos = pos.long()
    if section is None:
        return pos[:, None].expand(-1, 32)
    if pos.ndim == 1:
        pos = pos[:, None].expand(-1, 3)
    pair = torch.arange(32, device=pos.device)
    axis = torch.zeros_like(pair)
    axis[(pair % 3 == 1) & (pair < 3 * section[1])] = 1
    axis[(pair % 3 == 2) & (pair < 3 * section[2])] = 2
    return pos[:, axis]


def _rope(x: torch.Tensor, pair_pos: torch.Tensor, inv_freq: torch.Tensor):
    """GPT-J rope on dims 64..127, rotated half de-interleaved. The angle is the
    fp32 product position * inv_freq, as in the model."""
    angle = (pair_pos.float() * inv_freq).double()
    c, s = angle.cos(), angle.sin()
    xe, xo = x[..., 64::2], x[..., 65::2]
    return torch.cat([x[..., :64], xe * c - xo * s, xo * c + xe * s], -1)


def _rms(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + EPS) * weight.double()


def _pool_ref(raw4: torch.Tensor, pair_pos: torch.Tensor, w: dict) -> torch.Tensor:
    """Pooled key of each group from its 4 raw rows [G, 4, 128]: softmax over the
    rows of gates + ape per channel, weighted sum of values, RMSNorm, rope."""
    x = raw4.double()
    values = x @ w["wv_t"].double()
    gates = x @ w["wg_t"].double() + w["ape"].double()
    pooled = (values * gates.softmax(1)).sum(1)
    return _rope(_rms(pooled, w["key_norm"]), pair_pos, w["inv_freq"])


def _scores_ref(q: torch.Tensor, w: torch.Tensor, keys: torch.Tensor):
    """fp64 sum_h relu(q_h . k) * w_h for q [R, 16, 128], w [R, 16], keys [G, 128],
    and the error scale sum_h |w_h| * (|q_h| . |k|)."""
    qd, kd, wd = q.double(), keys.double(), w.double()[..., None]
    dots = torch.einsum("rhd,gd->rhg", qd, kd)
    scale = torch.einsum("rhd,gd->rhg", qd.abs(), kd.abs())
    return (dots.clamp_min(0) * wd).sum(1), (scale * wd.abs()).sum(1)


def _order_keys(v: torch.Tensor) -> torch.Tensor:
    """The kernel's order-preserving 32-bit key of fp32 scores (as int64)."""
    v = torch.where(v == 0, torch.zeros_like(v), v).contiguous()
    u = v.view(torch.int32).long() & 0xFFFFFFFF
    return torch.where(u >= 0x80000000, (~u) & 0xFFFFFFFF, u | 0x80000000)


def _hist(scores: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """[R, 2048] int32 counts of the top 11 key bits over each row's groups."""
    keys = _order_keys(scores) >> 21
    live = torch.arange(scores.shape[1], device=scores.device) < upper[:, None]
    rows = torch.arange(scores.shape[0], device=scores.device)[:, None]
    flat = (rows * 2048 + keys)[live]
    counts = torch.bincount(flat, minlength=scores.shape[0] * 2048)
    return counts.view(-1, 2048).int()


def _picked(scores: torch.Tensor, upper: torch.Tensor, k: int = K_GROUPS):
    """[R, G] bool: the k groups of each row's first upper[r] taken by the tie
    rule (higher score first, then the lower group index); all when upper <= k.
    -0 and +0 are equal."""
    col = torch.arange(scores.shape[1], device=scores.device)
    live = col[None, :] < upper[:, None]
    v = torch.where(live, scores.double(), float("-inf"))
    v = torch.where(v == 0, torch.zeros_like(v), v)
    order = torch.sort(-v, dim=1, stable=True).indices[:, :k]
    picked = torch.zeros_like(live)
    picked.scatter_(1, order, True)
    return picked & live


def _paged(pages: torch.Tensor, row_batch: torch.Tensor):
    def index(rows, toks):
        b = row_batch[rows].long()
        col = (toks // PAGE).clamp_max(pages.shape[1] - 1)
        return pages[b, col].long() * PAGE + toks % PAGE

    return index


def _ragged(offsets: torch.Tensor):
    return lambda rows, toks: offsets[rows] + toks


def _expand(picked, pos, token_index, topk=TOPK):
    """Index rows: the picked groups' 4 tokens in group order, then the row's
    own group up to the row, then -1."""
    rows_n = picked.shape[0]
    dev = picked.device
    pos = pos.long()
    out = torch.full((rows_n, topk), -1, dtype=torch.int64, device=dev)
    four = torch.arange(RATIO, device=dev)
    rows, groups = picked.nonzero(as_tuple=True)
    rank = (picked.long().cumsum(1) - 1)[rows, groups]
    out[rows[:, None], rank[:, None] * RATIO + four] = token_index(
        rows[:, None], groups[:, None] * RATIO + four
    )
    n = picked.sum(1)
    toks = (pos.clamp_min(0) // RATIO)[:, None] * RATIO + four
    live = (pos[:, None] >= 0) & (toks <= pos[:, None])
    every = torch.arange(rows_n, device=dev)[:, None].expand(-1, RATIO)
    cols = n[:, None] * RATIO + four
    out[every[live], cols[live]] = token_index(every, toks)[live]
    return out.int()


def _upper(pos: torch.Tensor, num_groups: int) -> torch.Tensor:
    return (pos.long().clamp_min(0) // RATIO).clamp_max(num_groups)


# ---------------------------------------------------------------------------
# Synthetic cache
# ---------------------------------------------------------------------------


def _weights(gen: torch.Generator) -> dict:
    return dict(
        wv_t=(torch.randn(DIM, DIM, generator=gen) / DIM**0.5).cuda(),
        wg_t=(torch.randn(DIM, DIM, generator=gen) / DIM**0.5).cuda(),
        ape=(torch.randn(RATIO, DIM, generator=gen) * 0.5).cuda(),
        key_norm=(1 + 0.1 * torch.randn(DIM, generator=gen)).cuda(),
        token_norm=(1 + 0.1 * torch.randn(DIM, generator=gen)).cuda(),
        inv_freq=_inv_freq(),
    )


def _pool_args(w: dict):
    return (w["wv_t"], w["wg_t"], w["ape"], w["key_norm"], EPS, w["inv_freq"])


class _Cache:
    """Requests on shuffled 64-token pages (page 0, the padding page, is never
    handed out), the DSA index-key cache [pages, 64 * 132] bytes filled with
    random bytes, and an fp16 raw-key buffer of pool size + 2 pages + 1 rows."""

    def __init__(self, lens, seed: int, spare_columns: int = 1):
        self.gen = torch.Generator().manual_seed(seed)
        self.lens = list(lens)
        counts = [-(-n // PAGE) for n in lens]
        self.num_pages = sum(counts) + 2
        perm = torch.randperm(self.num_pages - 1, generator=self.gen) + 1
        pages = torch.zeros(len(lens), max(counts) + spare_columns, dtype=torch.int32)
        offset = 0
        for b, n in enumerate(counts):
            pages[b, :n] = perm[offset : offset + n].int()
            offset += n
        self.pages = pages.cuda()
        self.buf = torch.randint(
            0, 256, (self.num_pages, PAGE_BYTES), generator=self.gen, dtype=torch.uint8
        ).cuda()
        self.raw = torch.zeros(
            self.num_pages * PAGE + 2 * PAGE + 1,
            DIM,
            dtype=torch.float16,
            device="cuda",
        )

    def randn(self, *shape, dtype=torch.float32, scale=1.0):
        return (torch.randn(*shape, generator=self.gen) * scale).to(dtype).cuda()

    def slots(self, b: int, toks) -> torch.Tensor:
        toks = torch.as_tensor(toks, device="cuda").long()
        return self.pages[b].long()[toks // PAGE] * PAGE + toks % PAGE

    def keys(self) -> torch.Tensor:
        """[pages, 16, 128] fp16 view: the key of group slot j of each page, i.e.
        the key bytes of the page's token rows 4j and 4j + 1."""
        groups = self.buf[:, : PAGE * DIM].view(-1, GROUPS_PER_PAGE, RATIO * DIM)
        return groups[:, :, : 2 * DIM].view(torch.float16)

    def group_keys(self, b: int, groups) -> torch.Tensor:
        groups = torch.as_tensor(groups, device="cuda").long()
        return self.keys()[
            self.pages[b].long()[groups // GROUPS_PER_PAGE], groups % GROUPS_PER_PAGE
        ]

    def write_group_keys(self, b: int, keys: torch.Tensor) -> None:
        groups = torch.arange(keys.shape[0], device="cuda")
        self.keys()[
            self.pages[b].long()[groups // GROUPS_PER_PAGE], groups % GROUPS_PER_PAGE
        ] = keys

    def key_byte_mask(self, b: int, groups) -> torch.Tensor:
        """[pages, 64 * 132] bool: the cache bytes the keys of `groups` occupy."""
        mask = torch.zeros_like(self.buf, dtype=torch.bool)
        groups = torch.as_tensor(groups, device="cuda").long()
        page = self.pages[b].long()[groups // GROUPS_PER_PAGE]
        start = (groups % GROUPS_PER_PAGE) * RATIO * DIM
        cols = start[:, None] + torch.arange(2 * DIM, device="cuda")
        mask[page[:, None].expand_as(cols), cols] = True
        return mask


@unittest.skipUnless(_sm70(), "Hy4 IndexPool SM70 kernels require an NVIDIA V100")
class TestHy4IndexPoolSm70(CustomTestCase):
    # -- assertions -------------------------------------------------------

    def assert_fp16(self, actual, expected, msg, atol=1e-5):
        """fp16 `actual` within one fp16 ulp (plus `atol`) of fp64 `expected`."""
        expected = expected.double()
        ulp = torch.exp2(
            torch.floor(torch.log2(expected.abs().clamp_min(2.0**-14))) - 10
        )
        excess = ((actual.double() - expected).abs() / (ulp + atol)).max().item()
        self.assertLessEqual(excess, 1.0, f"{msg}: error {excess:.2f}x the bound")

    def assert_scores(self, scores, ref, scale, upper, msg):
        """Scores within the error bound for groups below `upper`; the rest of
        the row (pre-filled with NaN) left unwritten."""
        live = torch.arange(scores.shape[1], device=scores.device) < upper[:, None]
        err = ((scores.double() - ref).abs() - SCORE_RTOL * scale - 1e-6)[live]
        self.assertTrue(torch.isfinite(scores[live]).all().item(), f"{msg}: non-finite")
        self.assertLessEqual(err.max().item() if err.numel() else 0.0, 0.0, msg)
        self.assertTrue(torch.isnan(scores[~live]).all().item(), f"{msg}: stray write")

    def assert_ranking(self, picked, ref, scale, upper, msg, k=K_GROUPS):
        """Each picked group scores at least each unpicked one, within the score
        error; min(upper, k) groups picked."""
        tol = SCORE_RTOL * scale + 1e-6
        live = torch.arange(ref.shape[1], device=ref.device) < upper[:, None]
        low = torch.where(picked, ref + tol, float("inf")).min(1).values
        high = torch.where(live & ~picked, ref - tol, float("-inf")).max(1).values
        self.assertTrue((low >= high).all().item(), f"{msg}: out of order")
        self.assertTrue(
            torch.equal(picked.sum(1), upper.clamp_max(k)), f"{msg}: picked count"
        )

    # -- stages -----------------------------------------------------------

    def test_query(self):
        ops = _ops()
        gen = torch.Generator().manual_seed(1)
        inv = _inv_freq()
        rows = 37
        for q_dtype in (torch.bfloat16, torch.float16):
            for hw_dtype in (torch.float32, torch.bfloat16, torch.float16):
                for pos_dtype in (torch.int64, torch.int32):
                    # A strided slice, as the projection output is consumed.
                    wide = (torch.randn(rows, HEADS * DIM + 64, generator=gen) * 2).to(
                        q_dtype
                    )
                    q_raw = wide.cuda()[:, 32 : 32 + HEADS * DIM]
                    hw = torch.randn(rows, HEADS, generator=gen).to(hw_dtype).cuda()
                    pos = torch.randint(0, 300000, (rows,), generator=gen)
                    pos[:4] = torch.tensor([0, 1, 262143, 262144])
                    pos = pos.to(pos_dtype).cuda()
                    _leave_stale_nan()
                    q, w, hist = ops.indexpool_query_sm70(
                        q_raw, pos, hw, inv, HEADS, with_hist=True
                    )
                    ref = _rope(
                        q_raw.view(rows, HEADS, DIM).double(),
                        _pair_positions(pos)[:, None, :],
                        inv,
                    )
                    msg = f"q {q_dtype} hw {hw_dtype} pos {pos_dtype}"
                    self.assert_fp16(q, ref, msg)
                    scale = torch.tensor(HEADS**-0.5 * DIM**-0.5, dtype=torch.float32)
                    self.assertTrue(torch.equal(w, hw.float() * scale.cuda()), msg)
                    self.assertTrue((hist == 0).all().item(), f"{msg}: histogram")

    def test_query_mrope(self):
        ops = _ops()
        gen = torch.Generator().manual_seed(2)
        inv = _inv_freq()
        rows = 21
        q_raw = torch.randn(rows, HEADS * DIM, generator=gen).bfloat16().cuda()
        hw = torch.randn(rows, HEADS, generator=gen).cuda()
        # The model passes positions[:, :rows].t() of a [3, N] tensor.
        pos3 = torch.randint(0, 5000, (3, rows + 5), generator=gen).cuda()
        pos = pos3[:, :rows].t()
        q, _, hist = ops.indexpool_query_sm70(
            q_raw, pos, hw, inv, HEADS, mrope_section=SECTION
        )
        self.assertIsNone(hist)
        ref = _rope(
            q_raw.view(rows, HEADS, DIM).double(),
            _pair_positions(pos, SECTION)[:, None, :],
            inv,
        )
        self.assert_fp16(q, ref, "MRoPE query")

    def test_store_raw(self):
        ops = _ops()
        gen = torch.Generator().manual_seed(3)
        norm = (1 + 0.1 * torch.randn(DIM, generator=gen)).cuda()
        rows = 77
        for dtype in (torch.bfloat16, torch.float16):
            for loc_dtype in (torch.int64, torch.int32):
                raw = torch.zeros(1000, DIM, dtype=torch.float16, device="cuda")
                wide = torch.randn(rows, DIM + 16, generator=gen).to(dtype).cuda()
                key = wide[:, :DIM]
                loc = torch.randperm(1000, generator=gen)[:rows].to(loc_dtype).cuda()
                ops.indexpool_store_raw_sm70(key, norm, EPS, loc, raw)
                msg = f"store_raw {dtype} {loc_dtype}"
                self.assert_fp16(raw[loc.long()], _rms(key.double(), norm), msg)
                untouched = torch.ones(1000, dtype=torch.bool, device="cuda")
                untouched[loc.long()] = False
                self.assertTrue(
                    (raw[untouched] == 0).all().item(), f"{msg}: stray write"
                )

    def _check_pool_write(self, section):
        ops = _ops()
        cache = _Cache([4096, 777], seed=4)
        w = _weights(cache.gen)
        cache.raw.copy_(cache.randn(*cache.raw.shape, dtype=torch.float16))
        before = cache.buf.clone()
        # Groups 0..1023 of request 0 and 0..193 of request 1; every third skipped.
        groups = [torch.arange(1024), torch.arange(194)]
        members, first, write, live = [], [], [], []
        for b, g in enumerate(groups):
            toks = g.cuda()[:, None] * RATIO + torch.arange(RATIO, device="cuda")
            slots = cache.slots(b, toks)
            members.append(slots)
            first.append(toks[:, 0])
            keep = (g % 3 != 2).cuda()
            write.append(torch.where(keep, slots[:, 0], -1))
            live.append(keep)
        members = torch.cat(members)
        first = torch.cat(first)
        write = torch.cat(write)
        live = torch.cat(live)
        positions = first
        if section is not None:
            positions = torch.stack(
                [
                    first,
                    first + 7,
                    torch.randint(0, 99, first.shape, generator=cache.gen).cuda(),
                ],
                1,
            ).int()
        _leave_stale_nan()
        ops.indexpool_pool_write_sm70(
            cache.raw,
            members,
            positions,
            write,
            *_pool_args(w),
            cache.buf,
            PAGE,
            mrope_section=section,
        )
        ref = _pool_ref(cache.raw[members], _pair_positions(positions, section), w)
        page, off = write[live] // PAGE, (write[live] % PAGE) * DIM
        cols = off[:, None] + torch.arange(2 * DIM, device="cuda")
        got = cache.buf[page[:, None].expand_as(cols), cols].view(torch.float16)
        self.assert_fp16(got, ref[live], f"pool_write section {section}")
        written = torch.zeros_like(cache.buf, dtype=torch.bool)
        written[page[:, None].expand_as(cols), cols] = True
        self.assertTrue(
            torch.equal(cache.buf[~written], before[~written]), "pool_write stray write"
        )

    def test_pool_write(self):
        self._check_pool_write(None)

    def test_pool_write_mrope(self):
        self._check_pool_write(SECTION)

    def test_pool_write_clamps_members(self):
        # Page-table entries past a row's length can be stale; slots are clamped
        # into the raw buffer instead of read out of bounds.
        ops = _ops()
        cache = _Cache([64], seed=5)
        w = _weights(cache.gen)
        cache.raw.copy_(cache.randn(*cache.raw.shape, dtype=torch.float16))
        n = cache.raw.shape[0]
        members = torch.tensor([[0, 1, n + 5, -3]], device="cuda")
        write = cache.slots(0, [0])
        ops.indexpool_pool_write_sm70(
            cache.raw,
            members,
            torch.tensor([0], device="cuda"),
            write,
            *_pool_args(w),
            cache.buf,
            PAGE,
        )
        clamped = torch.tensor([[0, 1, n - 1, 0]], device="cuda")
        ref = _pool_ref(
            cache.raw[clamped], _pair_positions(torch.zeros(1, device="cuda")), w
        )
        self.assert_fp16(cache.group_keys(0, [0]), ref, "clamped members")

    def _check_decode_after_prefill(self, key_dtype, section):
        """Prefill stores raw keys and pools the groups it completes; decode steps
        then store one token per request and pool each group they complete. Every
        completed group's key, and only those, end up in the cache."""
        ops = _ops()
        prompts = [1, 2, 3, 4, 5, 63, 64, 65, 130, 1001]
        steps = 9
        cache = _Cache([n + steps for n in prompts], seed=6)
        w = _weights(cache.gen)
        before = cache.buf.clone()
        proj = [
            cache.randn(n + steps, DIM + 16, dtype=key_dtype, scale=3.0)
            for n in prompts
        ]
        xyz = None
        if section is not None:
            xyz = torch.randint(0, 9999, (cache.raw.shape[0], 3), generator=cache.gen)
            xyz = xyz.int().cuda()

        def first_pos(slots, toks):
            return xyz[slots] if xyz is not None else toks

        # Prefill: the whole prompt of every request in one batch.
        keys = torch.cat([p[:n, :DIM] for p, n in zip(proj, prompts)])
        loc = torch.cat(
            [cache.slots(b, torch.arange(n)) for b, n in enumerate(prompts)]
        )
        ops.indexpool_store_raw_sm70(keys, w["token_norm"], EPS, loc, cache.raw)
        members = []
        for b, n in enumerate(prompts):
            toks = torch.arange(n // RATIO, device="cuda")[:, None] * RATIO
            members.append(cache.slots(b, toks + torch.arange(RATIO, device="cuda")))
        members = torch.cat(members)
        toks0 = torch.cat(
            [torch.arange(n // RATIO, device="cuda") * RATIO for n in prompts]
        )
        ops.indexpool_pool_write_sm70(
            cache.raw,
            members,
            first_pos(members[:, 0], toks0),
            members[:, 0],
            *_pool_args(w),
            cache.buf,
            PAGE,
            mrope_section=section,
        )
        # Decode steps.
        for s in range(steps):
            toks = [n + s for n in prompts]
            key = torch.stack([p[t] for p, t in zip(proj, toks)])[:, :DIM]
            loc = torch.cat([cache.slots(b, [t]) for b, t in enumerate(toks)])
            seq_lens = torch.tensor(
                [t + 1 for t in toks], dtype=torch.int32, device="cuda"
            )
            _leave_stale_nan()
            ops.indexpool_decode_update_sm70(
                key,
                w["token_norm"],
                EPS,
                loc,
                seq_lens,
                cache.pages,
                cache.raw,
                *_pool_args(w),
                cache.buf,
                PAGE,
                rope_xyz=xyz,
                mrope_section=section,
            )
        written = torch.zeros_like(cache.buf, dtype=torch.bool)
        for b, n in enumerate(prompts):
            total = n + steps
            slots = cache.slots(b, torch.arange(total))
            msg = f"{key_dtype} section {section} request {b}"
            self.assert_fp16(
                cache.raw[slots], _rms(proj[b][:, :DIM].double(), w["token_norm"]), msg
            )
            groups = torch.arange(total // RATIO, device="cuda")
            toks = groups[:, None] * RATIO + torch.arange(RATIO, device="cuda")
            member_slots = cache.slots(b, toks)
            pos = first_pos(member_slots[:, 0], toks[:, 0])
            ref = _pool_ref(cache.raw[member_slots], _pair_positions(pos, section), w)
            self.assert_fp16(cache.group_keys(b, groups), ref, msg)
            written |= cache.key_byte_mask(b, groups)
        self.assertTrue(
            torch.equal(cache.buf[~written], before[~written]), "decode stray write"
        )

    def test_decode_after_prefill(self):
        self._check_decode_after_prefill(torch.bfloat16, None)
        self._check_decode_after_prefill(torch.float16, None)

    def test_decode_after_prefill_mrope(self):
        self._check_decode_after_prefill(torch.bfloat16, SECTION)

    # -- scores -----------------------------------------------------------

    def _score_case(self, lens, seed):
        cache = _Cache(lens, seed=seed)
        for b, n in enumerate(lens):
            cache.write_group_keys(b, cache.randn(n // RATIO, DIM, dtype=torch.float16))
        return cache

    def _rows(self, cache, spans):
        """Query rows: for each (request, first position, count) span."""
        req = torch.cat([torch.full((c,), b) for b, _, c in spans]).cuda()
        pos = torch.cat([torch.arange(p, p + c) for _, p, c in spans]).cuda()
        rows = req.numel()
        q = cache.randn(rows, HEADS, DIM, dtype=torch.float16)
        w = cache.randn(rows, HEADS) * 0.03
        return req, pos, q, w

    def _scores_ref_rows(self, cache, req, q, w, upper, num_groups):
        ref = torch.zeros(q.shape[0], num_groups, dtype=torch.float64, device="cuda")
        scale = torch.zeros_like(ref)
        for b in req.unique().tolist():
            rows = (req == b).nonzero().squeeze(1)
            g = int(upper[rows].max())
            if g == 0:
                continue
            keys = cache.group_keys(b, torch.arange(g))
            for chunk in rows.split(16):
                s, sc = _scores_ref(q[chunk], w[chunk], keys)
                ref[chunk, :g] = s
                scale[chunk, :g] = sc
        return ref, scale

    def _logits(self, q, w, pages, row_batch, pos, cache, num_groups, rpb, hist=None):
        out = torch.full((q.shape[0], num_groups), float("nan"), device="cuda")
        _leave_stale_nan()
        return _ops().indexpool_logits_sm70(
            q,
            w,
            pages,
            row_batch,
            pos,
            cache.buf,
            PAGE,
            num_groups,
            rows_per_block=rpb,
            hist=hist,
            out=out,
        )

    def test_logits_decode_rows(self):
        lens = [5, 37, 300, 4100, 9001, 70000]
        cache = self._score_case(lens, seed=7)
        req = torch.arange(len(lens), device="cuda")
        pos = torch.tensor(lens, device="cuda") - 1
        q, w = self._rows(cache, [(b, n - 1, 1) for b, n in enumerate(lens)])[2:]
        num_groups = cache.pages.shape[1] * GROUPS_PER_PAGE
        upper = _upper(pos, num_groups)
        hist = torch.full((len(lens), 2048), 0, dtype=torch.int32, device="cuda")
        scores = self._logits(q, w, cache.pages, req, pos, cache, num_groups, 1, hist)
        ref, scale = self._scores_ref_rows(cache, req, q, w, upper, num_groups)
        self.assert_scores(scores, ref, scale, upper, "decode rows")
        self.assertTrue(torch.equal(hist, _hist(scores, upper)), "score histogram")

    def test_logits_row_runs(self):
        # Rows of several requests in one batch; runs of one request straddle the
        # 8-row blocks. Scores do not depend on the block shape, and each row's
        # score histogram matches its scores.
        lens = [37, 300, 4100, 9000]
        cache = self._score_case(lens, seed=8)
        spans = [(0, 0, 37), (1, 250, 50), (2, 4000, 100), (3, 8990, 10), (1, 299, 1)]
        req, pos, q, w = self._rows(cache, spans)
        num_groups = 64 * -(-(max(lens) - 1) // RATIO // 64)
        upper = _upper(pos, num_groups)
        ref, scale = self._scores_ref_rows(cache, req, q, w, upper, num_groups)
        first = None
        for rpb in (8, 5, 3, 1):
            for rb_dtype in (torch.int64, torch.int32):
                hist = torch.zeros(q.shape[0], 2048, dtype=torch.int32, device="cuda")
                scores = self._logits(
                    q,
                    w,
                    cache.pages,
                    req.to(rb_dtype),
                    pos.to(rb_dtype),
                    cache,
                    num_groups,
                    rpb,
                    hist if rb_dtype == torch.int64 else None,
                )
                msg = f"rows_per_block {rpb} {rb_dtype}"
                self.assert_scores(scores, ref, scale, upper, msg)
                if rb_dtype == torch.int64:
                    self.assertTrue(
                        torch.equal(hist, _hist(scores, upper)),
                        f"{msg}: score histogram",
                    )
                if first is None:
                    first = scores
                self.assertTrue(
                    torch.equal(scores.nan_to_num(-7.0), first.nan_to_num(-7.0)),
                    f"{msg}: differs from rows_per_block 8",
                )

    def test_logits_verify_rows(self):
        # Target verify: 6 rows per request; the page table repeats each request's
        # row for its 6 rows, and all 6 rows read the keys through the first.
        width = 6
        lens = [20, 1000, 33333]
        cache = self._score_case(lens, seed=9)
        spans = [(b, n - width, width) for b, n in enumerate(lens)]
        req, pos, q, w = self._rows(cache, spans)
        pages = cache.pages.repeat_interleave(width, 0)
        rows = torch.arange(req.numel(), device="cuda")
        num_groups = cache.pages.shape[1] * GROUPS_PER_PAGE
        upper = _upper(pos, num_groups)
        ref, scale = self._scores_ref_rows(cache, req, q, w, upper, num_groups)
        hist = torch.zeros(q.shape[0], 2048, dtype=torch.int32, device="cuda")
        shared = self._logits(
            q, w, pages, rows // width * width, pos, cache, num_groups, width, hist
        )
        self.assert_scores(shared, ref, scale, upper, "verify rows")
        self.assertTrue(
            torch.equal(hist, _hist(shared, upper)), "verify rows: score histogram"
        )
        own = self._logits(q, w, pages, rows, pos, cache, num_groups, 1)
        self.assertTrue(
            torch.equal(shared.nan_to_num(-7.0), own.nan_to_num(-7.0)),
            "verify rows: shared key reads differ from per-row reads",
        )

    # -- selection --------------------------------------------------------

    def _selection_rows(self, gen):
        """(upper, scores) per row, covering the selection paths: take-all rows,
        ties, -0 and +0, negatives, candidates gathered in shared memory at once
        or after more passes, ties too many to gather, and rows past the 65536
        groups the shared-memory output covers."""
        big = 65536

        def rand(n):
            return torch.randn(n, generator=gen)

        def ints(lo, hi, n):
            return torch.randint(lo, hi, (n,), generator=gen).float()

        # 0, -0.25 and -0.5, the zeros half -0 and half +0.
        signed_zero = -ints(0, 3, 30000) * 0.25
        flip = (signed_zero == 0) & (torch.rand(30000, generator=gen) < 0.5)
        signed_zero = torch.where(flip, torch.zeros_like(signed_zero), signed_zero)
        boundary = ints(0, 4, big) * 0.5
        hot = torch.randperm(big, generator=gen)
        boundary[hot[:400]] = 5.0
        boundary[hot[400:700]] = 4.0
        # 100 above, then 411 of 600 adjacent ties: the cut is inside a word.
        dense = rand(4000) * 0.1
        dense[1000:1600] = 1.0
        dense[3000:3100] = 2.0
        # 5800 keys in the threshold bin and 400 above: more than the 6144
        # candidates, so one more pass before they are gathered.
        crowd = -rand(big).abs() - 1
        crowd[hot[:5800]] = 1 + ints(0, 1 << 21, 5800) * 2.0**-23
        crowd[hot[5800:6200]] = 5.0
        # 400 above and about 16k equal at the threshold: no level fits in
        # shared memory, every pass is over the row.
        tied = ints(0, 4, big)
        tied[hot[:400]] = 5.0
        rows = [
            (0, torch.zeros(0)),
            (1, rand(1)),
            (511, rand(511)),
            (512, torch.ones(512)),
            (513, rand(513)),
            (big, ints(0, 4, big)),  # 4 values: one bin of ~16k candidates
            (big, rand(big)),
            (big, 1 + ints(0, 4096, big) * 2.0**-23),  # one bin through all levels
            (20000, 1 + ints(0, 1024, 20000) * 2.0**-23),
            (6000, 1 + ints(0, 1024, 6000) * 2.0**-23),  # refined in shared memory
            (40000, torch.ones(40000)),
            (30000, signed_zero),
            (3000, rand(3000) - 10),
            (big, boundary),  # 400 above, 111 of 300 equal at the threshold
            (4000, dense),
            (3000, signed_zero[:3000]),  # 511 of about 1000 zeros
            (big, 1 + ints(0, 1 << 21, big) * 2.0**-23),  # gathered after one pass
            (big, crowd),
            (big, tied),
            (big + 128, torch.cat([rand(128), boundary])),  # past 65536 groups
            (2000, rand(2000).abs() * 1e30),
            (777, -rand(777).abs() * 1e-30),
        ]
        return rows

    def _selection_case(self, seed):
        gen = torch.Generator().manual_seed(seed)
        rows = self._selection_rows(gen)
        n = len(rows)
        num_groups = 65536 + 128
        logits = torch.full((n, num_groups), float("nan"))
        pos = torch.zeros(n, dtype=torch.int64)
        for r, (u, v) in enumerate(rows):
            logits[r, :u] = v
            pos[r] = u * RATIO + r % RATIO
        pages = torch.randint(1, 1 << 20, (3, 4200), generator=gen, dtype=torch.int32)
        row_batch = torch.arange(n) % 3
        offsets = torch.arange(n, dtype=torch.int64) * 300000 + 17
        return (
            logits.cuda(),
            pos.cuda(),
            pages.cuda(),
            row_batch.cuda(),
            offsets.cuda(),
        )

    def test_select_tie_rule(self):
        ops = _ops()
        logits, pos, pages, row_batch, offsets = self._selection_case(10)
        upper = _upper(pos, logits.shape[1])
        picked = _picked(logits, upper)
        for ragged in (False, True):
            index = _ragged(offsets) if ragged else _paged(pages, row_batch)
            expected = _expand(picked, pos, index)
            for with_hist in (False, True):
                hist = _hist(logits, upper) if with_hist else None
                out = torch.full_like(expected, 0x7F7F7F7F)
                _leave_stale_nan()
                ops.indexpool_topk_expand_sm70(
                    logits,
                    pos,
                    pages,
                    row_batch,
                    offsets if ragged else None,
                    TOPK,
                    K_GROUPS,
                    hist=hist,
                    out=out,
                )
                for r in range(logits.shape[0]):
                    self.assertTrue(
                        torch.equal(out[r], expected[r]),
                        f"ragged {ragged} hist {with_hist} row {r} upper {int(upper[r])}",
                    )

    def test_select_histogram_mismatch(self):
        """A histogram that does not match the scores gives the same index rows as
        the right one: zeros, another row's, every group in the top bin, up to
        100 keys moved out of the threshold bin (one row then gathers more
        candidates than shared memory holds), and keys moved from above the
        threshold bin to the bottom bin."""
        ops = _ops()
        logits, pos, pages, row_batch, _ = self._selection_case(17)
        upper = _upper(pos, logits.shape[1])
        expected = _expand(_picked(logits, upper), pos, _paged(pages, row_batch))
        right = _hist(logits, upper)
        live = torch.arange(logits.shape[1], device="cuda") < upper[:, None]
        bins = torch.where(live, _order_keys(logits) >> 21, -1)
        threshold_bin = bins.sort(1, descending=True).values[:, K_GROUPS - 1]
        moved = right.clone()
        for r in range(logits.shape[0]):
            b = int(threshold_bin[r])
            if int(upper[r]) > K_GROUPS and b > 0:
                m = min(100, int(moved[r, b]))
                moved[r, b] -= m
                moved[r, b - 1] += m
        # Up to 50 keys of the top bin counted in bin 0 instead: the total
        # still matches the row, the count of keys above the threshold does not.
        lifted = right.clone()
        for r in range(logits.shape[0]):
            b = int(bins[r].max())
            if int(upper[r]) > K_GROUPS and b > int(threshold_bin[r]):
                m = min(50, int(lifted[r, b]))
                lifted[r, b] -= m
                lifted[r, 0] += m
        top = torch.zeros_like(right)
        top[:, -1] = upper.int()
        wrong = {
            "zeros": torch.zeros_like(right),
            "another row's": right.roll(1, 0),
            "all in the top bin": top,
            "keys moved out of the threshold bin": moved,
            "keys moved from above the threshold bin": lifted,
        }
        for name, hist in wrong.items():
            kept = hist.clone()
            out = ops.indexpool_topk_expand_sm70(
                logits, pos, pages, row_batch, None, TOPK, K_GROUPS, hist=hist
            )
            bad = (out != expected).any(1).nonzero().flatten().tolist()
            self.assertEqual(bad, [], f"histogram {name}: rows differ")
            self.assertTrue(torch.equal(hist, kept), f"histogram {name}: written")

    def test_select_own_group_past_scored_groups(self):
        """num_groups below a row's own group: the selection takes the scored
        groups only, and the tail is still the row's own group (as in
        hy4_flash_indexpool's expansion)."""
        ops = _ops()
        gen = torch.Generator().manual_seed(18)
        pos = torch.tensor([4100, 8191, 4098, 1500, 1027]).cuda()
        pages = torch.randint(1, 1 << 20, (5, 200), generator=gen, dtype=torch.int32)
        pages = pages.cuda()
        row_batch = torch.arange(5).cuda()
        offsets = (torch.arange(5, dtype=torch.int64) * 300000 + 3).cuda()
        for num_groups in (256, 1024):
            logits = torch.randn(5, num_groups, generator=gen).cuda()
            upper = _upper(pos, num_groups)
            picked = _picked(logits, upper)
            for ragged in (False, True):
                index = _ragged(offsets) if ragged else _paged(pages, row_batch)
                expected = _expand(picked, pos, index)
                for hist in (None, _hist(logits, upper)):
                    out = ops.indexpool_topk_expand_sm70(
                        logits,
                        pos,
                        pages,
                        row_batch,
                        offsets if ragged else None,
                        TOPK,
                        K_GROUPS,
                        hist=hist,
                    )
                    self.assertTrue(
                        torch.equal(out, expected),
                        f"{num_groups} groups, ragged {ragged}, hist {hist is not None}",
                    )

    def test_select_index_dtypes(self):
        ops = _ops()
        logits, pos, pages, row_batch, _ = self._selection_case(11)
        upper = _upper(pos, logits.shape[1])
        expected = _expand(_picked(logits, upper), pos, _paged(pages, row_batch))
        out = ops.indexpool_topk_expand_sm70(
            logits, pos.int(), pages, row_batch.int(), None, TOPK, K_GROUPS
        )
        self.assertTrue(torch.equal(out, expected), "int32 positions and batch rows")
        with self.assertRaisesRegex(RuntimeError, "topk too small"):
            ops.indexpool_topk_expand_sm70(
                logits, pos, pages, row_batch, None, 2044, K_GROUPS
            )

    def test_strided_index_vectors(self):
        """Index vectors given as strided views, and one-entry views with any
        stride, give the same results as packed copies."""
        ops = _ops()

        def views(x):
            # A stride-2 view of x, and its first entry (a size-1 view, stride 2).
            wide = torch.stack([x, torch.full_like(x, -7)], dim=-1)
            return [wide[..., 0], wide[:1, 0]]

        def packed(x):
            return x.clone(memory_format=torch.contiguous_format)

        cache = self._score_case([300, 41, 7], seed=21)
        w = _weights(cache.gen)
        cache.raw.copy_(cache.randn(*cache.raw.shape, dtype=torch.float16))

        key = cache.randn(9, DIM, dtype=torch.bfloat16)
        for v in views(cache.slots(0, torch.arange(100, 109))):
            n = v.numel()
            ref, got = cache.raw.clone(), cache.raw.clone()
            ops.indexpool_store_raw_sm70(key[:n], w["token_norm"], EPS, packed(v), ref)
            ops.indexpool_store_raw_sm70(key[:n], w["token_norm"], EPS, v, got)
            self.assertTrue(torch.equal(got, ref), f"store_raw, {n} rows")

        toks = torch.arange(64, device="cuda").view(16, RATIO)
        members = cache.slots(0, toks)
        wide = torch.cat([members, members], dim=1)[:, :RATIO]
        for m in (wide, wide[:1]):
            g = m.shape[0]
            ref, got = cache.buf.clone(), cache.buf.clone()
            for buf, mm, ws in ((ref, packed(m), packed(m[:, 0])), (got, m, m[:, 0])):
                ops.indexpool_pool_write_sm70(
                    cache.raw, mm, toks[:g, 0], ws, *_pool_args(w), buf, PAGE
                )
            self.assertTrue(torch.equal(got, ref), f"pool_write, {g} groups")

        lens = torch.tensor([300, 41, 7], dtype=torch.int32, device="cuda")
        loc = torch.cat([cache.slots(b, [n - 1]) for b, n in enumerate(lens.tolist())])
        key = cache.randn(3, DIM, dtype=torch.bfloat16)
        for lv, sv in zip(views(loc), views(lens)):
            n = lv.numel()
            results = []
            for slot_v, len_v in ((packed(lv), packed(sv)), (lv, sv)):
                raw, buf = cache.raw.clone(), cache.buf.clone()
                ops.indexpool_decode_update_sm70(
                    key[:n],
                    w["token_norm"],
                    EPS,
                    slot_v,
                    len_v,
                    cache.pages,
                    raw,
                    *_pool_args(w),
                    buf,
                    PAGE,
                )
                results.append((raw, buf))
            self.assertTrue(
                torch.equal(results[0][0], results[1][0])
                and torch.equal(results[0][1], results[1][1]),
                f"decode_update, {n} rows",
            )

        q = cache.randn(5, HEADS, DIM, dtype=torch.float16)
        qw = cache.randn(5, HEADS) * 0.03
        pos = torch.tensor([299, 250, 40, 6, 120], device="cuda")
        req = torch.tensor([0, 0, 1, 2, 0], device="cuda")
        for pv, rv in zip(views(pos), views(req)):
            n = pv.numel()
            outs = []
            for p, r in ((packed(pv), packed(rv)), (pv, rv)):
                out = torch.zeros(n, 128, device="cuda")
                ops.indexpool_logits_sm70(
                    q[:n], qw[:n], cache.pages, r, p, cache.buf, PAGE, 128, out=out
                )
                outs.append(out)
            self.assertTrue(torch.equal(outs[0], outs[1]), f"scores, {n} rows")

        logits, pos, pages, row_batch, offsets = self._selection_case(11)
        for pv, rv, ov in zip(views(pos), views(row_batch), views(offsets)):
            n = pv.numel()
            ref = ops.indexpool_topk_expand_sm70(
                logits[:n], packed(pv), pages, packed(rv), packed(ov), TOPK, K_GROUPS
            )
            got = ops.indexpool_topk_expand_sm70(
                logits[:n], pv, pages, rv, ov, TOPK, K_GROUPS
            )
            self.assertTrue(torch.equal(got, ref), f"selection, {n} rows")

    # -- end to end -------------------------------------------------------

    def _check_selection(
        self,
        out,
        q,
        w,
        req,
        pos,
        pages,
        row_batch,
        cache,
        num_groups,
        rpb,
        hist_used,
        msg,
    ):
        """`out` from indexpool_select_sm70 equals the tie-rule expansion of the
        scores the score kernel gives for the same rows, and those picks rank
        right under fp64 scores."""
        ng = 64 * max(1, -(-num_groups // 64))
        scores = self._logits(q, w, pages, row_batch, pos, cache, ng, rpb)
        upper = _upper(pos, ng)
        picked = _picked(scores, upper)
        expected = _expand(picked, pos, _paged(pages, row_batch))
        bad = (out != expected).any(1).nonzero().flatten().tolist()
        self.assertEqual(
            bad, [], f"{msg}: rows differ from the tie rule (hist {hist_used})"
        )
        ref, scale = self._scores_ref_rows(cache, req, q, w, upper, ng)
        self.assert_ranking(picked, ref, scale, upper, msg)

    def test_end_to_end(self):
        """Chunked prefill, decode steps and a verify step through the model's
        call sequence: raw-key store and pooling, query, scores, selection."""
        ops = _ops()
        prompts = [1, 2, 3, 4, 5, 63, 64, 65, 515, 2051, 9001]
        steps, width, chunk = 3, 6, 1500
        cache = _Cache([n + steps + width for n in prompts], seed=12)
        w8 = _weights(cache.gen)
        total = [n + steps + width for n in prompts]
        proj = [cache.randn(n, DIM, dtype=torch.bfloat16, scale=2.0) for n in total]
        inv = w8["inv_freq"]
        bs = len(prompts)

        def query(rows):
            q_raw = cache.randn(rows, HEADS * DIM, dtype=torch.bfloat16, scale=2.0)
            hw = cache.randn(rows, HEADS, dtype=torch.bfloat16)
            return q_raw, hw

        def pool(groups_by_req, write_zero=None):
            members, first = [], []
            for b, groups in groups_by_req:
                toks = groups[:, None] * RATIO + torch.arange(RATIO, device="cuda")
                members.append(cache.slots(b, toks))
                first.append(toks[:, 0])
            if not members:
                return
            members, first = torch.cat(members), torch.cat(first)
            ops.indexpool_pool_write_sm70(
                cache.raw,
                members,
                first,
                members[:, 0],
                *_pool_args(w8),
                cache.buf,
                PAGE,
            )

        # Chunked prefill: each chunk takes up to `chunk` tokens of each request.
        done = [0] * bs
        while any(d < n for d, n in zip(done, prompts)):
            spans = [(b, done[b], min(chunk, prompts[b] - done[b])) for b in range(bs)]
            spans = [s for s in spans if s[2] > 0]
            req = torch.cat([torch.full((c,), b) for b, _, c in spans]).cuda()
            pos = torch.cat([torch.arange(p, p + c) for _, p, c in spans]).cuda()
            keys = torch.cat([proj[b][p : p + c] for b, p, c in spans])
            loc = torch.cat(
                [cache.slots(b, torch.arange(p, p + c)) for b, p, c in spans]
            )
            ops.indexpool_store_raw_sm70(keys, w8["token_norm"], EPS, loc, cache.raw)
            pool(
                [
                    (b, torch.arange(p // RATIO, (p + c) // RATIO, device="cuda"))
                    for b, p, c in spans
                ]
            )
            q_raw, hw = query(req.numel())
            q, w, _ = ops.indexpool_query_sm70(q_raw, pos, hw, inv, HEADS)
            num_groups = max((max(p + c for _, p, c in spans) - 1) // RATIO, 1)
            # A score budget of 40 rows splits the chunk into several score passes.
            budget = 4 * 64 * -(-num_groups // 64) * 40
            out = ops.indexpool_select_sm70(
                q,
                w,
                pos,
                cache.pages,
                req,
                None,
                cache.buf,
                PAGE,
                num_groups,
                TOPK,
                K_GROUPS,
                8,
                logits_budget=budget,
            )
            self._check_selection(
                out,
                q,
                w,
                req,
                pos,
                cache.pages,
                req,
                cache,
                num_groups,
                8,
                False,
                f"prefill chunk at {done}",
            )
            for b, p, c in spans:
                done[b] = p + c

        # Decode steps; the first one is the first decode after the prefill.
        num_groups = cache.pages.shape[1] * GROUPS_PER_PAGE
        req = torch.arange(bs, device="cuda")
        for s in range(steps):
            toks = [n + s for n in prompts]
            pos = torch.tensor(toks, device="cuda")
            key = torch.stack([p[t] for p, t in zip(proj, toks)])
            loc = torch.cat([cache.slots(b, [t]) for b, t in enumerate(toks)])
            ops.indexpool_decode_update_sm70(
                key,
                w8["token_norm"],
                EPS,
                loc,
                (pos + 1).int(),
                cache.pages,
                cache.raw,
                *_pool_args(w8),
                cache.buf,
                PAGE,
            )
            q_raw, hw = query(bs)
            q, w, hist = ops.indexpool_query_sm70(
                q_raw, pos, hw, inv, HEADS, with_hist=True
            )
            out = ops.indexpool_select_sm70(
                q,
                w,
                pos,
                cache.pages,
                req,
                None,
                cache.buf,
                PAGE,
                num_groups,
                TOPK,
                K_GROUPS,
                1,
                hist=hist,
            )
            self._check_selection(
                out,
                q,
                w,
                req,
                pos,
                cache.pages,
                req,
                cache,
                num_groups,
                1,
                True,
                f"decode step {s}",
            )

        # One verify step of `width` rows per request.
        start = [n + steps for n in prompts]
        req = torch.arange(bs, device="cuda").repeat_interleave(width)
        pos = torch.cat([torch.arange(p, p + width) for p in start]).cuda()
        keys = torch.cat([proj[b][p : p + width] for b, p in enumerate(start)])
        loc = torch.cat(
            [cache.slots(b, torch.arange(p, p + width)) for b, p in enumerate(start)]
        )
        ops.indexpool_store_raw_sm70(keys, w8["token_norm"], EPS, loc, cache.raw)
        pool(
            [
                (b, torch.arange(p // RATIO, (p + width) // RATIO, device="cuda"))
                for b, p in enumerate(start)
            ]
        )
        q_raw, hw = query(bs * width)
        q, w, hist = ops.indexpool_query_sm70(
            q_raw, pos, hw, inv, HEADS, with_hist=True
        )
        pages = cache.pages.repeat_interleave(width, 0)
        first_rows = torch.arange(bs * width, device="cuda") // width * width
        out = ops.indexpool_select_sm70(
            q,
            w,
            pos,
            pages,
            first_rows,
            None,
            cache.buf,
            PAGE,
            num_groups,
            TOPK,
            K_GROUPS,
            width,
            hist=hist,
        )
        self._check_selection(
            out,
            q,
            w,
            req,
            pos,
            pages,
            first_rows,
            cache,
            num_groups,
            width,
            True,
            "verify",
        )

        # Every completed group's key matches the pooled raw keys.
        for b, n in enumerate(total):
            groups = torch.arange(n // RATIO, device="cuda")
            toks = groups[:, None] * RATIO + torch.arange(RATIO, device="cuda")
            ref = _pool_ref(
                cache.raw[cache.slots(b, toks)], _pair_positions(toks[:, 0]), w8
            )
            self.assert_fp16(cache.group_keys(b, groups), ref, f"pooled keys of {b}")

    def test_full_context(self):
        """262144 tokens: decode, verify and the last rows of a prefill chunk."""
        ops = _ops()
        n = 262144
        cache = self._score_case([n], seed=13)
        spans = [(0, n - 1, 1), (0, n - 6, 6), (0, n - 64, 64)]
        req, pos, q, w = self._rows(cache, spans)
        num_groups = cache.pages.shape[1] * GROUPS_PER_PAGE
        for name, rows, rpb in (
            ("decode", slice(0, 1), 1),
            ("verify", slice(1, 7), 6),
            ("prefill", slice(7, 71), 8),
        ):
            r_q, r_w, r_pos, r_req = q[rows], w[rows], pos[rows], req[rows]
            # Decode and verify pass the query kernel's histogram; prefill
            # leaves it to the wrapper.
            hist = None
            if name != "prefill":
                hist = torch.zeros(r_q.shape[0], 2048, dtype=torch.int32, device="cuda")
            out = ops.indexpool_select_sm70(
                r_q,
                r_w,
                r_pos,
                cache.pages,
                r_req,
                None,
                cache.buf,
                PAGE,
                num_groups,
                TOPK,
                K_GROUPS,
                rpb,
                hist=hist,
            )
            self._check_selection(
                out,
                r_q,
                r_w,
                r_req,
                r_pos,
                cache.pages,
                r_req,
                cache,
                num_groups,
                rpb,
                hist is not None,
                f"262144 {name}",
            )

    def _graph_replay(self, lens_a, lens_b, width, seed):
        """Query and selection captured in a CUDA graph at lengths lens_a and
        replayed at lens_b, ``width`` rows per request (rows of one request
        share key reads). The replay equals an eager run, follows the tie rule
        and ranks right under fp64 scores."""
        ops = _ops()
        cache = self._score_case([max(a, b) for a, b in zip(lens_a, lens_b)], seed=seed)
        bs = len(lens_a)
        rows = bs * width
        inv = _inv_freq()
        num_groups = cache.pages.shape[1] * GROUPS_PER_PAGE
        pages = cache.pages.repeat_interleave(width, 0)
        first_rows = torch.arange(rows, device="cuda") // width * width
        req = torch.arange(bs, device="cuda").repeat_interleave(width)

        def positions(lens):
            return torch.cat([torch.arange(n - width, n) for n in lens]).cuda()

        static = dict(
            q_raw=cache.randn(rows, HEADS * DIM, dtype=torch.bfloat16),
            hw=cache.randn(rows, HEADS, dtype=torch.bfloat16),
            pos=positions(lens_a),
        )

        def query():
            return ops.indexpool_query_sm70(
                static["q_raw"], static["pos"], static["hw"], inv, HEADS, with_hist=True
            )

        def run():
            q, w, hist = query()
            out = ops.indexpool_select_sm70(
                q,
                w,
                static["pos"],
                pages,
                first_rows,
                None,
                cache.buf,
                PAGE,
                num_groups,
                TOPK,
                K_GROUPS,
                width,
                hist=hist,
            )
            return q, w, out

        # The warm-up loads the kernels but launches no score kernel, so in a
        # process of its own the capture makes the first launch of a score
        # block shape that needs the shared-memory opt-in past 48 KB.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            query()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            q, w, out = run()
        static["pos"].copy_(positions(lens_b))
        static["q_raw"].copy_(cache.randn(rows, HEADS * DIM, dtype=torch.bfloat16))
        graph.replay()
        torch.cuda.synchronize()
        msg = f"graph replay, {width} rows per request"
        self.assertTrue(torch.equal(out, run()[2]), f"{msg}: differs from eager")
        self._check_selection(
            out,
            q,
            w,
            req,
            static["pos"],
            pages,
            first_rows,
            cache,
            num_groups,
            width,
            True,
            msg,
        )

    def test_decode_graph_replay_follows_live_lengths(self):
        self._graph_replay([4100, 1, 300, 20000], [7, 20000, 4096, 4097], 1, seed=14)

    def test_verify_graph_replay_follows_live_lengths(self):
        # Six rows per request: the 8-row score kernel with its 64 KB histogram.
        self._graph_replay([4100, 7, 300, 20000], [9, 20000, 4096, 4097], 6, seed=16)

    # -- model wiring -----------------------------------------------------

    def test_model_select_rows(self):
        """FlashIndexPool._select_sm70 on decode and target-verify batches built
        the model's way (_RowLayout; the verify page table repeats a request's
        row for each of its rows): index rows follow the tie rule and rank right
        under fp64 scores with each row reading its own request's pages. A
        verify batch with uneven rows per request fails."""
        import types

        from sglang.srt.model_executor.forward_mode import ForwardMode
        from sglang.srt.models.hunyuan_v4_flash import FlashIndexPool, _RowLayout

        ops = _ops()
        lens = [4100, 9, 300, 20000]
        width = 6
        bs = len(lens)
        cache = self._score_case(lens, seed=19)
        inv = _inv_freq()
        pool = types.SimpleNamespace(
            ratio=RATIO,
            num_heads=HEADS,
            inv_freq=inv,
            mrope_section=None,
            index_topk=TOPK,
            group_budget=K_GROUPS,
            _LOGITS_BUDGET=FlashIndexPool._LOGITS_BUDGET,
            _index_buffer=lambda: (cache.buf, PAGE),
        )
        pool._num_groups = types.MethodType(FlashIndexPool._num_groups, pool)

        class Metadata:
            topk_transform_method = None

            def __init__(self, seqlens, pages):
                self.seqlens, self.pages = seqlens, pages

            def get_seqlens_int32(self):
                return self.seqlens

            def get_seqlens_expanded(self):
                return self.seqlens

            def get_page_table_64(self):
                return self.pages

            def get_page_table_1(self):
                return None

        def select(mode, row_lens, pages):
            rows = row_lens.numel()
            batch = types.SimpleNamespace(
                forward_mode=mode,
                seq_lens=torch.tensor(lens, device="cuda"),
                seq_lens_cpu=torch.tensor(lens),
                batch_size=bs,
            )
            layout = _RowLayout(batch, Metadata(row_lens.int(), pages), rows)
            query = cache.randn(rows, HEADS * DIM, dtype=torch.bfloat16, scale=2.0)
            hw = cache.randn(rows, HEADS, dtype=torch.bfloat16)
            out = FlashIndexPool._select_sm70(
                pool, layout, batch, query, layout.positions, hw
            )
            q, w, _ = ops.indexpool_query_sm70(query, layout.positions, hw, inv, HEADS)
            return out, q, w, layout.positions

        num_groups = cache.pages.shape[1] * GROUPS_PER_PAGE
        req = torch.arange(bs, device="cuda")
        out, q, w, pos = select(
            ForwardMode.DECODE, torch.tensor(lens, device="cuda"), cache.pages
        )
        self._check_selection(
            out, q, w, req, pos, cache.pages, req, cache, num_groups, 1, True, "decode"
        )

        row_lens = torch.cat([torch.arange(n - width, n) + 1 for n in lens]).cuda()
        pages = cache.pages.repeat_interleave(width, 0)
        rows = torch.arange(bs * width, device="cuda")
        out, q, w, pos = select(ForwardMode.TARGET_VERIFY, row_lens, pages)
        self._check_selection(
            out,
            q,
            w,
            req.repeat_interleave(width),
            pos,
            pages,
            rows,
            cache,
            num_groups,
            1,
            True,
            "verify",
        )
        with self.assertRaisesRegex(AssertionError, "target verify"):
            select(ForwardMode.TARGET_VERIFY, row_lens[:-1], pages[:-1])

    # -- window layers ----------------------------------------------------

    def test_window_indices(self):
        from sglang.kernels.ops.attention.dsa.hy4_flash_indexpool import window_indices

        gen = torch.Generator().manual_seed(15)
        pos = torch.tensor(
            [0, 1, 63, 64, 510, 511, 512, 513, 5000, 70001, 262143]
        ).cuda()
        n = pos.numel()
        pages = torch.randint(
            1, 1 << 20, (4, 4100), generator=gen, dtype=torch.int32
        ).cuda()
        row_batch = (torch.arange(n) % 4).cuda()
        offsets = (torch.arange(n, dtype=torch.int64) * 300000 + 5).cuda()
        width = 64 * -(-WINDOW // 64)
        for ragged in (False, True):
            index = _ragged(offsets) if ragged else _paged(pages, row_batch)
            first = (pos - WINDOW + 1).clamp_min(0)
            toks = first[:, None] + torch.arange(width, device="cuda")
            live = toks <= pos[:, None]
            rows = torch.arange(n, device="cuda")[:, None]
            expected = torch.where(live, index(rows, toks), -1).int()
            out = window_indices(
                pos, pages, row_batch, offsets if ragged else None, WINDOW, width
            )
            self.assertTrue(torch.equal(out, expected), f"window ragged {ragged}")


if __name__ == "__main__":
    unittest.main()
