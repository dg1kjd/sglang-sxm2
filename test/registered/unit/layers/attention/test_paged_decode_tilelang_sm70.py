import math
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (7, 0),
    reason="The grouped TileLang decoder requires an NVIDIA V100",
)


def _leave_stale_nan():
    # Volta's L1 shares SRAM with shared memory, so this also seeds shared memory.
    junk = torch.full((64 << 20,), float("nan"), device="cuda")
    junk.sum()
    del junk


# stale-nan: the last tile of a split is partial, and a kernel that reads shared
# memory past it returns NaN when an earlier kernel left NaN there.
@pytest.mark.parametrize("stale_nan", [False, True], ids=["fresh", "stale-nan"])
@pytest.mark.parametrize("cache_dtype", [torch.float16, torch.float8_e5m2])
def test_grouped_decode_matches_paged_reference(cache_dtype, stale_nan):
    from sglang.srt.layers.attention.tilelang_fa_v100 import (
        grouped_decode_forward,
    )

    torch.manual_seed(7)
    tokens = 257
    page_size = 16
    pages = math.ceil(tokens / page_size)
    heads = 6
    dim = 256
    k_scale = 0.75
    v_scale = 1.25

    q = torch.randn(1, heads, dim, device="cuda", dtype=torch.float16)
    logical_k = torch.randn(
        pages,
        page_size,
        1,
        dim,
        device="cuda",
        dtype=torch.float16,
    ).to(cache_dtype)
    logical_v = torch.randn_like(logical_k.to(torch.float16)).to(cache_dtype)
    permutation = torch.randperm(pages, device="cuda")
    physical_k = torch.empty_like(logical_k)
    physical_v = torch.empty_like(logical_v)
    physical_k[permutation] = logical_k
    physical_v[permutation] = logical_v
    page_table = permutation.to(torch.int32).view(1, -1)
    seq_lens = torch.tensor([tokens], device="cuda", dtype=torch.int32)

    if stale_nan:
        _leave_stale_nan()
    actual = grouped_decode_forward(
        q,
        physical_k,
        physical_v,
        page_table,
        seq_lens,
        softmax_scale=dim**-0.5,
        k_scale=k_scale,
        v_scale=v_scale,
    )

    dense_k = logical_k.flatten(0, 1)[:tokens, 0].float() * k_scale
    dense_v = logical_v.flatten(0, 1)[:tokens, 0].float() * v_scale
    scores = torch.einsum("bhd,nd->bhn", q.float(), dense_k) * dim**-0.5
    expected = torch.softmax(scores, dim=-1) @ dense_v

    torch.testing.assert_close(actual.float(), expected, atol=2e-3, rtol=2e-3)


def _decode_backend(*, fp16_grouped_decode_enabled, triton_out, k_cache, v_cache):
    from sglang.srt.layers.attention.flash_attn_v100_backend import (
        FlashAttnV100Backend,
    )

    backend = FlashAttnV100Backend.__new__(FlashAttnV100Backend)
    backend.page_size = 16
    backend._grouped_decode_enabled = True
    backend._fp16_grouped_decode_enabled = fp16_grouped_decode_enabled
    backend._triton = SimpleNamespace(
        forward_metadata=None,
        forward_decode=lambda *args, **kwargs: triton_out,
    )
    backend.token_to_kv_pool = SimpleNamespace(
        get_kv_buffer=lambda _layer_id: (k_cache, v_cache)
    )
    return backend


def _decode_layer(heads, **overrides):
    fields = dict(
        layer_id=0,
        tp_q_head_num=heads,
        tp_k_head_num=1,
        qk_head_dim=256,
        v_head_dim=256,
        scaling=256**-0.5,
        k_scale=None,
        v_scale=None,
        logit_cap=0.0,
        xai_temperature_len=-1,
        sliding_window_size=-1,
        is_cross_attention=False,
        use_irope=False,
    )
    return SimpleNamespace(**(fields | overrides))


def test_group4_fp16_decode_uses_native_kernel():
    torch.manual_seed(3)
    lens = [1, 37, 1000]
    # The MHA pool's NHD buffers: one [tp_k, 256] row per token slot.
    k_cache = torch.randn(1100, 1, 256, device="cuda", dtype=torch.float16)
    v_cache = torch.randn_like(k_cache)
    slots = torch.randperm(1100, device="cuda")[: sum(lens)]
    kv_indptr = torch.tensor([0, 1, 38, 1038], device="cuda", dtype=torch.int32)
    q = torch.randn(len(lens), 4 * 256, device="cuda", dtype=torch.float16)

    backend = _decode_backend(
        fp16_grouped_decode_enabled=True,
        triton_out=None,
        k_cache=k_cache,
        v_cache=v_cache,
    )
    backend._triton.forward_metadata = SimpleNamespace(
        kv_indptr=kv_indptr, kv_indices=slots.to(torch.int64)
    )
    out = backend.forward_decode(
        q, None, None, _decode_layer(4), SimpleNamespace(), save_kv_cache=False
    )

    expected = []
    for b in range(len(lens)):
        rows = slots[kv_indptr[b] : kv_indptr[b + 1]]
        scores = q[b].view(4, 256).float() @ k_cache[rows, 0].float().T * 256**-0.5
        expected.append((scores.softmax(-1) @ v_cache[rows, 0].float()).flatten())
    torch.testing.assert_close(out.float(), torch.stack(expected), atol=2e-3, rtol=0)


# Each case changes one input of the group-4 case above, which the native kernel takes.
_TRITON_CASES = {
    "group6": dict(heads=6),
    "sliding-window": dict(layer=dict(sliding_window_size=128)),
    "logit-cap": dict(layer=dict(logit_cap=30.0)),
    "sinks": dict(kwargs=dict(sinks=torch.zeros(4))),
    "score-mod": dict(kwargs=dict(score_mod=lambda score, *_: score)),
    "disabled": dict(enabled=False),
    "fp8-kv": dict(cache_dtype=torch.float8_e5m2),
    # SGLANG_USE_HND_KVCACHE pools are [pages, heads, page, dim].
    "hnd-layout": dict(cache_shape=(70, 1, 16, 256)),
    # The kernel launches batch + 1 chunks or more, at most 1024.
    "batch-above-max": dict(batch=1024),
}


@pytest.mark.parametrize("case", _TRITON_CASES.values(), ids=_TRITON_CASES.keys())
def test_other_decode_inputs_keep_triton(case):
    triton_out = object()
    heads = case.get("heads", 4)
    cache = torch.zeros(
        case.get("cache_shape", (1100, 1, 256)), device="cuda", dtype=torch.float16
    ).to(case.get("cache_dtype", torch.float16))
    backend = _decode_backend(
        fp16_grouped_decode_enabled=case.get("enabled", True),
        triton_out=triton_out,
        k_cache=cache,
        v_cache=cache,
    )
    q = torch.zeros(
        case.get("batch", 3), heads * 256, device="cuda", dtype=torch.float16
    )
    layer = _decode_layer(heads, **case.get("layer", {}))

    out = backend.forward_decode(
        q, None, None, layer, SimpleNamespace(), **case.get("kwargs", {})
    )

    assert out is triton_out
