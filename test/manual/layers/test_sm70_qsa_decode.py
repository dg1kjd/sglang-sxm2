import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="Native QSA decode requires V100",
)


@pytest.mark.parametrize("batch", [1, 2, 4])
@pytest.mark.parametrize("topk", [2048, 2051])
@pytest.mark.parametrize("kv_dtype", [torch.float8_e5m2, torch.float16])
def test_selected_cache_attention_and_changing_graph_inputs(batch, topk, kv_dtype):
    from sglang.srt.layers.attention.tilelang_fa_v100._decode_cuda import (
        sm70_cuda_qsa_decode,
    )

    torch.manual_seed(31 + batch)
    pool = 70032
    q = torch.randn(batch, 6, 256, device="cuda", dtype=torch.float16)
    k = torch.randn(pool, 1, 256, device="cuda").to(kv_dtype)
    v = torch.randn(pool, 1, 256, device="cuda").to(kv_dtype)
    table = torch.stack(
        [torch.randperm(pool, device="cuda", dtype=torch.int32) for _ in range(2)]
    )
    requests = torch.zeros(batch, device="cuda", dtype=torch.int32)
    indices = torch.arange(topk, device="cuda", dtype=torch.int32).repeat(batch, 1)
    lengths = torch.full((batch,), 25000, device="cuda", dtype=torch.int32)

    def run():
        return sm70_cuda_qsa_decode(q, k, v, table, requests, indices, lengths, 0.0625)

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run()

    for iteration, length in enumerate([1, 31, 32, 33, 1000, 2048, 25000, 70000]):
        q.normal_()
        lengths.copy_(torch.arange(batch, device="cuda", dtype=torch.int32) + length)
        requests.copy_(
            (torch.arange(batch, device="cuda", dtype=torch.int32) + iteration) % 2
        )
        # Both the physical cache mapping and selected logical positions change.
        table.copy_(table.roll(17, dims=1))
        indices.fill_(-1)
        expected = []
        for row in range(batch):
            count = min(length + row, topk)
            selected = torch.randperm(length + row, device="cuda")[:count].sort().values
            if count > 4:
                # Include masked selections inside active splits, not only padding.
                selected[3::37] = -1
            indices[row, :count].copy_(selected)
            valid = selected >= 0
            slots = table[(row + iteration) % 2, selected[valid]]
            keys, values = k[slots, 0].float(), v[slots, 0].float()
            scores = (q[row].float() @ keys.T) * 0.0625
            expected.append(scores.softmax(-1) @ values)
        graph.replay()
        reference = torch.stack(expected)
        torch.testing.assert_close(output.float(), reference, rtol=0.005, atol=0.001)
        assert (output.float() - reference).norm() / reference.norm() < 0.001


@pytest.mark.parametrize(
    "prefix,rows,topk", [(0, 33, 31), (25000, 8, 2048), (262000, 9, 2051)]
)
@pytest.mark.parametrize("kv_dtype", [torch.float8_e5m2, torch.float16])
def test_chunk_prefill_causality_and_cache_mapping(prefix, rows, topk, kv_dtype):
    from sglang.srt.layers.attention.tilelang_fa_v100._decode_cuda import (
        sm70_cuda_qsa_prefill,
    )

    torch.manual_seed(41)
    length = prefix + rows
    q = torch.randn(rows, 6, 256, device="cuda", dtype=torch.float16)
    k = torch.randn(length, 1, 256, device="cuda").to(kv_dtype)
    v = torch.randn_like(k.float()).to(kv_dtype)
    table = torch.randperm(length, device="cuda", dtype=torch.int32)[None]
    req = torch.zeros(1, device="cuda", dtype=torch.int32)
    lens = torch.tensor([length], device="cuda", dtype=torch.int32)
    indices = torch.full((rows, topk), -1, device="cuda", dtype=torch.int32)
    expected = []
    for row in range(rows):
        visible = prefix + row + 1
        count = min(topk, visible)
        selected = torch.randperm(visible, device="cuda")[:count].sort().values
        if count > 4:
            selected[3::37] = -1
        indices[row, :count] = selected.int()
        slots = table[0, selected[selected >= 0]].long()
        scores = q[row].float() @ k[slots, 0].float().T * 0.0625
        expected.append(scores.softmax(-1) @ v[slots, 0].float())
    out = sm70_cuda_qsa_prefill(q, k, v, table, req, indices, lens, 0.0625)
    ref = torch.stack(expected)
    torch.testing.assert_close(out.float(), ref, rtol=0.005, atol=0.001)
    assert (out.float() - ref).norm() / ref.norm() < 0.001


def test_e5m2_values_in_fp16_have_identical_attention():
    from sglang.srt.layers.attention.tilelang_fa_v100._decode_cuda import (
        sm70_cuda_qsa_decode,
        sm70_cuda_qsa_prefill,
    )

    torch.manual_seed(72)
    rows, length, topk = 4, 4096, 2048
    q = torch.randn(rows, 6, 256, device="cuda", dtype=torch.float16)
    k = torch.randn(length, 1, 256, device="cuda").to(torch.float8_e5m2)
    v = torch.randn_like(k.float()).to(torch.float8_e5m2)
    table = torch.randperm(length, device="cuda", dtype=torch.int32)[None]
    indices = torch.arange(topk, device="cuda", dtype=torch.int32).repeat(rows, 1)
    for fn, requests in [(sm70_cuda_qsa_prefill, 1), (sm70_cuda_qsa_decode, rows)]:
        req = torch.zeros(requests, device="cuda", dtype=torch.int32)
        lens = torch.full((requests,), length, device="cuda", dtype=torch.int32)
        a = fn(q, k, v, table, req, indices, lens, 0.0625)
        b = fn(q, k.half(), v.half(), table, req, indices, lens, 0.0625)
        assert torch.equal(a, b)


@pytest.mark.parametrize("page_size", [4, 16])
def test_indexer_grouped_pages_match_independent_rows(page_size):
    from sglang.srt.layers.attention.qsa.mqa import torch_qsa_mqa_decode
    from sglang.srt.layers.attention.tilelang_fa_v100._decode_cuda import (
        sm70_cuda_qsa_indexer_decode,
    )

    torch.manual_seed(210)
    rows, pages = 9, 155
    device = "cuda"
    q = torch.randn(rows, 4, 128, device=device, dtype=torch.float16)
    k = torch.randn(pages, page_size, 1, 128, device=device, dtype=torch.float16)
    table = torch.randperm(pages, device=device, dtype=torch.int32).repeat(rows, 1)
    table[2] = table[0].roll(3)  # different physical pages in the same CTA
    table[3, -1] = table[3, 0]  # first pages shared, final page diverges
    width = pages * page_size
    lengths = torch.tensor(
        [width, width - 1, width - 3, width, 0, 1, 512, 513, width - 2],
        device=device,
        dtype=torch.int32,
    )
    scale = 128**0.5

    def run():
        return sm70_cuda_qsa_indexer_decode(
            q, k, table, lengths, width, scale, group_queries=True
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        grouped = run()
    for i in range(2):
        if i:
            table.copy_(table.roll(9, dims=1))
            lengths.copy_(lengths.roll(3))
            q.normal_()
        graph.replay()
        valid = torch.arange(width, device=device)[None, :] < lengths[:, None]
        separate = sm70_cuda_qsa_indexer_decode(q, k, table, lengths, width, scale)
        ref = torch_qsa_mqa_decode(q, k, table, lengths, width, scale)
        torch.testing.assert_close(grouped[valid], separate[valid], rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(grouped[valid], ref[valid], rtol=2e-5, atol=2e-5)

        from sglang.kernels.ops.attention.ordered_topk import ordered_topk

        short_rows_skipped = sm70_cuda_qsa_indexer_decode(
            q, k, table, lengths, width, scale, group_queries=True, min_scored_len=512
        )
        assert torch.equal(
            ordered_topk(short_rows_skipped, lengths, topk=512, row_starts=None),
            ordered_topk(separate, lengths, topk=512, row_starts=None),
        )


@pytest.mark.parametrize("kv_dtype", [torch.float8_e5m2, torch.float16])
def test_decode_resolves_compressed_indices_in_selection_order(kv_dtype):
    from sglang.srt.layers.attention.qsa.kernel import expand_qsa_block_indices
    from sglang.srt.layers.attention.tilelang_fa_v100._decode_cuda import (
        sm70_cuda_qsa_decode,
    )

    torch.manual_seed(211)
    device = "cuda"
    pool = 12_000
    lengths = torch.tensor(
        [0, 1, 2, 3, 4, 31, 32, 33, 2047, 2048, 2049, 2051, 4098, 11003],
        dtype=torch.int32,
        device=device,
    )
    rows = lengths.numel()
    q = torch.randn(rows, 6, 256, device=device, dtype=torch.float16)
    k = torch.randn(pool, 1, 256, device=device).to(kv_dtype)
    v = torch.randn_like(k.float()).to(kv_dtype)
    table = torch.stack(
        [torch.randperm(pool, device=device, dtype=torch.int32) for _ in range(3)]
    )
    reqs = torch.arange(rows, device=device, dtype=torch.int32) % 3
    positions = lengths - 1
    blocks = torch.full((rows, 512), -1, dtype=torch.int32, device=device)
    for row, length in enumerate(lengths.tolist()):
        count = length // 4
        blocks[row, : min(count, 512)] = torch.randperm(count, device=device)[:512]
    expanded = expand_qsa_block_indices(blocks, positions, lengths, 4, 2048)
    expected = sm70_cuda_qsa_decode(q, k, v, table, reqs, expanded, lengths, 0.0625)
    actual = sm70_cuda_qsa_decode(
        q, k, v, table, reqs, blocks, lengths, 0.0625, query_positions=positions
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_indexer_grouping_off_in_batch_invariant_mode():
    from sglang.srt.batch_invariant_ops import (
        disable_batch_invariant_mode,
        enable_batch_invariant_mode,
    )
    from sglang.srt.layers.attention.qsa.mqa import qsa_mqa_decode

    torch.manual_seed(212)
    rows, pages, page_size = 4, 64, 4
    q = torch.randn(rows, 4, 128, device="cuda", dtype=torch.float16)
    k = torch.randn(pages, page_size, 1, 128, device="cuda", dtype=torch.float16)
    table = torch.randperm(pages, device="cuda", dtype=torch.int32).repeat(rows, 1)
    width = pages * page_size
    lengths = torch.full((rows,), width, device="cuda", dtype=torch.int32)

    def run(group_queries):
        return qsa_mqa_decode(q, k, table, lengths, width, group_queries=group_queries)

    per_row = run(False)
    # The tensor-core scorer rounds differently, so a row's scores would depend
    # on whether its neighbours share its pages.
    assert not torch.equal(run(True), per_row)
    enable_batch_invariant_mode()
    try:
        assert torch.equal(run(True), per_row)
    finally:
        disable_batch_invariant_mode()
