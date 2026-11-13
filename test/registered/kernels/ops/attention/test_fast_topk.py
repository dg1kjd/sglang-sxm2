import pytest
import torch

from sglang.kernels.ops.attention.fast_topk import fast_topk
from sglang.kernels.ops.attention.ordered_topk import ordered_topk
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-small")


def _check_topk_values(score, lengths, indices, topk, row_starts):
    """fast_topk leaves order and tie-breaking unspecified,
    so compare the sorted top-k values rather than index sets."""
    for b in range(score.shape[0]):
        start = int(row_starts[b]) if row_starts is not None else 0
        length = int(lengths[b])
        section = score[b, start : start + length]
        row = indices[b]
        if length <= topk:
            # naive path: identity indices, then -1 fill
            assert torch.equal(
                row[:length].cpu(), torch.arange(length, dtype=torch.int32)
            )
            assert (row[length:] == -1).all()
            continue
        assert (row >= 0).all(), "long rows must fill every slot"
        picked = section[row.long()]
        expected = torch.topk(section, topk).values
        assert torch.equal(
            picked.sort(descending=True).values, expected.sort(descending=True).values
        ), f"row {b}: top-{topk} value multiset mismatch"


@pytest.mark.parametrize("topk", [512, 2048])
@pytest.mark.parametrize(
    "batch,length",
    [
        (1, 4096),
        (7, 3000),
        (33, 32768),
        (128, 2050),
    ],
)
def test_fast_topk_long_rows(topk, batch, length):
    torch.manual_seed(0)
    score = torch.randn(batch, length, dtype=torch.float32, device="cuda")
    lengths = torch.full((batch,), length, dtype=torch.int32, device="cuda")

    indices = fast_topk(score, lengths, topk)
    _check_topk_values(score, lengths, indices, topk, None)


@pytest.mark.parametrize("topk", [512, 2048])
def test_fast_topk_short_and_mixed_rows(topk):
    torch.manual_seed(0)
    max_len = topk + 128
    batch = 8
    score = torch.randn(batch, max_len, dtype=torch.float32, device="cuda")
    # rows shorter than k (naive path), exactly k, and longer than k
    lens = [1, topk // 3, topk - 1, topk, topk + 1, topk + 7, 17, max_len]
    lengths = torch.tensor(lens[:batch], dtype=torch.int32, device="cuda")

    indices = fast_topk(score, lengths, topk)
    _check_topk_values(score, lengths, indices, topk, None)


@pytest.mark.parametrize("topk", [512, 2048])
def test_fast_topk_ragged_with_row_starts(topk):
    torch.manual_seed(0)
    batch, width = 16, 8192
    score = torch.randn(batch, width, dtype=torch.float32, device="cuda")
    row_starts = torch.randint(0, 2048, (batch,), dtype=torch.int32, device="cuda")
    lengths = torch.randint(1, 2048, (batch,), dtype=torch.int32, device="cuda")
    lengths = torch.minimum(lengths, width - row_starts).to(torch.int32)
    # ensure some rows are longer than k
    lengths[0] = min(width - int(row_starts[0]), topk + 100)

    indices = fast_topk(score, lengths, topk, row_starts=row_starts)
    _check_topk_values(score, lengths, indices, topk, row_starts)


@pytest.mark.parametrize("topk", [512, 2048])
def test_fast_topk_row_stride(topk):
    torch.manual_seed(0)
    batch, length = 8, 4096
    base = torch.randn(batch, 2 * length, dtype=torch.float32, device="cuda")
    score = base[:, :length]  # stride(0) == 2*length, stride(1) == 1
    lengths = torch.full((batch,), length, dtype=torch.int32, device="cuda")

    indices = fast_topk(score, lengths, topk)
    _check_topk_values(score, lengths, indices, topk, None)


@pytest.mark.parametrize("topk", [512, 2048])
@pytest.mark.parametrize(
    "fill",
    [
        "binary",  # only 0s and 1s: extreme duplication at the threshold bin
        "few_levels",  # a handful of distinct levels incl. negatives
        "constant",  # whole rows of one value
    ],
)
def test_fast_topk_duplicate_heavy(topk, fill):
    torch.manual_seed(0)
    batch, length = 16, 8192
    if fill == "binary":
        score = torch.randint(0, 2, (batch, length), dtype=torch.float32, device="cuda")
    elif fill == "few_levels":
        levels = torch.tensor([-5.0, -1.0, 0.0, 0.5, 2.0], device="cuda")
        score = levels[torch.randint(0, 5, (batch, length), device="cuda")]
    else:
        score = torch.full((batch, length), 3.25, dtype=torch.float32, device="cuda")
    lengths = torch.full((batch,), length, dtype=torch.int32, device="cuda")

    indices = fast_topk(score, lengths, topk)
    _check_topk_values(score, lengths, indices, topk, None)


@pytest.mark.parametrize("topk", [512, 2048])
def test_fast_topk_negative_and_zero(topk):
    torch.manual_seed(0)
    batch, length = 8, 16384
    score = torch.randn(batch, length, dtype=torch.float32, device="cuda") * 100
    score[:, : length // 3] = 0.0  # long zero prefix
    score[:, length // 3 : length // 2] = -1e30  # very negative block
    lengths = torch.full((batch,), length, dtype=torch.int32, device="cuda")

    indices = fast_topk(score, lengths, topk)
    _check_topk_values(score, lengths, indices, topk, None)


def test_fast_topk_unsupported_k():
    score = torch.randn(2, 4096, dtype=torch.float32, device="cuda")
    lengths = torch.full((2,), 4096, dtype=torch.int32, device="cuda")
    with pytest.raises(RuntimeError, match="topk"):
        fast_topk(score, lengths, 1024)


def _ordered_reference(score, lengths, topk, row_starts=None):
    """Stable order by (score descending, index ascending), cut at topk, then
    sorted by index; -1 past the row length."""
    score = score.cpu()
    expected = torch.full((score.shape[0], topk), -1, dtype=torch.int32)
    for b in range(score.shape[0]):
        start = int(row_starts[b]) if row_starts is not None else 0
        length = int(lengths[b])
        order = torch.argsort(
            score[b, start : start + length], descending=True, stable=True
        )
        count = min(length, topk)
        expected[b, :count] = order[:count].sort().values.to(torch.int32)
    return expected


def _ordered_scores(kind, batch, width):
    if kind == "all_zero":
        return torch.zeros(batch, width, device="cuda")
    score = torch.randn(batch, width, device="cuda")
    if kind == "quarter_steps":
        # about 30 distinct values: thousands of exact ties around the cut-off
        return (score * 4).round() / 4
    if kind == "mostly_zero":
        # ReLU-style scores: the cut-off falls inside a long run of exact zeros
        return torch.relu(score - 2.5)
    if kind == "signed_zero":
        # -0.0 and +0.0 compare equal, so they must tie
        sign = torch.where(torch.rand_like(score) < 0.5, -1.0, 1.0)
        return torch.relu(score - 2.5) * sign
    if kind == "fp16_values":
        return score.half().float()
    if kind == "narrow_range":
        # every score shares its upper 24 bits with thousands of others
        return 1.0 + score.abs() * 1e-5
    if kind == "outside_fp16_range":
        return score * 1e30
    assert kind == "randn"
    return score


_ORDERED_SCORE_KINDS = [
    "randn",
    "quarter_steps",
    "all_zero",
    "mostly_zero",
    "signed_zero",
    "fp16_values",
    "narrow_range",
    "outside_fp16_range",
]


@pytest.mark.parametrize("topk", [512, 2048])
@pytest.mark.parametrize("kind", _ORDERED_SCORE_KINDS)
def test_ordered_topk_matches_reference(topk, kind):
    """Rows shorter than, equal to and longer than k in one batch."""
    torch.manual_seed(0)
    lens = [0, 1, topk - 1, topk, topk + 1, topk + 37, 4 * topk + 3, 40000]
    score = _ordered_scores(kind, len(lens), max(lens))
    lengths = torch.tensor(lens, dtype=torch.int32, device="cuda")

    indices = ordered_topk(score, lengths, topk)
    assert indices.dtype == torch.int32
    assert torch.equal(indices.cpu(), _ordered_reference(score, lengths, topk))


@pytest.mark.parametrize("topk", [512, 2048])
@pytest.mark.parametrize("kind", ["randn", "quarter_steps", "mostly_zero"])
def test_ordered_topk_row_starts_and_stride(topk, kind):
    """Rows start at any offset inside a strided score matrix; the scores before
    the start and past the end are larger than every score of the row."""
    torch.manual_seed(1)
    batch, width = 12, 9000
    base = _ordered_scores(kind, batch, 2 * width)
    score = base[:, :width]
    row_starts = torch.randint(0, 4000, (batch,), dtype=torch.int32, device="cuda")
    lengths = torch.randint(1, 5000, (batch,), dtype=torch.int32, device="cuda")
    lengths[:4] = torch.tensor([topk, topk + 1, topk + 2, topk + 3])
    row_starts[:4] = torch.tensor([1, 2, 3, 5])
    columns = torch.arange(2 * width, device="cuda")[None, :]
    outside = (columns < row_starts[:, None]) | (
        columns >= (row_starts + lengths)[:, None]
    )
    base.masked_fill_(outside, 1e9)

    indices = ordered_topk(score, lengths, topk, row_starts=row_starts)
    expected = _ordered_reference(score, lengths, topk, row_starts)
    assert torch.equal(indices.cpu(), expected)


@pytest.mark.parametrize("kind", ["randn", "quarter_steps", "mostly_zero"])
def test_ordered_topk_long_rows(kind):
    """Rows longer than 65,536 scores are read in more than one pass."""
    torch.manual_seed(2)
    lens = [65535, 65536, 65537, 131072, 200003]
    score = _ordered_scores(kind, len(lens), max(lens))
    lengths = torch.tensor(lens, dtype=torch.int32, device="cuda")

    indices = ordered_topk(score, lengths, 512)
    assert torch.equal(indices.cpu(), _ordered_reference(score, lengths, 512))


@pytest.mark.parametrize("kind", ["randn", "quarter_steps", "mostly_zero"])
def test_ordered_topk_repeated_calls_identical(kind):
    """fast_topk returns the same rows in a different order from call to call."""
    torch.manual_seed(3)
    batch, width = 4, 8192
    score = _ordered_scores(kind, batch, width)
    lengths = torch.full((batch,), width, dtype=torch.int32, device="cuda")

    first = ordered_topk(score, lengths, 512)
    for _ in range(50):
        assert torch.equal(ordered_topk(score, lengths, 512), first)


def test_ordered_topk_independent_of_batch():
    """A row gets the same indices alone and next to other rows."""
    torch.manual_seed(4)
    batch, width = 16, 8192
    score = _ordered_scores("quarter_steps", batch, width)
    lengths = torch.randint(600, width, (batch,), dtype=torch.int32, device="cuda")

    together = ordered_topk(score, lengths, 512)
    for b in range(batch):
        alone = ordered_topk(score[b : b + 1].contiguous(), lengths[b : b + 1], 512)
        assert torch.equal(alone[0], together[b])


def test_ordered_topk_cuda_graph_replay():
    """One captured call, replayed on new scores and lengths in the same buffers."""
    torch.manual_seed(5)
    batch, width = 4, 8192
    score = torch.empty(batch, width, device="cuda")
    lengths = torch.empty(batch, dtype=torch.int32, device="cuda")

    score.copy_(_ordered_scores("randn", batch, width))
    lengths.fill_(width)
    ordered_topk(score, lengths, 512)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = ordered_topk(score, lengths, 512)

    for kind, lens in [
        ("quarter_steps", [width, 513, 512, 100]),
        ("mostly_zero", [4000, width, 1, 2049]),
    ]:
        score.copy_(_ordered_scores(kind, batch, width))
        lengths.copy_(torch.tensor(lens, dtype=torch.int32))
        expected = _ordered_reference(score, lengths, 512)
        for _ in range(3):
            graph.replay()
            assert torch.equal(captured.cpu(), expected)


def test_ordered_topk_nan_and_inf():
    """NaN is ordered by its bits: above +inf with the sign bit clear, below
    -inf with it set. -inf is the smallest number."""
    torch.manual_seed(6)
    width, topk = 4096, 512
    score = torch.randn(3, width, device="cuda")
    lengths = torch.full((3,), width, dtype=torch.int32, device="cuda")
    positive_nan = torch.tensor(0x7FC00000, dtype=torch.int32).view(torch.float32)
    negative_nan = torch.tensor(-0x400000, dtype=torch.int32).view(torch.float32)
    # row 0: 5 positive NaN and 2 +inf are selected first
    score[0, [7, 100, 2000, 3000, 4095]] = positive_nan.item()
    score[0, [8, 9]] = float("inf")
    # row 1: only 300 finite scores; -inf fills the rest in index order and
    # negative NaN is taken last
    score[1] = float("-inf")
    score[1, 1000:1300] = torch.randn(300, device="cuda")
    score[1, :100] = negative_nan.item()
    # row 2: negative NaN and -inf are never selected while finite scores remain
    score[2, ::2] = negative_nan.item()
    score[2, 1::4] = float("-inf")

    indices = ordered_topk(score, lengths, topk).cpu()

    finite = score[0].cpu().nan_to_num(nan=float("inf"))
    assert torch.equal(
        indices[0], _ordered_reference(finite[None], lengths[:1], topk)[0]
    )
    row1 = torch.cat([torch.arange(100, 312), torch.arange(1000, 1300)])
    assert torch.equal(indices[1], row1.to(torch.int32))
    finite = score[2].cpu().nan_to_num(nan=float("-inf"))
    assert torch.equal(
        indices[2], _ordered_reference(finite[None], lengths[:1], topk)[0]
    )


def test_ordered_topk_rejects_topk_below_one():
    """topk = 0 would leave the kernel's cut-off unset and write out of bounds."""
    score = torch.randn(2, 64, device="cuda")
    lengths = torch.full((2,), 64, dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="topk >= 1"):
        ordered_topk(score, lengths, 0)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
