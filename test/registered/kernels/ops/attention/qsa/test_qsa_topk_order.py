"""QSA block top-k outside batch-invariant mode (the serving default): the same
blocks in ascending order on every call, ties at the cut-off to the smaller
index. The sparse attention sums the selected tokens in list order, so a
selection whose order or ties vary between calls changes the output of a
repeated greedy request."""

from types import SimpleNamespace

import pytest
import torch

import sglang.srt.layers.attention.qsa.qsa_indexer as qsa_indexer_module
from sglang.srt.layers.attention.qsa.kernel import qsa_fast_topk
from sglang.srt.layers.attention.qsa.qsa_indexer import QSAIndexer
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="nightly", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

BLOCK_TOPK = 512


def _reference(logits, starts, lengths, k):
    out = torch.full((logits.shape[0], k), -1, dtype=torch.int32)
    for r, (s, n) in enumerate(zip(starts.tolist(), lengths.tolist())):
        order = torch.sort(-logits[r, s : s + n].cpu(), stable=True).indices
        out[r, : min(n, k)] = order[:k].sort().values.int()
    return out.cuda()


def _logits(rows, width, ties):
    if ties:
        return torch.randint(0, 8, (rows, width), device="cuda").float()
    return torch.randn(rows, width, device="cuda")


@pytest.mark.parametrize("ties", [False, True])
@pytest.mark.parametrize("width", [2048, 16384, 65536])
def test_prefill_topk_ascending_ties_to_smaller(width, ties):
    torch.manual_seed(0)
    rows = 8
    logits = _logits(rows, width + 64, ties)
    starts = torch.randint(0, 64, (rows,), dtype=torch.int32, device="cuda")
    lengths = torch.randint(BLOCK_TOPK + 1, width + 1, (rows,), dtype=torch.int32, device="cuda")
    lengths[0] = width
    lengths[1] = 300
    expected = _reference(logits, starts, lengths, BLOCK_TOPK)
    for _ in range(5):
        actual = qsa_fast_topk(logits, starts, starts + lengths, BLOCK_TOPK)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("ties", [False, True])
def test_decode_topk_ascending_ties_to_smaller(monkeypatch, ties):
    torch.manual_seed(0)
    rows, width = 8, 32768
    logits = _logits(rows, width, ties)

    def mock_mqa_decode(*args, min_scored_len, group_queries=False):
        assert min_scored_len == BLOCK_TOPK
        return logits

    monkeypatch.setattr(qsa_indexer_module, "qsa_mqa_decode", mock_mqa_decode)
    lengths = torch.randint(BLOCK_TOPK + 1, width + 1, (rows,), dtype=torch.int32, device="cuda")
    expected = _reference(logits, torch.zeros_like(lengths), lengths, BLOCK_TOPK)
    for _ in range(5):
        actual = QSAIndexer.select_decode_tokens(
            SimpleNamespace(block_topk=BLOCK_TOPK),
            q=logits,
            compressed_cache=logits,
            compressed_page_table=lengths,
            compressed_lengths=lengths,
            max_model_len=width * 4,
            query_positions=lengths,
            sequence_lengths=lengths,
            defer_expansion=True,
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
