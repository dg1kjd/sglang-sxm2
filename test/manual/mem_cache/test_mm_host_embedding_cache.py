"""Multimodal embedding cache in pinned host memory (SGLANG_USE_HOST_VLM_CACHE).

Entries must give the same chunks as the GPU cache, hold no GPU memory while
cached, and leave the KV reservation at zero.
"""

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.managers import mm_schedule
from sglang.srt.managers.req_types import MultimodalDataItem
from sglang.srt.mem_cache.kv_cache_configurator import mm_runtime_reservation_gb
from sglang.srt.mem_cache.multimodal_cache import EmbeddingResult
from sglang.srt.multimodal.modality import Modality
from sglang.srt.runtime_context import get_context, get_parallel

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Host-cache copies need a GPU"
)

HIDDEN = 4096
ITEM_OFFSETS = [(2, 301), (310, 1309), (1320, 1339)]
TOTAL_LEN = 1400
CHUNKS = [(0, 512), (512, 512), (1024, 376)]
DEVICE = torch.device("cuda")


@pytest.fixture(autouse=True)
def single_process_runtime_context():
    override = get_context().override_server_args(tp_size=1)
    override.install()
    try:
        with get_parallel().override(attn_tp_rank=0, attn_tp_size=1, tp_size=1):
            yield
    finally:
        override.restore()


def _items():
    return [
        MultimodalDataItem(
            modality=Modality.IMAGE,
            hash=2000 + i,
            feature=torch.zeros(1),
            offsets=[offset],
        )
        for i, offset in enumerate(ITEM_OFFSETS)
    ]


def _embedding(item):
    start, end = item.offsets[0]
    gen = torch.Generator().manual_seed(item.hash)
    return torch.randn(end - start + 1, HIDDEN, generator=gen).half().to(DEVICE)


class _Encoder:
    def __init__(self):
        self.calls = 0

    def __call__(self, items):
        self.calls += 1
        return torch.cat([_embedding(item) for item in items], dim=0)


def _chunks(host, path):
    mm_schedule.init_mm_embedding_cache(1 << 30, host=host)
    encoder, items = _Encoder(), _items()
    out = []
    for prefix_len, extend_len in CHUNKS:
        if path == "by_item":
            parts = mm_schedule._get_chunked_embedding_by_item(
                encoder, items, prefix_len, extend_len, DEVICE
            )
            out.append(torch.cat(parts) if parts else None)
        else:
            chunk, _ = mm_schedule._get_chunked_embedding_full(
                encoder,
                items,
                ITEM_OFFSETS,
                prefix_len,
                extend_len,
                torch.zeros(TOTAL_LEN, dtype=torch.long, device=DEVICE),
                DEVICE,
            )
            out.append(chunk)
    torch.cuda.synchronize()
    return out, encoder.calls


@pytest.mark.parametrize("path", ["by_item", "full"])
def test_host_cache_gives_the_same_chunks(path):
    on_gpu, gpu_calls = _chunks(False, path)
    on_host, host_calls = _chunks(True, path)
    # Cache hits on later chunks, not re-encodes, in both modes.
    assert host_calls == gpu_calls
    assert host_calls == (1 if path == "full" else 2)
    for a, b in zip(on_gpu, on_host):
        assert b.device.type == "cuda"
        assert torch.equal(a, b)


def test_host_entries_hold_no_gpu_memory():
    mm_schedule.init_mm_embedding_cache(1 << 30, host=True)
    cache = mm_schedule.embedding_cache
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    gpu = torch.randn(8000, HIDDEN, device=DEVICE).half()
    assert cache.set(b"image", EmbeddingResult(embedding=gpu))
    stored = cache.mm_cache[b"image"].embedding
    assert stored.device.type == "cpu" and stored.is_pinned()
    assert cache.current_size == gpu.numel() * gpu.element_size()
    hit = cache.get_single(b"image").embedding
    assert hit.device == gpu.device and torch.equal(hit, gpu)
    del gpu, hit
    assert torch.cuda.memory_allocated() == before
    assert cache.free(b"image", None) and cache.current_size == 0


def test_host_cache_reserves_no_kv_memory():
    with envs.SGLANG_VLM_CACHE_SIZE_MB.override(512):
        with envs.SGLANG_USE_HOST_VLM_CACHE.override(True):
            assert (
                mm_runtime_reservation_gb(is_multimodal=True, mm_feature_transport=None)
                == 0
            )
        with envs.SGLANG_USE_HOST_VLM_CACHE.override(False):
            assert mm_runtime_reservation_gb(
                is_multimodal=True, mm_feature_transport=None
            ) == pytest.approx(0.5)
