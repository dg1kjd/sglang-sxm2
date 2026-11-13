"""Qwen3.5/Qwen4-Exp attention: the k-norm output made on the alt stream must survive
until the current stream is done with it, inside a captured CUDA graph.

In capture mode ``_apply_qk_norm`` runs the k-norm on the alt stream. Qwen4-Exp's MTP
draft gets 1-row positions, so MRoPE takes its out-of-place path: it reads the k-norm
output and drops it. Since upstream #39474 the QSA indexer is launched on the alt stream
right after that, without waiting for the current stream. Without ``record_stream`` the
indexer's first allocation got the k-norm's memory and wrote it while RoPE still read
it; MTP acceptance fell from ~2.8 to ~2.4 (gsm8k, t0.6).
"""

import pytest
import torch

from sglang.srt.layers.layernorm import GemmaRMSNorm
from sglang.srt.model_executor.runner_utils.capture_mode import model_capture_mode
from sglang.srt.models.qwen3_5 import Qwen3_5AttentionDecoderLayer

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

HEAD_DIM, ROT, ROWS = 256, 64, 4


class _Layer:
    """What ``_apply_qk_norm`` reads from the layer."""

    _apply_qk_norm = Qwen3_5AttentionDecoderLayer._apply_qk_norm

    def __init__(self, alt_stream):
        self.alt_stream = alt_stream
        self.head_dim = HEAD_DIM
        self.q_norm = GemmaRMSNorm(HEAD_DIM, eps=1e-6).cuda().half()
        self.k_norm = GemmaRMSNorm(HEAD_DIM, eps=1e-6).cuda().half()


def _rope_out_of_place(k):
    # The shape of MRotaryEmbedding.forward_native: slices, then a new tensor.
    rot, rest = k[..., :ROT], k[..., ROT:]
    out = torch.cat((rot * 0.5 - rot.flip(-1) * 0.25, rest), dim=-1)
    for _ in range(8):  # keep the current stream busy, as RoPE and the gate split do
        out = out * 1.0001
    return out


def test_k_norm_output_not_reused_by_next_alt_allocation():
    torch.manual_seed(0)
    layer = _Layer(torch.cuda.Stream())
    qkv = torch.randn(ROWS, 3 * HEAD_DIM, device="cuda", dtype=torch.float16)
    q, k = qkv[:, :HEAD_DIM], qkv[:, HEAD_DIM : 2 * HEAD_DIM]  # strided views

    def step():
        current = torch.cuda.current_stream()
        layer.alt_stream.wait_stream(current)
        q_out, k_out = layer._apply_qk_norm(q, k)
        k_norm_ptr = k_out.data_ptr()
        k_rope = _rope_out_of_place(k_out)
        del q_out, k_out
        with torch.cuda.stream(layer.alt_stream):
            # The indexer's allocations; enough of the k-norm's size to take every
            # freed block of that size on the alt stream. They write before RoPE is done.
            scratch = [
                torch.empty(ROWS, HEAD_DIM, device="cuda", dtype=torch.float16)
                for _ in range(4)
            ]
            for t in scratch:
                t.fill_(7.0)
        current.wait_stream(layer.alt_stream)
        return k_rope, k_norm_ptr, [t.data_ptr() for t in scratch]

    with torch.no_grad():
        expected = _rope_out_of_place(layer.k_norm(k.reshape(-1, HEAD_DIM)))
        stream = torch.cuda.Stream()
        with model_capture_mode(), torch.cuda.stream(stream):
            for _ in range(3):
                step()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with model_capture_mode(), torch.cuda.graph(graph, stream=stream):
            k_rope, k_norm_ptr, scratch_ptrs = step()
        assert k_norm_ptr not in scratch_ptrs, "alt stream reused the k-norm output"
        for _ in range(50):
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(k_rope, expected, rtol=0, atol=0)
