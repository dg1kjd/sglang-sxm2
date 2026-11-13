import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="The TileLang vision attention requires a V100",
)


def _reference(q, k, v, lengths, scale):
    outputs, start = [], 0
    for length in lengths:
        qs, ks, vs = (
            t[start : start + length].float().transpose(0, 1) for t in (q, k, v)
        )
        probs = torch.softmax(qs @ ks.transpose(1, 2) * scale, dim=-1)
        outputs.append((probs @ vs).transpose(0, 1))
        start += length
    return torch.cat(outputs)


@pytest.mark.parametrize("head_dim", [64, 72, 128])
@pytest.mark.parametrize("lengths", [[1], [333], [64, 1, 700, 257], [3072, 3072]])
def test_matches_fp32_reference(head_dim, lengths):
    from sglang.srt.layers.attention import vision

    torch.manual_seed(7)
    heads, total = 16, sum(lengths)
    q, k, v = (
        torch.randn(total, heads, head_dim, device="cuda", dtype=torch.float16)
        for _ in range(3)
    )
    cu_seqlens = torch.tensor([0] + lengths, device="cuda").cumsum(0).int()
    backend = vision.VisionTileLangV100Attention(num_heads=heads, head_dim=head_dim)
    assert backend._kernel_for(q, k, v) is not None

    output = backend(q, k, v, cu_seqlens, bsz=len(lengths), seq_len=max(lengths))

    reference = _reference(q, k, v, lengths, head_dim**-0.5)
    assert output.shape == q.shape and output.dtype == torch.float16
    assert (output.float() - reference).norm() / reference.norm() < 1e-3
    torch.testing.assert_close(output.float(), reference, rtol=0.01, atol=0.004)


def test_strided_inputs_and_explicit_scale():
    from sglang.srt.layers.attention import vision

    torch.manual_seed(8)
    heads, head_dim, total = 16, 72, 500
    packed = torch.randn(total, 3, heads, head_dim, device="cuda", dtype=torch.float16)
    q, k, v = packed.unbind(1)
    assert not q.is_contiguous()
    cu_seqlens = torch.tensor([0, 200, 500], device="cuda", dtype=torch.int32)
    backend = vision.VisionTileLangV100Attention(num_heads=heads, head_dim=head_dim)

    output = backend(q, k, v, cu_seqlens, bsz=2, seq_len=300, softmax_scale=0.05)

    reference = _reference(q, k, v, [200, 300], 0.05)
    assert (output.float() - reference).norm() / reference.norm() < 1e-3


def test_other_dtypes_use_the_triton_backend():
    from sglang.srt.layers.attention import vision

    q = torch.randn(32, 16, 72, device="cuda", dtype=torch.float32)
    backend = vision.VisionTileLangV100Attention(num_heads=16, head_dim=72)
    assert backend._kernel_for(q, q, q) is None
    cu_seqlens = torch.tensor([0, 32], device="cuda", dtype=torch.int32)

    output = backend(q, q, q, cu_seqlens, bsz=1, seq_len=32)

    reference = _reference(q, q, q, [32], 72**-0.5)
    torch.testing.assert_close(output, reference, rtol=0.01, atol=0.002)
