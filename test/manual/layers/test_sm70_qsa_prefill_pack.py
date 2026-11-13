import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="Checks the Triton packer on V100",
)


def _gather(buffer, req_to_token, req_indices, lens, dtype):
    # The index_select + cast the backend used before upstream's packer.
    parts = [
        buffer.index_select(0, req_to_token[req, :n].long())
        for req, n in zip(req_indices.tolist(), lens)
    ]
    return torch.cat(parts).to(dtype)


# The last case takes the packer's compact grid (one long request next to many
# short ones: max_k * bs > 2 * total_k and max_k / 4 * bs >= 4096); the others
# take the per-request grid.
@pytest.mark.parametrize(
    "lens",
    [[100, 700, 37], [1], [2048] + [10] * 7],
)
@torch.no_grad()
def test_pack_matches_gather_on_e5m2_pool(lens):
    from sglang.srt.layers.attention.qsa.sparse_attn import pack_qsa_prefill_kv

    torch.manual_seed(0)
    slots, heads, dim = 32768, 1, 256
    k = torch.randn(slots, heads, dim, device="cuda").to(torch.float8_e5m2)
    v = torch.randn(slots, heads, dim, device="cuda").to(torch.float8_e5m2)
    rows = len(lens) + 2
    req_to_token = torch.stack(
        [torch.randperm(slots, device="cuda")[:4096] for _ in range(rows)]
    ).to(torch.int32)
    req_indices = torch.randperm(rows, device="cuda")[: len(lens)].to(torch.int64)
    cu_k = torch.nn.functional.pad(
        torch.tensor(lens, dtype=torch.int32, device="cuda").cumsum(0), (1, 0)
    )

    packed_k, packed_v = pack_qsa_prefill_kv(
        k=k,
        v=v,
        req_to_token=req_to_token,
        req_indices=req_indices,
        cu_k=cu_k,
        total_k=sum(lens),
        max_k=max(lens),
        output_dtype=torch.float16,
    )
    torch.cuda.synchronize()

    expected_k = _gather(k, req_to_token, req_indices, lens, torch.float16)
    expected_v = _gather(v, req_to_token, req_indices, lens, torch.float16)
    assert packed_k.dtype == torch.float16 and packed_k.shape == expected_k.shape
    assert torch.equal(packed_k, expected_k)
    assert torch.equal(packed_v, expected_v)
