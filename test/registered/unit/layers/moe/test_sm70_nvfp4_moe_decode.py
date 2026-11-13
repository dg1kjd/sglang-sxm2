import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="The specialized NVFP4 MoE decode kernel requires an NVIDIA V100",
)


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_sm70_topk10_softmax_matches_torch(batch_size):
    from sglang.kernels.ops.moe.sm70_nvfp4_moe_decode import sm70_topk10_softmax

    torch.manual_seed(100 + batch_size)
    logits = torch.randn(batch_size, 512, dtype=torch.float16, device="cuda")
    weights, ids = sm70_topk10_softmax(logits)

    probabilities = torch.softmax(logits.float(), dim=-1)
    _, reference_ids = torch.topk(probabilities, 10, dim=-1)
    selected = torch.gather(probabilities, 1, ids.long())
    reference_weights = selected / selected.sum(dim=-1, keepdim=True)

    torch.testing.assert_close(
        torch.sort(ids).values,
        torch.sort(reference_ids.int()).values,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(weights, reference_weights, rtol=2e-6, atol=3e-8)


def test_sm70_nvfp4_moe_four_rows_match_individual_rows():
    from sglang.kernels.ops.moe.sm70_nvfp4_moe_decode import sm70_nvfp4_moe_decode

    torch.manual_seed(11)
    # Ten experts are sufficient for this route-layout test. The serving path
    # supplies all 512; keeping the fixture small avoids a >300 MB allocation.
    w13 = torch.empty((10, 160, 640), dtype=torch.int32, device="cuda").random_()
    w2 = torch.empty((10, 10, 5120), dtype=torch.int32, device="cuda").random_()
    w13_scales = torch.full((10, 160, 320), 0x70, dtype=torch.uint8, device="cuda")
    w2_scales = torch.full((10, 10, 2560), 0x70, dtype=torch.uint8, device="cuda")
    w13_global = torch.full((10,), 100.0, dtype=torch.float32, device="cuda")
    w2_global = torch.full((10,), 100.0, dtype=torch.float32, device="cuda")
    hidden_states = torch.randn((4, 2560), dtype=torch.float16, device="cuda")
    original_input = hidden_states.clone()
    ids = torch.arange(10, dtype=torch.int32, device="cuda").repeat(4, 1)
    weights = torch.softmax(torch.randn((4, 10), device="cuda"), dim=-1)

    batched = sm70_nvfp4_moe_decode(
        hidden_states,
        w13,
        w2,
        w13_scales,
        w2_scales,
        w13_global,
        w2_global,
        ids.flatten(),
        weights.flatten(),
    )
    individual = torch.cat(
        [
            sm70_nvfp4_moe_decode(
                hidden_states[row : row + 1],
                w13,
                w2,
                w13_scales,
                w2_scales,
                w13_global,
                w2_global,
                ids[row],
                weights[row],
            )
            for row in range(4)
        ]
    )

    assert torch.isfinite(batched).all()
    assert batched.data_ptr() != hidden_states.data_ptr()
    torch.testing.assert_close(hidden_states, original_input, rtol=0, atol=0)
    assert batched.abs().max() > 0
    torch.testing.assert_close(batched, individual, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_sm70_topk10_softmax_ties_and_signed_zero(dtype):
    from sglang.kernels.ops.moe.sm70_nvfp4_moe_decode import sm70_topk10_softmax

    logits = torch.zeros((4, 512), device="cuda", dtype=dtype)
    logits[1, ::2] = -0.0
    logits[2].fill_(-5)
    logits[2, [19, 31, 57]] = 4
    logits[3] = torch.arange(512, device="cuda", dtype=dtype) % 7
    weights, ids = sm70_topk10_softmax(logits)
    expected_ids = torch.argsort(logits.float(), descending=True, stable=True)[:, :10]
    expected_weights = torch.softmax(logits.float().gather(1, expected_ids), dim=1)
    torch.testing.assert_close(ids.long(), expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(weights, expected_weights, rtol=2e-6, atol=3e-8)


@pytest.mark.parametrize("batch_size", [1, 4])
def test_sm70_nvfp4_moe_matches_dequantized_weights(batch_size):
    # Load the quantization package before gptq_kernels: imported first,
    # gptq_kernels pulls in the package, whose compressed-tensors MoE scheme
    # imports gptq_kernels again while it is half loaded.
    import sglang.srt.layers.quantization  # noqa: F401
    from sglang.kernels.ops.moe.sm70_nvfp4_moe_decode import sm70_nvfp4_moe_decode
    from sglang.srt.hardware_backend.gpu.quantization.gptq_kernels import (
        gptq_marlin_moe_repack,
    )
    from sglang.srt.layers.quantization.marlin_utils import (
        sm70_nvfp4_marlin_process_global_scale,
        sm70_nvfp4_marlin_process_scales,
    )

    torch.manual_seed(412 + batch_size)
    device = "cuda"
    experts = 12
    levels = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device=device,
    )

    def make_weight(n, k):
        raw = torch.randint(256, (experts, n, k // 2), dtype=torch.uint8, device=device)
        scales = (torch.rand(experts, n, k // 16, device=device) + 0.5).to(
            torch.float8_e4m3fn
        )
        global_scale = torch.rand(experts, device=device) * 0.01 + 0.01
        codes = torch.stack((raw & 15, raw >> 4), dim=-1).reshape(experts, n, k)
        dequant = levels[codes.long()] * scales.float().repeat_interleave(16, dim=-1)
        dequant *= global_scale[:, None, None]
        packed = gptq_marlin_moe_repack(
            raw.view(torch.int32).transpose(1, 2).contiguous(),
            torch.empty((experts, 0), device=device, dtype=torch.int32),
            k,
            n,
            4,
        )
        encoded, factor = sm70_nvfp4_marlin_process_scales(
            scales.transpose(1, 2).contiguous(), torch.float16
        )
        processed_global = (
            sm70_nvfp4_marlin_process_global_scale(global_scale, torch.float16) / factor
        )
        return packed, encoded, processed_global, dequant

    w13, s13, g13, reference13 = make_weight(320, 2560)
    w2, s2, g2, reference2 = make_weight(2560, 160)
    x = torch.randn(batch_size, 2560, device=device, dtype=torch.float16)
    ids = torch.rand(batch_size, experts, device=device).topk(10, dim=-1).indices.int()
    ids[:, -1] = -1  # Masked routes must not read stale partials.
    weights = torch.softmax(torch.randn(batch_size, 10, device=device), dim=-1)
    output = sm70_nvfp4_moe_decode(x, w13, w2, s13, s2, g13, g2, ids, weights)
    reference = torch.zeros_like(x, dtype=torch.float32)
    for row in range(batch_size):
        for route in range(9):
            expert = int(ids[row, route])
            gate, up = (reference13[expert] @ x[row].float()).chunk(2)
            activated = (torch.nn.functional.silu(gate) * up).half().float()
            reference[row] += (reference2[expert] @ activated) * weights[row, route]

    assert torch.isfinite(output).all()
    # The existing decode contract accumulates each 64/32-element partial in
    # FP16, unlike the independent FP32 reference. Check normalized error so
    # cancellations near zero do not require a loose relative tolerance.
    relative_l2 = (output.float() - reference).norm() / reference.norm()
    assert relative_l2 < 0.003
    torch.testing.assert_close(output.float(), reference, rtol=0.02, atol=0.01)


_FUSED_EXPERTS = 32  # ids stay below this; 512 is still the kernel's invalid id


@pytest.fixture(scope="module")
def fused_weights():
    gen = torch.Generator(device="cuda")
    gen.manual_seed(1234)
    e = _FUSED_EXPERTS

    def words(*shape):
        return torch.randint(
            -(2**31), 2**31 - 1, shape, dtype=torch.int32, device="cuda", generator=gen
        )

    def scale_bytes(*shape):
        # FP8-E4M3 bytes; with global scales near 1e2 the outputs are O(1).
        return torch.randint(
            0x60, 0x78, shape, dtype=torch.uint8, device="cuda", generator=gen
        )

    def global_scales():
        return (torch.rand(e, device="cuda", generator=gen) * 2 + 0.25) * 128

    return (
        words(e, 160, 640),
        words(e, 10, 5120),
        scale_bytes(e, 160, 320),
        scale_bytes(e, 10, 2560),
        global_scales(),
        global_scales(),
    )


def _fused_ids(rows, mode, gen):
    if mode == "distinct":
        ids = torch.stack(
            [
                torch.randperm(_FUSED_EXPERTS, device="cuda", generator=gen)[:10]
                for _ in range(rows)
            ]
        )
    else:  # every row routes to the same ten experts, in a different order
        pool = torch.randperm(_FUSED_EXPERTS, device="cuda", generator=gen)[:10]
        ids = torch.stack(
            [
                pool[torch.randperm(10, device="cuda", generator=gen)]
                for _ in range(rows)
            ]
        )
    ids = ids.int()
    if mode == "invalid":
        ids[0, 3] = -1
        ids[-1, 7] = 512
    return ids.flatten().contiguous()


def _legacy_decode(weights, x, ids, wts):
    from sglang.kernels.ops.moe.sm70_nvfp4_moe_decode import (
        _load_sm70_nvfp4_moe_decode_ops,
    )

    routes = x.shape[0] * 10
    out = torch.empty_like(x)
    _load_sm70_nvfp4_moe_decode_ops().decode_legacy(
        x,
        *weights,
        ids,
        wts,
        torch.empty((40, routes, 320), dtype=torch.float32, device="cuda"),
        torch.empty((routes, 160), dtype=torch.float16, device="cuda"),
        torch.empty((routes, 5, 2560), dtype=torch.float32, device="cuda"),
        out,
    )
    return out


def _random_inputs(rows, mode, gen):
    x = torch.randn((rows, 2560), dtype=torch.float16, device="cuda", generator=gen)
    wts = torch.softmax(torch.randn((rows, 10), device="cuda", generator=gen), -1)
    return x, _fused_ids(rows, mode, gen), wts.flatten().contiguous()


@pytest.mark.parametrize("rows", [1, 2, 3, 4])
@pytest.mark.parametrize("mode", ["distinct", "shared", "invalid"])
def test_sm70_nvfp4_moe_two_kernel_path_matches_legacy_bits(fused_weights, rows, mode):
    """The two-kernel decode folds partials in the four-kernel order, so it must be
    bit-identical to it, invalid routes included; each row count has its own launch shape."""
    from sglang.kernels.ops.moe.sm70_nvfp4_moe_decode import sm70_nvfp4_moe_decode

    gen = torch.Generator(device="cuda")
    gen.manual_seed(rows * 31 + len(mode))
    x, ids, wts = _random_inputs(rows, mode, gen)
    expected = _legacy_decode(fused_weights, x, ids, wts)
    actual = sm70_nvfp4_moe_decode(x, *fused_weights, ids, wts)
    assert (expected != 0).float().mean() > 0.5
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize("rows", [1, 4])
def test_sm70_nvfp4_moe_two_kernel_graph_replay_with_changing_routes(
    fused_weights, rows
):
    """A captured decode must follow new routes, inputs and weights on every replay."""
    from sglang.kernels.ops.moe.sm70_nvfp4_moe_decode import sm70_nvfp4_moe_decode

    gen = torch.Generator(device="cuda")
    gen.manual_seed(900 + rows)
    x, ids, wts = _random_inputs(rows, "distinct", gen)
    sm70_nvfp4_moe_decode(x, *fused_weights, ids, wts)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = sm70_nvfp4_moe_decode(x, *fused_weights, ids, wts)
    for mode in ["shared", "distinct", "invalid"]:
        new_x, new_ids, new_wts = _random_inputs(rows, mode, gen)
        x.copy_(new_x)
        ids.copy_(new_ids)
        wts.copy_(new_wts)
        for _ in range(2):
            graph.replay()
            expected = _legacy_decode(fused_weights, x, ids, wts)
            assert torch.equal(out.view(torch.int16), expected.view(torch.int16))


def test_sm70_nvfp4_moe_misaligned_scales_fall_back_to_legacy(fused_weights):
    """Scales that are not 16-byte aligned cannot take the vector loads; the wrapper
    must hand the kernel full-size scratch for the four-kernel path."""
    from sglang.kernels.ops.moe.sm70_nvfp4_moe_decode import sm70_nvfp4_moe_decode

    w13, w2, s13, s2, g13, g2 = fused_weights
    buf = torch.empty(s13.numel() + 16, dtype=torch.uint8, device="cuda")
    s13_off = buf[8 : 8 + s13.numel()].view(s13.shape)
    s13_off.copy_(s13)
    assert s13_off.data_ptr() % 16 == 8
    gen = torch.Generator(device="cuda")
    gen.manual_seed(77)
    x, ids, wts = _random_inputs(4, "distinct", gen)
    weights = (w13, w2, s13_off, s2, g13, g2)
    actual = sm70_nvfp4_moe_decode(x, *weights, ids, wts)
    assert torch.equal(
        actual.view(torch.int16),
        _legacy_decode(fused_weights, x, ids, wts).view(torch.int16),
    )
