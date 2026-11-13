"""GLM-5.3 vision tower with its weights in host memory (V100).

The tower must give the same output as with resident weights, hold nothing on
the GPU between images, and keep only the small resident pieces.
"""

import os
import socket

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="The host-weight vision tower runs on V100 only",
)

# GLM-5.3-Flash's vision_config, with fewer blocks.
VISION_CONFIG = dict(
    attention_bias=True,
    depth=3,
    hidden_size=1024,
    in_channels=3,
    intermediate_size=4096,
    num_heads=16,
    out_hidden_size=4096,
    patch_size=14,
    projection_intermediate_size=10240,
    rms_norm_eps=1e-5,
    spatial_merge_size=2,
    swiglu_limit=10.0,
    temporal_patch_size=2,
)


@pytest.fixture(scope="module")
def one_rank():
    from sglang.srt.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from sglang.srt.runtime_context import get_context, get_parallel
    from sglang.test.test_utils import publish_build_topology

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method=f"tcp://127.0.0.1:{port}",
        backend="gloo",
    )
    publish_build_topology(tp_size=1)
    initialize_model_parallel(backend="gloo")
    with (
        get_parallel().override(tp_size=1, tp_rank=0, attn_tp_size=1, attn_tp_rank=0),
        get_context().override_server_args(),
    ):
        yield
    destroy_model_parallel()
    destroy_distributed_environment()


def _build_tower():
    from sglang.srt.configs.glm5_next import Glm5NextVisionConfig
    from sglang.srt.models.glm5_next import Glm5NextVisionModel

    torch.manual_seed(0)
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float16)
    try:
        with torch.device("cuda"):
            tower = Glm5NextVisionModel(Glm5NextVisionConfig(**VISION_CONFIG))
    finally:
        torch.set_default_dtype(default_dtype)
    with torch.no_grad():
        for name, p in tower.named_parameters():
            if "norm" in name and name.endswith("weight"):
                p.copy_(1 + 0.1 * torch.randn_like(p))
            else:
                p.copy_(0.02 * torch.randn_like(p))
    return tower.eval()


def _images(grids):
    grid_thw = torch.tensor(grids, dtype=torch.int64)
    patches = int(grid_thw.prod(-1).sum())
    channels = 3 * 2 * 14 * 14
    pixel_values = torch.randn(patches, channels, dtype=torch.float16)
    return pixel_values, grid_thw


def _offloaded_modules(tower):
    return (*tower.blocks, tower.downsample, tower.merger)


@pytest.mark.parametrize("grids", [[[1, 28, 40]], [[1, 16, 16], [1, 64, 46]]])
def test_host_weights_match_resident(one_rank, grids):
    tower = _build_tower()
    pixel_values, grid_thw = _images(grids)
    with torch.no_grad():
        resident = tower(pixel_values, grid_thw=grid_thw)
    torch.cuda.synchronize()

    sizes = [
        p.numel() * p.element_size()
        for module in _offloaded_modules(tower)
        for p in module.parameters()
    ]
    before = torch.cuda.memory_allocated()
    moved = tower.move_weights_to_host()
    after = torch.cuda.memory_allocated()

    assert moved == sum(sizes)
    # The caching allocator counts each block rounded up to 512 bytes.
    assert before - after == sum(-(-size // 512) * 512 for size in sizes)
    for module in _offloaded_modules(tower):
        for name, p in module.named_parameters():
            assert p.device.type == "cpu" and p.is_pinned(), name
    for name, p in tower.patch_embed.named_parameters():
        assert p.device.type == "cuda", name
    assert tower.device.type == "cuda"

    with torch.no_grad():
        on_host = tower(pixel_values, grid_thw=grid_thw)
    torch.cuda.synchronize()

    assert on_host.device.type == "cuda"
    assert torch.equal(on_host, resident)
    # The per-forward copies die with the forward.
    del on_host
    assert torch.cuda.memory_allocated() == after


# 256, 2944, 64 and 1120 patches. Passes join neighbouring images only: at 3000
# they are [0], [1], [2, 3]; at 1000 each image runs alone, the 2944-patch one
# above the limit.
GROUP_GRIDS = [[1, 16, 16], [1, 64, 46], [1, 8, 8], [1, 28, 40]]
GROUPS = {3000: [[0], [1], [2, 3]], 1000: [[0], [1], [2], [3]]}


@pytest.mark.parametrize("limit", sorted(GROUPS))
def test_grouped_passes_match_one_pass(one_rank, limit):
    tower = _build_tower()
    tower.move_weights_to_host()
    pixel_values, grid_thw = _images(GROUP_GRIDS)
    patches = grid_thw.prod(-1).tolist()
    starts = [sum(patches[:i]) for i in range(len(patches) + 1)]
    with torch.no_grad():
        one_pass = tower(pixel_values, grid_thw=grid_thw)
        alone = torch.cat(
            [
                tower(
                    pixel_values[starts[rows[0]] : starts[rows[-1] + 1]],
                    grid_thw=grid_thw[rows[0] : rows[-1] + 1],
                )
                for rows in GROUPS[limit]
            ]
        )
        tower.max_patches_per_forward = limit
        grouped = tower(pixel_values, grid_thw=grid_thw)
    assert grouped.shape == one_pass.shape == (sum(patches) // 4, 4096)
    # Each pass is exactly that group run on its own.
    assert torch.equal(grouped, alone)
    # Against one pass over all rows, cuBLAS picks other GEMM kernels for the
    # other row counts (4e-4 relative per Linear, measured), as it does for a
    # request with one image instead of several.
    error = (grouped.float() - one_pass.float()).norm() / one_pass.float().norm()
    assert error < 5e-3


def test_features_reach_the_tower_on_the_host():
    from sglang.srt.managers.mm_schedule import _can_skip_pre_embed_feature_move
    from sglang.srt.models.glm5_next import Glm5NextForConditionalGeneration

    # The scheduler leaves the pixels where they are; each pass uploads its own.
    model = Glm5NextForConditionalGeneration.__new__(Glm5NextForConditionalGeneration)
    assert _can_skip_pre_embed_feature_move(model.get_image_feature)
    assert _can_skip_pre_embed_feature_move(model.get_video_feature)


def test_move_is_idempotent(one_rank):
    tower = _build_tower()
    first = tower.move_weights_to_host()
    allocated = torch.cuda.memory_allocated()
    assert tower.move_weights_to_host() == first
    assert torch.cuda.memory_allocated() == allocated
    pixel_values, grid_thw = _images([[1, 8, 8]])
    with torch.no_grad():
        assert torch.isfinite(tower(pixel_values, grid_thw=grid_thw)).all()
