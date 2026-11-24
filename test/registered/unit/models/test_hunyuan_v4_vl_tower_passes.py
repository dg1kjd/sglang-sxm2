"""HYV4 vision tower: a call split into passes of at most max_patches_per_forward
patches gives each image and video back whole and in order (CPU only).

The tower's per-pass body (_forward) is replaced by one that returns, per grid
row, the frame ids it was given, so the test checks which frames each pass got
and how the passes are put back together.
"""

from types import SimpleNamespace

import torch

from sglang.srt.models.hunyuan_v4_vl import HYV4VisionTransformer
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _tower(limit, temporal_patch_size=1):
    passes = []

    def fake_forward(pixel_values, grid_thw):
        passes.append([list(g) for g in grid_thw])
        assert pixel_values.shape[0] == sum(t * h * w for t, h, w in grid_thw)
        out, offset = [], 0
        for t, h, w in grid_thw:
            frames = pixel_values[offset : offset + t * h * w : h * w, 0]
            out.append(frames.clone())
            offset += t * h * w
        return out

    tower = SimpleNamespace(
        max_patches_per_forward=limit,
        config=SimpleNamespace(temporal_patch_size=temporal_patch_size),
        _forward=fake_forward,
    )
    return tower, passes


def _inputs(grids):
    """Pixel rows carrying a global frame id, and the ids each grid row owns."""
    rows, want, frame = [], [], 0
    for t, h, w in grids:
        ids = torch.arange(frame, frame + t, dtype=torch.float32)
        rows.append(ids.repeat_interleave(h * w)[:, None])
        want.append(ids)
        frame += t
    return torch.cat(rows), want


def _run(tower, grids):
    pixels, want = _inputs(grids)
    got = HYV4VisionTransformer.forward(tower, pixels, grids)
    assert len(got) == len(want)
    for g, w in zip(got, want):
        assert torch.equal(g, w)


def test_no_limit_is_one_pass():
    tower, passes = _tower(None)
    grids = [[1, 64, 64], [3, 32, 32], [1, 128, 128]]
    _run(tower, grids)
    assert passes == [grids]


def test_images_grouped_whole():
    tower, passes = _tower(8192)
    grids = [[1, 64, 64], [1, 64, 64], [1, 64, 64], [1, 16, 16]]
    _run(tower, grids)
    assert passes == [[[1, 64, 64], [1, 64, 64]], [[1, 64, 64], [1, 16, 16]]]


def test_image_above_limit_runs_alone():
    tower, passes = _tower(4096)
    grids = [[1, 16, 16], [1, 128, 128], [1, 16, 16]]
    _run(tower, grids)
    assert passes == [[[1, 16, 16]], [[1, 128, 128]], [[1, 16, 16]]]


def test_video_split_into_frame_chunks():
    # 1024 patches a frame, 4 frames a pass; the last chunk shares its pass
    # with the image after it.
    tower, passes = _tower(4096)
    grids = [[10, 32, 32], [1, 32, 32]]
    _run(tower, grids)
    assert passes == [
        [[4, 32, 32]],
        [[4, 32, 32]],
        [[2, 32, 32], [1, 32, 32]],
    ]


def test_video_chunks_keep_temporal_groups():
    # The projector pools pairs of frames: chunks are cut at even frames only.
    tower, passes = _tower(3 * 1024, temporal_patch_size=2)
    grids = [[7, 32, 32]]
    _run(tower, grids)
    assert passes == [[[2, 32, 32]], [[2, 32, 32]], [[2, 32, 32], [1, 32, 32]]]


def test_frame_above_limit_runs_one_group_a_pass():
    tower, passes = _tower(1000, temporal_patch_size=2)
    grids = [[4, 32, 32]]
    _run(tower, grids)
    assert passes == [[[2, 32, 32]], [[2, 32, 32]]]
