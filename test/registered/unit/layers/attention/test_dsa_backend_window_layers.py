"""A hybrid DSA pool's sliding-window layers write and read their KV at the
window sub-pool's ids; the sparse layers keep the full pool's ids."""

import unittest
from types import MethodType, SimpleNamespace

import torch

from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
from sglang.srt.mem_cache.kv.alloc.id_space import IdSpaceKind
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_FULL_LAYER, _WINDOW_LAYER = 0, 1


def _backend(full_to_window: torch.Tensor):
    """A DSA backend stand-in whose layer 1 lives in the window sub-pool."""

    def ids_in(kind, ids):
        return full_to_window[ids] if kind == IdSpaceKind.SLIDING_WINDOW else ids

    backend = SimpleNamespace(
        runner_ids=SimpleNamespace(ids_in=ids_in, window_storage="swa"),
        window_layer_ids=frozenset({_WINDOW_LAYER}),
        hisparse_coordinator=None,
    )
    for name in ("_kv_write_loc", "_window_read_ids"):
        setattr(
            backend, name, MethodType(getattr(DeepseekSparseAttnBackend, name), backend)
        )
    return backend


def _layer(layer_id):
    return SimpleNamespace(layer_id=layer_id, is_cross_attention=False)


class TestDSABackendWindowLayers(CustomTestCase):
    def setUp(self):
        # Like the SWA allocator's table: full slot -> window slot, and a
        # trailing -1 so an unused (-1) selection stays -1.
        self.full_to_window = torch.tensor([7, 3, 5, 0, 2, -1], dtype=torch.int64)
        self.backend = _backend(self.full_to_window)
        self.batch = SimpleNamespace(
            out_cache_loc=torch.tensor([1, 2, 4], dtype=torch.int64),
            out_cache_loc_is_physical=False,
        )

    def test_window_layer_writes_at_window_ids(self):
        loc = self.backend._kv_write_loc(self.batch, _layer(_WINDOW_LAYER))
        self.assertEqual(loc.window_locs["swa"].tolist(), [3, 5, 2])
        self.assertEqual(loc.loc.tolist(), [1, 2, 4])

    def test_sparse_layer_writes_at_full_ids(self):
        loc = self.backend._kv_write_loc(self.batch, _layer(_FULL_LAYER))
        self.assertIsNone(loc.window_locs)
        self.assertEqual(loc.loc.tolist(), [1, 2, 4])

    def test_window_read_maps_selection_and_keeps_padding(self):
        selected = torch.tensor([[4, 1, -1, -1]], dtype=torch.int32)
        got = self.backend._window_read_ids(selected, ragged=False)
        self.assertEqual(got.dtype, torch.int32)
        self.assertEqual(got.tolist(), [[2, 3, -1, -1]])

    def test_window_read_refuses_ragged_offsets(self):
        with self.assertRaises(NotImplementedError):
            self.backend._window_read_ids(
                torch.zeros((1, 4), dtype=torch.int32), ragged=True
            )


if __name__ == "__main__":
    unittest.main()
