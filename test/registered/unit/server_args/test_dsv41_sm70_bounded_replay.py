"""DeepSeek-V4.1 decoder and encoder SWA bounded replay stay off on SM70.

DeepseekV4Sm70Model runs every row in every layer; with the flag on, the
backend and the prefix cache would act on a late-layer tail the model never
uses.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups import deepseek_v4_hook
from sglang.srt.arg_groups.overrides import resolving_view
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    CudaGraphConfig,
    PhaseConfig,
)
from sglang.srt.runtime_context import override_platform
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_ENV_GATE = "sglang.srt.layers.attention.dsv4.unified_kv_gate"


def _resolve(*, sm70: bool, **overrides) -> ServerArgs:
    args = ServerArgs(
        model_path="local-deepseek-v41",
        attention_backend="dsv4",
        cuda_graph_config=CudaGraphConfig(
            prefill=PhaseConfig(backend=Backend.DISABLED)
        ),
        **overrides,
    )
    args._model_config = SimpleNamespace(
        hf_config=SimpleNamespace(model_type="deepseek_v41")
    )
    with (
        override_platform(is_cuda=True, is_hip=False, is_sm70=sm70),
        patch(f"{_ENV_GATE}.is_unified_kv_triton", return_value=False),
    ):
        deepseek_v4_hook.validate_deepseek_v41_features(args)
    return args


class TestDsv41Sm70BoundedReplay(CustomTestCase):
    def test_default_resolves_off_on_sm70_only(self):
        for sm70, expected in ((False, True), (True, False)):
            with self.subTest(sm70=sm70):
                args = _resolve(sm70=sm70)
                self.assertIs(
                    resolving_view(args).enable_decoder_swa_bounded_replay, expected
                )

    def test_explicit_flag_is_rejected_on_sm70(self):
        with self.assertRaisesRegex(ValueError, "cannot be combined with SM70"):
            _resolve(sm70=True, enable_decoder_swa_bounded_replay=True)

    def test_encoder_replay_is_rejected_on_sm70(self):
        with self.assertRaisesRegex(ValueError, "does not support SM70"):
            _resolve(
                sm70=True,
                enable_encoder_swa_bounded_replay=True,
                max_running_requests=2,
                chunked_prefill_size=256,
            )


if __name__ == "__main__":
    unittest.main()
