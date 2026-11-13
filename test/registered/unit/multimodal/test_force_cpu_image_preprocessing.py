"""``SGLANG_FORCE_CPU_IMAGE_PREPROCESSING`` on the common image path.

With the switch on, neither the JPEG decode nor the fast image processor may
run on the tokenizer's GPU. With it off, the defaults are unchanged.

No server, no model loading — pure CPU.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from sglang.srt.environ import envs
from sglang.srt.multimodal.modality import Modality
from sglang.srt.multimodal.processors import base_processor
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sglang.test.test_utils import CustomTestCase


class _GpuDecodeProcessor(BaseMultimodalProcessor):
    gpu_image_decode = True

    async def process_mm_data_async(self, *args, **kwargs):
        raise NotImplementedError


class TestForceCpuImagePreprocessing(CustomTestCase):
    def _decode_flag(self, force_cpu: bool) -> bool:
        image = Image.new("RGB", (8, 8))
        with (
            patch.object(
                base_processor, "load_image", return_value=(image, image.size)
            ) as loader,
            envs.SGLANG_FORCE_CPU_IMAGE_PREPROCESSING.override(force_cpu),
        ):
            result = _GpuDecodeProcessor._load_single_item(b"jpeg", Modality.IMAGE)
        self.assertIs(result, image)
        loader.assert_called_once()
        return loader.call_args.args[1]

    def test_switch_turns_gpu_jpeg_decode_off(self):
        self.assertFalse(self._decode_flag(True))

    def test_default_keeps_gpu_jpeg_decode(self):
        self.assertTrue(self._decode_flag(False))

    def test_switch_puts_fast_image_processor_on_cpu(self):
        processor = object.__new__(_GpuDecodeProcessor)
        processor.server_args = SimpleNamespace(base_gpu_id=3)
        with (
            patch.object(base_processor, "_is_cpu", False),
            envs.SGLANG_FORCE_CPU_IMAGE_PREPROCESSING.override(True),
        ):
            device = processor._fast_image_processor_device(object())
        self.assertEqual(device, "cpu")


if __name__ == "__main__":
    unittest.main()
