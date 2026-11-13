"""GLM-5.3 MTP draft head: quantized like the target's lm_head.

The checkpoint has no draft head (the draft takes the target's), and ModelOpt
excludes ``lm_head`` but not ``model.shared_head.head``. Quantized, the draft
head's never-loaded scales went through the Marlin repack at load, and a start
failed whenever that memory held a negative byte.
"""

from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp4Config
from sglang.srt.models.glm5_next_nextn import Glm5NextForConditionalGenerationNextN


class _TextConfig:
    num_hidden_layers = 45


class _Config:
    text_config = _TextConfig


def _nvfp4_config(exclude_modules):
    quant_config = ModelOptFp4Config.from_config(
        {
            "quant_method": "modelopt",
            "quant_algo": "NVFP4",
            "config_groups": {
                "group_0": {
                    "targets": ["Linear"],
                    "weights": {"num_bits": 4, "type": "float", "group_size": 16},
                    "input_activations": {
                        "num_bits": 4,
                        "type": "float",
                        "group_size": 16,
                    },
                }
            },
            "exclude_modules": exclude_modules,
        }
    )
    quant_config.apply_weight_name_mapper(
        Glm5NextForConditionalGenerationNextN.get_hf_to_sglang_mapper(_Config)
    )
    return quant_config


def _resolve(quant_config):
    return Glm5NextForConditionalGenerationNextN._resolve_lm_head_quant_config(
        None, quant_config
    )


def test_draft_head_follows_excluded_target_head():
    # GLM-5.3-Flash-NVFP4's list, shortened.
    quant_config = _nvfp4_config(
        ["lm_head", "model.language_model.layers.45*", "model.layers.45*"]
    )
    assert not quant_config.is_layer_excluded("model.shared_head.head")
    assert _resolve(quant_config) is None


def test_draft_head_keeps_a_quantized_target_heads_config():
    quant_config = _nvfp4_config(["model.layers.45*"])
    assert _resolve(quant_config) is quant_config
    assert _resolve(None) is None
