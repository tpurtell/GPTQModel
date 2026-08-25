# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
import torch
from gptqmodel.exllamav3.modules.quant.exl3_lib.quantize import (
    reconstruction_error_metrics,
)
from gptqmodel.nn_modules.exllamav3 import ExllamaV3Linear
from gptqmodel.nn_modules.exllamav3_torch import ExllamaV3TorchLinear
from gptqmodel.quantization.config import (
    FORMAT,
    METHOD,
    AutoModuleDecoderConfig,
    EXL3Config,
    QuantizeConfig,
)
from gptqmodel.utils.exllamav3 import (
    build_exllamav3_tensor_storage,
    replace_exllamav3_placeholders,
)
from gptqmodel.utils.model_dequant import detect_format
from safetensors.torch import save_file
from torch import nn


def test_exllamav3_fractional_bits_fail_closed_instead_of_flooring():
    with pytest.raises(ValueError, match="fractional `bits` is not a matrix encoding"):
        QuantizeConfig(
            quant_method=METHOD.EXL3,
            format=FORMAT.EXL3,
            bits=2.25,
        )


def test_exllamav3_reconstruction_metrics_retain_bounded_tile_diagnostics():
    reference = torch.ones((32, 32), dtype=torch.float32)
    error = torch.zeros_like(reference)
    error[:16, :16] = 2.0

    metrics = reconstruction_error_metrics(error, reference, worst_tiles=2)

    assert metrics["domain"] == "regularized_exl3_search_space"
    assert metrics["element_count"] == 1024
    assert metrics["error_sum_sq"] == pytest.approx(1024.0)
    assert metrics["reference_sum_sq"] == pytest.approx(1024.0)
    assert metrics["mse"] == pytest.approx(1.0)
    assert metrics["nmse"] == pytest.approx(1.0)
    assert metrics["relative_frobenius"] == pytest.approx(1.0)
    assert metrics["tile_count"] == 4
    assert metrics["tile_sse_max"] == pytest.approx(1024.0)
    assert metrics["worst_tiles"][0] == {"row": 0, "column": 0, "sse": 1024.0}


def test_exllamav3_integer_and_dynamic_bits_round_trip():
    cfg = QuantizeConfig(
        quant_method=METHOD.EXL3,
        format=FORMAT.EXL3,
        bits=2.0,
        head_bits=4.0,
        out_scales="always",
        codebook="mul1",
        dynamic={r"^model\.layers\.0\.mlp\.experts\.7\.": {"bits": 3.0}},
    )

    assert isinstance(cfg, EXL3Config)
    assert cfg.quant_method == METHOD.EXL3
    assert cfg.format == FORMAT.EXL3
    assert cfg.runtime_bits == 2
    assert cfg.uses_weight_only_lifecycle() is False
    assert cfg.requires_calibration_dataset() is True

    payload = cfg.to_dict()
    assert payload["bits"] == 2.0
    assert payload["head_bits"] == 4.0
    assert payload["out_scales"] == "always"
    assert payload["codebook"] == "mul1"

    reloaded = QuantizeConfig.from_quant_config(payload)
    assert isinstance(reloaded, EXL3Config)
    assert reloaded.bits == 2.0
    assert reloaded.head_bits == 4.0
    assert reloaded.out_scales == "always"
    assert reloaded.codebook == "mul1"
    assert reloaded.runtime_bits == 2
    assert (
        reloaded.dynamic_get(
            "model.layers.0.mlp.experts.7.gate_proj", "bits", reloaded.bits
        )
        == 3.0
    )


def test_exllamav3_config_serialization_omits_machine_local_execution_state():
    provenance = {
        "family_join": {
            "schema_version": 1,
            "inventory_sha256": "a" * 64,
        }
    }
    cfg = EXL3Config(
        bits=3.0,
        offload_to_disk=True,
        offload_to_disk_path="/private/quant/offload",
        pack_impl="gpu",
        wait_for_submodule_finalizers=True,
        auto_forward_data_parallel=False,
        dense_vram_strategy="balanced",
        dense_vram_strategy_devices=["cuda:0", "cuda:1"],
        moe_vram_strategy="balanced",
        moe_vram_strategy_devices=["cuda:0", "cuda:1"],
        meta={
            "ds4rt_error_ledger": provenance,
            "quantizer": "gptqmodel-pinned",
        },
    )

    payload = cfg.to_dict()
    assert payload["meta"]["ds4rt_error_ledger"] == provenance
    assert payload["meta"]["quantizer"] == "gptqmodel-pinned"
    assert not {
        "offload_to_disk",
        "offload_to_disk_path",
        "pack_impl",
        "gc_mode",
        "wait_for_submodule_finalizers",
        "auto_forward_data_parallel",
        "dense_vram_strategy",
        "dense_vram_strategy_devices",
        "moe_vram_strategy",
        "moe_vram_strategy_devices",
        "weight_only",
    }.intersection(payload["meta"])


def test_exllamav3_config_accepts_and_round_trips_auto_module_decoder():
    cfg = EXL3Config(
        bits=2.0,
        preprocessors=[AutoModuleDecoderConfig(target_dtype=torch.bfloat16)],
    )

    assert len(cfg.preprocessors) == 1
    assert isinstance(cfg.preprocessors[0], AutoModuleDecoderConfig)
    assert cfg.preprocessors[0].target_dtype is torch.bfloat16

    reloaded = QuantizeConfig.from_quant_config(cfg.to_dict())
    assert isinstance(reloaded, EXL3Config)
    assert len(reloaded.preprocessors) == 1
    assert isinstance(reloaded.preprocessors[0], AutoModuleDecoderConfig)
    assert reloaded.preprocessors[0].target_dtype is torch.bfloat16


def test_exllamav3_serializes_standard_bits_as_json_integer():
    cfg = EXL3Config(bits=2.0, head_bits=3.0)

    payload = cfg.to_dict()

    assert payload["bits"] == 2
    assert type(payload["bits"]) is int
    assert payload["head_bits"] == 3
    assert type(payload["head_bits"]) is int


def test_exllamav3_module_include_is_a_positive_allowlist_and_round_trips():
    routed_expert_pattern = (
        r"^model\.layers\.\d+\.mlp\.experts\.\d+\."
        r"(?:gate_proj|up_proj|down_proj)$"
    )
    cfg = EXL3Config(bits=2.0, module_include=[routed_expert_pattern])

    assert cfg.module_is_included("model.layers.7.mlp.experts.31.gate_proj")
    assert not cfg.module_is_included("model.layers.7.mlp.shared_experts.gate_proj")
    assert not cfg.module_is_included("model.layers.7.self_attn.q_a_proj")
    assert not cfg.module_is_included("lm_head")

    reloaded = QuantizeConfig.from_quant_config(cfg.to_dict())
    assert isinstance(reloaded, EXL3Config)
    assert reloaded.module_include == [routed_expert_pattern]
    assert reloaded.module_is_included("model.layers.42.mlp.experts.255.down_proj")
    assert not reloaded.module_is_included(
        "model.layers.42.mlp.shared_experts.down_proj"
    )


def test_exllamav3_module_include_rejects_invalid_regex():
    with pytest.raises(ValueError, match="invalid module_include pattern"):
        EXL3Config(bits=2.0, module_include=["("])


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(16, 16, bias=True)


def test_replace_exllamav3_placeholders_uses_tensor_storage_metadata():
    model = _TinyModel()
    tensor_storage = {
        "proj": {
            "stored_tensors": {
                "proj.trellis": {"shape": [1, 1, 32], "torch_dtype": "int16"},
                "proj.suh": {"shape": [16], "torch_dtype": "float16"},
                "proj.svh": {"shape": [16], "torch_dtype": "float16"},
                "proj.bias": {"shape": [16], "torch_dtype": "float16"},
                "proj.mul1": {"shape": [], "torch_dtype": "int32"},
            },
            "quant_format": "exl3",
            "bits_per_weight": 2,
        }
    }

    replace_exllamav3_placeholders(
        model=model,
        module_names=["proj"],
        tensor_storage=tensor_storage,
    )

    assert isinstance(model.proj, ExllamaV3Linear)
    assert model.proj.trellis.device.type == "meta"
    assert tuple(model.proj.trellis.shape) == (1, 1, 32)
    assert model.proj.suh.dtype == torch.float16
    assert model.proj.svh.dtype == torch.float16
    assert model.proj.bias.dtype == torch.float16
    assert model.proj.mul1.dtype == torch.int32


def test_replace_exllamav3_placeholders_supports_torch_reference_kernel():
    model = _TinyModel()
    tensor_storage = {
        "proj": {
            "stored_tensors": {
                "proj.trellis": {"shape": [1, 1, 32], "torch_dtype": "int16"},
                "proj.suh": {"shape": [16], "torch_dtype": "float16"},
                "proj.svh": {"shape": [16], "torch_dtype": "float16"},
            },
            "quant_format": "exl3",
            "bits_per_weight": 2,
        }
    }

    replace_exllamav3_placeholders(
        model=model,
        module_names=["proj"],
        tensor_storage=tensor_storage,
        module_cls=ExllamaV3TorchLinear,
    )

    assert isinstance(model.proj, ExllamaV3TorchLinear)
    assert build_exllamav3_tensor_storage(model)["proj"]["quant_format"] == "exl3"


@pytest.mark.parametrize("module_cls", [ExllamaV3Linear, ExllamaV3TorchLinear])
def test_exllamav3_modules_support_recursive_to_empty(module_cls):
    module = module_cls.from_tensors(
        in_features=16,
        out_features=16,
        name="proj",
        tensors={
            "trellis": torch.zeros((1, 1, 32), dtype=torch.int16),
            "suh": torch.zeros(16, dtype=torch.float16),
            "svh": torch.zeros(16, dtype=torch.float16),
        },
    )

    module.to_empty(device=torch.device("meta"), recurse=True)

    assert module.trellis.device.type == "meta"
    assert module.suh.device.type == "meta"
    assert module.svh.device.type == "meta"


def test_detect_format_identifies_exllamav3(tmp_path):
    shard_path = tmp_path / "model.safetensors"
    save_file(
        {
            "model.layers.0.self_attn.q_proj.trellis": torch.zeros(
                (1, 1, 32), dtype=torch.int16
            ),
            "model.layers.0.self_attn.q_proj.suh": torch.zeros(
                (16,), dtype=torch.float16
            ),
            "model.layers.0.self_attn.q_proj.svh": torch.zeros(
                (16,), dtype=torch.float16
            ),
        },
        str(shard_path),
    )

    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "quantization_config": {
                    "quant_method": "exl3",
                    "format": "exl3",
                }
            }
        ),
        encoding="utf-8",
    )

    detected = detect_format(
        tmp_path, json.loads(config_path.read_text(encoding="utf-8"))
    )
    assert detected == "exl3"
