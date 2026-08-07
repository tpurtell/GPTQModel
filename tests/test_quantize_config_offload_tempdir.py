# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import gptqmodel.quantization.config as quant_config_module
import torch
from gptqmodel.looper.exllamav3_processor import EXL3Processor
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.quantization import EXL3Config, QuantizeConfig


def test_quantize_config_offload_path_defaults_to_tempdir(monkeypatch):
    class FakeTemporaryDirectory:
        def __init__(self, *, prefix):
            assert prefix == "gptqmodel_"
            self.name = "/tmp/gptqmodel_quant_cfg"

        def cleanup(self):
            return None

    monkeypatch.setattr(quant_config_module, "_SharedTemporaryDirectory", FakeTemporaryDirectory)

    cfg = QuantizeConfig(offload_to_disk=True)

    assert cfg.offload_to_disk_path == "/tmp/gptqmodel_quant_cfg"
    assert isinstance(cfg._offload_temp_dir, FakeTemporaryDirectory)


def test_exl3_config_offload_path_defaults_to_tempdir(monkeypatch):
    class FakeTemporaryDirectory:
        def __init__(self, *, prefix):
            assert prefix == "gptqmodel_"
            self.name = "/tmp/gptqmodel_exl3_cfg"

        def cleanup(self):
            return None

    monkeypatch.setattr(quant_config_module, "_SharedTemporaryDirectory", FakeTemporaryDirectory)

    cfg = EXL3Config(offload_to_disk=True)

    assert cfg.offload_to_disk_path == "/tmp/gptqmodel_exl3_cfg"
    assert isinstance(cfg._offload_temp_dir, FakeTemporaryDirectory)


def test_exl3_projection_capture_does_not_allocate_an_offload_tempdir(
    monkeypatch,
    tmp_path,
):
    def unexpected_tempdir():
        raise AssertionError("projection Hessian capture allocated a tempdir")

    monkeypatch.setattr(
        quant_config_module,
        "_create_temp_offload_dir",
        unexpected_tempdir,
    )
    qcfg = EXL3Config(
        bits=2.0,
        module_include=[r"^model\.layers\.0\.mlp\.experts\.0\.gate_proj$"],
        offload_to_disk=True,
        offload_to_disk_path=str(tmp_path / "shared-offload"),
        device="cpu",
    )
    processor = EXL3Processor.__new__(EXL3Processor)
    processor.qcfg = qcfg
    processor.tasks = {}
    processor.total_calibration_tokens = 8
    processor._remote_client_initialized = True
    processor._remote_client = None

    module = NamedModule(
        torch.nn.Linear(16, 16, bias=False),
        name="gate_proj",
        full_name="model.layers.0.mlp.experts.0.gate_proj",
        layer_index=0,
    )
    processor.preprocess(module)

    capture = processor.tasks[module.name]["capture"]
    assert capture.qcfg.offload_to_disk is False
    assert capture.qcfg.offload_to_disk_path is None
    assert processor.tasks[module.name]["qcfg"].offload_to_disk_path == str(
        tmp_path / "shared-offload"
    )
