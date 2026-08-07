# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace
import threading

import pytest
import torch
from torch import nn

import gptqmodel.looper.exllamav3_processor as processor_module
from gptqmodel.looper.exllamav3_processor import EXL3Processor
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.utils.exl3_projection_checkpoint import CHECKPOINT_CONTRACT


class _Capture:
    def __init__(self, module: NamedModule, hessian: torch.Tensor) -> None:
        self.module = module
        self.H = hessian
        self.nsamples = 1024

    def finalize_hessian(self, target_device=None):
        self.H = self.H.to(target_device)
        return self.H

    def clone_module(self, copy=True, device=None):
        return self.module.module.weight.detach().to(device=device, copy=copy).float()

    def free(self):
        self.H = None


def _processor_and_module(
    root: Path,
    weight: torch.Tensor,
    hessian: torch.Tensor,
) -> tuple[EXL3Processor, NamedModule]:
    linear = nn.Linear(128, 128, bias=False, dtype=torch.bfloat16, device="cuda:0")
    linear.weight.data.copy_(weight)
    module = NamedModule(
        linear,
        "mlp.experts.31.gate_proj",
        "model.layers.7.mlp.experts.31.gate_proj",
        7,
    )
    processor = EXL3Processor.__new__(EXL3Processor)
    processor.qcfg = SimpleNamespace(
        meta={
            "ds4rt_error_ledger": {
                "family_join": {"source_revision": "test-source"},
                "run": {
                    "projection_checkpoint": {
                        "contract": CHECKPOINT_CONTRACT,
                        "root": str(root),
                    }
                },
            }
        },
        dynamic=None,
    )
    processor.tasks = {
        module.name: {
            "capture": _Capture(module, hessian.clone()),
            "qcfg": SimpleNamespace(
                head_bits=None,
                runtime_bits=2,
                out_scales="auto",
                codebook="mcg",
            ),
        }
    }
    processor.lm_head_name = "lm_head"
    processor.error_journal_path = str(root.parent / "error-journal.jsonl")
    processor._stats_lock = threading.Lock()
    processor.durations = []
    processor.avg_losses = []
    processor.module_names = []
    processor.log = []
    processor.draw_progress = lambda *args, **kwargs: None
    processor.formatted_fwd_time = lambda: "0.000"
    processor.device_memory_report = lambda: "test"
    processor.log_new_row = lambda *args, **kwargs: None
    return processor, module


@pytest.mark.skipif(not torch.cuda.is_available(), reason="EXL3 requires CUDA")
def test_process_resumes_from_packed_checkpoint_without_requantizing(
    tmp_path,
    monkeypatch,
) -> None:
    torch.manual_seed(787)
    weight = (torch.randn((128, 128), dtype=torch.float32, device="cuda:0") * 0.02).to(
        torch.bfloat16
    )
    activations = torch.randn((1024, 128), dtype=torch.float32, device="cuda:0")
    hessian = (2.0 / activations.shape[0]) * activations.T @ activations
    checkpoint_root = tmp_path / "projection-checkpoints"

    first_processor, first_module = _processor_and_module(
        checkpoint_root,
        weight,
        hessian,
    )
    first_processor.process(first_module)
    first_module.stream_sync()
    first_replay_weight = first_module.module.weight.detach().cpu().clone()
    assert first_processor.log[-1]["exl3_projection_checkpoint_hit"] is False
    assert len(list(checkpoint_root.rglob("*.json"))) == 1
    assert len(list(checkpoint_root.rglob("*.safetensors"))) == 1

    second_processor, second_module = _processor_and_module(
        checkpoint_root,
        weight,
        hessian,
    )

    def fail_requantization(*args, **kwargs):
        raise AssertionError("checkpoint hit attempted to run trellis quantization")

    monkeypatch.setattr(processor_module, "quantize_exl3", fail_requantization)
    second_processor.process(second_module)
    second_module.stream_sync()
    assert second_processor.log[-1]["exl3_projection_checkpoint_hit"] is True
    assert torch.equal(
        second_module.module.weight.detach().cpu().view(torch.int16),
        first_replay_weight.view(torch.int16),
    )
