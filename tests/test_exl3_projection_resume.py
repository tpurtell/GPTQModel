# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace
import threading

import pytest
import torch
from torch import nn
from safetensors.torch import save_file as save_safetensors_file

import gptqmodel.looper.exllamav3_processor as processor_module
from gptqmodel.looper.exllamav3_processor import EXL3Processor
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.nn_modules.exllamav3 import ExllamaV3Linear
from gptqmodel.utils.exl3_error_ledger import append_exl3_error_journal
from gptqmodel.utils.exl3_capture_frontier import EXL3CaptureRecord
from gptqmodel.utils.exl3_projection_checkpoint import (
    CHECKPOINT_CONTRACT,
    EXL3ProjectionCheckpointStore,
    build_projection_request,
)


class _Capture:
    def __init__(self, module: NamedModule, hessian: torch.Tensor) -> None:
        self.module = module
        self.H = hessian
        self.nsamples = 1024
        self._device_hessian_partials = {}
        self._device_sample_counts = {}
        self._hessian_dirty = False
        self._final_hessian_device_hint = None
        self.clone_devices = []

    def finalize_hessian(self, target_device=None):
        self.H = self.H.to(target_device)
        return self.H

    def clone_module(self, copy=True, device=None):
        self.clone_devices.append(torch.device(device))
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
    frontier_root = tmp_path / "capture-frontier"
    frontier_root.mkdir()
    frontier_payload = frontier_root / "expert-31-gate.safetensors"
    save_safetensors_file({"H": hessian.cpu()}, frontier_payload)
    second_capture = second_processor.tasks[second_module.name]["capture"]
    second_capture.H = None
    second_processor.tasks[second_module.name]["capture_frontier_record"] = (
        EXL3CaptureRecord(
            module=second_module.full_name,
            path=frontier_payload,
            payload={
                "tensor": {
                    "dtype": "torch.float32",
                    "shape": [128, 128],
                    "bytes": 128 * 128 * 4,
                }
            },
            sample_count=1024,
            route_evidence=None,
        )
    )
    monkeypatch.setenv("GPTQMODEL_EXL3_CAPTURE_FRONTIER", str(frontier_root))

    def fail_requantization(*args, **kwargs):
        raise AssertionError("checkpoint hit attempted to run trellis quantization")

    monkeypatch.setattr(processor_module, "quantize_exl3", fail_requantization)
    second_processor.process(second_module)
    second_module.stream_sync()
    assert second_capture.clone_devices == [torch.device("cpu")]
    assert second_processor.log[-1]["exl3_projection_checkpoint_hit"] is True
    assert torch.equal(
        second_module.module.weight.detach().cpu().view(torch.int16),
        first_replay_weight.view(torch.int16),
    )


def test_capture_memory_summary_attributes_cache_model_and_heap_release(
    monkeypatch,
) -> None:
    processor = EXL3Processor.__new__(EXL3Processor)
    shared = torch.zeros(8, dtype=torch.float32)
    position_ids = torch.zeros(3, dtype=torch.int64)
    processor.inputs_cache = SimpleNamespace(
        layer_inputs=[[shared]],
        layer_input_kwargs=[{"alias": shared[:4]}],
        position_ids=[position_ids],
        attention_masks=[],
    )
    processor.tasks = {}

    model_root = nn.Module()
    model_root.register_parameter(
        "weight", nn.Parameter(torch.zeros(5, dtype=torch.bfloat16))
    )
    model_root.register_buffer("deferred", torch.empty(7, device="meta"))
    model = SimpleNamespace(model=model_root)

    summary = processor.capture_memory_summary(model=model)
    assert summary["process_rss_bytes"] > 0
    assert summary["input_cache_tensor_count"] == 3
    assert summary["input_cache_storage_count"] == 2
    assert summary["input_cache_host_bytes"] == (
        shared.untyped_storage().nbytes()
        + position_ids.untyped_storage().nbytes()
    )
    assert summary["model_tensor_count"] == 2
    assert summary["model_storage_count"] == 1
    assert summary["model_meta_tensor_count"] == 1
    assert summary["model_host_bytes"] == 5 * torch.bfloat16.itemsize

    monkeypatch.setattr(processor_module.gc, "collect", lambda: 7)
    monkeypatch.setattr(
        EXL3Processor,
        "_malloc_trim",
        staticmethod(lambda: 1),
    )
    released = processor.release_host_memory("test", model=model)
    assert released["gc_collected"] == 7
    assert released["malloc_trim_result"] == 1
    assert released["before"]["input_cache_host_bytes"] == summary[
        "input_cache_host_bytes"
    ]
    assert released["after"]["model_host_bytes"] == summary["model_host_bytes"]


def test_restore_completed_layer_installs_packed_modules_without_hessian(
    tmp_path,
) -> None:
    checkpoint_root = tmp_path / "projection-checkpoints"
    offload_root = tmp_path / "offload"
    journal = tmp_path / "error-journal.jsonl"
    family_join = {"source_revision": "test-source", "corpus": "test-corpus"}
    store = EXL3ProjectionCheckpointStore(checkpoint_root)

    class Expert(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Linear(2, 4, bias=False, dtype=torch.bfloat16)
            self.up_proj = nn.Linear(2, 4, bias=False, dtype=torch.bfloat16)
            self.down_proj = nn.Linear(4, 2, bias=False, dtype=torch.bfloat16)

    root = nn.Module()
    root.model = nn.Module()
    root.model.layers = nn.ModuleList([nn.Module()])
    root.model.layers[0].mlp = nn.Module()
    root.model.layers[0].mlp.experts = nn.ModuleList([Expert(), Expert()])
    model = SimpleNamespace(model=root)

    entries = []
    for expert in range(2):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            module_name = f"model.layers.0.mlp.experts.{expert}.{projection}"
            request = build_projection_request(
                module_full_name=module_name,
                layer_index=0,
                input_weight=torch.arange(8, dtype=torch.float32).reshape(4, 2),
                hessian=torch.eye(4, dtype=torch.float32),
                sample_count=32,
                quantizer_contract={"bits": 2, "codebook": "mcg"},
                family_join=family_join,
                route_evidence=None,
            )
            ledger_record = {
                "schema": "ds4rt.exl3-error-ledger",
                "schema_version": 1,
                "record_kind": "projection",
                "module": module_name,
                "processor_layer_index": 0,
                "provenance": {"family_join": family_join},
            }
            record_sha256 = append_exl3_error_journal(journal, ledger_record)
            store.commit(
                request,
                {
                    "trellis": torch.arange(4096, dtype=torch.int16).reshape(1, 1, 4096),
                    "suh": torch.ones(4, dtype=torch.float16),
                    "svh": torch.ones(4, dtype=torch.float16),
                    "mcg": torch.tensor([123], dtype=torch.int32),
                },
                {
                    "duration_seconds": 1.0,
                    "proxy_error": 0.1,
                    "device_names": ["cuda:0"],
                    "quantizer_metrics": {"reported_metric_kind": "test"},
                    "ledger_record": ledger_record,
                    "execution_contract": None,
                    "execution_result": {"kind": "test"},
                },
            )
            entries.append(
                {
                    "module": module_name,
                    "request_sha256": request["request_sha256"],
                    "record_sha256": record_sha256,
                }
            )

    processor = EXL3Processor.__new__(EXL3Processor)
    processor.qcfg = SimpleNamespace(
        meta={
            "ds4rt_error_ledger": {
                "family_join": family_join,
                "run": {
                    "projection_checkpoint": {
                        "contract": CHECKPOINT_CONTRACT,
                        "root": str(checkpoint_root),
                    }
                },
            }
        },
        offload_to_disk=True,
        offload_to_disk_path=str(offload_root),
    )
    processor.error_journal_path = str(journal)
    processor._stats_lock = threading.Lock()
    processor.durations = []
    processor.avg_losses = []
    processor.module_names = []
    processor.log = []

    processor.restore_completed_layer_checkpoints(
        model=model,
        layer_index=0,
        projection_entries=entries,
    )

    restored = [
        root.get_submodule(entry["module"])
        for entry in entries
    ]
    assert all(isinstance(module, ExllamaV3Linear) for module in restored)
    assert all(hasattr(module, "_hf_hook") for module in restored)
    assert len(processor.log) == 6
    assert all(stat["exl3_layer_boundary_restore"] for stat in processor.log)
    assert len(list(offload_root.rglob("module.safetensors"))) == 6
