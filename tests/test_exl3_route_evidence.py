# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
import threading

import pytest
import torch
from torch import nn

from gptqmodel.looper.exllamav3_processor import EXL3Processor
from gptqmodel.utils.exl3_error_ledger import ROUTE_EVIDENCE_SCHEMA


class _Router(nn.Module):
    def forward(self, hidden_states):
        del hidden_states
        logits = torch.zeros((3, 3), dtype=torch.float32)
        weights = torch.tensor(
            [[0.8, 0.2], [0.6, 0.4], [0.7, 0.3]],
            dtype=torch.float32,
        )
        indices = torch.tensor([[0, 1], [1, 2], [0, 2]], dtype=torch.int64)
        return logits, weights, indices


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.gate = _Router()


def _processor_and_subset():
    processor = EXL3Processor.__new__(EXL3Processor)
    processor.qcfg = SimpleNamespace(
        meta={
            "ds4rt_error_ledger": {
                "family_join": {
                    "route_evidence_contract": ROUTE_EVIDENCE_SCHEMA,
                }
            }
        }
    )
    processor._mask_tls = threading.local()
    processor._hooks_paused_tls = threading.local()
    processor.tasks = {}
    subset = {}
    for expert in range(2):
        for projection in ("gate_proj", "up_proj"):
            task_name = f"mlp.experts.{expert}.{projection}"
            processor.tasks[task_name] = {}
            subset[task_name] = SimpleNamespace(full_name=f"model.layers.7.{task_name}")
    return processor, subset


def test_existing_subset_forward_captures_masked_route_count_and_gate_mass():
    processor, subset = _processor_and_subset()
    layer = _Layer()
    processor._mask_tls.value = torch.tensor([[True, False, True]])

    context = processor.subset_forward_capture_context(
        layer_module=layer,
        subset=subset,
    )
    with context:
        layer.mlp.gate(torch.zeros((3, 4)))

    expert0 = processor.tasks["mlp.experts.0.gate_proj"]["route_evidence"]
    expert1 = processor.tasks["mlp.experts.1.up_proj"]["route_evidence"]
    assert expert0["router_calls"] == 1
    assert expert0["router_token_count"] == 2
    assert expert0["router_selected_route_count"] == 4
    assert expert0["router_top_k"] == 2
    assert expert0["expert_route_count"] == 2
    assert expert0["expert_gate_weight_sum"] == pytest.approx(1.5)
    assert expert0["expert_gate_squared_mass"] == pytest.approx(1.13)
    assert expert0["total_gate_weight_sum"] == pytest.approx(2.0)
    assert expert0["total_gate_squared_mass"] == pytest.approx(1.26)
    assert expert0["mask_modes"] == ["filtered"]
    assert processor.tasks["mlp.experts.0.up_proj"]["route_evidence"] == expert0

    assert expert1["expert_route_count"] == 1
    assert expert1["expert_gate_weight_sum"] == pytest.approx(0.2)
    assert expert1["expert_gate_squared_mass"] == pytest.approx(0.04)

    down_subset = {}
    for expert in range(2):
        task_name = f"mlp.experts.{expert}.down_proj"
        processor.tasks[task_name] = {}
        down_subset[task_name] = SimpleNamespace(
            full_name=f"model.layers.7.{task_name}"
        )
    cached_context = processor.subset_forward_capture_context(
        layer_module=layer,
        subset=down_subset,
    )
    with cached_context:
        pass
    assert processor.tasks["mlp.experts.0.down_proj"]["route_evidence"] == expert0
    assert not layer.mlp.gate._forward_hooks


def test_failed_subset_forward_removes_router_hook_without_committing_evidence():
    processor, subset = _processor_and_subset()
    layer = _Layer()
    context = processor.subset_forward_capture_context(
        layer_module=layer,
        subset=subset,
    )
    with pytest.raises(RuntimeError, match="forward failed"):
        with context:
            layer.mlp.gate(torch.zeros((3, 4)))
            raise RuntimeError("forward failed")

    assert not layer.mlp.gate._forward_hooks
    assert all("route_evidence" not in task for task in processor.tasks.values())
