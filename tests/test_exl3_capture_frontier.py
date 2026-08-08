# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
import threading

import pytest
import torch

from gptqmodel.utils.exl3_capture_frontier import (
    EXL3CaptureFrontierError,
    EXL3CaptureFrontierStore,
    EXL3CaptureState,
)
from gptqmodel.looper.exllamav3_processor import EXL3Processor


FAMILY_JOIN = {"source": {"revision": "test"}, "corpus": {"sha256": "a" * 64}}


def _subset():
    return {
        "mlp.experts.7.gate_proj": SimpleNamespace(
            full_name="model.layers.3.mlp.experts.7.gate_proj"
        ),
        "mlp.experts.7.up_proj": SimpleNamespace(
            full_name="model.layers.3.mlp.experts.7.up_proj"
        ),
    }


def _states(hessian: torch.Tensor):
    evidence = {"schema": "test-route", "expert": 7, "count": 32}
    return [
        EXL3CaptureState(
            module=named.full_name,
            hessian=hessian.clone(),
            sample_count=32,
            route_evidence=evidence,
        )
        for named in _subset().values()
    ]


def test_capture_frontier_round_trip_deduplicates_gate_up(tmp_path) -> None:
    store = EXL3CaptureFrontierStore(tmp_path / "frontier", family_join=FAMILY_JOIN)
    hessian = torch.eye(4, dtype=torch.float32)
    manifest = store.commit(
        layer_index=3,
        subset_index=1,
        subset_total=4,
        subset=_subset(),
        states=_states(hessian),
    )
    assert manifest["payload_hash_algorithm"] == "xxh3-128"
    assert len({record["hessian"]["file"] for record in manifest["captures"]}) == 1
    assert len(list(store.root.rglob("*.safetensors"))) == 1

    restored = store.restore(
        layer_index=3,
        subset_index=1,
        subset_total=4,
        subset=_subset(),
    )
    assert restored is not None
    gate = restored["model.layers.3.mlp.experts.7.gate_proj"]
    up = restored["model.layers.3.mlp.experts.7.up_proj"]
    assert torch.equal(gate.hessian, hessian)
    assert gate.hessian.data_ptr() == up.hessian.data_ptr()
    assert gate.sample_count == up.sample_count == 32

    store.discard_through(2)
    assert list(store.root.iterdir())
    store.discard_through(3)
    assert list(store.root.iterdir()) == []


def test_capture_frontier_rejects_gate_up_drift(tmp_path) -> None:
    store = EXL3CaptureFrontierStore(tmp_path / "frontier", family_join=FAMILY_JOIN)
    states = _states(torch.eye(4, dtype=torch.float32))
    states[1].hessian[0, 0] += 1
    with pytest.raises(EXL3CaptureFrontierError, match="not identical"):
        store.commit(
            layer_index=3,
            subset_index=1,
            subset_total=4,
            subset=_subset(),
            states=states,
        )


def test_capture_frontier_rejects_payload_tampering(tmp_path) -> None:
    store = EXL3CaptureFrontierStore(tmp_path / "frontier", family_join=FAMILY_JOIN)
    store.commit(
        layer_index=3,
        subset_index=1,
        subset_total=4,
        subset=_subset(),
        states=_states(torch.eye(4, dtype=torch.float32)),
    )
    payload = next(store.root.rglob("*.safetensors"))
    damaged = bytearray(payload.read_bytes())
    damaged[-1] ^= 1
    payload.write_bytes(damaged)
    with pytest.raises(EXL3CaptureFrontierError, match="failed validation"):
        store.restore(
            layer_index=3,
            subset_index=1,
            subset_total=4,
            subset=_subset(),
        )


class _Capture:
    def __init__(self, hessian: torch.Tensor | None, sample_count: int) -> None:
        self.H = hessian
        self.nsamples = sample_count
        self._device_hessian_partials = {}
        self._device_sample_counts = {}
        self._hessian_dirty = False

    def finalize_hessian(self, target_device=None):
        self.H = self.H.to(device=target_device)
        return self.H


def _processor(subset, *, captured: bool) -> EXL3Processor:
    processor = EXL3Processor.__new__(EXL3Processor)
    processor.qcfg = SimpleNamespace(
        meta={"ds4rt_error_ledger": {"family_join": FAMILY_JOIN}}
    )
    processor._stats_lock = threading.Lock()
    processor._natural_route_evidence_cache = {}
    processor.tasks = {}
    for task_name in subset:
        processor.tasks[task_name] = {
            "capture": _Capture(
                torch.eye(4, dtype=torch.float32) if captured else None,
                32 if captured else 0,
            ),
            "route_evidence": {"expert": 7} if captured else None,
        }
    return processor


def test_processor_commits_and_restores_before_replay(tmp_path, monkeypatch) -> None:
    subset = _subset()
    root = tmp_path / "frontier"
    monkeypatch.setenv("GPTQMODEL_EXL3_CAPTURE_FRONTIER", str(root))
    captured = _processor(subset, captured=True)
    captured.commit_subset_capture_frontier(
        layer_index=3,
        subset_index=1,
        subset_total=4,
        subset=subset,
    )
    captured_tasks = list(captured.tasks.values())
    assert (
        captured_tasks[0]["capture"].H.data_ptr()
        == captured_tasks[1]["capture"].H.data_ptr()
    )

    resumed = _processor(subset, captured=False)
    assert resumed.restore_subset_capture_frontier(
        layer_index=3,
        subset_index=1,
        subset_total=4,
        subset=subset,
    )
    for task in resumed.tasks.values():
        capture = task["capture"]
        assert capture.nsamples == 32
        assert torch.equal(capture.H, torch.eye(4, dtype=torch.float32))
        assert capture._hessian_dirty is False
        assert task["route_evidence"] == {"expert": 7}
