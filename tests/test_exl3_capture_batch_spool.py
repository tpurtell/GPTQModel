from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file as save_safetensors_file

from gptqmodel.utils.exl3_capture_batch_spool import (
    EXL3CaptureBatchSpool,
    EXL3CaptureBatchSpoolError,
)


def _open(tmp_path):
    return EXL3CaptureBatchSpool(
        tmp_path / "spool",
        layer_index=4,
        subset_index=0,
        subset_total=2,
        expected_batches=3,
        phase="gate-up",
        module_names=["model.layers.4.mlp.experts.0.gate_proj"],
        provenance={"plan_sha256": "a" * 64},
        ownership={"0": "cuda:0"},
    )


def test_capture_batch_spool_round_trip_and_resume(tmp_path) -> None:
    spool = _open(tmp_path)
    spool.commit(
        1,
        tensors={
            "router_input": torch.arange(12, dtype=torch.bfloat16).reshape(3, 4),
            "top_indices": torch.tensor([[0], [1], [0]]),
        },
        metadata={"corpus_batch_sha256": "b" * 64},
    )
    resumed = _open(tmp_path)
    assert resumed.committed_indices == frozenset({1})
    tensors, metadata = resumed.load(1)
    assert torch.equal(tensors["router_input"], torch.arange(12).reshape(3, 4))
    assert metadata == {"corpus_batch_sha256": "b" * 64}


def test_capture_batch_spool_drops_interrupted_temporary(tmp_path) -> None:
    spool = _open(tmp_path)
    partial = spool.directory / ".batch-000000000.safetensors.dead.tmp"
    partial.write_bytes(b"partial")
    resumed = _open(tmp_path)
    assert resumed.committed_indices == frozenset()
    assert not partial.exists()


def test_capture_batch_spool_drops_unmanifested_renamed_payload(tmp_path) -> None:
    spool = _open(tmp_path)
    partial = spool.directory / "batch-000000000.safetensors"
    save_safetensors_file({"router_input": torch.zeros(1, 4)}, partial)
    resumed = _open(tmp_path)
    assert resumed.committed_indices == frozenset()
    assert not partial.exists()


def test_capture_batch_spool_rejects_payload_tampering(tmp_path) -> None:
    spool = _open(tmp_path)
    record = spool.commit(
        0,
        tensors={"router_input": torch.zeros(2, 4, dtype=torch.bfloat16)},
        metadata={"batch": 0},
    )
    (spool.directory / record["file"]).write_bytes(b"corrupt")
    with pytest.raises(EXL3CaptureBatchSpoolError, match="payload differs"):
        _open(tmp_path)


def test_capture_batch_spool_rejects_identity_drift(tmp_path) -> None:
    spool = _open(tmp_path)
    progress_path = spool.directory / "progress.json"
    progress = json.loads(progress_path.read_text())
    progress["capture_key"]["phase"] = "down"
    progress_path.write_text(json.dumps(progress))
    with pytest.raises(EXL3CaptureBatchSpoolError, match="identity differs"):
        _open(tmp_path)
