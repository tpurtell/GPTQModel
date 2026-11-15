from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file as save_safetensors_file

from gptqmodel.utils.exl3_capture_batch_spool import (
    EXL3CaptureBatchSpool,
    EXL3CaptureBatchSpoolError,
)


def _open(tmp_path, *, checkpoint_interval=1):
    return EXL3CaptureBatchSpool(
        tmp_path / "spool",
        layer_index=4,
        subset_index=0,
        subset_total=2,
        expected_batches=3,
        payload_contract="test.exl3-capture-payload-v1",
        phase="gate-up",
        module_names=["model.layers.4.mlp.experts.0.gate_proj"],
        provenance={"plan_sha256": "a" * 64},
        ownership={"0": "cuda:0"},
        checkpoint_interval=checkpoint_interval,
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


def test_capture_batch_spool_checkpoints_bounded_groups(tmp_path) -> None:
    spool = _open(tmp_path, checkpoint_interval=2)
    spool.commit(
        0,
        tensors={"router_input": torch.zeros(2, 4, dtype=torch.bfloat16)},
        metadata={"batch": 0},
    )
    assert spool.committed_indices == frozenset()
    assert spool.pending_indices == frozenset({0})

    spool.commit(
        1,
        tensors={"router_input": torch.ones(2, 4, dtype=torch.bfloat16)},
        metadata={"batch": 1},
    )
    assert spool.committed_indices == frozenset({0, 1})
    assert spool.pending_indices == frozenset()

    # The final short group is forced durable even though it does not fill the
    # configured interval.
    spool.commit(
        2,
        tensors={"router_input": torch.full((2, 4), 2, dtype=torch.bfloat16)},
        metadata={"batch": 2},
    )
    assert spool.committed_indices == frozenset({0, 1, 2})
    resumed = _open(tmp_path, checkpoint_interval=3)
    assert resumed.committed_indices == frozenset({0, 1, 2})


def test_capture_batch_spool_discards_uncheckpointed_group(tmp_path) -> None:
    spool = _open(tmp_path, checkpoint_interval=3)
    spool.commit(
        0,
        tensors={"router_input": torch.zeros(2, 4, dtype=torch.bfloat16)},
        metadata={"batch": 0},
    )
    assert spool.pending_indices == frozenset({0})

    resumed = _open(tmp_path, checkpoint_interval=3)
    assert resumed.committed_indices == frozenset()
    assert resumed.pending_indices == frozenset()
    assert not (resumed.directory / "batch-000000000.safetensors").exists()


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


def test_capture_batch_spool_payload_contract_changes_storage_identity(tmp_path) -> None:
    original = _open(tmp_path)
    replacement = EXL3CaptureBatchSpool(
        tmp_path / "spool",
        layer_index=4,
        subset_index=0,
        subset_total=2,
        expected_batches=3,
        payload_contract="test.exl3-capture-payload-v2",
        phase="gate-up",
        module_names=["model.layers.4.mlp.experts.0.gate_proj"],
        provenance={"plan_sha256": "a" * 64},
        ownership={"0": "cuda:0"},
    )

    assert replacement.directory != original.directory
    assert replacement.committed_indices == frozenset()
    assert not original.directory.exists()


def test_capture_batch_spool_retries_after_checkpoint_failure(
    tmp_path, monkeypatch
) -> None:
    import gptqmodel.utils.exl3_capture_batch_spool as spool_module

    spool = _open(tmp_path)
    original_atomic_json = spool_module._atomic_json

    def fail(path, body):
        raise OSError("disk full")

    monkeypatch.setattr(spool_module, "_atomic_json", fail)
    with pytest.raises(OSError, match="disk full"):
        spool.commit(
            0,
            tensors={"router_input": torch.zeros(2, 4, dtype=torch.bfloat16)},
            metadata={"batch": 0},
        )
    assert spool.pending_indices == frozenset()
    assert not (spool.directory / "batch-000000000.safetensors").exists()

    monkeypatch.setattr(spool_module, "_atomic_json", original_atomic_json)
    record = spool.commit(
        0,
        tensors={"router_input": torch.zeros(2, 4, dtype=torch.bfloat16)},
        metadata={"batch": 0},
    )
    assert record["batch_index"] == 0
    assert (spool.directory / record["file"]).exists()
    assert spool.committed_indices == frozenset({0})


def test_capture_group_uses_one_payload_sync_before_manifest(tmp_path, monkeypatch):
    import os
    import gptqmodel.utils.exl3_capture_batch_spool as module

    spool = _open(tmp_path, checkpoint_interval=2)
    events = []
    original_sync = os.fsync
    original_json = module._atomic_json

    def sync(fd):
        name = os.readlink(f"/proc/self/fd/{fd}")
        if name.endswith(".bin"):
            events.append("payload")
        return original_sync(fd)

    def manifest(path, body):
        events.append("manifest")
        return original_json(path, body)

    monkeypatch.setattr(os, "fsync", sync)
    monkeypatch.setattr(module, "_atomic_json", manifest)
    monkeypatch.setattr(module, "hash_file", lambda *a, **k: pytest.fail("write-time reread"))
    records = [spool.commit(i, tensors={"hidden": torch.full((2, 4), i)}, metadata={}) for i in (0, 1)]
    assert records[0]["file"] == records[1]["file"]
    assert records[1]["offset"] == records[0]["bytes"]
    assert events == ["payload", "manifest"]


def test_direct_checkpoint_reconciles_failure_after_manifest_rename(tmp_path, monkeypatch):
    import gptqmodel.utils.exl3_capture_batch_spool as module

    spool = _open(tmp_path, checkpoint_interval=3)
    first = spool.commit(0, tensors={"hidden": torch.zeros(1)}, metadata={})
    original = module._atomic_json

    def after_rename(path, body):
        original(path, body)
        raise OSError("directory sync failed after rename")

    with monkeypatch.context() as patch:
        patch.setattr(module, "_atomic_json", after_rename)
        with pytest.raises(OSError, match="after rename"):
            spool.checkpoint()
    assert spool.committed_indices == frozenset({0})
    second = spool.commit(1, tensors={"hidden": torch.ones(1)}, metadata={})
    assert first["file"] != second["file"]
    spool.commit(2, tensors={"hidden": torch.ones(1)}, metadata={})
    assert _open(tmp_path).committed_indices == frozenset({0, 1, 2})


@pytest.mark.parametrize("mutation", ["overlap", "tail"])
def test_segment_restore_rejects_overlapping_ranges_and_unbound_tail(tmp_path, mutation):
    spool = _open(tmp_path, checkpoint_interval=2)
    first = spool.commit(0, tensors={"hidden": torch.zeros(1)}, metadata={})
    spool.commit(1, tensors={"hidden": torch.zeros(1)}, metadata={})
    if mutation == "tail":
        with (spool.directory / first["file"]).open("ab") as stream:
            stream.write(b"orphan tail")
    else:
        path = spool.directory / "progress.json"
        body = json.loads(path.read_text())
        body["records"][1]["offset"] = 0
        path.write_text(json.dumps(body))
    with pytest.raises(EXL3CaptureBatchSpoolError):
        _open(tmp_path)


def test_legacy_v1_capture_standalone_payload_is_still_readable(tmp_path):
    import gptqmodel.utils.exl3_capture_batch_spool as module

    spool = _open(tmp_path)
    tensor = torch.arange(4, dtype=torch.bfloat16)
    path = spool.directory / "batch-000000000.safetensors"
    save_safetensors_file({"hidden": tensor}, path)
    body = json.loads((spool.directory / "progress.json").read_text())
    body["schema_version"] = 1
    body["records"] = [{"batch_index": 0, "file": path.name, "bytes": path.stat().st_size,
                        "xxh3_128": module.hash_file(path), "tensors": {"hidden": module._tensor_spec(tensor)}, "metadata": {}}]
    (spool.directory / "progress.json").write_text(json.dumps(body))
    restored = _open(tmp_path)
    assert torch.equal(restored.load(0)[0]["hidden"], tensor)
    with path.open("ab") as stream:
        stream.write(b"x")
    with pytest.raises(EXL3CaptureBatchSpoolError, match="payload differs"):
        _open(tmp_path)
