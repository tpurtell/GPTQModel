"""CPU-only byte and failure contracts for streamed scratch payloads."""
import hashlib
import io
from pathlib import Path

import pytest
import torch
import xxhash
from safetensors import safe_open

from gptqmodel.utils import checkpoint_io


def test_streamed_document_preserves_bf16_bytes_without_payload_reread(tmp_path, monkeypatch):
    tensors = {"bf16": torch.arange(24, dtype=torch.bfloat16).reshape(6, 4),
               "indices": torch.tensor([0, 2, 1], dtype=torch.int64),
               "empty": torch.empty((0, 4))}
    path = tmp_path / "weights.safetensors"
    original = Path.open

    def forbid_reads(self, mode="r", *args, **kwargs):
        if "r" in mode:
            raise AssertionError("write-time payload reread")
        return original(self, mode, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", forbid_reads)
        receipt = checkpoint_io.write_safetensors(path, tensors, {"kind": "scratch"})
    assert receipt["hash_algorithm"] == "xxh3-128"
    assert receipt["bytes"] == path.stat().st_size
    assert receipt["xxh3_128"] == xxhash.xxh3_128_hexdigest(path.read_bytes())
    with safe_open(path, framework="pt", device="cpu") as source:
        assert source.metadata() == {"kind": "scratch"}
        for name, value in tensors.items():
            assert torch.equal(source.get_tensor(name), value)


def test_append_short_writes_and_offset_receipts(monkeypatch):
    class ShortWriter(io.BytesIO):
        def write(self, value):
            return super().write(value[:5])

    target = ShortWriter(b"prefix")
    target.seek(0, 2)
    monkeypatch.setattr(checkpoint_io, "BUFFER_BYTES", 7)
    receipt = checkpoint_io.append_safetensors(target, {"value": torch.arange(11)}, hash_algorithm="sha256")
    assert receipt["offset"] == 6
    payload = target.getvalue()[6:]
    assert receipt["bytes"] == len(payload)
    assert receipt["sha256"] == hashlib.sha256(payload).hexdigest()


@pytest.mark.parametrize("offset,length", [(-1, 1), (True, 1), (0, -1), (0, False), (1.2, None)])
def test_payload_ranges_reject_invalid_values(tmp_path, offset, length):
    path = tmp_path / "record"
    path.write_bytes(b"x")
    for read in (checkpoint_io.hash_file, checkpoint_io.read_safetensors):
        with pytest.raises(ValueError, match="nonnegative integer"):
            read(path, offset=offset, length=length)


def test_range_hash_rejects_truncation(tmp_path):
    path = tmp_path / "record"
    path.write_bytes(b"x")
    with pytest.raises(ValueError, match="truncated"):
        checkpoint_io.hash_file(path, length=2)


def test_failed_writer_preserves_prior_publication(tmp_path, monkeypatch):
    path = tmp_path / "record"
    path.write_bytes(b"old")

    def fail(target, *args, **kwargs):
        target.write(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(checkpoint_io, "append_safetensors", fail)
    with pytest.raises(OSError, match="disk full"):
        checkpoint_io.write_safetensors(path, {"value": torch.zeros(1)})
    assert path.read_bytes() == b"old"
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("view", [lambda: torch._neg_view(torch.ones(2)),
                                  lambda: torch.tensor([1 + 2j]).conj()])
def test_unresolved_logical_views_fail_closed(view):
    with pytest.raises(ValueError, match="unresolved"):
        checkpoint_io.append_safetensors(io.BytesIO(), {"view": view()})


def test_non_posix_directory_sync_does_not_open_directory(tmp_path, monkeypatch):
    # Patch this module's capability, not global os.name (which changes
    # pathlib's platform selection throughout pytest).
    monkeypatch.setattr(checkpoint_io, "_DIRECTORY_FSYNC", False)

    def fail_open(*args, **kwargs):
        pytest.fail("unsupported directory handle was opened")

    with monkeypatch.context() as patch:
        patch.setattr(checkpoint_io.os, "open", fail_open)
        checkpoint_io.sync_directory(tmp_path)


def test_non_posix_writer_without_fchmod_still_syncs_file(tmp_path, monkeypatch):
    monkeypatch.setattr(checkpoint_io, "_DIRECTORY_FSYNC", False)
    monkeypatch.setattr(checkpoint_io, "_FCHMOD", None)
    original_sync = checkpoint_io.os.fsync
    synced = []

    def sync(fd):
        synced.append(fd)
        return original_sync(fd)

    monkeypatch.setattr(checkpoint_io.os, "fsync", sync)
    path = tmp_path / "portable.safetensors"
    checkpoint_io.write_safetensors(path, {"value": torch.arange(3)})
    assert len(synced) == 1
    assert torch.equal(checkpoint_io.read_safetensors(path)["value"], torch.arange(3))


@pytest.mark.skipif(not checkpoint_io._DIRECTORY_FSYNC, reason="POSIX directory fsync contract")
def test_posix_directory_sync_failures_propagate(tmp_path, monkeypatch):
    monkeypatch.setattr(checkpoint_io, "_DIRECTORY_FSYNC", True)

    def fail(fd):
        raise OSError("directory sync failed")

    monkeypatch.setattr(checkpoint_io.os, "fsync", fail)
    with pytest.raises(OSError, match="directory sync failed"):
        checkpoint_io.sync_directory(tmp_path)
