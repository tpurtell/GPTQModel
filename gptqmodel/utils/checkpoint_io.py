# SPDX-License-Identifier: Apache-2.0
"""Bounded, streamed scratch payload I/O; durability belongs to the commit boundary.

The caller owns tensor immutability until append_safetensors returns. CPU tensor
storage is borrowed only while writing; no packed payload is expanded or cached.
"""
from __future__ import annotations

import ctypes
import hashlib
import threading
import sys
import json
import os
import struct
import tempfile
from pathlib import Path

import xxhash
from safetensors.torch import load as load_safetensors
from safetensors.torch import _TYPES

HASH_ALGORITHM = "xxh3-128"
BUFFER_BYTES = 8 * 1024 * 1024
# Use the installed format's dtype inventory, including native FP8 scales.
_DTYPES = {str(dtype): name for name, dtype in _TYPES.items()}
_WRITER_LOCK = threading.RLock()


def _validate_range(offset, length):
    if type(offset) is not int or offset < 0 or (length is not None and (type(length) is not int or length < 0)):
        raise ValueError("checkpoint ranges require nonnegative integer offsets/lengths")



def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def durable_makedirs(path):
    """Publish newly created directory ancestry once, outside the batch loop."""
    path = Path(path)
    missing = []
    cursor = path
    while not cursor.exists():
        missing.append(cursor)
        cursor = cursor.parent
    for directory in reversed(missing):
        directory.mkdir(exist_ok=True)
        sync_directory(directory)
        sync_directory(directory.parent)


def hash_file(path, *, offset=0, length=None):
    _validate_range(offset, length)
    digest = xxhash.xxh3_128()
    with Path(path).open("rb") as source:
        source.seek(offset)
        remaining = length
        while remaining is None or remaining:
            chunk = source.read(BUFFER_BYTES if remaining is None else min(BUFFER_BYTES, remaining))
            if not chunk:
                if remaining:
                    raise ValueError("truncated checkpoint payload")
                break
            digest.update(chunk)
            if remaining is not None:
                remaining -= len(chunk)
    return digest.hexdigest()


def _append_safetensors(target, tensors, metadata=None, *, hash_algorithm=HASH_ALGORITHM):
    """Append one independently decodable document, hashing its write stream.

    Header memory scales with tensor count; payload views are at most 8 MiB.
    Unlike safetensors.torch.save, this never creates a whole-payload bytes copy.
    The caller supplies already quiesced contiguous CPU tensors, as save_file
    does. Borrowing storage through ctypes avoids new Torch conversion kernels
    or NumPy dtype conversions (in particular for BF16/FP8).
    """
    if sys.byteorder != "little":
        raise ValueError("streamed safetensors require little-endian CPU storage")
    if hash_algorithm not in {HASH_ALGORITHM, "sha256"}:
        raise ValueError("unsupported checkpoint hash algorithm")
    header = {}
    cursor = 0
    for name, tensor in sorted(tensors.items()):
        if not isinstance(name, str) or not name or name == "__metadata__":
            raise ValueError("invalid checkpoint tensor name")
        if tensor.device.type != "cpu" or not tensor.is_contiguous():
            raise ValueError("streamed checkpoints require contiguous CPU tensors")
        if tensor.is_conj() or tensor.is_neg():
            raise ValueError("checkpoint tensor has an unresolved conjugate/negative view")
        size = tensor.numel() * tensor.element_size()
        try:
            dtype = _DTYPES[str(tensor.dtype)]
        except KeyError as exc:
            raise ValueError(f"unsupported checkpoint dtype: {tensor.dtype}") from exc
        header[name] = {"dtype": dtype, "shape": list(tensor.shape), "data_offsets": [cursor, cursor + size]}
        cursor += size
    if metadata is not None:
        if not all(isinstance(k, str) and isinstance(v, str) for k, v in metadata.items()):
            raise ValueError("safetensors metadata must contain strings")
        header["__metadata__"] = metadata
    encoded = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode()
    encoded += b" " * (-len(encoded) % 8)
    offset = target.tell()
    digest = xxhash.xxh3_128() if hash_algorithm == HASH_ALGORITHM else hashlib.sha256()

    def write(chunk):
        # Buffered files normally write all bytes, but also support raw writers.
        view = memoryview(chunk)
        while view:
            count = target.write(view)
            if count is None or count <= 0:
                raise OSError("checkpoint writer made no progress")
            digest.update(view[:count])
            view = view[count:]

    write(struct.pack("<Q", len(encoded)))
    write(encoded)
    specs = {}
    for name, tensor in sorted(tensors.items()):
        size = tensor.numel() * tensor.element_size()
        tensor_hash = xxhash.xxh3_128()
        address = tensor.data_ptr()
        for start in range(0, size, BUFFER_BYTES):
            # The tensor retains ownership of its storage for the entire call.
            chunk = memoryview((ctypes.c_ubyte * min(BUFFER_BYTES, size - start)).from_address(address + start)).cast("B")
            tensor_hash.update(chunk)
            write(chunk)
        specs[name] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "numel": tensor.numel(), "bytes": size,
                       "hash_algorithm": HASH_ALGORITHM, "xxh3_128": tensor_hash.hexdigest()}
    return {"offset": offset, "bytes": target.tell() - offset, "hash_algorithm": hash_algorithm,
            ("xxh3_128" if hash_algorithm == HASH_ALGORITHM else "sha256"): digest.hexdigest(), "tensors": specs}


def append_safetensors(target, tensors, metadata=None, *, hash_algorithm=HASH_ALGORITHM):
    # Serialize CPU storage metadata access as well as writer ownership. No
    # free-threaded Torch/GPU qualification is implied by this I/O helper.
    with _WRITER_LOCK:
        return _append_safetensors(target, tensors, metadata, hash_algorithm=hash_algorithm)


def write_safetensors(path, tensors, metadata=None, *, sync=True, hash_algorithm=HASH_ALGORITHM):
    """Atomically write a regular document; sync=False requires a later barrier."""
    path = Path(path)
    durable_makedirs(path.parent)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as target:
            receipt = append_safetensors(target, tensors, metadata, hash_algorithm=hash_algorithm)
            target.flush()
            os.fchmod(target.fileno(), 0o644)
            if sync:
                os.fsync(target.fileno())
        os.replace(temporary, path)
        if sync:
            sync_directory(path.parent)
        return receipt
    finally:
        Path(temporary).unlink(missing_ok=True)


def read_safetensors(path, *, offset=0, length=None):
    """Decode one document (including documents inside a sealed segment)."""
    _validate_range(offset, length)
    with Path(path).open("rb") as source:
        source.seek(offset)
        payload = source.read() if length is None else source.read(length)
    if length is not None and len(payload) != length:
        raise ValueError("truncated checkpoint payload")
    return load_safetensors(payload)
