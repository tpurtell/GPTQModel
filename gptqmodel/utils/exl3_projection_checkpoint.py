# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Atomic, content-addressed checkpoints for completed EXL3 projections."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors


CHECKPOINT_SCHEMA = "ds4rt.exl3-projection-checkpoint"
CHECKPOINT_SCHEMA_VERSION = 1
CHECKPOINT_CONTRACT = f"{CHECKPOINT_SCHEMA}-v{CHECKPOINT_SCHEMA_VERSION}"


def _finite_json_value(value: Any, path: str = "value") -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite EXL3 checkpoint value at {path}: {value}")
        return value
    if isinstance(value, dict):
        return {
            str(key): _finite_json_value(item, f"{path}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _finite_json_value(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(
        f"unsupported EXL3 checkpoint value at {path}: {type(value).__name__}"
    )


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_identity(tensor: torch.Tensor) -> dict[str, Any]:
    """Hash logical tensor bytes together with their exact shape and dtype."""

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("EXL3 checkpoint identities require torch tensors")
    host = tensor.detach().contiguous().to(device="cpu")
    byte_view = host.reshape(-1).view(torch.uint8).numpy()
    digest = hashlib.sha256(memoryview(byte_view)).hexdigest()
    return {
        "shape": list(host.shape),
        "dtype": str(host.dtype),
        "numel": host.numel(),
        "bytes": host.numel() * host.element_size(),
        "sha256": digest,
    }


def build_projection_request(
    *,
    module_full_name: str,
    layer_index: int | None,
    input_weight: torch.Tensor,
    hessian: torch.Tensor,
    sample_count: int,
    quantizer_contract: dict[str, Any],
    family_join: dict[str, Any] | None,
    route_evidence: dict[str, Any] | None,
) -> dict[str, Any]:
    """Bind every input capable of changing one packed projection result."""

    request = {
        "schema": CHECKPOINT_SCHEMA,
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "module": module_full_name,
        "processor_layer_index": layer_index,
        "sample_count": int(sample_count),
        "input_weight": tensor_identity(input_weight),
        "hessian": tensor_identity(hessian),
        "quantizer_contract": deepcopy(quantizer_contract),
        "family_join": deepcopy(family_join),
        "route_evidence": deepcopy(route_evidence),
    }
    clean = _finite_json_value(request, "request")
    clean["request_sha256"] = sha256_bytes(canonical_json_bytes(clean))
    return clean


def checkpoint_root_from_provenance(
    provenance: dict[str, Any] | None,
) -> Path | None:
    """Resolve the opt-in production checkpoint root from run provenance."""

    if not isinstance(provenance, dict):
        return None
    run = provenance.get("run")
    if not isinstance(run, dict):
        return None
    checkpoint = run.get("projection_checkpoint")
    if checkpoint is None:
        return None
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("contract") != CHECKPOINT_CONTRACT
        or not isinstance(checkpoint.get("root"), str)
        or not checkpoint["root"]
    ):
        raise ValueError("invalid EXL3 projection-checkpoint run contract")
    return Path(checkpoint["root"]).expanduser().resolve()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as target:
            target.write(payload)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class EXL3ProjectionCheckpointStore:
    """Publish and validate immutable packed projection results."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root).expanduser().resolve()

    def _paths(self, request_sha256: str) -> tuple[Path, Path]:
        if len(request_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in request_sha256
        ):
            raise ValueError("invalid EXL3 checkpoint request digest")
        prefix = self.root / request_sha256[:2] / request_sha256[2:4]
        return (
            prefix / f"{request_sha256}.json",
            prefix / f"{request_sha256}.safetensors",
        )

    def load(
        self,
        request: dict[str, Any],
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]] | None:
        request_sha256 = request.get("request_sha256")
        manifest_path, tensor_path = self._paths(str(request_sha256))
        if not manifest_path.exists():
            return None
        if (
            not manifest_path.is_file()
            or manifest_path.is_symlink()
            or not tensor_path.is_file()
            or tensor_path.is_symlink()
        ):
            raise ValueError("EXL3 checkpoint contains a non-regular committed path")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError("cannot read EXL3 projection checkpoint") from error
        if not isinstance(manifest, dict):
            raise ValueError("EXL3 projection checkpoint manifest is not an object")
        manifest_digest = manifest.pop("manifest_sha256", None)
        if (
            manifest.get("schema") != CHECKPOINT_SCHEMA
            or manifest.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
            or manifest.get("request") != request
            or manifest.get("request_sha256") != request_sha256
            or manifest.get("tensor_file") != tensor_path.name
            or manifest_digest != sha256_bytes(canonical_json_bytes(manifest))
            or manifest.get("tensor_sha256") != sha256_file(tensor_path)
        ):
            raise ValueError("EXL3 projection checkpoint failed content validation")
        payload = tensor_path.read_bytes()
        try:
            tensors = load_safetensors(payload)
        except Exception as error:
            raise ValueError(
                "cannot decode EXL3 projection checkpoint tensors"
            ) from error
        tensor_specs = manifest.get("tensors")
        if not isinstance(tensor_specs, dict) or set(tensor_specs) != set(tensors):
            raise ValueError("EXL3 projection checkpoint tensor set is inconsistent")
        for name, tensor in tensors.items():
            if tensor_specs[name] != tensor_identity(tensor):
                raise ValueError(
                    f"EXL3 projection checkpoint tensor `{name}` is inconsistent"
                )
        result = manifest.get("result")
        if not isinstance(result, dict):
            raise ValueError("EXL3 projection checkpoint result is malformed")
        return tensors, result

    def commit(
        self,
        request: dict[str, Any],
        tensors: dict[str, torch.Tensor],
        result: dict[str, Any],
    ) -> dict[str, Any]:
        request_sha256 = request.get("request_sha256")
        manifest_path, tensor_path = self._paths(str(request_sha256))
        existing = self.load(request)
        if existing is not None:
            existing_tensors, existing_result = existing
            if existing_result != result or any(
                not torch.equal(existing_tensors[name], tensor.detach().cpu())
                for name, tensor in tensors.items()
            ):
                raise ValueError("EXL3 projection checkpoint collision")
            return existing_result

        host_tensors = {
            str(name): tensor.detach().contiguous().to(device="cpu")
            for name, tensor in tensors.items()
        }
        if not host_tensors:
            raise ValueError("EXL3 projection checkpoint cannot commit no tensors")
        tensor_payload = save_safetensors(host_tensors)
        clean_result = _finite_json_value(result, "result")
        manifest = {
            "schema": CHECKPOINT_SCHEMA,
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "request": deepcopy(request),
            "request_sha256": request_sha256,
            "tensor_file": tensor_path.name,
            "tensor_sha256": sha256_bytes(tensor_payload),
            "tensors": {
                name: tensor_identity(tensor) for name, tensor in host_tensors.items()
            },
            "result": clean_result,
        }
        manifest["manifest_sha256"] = sha256_bytes(canonical_json_bytes(manifest))
        _atomic_write(tensor_path, tensor_payload)
        _atomic_write(manifest_path, canonical_json_bytes(manifest) + b"\n")
        _fsync_directory(manifest_path.parent)
        loaded = self.load(request)
        if loaded is None or loaded[1] != clean_result:
            raise RuntimeError("EXL3 projection checkpoint did not commit durably")
        return clean_result


__all__ = [
    "CHECKPOINT_CONTRACT",
    "CHECKPOINT_SCHEMA",
    "CHECKPOINT_SCHEMA_VERSION",
    "EXL3ProjectionCheckpointStore",
    "build_projection_request",
    "canonical_json_bytes",
    "checkpoint_root_from_provenance",
    "tensor_identity",
]
