# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Durable, content-bound EXL3 quantization error ledgers."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Iterable
from copy import deepcopy
from pathlib import Path
from typing import Any

LEDGER_SCHEMA = "ds4rt.exl3-error-ledger"
LEDGER_SCHEMA_VERSION = 1
LEDGER_FILENAME = "ds4rt-exl3-error-ledger.jsonl"
LEDGER_MANIFEST_FILENAME = "ds4rt-exl3-error-ledger.manifest.json"
JOURNAL_ENV = "GPTQMODEL_EXL3_ERROR_JOURNAL"

_BASE_EXPERT = re.compile(
    r"^(?:model\.)?layers\.(?P<layer>\d+)\.mlp\.experts\."
    r"(?P<expert>\d+)\.(?P<projection>gate_proj|up_proj|down_proj)$"
)
_MTP_EXPERT = re.compile(
    r"^mtp\.(?P<layer>\d+)\.mlp\.experts\."
    r"(?P<expert>\d+)\.(?P<projection>gate_proj|up_proj|down_proj)$"
)
_PROJECTION_NAMES = {
    "gate_proj": "w1",
    "down_proj": "w2",
    "up_proj": "w3",
}


def _finite_json_value(value: Any, path: str = "record") -> Any:
    """Deep-copy a record while rejecting values JSON cannot bind safely."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite EXL3 ledger value at {path}: {value}")
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
    raise TypeError(f"unsupported EXL3 ledger value at {path}: {type(value).__name__}")


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def routed_expert_identity(module_full_name: str) -> dict[str, Any] | None:
    """Map a GPTQModel routed projection name to its stable DS4 identity."""

    for namespace, pattern in (("base", _BASE_EXPERT), ("mtp", _MTP_EXPERT)):
        match = pattern.fullmatch(module_full_name)
        if match is None:
            continue
        return {
            "block_namespace": namespace,
            "logical_layer": int(match.group("layer")),
            "expert": int(match.group("expert")),
            "projection": _PROJECTION_NAMES[match.group("projection")],
        }
    return None


def build_projection_record(
    *,
    module_full_name: str,
    layer_index: int | None,
    bits: int,
    codebook: str,
    sample_count: int,
    duration_seconds: float,
    encoded_bytes: int,
    device_names: list[str],
    quantizer_metrics: dict[str, Any],
    provenance: dict[str, Any] | None,
) -> dict[str, Any]:
    """Construct one projection record without discarding raw error terms."""

    record = {
        "schema": LEDGER_SCHEMA,
        "schema_version": LEDGER_SCHEMA_VERSION,
        "record_kind": "projection",
        "module": module_full_name,
        "processor_layer_index": layer_index,
        "bits": int(bits),
        "codebook": str(codebook),
        "sample_count": int(sample_count),
        "duration_seconds": float(duration_seconds),
        "encoded_bytes": int(encoded_bytes),
        "devices": [str(device) for device in device_names],
        "quantizer_metrics": deepcopy(quantizer_metrics),
        "provenance": deepcopy(provenance) if provenance is not None else None,
    }
    identity = routed_expert_identity(module_full_name)
    if identity is not None:
        record.update(identity)
    return _finite_json_value(record)


def _family_record(records: list[dict[str, Any]]) -> dict[str, Any]:
    first = records[0]
    reconstructions = [
        record["quantizer_metrics"]["reconstruction"] for record in records
    ]
    error_sum_sq = sum(item["error_sum_sq"] for item in reconstructions)
    reference_sum_sq = sum(item["reference_sum_sq"] for item in reconstructions)
    element_count = sum(item["element_count"] for item in reconstructions)

    hessian_numerators = [
        record["quantizer_metrics"].get("hessian_weighted_error_numerator")
        for record in records
    ]
    hessian_denominators = [
        record["quantizer_metrics"].get("hessian_weighted_reference_denominator")
        for record in records
    ]
    hessian_complete = all(
        value is not None for value in hessian_numerators + hessian_denominators
    )
    hessian_numerator = sum(hessian_numerators) if hessian_complete else None
    hessian_denominator = sum(hessian_denominators) if hessian_complete else None

    return _finite_json_value(
        {
            "schema": LEDGER_SCHEMA,
            "schema_version": LEDGER_SCHEMA_VERSION,
            "record_kind": "expert_family",
            "block_namespace": first["block_namespace"],
            "logical_layer": first["logical_layer"],
            "expert": first["expert"],
            "bits": first["bits"],
            "codebook": first["codebook"],
            "projections": [record["projection"] for record in records],
            "projection_modules": [record["module"] for record in records],
            "sample_counts": [record["sample_count"] for record in records],
            "duration_seconds": sum(record["duration_seconds"] for record in records),
            "encoded_bytes": sum(record["encoded_bytes"] for record in records),
            "provenance": {
                "family_join": deepcopy(_family_join_provenance(first)),
                "projections": {
                    record["projection"]: deepcopy(record.get("provenance"))
                    for record in records
                },
            },
            "aggregate_metrics": {
                "error_sum_sq": error_sum_sq,
                "reference_sum_sq": reference_sum_sq,
                "element_count": element_count,
                "mse": error_sum_sq / max(element_count, 1),
                "nmse": error_sum_sq / max(reference_sum_sq, 1e-20),
                "relative_frobenius": math.sqrt(
                    max(error_sum_sq / max(reference_sum_sq, 1e-20), 0.0)
                ),
                "max_abs_error": max(item["max_abs_error"] for item in reconstructions),
                "hessian_weighted_error_numerator": hessian_numerator,
                "hessian_weighted_reference_denominator": hessian_denominator,
                "hessian_weighted_relative_error": (
                    hessian_numerator / max(hessian_denominator, 1e-8)
                    if hessian_complete
                    else None
                ),
                "hessian_metric_status": "ok" if hessian_complete else "incomplete",
            },
        }
    )


def _family_join_provenance(record: dict[str, Any]) -> Any:
    provenance = record.get("provenance")
    if isinstance(provenance, dict) and "family_join" in provenance:
        return provenance["family_join"]
    return provenance


def derive_family_records(
    projection_records: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Join complete w1/w2/w3 projection triples into expert-family records."""

    groups: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
    for record in projection_records:
        required = ("block_namespace", "logical_layer", "expert", "projection")
        if any(field not in record for field in required):
            continue
        key = (
            record["block_namespace"],
            record["logical_layer"],
            record["expert"],
            record["bits"],
            record["codebook"],
            _canonical_json_bytes(_family_join_provenance(record)),
        )
        projections = groups.setdefault(key, {})
        projection = record["projection"]
        if projection in projections:
            raise ValueError(
                "duplicate EXL3 ledger projection for "
                f"{key[0]} layer {key[1]} expert {key[2]} {projection}"
            )
        projections[projection] = record

    families = []
    for projections in groups.values():
        if set(projections) != {"w1", "w2", "w3"}:
            continue
        families.append(
            _family_record([projections[name] for name in ("w1", "w2", "w3")])
        )
    return families


def _record_sort_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        record.get("block_namespace", "unknown"),
        record.get("logical_layer", -1),
        record.get("expert", -1),
        0 if record["record_kind"] == "projection" else 1,
        record.get("projection", ""),
        record.get("bits", -1),
        record.get("module", ""),
    )


def _bind_record(record: dict[str, Any]) -> dict[str, Any]:
    clean = _finite_json_value(record)
    digest = hashlib.sha256(_canonical_json_bytes(clean)).hexdigest()
    return {**clean, "record_sha256": digest}


def append_exl3_error_journal(
    journal_path: str | os.PathLike[str],
    projection_record: dict[str, Any],
) -> str:
    """Append and fsync one bound projection before its tensors may commit."""

    if projection_record.get("record_kind") != "projection":
        raise ValueError("EXL3 error journal accepts projection records only")
    bound = _bind_record(projection_record)
    payload = _canonical_json_bytes(bound) + b"\n"
    path = Path(journal_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existed = path.exists()
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        written = os.write(descriptor, payload)
        if written != len(payload):
            raise OSError(
                f"short EXL3 error-journal write: expected {len(payload)}, wrote {written}"
            )
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if not existed:
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    return bound["record_sha256"]


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def write_exl3_error_ledger(
    save_dir: str | os.PathLike[str],
    projection_records: Iterable[dict[str, Any]],
) -> dict[str, Any] | None:
    """Atomically persist projections, family joins, and a digest manifest."""

    projections = [
        _finite_json_value(record)
        for record in projection_records
        if record.get("record_kind") == "projection"
        and record.get("schema") == LEDGER_SCHEMA
        and record.get("schema_version") == LEDGER_SCHEMA_VERSION
    ]
    if not projections:
        return None

    families = derive_family_records(projections)
    records = [_bind_record(record) for record in projections + families]
    records.sort(key=_record_sort_key)
    ledger_payload = b"".join(
        _canonical_json_bytes(record) + b"\n" for record in records
    )
    ledger_sha256 = hashlib.sha256(ledger_payload).hexdigest()

    save_path = Path(save_dir)
    ledger_path = save_path / LEDGER_FILENAME
    manifest_path = save_path / LEDGER_MANIFEST_FILENAME
    _atomic_write(ledger_path, ledger_payload)

    manifest = {
        "schema": LEDGER_SCHEMA,
        "schema_version": LEDGER_SCHEMA_VERSION,
        "ledger": LEDGER_FILENAME,
        "ledger_sha256": ledger_sha256,
        "projection_records": len(projections),
        "complete_family_records": len(families),
        "total_records": len(records),
    }
    _atomic_write(manifest_path, _canonical_json_bytes(manifest) + b"\n")
    return manifest


__all__ = [
    "JOURNAL_ENV",
    "LEDGER_FILENAME",
    "LEDGER_MANIFEST_FILENAME",
    "LEDGER_SCHEMA",
    "LEDGER_SCHEMA_VERSION",
    "append_exl3_error_journal",
    "build_projection_record",
    "derive_family_records",
    "routed_expert_identity",
    "write_exl3_error_ledger",
]
