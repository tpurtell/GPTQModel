# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json

import pytest
from gptqmodel.utils.exl3_error_ledger import (
    LEDGER_FILENAME,
    LEDGER_MANIFEST_FILENAME,
    ROUTE_EVIDENCE_SCHEMA,
    append_exl3_error_journal,
    build_projection_record,
    derive_family_records,
    routed_expert_identity,
    write_exl3_error_ledger,
)


def _metrics(scale: float) -> dict:
    return {
        "schema": "gptqmodel.exl3-trellis-error",
        "schema_version": 1,
        "quantizer_path": "hessian_ldlq",
        "reported_metric_kind": "hessian_weighted_relative_error",
        "reported_metric_value": 0.1 * scale,
        "hessian_weighted_error_numerator": 2.0 * scale,
        "hessian_weighted_reference_denominator": 20.0 * scale,
        "hessian_weighted_relative_error": 0.1,
        "hessian_metric_status": "ok",
        "selected_global_scale": 0.9,
        "scale_search_mse": 0.01,
        "apply_out_scales": True,
        "reconstruction": {
            "domain": "regularized_exl3_search_space",
            "shape": [16, 16],
            "element_count": 256,
            "error_sum_sq": 4.0 * scale,
            "reference_sum_sq": 40.0 * scale,
            "mse": 4.0 * scale / 256,
            "nmse": 0.1,
            "relative_frobenius": 0.1**0.5,
            "mean_abs_error": 0.01,
            "max_abs_error": 0.2 * scale,
            "reference_finite": True,
            "error_finite": True,
            "tile_shape": [16, 16],
            "tile_count": 1,
            "tile_sse_sum": 4.0 * scale,
            "tile_sse_max": 4.0 * scale,
            "tile_sse_percentiles": {
                "p50": 4.0 * scale,
                "p90": 4.0 * scale,
                "p99": 4.0 * scale,
                "p99_9": 4.0 * scale,
            },
            "worst_tiles": [{"row": 0, "column": 0, "sse": 4.0 * scale}],
        },
    }


def _route_evidence(scale: float = 1.0) -> dict:
    route_count = 1024
    selected_count = 8192
    gate_sum = 100.0 * scale
    gate_sq = 20.0 * scale
    total_gate_sum = 1000.0 * scale
    total_gate_sq = 200.0 * scale
    return {
        "schema": ROUTE_EVIDENCE_SCHEMA,
        "schema_version": 1,
        "block_namespace": "base",
        "logical_layer": 7,
        "expert": 31,
        "router_calls": 8,
        "router_token_count": 1024,
        "router_selected_route_count": selected_count,
        "router_top_k": 8,
        "expert_route_count": route_count,
        "expert_gate_weight_sum": gate_sum,
        "expert_gate_squared_mass": gate_sq,
        "total_gate_weight_sum": total_gate_sum,
        "total_gate_squared_mass": total_gate_sq,
        "expert_route_fraction": route_count / selected_count,
        "expert_gate_weight_mass_fraction": gate_sum / total_gate_sum,
        "expert_gate_squared_mass_fraction": gate_sq / total_gate_sq,
        "expert_gate_weight_mean": gate_sum / route_count,
        "expert_gate_weight_rms": (gate_sq / route_count) ** 0.5,
        "router_weight_dtypes": ["torch.float32"],
        "mask_modes": ["all-valid"],
    }


def _record(
    projection: str,
    scale: float = 1.0,
    provenance=None,
    route_evidence=None,
) -> dict:
    return build_projection_record(
        module_full_name=f"model.layers.7.mlp.experts.31.{projection}",
        layer_index=7,
        bits=2,
        codebook="mcg",
        sample_count=1024,
        duration_seconds=1.25 * scale,
        encoded_bytes=128,
        device_names=["cuda:0"],
        quantizer_metrics=_metrics(scale),
        provenance=provenance or {"source_revision": "abc", "hessian_sha256": "def"},
        route_evidence=route_evidence,
    )


def test_routed_expert_identity_covers_base_and_mtp_namespaces():
    assert routed_expert_identity("model.layers.7.mlp.experts.31.gate_proj") == {
        "block_namespace": "base",
        "logical_layer": 7,
        "expert": 31,
        "projection": "w1",
    }
    assert routed_expert_identity("mtp.2.mlp.experts.255.down_proj") == {
        "block_namespace": "mtp",
        "logical_layer": 2,
        "expert": 255,
        "projection": "w2",
    }
    assert routed_expert_identity("model.layers.7.mlp.shared_experts.up_proj") is None


def test_family_join_aggregates_raw_terms_only_for_exact_provenance():
    records = [
        _record("gate_proj", 1.0),
        _record("down_proj", 2.0),
        _record("up_proj", 3.0),
    ]
    families = derive_family_records(records)
    assert len(families) == 1
    family = families[0]
    assert family["projections"] == ["w1", "w2", "w3"]
    assert family["aggregate_metrics"]["error_sum_sq"] == pytest.approx(24.0)
    assert family["aggregate_metrics"]["reference_sum_sq"] == pytest.approx(240.0)
    assert family["aggregate_metrics"][
        "hessian_weighted_error_numerator"
    ] == pytest.approx(12.0)
    assert family["aggregate_metrics"][
        "hessian_weighted_reference_denominator"
    ] == pytest.approx(120.0)
    assert family["aggregate_metrics"][
        "hessian_weighted_relative_error"
    ] == pytest.approx(0.1)

    records[-1] = _record(
        "up_proj",
        3.0,
        provenance={"source_revision": "different", "hessian_sha256": "def"},
    )
    assert derive_family_records(records) == []

    common = {"source_revision": "abc", "corpus_sha256": "def"}
    records = [
        _record(
            projection,
            float(index),
            provenance={
                "family_join": common,
                "hessian_sha256": str(index) * 64,
            },
        )
        for index, projection in enumerate(
            ("gate_proj", "down_proj", "up_proj"), start=1
        )
    ]
    families = derive_family_records(records)
    assert len(families) == 1
    assert families[0]["provenance"]["family_join"] == common
    assert set(families[0]["provenance"]["projections"]) == {"w1", "w2", "w3"}


def test_natural_route_evidence_is_required_and_shared_by_one_expert_family():
    provenance = {
        "family_join": {
            "source_revision": "abc",
            "route_evidence_contract": ROUTE_EVIDENCE_SCHEMA,
        }
    }
    with pytest.raises(ValueError, match="natural-route evidence is required"):
        _record("gate_proj", provenance=provenance)

    evidence = _route_evidence()
    records = [
        _record(projection, provenance=provenance, route_evidence=evidence)
        for projection in ("gate_proj", "down_proj", "up_proj")
    ]
    family = derive_family_records(records)[0]
    assert family["route_evidence"] == evidence
    assert family["route_evidence"]["expert_gate_squared_mass"] == 20.0

    records[-1] = _record(
        "up_proj",
        provenance=provenance,
        route_evidence=_route_evidence(scale=2.0),
    )
    with pytest.raises(ValueError, match="inconsistent route evidence"):
        derive_family_records(records)


def test_ledger_is_canonical_content_bound_and_contains_family_record(tmp_path):
    records = [
        _record("up_proj", 3.0),
        _record("gate_proj", 1.0),
        _record("down_proj", 2.0),
    ]
    manifest = write_exl3_error_ledger(tmp_path, records)
    assert manifest is not None
    assert manifest["projection_records"] == 3
    assert manifest["complete_family_records"] == 1
    assert manifest["total_records"] == 4

    payload = (tmp_path / LEDGER_FILENAME).read_bytes()
    assert hashlib.sha256(payload).hexdigest() == manifest["ledger_sha256"]
    assert json.loads((tmp_path / LEDGER_MANIFEST_FILENAME).read_text()) == manifest

    rows = [json.loads(line) for line in payload.splitlines()]
    assert [row.get("projection", "family") for row in rows] == [
        "w1",
        "w2",
        "w3",
        "family",
    ]
    for row in rows:
        unbound = {key: value for key, value in row.items() if key != "record_sha256"}
        canonical = json.dumps(
            unbound,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        assert hashlib.sha256(canonical).hexdigest() == row["record_sha256"]


def test_projection_journal_fsyncs_individually_bound_records(tmp_path):
    journal = tmp_path / "in-progress.jsonl"
    expected = [_record("gate_proj", 1.0), _record("down_proj", 2.0)]
    digests = [append_exl3_error_journal(journal, record) for record in expected]

    rows = [json.loads(line) for line in journal.read_bytes().splitlines()]
    assert [row["record_sha256"] for row in rows] == digests
    assert [row["projection"] for row in rows] == ["w1", "w2"]


def test_ledger_rejects_non_finite_metrics():
    metrics = _metrics(1.0)
    metrics["reported_metric_value"] = float("nan")
    with pytest.raises(ValueError, match="non-finite EXL3 ledger value"):
        build_projection_record(
            module_full_name="model.layers.0.mlp.experts.0.gate_proj",
            layer_index=0,
            bits=2,
            codebook="mcg",
            sample_count=1,
            duration_seconds=1.0,
            encoded_bytes=1,
            device_names=["cuda:0"],
            quantizer_metrics=metrics,
            provenance=None,
        )
