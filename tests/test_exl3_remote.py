# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
import torch

from gptqmodel.utils.exl3_projection_checkpoint import (
    EXL3ProjectionCheckpointStore,
    build_projection_request,
)
from gptqmodel.utils.exl3_remote import (
    REMOTE_CONTRACT,
    REMOTE_RESULT_SCHEMA,
    REMOTE_SCHEDULER,
    EXL3RemoteClient,
    RemoteEndpoint,
    decode_tensor_envelope,
    encode_tensor_envelope,
    execute_remote_projection,
    remote_client_from_provenance,
    validate_remote_output_tensors,
)


def _endpoint(name: str) -> RemoteEndpoint:
    return RemoteEndpoint(
        name=name,
        url=f"http://{name}:17841",
        preflight_sha256=(name.encode().hex() + "0" * 64)[:64],
        image_digest="sha256:" + (name.encode().hex() + "1" * 64)[:64],
    )


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(787)
    weight = torch.randn((128, 128), generator=generator, dtype=torch.float32) * 0.02
    activations = torch.randn((1024, 128), generator=generator, dtype=torch.float32)
    hessian = (2.0 / activations.shape[0]) * activations.T @ activations
    return weight, hessian


def _request(
    weight: torch.Tensor,
    hessian: torch.Tensor,
    endpoint: RemoteEndpoint,
) -> dict:
    return build_projection_request(
        module_full_name="model.layers.7.mlp.experts.31.gate_proj",
        layer_index=7,
        input_weight=weight,
        hessian=hessian,
        sample_count=1024,
        quantizer_contract={
            "bits": 2,
            "codebook": "mcg",
            "apply_out_scales": None,
            "sigma_reg": 0.025,
            "seed": 787,
            "execution": EXL3RemoteClient.execution_contract(endpoint),
        },
        family_join={"source_revision": "test-source"},
        route_evidence=None,
    )


def _packed_tensors() -> dict[str, torch.Tensor]:
    return {
        "trellis": torch.zeros((8, 8, 32), dtype=torch.int16),
        "suh": torch.ones(128, dtype=torch.float16),
        "svh": torch.ones(128, dtype=torch.float16),
        "mcg": torch.tensor([-877912083], dtype=torch.int32),
    }


def test_tensor_envelope_is_content_bound() -> None:
    payload = encode_tensor_envelope(
        {"schema": "test", "value": 7},
        {"weight": torch.arange(16, dtype=torch.float32).reshape(4, 4)},
    )
    manifest, tensors = decode_tensor_envelope(payload)
    assert manifest == {"schema": "test", "value": 7}
    assert torch.equal(
        tensors["weight"], torch.arange(16, dtype=torch.float32).reshape(4, 4)
    )

    corrupt = bytearray(payload)
    corrupt[-1] ^= 1
    with pytest.raises(ValueError, match="content validation"):
        decode_tensor_envelope(bytes(corrupt))


def test_scheduler_is_order_independent_and_binds_worker_identity() -> None:
    endpoints = [_endpoint("spark-d"), _endpoint("spark-a"), _endpoint("spark-c")]
    forward = EXL3RemoteClient(
        endpoints=endpoints,
        token=b"secret",
        coordinator_slot=True,
        timeout_seconds=1,
    )
    reverse = EXL3RemoteClient(
        endpoints=list(reversed(endpoints)),
        token=b"secret",
        coordinator_slot=True,
        timeout_seconds=1,
    )
    keys = [f"base:17:{expert}" for expert in range(128)]
    assert [forward.assigned_endpoint(key) for key in keys] == [
        reverse.assigned_endpoint(key) for key in keys
    ]
    assert {forward.assigned_endpoint(key) for key in keys} == {
        None,
        *endpoints,
    }
    endpoint = forward.assigned_endpoint("base:17:31")
    if endpoint is None:
        endpoint = endpoints[0]
    assert forward.execution_contract(endpoint) == {
        "kind": "remote_worker",
        "remote_contract": REMOTE_CONTRACT,
        "name": endpoint.name,
        "preflight_sha256": endpoint.preflight_sha256,
        "image_digest": endpoint.image_digest,
    }


def test_provenance_requires_token_and_preserves_retry_contract(monkeypatch) -> None:
    endpoint = _endpoint("spark-a")
    provenance = {
        "run": {
            "remote_workers": {
                "contract": REMOTE_CONTRACT,
                "scheduler": REMOTE_SCHEDULER,
                "token_env": "TEST_EXL3_TOKEN",
                "coordinator_slot": True,
                "timeout_seconds": 42,
                "max_attempts": 3,
                "orchestration_workers": 2,
                "endpoints": [endpoint.__dict__],
            }
        }
    }
    monkeypatch.delenv("TEST_EXL3_TOKEN", raising=False)
    with pytest.raises(ValueError, match="is unset"):
        remote_client_from_provenance(provenance)
    monkeypatch.setenv("TEST_EXL3_TOKEN", "secret")
    client = remote_client_from_provenance(provenance)
    assert client is not None
    assert client.max_attempts == 3
    assert client.timeout_seconds == 42

    invalid = json.loads(json.dumps(provenance))
    invalid["run"]["remote_workers"]["max_attempts"] = 0
    with pytest.raises(ValueError, match="run contract"):
        remote_client_from_provenance(invalid)


def test_output_geometry_is_bound_to_request() -> None:
    endpoint = _endpoint("spark-a")
    weight, hessian = _inputs()
    request = _request(weight, hessian, endpoint)
    tensors = _packed_tensors()
    validate_remote_output_tensors(request, tensors)
    tensors["trellis"] = torch.zeros((8, 8, 16), dtype=torch.int16)
    with pytest.raises(ValueError, match="geometry"):
        validate_remote_output_tensors(request, tensors)


def test_retry_stays_on_the_assigned_worker_and_retains_history(monkeypatch) -> None:
    endpoint = _endpoint("spark-a")
    client = EXL3RemoteClient(
        endpoints=[endpoint],
        token=b"secret",
        coordinator_slot=False,
        timeout_seconds=1,
        max_attempts=2,
    )
    weight, hessian = _inputs()
    request = _request(weight, hessian, endpoint)
    result = {
        "duration_seconds": 1.0,
        "proxy_error": 0.125,
        "device_names": ["remote:spark-a/cuda:0"],
        "quantizer_metrics": {"reported_metric_kind": "test"},
        "worker": {
            "name": endpoint.name,
            "preflight_sha256": endpoint.preflight_sha256,
            "image_digest": endpoint.image_digest,
        },
    }
    response = encode_tensor_envelope(
        {
            "schema": REMOTE_RESULT_SCHEMA,
            "contract": REMOTE_CONTRACT,
            "request_sha256": request["request_sha256"],
            "checkpoint_hit": True,
            "result": result,
        },
        _packed_tensors(),
    )
    calls: list[str] = []

    monkeypatch.setattr(client, "qualify", lambda value: None)

    def request_once_then_succeed(value, _request):
        calls.append(value.name)
        if len(calls) == 1:
            raise RuntimeError("simulated lost response")
        return response

    monkeypatch.setattr(client, "_request", request_once_then_succeed)
    tensors, returned, transport = client.quantize(
        endpoint=endpoint,
        request_manifest=request,
        input_weight=weight,
        hessian=hessian,
    )
    assert calls == ["spark-a", "spark-a"]
    assert returned == result
    assert set(tensors) == {"trellis", "suh", "svh", "mcg"}
    assert transport["attempts"] == 2
    assert transport["worker_checkpoint_hit"] is True
    assert transport["retry_errors"] == [
        {
            "attempt": 1,
            "error_type": "RuntimeError",
            "message": "simulated lost response",
        }
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="EXL3 requires CUDA")
def test_worker_quantization_resumes_from_its_packed_checkpoint(tmp_path) -> None:
    endpoint = _endpoint("spark-a")
    identity = {
        "contract": REMOTE_CONTRACT,
        "name": endpoint.name,
        "preflight_sha256": endpoint.preflight_sha256,
        "image_digest": endpoint.image_digest,
    }
    weight, hessian = _inputs()
    request = _request(weight, hessian, endpoint)
    store = EXL3ProjectionCheckpointStore(tmp_path / "worker-checkpoints")

    first_tensors, first_result, first_hit = execute_remote_projection(
        request=request,
        tensors={"input_weight": weight, "hessian": hessian},
        device="cuda:0",
        worker_identity=identity,
        checkpoint_store=store,
    )
    second_tensors, second_result, second_hit = execute_remote_projection(
        request=request,
        tensors={"input_weight": weight, "hessian": hessian},
        device="cuda:0",
        worker_identity=identity,
        checkpoint_store=store,
    )
    assert first_hit is False
    assert second_hit is True
    assert first_result == second_result
    assert set(first_tensors) == {"trellis", "suh", "svh", "mcg"}
    assert all(
        torch.equal(first_tensors[name], second_tensors[name])
        for name in first_tensors
    )
