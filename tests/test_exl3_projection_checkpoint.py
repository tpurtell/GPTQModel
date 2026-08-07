# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
import torch

from gptqmodel.utils.exl3_projection_checkpoint import (
    CHECKPOINT_CONTRACT,
    EXL3ProjectionCheckpointStore,
    build_projection_request,
    checkpoint_root_from_provenance,
)


def _request(scale: float = 1.0) -> dict:
    return build_projection_request(
        module_full_name="model.layers.7.mlp.experts.31.gate_proj",
        layer_index=7,
        input_weight=torch.arange(32, dtype=torch.float32).reshape(8, 4) * scale,
        hessian=torch.eye(8, dtype=torch.float32),
        sample_count=1024,
        quantizer_contract={
            "bits": 2,
            "codebook": "mcg",
            "apply_out_scales": None,
            "sigma_reg": 0.025,
            "seed": 787,
        },
        family_join={"source_revision": "abc", "corpus_sha256": "def"},
        route_evidence={"expert_route_count": 1024},
    )


def _tensors() -> dict[str, torch.Tensor]:
    return {
        "trellis": torch.arange(32, dtype=torch.int16).reshape(1, 1, 32),
        "suh": torch.ones(16, dtype=torch.float16),
        "svh": torch.ones(16, dtype=torch.float16),
        "mcg": torch.tensor([-877912083], dtype=torch.int32),
    }


def _result() -> dict:
    return {
        "duration_seconds": 12.5,
        "proxy_error": 0.125,
        "device_names": ["cuda:0"],
        "quantizer_metrics": {"reported_metric_kind": "test"},
        "ledger_record": {"record_kind": "projection", "module": "test"},
    }


def test_projection_checkpoint_round_trips_and_is_idempotent(tmp_path) -> None:
    store = EXL3ProjectionCheckpointStore(tmp_path / "checkpoints")
    request = _request()
    tensors = _tensors()
    result = _result()

    assert store.load(request) is None
    assert store.commit(request, tensors, result) == result
    first_files = {
        path.relative_to(store.root): path.read_bytes()
        for path in store.root.rglob("*")
        if path.is_file()
    }
    assert store.commit(request, tensors, result) == result
    second_files = {
        path.relative_to(store.root): path.read_bytes()
        for path in store.root.rglob("*")
        if path.is_file()
    }
    assert second_files == first_files

    loaded_tensors, loaded_result = store.load(request)
    assert loaded_result == result
    assert set(loaded_tensors) == set(tensors)
    assert all(
        torch.equal(loaded_tensors[name], tensor) for name, tensor in tensors.items()
    )
    assert store.load(_request(scale=2.0)) is None


def test_projection_checkpoint_rejects_corrupt_payload_and_manifest(tmp_path) -> None:
    store = EXL3ProjectionCheckpointStore(tmp_path / "checkpoints")
    request = _request()
    store.commit(request, _tensors(), _result())
    manifest_path = next(store.root.rglob("*.json"))
    tensor_path = next(store.root.rglob("*.safetensors"))

    payload = bytearray(tensor_path.read_bytes())
    payload[-1] ^= 1
    tensor_path.write_bytes(payload)
    with pytest.raises(ValueError, match="content validation"):
        store.load(request)

    store = EXL3ProjectionCheckpointStore(tmp_path / "other")
    store.commit(request, _tensors(), _result())
    manifest_path = next(store.root.rglob("*.json"))
    manifest = json.loads(manifest_path.read_text())
    manifest["result"]["proxy_error"] = 9.0
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="content validation"):
        store.load(request)


def test_projection_checkpoint_contract_is_explicit_in_run_provenance(tmp_path) -> None:
    root = tmp_path / "checkpoint-root"
    provenance = {
        "run": {
            "projection_checkpoint": {
                "contract": CHECKPOINT_CONTRACT,
                "root": str(root),
            }
        }
    }
    assert checkpoint_root_from_provenance(provenance) == root.resolve()

    provenance["run"]["projection_checkpoint"]["contract"] = "wrong"
    with pytest.raises(ValueError, match="run contract"):
        checkpoint_root_from_provenance(provenance)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="EXL3 requires CUDA")
def test_packed_replay_matches_exllamav3_runtime_reconstruction() -> None:
    from gptqmodel.exllamav3.modules.quant.exl3 import LinearEXL3
    from gptqmodel.exllamav3.modules.quant.exl3_lib.quantize import (
        quantize_exl3,
        reconstruct_exl3_tensors,
    )

    torch.manual_seed(787)
    device = torch.device("cuda:0")
    weight = torch.randn((128, 128), dtype=torch.float32, device=device) * 0.02
    activations = torch.randn((1024, 128), dtype=torch.float32, device=device)
    hessian = (2.0 / activations.shape[0]) * activations.T @ activations
    quant_args = {
        "K": 2,
        "devices": [device],
        "apply_out_scales": None,
        "sigma_reg": 0.025,
        "seed": 787,
        "mcg": True,
    }
    _, _, tensors = quantize_exl3(
        weight,
        {"H": hessian, "count": activations.shape[0], "finalized": False},
        quant_args,
        return_weight_q=False,
    )
    replay = reconstruct_exl3_tensors(tensors, device=device, dtype=torch.float16)
    runtime = LinearEXL3(
        config=None,
        in_features=128,
        out_features=128,
        suh=tensors["suh"],
        svh=tensors["svh"],
        trellis=tensors["trellis"],
        mcg=tensors["mcg"],
        out_dtype=torch.float16,
    )
    try:
        reference = runtime.get_weight_tensor()
        assert torch.equal(replay.view(torch.int16), reference.view(torch.int16))
    finally:
        runtime.unload()
