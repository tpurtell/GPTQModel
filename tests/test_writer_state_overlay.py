# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from gptqmodel.models import writer
from gptqmodel.utils import model as model_utils


class _Owner:
    def __init__(self, prefixes, expected_suffixes=None):
        self.prefixes = prefixes
        self.expected_suffixes = expected_suffixes
        self.model = torch.nn.Linear(1, 1)

    def save_state_overlay(self):
        contract = {
            "model": torch.nn.Linear(1, 1),
            "offload_root": "/offload",
            "replace_prefixes": self.prefixes,
        }
        if self.expected_suffixes is not None:
            contract["expected_suffixes"] = self.expected_suffixes
        return contract


def test_save_state_overlay_replaces_only_declared_prefixes(monkeypatch):
    old = SimpleNamespace(value="old")
    native = SimpleNamespace(value="native")
    trellis = SimpleNamespace(value="trellis")
    suh = SimpleNamespace(value="suh")
    state = {
        "mtp.0.mlp.experts.7.gate_proj.weight": old,
        "mtp.0.mlp.experts.7.gate_proj.weight_scale_inv": old,
        "mtp.0.main_norm.weight": native,
    }
    monkeypatch.setattr(
        writer,
        "get_state_dict_for_save",
        lambda model, offload_root=None, include_prefixes=None: {
            "mtp.0.mlp.experts.7.gate_proj.trellis": trellis,
            "mtp.0.mlp.experts.7.gate_proj.suh": suh,
            "mtp.0.main_norm.weight": SimpleNamespace(value="not-selected"),
        },
    )

    writer._apply_save_state_overlay(
        _Owner(["mtp.0.mlp.experts.7.gate_proj"]), state
    )

    assert state == {
        "mtp.0.main_norm.weight": native,
        "mtp.0.mlp.experts.7.gate_proj.trellis": trellis,
        "mtp.0.mlp.experts.7.gate_proj.suh": suh,
    }


def test_save_tensor_storage_includes_exact_overlay_modules(monkeypatch):
    base = {"model.layers.0.expert": {"stored_tensors": {}}}
    mtp_module = "mtp.0.mlp.experts.7.gate_proj"
    overlay = {
        mtp_module: {
            "stored_tensors": {
                f"{mtp_module}.trellis": {},
                f"{mtp_module}.suh": {},
                f"{mtp_module}.svh": {},
                f"{mtp_module}.mcg": {},
            }
        }
    }
    storage_by_model = iter((base, overlay))
    monkeypatch.setattr(
        writer,
        "build_exllamav3_tensor_storage",
        lambda _model: next(storage_by_model),
    )

    storage = writer._build_exllamav3_tensor_storage_for_save(
        _Owner([mtp_module], ["trellis", "suh", "svh", "mcg"])
    )

    assert storage == {**base, **overlay}


def test_save_tensor_storage_fails_closed_on_missing_overlay_module(monkeypatch):
    monkeypatch.setattr(
        writer,
        "build_exllamav3_tensor_storage",
        lambda _model: {},
    )
    with pytest.raises(ValueError, match="no EXL3 tensor-storage metadata"):
        writer._build_exllamav3_tensor_storage_for_save(
            _Owner(["mtp.0.missing"])
        )


def test_save_tensor_storage_fails_closed_on_tensor_contract(monkeypatch):
    module = "mtp.0.expert"
    storage_by_model = iter(
        (
            {},
            {module: {"stored_tensors": {f"{module}.trellis": {}}}},
        )
    )
    monkeypatch.setattr(
        writer,
        "build_exllamav3_tensor_storage",
        lambda _model: next(storage_by_model),
    )
    with pytest.raises(ValueError, match="tensor-storage contract differs"):
        writer._build_exllamav3_tensor_storage_for_save(
            _Owner([module], ["trellis", "suh", "svh", "mcg"])
        )


def test_save_state_overlay_fails_closed_on_missing_prefix(monkeypatch):
    monkeypatch.setattr(writer, "get_state_dict_for_save", lambda *args, **kwargs: {})
    with pytest.raises(ValueError, match="no replacement tensors"):
        writer._apply_save_state_overlay(_Owner(["mtp.0.missing"]), {})


def test_save_state_overlay_fails_closed_on_tensor_contract(monkeypatch):
    monkeypatch.setattr(
        writer,
        "get_state_dict_for_save",
        lambda *args, **kwargs: {"mtp.0.expert.weight": object()},
    )
    with pytest.raises(ValueError, match="tensor contract differs"):
        writer._apply_save_state_overlay(
            _Owner(["mtp.0.expert"], ["trellis", "suh", "svh", "mcg"]),
            {},
        )


@pytest.mark.parametrize("prefixes", ([], [""], ["mtp.0", "mtp.0"]))
def test_save_state_overlay_rejects_ambiguous_prefix_contract(prefixes):
    with pytest.raises(ValueError, match="unique non-empty"):
        writer._apply_save_state_overlay(_Owner(prefixes), {})


def test_prefixed_state_collection_does_not_resolve_unrelated_meta_tensors(
    monkeypatch,
):
    tree = torch.nn.Module()
    tree.selected = torch.nn.Linear(2, 2, device="meta")
    tree.unrelated = torch.nn.Linear(2, 2, device="meta")
    resolved = []

    def fake_resolve(root, module_path, leaf, dtype, shape, cache):
        resolved.append((module_path, leaf))
        return SimpleNamespace(path=f"{root}/{module_path}/{leaf}")

    monkeypatch.setattr(model_utils, "_resolve_offload_entry", fake_resolve)
    state = model_utils.get_state_dict_for_save(
        tree,
        offload_root="/offload",
        include_prefixes=("selected.",),
    )

    assert set(state) == {"selected.weight", "selected.bias"}
    assert resolved == [("selected", "weight"), ("selected", "bias")]
