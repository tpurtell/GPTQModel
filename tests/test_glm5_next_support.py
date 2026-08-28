# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
from torch import nn

from gptqmodel.models.definitions.glm5_next import (
    GLM5_NEXT_ROUTED_EXPERT_PATTERN,
    Glm5NextQModel,
)


def test_routed_expert_pattern_includes_target_and_checkpoint_mtp_only():
    import re

    pattern = re.compile(GLM5_NEXT_ROUTED_EXPERT_PATTERN)
    assert pattern.fullmatch(
        "model.language_model.layers.3.mlp.experts.0.gate_proj"
    )
    assert pattern.fullmatch(
        "model.language_model.layers.45.mlp.experts.287.down_proj"
    )
    assert not pattern.fullmatch(
        "model.language_model.layers.2.mlp.experts.0.gate_proj"
    )
    assert not pattern.fullmatch(
        "model.language_model.layers.46.mlp.experts.0.gate_proj"
    )
    assert not pattern.fullmatch(
        "model.language_model.layers.3.mlp.shared_experts.gate_proj"
    )


def test_route_recovery_resolves_nested_mtp_as_conventional_replay():
    layers = [nn.Identity() for _ in range(46)]
    adapter = Glm5NextQModel.__new__(Glm5NextQModel)
    adapter.model = SimpleNamespace(
        model=SimpleNamespace(
            language_model=SimpleNamespace(layers=nn.ModuleList(layers))
        )
    )

    assert adapter.zero_route_recovery_block_identity(layers[3]) == (
        "base",
        3,
        "model.language_model.layers.3.mlp.experts.",
    )
    assert adapter.zero_route_recovery_block_identity(layers[45]) == (
        "base",
        45,
        "model.language_model.layers.45.mlp.experts.",
    )
    with pytest.raises(RuntimeError, match="canonical decoder block"):
        adapter.zero_route_recovery_block_identity(nn.Identity())
