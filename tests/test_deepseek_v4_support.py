from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F

from gptqmodel.models import auto
from gptqmodel.models.definitions.deepseek_v4 import (
    DeepSeekV4QModel,
    patch_deepseek_v4_router_precision,
)


def test_deepseek_v4_model_type_selects_definition(monkeypatch):
    fake_config = SimpleNamespace(model_type="deepseek_v4")

    monkeypatch.setattr(auto, "resolve_trust_remote_code", lambda path, trust_remote_code=False: trust_remote_code)
    monkeypatch.setattr(auto.AutoConfig, "from_pretrained", lambda *args, **kwargs: fake_config)

    assert auto.check_and_get_model_definition("/tmp/deepseek-v4") is DeepSeekV4QModel


def test_deepseek_v4_module_tree_matches_v4_attention_and_fused_experts():
    layer_modules = DeepSeekV4QModel.simple_layer_modules(
        model_config=SimpleNamespace(n_routed_experts=256),
        quantize_config=SimpleNamespace(dynamic=None),
    )
    flat_modules = {name for block in layer_modules for name in block}

    assert "self_attn.q_a_proj" in flat_modules
    assert "self_attn.q_b_proj" in flat_modules
    assert "self_attn.kv_proj" in flat_modules
    assert "self_attn.o_b_proj" in flat_modules
    # grouped projection must stay native and should not be part of quant blocks
    assert "self_attn.o_a_proj" not in flat_modules
    assert "mlp.experts.99.gate_proj" in flat_modules
    assert "mlp.experts.99.up_proj" in flat_modules
    assert "mlp.experts.99.down_proj" in flat_modules
    assert "mlp.shared_experts.gate_proj" in flat_modules


def test_deepseek_v4_preserves_integrated_mtp_namespace() -> None:
    assert DeepSeekV4QModel.out_of_model_tensors == {"prefixes": ["mtp"]}


class _LearnedRouter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden_dim = 4
        self.top_k = 2
        self.routed_scaling_factor = 1.5
        self.score_fn = nn.Softplus()
        self.weight = nn.Parameter(
            torch.tensor(
                [[1, 2, 3, 4], [4, 3, 2, 1], [1, -1, 1, -1]],
                dtype=torch.bfloat16,
            ),
            requires_grad=False,
        )
        self.register_buffer(
            "e_score_correction_bias", torch.tensor([0.1, -0.2, 0.3])
        )


class _HashRouter(_LearnedRouter):
    def __init__(self) -> None:
        super().__init__()
        del self.e_score_correction_bias
        self.register_buffer("tid2eid", torch.tensor([[0, 2], [1, 0]]))


class _Mlp(nn.Module):
    def __init__(self, gate: nn.Module) -> None:
        super().__init__()
        self.gate = gate


class _Layer(nn.Module):
    def __init__(self, gate: nn.Module) -> None:
        super().__init__()
        self.mlp = _Mlp(gate)


class _RouterModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(num_hidden_layers=2)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_Layer(_HashRouter()), _Layer(_LearnedRouter())])


def test_deepseek_v4_router_patch_uses_fp32_without_promoting_stored_weights() -> None:
    model = _RouterModel()
    assert patch_deepseek_v4_router_precision(model) == 2
    assert patch_deepseek_v4_router_precision(model) == 2
    hidden = torch.tensor([[1.0, 0.5, -0.25, 2.0]], dtype=torch.bfloat16)

    learned = model.model.layers[1].mlp.gate
    logits, weights, indices = learned(hidden)
    expected_logits = F.linear(hidden.float(), learned.weight.float())
    expected_scores = learned.score_fn(expected_logits)
    expected_indices = torch.topk(
        expected_scores + learned.e_score_correction_bias,
        2,
        dim=-1,
        sorted=False,
    ).indices
    expected_weights = expected_scores.gather(1, expected_indices)
    expected_weights /= expected_weights.sum(dim=-1, keepdim=True) + 1e-20
    torch.testing.assert_close(logits, expected_logits)
    torch.testing.assert_close(weights, expected_weights * 1.5)
    assert torch.equal(indices, expected_indices)
    assert learned.weight.dtype is torch.bfloat16
    assert logits.dtype is torch.float32
    assert weights.dtype is torch.float32

    hashed = model.model.layers[0].mlp.gate
    _, hash_weights, hash_indices = hashed(hidden, torch.tensor([[1]]))
    assert hash_indices.tolist() == [[1, 0]]
    assert hash_weights.dtype is torch.float32


def test_deepseek_v4_after_load_requires_every_router_to_be_patched() -> None:
    model = _RouterModel()
    harness = object.__new__(DeepSeekV4QModel)
    assert DeepSeekV4QModel.after_model_load(harness, model) is model
    model.config.num_hidden_layers = 3
    try:
        DeepSeekV4QModel.after_model_load(harness, model)
    except RuntimeError as exc:
        assert "coverage mismatch" in str(exc)
    else:
        raise AssertionError("missing DeepSeek V4 router coverage was accepted")
