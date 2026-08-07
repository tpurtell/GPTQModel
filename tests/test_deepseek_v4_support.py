from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config

from gptqmodel.models import auto
from gptqmodel.models.definitions.deepseek_v4 import (
    DeepSeekV4MTPAuxiliaryShell,
    DeepSeekV4QModel,
    expected_deepseek_v4_mtp_checkpoint_keys,
    patch_deepseek_v4_router_precision,
    validate_deepseek_v4_mtp_checkpoint_keys,
)


def _tiny_v4_config() -> DeepseekV4Config:
    return DeepseekV4Config(
        vocab_size=32,
        hidden_size=16,
        moe_intermediate_size=8,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        q_lora_rank=8,
        n_routed_experts=3,
        num_experts_per_tok=2,
        hc_mult=2,
        o_groups=2,
        o_lora_rank=8,
        sliding_window=8,
        layer_types=["sliding_attention"] * 3,
        mlp_layer_types=["moe"] * 3,
        dspark_target_layer_ids=[0, 1, 2],
        dspark_markov_rank=4,
        partial_rotary_factor=0.5,
        dtype="bfloat16",
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


def test_deepseek_v4_mtp_checkpoint_contract_is_exact_and_does_not_trust_nextn_count() -> None:
    config = _tiny_v4_config()
    assert config.num_nextn_predict_layers == 1
    keys = expected_deepseek_v4_mtp_checkpoint_keys(config)
    report = validate_deepseek_v4_mtp_checkpoint_keys(config, keys)

    assert report == {
        "block_count": 3,
        "target_layer_ids": [0, 1, 2],
        "routed_experts_per_block": 3,
        "tensor_count": len(keys),
    }
    assert "mtp.0.ffn.experts.0.w1.weight" in keys
    assert "mtp.2.ffn.experts.2.w3.scale" in keys
    assert "mtp.0.main_proj.weight" in keys
    assert "mtp.2.confidence_head.proj.weight" in keys

    missing = set(keys)
    missing.remove("mtp.1.ffn.experts.2.w2.scale")
    try:
        validate_deepseek_v4_mtp_checkpoint_keys(config, missing)
    except RuntimeError as exc:
        assert "missing=1" in str(exc)
    else:
        raise AssertionError("an incomplete MTP namespace was accepted")

    unexpected = set(keys)
    unexpected.add("mtp.3.ffn.gate.weight")
    try:
        validate_deepseek_v4_mtp_checkpoint_keys(config, unexpected)
    except RuntimeError as exc:
        assert "unexpected=1" in str(exc)
    else:
        raise AssertionError("an unknown MTP block was accepted")


def test_deepseek_v4_mtp_shell_is_defused_patched_and_fail_closed() -> None:
    shell = DeepSeekV4MTPAuxiliaryShell(_tiny_v4_config())

    assert shell.target_layer_ids == (0, 1, 2)
    assert shell.base_num_hidden_layers == 3
    assert shell.config.num_hidden_layers == 6
    assert shell.config.layer_types[-3:] == ["sliding_attention"] * 3
    assert shell.config.mlp_layer_types[-3:] == ["moe"] * 3
    assert len(shell.mtp) == 3
    assert shell.mtp[0].main_proj.in_features == 48
    assert shell.mtp[0].main_proj.out_features == 16
    assert shell.mtp[2].confidence_head.proj.in_features == 20
    assert shell.mtp[2].markov_head.markov_w1.weight.shape == (32, 4)

    for block in shell.mtp:
        assert block.self_attn.layer_type == "sliding_attention"
        assert len(block.mlp.experts) == 3
        assert hasattr(block.mlp.experts[0], "gate_proj")
        assert hasattr(block.mlp.experts[0], "up_proj")
        assert hasattr(block.mlp.experts[0], "down_proj")
        assert block.mlp.gate._gptqmodel_v4_fp32_router
        assert block.mlp.gate.weight.dtype is torch.bfloat16
        assert block.mlp.gate.e_score_correction_bias.dtype is torch.float32
        assert block.self_attn.sinks.dtype is torch.float32
        assert block.attn_hc.fn.dtype is torch.float32

    try:
        shell()
    except RuntimeError as exc:
        assert "must not be appended to target layers" in str(exc)
    else:
        raise AssertionError("generic MTP shell forward did not fail closed")


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
