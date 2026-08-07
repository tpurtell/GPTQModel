from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4ForCausalLM,
    DeepseekV4HyperHead,
    DeepseekV4RMSNorm,
)

from gptqmodel.models import auto
from gptqmodel.models.definitions.deepseek_v4 import (
    DeepSeekV4MTPAuxiliary,
    DeepSeekV4MTPAuxiliaryShell,
    DeepSeekV4MTPPrefixRuntime,
    DeepSeekV4MTPQuantizationModel,
    DeepSeekV4MTPReplay,
    DeepSeekV4MTPReplayBatch,
    DeepSeekV4MTPTargetTapEvent,
    DeepSeekV4QModel,
    DeepSeekV4TargetAnchorResolver,
    MTP_CAPTURE_ATTENTION_MASK,
    MTP_CAPTURE_DECODE_MASK,
    MTP_CAPTURE_INPUT_IDS,
    expected_deepseek_v4_mtp_checkpoint_keys,
    patch_deepseek_v4_router_precision,
    validate_deepseek_v4_mtp_checkpoint_keys,
)
from gptqmodel.quantization.config import EXL3Config
from gptqmodel.looper.stage_inputs_capture import StageInputsCapture


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
        dspark_block_size=5,
        dspark_noise_token_id=31,
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


def test_deepseek_v4_mtp_replay_keeps_five_rows_joint_and_uses_target_lane_means() -> None:
    torch.manual_seed(0xD54)
    config = _tiny_v4_config()
    shell = DeepSeekV4MTPAuxiliaryShell(config, device="cpu")
    with torch.no_grad():
        for parameter in shell.parameters():
            if parameter.is_floating_point():
                parameter.normal_(mean=0.0, std=0.02)
        for block in shell.mtp:
            block.mlp.gate.e_score_correction_bias.zero_()
    embedding = torch.randn(
        config.vocab_size, config.hidden_size, dtype=torch.bfloat16
    )
    replay = DeepSeekV4MTPReplay(shell, embedding_weight=embedding)

    target_outputs = tuple(
        torch.randn(2, 4, config.hc_mult, config.hidden_size, dtype=torch.bfloat16)
        for _ in range(3)
    )
    collapsed = tuple(
        DeepSeekV4MTPReplay.collapse_target_layer_output(value)
        for value in target_outputs
    )
    for source, tap in zip(target_outputs, collapsed):
        torch.testing.assert_close(tap, source.mean(dim=2))

    batch = DeepSeekV4MTPReplayBatch(
        target_taps=collapsed,
        anchor_token_ids=torch.tensor([7, 9]),
        main_position_ids=torch.tensor([[0, 1, 2, 3], [0, 0, 5, 6]]),
        main_attention_mask=torch.tensor(
            [[True, True, True, True], [False, False, True, True]]
        ),
    )
    ffn_shapes = []
    result = replay.replay(
        batch,
        prepare_ffn=lambda block_index, _block, hidden, token_ids: ffn_shapes.append(
            (block_index, tuple(hidden.shape), tuple(token_ids.shape))
        ),
    )

    assert result.proposal_token_ids.tolist() == [
        [7, 31, 31, 31, 31],
        [9, 31, 31, 31, 31],
    ]
    assert result.proposal_position_ids.tolist() == [
        [4, 5, 6, 7, 8],
        [7, 8, 9, 10, 11],
    ]
    assert result.projected_main.shape == (2, 4, config.hidden_size)
    assert result.terminal_residual.shape == (
        2,
        5,
        config.hc_mult,
        config.hidden_size,
    )
    assert ffn_shapes == [
        (0, (2, 5, config.hidden_size), (2, 5)),
        (1, (2, 5, config.hidden_size), (2, 5)),
        (2, (2, 5, config.hidden_size), (2, 5)),
    ]
    assert len(result.routes) == 3
    for block_index, route in enumerate(result.routes):
        assert route.block_index == block_index
        assert route.logits.shape == (2, 5, config.n_routed_experts)
        assert route.weights.shape == (2, 5, config.num_experts_per_tok)
        assert route.indices.shape == (2, 5, config.num_experts_per_tok)
        assert route.logits.dtype is torch.float32
        assert route.weights.dtype is torch.float32

    projected_result = replay.replay(
        DeepSeekV4MTPReplayBatch(
            target_taps=None,
            projected_main=result.projected_main,
            anchor_token_ids=batch.anchor_token_ids,
            main_position_ids=batch.main_position_ids,
            main_attention_mask=batch.main_attention_mask,
        )
    )
    assert torch.equal(projected_result.terminal_residual, result.terminal_residual)
    for projected_route, tap_route in zip(projected_result.routes, result.routes):
        assert torch.equal(projected_route.indices, tap_route.indices)
        assert torch.equal(projected_route.weights, tap_route.weights)


def test_deepseek_v4_mtp_quantization_adapter_replays_one_exact_block() -> None:
    torch.manual_seed(0xD55)
    config = _tiny_v4_config()
    shell = DeepSeekV4MTPAuxiliaryShell(config, device="cpu")
    with torch.no_grad():
        for parameter in shell.parameters():
            if parameter.is_floating_point():
                parameter.normal_(mean=0.0, std=0.02)
        for block in shell.mtp:
            block.mlp.gate.e_score_correction_bias.zero_()
    auxiliary = DeepSeekV4MTPAuxiliary(
        model=shell,
        turtle_model=SimpleNamespace(),
        checkpoint_contract={"test": True},
    )
    embedding = torch.randn(
        config.vocab_size, config.hidden_size, dtype=torch.bfloat16
    )
    target = object.__new__(DeepSeekV4QModel)
    nn.Module.__init__(target)
    target.quantize_config = EXL3Config(bits=2.0, device="cpu")
    target.qlinear_kernel = None
    target.trust_remote_code = False
    target.model_local_path = "/tmp/deepseek-v4-test"

    adapter = DeepSeekV4MTPQuantizationModel.from_target_model(
        target,
        auxiliary=auxiliary,
        embedding_weight=embedding,
    )
    assert target.quantize_config.module_include is None
    assert adapter.quantize_config.module_is_included(
        "mtp.2.mlp.experts.2.down_proj"
    )
    assert not adapter.quantize_config.module_is_included(
        "mtp.2.mlp.shared_experts.down_proj"
    )
    assert adapter.extract_layers_node() == ["mtp"]
    modules = {
            name
            for block in adapter.simple_layer_modules(
                model_config=config,
                quantize_config=adapter.quantize_config,
            )
        for name in block
    }
    assert "mlp.experts.2.gate_proj" in modules
    assert "mlp.experts.2.up_proj" in modules
    assert "mlp.experts.2.down_proj" in modules
    assert not any(name.startswith("self_attn.") for name in modules)
    assert not any(name.startswith("mlp.shared_experts.") for name in modules)

    batch = DeepSeekV4MTPReplayBatch(
        target_taps=None,
        projected_main=torch.randn(2, 4, config.hidden_size, dtype=torch.bfloat16),
        anchor_token_ids=torch.tensor([3, 7]),
        main_position_ids=torch.tensor([[0, 1, 2, 3], [0, 0, 5, 6]]),
        main_attention_mask=torch.tensor(
            [[True, True, True, True], [False, False, True, True]]
        ),
    )
    prepared = adapter.prepare_dataset(
        [batch], calibration_dataset_sort=None, batch_size=1
    )
    assert prepared[0]["input_ids"].shape == (2, 5)
    assert prepared[0]["attention_mask"].all()

    reference = DeepSeekV4MTPReplay(shell, embedding_weight=embedding)
    state = reference.prepare_batch(batch)
    expected, _ = reference.replay_block(0, state, capture_route=False)
    actual = adapter.run_input_capture(
        prepared[0], use_cache=False, data_device=torch.device("cpu")
    )
    torch.testing.assert_close(actual, expected)

    looper = SimpleNamespace(
        gptq_model=adapter,
        _batch_row_count=lambda value: int(value[0].shape[0]),
    )
    cache = StageInputsCapture(looper).cache_inputs(
        layers=list(adapter.model.mtp),
        layer_names=[f"mtp.{index}" for index in range(3)],
        calibration_data=prepared,
        use_cache=False,
    )
    assert len(cache.layer_inputs) == 1
    torch.testing.assert_close(cache.layer_inputs[0][0], state.residual)
    assert torch.equal(cache.position_ids[0], state.proposal_position_ids)
    assert torch.equal(cache.attention_masks[0], prepared[0]["attention_mask"])
    assert torch.equal(
        cache.layer_input_kwargs[0]["_gptqmodel_mtp_projected_main"],
        state.projected_main,
    )
    assert torch.equal(
        cache.layer_input_kwargs[0]["_gptqmodel_mtp_proposal_token_ids"],
        state.proposal_token_ids,
    )
    replay_kwargs = dict(cache.layer_input_kwargs[0])
    routes = []
    handles = [
        block.mlp.gate.register_forward_hook(
            lambda _module, _args, output: routes.append(output)
        )
        for block in adapter.model.mtp
    ]
    try:
        residual = cache.layer_inputs[0][0]
        for block in adapter.model.mtp:
            residual = block(
                residual,
                attention_mask=cache.attention_masks[0],
                position_ids=cache.position_ids[0],
                **replay_kwargs,
            )
    finally:
        for handle in handles:
            handle.remove()
    full_reference = reference.replay(batch)
    torch.testing.assert_close(residual, full_reference.terminal_residual)
    assert len(routes) == 3
    for output in routes:
        _, _, indices = output
        assert indices.shape == (2 * 5, config.num_experts_per_tok)


def test_deepseek_v4_target_anchor_resolver_matches_native_fp32_greedy_head() -> None:
    torch.manual_seed(0xA4C40)
    config = _tiny_v4_config()
    hc_head = DeepseekV4HyperHead(config).to(dtype=torch.bfloat16)
    norm = DeepseekV4RMSNorm(config.hidden_size, eps=config.rms_norm_eps).to(
        dtype=torch.bfloat16
    )
    lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False).to(
        dtype=torch.bfloat16
    )
    resolver = DeepSeekV4TargetAnchorResolver(
        hc_head=hc_head,
        norm=norm,
        lm_head=lm_head,
        position_chunk_size=2,
        vocab_chunk_size=7,
    )
    raw = torch.randn(
        2, 4, config.hc_mult, config.hidden_size, dtype=torch.bfloat16
    )
    input_ids = torch.arange(8).reshape(2, 4)
    attention_mask = torch.tensor(
        [[False, True, True, True], [True, True, True, False]]
    )
    decode_mask = torch.tensor(
        [[False, True, False, True], [True, False, True, True]]
    )
    position_ids = torch.tensor([[0, 3, 4, 5], [8, 9, 10, 0]])

    anchors = resolver(
        raw,
        input_ids,
        attention_mask,
        decode_mask,
        position_ids,
    )
    eligible = attention_mask & decode_mask
    selected = raw[eligible].unsqueeze(0)
    reference_hidden = norm(hc_head(selected)).squeeze(0)
    reference = F.linear(reference_hidden.float(), lm_head.weight.float()).argmax(-1)
    assert torch.equal(anchors[eligible], reference)
    assert torch.equal(anchors[~eligible], torch.full_like(anchors[~eligible], -1))


def test_deepseek_v4_target_tap_sink_receives_only_official_lane_means() -> None:
    config = _tiny_v4_config()
    harness = object.__new__(DeepSeekV4QModel)
    harness.model = SimpleNamespace(config=config)
    events = []

    assert not harness.quantization_layer_output_required(
        layer_index=2, layer_name="model.layers.2", layer_count=3
    )
    harness.set_mtp_target_tap_sink(events.append)
    assert harness.quantization_layer_output_required(
        layer_index=2, layer_name="model.layers.2", layer_count=3
    )
    assert not harness.quantization_layer_output_required(
        layer_index=3, layer_name="model.layers.3", layer_count=3
    )

    raw = torch.randn(2, 5, config.hc_mult, config.hidden_size, dtype=torch.bfloat16)
    positions = torch.arange(5).unsqueeze(0).expand(2, -1)
    mask = torch.ones(2, 5, dtype=torch.bool)
    harness.receive_quantization_layer_outputs(
        layer_index=2,
        layer_name="model.layers.2",
        layer_outputs=[[raw]],
        layer_input_kwargs=[{"input_ids": torch.arange(10).reshape(2, 5)}],
        position_ids=[positions],
        attention_masks=[mask],
    )

    assert len(events) == 1
    event = events[0]
    assert isinstance(event, DeepSeekV4MTPTargetTapEvent)
    assert event.layer_index == 2
    assert event.layer_name == "model.layers.2"
    assert event.raw_layer_outputs[0] is raw
    assert event.position_ids[0] is positions
    assert event.attention_masks[0] is mask
    torch.testing.assert_close(event.collapsed_target_taps[0], raw.mean(dim=2))


def test_deepseek_v4_target_tap_sink_rejects_uncollapsible_output() -> None:
    config = _tiny_v4_config()
    harness = object.__new__(DeepSeekV4QModel)
    harness.model = SimpleNamespace(config=config)
    harness.set_mtp_target_tap_sink(lambda _event: None)
    try:
        harness.receive_quantization_layer_outputs(
            layer_index=2,
            layer_name="model.layers.2",
            layer_outputs=[[torch.zeros(2, 5, config.hidden_size)]],
            layer_input_kwargs=[{}],
            position_ids=[],
            attention_masks=[],
        )
    except RuntimeError as exc:
        assert "[batch, sequence, hc, hidden]" in str(exc)
    else:
        raise AssertionError("a rank-3 target boundary was accepted as a raw mHC tap")


def test_deepseek_v4_preserves_original_token_mask_metadata_without_forwarding_it() -> None:
    config = _tiny_v4_config()
    harness = object.__new__(DeepSeekV4QModel)
    harness.model = SimpleNamespace(config=config)
    input_ids = torch.tensor([[1, 2, 3]])
    attention_mask = torch.tensor([[0, 1, 1]])
    labels = torch.tensor([[-100, -100, 3]])
    harness.begin_input_capture_example(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        },
        batch_device=torch.device("cpu"),
    )
    captured = harness.capture_first_layer_input_kwargs(
        args=(),
        kwargs={},
        batch_device=torch.device("cpu"),
        layer_input_kwargs={"ordinary": torch.tensor(1)},
    )
    harness.end_input_capture_example()

    assert torch.equal(captured[MTP_CAPTURE_INPUT_IDS], input_ids)
    assert torch.equal(captured[MTP_CAPTURE_ATTENTION_MASK], attention_mask)
    assert captured[MTP_CAPTURE_DECODE_MASK].tolist() == [[False, True, False]]
    replay_kwargs = harness.prepare_layer_replay_kwargs(
        layer=nn.Identity(),
        layer_input=[torch.zeros(1, 3, config.hidden_size)],
        additional_inputs=dict(captured),
        target_device=torch.device("cpu"),
    )
    assert MTP_CAPTURE_INPUT_IDS not in replay_kwargs
    assert MTP_CAPTURE_ATTENTION_MASK not in replay_kwargs
    assert MTP_CAPTURE_DECODE_MASK not in replay_kwargs
    assert "ordinary" in replay_kwargs


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


def test_deepseek_v4_after_load_preserves_source_fp32_and_requires_router_coverage() -> None:
    model = DeepseekV4ForCausalLM(_tiny_v4_config()).to(dtype=torch.bfloat16)
    harness = object.__new__(DeepSeekV4QModel)
    assert DeepSeekV4QModel.after_model_load(harness, model) is model

    assert model.model.hc_head.hc_fn.dtype is torch.float32
    assert model.model.hc_head.hc_base.dtype is torch.float32
    assert model.model.hc_head.hc_scale.dtype is torch.float32
    assert model.model.norm.weight.dtype is torch.bfloat16
    assert model.lm_head.weight.dtype is torch.bfloat16
    for layer in model.model.layers:
        assert layer.attn_hc.fn.dtype is torch.float32
        assert layer.attn_hc.base.dtype is torch.float32
        assert layer.attn_hc.scale.dtype is torch.float32
        assert layer.ffn_hc.fn.dtype is torch.float32
        assert layer.ffn_hc.base.dtype is torch.float32
        assert layer.ffn_hc.scale.dtype is torch.float32
        assert layer.self_attn.sinks.dtype is torch.float32

    model.model.layers[-1].mlp.gate = nn.Identity()
    try:
        DeepSeekV4QModel.after_model_load(harness, model)
    except RuntimeError as exc:
        assert "router" in str(exc)
        assert "coverage mismatch" in str(exc)
    else:
        raise AssertionError("missing DeepSeek V4 router coverage was accepted")


def test_deepseek_v4_prefix_runtime_owns_exact_projector_anchor_and_embedding(
    monkeypatch,
) -> None:
    torch.manual_seed(0xA4C41)
    config = _tiny_v4_config()
    model = DeepseekV4ForCausalLM(config).to(dtype=torch.bfloat16)
    harness = object.__new__(DeepSeekV4QModel)
    nn.Module.__init__(harness)
    harness.model = DeepSeekV4QModel.after_model_load(harness, model)

    shell = DeepSeekV4MTPAuxiliaryShell(config, device="cpu")
    with torch.no_grad():
        for parameter in shell.parameters():
            if parameter.is_floating_point():
                parameter.normal_(mean=0.0, std=0.02)
    auxiliary = DeepSeekV4MTPAuxiliary(
        model=shell,
        turtle_model=SimpleNamespace(),
        checkpoint_contract={"test": True},
    )

    from gptqmodel.utils.structure import LazyTurtle

    harness.turtle_model = object.__new__(LazyTurtle)
    mtp_materialized = []
    target_materialized = []
    monkeypatch.setattr(
        harness,
        "build_mtp_auxiliary",
        lambda *, device="meta": auxiliary,
    )
    monkeypatch.setattr(
        harness,
        "materialize_mtp_replay_submodule",
        lambda _auxiliary, module, **_kwargs: mtp_materialized.append(module) or module,
    )
    monkeypatch.setattr(
        harness,
        "shell_module_materialize",
        lambda *, target_submodule, module_path, **_kwargs: (
            target_materialized.append(module_path) or target_submodule
        ),
    )

    runtime = harness.build_mtp_prefix_runtime(
        device="cpu",
        position_chunk_size=2,
        vocab_chunk_size=7,
    )
    assert isinstance(runtime, DeepSeekV4MTPPrefixRuntime)
    assert mtp_materialized == [shell.mtp[0].main_proj, shell.mtp[0].main_norm]
    assert target_materialized == [
        "model.hc_head",
        "model.norm",
        "lm_head",
        "model.embed_tokens",
    ]

    taps = tuple(
        torch.randn(2, 3, config.hidden_size, dtype=torch.bfloat16)
        for _ in range(3)
    )
    expected = shell.mtp[0].main_norm(shell.mtp[0].main_proj(torch.cat(taps, -1)))
    torch.testing.assert_close(runtime.project_target_taps(taps), expected)
    assert runtime.build_replay().embedding_weight is model.model.embed_tokens.weight

    raw = torch.randn(
        2, 3, config.hc_mult, config.hidden_size, dtype=torch.bfloat16
    )
    input_ids = torch.arange(6).reshape(2, 3)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    decode_mask = torch.tensor([[False, True, True], [True, False, True]])
    position_ids = torch.arange(3).expand(2, -1)
    anchors = runtime.anchor_resolver(
        raw, input_ids, attention_mask, decode_mask, position_ids
    )
    assert torch.all(anchors[decode_mask] >= 0)
    assert torch.all(anchors[~decode_mask] == -1)
