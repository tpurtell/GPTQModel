# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import copy
from dataclasses import dataclass
from types import MethodType
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from .deepseek_v3 import DeepSeekV3QModel


MTP_BLOCK_COUNT = 3


def deepseek_v4_mtp_target_layer_ids(config) -> tuple[int, ...]:
    """Return and validate the three target taps that seed integrated dSpark."""

    target_ids = tuple(int(value) for value in (getattr(config, "dspark_target_layer_ids", None) or ()))
    num_hidden_layers = int(getattr(config, "num_hidden_layers", 0))
    expected = tuple(range(num_hidden_layers - MTP_BLOCK_COUNT, num_hidden_layers))
    if num_hidden_layers < MTP_BLOCK_COUNT or target_ids != expected:
        raise ValueError(
            "DeepSeek V4 integrated dSpark requires the final three target layers: "
            f"actual={target_ids} expected={expected}"
        )
    return target_ids


def expected_deepseek_v4_mtp_checkpoint_keys(config) -> set[str]:
    """Build the complete checkpoint namespace consumed by the MTP shell."""

    deepseek_v4_mtp_target_layer_ids(config)
    num_experts = int(getattr(config, "n_routed_experts", 0))
    if num_experts <= 0:
        raise ValueError(f"DeepSeek V4 MTP requires routed experts, got {num_experts}")

    expected: set[str] = set()
    common = {
        "attn.attn_sink",
        "attn.kv_norm.weight",
        "attn.q_norm.weight",
        "attn_norm.weight",
        "ffn.gate.bias",
        "ffn.gate.weight",
        "ffn_norm.weight",
        "hc_attn_base",
        "hc_attn_fn",
        "hc_attn_scale",
        "hc_ffn_base",
        "hc_ffn_fn",
        "hc_ffn_scale",
    }
    for projection in ("wkv", "wo_a", "wo_b", "wq_a", "wq_b"):
        common.update({f"attn.{projection}.weight", f"attn.{projection}.scale"})
    for projection in ("w1", "w2", "w3"):
        common.update(
            {
                f"ffn.shared_experts.{projection}.weight",
                f"ffn.shared_experts.{projection}.scale",
            }
        )

    for block_index in range(MTP_BLOCK_COUNT):
        prefix = f"mtp.{block_index}."
        expected.update(prefix + suffix for suffix in common)
        for expert_index in range(num_experts):
            for projection in ("w1", "w2", "w3"):
                expert_prefix = f"{prefix}ffn.experts.{expert_index}.{projection}"
                expected.update({f"{expert_prefix}.weight", f"{expert_prefix}.scale"})

    expected.update(
        {
            "mtp.0.main_norm.weight",
            "mtp.0.main_proj.scale",
            "mtp.0.main_proj.weight",
            "mtp.2.confidence_head.proj.weight",
            "mtp.2.hc_head_base",
            "mtp.2.hc_head_fn",
            "mtp.2.hc_head_scale",
            "mtp.2.markov_head.markov_w1.weight",
            "mtp.2.markov_head.markov_w2.weight",
            "mtp.2.norm.weight",
        }
    )
    return expected


def validate_deepseek_v4_mtp_checkpoint_keys(config, keys: Iterable[str]) -> dict:
    """Fail closed unless the source contains exactly the supported MTP graph."""

    expected = expected_deepseek_v4_mtp_checkpoint_keys(config)
    actual = {str(key) for key in keys if str(key).startswith("mtp.")}
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing={len(missing)} sample={missing[:8]}")
        if unexpected:
            details.append(f"unexpected={len(unexpected)} sample={unexpected[:8]}")
        raise RuntimeError("DeepSeek V4 MTP checkpoint namespace mismatch: " + "; ".join(details))
    return {
        "block_count": MTP_BLOCK_COUNT,
        "target_layer_ids": list(deepseek_v4_mtp_target_layer_ids(config)),
        "routed_experts_per_block": int(config.n_routed_experts),
        "tensor_count": len(actual),
    }


def deepseek_v4_mtp_module_tree() -> list:
    """Return the auxiliary layer tree without joining it to target traversal."""

    return [
        "mtp",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": (
                "q_a_norm:!",
                "q_a_proj:0",
                "q_b_norm:!",
                "q_b_proj:0",
                "o_a_proj:!",
                "o_b_proj:1",
                "kv_norm:!",
                "kv_proj:2",
            ),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp|ffn:moe": {
                "gate": ("gate:!",),
                "experts": {
                    "#": (
                        "gate_proj|w1:0",
                        "up_proj|w3:0",
                        "down_proj|w2:1",
                    ),
                },
                "shared_experts": (
                    "gate_proj|w1:0",
                    "up_proj|w3:0",
                    "down_proj|w2:1",
                ),
            },
        },
    ]


def deepseek_v4_mtp_checkpoint_mapping_reversed():
    """Map the checkpoint-native ``mtp.*`` spellings to the execution shell."""

    from transformers.core_model_loading import WeightRenaming

    from ...utils.structure import LazyTurtle

    forward_mapping = [
        WeightRenaming(r"^mtp\.(\d+)\.attn_norm\.", r"mtp.\1.input_layernorm."),
        WeightRenaming(r"^mtp\.(\d+)\.ffn_norm\.", r"mtp.\1.post_attention_layernorm."),
        WeightRenaming(r"^mtp\.(\d+)\.hc_attn_fn$", r"mtp.\1.attn_hc.fn"),
        WeightRenaming(r"^mtp\.(\d+)\.hc_attn_base$", r"mtp.\1.attn_hc.base"),
        WeightRenaming(r"^mtp\.(\d+)\.hc_attn_scale$", r"mtp.\1.attn_hc.scale"),
        WeightRenaming(r"^mtp\.(\d+)\.hc_ffn_fn$", r"mtp.\1.ffn_hc.fn"),
        WeightRenaming(r"^mtp\.(\d+)\.hc_ffn_base$", r"mtp.\1.ffn_hc.base"),
        WeightRenaming(r"^mtp\.(\d+)\.hc_ffn_scale$", r"mtp.\1.ffn_hc.scale"),
        WeightRenaming(r"^mtp\.(\d+)\.attn\.", r"mtp.\1.self_attn."),
        WeightRenaming(r"^mtp\.(\d+)\.ffn\.", r"mtp.\1.mlp."),
        WeightRenaming(r"^mtp\.(\d+)\.self_attn\.attn_sink$", r"mtp.\1.self_attn.sinks"),
        WeightRenaming(r"^mtp\.(\d+)\.self_attn\.wq_a\.", r"mtp.\1.self_attn.q_a_proj."),
        WeightRenaming(r"^mtp\.(\d+)\.self_attn\.wq_b\.", r"mtp.\1.self_attn.q_b_proj."),
        WeightRenaming(r"^mtp\.(\d+)\.self_attn\.wkv\.", r"mtp.\1.self_attn.kv_proj."),
        WeightRenaming(r"^mtp\.(\d+)\.self_attn\.wo_a\.", r"mtp.\1.self_attn.o_a_proj."),
        WeightRenaming(r"^mtp\.(\d+)\.self_attn\.wo_b\.", r"mtp.\1.self_attn.o_b_proj."),
        WeightRenaming(r"^mtp\.(\d+)\.self_attn\.q_norm\.", r"mtp.\1.self_attn.q_a_norm."),
        WeightRenaming(r"^mtp\.(\d+)\.mlp\.gate\.bias$", r"mtp.\1.mlp.gate.e_score_correction_bias"),
        WeightRenaming(
            r"^mtp\.(\d+)\.mlp\.shared_experts\.w1\.",
            r"mtp.\1.mlp.shared_experts.gate_proj.",
        ),
        WeightRenaming(
            r"^mtp\.(\d+)\.mlp\.shared_experts\.w2\.",
            r"mtp.\1.mlp.shared_experts.down_proj.",
        ),
        WeightRenaming(
            r"^mtp\.(\d+)\.mlp\.shared_experts\.w3\.",
            r"mtp.\1.mlp.shared_experts.up_proj.",
        ),
        WeightRenaming(r"^mtp\.2\.hc_head_fn$", r"mtp.2.hc_head.hc_fn"),
        WeightRenaming(r"^mtp\.2\.hc_head_base$", r"mtp.2.hc_head.hc_base"),
        WeightRenaming(r"^mtp\.2\.hc_head_scale$", r"mtp.2.hc_head.hc_scale"),
    ]
    reversed_mapping = LazyTurtle.reverse_hf_conversion_map(forward_mapping)
    if reversed_mapping is None:
        raise RuntimeError("DeepSeek V4 MTP checkpoint mapping could not be reversed")
    return reversed_mapping


class _DeepSeekV4MTPMarkovHead(nn.Module):
    def __init__(self, *, vocab_size: int, rank: int):
        super().__init__()
        self.markov_w1 = nn.Embedding(vocab_size, rank)
        self.markov_w2 = nn.Embedding(vocab_size, rank)


class _DeepSeekV4MTPConfidenceHead(nn.Module):
    def __init__(self, *, hidden_size: int, rank: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size + rank, 1, bias=False)


class DeepSeekV4MTPAuxiliaryShell(nn.Module):
    """Checkpoint-shaped MTP graph that cannot enter ordinary target traversal."""

    def __init__(self, config, *, device: torch.device | str = "meta"):
        super().__init__()
        from transformers.modeling_utils import local_torch_dtype
        from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
            DeepseekV4DecoderLayer,
            DeepseekV4HyperConnection,
            DeepseekV4HyperHead,
            DeepseekV4RMSNorm,
            DeepseekV4TopKRouter,
        )

        target_layer_ids = deepseek_v4_mtp_target_layer_ids(config)
        self.base_num_hidden_layers = int(config.num_hidden_layers)
        self.target_layer_ids = target_layer_ids
        self.config = copy.deepcopy(config)
        self.config.num_hidden_layers = self.base_num_hidden_layers + MTP_BLOCK_COUNT
        self.config.layer_types = list(config.layer_types) + ["sliding_attention"] * MTP_BLOCK_COUNT
        self.config.mlp_layer_types = list(config.mlp_layer_types) + ["moe"] * MTP_BLOCK_COUNT

        source_dtype = getattr(config, "dtype", None)
        if not isinstance(source_dtype, torch.dtype) or not source_dtype.is_floating_point:
            source_dtype = torch.bfloat16
        with local_torch_dtype(source_dtype, model_class_name=type(self).__name__):
            with torch.device(device):
                self.mtp = nn.ModuleList(
                    [
                        DeepseekV4DecoderLayer(self.config, self.base_num_hidden_layers + index)
                        for index in range(MTP_BLOCK_COUNT)
                    ]
                )
                self.mtp[0].main_proj = nn.Linear(
                    len(target_layer_ids) * int(config.hidden_size),
                    int(config.hidden_size),
                    bias=False,
                )
                self.mtp[0].main_norm = DeepseekV4RMSNorm(
                    int(config.hidden_size), eps=float(config.rms_norm_eps)
                )

                terminal = self.mtp[-1]
                terminal.hc_head = DeepseekV4HyperHead(self.config)
                terminal.norm = DeepseekV4RMSNorm(
                    int(config.hidden_size), eps=float(config.rms_norm_eps)
                )
                markov_rank = int(getattr(config, "dspark_markov_rank", 0))
                if markov_rank <= 0:
                    raise ValueError(f"DeepSeek V4 MTP requires dspark_markov_rank, got {markov_rank}")
                terminal.markov_head = _DeepSeekV4MTPMarkovHead(
                    vocab_size=int(config.vocab_size), rank=markov_rank
                )
                terminal.confidence_head = _DeepSeekV4MTPConfidenceHead(
                    hidden_size=int(config.hidden_size), rank=markov_rank
                )

        # These tensors are intentionally FP32 in the source checkpoint even
        # though the projection/norm/expert storage dtype is BF16.
        for module in self.modules():
            if isinstance(module, DeepseekV4HyperConnection):
                for name in ("fn", "base", "scale"):
                    setattr(module, name, nn.Parameter(getattr(module, name).to(dtype=torch.float32)))
            elif isinstance(module, DeepseekV4HyperHead):
                for name in ("hc_fn", "hc_base", "hc_scale"):
                    setattr(module, name, nn.Parameter(getattr(module, name).to(dtype=torch.float32)))
            elif isinstance(module, DeepseekV4TopKRouter):
                module.register_buffer(
                    "e_score_correction_bias",
                    module.e_score_correction_bias.to(dtype=torch.float32),
                    persistent=True,
                )
        for block in self.mtp:
            block.self_attn.sinks = nn.Parameter(block.self_attn.sinks.to(dtype=torch.float32))

        import defuser

        if not defuser.convert_model(self, cleanup_original=False):
            raise RuntimeError("Defuser did not expand DeepSeek V4 MTP expert tensors")
        patched = patch_deepseek_v4_router_precision(self)
        if patched != MTP_BLOCK_COUNT:
            raise RuntimeError(
                "DeepSeek V4 MTP FP32 router coverage mismatch: "
                f"patched={patched} expected={MTP_BLOCK_COUNT}"
            )
        self.eval()

    def forward(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError(
            "DeepSeek V4 MTP is an auxiliary graph: supply the three target taps, "
            "projected main KV, and joint anchor/noise proposal rows through the "
            "explicit MTP replay path; it must not be appended to target layers."
        )


@dataclass(frozen=True)
class DeepSeekV4MTPAuxiliary:
    """A separate MTP shell and checkpoint source for custom replay/quantization."""

    model: DeepSeekV4MTPAuxiliaryShell
    turtle_model: object
    checkpoint_contract: dict

    def block(self, index: int) -> nn.Module:
        if index < 0 or index >= MTP_BLOCK_COUNT:
            raise IndexError(f"MTP block index {index} outside [0, {MTP_BLOCK_COUNT})")
        return self.model.mtp[index]

    def checkpoint_tensors_for_submodule(self, target_submodule: nn.Module, *, recurse: bool = False):
        return self.turtle_model.checkpoint_tensors_for_submodule(
            target_model=self.model,
            target_submodule=target_submodule,
            recurse=recurse,
        )

    def materialize_nonquant_submodule(
        self,
        target_submodule: nn.Module,
        *,
        device: torch.device | str,
        recurse: bool = True,
    ) -> nn.Module:
        checkpoint_tensors = self.checkpoint_tensors_for_submodule(target_submodule, recurse=False)
        weight = checkpoint_tensors.get("weight")
        if isinstance(weight, torch.Tensor) and (
            weight.dtype in {torch.uint8, torch.int8}
            or str(weight.dtype).startswith(("torch.float8_", "torch.float4_"))
        ):
            raise RuntimeError(
                "MTP floatx projection materialization requires the auto module decoder; "
                "use DeepSeekV4QModel.build_mtp_quant_source_module()."
            )
        return self.turtle_model.materialize_submodule(
            target_model=self.model,
            target_submodule=target_submodule,
            device=torch.device(device),
            recurse=recurse,
        )


def _fp32_topk_router_forward(self, hidden_states: torch.Tensor):
    """Match V4's serving router without changing the stored BF16 weights."""

    flat = hidden_states.reshape(-1, self.hidden_dim)
    logits = F.linear(flat.float(), self.weight.float())
    scores = self.score_fn(logits)
    indices = torch.topk(
        scores + self.e_score_correction_bias.float(),
        self.top_k,
        dim=-1,
        sorted=False,
    ).indices
    weights = scores.gather(1, indices)
    weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    return logits, weights * self.routed_scaling_factor, indices


def _fp32_hash_router_forward(
    self, hidden_states: torch.Tensor, input_ids: torch.Tensor
):
    """Keep hash-selected routes while calculating their weights in FP32."""

    flat = hidden_states.reshape(-1, self.hidden_dim)
    logits = F.linear(flat.float(), self.weight.float())
    scores = self.score_fn(logits)
    indices = self.tid2eid[input_ids.reshape(-1)].long()
    weights = scores.gather(1, indices)
    weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    return logits, weights * self.routed_scaling_factor, indices


def patch_deepseek_v4_router_precision(model) -> int:
    """Use FP32 router math during calibration/replay and return patch count.

    The checkpoint weights remain BF16. Stock Transformers returns BF16 logits
    for BF16 inputs and weights; near top-k boundaries this changes selected
    experts relative to the native V4 serving kernel. Explicit FP32 math keeps
    the source representation untouched and matches natural serving routes.
    """

    patched = 0
    for name, module in model.named_modules():
        if not name.endswith(".mlp.gate"):
            continue
        if getattr(module, "_gptqmodel_v4_fp32_router", False):
            patched += 1
            continue
        if hasattr(module, "e_score_correction_bias"):
            forward = _fp32_topk_router_forward
        elif hasattr(module, "tid2eid"):
            forward = _fp32_hash_router_forward
        else:
            continue
        module.forward = MethodType(forward, module)
        module._gptqmodel_v4_fp32_router = True
        patched += 1
    return patched


class DeepSeekV4QModel(DeepSeekV3QModel):
    dynamic_expert_index = "n_routed_experts"
    rotary_embedding = "model.rotary_emb"
    # Transformers intentionally ignores the integrated dSpark/MTP checkpoint
    # namespace. Preserve it byte-for-byte until a quantizer explicitly
    # replaces those tensors; silently dropping it produces an incomplete V4
    # artifact even when normal target-layer quantization succeeds.
    out_of_model_tensors = {"prefixes": ["mtp"]}
    router_compute_dtype = torch.float32
    # This tree is deliberately separate from ``module_tree``. Feeding it to
    # the normal ModuleLooper would serialize target layer 42 into MTP block 0
    # and silently lose the three target taps and five proposal rows.
    mtp_auxiliary_module_tree = deepseek_v4_mtp_module_tree()
    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": (
                "q_a_norm:!",
                "q_a_proj:0",
                "q_b_norm:!",
                "q_b_proj:0",
                "o_a_proj:!",
                "o_b_proj:1",
                "kv_norm:!",
                "kv_proj:2",
            ),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp:moe": {
                "gate": ("gate:!",),
                "experts": {
                    "#": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                },
                "shared_experts": ("gate_proj:0", "up_proj:0", "down_proj:1"),
            },
        },
    ]

    def after_model_load(self, model, load_quantized_model=False):
        patched = patch_deepseek_v4_router_precision(model)
        expected = int(getattr(model.config, "num_hidden_layers", 0))
        if expected <= 0 or patched != expected:
            raise RuntimeError(
                "DeepSeek V4 FP32 router coverage mismatch: "
                f"patched={patched} expected={expected}"
            )
        return model

    def build_mtp_auxiliary(
        self,
        *,
        device: torch.device | str = "meta",
    ) -> DeepSeekV4MTPAuxiliary:
        """Build the checkpoint-backed MTP shell without attaching it to target layers."""

        from ...utils.structure import LazyTurtle

        model_local_path = getattr(self, "model_local_path", None)
        if not model_local_path:
            raise RuntimeError("DeepSeek V4 MTP auxiliary loading requires a local checkpoint snapshot")
        shell = DeepSeekV4MTPAuxiliaryShell(self.model.config, device=device)
        turtle = LazyTurtle.maybe_create(
            model_local_path=model_local_path,
            config=shell.config,
            model_init_kwargs={"device_map": {"": "cpu"}},
            module_tree=copy.deepcopy(self.mtp_auxiliary_module_tree),
            hf_conversion_map_reversed=deepseek_v4_mtp_checkpoint_mapping_reversed(),
            target_model=shell,
        )
        if turtle is None:
            raise RuntimeError(f"DeepSeek V4 MTP cannot open checkpoint snapshot: {model_local_path}")
        contract = validate_deepseek_v4_mtp_checkpoint_keys(
            self.model.config,
            turtle._weight_map.keys(),
        )
        return DeepSeekV4MTPAuxiliary(
            model=shell,
            turtle_model=turtle,
            checkpoint_contract=contract,
        )

    def build_mtp_quant_source_module(
        self,
        auxiliary: DeepSeekV4MTPAuxiliary,
        target_submodule: nn.Module,
        *,
        target_dtype: torch.dtype | None = None,
    ) -> nn.Module:
        """Decode one MTP FP8/FP4 projection into the normal dense quant source."""

        if not isinstance(auxiliary, DeepSeekV4MTPAuxiliary):
            raise TypeError("auxiliary must be a DeepSeekV4MTPAuxiliary")
        checkpoint_tensors = auxiliary.checkpoint_tensors_for_submodule(target_submodule)
        if not isinstance(checkpoint_tensors.get("weight"), torch.Tensor):
            raise RuntimeError("MTP quant source module has no checkpoint weight")
        if target_dtype is None:
            decoder_config = self._active_auto_module_decoder_config()
            target_dtype = (
                decoder_config.target_dtype
                if decoder_config is not None
                else torch.bfloat16
            )
        if target_dtype not in {torch.bfloat16, torch.float16}:
            raise ValueError(f"Unsupported MTP quant source dtype: {target_dtype}")
        return self._build_decoder_quant_source_module(
            target_submodule,
            checkpoint_tensors=checkpoint_tensors,
            target_dtype=target_dtype,
        )



__all__ = [
    "MTP_BLOCK_COUNT",
    "DeepSeekV4MTPAuxiliary",
    "DeepSeekV4MTPAuxiliaryShell",
    "DeepSeekV4QModel",
    "deepseek_v4_mtp_checkpoint_mapping_reversed",
    "deepseek_v4_mtp_module_tree",
    "deepseek_v4_mtp_target_layer_ids",
    "expected_deepseek_v4_mtp_checkpoint_keys",
    "patch_deepseek_v4_router_precision",
    "validate_deepseek_v4_mtp_checkpoint_keys",
]
