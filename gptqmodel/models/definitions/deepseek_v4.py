# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import copy
from dataclasses import dataclass
from types import MethodType
from typing import Callable, Iterable

import torch
import torch.nn.functional as F
from torch import nn

from .deepseek_v3 import DeepSeekV3QModel


MTP_BLOCK_COUNT = 3
MTP_CAPTURE_INPUT_IDS = "_gptqmodel_mtp_input_ids"
MTP_CAPTURE_ATTENTION_MASK = "_gptqmodel_mtp_attention_mask"
MTP_CAPTURE_DECODE_MASK = "_gptqmodel_mtp_decode_mask"


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
            DeepseekV4HyperHead,
            DeepseekV4RMSNorm,
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

        precision = patch_deepseek_v4_checkpoint_precision(self)
        expected_precision = {
            "hyper_connections": 2 * MTP_BLOCK_COUNT,
            "hyper_heads": 1,
            "attention_sinks": MTP_BLOCK_COUNT,
            "compressor_position_biases": 0,
            "router_correction_biases": MTP_BLOCK_COUNT,
        }
        if precision != expected_precision:
            raise RuntimeError(
                "DeepSeek V4 MTP FP32 source-tensor coverage mismatch: "
                f"patched={precision} expected={expected_precision}"
            )

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


@dataclass(frozen=True)
class DeepSeekV4MTPReplayBatch:
    """One jointly issued batch of natural dSpark decode positions.

    ``target_taps`` are the already-collapsed (four-lane mean) hidden states
    after the three configured target layers.  They retain a main-context
    window, not merely the final decode row, because every dSpark block reads
    up to 128 projected target-main KV rows.
    """

    target_taps: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None
    anchor_token_ids: torch.Tensor
    main_position_ids: torch.Tensor
    main_attention_mask: torch.Tensor
    projected_main: torch.Tensor | None = None


@dataclass(frozen=True)
class DeepSeekV4MTPReplayRoute:
    block_index: int
    logits: torch.Tensor
    weights: torch.Tensor
    indices: torch.Tensor


@dataclass(frozen=True)
class DeepSeekV4MTPReplayResult:
    projected_main: torch.Tensor
    proposal_token_ids: torch.Tensor
    proposal_position_ids: torch.Tensor
    terminal_residual: torch.Tensor
    routes: tuple[DeepSeekV4MTPReplayRoute, ...]


@dataclass(frozen=True)
class DeepSeekV4MTPTargetTapEvent:
    """Post-quantized target boundary consumed by an external durable sink."""

    layer_index: int
    layer_name: str
    collapsed_target_taps: tuple[torch.Tensor, ...]
    raw_layer_outputs: tuple[torch.Tensor, ...]
    layer_input_kwargs: tuple[dict, ...]
    position_ids: tuple[torch.Tensor, ...]
    attention_masks: tuple[torch.Tensor | None, ...]


class DeepSeekV4TargetAnchorResolver:
    """Resolve deterministic target tokens from the native V4 output head.

    The reference implementation collapses the final mHC streams, applies the
    target RMSNorm, and computes the vocabulary projection in FP32 even though
    the checkpoint head is stored in BF16.  Chunking positions and vocabulary
    rows preserves that arithmetic without materializing a corpus-sized logits
    tensor.  Ineligible rows remain ``-1`` and must never enter MTP replay.
    """

    def __init__(
        self,
        *,
        hc_head: nn.Module,
        norm: nn.Module,
        lm_head: nn.Module,
        position_chunk_size: int = 32,
        vocab_chunk_size: int = 8192,
    ) -> None:
        if not isinstance(hc_head, nn.Module) or not isinstance(norm, nn.Module):
            raise TypeError("DeepSeek V4 target anchor head and norm must be modules")
        if not isinstance(lm_head, nn.Module) or not isinstance(
            getattr(lm_head, "weight", None), torch.Tensor
        ):
            raise TypeError("DeepSeek V4 target anchor lm_head must expose a weight")
        if getattr(lm_head, "bias", None) is not None:
            raise ValueError("DeepSeek V4 target anchor lm_head must be bias-free")
        if position_chunk_size <= 0 or vocab_chunk_size <= 0:
            raise ValueError("DeepSeek V4 target anchor chunk sizes must be positive")
        weight = lm_head.weight
        if weight.ndim != 2 or not weight.dtype.is_floating_point:
            raise ValueError("DeepSeek V4 target anchor weight must be a floating rank-2 tensor")
        self.hc_head = hc_head
        self.norm = norm
        self.lm_head = lm_head
        self.position_chunk_size = int(position_chunk_size)
        self.vocab_chunk_size = int(vocab_chunk_size)

    def __call__(
        self,
        raw_hidden: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decode_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(raw_hidden, torch.Tensor) or raw_hidden.ndim != 4:
            raise ValueError(
                "DeepSeek V4 target anchor input must be [batch, sequence, hc, hidden]"
            )
        batch_sequence = tuple(raw_hidden.shape[:2])
        for name, value in (
            ("input IDs", input_ids),
            ("attention mask", attention_mask),
            ("decode mask", decode_mask),
            ("position IDs", position_ids),
        ):
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != batch_sequence:
                raise ValueError(
                    f"DeepSeek V4 target anchor {name} must have shape {batch_sequence}"
                )
            if value.device != raw_hidden.device:
                raise ValueError(
                    f"DeepSeek V4 target anchor {name} must share the residual device"
                )
        if raw_hidden.shape[-1] != int(self.lm_head.weight.shape[1]):
            raise ValueError("DeepSeek V4 target anchor residual/head width mismatch")
        for module_name, module in (
            ("HC head", self.hc_head),
            ("norm", self.norm),
            ("lm_head", self.lm_head),
        ):
            parameters = tuple(module.parameters())
            if any(parameter.is_meta for parameter in parameters):
                raise ValueError(f"DeepSeek V4 target anchor {module_name} is still meta")
            if any(parameter.device != raw_hidden.device for parameter in parameters):
                raise ValueError(
                    f"DeepSeek V4 target anchor {module_name} must share the residual device"
                )

        eligible = attention_mask.to(dtype=torch.bool) & decode_mask.to(dtype=torch.bool)
        anchors = torch.full(
            batch_sequence,
            -1,
            dtype=torch.long,
            device=raw_hidden.device,
        )
        coordinates = torch.nonzero(eligible, as_tuple=False)
        vocab_size = int(self.lm_head.weight.shape[0])
        for position_start in range(0, int(coordinates.shape[0]), self.position_chunk_size):
            coordinate_chunk = coordinates[
                position_start : position_start + self.position_chunk_size
            ]
            if coordinate_chunk.numel() == 0:
                continue
            selected = raw_hidden[
                coordinate_chunk[:, 0], coordinate_chunk[:, 1]
            ].unsqueeze(0)
            collapsed = self.hc_head(selected).squeeze(0)
            normalized = self.norm(collapsed)
            best_values = torch.full(
                (int(coordinate_chunk.shape[0]),),
                -torch.inf,
                dtype=torch.float32,
                device=raw_hidden.device,
            )
            best_ids = torch.zeros_like(best_values, dtype=torch.long)
            for vocab_start in range(0, vocab_size, self.vocab_chunk_size):
                vocab_end = min(vocab_start + self.vocab_chunk_size, vocab_size)
                logits = F.linear(
                    normalized.float(),
                    self.lm_head.weight[vocab_start:vocab_end].float(),
                )
                chunk_values, chunk_ids = torch.max(logits, dim=-1)
                # Strict comparison preserves torch.argmax's first-index tie rule
                # while vocabulary chunks are visited in ascending order.
                replace = chunk_values > best_values
                best_values = torch.where(replace, chunk_values, best_values)
                best_ids = torch.where(
                    replace,
                    chunk_ids.to(dtype=torch.long) + vocab_start,
                    best_ids,
                )
            anchors[coordinate_chunk[:, 0], coordinate_chunk[:, 1]] = best_ids
        return anchors


@dataclass(frozen=True)
class DeepSeekV4MTPPrefixRuntime:
    """Materialized target-head and MTP-prefix state for natural replay.

    This object deliberately owns the modules whose tensors back its callables.
    Keeping those references together prevents an external launcher from
    accidentally offloading the target head or MTP main projector while the
    synchronous target-tap sink is still consuming the calibration stream.
    """

    auxiliary: DeepSeekV4MTPAuxiliary
    target_hc_head: nn.Module
    target_norm: nn.Module
    target_lm_head: nn.Module
    target_embedding: nn.Module
    anchor_resolver: DeepSeekV4TargetAnchorResolver
    device: torch.device
    dtype: torch.dtype

    def project_target_taps(
        self,
        target_taps: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Apply the official concatenation, FP4 decode, projection, and norm."""

        if not isinstance(target_taps, tuple) or len(target_taps) != MTP_BLOCK_COUNT:
            raise ValueError(
                f"DeepSeek V4 MTP prefix projection requires {MTP_BLOCK_COUNT} target taps"
            )
        expected_shape = None
        for index, tap in enumerate(target_taps):
            if not isinstance(tap, torch.Tensor) or tap.ndim != 3:
                raise ValueError(
                    f"DeepSeek V4 MTP target tap {index} must be [batch, sequence, hidden]"
                )
            if tap.device != self.device or tap.dtype != self.dtype:
                raise ValueError(
                    "DeepSeek V4 MTP target taps must share the materialized "
                    f"device/dtype {(self.device, self.dtype)}"
                )
            if expected_shape is None:
                expected_shape = tuple(tap.shape)
            elif tuple(tap.shape) != expected_shape:
                raise ValueError("DeepSeek V4 MTP target tap shapes differ")

        concatenated = torch.cat(target_taps, dim=-1)
        block_zero = self.auxiliary.block(0)
        projected = block_zero.main_norm(block_zero.main_proj(concatenated))
        if tuple(projected.shape) != expected_shape:
            raise RuntimeError(
                "DeepSeek V4 MTP main projector returned unexpected geometry: "
                f"actual={tuple(projected.shape)} expected={expected_shape}"
            )
        if projected.device != self.device or projected.dtype != self.dtype:
            raise RuntimeError(
                "DeepSeek V4 MTP main projector changed the replay device or dtype"
            )
        return projected

    def build_replay(self) -> "DeepSeekV4MTPReplay":
        """Construct the exact joint five-row replay over this materialization."""

        weight = getattr(self.target_embedding, "weight", None)
        if not isinstance(weight, torch.Tensor):
            raise RuntimeError("DeepSeek V4 target embedding has no materialized weight")
        return DeepSeekV4MTPReplay(self.auxiliary, embedding_weight=weight)


class DeepSeekV4MTPReplay:
    """Exact, allocation-tolerant replay of the integrated V4 dSpark body.

    This path follows the checkpoint's reference inference implementation:
    mean the four target mHC lanes before persistence, concatenate the three
    target taps, project once, build one anchor plus four noise rows together,
    prime each block with the same target-main window, and expose all five
    proposal KV rows non-causally.  It is an offline calibration/reference
    path; serving continues to use its preplanned native arenas.
    """

    def __init__(
        self,
        auxiliary: DeepSeekV4MTPAuxiliary | DeepSeekV4MTPAuxiliaryShell,
        *,
        embedding_weight: torch.Tensor,
        noise_token_id: int | None = None,
    ) -> None:
        shell = auxiliary.model if isinstance(auxiliary, DeepSeekV4MTPAuxiliary) else auxiliary
        if not isinstance(shell, DeepSeekV4MTPAuxiliaryShell):
            raise TypeError("DeepSeek V4 MTP replay requires an auxiliary shell")
        if not isinstance(embedding_weight, torch.Tensor) or embedding_weight.ndim != 2:
            raise TypeError("DeepSeek V4 MTP replay embedding_weight must be rank-2")
        hidden_size = int(shell.config.hidden_size)
        vocab_size = int(shell.config.vocab_size)
        if tuple(embedding_weight.shape) != (vocab_size, hidden_size):
            raise ValueError(
                "DeepSeek V4 MTP replay embedding geometry mismatch: "
                f"actual={tuple(embedding_weight.shape)} expected={(vocab_size, hidden_size)}"
            )
        configured_noise = int(getattr(shell.config, "dspark_noise_token_id", -1))
        self.noise_token_id = configured_noise if noise_token_id is None else int(noise_token_id)
        if not 0 <= self.noise_token_id < vocab_size:
            raise ValueError(
                f"DeepSeek V4 MTP replay noise token {self.noise_token_id} outside [0, {vocab_size})"
            )
        block_size = int(getattr(shell.config, "dspark_block_size", 0))
        if block_size != 5:
            raise ValueError(f"DeepSeek V4 MTP replay requires five proposal rows, got {block_size}")
        self.shell = shell
        self.embedding_weight = embedding_weight

    @staticmethod
    def collapse_target_layer_output(hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply the official target-tap four-lane mean before persistence."""

        if not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim != 4:
            raise ValueError(
                "DeepSeek V4 target output must be [batch, sequence, hc, hidden]"
            )
        if hidden_states.shape[2] <= 0:
            raise ValueError("DeepSeek V4 target output has no mHC lanes")
        return hidden_states.mean(dim=2)

    def _validate_batch(self, batch: DeepSeekV4MTPReplayBatch) -> tuple[int, int, int]:
        if not isinstance(batch, DeepSeekV4MTPReplayBatch):
            raise TypeError("batch must be DeepSeekV4MTPReplayBatch")
        if batch.projected_main is not None:
            if batch.target_taps is not None:
                raise ValueError(
                    "DeepSeek V4 MTP replay accepts target taps or projected main, not both"
                )
            first = batch.projected_main
        else:
            if batch.target_taps is None or len(batch.target_taps) != MTP_BLOCK_COUNT:
                raise ValueError(
                    f"DeepSeek V4 MTP replay requires {MTP_BLOCK_COUNT} target taps"
                )
            first = batch.target_taps[0]
        if not isinstance(first, torch.Tensor) or first.ndim != 3:
            raise ValueError(
                "collapsed target taps/projected main must be [batch, main_rows, hidden]"
            )
        batch_size, main_rows, hidden_size = map(int, first.shape)
        expected_hidden = int(self.shell.config.hidden_size)
        if batch_size <= 0 or not 1 <= main_rows <= int(self.shell.config.sliding_window):
            raise ValueError(
                "DeepSeek V4 MTP main window must contain 1.."
                f"{int(self.shell.config.sliding_window)} rows"
            )
        if hidden_size != expected_hidden:
            raise ValueError(
                f"DeepSeek V4 MTP target width {hidden_size} != {expected_hidden}"
            )
        if batch.target_taps is not None:
            for index, tap in enumerate(batch.target_taps):
                if not isinstance(tap, torch.Tensor) or tuple(tap.shape) != tuple(first.shape):
                    raise ValueError(
                        f"DeepSeek V4 MTP target tap {index} shape mismatch: "
                        f"actual={getattr(tap, 'shape', None)} expected={tuple(first.shape)}"
                    )
                if tap.device != first.device or tap.dtype != first.dtype:
                    raise ValueError("DeepSeek V4 MTP target taps must share dtype and device")
        if tuple(batch.anchor_token_ids.shape) != (batch_size,):
            raise ValueError(
                f"DeepSeek V4 MTP anchors must have shape {(batch_size,)}"
            )
        if batch.anchor_token_ids.device != first.device:
            raise ValueError("DeepSeek V4 MTP anchors must share the target-tap device")
        if batch.anchor_token_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("DeepSeek V4 MTP anchors must be integer token IDs")
        if bool(torch.any(batch.anchor_token_ids < 0)) or bool(
            torch.any(batch.anchor_token_ids >= int(self.shell.config.vocab_size))
        ):
            raise ValueError("DeepSeek V4 MTP anchor token is outside the vocabulary")
        expected_main = (batch_size, main_rows)
        if tuple(batch.main_position_ids.shape) != expected_main:
            raise ValueError(
                f"DeepSeek V4 MTP main positions must have shape {expected_main}"
            )
        if tuple(batch.main_attention_mask.shape) != expected_main:
            raise ValueError(
                f"DeepSeek V4 MTP main mask must have shape {expected_main}"
            )
        if batch.main_position_ids.device != first.device or batch.main_attention_mask.device != first.device:
            raise ValueError("DeepSeek V4 MTP metadata must share the target-tap device")
        mask = batch.main_attention_mask.to(dtype=torch.bool)
        for row in range(batch_size):
            valid = torch.nonzero(mask[row], as_tuple=False).flatten()
            if valid.numel() == 0:
                raise ValueError(f"DeepSeek V4 MTP batch row {row} has no main context")
            expected_start = main_rows - int(valid.numel())
            expected_indices = torch.arange(expected_start, main_rows, device=valid.device)
            if not torch.equal(valid, expected_indices):
                raise ValueError(
                    "DeepSeek V4 MTP main context must be a contiguous right-aligned suffix"
                )
            positions = batch.main_position_ids[row, valid].to(dtype=torch.long)
            if bool(torch.any(positions < 0)):
                raise ValueError("DeepSeek V4 MTP positions must be non-negative")
            if positions.numel() > 1 and not torch.equal(
                positions[1:], positions[:-1] + 1
            ):
                raise ValueError("DeepSeek V4 MTP valid main positions must be contiguous")
        return batch_size, main_rows, hidden_size

    def _proposal_metadata(
        self, batch: DeepSeekV4MTPReplayBatch, *, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = int(batch.anchor_token_ids.shape[0])
        proposal_ids = torch.full(
            (batch_size, 5),
            self.noise_token_id,
            dtype=batch.anchor_token_ids.dtype,
            device=batch.anchor_token_ids.device,
        )
        proposal_ids[:, 0].copy_(batch.anchor_token_ids)
        mask = batch.main_attention_mask.to(dtype=torch.bool)
        last_main = torch.stack(
            [batch.main_position_ids[row, torch.nonzero(mask[row], as_tuple=False)[-1, 0]] for row in range(batch_size)]
        ).to(dtype=torch.long)
        proposal_positions = last_main[:, None] + torch.arange(
            1, 6, dtype=torch.long, device=last_main.device
        )[None, :]
        proposal_visible = torch.ones(
            (batch_size, 5), dtype=torch.bool, device=mask.device
        )
        kv_visible = torch.cat([mask, proposal_visible], dim=1)
        attention_mask = torch.zeros(
            (batch_size, 1, 5, kv_visible.shape[1]),
            dtype=dtype,
            device=mask.device,
        )
        attention_mask.masked_fill_(
            ~kv_visible[:, None, None, :],
            torch.finfo(attention_mask.dtype).min,
        )
        return proposal_ids, proposal_positions, attention_mask

    def _prime_main_cache(
        self,
        *,
        block: nn.Module,
        projected_main: torch.Tensor,
        main_position_ids: torch.Tensor,
        rotary: nn.Module,
    ):
        from transformers.cache_utils import DynamicCache
        from transformers.models.deepseek_v4.modeling_deepseek_v4 import apply_rotary_pos_emb

        input_shape = projected_main.shape[:-1]
        hidden_shape = (*input_shape, -1, int(block.self_attn.head_dim))
        cos, sin = rotary(
            projected_main, position_ids=main_position_ids, layer_type="main"
        )
        kv = block.self_attn.kv_norm(block.self_attn.kv_proj(projected_main))
        kv = kv.view(*hidden_shape).transpose(1, 2)
        kv = apply_rotary_pos_emb(kv, cos, sin)
        # Do not configure this as a Transformers sliding cache: that class
        # retains 127 prior rows for a five-row query, while native dSpark
        # deliberately attends to 128 target-main plus five proposal rows.
        cache = DynamicCache()
        cache.update(kv, kv, block.layer_idx)
        return cache

    def replay(
        self,
        batch: DeepSeekV4MTPReplayBatch,
        *,
        prepare_ffn: Callable[[int, nn.Module, torch.Tensor, torch.Tensor], None] | None = None,
    ) -> DeepSeekV4MTPReplayResult:
        """Run all three blocks while preserving the joint five-row batch."""

        from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
            DeepseekV4RotaryEmbedding,
        )

        self._validate_batch(batch)
        first = (
            batch.projected_main
            if batch.projected_main is not None
            else batch.target_taps[0]
        )
        if self.embedding_weight.device != first.device:
            raise ValueError("DeepSeek V4 MTP embedding and target taps must share a device")
        if self.embedding_weight.dtype != first.dtype:
            raise ValueError("DeepSeek V4 MTP embedding and target taps must share a dtype")
        if batch.projected_main is None:
            concatenated = torch.cat(batch.target_taps, dim=-1)
            projected_main = self.shell.mtp[0].main_norm(
                self.shell.mtp[0].main_proj(concatenated)
            )
        else:
            projected_main = batch.projected_main
        proposal_ids, proposal_positions, attention_mask = self._proposal_metadata(
            batch, dtype=first.dtype
        )
        proposal_embedding = F.embedding(proposal_ids, self.embedding_weight)
        residual = proposal_embedding.unsqueeze(2).expand(
            -1, -1, int(self.shell.config.hc_mult), -1
        ).contiguous()
        rotary = DeepseekV4RotaryEmbedding(self.shell.config).to(
            device=first.device, dtype=first.dtype
        )
        proposal_position_embeddings = {
            "main": rotary(
                proposal_embedding,
                position_ids=proposal_positions,
                layer_type="main",
            )
        }
        routes: list[DeepSeekV4MTPReplayRoute] = []
        for block_index, block in enumerate(self.shell.mtp):
            cache = self._prime_main_cache(
                block=block,
                projected_main=projected_main,
                main_position_ids=batch.main_position_ids,
                rotary=rotary,
            )
            dtype = residual.dtype
            post, comb, collapsed = block.attn_hc(residual)
            attention_output, _ = block.self_attn(
                block.input_layernorm(collapsed),
                position_embeddings=proposal_position_embeddings,
                position_ids=proposal_positions,
                attention_mask=attention_mask,
                past_key_values=cache,
            )
            residual = post.to(dtype).unsqueeze(-1) * attention_output.unsqueeze(
                -2
            ) + torch.matmul(comb.to(dtype).transpose(-1, -2), residual)

            post, comb, collapsed = block.ffn_hc(residual)
            ffn_input = block.post_attention_layernorm(collapsed)
            if prepare_ffn is not None:
                prepare_ffn(block_index, block, ffn_input, proposal_ids)

            captured: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

            def capture_router(_module, _args, _kwargs, output):
                captured.append(output)

            handle = block.mlp.gate.register_forward_hook(capture_router, with_kwargs=True)
            try:
                mlp_output = block.mlp(ffn_input, input_ids=proposal_ids)
            finally:
                handle.remove()
            if len(captured) != 1 or len(captured[0]) != 3:
                raise RuntimeError(
                    f"DeepSeek V4 MTP block {block_index} router was not captured exactly once"
                )
            logits, weights, indices = captured[0]
            routes.append(
                DeepSeekV4MTPReplayRoute(
                    block_index=block_index,
                    logits=logits.reshape(*proposal_ids.shape, -1),
                    weights=weights.reshape(*proposal_ids.shape, -1),
                    indices=indices.reshape(*proposal_ids.shape, -1),
                )
            )
            residual = post.to(dtype).unsqueeze(-1) * mlp_output.unsqueeze(
                -2
            ) + torch.matmul(comb.to(dtype).transpose(-1, -2), residual)

        return DeepSeekV4MTPReplayResult(
            projected_main=projected_main,
            proposal_token_ids=proposal_ids,
            proposal_position_ids=proposal_positions,
            terminal_residual=residual,
            routes=tuple(routes),
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


def patch_deepseek_v4_checkpoint_precision(model) -> dict[str, int]:
    """Preserve every V4 checkpoint tensor intentionally stored in FP32.

    Transformers constructs these modules under the model-wide BF16 dtype.
    Lazy checkpoint materialization then follows the shell dtype, so leaving
    the shell untouched silently rounds native FP32 control tensors before the
    first calibration forward.  Change only their destination storage dtype;
    ordinary projections, norms, embeddings, and expert weights stay BF16 or
    in their native packed source format.
    """

    from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
        DeepseekV4Attention,
        DeepseekV4CSACompressor,
        DeepseekV4HCACompressor,
        DeepseekV4HyperConnection,
        DeepseekV4HyperHead,
        DeepseekV4Indexer,
        DeepseekV4TopKRouter,
    )

    counts = {
        "hyper_connections": 0,
        "hyper_heads": 0,
        "attention_sinks": 0,
        "compressor_position_biases": 0,
        "router_correction_biases": 0,
    }

    def fp32_parameter(module: nn.Module, name: str) -> None:
        parameter = getattr(module, name)
        if not isinstance(parameter, nn.Parameter):
            raise RuntimeError(
                f"DeepSeek V4 source-precision tensor {type(module).__name__}.{name} "
                "is not a parameter"
            )
        setattr(
            module,
            name,
            nn.Parameter(
                parameter.detach().to(dtype=torch.float32),
                requires_grad=parameter.requires_grad,
            ),
        )

    for module in model.modules():
        if isinstance(module, DeepseekV4HyperConnection):
            for name in ("fn", "base", "scale"):
                fp32_parameter(module, name)
            counts["hyper_connections"] += 1
        elif isinstance(module, DeepseekV4HyperHead):
            for name in ("hc_fn", "hc_base", "hc_scale"):
                fp32_parameter(module, name)
            counts["hyper_heads"] += 1

        if isinstance(module, DeepseekV4Attention):
            fp32_parameter(module, "sinks")
            counts["attention_sinks"] += 1
        if isinstance(
            module,
            (DeepseekV4HCACompressor, DeepseekV4CSACompressor, DeepseekV4Indexer),
        ):
            fp32_parameter(module, "position_bias")
            counts["compressor_position_biases"] += 1
        if isinstance(module, DeepseekV4TopKRouter):
            bias = getattr(module, "e_score_correction_bias", None)
            if not isinstance(bias, torch.Tensor):
                raise RuntimeError("DeepSeek V4 learned router lacks correction bias")
            module.register_buffer(
                "e_score_correction_bias",
                bias.detach().to(dtype=torch.float32),
                persistent=True,
            )
            counts["router_correction_biases"] += 1
    return counts


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
        precision = patch_deepseek_v4_checkpoint_precision(model)
        expected_layers = int(getattr(model.config, "num_hidden_layers", 0))
        if expected_layers <= 0:
            raise RuntimeError("DeepSeek V4 source-precision patch has no target layers")
        layer_types = tuple(getattr(model.config, "layer_types", ()) or ())
        if len(layer_types) != expected_layers:
            raise RuntimeError(
                "DeepSeek V4 attention-layer precision contract mismatch: "
                f"layer_types={len(layer_types)} expected={expected_layers}"
            )
        unsupported_layer_types = sorted(
            set(layer_types)
            - {
                "sliding_attention",
                "compressed_sparse_attention",
                "heavily_compressed_attention",
            }
        )
        if unsupported_layer_types:
            raise RuntimeError(
                "DeepSeek V4 source-precision patch does not recognize attention "
                f"types {unsupported_layer_types}"
            )
        expected_compressor_biases = sum(
            2 if layer_type == "compressed_sparse_attention" else 1
            for layer_type in layer_types
            if layer_type != "sliding_attention"
        )
        mlp_layer_types = tuple(getattr(model.config, "mlp_layer_types", ()) or ())
        if len(mlp_layer_types) != expected_layers:
            raise RuntimeError(
                "DeepSeek V4 MLP-layer precision contract mismatch: "
                f"mlp_layer_types={len(mlp_layer_types)} expected={expected_layers}"
            )
        unsupported_mlp_types = sorted(set(mlp_layer_types) - {"hash_moe", "moe"})
        if unsupported_mlp_types:
            raise RuntimeError(
                "DeepSeek V4 source-precision patch does not recognize MLP types "
                f"{unsupported_mlp_types}"
            )
        expected_router_biases = sum(
            layer_type == "moe" for layer_type in mlp_layer_types
        )
        if precision["hyper_connections"] != 2 * expected_layers:
            raise RuntimeError(
                "DeepSeek V4 FP32 Hyper-Connection coverage mismatch: "
                f"patched={precision['hyper_connections']} expected={2 * expected_layers}"
            )
        if precision["hyper_heads"] != 1:
            raise RuntimeError(
                "DeepSeek V4 FP32 target HC-head coverage mismatch: "
                f"patched={precision['hyper_heads']} expected=1"
            )
        if precision["attention_sinks"] != expected_layers:
            raise RuntimeError(
                "DeepSeek V4 FP32 attention-sink coverage mismatch: "
                f"patched={precision['attention_sinks']} expected={expected_layers}"
            )
        if precision["compressor_position_biases"] != expected_compressor_biases:
            raise RuntimeError(
                "DeepSeek V4 FP32 compressor/indexer position-bias coverage mismatch: "
                f"patched={precision['compressor_position_biases']} "
                f"expected={expected_compressor_biases}"
            )
        if precision["router_correction_biases"] != expected_router_biases:
            raise RuntimeError(
                "DeepSeek V4 FP32 router correction-bias coverage mismatch: "
                f"patched={precision['router_correction_biases']} "
                f"expected={expected_router_biases}"
            )
        patched = patch_deepseek_v4_router_precision(model)
        if patched != expected_layers:
            raise RuntimeError(
                "DeepSeek V4 FP32 router coverage mismatch: "
                f"patched={patched} expected={expected_layers}"
            )
        return model

    def begin_input_capture_example(
        self,
        example: dict,
        batch_device: torch.device,
    ) -> None:
        metadata = {}
        for source_name, capture_name in (
            ("input_ids", MTP_CAPTURE_INPUT_IDS),
            ("attention_mask", MTP_CAPTURE_ATTENTION_MASK),
        ):
            value = example.get(source_name)
            if isinstance(value, torch.Tensor):
                metadata[capture_name] = value.detach().to(device=batch_device)
        labels = example.get("labels")
        input_ids = example.get("input_ids")
        attention_mask = example.get("attention_mask")
        if (
            isinstance(labels, torch.Tensor)
            and isinstance(input_ids, torch.Tensor)
            and tuple(labels.shape) == tuple(input_ids.shape)
        ):
            valid = (
                attention_mask.to(dtype=torch.bool)
                if isinstance(attention_mask, torch.Tensor)
                and tuple(attention_mask.shape) == tuple(input_ids.shape)
                else torch.ones_like(input_ids, dtype=torch.bool)
            )
            decode_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            decode_mask[:, :-1] = (labels[:, 1:] != -100) & valid[:, :-1] & valid[:, 1:]
            metadata[MTP_CAPTURE_DECODE_MASK] = decode_mask.detach().to(
                device=batch_device
            )
        self._mtp_input_capture_metadata = metadata

    def end_input_capture_example(self) -> None:
        self._mtp_input_capture_metadata = None

    def capture_first_layer_input_kwargs(
        self,
        args: tuple,
        kwargs: dict,
        batch_device: torch.device,
        layer_input_kwargs: dict,
    ) -> dict:
        result = super().capture_first_layer_input_kwargs(
            args=args,
            kwargs=kwargs,
            batch_device=batch_device,
            layer_input_kwargs=layer_input_kwargs,
        )
        metadata = getattr(self, "_mtp_input_capture_metadata", None) or {}
        result.update(
            {
                name: value.detach().to(device=batch_device)
                for name, value in metadata.items()
            }
        )
        return result

    def prepare_layer_replay_kwargs(
        self,
        layer: nn.Module,
        layer_input: list[torch.Tensor],
        additional_inputs: dict,
        target_device: torch.device,
    ) -> dict:
        result = super().prepare_layer_replay_kwargs(
            layer=layer,
            layer_input=layer_input,
            additional_inputs=additional_inputs,
            target_device=target_device,
        )
        result.pop(MTP_CAPTURE_INPUT_IDS, None)
        result.pop(MTP_CAPTURE_ATTENTION_MASK, None)
        result.pop(MTP_CAPTURE_DECODE_MASK, None)
        return result

    def set_mtp_target_tap_sink(
        self,
        sink: Callable[[DeepSeekV4MTPTargetTapEvent], None] | None,
    ) -> None:
        """Install the synchronous sink used by streamed MTP calibration.

        The sink must clone or durably persist anything it needs before it
        returns. The layer loop remains free to release the replay outputs
        immediately afterward.
        """

        if sink is not None and not callable(sink):
            raise TypeError("DeepSeek V4 MTP target-tap sink must be callable")
        self._mtp_target_tap_sink = sink

    def quantization_layer_output_required(
        self,
        *,
        layer_index: int,
        layer_name: str,
        layer_count: int,
    ) -> bool:
        del layer_name
        sink = getattr(self, "_mtp_target_tap_sink", None)
        if not callable(sink):
            return False
        target_ids = deepseek_v4_mtp_target_layer_ids(self.model.config)
        if layer_count != int(self.model.config.num_hidden_layers):
            raise RuntimeError(
                "DeepSeek V4 target-tap capture layer count mismatch: "
                f"looper={layer_count} config={self.model.config.num_hidden_layers}"
            )
        return int(layer_index) in target_ids

    def receive_quantization_layer_outputs(
        self,
        *,
        layer_index: int,
        layer_name: str,
        layer_outputs: list[list[torch.Tensor]],
        layer_input_kwargs: list[dict],
        position_ids: list[torch.Tensor],
        attention_masks: list[torch.Tensor | None],
    ) -> None:
        sink = getattr(self, "_mtp_target_tap_sink", None)
        if not callable(sink):
            raise RuntimeError("DeepSeek V4 MTP target-tap output has no installed sink")
        if int(layer_index) not in deepseek_v4_mtp_target_layer_ids(self.model.config):
            raise RuntimeError(
                f"DeepSeek V4 received an unexpected target-tap layer {layer_index}"
            )
        raw: list[torch.Tensor] = []
        collapsed: list[torch.Tensor] = []
        expected_hc = int(self.model.config.hc_mult)
        expected_hidden = int(self.model.config.hidden_size)
        for batch_index, outputs in enumerate(layer_outputs):
            if not isinstance(outputs, (list, tuple)) or not outputs:
                raise RuntimeError(
                    f"DeepSeek V4 target-tap batch {batch_index} has no primary output"
                )
            hidden = outputs[0]
            if not isinstance(hidden, torch.Tensor) or hidden.ndim != 4:
                raise RuntimeError(
                    "DeepSeek V4 target-tap output must be "
                    "[batch, sequence, hc, hidden]"
                )
            if tuple(hidden.shape[2:]) != (expected_hc, expected_hidden):
                raise RuntimeError(
                    "DeepSeek V4 target-tap output geometry mismatch: "
                    f"actual={tuple(hidden.shape)} expected_hc_hidden="
                    f"{(expected_hc, expected_hidden)}"
                )
            raw.append(hidden)
            collapsed.append(DeepSeekV4MTPReplay.collapse_target_layer_output(hidden))
        sink(
            DeepSeekV4MTPTargetTapEvent(
                layer_index=int(layer_index),
                layer_name=str(layer_name),
                collapsed_target_taps=tuple(collapsed),
                raw_layer_outputs=tuple(raw),
                layer_input_kwargs=tuple(layer_input_kwargs),
                position_ids=tuple(position_ids),
                attention_masks=tuple(attention_masks),
            )
        )

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

    def build_mtp_prefix_runtime(
        self,
        *,
        device: torch.device | str,
        position_chunk_size: int = 32,
        vocab_chunk_size: int = 8192,
    ) -> DeepSeekV4MTPPrefixRuntime:
        """Materialize the exact target-to-MTP calibration prefix.

        Only the two MTP main-projection leaves and the four target-side leaves
        needed for anchors/replay are loaded.  Routed experts and the rest of
        the target model remain checkpoint-backed and layerwise.
        """

        from ...utils.structure import LazyTurtle

        if not isinstance(getattr(self, "model", None), nn.Module):
            raise RuntimeError("DeepSeek V4 MTP prefix runtime requires a model shell")
        if not isinstance(getattr(self, "turtle_model", None), LazyTurtle):
            raise RuntimeError(
                "DeepSeek V4 MTP prefix runtime requires a checkpoint-backed LazyTurtle"
            )
        replay_device = torch.device(device)
        source_dtype = getattr(self.model.config, "dtype", None)
        if not isinstance(source_dtype, torch.dtype) or not source_dtype.is_floating_point:
            raise RuntimeError(
                f"DeepSeek V4 MTP prefix runtime has invalid source dtype {source_dtype!r}"
            )
        if source_dtype is not torch.bfloat16:
            raise RuntimeError(
                "DeepSeek V4 MTP prefix qualification currently requires a BF16 "
                f"checkpoint, got {source_dtype}"
            )

        auxiliary = self.build_mtp_auxiliary(device="meta")
        block_zero = auxiliary.block(0)
        self.materialize_mtp_replay_submodule(
            auxiliary,
            block_zero.main_proj,
            device=replay_device,
            target_dtype=source_dtype,
        )
        self.materialize_mtp_replay_submodule(
            auxiliary,
            block_zero.main_norm,
            device=replay_device,
            target_dtype=source_dtype,
        )

        def target_leaf(path: str) -> nn.Module:
            module = self.model.get_submodule(path)
            materialized = self.shell_module_materialize(
                target_submodule=module,
                device=replay_device,
                module_path=path,
            )
            parameters = tuple(materialized.parameters())
            buffers = tuple(materialized.buffers())
            if any(tensor.is_meta for tensor in (*parameters, *buffers)):
                raise RuntimeError(
                    f"DeepSeek V4 MTP prefix module {path} remained on the meta device"
                )
            if any(tensor.device != replay_device for tensor in (*parameters, *buffers)):
                raise RuntimeError(
                    f"DeepSeek V4 MTP prefix module {path} was materialized on the wrong device"
                )
            return materialized

        target_hc_head = target_leaf("model.hc_head")
        target_norm = target_leaf("model.norm")
        target_lm_head = target_leaf("lm_head")
        target_embedding = target_leaf("model.embed_tokens")

        hc_parameters = tuple(target_hc_head.parameters())
        if not hc_parameters or any(
            parameter.dtype is not torch.float32 for parameter in hc_parameters
        ):
            raise RuntimeError(
                "DeepSeek V4 target HC head did not retain checkpoint FP32 precision"
            )
        for path, module in (
            ("mtp.0.main_proj", block_zero.main_proj),
            ("mtp.0.main_norm", block_zero.main_norm),
            ("model.norm", target_norm),
            ("lm_head", target_lm_head),
            ("model.embed_tokens", target_embedding),
        ):
            floating = tuple(
                tensor
                for tensor in (*module.parameters(), *module.buffers())
                if tensor.is_floating_point()
            )
            if not floating or any(tensor.dtype is not source_dtype for tensor in floating):
                raise RuntimeError(
                    f"DeepSeek V4 MTP prefix module {path} did not retain {source_dtype}"
                )

        anchor_resolver = DeepSeekV4TargetAnchorResolver(
            hc_head=target_hc_head,
            norm=target_norm,
            lm_head=target_lm_head,
            position_chunk_size=position_chunk_size,
            vocab_chunk_size=vocab_chunk_size,
        )
        return DeepSeekV4MTPPrefixRuntime(
            auxiliary=auxiliary,
            target_hc_head=target_hc_head,
            target_norm=target_norm,
            target_lm_head=target_lm_head,
            target_embedding=target_embedding,
            anchor_resolver=anchor_resolver,
            device=replay_device,
            dtype=source_dtype,
        )

    def materialize_mtp_replay_submodule(
        self,
        auxiliary: DeepSeekV4MTPAuxiliary,
        target_submodule: nn.Module,
        *,
        device: torch.device | str,
        target_dtype: torch.dtype = torch.bfloat16,
    ) -> nn.Module:
        """Materialize one replay leaf while decoding native FP8/FP4 weights.

        The target object is retained so V4's patched FP32 router methods and
        Defuser expert containers remain attached to the auxiliary graph.
        """

        if not isinstance(auxiliary, DeepSeekV4MTPAuxiliary):
            raise TypeError("auxiliary must be a DeepSeekV4MTPAuxiliary")
        checkpoint_tensors = auxiliary.checkpoint_tensors_for_submodule(
            target_submodule, recurse=False
        )
        weight = checkpoint_tensors.get("weight")
        is_floatx = isinstance(weight, torch.Tensor) and (
            weight.dtype in {torch.uint8, torch.int8}
            or str(weight.dtype).startswith(("torch.float8_", "torch.float4_"))
        )
        if not is_floatx:
            return auxiliary.materialize_nonquant_submodule(
                target_submodule, device=device, recurse=True
            )

        decoded = self.build_mtp_quant_source_module(
            auxiliary,
            target_submodule,
            target_dtype=target_dtype,
        ).to(device=device)
        decoded_parameters = dict(decoded.named_parameters(recurse=False))
        if "weight" not in decoded_parameters:
            raise RuntimeError("decoded MTP replay module did not expose a direct weight")
        for name, parameter in decoded_parameters.items():
            if name not in target_submodule._parameters:
                raise RuntimeError(
                    f"decoded MTP replay parameter {name!r} is absent from the target module"
                )
            target_submodule._parameters[name] = nn.Parameter(
                parameter.detach(), requires_grad=False
            )
        return target_submodule



__all__ = [
    "MTP_BLOCK_COUNT",
    "MTP_CAPTURE_ATTENTION_MASK",
    "MTP_CAPTURE_DECODE_MASK",
    "MTP_CAPTURE_INPUT_IDS",
    "DeepSeekV4MTPAuxiliary",
    "DeepSeekV4MTPAuxiliaryShell",
    "DeepSeekV4MTPPrefixRuntime",
    "DeepSeekV4MTPReplay",
    "DeepSeekV4MTPReplayBatch",
    "DeepSeekV4MTPReplayResult",
    "DeepSeekV4MTPReplayRoute",
    "DeepSeekV4MTPTargetTapEvent",
    "DeepSeekV4TargetAnchorResolver",
    "DeepSeekV4QModel",
    "deepseek_v4_mtp_checkpoint_mapping_reversed",
    "deepseek_v4_mtp_module_tree",
    "deepseek_v4_mtp_target_layer_ids",
    "expected_deepseek_v4_mtp_checkpoint_keys",
    "patch_deepseek_v4_router_precision",
    "patch_deepseek_v4_checkpoint_precision",
    "validate_deepseek_v4_mtp_checkpoint_keys",
]
