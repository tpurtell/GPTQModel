# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0

"""GPTQModel support for the multimodal GLM-5.3 Flash checkpoint.

The public checkpoint stores one non-mHC MTP block at text layer 45 while
Transformers intentionally instantiates only target layers 0..44.  The loader
below attaches a checkpoint-shaped execution block to the lazy shell so target
and MTP routed experts are quantized in one ordinary layer traversal.
"""

from __future__ import annotations

import copy
import os
import shutil
import weakref
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import AutoModelForImageTextToText

from ...utils.exl3_router_candidates import bind_sigmoid_grouped_router_recovery
from ...utils.model import move_to
from ..base import BaseQModel
from ..moe_lifecycle import GateUpDownMoELifecycleHooks


GLM5_NEXT_MTP_LAYER = 45
GLM5_NEXT_ROUTED_EXPERT_PATTERN = (
    r"^model\.language_model\.layers\."
    r"(?:[3-9]|[1-3][0-9]|4[0-5])\.mlp\.experts\.\d+\."
    r"(?:gate_proj|up_proj|down_proj)$"
)


class _Glm5NextSharedHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        from transformers.models.glm5_next.modeling_glm5_next import (
            Glm5NextTextRMSNorm,
        )

        self.norm = Glm5NextTextRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )


class _Glm5NextDefusedExpert(nn.Module):
    def __init__(self, config, *, device: torch.device, dtype: torch.dtype):
        super().__init__()
        self.swiglu_limit = config.swiglu_limit
        self.gate_proj = nn.Linear(
            config.hidden_size,
            config.moe_intermediate_size,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.up_proj = nn.Linear(
            config.hidden_size,
            config.moe_intermediate_size,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.down_proj = nn.Linear(
            config.moe_intermediate_size,
            config.hidden_size,
            bias=False,
            device=device,
            dtype=dtype,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(hidden_states).clamp(max=self.swiglu_limit)
        up = self.up_proj(hidden_states).clamp(
            min=-self.swiglu_limit,
            max=self.swiglu_limit,
        )
        return self.down_proj(torch.nn.functional.silu(gate) * up)


class _Glm5NextDefusedExperts(nn.ModuleList):
    """Callable expert list matching the checkpoint's per-expert namespace."""

    def __init__(self, config, *, device: torch.device, dtype: torch.dtype):
        super().__init__(
            [
                _Glm5NextDefusedExpert(config, device=device, dtype=dtype)
                for _ in range(int(config.n_routed_experts))
            ]
        )
        self.num_experts = int(config.n_routed_experts)

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final = torch.zeros_like(hidden_states)
        with torch.no_grad():
            mask = torch.nn.functional.one_hot(
                top_k_index,
                num_classes=self.num_experts,
            ).permute(2, 1, 0)
            hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx_tensor in hit:
            expert_idx = int(expert_idx_tensor[0])
            top_k_pos, token_idx = torch.where(mask[expert_idx])
            current = self[expert_idx](hidden_states[token_idx])
            current = current * top_k_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, current.to(final.dtype))
        return final


def _defuse_glm5_next_experts(model: nn.Module, *, cleanup_original: bool) -> int:
    from transformers.models.glm5_next.modeling_glm5_next import (
        Glm5NextTextExperts,
    )

    replacements: list[tuple[nn.Module, Glm5NextTextExperts]] = []
    for module in model.modules():
        experts = getattr(module, "experts", None)
        if isinstance(experts, Glm5NextTextExperts):
            replacements.append((module, experts))

    for parent, source in replacements:
        gate_up = source.gate_up_proj
        down = source.down_proj
        replacement = _Glm5NextDefusedExperts(
            parent.config,
            device=gate_up.device,
            dtype=gate_up.dtype,
        )
        if not gate_up.is_meta:
            for expert_idx, expert in enumerate(replacement):
                gate, up = gate_up[expert_idx].chunk(2, dim=0)
                expert.gate_proj.weight = nn.Parameter(gate.clone())
                expert.up_proj.weight = nn.Parameter(up.clone())
                expert.down_proj.weight = nn.Parameter(down[expert_idx].clone())
        parent.experts = replacement
        if cleanup_original:
            del source
    return len(replacements)


class Glm5NextMTPDecoderLayer(nn.Module):
    """Checkpoint-exact layer-45 MTP body used only during quantization.

    GLM-5.3 MTP is not an mHC block.  It consumes the target model's final
    normalized state at token t plus the normalized embedding at token t+1,
    then runs a conventional residual MLA+MoE block.  Its final shared-head
    norm is retained natively; only routed gate/up/down projections are in the
    EXL3 allowlist.
    """

    def __init__(self, config, *, language_model: nn.Module):
        super().__init__()
        from transformers.models.glm5_next.modeling_glm5_next import (
            Glm5NextTextAttention,
            Glm5NextTextMoE,
            Glm5NextTextRMSNorm,
        )

        self.layer_idx = GLM5_NEXT_MTP_LAYER
        self.config = copy.deepcopy(config)
        self.config.layer_types = list(config.layer_types) + [
            "deepseek_sparse_attention"
        ]
        self.config.mlp_layer_types = list(config.mlp_layer_types) + ["sparse"]
        self.config.indexer_types = list(config.indexer_types) + ["full"]

        self.enorm = Glm5NextTextRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.hnorm = Glm5NextTextRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.eh_proj = nn.Linear(
            config.hidden_size * 2,
            config.hidden_size,
            bias=False,
        )
        self.input_layernorm = Glm5NextTextRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.self_attn = Glm5NextTextAttention(
            self.config,
            GLM5_NEXT_MTP_LAYER,
        )
        self.post_attention_layernorm = Glm5NextTextRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.mlp = Glm5NextTextMoE(self.config)
        self.shared_head = _Glm5NextSharedHead(config)

        # Avoid registering aliases of native target modules. LazyTurtle must
        # see exactly the checkpoint namespace and no duplicate embedding/norm
        # tensors under layer 45.
        object.__setattr__(self, "_language_model_ref", weakref.ref(language_model))

    def _target_hidden(self, hidden_streams: torch.Tensor) -> torch.Tensor:
        language_model = self._language_model_ref()
        if language_model is None:
            raise RuntimeError("GLM-5.3 MTP lost its target language-model reference")
        return language_model.norm(language_model.hc_head(hidden_streams))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        use_cache: bool | None = False,
        prev_topk_indices: torch.Tensor | None = None,
        input_ids: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if use_cache or past_key_values is not None:
            raise ValueError("GLM-5.3 MTP calibration requires cache-free replay")
        if input_ids is None or position_ids is None:
            raise ValueError("GLM-5.3 MTP calibration requires token and position IDs")
        if hidden_states.ndim != 4:
            raise ValueError(
                "GLM-5.3 target frontier must be [batch, sequence, hc, hidden]"
            )
        if hidden_states.shape[1] != input_ids.shape[1] + 1:
            raise ValueError(
                "GLM-5.3 MTP replay requires target[t] paired with token[t+1]"
            )

        language_model = self._language_model_ref()
        if language_model is None:
            raise RuntimeError("GLM-5.3 MTP lost its embedding reference")
        previous_hidden = self._target_hidden(hidden_states)[:, :-1]
        inputs_embeds = language_model.embed_tokens(input_ids)
        inputs_embeds = torch.where(
            position_ids.unsqueeze(-1).eq(0),
            torch.zeros((), dtype=inputs_embeds.dtype, device=inputs_embeds.device),
            inputs_embeds,
        )
        hidden_states = self.eh_proj(
            torch.cat(
                (self.enorm(inputs_embeds), self.hnorm(previous_hidden)),
                dim=-1,
            )
        )

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _, topk_indices = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            position_embeddings=None,
            prev_topk_indices=prev_topk_indices,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.shared_head.norm(residual + hidden_states)
        return hidden_states, topk_indices


def _attach_glm5_next_mtp(model: nn.Module) -> nn.Module:
    language_model = model.model.language_model
    if len(language_model.layers) == GLM5_NEXT_MTP_LAYER + 1:
        return model
    if len(language_model.layers) != GLM5_NEXT_MTP_LAYER:
        raise RuntimeError(
            "GLM-5.3 target layer count differs: "
            f"actual={len(language_model.layers)} expected={GLM5_NEXT_MTP_LAYER}"
        )
    language_model.layers.append(
        Glm5NextMTPDecoderLayer(
            model.config.text_config,
            language_model=language_model,
        )
    )
    return model


def _prune_glm5_next_replay_frontier(
    root: str | os.PathLike[str],
    *,
    before_layer: int,
) -> None:
    """Retain only the current durable post-quant replay frontier."""

    parent = Path(root).expanduser().resolve(strict=True)
    for candidate in parent.iterdir():
        name = candidate.name
        if name.startswith("layer-"):
            suffix = name.removeprefix("layer-")
        elif name.startswith(".layer-") and name.endswith(".partial"):
            suffix = name.removeprefix(".layer-").removesuffix(".partial")
        else:
            continue
        if len(suffix) != 6 or not suffix.isdigit():
            continue
        if int(suffix) >= before_layer:
            continue
        if not candidate.is_dir() or candidate.is_symlink():
            raise RuntimeError(
                f"GLM-5.3 replay frontier is not a real directory: {candidate}"
            )
        shutil.rmtree(candidate)


class Glm5NextQuantizationLoader:
    """Auto loader that exposes checkpoint-only MTP layer 45 to LazyTurtle."""

    @classmethod
    def from_config(cls, config, **kwargs):
        model = AutoModelForImageTextToText.from_config(config, **kwargs)
        return _attach_glm5_next_mtp(model)

    @classmethod
    def from_pretrained(cls, model_local_path, **kwargs):
        model = AutoModelForImageTextToText.from_pretrained(
            model_local_path,
            **kwargs,
        )
        return _attach_glm5_next_mtp(model)


class Glm5NextQModel(BaseQModel):
    loader = Glm5NextQuantizationLoader
    require_pkgs = ["transformers>=5.16.0"]
    require_load_processor = False
    layer_modules_strict = False

    dynamic_expert_index = "n_routed_experts"
    pre_lm_head_norm_module = "model.language_model.norm"
    rotary_embedding = None
    moe_lifecycle_hooks = GateUpDownMoELifecycleHooks()

    @classmethod
    def convert_model_structure(cls, model, *, cleanup_original: bool) -> bool:
        converted = _defuse_glm5_next_experts(
            model,
            cleanup_original=cleanup_original,
        )
        expected = int(model.config.text_config.num_hidden_layers) - int(
            model.config.text_config.first_k_dense_replace
        ) + 1
        if converted != expected:
            raise RuntimeError(
                "GLM-5.3 expert defusion coverage mismatch: "
                f"converted={converted} expected={expected}"
            )
        return True

    module_tree = [
        "model",
        "language_model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": (
                "q_proj:0",
                "k_proj:0",
                "v_proj:0",
                "q_a_proj:0",
                "kv_a_proj_with_mqa:0",
                "q_b_proj:1",
                "kv_b_proj:1",
                "o_proj:2",
            ),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp:moe": {
                "gate": ("gate:!",),
                "experts": {
                    "#": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                },
                "shared_experts": (
                    "gate_proj:0",
                    "up_proj:0",
                    "down_proj:1",
                ),
                "": ("gate_proj:0", "up_proj:0", "down_proj:1"),
            },
        },
    ]

    def after_model_load(self, model, load_quantized_model=False):
        del load_quantized_model
        config = model.config.text_config
        expected = int(config.num_hidden_layers) - int(config.first_k_dense_replace) + 1
        patched = 0
        for name, module in model.named_modules():
            if not name.endswith(".mlp.gate"):
                continue
            correction = getattr(module, "e_score_correction_bias", None)
            if not isinstance(correction, torch.Tensor):
                continue
            bind_sigmoid_grouped_router_recovery(module)
            patched += 1
        if patched != expected:
            raise RuntimeError(
                "GLM-5.3 learned-router coverage mismatch: "
                f"patched={patched} expected={expected}"
            )
        return model

    def configure_base_replay_store(
        self,
        root: str | os.PathLike[str],
        *,
        provenance: dict[str, Any],
    ) -> None:
        if not isinstance(provenance, dict) or not provenance:
            raise ValueError("base replay storage requires immutable provenance")
        self._base_replay_store_root = os.fspath(Path(root).expanduser().resolve())
        self._base_replay_store_provenance = copy.deepcopy(provenance)

    def create_quantization_layer_output_writer(
        self,
        *,
        layer_index: int,
        expected_batches: int,
        progress_stage: str | None,
        apply_moe_config: bool,
    ):
        if layer_index < 3 or apply_moe_config or progress_stage != "Forward replay":
            return None
        root = getattr(self, "_base_replay_store_root", None)
        provenance = getattr(self, "_base_replay_store_provenance", None)
        if not isinstance(root, str) or not isinstance(provenance, dict):
            return None
        from ...looper.input_cache import DiskBackedLayerOutputWriter

        return DiskBackedLayerOutputWriter(
            root,
            layer_index=layer_index,
            expected_batches=expected_batches,
            provenance={
                **copy.deepcopy(provenance),
                "layer_index": int(layer_index),
                "replay_contract": "glm5-next-target-mtp-post-quant-bf16-v1",
            },
            shard_batches=1,
            on_finalize=lambda _sequence: _prune_glm5_next_replay_frontier(
                root,
                before_layer=layer_index,
            ),
        )

    def zero_route_recovery_context(
        self,
        *,
        looper,
        processor,
        layer_module,
        subset: dict[str, Any],
        task_names: tuple[str, ...],
    ):
        from .deepseek_v4 import DeepSeekV4MTPQuantizationModel

        return DeepSeekV4MTPQuantizationModel.zero_route_recovery_context(
            self,
            looper=looper,
            processor=processor,
            layer_module=layer_module,
            subset=subset,
            task_names=task_names,
        )

    def zero_route_recovery_block_identity(
        self,
        layer_module: nn.Module,
    ) -> tuple[str, int, str]:
        """Resolve GLM's nested target-plus-MTP layer namespace.

        Layer 45 is checkpoint MTP, but its replay is a conventional decoder
        block rather than DeepSeek's proposal-token replay.  It therefore uses
        the ordinary ``base`` recovery mechanics with its exact layer-45 path.
        """

        layers = tuple(self.model.model.language_model.layers)
        if layer_module not in layers:
            raise RuntimeError(
                "GLM-5.3 route recovery requires one canonical decoder block"
            )
        block_index = layers.index(layer_module)
        return (
            "base",
            block_index,
            f"model.language_model.layers.{block_index}.mlp.experts.",
        )

    def update_layer_replay_kwargs_from_output(
        self,
        layer,
        layer_output,
        layer_input_kwargs,
        target_device,
    ):
        if isinstance(layer_output, tuple) and len(layer_output) >= 2:
            topk_indices = layer_output[1]
            if topk_indices is not None:
                layer_input_kwargs["prev_topk_indices"] = move_to(
                    topk_indices,
                    device=target_device,
                )

        layer_idx = getattr(layer, "layer_idx", None)
        if layer_idx == GLM5_NEXT_MTP_LAYER - 1:
            for name in ("input_ids", "attention_mask", "position_ids"):
                value = layer_input_kwargs.get(name)
                if isinstance(value, torch.Tensor) and value.ndim >= 2:
                    layer_input_kwargs[name] = value[:, 1:]
            layer_input_kwargs.pop("prev_topk_indices", None)
        return layer_input_kwargs


__all__ = [
    "GLM5_NEXT_MTP_LAYER",
    "GLM5_NEXT_ROUTED_EXPERT_PATTERN",
    "Glm5NextMTPDecoderLayer",
    "Glm5NextQModel",
    "_prune_glm5_next_replay_frontier",
]
