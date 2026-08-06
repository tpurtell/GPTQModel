# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from .deepseek_v3 import DeepSeekV3QModel


class DeepSeekV4QModel(DeepSeekV3QModel):
    dynamic_expert_index = "n_routed_experts"
    rotary_embedding = "model.rotary_emb"
    # Transformers intentionally ignores the integrated dSpark/MTP checkpoint
    # namespace. Preserve it byte-for-byte until a quantizer explicitly
    # replaces those tensors; silently dropping it produces an incomplete V4
    # artifact even when normal target-layer quantization succeeds.
    out_of_model_tensors = {"prefixes": ["mtp"]}
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



__all__ = ["DeepSeekV4QModel"]
