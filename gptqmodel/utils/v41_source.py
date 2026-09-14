"""Bounded, one-tensor-at-a-time V4.1 source access.

Decoded block loading is for numerical diagnostics and trellis source weights.
It does not reproduce the native FP8 activation quantization of source GEMMs.
"""

import json
from pathlib import Path
import re

import torch
from safetensors import safe_open

from .model_dequant import dequantize_deepseek_v4_fp4_expert


def source_layer_key(layer, runtime_key, namespace="layers"):
    if namespace not in ("layers", "mtp") or type(layer) is not int or layer < 0:
        raise ValueError("invalid V4.1 source namespace/layer")
    key = runtime_key
    for family, native in (("attn_hc", "hc_attn"), ("ffn_hc", "hc_ffn")):
        if key.startswith(family + "."):
            return f"{namespace}.{layer}.{native}_{key.split('.', 1)[1]}"
    replacements = (
        ("input_layernorm.", "attn_norm."), ("post_attention_layernorm.", "ffn_norm."),
        ("self_attn.", "attn."), ("mlp.", "ffn."),
        ("attn.sinks", "attn.attn_sink"),
        (".q_a_proj.", ".wq_a."), (".q_a_norm.", ".q_norm."),
        (".q_b_proj.", ".wq_b."), (".kv_proj.", ".wkv."),
        (".o_a_proj.", ".wo_a."), (".o_b_proj.", ".wo_b."),
        (".compressor.gate_proj.", ".compressor.wgate."),
        (".compressor.kv_norm.", ".compressor.norm."),
        (".indexer.k_proj.", ".indexer.wk."),
        (".gate.e_score_correction_bias_vl", ".gate.bias_vl"),
        (".gate.e_score_correction_bias", ".gate.bias"),
        (".gate_proj.", ".w1."), (".up_proj.", ".w3."), (".down_proj.", ".w2."),
    )
    for old, new in replacements:
        key = key.replace(old, new)
    return f"{namespace}.{layer}.{key}"


class V41Source:
    def __init__(self, snapshot):
        self.snapshot = Path(snapshot).resolve()
        self.config = json.loads((self.snapshot / "config.json").read_text())
        if self.config.get("model_type") != "deepseek_v41":
            raise ValueError("expected DeepSeek V4.1 source config")
        self.weight_map = json.loads((self.snapshot / "model.safetensors.index.json").read_text())["weight_map"]
        for shard in set(self.weight_map.values()):
            if Path(shard).name != shard or not (self.snapshot / shard).is_file():
                raise ValueError(f"invalid or missing V4.1 shard: {shard}")

    def tensor(self, name, device="cpu"):
        if ".engram.embed." in name:
            raise ValueError("PLE payloads must use MappedPLETable, never full-tensor loading")
        with safe_open(self.snapshot / self.weight_map[name], framework="pt", device="cpu") as shard:
            return shard.get_tensor(name).to(device=device, copy=True)

    def decoded(self, name, device="cpu", dtype=torch.bfloat16):
        value = self.tensor(name, device)
        scale_name = name.removesuffix(".weight") + ".scale"
        if not name.endswith(".weight") or scale_name not in self.weight_map:
            return value
        scale = self.tensor(scale_name, device)
        if value.dtype == torch.int8:
            if not re.fullmatch(r"(?:layers|mtp)\.\d+\.ffn\.experts\.\d+\.w[123]\.weight", name):
                raise ValueError(f"unexpected packed INT8 source: {name}")
            return dequantize_deepseek_v4_fp4_expert(value, scale, target_dtype=dtype)
        if value.dtype != torch.float8_e4m3fn or value.ndim != 2:
            raise ValueError(f"unsupported scaled source tensor: {name}")
        rows, cols = value.shape
        if tuple(scale.shape) != ((rows + 31) // 32, (cols + 31) // 32):
            raise ValueError(f"invalid block-32 FP8 scales: {name}")
        result = torch.empty(value.shape, dtype=dtype, device=device)
        # Keep FP32 work bounded to 32 rows instead of expanding the whole matrix.
        for start in range(0, rows, 32):
            expanded = scale[start // 32].float().repeat_interleave(32)[:cols]
            result[start:start + 32] = value[start:start + 32].float() * expanded
        return result

    @torch.no_grad()
    def load_decoded_block(self, layer_index, device="cuda:0"):
        from transformers import DeepseekV41Config
        from transformers.models.deepseek_v41.modeling_deepseek_v41 import DeepseekV41DecoderLayer
        from ..models.definitions.deepseek_v41 import DeepSeekV41Experts

        config = DeepseekV41Config.from_dict(self.config).get_text_config()
        if type(layer_index) is not int or not 0 <= layer_index < config.num_hidden_layers:
            raise ValueError("main block index outside source config")
        config._experts_implementation = "eager"
        config._attn_implementation = "eager"
        with torch.device("meta"):
            block = DeepseekV41DecoderLayer(config, layer_index)
            block.mlp.experts = DeepSeekV41Experts.from_fused(block.mlp.experts)
        expected = block.state_dict()
        for key, template in expected.items():
            source_key = source_layer_key(layer_index, key)
            value = self.decoded(source_key, device)
            if value.shape != template.shape:
                raise ValueError(f"V4.1 shape mismatch: {source_key}: {value.shape} != {template.shape}")
            parent_key, _, leaf = key.rpartition(".")
            parent = block.get_submodule(parent_key) if parent_key else block
            if leaf in parent._parameters:
                setattr(parent, leaf, torch.nn.Parameter(value, requires_grad=False))
            else:
                parent.register_buffer(leaf, value)
        # Rotary embedding constants are nonpersistent buffers initialized on meta.
        # Recreate these tiny modules on the execution device from their config.
        from transformers.models.deepseek_v41.modeling_deepseek_v41 import DeepseekV41RotaryEmbedding
        for name, module in list(block.named_modules()):
            if isinstance(module, DeepseekV41RotaryEmbedding):
                parent_key, _, leaf = name.rpartition(".")
                parent = block.get_submodule(parent_key) if parent_key else block
                with torch.device(device):
                    setattr(parent, leaf, DeepseekV41RotaryEmbedding(config))
        if any(tensor.is_meta for tensor in list(block.parameters()) + list(block.buffers())):
            raise ValueError("V4.1 block retains uninitialized meta tensors")
        return block.eval()
