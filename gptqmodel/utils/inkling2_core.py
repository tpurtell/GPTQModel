# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Bounded, layer-streamed Inkling 2 calibration reference.

Math authority: sgl-project/sglang, inkling-2-small-support, revision below.
This module deliberately has no model registry, serving, tokenizer or device
selection side effects. Native quantized weights are decoded only for their
current consumer. The caller owns scheduling, K8 reconstruction callbacks and
activation/Hessian storage; this is not an optimized inference implementation.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from safetensors import safe_open

REFERENCE_REVISION = "ec5e7ff575beb33e54509fe01934d6a4483b5830"
REFERENCE_REPOSITORY = "https://github.com/sgl-project/sglang"
# Return None for the source projection, or a replacement output. Expert names
# carry their physical expert index as `...w13_weight[17]` / `...w2_weight[17]`.
LinearOverride = Callable[[str, torch.Tensor, torch.Tensor], torch.Tensor | None]


class SourceReader:
    """Read checkpoint-owned tensors without retaining weights between calls.

    Only the requested first-axis expert slice is paged for quantized experts.
    The separately indexed MTP shard is exposed with a logical ``mtp.`` prefix.
    """

    def __init__(self, snapshot: str | Path):
        self.snapshot = Path(snapshot)
        self.config = json.loads((self.snapshot / "config.json").read_text())
        index = json.loads((self.snapshot / "model.safetensors.index.json").read_text())
        self.weight_map = dict(index["weight_map"])
        self._raw_names = {key: key for key in self.weight_map}
        mtp = self.snapshot / "mtp.safetensors"
        if mtp.exists():
            with safe_open(mtp, framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    logical = key if key.startswith("mtp.") else "mtp." + key
                    if logical in self.weight_map and self.weight_map[logical] != mtp.name:
                        raise ValueError(f"Duplicate MTP tensor: {logical}")
                    self.weight_map[logical] = mtp.name
                    self._raw_names[logical] = key

    def tensor(self, name: str, *, expert: int | None = None, rows: slice | None = None, device="cpu"):
        path = self.snapshot / self.weight_map[name]
        with safe_open(path, framework="pt", device="cpu") as handle:
            key = self._raw_names[name]
            if expert is not None and rows is not None:
                raise ValueError("Select expert or rows, not both")
            if rows is not None:
                value = handle.get_slice(key)[rows]
            elif expert is None:
                value = handle.get_tensor(key)
            else:
                view = handle.get_slice(key)
                shape = view.get_shape()
                if not shape or not 0 <= expert < shape[0]:
                    raise IndexError(f"{name}: expert {expert} outside {shape}")
                value = view[expert]
            # Own only this selected tensor; no mapping or expanded cache lives
            # on this object. CPU clone also makes caller mutation safe.
            return value.to(device=device, copy=True)

    def embedding(self, name: str, ids: torch.Tensor, *, device="cpu"):
        """Gather only selected vocabulary rows, without copying the whole table."""
        flat = ids.detach().cpu().reshape(-1)
        unique, inverse = torch.unique(flat, sorted=True, return_inverse=True)
        with safe_open(self.snapshot / self.weight_map[name], framework="pt", device="cpu") as handle:
            view = handle.get_slice(self._raw_names[name])
            count, width = view.get_shape()
            if len(unique) and (unique[0] < 0 or unique[-1] >= count):
                raise IndexError("Embedding token outside checkpoint vocabulary")
            if not len(unique):
                return torch.empty((*ids.shape, width), dtype=torch.bfloat16, device=device)
            rows = torch.stack([view[int(token)] for token in unique])
            selected = rows[inverse].reshape(*ids.shape, width)
            return selected.to(device=device, copy=True)

    def expert(self, name: str, index: int, *, projection: str | None = None, device="cpu"):
        """Decode one native expert, optionally expose logical gate/up/down.

        ``name`` is a physical w13_weight or w2_weight checkpoint key. w13 is
        interleaved in its output dimension; scales follow the same row order.
        """
        payload = self.tensor(name, expert=index, device=device)
        scales = self.tensor(name + "_scale", expert=index, device=device)
        weight = mxfp8_decode(payload, scales)
        if projection in ("gate", "up"):
            if not name.endswith("w13_weight"):
                raise ValueError("gate/up views require interleaved w13_weight")
            return weight[0 if projection == "gate" else 1 :: 2].contiguous()
        if projection not in (None, "down") or (projection == "down" and not name.endswith("w2_weight")):
            raise ValueError(f"Invalid expert projection: {projection}")
        return weight


def mxfp8_decode(payload: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Per-row E4M3 x UE8M0, one U8 exponent for each 32 input values."""
    if payload.dtype != torch.float8_e4m3fn or scales.dtype != torch.uint8:
        raise TypeError("MXFP8 requires E4M3 payload and U8 UE8M0 scales")
    if payload.shape[-1] % 32 or scales.shape != (*payload.shape[:-1], payload.shape[-1] // 32):
        raise ValueError("MXFP8 scale shape must be (..., output_rows, input_width/32)")
    exponent = scales.to(torch.int32)
    if (exponent == 255).any():
        raise ValueError("NaN UE8M0 scale in source checkpoint")
    factor = torch.ldexp(torch.ones_like(exponent, dtype=torch.float32), exponent - 127)
    return (payload.float().unflatten(-1, (-1, 32)) * factor[..., None]).flatten(-2)


def mxfp8_encode(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Native nearest E4M3 quantization; power-of-two block scale rounds up.

    Implements ops/inkling2/moe.py::_mxfp8_block. UE8M0 byte zero represents
    2**-127. Signed zero payload bits are preserved. Nonfinite calibration
    activations fail closed rather than yielding reusable numerical state.
    """
    if not x.is_floating_point() or x.shape[-1] % 32:
        raise ValueError("MXFP8 activations need a floating width divisible by 32")
    groups = x.float().unflatten(-1, (-1, 32))
    if not torch.isfinite(groups).all():
        raise ValueError("Nonfinite MXFP8 calibration activation")
    raw = (groups.abs().amax(-1) * (1.0 / 448.0)).contiguous()
    exponent = ((raw.view(torch.int32) + 0x007FFFFF) & 0x7F800000) >> 23
    exponent = torch.where(raw <= 2.0**-127, 0, exponent.clamp(max=254))
    scale = torch.ldexp(torch.ones_like(raw), exponent - 127)
    inverse = torch.where(exponent == 0, 2.0**127, scale.reciprocal())
    values = (groups * inverse[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return values.flatten(-2), exponent.to(torch.uint8)


def mxfp8_qdq(x: torch.Tensor) -> torch.Tensor:
    return mxfp8_decode(*mxfp8_encode(x)).to(x.dtype)


def rms_norm(x, weight=None, eps=1e-6, *, output_dtype=None):
    value = x.float()
    value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps)
    if weight is not None:
        value = value * weight.float()
    return value.to(output_dtype or x.dtype)


def l2_clip(x, threshold, scale=None, eps=1e-6, output_scale=None):
    value = x.float()
    if scale is not None:
        value = value * scale.float()
    value = value * (threshold / (value.square().sum(-1, keepdim=True).sqrt() + eps)).clamp(max=1)
    if output_scale is not None:
        value = value.to(x.dtype).float() * output_scale
    return value.to(x.dtype)


def block_attention_residual(blocks, partial, queries, *, addend=None, eps=1e-6):
    """FP32 key/value normalization, query scoring and weighted value sum."""
    values = [value.float() for value in blocks]
    tail = partial.float()
    if addend is not None:
        tail = tail + addend.float()
    values.append(tail)
    stacked = torch.stack(values)
    normalized = stacked * torch.rsqrt(stacked.square().mean(-1, keepdim=True) + eps)
    q = queries.float().reshape(-1, partial.shape[-1]) * partial.shape[-1]**-0.5
    scores = torch.einsum("std,qd->qst", normalized, q)
    probabilities = scores.softmax(1)
    return tuple(torch.einsum("qst,std->qtd", probabilities, normalized).unbind(0))


def short_convolution(x, weight, *, dynamic_weights=None, output_dtype=None):
    """Single uninterrupted sequence, causal depthwise residual convolution.

    Static checkpoint taps run oldest->current. Dynamic taps run current->oldest
    and add per-token/per-KV-head weights to FP32 static weights.
    """
    if x.ndim != 2:
        raise ValueError("short_convolution expects one [tokens,hidden] sequence")
    dynamic = dynamic_weights is not None
    if dynamic:
        taps, hidden = weight.shape
        heads = dynamic_weights.shape[1]
        if dynamic_weights.shape != (x.shape[0], heads, taps) or hidden % heads:
            raise ValueError("Dynamic convolution shape mismatch")
    else:
        weight = weight.squeeze(1) if weight.ndim == 3 else weight
        hidden, taps = weight.shape
    value = x.float()
    result = torch.zeros_like(value)
    for tap in range(taps):
        lag = tap if dynamic else taps - 1 - tap
        source = torch.zeros_like(value)
        if lag < len(value):
            source[lag:] = value[:len(value) - lag] if lag else value
        if dynamic:
            coefficient = weight[tap].float()[None] + dynamic_weights[:, :, tap].float().repeat_interleave(hidden // heads, -1)
        else:
            # Official static convolution rounds F32 checkpoint weights to the
            # input activation dtype before accumulation. Dynamic static_w does not.
            coefficient = weight[:, tap].to(x.dtype).float()
        result = result + source * coefficient
    result = (result + value) * ((taps * (2 if dynamic else 1) + 1)**-0.5)
    return result.to(output_dtype or x.dtype)


def capped_swiglu(interleaved, route_gamma=None, gate_cap=4.0, up_cap=25.0):
    gate, up = interleaved[..., 0::2].float(), interleaved[..., 1::2].float()
    sigmoid = gate.sigmoid()  # raw gate, before the softcap
    gate = gate_cap * (gate / gate_cap).tanh() if gate_cap else gate
    up = up_cap * (up / up_cap).tanh() if up_cap else up
    result = (gate * sigmoid) * up
    if route_gamma is not None:
        result = result * route_gamma[..., None].float()
    return result.to(interleaved.dtype)


def relative_attention(q, k, v, r, relative_bank, gate, sinks, *, window=None):
    """GQA + bounded relative bias + zero-valued sink; output BF16 gate input.

    All tensors have token-major [T,heads,width] shape except flat gate/sinks.
    This dense attention oracle is for bounded calibration prompt lengths.
    """
    tokens, heads, dim = q.shape
    repeats = heads // k.shape[1]
    key = k.repeat_interleave(repeats, 1).transpose(0, 1).float()
    value = v.repeat_interleave(repeats, 1).transpose(0, 1).float()
    query = q.transpose(0, 1).float()
    scores = (query @ key.transpose(-1, -2)) * dim**-0.5
    # Projection rounds to BF16 before inserting bias into FP32 scores.
    relative = (r @ relative_bank.to(r.dtype)).transpose(0, 1).float()
    distance = torch.arange(tokens, device=q.device)[:, None] - torch.arange(tokens, device=q.device)[None, :]
    extent = relative.shape[-1]
    bias = relative.gather(-1, distance.clamp(0, extent - 1)[None].expand(heads, -1, -1))
    scores = scores + torch.where(((distance >= 0) & (distance < extent))[None], bias, 0)
    permitted = distance >= 0
    if window is not None:
        permitted &= distance < window
    scores = scores.masked_fill(~permitted[None], -torch.inf)
    sink = sinks.reshape(tokens, heads).T[..., None].float()
    probability = torch.cat((scores, sink), -1).softmax(-1)[..., :-1]
    # FA returns FP32 PV, fuses gate/sink and only then rounds to BF16.
    result = (probability @ value).transpose(0, 1).reshape(tokens, -1)
    return (result * gate.float().sigmoid()).to(q.dtype)


@dataclass
class AttnResState:
    partial: torch.Tensor
    blocks: list[torch.Tensor] = field(default_factory=list)


@dataclass
class PendingLayer:
    state: AttnResState
    attention: torch.Tensor
    layer_id: int


class Inkling2Reference:
    """One sequence / one layer at a time, with caller-owned projection replay."""

    def __init__(self, reader: SourceReader, linear: LinearOverride | None = None, *, device="cpu"):
        self.reader, self.linear_override, self.device = reader, linear, device
        self.config = reader.config.get("text_config", reader.config)
        self.mtp_config = reader.config.get("mtp_config")
        self.hidden = self.config["hidden_size"]
        self.heads = self.config["num_attention_heads"]
        self.kv_heads = self.config["num_key_value_heads"]
        self.dim = self.config["head_dim"]
        self.eps = self.config.get("rms_norm_eps", 1e-6)
        self.local_ids = set(self.config["local_layer_ids"])
        self.block_size = self.config.get("n_layers_per_attn_res_block", 6)

    def weight(self, name):
        return self.reader.tensor(name, device=self.device)

    def project(self, name, x, *, weight=None, hook=None, output_dtype=None):
        weight = self.weight(name) if weight is None else weight
        callback = hook if hook is not None else self.linear_override
        replacement = callback(name, x, weight) if callback is not None else None
        if replacement is not None:
            return replacement
        # FP32 output requests (router/head) must not pass through BF16 GEMM
        # output before the cast. Inputs/weights already have checkpoint precision.
        dtype = output_dtype or x.dtype
        if dtype == torch.float32:
            return F.linear(x.float(), weight.float())
        return F.linear(x.to(weight.dtype), weight).to(dtype)

    def norm(self, x, name, *, output_dtype=torch.bfloat16):
        return rms_norm(x, self.weight(name), self.eps, output_dtype=output_dtype)

    def embed(self, ids):
        # Read embedding once per call, release after gather; caller retains only
        # selected normalized activation rows. No full-model resident module.
        result = self.reader.embedding("model.llm.embed.weight", ids, device=self.device)
        return self.norm(result, "model.llm.embed_norm.weight")

    def start(self, ids):
        return AttnResState(self.embed(ids))

    def attention_inputs(self, state, layer_id):
        prefix = f"model.llm.layers.{layer_id}"
        if layer_id == 0:
            streams = (state.partial,) * (1 if layer_id in self.local_ids else 3)
        else:
            query = self.weight(f"model.llm.attn_res.attn_queries.{layer_id - 1}")
            streams = block_attention_residual(state.blocks, state.partial, query)
        if layer_id in self.local_ids:
            q = k = v = self.norm(streams[0], prefix + ".attn_norm.weight")
        else:
            q, k, v = (self.norm(stream, prefix + f".{name}_input_norm.weight") for stream, name in zip(streams, ("q", "k", "v")))
        return {prefix + ".attn." + name + ".weight": source for name, source in (("wq_du", q), ("wk_dv", k), ("wv_dv", v), ("wr_du", q), ("gate_attn", q), ("attn_sink_proj", q))}

    def attention(self, state, layer_id, linear_override=None):
        prefix = f"model.llm.layers.{layer_id}.attn."
        inputs = self.attention_inputs(state, layer_id)
        values = {name: self.project(prefix + name + ".weight", inputs[prefix + name + ".weight"], hook=linear_override) for name in ("wq_du", "wk_dv", "wv_dv", "wr_du", "gate_attn", "attn_sink_proj")}
        q, k, v, r = (values[name] for name in ("wq_du", "wk_dv", "wv_dv", "wr_du"))
        local = layer_id in self.local_ids
        for kind in ("k", "v"):
            value = k if kind == "k" else v
            conv = prefix + kind + "_sconv."
            if local:
                value = short_convolution(value, self.weight(conv + "weight"))
            else:
                source = inputs[prefix + ("wk_dv" if kind == "k" else "wv_dv") + ".weight"]
                dynamic = self.project(conv + "weight_proj", source).reshape(len(source), self.kv_heads, -1)
                value = short_convolution(value, self.weight(conv + "static_w"), dynamic_weights=dynamic)
            if kind == "k":
                k = value
            else:
                v = value
        q = q.reshape(-1, self.heads, self.dim)
        k = k.reshape(-1, self.kv_heads, self.dim)
        v = v.reshape(-1, self.kv_heads, self.dim)
        r = r.reshape(-1, self.heads, self.config["d_rel"])
        tau = None
        floor = self.config.get("log_scaling_n_floor")
        if not local and floor is not None:
            positions = torch.arange(len(q), device=q.device, dtype=torch.float32)
            tau = (1 + self.config.get("log_scaling_alpha", 0.1) * ((positions + 1) / floor).clamp(min=1).log())[:, None, None]
        q = l2_clip(q, self.config.get("qk_l2_clip_threshold", 22.627416997969522), self.weight(prefix + "q_l2_clip.scale"), output_scale=tau)
        k = l2_clip(k, self.config.get("qk_l2_clip_threshold", 22.627416997969522))
        r = l2_clip(r, self.config.get("rel_l2_clip_threshold", 8.0), self.weight(prefix + "rel_l2_clip.scale"), output_scale=tau)
        v = rms_norm(v, eps=self.eps)
        o_input = relative_attention(q, k, v, r, self.weight(prefix + "rel_logits_proj.proj"), values["gate_attn"], values["attn_sink_proj"], window=self.config["sliding_window_size"] if local else None)
        attention = self.project(prefix + "wo_ud.weight", o_input, hook=linear_override)
        return attention, o_input

    def attention_finish(self, state, attention, layer_id):
        prefix = f"model.llm.layers.{layer_id}"
        attention = short_convolution(attention, self.weight(prefix + ".attn_sconv.weight"))
        query = self.weight(f"model.llm.attn_res.mlp_queries.{layer_id}")
        if layer_id % self.block_size == 0:
            streams = block_attention_residual([*state.blocks, state.partial], attention, query)
        else:
            streams = block_attention_residual(state.blocks, state.partial, query, addend=attention)
        mlp_input = self.norm(streams[0], prefix + ".mlp_norm.weight")
        return PendingLayer(state, attention, layer_id), mlp_input

    def route(self, layer_id, mlp_input):
        prefix = f"model.llm.layers.{layer_id}.mlp.gate."
        logits = self.project(prefix + "weight", mlp_input, output_dtype=torch.float32)
        selection = logits.sigmoid() + self.weight(prefix + "bias").float()
        # Official tie break favors the smaller expert index, then sorts selected
        # experts in descending selection-score order.
        ids = torch.argsort(selection, dim=-1, descending=True, stable=True)[..., :self.config["num_experts_per_tok"]]
        chosen = logits.gather(-1, ids)
        log_prob = F.logsigmoid(chosen)
        if ids.shape[-1] == 8:
            # Same fixed reduction order as _logsigmoid_renorm8; softmax's
            # backend-selected reduction can perturb these routing weights.
            maximum = log_prob.max(-1, keepdim=True).values
            shifted = (log_prob - maximum).exp()
            total = ((shifted[..., 0] + shifted[..., 2]) + (shifted[..., 1] + shifted[..., 3])) + ((shifted[..., 4] + shifted[..., 6]) + (shifted[..., 5] + shifted[..., 7]))
            gamma = (log_prob - (maximum + total.log()[..., None])).exp()
        else:
            gamma = log_prob.softmax(-1)
        gamma = gamma * self.config.get("route_scale", 8.0)
        ranks = torch.arange(ids.shape[-1], device=ids.device).expand_as(ids)
        return ids, gamma, ranks

    def expert_project(self, layer_id, expert, role, x, *, hook=None):
        if role not in ("w13", "w2", "gate", "up", "down"):
            raise ValueError(f"Unknown expert projection role: {role}")
        prefix = f"model.llm.layers.{layer_id}.mlp.experts."
        name = prefix + ("w13_weight" if role in ("w13", "gate", "up") else "w2_weight")
        weight = self.reader.expert(name, expert, projection=role if role in ("gate", "up", "down") else None, device=self.device)
        return self.project(f"{name}[{expert}]", x, weight=weight.to(x.dtype), hook=hook)

    def experts(self, layer_id, mlp_input, ids, gamma, *, hook=None):
        # Accumulate in routing rank order, as post_reorder does; route gamma is
        # inside the SwiGLU BEFORE down-input MXFP8 rounding, never applied here.
        rows = mlp_input.shape[0]
        partials = mlp_input.new_empty((rows, ids.shape[-1], self.hidden))
        for expert in ids.unique().tolist():
            token, rank = torch.where(ids == expert)
            x = mxfp8_qdq(mlp_input[token])
            gate_up = self.expert_project(layer_id, expert, "w13", x, hook=hook)
            activated = capped_swiglu(gate_up, gamma[token, rank], self.config.get("mlp_gate_softcap", 4.0), self.config.get("mlp_up_softcap", 25.0))
            partials[token, rank] = self.expert_project(layer_id, expert, "w2", mxfp8_qdq(activated), hook=hook)
        result = torch.zeros((rows, self.hidden), device=mlp_input.device, dtype=torch.float32)
        for rank in range(ids.shape[-1]):
            result = result + partials[:, rank].float()
        return result.to(mlp_input.dtype)

    def mlp_finish(self, pending, expert_output, layer_id=None):
        layer_id = pending.layer_id if layer_id is None else layer_id
        if layer_id != pending.layer_id:
            raise ValueError("Pending attention belongs to a different layer")
        mlp = short_convolution(expert_output, self.weight(f"model.llm.layers.{layer_id}.mlp_sconv.weight"))
        state, attention = pending.state, pending.attention
        if layer_id % self.block_size == 0:
            return AttnResState((attention.float() + mlp.float()).to(state.partial.dtype), [*state.blocks, state.partial])
        return AttnResState(((state.partial.float() + attention.float()) + mlp.float()).to(state.partial.dtype), list(state.blocks))

    def layer(self, state, layer_id, hook=None):
        attention, _ = self.attention(state, layer_id, hook)
        pending, x = self.attention_finish(state, attention, layer_id)
        ids, gamma, _ = self.route(layer_id, x)
        return self.mlp_finish(pending, self.experts(layer_id, x, ids, gamma, hook=hook))

    def finish(self, state):
        output = block_attention_residual(state.blocks, state.partial, self.weight("model.llm.attn_res.output_query"))[0]
        return self.norm(output, "model.llm.norm.weight")

    def logits(self, hidden):
        logits = self.project("model.llm.unembed.weight", hidden, output_dtype=torch.float32)
        cap = self.config.get("final_logit_softcapping", 30.0)
        if cap:
            logits = (logits / cap).tanh() * cap
        logits[..., self.config["vocab_size"]:] = -torch.inf
        return logits

    def mtp_context(self, features, layer_id):
        """features [context_tokens,8,hidden], captured from the K8 target pass."""
        if features.ndim != 3:
            raise ValueError("MTP features must be [tokens,target_taps,hidden]")
        alpha = self.weight("mtp.fuse_weight").float().softmax(-1).to(features.dtype)
        mixed = torch.zeros_like(features[:, 0], dtype=torch.float32)
        for tap in range(features.shape[1]):
            mixed = mixed + features[:, tap].float() * alpha[layer_id, tap].float()
        norm = "mtp.markov_head.dflare_norm.weight" if layer_id == -1 or layer_id == len(alpha) - 1 else f"mtp.layers.{layer_id}.dflare_norm.weight"
        return self.norm(mixed, norm)

    def mtp_embed(self, anchor_ids):
        cfg = self.mtp_config
        if cfg is None:
            raise ValueError("Checkpoint has no MTP config")
        anchor = self.embed(anchor_ids)
        blocks = self.weight("mtp.mask_embed").expand(*anchor.shape[:-1], cfg["block_size"], -1).clone()
        blocks[..., 0, :] = anchor
        return blocks

    def mtp_layer(self, block, features, layer_id, anchor_position, hook=None):
        """One bidirectional block; context rows strictly precede its anchor.

        Context positions are [0,anchor_position), query positions are
        [anchor_position,anchor_position+block_size). Local context windows are
        anchored at the block START, not separately at each future token.
        """
        cfg = self.mtp_config
        if block.ndim != 2 or block.shape[0] != cfg["block_size"] or len(features) != anchor_position:
            raise ValueError("MTP replay requires one block and all preceding context rows")
        prefix = f"mtp.layers.{layer_id}."
        attn = prefix + "self_attn."
        x = self.norm(block, prefix + "input_layernorm.weight")
        context = self.mtp_context(features, layer_id)
        q = self.project(attn + "q_proj.weight", x, hook=hook).reshape(-1, cfg["num_attention_heads"], cfg["head_dim"])
        k = self.project(attn + "k_proj.weight", x, hook=hook).reshape(-1, cfg["num_key_value_heads"], cfg["head_dim"])
        v = self.project(attn + "v_proj.weight", x, hook=hook).reshape_as(k)
        ck = self.project(attn + "k_proj_ctx.weight", context, hook=hook).reshape(-1, cfg["num_key_value_heads"], cfg["head_dim"])
        cv = self.project(attn + "v_proj_ctx.weight", context, hook=hook).reshape_as(ck)
        q = rms_norm(q, self.weight(attn + "q_norm.weight"), cfg["rms_norm_eps"])
        k = rms_norm(k, self.weight(attn + "k_norm.weight"), cfg["rms_norm_eps"])
        ck = rms_norm(ck, self.weight(attn + "k_norm_ctx.weight"), cfg["rms_norm_eps"])
        positions = torch.arange(anchor_position, anchor_position + len(block), device=x.device)
        q, k = rope(q, positions, cfg["rope_theta"]), rope(k, positions, cfg["rope_theta"])
        ck = rope(ck, torch.arange(anchor_position, device=x.device), cfg["rope_theta"])
        keys, values = torch.cat((ck, k)), torch.cat((cv, v))
        repeat = cfg["num_attention_heads"] // cfg["num_key_value_heads"]
        key = keys.repeat_interleave(repeat, 1).transpose(0, 1).float()
        value = values.repeat_interleave(repeat, 1).transpose(0, 1).float()
        scores = (q.transpose(0, 1).float() @ key.transpose(-1, -2)) * cfg["head_dim"]**-0.5
        if layer_id in cfg["local_layer_ids"]:
            start = max(0, anchor_position - cfg["sliding_window_size"] + 1)
            scores[..., :start] = -torch.inf
        o_input = (scores.softmax(-1) @ value).transpose(0, 1).reshape(len(block), -1).to(block.dtype)
        attention = self.project(attn + "o_proj.weight", o_input, hook=hook)
        residual = (block.float() + attention.float()).to(block.dtype)
        x = self.norm(residual, prefix + "post_attention_layernorm.weight")
        gate = self.project(prefix + "mlp.gate_proj.weight", x, hook=hook)
        up = self.project(prefix + "mlp.up_proj.weight", x, hook=hook)
        activated = (F.silu(gate.float()) * up.float()).to(gate.dtype)
        down = self.project(prefix + "mlp.down_proj.weight", activated, hook=hook)
        return (residual.float() + down.float()).to(block.dtype)

    def mtp_finish(self, block):
        return self.norm(block, "mtp.norm.weight")


def rope(x, positions, theta):
    """Drafter's full-width NeoX rotary convention, not target attention."""
    width = x.shape[-1]
    inverse = theta**(-torch.arange(0, width, 2, device=x.device, dtype=torch.float32) / width)
    angle = positions.float()[:, None] * inverse[None]
    cos, sin = angle.cos().to(x.dtype)[:, None], angle.sin().to(x.dtype)[:, None]
    first, second = x[..., :width // 2], x[..., width // 2:]
    # Fused rotary evaluates in FP32 from stored BF16 cos/sin then rounds once.
    return torch.cat((first.float() * cos.float() - second.float() * sin.float(), second.float() * cos.float() + first.float() * sin.float()), -1).to(x.dtype)
