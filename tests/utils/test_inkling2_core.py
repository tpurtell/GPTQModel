# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Independent tiny CPU math/source oracles; no model download or GPU access."""
import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

# The utility intentionally works without importing the framework's eager CUDA
# dependency graph. Production runner uses the same standalone loading contract.
_PATH = Path(__file__).parents[2] / "gptqmodel/utils/inkling2_core.py"
_SPEC = importlib.util.spec_from_file_location("inkling2_core_oracle_test", _PATH)
core = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = core
_SPEC.loader.exec_module(core)


def oracle_encode(x):
    # Independent scalar scale selection in float64; ties in E4M3 conversion
    # are specified round-to-nearest-even by Torch's native format cast.
    groups = x.double().reshape(-1, 32)
    encoded, scales = [], []
    for row in groups:
        peak = float(row.abs().max())
        power = max(-127, math.ceil(math.log2(peak / 448))) if peak else -127
        factor = 2.0**power
        encoded.append((row / factor).clamp(-448, 448).float().to(torch.float8_e4m3fn))
        scales.append(power + 127)
    return torch.stack(encoded).reshape(x.shape), torch.tensor(scales, dtype=torch.uint8).reshape(*x.shape[:-1], -1)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_mxfp8_complete_bytes_and_boundary_neighbors(dtype):
    # Fixed max448 forces scale1; test every E4M3 midpoint and its adjacent
    # input representable values, both signs, saturation, signed zero.
    levels = torch.arange(0, 127, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    levels = levels[torch.isfinite(levels) & (levels <= 448)]
    mid = ((levels[:-1] + levels[1:]) * .5).to(dtype)
    lower = torch.nextafter(mid, torch.full_like(mid, -torch.inf))
    upper = torch.nextafter(mid, torch.full_like(mid, torch.inf))
    values = torch.cat((mid, lower, upper, -mid, -lower, -upper, torch.tensor([0., -0., 448., -448.], dtype=dtype)))
    groups = []
    for value in values:
        row = torch.zeros(32, dtype=dtype)
        row[0], row[1] = value, 448
        groups.append(row)
    x = torch.stack(groups)
    actual, scales = core.mxfp8_encode(x)
    expected, expected_scales = oracle_encode(x)
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    assert torch.equal(scales, expected_scales)
    assert torch.equal(scales, torch.full_like(scales, 127))
    assert torch.signbit(actual.float()[len(values) - 3, 0])


@pytest.mark.parametrize("power", [-120, -10, 0, 80])
def test_mxfp8_scale_boundary_neighbors(power):
    # A one-step input move above max448*2^power must advance scale to2^(power+1).
    edge = torch.tensor(448 * 2.0**power, dtype=torch.float32)
    peaks = torch.stack((torch.nextafter(edge, torch.tensor(-torch.inf)), edge, torch.nextafter(edge, torch.tensor(torch.inf))))
    x = torch.zeros(3, 32)
    x[:, 0] = peaks
    values, scales = core.mxfp8_encode(x)
    expected, exponent = oracle_encode(x)
    assert torch.equal(values.view(torch.uint8), expected.view(torch.uint8))
    assert torch.equal(scales, exponent)
    assert torch.equal(scales[:, 0], torch.tensor([power+127, power+127, power+128], dtype=torch.uint8))


def test_zero_tiny_scale_and_invalid_inputs():
    x = torch.zeros(2, 32)
    x[0, 1] = -0.
    x[1, 0] = torch.finfo(torch.float32).tiny / 32
    packed, scale = core.mxfp8_encode(x)
    assert not scale.any()
    assert packed.view(torch.uint8)[0, 1] == 128
    assert torch.equal(core.mxfp8_decode(packed, scale), packed.float() * 2.0**-127)
    for value in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match="Nonfinite"):
            core.mxfp8_encode(torch.full((1, 32), value))
    with pytest.raises(ValueError, match="scale shape"):
        core.mxfp8_decode(packed, torch.zeros((2, 2), dtype=torch.uint8))


def test_source_reader_uses_selected_expert_scales_and_mtp_namespace(tmp_path):
    native = torch.arange(2*6*64, dtype=torch.float32).reshape(2, 6, 64) % 16
    native = native.to(torch.float8_e4m3fn)
    scales = torch.tensor([[[126,127]]*6, [[128,129]]*6], dtype=torch.uint8)
    embedding = torch.randn(17, 4).bfloat16()
    save_file({"model.llm.layers.0.mlp.experts.w13_weight": native, "model.llm.layers.0.mlp.experts.w13_weight_scale": scales, "model.llm.embed.weight": embedding}, tmp_path / "model.safetensors")
    save_file({"fuse_weight": torch.randn(6,8).bfloat16()}, tmp_path / "mtp.safetensors")
    (tmp_path / "config.json").write_text("{}")
    keys = ["model.llm.layers.0.mlp.experts.w13_weight", "model.llm.layers.0.mlp.experts.w13_weight_scale", "model.llm.embed.weight"]
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {k: "model.safetensors" for k in keys}}))
    reader = core.SourceReader(tmp_path)
    weight = reader.expert(keys[0], 1)
    expected = native[1].float().reshape(6,2,32) * torch.tensor([2.,4.])[None,:,None]
    assert torch.equal(weight, expected.reshape(6,64))
    assert torch.equal(reader.expert(keys[0],1,projection="gate"), weight[::2])
    assert torch.equal(reader.expert(keys[0],1,projection="up"), weight[1::2])
    assert reader.tensor("mtp.fuse_weight").shape == (6,8)
    ids = torch.tensor([[1,16,1], [4,2,4]])
    assert torch.equal(reader.embedding("model.llm.embed.weight", ids), embedding[ids])
    # Reader owns metadata only, no tensor or expanded cache after repeated reads.
    reader.expert(keys[0], 0)
    assert not any(isinstance(value, torch.Tensor) for value in vars(reader).values())


def test_interleaved_softcap_raw_sigmoid_and_route_before_down_qdq():
    x = torch.tensor([[8.,40., -3.,-4.], [2.,-7., -9.,90.]], dtype=torch.bfloat16)
    gamma = torch.tensor([.17, 2.31])
    expected = []
    for row, g in zip(x.tolist(), gamma.tolist()):
        expected.append([4*math.tanh(row[i]/4) / (1+math.exp(-row[i])) * 25*math.tanh(row[i+1]/25)*g for i in (0,2)])
    actual = core.capped_swiglu(x,gamma)
    torch.testing.assert_close(actual, torch.tensor(expected).bfloat16(), rtol=0, atol=0)
    # Multiplying route weights after native activation rounding is distinguishable.
    x = torch.randn(4,64, generator=torch.Generator().manual_seed(3)).bfloat16()
    g = torch.tensor([.13,.37,1.75,3.13])
    before = core.mxfp8_qdq(core.capped_swiglu(x,g))
    after = core.mxfp8_qdq(core.capped_swiglu(x))*g[:,None]
    assert not torch.equal(before.float(), after.float())


def test_attention_residual_normalizes_values_and_adds_partial_in_fp32():
    generator=torch.Generator().manual_seed(8)
    blocks=[torch.randn(3,6,generator=generator).bfloat16() for _ in range(2)]
    partial=torch.randn(3,6,generator=generator).bfloat16()
    addend=torch.randn(3,6,generator=generator).bfloat16()
    queries=torch.randn(3,6,generator=generator)
    actual=core.block_attention_residual(blocks,partial,queries,addend=addend)
    sources=torch.stack([*blocks,partial.float()+addend.float()]).double()
    normalized=sources / (sources.square().mean(-1,keepdim=True)+1e-6).sqrt()
    expected=[]
    for query in queries.double():
        scores=(normalized*query/math.sqrt(6)).sum(-1)
        p=scores.softmax(0)
        expected.append((p[...,None]*normalized).sum(0))
    for a,b in zip(actual,expected):
        assert a.dtype==torch.float32
        torch.testing.assert_close(a.double(),b,rtol=1e-6,atol=1e-6)


@pytest.mark.parametrize("dynamic",[False,True])
def test_sconv_tap_direction_residual_and_normalization(dynamic):
    x=torch.arange(15,dtype=torch.float32).reshape(5,3)/10
    if dynamic:
        weight=torch.arange(12,dtype=torch.float32).reshape(4,3)/10
        delta=torch.arange(20,dtype=torch.float32).reshape(5,1,4)/30
    else:
        weight=torch.arange(12,dtype=torch.float32).reshape(3,1,4)/10
        delta=None
    actual=core.short_convolution(x,weight,dynamic_weights=delta)
    expected=torch.empty_like(x)
    for t in range(5):
        for h in range(3):
            acc=float(x[t,h])
            for lag in range(4):
                if t>=lag:
                    w=float(weight[lag,h]+delta[t,0,lag]) if dynamic else float(weight[h,0,3-lag])
                    acc+=float(x[t-lag,h])*w
            expected[t,h]=acc/math.sqrt(9 if dynamic else 5)
    torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-6)


def test_relative_attention_sink_gqa_causality_window_and_gate():
    generator=torch.Generator().manual_seed(4)
    q=torch.randn(5,4,2,generator=generator)
    k=torch.randn(5,2,2,generator=generator)
    v=torch.randn(5,2,2,generator=generator)
    r=torch.randn(5,4,2,generator=generator)
    bank=torch.randn(2,3,generator=generator)
    gate=torch.randn(5,8,generator=generator)
    sink=torch.randn(5,4,generator=generator)
    actual=core.relative_attention(q,k,v,r,bank,gate,sink,window=2)
    expected=torch.zeros_like(actual)
    for t in range(5):
        for h in range(4):
            indices=list(range(max(0,t-1),t+1))
            scores=[]
            for j in indices:
                scores.append(float(q[t,h].double()@k[j,h//2].double())/math.sqrt(2)+float(r[t,h].double()@bank[:,t-j].double()))
            p=torch.tensor([*scores,float(sink[t,h])],dtype=torch.float64).softmax(0)
            out=sum((p[i]*v[j,h//2].double() for i,j in enumerate(indices)))
            expected[t,h*2:h*2+2]=out*gate[t,h*2:h*2+2].double().sigmoid()
    torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-6)


class TinyReader:
    """Synthetic native checkpoint with one local and one global layer."""
    def __init__(self):
        self.config={"text_config":dict(hidden_size=32,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=8,d_rel=4,local_layer_ids=[0],sliding_window_size=4,rel_extent=4,n_layers_per_attn_res_block=6,num_experts_per_tok=2,route_scale=2.,rms_norm_eps=1e-6,vocab_size=19,final_logit_softcapping=30.),"mtp_config":dict(hidden_size=32,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=8,intermediate_size=64,local_layer_ids=[0],sliding_window_size=4,block_size=8,rms_norm_eps=1e-6,rope_theta=1000000.)}
        self.data={}
        gen=torch.Generator().manual_seed(90)
        def weight(name,shape,dtype=torch.bfloat16):
            self.data[name]=(torch.randn(shape,generator=gen)*.07).to(dtype)
        for name in ("embed","unembed"):
            weight(f"model.llm.{name}.weight",(19,32))
        for name in ("embed_norm","norm"):
            self.data[f"model.llm.{name}.weight"]=torch.ones(32,dtype=torch.bfloat16)
        for layer in range(2):
            p=f"model.llm.layers.{layer}."
            for name in (("attn_norm",) if layer==0 else ("q_input_norm","k_input_norm","v_input_norm"))+("mlp_norm",):
                self.data[p+name+".weight"]=torch.ones(32,dtype=torch.bfloat16)
            for name in ("attn_sconv","mlp_sconv"):
                weight(p+name+".weight",(32,1,4),torch.float32)
            for name,out in (("wq_du",32),("wk_dv",16),("wv_dv",16),("wo_ud",32),("wr_du",16),("gate_attn",32),("attn_sink_proj",4)):
                weight(p+"attn."+name+".weight",(out,32))
            self.data[p+"attn.q_l2_clip.scale"]=torch.ones(4,8)
            self.data[p+"attn.rel_l2_clip.scale"]=torch.ones(4,4)
            weight(p+"attn.rel_logits_proj.proj",(4,4))
            for kind in ("k","v"):
                if layer==0:
                    weight(p+f"attn.{kind}_sconv.weight",(16,1,4),torch.float32)
                else:
                    weight(p+f"attn.{kind}_sconv.static_w",(4,16),torch.float32)
                    weight(p+f"attn.{kind}_sconv.weight_proj",(8,32))
            weight(p+"mlp.gate.weight",(3,32))
            weight(p+"mlp.gate.bias",(3,),torch.float32)
            for role,out in (("w13",64),("w2",32)):
                weight(p+f"mlp.experts.{role}_weight",(3,out,32))
                payload,scales=core.mxfp8_encode(self.data[p+f"mlp.experts.{role}_weight"])
                self.data[p+f"mlp.experts.{role}_weight"]=payload
                self.data[p+f"mlp.experts.{role}_weight_scale"]=scales
            weight(f"model.llm.attn_res.mlp_queries.{layer}",(32,),torch.float32)
        weight("model.llm.attn_res.attn_queries.0",(3,32),torch.float32)
        weight("model.llm.attn_res.output_query",(32,),torch.float32)
        weight("mtp.fuse_weight",(3,3))
        weight("mtp.mask_embed",(32,))
        self.data["mtp.norm.weight"]=torch.ones(32,dtype=torch.bfloat16)
        self.data["mtp.markov_head.dflare_norm.weight"]=torch.ones(32,dtype=torch.bfloat16)
        for layer in range(2):
            p=f"mtp.layers.{layer}."
            for name in ("dflare_norm","input_layernorm","post_attention_layernorm"):
                self.data[p+name+".weight"]=torch.ones(32,dtype=torch.bfloat16)
            for name in ("q_norm","k_norm","k_norm_ctx"):
                self.data[p+"self_attn."+name+".weight"]=torch.ones(8,dtype=torch.bfloat16)
            for name,out in (("q_proj",32),("k_proj",16),("v_proj",16),("k_proj_ctx",16),("v_proj_ctx",16),("o_proj",32)):
                weight(p+"self_attn."+name+".weight",(out,32))
            for name,shape in (("gate_proj",(64,32)),("up_proj",(64,32)),("down_proj",(32,64))):
                weight(p+"mlp."+name+".weight",shape)
    def tensor(self,name,*,expert=None,device="cpu"):
        x=self.data[name]
        return (x if expert is None else x[expert]).to(device=device,copy=True)
    def embedding(self,name,ids,*,device="cpu"):
        return self.data[name][ids].to(device)
    def expert(self,name,index,*,projection=None,device="cpu"):
        result=core.mxfp8_decode(self.tensor(name,expert=index,device=device),self.tensor(name+"_scale",expert=index,device=device))
        return result


def test_phased_local_global_replay_captures_causal_projection_inputs():
    model=core.Inkling2Reference(TinyReader())
    ids=torch.tensor([1,2,3,4])
    state=model.start(ids)
    assert state.partial.dtype==torch.bfloat16
    for layer in range(2):
        captures={}
        def capture(name,x,weight):
            captures[name]=x.clone()
            return None
        attention,o_input=model.attention(state,layer,capture)
        key=f"model.llm.layers.{layer}.attn.wo_ud.weight"
        assert torch.equal(captures[key],o_input)
        pending,x=model.attention_finish(state,attention,layer)
        expert_ids,gamma,ranks=model.route(layer,x)
        assert torch.equal(ranks,torch.tensor([[0,1]]*4))
        torch.testing.assert_close(gamma.sum(-1),torch.full((4,),2.))
        expert_output=model.experts(layer,x,expert_ids,gamma,hook=capture)
        phased=model.mlp_finish(pending,expert_output)
        direct=model.layer(state,layer)
        assert torch.equal(phased.partial,direct.partial)
        assert all(torch.equal(a,b) for a,b in zip(phased.blocks,direct.blocks))
        assert phased.partial.dtype==torch.bfloat16
        # Replace K8 Q projection and observe a new O Hessian input; hooks
        # execute at the actual causal site rather than a source-only replay.
        def zero_q(name,x,weight):
            if name.endswith(".wq_du.weight"):
                return x.new_zeros((len(x),weight.shape[0]))
            return None
        _,changed=model.attention(state,layer,zero_q)
        assert not torch.equal(changed,o_input)
        state=phased
    hidden=model.finish(state)
    assert hidden.dtype==torch.bfloat16
    assert model.logits(hidden).shape==(4,19)
    with pytest.raises(ValueError,match="different layer"):
        model.mlp_finish(pending,expert_output,0)


def test_mtp_mixture_and_nine_projection_calibration_replay():
    model=core.Inkling2Reference(TinyReader())
    features=torch.randn(3,3,32,generator=torch.Generator().manual_seed(19)).bfloat16()
    alpha=model.reader.data["mtp.fuse_weight"].float().softmax(-1).bfloat16()
    expected=(features.double()*alpha[0].double()[None,:,None]).sum(1)
    expected=expected/(expected.square().mean(-1,keepdim=True)+1e-6).sqrt()
    torch.testing.assert_close(model.mtp_context(features,0),expected.bfloat16(),rtol=0,atol=0)
    anchor=torch.tensor(4)
    block=model.mtp_embed(anchor)
    assert torch.equal(block[0],model.embed(anchor))
    assert torch.equal(block[1:],model.reader.data["mtp.mask_embed"].expand(7,-1))
    captures={}
    def capture(name,x,weight):
        captures[name]=x.clone()
    output=model.mtp_layer(block,features,0,3,capture)
    assert len(captures)==9
    assert torch.equal(captures["mtp.layers.0.self_attn.k_proj_ctx.weight"],model.mtp_context(features,0))
    assert torch.equal(captures["mtp.layers.0.self_attn.v_proj_ctx.weight"],model.mtp_context(features,0))
    assert captures["mtp.layers.0.mlp.down_proj.weight"].shape==(8,64)
    assert model.mtp_finish(output).shape==(8,32)
    with pytest.raises(ValueError,match="preceding context"):
        model.mtp_layer(block,features,0,2)
    # No dequantized weight tensor is retained by the reference object.
    assert not any(isinstance(value,torch.Tensor) for value in vars(model).values())
