"""Packed V4.1 source linear execution using pinned reference kernels.

Allocate explicit BF16 outputs instead of relying on process-global default
dtype. The kernel module is supplied by the caller from an authenticated source.
"""

import torch
from torch import nn


class V41NativeLinear(nn.Module):
    def __init__(self, weight, scale, kernels):
        super().__init__()
        if weight.ndim != 2 or weight.dtype not in (torch.int8, torch.float8_e4m3fn):
            raise ValueError("expected packed FP4 or FP8 V4.1 matrix")
        if scale.dtype != torch.float8_e8m0fnu:
            raise ValueError("V4.1 source linear scales must be E8M0")
        self.packed_fp4 = weight.dtype == torch.int8
        self.out_features = weight.shape[0]
        self.in_features = weight.shape[1] * (2 if self.packed_fp4 else 1)
        expected = ((self.out_features, self.in_features // 32) if self.packed_fp4
                    else ((self.out_features + 31) // 32, self.in_features // 32))
        if self.in_features % 32 or tuple(scale.shape) != expected:
            raise ValueError("invalid V4.1 matrix/scale geometry")
        if self.packed_fp4:
            weight = weight.view(torch.float4_e2m1fn_x2)
        self.register_buffer("weight", weight)
        self.register_buffer("scale", scale)
        self.kernels = kernels

    @torch.no_grad()
    def forward(self, x):
        if x.dtype != torch.bfloat16 or x.shape[-1] != self.in_features:
            raise ValueError("native V4.1 GEMM requires BF16 inputs of the declared width")
        rows = x.numel() // self.in_features
        result = torch.empty((*x.shape[:-1], self.out_features), device=x.device, dtype=torch.bfloat16)
        if not rows:
            return result
        a, a_scale = self.kernels.act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu)
        if self.packed_fp4:
            kernel = self.kernels.fp4_gemm_kernel(self.out_features, self.in_features,
                                                 act_block_size=32, scale_dtype=self.kernels.FE8M0)
        else:
            kernel = self.kernels.fp8_gemm_kernel(self.out_features, self.in_features,
                                                 block_size=32, scale_dtype=self.kernels.FE8M0)
        kernel(a.reshape(rows, self.in_features), self.weight,
               result.reshape(rows, self.out_features), a_scale.reshape(rows, -1), self.scale)
        return result
