# SPDX-License-Identifier: Apache-2.0
"""E4M3 format laws, circular packing, and native encoder round trips."""
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from gptqmodel.nn_modules.exllamav3_torch import ExllamaV3TorchLinear, _codebook_lut
from gptqmodel.utils.kimi_k3_core_format import validate_core_tensors


def packed(trellis):
    return dict(trellis=trellis, suh=torch.ones(trellis.shape[0] * 16).half(),
                svh=torch.ones(trellis.shape[1] * 16).half(),
                e4m3=torch.tensor(0xCBAC1FED, dtype=torch.uint32).view(torch.int32))


def test_all_states_round_native_half_sum_before_e4m3():
    states = np.arange(65536, dtype=np.uint32)
    raw = (states * np.uint32(0xCBAC1FED) & np.uint32(0x8FFF8FFF)) ^ np.uint32(0x3B603B60)
    lo = (raw & 65535).astype('<u2').view('<f2').astype(np.float64)
    hi = (raw >> 16).astype('<u2').view('<f2').astype(np.float64)
    value = (lo + hi).astype(np.float16).astype(np.float64)
    magnitude = np.minimum(np.abs(value), 448.)
    step = np.exp2(np.floor(np.log2(np.maximum(magnitude, 2. ** -6))) - 3)
    expected = np.copysign(np.minimum(np.rint(magnitude / step) * step, 448.), value)
    actual = _codebook_lut('e4m3', 'cpu', None)
    np.testing.assert_array_equal(actual.numpy(), expected.astype(np.float32))
    assert torch.equal(actual, actual.to(torch.float8_e4m3fn).float())


def test_k6_cpu_decode_matches_engine_oracle_bitwise():
    # Point at the engine's independent oracle rather than duplicating its codec.
    path = os.environ.get('EXL3_ENGINE_TRELLIS_REFERENCE')
    if not path:
        pytest.skip('set EXL3_ENGINE_TRELLIS_REFERENCE to engine tests/_reference/native_trellis.py')
    spec = importlib.util.spec_from_file_location('engine_trellis_reference', Path(path))
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    generator = torch.Generator().manual_seed(787)
    trellis = torch.randint(-32768, 32768, (8, 8, 96), dtype=torch.int16, generator=generator)
    module = ExllamaV3TorchLinear.from_tensors(in_features=128, out_features=128,
                                               name='pilot', tensors=packed(trellis))
    expected = oracle.Reference('e4m3').decode_tiles(trellis.numpy().tobytes(), 6)
    expected = expected.reshape(8, 8, 16, 16).transpose(0, 2, 1, 3).reshape(128, 128)
    assert torch.equal(module.get_inner_weight_tensor(dtype=torch.float32), torch.from_numpy(expected))
    validate_core_tensors(packed(trellis), (128, 128))


def test_marker_rejects_ambiguous_decode():
    tensors = packed(torch.zeros((8, 8, 96), dtype=torch.int16))
    tensors['mcg'] = tensors['e4m3'].clone()
    module = ExllamaV3TorchLinear.from_tensors(in_features=128, out_features=128,
                                               name='pilot', tensors=tensors)
    with pytest.raises(ValueError, match='ambiguous'):
        module.get_inner_weight_tensor()
    with pytest.raises(ValueError):
        validate_core_tensors(tensors, (128, 128))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA encoder oracle')
def test_k6_viterbi_pack_reconstruct_round_trip():
    from gptqmodel.exllamav3.ext import exllamav3_ext as ext
    from gptqmodel.exllamav3.modules.quant.exl3_lib.quantize import (
        pack_trellis, quantize_tiles, tensor_core_perm_i)
    torch.manual_seed(787)
    tiles = torch.randn((64, 256), device='cuda', dtype=torch.float32)
    decoded, states = quantize_tiles(tiles, dict(K=6, e4m3=True))
    trellis = pack_trellis(states.reshape(8, 8, 256), dict(K=6))
    actual = torch.empty((128, 128), device='cuda', dtype=torch.float16)
    ext.reconstruct(actual, trellis, 6, False, False, True)
    expected = decoded[:, tensor_core_perm_i('cuda')].reshape(8, 8, 16, 16)
    expected = expected.permute(0, 2, 1, 3).reshape(128, 128).half()
    assert torch.equal(actual, expected)
    assert torch.equal(decoded, decoded.to(torch.float8_e4m3fn).float())
    cpu = ExllamaV3TorchLinear.from_tensors(in_features=128, out_features=128,
        name='roundtrip', tensors=packed(trellis.cpu()))
    assert torch.equal(actual.cpu().float(), cpu.get_inner_weight_tensor(dtype=torch.float32))
