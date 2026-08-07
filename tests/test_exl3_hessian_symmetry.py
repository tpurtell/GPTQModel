import pytest
import torch

from gptqmodel.exllamav3.modules.quant.exl3_lib.quantize import (
    restore_hessian_symmetry_,
)


def test_restore_hessian_symmetry_averages_without_diagonal_damping() -> None:
    hessian = torch.tensor(
        [[2.0, 0.75, -0.2], [0.25, 3.0, 0.6], [0.4, 0.2, 1.5]],
        dtype=torch.float32,
    )
    expected = (hessian + hessian.T) * 0.5
    expected_diagonal = torch.diagonal(hessian).clone()
    expected_correction = float((hessian - expected).abs().max().item())
    quant_args = {}

    restore_hessian_symmetry_(hessian, quant_args)

    assert torch.equal(hessian, hessian.T)
    assert torch.equal(hessian, expected)
    assert torch.equal(torch.diagonal(hessian), expected_diagonal)
    assert quant_args["hessian_symmetry_restoration"] == "mean-with-transpose-fp32"
    assert quant_args["hessian_symmetry_correction_max_abs"] == pytest.approx(
        expected_correction
    )


def test_restore_hessian_symmetry_rejects_non_square_input() -> None:
    with pytest.raises(ValueError, match="square matrix"):
        restore_hessian_symmetry_(torch.zeros((2, 3)), {})
