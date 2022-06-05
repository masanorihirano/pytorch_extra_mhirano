from typing import List

import pytest
import torch
from torch.testing import assert_close

from pytorch_extra_mhirano.experimental.variance_decomposition import (
    variance_decomposition,
)


@pytest.mark.parametrize("size", [[4, 2], [8, 2, 3], [4, 3], [30, 5, 6]])
@pytest.mark.parametrize("zero_intercept", [True, False])
def test_variance_decomposition_1(
    size: List[int], zero_intercept: bool, device: torch.device = torch.device("cpu")
) -> None:
    torch.manual_seed(42)
    inputs = torch.rand(size).to(device)
    target = torch.rand([size[0], 1]).to(device)
    res, intercept, coefficient = variance_decomposition(
        inputs=inputs, targets=target, zero_intercept=zero_intercept
    )
    pred = (
        torch.mul(inputs, coefficient).reshape(size[0], -1).sum(dim=-1)
        + intercept
        + res
    )
    assert_close(target.reshape(-1), pred, rtol=1e-3, atol=1e-3)
    if zero_intercept:
        assert_close(intercept, torch.zeros_like(intercept))


@pytest.mark.gpu
@pytest.mark.parametrize("size", [[4, 2], [8, 2, 3], [4, 3], [30, 5, 6]])
@pytest.mark.parametrize("zero_intercept", [True, False])
def test_variance_decomposition_1_gpu(size: List[int], zero_intercept: bool) -> None:
    test_variance_decomposition_1(
        size=size, zero_intercept=zero_intercept, device=torch.device("cuda")
    )
