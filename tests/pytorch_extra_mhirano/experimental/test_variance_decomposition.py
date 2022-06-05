from typing import List
from typing import Optional
from typing import Tuple

import pytest
import torch
from sklearn.linear_model import LinearRegression
from torch.testing import assert_close

from pytorch_extra_mhirano.experimental.variance_decomposition import (
    VarianceDecomposition,
)
from pytorch_extra_mhirano.experimental.variance_decomposition import (
    variance_decomposition,
)


@pytest.mark.parametrize("size", [[4, 2], [8, 2, 3], [4, 3], [32, 5, 6]])
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
    model = LinearRegression(fit_intercept=not zero_intercept)
    x = inputs.reshape(size[0], -1).numpy()
    y = target.reshape(-1).numpy()
    model.fit(x, y)
    assert_close(
        intercept, torch.Tensor([model.intercept_], device=device), atol=1e-3, rtol=1e-3
    )
    assert_close(
        coefficient,
        torch.Tensor(model.coef_, device=device).reshape(coefficient.size()),
        atol=1e-3,
        rtol=1e-3,
    )


@pytest.mark.gpu
@pytest.mark.parametrize("size", [[4, 2], [8, 2, 3], [4, 3], [32, 5, 6]])
@pytest.mark.parametrize("zero_intercept", [True, False])
def test_variance_decomposition_1_gpu(size: List[int], zero_intercept: bool) -> None:
    test_variance_decomposition_1(
        size=size, zero_intercept=zero_intercept, device=torch.device("cuda")
    )


class TestVarianceDecomposition:
    @pytest.mark.gpu
    @pytest.mark.parametrize(
        "size", [(4, None, 2), (8, 2, 3), (4, None, 3), (32, 5, 6)]
    )
    @pytest.mark.parametrize("zero_intercept", [True, False])
    def test__init__(
        self,
        size: Tuple[int, Optional[int], int],
        zero_intercept: bool,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        batch_size, input_len, input_dim = size
        vd = VarianceDecomposition(
            inputs_dim=input_dim, inputs_len=input_len, zero_intercept=zero_intercept
        )
        assert vd.inputs_len == input_len
        assert vd.inputs_dim == input_dim
        vd.to(device)
        torch.manual_seed(42)
        _size: List[int] = [x for x in size if x]
        inputs = torch.rand(_size).to(device)
        target = torch.rand([size[0], 1]).to(device)
        vd.forward(inputs=inputs, targets=target)
        model = LinearRegression(fit_intercept=not zero_intercept)
        x = inputs.reshape(size[0], -1).numpy()
        y = target.reshape(-1).numpy()
        model.fit(x, y)
        assert_close(
            vd.intercept,
            torch.Tensor([model.intercept_], device=device),
            atol=1e-3,
            rtol=1e-3,
        )
        assert_close(
            vd.coefficient,
            torch.Tensor(model.coef_, device=device).reshape(vd.coefficient.size()),
            atol=1e-3,
            rtol=1e-3,
        )
