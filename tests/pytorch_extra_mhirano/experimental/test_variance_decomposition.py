from typing import List
from typing import Optional
from typing import Tuple

import pytest
import torch
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.vector_ar.var_model import VAR
from torch.testing import assert_close

from pytorch_extra_mhirano.experimental.variance_decomposition import (
    VarianceDecomposition,
)
from pytorch_extra_mhirano.experimental.variance_decomposition import (
    variance_decomposition,
)


@pytest.mark.parametrize("size", [[4, 2], [8, 2, 3], [5, 3], [33, 5, 6]])
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
    x = inputs.reshape(size[0], -1).cpu().numpy()
    y = target.reshape(-1).cpu().numpy()
    model.fit(x, y)
    assert_close(
        intercept,
        torch.Tensor([model.intercept_]).to(device),
        atol=1e-3,
        rtol=1e-3,
        check_stride=False,
    )
    assert_close(
        coefficient,
        torch.Tensor(model.coef_).to(device).reshape(coefficient.size()),
        atol=1e-3,
        rtol=1e-3,
    )


@pytest.mark.gpu
@pytest.mark.parametrize("size", [[4, 2], [8, 2, 3], [5, 3], [33, 5, 6]])
@pytest.mark.parametrize("zero_intercept", [True, False])
def test_variance_decomposition_1_gpu(size: List[int], zero_intercept: bool) -> None:
    test_variance_decomposition_1(
        size=size, zero_intercept=zero_intercept, device=torch.device("cuda")
    )


class TestVarianceDecomposition:
    @pytest.mark.parametrize(
        "size", [(4, None, 2), (8, 2, 3), (5, None, 3), (33, 5, 6)]
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
        x = inputs.reshape(size[0], -1).cpu().numpy()
        y = target.reshape(-1).cpu().numpy()
        model.fit(x, y)
        assert_close(
            vd.intercept,
            torch.Tensor([model.intercept_]).to(device),
            atol=1e-3,
            rtol=1e-3,
        )
        assert_close(
            vd.coefficient,
            torch.Tensor(model.coef_).to(device).reshape(vd.coefficient.size()),
            atol=1e-3,
            rtol=1e-3,
        )

    @pytest.mark.gpu
    @pytest.mark.parametrize(
        "size", [(4, None, 2), (8, 2, 3), (5, None, 3), (33, 5, 6)]
    )
    @pytest.mark.parametrize("zero_intercept", [True, False])
    def test__init__gpu(
        self, size: Tuple[int, Optional[int], int], zero_intercept: bool
    ) -> None:
        self.test__init__(
            size=size, zero_intercept=zero_intercept, device=torch.device("cuda")
        )

    @pytest.mark.parametrize(
        "size", [(4, None, 2), (8, 2, 3), (5, None, 3), (33, 5, 6)]
    )
    @pytest.mark.parametrize("zero_intercept", [True, False])
    @pytest.mark.parametrize("n_batch", [2, 10, 100])
    def test2(
        self,
        size: Tuple[int, Optional[int], int],
        zero_intercept: bool,
        n_batch: int,
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
        inputs_all = []
        targets_all = []
        for _ in range(n_batch):
            _size: List[int] = [x for x in size if x]
            inputs = torch.rand(_size)
            target = torch.rand([size[0], 1])
            inputs_all.append(inputs)
            targets_all.append(target)
        for inputs, target in zip(inputs_all, targets_all):
            inputs = inputs.to(device)
            target = target.to(device)
            vd.forward(inputs=inputs, targets=target)
        inputs_all_ = torch.cat(inputs_all, dim=0)
        targets_all_ = torch.cat(targets_all, dim=0)
        model = LinearRegression(fit_intercept=not zero_intercept)
        x = inputs_all_.reshape(size[0] * n_batch, -1).numpy()
        y = targets_all_.reshape(-1).numpy()
        model.fit(x, y)
        assert_close(
            vd.intercept,
            torch.Tensor([model.intercept_]).to(device),
            atol=1e-2,
            rtol=1e-2,
        )
        assert_close(
            vd.coefficient,
            torch.Tensor(model.coef_).to(device).reshape(vd.coefficient.size()),
            atol=1e-2,
            rtol=1e-2,
        )

    @pytest.mark.gpu
    @pytest.mark.parametrize(
        "size", [(4, None, 2), (8, 2, 3), (5, None, 3), (33, 5, 6)]
    )
    @pytest.mark.parametrize("zero_intercept", [True, False])
    @pytest.mark.parametrize("n_batch", [2, 10, 100, 1000])
    def test2_gpu(
        self, size: Tuple[int, Optional[int], int], zero_intercept: bool, n_batch: int
    ) -> None:
        self.test2(
            size=size,
            zero_intercept=zero_intercept,
            n_batch=n_batch,
            device=torch.device("cuda"),
        )

    @pytest.mark.parametrize(
        "size", [(4, None, 2), (8, 2, 3), (5, None, 3), (33, 5, 6)]
    )
    @pytest.mark.parametrize("zero_intercept", [True, False])
    @pytest.mark.parametrize("n_batch", [2, 10, 100])
    def test_grad(
        self,
        size: Tuple[int, Optional[int], int],
        zero_intercept: bool,
        n_batch: int,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        batch_size, input_len, input_dim = size
        vd = VarianceDecomposition(
            inputs_dim=input_dim, inputs_len=input_len, zero_intercept=zero_intercept
        )
        assert vd.inputs_len == input_len
        assert vd.inputs_dim == input_dim
        vd.to(device)
        layer1 = torch.nn.Linear(input_dim, input_dim).to(device)
        optimizer = torch.optim.Adam(layer1.parameters())
        torch.manual_seed(42)
        inputs_all = []
        targets_all = []
        for _ in range(n_batch):
            _size: List[int] = [x for x in size if x]
            inputs = torch.rand(_size)
            target = torch.rand([size[0], 1])
            inputs_all.append(inputs)
            targets_all.append(target)
        for inputs, target in zip(inputs_all, targets_all):
            inputs = inputs.to(device)
            target = target.to(device)
            inputs = layer1.forward(inputs)
            res, pred = vd.forward(inputs=inputs, targets=target)
            if res is None:
                raise AssertionError
            optimizer.zero_grad()
            res.square().sum().backward()
            optimizer.step()

    @pytest.mark.gpu
    @pytest.mark.parametrize(
        "size", [(4, None, 2), (8, 2, 3), (5, None, 3), (33, 5, 6)]
    )
    @pytest.mark.parametrize("zero_intercept", [True, False])
    @pytest.mark.parametrize("n_batch", [2, 10, 100, 1000])
    def test_grad_gpu(
        self, size: Tuple[int, Optional[int], int], zero_intercept: bool, n_batch: int
    ) -> None:
        self.test_grad(
            size=size,
            zero_intercept=zero_intercept,
            n_batch=n_batch,
            device=torch.device("cuda"),
        )

    @pytest.mark.parametrize(
        "size", [(8, None, 2), (16, 2, 3), (12, None, 3), (70, 5, 6)]
    )
    @pytest.mark.parametrize("zero_intercept", [True, False])
    def test_enable_analysis(
        self,
        size: Tuple[int, Optional[int], int],
        zero_intercept: bool,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        batch_size, input_len, input_dim = size
        vd = VarianceDecomposition(
            inputs_dim=input_dim, inputs_len=input_len, zero_intercept=zero_intercept
        ).to(device)
        torch.manual_seed(42)
        X = torch.rand([batch_size + (input_len or 1), input_dim]).to(device)
        inputs = torch.stack(
            [X[i : i + (input_len or 1), :].squeeze() for i in range(batch_size)], dim=0
        )
        target = torch.stack(
            [
                X[(input_len or 1) + i : (input_len or 1) + 1 + i, 0]
                for i in range(batch_size)
            ],
            dim=0,
        )
        vd.forward(inputs=inputs, targets=target)
        with pytest.raises(RuntimeError):
            with vd.enable_analysis_first_step():
                pass
        vd.eval()
        with vd.enable_analysis_first_step():
            vd.forward(inputs=inputs, targets=target)
        with vd.enable_analysis_second_step():
            vd.forward(inputs=inputs, targets=target)
        with vd.enable_analysis_first_step():
            vd.forward(
                inputs=inputs[: batch_size // 2], targets=target[: batch_size // 2]
            )
            vd.forward(
                inputs=inputs[batch_size // 2 :], targets=target[batch_size // 2 :]
            )
        with vd.enable_analysis_second_step():
            vd.forward(
                inputs=inputs[: batch_size // 2], targets=target[: batch_size // 2]
            )
            vd.forward(
                inputs=inputs[batch_size // 2 :], targets=target[batch_size // 2 :]
            )

        if not zero_intercept:
            model = VAR(X.cpu().numpy())
            results = model.fit(input_len, trend=("n" if zero_intercept else "c"))
            test_results = [
                results.test_causality(causing=i, caused=0, kind="wald")
                for i in range(input_dim)
            ]
            statistics = [test_results[i].test_statistic for i in range(input_dim)]
            pvalues = [test_results[i].pvalue for i in range(input_dim)]
            assert_close(
                torch.as_tensor(statistics).to(vd.granger_causality_statistics),
                vd.granger_causality_statistics,
                rtol=1e-4,
                atol=1e-4,
            )
            assert_close(
                torch.as_tensor(pvalues).to(vd.granger_causality_pvalues),
                vd.granger_causality_pvalues,
                rtol=1e-4,
                atol=1e-4,
            )

    @pytest.mark.gpu
    @pytest.mark.parametrize(
        "size", [(8, None, 2), (16, 2, 3), (10, None, 3), (70, 5, 6)]
    )
    @pytest.mark.parametrize("zero_intercept", [True, False])
    def test_enable_analysis_gpu(
        self, size: Tuple[int, Optional[int], int], zero_intercept: bool
    ) -> None:
        self.test_enable_analysis(
            size=size, zero_intercept=zero_intercept, device=torch.device("cuda")
        )
