import warnings
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import scipy.stats
import torch
import torch.nn as nn


def variance_decomposition(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    rcond: Optional[float] = None,
    zero_intercept: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Args:
        inputs (torch.Tensor): inputs data. B (batch size) x D (Dimension) or B (batch size) x L (input data length) x D (Dimension)
        targets (torch.Tensor, optional): target data. Usually, teaching data. B x 1. For training, this is required.
        rcond (float, optional): See https://pytorch.org/docs/stable/generated/torch.linalg.lstsq.html
        zero_intercept (bool, optional): if True, set intercept to 0.

    Returns:
        residual (torch.Tensor): residual of variance decomposition
        intercept (torch.Tensor): 1 Dim. Zero when zero_intercept is True
        coefficients (torch.Tensor): D or L x D. Coefficient for each factors.
    """
    batch_size = inputs.size(0)
    other_shape = inputs.size()[1:]
    if targets.size() != torch.Size([batch_size, 1]):
        raise ValueError("targets have to be (batch size) x 1")
    _inputs = inputs.reshape(batch_size, -1)
    total_param_dim = _inputs.size(1) + (0 if zero_intercept else 1)
    if batch_size < total_param_dim:
        raise AssertionError("batch_size is too small to fit.")
    if not zero_intercept:
        _inputs = torch.cat([torch.ones(batch_size, 1).to(_inputs), _inputs], dim=-1)
    torch_coefficient, _, _, _ = torch.linalg.lstsq(_inputs, targets, rcond=rcond)
    res = targets.reshape(batch_size) - (
        _inputs * torch_coefficient.squeeze(dim=-1)
    ).sum(dim=-1)
    if zero_intercept:
        torch_coefficient = torch.cat(
            [torch.zeros(1, 1).to(torch_coefficient), torch_coefficient], dim=0
        )
    return (
        res,
        torch_coefficient[0:1].reshape(-1),
        torch_coefficient[1:].reshape(other_shape),
    )


class VarianceDecompositionContextManagerFirst:
    def __init__(self, parent: "VarianceDecomposition"):
        self.parent = parent

    def __enter__(self) -> None:
        self.parent.enabled_analysis_first = True
        self.parent.analysis_init()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore
        self.parent.enabled_analysis_first = False
        self.parent.analysis_first_end()


class VarianceDecompositionContextManagerSecond:
    def __init__(self, parent: "VarianceDecomposition"):
        self.parent = parent

    def __enter__(self) -> None:
        self.parent.enabled_analysis_second = True

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore
        self.parent.analysis_second_end()
        self.parent.enabled_analysis_second = False


class VarianceDecomposition(nn.Module):
    """Variance decomposition module

    .. Note::
        This is under development. No additional documentation.
    """

    X_left: torch.Tensor
    X_right: torch.Tensor
    intercept: torch.Tensor
    coefficient: torch.Tensor
    ssr: torch.Tensor
    ssr_lim: torch.Tensor
    X_lefts_for_ssr: List[torch.Tensor]
    X_rights_for_ssr: List[torch.Tensor]
    As_for_ssr: List[torch.Tensor]
    N: float

    def __init__(
        self,
        inputs_dim: int,
        inputs_len: Optional[int] = None,
        zero_intercept: bool = False,
        momentum: Optional[float] = None,
    ):
        super(VarianceDecomposition, self).__init__()
        warnings.warn(
            "VarianceDecomposition module is under development. This API could be changed in future."
        )
        self.inputs_dim = inputs_dim
        self.inputs_len = inputs_len
        self.zero_intercept = zero_intercept
        self.register_buffer("intercept", torch.zeros(1))
        self.params_dim_for_solver = 0 if zero_intercept else 1
        self.coefficient_size: Union[int, Tuple[int, int]]
        if self.inputs_len:
            self.coefficient_size = (self.inputs_len, self.inputs_dim)
            self.params_dim_for_solver += self.inputs_dim * self.inputs_len
        else:
            self.coefficient_size = self.inputs_dim
            self.params_dim_for_solver += self.inputs_dim
        self.register_buffer("coefficient", torch.zeros(self.coefficient_size))
        # (X_1^T X_1 + X_2^T X_2 + ...) A = X_1^T X_1 A_1 + X_2^T X_2 A_2 + ...
        # X_left := X_1^T X_1 + X_2^T X_2 + ...
        # X_right := X_1^T X_1 A_1 + X_2^T X_2 A_2 + ...
        self.register_buffer(
            "X_left",
            torch.zeros(self.params_dim_for_solver, self.params_dim_for_solver),
        )
        self.register_buffer("X_right", torch.zeros(self.params_dim_for_solver, 1))
        self.momentum: float = momentum if momentum else 1.0
        self.enabled_analysis_first: bool = False
        self.enabled_analysis_second: bool = False
        self.analysis_step: int = 0
        self.granger_causality_statistics: Optional[torch.Tensor] = None
        self.register_buffer("ssr", torch.zeros(1))
        self.register_buffer("ssr_lim", torch.zeros(self.inputs_dim))

    def update_param(
        self,
        sample_intercept: torch.Tensor,
        sample_coefficient: torch.Tensor,
        inputs: torch.Tensor,
    ) -> None:
        sample_intercept = sample_intercept.detach()
        sample_coefficient = sample_coefficient.detach()
        inputs = inputs.detach()
        if self.zero_intercept:
            Ai = sample_coefficient.reshape(self.params_dim_for_solver, 1)
        else:
            Ai = torch.cat(
                [sample_intercept, sample_coefficient.reshape(-1)], dim=0
            ).reshape(self.params_dim_for_solver, 1)
        if not self.zero_intercept:
            inputs = inputs.reshape(-1, self.params_dim_for_solver - 1)
            inputs = torch.cat(
                [torch.ones(inputs.size(0), 1).to(inputs), inputs], dim=-1
            )
        _inputs = inputs.reshape(-1, self.params_dim_for_solver)
        XiTXi = torch.mm(_inputs.T, _inputs)
        self.X_left = self.momentum * self.X_left + XiTXi
        self.X_right = self.momentum * self.X_right + torch.mm(XiTXi, Ai)
        A = torch.mm(self.X_left.inverse(), self.X_right).reshape(
            self.params_dim_for_solver
        )
        if not self.zero_intercept:
            self.intercept = A[:1]
            self.coefficient = A[1:].reshape(self.coefficient_size)
        else:
            self.coefficient = A[:].reshape(self.coefficient_size)

    def forward(
        self,
        inputs: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        rcond: Optional[float] = None,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """forward

        Args:
            inputs (torch.Tensor): inputs data. B (batch size) x D (Dimension) or B (batch size) x L (input data length) x D (Dimension)
            targets (torch.Tensor, optional): target data. Usually, teaching data. B x 1. For training, this is required.
            rcond (float, optional): See https://pytorch.org/docs/stable/generated/torch.linalg.lstsq.html
            zero_intercept (bool, optional): if True, set intercept to 0.

        Returns:
            residual (torch.Tensor, optional): residual of variance decomposition. None when targets is None.
            model prediction (torch.Tensor): prediction of this model

        Notes: Under developing
        """
        if self.training or self.enabled_analysis_first or self.enabled_analysis_second:
            batch_size = inputs.size(0)
            if batch_size < self.inputs_dim * (self.inputs_len or 1) + (
                0 if self.zero_intercept else 1
            ):
                raise AssertionError("batch_size is too small.")
        if self.training:
            if targets is None:
                raise ValueError(
                    "targets is required for training. Please set targets or change to eval"
                )
            sample_res, sample_intercept, sample_coefficient = variance_decomposition(
                inputs=inputs,
                targets=targets,
                rcond=rcond,
                zero_intercept=self.zero_intercept,
            )
            self.update_param(
                sample_intercept=sample_intercept,
                sample_coefficient=sample_coefficient,
                inputs=inputs,
            )
        pred = (inputs * self.coefficient).reshape(inputs.size(0), -1).sum(
            dim=-1, keepdim=True
        ) + self.intercept
        global_res = (targets - pred) if targets is not None else None
        if self.enabled_analysis_first or self.enabled_analysis_second:
            if targets is None:
                raise ValueError("targets is required for analysis")
            if global_res is None:
                raise AssertionError
            if self.enabled_analysis_first:
                self._calc_granger_causality_for_batch(
                    inputs=inputs, targets=targets, res=global_res
                )
            if self.enabled_analysis_second:
                self._calc_granger_causality_final_for_batch(
                    inputs=inputs, targets=targets, res=global_res
                )
        return global_res, pred

    def analysis_init(self) -> None:
        if self.analysis_step != 0:
            raise AssertionError("previous analysis is not finished")
        self.analysis_step = 1
        self.ssr = torch.zeros_like(self.ssr)
        self.ssr_lim = torch.zeros_like(self.ssr_lim)
        params_dim_for_solver = self.params_dim_for_solver - (
            1 if self.inputs_len is None else self.inputs_len
        )
        self.X_lefts_for_ssr: List[torch.Tensor] = [
            torch.zeros(params_dim_for_solver, params_dim_for_solver).to(self.X_left)
            for _ in range(self.inputs_dim)
        ]
        self.X_rights_for_ssr: List[torch.Tensor] = [
            torch.zeros(params_dim_for_solver, 1).to(self.X_right)
            for _ in range(self.inputs_dim)
        ]
        self.N = 0

    def analysis_first_end(self) -> None:
        params_dim_for_solver = self.params_dim_for_solver - (
            1 if self.inputs_len is None else self.inputs_len
        )
        self.As_for_ssr = [
            torch.mm(
                self.X_lefts_for_ssr[i].inverse(), self.X_rights_for_ssr[i]
            ).reshape(params_dim_for_solver)
            for i in range(self.inputs_dim)
        ]
        self.analysis_step = 2

    def analysis_second_end(self) -> None:
        r = self.inputs_len if self.inputs_len else 1
        F = ((self.ssr_lim - self.ssr) / r) / (
            self.ssr / (self.N - self.inputs_dim * r - 1)
        )
        self.granger_causality_statistics = r * F
        # ToDo: torch.distributions.chi2.Chi2 does not support cdf at v1.11.0
        # chi2 = torch.distributions.chi2.Chi2(df=r)
        self.granger_causality_pvalues = 1 - torch.as_tensor(
            scipy.stats.chi2.cdf(self.granger_causality_statistics.cpu().numpy(), r)
        ).to(self.granger_causality_statistics)

        self.analysis_step = 0

    def _calc_granger_causality_for_batch(
        self, inputs: torch.Tensor, targets: torch.Tensor, res: torch.Tensor
    ) -> None:
        _inputs = [
            inputs[..., [j for j in range(self.inputs_dim) if i != j]]
            for i in range(self.inputs_dim)
        ]
        params_dim_for_solver = self.params_dim_for_solver - (
            1 if self.inputs_len is None else self.inputs_len
        )
        # res intercept coef
        vd_results_limited: List[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = list(
            map(
                lambda x: variance_decomposition(
                    inputs=x, targets=targets, zero_intercept=self.zero_intercept
                ),
                _inputs,
            )
        )
        if not self.zero_intercept:
            _inputs = [x.reshape(-1, params_dim_for_solver - 1) for x in _inputs]
            _inputs = [
                torch.cat([torch.ones(x.size(0), 1).to(x), x], dim=-1) for x in _inputs
            ]
        XiTXis = [
            torch.mm(
                _inputs[i].reshape(-1, params_dim_for_solver).T,
                _inputs[i].reshape(-1, params_dim_for_solver),
            )
            for i in range(self.inputs_dim)
        ]
        if self.zero_intercept:
            Ais = [
                vd_results_limited[i][2].reshape(params_dim_for_solver, 1)
                for i in range(self.inputs_dim)
            ]
        else:
            Ais = [
                torch.cat(
                    [vd_results_limited[i][1], vd_results_limited[i][2].reshape(-1)],
                    dim=0,
                ).reshape(params_dim_for_solver, 1)
                for i in range(self.inputs_dim)
            ]
        self.X_lefts_for_ssr = [
            self.X_lefts_for_ssr[i] + XiTXis[i] for i in range(self.inputs_dim)
        ]
        self.X_rights_for_ssr = [
            self.X_rights_for_ssr[i] + torch.mm(XiTXis[i], Ais[i])
            for i in range(self.inputs_dim)
        ]

    def _calc_granger_causality_final_for_batch(
        self, inputs: torch.Tensor, targets: torch.Tensor, res: torch.Tensor
    ) -> None:
        self.N += len(inputs)
        self.ssr += torch.square(res).sum()
        _inputs = [
            inputs[..., [j for j in range(self.inputs_dim) if i != j]]
            for i in range(self.inputs_dim)
        ]
        params_dim_for_solver = self.params_dim_for_solver - (
            1 if self.inputs_len is None else self.inputs_len
        )
        if not self.zero_intercept:
            _inputs = [x.reshape(-1, params_dim_for_solver - 1) for x in _inputs]
            _inputs = [
                torch.cat([torch.ones(x.size(0), 1).to(x), x], dim=-1) for x in _inputs
            ]
        else:
            _inputs = [x.reshape(-1, params_dim_for_solver) for x in _inputs]
        self.ssr_lim += torch.stack(
            [
                (
                    targets
                    - (_inputs[i] * self.As_for_ssr[i])
                    .reshape(_inputs[i].size(0), -1)
                    .sum(dim=-1, keepdim=True)
                )
                .square()
                .sum()
                for i in range(self.inputs_dim)
            ],
            dim=0,
        )

    def enable_analysis_first_step(self) -> VarianceDecompositionContextManagerFirst:
        if self.training:
            raise RuntimeError("eval() mode is required to enable analysis")
        if self.enabled_analysis_first or self.enabled_analysis_second:
            raise RuntimeError("invalid analysis procedure")
        return VarianceDecompositionContextManagerFirst(self)

    def enable_analysis_second_step(self) -> VarianceDecompositionContextManagerSecond:
        if self.training:
            raise RuntimeError("eval() mode is required to enable analysis")
        if (
            self.enabled_analysis_first
            or self.enabled_analysis_second
            or self.analysis_step != 2
        ):
            raise RuntimeError("invalid analysis procedure")
        return VarianceDecompositionContextManagerSecond(self)
