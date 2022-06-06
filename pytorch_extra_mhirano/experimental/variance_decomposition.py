import warnings
from typing import Optional
from typing import Tuple
from typing import Union

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
        _inputs = torch.cat([torch.ones_like(_inputs[:, :1]), _inputs], dim=-1)
    torch_coefficient, _, _, _ = torch.linalg.lstsq(_inputs, targets, rcond=rcond)
    res = targets.reshape(batch_size) - (
        _inputs * torch_coefficient.squeeze(dim=-1)
    ).sum(dim=-1)
    if zero_intercept:
        torch_coefficient = torch.cat(
            [torch.zeros_like(torch_coefficient[:1, :1]), torch_coefficient], dim=0
        )
    return (
        res,
        torch_coefficient[0:1].reshape(-1),
        torch_coefficient[1:].reshape(other_shape),
    )


class VarianceDecomposition(nn.Module):
    X_left: torch.Tensor
    X_right: torch.Tensor
    intercept: torch.Tensor
    coefficient: torch.Tensor

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
            inputs = torch.cat([torch.ones_like(inputs[:, :1]), inputs], dim=-1)
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
        """

        Args:
            inputs:
            targets:
            rcond:

        Returns:
            residual:
            model prediction:

        Notes: Under developing
        """
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
        return global_res, pred
