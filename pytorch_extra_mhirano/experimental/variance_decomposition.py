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
