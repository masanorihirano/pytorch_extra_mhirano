from typing import Optional

import torch
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss

__all__ = ["JSDivLoss", "KLDivLoss"]


class KLDivLoss(_Loss):
    def __init__(
        self,
        size_average: Optional[bool] = None,
        reduce: Optional[bool] = None,
        reduction: str = "mean",
    ) -> None:
        super(KLDivLoss, self).__init__(size_average, reduce, reduction)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_input = torch.log(inputs)
        return F.kl_div(log_input, targets, reduction=self.reduction)


class JSDivLoss(_Loss):
    def __init__(
        self,
        size_average: Optional[bool] = None,
        reduce: Optional[bool] = None,
        reduction: str = "mean",
    ) -> None:
        super(JSDivLoss, self).__init__(size_average, reduce, reduction)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return js_div(inputs, targets, reduction=self.reduction)


def js_div(
    p: torch.Tensor,
    q: torch.Tensor,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    m = (p + q) * 0.5

    reduced1 = torch.kl_div(
        torch.log(m), p, size_average=size_average, reduce=reduce, reduction=reduction
    )
    reduced2 = torch.kl_div(
        torch.log(m), q, size_average=size_average, reduce=reduce, reduction=reduction
    )

    reduced = (reduced1 + reduced2) * 0.5

    return reduced
