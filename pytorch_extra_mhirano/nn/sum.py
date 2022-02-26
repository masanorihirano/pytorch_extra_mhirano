import torch
import torch.nn as nn

__all__ = ["SumLayer"]


class SumLayer(nn.Module):
    def __init__(self, dim: int) -> None:
        super(SumLayer, self).__init__()
        self.dim = dim

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.sum(inputs, self.dim)
