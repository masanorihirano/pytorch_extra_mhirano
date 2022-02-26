from typing import Optional

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    r"""
    Residual Block
    If bottleneck=None, this is plain Residual Block with 2 FC layers (input_dim=>input_dim=>input_dim) and 2 activation layers.
    If bottleneck=x, this is bottleneck residual block with 3 FC layers(input_dim=>x=>input_dim) and 3 activation layers.
    """

    def __init__(
        self,
        input_dim: int,
        bottleneck: Optional[int] = None,
        activation_func: nn.Module = nn.ReLU(),
        bias: bool = True,
    ):
        super(ResidualBlock, self).__init__()
        layers = nn.ModuleList()
        layers.append(
            nn.Linear(
                input_dim, input_dim if bottleneck is None else bottleneck, bias=bias
            )
        )
        layers.append(activation_func)
        if bottleneck is not None:
            layers.append(nn.Linear(bottleneck, bottleneck, bias=bias))
            layers.append(activation_func)
        layers.append(
            nn.Linear(
                input_dim if bottleneck is None else bottleneck, input_dim, bias=bias
            )
        )
        layers.append(activation_func)
        self.layers = nn.Sequential(*layers)
        self.activation = activation_func

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        y = self.layers(inputs)
        return self.activation(y + inputs)
