from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ref. https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_pyramids/py_pyramids.html


class PyramidDown(nn.Module):
    def __init__(self) -> None:
        super(PyramidDown, self).__init__()
        # [out_ch, in_ch, .., ..]
        self.filter = nn.Parameter(
            torch.tensor(
                [
                    [1, 4, 6, 4, 1],
                    [4, 16, 24, 16, 4],
                    [6, 24, 36, 24, 6],
                    [4, 16, 24, 16, 4],
                    [1, 4, 6, 4, 1],
                ],
                dtype=torch.float,
            ).reshape(1, 1, 5, 5)
            / 256,
            requires_grad=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        results = []
        for i in range(x.shape[1]):
            results.append(
                F.conv2d(x[:, i : i + 1, :, :], self.filter, padding=2, stride=2)
            )
        return torch.cat(results, dim=1)


class PyramidUp(nn.Module):
    def __init__(self) -> None:
        super(PyramidUp, self).__init__()
        # [out_ch, in_ch, .., ..]
        self.filter = nn.Parameter(
            torch.tensor(
                [
                    [1, 4, 6, 4, 1],
                    [4, 16, 24, 16, 4],
                    [6, 24, 36, 24, 6],
                    [4, 16, 24, 16, 4],
                    [1, 4, 6, 4, 1],
                ],
                dtype=torch.float,
            ).reshape(1, 1, 5, 5)
            / 256,
            requires_grad=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        upsample = F.interpolate(x, scale_factor=2)
        results = []
        for i in range(x.shape[1]):
            results.append(
                F.conv2d(upsample[:, i : i + 1, :, :], self.filter, padding=2)
            )
        return torch.cat(results, dim=1)


class LaplacianPyramidLayer(nn.Module):
    def __init__(self) -> None:
        super(LaplacianPyramidLayer, self).__init__()
        self.pyramid_down = PyramidDown()
        self.pyramid_up = PyramidUp()

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y = x
        if x.shape[-1] % 2 != 0:
            y = torch.cat([y, torch.zeros(y.shape[:-1]).unsqueeze(dim=-1)], dim=-1)
        if x.shape[-2] % 2 != 0:
            y = y.transpose(-1, -2)
            y = torch.cat([y, torch.zeros(y.shape[:-1]).unsqueeze(dim=-1)], dim=-1)
            y = y.transpose(-1, -2)
        down: torch.Tensor = self.pyramid_down(y)
        remade: torch.Tensor = self.pyramid_up(down)
        diff: torch.Tensor = y - remade
        if x.shape[-1] % 2 != 0:
            diff = diff[:, :, :, :-1]
        if x.shape[-1] % 2 != 0:
            diff = diff[:, :, :-1, :]
        return diff, down, remade
