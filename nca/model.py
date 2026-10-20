"""The Neural Cellular Automaton module.

The update rule follows the reference TensorFlow implementation from
https://distill.pub/2020/growing-ca/ as closely as possible, with tensors in
channels-first ``[B, C, H, W]`` layout.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

ALPHA_CHANNEL = 3
ALIVE_THRESHOLD = 0.1


def _circular_pad2d(x: torch.Tensor, pad: int) -> torch.Tensor:
    """Wrap-pad the last two dimensions (periodic / toroidal boundaries).

    Implemented with slicing + ``cat`` (rather than ``F.pad(..., mode='circular')``)
    so the exact same graph exports cleanly to ONNX on opset 18.
    """
    if pad == 0:
        return x
    # width (last dim): left/right wraps
    x = torch.cat([x[..., -pad:], x, x[..., :pad]], dim=-1)
    # height (second-to-last dim): top/bottom wraps
    x = torch.cat([x[..., -pad:, :], x, x[..., :pad, :]], dim=-2)
    return x


def get_living_mask(x: torch.Tensor, wrap: bool = True) -> torch.Tensor:
    """Return a ``[B, 1, H, W]`` boolean mask of cells that are alive.

    A cell is alive when any cell in its 3x3 neighbourhood has an alpha
    value above ``ALIVE_THRESHOLD``. With ``wrap`` the neighbourhood is
    periodic.
    """
    alpha = x[:, ALPHA_CHANNEL : ALPHA_CHANNEL + 1]
    if wrap:
        pooled = F.max_pool2d(_circular_pad2d(alpha, 1), kernel_size=3, stride=1, padding=0)
    else:
        pooled = F.max_pool2d(alpha, kernel_size=3, stride=1, padding=1)
    return pooled > ALIVE_THRESHOLD


def _perception_kernels(channel_n: int, angle: float = 0.0) -> torch.Tensor:
    """Build the fixed depthwise Sobel perception kernel.

    Returns a ``[3 * channel_n, 1, 3, 3]`` weight for a depthwise convolution
    with ``groups=channel_n``. For every input channel the three output
    channels are ``[identity, d/dx, d/dy]``.
    """
    ident = torch.tensor([0.0, 1.0, 0.0])
    ident = torch.outer(ident, ident)

    # Sobel filter, normalised by 8 as in the reference implementation.
    dx = torch.outer(
        torch.tensor([1.0, 2.0, 1.0]), torch.tensor([-1.0, 0.0, 1.0])
    ) / 8.0
    dy = dx.t()

    c, s = math.cos(angle), math.sin(angle)
    # [3, 3, 3] -> last dim indexes (identity, dx, dy)
    kernels = torch.stack([ident, c * dx - s * dy, s * dx + c * dy], dim=0)

    weight = torch.zeros(3 * channel_n, 1, 3, 3)
    for ci in range(channel_n):
        weight[3 * ci : 3 * ci + 3, 0] = kernels
    return weight


class NeuralCA(nn.Module):
    """A neural cellular automaton with ``channel_n`` state channels.

    The first three channels are RGB and channel 3 is alpha; the remaining
    channels are hidden and carry whatever signalling the update rule learns.
    """

    def __init__(
        self,
        channel_n: int = 16,
        fire_rate: float = 0.5,
        hidden_n: int = 128,
        wrap: bool = True,
    ) -> None:
        super().__init__()
        self.channel_n = channel_n
        self.fire_rate = fire_rate
        self.wrap = wrap

        # Perception output has 3 * channel_n channels; the update rule is a
        # small MLP applied per-cell via 1x1 convolutions.
        self.update = nn.Sequential(
            nn.Conv2d(3 * channel_n, hidden_n, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_n, channel_n, kernel_size=1),
        )
        # "Do-nothing" initialisation: the final layer starts at zero so the
        # CA initially leaves the state untouched.
        nn.init.zeros_(self.update[2].weight)
        nn.init.zeros_(self.update[2].bias)

        self.register_buffer(
            "_perception_weight",
            _perception_kernels(channel_n),
            persistent=False,
        )

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def perceive(self, x: torch.Tensor, angle: float = 0.0) -> torch.Tensor:
        """Concatenate each cell's state with its x/y spatial gradients."""
        if angle == 0.0:
            weight = self._perception_weight
        else:
            weight = _perception_kernels(self.channel_n, angle).to(
                device=x.device, dtype=x.dtype
            )
        if self.wrap:
            return F.conv2d(
                _circular_pad2d(x, 1), weight, stride=1, padding=0, groups=self.channel_n
            )
        return F.conv2d(x, weight, stride=1, padding=1, groups=self.channel_n)

    def forward(
        self,
        x: torch.Tensor,
        fire_rate: float | None = None,
        angle: float = 0.0,
        step_size: float = 1.0,
    ) -> torch.Tensor:
        pre_life_mask = get_living_mask(x, self.wrap)

        y = self.perceive(x, angle)
        dx = self.update(y) * step_size

        if fire_rate is None:
            fire_rate = self.fire_rate
        # Asynchronous update: each cell updates with probability fire_rate.
        update_mask = torch.rand_like(x[:, :1]) <= fire_rate
        x = x + dx * update_mask

        post_life_mask = get_living_mask(x, self.wrap)
        life_mask = pre_life_mask & post_life_mask
        return x * life_mask
