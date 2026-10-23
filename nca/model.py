"""The Neural Cellular Automaton module.

The update rule follows the reference TensorFlow implementation from
https://distill.pub/2020/growing-ca/ as closely as possible, with tensors in
channels-first ``[B, C, H, W]`` layout.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import step
from .spec import ModelSpec

ALPHA_CHANNEL = step.ALPHA_CHANNEL
ALIVE_THRESHOLD = step.ALIVE_THRESHOLD


def _circular_pad2d(x: torch.Tensor, pad: int) -> torch.Tensor:
    """Wrap-pad the last two dimensions (periodic / toroidal boundaries).

    Exports as a single ONNX ``Pad`` (mode ``wrap``, opset 19) instead of a
    Slice/Concat sandwich, which keeps the per-step graph small - and every graph
    node is a separate WebGPU dispatch.
    """
    if pad == 0:
        return x
    return F.pad(x, (pad, pad, pad, pad), mode="circular")


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


def _canonical_kernels(angle: float = 0.0) -> torch.Tensor:
    """The three 3x3 perception kernels, ``(identity, d/dx, d/dy)``, as ``[3,3,3]``.

    The numbers live in `nca.step` (the torch-free definition shared with the GPU
    bench, `nca.viz` and the frontend); this is only the torch view of them.
    ``angle`` rotates dx/dy, which is what the frontend's kernel lab exposes.
    """
    return torch.from_numpy(step.canonical_kernels(angle))


def _kernels_to_weight(channel_n: int, kernels: torch.Tensor) -> torch.Tensor:
    """Expand shared ``[3,3,3]`` kernels to a ``[3C, 1, 3, 3]`` depthwise weight.

    Every channel gets the same ``(identity, d/dx, d/dy)`` triple, so this is just
    a repeat: the frontend and the WGSL step likewise assume one shared kernel.
    """
    return kernels.repeat(channel_n, 1, 1).unsqueeze(1)


def _perception_kernels(channel_n: int, angle: float = 0.0) -> torch.Tensor:
    """The fixed (canonical) depthwise Sobel perception weight.

    Returns a ``[3 * channel_n, 1, 3, 3]`` weight for a depthwise convolution
    with ``groups=channel_n``. For every input channel the three output
    channels are ``[identity, d/dx, d/dy]``.
    """
    return _kernels_to_weight(channel_n, _canonical_kernels(angle))


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
        learn_perception: bool = False,
        cond_n: int = 0,
        hidden_layers: int = 1,
    ) -> None:
        super().__init__()
        self.channel_n = channel_n
        self.fire_rate = fire_rate
        self.wrap = wrap
        self.learn_perception = learn_perception
        # cond_n > 0 makes the CA conditional: a one-hot vector of length cond_n
        # is appended to every cell's perception, telling the update rule *which*
        # target to grow. It is a separate signal, not part of the state, so the
        # living mask cannot zero it away.
        self.cond_n = cond_n
        self.hidden_layers = hidden_layers

        # Perception output has 3 * channel_n channels (+ cond_n for a conditional
        # model); the update rule is a small MLP per cell via 1x1 convolutions.
        # hidden_layers counts the hidden (ReLU) blocks, so 1 is the paper shape and
        # 2 adds a hidden -> hidden projection.
        layers: list[nn.Module] = [
            nn.Conv2d(3 * channel_n + cond_n, hidden_n, kernel_size=1),
            nn.ReLU(),
        ]
        for _ in range(max(hidden_layers - 1, 0)):
            layers += [nn.Conv2d(hidden_n, hidden_n, kernel_size=1), nn.ReLU()]
        layers += [nn.Conv2d(hidden_n, channel_n, kernel_size=1)]
        self.update = nn.Sequential(*layers)

        # "Do-nothing" initialisation: the final layer starts at zero so the
        # CA initially leaves the state untouched.
        nn.init.zeros_(self.update[-1].weight)
        nn.init.zeros_(self.update[-1].bias)

        if learn_perception:
            # Learning the perception means learning dx/dy; the identity kernel is
            # kept fixed (it is what lets a cell see its own state). The kernels
            # are shared across channels, matching the WGSL 27-float kernel buffer
            # and the frontend's kernel lab.
            canonical = _canonical_kernels()
            self.register_buffer("_identity_kernel", canonical[0:1].clone(), persistent=False)
            self.sobel = nn.Parameter(canonical[1:3].clone())  # [2,3,3] = (dx, dy)
        else:
            self.register_buffer(
                "_perception_weight",
                _perception_kernels(channel_n),
                persistent=False,
            )

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @classmethod
    def from_spec(cls, spec: "ModelSpec") -> "NeuralCA":
        """Build the module described by a :class:`nca.spec.ModelSpec`."""
        return cls(
            channel_n=spec.channel_n,
            fire_rate=spec.fire_rate,
            hidden_n=spec.hidden_n,
            wrap=spec.wrap,
            learn_perception=spec.learn_perception,
            cond_n=spec.cond_n,
            hidden_layers=spec.hidden_layers,
        )

    @property
    def spec(self) -> "ModelSpec":
        """The shape of this module, as a :class:`nca.spec.ModelSpec`."""
        return ModelSpec.from_model(self)

    def kernels(self, angle: float = 0.0) -> torch.Tensor:
        """The ``[3,3,3]`` ``(identity, d/dx, d/dy)`` kernels currently in use.

        With ``learn_perception`` these are the trained parameters; ``angle`` only
        applies to the fixed canonical kernels.
        """
        if self.learn_perception:
            return torch.cat([self._identity_kernel, self.sobel], dim=0)
        return _canonical_kernels(angle)

    def perception_weight(self, angle: float = 0.0) -> torch.Tensor:
        """The ``[3C, 1, 3, 3]`` depthwise perception weight currently in use."""
        return _kernels_to_weight(self.channel_n, self.kernels(angle))

    def perceive(self, x: torch.Tensor, angle: float = 0.0) -> torch.Tensor:
        """Concatenate each cell's state with its x/y spatial gradients."""
        if self.learn_perception:
            weight = self.perception_weight()
        elif angle == 0.0:
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
        update_mask: torch.Tensor | None = None,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """One CA step.

        ``update_mask`` (``[B, 1, H, W]``) lets a caller supply the stochastic
        fire mask instead of drawing one here. Training pre-draws a whole batch
        of masks for the unroll, which removes one RNG kernel + one comparison
        per step and makes the step a pure function of its inputs.

        ``cond`` (``[B, cond_n]`` or ``[B, cond_n, 1, 1]``) is the one-hot target
        selector for a conditional model; it is broadcast over the grid.
        """
        pre_life_mask = get_living_mask(x, self.wrap)

        y = self.perceive(x, angle)
        if self.cond_n:
            if cond is None:
                raise ValueError(f"model has cond_n={self.cond_n} but no cond was given")
            if cond.dim() == 2:
                cond = cond[:, :, None, None]
            y = torch.cat([y, cond.expand(-1, -1, y.shape[-2], y.shape[-1])], dim=1)
        dx = self.update(y) * step_size

        if update_mask is None:
            if fire_rate is None:
                fire_rate = self.fire_rate
            # Asynchronous update: each cell updates with probability fire_rate.
            update_mask = torch.rand_like(x[:, :1]) <= fire_rate
        x = x + dx * update_mask

        post_life_mask = get_living_mask(x, self.wrap)
        life_mask = pre_life_mask & post_life_mask
        return x * life_mask
