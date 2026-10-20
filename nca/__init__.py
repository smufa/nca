"""Neural Cellular Automata (a PyTorch reimplementation of "Growing Neural
Cellular Automata", Mordvintsev et al., Distill 2020).

State tensors use the channels-first layout ``[B, C, H, W]``.
"""

from .model import NeuralCA, get_living_mask
from .pool import SamplePool
from .targets import load_target, make_seed, prepare_target

__all__ = [
    "NeuralCA",
    "get_living_mask",
    "SamplePool",
    "load_target",
    "make_seed",
    "prepare_target",
]
