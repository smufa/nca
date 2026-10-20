"""Sample pool used to train persistent / regenerating models.

Instead of always starting from the seed, we keep a pool of previously
observed states, train on a random batch of them and write the resulting
states back. This lets the model learn to *persist* an already-formed
pattern, not just grow it from a single cell.
"""

from __future__ import annotations

import torch


class SamplePool:
    def __init__(self, x: torch.Tensor) -> None:
        """``x`` has shape ``[pool_size, C, H, W]``."""
        self.x = x

    def __len__(self) -> int:
        return self.x.shape[0]

    def sample(self, n: int, generator: torch.Generator | None = None):
        """Return ``(indices, batch)`` for ``n`` distinct pool entries."""
        # randperm uses the (CPU) generator for reproducibility, then the indices
        # are moved to the pool's device (the pool may live on CUDA).
        idx = torch.randperm(len(self.x), generator=generator)[:n].to(self.x.device)
        return idx, self.x[idx].clone()

    def commit(self, idx: torch.Tensor, x: torch.Tensor) -> None:
        """Write ``x`` back to the slots at ``idx``."""
        with torch.no_grad():
            self.x[idx] = x.to(self.x.dtype)
