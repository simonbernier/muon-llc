"""The interface a testbed exposes to the samplers.

A problem owns a dataset and a parameterisation by flat vectors theta of length `dim`. Losses are
batched: theta may carry leading dims (chains, temperatures, centres) and the loss returns one
value per leading index, so chains never mix. `DLNProblem` and `TMSProblem` both satisfy this;
a new testbed only has to provide the same methods to use `run_sgld` / `run_mala`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class Problem(Protocol):
    @property
    def n(self) -> int:
        """Number of samples; the default inverse temperature is n / log n."""

    @property
    def dim(self) -> int:
        """Number of parameters."""

    @property
    def device(self) -> torch.device: ...

    @property
    def dtype(self) -> torch.dtype: ...

    def loss(self, theta: torch.Tensor, idx: torch.Tensor | None = None) -> torch.Tensor:
        """Average negative log-likelihood L_n for theta (..., dim) -> (...); `idx` selects a minibatch."""

    def loss_and_grad(self, theta: torch.Tensor, idx: torch.Tensor | None = None
                      ) -> tuple[torch.Tensor, torch.Tensor]:
        """(loss (...), gradient (..., dim)), both detached."""

    def minibatch_indices(self, batch_size: int, generator: torch.Generator | None) -> torch.Tensor:
        """Indices of one minibatch, drawn from `generator` on the CPU and moved to `device`."""
