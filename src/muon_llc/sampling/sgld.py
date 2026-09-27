"""Vectorised SGLD with the devinterp update rule.

    w <- w - (eps/2) [ nβ ∇L_batch(w) + γ (w - w0) ] + sqrt(eps) ξ,   ξ ~ N(0, I)

All chains run in one batched tensor of shape (C, D). The recorded loss is the exact
full-batch L_n(w) (cheap here), alongside the minibatch loss the update actually used.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..dln import DLNProblem


@dataclass
class ChainRecord:
    """delta_loss[..., t] = L_n(w_t) - L_n(w0) at each recorded draw (after burn-in)."""

    delta_loss: np.ndarray  # (..., T)
    delta_loss_burnin: np.ndarray  # (..., B)
    nbeta: np.ndarray  # (...) broadcastable to delta_loss[..., 0]
    gamma: float
    eps: np.ndarray  # final step size(s)
    acceptance: np.ndarray | None = None  # MALA only
    extras: dict | None = None


def run_sgld(
    problem: DLNProblem,
    theta0: torch.Tensor,
    nbeta: float,
    gamma: float,
    eps: float,
    n_chains: int,
    n_burnin: int,
    n_draws: int,
    batch_size: int,
    seed: int,
    init_jitter: float = 0.0,
) -> ChainRecord:
    device, dtype = problem.device, problem.dtype
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    theta0 = theta0.to(device=device, dtype=dtype).reshape(1, -1)
    D = theta0.shape[1]
    theta = theta0.repeat(n_chains, 1)
    if init_jitter > 0:
        theta = theta + init_jitter * torch.randn(n_chains, D, generator=g).to(device, dtype)
    L0 = problem.loss(theta0).item()
    total = n_burnin + n_draws
    rec = torch.empty(n_chains, total, dtype=torch.float64)
    rec_mb = torch.empty(n_chains, total, dtype=torch.float64)
    half_eps, sqrt_eps = 0.5 * eps, eps**0.5
    for t in range(total):
        idx = problem.minibatch_indices(batch_size, g)
        loss_mb, grad = problem.loss_and_grad(theta, idx)
        with torch.no_grad():
            drift = nbeta * grad + gamma * (theta - theta0)
            noise = torch.randn(n_chains, D, generator=g).to(device, dtype)
            theta = theta - half_eps * drift + sqrt_eps * noise
            rec[:, t] = (problem.loss(theta) - L0).double().cpu()
            rec_mb[:, t] = loss_mb.double().cpu()
        if not torch.isfinite(theta).all():
            raise FloatingPointError("SGLD diverged; reduce eps")
    return ChainRecord(
        delta_loss=rec[:, n_burnin:].numpy(),
        delta_loss_burnin=rec[:, :n_burnin].numpy(),
        nbeta=np.asarray(nbeta, dtype=np.float64),
        gamma=float(gamma),
        eps=np.asarray(eps, dtype=np.float64),
        extras={"minibatch_loss": rec_mb[:, n_burnin:].numpy(), "L0": L0},
    )
