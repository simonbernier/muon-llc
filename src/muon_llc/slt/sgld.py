"""Vectorised SGLD with the devinterp update rule, batched over centres and chains.

    w <- w - (eps/2) [ nβ ∇L_batch(w) + γ (w - w0) ] + sqrt(eps) ξ,   ξ ~ N(0, I)

One call samples around B centres with C chains each; the state is a single tensor (B, C, D).
`nbeta`, `gamma` and `eps` are scalars or one value per centre, so a whole resolution sweep or
every checkpoint of a run goes through one call. On a GPU the cost of a step barely depends on
B·C, which is where the batching pays.

Chains never mix: the problem's loss returns one value per leading index. A problem with one
dataset per centre (TMS with per-run data) only has to broadcast over (B, C).

Recorded: the full-batch loss L_n(w_t) - L_n(w0) at every step, burn-in included, and the
minibatch loss the update used (when minibatching). The trace stays on the device and is copied
once at the end. Random numbers come from a CPU generator and are moved to the device, so a
seed gives the same chains on every device up to floating-point rounding.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .problem import Problem


@dataclass
class ChainRecord:
    """delta_loss[..., t] = L_n(w_t) - L_n(w0) at each recorded draw (after burn-in).

    Leading dims: (C,) for SGLD at one centre, (B, C) for SGLD at B centres, (K, C) for the
    MALA β-ladder.
    """

    delta_loss: np.ndarray  # (..., T)
    delta_loss_burnin: np.ndarray  # (..., B)
    nbeta: np.ndarray  # (...) broadcastable to delta_loss[..., 0]
    gamma: float | np.ndarray
    eps: np.ndarray  # final step size(s)
    acceptance: np.ndarray | None = None  # MALA only
    extras: dict | None = None

    def frac_below(self, tol: float = 0.0) -> np.ndarray:
        """Share of draws of each chain with loss below the centre's loss: the escape test of
        Chen et al. (2023), Appendix K. A chain that left the basin of w0 spends its time below."""
        return (self.delta_loss < -tol).mean(axis=-1)


def _per_centre(x, B: int, device, dtype) -> torch.Tensor:
    """Scalar or (B,) -> tensor (B, 1) on the device."""
    t = torch.as_tensor(np.broadcast_to(np.asarray(x, dtype=np.float64), (B,)).copy())
    return t.to(device=device, dtype=dtype).reshape(B, 1)


def run_sgld(
    problem: Problem,
    theta0: torch.Tensor,
    nbeta,
    gamma,
    eps,
    n_chains: int,
    n_burnin: int,
    n_draws: int,
    batch_size: int | None,
    seed: int | torch.Generator,
    init_jitter: float = 0.0,
    on_diverge: str = "raise",
) -> ChainRecord:
    """SGLD around `theta0`: (D,) for one centre or (B, D) for B centres.

    nbeta, gamma, eps   scalar, or one value per centre
    batch_size          minibatch size for the drift; None = full batch (no index draw)
    seed                int, or a CPU generator to continue an existing stream
    on_diverge          "raise" (FloatingPointError as soon as any chain is non-finite) or "nan"
                        (let that chain carry nan; the others are unaffected)

    Returns a ChainRecord whose arrays lead with (C,) for a 1-d `theta0` and (B, C) otherwise.
    """
    if on_diverge not in ("raise", "nan"):
        raise ValueError("on_diverge must be 'raise' or 'nan'")
    device, dtype = problem.device, problem.dtype
    g = seed if isinstance(seed, torch.Generator) else torch.Generator(device="cpu").manual_seed(int(seed))
    single = theta0.dim() == 1
    centres = theta0.to(device=device, dtype=dtype).reshape(-1, 1, theta0.shape[-1])  # (B, 1, D)
    B, _, D = centres.shape
    nb, gam, ep = (_per_centre(x, B, device, dtype) for x in (nbeta, gamma, eps))
    half_eps, sqrt_eps = (0.5 * ep).unsqueeze(-1), ep.sqrt().unsqueeze(-1)  # (B, 1, 1)
    nb3, gam3 = nb.unsqueeze(-1), gam.unsqueeze(-1)

    theta = centres.repeat(1, n_chains, 1)  # (B, C, D)
    if init_jitter > 0:
        theta = theta + init_jitter * torch.randn(B, n_chains, D, generator=g, dtype=dtype).to(device)
    with torch.no_grad():
        L0 = problem.loss(centres).double()  # (B, 1)
    total = n_burnin + n_draws
    rec = torch.empty(B, n_chains, total, dtype=torch.float64, device=device)
    rec_mb = torch.empty(B, n_chains, total, dtype=torch.float64, device=device) if batch_size is not None else None
    for t in range(total):
        idx = None if batch_size is None else problem.minibatch_indices(batch_size, g)
        loss_mb, grad = problem.loss_and_grad(theta, idx)
        with torch.no_grad():
            drift = nb3 * grad + gam3 * (theta - centres)
            noise = torch.randn(B, n_chains, D, generator=g, dtype=dtype).to(device)
            theta = theta - half_eps * drift + sqrt_eps * noise
            rec[:, :, t] = problem.loss(theta).double() - L0
            if rec_mb is not None:
                rec_mb[:, :, t] = loss_mb.double()
        if on_diverge == "raise" and not torch.isfinite(theta).all():
            raise FloatingPointError("SGLD diverged; reduce eps")

    def out(x: torch.Tensor) -> np.ndarray:
        a = x.cpu().numpy()
        return a[0] if single else a

    extras = {"L0": float(L0[0, 0]) if single else L0[:, 0].cpu().numpy()}
    if rec_mb is not None:
        extras["minibatch_loss"] = out(rec_mb[:, :, n_burnin:])

    def param(x) -> np.ndarray:
        a = np.broadcast_to(np.asarray(x, dtype=np.float64), (B,))
        return np.asarray(a[0]) if single else a.reshape(B, 1).copy()

    g_out = param(gamma)
    return ChainRecord(
        delta_loss=out(rec[:, :, n_burnin:]),
        delta_loss_burnin=out(rec[:, :, :n_burnin]),
        nbeta=param(nbeta),
        gamma=float(g_out) if single else g_out,
        eps=param(eps),
        extras=extras,
    )
