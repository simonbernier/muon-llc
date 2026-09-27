"""Full-batch Metropolis-adjusted Langevin (MALA) on a ladder of inverse temperatures.

Target for ladder rung k:  U_k(w) = nβ_k L_n(w) + γ/2 ||w - w0||².
Proposal:                   w' = w - (eps_k/2) ∇U_k(w) + sqrt(eps_k) ξ
Accept with probability     min(1, exp(-U(w') + U(w) + log q(w|w') - log q(w'|w))).

Chains are a tensor of shape (K, C, D); eps_k is adapted during burn-in toward a target
acceptance rate (default 0.574, the optimal MALA rate) by Robbins-Monro on log eps.
The Metropolis step removes the discretisation bias of unadjusted Langevin and the
full batch removes minibatch noise, so this is the reference estimator of Phase B.
"""

from __future__ import annotations

import numpy as np
import torch

from ..dln import DLNProblem
from .sgld import ChainRecord


def _potential_and_grad(problem, theta, nbeta_col, gamma, theta0):
    loss, grad_loss = problem.loss_and_grad(theta)  # (K, C), (K, C, D)
    diff = theta - theta0
    U = nbeta_col * loss + 0.5 * gamma * (diff * diff).sum(-1)
    gradU = nbeta_col.unsqueeze(-1) * grad_loss + gamma * diff
    return loss.detach(), U.detach(), gradU.detach()


def run_mala(
    problem: DLNProblem,
    theta0: torch.Tensor,
    nbetas,
    gamma: float,
    eps0,
    n_chains: int,
    n_burnin: int,
    n_draws: int,
    seed: int,
    target_accept: float = 0.574,
    adapt_every: int = 20,
    adapt_rate: float = 0.5,
    init_jitter: float = 0.0,
) -> ChainRecord:
    device, dtype = problem.device, problem.dtype
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    nbetas_t = torch.as_tensor(np.atleast_1d(np.asarray(nbetas, dtype=np.float64)))
    K = nbetas_t.numel()
    nbeta_col = nbetas_t.to(device, dtype).reshape(K, 1)
    theta0 = theta0.to(device=device, dtype=dtype).reshape(1, 1, -1)
    D = theta0.shape[-1]
    theta = theta0.repeat(K, n_chains, 1)
    if init_jitter > 0:
        theta = theta + init_jitter * torch.randn(K, n_chains, D, generator=g).to(device, dtype)
    L0 = problem.loss(theta0.reshape(1, -1)).item()

    eps = torch.as_tensor(np.broadcast_to(np.asarray(eps0, dtype=np.float64), (K,)).copy()).to(
        device, dtype
    ).reshape(K, 1)
    total = n_burnin + n_draws
    rec = torch.empty(K, n_chains, total, dtype=torch.float64)
    acc_rec = torch.zeros(K, n_chains, total, dtype=torch.float64)

    loss, U, gradU = _potential_and_grad(problem, theta, nbeta_col, gamma, theta0)
    acc_window = torch.zeros(K, device=device, dtype=dtype)
    for t in range(total):
        with torch.no_grad():
            noise = torch.randn(K, n_chains, D, generator=g).to(device, dtype)
            mean_fwd = theta - 0.5 * eps.unsqueeze(-1) * gradU
            prop = mean_fwd + eps.sqrt().unsqueeze(-1) * noise
        loss_p, U_p, gradU_p = _potential_and_grad(problem, prop, nbeta_col, gamma, theta0)
        with torch.no_grad():
            mean_bwd = prop - 0.5 * eps.unsqueeze(-1) * gradU_p
            # log q(w'|w) = -||w' - mean_fwd||^2 / (2 eps); log q(w|w') = -||w - mean_bwd||^2 / (2 eps)
            log_q_fwd = -((prop - mean_fwd) ** 2).sum(-1) / (2 * eps)
            log_q_bwd = -((theta - mean_bwd) ** 2).sum(-1) / (2 * eps)
            log_alpha = -U_p + U + log_q_bwd - log_q_fwd
            u = torch.rand(K, n_chains, generator=g).to(device, dtype)
            accept = torch.log(u) < log_alpha
            a3 = accept.unsqueeze(-1)
            theta = torch.where(a3, prop, theta)
            gradU = torch.where(a3, gradU_p, gradU)
            U = torch.where(accept, U_p, U)
            loss = torch.where(accept, loss_p, loss)
            rec[:, :, t] = (loss - L0).double().cpu()
            acc_rec[:, :, t] = accept.double().cpu()
            acc_window += accept.to(dtype).mean(dim=1)
            if t < n_burnin and (t + 1) % adapt_every == 0:
                rate = acc_window / adapt_every
                eps = eps * torch.exp(adapt_rate * (rate - target_accept)).reshape(K, 1)
                acc_window.zero_()
        if not torch.isfinite(theta).all():
            raise FloatingPointError("MALA diverged")
    return ChainRecord(
        delta_loss=rec[:, :, n_burnin:].numpy(),
        delta_loss_burnin=rec[:, :, :n_burnin].numpy(),
        nbeta=nbetas_t.numpy().reshape(K, 1),
        gamma=float(gamma),
        eps=eps.reshape(K).double().cpu().numpy(),
        acceptance=acc_rec[:, :, n_burnin:].mean(-1).numpy(),
        extras={"L0": L0},
    )
