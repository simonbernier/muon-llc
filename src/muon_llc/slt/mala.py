"""Full-batch Metropolis-adjusted Langevin (MALA) on a ladder of inverse temperatures.

Target for ladder rung k:  U_k(w) = nβ_k L_n(w) + γ/2 ||w - w0||².
Proposal:                   w' = w - (eps_k/2) ∇U_k(w) + sqrt(eps_k) ξ
Accept with probability     min(1, exp(-U(w') + U(w) + log q(w|w') - log q(w'|w))).

Chains are one tensor of shape (B, K, C, D): B centres, K rungs, C chains. γ is a scalar or one
value per centre, so every checkpoint and every γ of a resolution sweep can share a call; on a
GPU the cost of a step barely depends on B·K·C. eps is adapted per (centre, rung) during burn-in
toward a target acceptance rate (default 0.574, the optimal MALA rate) by Robbins-Monro on log eps.
The step size is frozen after burn-in. If the chains are still moving away from the centre at that
point, the frozen step is tuned for the wrong region and the later acceptance rate drifts from the
target: compare `acceptance` with `extras["accept_burnin_tail"]`.
The Metropolis step removes the discretisation bias of unadjusted Langevin and the
full batch removes minibatch noise, so this is the reference estimator of Phase B.
"""

from __future__ import annotations

import numpy as np
import torch

from .problem import Problem
from .sgld import ChainRecord


def _potential_and_grad(problem, theta, nbeta, gamma, theta0):
    """theta (B, K, C, D); nbeta (1, K, 1); gamma (B, 1, 1). Returns loss, U (B, K, C) and ∇U."""
    loss, grad_loss = problem.loss_and_grad(theta)
    diff = theta - theta0
    U = nbeta * loss + 0.5 * gamma * (diff * diff).sum(-1)
    gradU = nbeta.unsqueeze(-1) * grad_loss + gamma.unsqueeze(-1) * diff
    return loss.detach(), U.detach(), gradU.detach()


def run_mala(
    problem: Problem,
    theta0: torch.Tensor,
    nbetas,
    gamma,
    eps0,
    n_chains: int,
    n_burnin: int,
    n_draws: int,
    seed: int | torch.Generator,
    target_accept: float = 0.574,
    adapt_every: int = 20,
    adapt_rate: float = 0.5,
    init_jitter=0.0,
    adapt_decay: float = 0.0,
    thin: int = 1,
) -> ChainRecord:
    """MALA around `theta0`: (D,) for one centre or (B, D) for B centres, on the ladder `nbetas` (K,).

    gamma   scalar, or one value per centre
    eps0    initial step size: scalar, (K,) per rung, or (B, K)
    init_jitter   std of a Gaussian offset of each chain's start from the centre: scalar or one
                  value per centre (0 = every chain starts at the centre)
    adapt_decay   the i-th adaptation uses gain adapt_rate / (1 + i)^adapt_decay; 0 keeps the gain
                  constant, 0.5 lets the step size settle
    thin    record the loss every `thin` steps (burn-in and draws must be multiples of it)

    Returns a ChainRecord whose arrays lead with (K, C) for a 1-d `theta0` and (B, K, C) otherwise;
    `eps` is (K,) or (B, K), `acceptance` the post-burn-in rate per chain.
    """
    device, dtype = problem.device, problem.dtype
    g = seed if isinstance(seed, torch.Generator) else torch.Generator(device="cpu").manual_seed(int(seed))
    single = theta0.dim() == 1
    nbetas_t = torch.as_tensor(np.atleast_1d(np.asarray(nbetas, dtype=np.float64)))
    K = nbetas_t.numel()
    centres = theta0.to(device=device, dtype=dtype).reshape(-1, 1, 1, theta0.shape[-1])  # (B, 1, 1, D)
    B, D = centres.shape[0], centres.shape[-1]
    nbeta = nbetas_t.to(device, dtype).reshape(1, K, 1)
    gam = torch.as_tensor(np.broadcast_to(np.asarray(gamma, dtype=np.float64), (B,)).copy()).to(
        device, dtype).reshape(B, 1, 1)
    eps = torch.as_tensor(np.broadcast_to(np.asarray(eps0, dtype=np.float64), (B, K)).copy()).to(
        device, dtype).reshape(B, K, 1)
    theta = centres.repeat(1, K, n_chains, 1)
    jit = np.broadcast_to(np.asarray(init_jitter, dtype=np.float64), (B,))
    if (jit > 0).any():
        sd = torch.as_tensor(jit.copy()).to(device, dtype).reshape(B, 1, 1, 1)
        theta = theta + sd * torch.randn(B, K, n_chains, D, generator=g, dtype=dtype).to(device)
    with torch.no_grad():
        L0 = problem.loss(centres).double()  # (B, 1, 1)

    thin = int(thin)
    if thin < 1 or n_burnin % thin or n_draws % thin:
        raise ValueError("n_burnin and n_draws must be multiples of thin")
    total = n_burnin + n_draws
    rec = torch.empty(B, K, n_chains, total // thin, dtype=torch.float64, device=device)
    n_acc = torch.zeros(B, K, n_chains, dtype=torch.float64, device=device)
    tail_start = n_burnin - max(1, n_burnin // 10)
    acc_tail = torch.zeros(B, K, dtype=torch.float64, device=device)
    n_adapt = 0

    loss, U, gradU = _potential_and_grad(problem, theta, nbeta, gam, centres)
    acc_window = torch.zeros(B, K, device=device, dtype=dtype)
    for t in range(total):
        with torch.no_grad():
            noise = torch.randn(B, K, n_chains, D, generator=g, dtype=dtype).to(device)
            mean_fwd = theta - 0.5 * eps.unsqueeze(-1) * gradU
            prop = mean_fwd + eps.sqrt().unsqueeze(-1) * noise
        loss_p, U_p, gradU_p = _potential_and_grad(problem, prop, nbeta, gam, centres)
        with torch.no_grad():
            mean_bwd = prop - 0.5 * eps.unsqueeze(-1) * gradU_p
            # log q(w'|w) = -||w' - mean_fwd||^2 / (2 eps); log q(w|w') = -||w - mean_bwd||^2 / (2 eps)
            log_q_fwd = -((prop - mean_fwd) ** 2).sum(-1) / (2 * eps)
            log_q_bwd = -((theta - mean_bwd) ** 2).sum(-1) / (2 * eps)
            log_alpha = -U_p + U + log_q_bwd - log_q_fwd
            u = torch.rand(B, K, n_chains, generator=g, dtype=dtype).to(device)
            accept = torch.log(u) < log_alpha
            a3 = accept.unsqueeze(-1)
            theta = torch.where(a3, prop, theta)
            gradU = torch.where(a3, gradU_p, gradU)
            U = torch.where(accept, U_p, U)
            loss = torch.where(accept, loss_p, loss)
            if (t + 1) % thin == 0:
                rec[..., (t + 1) // thin - 1] = loss.double() - L0
            if t >= n_burnin:
                n_acc += accept
            elif t >= tail_start:
                acc_tail += accept.double().mean(dim=-1)
            acc_window += accept.to(dtype).mean(dim=-1)
            if t < n_burnin and (t + 1) % adapt_every == 0:
                rate = acc_window / adapt_every
                gain = adapt_rate / (1.0 + n_adapt) ** adapt_decay
                eps = eps * torch.exp(gain * (rate - target_accept)).unsqueeze(-1)
                acc_window.zero_()
                n_adapt += 1
        if not torch.isfinite(theta).all():
            raise FloatingPointError("MALA diverged")

    def out(x: torch.Tensor) -> np.ndarray:
        a = x.double().cpu().numpy()
        return a[0] if single else a

    g_out = np.broadcast_to(np.asarray(gamma, dtype=np.float64), (B,))
    return ChainRecord(
        delta_loss=out(rec[..., n_burnin // thin:]),
        delta_loss_burnin=out(rec[..., :n_burnin // thin]),
        nbeta=nbetas_t.numpy().reshape(K, 1),
        gamma=float(g_out[0]) if single else g_out.copy(),
        eps=out(eps.reshape(B, K)),
        acceptance=out(n_acc / max(n_draws, 1)),
        extras={"L0": float(L0.reshape(-1)[0]) if single else L0.reshape(B).cpu().numpy(),
                "accept_burnin_tail": out(acc_tail / max(1, n_burnin - tail_start)), "thin": thin},
    )
