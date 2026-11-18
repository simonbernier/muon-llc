"""Block Gibbs sampler for the localised, tempered DLN posterior: a check on MALA that has no
step size and no rejection.

Target, the same as `slt.mala`:  U(w) = nβ · ½‖W_L ⋯ W_1 − A‖² + γ/2 ‖w − w0‖².

Given the other layers, write the product as P W_l Q with P = W_L ⋯ W_{l+1} and
Q = W_{l−1} ⋯ W_1. U is quadratic in W_l with precision  X ↦ nβ PᵀP X QQᵀ + γ X, so with
PᵀP = E_p diag(μ) E_pᵀ and QQᵀ = E_q diag(ν) E_qᵀ the rotated layer E_pᵀ W_l E_q has independent
Gaussian entries with precision nβ μ_i ν_j + γ and mean
(E_pᵀ (nβ Pᵀ A Qᵀ + γ W_l⁰) E_q)_ij / (nβ μ_i ν_j + γ). One sweep draws every layer once, exactly.

This needs the loss to be exactly ½‖W − A‖² (whitened inputs, noiseless targets, unit noise
variance), which is what `DLNProblem.loss` evaluates, and γ > 0.
"""

from __future__ import annotations

import numpy as np
import torch

from ..slt.sgld import ChainRecord
from .model import DLNProblem, flatten, layer_shapes, product, unflatten


def _chain(mats, eye):
    """Product of a list of batched matrices, later layers on the left; identity if empty."""
    return product(mats) if mats else eye


def run_gibbs(
    problem: DLNProblem,
    theta0: torch.Tensor,
    nbetas,
    gamma,
    n_chains: int,
    n_burnin: int,
    n_draws: int,
    seed: int | torch.Generator,
    init_jitter=0.0,
    thin: int = 1,
) -> ChainRecord:
    """Gibbs sweeps around `theta0` (D,) or (B, D) on the ladder `nbetas` (K,).

    Arguments and the returned ChainRecord follow `slt.mala.run_mala`; a "step" is one sweep over
    the layers. `problem.A_star` may carry leading dims (B, 1, 1) for one true map per centre.
    """
    if problem.noise_var != 1.0:
        raise ValueError("the Gibbs conditionals assume unit noise variance")
    # always double precision: the eigendecompositions fail to converge in single precision on
    # the nearly rank-deficient Gram matrices of a low-rank point
    problem = problem.to(dtype=torch.float64)
    device, dtype, widths = problem.device, problem.dtype, problem.widths
    g = seed if isinstance(seed, torch.Generator) else torch.Generator(device="cpu").manual_seed(int(seed))
    single = theta0.dim() == 1
    nbetas_t = torch.as_tensor(np.atleast_1d(np.asarray(nbetas, dtype=np.float64)))
    K = nbetas_t.numel()
    centres = theta0.to(device=device, dtype=dtype).reshape(-1, 1, 1, theta0.shape[-1])
    B, D = centres.shape[0], centres.shape[-1]
    gam_np = np.broadcast_to(np.asarray(gamma, dtype=np.float64), (B,))
    if (gam_np <= 0).any():
        raise ValueError("Gibbs needs gamma > 0")
    nbeta = nbetas_t.to(device, dtype).reshape(1, K, 1, 1, 1)
    gam = torch.as_tensor(gam_np.copy()).to(device, dtype).reshape(B, 1, 1, 1, 1)
    A = problem.A_star
    shapes = layer_shapes(widths)
    L = len(shapes)

    theta = centres.repeat(1, K, n_chains, 1)
    jit = np.broadcast_to(np.asarray(init_jitter, dtype=np.float64), (B,))
    if (jit > 0).any():
        sd = torch.as_tensor(jit.copy()).to(device, dtype).reshape(B, 1, 1, 1)
        theta = theta + sd * torch.randn(B, K, n_chains, D, generator=g, dtype=dtype).to(device)
    ws = unflatten(theta, widths)
    w0 = unflatten(centres, widths)
    eye_out = torch.eye(widths[-1], device=device, dtype=dtype)
    eye_in = torch.eye(widths[0], device=device, dtype=dtype)

    thin = int(thin)
    if thin < 1 or n_burnin % thin or n_draws % thin:
        raise ValueError("n_burnin and n_draws must be multiples of thin")
    total = n_burnin + n_draws
    rec = torch.empty(B, K, n_chains, total // thin, dtype=torch.float64, device=device)
    with torch.no_grad():
        L0 = problem.loss(centres).double()
        for t in range(total):
            noise = unflatten(torch.randn(B, K, n_chains, D, generator=g, dtype=dtype).to(device), widths)
            for l in range(L):
                P = _chain(ws[l + 1:], eye_out)  # (..., H_L, H_{l+1})
                Q = _chain(ws[:l], eye_in)  # (..., H_l, H_0)
                mu, Ep = torch.linalg.eigh(P.mT @ P)
                nu, Eq = torch.linalg.eigh(Q @ Q.mT)
                prec = nbeta * mu.clamp_min(0).unsqueeze(-1) * nu.clamp_min(0).unsqueeze(-2) + gam
                lin = nbeta * (P.mT @ A @ Q.mT) + gam * w0[l]
                rot = (Ep.mT @ lin @ Eq) / prec + noise[l] / prec.sqrt()
                ws[l] = Ep @ rot @ Eq.mT
            if (t + 1) % thin == 0:
                rec[..., (t + 1) // thin - 1] = problem.loss_weights(ws).double() - L0
            if not all(torch.isfinite(w).all() for w in ws):
                raise FloatingPointError("Gibbs produced a non-finite state")

    def out(x: torch.Tensor) -> np.ndarray:
        a = x.double().cpu().numpy()
        return a[0] if single else a

    return ChainRecord(
        delta_loss=out(rec[..., n_burnin // thin:]),
        delta_loss_burnin=out(rec[..., :n_burnin // thin]),
        nbeta=nbetas_t.numpy().reshape(K, 1),
        gamma=float(gam_np[0]) if single else gam_np.copy(),
        eps=np.full(K if single else (B, K), np.nan),
        acceptance=np.ones((K, n_chains) if single else (B, K, n_chains)),
        extras={"L0": float(L0.reshape(-1)[0]) if single else L0.reshape(B).cpu().numpy(), "thin": thin,
                "theta": flatten(ws).cpu().numpy()},
    )
