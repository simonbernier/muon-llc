"""A1 soft-rank prediction from the exact Hessian at a checkpoint.

For a quadratic direction with Hessian eigenvalue h under the localised tempered posterior
the contribution to nβ E[ΔL] is ½ · nβh / (nβh + γ). Summing over positive eigenvalues gives
λ_quad(γ/nβ), the "half the soft rank of the Hessian at resolution γ/nβ" baseline. The count
of negative eigenvalues is a saddle diagnostic (checkpoints along a trajectory are not minima).
The DLN has ~200 parameters, so torch.func.hessian is exact and cheap.
"""

from __future__ import annotations

import numpy as np
import torch

from ..model import DLNProblem


def hessian_eigenvalues(problem: DLNProblem, theta0: torch.Tensor) -> np.ndarray:
    theta0 = theta0.to(device=problem.device, dtype=torch.float64).reshape(-1)
    prob64 = problem.to(dtype=torch.float64)

    def f(th):
        return prob64.loss(th.reshape(1, -1))[0]

    H = torch.func.hessian(f)(theta0)
    H = 0.5 * (H + H.T)
    return torch.linalg.eigvalsh(H).cpu().numpy()


def lambda_quad(eigs: np.ndarray, nbeta: float, gamma: float) -> float:
    h = np.asarray(eigs, dtype=np.float64)
    h = h[h > 0]
    return float(0.5 * np.sum(nbeta * h / (nbeta * h + gamma)))


def hessian_summary(eigs: np.ndarray, nbeta: float, gamma: float, tol: float = 1e-10) -> dict:
    eigs = np.asarray(eigs, dtype=np.float64)
    scale = max(np.abs(eigs).max(), 1e-30)
    return dict(
        lambda_quad=lambda_quad(eigs, nbeta, gamma),
        n_neg_hessian=int((eigs < -tol * scale).sum()),
        n_pos_hessian=int((eigs > tol * scale).sum()),
        h_max=float(eigs.max()),
        h_min=float(eigs.min()),
        # eigenvalue at which a direction is half-resolved: nβh = γ
        resolution_h=float(gamma / nbeta),
    )
