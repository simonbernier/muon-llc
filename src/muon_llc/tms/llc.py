"""Local learning coefficient estimates for TMS with SGLD (devinterp / Lau et al. update rule),
vectorised over many centres at once.

    w <- w - (eps/2) [ n beta grad L_n(w) + gamma (w - w0) ] + sqrt(eps) xi
    lambda_hat(w0) = n beta ( mean_t L_n(w_t) - L_n(w0) ),   n beta = n / log n by default

L_n is the paper's loss (mean of the squared reconstruction error), whatever scale was used for
training. `frac_below[b, c]` is the share of draws of chain c with loss below the centre's loss,
the paper's health test (Appendix K: discard chains with more than 5 %).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .model import TMSProblem, split, tms_loss


@dataclass
class LLCResult:
    llc_chain: np.ndarray  # (B, C)
    frac_below: np.ndarray  # (B, C)
    loss0: np.ndarray  # (B,)
    nbeta: float

    @property
    def llc(self) -> np.ndarray:
        return self.llc_chain.mean(1)

    def healthy(self, max_frac: float = 0.05):
        """Mean, std and count over chains passing the escape test (nan if none pass)."""
        ok = self.frac_below <= max_frac
        cnt = ok.sum(1)
        x = np.where(ok, self.llc_chain, np.nan)
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return np.nanmean(x, 1), np.nanstd(x, 1), cnt


def sgld_llc(problem: TMSProblem, centres, eps: float = 1e-3, gamma: float = 1.0,
             n_steps: int = 500, n_burnin: int = 0, n_chains: int = 1, nbeta: float | None = None,
             batch_size: int | None = None, seed: int = 0, chunk: int = 2048,
             below_tol: float = 0.0, data_index=None) -> LLCResult:
    """centres (B, D). With per-run datasets, `data_index` (B,) says which dataset each centre uses."""
    centres = torch.as_tensor(np.asarray(centres), dtype=problem.dtype).reshape(-1, problem.dim)
    n, r, c = problem.n, problem.r, problem.c
    nb = float(n / np.log(n)) if nbeta is None else float(nbeta)
    g = torch.Generator().manual_seed(int(seed))
    out_llc, out_fb, out_l0 = [], [], []

    if problem.per_run and data_index is None:
        raise ValueError("per-run datasets need data_index")
    data_index = None if data_index is None else torch.as_tensor(np.asarray(data_index), dtype=torch.long)

    for start in range(0, centres.shape[0], chunk):
        w0 = centres[start:start + chunk].unsqueeze(1)  # (B, 1, D)
        B = w0.shape[0]
        if problem.per_run:
            di = data_index[start:start + chunk]
            feat, mu = problem.feat[di].unsqueeze(1), problem.mu[di].unsqueeze(1)  # (B, 1, n)
        else:
            feat, mu = problem.feat, problem.mu

        def full_loss(theta, feat=feat, mu=mu):
            W, b = split(theta, r, c)
            return tms_loss(W, b, feat, mu)

        with torch.no_grad():
            L0 = full_loss(w0)  # (B, 1)
        theta = w0.repeat(1, n_chains, 1)
        acc = torch.zeros(B, n_chains, dtype=problem.dtype)
        below = torch.zeros(B, n_chains, dtype=problem.dtype)
        for t in range(n_steps):
            theta.requires_grad_(True)
            W, b = split(theta, r, c)
            if batch_size is None:
                loss = tms_loss(W, b, feat, mu)
            else:
                idx = torch.randint(0, n, (batch_size,), generator=g)
                loss = tms_loss(W, b, feat[..., idx], mu[..., idx])
            (grad,) = torch.autograd.grad(loss.sum(), theta)
            with torch.no_grad():
                theta = theta - 0.5 * eps * (nb * grad + gamma * (theta - w0)) \
                    + eps**0.5 * torch.randn(theta.shape, generator=g, dtype=problem.dtype)
                if t >= n_burnin:
                    Lt = full_loss(theta)
                    acc += Lt
                    below += (Lt < L0 - below_tol).to(problem.dtype)
        T = n_steps - n_burnin
        out_llc.append((nb * (acc / T - L0)).numpy())
        out_fb.append((below / T).numpy())
        out_l0.append(L0.squeeze(1).numpy())
    return LLCResult(np.concatenate(out_llc), np.concatenate(out_fb), np.concatenate(out_l0), nb)
