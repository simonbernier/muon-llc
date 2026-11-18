"""Local learning coefficient estimates for TMS: the shared SGLD sampler (`muon_llc.slt.sgld`,
devinterp / Lau et al. update rule) run around many centres at once.

    lambda_hat(w0) = n beta ( mean_t L_n(w_t) - L_n(w0) ),   n beta = n / log n by default

L_n is the paper's loss (mean of the squared reconstruction error), whatever scale was used for
training. `frac_below[b, c]` is the share of draws of chain c with loss below the centre's loss,
the paper's health test (Appendix K: discard chains with more than 5 %).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import torch

from ..slt.sgld import run_sgld
from .model import TMSProblem


@dataclass
class LLCResult:
    llc_chain: np.ndarray  # (B, C) per-chain estimates
    frac_below: np.ndarray  # (B, C)
    loss0: np.ndarray  # (B,)
    nbeta: float
    var_over_mean: np.ndarray | None = None  # (B,) Var(nβ ΔL) / E[nβ ΔL] over all draws; ≈ 1 if scale-free

    @property
    def llc(self) -> np.ndarray:
        return self.llc_chain.mean(1)

    @property
    def sem(self) -> np.ndarray:
        """Standard error over chains (nan with a single chain)."""
        C = self.llc_chain.shape[1]
        return self.llc_chain.std(1, ddof=1) / np.sqrt(C) if C > 1 else np.full(self.llc_chain.shape[0], np.nan)

    def healthy(self, max_frac: float = 0.05):
        """Mean, std and count over chains passing the escape test (nan if none pass)."""
        ok = self.frac_below <= max_frac
        cnt = ok.sum(1)
        x = np.where(ok, self.llc_chain, np.nan)
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.nanmean(x, 1), np.nanstd(x, 1), cnt


def sgld_llc(problem: TMSProblem, centres, eps: float = 1e-3, gamma: float = 1.0,
             n_steps: int = 500, n_burnin: int = 0, n_chains: int = 1, nbeta: float | None = None,
             batch_size: int | None = None, seed: int = 0, chunk: int = 2048,
             below_tol: float = 0.0, data_index=None) -> LLCResult:
    """centres (B, D). With per-run datasets, `data_index` (B,) says which dataset each centre uses.

    `n_steps` counts burn-in; `chunk` bounds the number of centres held in memory at once. A chain
    that diverges carries nan and fails the health test instead of aborting the batch.
    """
    centres = torch.as_tensor(np.asarray(centres), dtype=problem.dtype).reshape(-1, problem.dim)
    n, r, c = problem.n, problem.r, problem.c
    nb = float(n / np.log(n)) if nbeta is None else float(nbeta)
    g = torch.Generator().manual_seed(int(seed))
    if problem.per_run and data_index is None:
        raise ValueError("per-run datasets need data_index")
    if data_index is not None:
        data_index = torch.as_tensor(np.asarray(data_index), dtype=torch.long, device=problem.feat.device)

    out_llc, out_fb, out_l0, out_vm = [], [], [], []
    for start in range(0, centres.shape[0], chunk):
        w0 = centres[start:start + chunk]
        if problem.per_run:  # one dataset per centre, broadcast over chains: (B, 1, n)
            di = data_index[start:start + chunk]
            # per-run slope and importance get the same chain axis: (B, 1) and (B, 1, c)
            per = lambda x, nd: x[di].unsqueeze(1) if isinstance(x, torch.Tensor) and x.dim() == nd else x  # noqa: E731
            sub = TMSProblem(problem.feat[di].unsqueeze(1), problem.mu[di].unsqueeze(1), r, c, 1.0,
                             per(problem.slope, 1), per(problem.importance, 2))
        else:
            sub = TMSProblem(problem.feat, problem.mu, r, c, 1.0, problem.slope, problem.importance)
        rec = run_sgld(sub, w0, nb, gamma, eps, n_chains, n_burnin, n_steps - n_burnin, batch_size,
                       seed=g, on_diverge="nan")
        x = nb * rec.delta_loss  # (B, C, T)
        mean = x.mean(axis=(1, 2))
        out_llc.append(x.mean(axis=2))
        out_fb.append(rec.frac_below(below_tol))
        out_l0.append(np.asarray(rec.extras["L0"]))
        with np.errstate(all="ignore"):
            out_vm.append(x.var(axis=(1, 2)) / mean)
    return LLCResult(np.concatenate(out_llc), np.concatenate(out_fb), np.concatenate(out_l0), nb,
                     np.concatenate(out_vm))
