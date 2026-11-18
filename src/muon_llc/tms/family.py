"""The family between the toy model of superposition and a tied linear network.

    f(x) = leaky_slope(W^T W x + b),   loss = mean_n sum_j rho^j (x_j - f_j(x))^2

A scan is a grid over the parameters below; every grid cell is trained with the same seeds, and
all cells of a scan train in one batch (cells x seeds runs in one tensor) under one optimizer.

    slope         activation slope for negative arguments (0 = ReLU, the paper; 1 = linear)
    rho           importance of output j is rho^j (1 = the paper's loss)
    init_length   length of the initial k-gon (r = 2 only; 0 = no structure in the initial point)
    init_noise    standard deviation of the Gaussian added to the initial W
    bias_std      standard deviation of the initial bias

Run (cell, seed) has index cell * seeds + seed. With the same seed every cell has the same
dataset, the same k-gon rotation and the same Gaussian draws for W and b (scaled by the cell's
values). The minibatch order is drawn per run and differs between cells.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import torch

from .model import TMSProblem, init_kgon, sample_data, split

GRID_KEYS = ("slope", "rho", "init_length", "init_noise", "bias_std")
GRID_DEFAULTS = {"slope": 0.0, "rho": 1.0, "init_length": 0.9, "init_noise": 0.01, "bias_std": 1.0}


def cells_of(cfg: dict) -> pd.DataFrame:
    """One row per grid cell, columns GRID_KEYS, in the order of itertools.product."""
    grid = {k: list(np.atleast_1d(cfg.get("grid", {}).get(k, GRID_DEFAULTS[k]))) for k in GRID_KEYS}
    rows = list(itertools.product(*grid.values()))
    return pd.DataFrame(rows, columns=list(GRID_KEYS)).astype(float)


def initial_points(cfg: dict, cells: pd.DataFrame) -> torch.Tensor:
    """(cells * seeds, r*c + c). r = 2: the reference k-gon init. Other r: Gaussian W (init_length
    must be 0, there is no k-gon), with the same draws for every cell."""
    r, c, seeds, seed = int(cfg["r"]), int(cfg["c"]), int(cfg["seeds"]), int(cfg.get("seed", 0))
    out = []
    for cell in cells.itertuples(index=False):
        if r == 2:
            out.append(init_kgon(seeds, c, int(cfg.get("init_k", 4)), cell.init_length, cell.init_noise,
                                 cell.bias_std, seed))
            continue
        if cell.init_length != 0:
            raise ValueError("init_length > 0 needs r = 2 (the k-gon lives in the plane)")
        th = np.empty((seeds, r * c + c))
        for j in range(seeds):
            rng = np.random.default_rng(seed + j)
            th[j] = np.concatenate([cell.init_noise * rng.normal(size=r * c), cell.bias_std * rng.normal(size=c)])
        out.append(torch.tensor(th, dtype=torch.float64))
    return torch.cat(out)


def family_problem(cfg: dict, cells: pd.DataFrame, device=None) -> TMSProblem:
    """One dataset per seed, repeated over cells; per-run slope and importance."""
    r, c, seeds, n = int(cfg["r"]), int(cfg["c"]), int(cfg["seeds"]), int(cfg["n"])
    pairs = [sample_data(n, c, torch.Generator().manual_seed(int(cfg.get("data_seed", 1000)) + j))
             for j in range(seeds)]
    feat = torch.stack([p[0] for p in pairs]).repeat(len(cells), 1)
    mu = torch.stack([p[1] for p in pairs]).repeat(len(cells), 1)
    slope = torch.tensor(np.repeat(cells["slope"].to_numpy(), seeds))
    imp = torch.tensor(np.repeat(cells["rho"].to_numpy()[:, None] ** np.arange(c)[None, :], seeds, axis=0))
    problem = TMSProblem(feat, mu, r, c, 1.0, slope, imp)
    return problem if device is None else problem.to(device)


@torch.no_grad()
def feature_losses(problem: TMSProblem, thetas: np.ndarray, chunk: int = 64) -> np.ndarray:
    """Loss split by which feature the input carries: (K, R, c), summing to the loss over the last
    axis. thetas (K, R, D) are the checkpoints of the runs of `problem` (one dataset per run)."""
    r, c = problem.r, problem.c
    K = thetas.shape[0]
    onehot = torch.nn.functional.one_hot(problem.feat, c).to(problem.dtype)  # (R, n, c)
    out = np.empty((K, thetas.shape[1], c))
    slope = problem.slope[:, None, None] if isinstance(problem.slope, torch.Tensor) else problem.slope
    imp = problem.importance
    imp = 1.0 if imp is None else (imp.unsqueeze(-2) if imp.dim() == 2 else imp)
    for s in range(0, K, chunk):
        th = torch.as_tensor(thetas[s:s + chunk], dtype=problem.dtype, device=problem.device)
        W, b = split(th, r, c)  # (k, R, r, c), (k, R, c)
        G = W.mT @ W
        idx = problem.feat.unsqueeze(-1).expand(*problem.feat.shape, c).expand(th.shape[0], -1, -1, -1)
        z = problem.mu.unsqueeze(-1) * torch.gather(G, -2, idx) + b.unsqueeze(-2)
        f = torch.where(z > 0, z, slope * z)
        sq = (imp * (onehot * problem.mu.unsqueeze(-1) - f) ** 2).sum(-1)  # (k, R, n)
        out[s:s + chunk] = (torch.einsum("krn,rnj->krj", sq, onehot) / problem.n).cpu().numpy()
    return out


def unlearned_level(problem: TMSProblem) -> np.ndarray:
    """(R, c): the loss feature j contributes while the model outputs zero on it,
    importance_j * sum of mu^2 over the inputs carrying j, divided by n."""
    c = problem.c
    onehot = torch.nn.functional.one_hot(problem.feat, c).to(problem.dtype)
    lev = torch.einsum("rn,rnj->rj", problem.mu ** 2, onehot) / problem.n
    if problem.importance is not None:
        lev = lev * problem.importance
    return lev.cpu().numpy()


def learning_times(epochs: np.ndarray, feat_loss: np.ndarray, unlearned: np.ndarray,
                   learned_frac: float = 0.5) -> np.ndarray:
    """(R, c) epoch from which feature j stays learned: the first checkpoint after the last one
    where its loss was above `learned_frac` of its unlearned level. 0 if it never was above,
    nan if it is above at the end (not learned)."""
    above = feat_loss > learned_frac * unlearned[None]  # (K, R, c)
    K = above.shape[0]
    last = K - 1 - np.argmax(above[::-1], axis=0)  # index of the last checkpoint above
    never = ~above.any(axis=0)
    t = np.where(last + 1 < K, epochs[np.minimum(last + 1, K - 1)], np.nan).astype(float)
    t[never] = 0.0
    return t


def state_labels(thetas: np.ndarray, r: int, c: int, act_thresh: float = 0.25, pos_thresh: float = 0.02) -> np.ndarray:
    """(K, R) labels "<A>a<S>p": A features represented (|W_j|^2 + b_j > act_thresh, the test of
    `critical_points.classify`) and S unrepresented features with a positive bias. For r = 2,
    c = 6 the paper's k^{sigma+, phi-} has A = k - phi and S = sigma.
    Unlike `classify` there is no test against a loss level, so the label exists for every r,
    importance and slope; a plateau is a stretch of constant label (`transitions.plateaus`)."""
    th = np.asarray(thetas, dtype=np.float64)
    W = th[..., : r * c].reshape(*th.shape[:-1], r, c)
    b = th[..., r * c:]
    active = (W ** 2).sum(-2) + b > act_thresh
    positive = ~active & (b > pos_thresh)
    A, S = active.sum(-1), positive.sum(-1)
    return np.char.add(np.char.add(A.astype(str), "a"), np.char.add(S.astype(str), "p")).astype(object)
