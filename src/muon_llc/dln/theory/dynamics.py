"""A4: predicted singular-value trajectories and λ(t) for GD vs spectral GD.

GD (aligned / balanced idealisation, Saxe et al.; Jacot et al. for the saddle-to-saddle
limit). Each target singular value s_i evolves independently through a depth-L scalar
product a^L with all factors equal, a(0) = alpha:
      a <- a - lr * (a^L - s_i) * a^{L-1}          (one GD step on ½ (a^L - s_i)²)
For small alpha and L >= 3 the escape time from the origin scales like
alpha^{2-L} / ((L-2) * lr * s_i), so singular values are learned one at a time in
decreasing order: a staircase in the product rank and hence in λ.

Spectral GD (Dragutinović & Ranganath): the update has unit spectral norm along every
singular direction of the gradient, so every layer singular value grows by lr per step
until the product saturates:
      s_i(t) = min((alpha + lr t)^L, s_i)
All s_i cross any threshold at the same time -> a single jump in λ.

λ_pred(t) = λ_Aoyagi(r(t)) with r(t) = #{ i : s_i(t) > threshold }. The threshold is the
resolution of the estimator: a direction whose Hessian eigenvalue h satisfies
nβ h << γ is invisible (A1), and the product-singular-value Hessian scale is derived in
theory/hessian.py numerically rather than assumed.
"""

from __future__ import annotations

import numpy as np

from .aoyagi import dln_learning_coefficient


def gd_aligned_trajectory(singular_values, depth: int, alpha: float, lr: float, steps: int,
                          record_every: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Return (t, s) with s of shape (T, len(singular_values)) = predicted product singular values."""
    s_t = np.asarray(singular_values, dtype=np.float64)
    a = np.full_like(s_t, float(alpha))
    L = int(depth)
    ts, out = [], []
    for t in range(int(steps) + 1):
        if t % record_every == 0:
            ts.append(t)
            out.append(a**L)
        prod = a**L
        a = a - lr * (prod - s_t) * a ** (L - 1)
        a = np.clip(a, 0.0, None)
    return np.array(ts), np.array(out)


def gd_escape_time_estimate(s: float, depth: int, alpha: float, lr: float) -> float:
    """Leading-order time (in steps) for the aligned scalar to escape the origin saddle.

    Small-a limit of da/dt = lr s a^{L-1} (each of the L factors gets the same gradient):
    L = 2: (1/(2 lr s)) log(s / alpha^2);   L >= 3: alpha^{2-L} / ((L-2) lr s).
    """
    L = int(depth)
    if L == 2:
        return np.log(s / alpha**2) / (2 * lr * s)
    return alpha ** (2 - L) / ((L - 2) * lr * s)


def spectral_trajectory(singular_values, depth: int, alpha: float, lr: float, steps: int,
                        record_every: int = 1) -> tuple[np.ndarray, np.ndarray]:
    s_t = np.asarray(singular_values, dtype=np.float64)
    ts = np.arange(0, int(steps) + 1, record_every)
    grown = (alpha + lr * ts[:, None]) ** int(depth)
    return ts, np.minimum(grown, s_t[None, :])


def spectral_saturation_time(s: float, depth: int, alpha: float, lr: float) -> float:
    return (s ** (1.0 / depth) - alpha) / lr


def crossing_times(ts: np.ndarray, s: np.ndarray, threshold: float) -> np.ndarray:
    """First recorded time each column of s exceeds threshold (nan if never)."""
    out = np.full(s.shape[1], np.nan)
    for i in range(s.shape[1]):
        idx = np.nonzero(s[:, i] > threshold)[0]
        if idx.size:
            out[i] = ts[idx[0]]
    return out


def effective_rank(s: np.ndarray, threshold: float) -> np.ndarray:
    return (np.asarray(s) > threshold).sum(axis=-1)


def predicted_lambda_curve(ts: np.ndarray, s: np.ndarray, threshold: float, widths) -> np.ndarray:
    r = effective_rank(s, threshold)
    table = {k: dln_learning_coefficient(widths, k).value for k in range(int(r.max()) + 1)}
    return np.array([table[int(k)] for k in r])
