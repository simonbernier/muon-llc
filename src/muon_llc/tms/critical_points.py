"""Known critical points of the TMS potential for r = 2 (the k-gons of the paper) and a phase
classifier for arbitrary parameters.

Naming. `k` standard k-gon; a `+` per positive bias (value 1/(2c)) on a zero column; a `-` per
4-gon vertex killed by a large negative bias. For c = 6 this gives the paper's 18 families:
6, 5, 5+, and 4 with sigma in {0,1,2} pluses and phi in {0..4} minuses.

Losses. L(4^{sigma+, phi-}) = (c - 4 + phi)/(3c) - sigma/(4 c^2)   (dead feature: 1/(3c); positive
bias on a zero column gains 1/(4c^2)); k >= 5 from the numerically solved (l*, b*).

Classifier. A column is *active* if its own feature is reconstructed (l_i^2 + b_i above a
threshold), *positive* if it is not active and b_i > threshold. With A active columns the name
is k = A for A >= 5 and 4 with phi = 4 - A minuses otherwise, sigma = #positive. The label is kept
only if the exact population loss is within `tol` of that family's level, else "other". This is
the paper's (convex hull, positive bias) scheme made robust to vestigial columns, plus a loss check.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch
from scipy.optimize import minimize

from .model import kgon_vertices
from .potential import population_loss

# Table K.1 of the paper: estimated local learning coefficients (std over SGLD chains), c = 6.
PAPER_LLC_HAT = {
    "4----": (0.000, 0.00), "4+----": (0.540, 0.16), "4++----": (0.998, 0.54),
    "4---": (1.024, 0.74), "4+---": (1.619, 0.89), "4++---": (1.899, 0.76),
    "4--": (1.689, 0.88), "4+--": (2.096, 1.00), "4++--": (2.597, 0.88),
    "4-": (2.991, 0.35), "4+-": (3.393, 0.65), "4++-": (4.097, 0.65),
    "4": (5.297, 0.04), "4+": (5.761, 1.53), "4++": (6.203, 0.99),
    "5": (7.705, 0.85), "5+": (9.906, 1.27), "6": (9.027, 0.59),
}
# Table 1: theoretically derived values.
THEORY_LLC = {"5": 7.0, "5+": 8.5, "6": 8.5}


def name_of(k: int, sigma: int, phi: int) -> str:
    return f"{k}" + "+" * sigma + "-" * phi


@lru_cache(maxsize=None)
def kgon_optimum(k: int, c: int = 6) -> tuple[float, float]:
    """(l*, b*) of the standard k-gon: the stationary point of L restricted to regular k-gons."""
    if k == 4:
        return 1.0, 0.0

    def f(x):
        W = torch.tensor(kgon_vertices(k, 0.0, c, x[0]), dtype=torch.float64)
        b = torch.full((c,), -1.0, dtype=torch.float64)
        b[:k] = x[1]
        return float(population_loss(W, b))

    res = minimize(f, x0=[1.2, -0.3], method="Nelder-Mead",
                   options=dict(xatol=1e-12, fatol=1e-15, maxiter=20000))
    return float(res.x[0]), float(res.x[1])


@dataclass(frozen=True)
class CriticalPoint:
    name: str
    k: int
    sigma: int
    phi: int
    loss: float
    prior_half_norm: float  # lower bound of ||w*||^2 / 2 (Table C.2)
    W: np.ndarray  # representative (2, c)
    b: np.ndarray  # (c,)

    @property
    def theta(self) -> np.ndarray:
        return np.concatenate([self.W.reshape(-1), self.b])


def representative(k: int, sigma: int, phi: int, c: int = 6, dead_bias: float = -1.0,
                   dead_len: float = 1.0, rot: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """A member of the family. Dead vertices get bias 2*dead_bias and length `dead_len` (must
    satisfy dead_len^2 < -bias); vestigial columns are zero with bias `dead_bias` or 1/(2c).

    Note: with sigma > 0 the gradient w.r.t. a positive-bias zero column is (b/c) * sum_i W_i, so
    the point is critical only if the columns sum to zero. Dead vertices therefore keep the
    4-gon length by default (the paper's "l_i subject only to l_i^2 < -b_i" holds for sigma = 0)."""
    l, bstar = kgon_optimum(k, c)
    W = kgon_vertices(k, rot, c, l)
    b = np.full(c, dead_bias)
    b[:k] = bstar
    for j in range(phi):  # kill vertices 0..phi-1 (adjacent ones first)
        b[j] = 2 * dead_bias
        W[:, j] *= dead_len / l
    b[k:k + sigma] = 1.0 / (2 * c)
    return W, b


@lru_cache(maxsize=None)
def critical_points(c: int = 6) -> tuple[CriticalPoint, ...]:
    out = []
    fams = [(k, s, 0) for k in range(c, 4, -1) for s in range(0, c - k + 1) if k in (5, 6, 7, 8)]
    fams += [(4, s, p) for p in range(0, 5) for s in range(0, c - 4 + 1)]
    for k, s, p in fams:
        W, b = representative(k, s, p, c)
        loss = float(population_loss(torch.tensor(W), torch.tensor(b)))
        l, bstar = kgon_optimum(k, c)
        half = 0.5 * (k - p) * (l * l + bstar * bstar) + 0.5 * s / (2 * c) ** 2
        out.append(CriticalPoint(name_of(k, s, p), k, s, p, loss, half, W, b))
    return tuple(sorted(out, key=lambda cp: -cp.loss))


def by_name(c: int = 6) -> dict[str, CriticalPoint]:
    return {cp.name: cp for cp in critical_points(c)}


def level(k: int, sigma: int, phi: int, c: int = 6) -> float:
    """Loss level, also for combinations outside the paper's list (e.g. 4 with 3 pluses)."""
    if k == 4:
        return (c - 4 + phi) / (3 * c) - sigma / (4 * c * c)
    W, b = representative(k, 0, 0, c)
    return float(population_loss(torch.tensor(W), torch.tensor(b))) - sigma / (4 * c * c)


def classify(theta, c: int = 6, tol: float = 0.004, act_thresh: float = 0.25,
             pos_thresh: float = 0.02, return_loss: bool = False):
    """theta: (N, 3c) array/tensor. Returns list of names ("other" if off every level)."""
    th = torch.as_tensor(np.asarray(theta), dtype=torch.float64).reshape(-1, 3 * c)
    W = th[:, : 2 * c].reshape(-1, 2, c)
    b = th[:, 2 * c:]
    L = population_loss(W, b).numpy()
    l2 = (W**2).sum(1)
    active = (l2 + b) > act_thresh
    positive = (~active) & (b > pos_thresh)
    A = active.sum(1).numpy()
    S = positive.sum(1).numpy()
    names = []
    for a, s, loss in zip(A, S, L):
        k, phi = (int(a), 0) if a >= 5 else (4, 4 - int(a))
        if k + s > c + phi or not np.isfinite(loss):
            names.append("other")
            continue
        names.append(name_of(k, int(s), phi) if abs(loss - level(k, int(s), phi, c)) < tol else "other")
    return (names, L) if return_loss else names


def free_energy(name: str, n, c: int = 6, llc: dict | None = None, with_const: bool = True):
    """f_alpha(n) = n L_alpha + lambda_alpha log n + c_alpha  (eq. 11 with the prior constant of C.1)."""
    cp = by_name(c)[name]
    lam = (llc or {}).get(name, THEORY_LLC.get(name, PAPER_LLC_HAT.get(name, (np.nan,))[0]))
    n = np.asarray(n, dtype=float)
    return n * cp.loss + lam * np.log(n) + (cp.prior_half_norm if with_const else 0.0)


def critical_n(alpha: str, beta: str, c: int = 6, llc: dict | None = None, with_const: bool = True,
               n_max: float = 1e6) -> float:
    """Smallest n > 3 where F(beta) drops below F(alpha) (eq. 12). nan if no crossing / already below."""
    ns = np.geomspace(3.0, n_max, 20000)
    d = free_energy(beta, ns, c, llc, with_const) - free_energy(alpha, ns, c, llc, with_const)
    idx = np.where((d[:-1] > 0) & (d[1:] <= 0))[0]
    return float(ns[idx[0] + 1]) if len(idx) else float("nan")
