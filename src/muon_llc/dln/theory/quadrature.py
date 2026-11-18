"""Exact tempered expectation for the smallest deep linear network, widths 1-1-1-1, rank 0.

Loss ½(abc)², localised at the origin:  U = nβ ½(abc)² + γ/2 (a² + b² + c²). The third weight
integrates out in closed form (it is Gaussian given a and b), leaving a two-dimensional integral
done on a grid. The result is E[nβ ΔL] with no sampling error, to compare with a sampler and with
the asymptotic form λ − (m − 1)/log nβ (here λ = 1/2, m = 3).
"""

from __future__ import annotations

import numpy as np


def scalar_chain_expectation(nbeta: float, gamma: float, n_grid: int = 4001, extent: float = 9.0) -> float:
    """E[nβ · ½(abc)²] under exp(−U) for widths 1-1-1-1 centred at 0. Needs gamma > 0."""
    if gamma <= 0:
        raise ValueError("needs gamma > 0")
    # x = sqrt(γ) a, y = sqrt(γ) b: prior N(0, 1) on both, s = nβ a² b² = (nβ/γ²) x² y²
    # grid x = extent·u², dense near the axes where the integrand varies on the scale γ/sqrt(nβ)
    u = np.linspace(0.0, 1.0, n_grid)
    x = extent * u * u
    w = 2.0 * extent * u * (u[1] - u[0])
    w[-1] *= 0.5
    X2 = (x * x)[:, None]
    Y2 = (x * x)[None, :]
    s = (nbeta / gamma**2) * X2 * Y2
    prior = np.exp(-0.5 * (X2 + Y2)) * w[:, None] * w[None, :]
    # ∫ exp(−(s+γ)c²/2) dc ∝ (s+γ)^(−1/2);  ∫ ½ s c² exp(−(s+γ)c²/2) dc ∝ ½ s (s+γ)^(−3/2)
    den = (prior * (s + gamma) ** -0.5).sum()
    num = (prior * 0.5 * s * (s + gamma) ** -1.5).sum()
    return float(num / den)
