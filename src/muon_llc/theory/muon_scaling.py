"""A5: full Muon as perturbations of spectral GD — three scaling laws to test in C2.

1. Newton-Schulz with k iterations. The quintic acts on singular values as
   σ -> a σ + b σ³ + c σ⁵ with (a, b, c) = (3.4445, -4.7750, 2.0315). For small σ this is
   σ -> a σ, so a singular value of the Frobenius-normalised update below ~a^{-k} is not
   flattened to 1: it is merely amplified by a^k. Uniform growth therefore fails when the
   gradient's condition number κ exceeds ~a^k, and intermediate LLC plateaus reappear.
   Prediction: log κ* = k log a + const  (slope log a ≈ 1.237 per NS step).

2. Decoupled weight decay λ_wd. W <- (1 - η λ_wd) W - η O with ||O||_op = 1, so per-layer
   spectral norms equilibrate at 1/λ_wd; in an L-layer DLN product singular values saturate at
   min(s_i, λ_wd^{-L}), independent of η.

3. Momentum μ. Under alignment orth(M) = orth(G), so uniform growth is preserved; at
   saturation overshoot gives fixed-size ±η steps in each layer, i.e. oscillation of the
   layer singular values with amplitude ~η (product amplitude ~L η s^{(L-1)/L}).
"""

from __future__ import annotations

import numpy as np

from ..optim import KJ_COEFFS, ns_scalar_map

NS_SMALL_SIGMA_GAIN = KJ_COEFFS[0]  # a = 3.4445


def ns_flattening_threshold(k: int, a: float = NS_SMALL_SIGMA_GAIN) -> float:
    """Singular value (of the Frobenius-normalised update) below which k NS steps leave it small."""
    return float(a ** (-k))


def predicted_breakdown_kappa(k: int, const: float = 1.0, a: float = NS_SMALL_SIGMA_GAIN) -> float:
    return float(const * a**k)


def ns_response_curve(k: int, sigmas: np.ndarray, coeffs=KJ_COEFFS) -> np.ndarray:
    """σ -> p^k(σ) for the quintic p; exact per-singular-value action of Newton-Schulz."""
    import torch

    return ns_scalar_map(torch.as_tensor(np.asarray(sigmas, dtype=np.float64)), k, coeffs).numpy()


def wd_layer_cap(weight_decay: float) -> float:
    return float("inf") if weight_decay <= 0 else 1.0 / weight_decay


def wd_product_cap(weight_decay: float, depth: int) -> float:
    return float("inf") if weight_decay <= 0 else weight_decay ** (-int(depth))


def predicted_final_singular_values(singular_values, weight_decay: float, depth: int) -> np.ndarray:
    return np.minimum(np.asarray(singular_values, dtype=np.float64), wd_product_cap(weight_decay, depth))


def momentum_overshoot_amplitude(lr: float, s: float, depth: int) -> float:
    """Product-level oscillation amplitude from ±lr steps on each of L layer singular values."""
    return float(depth * lr * s ** ((depth - 1) / depth))


def fit_loglog(x, y) -> tuple[float, float, float]:
    """Fit log y = a + b log x. Returns (exponent b, prefactor exp(a), R^2)."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    lx, ly = np.log(x[ok]), np.log(y[ok])
    if lx.size < 2:
        return float("nan"), float("nan"), float("nan")
    b, a = np.polyfit(lx, ly, 1)
    pred = a + b * lx
    ss_res = float(((ly - pred) ** 2).sum())
    ss_tot = float(((ly - ly.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(b), float(np.exp(a)), r2
