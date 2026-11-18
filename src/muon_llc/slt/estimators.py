"""LLC estimators and regime diagnostics from sampled losses.

Notation: ΔL_t = L_n(w_t) - L_n(w0) along a chain at inverse temperature β (nβ = n β).

WBIC estimate at one β:     λ̂(β) = nβ · mean_t ΔL_t
Scale-free theory:          -log Z(β) = λ log(nβ) - (m - 1) log log(nβ) + const
                            ⇒  E_β[nβ ΔL] = λ - (m - 1) / log(nβ)
Ladder fit (A2):            regress λ̂(β_k) on 1/log(nβ_k): intercept λ, slope -(m - 1)
Thermodynamic route:        F(nβ_k) = ∫_{nβ_0}^{nβ_k} E[ΔL] d(nβ) = ∫ nβ E[ΔL] d log(nβ)
                            (trapezoid in log nβ), then fit
                            F = λ log(nβ_k/nβ_0) - (m-1) log(log nβ_k / log nβ_0)
Fluctuation-dissipation:    Var(nβ ΔL) = E[nβ ΔL] - d E[nβ ΔL] / d log β; scale-free ⇒ Var = mean
Quadratic bracket (A1):     with inflation factors c_i, mean = ½Σc_i, Var = ½Σc_i², so
                            mean²/Var ≤ λ_quad; var/mean ≈ 1 signals the scale-free regime.

Error bars come from per-chain means (standard error over chains) or from a chain bootstrap.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class LLCEstimate:
    lambda_hat: float
    err: float
    nbeta: float
    gamma: float
    var_over_mean: float
    mean2_over_var: float
    n_chains: int
    n_draws: int
    acceptance: float | None = None
    eps: float | None = None
    per_chain: np.ndarray | None = None

    def row(self) -> dict:
        d = asdict(self)
        d.pop("per_chain")
        return d


def wbic_estimate(delta_loss: np.ndarray, nbeta: float, gamma: float,
                  acceptance=None, eps=None) -> LLCEstimate:
    """delta_loss: (C, T) for one β."""
    x = nbeta * np.asarray(delta_loss, dtype=np.float64)
    per_chain = x.mean(axis=1)
    lam = float(per_chain.mean())
    C = x.shape[0]
    err = float(per_chain.std(ddof=1) / np.sqrt(C)) if C > 1 else float("nan")
    var = float(x.var())
    return LLCEstimate(
        lambda_hat=lam, err=err, nbeta=float(nbeta), gamma=float(gamma),
        var_over_mean=var / lam if lam != 0 else float("nan"),
        mean2_over_var=lam * lam / var if var > 0 else float("nan"),
        n_chains=C, n_draws=x.shape[1],
        acceptance=None if acceptance is None else float(np.mean(acceptance)),
        eps=None if eps is None else float(np.mean(eps)),
        per_chain=per_chain,
    )


@dataclass
class LadderFit:
    lambda_slope: float
    lambda_slope_err: float
    m_hat: float
    m_hat_err: float
    lambda_thermo: float
    lambda_thermo_err: float
    m_thermo: float
    nbetas: np.ndarray
    lambda_wbic: np.ndarray  # per rung
    lambda_wbic_err: np.ndarray
    used: np.ndarray  # mask of rungs used in the fit

    def row(self) -> dict:
        return dict(
            lambda_slope=self.lambda_slope, lambda_slope_err=self.lambda_slope_err,
            m_hat=self.m_hat, m_hat_err=self.m_hat_err,
            lambda_thermo=self.lambda_thermo, lambda_thermo_err=self.lambda_thermo_err,
            m_thermo=self.m_thermo,
            lambda_wbic_min=float(np.min(self.lambda_wbic[self.used])),
            lambda_wbic_max=float(np.max(self.lambda_wbic[self.used])),
        )


def _fit_intercept_slope(x: np.ndarray, y: np.ndarray, w: np.ndarray | None = None):
    """Weighted least squares y = a + b x. Returns (a, b)."""
    A = np.stack([np.ones_like(x), x], axis=1)
    if w is not None:
        sw = np.sqrt(w)
        A, y = A * sw[:, None], y * sw
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(coef[0]), float(coef[1])


def fit_ladder(delta_loss: np.ndarray, nbetas: np.ndarray, gamma: float, n_boot: int = 200,
               seed: int = 0, use: np.ndarray | None = None) -> LadderFit:
    """delta_loss: (K, C, T) over rungs, chains, draws; nbetas: (K,)."""
    delta_loss = np.asarray(delta_loss, dtype=np.float64)
    nbetas = np.asarray(nbetas, dtype=np.float64).reshape(-1)
    K, C, T = delta_loss.shape
    order = np.argsort(nbetas)
    nbetas, delta_loss = nbetas[order], delta_loss[order]
    used = np.ones(K, dtype=bool) if use is None else np.asarray(use, dtype=bool)[order]

    chain_means = delta_loss.mean(axis=2)  # (K, C) : E[ΔL] per chain
    lam_k = nbetas * chain_means.mean(axis=1)
    lam_k_err = nbetas * chain_means.std(axis=1, ddof=1) / np.sqrt(C) if C > 1 else np.full(K, np.nan)

    def fits(cm):  # cm: (K, C) chain means (possibly resampled)
        E = cm.mean(axis=1)  # (K,)
        lam = nbetas * E
        x = 1.0 / np.log(nbetas)
        a, b = _fit_intercept_slope(x[used], lam[used])
        lam_slope, m = a, 1.0 - b
        # thermodynamic integration on the used rungs, in log nβ (the integrand nβ·E[ΔL] is
        # flat in the scale-free regime, so the trapezoid rule is accurate on a log-spaced ladder)
        nb, lam_u = nbetas[used], lam[used]
        F = np.concatenate([[0.0], np.cumsum(0.5 * (lam_u[1:] + lam_u[:-1]) * np.diff(np.log(nb)))])
        X1 = np.log(nb / nb[0])
        X2 = -np.log(np.log(nb) / np.log(nb[0]))
        A = np.stack([X1, X2], axis=1)[1:]
        coef, *_ = np.linalg.lstsq(A, F[1:], rcond=None)
        lam_th, m_th = float(coef[0]), 1.0 + float(coef[1])
        return lam_slope, m, lam_th, m_th

    lam_slope, m_hat, lam_th, m_th = fits(chain_means)
    rng = np.random.default_rng(seed)
    boots = []
    if C > 1 and n_boot > 0:
        for _ in range(n_boot):
            idx = rng.integers(0, C, size=C)
            boots.append(fits(chain_means[:, idx]))
    boots = np.array(boots) if boots else np.full((1, 4), np.nan)
    return LadderFit(
        lambda_slope=lam_slope, lambda_slope_err=float(np.nanstd(boots[:, 0])),
        m_hat=m_hat, m_hat_err=float(np.nanstd(boots[:, 1])),
        lambda_thermo=lam_th, lambda_thermo_err=float(np.nanstd(boots[:, 2])), m_thermo=m_th,
        nbetas=nbetas, lambda_wbic=lam_k, lambda_wbic_err=lam_k_err, used=used,
    )


def fluctuation_dissipation_residual(delta_loss: np.ndarray, nbetas: np.ndarray) -> np.ndarray:
    """Var(nβΔL) - [E(nβΔL) - dE(nβΔL)/dlog β] per rung (finite differences). ≈0 at equilibrium."""
    x = np.asarray(delta_loss, dtype=np.float64) * np.asarray(nbetas).reshape(-1, 1, 1)
    mean = x.mean(axis=(1, 2))
    var = x.var(axis=(1, 2))
    logb = np.log(np.asarray(nbetas, dtype=np.float64).reshape(-1))
    dmean = np.gradient(mean, logb) if len(logb) > 1 else np.zeros_like(mean)
    return var - (mean - dmean)


# ----------------------------------------------------------------------------- convergence
def window_means(delta_loss: np.ndarray, k: int) -> np.ndarray:
    """delta_loss (C, T) -> (k,) mean over chains and over each of k consecutive time windows."""
    x = np.asarray(delta_loss, dtype=np.float64)
    edges = np.linspace(0, x.shape[1], k + 1).astype(int)
    return np.array([x[:, a:b].mean() for a, b in zip(edges[:-1], edges[1:])])


def split_rhat(delta_loss: np.ndarray) -> float:
    """Split-R̂ of Gelman et al.: each chain is cut in two halves, and the variance between the
    2C half-chain means is compared with the variance within them. 1 at convergence; values above
    about 1.01 mean the chains (or the two halves of each chain) disagree."""
    x = np.asarray(delta_loss, dtype=np.float64)
    h = x.shape[1] // 2
    if h < 2:
        return float("nan")
    y = np.concatenate([x[:, :h], x[:, h:2 * h]], axis=0)
    W = y.var(axis=1, ddof=1).mean()
    B = h * y.mean(axis=1).var(ddof=1)
    if W <= 0:
        return float("nan")
    return float(np.sqrt(((h - 1) / h * W + B / h) / W))


def ess(delta_loss: np.ndarray) -> float:
    """Effective number of independent draws in (C, T) correlated draws, from the autocorrelation
    averaged over chains and summed over lags in pairs while the pair sums stay positive (Geyer)."""
    x = np.asarray(delta_loss, dtype=np.float64)
    C, T = x.shape
    if T < 4:
        return float("nan")
    xc = x - x.mean(axis=1, keepdims=True)
    nfft = 1 << int(np.ceil(np.log2(2 * T)))
    f = np.fft.rfft(xc, nfft, axis=1)
    acov = np.fft.irfft(f * np.conj(f), nfft, axis=1)[:, :T] / T  # biased autocovariance per chain
    W = acov[:, 0].mean() * T / (T - 1)
    var_plus = (T - 1) / T * W + (x.mean(axis=1).var(ddof=1) if C > 1 else 0.0)
    if var_plus <= 0:
        return float("nan")
    rho = 1.0 - (W - acov.mean(axis=0)) / var_plus
    tau = 0.0
    for t in range(0, T - 1, 2):
        pair = rho[t] + rho[t + 1]
        if pair < 0:
            break
        tau += 2.0 * pair
    return float(C * T / max(tau - 1.0, 1e-12))
