"""Bayesian posterior of TMS sampled with NUTS (numpyro), as in Section 4.2 / Appendix F.

Prior N(0, 1) on every entry of (W, b). Likelihood exp(-n * nll_scale * L_n(w)) with L_n the
paper's loss (mean squared reconstruction error). nll_scale = 1 is the convention under which
the paper's free-energy formula n L + lambda log n is written (it reproduces their critical
sample sizes); nll_scale = 0.5 is the unit-variance Gaussian model p(x|w) ~ exp(-||x - f||^2 / 2)
as literally stated in eq. (3).

`classify_hull` is the paper's region classifier W_{k, sigma}: k = number of columns on the
convex hull, sigma = number of positive biases.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull


def sample_posterior(feat: np.ndarray, mu: np.ndarray, c: int = 6, r: int = 2, n_chains: int = 6,
                     n_warmup: int = 500, n_samples: int = 5000, thin: int = 10, seed: int = 0,
                     nll_scale: float = 1.0, prior_std: float = 1.0, progress: bool = False) -> np.ndarray:
    """Returns flat samples (n_chains, n_samples // thin, r*c + c)."""
    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS

    jax.config.update("jax_enable_x64", True)
    feat_j, mu_j = jnp.asarray(feat), jnp.asarray(mu)
    n = feat_j.shape[0]
    target = jax.nn.one_hot(feat_j, c) * mu_j[:, None]

    def model():
        W = numpyro.sample("W", dist.Normal(0.0, prior_std).expand([r, c]).to_event(2))
        b = numpyro.sample("b", dist.Normal(0.0, prior_std).expand([c]).to_event(1))
        G = W.T @ W
        out = jax.nn.relu(mu_j[:, None] * G[feat_j] + b)
        numpyro.factor("ll", -nll_scale * jnp.sum((target - out) ** 2))

    mcmc = MCMC(NUTS(model), num_warmup=n_warmup, num_samples=n_samples, thinning=thin,
                num_chains=n_chains, chain_method="vectorized", progress_bar=progress)
    mcmc.run(jax.random.PRNGKey(seed))
    s = mcmc.get_samples(group_by_chain=True)
    W = np.asarray(s["W"])
    b = np.asarray(s["b"])
    assert n > 0
    return np.concatenate([W.reshape(*W.shape[:2], r * c), b], axis=-1)


def classify_hull(theta: np.ndarray, c: int = 6) -> list[str]:
    """Paper's W_{k, sigma} label: '4', '4+', '5', '5+', '6', else 'other'."""
    theta = np.asarray(theta).reshape(-1, 3 * c)
    out = []
    for th in theta:
        W, b = th[: 2 * c].reshape(2, c), th[2 * c:]
        try:
            k = len(ConvexHull(W.T).vertices)
        except Exception:
            out.append("other")
            continue
        sigma = int((b > 0).sum())
        name = f"{k}" + "+" * sigma
        out.append(name if name in ("4", "4+", "5", "5+", "6") else "other")
    return out
