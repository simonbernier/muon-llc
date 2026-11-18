"""The Gibbs sampler, MALA and quadrature must agree where all three apply."""

import numpy as np
import torch

from muon_llc.dln.gibbs import run_gibbs
from muon_llc.dln.model import DLNProblem, balanced_weights, flatten, make_problem
from muon_llc.dln.theory.quadrature import scalar_chain_expectation
from muon_llc.slt.estimators import ess, split_rhat, wbic_estimate, window_means
from muon_llc.slt.mala import run_mala


def scalar_problem():
    z = torch.zeros(1, 1, dtype=torch.float64)
    return DLNProblem((1, 1, 1, 1), torch.ones(1, 4, dtype=torch.float64), torch.zeros(1, 4, dtype=torch.float64), z)


def test_quadrature_is_grid_independent_and_below_one_half():
    a = scalar_chain_expectation(50.0, 1.0, n_grid=2001)
    b = scalar_chain_expectation(50.0, 1.0, n_grid=4001)
    assert abs(a - b) < 1e-5
    assert 0.0 < b < 0.5


def test_gibbs_and_mala_match_quadrature_on_the_scalar_chain():
    prob = scalar_problem()
    nbeta, gamma = 50.0, 1.0
    want = scalar_chain_expectation(nbeta, gamma)
    rg = run_gibbs(prob, torch.zeros(3, dtype=torch.float64), [nbeta], gamma, n_chains=64, n_burnin=200,
                   n_draws=4000, seed=0)
    eg = wbic_estimate(rg.delta_loss[0], nbeta, gamma)
    assert abs(eg.lambda_hat - want) < 5 * eg.err + 0.005, (eg.lambda_hat, want, eg.err)
    rm = run_mala(prob, torch.zeros(3, dtype=torch.float64), [nbeta], gamma, 0.1, n_chains=64, n_burnin=1000,
                  n_draws=6000, seed=1, init_jitter=0.3)
    em = wbic_estimate(rm.delta_loss[0], nbeta, gamma)
    assert abs(em.lambda_hat - want) < 5 * em.err + 0.01, (em.lambda_hat, want, em.err)


def test_gibbs_matches_mala_on_a_rank_one_point():
    prob = make_problem((3, 3, 3), 64, [1.0], seed=0, dtype=torch.float64)
    ws = balanced_weights(prob.A_star, prob.widths, torch.Generator().manual_seed(1))
    theta = flatten(ws)
    assert float(prob.loss(theta)) < 1e-20
    nbeta, gamma = 40.0, 2.0
    rg = run_gibbs(prob, theta, [nbeta], gamma, n_chains=32, n_burnin=500, n_draws=4000, seed=0)
    rm = run_mala(prob, theta, [nbeta], gamma, 0.01, n_chains=32, n_burnin=1000, n_draws=6000, seed=0)
    eg, em = wbic_estimate(rg.delta_loss[0], nbeta, gamma), wbic_estimate(rm.delta_loss[0], nbeta, gamma)
    assert abs(eg.lambda_hat - em.lambda_hat) < 5 * np.hypot(eg.err, em.err) + 0.02, (eg.lambda_hat, em.lambda_hat)


def test_mala_thinning_keeps_the_same_chain():
    prob = scalar_problem()
    kw = dict(n_chains=4, n_burnin=40, n_draws=200, seed=3)
    a = run_mala(prob, torch.zeros(3, dtype=torch.float64), [20.0], 1.0, 0.1, **kw)
    b = run_mala(prob, torch.zeros(3, dtype=torch.float64), [20.0], 1.0, 0.1, thin=10, **kw)
    assert b.delta_loss.shape == (1, 4, 20) and b.delta_loss_burnin.shape == (1, 4, 4)
    assert np.array_equal(a.delta_loss[..., 9::10], b.delta_loss)
    assert np.array_equal(a.acceptance, b.acceptance)
    assert a.extras["accept_burnin_tail"].shape == (1,)


def test_diagnostics_on_independent_and_on_drifting_draws():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(8, 4000))
    assert abs(split_rhat(x) - 1.0) < 0.01
    assert 0.8 * x.size < ess(x) < 1.25 * x.size
    assert window_means(x, 5).shape == (5,)
    drift = x + np.linspace(0, 3, 4000)
    assert split_rhat(drift) > 1.2
    ar = np.zeros_like(x)
    for t in range(1, x.shape[1]):
        ar[:, t] = 0.9 * ar[:, t - 1] + x[:, t]
    assert ess(ar) < 0.15 * x.size  # AR(1) with ρ = 0.9 keeps (1 − ρ)/(1 + ρ) ≈ 5 % of the draws


def test_estimator_rows_with_the_gibbs_sampler_in_single_precision():
    from muon_llc.dln.llc import estimate_checkpoint, headline_estimator

    prob = make_problem((3, 3, 3), 64, [1.0, 0.5], seed=0, device="cpu", dtype=torch.float32)
    theta = flatten(balanced_weights(prob.A_star.double(), (3, 3, 3), torch.Generator().manual_seed(0)))
    cfg = {"nbeta_ladder": {"n": 3, "lo": 0.5, "hi": 2.0}, "hessian": False, "surrogate_sampler": "gibbs",
           "gibbs": {"n_chains": 4, "n_burnin": 20, "n_draws": 60, "n_boot": 20}}
    rows, rungs = estimate_checkpoint(prob, theta.float(), cfg, step=0, seed=0, gammas=[1.0],
                                      estimators=["gibbs", "surrogate"])
    names = [r["estimator"] for r in rows]
    assert names == ["surrogate", "surrogate_ladder", "gibbs", "gibbs_ladder"]
    assert headline_estimator(names) == "gibbs" and headline_estimator(["mala", "sgld"]) == "mala"
    assert {r["estimator"] for r in rungs} == {"surrogate", "gibbs"} and len(rungs) == 6
    assert all(np.isfinite(r["lambda_hat"]) and r["lambda_hat"] > 0 for r in rows)
