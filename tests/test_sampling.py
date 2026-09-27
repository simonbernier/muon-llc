"""Estimator checks on problems with known answers (A1 / A2)."""

import numpy as np
import torch

from muon_llc.sampling.estimators import fit_ladder, wbic_estimate
from muon_llc.sampling.mala import run_mala
from muon_llc.sampling.sgld import run_sgld
from muon_llc.theory.hessian import lambda_quad


class QuadraticProblem:
    """L(theta) = 1/2 sum_i h_i theta_i^2. Mimics the DLNProblem interface the samplers use."""

    def __init__(self, h, n=1000, device="cpu", dtype=torch.float64):
        self.h = torch.as_tensor(h, dtype=dtype)
        self._n = n
        self.device = torch.device(device)
        self.dtype = dtype

    @property
    def n(self):
        return self._n

    def loss(self, theta, idx=None):
        return 0.5 * (self.h * theta * theta).sum(-1)

    def loss_and_grad(self, theta, idx=None):
        return self.loss(theta), self.h * theta

    def minibatch_indices(self, batch_size, generator):
        return torch.arange(batch_size)


class MonomialProblem(QuadraticProblem):
    """L(theta) = sum_i theta_i^(2k). For 2k = 4 each direction contributes 1/(2k) = 1/4."""

    def __init__(self, dim, k=2, **kw):
        super().__init__(np.ones(dim), **kw)
        self.k = k

    def loss(self, theta, idx=None):
        return (theta ** (2 * self.k)).sum(-1)

    def loss_and_grad(self, theta, idx=None):
        return self.loss(theta), 2 * self.k * theta ** (2 * self.k - 1)


def test_mala_quadratic_matches_soft_rank():
    h = np.array([10.0, 3.0, 1.0, 0.3, 0.1, 0.03])
    prob = QuadraticProblem(h)
    nbeta, gamma = 50.0, 5.0
    rec = run_mala(prob, torch.zeros(6, dtype=torch.float64), [nbeta], gamma, 0.05, n_chains=32,
                   n_burnin=500, n_draws=3000, seed=0)
    e = wbic_estimate(rec.delta_loss[0], nbeta, gamma, rec.acceptance[0])
    want = lambda_quad(h, nbeta, gamma)
    assert abs(e.lambda_hat - want) < 4 * e.err + 0.03, (e.lambda_hat, want, e.err)
    assert 0.35 < e.acceptance < 0.8
    # fluctuation identity for quadratics: var = (1/2) sum c_i^2 <= mean, so var/mean <= 1
    assert e.var_over_mean < 1.05


def test_sgld_quadratic_small_eps_matches_soft_rank():
    h = np.array([2.0, 1.0, 0.5, 0.1])
    prob = QuadraticProblem(h)
    nbeta, gamma = 20.0, 4.0
    rec = run_sgld(prob, torch.zeros(4, dtype=torch.float64), nbeta, gamma, eps=2e-3, n_chains=32,
                   n_burnin=2000, n_draws=6000, batch_size=1, seed=0)
    e = wbic_estimate(rec.delta_loss, nbeta, gamma)
    want = lambda_quad(h, nbeta, gamma)
    # unadjusted Langevin inflates each direction by 1/(1 - eps A/4); eps A <= 0.09 -> <2.5% bias
    assert abs(e.lambda_hat - want) < 0.05 * want + 4 * e.err


def test_mala_monomial_is_quarter_per_direction_independent_of_beta():
    prob = MonomialProblem(dim=3, k=2)
    nbetas = np.array([30.0, 300.0, 3000.0])
    rec = run_mala(prob, torch.zeros(3, dtype=torch.float64), nbetas, gamma=0.0, eps0=[0.3, 0.1, 0.03],
                   n_chains=32, n_burnin=1000, n_draws=4000, seed=1)
    for k in range(3):
        e = wbic_estimate(rec.delta_loss[k], nbetas[k], 0.0)
        assert abs(e.lambda_hat - 0.75) < 5 * e.err + 0.03, (nbetas[k], e.lambda_hat, e.err)


def test_ladder_fit_recovers_lambda_and_m_on_scale_free_problem():
    # monomial with no localiser: E[nβΔL] = 3/4 exactly at every β, so λ = 3/4 and m = 1
    prob = MonomialProblem(dim=3, k=2)
    nbetas = np.logspace(1.5, 3.5, 5)
    rec = run_mala(prob, torch.zeros(3, dtype=torch.float64), nbetas, gamma=0.0,
                   eps0=0.3 * (nbetas / nbetas[0]) ** -0.5, n_chains=32, n_burnin=1000, n_draws=4000, seed=2)
    lf = fit_ladder(rec.delta_loss, nbetas, 0.0, n_boot=100)
    assert abs(lf.lambda_slope - 0.75) < 0.1, lf.row()
    assert abs(lf.m_hat - 1.0) < 0.5, lf.row()
    assert abs(lf.lambda_thermo - 0.75) < 0.1, lf.row()


def test_mala_adaptation_targets_acceptance():
    prob = QuadraticProblem(np.ones(10))
    rec = run_mala(prob, torch.zeros(10, dtype=torch.float64), [100.0], 1.0, eps0=1.0, n_chains=16,
                   n_burnin=600, n_draws=500, seed=0)
    acc = rec.acceptance.mean()
    assert 0.4 < acc < 0.75, acc
