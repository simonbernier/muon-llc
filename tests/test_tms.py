import numpy as np
import pytest
import torch

from muon_llc.tms.critical_points import by_name, classify, critical_n, critical_points, kgon_optimum
from muon_llc.tms.model import init_kgon, make_problem, tms_loss
from muon_llc.tms.potential import population_loss, population_loss_flat

PAPER_LOSS = {  # Figure A.2 / Table 1
    "6": 0.04819, "5": 0.06874, "5+": 0.06180, "4": 0.11111, "4+": 0.10417, "4++": 0.09722,
    "4-": 0.16667, "4+-": 0.15972, "4++-": 0.15278, "4--": 0.22222, "4+--": 0.21528, "4++--": 0.20833,
    "4---": 0.27778, "4+---": 0.27083, "4++---": 0.26389, "4----": 0.33333, "4+----": 0.32639,
    "4++----": 0.31944,
}
PAPER_PRIOR = {"6": 6.37767, "5": 3.62417, "5+": 3.62764, "4": 2.0, "4+": 2.00347, "4++-": 1.50694}


def test_kgon_parameters_table_a1():
    assert kgon_optimum(5) == pytest.approx((1.17046, -0.28230), abs=2e-5)
    assert kgon_optimum(6) == pytest.approx((1.32053, -0.61814), abs=2e-5)


def test_losses_and_prior_constants():
    cps = by_name()
    assert set(cps) == set(PAPER_LOSS)
    for name, L in PAPER_LOSS.items():
        assert cps[name].loss == pytest.approx(L, abs=6e-6), name
    for name, c in PAPER_PRIOR.items():
        assert cps[name].prior_half_norm == pytest.approx(c, abs=5e-5), name


def test_critical_points_have_zero_gradient_and_classify():
    for cp in critical_points():
        W = torch.tensor(cp.W, requires_grad=True)
        b = torch.tensor(cp.b, requires_grad=True)
        gW, gb = torch.autograd.grad(population_loss(W, b), [W, b])
        assert max(gW.abs().max(), gb.abs().max()) < 1e-7, cp.name
        assert classify(cp.theta[None])[0] == cp.name


def test_critical_sample_size_5_to_6():
    assert critical_n("5", "6") == pytest.approx(601, abs=2)  # Appendix C.2


def test_population_loss_matches_monte_carlo():
    p = make_problem(400_000, seed=3)
    theta = 0.7 * torch.randn(5, 18, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    assert torch.allclose(p.loss(theta), population_loss_flat(theta), rtol=2e-2, atol=2e-3)


def test_batched_loss_matches_loop_and_per_run_batches():
    p = make_problem(64, seed=1)
    theta = init_kgon(3, seed=5)
    W, b = theta[:, :12].reshape(3, 2, 6), theta[:, 12:]
    idx = torch.stack([torch.randperm(64)[:20] for _ in range(3)])
    batched = tms_loss(W, b, p.feat[idx], p.mu[idx])
    for j in range(3):
        assert torch.allclose(batched[j], tms_loss(W[j], b[j], p.feat[idx[j]], p.mu[idx[j]]))


def test_classify_other_off_level():
    theta = init_kgon(1, seed=0, bias_std=0.0)
    theta[:, :12] *= 0.5
    assert classify(theta)[0] == "other"
    assert np.isfinite(float(population_loss_flat(theta)))


def test_batched_spectral_step_equals_per_matrix_step():
    from muon_llc.optim import SpectralOptimizer

    g = torch.Generator().manual_seed(0)
    W0 = torch.randn(4, 2, 6, generator=g, dtype=torch.float64)
    G = torch.randn(4, 2, 6, generator=g, dtype=torch.float64)
    for kw in (dict(orth="exact"), dict(orth="ns", momentum=0.9, nesterov=True, ns_dtype=None)):
        Wb = W0.clone().requires_grad_(True)
        ob = SpectralOptimizer([Wb], lr=0.1, **kw)
        singles = [W0[j].clone().requires_grad_(True) for j in range(4)]
        os_ = [SpectralOptimizer([w], lr=0.1, **kw) for w in singles]
        for _ in range(3):
            Wb.grad = G.clone()
            ob.step()
            for j, (w, o) in enumerate(zip(singles, os_)):
                w.grad = G[j].clone()
                o.step()
        assert torch.allclose(Wb.detach(), torch.stack([w.detach() for w in singles]), atol=1e-10)


def test_batched_training_runs_are_independent():
    from muon_llc.tms.train import train_batched

    p = make_problem(100, seed=0)
    th = init_kgon(3, seed=1)
    cfg = dict(name="muon", lr=0.002, momentum=0.9, nesterov=True, orth="ns", ns_dtype=None)
    full = train_batched(p, th, cfg, epochs=2, ckpt_every=1, seed=0)
    assert np.isfinite(full.thetas).all()
    assert (full.path_len[-1] > 0).all()
    # changing run 2's init must not change run 0
    th2 = th.clone()
    th2[2] += 0.3
    other = train_batched(p, th2, cfg, epochs=2, ckpt_every=1, seed=0)
    assert np.allclose(full.thetas[:, 0], other.thetas[:, 0])
    assert not np.allclose(full.thetas[:, 2], other.thetas[:, 2])


def test_plateau_segmentation():
    from muon_llc.tms.transitions import plateaus, summarize

    lab = np.array(["4+-"] * 6 + ["other"] * 4 + ["4"] * 5 + ["other"] + ["4"] * 3 + ["5", "other"] + ["5"] * 4,
                   dtype=object)
    ps = plateaus(lab, min_ckpts=3, gap=2)
    assert [p.name for p in ps] == ["4+-", "4", "5"]
    s = summarize(lab[:, None])
    assert s["transitions"][("4+-", "4")] == 1 and s["finals"]["5"] == 1


def test_sgld_llc_regular_point_and_batching():
    from muon_llc.tms.llc import sgld_llc

    p = make_problem(1000, seed=0)
    six = by_name()["6"].theta
    r = sgld_llc(p, np.stack([six, six]), eps=1e-3, gamma=1.0, n_steps=400, n_burnin=100, n_chains=8, seed=0)
    assert r.llc.shape == (2,)
    assert 6.5 < r.llc.mean() < 10.5  # theory 8.5


def test_per_run_datasets_match_single_dataset_runs():
    from muon_llc.tms.llc import sgld_llc
    from muon_llc.tms.train import train_batched

    multi = make_problem(100, seed=7, n_datasets=3)
    th = init_kgon(3, seed=1)
    cfg = dict(name="gd", lr=0.005)
    full = train_batched(multi, th, cfg, epochs=2, ckpt_every=1, seed=0)
    single = make_problem(100, seed=8)  # dataset of run 1
    assert torch.equal(single.feat, multi.feat[1])
    assert torch.allclose(multi.loss(th)[1], single.loss(th[1]))
    r = sgld_llc(multi, full.thetas[-1], n_steps=20, n_chains=2, data_index=np.arange(3), seed=0)
    assert np.allclose(r.loss0[1], float(single.loss(torch.tensor(full.thetas[-1, 1]))))
