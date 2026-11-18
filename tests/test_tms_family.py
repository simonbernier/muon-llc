import numpy as np
import torch

from muon_llc.tms.family import (
    cells_of,
    family_problem,
    feature_losses,
    initial_points,
    learning_times,
    unlearned_level,
)
from muon_llc.tms.llc import sgld_llc
from muon_llc.tms.model import make_problem, split, tms_loss
from muon_llc.tms.train import train_batched

CFG = {"r": 2, "c": 6, "n": 200, "seeds": 3, "grid": {"slope": [0.0, 0.3, 1.0], "rho": [1.0, 0.5]}}


def test_defaults_are_the_paper_loss_and_slope_one_is_linear():
    p = make_problem(100, seed=0)
    g = torch.Generator().manual_seed(0)
    W, b = torch.randn(2, 6, generator=g, dtype=torch.float64), torch.randn(6, generator=g, dtype=torch.float64)
    assert torch.equal(tms_loss(W, b, p.feat, p.mu), tms_loss(W, b, p.feat, p.mu, 0.0, torch.ones(6)))
    imp = torch.tensor(0.5) ** torch.arange(6)
    x = torch.nn.functional.one_hot(p.feat, 6).double() * p.mu[:, None]
    lin = (imp * (x - (x @ (W.T @ W) + b)) ** 2).sum(-1).mean()
    assert torch.allclose(tms_loss(W, b, p.feat, p.mu, 1.0, imp), lin)


def test_batched_cells_match_single_runs_and_feature_losses_sum_to_the_loss():
    cells = cells_of(CFG)
    assert len(cells) == 6
    prob = family_problem(CFG, cells)
    th = initial_points(CFG, cells)
    assert th.shape == (18, 18) and torch.equal(th[:3], th[3:6])  # same seeds in every cell
    loss = prob.loss(th)
    for run in (0, 4, 10, 17):
        cell = cells.iloc[run // 3]
        W, b = split(th[run], 2, 6)
        single = tms_loss(W, b, prob.feat[run], prob.mu[run], cell.slope, cell.rho ** torch.arange(6))
        assert torch.allclose(loss[run], single)
    fl = feature_losses(prob, th.numpy()[None])
    assert np.allclose(fl.sum(-1)[0], loss.numpy())
    sub = prob.select([4, 17])
    assert torch.allclose(sub.loss(th[[4, 17]]), loss[[4, 17]])


def test_training_and_llc_run_on_the_family():
    cells = cells_of(CFG | {"grid": {"slope": [0.0, 1.0], "rho": 0.5, "init_length": 0.0, "bias_std": 0.0}})
    cfg = CFG | {"seeds": 2}
    prob = family_problem(cfg, cells)
    run = train_batched(prob, initial_points(cfg, cells), {"name": "spectral", "lr": 3e-3, "orth": "exact"},
                        epochs=30, ckpt_every=10)
    fl = feature_losses(prob, run.thetas)
    assert fl.sum(-1)[-1].max() < fl.sum(-1)[0].min()  # every run made progress
    t = learning_times(run.epochs, fl, unlearned_level(prob))
    assert t.shape == (4, 6) and np.isfinite(t[:, 0]).all()  # the most important feature is learned
    res = sgld_llc(prob, run.thetas[-1], n_steps=50, n_chains=2, data_index=np.arange(4))
    assert res.llc.shape == (4,) and np.isfinite(res.llc).all()
    assert np.allclose(res.loss0, fl.sum(-1)[-1], rtol=1e-5)
