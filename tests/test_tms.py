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


def test_default_bias_rule_muon_adamw_spectral_sgd():
    from muon_llc.optim import SpectralOptimizer
    from muon_llc.tms.train import build_optimizers

    W = torch.zeros(2, 6, requires_grad=True)
    b = torch.zeros(6, requires_grad=True)
    muon = build_optimizers(W, b, dict(name="muon", lr=3e-4, orth="ns", momentum=0.95, nesterov=True))
    assert isinstance(muon[0], SpectralOptimizer) and isinstance(muon[1], torch.optim.AdamW)
    assert muon[1].param_groups[0]["lr"] == 1e-3 and muon[1].param_groups[0]["weight_decay"] == 0.0
    spec = build_optimizers(W, b, dict(name="spectral", lr=3e-4, orth="exact"))
    assert isinstance(spec[1], torch.optim.SGD)
    explicit = build_optimizers(W, b, dict(name="muon", lr=3e-4, orth="ns", bias=dict(name="gd", lr=0.005)))
    assert isinstance(explicit[1], torch.optim.SGD)


def test_llc_jumps_medians_inside_plateaus():
    from muon_llc.tms.transitions import llc_jumps

    #        ckpt: 0..5 "4+-", 6..7 other, 8..13 "4", 14 other, 15..19 "4" (same plateau), 20..24 "5"
    lab = np.array(["4+-"] * 6 + ["other"] * 2 + ["4"] * 6 + ["other"] + ["4"] * 5 + ["5"] * 5, dtype=object)
    labels = np.stack([lab, np.array(["6"] * 25, dtype=object)], axis=1)  # run 1 never moves
    ck_idx = np.arange(0, 25, 3)  # 0 3 6 9 12 15 18 21 24
    llc = np.zeros((len(ck_idx), 2))
    llc[:, 0] = [1.0, 3.0, 99.0, 4.0, np.nan, 5.0, 9.0, 7.0, 8.0]
    rows = llc_jumps(labels, ck_idx, llc, seeds=[7, 11])
    assert [(r["seed"], r["alpha"], r["beta"]) for r in rows] == [(7, "4+-", "4"), (7, "4", "5")]
    # "4+-" holds ckpts 0, 3; "4" holds 9, (12 is nan), 15, 18; "5" holds 21, 24; ckpt 6 is off-plateau
    assert rows[0]["lam_before"] == pytest.approx(2.0) and rows[0]["lam_after"] == pytest.approx(5.0)
    assert rows[0]["rise"] == pytest.approx(3.0)
    assert rows[1]["lam_before"] == pytest.approx(5.0) and rows[1]["rise"] == pytest.approx(2.5)
    # no estimated checkpoint inside a plateau -> nan, not an error
    sparse = llc_jumps(labels, np.array([0, 24]), llc[[0, 8]], seeds=[7, 11])
    assert np.isnan(sparse[0]["lam_after"]) and np.isnan(sparse[0]["rise"])


def test_bah_writes_trajectory_llc_jumps(tmp_path):
    import json

    import pandas as pd

    from muon_llc.tms.experiments import bah

    cps = by_name()
    K = 12
    thetas = np.stack([np.stack([cps["4+"].theta] * 6 + [cps["5"].theta] * 6),  # run 0: 4+ -> 5
                       np.stack([cps["6"].theta] * K),  # run 1: no transition
                       np.stack([cps["4"].theta] * 6 + [cps["5"].theta] * 6)], axis=1)  # run 2: not estimated
    for name, opt in (("sgd", {"name": "gd", "lr": 0.005}), ("adamw_lr3e-4", {"name": "adamw", "lr": 3e-4})):
        d = tmp_path / "arms" / name
        d.mkdir(parents=True)
        np.savez(d / "run.npz", epochs=10 * np.arange(K), thetas=thetas.astype(np.float32),
                 path_len=np.cumsum(np.ones((K, 3)), axis=0), steps_per_epoch=50)
        (d / "config.json").write_text(json.dumps({"opt": opt, "n": 100, "c": 6, "runs": 3, "data_seed": 0}),
                                       encoding="utf-8")
    d = tmp_path / "arms" / "sgd"  # only SGD has estimates
    ck = np.arange(0, K, 3)  # 0 3 | 6 9
    np.savez(d / "llc.npz", ck_idx=ck, run_idx=np.arange(2), llc=np.array([[4.0, 8.0], [5.0, 8.0], [7.0, 8.0],
                                                                         [7.5, 8.0]]))
    out = tmp_path / "out"
    bah.main(["--arms", str(tmp_path / "arms"), "--out", str(out), "--table", str(tmp_path / "none.csv")])
    jumps = pd.read_csv(out / "trajectory_llc_jumps.csv", dtype={"alpha": str, "beta": str})
    assert list(jumps.columns) == ["arm", "seed", "alpha", "beta", "lam_before", "lam_after", "rise"]
    assert len(jumps) == 1
    row = jumps.iloc[0]
    assert (row.arm, row.seed, row.alpha, row.beta) == ("sgd", 0, "4+", "5")
    assert (row.lam_before, row.lam_after, row.rise) == pytest.approx((4.5, 7.25, 2.75))
    assert len(pd.read_csv(out / "transitions.csv")) == 4  # both arms: 4+ -> 5 and 4 -> 5


needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@needs_cuda
@pytest.mark.parametrize("opt", [
    dict(name="gd", lr=0.005),
    dict(name="adamw", lr=3e-4, weight_decay=0.01),
    dict(name="spectral", lr=3e-4, orth="exact", bias=dict(name="gd", lr=0.005)),
    dict(name="muon", lr=3e-4, orth="ns", ns_steps=5, momentum=0.95, nesterov=True, ns_dtype=None),
], ids=lambda o: o["name"])
@pytest.mark.parametrize("per_run", [True, False], ids=["per-run data", "shared data"])
def test_training_agrees_between_cpu_and_cuda(opt, per_run):
    from muon_llc.tms.train import train_batched

    th = init_kgon(5, seed=3)
    runs = []
    for device in ("cpu", "cuda"):
        p = make_problem(200, seed=11, n_datasets=5 if per_run else None, device=device)
        assert p.device.type == device
        runs.append(train_batched(p, th, opt, epochs=3, ckpt_every=1, seed=2))
    cpu, gpu = runs
    assert isinstance(gpu.thetas, np.ndarray) and gpu.thetas.shape == cpu.thetas.shape == (4, 5, 18)
    assert np.abs(cpu.thetas - gpu.thetas).max() < 1e-9
    assert np.abs(cpu.path_len - gpu.path_len).max() < 1e-9
    assert not np.array_equal(cpu.thetas[0], cpu.thetas[-1])


@needs_cuda
def test_float32_newton_schulz_agrees_between_devices_to_single_precision():
    """Standard Muon orthogonalises in float32; CPU and CUDA round those products differently and
    the small-singular-value gain of Newton-Schulz amplifies it, so the agreement is looser."""
    from muon_llc.tms.train import train_batched

    opt = dict(name="muon", lr=3e-4, orth="ns", ns_steps=5, momentum=0.95, nesterov=True)
    th = init_kgon(5, seed=3)
    out = [train_batched(make_problem(200, seed=11, n_datasets=5, device=d), th, opt, epochs=3, ckpt_every=1,
                         seed=2).thetas for d in ("cpu", "cuda")]
    assert np.abs(out[0] - out[1]).max() < 1e-5


@needs_cuda
def test_sgld_llc_agrees_between_cpu_and_cuda():
    from muon_llc.tms.llc import sgld_llc

    centres = np.stack([by_name()[k].theta for k in ("6", "5", "4")])
    res = []
    for device in ("cpu", "cuda"):
        p = make_problem(300, seed=5, n_datasets=3, device=device)
        res.append(sgld_llc(p, centres, n_steps=40, n_chains=2, data_index=np.arange(3), seed=1))
    assert np.allclose(res[0].llc_chain, res[1].llc_chain, rtol=0, atol=1e-8)
    assert np.allclose(res[0].loss0, res[1].loss0, rtol=0, atol=1e-12)


@needs_cuda
def test_polar_exact_on_cuda_matches_cpu():
    from muon_llc.optim import polar_exact

    G = torch.randn(7, 2, 6, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    O = polar_exact(G.cuda())
    assert O.is_cuda and torch.allclose(O.cpu(), polar_exact(G), atol=1e-13)
    big = torch.randn(80, 80, dtype=torch.float64, generator=torch.Generator().manual_seed(1))  # on-device SVD
    assert torch.allclose(polar_exact(big.cuda()).cpu(), polar_exact(big), atol=1e-9)
