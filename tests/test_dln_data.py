import torch

from muon_llc.dln.model import (
    balanced_weights,
    flatten,
    init_weights,
    make_problem,
    num_params,
    product,
    unflatten,
)


def test_whitening_and_loss_identity():
    p = make_problem([8, 8, 8, 8], 512, [4, 2, 1, 0.5], seed=3, dtype=torch.float64)
    XXt = p.X @ p.X.T / p.n
    assert torch.allclose(XXt, torch.eye(8, dtype=torch.float64), atol=1e-10)
    g = torch.Generator().manual_seed(0)
    W = product(init_weights(p.widths, 0.5, g, dtype=torch.float64))
    assert torch.allclose(p.loss_from_data(W), p.loss_from_product(W), atol=1e-10)
    assert torch.allclose(p.loss_from_product(W), 0.5 * ((W - p.A_star) ** 2).sum(), atol=1e-12)


def test_target_spectrum():
    p = make_problem([8, 8, 8, 8], 256, [4, 2, 1, 0.5], seed=1, dtype=torch.float64)
    sv = torch.linalg.svdvals(p.A_star)
    assert torch.allclose(sv[:4], torch.tensor([4.0, 2.0, 1.0, 0.5], dtype=torch.float64), atol=1e-10)
    assert torch.all(sv[4:] < 1e-10)


def test_flatten_roundtrip_and_batched_loss():
    widths = [8, 6, 8]
    p = make_problem(widths, 128, [3, 1], seed=0, dtype=torch.float64)
    g = torch.Generator().manual_seed(1)
    ws = init_weights(widths, 0.3, g, dtype=torch.float64)
    theta = flatten(ws)
    assert theta.numel() == num_params(widths)
    back = unflatten(theta, widths)
    assert all(torch.equal(a, b) for a, b in zip(ws, back))
    # batched chains: loss(theta[C, D]) equals per-chain losses
    thetas = torch.stack([theta, 2 * theta, 0 * theta])
    L = p.loss(thetas)
    assert L.shape == (3,)
    assert torch.allclose(L[0], p.loss_weights(ws))
    assert torch.allclose(L[2], 0.5 * (p.A_star**2).sum())


def test_balanced_weights_have_exact_product_and_equal_layer_spectra():
    p = make_problem([8, 8, 8, 8], 128, [4, 2, 1, 0.5], seed=0, dtype=torch.float64)
    g = torch.Generator().manual_seed(0)
    ws = balanced_weights(p.A_star, p.widths, g)
    assert torch.allclose(product(ws), p.A_star, atol=1e-10)
    spectra = [torch.linalg.svdvals(w)[:4] for w in ws]
    for s in spectra[1:]:
        assert torch.allclose(s, spectra[0], atol=1e-10)
    assert torch.allclose(spectra[0], torch.tensor([4.0, 2.0, 1.0, 0.5], dtype=torch.float64) ** (1 / 3))


def test_minibatch_loss_unbiased_on_average():
    p = make_problem([8, 8, 8, 8], 1024, [4, 2, 1, 0.5], seed=0, dtype=torch.float64)
    g = torch.Generator().manual_seed(0)
    W = product(init_weights(p.widths, 0.4, g, dtype=torch.float64))
    full = p.loss_from_product(W)
    mb = torch.stack([p.loss_from_product(W, p.minibatch_indices(128, g)) for _ in range(200)])
    assert abs(mb.mean() - full) < 0.05 * full
