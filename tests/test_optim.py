import numpy as np
import torch

from muon_llc.optim import (
    KJ_COEFFS,
    SpectralOptimizer,
    build_optimizer,
    newton_schulz,
    ns_scalar_map,
    polar_exact,
)


def test_polar_exact_matches_svd_polar_factor():
    g = torch.Generator().manual_seed(0)
    G = torch.randn(8, 8, generator=g, dtype=torch.float64)
    U, S, Vh = torch.linalg.svd(G)
    assert torch.allclose(polar_exact(G), U @ Vh, atol=1e-12)
    O = polar_exact(G)
    assert torch.allclose(O.T @ O, torch.eye(8, dtype=torch.float64), atol=1e-10)


def test_polar_exact_keeps_null_directions_at_zero():
    g = torch.Generator().manual_seed(0)
    A = torch.randn(8, 3, generator=g, dtype=torch.float64)
    B = torch.randn(3, 8, generator=g, dtype=torch.float64)
    O = polar_exact(A @ B)
    assert torch.allclose(torch.linalg.svdvals(O)[:3], torch.ones(3, dtype=torch.float64), atol=1e-10)
    assert torch.all(torch.linalg.svdvals(O)[3:] < 1e-10)


def test_newton_schulz_flattens_well_conditioned_matrix():
    g = torch.Generator().manual_seed(0)
    G = torch.randn(8, 8, generator=g, dtype=torch.float64)
    X = newton_schulz(G, steps=10, dtype=torch.float64)
    sv = torch.linalg.svdvals(X)
    # KJ's quintic converges to singular values in a band around 1, not to exactly 1
    assert torch.all(sv > 0.65) and torch.all(sv < 1.35)
    # and preserves the singular vectors: X should be close to the polar factor in direction
    U, S, Vh = torch.linalg.svd(G)
    assert torch.allclose(torch.sign((X * (U @ Vh)).sum()), torch.tensor(1.0, dtype=torch.float64))


def test_newton_schulz_small_sigma_gain():
    sig = torch.tensor([1e-4, 1e-3], dtype=torch.float64)
    out = ns_scalar_map(sig, 1)
    assert torch.allclose(out / sig, torch.full_like(sig, KJ_COEFFS[0]), rtol=1e-5)
    # after k steps the gain is a^k while still in the linear regime
    out3 = ns_scalar_map(torch.tensor([1e-6], dtype=torch.float64), 3)
    assert abs(out3.item() / 1e-6 - KJ_COEFFS[0] ** 3) / KJ_COEFFS[0] ** 3 < 1e-4


def test_newton_schulz_leaves_tiny_singular_values_unflattened():
    # matrix with singular values (1, a^-(k+2)): after k steps the small one is still << 1
    g = torch.Generator().manual_seed(0)
    Q1, _ = torch.linalg.qr(torch.randn(8, 8, generator=g, dtype=torch.float64))
    Q2, _ = torch.linalg.qr(torch.randn(8, 8, generator=g, dtype=torch.float64))
    k = 3
    small = KJ_COEFFS[0] ** (-(k + 2))
    S = torch.zeros(8, dtype=torch.float64)
    S[0], S[1] = 1.0, small
    G = Q1 @ torch.diag(S) @ Q2.T
    X = newton_schulz(G, steps=k, dtype=torch.float64)
    sv = torch.linalg.svdvals(X)
    assert sv[0] > 0.65
    assert sv[1] < 0.2


def test_spectral_step_on_diagonal_toy_grows_all_singular_values_by_lr():
    # one layer W (8x8) with gradient W - A*: spectral step should add lr to each singular
    # value direction of the gradient's polar factor
    torch.manual_seed(0)
    A = torch.diag(torch.tensor([4.0, 2.0, 1.0, 0.5, 0, 0, 0, 0], dtype=torch.float64))
    W = torch.zeros(8, 8, dtype=torch.float64, requires_grad=True)
    opt = SpectralOptimizer([W], lr=0.1, orth="exact")
    for _ in range(3):
        opt.zero_grad()
        loss = 0.5 * ((W - A) ** 2).sum()
        loss.backward()
        opt.step()
    sv = torch.linalg.svdvals(W.detach())
    assert torch.allclose(sv[:4], torch.full((4,), 0.3, dtype=torch.float64), atol=1e-10)
    assert torch.all(sv[4:] < 1e-10)


def test_weight_decay_fixed_point_is_one_over_wd():
    torch.manual_seed(0)
    A = 10.0 * torch.eye(4, dtype=torch.float64)  # target far above the cap
    W = (0.01 * torch.randn(4, 4, dtype=torch.float64)).requires_grad_(True)
    wd, lr = 0.5, 0.05
    opt = SpectralOptimizer([W], lr=lr, orth="exact", weight_decay=wd)
    for _ in range(2000):
        opt.zero_grad()
        (0.5 * ((W - A) ** 2).sum()).backward()
        opt.step()
    norm = torch.linalg.matrix_norm(W.detach(), ord=2).item()
    assert abs(norm - 1 / wd) < 0.1  # ±lr oscillation around the fixed point


def test_momentum_zero_exact_matches_manual_spectral_step():
    torch.manual_seed(0)
    A = torch.randn(6, 6, dtype=torch.float64)
    W = torch.randn(6, 6, dtype=torch.float64).requires_grad_(True)
    W0 = W.detach().clone()
    opt = SpectralOptimizer([W], lr=0.2, orth="exact")
    opt.zero_grad()
    (0.5 * ((W - A) ** 2).sum()).backward()
    opt.step()
    G = W0 - A
    assert torch.allclose(W.detach(), W0 - 0.2 * polar_exact(G), atol=1e-12)


def test_muon_nesterov_momentum_buffer_semantics():
    torch.manual_seed(0)
    W = torch.randn(4, 4, dtype=torch.float64).requires_grad_(True)
    opt = SpectralOptimizer([W], lr=0.1, momentum=0.9, nesterov=True, orth="exact")
    grads = []
    for _ in range(2):
        opt.zero_grad()
        (W**2).sum().backward()
        grads.append(W.grad.clone())
        opt.step()
    buf = opt.state[W]["momentum_buffer"]
    # buf_t = lerp(buf_{t-1}, g_t, 1-mu) = mu*buf + (1-mu)*g
    expect = 0.9 * (0.1 * grads[0]) + 0.1 * grads[1]
    assert torch.allclose(buf, expect, atol=1e-12)


def test_build_optimizer_gd():
    W = torch.zeros(2, 2, requires_grad=True)
    opt = build_optimizer([W], {"name": "gd", "lr": 0.1})
    assert isinstance(opt, torch.optim.SGD)
    opt2 = build_optimizer([W], {"name": "spectral", "lr": 0.1, "orth": "ns", "ns_steps": 5})
    assert isinstance(opt2, SpectralOptimizer)
    assert np.isclose(opt2.param_groups[0]["ns_coeffs"][0], 3.4445)
