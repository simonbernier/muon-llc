"""Exact population loss of TMS in the high-sparsity limit (equivalent to Lemma 3.1 of the paper).

    L(W, b) = (1/c) sum_i int_0^1 [ (mu - relu(g_ii mu + b_i))^2 + sum_{j != i} relu(g_ij mu + b_j)^2 ] dmu

with g = W^T W. Each relu(a mu + b) is active on one sub-interval [lo, hi] of [0, 1], so every
term is a cubic polynomial evaluated at the interval ends. Batched over leading dims and
differentiable almost everywhere (autograd gives the gradient used in the critical-point tests).
"""

from __future__ import annotations

import torch

Tensor = torch.Tensor


def population_loss(W: Tensor, b: Tensor) -> Tensor:
    G = W.mT @ W  # (..., c, c): a[i, j] = g_ij is the slope of output j on input i
    c = G.shape[-1]
    a = G
    bb = b.unsqueeze(-2).expand_as(a)  # bias of output j
    tiny = 1e-300
    safe_a = torch.where(a.abs() > tiny, a, torch.ones_like(a))
    mu0 = (-bb / safe_a).clamp(0.0, 1.0)
    zero, one = torch.zeros_like(a), torch.ones_like(a)
    flat_on = (a.abs() <= tiny) & (bb > 0)
    lo = torch.where(a > tiny, mu0, zero)
    hi = torch.where(a > tiny, one, torch.where(a < -tiny, mu0, torch.where(flat_on, one, zero)))

    def F2(m):  # antiderivative of (a m + b)^2
        return a * a * m**3 / 3 + a * bb * m**2 + bb * bb * m

    def F1(m):  # antiderivative of 2 m (a m + b)
        return 2 * (a * m**3 / 3 + bb * m**2 / 2)

    sq = F2(hi) - F2(lo)  # int relu^2
    cross = F1(hi) - F1(lo)  # int 2 mu relu
    eye = torch.eye(c, dtype=W.dtype, device=W.device)
    per_input = sq.sum(-1) + ((1.0 / 3.0 - cross) * eye).sum(-1)
    return per_input.sum(-1) / c


def population_loss_flat(theta: Tensor, r: int = 2, c: int = 6) -> Tensor:
    W = theta[..., : r * c].reshape(*theta.shape[:-1], r, c)
    return population_loss(W, theta[..., r * c:])
