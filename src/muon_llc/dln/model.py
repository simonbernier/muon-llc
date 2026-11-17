"""Deep linear network, whitened regression data, and the batched loss used by samplers.

Parameterisation. Widths (H_0, ..., H_L). Weight l has shape (H_l, H_{l-1}); the network
map is W = W_L @ ... @ W_1 of shape (H_L, H_0). Parameters are handled either as a list of
tensors with arbitrary leading batch dims (for training) or as a flat vector theta of
length D = sum_l H_l H_{l-1} with leading batch dims (for samplers).

Loss. L_n(theta) = (1/(2n)) sum_i ||y_i - W x_i||^2, the average Gaussian negative
log-likelihood with unit noise variance. With exactly whitened inputs (X X^T / n = I) and
noiseless targets Y = A* X this equals (1/2) ||W - A*||_F^2 identically, so the
closed-form DLN dynamics apply and the full-batch loss can be evaluated without data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

Tensor = torch.Tensor


# ----------------------------------------------------------------------------- parameters
def layer_shapes(widths: Sequence[int]) -> list[tuple[int, int]]:
    return [(int(widths[l + 1]), int(widths[l])) for l in range(len(widths) - 1)]


def num_params(widths: Sequence[int]) -> int:
    return sum(o * i for o, i in layer_shapes(widths))


def init_weights(
    widths: Sequence[int],
    init_scale: float,
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> list[Tensor]:
    """iid Gaussian init W_l ~ N(0, init_scale^2)."""
    ws = []
    for out_dim, in_dim in layer_shapes(widths):
        w = torch.randn(out_dim, in_dim, generator=generator, dtype=dtype)
        ws.append((init_scale * w).to(device))
    return ws


def flatten(weights: Sequence[Tensor]) -> Tensor:
    """List of (..., H_l, H_{l-1}) -> (..., D). Leading dims must agree."""
    return torch.cat([w.reshape(*w.shape[:-2], -1) for w in weights], dim=-1)


def unflatten(theta: Tensor, widths: Sequence[int]) -> list[Tensor]:
    """(..., D) -> list of (..., H_l, H_{l-1})."""
    ws, off = [], 0
    for out_dim, in_dim in layer_shapes(widths):
        k = out_dim * in_dim
        ws.append(theta[..., off : off + k].reshape(*theta.shape[:-1], out_dim, in_dim))
        off += k
    if off != theta.shape[-1]:
        raise ValueError(f"theta has {theta.shape[-1]} entries, widths need {off}")
    return ws


def product(weights: Sequence[Tensor]) -> Tensor:
    """W = W_L @ ... @ W_1 with broadcasting over leading dims."""
    W = weights[0]
    for w in weights[1:]:
        W = w @ W
    return W


def product_from_flat(theta: Tensor, widths: Sequence[int]) -> Tensor:
    return product(unflatten(theta, widths))


def balanced_weights(
    A: Tensor, widths: Sequence[int], generator: torch.Generator | None = None
) -> list[Tensor]:
    """Weights whose product is exactly A, with every layer carrying s_i^{1/L}.

    A is (H_L, H_0). Requires min(widths) >= rank(A). Used for the static rank-r
    checks (Phase B2): A = top-r truncation of the target gives a rank-r DLN point.
    Hidden layers are padded with zero rows/cols beyond the rank.
    """
    L = len(widths) - 1
    U, S, Vh = torch.linalg.svd(A, full_matrices=True)
    r = int((S > 1e-12 * S.max().clamp_min(1e-30)).sum()) if S.numel() else 0
    root = S[:r] ** (1.0 / L)
    # Random orthogonal bases for hidden layers, so the point is generic on the fibre.
    bases = [Vh.T[:, :r]]  # (H_0, r)
    for l in range(1, L):
        H = widths[l]
        Q, _ = torch.linalg.qr(torch.randn(H, H, generator=generator, dtype=A.dtype))
        bases.append(Q[:, :r])
    bases.append(U[:, :r])  # (H_L, r)
    ws = []
    for l in range(L):
        Bin, Bout = bases[l], bases[l + 1]  # (H_{l}, r), (H_{l+1}, r)
        ws.append(Bout @ torch.diag(root) @ Bin.T)
    return ws


# ----------------------------------------------------------------------------- data
def make_target(
    d_out: int, d_in: int, singular_values: Sequence[float], generator: torch.Generator | None
) -> Tensor:
    """A* = U diag(s) V^T with random orthogonal U, V (seeded through `generator`)."""
    s = torch.as_tensor(list(singular_values), dtype=torch.float64)
    if s.numel() > min(d_out, d_in):
        raise ValueError("more singular values than min(d_out, d_in)")
    U, _ = torch.linalg.qr(torch.randn(d_out, d_out, generator=generator, dtype=torch.float64))
    V, _ = torch.linalg.qr(torch.randn(d_in, d_in, generator=generator, dtype=torch.float64))
    S = torch.zeros(d_out, d_in, dtype=torch.float64)
    S[: s.numel(), : s.numel()] = torch.diag(s)
    return U @ S @ V.T


def make_whitened_inputs(d_in: int, n: int, generator: torch.Generator | None) -> Tensor:
    """X of shape (d_in, n) with X X^T / n = I exactly (Gaussian directions, exact whitening)."""
    if n < d_in:
        raise ValueError("need n >= d_in to whiten")
    Z = torch.randn(n, d_in, generator=generator, dtype=torch.float64)
    Q, _ = torch.linalg.qr(Z)  # (n, d_in), orthonormal columns
    return (Q.T * (n**0.5)).contiguous()


@dataclass
class DLNProblem:
    """Everything the loss needs: widths, data, target. Tensors live on `device`."""

    widths: tuple[int, ...]
    X: Tensor  # (H_0, n)
    Y: Tensor  # (H_L, n)
    A_star: Tensor  # (H_L, H_0)
    noise_var: float = 1.0

    @property
    def n(self) -> int:
        return int(self.X.shape[1])

    @property
    def dim(self) -> int:
        return num_params(self.widths)

    @property
    def device(self) -> torch.device:
        return self.X.device

    @property
    def dtype(self) -> torch.dtype:
        return self.X.dtype

    def to(self, device=None, dtype=None) -> "DLNProblem":
        kw = {}
        if device is not None:
            kw["device"] = device
        if dtype is not None:
            kw["dtype"] = dtype
        return DLNProblem(
            self.widths, self.X.to(**kw), self.Y.to(**kw), self.A_star.to(**kw), self.noise_var
        )

    # -- losses -------------------------------------------------------------------------
    def loss_from_product(self, W: Tensor, idx: Tensor | None = None) -> Tensor:
        """L_n (or minibatch loss) for W of shape (..., H_L, H_0). Returns (...)."""
        if idx is None:
            # Whitened, noiseless: exactly (1/2)||W - A*||^2. Use it (cheap, exact).
            diff = W - self.A_star
            return 0.5 * (diff * diff).sum(dim=(-2, -1)) / self.noise_var
        Xb, Yb = self.X[:, idx], self.Y[:, idx]
        resid = W @ Xb - Yb
        return 0.5 * (resid * resid).sum(dim=(-2, -1)) / (idx.numel() * self.noise_var)

    def loss_from_data(self, W: Tensor) -> Tensor:
        """Full-batch loss evaluated from the data (used in tests to check the identity)."""
        resid = W @ self.X - self.Y
        return 0.5 * (resid * resid).sum(dim=(-2, -1)) / (self.n * self.noise_var)

    def loss(self, theta: Tensor, idx: Tensor | None = None) -> Tensor:
        """Loss for flat parameters theta of shape (..., D). Returns (...)."""
        return self.loss_from_product(product_from_flat(theta, self.widths), idx)

    def loss_weights(self, weights: Sequence[Tensor], idx: Tensor | None = None) -> Tensor:
        return self.loss_from_product(product(weights), idx)

    def loss_and_grad(self, theta: Tensor, idx: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Batched loss and per-sample gradient (chains are independent so sum() works)."""
        theta = theta.detach().requires_grad_(True)
        loss = self.loss(theta, idx)
        (grad,) = torch.autograd.grad(loss.sum(), theta)
        return loss.detach(), grad

    # -- observables ------------------------------------------------------------------
    def product_singular_values(self, theta_or_weights) -> Tensor:
        W = (
            product_from_flat(theta_or_weights, self.widths)
            if torch.is_tensor(theta_or_weights)
            else product(theta_or_weights)
        )
        return torch.linalg.svdvals(W)

    def minibatch_indices(self, batch_size: int, generator: torch.Generator | None) -> Tensor:
        return torch.randperm(self.n, generator=generator, device="cpu")[:batch_size].to(self.device)


def make_problem(
    widths: Sequence[int],
    n: int,
    singular_values: Sequence[float],
    seed: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> DLNProblem:
    """Seeded target + whitened data. Same seed -> same A*, X for every optimizer."""
    g = torch.Generator().manual_seed(int(seed))
    widths = tuple(int(h) for h in widths)
    A = make_target(widths[-1], widths[0], singular_values, g)
    X = make_whitened_inputs(widths[0], n, g)
    Y = A @ X
    return DLNProblem(widths, X.to(device=device, dtype=dtype), Y.to(device=device, dtype=dtype),
                      A.to(device=device, dtype=dtype))
