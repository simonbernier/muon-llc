"""TMS autoencoder f(x) = ReLU(W^T W x + b), W in R^{r x c}, b in R^c, on one-hot-times-length data.

Data. q(x): pick a coordinate i uniformly, a length mu ~ U[0, 1], return mu * e_i. A dataset is
stored as (feat, mu) with feat in {0..c-1}^n and mu in [0,1]^n.

Parameters. Either (W, b) with arbitrary leading batch dims, or flat theta (..., r*c + c) with W
row-major first.

Loss. loss = mean_n sum_j (x_j - f_j(x))^2, the paper's L (no factor 1/2, no division by c).
`TMSProblem.loss_scale` multiplies it (1/c reproduces torch's MSELoss used in the reference
notebook). LLC estimation always uses the unscaled paper loss.

Family. Two optional changes turn the model into a family that contains a tied linear network:

    f(x) = leaky_slope(W^T W x + b),   loss = mean_n sum_j importance_j (x_j - f_j(x))^2

`slope` is the gradient of the activation for negative arguments (0 = ReLU = the paper's model,
1 = linear) and `importance` weights the output coordinates (None = all 1 = the paper's loss).
Both may carry leading run dimensions, slope as (R,) and importance as (R, c), so that runs with
different values train in one batch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

Tensor = torch.Tensor


def split(theta: Tensor, r: int, c: int) -> tuple[Tensor, Tensor]:
    W = theta[..., : r * c].reshape(*theta.shape[:-1], r, c)
    return W, theta[..., r * c:]


def join(W: Tensor, b: Tensor) -> Tensor:
    return torch.cat([W.reshape(*W.shape[:-2], -1), b], dim=-1)


def sample_data(n: int, c: int, generator: torch.Generator | None = None,
                dtype=torch.float64) -> tuple[Tensor, Tensor]:
    feat = torch.randint(0, c, (n,), generator=generator)
    mu = torch.rand(n, generator=generator, dtype=dtype)
    return feat, mu


def _per_run(x, dtype, device, trailing: int):
    """A float, or a tensor whose leading dims are run dims, with `trailing` singleton dims added."""
    if x is None or isinstance(x, (int, float)):
        return x
    x = torch.as_tensor(x, dtype=dtype, device=device)
    return x.reshape(*x.shape, *([1] * trailing))


def tms_loss(W: Tensor, b: Tensor, feat: Tensor, mu: Tensor, slope=0.0, importance=None) -> Tensor:
    """Loss of the family (the paper's loss for slope 0 and no importance). W (..., r, c), b (..., c);
    feat, mu of shape (n,) or (..., n); slope a float or (...,); importance None, (c,) or (..., c).
    Returns (...)."""
    G = W.mT @ W  # (..., c, c)
    c = G.shape[-1]
    if feat.dim() == 1:
        rows = G[..., feat, :]  # (..., n, c)
    else:
        lead = torch.broadcast_shapes(G.shape[:-2], feat.shape[:-1])
        feat = feat.expand(*lead, feat.shape[-1])
        mu = mu.expand(*lead, mu.shape[-1])
        rows = torch.gather(G.expand(*lead, c, c), -2, feat.unsqueeze(-1).expand(*feat.shape, c))
    z = mu.unsqueeze(-1) * rows + b.unsqueeze(-2)
    if isinstance(slope, (int, float)) and slope == 0.0:
        out = torch.relu(z)
    else:
        out = torch.where(z > 0, z, _per_run(slope, z.dtype, z.device, 2) * z)
    target = torch.nn.functional.one_hot(feat, c).to(out.dtype) * mu.unsqueeze(-1)
    sq = (target - out) ** 2
    if importance is not None:
        sq = sq * torch.as_tensor(importance, dtype=sq.dtype, device=sq.device).unsqueeze(-2)
    return sq.sum(-1).mean(-1)


@dataclass
class TMSProblem:
    feat: Tensor  # (n,) one dataset, or (R, n) one dataset per run
    mu: Tensor  # same shape
    r: int = 2
    c: int = 6
    loss_scale: float = 1.0
    slope: float | Tensor = 0.0  # float, or (R,) one per run
    importance: Tensor | None = None  # (c,), or (R, c) one per run

    @property
    def n(self) -> int:
        return int(self.feat.shape[-1])

    @property
    def per_run(self) -> bool:
        return self.feat.dim() == 2

    def select(self, index) -> "TMSProblem":
        """Datasets of the given runs (index: (B,) long); identity for a shared dataset."""
        if not self.per_run:
            return self
        index = torch.as_tensor(index, dtype=torch.long, device=self.device)
        return TMSProblem(self.feat[index], self.mu[index], self.r, self.c, self.loss_scale,
                          self._take(self.slope, index, 1), self._take(self.importance, index, 2))

    @staticmethod
    def _take(x, index, ndim: int):
        """Rows `index` of a per-run value (a tensor with `ndim` dims); shared values pass through."""
        return x[index] if isinstance(x, Tensor) and x.dim() == ndim else x

    def to(self, device) -> "TMSProblem":
        """The same problem with its data on `device` (no copy if it is already there)."""
        move = lambda x: x.to(device) if isinstance(x, Tensor) else x  # noqa: E731
        return TMSProblem(self.feat.to(device), self.mu.to(device), self.r, self.c, self.loss_scale,
                          move(self.slope), move(self.importance))

    @property
    def dim(self) -> int:
        return self.r * self.c + self.c

    @property
    def device(self) -> torch.device:
        return self.mu.device

    @property
    def dtype(self) -> torch.dtype:
        return self.mu.dtype

    def loss_wb(self, W: Tensor, b: Tensor, idx: Tensor | None = None) -> Tensor:
        feat, mu = (self.feat, self.mu) if idx is None else (self.feat[..., idx], self.mu[..., idx])
        return self.loss_on(W, b, feat, mu)

    def loss_on(self, W: Tensor, b: Tensor, feat: Tensor, mu: Tensor) -> Tensor:
        """Loss on the given data (for minibatches gathered by the caller)."""
        return self.loss_scale * tms_loss(W, b, feat, mu, self.slope, self.importance)

    def loss(self, theta: Tensor, idx: Tensor | None = None) -> Tensor:
        W, b = split(theta, self.r, self.c)
        return self.loss_wb(W, b, idx)

    def loss_and_grad(self, theta: Tensor, idx: Tensor | None = None) -> tuple[Tensor, Tensor]:
        theta = theta.detach().requires_grad_(True)
        loss = self.loss(theta, idx)
        (grad,) = torch.autograd.grad(loss.sum(), theta)
        return loss.detach(), grad

    def minibatch_indices(self, batch_size: int, generator: torch.Generator | None) -> Tensor:
        return torch.randperm(self.n, generator=generator, device="cpu")[:batch_size].to(self.device)


def make_problem(n: int, c: int = 6, r: int = 2, seed: int = 0, loss_scale: float = 1.0,
                 dtype=torch.float64, n_datasets: int | None = None, device=None) -> TMSProblem:
    """One dataset (seed), or `n_datasets` datasets with seeds seed, seed+1, ... (one per run).

    The data are always drawn from CPU generators and then moved to `device` (None = CPU), so a
    seed gives the same dataset on every device."""
    if n_datasets is None:
        feat, mu = sample_data(n, c, torch.Generator().manual_seed(int(seed)), dtype)
        problem = TMSProblem(feat, mu, r, c, loss_scale)
    else:
        pairs = [sample_data(n, c, torch.Generator().manual_seed(int(seed) + j), dtype) for j in range(n_datasets)]
        problem = TMSProblem(torch.stack([p[0] for p in pairs]), torch.stack([p[1] for p in pairs]), r, c,
                             loss_scale)
    return problem if device is None else problem.to(device)


def kgon_vertices(k: int, rot: float = 0.0, pad_to: int | None = None, length: float = 1.0) -> np.ndarray:
    ang = np.linspace(0, 2 * np.pi, k, endpoint=False) + rot
    W = np.vstack([np.cos(ang), np.sin(ang)]) * length
    if pad_to is not None and k < pad_to:
        W = np.hstack([W, np.zeros((2, pad_to - k))])
    return W


def init_kgon(n_runs: int, c: int = 6, k: int = 4, length: float = 0.9, noise: float = 0.01,
              bias_std: float = 1.0, seed: int = 0, dtype=torch.float64) -> Tensor:
    """Reference-notebook init (devinterp examples/tms.ipynb `generate_init_param`): a regular
    k-gon of the given length at a random rotation, zero-padded to c columns, plus N(0, noise^2)
    on W; bias ~ N(0, bias_std^2). Returns flat theta (n_runs, 2c + c). Run j uses seed + j."""
    out = np.empty((n_runs, 3 * c))
    for j in range(n_runs):
        rng = np.random.default_rng(int(seed) + j)
        rot = rng.uniform(0, 2 * np.pi)
        W = kgon_vertices(k, rot, c, length) + noise * rng.normal(size=(2, c))
        b = bias_std * rng.normal(size=c)
        out[j] = np.concatenate([W.reshape(-1), b])
    return torch.tensor(out, dtype=dtype)
