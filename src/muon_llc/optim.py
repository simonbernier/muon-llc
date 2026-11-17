"""Muon-family optimizer with every ingredient exposed, plus plain GD.

SpectralOptimizer(lr, momentum=0, nesterov=False, orth="exact"|"ns", ns_steps=5,
                  ns_coeffs=(3.4445, -4.7750, 2.0315), ns_dtype=float32, weight_decay=0,
                  shape_scale=False)

  momentum = 0, orth = "exact", weight_decay = 0   -> pure spectral GD (idealised Muon)
  momentum = 0.95, nesterov, orth = "ns", k = 5, wd -> standard Muon (Keller Jordan's recipe)

Semantics follow Keller Jordan's muon.py: momentum buffer `buf.lerp_(g, 1 - mu)`, Nesterov
`g.lerp(buf, mu)`, Frobenius normalisation with +1e-7 before Newton-Schulz, transpose for
tall matrices, decoupled weight decay `p *= 1 - lr * wd` applied before the update.
Two deliberate differences: Newton-Schulz runs in float32 by default (bf16 adds its own noise
floor that would confound the flattening-threshold sweep), and the sqrt(rows/cols) shape
factor is off by default (all layers here are square).

Newton-Schulz acts on singular values independently: sigma -> a sigma + b sigma^3 + c sigma^5.
For small sigma this is sigma -> a sigma with a = 3.4445, which is the A5 prediction that
singular values below ~a^-k of the Frobenius-normalised update are not flattened.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch

KJ_COEFFS = (3.4445, -4.7750, 2.0315)


def newton_schulz(
    G: torch.Tensor,
    steps: int,
    coeffs: Sequence[float] = KJ_COEFFS,
    eps: float = 1e-7,
    dtype: torch.dtype | None = torch.float32,
) -> torch.Tensor:
    """Approximate polar factor of G via the quintic Newton-Schulz iteration (batched ok)."""
    a, b, c = coeffs
    X = G if dtype is None else G.to(dtype)
    transposed = G.size(-2) > G.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(int(steps)):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X.to(G.dtype)


SVD_ON_CPU_MAX_DIM = 64  # CUDA matrices up to this size are decomposed on the CPU (see polar_exact)


def polar_exact(G: torch.Tensor, rtol: float = 1e-12) -> torch.Tensor:
    """Exact orthogonalisation U V^T restricted to the numerically nonzero singular values.

    For small matrices on a GPU the SVD itself runs on the CPU and the factor is moved back:
    cuSOLVER's batched SVD of tiny matrices is 3-4x slower than LAPACK plus the two copies
    (measured for 200 x (2, 6) and for (8, 8) .. (64, 64)), and this makes the result the same as
    a CPU run's for the same input. On the CPU nothing changes."""
    if G.device.type != "cpu" and max(G.shape[-2:]) <= SVD_ON_CPU_MAX_DIM:
        return polar_exact(G.cpu(), rtol).to(G.device)
    U, S, Vh = torch.linalg.svd(G, full_matrices=False)
    keep = (S > rtol * S.amax(dim=-1, keepdim=True)).to(G.dtype)
    return (U * keep.unsqueeze(-2)) @ Vh


def ns_scalar_map(sigma: torch.Tensor, steps: int, coeffs: Sequence[float] = KJ_COEFFS):
    """The map a singular value undergoes under `steps` NS iterations (no normalisation)."""
    a, b, c = coeffs
    x = sigma
    for _ in range(int(steps)):
        x = a * x + b * x**3 + c * x**5
    return x


class SpectralOptimizer(torch.optim.Optimizer):
    """Muon family for matrix parameters (leading batch dims = independent matrices)."""

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 1e-2,
        momentum: float = 0.0,
        nesterov: bool = False,
        orth: str = "exact",
        ns_steps: int = 5,
        ns_coeffs: Sequence[float] = KJ_COEFFS,
        ns_dtype: str | None = "float32",
        ns_eps: float = 1e-7,
        weight_decay: float = 0.0,
        shape_scale: bool = False,
    ):
        if orth not in ("exact", "ns"):
            raise ValueError("orth must be 'exact' or 'ns'")
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in [0, 1)")
        defaults = dict(
            lr=lr, momentum=momentum, nesterov=nesterov, orth=orth, ns_steps=ns_steps,
            ns_coeffs=tuple(ns_coeffs), ns_dtype=ns_dtype, ns_eps=ns_eps,
            weight_decay=weight_decay, shape_scale=shape_scale,
        )
        super().__init__(params, defaults)

    @staticmethod
    def orthogonalise(u: torch.Tensor, group: dict) -> torch.Tensor:
        if group["orth"] == "exact":
            return polar_exact(u)
        dtype = None if group["ns_dtype"] is None else getattr(torch, group["ns_dtype"])
        return newton_schulz(u, group["ns_steps"], group["ns_coeffs"], group["ns_eps"], dtype)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, mu, wd = group["lr"], group["momentum"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.ndim < 2:
                    raise ValueError("SpectralOptimizer expects matrices (leading batch dims allowed)")
                g = p.grad
                if mu > 0:
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.lerp_(g, 1.0 - mu)
                    u = g.lerp(buf, mu) if group["nesterov"] else buf
                else:
                    u = g
                O = self.orthogonalise(u, group)
                if group["shape_scale"]:
                    O = O * max(1.0, O.size(-2) / O.size(-1)) ** 0.5
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                p.add_(O, alpha=-lr)
        return loss


def build_optimizer(params: Iterable[torch.Tensor], cfg: dict) -> torch.optim.Optimizer:
    """cfg: {"name": "gd" | "spectral", "lr": ..., other SpectralOptimizer kwargs}."""
    cfg = dict(cfg)
    name = cfg.pop("name")
    params = list(params)
    if name == "gd":
        return torch.optim.SGD(params, lr=cfg["lr"], momentum=cfg.get("momentum", 0.0),
                               nesterov=cfg.get("nesterov", False),
                               weight_decay=cfg.get("weight_decay", 0.0))
    if name == "adamw":
        return torch.optim.AdamW(params, lr=cfg["lr"], betas=tuple(cfg.get("betas", (0.9, 0.999))),
                                 eps=cfg.get("eps", 1e-8), weight_decay=cfg.get("weight_decay", 0.0))
    if name in ("spectral", "muon"):
        return SpectralOptimizer(params, **cfg)
    raise ValueError(f"unknown optimizer {name!r}")
