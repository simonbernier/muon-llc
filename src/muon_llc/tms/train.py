"""Batched TMS training: R independent runs live in one tensor (W: (R, r, c), b: (R, c)).

Every run has its own initial point and its own minibatch order; the loss is summed over runs so
gradients do not mix, and every optimizer used here acts per run (SGD/AdamW elementwise, the
spectral optimizers per matrix). The same `seed` gives the same minibatch order for every
optimizer, so arms differ only in the update rule.

Optimizer config:
    {"name": "sgd" | "adamw", "lr": ..., ...}                     both W and b under one rule
    {"name": "spectral" | "muon", "lr": ..., ..., "bias": {...}}  W under SpectralOptimizer, b under
                                                                  the `bias` rule (default: SGD at
                                                                  `bias_lr`, no momentum)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..optim import build_optimizer
from .model import TMSProblem, join, split, tms_loss


@dataclass
class TMSRun:
    epochs: np.ndarray  # (K,) checkpoint epochs
    thetas: np.ndarray  # (K, R, D)
    path_len: np.ndarray  # (K, R) cumulative parameter path length
    steps_per_epoch: int


def build_optimizers(W, b, cfg: dict) -> list[torch.optim.Optimizer]:
    cfg = dict(cfg)
    bias_cfg = cfg.pop("bias", None)
    if cfg["name"] in ("spectral", "muon"):
        bias_cfg = dict(bias_cfg or {"name": "gd", "lr": cfg["lr"]})
        return [build_optimizer([W], cfg), build_optimizer([b], bias_cfg)]
    return [build_optimizer([W, b], cfg)]


def train_batched(problem: TMSProblem, theta0: torch.Tensor, opt_cfg: dict, epochs: int,
                  batch_size: int = 20, seed: int = 0, ckpt_every: int = 10,
                  verbose: bool = False) -> TMSRun:
    R = theta0.shape[0]
    n, r, c = problem.n, problem.r, problem.c
    W0, b0 = split(theta0.to(problem.dtype), r, c)
    W = W0.clone().requires_grad_(True)
    b = b0.clone().requires_grad_(True)
    opts = build_optimizers(W, b, opt_cfg)
    g = torch.Generator().manual_seed(int(seed))
    spe = n // batch_size
    ck_epochs, cks, plen = [0], [join(W, b).detach().clone()], [torch.zeros(R, dtype=problem.dtype)]
    path = torch.zeros(R, dtype=problem.dtype)
    prev = join(W, b).detach().clone()
    for ep in range(1, epochs + 1):
        perm = torch.argsort(torch.rand(R, n, generator=g), dim=1)[:, : spe * batch_size]
        if problem.per_run:
            feat_ep = torch.gather(problem.feat, 1, perm).view(R, spe, batch_size)
            mu_ep = torch.gather(problem.mu, 1, perm).view(R, spe, batch_size)
        else:
            feat_ep = problem.feat[perm].view(R, spe, batch_size)
            mu_ep = problem.mu[perm].view(R, spe, batch_size)
        for s in range(spe):
            for o in opts:
                o.zero_grad(set_to_none=True)
            loss = problem.loss_scale * tms_loss(W, b, feat_ep[:, s], mu_ep[:, s])
            loss.sum().backward()
            for o in opts:
                o.step()
            with torch.no_grad():
                cur = join(W, b)
                path += (cur - prev).norm(dim=-1)
                prev = cur.clone()
        if ep % ckpt_every == 0 or ep == epochs:
            ck_epochs.append(ep)
            cks.append(prev.clone())
            plen.append(path.clone())
            if verbose and (ep % (50 * ckpt_every) == 0):
                print(f"epoch {ep:>6d}  median batch loss {loss.median().item():.4f}", flush=True)
        if not torch.isfinite(prev).all():
            raise FloatingPointError(f"diverged at epoch {ep}")
    return TMSRun(np.array(ck_epochs), torch.stack(cks).numpy(), torch.stack(plen).numpy(), spe)
