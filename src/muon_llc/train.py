"""Training loop for the DLN: records loss, product singular values, layer spectral norms,
and flat-parameter checkpoints at a chosen schedule."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .dln import DLNProblem, flatten, init_weights
from .optim import build_optimizer


@dataclass
class TrainResult:
    steps: np.ndarray  # logged steps
    loss: np.ndarray  # (T,)
    singular_values: np.ndarray  # (T, min(H_L, H_0))
    layer_spectral_norms: np.ndarray  # (T, L)
    checkpoint_steps: np.ndarray  # (K,)
    checkpoints: np.ndarray  # (K, D) flat parameters
    extras: dict = field(default_factory=dict)


def checkpoint_schedule(total_steps: int, n_log: int = 40, extra=None) -> np.ndarray:
    """0, then ~n_log log-spaced steps up to total_steps, plus any extra steps."""
    pts = {0, int(total_steps)}
    if total_steps > 0:
        for s in np.logspace(0, np.log10(max(total_steps, 1)), n_log):
            pts.add(int(round(s)))
    for s in extra or []:
        if 0 <= int(s) <= total_steps:
            pts.add(int(s))
    return np.array(sorted(pts), dtype=np.int64)


def train(
    problem: DLNProblem,
    optimizer_cfg: dict,
    steps: int,
    init_scale: float,
    seed: int,
    log_every: int = 10,
    checkpoint_steps=None,
    n_checkpoints: int = 40,
    full_batch: bool = True,
    batch_size: int | None = None,
    verbose: bool = False,
) -> TrainResult:
    """Full-batch GD/spectral training on L_n. Same `seed` -> same init for every optimizer."""
    device = problem.device
    g = torch.Generator().manual_seed(int(seed))
    init = init_weights(problem.widths, init_scale, g, dtype=problem.dtype)
    weights = [w.to(device).requires_grad_(True) for w in init]
    opt = build_optimizer(weights, optimizer_cfg)
    ckpt_steps = (
        np.asarray(sorted(set(int(s) for s in checkpoint_steps)), dtype=np.int64)
        if checkpoint_steps is not None
        else checkpoint_schedule(steps, n_checkpoints)
    )
    ckpt_set = set(ckpt_steps.tolist())
    log_steps = sorted(set(range(0, steps + 1, max(1, log_every))) | ckpt_set | {steps})

    rec_steps, rec_loss, rec_sv, rec_norms, ckpts = [], [], [], [], {}
    log_ptr = 0

    def record(step: int):
        with torch.no_grad():
            loss = problem.loss_weights(weights).item()
            W = weights[0]
            for w in weights[1:]:
                W = w @ W
            sv = torch.linalg.svdvals(W).cpu().numpy()
            norms = np.array([torch.linalg.matrix_norm(w, ord=2).item() for w in weights])
        rec_steps.append(step)
        rec_loss.append(loss)
        rec_sv.append(sv)
        rec_norms.append(norms)
        if step in ckpt_set:
            ckpts[step] = flatten([w.detach() for w in weights]).cpu().numpy().astype(np.float64)
        if verbose and (len(rec_steps) % 50 == 0):
            print(f"step {step:>8d}  loss {loss:.3e}  sv {np.array2string(sv[:5], precision=3)}")

    for step in range(steps + 1):
        if log_ptr < len(log_steps) and step == log_steps[log_ptr]:
            record(step)
            log_ptr += 1
        if step == steps:
            break
        opt.zero_grad(set_to_none=True)
        idx = None if full_batch else problem.minibatch_indices(batch_size, g)
        loss = problem.loss_weights(weights, idx)
        loss.backward()
        opt.step()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"loss diverged at step {step}")

    ks = np.array(sorted(ckpts), dtype=np.int64)
    return TrainResult(
        steps=np.array(rec_steps, dtype=np.int64),
        loss=np.array(rec_loss),
        singular_values=np.stack(rec_sv),
        layer_spectral_norms=np.stack(rec_norms),
        checkpoint_steps=ks,
        checkpoints=np.stack([ckpts[k] for k in ks]) if len(ks) else np.zeros((0, problem.dim)),
    )
