"""Load TMS runs and compute the quantities the figures need."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .model import TMSProblem, make_problem, split, tms_loss
from .potential import population_loss_flat

ARM_STYLE = {  # fixed categorical order: SGD, spectral GD, Muon, AdamW
    "gd": ("SGD", "#2a78d6"), "spectral": ("spectral GD", "#eb6834"),
    "muon": ("Muon", "#1baf7a"), "adamw": ("AdamW", "#eda100"),
}


def arm_label(cfg: dict) -> tuple[str, str]:
    opt = cfg["opt"]
    base, color = ARM_STYLE[opt["name"]]
    extra = f"lr {opt['lr']:g}"
    if opt.get("weight_decay"):
        extra += f", wd {opt['weight_decay']:g}"
    if opt["name"] == "muon" and opt.get("bias", {}).get("name") == "gd":
        extra += ", SGD bias"
    return f"{base} ({extra})", color


def problem_of(cfg: dict, device=None) -> TMSProblem:
    """The datasets a run was trained on, with the unscaled paper loss (None = CPU)."""
    return make_problem(cfg["n"], cfg["c"], seed=cfg["data_seed"], loss_scale=1.0,
                        n_datasets=None if cfg.get("shared_data") else cfg["runs"], device=device)


def load_run(d) -> dict:
    d = Path(d)
    z = np.load(d / "run.npz")
    cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
    label, color = arm_label(cfg)
    return dict(dir=d, name=d.name, cfg=cfg, epochs=z["epochs"], thetas=z["thetas"].astype(np.float64),
                path_len=z["path_len"], steps_per_epoch=int(z["steps_per_epoch"]), label=label, color=color)


def theory_loss(thetas: np.ndarray) -> np.ndarray:
    c = thetas.shape[-1] // 3
    return population_loss_flat(torch.as_tensor(thetas), 2, c).numpy()


def data_losses(run: dict, run_idx, test_n: int = 5000, test_seed: int = 99) -> tuple[np.ndarray, np.ndarray]:
    """Train loss on each run's own dataset and loss on a shared test set. (K, len(run_idx)) each."""
    cfg = run["cfg"]
    prob = problem_of(cfg).select(run_idx)
    test = make_problem(test_n, cfg["c"], seed=test_seed)
    th = torch.as_tensor(run["thetas"][:, run_idx])
    W, b = split(th, 2, cfg["c"])
    train = torch.stack([tms_loss(W[k], b[k], prob.feat, prob.mu) for k in range(th.shape[0])])
    tst = torch.stack([tms_loss(W[k], b[k], test.feat, test.mu) for k in range(th.shape[0])])
    return train.numpy(), tst.numpy()


def smooth(x: np.ndarray, window: int = 6) -> np.ndarray:
    """Trailing-centred moving average along axis 0, edges handled by shrinking the window."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    h = window // 2
    for t in range(x.shape[0]):
        out[t] = np.nanmean(x[max(0, t - h): t + h + 1], axis=0)
    return out
