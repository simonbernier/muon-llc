"""DLN run-directory I/O.

Run directory layout (runs/dln/<name>/):
    config.json        resolved config
    trajectory.csv     step, loss, sv_1..sv_d, norm_1..norm_L
    checkpoints.npz    steps (K,), thetas (K, D)
    llc.csv            one row per (checkpoint, estimator, nbeta, gamma)
    ladder.csv         per-rung WBIC estimates for the MALA ladder
    *.png              figures
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from .train import TrainResult


def save_train_result(res: TrainResult, d: Path) -> None:
    sv_cols = {f"sv_{i + 1}": res.singular_values[:, i] for i in range(res.singular_values.shape[1])}
    norm_cols = {f"norm_{l + 1}": res.layer_spectral_norms[:, l]
                 for l in range(res.layer_spectral_norms.shape[1])}
    pd.DataFrame({"step": res.steps, "loss": res.loss, **sv_cols, **norm_cols}).to_csv(
        d / "trajectory.csv", index=False
    )
    np.savez(d / "checkpoints.npz", steps=res.checkpoint_steps, thetas=res.checkpoints)


def load_trajectory(d: str | os.PathLike) -> pd.DataFrame:
    return pd.read_csv(Path(d) / "trajectory.csv")


def load_checkpoints(d: str | os.PathLike) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(Path(d) / "checkpoints.npz")
    return z["steps"], z["thetas"]


def load_llc(d: str | os.PathLike) -> pd.DataFrame:
    return pd.read_csv(Path(d) / "llc.csv")
