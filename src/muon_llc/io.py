"""Config loading (YAML with `extends` and dotted overrides) and run-directory I/O.

Run directory layout (runs/<name>/):
    config.json        resolved config
    trajectory.csv     step, loss, sv_1..sv_d, norm_1..norm_L
    checkpoints.npz    steps (K,), thetas (K, D)
    llc.csv            one row per (checkpoint, estimator, nbeta, gamma)
    ladder.csv         per-rung WBIC estimates for the MALA ladder
    *.png              figures
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from .train import TrainResult

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def _deep_update(base: dict, upd: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | os.PathLike, overrides: list[str] | None = None) -> dict:
    """Load YAML; `extends: other.yaml` (relative to the file or to configs/) is merged first."""
    path = Path(path)
    if not path.exists() and (CONFIG_DIR / path).exists():
        path = CONFIG_DIR / path
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("extends", None)
    if parent:
        ppath = path.parent / parent
        if not ppath.exists():
            ppath = CONFIG_DIR / parent
        cfg = _deep_update(load_config(ppath), cfg)
    for ov in overrides or []:
        key, _, val = ov.partition("=")
        set_dotted(cfg, key.strip(), yaml.safe_load(val))
    return cfg


def set_dotted(cfg: dict, key: str, value: Any) -> None:
    parts = key.split(".")
    d = cfg
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = value


def get_device(cfg: dict) -> torch.device:
    dev = cfg.get("device", "auto")
    if dev == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(dev)


def get_dtype(cfg: dict) -> torch.dtype:
    return getattr(torch, cfg.get("dtype", "float32"))


def run_dir(cfg: dict, root: str | os.PathLike | None = None) -> Path:
    root = Path(root or cfg.get("runs_root", "runs"))
    d = root / cfg["name"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_config(cfg: dict, d: Path) -> None:
    with open(d / "config.json", "w") as f:
        json.dump(cfg, f, indent=2, default=str)


def load_run_config(d: str | os.PathLike) -> dict:
    with open(Path(d) / "config.json") as f:
        return json.load(f)


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


def append_rows(rows: list[dict], path: Path) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path)
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(path, index=False)
    return df


def load_llc(d: str | os.PathLike) -> pd.DataFrame:
    return pd.read_csv(Path(d) / "llc.csv")
