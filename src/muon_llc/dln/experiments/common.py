"""Shared helpers: argument parsing, problem construction, derived config keys, theory predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from ...config import get_device, get_dtype, load_config
from ..model import DLNProblem, make_problem
from ..theory.dynamics import (
    gd_aligned_trajectory,
    gd_escape_time_estimate,
    spectral_saturation_time,
    spectral_trajectory,
)


def base_parser(desc: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--config", required=True, help="YAML config (path or name under configs/)")
    p.add_argument("--set", nargs="*", default=[], help="dotted overrides, e.g. train.steps=5000")
    p.add_argument("--runs-root", default=None)
    return p


def load(args) -> dict:
    cfg = load_config(args.config, args.set)
    if getattr(args, "runs_root", None):
        cfg["runs_root"] = args.runs_root
    return resolve_derived(cfg)


def resolve_derived(cfg: dict) -> dict:
    """Expand convenience keys used by sweeps: data.kappa -> singular_values, model.depth -> widths."""
    d = cfg.get("data", {})
    if d.get("kappa") is not None:
        k = float(d["kappa"])
        s1 = float(d.get("s_max", 4.0))
        r = int(d.get("rank", 4))
        d["singular_values"] = [s1 * k ** (-(i) / max(1, r - 1)) for i in range(r)]
    m = cfg.get("model", {})
    if m.get("depth") is not None:
        w = int(m.get("width", 8))
        m["widths"] = [w] * (int(m["depth"]) + 1)
    return cfg


def build_problem(cfg: dict) -> DLNProblem:
    dev, dt = get_device(cfg), get_dtype(cfg)
    return make_problem(cfg["model"]["widths"], cfg["data"]["n"], cfg["data"]["singular_values"],
                        cfg["data"]["seed"], device=dev, dtype=dt)


def effective_alpha(cfg: dict) -> float:
    """Per-layer singular-value scale of the random init: init_scale * (sqrt(H_in) + sqrt(H_out)) / 2-ish.

    The aligned prediction needs one scalar; the median singular value of a Gaussian matrix is a
    reasonable stand-in for the 'typical' factor in each direction.
    """
    w = cfg["model"]["widths"]
    return float(cfg["model"]["init_scale"]) * float(np.sqrt(np.mean(w)))


def predicted_trajectory(cfg: dict, steps: int | None = None):
    """(steps, s_pred) from theory/dynamics for this config's optimizer."""
    opt = cfg["optimizer"]
    steps = int(cfg["train"]["steps"]) if steps is None else steps
    L = len(cfg["model"]["widths"]) - 1
    alpha = effective_alpha(cfg)
    s = cfg["data"]["singular_values"]
    every = max(1, steps // 2000)
    if opt["name"] == "gd":
        return gd_aligned_trajectory(s, L, alpha, float(opt["lr"]), steps, every)
    return spectral_trajectory(s, L, alpha, float(opt["lr"]), steps, every)


def predicted_event_times(cfg: dict) -> list[float]:
    opt = cfg["optimizer"]
    L = len(cfg["model"]["widths"]) - 1
    alpha = effective_alpha(cfg)
    s = cfg["data"]["singular_values"]
    if opt["name"] == "gd":
        return [gd_escape_time_estimate(si, L, alpha, float(opt["lr"])) for si in s]
    return [spectral_saturation_time(si, L, alpha, float(opt["lr"])) for si in s]


def dense_band(times, factors=(0.5, 0.7, 1.0, 1.4, 2.0), total_steps: int | None = None) -> list[int]:
    out = []
    for t in times:
        if not np.isfinite(t) or t <= 0:
            continue
        for f in factors:
            st = int(round(t * f))
            if total_steps is None or 0 < st <= total_steps:
                out.append(st)
    return sorted(set(out))


def theta_tensor(theta_np: np.ndarray, problem: DLNProblem) -> torch.Tensor:
    return torch.as_tensor(theta_np, dtype=problem.dtype, device=problem.device)


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p
