"""Run a grid sweep from a config's `sweep.grid` (dotted keys -> value lists).

    uv run python -m muon_llc.dln.experiments.sweep --config configs/dln/c2_ns_steps.yaml

Each grid point trains one run (runs/dln/<name>/<point>/), optionally estimates the LLC at a few
checkpoints, and appends a row of singular-value observables to runs/dln/<name>/summary.csv:
    final s_i, predicted caps, uniformity ratio, residual amplitude, half-target times.
"""

from __future__ import annotations

import copy
import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ...config import set_dotted
from ..llc import headline_estimator
from ..theory.muon_scaling import predicted_final_singular_values
from .common import base_parser, load, resolve_derived
from .estimate import estimate_run
from .train import run_training


def observables(tr: pd.DataFrame, targets, depth: int, weight_decay: float) -> dict:
    """Singular-value observables used by the C2 analysis."""
    s = np.asarray(targets, dtype=float)
    r = len(s)
    sv_cols = [f"sv_{i + 1}" for i in range(r)]
    steps = tr["step"].values
    sv = tr[sv_cols].values  # (T, r)
    T = len(steps)
    tail = slice(max(0, int(0.8 * T)), T)
    final = sv[tail].mean(axis=0)
    resid = (sv[tail] - s[None, :]).std(axis=0)
    out = {f"final_s{i + 1}": float(final[i]) for i in range(r)}
    out.update({f"resid_amp_s{i + 1}": float(resid[i]) for i in range(r)})
    out["resid_amp_max"] = float(resid.max())
    pred = predicted_final_singular_values(s, weight_decay, depth)
    out.update({f"pred_final_s{i + 1}": float(pred[i]) for i in range(r)})
    # half-target crossing times
    for i in range(r):
        idx = np.nonzero(sv[:, i] >= 0.5 * s[i])[0]
        out[f"t_half_s{i + 1}"] = float(steps[idx[0]]) if idx.size else float("nan")
    # uniformity: s_r / s_1 at the time the smallest target direction reaches half its target.
    # Under uniform (spectral) growth every unsaturated product singular value equals (α+ηt)^L,
    # so the ratio is 1; under saddle-to-saddle it is ≈ 0.5·s_r/s_1 ≪ 1.
    t_star = out[f"t_half_s{r}"]
    if np.isfinite(t_star):
        j = int(np.nonzero(steps >= t_star)[0][0])
        out["uniformity"] = float(sv[j, r - 1] / max(sv[j, 0], 1e-30))
    else:
        out["uniformity"] = 0.0  # never reached half target: growth is certainly not uniform
    # lag ratio: t½(s_r) / t½(s_1) relative to the uniform-growth prediction (s_r/s_1)^{1/L}
    # (from (α+ηt)^L = s/2). ≈ 1 under uniform growth, ≫ 1 under saddle-to-saddle, inf if s_r
    # never reached half its target. Immune to the early alignment transient that lowers
    # `uniformity` even for uniform growth.
    t1, tr = out["t_half_s1"], out[f"t_half_s{r}"]
    pred = (s[r - 1] / s[0]) ** (1.0 / depth)
    if np.isfinite(t1) and t1 > 0:
        out["lag_ratio"] = float((tr / t1) / pred) if np.isfinite(tr) else float("inf")
    else:
        out["lag_ratio"] = float("nan")
    return out


def flatten_grid(grid: dict, prefix: str = "") -> dict:
    """{'optimizer.lr': [...]} and {'optimizer': {'lr': [...]}} (from --set) both -> dotted keys."""
    out = {}
    for k, v in grid.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten_grid(v, key + "."))
        else:
            out[key] = list(v)
    return out


def run_sweep(cfg: dict, verbose: bool = True):
    sw = cfg["sweep"]
    grid = flatten_grid(sw["grid"])
    keys = list(grid)
    root = Path(cfg.get("runs_root", "runs")) / cfg["name"]
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = time.time()
    for combo in itertools.product(*[grid[k] for k in keys]):
        c = copy.deepcopy(cfg)
        c.pop("sweep", None)
        tag = "_".join(f"{k.split('.')[-1]}={v}" for k, v in zip(keys, combo))
        c["name"] = tag
        c["runs_root"] = str(root)
        for k, v in zip(keys, combo):
            set_dotted(c, k, v)
        c = resolve_derived(c)
        try:
            d, res = run_training(c, verbose=verbose)
        except FloatingPointError as e:
            print(f"[{tag}] diverged: {e}")
            rows.append({k: v for k, v in zip(keys, combo)} | {"diverged": True})
            continue
        tr = pd.read_csv(d / "trajectory.csv")
        depth = len(c["model"]["widths"]) - 1
        obs = observables(tr, c["data"]["singular_values"], depth, float(c["optimizer"].get("weight_decay", 0.0)))
        row = {k: v for k, v in zip(keys, combo)}
        row.update(run=str(d), depth=depth, lr=c["optimizer"]["lr"], diverged=False, **obs)
        if sw.get("llc", False):
            df, _ = estimate_run(d, ckpt_spec=sw.get("llc_checkpoints", {"max": 6}), verbose=False)
            m = df[df["estimator"] == headline_estimator(df["estimator"].unique())].sort_values("step")
            row["lambda_final"] = float(m["lambda_hat"].iloc[-1]) if len(m) else float("nan")
            row["lambda_final_err"] = float(m["err"].iloc[-1]) if len(m) else float("nan")
            # number of distinct plateaus: count jumps larger than 3σ between consecutive checkpoints
            if len(m) > 1:
                lam, err = m["lambda_hat"].values, m["err"].values
                jumps = np.abs(np.diff(lam)) > 3 * np.sqrt(err[1:] ** 2 + err[:-1] ** 2) + 0.5
                row["n_jumps"] = int(jumps.sum())
        rows.append(row)
        pd.DataFrame(rows).to_csv(root / "summary.csv", index=False)
    if verbose:
        print(f"sweep {cfg['name']}: {len(rows)} runs in {time.time() - t0:.0f}s -> {root / 'summary.csv'}")
    return pd.DataFrame(rows)


def main():
    run_sweep(load(base_parser("Grid sweep over optimizer / data knobs").parse_args()))


if __name__ == "__main__":
    main()
