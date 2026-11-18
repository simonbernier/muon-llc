"""Estimate the LLC at the checkpoints of a saved run.

    uv run python -m muon_llc.dln.experiments.estimate --run runs/dln/c1_gd [--estimators sgld gibbs]
        [--checkpoints all|max:12|every:2] [--resolution-sweep] [--gammas 1 10 100] [--device cuda]

Writes llc.csv and ladder.csv into the run directory (overwriting unless --append).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ...config import load_run_config
from ..io import load_checkpoints
from ..llc import estimate_checkpoints, select_checkpoints
from .common import build_problem, resolve_derived, theta_tensor


def parse_ckpt_spec(s):
    if s is None or s == "all":
        return "all"
    if s.startswith("max:"):
        return {"max": int(s[4:])}
    if s.startswith("every:"):
        return {"every": int(s[6:])}
    return [int(x) for x in s.split(",")]


def estimate_run(run: Path, estimators=None, ckpt_spec="all", resolution_sweep=False, gammas=None,
                 res_ckpt_spec=None, append=False, verbose=True, cfg_override: dict | None = None,
                 device: str | None = None):
    cfg = load_run_config(run)
    if device:  # sample on another device than the one the run was trained on
        cfg["device"] = device
    if cfg_override:
        cfg["llc"].update(cfg_override)
    cfg = resolve_derived(cfg)
    problem = build_problem(cfg)
    steps, thetas = load_checkpoints(run)
    llc_cfg = cfg["llc"]
    estimators = estimators or llc_cfg.get("estimators", ["sgld", "gibbs"])
    chosen = select_checkpoints(steps, ckpt_spec if ckpt_spec != "cfg" else llc_cfg.get("checkpoints", "all"))
    res_spec = res_ckpt_spec if res_ckpt_spec is not None else llc_cfg.get("resolution_sweep_checkpoints", "none")
    res_steps = set() if (not resolution_sweep and res_spec in (None, "none")) else set(
        select_checkpoints(chosen, "all" if res_spec in (None, "none") else res_spec).tolist())
    gam_sweep = [float(g) for g in (gammas or llc_cfg.get("gamma_sweep", [llc_cfg.get("gamma", 10.0)]))]
    gam_main = float(llc_cfg.get("gamma", 10.0))

    t0 = time.time()
    idx = [int(np.nonzero(steps == step)[0][0]) for step in chosen]
    gam = [sorted(set([gam_main] + (gam_sweep if step in res_steps else []))) for step in chosen]
    rows, ladder_rows = estimate_checkpoints(problem, theta_tensor(thetas[idx], problem), chosen, gam, llc_cfg,
                                             int(llc_cfg.get("seed", 0)), estimators=estimators, verbose=verbose)
    df = pd.DataFrame(rows)
    lad = pd.DataFrame(ladder_rows)
    for name, new in (("llc.csv", df), ("ladder.csv", lad)):
        p = run / name
        if append and p.exists():
            new = pd.concat([pd.read_csv(p), new], ignore_index=True)
        new.to_csv(p, index=False)
    if verbose:
        print(f"[{cfg['name']}] estimated {len(chosen)} checkpoints in {time.time() - t0:.1f}s -> {run / 'llc.csv'}")
    return df, lad


def main():
    p = argparse.ArgumentParser(description="LLC estimation at saved checkpoints")
    p.add_argument("--run", required=True)
    p.add_argument("--estimators", nargs="*", default=None)
    p.add_argument("--checkpoints", default="cfg", help="all | max:K | every:k | s1,s2,... | cfg")
    p.add_argument("--resolution-sweep", action="store_true")
    p.add_argument("--resolution-checkpoints", default=None, help="all | max:K | every:k | steps")
    p.add_argument("--gammas", nargs="*", type=float, default=None)
    p.add_argument("--append", action="store_true")
    p.add_argument("--device", default=None, help="cpu | cuda | auto; default: the run's own config")
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args()
    estimate_run(Path(a.run), a.estimators, parse_ckpt_spec(a.checkpoints) if a.checkpoints != "cfg" else "cfg",
                 a.resolution_sweep, a.gammas,
                 None if a.resolution_checkpoints is None else parse_ckpt_spec(a.resolution_checkpoints),
                 a.append, not a.quiet, device=a.device)


if __name__ == "__main__":
    main()
