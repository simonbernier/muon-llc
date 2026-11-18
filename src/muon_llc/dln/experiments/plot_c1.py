"""C1 figures: λ̂(t) for the given runs (GD vs spectral GD) with the Aoyagi staircase and, where
the theory has one, the A4 prediction; the product singular values, the β-ladder, and (if present) the resolution sweep.

    uv run python -m muon_llc.dln.experiments.plot_c1 --runs runs/dln/c1_gd runs/dln/c1_spectral --out runs/dln/c1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ...config import load_run_config
from ..io import load_llc, load_trajectory
from ..llc import headline_estimator
from ..plots import (
    SERIES,
    label_for,
    plot_ladder,
    plot_lambda_family,
    plot_lambda_overlay,
    plot_resolution,
    plot_singular_values,
)
from ..theory.aoyagi import dln_learning_coefficient
from ..theory.dynamics import predicted_lambda_curve
from .common import ensure_dir, predicted_trajectory, resolve_derived


def resolution_threshold(cfg: dict, hess: pd.DataFrame | None = None) -> float:
    """Product singular value at which a direction is resolved. Heuristic: for a balanced depth-L
    factor s^{1/L} per layer, the Hessian eigenvalue along the product direction scales as
    s^{2(L-1)/L}; resolved when nβ h ≳ γ. Solve s^{2(L-1)/L} = γ / nβ."""
    L = len(cfg["model"]["widths"]) - 1
    n = cfg["data"]["n"]
    nb = cfg["llc"].get("nbeta") or n / np.log(n)
    gamma = float(cfg["llc"].get("gamma", 10.0))
    return float((gamma / nb) ** (L / (2 * (L - 1))))


def make_c1_figures(run_dirs: list[Path], out: Path, estimator: str | None = None):
    """`estimator` None: the Gibbs ladder if the runs have it, else MALA."""
    ensure_dir(out)
    runs_sv, runs_lam, preds = [], [], {}
    staircase = None
    for rd in run_dirs:
        cfg = resolve_derived(load_run_config(rd))
        label, ckey = label_for(cfg["optimizer"])
        color = SERIES[ckey]
        tr = load_trajectory(rd)
        pred = predicted_trajectory(cfg)  # None for an optimizer without a closed-form prediction
        if pred is not None:
            preds[label] = pred
        runs_sv.append(dict(label=label, color=color, trajectory=tr))
        if staircase is None:
            widths = cfg["model"]["widths"]
            staircase = {r: dln_learning_coefficient(widths, r).value
                         for r in range(len(cfg["data"]["singular_values"]) + 1)}
        llc_path = rd / "llc.csv"
        if llc_path.exists():
            df = load_llc(rd)
            est0 = estimator or headline_estimator(df["estimator"].unique())
            thr = resolution_threshold(cfg)
            lam_pred = None
            if pred is not None:
                lam_pred = (pred[0], predicted_lambda_curve(pred[0], pred[1], thr, cfg["model"]["widths"]))
            runs_lam.append(dict(label=label, color=color, llc=df, gamma=float(cfg["llc"]["gamma"]),
                                 pred=lam_pred))
            lad_path = rd / "ladder.csv"
            if lad_path.exists():
                lad = pd.read_csv(lad_path)
                if "estimator" in lad.columns:
                    lad = lad[lad["estimator"] == est0]
                if not lad.empty:
                    plot_ladder(lad, out / f"ladder_{cfg['name']}_full.png", title=f"{label}: {est0} β-ladder")
                    plot_ladder(lad, out / f"ladder_{cfg['name']}.png", title=f"{label}: {est0} β-ladder",
                                ylim=(min(staircase.values()) - 8, max(staircase.values()) + 8))
            sweep = df[df["estimator"] == est0].groupby("step")["gamma"].nunique()
            sweep_steps = sweep[sweep > 1].index.tolist()
            if sweep_steps:
                sub = df[df["step"].isin(sweep_steps)]
                plot_resolution(sub, out / f"resolution_{cfg['name']}.png", group_col="step",
                                group_labels={s: f"step {int(s)}" for s in sweep_steps}, estimator=est0,
                                title=f"{label}: λ̂ vs resolution at checkpoints")
    targets = resolve_derived(load_run_config(run_dirs[0]))["data"]["singular_values"]
    plot_singular_values(runs_sv, out / "singular_values.png", targets=targets, predictions=preds)
    if runs_lam:
        present = {e for r in runs_lam for e in r["llc"]["estimator"].unique()}
        for est in sorted(present & {"gibbs", "gibbs_ladder", "mala", "sgld", "mala_ladder", "surrogate"}):
            plot_lambda_overlay(runs_lam, out / f"c1_overlay_{est}.png", staircase=staircase, estimator=est,
                                show_quad=(est != "surrogate"))
        lo = min(staircase.values()) - 5 if staircase else None
        hi = max(staircase.values()) + 8 if staircase else None
        for est in sorted(present & {"gibbs", "mala", "sgld", "surrogate", "hessian"}):
            plot_lambda_family(runs_lam, out / f"c1_family_{est}.png", estimator=est, staircase=staircase,
                               ylim=(lo, hi) if lo is not None else None,
                               title="grey lines: Aoyagi λ(r); values below the axis are basin escapes")
            plot_lambda_family(runs_lam, out / f"c1_family_{est}_full.png", estimator=est, staircase=staircase)
    print(f"figures -> {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--estimator", default=None)
    a = p.parse_args()
    make_c1_figures([Path(r) for r in a.runs], Path(a.out), a.estimator)


if __name__ == "__main__":
    main()
