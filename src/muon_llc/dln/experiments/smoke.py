"""Reduced end-to-end run on CPU: B2 static check, C1 GD + spectral GD, both estimators, figures.

    uv run python -m muon_llc.dln.experiments.smoke [--config configs/dln/smoke.yaml] [--set ...]
"""

from __future__ import annotations

import copy
import time
from pathlib import Path

from .common import load, resolve_derived
from .estimate import estimate_run
from .plot_c1 import make_c1_figures
from .static_dln import run_static
from .train import run_training


def main():
    import argparse

    p = argparse.ArgumentParser(description="Smoke test: B2 + C1 at reduced size")
    p.add_argument("--config", default="configs/dln/smoke.yaml")
    p.add_argument("--set", nargs="*", default=[])
    p.add_argument("--runs-root", default=None)
    args = p.parse_args()
    cfg = load(args)
    t0 = time.time()

    b2 = copy.deepcopy(cfg)
    b2["name"] = "b2_static"
    b2["llc"]["estimators"] = ["mala"]
    run_static(b2)

    gd = copy.deepcopy(cfg)
    gd["name"] = "c1_gd"
    gd["optimizer"] = {"name": "gd", "lr": 0.05}
    sp = copy.deepcopy(cfg)
    sp["name"] = "c1_spectral"
    sp["optimizer"] = {"name": "spectral", "lr": cfg["smoke"]["spectral_lr"], "momentum": 0.0,
                       "orth": "exact", "weight_decay": 0.0}
    sp["train"]["steps"] = int(cfg["smoke"]["spectral_steps"])
    dirs = []
    for c in (gd, sp):
        c = resolve_derived(c)
        d, _ = run_training(c)
        estimate_run(d, ckpt_spec="cfg", verbose=True)
        dirs.append(d)
    make_c1_figures(dirs, Path(cfg["runs_root"]) / "c1")
    print(f"smoke test done in {time.time() - t0:.0f}s -> {cfg['runs_root']}")


if __name__ == "__main__":
    main()
