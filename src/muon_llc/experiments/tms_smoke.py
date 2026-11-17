"""Reduced end-to-end TMS pipeline on CPU in about a minute: train SGD and Muon on 8 seeds,
estimate lambda-hat along the trajectories, build the transition table and two figures.
Output in runs/tms_smoke/."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from . import tms_bah, tms_estimate, tms_figures, tms_train

ARMS = {
    "sgd": {"name": "gd", "lr": 0.005},
    "muon_lr3e-4": {"name": "muon", "lr": 3e-4, "orth": "ns", "ns_steps": 5, "momentum": 0.95,
                    "nesterov": True, "bias": {"name": "gd", "lr": 0.005}},
}


def main(root: str = "runs/tms_smoke", epochs: int = 300, runs: int = 8):
    root = Path(root)
    shutil.rmtree(root, ignore_errors=True)
    for name, opt in ARMS.items():
        tms_train.main(["--name", name, "--opt", json.dumps(opt), "--out", str(root / "arms" / name),
                        "--runs", str(runs), "--epochs", str(epochs), "--threads", "2"])
        tms_estimate.main(["--run", str(root / "arms" / name), "--n-runs", "4", "--stride", "5",
                           "--steps", "100", "--chains", "1", "--threads", "2"])
    tms_bah.main(["--arms", str(root / "arms"), "--out", str(root), "--table", str(root / "none.csv")])
    tms_figures.main(["--arms", str(root / "arms"), "--out", str(root), "--headline", *ARMS,
                      "--seeds", "0", "--skip", "posterior"])
    return root


if __name__ == "__main__":
    main()
