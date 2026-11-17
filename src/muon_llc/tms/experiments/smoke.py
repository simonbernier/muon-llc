"""Reduced end-to-end TMS pipeline in about a minute: train SGD and Muon on 8 seeds, estimate
lambda-hat along the trajectories, build the transition tables (with the on-trajectory jumps)
and two figures. Output in runs/tms/smoke/.

    python -m muon_llc.tms.experiments.smoke [--device auto|cpu|cuda] [--root runs/tms/smoke]
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from . import bah, estimate, figures, train

ARMS = {
    "sgd": {"name": "gd", "lr": 0.005},
    "muon_lr3e-4": {"name": "muon", "lr": 3e-4, "orth": "ns", "ns_steps": 5, "momentum": 0.95,
                    "nesterov": True},
}


def main(root: str = "runs/tms/smoke", epochs: int = 300, runs: int = 8, device: str = "auto"):
    root = Path(root)
    shutil.rmtree(root, ignore_errors=True)
    for name, opt in ARMS.items():
        train.main(["--name", name, "--opt", json.dumps(opt), "--out", str(root / "arms" / name),
                    "--runs", str(runs), "--epochs", str(epochs), "--threads", "2", "--device", device])
        estimate.main(["--run", str(root / "arms" / name), "--n-runs", "4", "--stride", "5",
                       "--steps", "100", "--chains", "1", "--threads", "2", "--device", device])
    bah.main(["--arms", str(root / "arms"), "--out", str(root), "--table", str(root / "none.csv")])
    figures.main(["--arms", str(root / "arms"), "--out", str(root), "--headline", *ARMS,
                  "--seeds", "0", "--skip", "posterior"])
    return root


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="runs/tms/smoke")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--runs", type=int, default=8)
    ap.add_argument("--device", default="auto", help="auto (cuda if available) | cpu | cuda")
    a = ap.parse_args()
    main(a.root, a.epochs, a.runs, a.device)
