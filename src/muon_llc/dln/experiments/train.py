"""Train a DLN from a config and save the run directory.

    uv run python -m muon_llc.dln.experiments.train --config configs/dln/c1_gd.yaml
"""

from __future__ import annotations

import time

from ...config import run_dir, save_config
from ..io import save_train_result
from ..train import checkpoint_schedule, train
from .common import base_parser, build_problem, dense_band, load, predicted_event_times


def run_training(cfg: dict, verbose: bool = True):
    problem = build_problem(cfg)
    tr = cfg["train"]
    extra = tr.get("extra_checkpoints", [])
    if extra == "predicted":
        extra = dense_band(predicted_event_times(cfg), total_steps=int(tr["steps"]))
    ckpts = checkpoint_schedule(int(tr["steps"]), int(tr.get("n_checkpoints", 40)), extra)
    t0 = time.time()
    res = train(
        problem, cfg["optimizer"], int(tr["steps"]), float(cfg["model"]["init_scale"]),
        int(cfg["model"]["seed"]), log_every=int(tr.get("log_every", 20)), checkpoint_steps=ckpts,
        verbose=bool(tr.get("verbose", False)),
    )
    d = run_dir(cfg)
    save_config(cfg, d)
    save_train_result(res, d)
    if verbose:
        sv = res.singular_values[-1][: len(cfg["data"]["singular_values"])]
        print(f"[{cfg['name']}] trained {tr['steps']} steps in {time.time() - t0:.1f}s; final loss "
              f"{res.loss[-1]:.3e}; final s_i = {sv.round(3)}; {len(res.checkpoint_steps)} checkpoints -> {d}")
    if cfg.get("wandb", {}).get("enabled"):
        _log_wandb(cfg, res)
    return d, res


def _log_wandb(cfg, res):
    import wandb

    run = wandb.init(project=cfg["wandb"].get("project", "muon-llc"), name=cfg["name"], config=cfg)
    for i, step in enumerate(res.steps):
        row = {"loss": float(res.loss[i])}
        row.update({f"sv/{j + 1}": float(v) for j, v in enumerate(res.singular_values[i])})
        run.log(row, step=int(step))
    run.finish()


def main():
    args = base_parser("Train a DLN with GD or the Muon family").parse_args()
    run_training(load(args))


if __name__ == "__main__":
    main()
