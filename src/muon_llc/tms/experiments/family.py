"""Scans of the family between the toy model of superposition and a tied linear network
(`muon_llc.tms.family`).

    python -m muon_llc.tms.experiments.family train   --config configs/tms/family_scan.yaml --opt sgd
    python -m muon_llc.tms.experiments.family analyse --config configs/tms/family_scan.yaml

`train` trains every grid cell x seed under one optimizer of the config, in one batch, and writes
<out>/<opt>/run.npz (epochs, thetas (K, R, D) in float32) and cells.csv. `analyse` reads every
optimizer that has been trained and writes into <out>/:

    feature_times.csv   one row per optimizer, cell, seed: epoch at which each feature is learned
                        (see `family.learning_times`), number learned, final loss
    summary.csv         one row per optimizer and cell: medians over seeds
    states.csv          one row per optimizer and cell: plateaus of the state label
                        (`family.state_labels`): transitions per run, how many come after
                        `--late` epochs, dwell of the plateaus that were left, final states
    loss_<opt>.png      training loss of every seed, one panel per cell
    times_<opt>.png     learning time of each feature, one panel per cell
    spread_<key>.png    last / first learning time against the first grid key that varies, one line
                        per optimizer, one panel per value of the other keys

`llc` estimates lambda-hat along the trajectories with the shared SGLD sampler (the paper's
settings by default: full batch, eps = 0.001, gamma = 1, 500 steps, n beta = n / log n) at
checkpoints spaced evenly in log time, and writes <out>/<opt>/llc.npz and <out>/llc.csv (one row
per optimizer, cell, seed and checkpoint: loss, lambda-hat, share of draws below the starting
loss, number of features learned so far), llc_by_state.csv (median lambda-hat, number of estimates
and share of escaped chains per optimizer, cell and number of features learned) and llc_<opt>.png.

<out> defaults to runs/tms/<config name>.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ...config import get_device, load_config
from ..family import (
    GRID_KEYS,
    cells_of,
    family_problem,
    feature_losses,
    initial_points,
    learning_times,
    state_labels,
    unlearned_level,
)
from ..train import train_batched
from ..transitions import plateaus


def out_dir(cfg: dict, args) -> Path:
    return Path(args.out) if args.out else Path("runs/tms") / cfg["name"]


def varying(cells: pd.DataFrame) -> list[str]:
    return [k for k in GRID_KEYS if cells[k].nunique() > 1]


OPT_STYLE = {"sgd": ("SGD", "#2a78d6"), "spectral": ("spectral GD", "#eb6834"), "muon": ("Muon", "#1baf7a")}


def llc_by_state(df: pd.DataFrame, keys: list[str], max_below: float = 0.05) -> pd.DataFrame:
    """Median lambda-hat per optimizer, cell and number of features learned; `escaped` is the share
    of estimates whose chains spend more than `max_below` of their draws below the starting loss."""
    df = df.assign(escaped=df.frac_below > max_below)
    return (df.groupby(["opt", "cell"] + keys + ["n_learned"], sort=False)
            .agg(llc=("llc", "median"), count=("llc", "size"), escaped=("escaped", "mean")).reset_index())


def plot_spread(summ: pd.DataFrame, keys: list[str], path_of, name: str) -> None:
    """Median last / first learning time against keys[0], one panel per value of the other keys."""
    import matplotlib.pyplot as plt

    x, rest = keys[0], keys[1:]
    groups = list(summ.groupby(rest, sort=False)) if rest else [((), summ)]
    fig, axes = plt.subplots(1, len(groups), figsize=(4.2 * len(groups), 3.4), sharey=True, squeeze=False)
    for ax, (val, g) in zip(axes[0], groups):
        for opt, go in g.groupby("opt", sort=False):
            label, color = OPT_STYLE.get(opt, (opt, None))
            go = go.sort_values(x)
            ax.plot(go[x], go["spread"], marker="o", ms=4, lw=1.6, color=color, label=label)
        ax.set_yscale("log")
        ax.set_xlabel(x.replace("_", " "))
        val = val if isinstance(val, tuple) else (val,)
        ax.set_title(", ".join(f"{k.replace('init_', 'initial ')} {v:g}" for k, v in zip(rest, val)), fontsize=9)
        ax.grid(alpha=0.25, lw=0.5)
    axes[0, 0].set_ylabel("last / first feature-learning time")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"{name}: how staggered the features are learned (median over seeds)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path_of(f"spread_{x}.png"), dpi=130)
    plt.close(fig)


def cell_title(cell, keys) -> str:
    return ", ".join(f"{k.replace('init_', '')} {getattr(cell, k):g}" for k in keys) or "single cell"


def train(cfg: dict, args) -> None:
    torch.set_num_threads(args.threads)
    device = get_device({"device": args.device})
    cells = cells_of(cfg)
    problem = family_problem(cfg, cells, device)
    theta0 = initial_points(cfg, cells)
    opt = cfg["optimizers"][args.opt]
    out = out_dir(cfg, args) / args.opt
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    run = train_batched(problem, theta0, opt, int(cfg["epochs"]), int(cfg.get("batch_size", 20)),
                        int(cfg.get("seed", 0)), int(cfg.get("ckpt_every", 10)), verbose=True)
    np.savez_compressed(out / "run.npz", epochs=run.epochs, thetas=run.thetas.astype(np.float32),
                        path_len=run.path_len[-1], steps_per_epoch=run.steps_per_epoch)
    cells.to_csv(out / "cells.csv", index_label="cell")
    (out / "config.json").write_text(json.dumps(cfg | {"opt": opt, "device": str(device),
                                                       "seconds": time.time() - t0}, indent=1), encoding="utf-8")
    print(f"[{cfg['name']}/{args.opt}] {len(cells)} cells x {cfg['seeds']} seeds, {time.time() - t0:.0f}s on {device}")


def analyse(cfg: dict, args) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = out_dir(cfg, args)
    cells = cells_of(cfg)
    seeds, c = int(cfg["seeds"]), int(cfg["c"])
    keys = varying(cells)
    problem = family_problem(cfg, cells)
    unlearned = unlearned_level(problem)
    rows, srows = [], []
    ncol = int(np.ceil(np.sqrt(len(cells)))) if len(keys) != 1 else len(cells)
    if len(keys) >= 2:
        ncol = cells[keys[-1]].nunique()
    nrow = int(np.ceil(len(cells) / ncol))
    for opt in cfg["optimizers"]:
        f = root / opt / "run.npz"
        if not f.exists():
            continue
        z = np.load(f)
        ep, th = z["epochs"], z["thetas"].astype(np.float64)
        fl = feature_losses(problem, th)  # (K, R, c)
        loss = fl.sum(-1)
        t = learning_times(ep, fl, unlearned)  # (R, c)
        fig, axes = plt.subplots(nrow, ncol, figsize=(2.9 * ncol, 2.4 * nrow), sharex=True, squeeze=False)
        fig2, axes2 = plt.subplots(nrow, ncol, figsize=(2.9 * ncol, 2.4 * nrow), sharex=True, sharey=True,
                                   squeeze=False)
        for ci, cell in enumerate(cells.itertuples(index=False)):
            sl = slice(ci * seeds, (ci + 1) * seeds)
            for s in range(seeds):
                r = ci * seeds + s
                rows.append({"opt": opt, "cell": ci, **cell._asdict(), "seed": s,
                             **{f"t{j}": t[r, j] for j in range(c)},
                             "n_learned": int(np.isfinite(t[r]).sum()), "final_loss": loss[-1, r]})
            ax = axes[ci // ncol, ci % ncol]
            ax.plot(ep[1:], loss[1:, sl], lw=0.6, alpha=0.7)
            ax.set_xscale("log")
            lo, hi = loss[-1, sl].min(), np.quantile(loss[1:, sl].max(0), 0.75)
            ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))
            ax.set_title(cell_title(cell, keys), fontsize=8)
            ax2 = axes2[ci // ncol, ci % ncol]
            tt = np.where(t[sl] <= 0, ep[1] / 2, t[sl])  # learned from the start: left edge
            for j in range(c):
                ax2.scatter(tt[:, j], np.full(seeds, j) + np.linspace(-0.3, 0.3, seeds), s=4)
            ax2.set_xscale("log")
            ax2.set_title(cell_title(cell, keys), fontsize=8)
        for a in axes[-1]:
            a.set_xlabel("epoch")
        for a in axes2[-1]:
            a.set_xlabel("epoch the feature is learned")
        for a in axes[:, 0]:
            a.set_ylabel("training loss")
        for a in axes2[:, 0]:
            a.set_ylabel("feature")
        fig.suptitle(f"{cfg['name']}, {opt}: training loss, {seeds} seeds per panel")
        fig2.suptitle(f"{cfg['name']}, {opt}: when each feature is learned (one dot per seed; missing = not learned)")
        for fg, nm in ((fig, "loss"), (fig2, "times")):
            fg.tight_layout()
            fg.savefig(root / f"{nm}_{opt}.png", dpi=110)
            plt.close(fg)
        labels = state_labels(th, int(cfg["r"]), c)
        for ci, cell in enumerate(cells.itertuples(index=False)):
            n_tr, n_late, dwell, finals = [], [], [], Counter()
            for s in range(seeds):
                ps = plateaus(labels[:, ci * seeds + s])
                n_tr.append(len(ps) - 1)
                n_late.append(sum(ep[q.start] > args.late for q in ps[1:]))
                dwell += [ep[q.end] - ep[q.start] for q in ps[:-1] if q.start > 0 or len(ps) > 1]
                finals[ps[-1].name if ps else "none"] += 1
            srows.append({"opt": opt, "cell": ci, **{k: getattr(cell, k) for k in keys},
                          "transitions_per_run": np.mean(n_tr), "runs_with_late": int(np.sum(np.array(n_late) > 0)),
                          "late_per_run": np.mean(n_late),
                          "median_dwell": float(np.median(dwell)) if dwell else np.nan,
                          "finals": " ".join(f"{k}:{v}" for k, v in finals.most_common(4))})
    df = pd.DataFrame(rows)
    pd.DataFrame(srows).to_csv(root / "states.csv", index=False)
    df.to_csv(root / "feature_times.csv", index=False)
    tcols = [f"t{j}" for j in range(c)]
    tv = df[tcols].to_numpy()
    pos = np.where(tv > 0, tv, np.nan)  # features learned during training (not from the start)
    df["t_first"] = np.nanmin(np.where(np.isnan(pos), np.inf, pos), axis=1)
    df["t_last"] = np.nanmax(np.where(np.isnan(pos), -np.inf, pos), axis=1)
    df.loc[~np.isfinite(df.t_first), ["t_first", "t_last"]] = np.nan
    df["n_during"] = np.isfinite(pos).sum(1)
    df["spread"] = df.t_last / df.t_first
    g = df.groupby(["opt", "cell"] + keys, sort=False)
    summ = g.agg(n_learned=("n_learned", "median"), n_learned_min=("n_learned", "min"),
                 n_during=("n_during", "median"), t_first=("t_first", "median"), t_last=("t_last", "median"),
                 spread=("spread", "median"), final_loss=("final_loss", "median"),
                 final_loss_max=("final_loss", "max")).reset_index()
    summ.to_csv(root / "summary.csv", index=False)
    if keys:
        plot_spread(summ, keys, lambda f: root / f, cfg["name"])
    with pd.option_context("display.width", 250, "display.max_rows", 500):
        print(summ.round(4).to_string(index=False))
        print(pd.DataFrame(srows).round(2).to_string(index=False))


def llc(cfg: dict, args) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ..llc import sgld_llc

    root = out_dir(cfg, args)
    cells = cells_of(cfg)
    seeds, c = int(cfg["seeds"]), int(cfg["c"])
    keys = varying(cells)
    device = get_device({"device": args.device})
    problem = family_problem(cfg, cells, device)
    cpu_problem = family_problem(cfg, cells)
    unlearned = unlearned_level(cpu_problem)
    ns = min(args.llc_seeds, seeds)
    runs = np.concatenate([ci * seeds + np.arange(ns) for ci in range(len(cells))])
    rows = []
    ncol = cells[keys[-1]].nunique() if len(keys) >= 2 else len(cells)
    nrow = int(np.ceil(len(cells) / ncol))
    for opt in cfg["optimizers"]:
        f = root / opt / "run.npz"
        if not f.exists():
            continue
        z = np.load(f)
        ep, th = z["epochs"], z["thetas"].astype(np.float64)
        ck = np.unique(np.round(np.geomspace(1, len(ep) - 1, args.llc_checkpoints)).astype(int))
        ck = np.r_[0, ck]
        centres = th[ck][:, runs]  # (k, m, D)
        k, m, D = centres.shape
        t0 = time.time()
        res = sgld_llc(problem, centres.reshape(k * m, D), eps=args.eps, gamma=args.gamma, n_steps=args.steps,
                       n_chains=args.chains, chunk=args.chunk, data_index=np.tile(runs, k), seed=0)
        lam = res.llc.reshape(k, m)
        below = res.frac_below.mean(1).reshape(k, m)
        np.savez_compressed(root / opt / "llc.npz", ck_idx=ck, run_idx=runs, llc=lam,
                            llc_chain=res.llc_chain.reshape(k, m, -1), frac_below=res.frac_below.reshape(k, m, -1),
                            nbeta=res.nbeta, eps=args.eps, gamma=args.gamma, steps=args.steps)
        fl = feature_losses(cpu_problem, th[ck])  # (k, R, c)
        n_learned = (fl[:, runs] <= 0.5 * unlearned[runs][None]).sum(-1)
        loss = fl.sum(-1)[:, runs]
        print(f"[{cfg['name']}/{opt}] {k} checkpoints x {m} runs in {time.time() - t0:.0f}s on {device}")
        fig, axes = plt.subplots(nrow, ncol, figsize=(2.9 * ncol, 2.4 * nrow), sharex=True, squeeze=False)
        for ci, cell in enumerate(cells.itertuples(index=False)):
            sl = slice(ci * ns, (ci + 1) * ns)
            for a in range(k):
                for s in range(ns):
                    rows.append({"opt": opt, "cell": ci, **cell._asdict(), "seed": s, "epoch": int(ep[ck[a]]),
                                 "loss": loss[a, ci * ns + s], "llc": lam[a, ci * ns + s],
                                 "frac_below": below[a, ci * ns + s], "n_learned": int(n_learned[a, ci * ns + s])})
            ax = axes[ci // ncol, ci % ncol]
            x = np.maximum(ep[ck], ep[1] / 2)
            ax.plot(x, lam[:, sl], lw=0.5, alpha=0.5, color="C0")
            ax.plot(x, np.median(lam[:, sl], axis=1), lw=1.6, color="k")
            ax.set_xscale("log")
            ax.set_title(cell_title(cell, keys), fontsize=8)
            tw = ax.twinx()
            tw.plot(x, np.median(n_learned[:, sl], axis=1), lw=1.2, color="C3", ls="--")
            tw.set_ylim(-0.2, c + 0.2)
            tw.tick_params(labelsize=6, colors="C3")
        for a in axes[-1]:
            a.set_xlabel("epoch")
        for a in axes[:, 0]:
            a.set_ylabel("lambda-hat")
        fig.suptitle(f"{cfg['name']}, {opt}: lambda-hat per seed (blue), median (black); "
                     f"features learned, median (red dashed, right axis)")
        fig.tight_layout()
        fig.savefig(root / f"llc_{opt}.png", dpi=110)
        plt.close(fig)
    df = pd.DataFrame(rows)
    df.to_csv(root / "llc.csv", index=False)
    llc_by_state(df, keys).to_csv(root / "llc_by_state.csv", index=False)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["train", "analyse", "llc"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="dotted overrides, key=value")
    ap.add_argument("--opt", help="optimizer name in the config (train)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cpu", help="cpu | cuda | auto")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--late", type=float, default=300, help="epoch after which a transition counts as late")
    ap.add_argument("--llc-seeds", type=int, default=8, help="seeds per cell (llc)")
    ap.add_argument("--llc-checkpoints", type=int, default=40, help="checkpoints, evenly spaced in log time (llc)")
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=600)
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)
    {"train": train, "analyse": analyse, "llc": llc}[args.command](cfg, args)


if __name__ == "__main__":
    main()
