"""Build every TMS figure from saved runs.

    python -m muon_llc.tms.experiments.figures --arms runs/tms/arms --out results/tms \
        --headline sgd spectral_lr3e-4 muon_lr3e-4 adamw_lr3e-4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .. import plots
from ..analysis import load_run
from ..transitions import label_runs, summarize
from .bah import llc_table


def _with_llc(run, tag="llc"):
    p = run["dir"] / f"{tag}.npz"
    run["llc"] = dict(np.load(p)) if p.exists() else None
    return run


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", default="runs/tms/arms")
    ap.add_argument("--post", default="runs/tms/post")
    ap.add_argument("--out", default="results/tms")
    ap.add_argument("--headline", nargs="*", default=["sgd", "spectral_lr3e-4", "muon_lr3e-4", "adamw_lr3e-4"])
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--n-seed-figs", type=int, default=8)
    ap.add_argument("--skip", nargs="*", default=[])
    args = ap.parse_args(argv)
    out = Path(args.out)
    fig = out / "figures"
    fig.mkdir(parents=True, exist_ok=True)
    arms_dir = Path(args.arms)
    runs = [_with_llc(load_run(arms_dir / a)) for a in args.headline if (arms_dir / a / "run.npz").exists()]
    lam, err, _ = llc_table(out / "table_k1.csv")

    if "trajectory" not in args.skip and runs:
        seeds = args.seeds
        if seeds is None:  # seeds (among those with lambda-hat) where SGD shows a transition, varied by type
            base = runs[0]
            n_est = len(base["llc"]["run_idx"]) if base["llc"] is not None else 30
            labels, _ = label_runs(base["thetas"][:, :n_est])
            s = summarize(labels, base["path_len"][:, :n_est])
            seen, seeds = set(), []
            for j, ps in enumerate(s["sequences"]):
                key = tuple(p.name for p in ps)
                if len(ps) > 1 and key not in seen:
                    seen.add(key)
                    seeds.append(j)
            seeds = seeds[: args.n_seed_figs]
        for j in seeds:
            print(plots.fig_trajectory(runs, j, fig / f"trajectory_seed{j:03d}.png"), flush=True)

    if "ensemble" not in args.skip and runs:
        print(plots.fig_ensemble(runs, fig / "ensemble_loss.png"))
        allr = [load_run(d) for d in sorted(Path(args.arms).iterdir()) if (d / "run.npz").exists()]
        for fam in ("spectral", "muon", "adamw"):
            sub = [r for r in allr if r["cfg"]["opt"]["name"] == fam]
            sub.sort(key=lambda r: (r["cfg"]["opt"]["lr"], r["name"]))
            if sub:
                print(plots.fig_ensemble(sub, fig / f"ensemble_loss_{fam}_lr_sweep.png",
                                         title=f"{sub[0]['label'].split(' (')[0]}: learning-rate sweep, all seeds"))

    tr_path = out / "transitions.csv"
    if "graph" not in args.skip and tr_path.exists():
        try:
            trans = pd.read_csv(tr_path)
        except pd.errors.EmptyDataError:  # written by bah when no arm had a transition
            trans = pd.DataFrame()
        if len(trans):
            arms = [(r["name"], r["label"], r["color"]) for r in runs]
            print(plots.fig_transition_graph(trans, lam, err, arms, fig / "transition_graph.png"))
        else:
            print("no transitions: transition graph skipped")

    if "occupancy" not in args.skip:
        occs = []
        for fname, lab in (("occupancy.csv", "NUTS, posterior ∝ exp(−n·L_n)"),
                           ("occupancy_nll0.5.csv", "NUTS, posterior ∝ exp(−n·L_n / 2)")):
            if (out / fname).exists():
                occs.append((lab, pd.read_csv(out / fname)))
        print(plots.fig_occupancy(occs, fig / "occupancy.png", llc=None))

    if "posterior" not in args.skip:
        pruns = [_with_llc(load_run(Path(args.post) / a)) for a in args.headline
                 if (Path(args.post) / a / "llc.npz").exists()]
        if pruns:
            print(plots.fig_loss_vs_llc(pruns, fig / "posterior_init_loss_vs_llc.png"))


if __name__ == "__main__":
    main()
