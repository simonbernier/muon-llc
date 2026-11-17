"""Transition tables and the Bayesian-antecedent check for every optimizer arm.

For each arm: label checkpoints, segment plateaus, count transitions alpha -> beta. For each
transition: Delta L (theory), Delta lambda (theory value where known, else our Table K.1
estimate), and the critical sample size from n dL + dlambda log n + dc = 0 with and without
the prior constant. Verdict per transition:

    antecedent     dL < 0, dlambda > 0 beyond its error, and the free energies cross
    inconclusive   dL < 0 but dlambda is within its error of 0, or the crossing exists only
                   with / only without the constant term
    violation      dL < 0, dlambda < 0 beyond its error and no crossing either way
    uphill         dL > 0 (not a transition the BAH speaks about; reported separately)

Writes arms_summary.csv, transitions.csv into --out.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ..tms.analysis import load_run
from ..tms.critical_points import PAPER_LLC_HAT, THEORY_LLC, by_name, critical_n
from ..tms.transitions import label_runs, summarize


def llc_table(path: Path | None) -> tuple[dict, dict, dict]:
    """name -> (lambda, err, source). Theory first, then our estimate (error = standard error
    over kept chains), then the paper's Table K.1 (error = their std over chains)."""
    lam, err, src = {}, {}, {}
    ours = pd.read_csv(path).set_index("name") if path is not None and path.exists() else None
    for name in by_name():
        if name in THEORY_LLC:
            lam[name], err[name], src[name] = THEORY_LLC[name], 0.0, "theory"
        elif ours is not None and name in ours.index and np.isfinite(ours.loc[name, "llc_hat"]):
            sem = ours.loc[name, "llc_std"] / np.sqrt(max(ours.loc[name, "chains_kept"], 1))
            lam[name], err[name], src[name] = ours.loc[name, "llc_hat"], sem, "ours"
        else:
            lam[name], err[name], src[name] = *PAPER_LLC_HAT[name], "paper"
    return lam, err, src


def verdict(a: str, b: str, lam, err) -> dict:
    cps = by_name()
    if a not in cps or b not in cps:
        return dict(dL=np.nan, dlam=np.nan, dlam_err=np.nan, n_cr=np.nan, n_cr_noconst=np.nan,
                    verdict="unlisted phase")
    dL = cps[b].loss - cps[a].loss
    dlam = lam[b] - lam[a]
    derr = float(np.hypot(err[a], err[b]))
    n1 = critical_n(a, b, llc=lam, with_const=True)
    n0 = critical_n(a, b, llc=lam, with_const=False)
    if dL >= 0:
        v = "uphill"
    elif dlam > derr and np.isfinite(n1) and np.isfinite(n0):
        v = "antecedent"
    elif dlam < -derr and not np.isfinite(n1) and not np.isfinite(n0):
        v = "violation"
    else:
        v = "inconclusive"
    return dict(dL=dL, dlam=dlam, dlam_err=derr, n_cr=n1, n_cr_noconst=n0, verdict=v)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", default="runs/tms/arms")
    ap.add_argument("--out", default="results/tms")
    ap.add_argument("--table", default="results/tms/table_k1.csv")
    ap.add_argument("--min-ckpts", type=int, default=3)
    ap.add_argument("--gap", type=int, default=2)
    ap.add_argument("--prefix", default="")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lam, err, src = llc_table(Path(args.table))
    arm_rows, tr_rows = [], []
    for d in sorted(Path(args.arms).iterdir()):
        if not (d / "run.npz").exists():
            continue
        run = load_run(d)
        labels, L = label_runs(run["thetas"])
        s = summarize(labels, run["path_len"], args.min_ckpts, args.gap)
        dw = [x for v in s["dwell"].values() for x in v]
        ck_epochs = float(np.median(np.diff(run["epochs"])))
        arm_rows.append(dict(
            arm=run["name"], label=run["label"], n_runs=s["n_runs"],
            runs_with_transition=s["runs_with_transition"], n_transitions=sum(s["transitions"].values()),
            frac_ckpts_off_critical=s["frac_other"], frac_off_critical_2nd_half=s["frac_other_late"],
            median_plateau_epochs=np.median([x[0] for x in dw]) * ck_epochs if dw else np.nan,
            median_plateau_path=np.median([x[1] for x in dw]) if dw else np.nan,
            total_path_median=float(np.median(run["path_len"][-1])),
            final_theory_loss_median=float(np.median(L[-1])),
            finals="; ".join(f"{k}:{v}" for k, v in s["finals"].most_common()),
        ))
        for (a, b), cnt in s["transitions"].most_common():
            tr_rows.append(dict(arm=run["name"], alpha=a, beta=b, count=cnt,
                                lam_alpha=lam.get(a, np.nan), lam_beta=lam.get(b, np.nan),
                                lam_src=f"{src.get(a, '?')}/{src.get(b, '?')}", **verdict(a, b, lam, err)))
    arms = pd.DataFrame(arm_rows)
    trans = pd.DataFrame(tr_rows)
    if len(trans):
        sgd_pairs = {(r.alpha, r.beta) for r in trans.itertuples() if r.arm == "sgd"}
        trans["seen_under_sgd"] = [(r.alpha, r.beta) in sgd_pairs for r in trans.itertuples()]
    arms.to_csv(out / f"{args.prefix}arms_summary.csv", index=False)
    trans.to_csv(out / f"{args.prefix}transitions.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 90)
    print(arms.drop(columns=["label"]).to_string(index=False, float_format=lambda x: f"{x:.3g}"))
    if len(trans):
        piv = trans.pivot_table(index="arm", columns="verdict", values="count", aggfunc="sum", fill_value=0)
        print("\ntransitions by verdict (counts over runs):\n", piv.to_string())
        print("\n", trans.to_string(index=False, float_format=lambda x: f"{x:.3g}"))


if __name__ == "__main__":
    main()
