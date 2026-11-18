"""B3: why the static calibration (B2) reads below Aoyagi's values.

Same points as B2 (balanced weights whose product is the rank-r truncation, sampled on the problem
whose true map is that truncation). Each arm of `offset.arms` samples every (rank, γ) pair on the
rungs nβ = (n/log n)·multiplier and reports λ̂ with convergence diagnostics:

    sampler: mala | gibbs      MALA is exact at any step size; Gibbs has no step size at all
    start_nbeta_dl: x          start each chain off the centre, far enough that nβ·ΔL ≈ x at the
                               first rung. Chains from the centre approach the answer from below,
                               these from above; converged chains agree.
    adapt_decay, thin, n_burnin, n_draws    see slt.mala.run_mala

    uv run python -m muon_llc.dln.experiments.offset_study --config configs/dln/b3_offset.yaml --arms B
    uv run python -m muon_llc.dln.experiments.offset_study --config configs/dln/b3_offset.yaml --report

Each arm writes offset_<arm>.csv and windows_<arm>.csv; --report merges them into offset.csv and
windows.csv and writes offset_summary.txt, scalar_chain.csv and offset_convergence_x<mult>.png.
"""

from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from ...config import run_dir, save_config
from ...slt.estimators import ess, split_rhat, wbic_estimate, window_means
from ...slt.mala import run_mala
from ...style import INK2, MUTED, SERIES
from ..gibbs import run_gibbs
from ..llc import resolve_nbeta
from ..model import DLNProblem, balanced_weights, flatten
from ..theory.aoyagi import dln_learning_coefficient
from ..theory.quadrature import scalar_chain_expectation
from .common import base_parser, build_problem, load
from .static_dln import truncated_target

ARM_COLOURS = [SERIES["gd"], SERIES["spectral"], SERIES["muon"], SERIES["adamw"], SERIES["fifth"], INK2]


def jitter_for(problem: DLNProblem, thetas: torch.Tensor, nbeta: float, target: float, seed: int) -> np.ndarray:
    """Per-centre std σ such that nβ·E[L(θ0 + σξ) − L(θ0)] = target (bisection on log σ)."""
    g = torch.Generator().manual_seed(seed)
    B, D = thetas.shape
    xi = torch.randn(B, 1, 64, D, generator=g, dtype=problem.dtype).to(problem.device)
    c = thetas.reshape(B, 1, 1, D)
    L0 = problem.loss(c)

    def excess(log_sd: torch.Tensor) -> torch.Tensor:
        return nbeta * (problem.loss(c + log_sd.exp().reshape(B, 1, 1, 1) * xi) - L0).mean(dim=(1, 2))

    lo = torch.full((B,), np.log(1e-4), dtype=problem.dtype, device=problem.device)
    hi = torch.full((B,), np.log(10.0), dtype=problem.dtype, device=problem.device)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        above = excess(mid) > target
        hi, lo = torch.where(above, mid, hi), torch.where(above, lo, mid)
    return (0.5 * (lo + hi)).exp().cpu().numpy()


def run_arm(cfg: dict, arm: str, verbose: bool = True) -> None:
    spec = dict(cfg["offset"]["arms"][arm])
    if spec.get("device"):
        cfg = {**cfg, "device": spec["device"]}
    problem = build_problem(cfg)
    st, off = cfg["static"], cfg["offset"]
    widths = cfg["model"]["widths"]
    d = run_dir(cfg)
    save_config(cfg, d)
    nb0 = resolve_nbeta(cfg["llc"], problem.n)
    nbs = nb0 * np.asarray(st["nbeta_multipliers"], dtype=np.float64)
    # the same generator and order as static_dln, so these are B2's centres
    g = torch.Generator().manual_seed(int(cfg["model"]["seed"]) + 123)
    pairs, thetas, targets, truth = [], [], [], {}
    for r in st["ranks"]:
        A_r = truncated_target(problem.A_star.double().cpu(), int(r))
        theta = flatten(balanced_weights(A_r, widths, g))
        lc = dln_learning_coefficient(widths, int(r))
        truth[int(r)] = (lc.value, lc.multiplicity)
        for gamma in st["gammas"]:
            pairs.append((int(r), float(gamma)))
            thetas.append(theta)
            targets.append(A_r)
    thetas = torch.stack(thetas).to(problem.device, problem.dtype)
    A = torch.stack(targets).to(problem.device, problem.dtype).reshape(len(pairs), 1, 1, *targets[0].shape)
    prob = DLNProblem(problem.widths, problem.X, A @ problem.X, A)
    gammas = [gm for _, gm in pairs]
    seed = int(cfg["llc"].get("seed", 0)) + int(spec.get("seed", 0))
    jitter = 0.0
    if spec.get("start_nbeta_dl"):
        jitter = jitter_for(prob, thetas, float(nbs[0]), float(spec["start_nbeta_dl"]), seed + 7)
    kw = dict(n_chains=int(off.get("n_chains", 16)), n_burnin=int(spec["n_burnin"]), n_draws=int(spec["n_draws"]),
              seed=seed, init_jitter=jitter, thin=int(spec.get("thin", 1)))
    t0 = time.time()
    if spec["sampler"] == "gibbs":
        rec = run_gibbs(prob, thetas, nbs, gammas, **kw)
    else:
        rec = run_mala(prob, thetas, nbs, gammas, float(cfg["llc"]["mala"].get("eps0", 1e-3)),
                       target_accept=float(cfg["llc"]["mala"].get("target_accept", 0.574)),
                       adapt_decay=float(spec.get("adapt_decay", 0.0)), **kw)
    secs = time.time() - t0
    tail = rec.extras.get("accept_burnin_tail")
    n_win = int(off.get("windows", 20))
    rows, wins = [], []
    for i, (r, gamma) in enumerate(pairs):
        lam, m = truth[r]
        for j, nb in enumerate(nbs):
            x = rec.delta_loss[i, j]
            e = wbic_estimate(x, nb, gamma, rec.acceptance[i, j], rec.eps[i, j])
            h = x.shape[1] // 2
            burn = rec.delta_loss_burnin[i, j]
            base = dict(arm=arm, sampler=spec["sampler"], rank=r, gamma=gamma, nbeta=float(nb),
                        nbeta_mult=float(st["nbeta_multipliers"][j]))
            rows.append(dict(
                **base, lambda_true=lam, multiplicity=m, lambda_asymptotic=lam - (m - 1) / np.log(nb),
                lambda_hat=e.lambda_hat, err=e.err, rhat=split_rhat(x), ess=ess(x),
                first_half=float(nb * x[:, :h].mean()), second_half=float(nb * x[:, h:].mean()),
                burnin_last_tenth=float(nb * burn[:, -max(1, burn.shape[1] // 10):].mean()),
                var_over_mean=e.var_over_mean, acceptance=e.acceptance,
                accept_burnin_tail=float(tail[i, j]) if tail is not None else np.nan, eps=e.eps,
                start_jitter=float(np.broadcast_to(jitter, (len(pairs),))[i]),
                n_burnin=kw["n_burnin"], n_draws=kw["n_draws"], thin=kw["thin"], n_chains=kw["n_chains"]))
            for w, v in enumerate(nb * window_means(x, n_win)):
                wins.append(dict(**base, window=w, lambda_hat=float(v)))
    pd.DataFrame(rows).to_csv(d / f"offset_{arm}.csv", index=False)
    pd.DataFrame(wins).to_csv(d / f"windows_{arm}.csv", index=False)
    if verbose:
        df = pd.DataFrame(rows)
        print(f"arm {arm} ({spec['sampler']}, {kw['n_burnin']}+{kw['n_draws']} steps, {secs:.0f}s on {problem.device})")
        print(df[["rank", "gamma", "nbeta_mult", "lambda_true", "lambda_hat", "err", "rhat", "ess",
                  "first_half", "second_half", "acceptance"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))


def report(cfg: dict) -> None:
    d = run_dir(cfg)
    arms = [a for a in cfg["offset"]["arms"] if (d / f"offset_{a}.csv").exists()]
    if not arms:
        raise FileNotFoundError(f"no offset_<arm>.csv under {d}")
    df = pd.concat([pd.read_csv(d / f"offset_{a}.csv") for a in arms], ignore_index=True)
    win = pd.concat([pd.read_csv(d / f"windows_{a}.csv") for a in arms], ignore_index=True)
    df.to_csv(d / "offset.csv", index=False)
    win.to_csv(d / "windows.csv", index=False)

    # the smallest network of the family, where the expectation is known without sampling
    sc = pd.DataFrame([dict(nbeta=nb, gamma=gm, exact=scalar_chain_expectation(nb, gm),
                            asymptotic=0.5 - 2.0 / np.log(nb))
                       for gm in (0.1, 1.0) for nb in (1e1, 1e2, 1e3, 1e4, 1e5, 1e6)])
    sc.to_csv(d / "scalar_chain.csv", index=False)

    def table(sub, col, fmt):
        t = sub.pivot_table(index=["gamma", "rank"], columns="arm", values=col, dropna=False)
        return t.reindex(columns=arms).to_string(float_format=lambda v: format(v, fmt), na_rep="-")

    with open(d / "offset_summary.txt", "w", encoding="utf-8") as f:
        for mult in sorted(df["nbeta_mult"].unique()):
            sub = df[df["nbeta_mult"] == mult]
            ref = sub.groupby(["gamma", "rank"])[["lambda_true", "lambda_asymptotic"]].first()
            lam = sub.pivot_table(index=["gamma", "rank"], columns="arm", values="lambda_hat").reindex(columns=arms)
            f.write(f"lambda_hat by arm, nbeta = {mult:g} x n/log n\n")
            f.write(ref.join(lam).to_string(float_format=lambda v: f"{v:.2f}") + "\n\n")
            f.write("standard error over chains\n" + table(sub, "err", ".2f") + "\n\n")
            f.write("split R-hat\n" + table(sub, "rhat", ".3f") + "\n\n")
            f.write("effective sample size\n" + table(sub, "ess", ".0f") + "\n\n")
            f.write("acceptance after burn-in\n" + table(sub, "acceptance", ".2f") + "\n\n")
            f.write("acceptance in the last tenth of burn-in\n" + table(sub, "accept_burnin_tail", ".2f") + "\n\n")
        f.write("widths 1-1-1-1, rank 0 (lambda = 1/2, m = 3): exact E[n beta dL] by quadrature\n")
        f.write(sc.to_string(index=False, float_format=lambda v: f"{v:.4f}") + "\n")

    ranks, gammas = sorted(df["rank"].unique()), sorted(df["gamma"].unique())
    labels = {a: f"{a}: {s['sampler']}, {'off-centre start' if s.get('start_nbeta_dl') else 'centre start'}, "
                 f"{int(s['n_draws']):,} steps" for a, s in cfg["offset"]["arms"].items()}
    for mult in sorted(df["nbeta_mult"].unique()):
        fig, axes = plt.subplots(len(gammas), len(ranks), figsize=(2.7 * len(ranks), 2.5 * len(gammas)),
                                 sharex=True, squeeze=False)
        for a_i, gm in enumerate(gammas):
            for b_i, r in enumerate(ranks):
                ax = axes[a_i][b_i]
                ref = df[(df["rank"] == r) & (df["gamma"] == gm) & (df["nbeta_mult"] == mult)].iloc[0]
                ax.axhline(ref["lambda_true"], color=INK2, lw=1.0, ls="--")
                ax.axhline(ref["lambda_asymptotic"], color=MUTED, lw=1.0, ls=":")
                for k, arm in enumerate(arms):
                    w = win[(win["arm"] == arm) & (win["rank"] == r) & (win["gamma"] == gm)
                            & (win["nbeta_mult"] == mult)].sort_values("window")
                    ax.plot((w["window"] + 0.5) / len(w), w["lambda_hat"], color=ARM_COLOURS[k % len(ARM_COLOURS)],
                            lw=1.2, label=labels[arm] if (a_i, b_i) == (0, 0) else None)
                ax.set_title(f"rank {r}, γ = {gm:g}", fontsize=8.5)
                if a_i == len(gammas) - 1:
                    ax.set_xlabel("fraction of the chain")
                if b_i == 0:
                    ax.set_ylabel("λ̂ in a time window")
        fig.legend(loc="lower center", ncol=2, fontsize=7.5)
        fig.suptitle(f"B3: λ̂ along the chain at nβ = {mult:g} × n/log n   "
                     "(dashed: Aoyagi's λ, dotted: λ − (m−1)/log nβ)", fontsize=9.5)
        fig.tight_layout(rect=(0, 0.04 + 0.025 * ((len(arms) + 1) // 2), 1, 0.96))
        fig.savefig(d / f"offset_convergence_x{mult:g}.png", dpi=160)
        plt.close(fig)
    print(open(d / "offset_summary.txt", encoding="utf-8").read())


def main():
    p = base_parser("B3 calibration offset study")
    p.add_argument("--arms", nargs="*", default=None, help="arms to sample (default: all)")
    p.add_argument("--report", action="store_true", help="merge the arms' tables and draw the figures")
    a = p.parse_args()
    cfg = load(a)
    arms = list(cfg["offset"]["arms"]) if a.arms is None and not a.report else (a.arms or [])
    for arm in arms:
        run_arm(cfg, arm)
    if a.report or a.arms is None:
        report(cfg)


if __name__ == "__main__":
    main()
