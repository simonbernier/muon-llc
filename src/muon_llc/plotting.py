"""Figures for C1/C2/C3 and B2. Matplotlib, light surface, fixed categorical colour order:
GD = blue, spectral GD = orange, full Muon = aqua; sequential blues for ordered families
(singular values, gamma sweep); text and chrome in neutral inks. One y-axis per panel."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SERIES = {"gd": "#2a78d6", "spectral": "#eb6834", "muon": "#1baf7a", "other": "#eda100",
          "fifth": "#e87ba4"}
SEQ_BLUE = ["#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#b7d3f6", "#cde2fb"]
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.6,
    "font.size": 9, "legend.frameon": False, "figure.dpi": 130,
})


def label_for(optimizer_cfg: dict) -> tuple[str, str]:
    """(display label, colour key) for an optimizer config."""
    if optimizer_cfg.get("name") == "gd":
        return "GD", "gd"
    mom = optimizer_cfg.get("momentum", 0)
    orth, wd = optimizer_cfg.get("orth", "exact"), optimizer_cfg.get("weight_decay", 0)
    if mom == 0 and orth == "exact" and wd == 0:
        return "spectral GD", "spectral"
    return "Muon", "muon"


def _xlog(ax, steps):
    steps = np.asarray(steps)
    ax.set_xscale("symlog", linthresh=max(1, float(np.min(steps[steps > 0])) if np.any(steps > 0) else 1))
    ax.set_xlim(left=-0.3, right=float(steps.max()) * 1.3 if steps.size else 1)
    ax.set_xlabel("training step")


# ------------------------------------------------------------------------------ C1 / C3
def plot_singular_values(runs: list[dict], out: Path, targets=None, predictions: dict | None = None):
    """runs: [{label, color, trajectory (DataFrame)}]. One panel per run, sequential blues by rank."""
    fig, axes = plt.subplots(1, len(runs), figsize=(4.2 * len(runs), 3.2), sharey=True, squeeze=False)
    for ax, run in zip(axes[0], runs):
        tr = run["trajectory"]
        sv_cols = [c for c in tr.columns if c.startswith("sv_")]
        for i, c in enumerate(sv_cols):
            col = SEQ_BLUE[min(i, len(SEQ_BLUE) - 1)]
            ax.plot(tr["step"], tr[c], color=col, lw=1.4 if i < 4 else 0.9)
            if i < 4:
                ax.annotate(f"s{i + 1}", (tr["step"].iloc[-1], tr[c].iloc[-1]), xytext=(3, 0),
                            textcoords="offset points", color=INK2, fontsize=8, va="center")
        if targets is not None:
            for s in targets:
                ax.axhline(s, color=AXIS, lw=0.6, ls=":")
        if predictions and run["label"] in predictions:
            pt, ps = predictions[run["label"]]
            for i in range(ps.shape[1]):
                ax.plot(pt, ps[:, i], color=run["color"], lw=0.9, ls="--", alpha=0.8)
        _xlog(ax, tr["step"].values)
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 10)
        ax.set_title(f"{run['label']}: product singular values", color=INK, loc="left")
    axes[0][0].set_ylabel("singular value of W")
    if predictions:
        axes[0][0].plot([], [], color=INK2, ls="--", lw=0.9, label="A4 aligned prediction")
        axes[0][0].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_lambda_overlay(runs: list[dict], out: Path, staircase: dict[int, float] | None = None,
                        estimator: str = "mala", show_quad: bool = True, title: str | None = None):
    """runs: [{label, color, llc (DataFrame), gamma, pred (steps, lambda) optional}]."""
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    all_steps = []
    for run in runs:
        df = run["llc"]
        df = df[(df["gamma"] == run["gamma"])]
        d = df[df["estimator"] == estimator].sort_values("step")
        if d.empty:
            continue
        all_steps.append(d["step"].values)
        ax.errorbar(d["step"], d["lambda_hat"], yerr=d["err"], color=run["color"], marker="o", ms=4,
                    capsize=2, lw=1.4, label=f"{run['label']}  λ̂ ({estimator})")
        if show_quad:
            h = df[df["estimator"] == "hessian"].sort_values("step")
            if not h.empty:
                ax.plot(h["step"], h["lambda_hat"], color=run["color"], lw=0.9, ls=":", alpha=0.9,
                        label=f"{run['label']}  ½·soft-rank(Hessian)")
        if run.get("pred") is not None:
            pt, pl = run["pred"]
            ax.step(pt, pl, where="post", color=run["color"], lw=0.9, ls="--", alpha=0.8,
                    label=f"{run['label']}  A4 prediction")
    if staircase:
        for r, lam in staircase.items():
            ax.axhline(lam, color=AXIS, lw=0.6, ls="-")
            ax.annotate(f"λ(r={r}) = {lam:g}", (1.0, lam), xycoords=("axes fraction", "data"),
                        xytext=(-4, 2), textcoords="offset points", ha="right", fontsize=7, color=MUTED)
    if all_steps:
        _xlog(ax, np.concatenate(all_steps))
    ax.set_ylabel("local learning coefficient λ̂")
    g = runs[0]["gamma"] if runs else None
    ax.set_title(title or f"λ̂ along training (γ = {g:g}, nβ = n/log n)", loc="left", color=INK)
    ax.legend(loc="lower right", fontsize=7.5, ncol=1)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_lambda_family(runs: list[dict], out: Path, estimator: str = "mala", staircase=None,
                       ylim=None, title=None):
    """λ̂(t) at every γ of the resolution sweep, one panel per run, sequential blues by γ
    (dark = small γ, fine resolution). The plan's 'report λ(resolution), not the corner value'."""
    fig, axes = plt.subplots(1, len(runs), figsize=(4.6 * len(runs), 3.6), sharey=True, squeeze=False)
    for ax, run in zip(axes[0], runs):
        df = run["llc"]
        d = df[df["estimator"] == estimator]
        gammas = sorted(d["gamma"].unique())
        for i, g in enumerate(gammas):
            dd = d[d["gamma"] == g].sort_values("step")
            if len(dd) < 2:
                continue
            col = SEQ_BLUE[int(round(i / max(1, len(gammas) - 1) * (len(SEQ_BLUE) - 2)))]
            ax.errorbar(dd["step"], dd["lambda_hat"], yerr=dd["err"], color=col, marker="o", ms=3, lw=1.2,
                        capsize=1.5, label=f"γ = {g:g}")
        if staircase:
            for r, lam in staircase.items():
                ax.axhline(lam, color=AXIS, lw=0.6)
        _xlog(ax, d["step"].values if len(d) else np.array([1]))
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_title(f"{run['label']}: λ̂ ({estimator}) by resolution", loc="left", color=INK)
        ax.legend(fontsize=7, ncol=2, loc="upper left")
    axes[0][0].set_ylabel("λ̂")
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=9, color=INK2)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_ladder(ladder: pd.DataFrame, out: Path, steps=None, title: str = "MALA β-ladder", ylim=None):
    """λ̂_WBIC(β) per checkpoint, with the Hessian soft-rank curve; flat = scale-free regime."""
    gam = ladder["gamma"].iloc[0]
    ladder = ladder[ladder["gamma"] == gam]
    steps = sorted(ladder["step"].unique()) if steps is None else steps
    steps = steps[:: max(1, len(steps) // 8)] if len(steps) > 8 else steps
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for i, s in enumerate(steps):
        d = ladder[ladder["step"] == s].sort_values("nbeta")
        col = SEQ_BLUE[int(round(i / max(1, len(steps) - 1) * (len(SEQ_BLUE) - 3)))]
        ax.errorbar(d["nbeta"], d["lambda_hat"], yerr=d["err"], color=col, marker="o", ms=3.5, lw=1.2,
                    label=f"step {int(s)}")
        ax.plot(d["nbeta"], d["lambda_quad"], color=col, lw=0.8, ls=":")
    ax.set_xscale("log")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel("nβ")
    ax.set_ylabel("nβ · E[ΔL]  (WBIC estimate)")
    ax.set_title(f"{title} (γ = {gam:g}); dotted = ½·soft-rank(Hessian)", loc="left")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_resolution(df: pd.DataFrame, out: Path, group_col: str, group_labels: dict | None = None,
                    truth: dict | None = None, estimator: str = "mala", title: str | None = None):
    """λ̂ vs γ/nβ for several groups (ranks or checkpoints), with Aoyagi truth lines."""
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    groups = sorted(df[group_col].unique())
    for i, gkey in enumerate(groups):
        d = df[(df[group_col] == gkey) & (df["estimator"] == estimator)].sort_values("gamma")
        col = SEQ_BLUE[int(round(i / max(1, len(groups) - 1) * (len(SEQ_BLUE) - 3)))]
        lab = group_labels.get(gkey, str(gkey)) if group_labels else str(gkey)
        ax.errorbar(d["gamma"] / d["nbeta"], d["lambda_hat"], yerr=d["err"], color=col, marker="o",
                    ms=4, lw=1.3, label=lab)
        h = df[(df[group_col] == gkey) & (df["estimator"] == "hessian")].sort_values("gamma")
        if not h.empty:
            ax.plot(h["gamma"] / h["nbeta"], h["lambda_hat"], color=col, lw=0.8, ls=":")
        if truth and gkey in truth:
            ax.axhline(truth[gkey], color=col, lw=0.6, ls="--", alpha=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("resolution  γ / nβ")
    ax.set_ylabel("λ̂")
    ax.set_title(title or "λ̂ against resolution (dashed = Aoyagi, dotted = ½·soft-rank)", loc="left")
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


# ------------------------------------------------------------------------------ C2
def plot_loglog_fit(x, y, out: Path, xlabel: str, ylabel: str, fit: tuple | None = None,
                    predicted_exponent: float | None = None, series: dict | None = None, title=""):
    """Log-log scatter with the fitted power law and the predicted slope for comparison.
    series: optional {label: (x, y)} to draw several groups in the categorical order."""
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    if series:
        for i, (lab, (xs, ys)) in enumerate(series.items()):
            col = list(SERIES.values())[i % len(SERIES)]
            ax.plot(xs, ys, "o", ms=5, color=col, label=lab)
    else:
        ax.plot(x, y, "o", ms=5, color=SERIES["gd"])
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(x) & (x > 0)
    if fit is not None and ok.any():
        b, A, r2 = fit
        xx = np.logspace(np.log10(x[ok].min()), np.log10(x[ok].max()), 50)
        ax.plot(xx, A * xx**b, color=INK2, lw=1.2, label=f"fit: slope {b:.2f} (R²={r2:.2f})")
        if predicted_exponent is not None:
            ax.plot(xx, A * (xx / xx[0]) ** predicted_exponent * (A * xx[0] ** b) / A, color=SERIES["spectral"],
                    lw=1.0, ls="--", label=f"predicted slope {predicted_exponent:.2f}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_semilog_lines(series: dict, out: Path, xlabel: str, ylabel: str, title="", logx=True, logy=False,
                       hline=None):
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for i, (lab, (xs, ys)) in enumerate(series.items()):
        col = SEQ_BLUE[int(round(i / max(1, len(series) - 1) * (len(SEQ_BLUE) - 3)))]
        ax.plot(xs, ys, "o-", ms=4, lw=1.3, color=col, label=str(lab))
    if hline is not None:
        ax.axhline(hline, color=AXIS, lw=0.8, ls="--")
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
