"""Figures for the TMS reproduction. Same visual system as muon_llc.plotting: light surface,
fixed colour per optimizer (SGD blue, spectral GD orange, Muon aqua, AdamW yellow), one y-axis
per panel (loss and lambda-hat are stacked, not twinned), recessive grid."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial import ConvexHull  # noqa: E402

from ..plotting import AXIS, GRID, INK, INK2, MUTED, SURFACE  # noqa: E402,F401
from .analysis import data_losses, smooth, theory_loss  # noqa: E402
from .critical_points import critical_points  # noqa: E402
from .transitions import label_runs, plateaus  # noqa: E402

NEG, POS = "#d03238", "#1e8a4c"  # bias sign in the bar panels (status-like, not series colours)


def _levels(ax, lo, hi, c=6, side="right"):
    """Dashed lines at the critical-point loss levels inside [lo, hi], labelled at the edge."""
    last = None
    for cp in sorted(critical_points(c), key=lambda p: p.loss):
        if lo <= cp.loss <= hi:
            ax.axhline(cp.loss, color=AXIS, lw=0.7, ls=(0, (4, 3)), zorder=0)
            if last is None or cp.loss - last > 0.035 * (hi - lo):
                ax.annotate(cp.name, (1.0, cp.loss), xycoords=("axes fraction", "data"), xytext=(3, 0),
                            textcoords="offset points", va="center", fontsize=6.5, color=INK2,
                            annotation_clip=False)
                last = cp.loss


def draw_polygon(ax, W, lim=1.45):
    order = np.argsort(np.arctan2(W[1], W[0]))
    Ws = W[:, order]
    poly = np.column_stack([Ws, Ws[:, 0]])
    ax.plot(poly[0], poly[1], color="#6da7ec", lw=1.0)
    try:
        hull = ConvexHull(W.T)
        vs = list(hull.vertices) + [hull.vertices[0]]
        ax.plot(W[0, vs], W[1, vs], color=NEG, lw=0.8, ls="--", alpha=0.6)
    except Exception:
        pass
    for i in range(W.shape[1]):
        ax.plot([0, W[0, i]], [0, W[1, i]], color=INK, lw=0.8)
    ax.scatter([0], [0], s=10, color=NEG, zorder=3)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color(GRID)


def draw_bars(ax, W, b, xmax=1.6):
    order = np.argsort(np.arctan2(W[1], W[0]))
    norms, bs = np.linalg.norm(W, axis=0)[order], b[order]
    y = np.arange(1, len(bs) + 1)
    ax.barh(y + 0.2, norms, height=0.34, color=INK)
    ax.barh(y - 0.2, np.minimum(np.abs(bs), xmax), height=0.34, color=[NEG if v < 0 else POS for v in bs])
    for yi, v in zip(y, bs):
        if abs(v) > xmax:
            ax.text(xmax, yi - 0.2, "+", ha="right", va="center", color="white", fontsize=7, fontweight="bold")
    ax.set_xlim(0, xmax)
    ax.set_yticks([])
    ax.tick_params(labelsize=6)
    ax.grid(False)


def fig_trajectory(runs: list[dict], j: int, out: Path, title: str | None = None, n_snap: int = 3):
    """Fig. 1 / B.x analogue. One column per optimizer for the same seed j (same init, dataset,
    minibatch order): weight snapshots, ||W_i|| (black) and |b_i| (red < 0, green > 0), losses,
    and lambda-hat underneath (window-6 moving average of estimates every 30 epochs).
    Each run dict needs the output of analysis.load_run plus optional 'llc' (npz dict)."""
    A = len(runs)
    fig = plt.figure(figsize=(4.4 * A + 0.4, 7.6))
    gs = fig.add_gridspec(4, A, height_ratios=[1.25, 0.8, 2.0, 1.5], hspace=0.42, wspace=0.3,
                          left=0.07, right=0.95, top=0.9, bottom=0.07)
    for a, run in enumerate(runs):
        ep = run["epochs"]
        th = run["thetas"][:, j]
        L = theory_loss(th)
        train, test = data_losses(run, [j])
        labels, _ = label_runs(run["thetas"][:, j:j + 1])
        ps = plateaus(labels[:, 0], run["path_len"][:, j])
        # snapshots: middle of up to n_snap plateaus (last ones), else evenly spaced late times
        if ps:
            sel = ps[-n_snap:]
            snaps = [(p.start + p.end) // 2 for p in sel]
            snap_names = [p.name for p in sel]
        else:
            snaps = [int(f * (len(ep) - 1)) for f in np.linspace(0.3, 1.0, n_snap)]
            snap_names = ["other"] * n_snap
        sub = gs[0, a].subgridspec(1, n_snap, wspace=0.05)
        subb = gs[1, a].subgridspec(1, n_snap, wspace=0.12)
        for s, (t, nm) in enumerate(zip(snaps, snap_names)):
            W, b = th[t, :12].reshape(2, 6), th[t, 12:]
            ax = fig.add_subplot(sub[0, s])
            draw_polygon(ax, W)
            ax.set_title(f"{nm}  (epoch {int(ep[t])})", fontsize=7, color=INK2, pad=2)
            axb = fig.add_subplot(subb[0, s])
            draw_bars(axb, W, b)
            if s == 0 and a == 0:
                axb.set_ylabel("column", fontsize=7)
        ax = fig.add_subplot(gs[2, a])
        t0 = 2  # skip the first checkpoints (initial transient dominates the scale)
        lo, hi = L[t0:].min() - 0.008, min(L[t0:].max(), L[t0:].min() + 0.25) + 0.008
        ax.plot(ep, train[:, 0], color=MUTED, lw=1.0, label="train loss")
        ax.plot(ep, test[:, 0], color=MUTED, lw=1.0, ls=":", label="test loss")
        ax.plot(ep, L, color=run["color"], lw=2.0, label="theory loss L(w)")
        ax.scatter(ep[snaps], L[snaps], s=22, color=INK, zorder=4)
        ax.set_ylim(lo, hi)
        _levels(ax, lo, hi)
        ax.set_title(run["label"], fontsize=9.5, loc="left", color=INK)
        ax.tick_params(labelbottom=False)
        if a == 0:
            ax.set_ylabel("loss")
            ax.legend(fontsize=7, loc="upper right")
        ax2 = fig.add_subplot(gs[3, a], sharex=ax)
        if run.get("llc") is not None:
            z = run["llc"]
            jj = int(np.where(z["run_idx"] == j)[0][0])
            lam = z["llc"][:, jj]
            ax2.plot(ep[z["ck_idx"]], lam, color=run["color"], lw=0.6, alpha=0.35)
            ax2.plot(ep[z["ck_idx"]], smooth(lam, 6), color=run["color"], lw=2.0)
            ax2.axhline(0, color=AXIS, lw=0.7)
            lo2 = max(np.nanpercentile(lam[2:], 1) - 0.5, -6)
            ax2.set_ylim(min(lo2, -0.5), max(10.0, np.nanmax(smooth(lam, 6)[2:]) + 0.5))
        ax2.set_xlabel("epoch")
        if a == 0:
            ax2.set_ylabel(r"estimated LLC  $\hat\lambda$")
    fig.suptitle(title or f"Seed {j}: same initial point, dataset and minibatch order under each optimizer",
                 fontsize=11, x=0.07, ha="left")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------- ensembles
def fig_ensemble(runs: list[dict], out: Path, max_runs: int = 200, ylim=(0.04, 0.34), title=None):
    """Theory loss of every run against epoch, one panel per optimizer, critical levels dashed."""
    A = len(runs)
    fig, axes = plt.subplots(1, A, figsize=(3.6 * A + 0.6, 4.6), sharey=True, squeeze=False)
    for ax, run in zip(axes[0], runs):
        L = theory_loss(run["thetas"][:, :max_runs])
        ax.plot(run["epochs"], L, color=run["color"], lw=0.5, alpha=0.25)
        ax.set_ylim(*ylim)
        _levels(ax, *ylim)
        ax.set_title(run["label"], fontsize=9.5, loc="left")
        ax.set_xlabel("epoch")
    axes[0, 0].set_ylabel("theory loss L(w)")
    fig.suptitle(title or f"All trajectories ({min(max_runs, runs[0]['thetas'].shape[1])} seeds per optimizer): "
                 "plateaus sit on the k-gon loss levels", fontsize=11, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 0.98, 0.95))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_loss_vs_llc(runs: list[dict], out: Path, title=None, xlim=(-16, 12), ylim=(0.03, 0.30)):
    """Fig. 3 analogue. Top: (lambda-hat, L) for every estimated checkpoint, shaded by training
    time (light = early, dark = late). Bottom: theory loss against SGD steps. One column per arm."""
    from matplotlib.colors import LinearSegmentedColormap

    A = len(runs)
    fig, axes = plt.subplots(2, A, figsize=(3.7 * A + 0.6, 7.4), squeeze=False)
    for a, run in enumerate(runs):
        z = run["llc"]
        ep = run["epochs"][z["ck_idx"]]
        L = theory_loss(run["thetas"][z["ck_idx"]][:, z["run_idx"]])
        lam = z["llc"]
        cmap = LinearSegmentedColormap.from_list("seq", ["#e3e2dc", run["color"], "#111111"])
        ax = axes[0, a]
        t = np.repeat(ep[:, None], lam.shape[1], 1)
        sc = ax.scatter(lam.ravel(), L.ravel(), c=t.ravel(), cmap=cmap, s=4, lw=0, rasterized=True)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        _levels(ax, *ylim)
        ax.set_title(run["label"], fontsize=9.5, loc="left")
        ax.set_xlabel(r"estimated LLC  $\hat\lambda$")
        cb = fig.colorbar(sc, ax=ax, orientation="horizontal", pad=0.14, fraction=0.04, aspect=40)
        cb.set_label("epoch", fontsize=7)
        cb.ax.tick_params(labelsize=6)
        ax = axes[1, a]
        Lall = theory_loss(run["thetas"])
        ax.plot(run["epochs"] * run["steps_per_epoch"], Lall, color=run["color"], lw=0.4, alpha=0.3)
        ax.set_ylim(*ylim)
        _levels(ax, *ylim)
        ax.set_xlabel("optimizer steps")
        if a == 0:
            axes[0, a].set_ylabel("theory loss L(w)")
            ax.set_ylabel("theory loss L(w)")
    fig.suptitle(title or "Trajectories started from posterior samples (n = 100)", fontsize=11, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 0.98, 0.96))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_transition_graph(trans, lam: dict, err: dict, arms: list[tuple[str, str, str]], out: Path,
                         title=None):
    """Fig. D.1 analogue, one panel per optimizer. Points: critical points at (lambda, L).
    Lines: observed dynamical transitions, width ~ log(count). Solid = loss down and lambda up
    (what the BAH needs); dashed = anything else. arms: [(arm name, label, colour)]."""
    cps = critical_points(6)
    A = len(arms)
    fig, axes = plt.subplots(1, A, figsize=(4.1 * A + 0.4, 5.0), sharey=True, sharex=True, squeeze=False)
    for ax, (arm, label, color) in zip(axes[0], arms):
        for cp in cps:
            ax.errorbar(lam[cp.name], cp.loss, xerr=err[cp.name], fmt="o", ms=4, color=INK,
                        ecolor=AXIS, elinewidth=1.0, capsize=2, zorder=3)
            ax.annotate(cp.name, (lam[cp.name], cp.loss), xytext=(0, 5), textcoords="offset points",
                        fontsize=6.5, color=INK2, ha="center")
        sub = trans[trans["arm"] == arm]
        for _, r in sub.iterrows():
            if r["alpha"] not in lam or r["beta"] not in lam or not np.isfinite(r["dL"]):
                continue
            a, b = r["alpha"], r["beta"]
            good = (r["dL"] < 0) and (r["dlam"] > 0)
            pa = (lam[a], [c.loss for c in cps if c.name == a][0])
            pb = (lam[b], [c.loss for c in cps if c.name == b][0])
            ax.annotate("", xy=pb, xytext=pa, zorder=2,
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=0.8 + 0.9 * np.log10(r["count"] + 1),
                                        ls="-" if good else (0, (3, 2)), shrinkA=3, shrinkB=3, alpha=0.9))
        n_t = int(sub["count"].sum()) if len(sub) else 0
        ax.set_title(f"{label}\n{n_t} transitions", fontsize=9, loc="left")
        ax.set_xlabel(r"local learning coefficient $\lambda$")
    axes[0, 0].set_ylabel("theory loss L")
    fig.suptitle(title or "Observed dynamical transitions between critical points", fontsize=11, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


PHASE_COLORS = {"4": "#eda100", "4+": "#e87ba4", "5": "#1baf7a", "5+": "#2a78d6", "6": "#eb6834",
                "other": "#898781"}


def fig_occupancy(occ, out: Path, llc: dict | None = None, title=None):
    """Fig. 2: posterior share of each phase against n. Left: free-energy formula. Right: NUTS."""
    from .critical_points import free_energy

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    ns = np.arange(100, 1201)
    names = ["4", "4+", "5", "5+", "6"]
    F = np.stack([free_energy(nm, ns, llc=llc) for nm in names])
    P = np.exp(-(F - F.min(0)))
    P /= P.sum(0)
    for nm, p in zip(names, P):
        axes[0].plot(ns, p, color=PHASE_COLORS[nm], lw=2)
        k = int(np.argmax(p))
        axes[0].annotate(nm, (ns[k], p[k]), xytext=(0, 4), textcoords="offset points", fontsize=8,
                         color=INK2, ha="center")
    axes[0].set_title("Free-energy formula", fontsize=9.5, loc="left")
    axes[0].set_xlabel("sample size n")
    axes[0].set_ylabel("share of posterior")
    if occ is not None and len(occ):
        g = occ.groupby("n")
        for nm in names + ["other"]:
            m, se = g[nm].mean(), g[nm].sem().fillna(0)
            axes[1].errorbar(m.index, m.values, yerr=se.values, color=PHASE_COLORS[nm], lw=1.6, marker="o",
                             ms=4, capsize=2, label=nm)
        axes[1].legend(ncol=6, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, 1.0))
        axes[1].set_title(f"NUTS posterior samples ({int(g.size().median())} datasets per n, mean ± s.e.)",
                          fontsize=9.5, loc="left", pad=14)
    axes[1].set_xlabel("sample size n")
    fig.suptitle(title or "Bayesian phase transition 5 → 6 (optimizer-independent)", fontsize=11, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
