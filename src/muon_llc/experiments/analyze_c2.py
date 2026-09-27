"""C2 analysis: fit the three A5 scaling laws from sweep summaries.

    uv run python -m muon_llc.experiments.analyze_c2 --sweep runs/c2_ns_steps --kind ns
    uv run python -m muon_llc.experiments.analyze_c2 --sweep runs/c2_weight_decay --kind wd
    uv run python -m muon_llc.experiments.analyze_c2 --sweep runs/c2_momentum --kind momentum

ns:       breakdown κ*(k) where the lag ratio [t½(s_r)/t½(s_1)] / (s_r/s_1)^{1/L} exceeds 2
          (≈1 under uniform growth, ≫1 under saddle-to-saddle); fit log κ* = slope·k + c;
          predicted slope log a = log 3.4445 ≈ 1.237 (also plotted as κ* vs a^k, exponent 1).
wd:       final s_1 vs λ_wd for λ_wd above the cap threshold; predicted log-log slope -L per depth;
          cap independence of lr.
momentum: residual amplitude at saturation vs lr (predicted slope 1) and vs μ.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ..plotting import plot_loglog_fit, plot_semilog_lines
from ..theory.muon_scaling import NS_SMALL_SIGMA_GAIN, fit_loglog, wd_product_cap


def breakdown_kappa(sub: pd.DataFrame, col: str = "lag_ratio", threshold: float = 2.0) -> float:
    """κ at which `col` first exceeds `threshold` (log-interpolated). lag_ratio ≈ 1 under uniform
    growth and ≫ 1 (or inf) under saddle-to-saddle; inf means no breakdown within the grid."""
    sub = sub.sort_values("data.kappa")
    k = sub["data.kappa"].values.astype(float)
    u = sub[col].values.astype(float)
    u = np.where(np.isnan(u), np.inf, u)
    broken = u > threshold
    if not broken.any():
        return float("inf")
    i = int(np.nonzero(broken)[0][0])
    if i == 0 or not np.isfinite(u[i]):
        return float(k[i]) if i == 0 else float(np.sqrt(k[i - 1] * k[i]))
    x0, x1, y0, y1 = np.log(k[i - 1]), np.log(k[i]), np.log(u[i - 1]), np.log(u[i])
    return float(np.exp(x0 + (np.log(threshold) - y0) * (x1 - x0) / (y1 - y0)))


def analyze_ns(df: pd.DataFrame, out: Path, threshold: float = 2.0):
    rows = []
    for k, sub in df.groupby("optimizer.ns_steps"):
        rows.append(dict(ns_steps=int(k), kappa_star=breakdown_kappa(sub, "lag_ratio", threshold),
                         kappa_star_uniformity=breakdown_kappa(sub.assign(inv=1 / sub["uniformity"].clip(1e-9)),
                                                               "inv", 2.0)))
    res = pd.DataFrame(rows).sort_values("ns_steps")
    finite = res[np.isfinite(res["kappa_star"])]
    slope = float("nan")
    if len(finite) >= 2:
        slope, _ = np.polyfit(finite["ns_steps"], np.log(finite["kappa_star"]), 1)
    res.to_csv(out / "ns_breakdown.csv", index=False)
    txt = (f"breakdown kappa* by NS steps (lag_ratio > {threshold}; inf = no breakdown in the grid):\n"
           f"{res.to_string(index=False)}\n\n"
           f"fitted d log kappa* / dk = {slope:.3f}   predicted log a = {np.log(NS_SMALL_SIGMA_GAIN):.3f}\n")
    print(txt)
    (out / "ns_summary.txt").write_text(txt)
    # lag-ratio and uniformity curves vs kappa, one line per k
    series = {f"k = {int(k)}": (sub.sort_values("data.kappa")["data.kappa"].values,
                               np.minimum(sub.sort_values("data.kappa")["lag_ratio"].fillna(np.inf).values, 1e3))
              for k, sub in df.groupby("optimizer.ns_steps")}
    plot_semilog_lines(series, out / "ns_lag_ratio.png", "target condition number κ",
                       "lag ratio  [t½(s_r)/t½(s_1)] / (s_r/s_1)^{1/L}", logy=True,
                       title="Newton-Schulz: saddle-to-saddle lag vs κ (1 = uniform growth)", hline=threshold)
    series = {f"k = {int(k)}": (sub.sort_values("data.kappa")["data.kappa"].values,
                               sub.sort_values("data.kappa")["uniformity"].values)
              for k, sub in df.groupby("optimizer.ns_steps")}
    plot_semilog_lines(series, out / "ns_uniformity.png", "target condition number κ",
                       "uniformity  s_r / s_1 at t½(s_r)", title="Newton-Schulz: uniform growth vs κ", hline=0.5)
    if len(finite) >= 2:
        x = NS_SMALL_SIGMA_GAIN ** finite["ns_steps"].values.astype(float)
        fit = fit_loglog(x, finite["kappa_star"].values)
        plot_loglog_fit(x, finite["kappa_star"].values, out / "ns_breakdown.png", "a^k  (a = 3.4445)",
                        "breakdown κ*", fit=fit, predicted_exponent=1.0, title="NS breakdown: κ* ∝ a^k ?")
    return res


def analyze_wd(df: pd.DataFrame, out: Path):
    s1 = df["pred_final_s1"].max()  # the uncapped target s_1
    txt = []
    series = {}
    for (depth, lr), sub in df.groupby(["depth", "lr"]):
        sub = sub.sort_values("optimizer.weight_decay")
        wd = sub["optimizer.weight_decay"].values.astype(float)
        capped = np.array([wd_product_cap(w, int(depth)) < s1 for w in wd])
        series[f"L={depth}, lr={lr}"] = (wd, sub["final_s1"].values)
        if capped.sum() >= 2:
            b, A, r2 = fit_loglog(wd[capped], sub["final_s1"].values[capped])
            txt.append(f"depth {depth} lr {lr}: slope {b:.3f} (predicted {-int(depth)}), R2 {r2:.3f}")
    # cap independence of lr: spread of final_s1 across lr at fixed (depth, wd)
    spread = df.groupby(["depth", "optimizer.weight_decay"])["final_s1"].agg(["mean", "std"])
    txt.append("\nfinal s_1 across lr (mean, std) by depth and wd:\n" + spread.to_string())
    text = "\n".join(txt)
    print(text)
    (out / "wd_summary.txt").write_text(text)
    plot_semilog_lines(series, out / "wd_cap.png", "weight decay λ_wd", "final s_1", logy=True,
                       title="Weight decay cap: final s_1 vs λ_wd (prediction: min(s_1, λ_wd^-L))")
    # prediction overlay figure per depth
    for depth, sub in df.groupby("depth"):
        sub = sub.sort_values("optimizer.weight_decay")
        wd = sub["optimizer.weight_decay"].values.astype(float)
        pred = np.minimum(s1, np.array([wd_product_cap(w, int(depth)) for w in wd]))
        ser = {"measured": (wd, sub["final_s1"].values), "predicted": (wd, pred)}
        plot_semilog_lines(ser, out / f"wd_cap_L{depth}.png", "λ_wd", "final s_1", logy=True,
                           title=f"depth {depth}: measured vs min(s_1, λ_wd^-L)")


def analyze_momentum(df: pd.DataFrame, out: Path):
    txt = []
    series = {}
    for mu, sub in df.groupby("optimizer.momentum"):
        agg = sub.groupby("optimizer.lr")["resid_amp_max"].mean()
        series[f"μ = {mu}"] = (agg.index.values.astype(float), agg.values)
        b, A, r2 = fit_loglog(agg.index.values.astype(float), agg.values)
        txt.append(f"mu {mu}: residual amplitude ∝ lr^{b:.2f} (predicted 1), R2 {r2:.2f}")
    uni = df.groupby(["optimizer.momentum", "model.init_scale"])["uniformity"].mean().unstack()
    txt.append("\nuniformity (rows μ, cols init scale):\n" + uni.to_string())
    text = "\n".join(txt)
    print(text)
    (out / "momentum_summary.txt").write_text(text)
    plot_semilog_lines(series, out / "momentum_residual.png", "learning rate η", "residual amplitude at saturation",
                       logy=True, title="Momentum: overshoot amplitude vs η (prediction ∝ η)")
    series2 = {f"init {a}": (sub.sort_values("optimizer.momentum")["optimizer.momentum"].values + 1e-3,
                             sub.sort_values("optimizer.momentum")["uniformity"].values)
               for a, sub in df.groupby("model.init_scale")}
    plot_semilog_lines(series2, out / "momentum_uniformity.png", "momentum μ (+1e-3)", "uniformity",
                       title="Momentum: uniform growth preserved?", hline=0.5)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep", required=True)
    p.add_argument("--kind", choices=["ns", "wd", "momentum"], required=True)
    a = p.parse_args()
    root = Path(a.sweep)
    df = pd.read_csv(root / "summary.csv")
    df = df[~df.get("diverged", pd.Series(False, index=df.index)).astype(bool)]
    {"ns": analyze_ns, "wd": analyze_wd, "momentum": analyze_momentum}[a.kind](df, root)


if __name__ == "__main__":
    main()
