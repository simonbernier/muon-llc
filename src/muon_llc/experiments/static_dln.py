"""Phase B2: static DLN calibration against Aoyagi's table.

For each rank r, build the regression problem whose TRUE map is the top-r truncation A_r of A*
(same U, V, data), and balanced weights whose product is exactly A_r, so the point is a global
minimum of that problem (Aoyagi's setting). Run the MALA ladder over a gamma sweep and compare
λ̂ with λ_Aoyagi(r). Writes b2.csv, b2_ladder.csv, b2_resolution.png and prints the recommended
gamma (smallest mean |λ̂ - λ| across ranks).

Note: the truncated point is a *saddle* of the rank-4 loss, not a minimum, and the tempered
posterior at nβ = n/log n escapes into the deeper basin unless γ is very large (measured λ̂
around -1500 = nβ · (loss drop)). That is why B2 uses the truncated target rather than the
full one; the trajectory experiments (C1) face the same issue and report λ̂ along the
resolution axis γ/nβ instead of a single number.

    uv run python -m muon_llc.experiments.static_dln --config configs/b2_static.yaml
"""

from __future__ import annotations

import time

import pandas as pd
import torch

from ..dln import DLNProblem, balanced_weights, flatten
from ..io import run_dir, save_config
from ..llc import estimate_checkpoint, resolve_nbeta
from ..plotting import plot_resolution
from ..theory.aoyagi import dln_learning_coefficient
from .common import base_parser, build_problem, load


def truncated_target(A: torch.Tensor, r: int) -> torch.Tensor:
    U, S, Vh = torch.linalg.svd(A)
    S = S.clone()
    S[r:] = 0
    return U @ torch.diag(S) @ Vh


def run_static(cfg: dict, verbose: bool = True):
    problem = build_problem(cfg)
    st = cfg["static"]
    d = run_dir(cfg)
    save_config(cfg, d)
    widths = cfg["model"]["widths"]
    nb0 = resolve_nbeta(cfg["llc"], problem.n)
    rows, ladder = [], []
    g = torch.Generator().manual_seed(int(cfg["model"]["seed"]) + 123)
    t0 = time.time()
    for r in st["ranks"]:
        A_r = truncated_target(problem.A_star.double().cpu(), int(r))
        A_r_dev = A_r.to(problem.device, problem.dtype)
        # problem_r: same inputs, true map A_r -> the constructed point is a global minimum
        problem_r = DLNProblem(problem.widths, problem.X, A_r_dev @ problem.X, A_r_dev)
        ws = balanced_weights(A_r, widths, g)
        theta = flatten(ws).to(problem.device, problem.dtype)
        truth = dln_learning_coefficient(widths, int(r)).value
        for mult in st.get("nbeta_multipliers", [1.0]):
            llc_cfg = dict(cfg["llc"])
            llc_cfg["nbeta"] = nb0 * float(mult)
            rr, lr = estimate_checkpoint(problem_r, theta, llc_cfg, step=int(r), seed=int(cfg["llc"].get("seed", 0)),
                                         gammas=st["gammas"], estimators=llc_cfg.get("estimators", ["mala"]),
                                         verbose=False)
            for x in rr:
                x.update(rank=int(r), lambda_true=truth, nbeta_mult=float(mult))
            for x in lr:
                x.update(rank=int(r), lambda_true=truth, nbeta_mult=float(mult))
            rows += rr
            ladder += lr
            if verbose:
                sub = [x for x in rr if x["estimator"] == "mala"]
                s = "  ".join(f"γ={x['gamma']:g}: {x['lambda_hat']:.2f}±{x['err']:.2f}" for x in sub)
                print(f"rank {r}  λ_Aoyagi={truth:g}  nβ={nb0 * mult:.0f}  {s}")
    df = pd.DataFrame(rows)
    lad = pd.DataFrame(ladder)
    df.to_csv(d / "b2.csv", index=False)
    lad.to_csv(d / "b2_ladder.csv", index=False)

    # recommended gamma: minimise mean |λ̂ - λ_true| over ranks (headline MALA estimate, mult 1)
    m = df[(df["estimator"] == "mala") & (df["nbeta_mult"] == 1.0)]
    score = m.assign(dev=(m["lambda_hat"] - m["lambda_true"]).abs()).groupby("gamma")["dev"].mean()
    best_gamma = float(score.idxmin()) if len(score) else float("nan")
    with open(d / "b2_summary.txt", "w") as f:
        f.write("mean |lambda_hat - lambda_Aoyagi| over ranks, by gamma (MALA, nbeta = n/log n)\n")
        f.write(score.to_string() + f"\n\nrecommended gamma: {best_gamma}\n")
    if verbose:
        print(score.to_string())
        print(f"recommended gamma = {best_gamma:g}  ({time.time() - t0:.0f}s)")
    truth = {int(r): dln_learning_coefficient(widths, int(r)).value for r in st["ranks"]}
    plot_resolution(m, d / "b2_resolution.png", group_col="rank",
                    group_labels={r: f"rank {r} (λ={truth[r]:g})" for r in truth}, truth=truth,
                    title="B2 static DLN: λ̂ vs resolution (dashed = Aoyagi, dotted = ½·soft-rank)")
    return df, lad, best_gamma


def main():
    run_static(load(base_parser("B2 static DLN calibration").parse_args()))


if __name__ == "__main__":
    main()
