"""Per-checkpoint LLC estimation: SGLD baseline, MALA beta ladder with slope fit, Hessian
soft-rank baseline, the resolution (gamma) sweep, and the optional surrogate-target estimate.
Produces flat rows for llc.csv.

Estimators (the `estimator` column):
    sgld           devinterp-style SGLD at (nβ, γ), centred at the checkpoint w_t
    mala           full-batch MALA, headline rung of the β-ladder (nβ ≈ n/log n)
    mala_ladder    slope fit over the ladder: λ (intercept) and m̂, plus thermodynamic route
    hessian        ½ · soft-rank of the exact Hessian at resolution γ/nβ (A1 prediction)
    surrogate      MALA on the surrogate problem whose TRUE map is the checkpoint's own product
                   W_t (loss ½||W - W_t||²). w_t is then a global minimum, so the estimate is the
                   learning coefficient of the current function's fibre (Aoyagi's λ at the
                   effective rank of W_t) without the basin-escape artefact that afflicts the
                   direct estimate at a saddle. It is NOT the LLC of the actual loss at w_t.
    surrogate_ladder  slope fit for the surrogate

Basin escape: at a rank-r saddle of the true loss the tempered posterior with nβ = n/log n
escapes to the deeper basin (loss drop ½ s_{r+1}²) unless γ/nβ exceeds roughly s_{r+1}^{4/3}/3,
which is the same scale at which learned directions (Hessian eigenvalues ~ s_r^{4/3}) start to be
resolved. There is no single γ that works for every plateau, which is why the trajectory
experiments report λ̂ as a family over γ/nβ (the resolution axis of the research plan).
"""

from __future__ import annotations

import numpy as np
import torch

from .dln import DLNProblem, product_from_flat
from .sampling.estimators import fit_ladder, wbic_estimate
from .sampling.mala import run_mala
from .sampling.sgld import run_sgld
from .theory.hessian import hessian_eigenvalues, hessian_summary

ROW_KEYS = [
    "step", "estimator", "nbeta", "gamma", "lambda_hat", "err", "m_hat", "m_hat_err",
    "var_over_mean", "mean2_over_var", "acceptance", "eps", "lambda_quad", "n_neg_hessian",
    "n_pos_hessian", "lambda_wbic_min", "lambda_wbic_max", "lambda_thermo", "lambda_thermo_err",
]


def default_nbeta(n: int) -> float:
    """devinterp's default: n / log n."""
    return float(n / np.log(n))


def resolve_nbeta(cfg_llc: dict, n: int) -> float:
    nb = cfg_llc.get("nbeta")
    return default_nbeta(n) if nb is None else float(nb)


def nbeta_ladder(cfg_llc: dict, n: int) -> np.ndarray:
    lad = cfg_llc.get("nbeta_ladder", {}) or {}
    nb = resolve_nbeta(cfg_llc, n)
    k = int(lad.get("n", 8))
    lo, hi = float(lad.get("lo", 0.1)), float(lad.get("hi", 10.0))
    return nb * np.logspace(np.log10(lo), np.log10(hi), k)


def _row(step, estimator, **kw) -> dict:
    r = {k: np.nan for k in ROW_KEYS}
    r.update(step=int(step), estimator=estimator)
    r.update(kw)
    return r


def estimate_checkpoint(
    problem: DLNProblem,
    theta: torch.Tensor,
    cfg_llc: dict,
    step: int,
    seed: int,
    gammas=None,
    estimators=None,
    verbose: bool = False,
) -> tuple[list[dict], list[dict]]:
    """Run the configured estimators at one checkpoint. Returns (llc rows, ladder rows)."""
    n = problem.n
    nb = resolve_nbeta(cfg_llc, n)
    gammas = [float(cfg_llc.get("gamma", 10.0))] if gammas is None else [float(g) for g in gammas]
    estimators = list(cfg_llc.get("estimators", ["sgld", "mala"])) if estimators is None else estimators
    rows, ladder_rows = [], []

    eigs = None
    if cfg_llc.get("hessian", True):
        eigs = hessian_eigenvalues(problem, theta)

    if "surrogate" in estimators:
        W_t = product_from_flat(theta.to(problem.device, problem.dtype), problem.widths).detach()
        surrogate = DLNProblem(problem.widths, problem.X, W_t @ problem.X, W_t)
        sub_est = ["mala"]
        r2, l2 = estimate_checkpoint(surrogate, theta, {**cfg_llc, "hessian": cfg_llc.get("hessian", True)},
                                     step, seed + 99991, gammas=gammas, estimators=sub_est, verbose=False)
        rename = {"mala": "surrogate", "mala_ladder": "surrogate_ladder", "hessian": "surrogate_hessian"}
        for x in r2:
            x["estimator"] = rename.get(x["estimator"], x["estimator"])
        for x in l2:
            x["estimator"] = "surrogate"
        rows += r2
        ladder_rows += l2
        if verbose:
            for x in r2:
                if x["estimator"] == "surrogate":
                    print(f"  step {step:>7d} γ={x['gamma']:g} surrogate λ={x['lambda_hat']:7.2f} ± {x['err']:.2f}"
                          f"  (½·soft-rank {x['lambda_quad']:.2f})")

    for gi, gamma in enumerate(gammas):
        hs = hessian_summary(eigs, nb, gamma) if eigs is not None else {}
        if eigs is not None:
            rows.append(_row(step, "hessian", nbeta=nb, gamma=gamma, lambda_hat=hs["lambda_quad"],
                             err=0.0, **{k: hs[k] for k in ("lambda_quad", "n_neg_hessian", "n_pos_hessian")}))

        if "sgld" in estimators:
            s = cfg_llc.get("sgld", {})
            rec = run_sgld(
                problem, theta, nb, gamma, float(s.get("eps", 1e-3)), int(s.get("n_chains", 16)),
                int(s.get("n_burnin", 200)), int(s.get("n_draws", 1000)), int(s.get("batch_size", 128)),
                seed=seed + 1000 * gi,
            )
            e = wbic_estimate(rec.delta_loss, nb, gamma)
            rows.append(_row(step, "sgld", nbeta=nb, gamma=gamma, lambda_hat=e.lambda_hat, err=e.err,
                             var_over_mean=e.var_over_mean, mean2_over_var=e.mean2_over_var,
                             eps=float(s.get("eps", 1e-3)),
                             lambda_quad=hs.get("lambda_quad", np.nan),
                             n_neg_hessian=hs.get("n_neg_hessian", np.nan)))
            if verbose:
                print(f"  step {step:>7d} γ={gamma:g} sgld  λ={e.lambda_hat:7.2f} ± {e.err:.2f}"
                      f"  var/mean={e.var_over_mean:.2f}")

        if "mala" in estimators:
            m = cfg_llc.get("mala", {})
            nbs = nbeta_ladder(cfg_llc, n)
            rec = run_mala(
                problem, theta, nbs, gamma, float(m.get("eps0", 1e-3)), int(m.get("n_chains", 16)),
                int(m.get("n_burnin", 300)), int(m.get("n_draws", 1000)), seed=seed + 1000 * gi + 500,
                target_accept=float(m.get("target_accept", 0.574)),
            )
            per_rung = []
            for k in range(len(nbs)):
                e = wbic_estimate(rec.delta_loss[k], nbs[k], gamma, rec.acceptance[k], rec.eps[k])
                per_rung.append(e)
                ladder_rows.append(dict(step=int(step), estimator="mala", gamma=gamma, nbeta=float(nbs[k]),
                                        lambda_hat=e.lambda_hat, err=e.err, var_over_mean=e.var_over_mean,
                                        mean2_over_var=e.mean2_over_var, acceptance=e.acceptance, eps=e.eps,
                                        lambda_quad=hessian_summary(eigs, nbs[k], gamma)["lambda_quad"]
                                        if eigs is not None else np.nan))
            # WBIC at the default nbeta (closest rung) as the headline MALA number
            k0 = int(np.argmin(np.abs(np.log(nbs / nb))))
            e0 = per_rung[k0]
            rows.append(_row(step, "mala", nbeta=float(nbs[k0]), gamma=gamma, lambda_hat=e0.lambda_hat,
                             err=e0.err, var_over_mean=e0.var_over_mean, mean2_over_var=e0.mean2_over_var,
                             acceptance=e0.acceptance, eps=e0.eps,
                             lambda_quad=hs.get("lambda_quad", np.nan),
                             n_neg_hessian=hs.get("n_neg_hessian", np.nan)))
            # slope estimate over the rungs in the scale-free regime (var/mean above a floor)
            floor = float(m.get("fit_min_var_over_mean", 0.0))
            use = np.array([e.var_over_mean >= floor for e in per_rung])
            if use.sum() < 3:
                use = np.ones(len(nbs), dtype=bool)
            lf = fit_ladder(rec.delta_loss, nbs, gamma, n_boot=int(m.get("n_boot", 200)), seed=seed, use=use)
            rows.append(_row(step, "mala_ladder", nbeta=nb, gamma=gamma, lambda_hat=lf.lambda_slope,
                             err=lf.lambda_slope_err, m_hat=lf.m_hat, m_hat_err=lf.m_hat_err,
                             lambda_thermo=lf.lambda_thermo, lambda_thermo_err=lf.lambda_thermo_err,
                             lambda_wbic_min=lf.row()["lambda_wbic_min"],
                             lambda_wbic_max=lf.row()["lambda_wbic_max"],
                             acceptance=float(np.mean(rec.acceptance)),
                             lambda_quad=hs.get("lambda_quad", np.nan),
                             n_neg_hessian=hs.get("n_neg_hessian", np.nan)))
            if verbose:
                print(f"  step {step:>7d} γ={gamma:g} mala  λ(nβ≈{nbs[k0]:.0f})={e0.lambda_hat:7.2f} ± {e0.err:.2f}"
                      f"  slope λ={lf.lambda_slope:7.2f} ± {lf.lambda_slope_err:.2f} m̂={lf.m_hat:.2f}"
                      f"  acc={np.mean(rec.acceptance):.2f}  λ_quad={hs.get('lambda_quad', np.nan):.2f}")
    return rows, ladder_rows


def select_checkpoints(steps: np.ndarray, spec) -> np.ndarray:
    """spec: 'all' | int stride | list of steps | {'every': k} | {'max': K} (K evenly chosen)."""
    steps = np.asarray(steps)
    if spec is None or spec == "all":
        return steps
    if isinstance(spec, int):
        return steps[:: max(1, spec)]
    if isinstance(spec, dict):
        if "every" in spec:
            return steps[:: int(spec["every"])]
        if "max" in spec:
            K = int(spec["max"])
            if len(steps) <= K:
                return steps
            idx = np.unique(np.round(np.linspace(0, len(steps) - 1, K)).astype(int))
            return steps[idx]
    wanted = set(int(s) for s in spec)
    return np.array([s for s in steps if int(s) in wanted])
