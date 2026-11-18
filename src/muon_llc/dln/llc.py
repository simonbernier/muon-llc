"""Per-checkpoint LLC estimation: SGLD baseline, Gibbs or MALA beta ladder with slope fit, Hessian
soft-rank baseline, the resolution (gamma) sweep, and the optional surrogate-target estimate.
Produces flat rows for llc.csv. Every estimator is batched over checkpoints and γ: all
(checkpoint, γ) pairs of a run share one SGLD call and one ladder call per `llc.max_centres` pairs.

Estimators (the `estimator` column):
    sgld           devinterp-style SGLD at (nβ, γ), centred at the checkpoint w_t
    gibbs          block Gibbs (`dln.gibbs`), headline rung of the β-ladder (nβ ≈ n/log n). Exact
                   draws from each layer's conditional: no step size, no rejection. The default,
                   because MALA does not converge on these posteriors (results/dln/REPORT.md, B3)
    gibbs_ladder   slope fit over the ladder: λ (intercept) and m̂, plus thermodynamic route
    mala           full-batch MALA on the same ladder, kept for comparison
    mala_ladder    its slope fit
    hessian        ½ · soft-rank of the exact Hessian at resolution γ/nβ (A1 prediction)
    surrogate      the ladder (sampler `llc.surrogate_sampler`) on the surrogate problem whose TRUE
                   map is the checkpoint's own product
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

from ..slt.estimators import fit_ladder, wbic_estimate
from ..slt.mala import run_mala
from ..slt.sgld import run_sgld
from .gibbs import run_gibbs
from .model import DLNProblem, product_from_flat
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


def sgld_rows(problem: DLNProblem, thetas: torch.Tensor, steps, gammas, cfg_llc: dict, seed: int,
              hessian: dict | None = None) -> list[dict]:
    """SGLD rows for every (checkpoint, γ) pair in one batched sampler call.

    thetas (K, D); steps (K,); gammas: one list of γ per checkpoint. `hessian` optionally maps
    (step, γ) to a `hessian_summary` dict, copied into the rows for comparison.
    """
    s = cfg_llc.get("sgld", {})
    nb = resolve_nbeta(cfg_llc, problem.n)
    eps = float(s.get("eps", 1e-3))
    pairs = [(k, float(g)) for k in range(len(steps)) for g in gammas[k]]
    if not pairs:
        return []
    centres = torch.stack([thetas[k] for k, _ in pairs])
    rec = run_sgld(problem, centres, nb, [g for _, g in pairs], eps, int(s.get("n_chains", 16)),
                   int(s.get("n_burnin", 200)), int(s.get("n_draws", 1000)), int(s.get("batch_size", 128)),
                   seed=seed)
    rows = []
    for i, (k, gamma) in enumerate(pairs):
        e = wbic_estimate(rec.delta_loss[i], nb, gamma)
        hs = (hessian or {}).get((int(steps[k]), gamma), {})
        rows.append(_row(steps[k], "sgld", nbeta=nb, gamma=gamma, lambda_hat=e.lambda_hat, err=e.err,
                         var_over_mean=e.var_over_mean, mean2_over_var=e.mean2_over_var, eps=eps,
                         lambda_quad=hs.get("lambda_quad", np.nan),
                         n_neg_hessian=hs.get("n_neg_hessian", np.nan)))
    return rows


def headline_estimator(names) -> str:
    """The exact-target ladder estimate present in a table: `gibbs` if it was run, else `mala`."""
    return "gibbs" if "gibbs" in set(names) else "mala"


def ladder_rows(problem: DLNProblem, thetas: torch.Tensor, steps, gammas, cfg_llc: dict, seed: int,
                eigs: list | None = None, boot_seeds=None, targets: torch.Tensor | None = None,
                names: tuple[str, str] = ("mala", "mala_ladder"),
                sampler: str = "mala") -> tuple[list[dict], list[dict]]:
    """β-ladder rows for every (checkpoint, γ) pair, `llc.max_centres` pairs per sampler call.
    `sampler` is "mala" (settings `llc.mala`) or "gibbs" (settings `llc.gibbs`).

    `targets` (K, H_L, H_0), if given, replaces the true map per checkpoint (the surrogate
    estimator: each checkpoint is sampled on the problem whose true map is its own product).
    Returns (llc rows named `names[0]` for the headline rung and `names[1]` for the ladder fit,
    per-rung ladder rows named `names[0]`).
    """
    if sampler not in ("mala", "gibbs"):
        raise ValueError(f"unknown sampler {sampler!r}")
    m = cfg_llc.get(sampler, {})
    n_boot = int(m.get("n_boot", cfg_llc.get("mala", {}).get("n_boot", 200)))
    n = problem.n
    nb = resolve_nbeta(cfg_llc, n)
    nbs = nbeta_ladder(cfg_llc, n)
    k0 = int(np.argmin(np.abs(np.log(nbs / nb))))  # rung closest to the default nβ: the headline number
    floor = float(m.get("fit_min_var_over_mean", 0.0))
    pairs = [(k, float(g)) for k in range(len(steps)) for g in gammas[k]]
    chunk = max(1, int(m.get("max_centres", cfg_llc.get("max_centres", 256))))  # a sampler may set its own
    rows, rungs = [], []
    for ci, start in enumerate(range(0, len(pairs), chunk)):
        part = pairs[start:start + chunk]
        ks = [k for k, _ in part]
        prob = problem
        if targets is not None:  # one true map per centre, broadcast over rungs and chains
            A = targets[ks].reshape(len(part), 1, 1, *targets.shape[-2:])
            prob = DLNProblem(problem.widths, problem.X, A @ problem.X, A, problem.noise_var)
        if sampler == "gibbs":
            rec = run_gibbs(prob, thetas[ks], nbs, [g for _, g in part], int(m.get("n_chains", 16)),
                            int(m.get("n_burnin", 200)), int(m.get("n_draws", 500)), seed=seed + ci)
        else:
            rec = run_mala(prob, thetas[ks], nbs, [g for _, g in part], float(m.get("eps0", 1e-3)),
                           int(m.get("n_chains", 16)), int(m.get("n_burnin", 300)), int(m.get("n_draws", 1000)),
                           seed=seed + ci, target_accept=float(m.get("target_accept", 0.574)))
        for i, (k, gamma) in enumerate(part):
            step = int(steps[k])
            e_k = None if eigs is None else eigs[k]
            hs = hessian_summary(e_k, nb, gamma) if e_k is not None else {}
            per_rung = []
            for j in range(len(nbs)):
                e = wbic_estimate(rec.delta_loss[i, j], nbs[j], gamma, rec.acceptance[i, j], rec.eps[i, j])
                per_rung.append(e)
                rungs.append(dict(step=step, estimator=names[0], gamma=gamma, nbeta=float(nbs[j]),
                                        lambda_hat=e.lambda_hat, err=e.err, var_over_mean=e.var_over_mean,
                                        mean2_over_var=e.mean2_over_var, acceptance=e.acceptance, eps=e.eps,
                                        lambda_quad=hessian_summary(e_k, nbs[j], gamma)["lambda_quad"]
                                        if e_k is not None else np.nan))
            e0 = per_rung[k0]
            rows.append(_row(step, names[0], nbeta=float(nbs[k0]), gamma=gamma, lambda_hat=e0.lambda_hat,
                             err=e0.err, var_over_mean=e0.var_over_mean, mean2_over_var=e0.mean2_over_var,
                             acceptance=e0.acceptance, eps=e0.eps,
                             lambda_quad=hs.get("lambda_quad", np.nan),
                             n_neg_hessian=hs.get("n_neg_hessian", np.nan)))
            # slope estimate over the rungs in the scale-free regime (var/mean above a floor)
            use = np.array([e.var_over_mean >= floor for e in per_rung])
            if use.sum() < 3:
                use = np.ones(len(nbs), dtype=bool)
            lf = fit_ladder(rec.delta_loss[i], nbs, gamma, n_boot=n_boot,
                            seed=seed if boot_seeds is None else int(boot_seeds[k]), use=use)
            rows.append(_row(step, names[1], nbeta=nb, gamma=gamma, lambda_hat=lf.lambda_slope,
                             err=lf.lambda_slope_err, m_hat=lf.m_hat, m_hat_err=lf.m_hat_err,
                             lambda_thermo=lf.lambda_thermo, lambda_thermo_err=lf.lambda_thermo_err,
                             lambda_wbic_min=lf.row()["lambda_wbic_min"],
                             lambda_wbic_max=lf.row()["lambda_wbic_max"],
                             acceptance=float(np.mean(rec.acceptance[i])),
                             lambda_quad=hs.get("lambda_quad", np.nan),
                             n_neg_hessian=hs.get("n_neg_hessian", np.nan)))
    return rows, rungs


def estimate_checkpoints(
    problem: DLNProblem,
    thetas: torch.Tensor,
    steps,
    gammas,
    cfg_llc: dict,
    seed: int,
    estimators=None,
    verbose: bool = False,
) -> tuple[list[dict], list[dict]]:
    """Run the configured estimators at K checkpoints. thetas (K, D); steps (K,); gammas: one
    list of γ per checkpoint. Every (checkpoint, γ) pair shares a sampler call per estimator.
    Returns (llc rows, ladder rows)."""
    nb = resolve_nbeta(cfg_llc, problem.n)
    steps = [int(s) for s in steps]
    K = len(steps)
    thetas = thetas.to(problem.device, problem.dtype).reshape(K, -1)
    gammas = [[float(g) for g in gs] for gs in gammas]
    estimators = list(cfg_llc.get("estimators", ["sgld", "mala"])) if estimators is None else list(estimators)
    use_hessian = bool(cfg_llc.get("hessian", True))
    boot_seeds = [seed + 17 * k for k in range(K)]
    rows, ladder = [], []

    def hessian_rows(eigs, name):
        out = []
        for k in range(K):
            for gamma in gammas[k]:
                hs = hessian_summary(eigs[k], nb, gamma)
                out.append(_row(steps[k], name, nbeta=nb, gamma=gamma, lambda_hat=hs["lambda_quad"], err=0.0,
                                **{c: hs[c] for c in ("lambda_quad", "n_neg_hessian", "n_pos_hessian")}))
        return out

    eigs = None
    if use_hessian:
        eigs = [hessian_eigenvalues(problem, thetas[k]) for k in range(K)]
        rows += hessian_rows(eigs, "hessian")

    if "surrogate" in estimators:
        W = product_from_flat(thetas, problem.widths).detach()  # (K, H_L, H_0): each checkpoint's own map
        eigs_s = None
        if use_hessian:
            eigs_s = [hessian_eigenvalues(DLNProblem(problem.widths, problem.X, W[k] @ problem.X, W[k]),
                                          thetas[k]) for k in range(K)]
            rows += hessian_rows(eigs_s, "surrogate_hessian")
        r, lr = ladder_rows(problem, thetas, steps, gammas, cfg_llc, seed + 99991 + 500, eigs_s,
                            [s + 99991 for s in boot_seeds], targets=W, names=("surrogate", "surrogate_ladder"),
                            sampler=str(cfg_llc.get("surrogate_sampler", "mala")))
        rows += r
        ladder += lr

    if "sgld" in estimators:
        hess = None if eigs is None else {(steps[k], g): hessian_summary(eigs[k], nb, g)
                                          for k in range(K) for g in gammas[k]}
        rows += sgld_rows(problem, thetas, steps, gammas, cfg_llc, seed, hessian=hess)

    for name in ("gibbs", "mala"):
        if name in estimators:
            r, lr = ladder_rows(problem, thetas, steps, gammas, cfg_llc, seed + 500, eigs, boot_seeds,
                                names=(name, name + "_ladder"), sampler=name)
            rows += r
            ladder += lr

    if verbose:
        order = {s: i for i, s in enumerate(steps)}
        by: dict = {}
        for x in rows:
            by.setdefault((order[x["step"]], x["gamma"]), {})[x["estimator"]] = x
        for (k, gamma), d in sorted(by.items()):
            parts = []
            if "sgld" in d:
                parts.append(f"sgld {d['sgld']['lambda_hat']:7.2f}±{d['sgld']['err']:.2f}")
            for nm in ("gibbs", "mala"):
                if nm in d:
                    lad = d[nm + "_ladder"]
                    parts.append(f"{nm} {d[nm]['lambda_hat']:7.2f}±{d[nm]['err']:.2f}"
                                 f" (slope {lad['lambda_hat']:.2f}, m̂ {lad['m_hat']:.2f},"
                                 f" acc {lad['acceptance']:.2f})")
            if "surrogate" in d:
                parts.append(f"surrogate {d['surrogate']['lambda_hat']:7.2f}±{d['surrogate']['err']:.2f}")
            if "hessian" in d:
                parts.append(f"½·soft-rank {d['hessian']['lambda_hat']:.2f}")
            print(f"  step {steps[k]:>7d} γ={gamma:g}  " + "  ".join(parts))
    return rows, ladder


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
    """One checkpoint: `estimate_checkpoints` with K = 1 (its γ values still share sampler calls)."""
    gammas = [float(cfg_llc.get("gamma", 10.0))] if gammas is None else [float(g) for g in gammas]
    return estimate_checkpoints(problem, theta.reshape(1, -1), [step], [gammas], cfg_llc, seed,
                                estimators=estimators, verbose=verbose)


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
