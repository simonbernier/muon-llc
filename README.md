# muon-llc — LLC trajectories under Muon vs GD in deep linear networks

Phase C of the research plan *LLC trajectories under Muon*: does Muon change the sequence of
developmental phase transitions during training, as measured by the local learning coefficient
(LLC) over time? Deep linear networks (DLNs) are the test bed because the training dynamics are
known in closed form (GD from small init learns singular values one at a time; spectral GD grows
them all at the same rate) and the learning coefficient is known exactly (Aoyagi 2024).

This repository contains the parts of Phases A and B that Phase C needs (the Aoyagi table, the
trajectory predictions, the estimators and their static calibration) plus the C1/C2/C3
experiments as runnable configs. Everything is a `uv` project in PyTorch. Models are tiny
(192 parameters); the cost is chains × temperatures × checkpoints × resolutions.

## TMS: dynamical vs Bayesian phase transitions under Muon

`src/muon_llc/tms/` reproduces Chen et al. (2023), arXiv 2310.06301 (Toy Model of Superposition,
r = 2, c = 6) and adds AdamW, spectral GD and Muon to its figures. Results, figures and the
Bayesian-antecedent analysis are in [`results/tms/REPORT.md`](results/tms/REPORT.md). In short:
under every optimizer trajectories plateau on the same k-gon critical points, and none of the 925
observed transitions lowers the loss while lowering the learning coefficient.

| Path | Purpose |
|---|---|
| `tms/model.py`, `tms/potential.py` | model, data, exact population loss |
| `tms/critical_points.py` | the 18 k-gon families, losses, prior constants, phase classifier, free energy |
| `tms/train.py` | batched multi-seed training (SGD, AdamW, spectral GD, Muon) |
| `tms/llc.py`, `tms/posterior.py` | SGLD λ̂ (batched over centres), NUTS posterior (extra `mcmc`) |
| `tms/transitions.py`, `tms/plots.py` | plateau segmentation, transitions, figures |
| `experiments/tms_*.py` | `tms_arms`, `tms_estimate`, `tms_critical_llc`, `tms_posterior`, `tms_bah`, `tms_figures`, `tms_smoke` |

## Quickstart

```bash
uv sync --extra dev                                   # torch from PyPI (CUDA wheels on Linux)
uv run pytest                                         # ~2 min on CPU
uv run python -m muon_llc.theory.aoyagi --widths 8 8 8 8   # A3 table
uv run python -m muon_llc.experiments.smoke           # reduced B2 + C1 on CPU, ~2 min -> runs/smoke/
bash scripts/run_all_gpu.sh                           # the full Phase C on a GPU box (vast.ai)
```

Every script takes `--config <yaml>` and `--set key.sub=value` overrides, e.g.
`--set train.steps=50000 llc.mala.n_chains=64`. Configs live in `configs/` and extend
`base.yaml`.

## What is where

| Path | Purpose |
|---|---|
| `src/muon_llc/dln.py` | DLN (widths `H_0..H_L`), whitened data, batched loss `½‖W − A*‖²_F` |
| `src/muon_llc/optim.py` | `SpectralOptimizer`: momentum, exact polar or Newton–Schulz(k), decoupled wd; plain GD |
| `src/muon_llc/train.py` | training loop; logs loss, product singular values `s_i(t)`, layer spectral norms, checkpoints |
| `src/muon_llc/theory/aoyagi.py` | Aoyagi's learning coefficient `λ(widths, r)`; CLI prints the staircase |
| `src/muon_llc/theory/dynamics.py` | A4 predictions: aligned GD ODE (staircase), spectral GD (single jump) |
| `src/muon_llc/theory/hessian.py` | exact Hessian, `½·soft-rank` prediction of the estimator (A1) |
| `src/muon_llc/theory/muon_scaling.py` | A5 laws: NS threshold `a^-k`, wd cap `λ_wd^-L`, momentum overshoot `~η` |
| `src/muon_llc/sampling/` | SGLD (devinterp rule), full-batch MALA on a β-ladder, WBIC and slope estimators |
| `src/muon_llc/llc.py` | per-checkpoint estimation driver, resolution sweep, surrogate estimate |
| `src/muon_llc/experiments/` | `train`, `estimate`, `static_dln` (B2), `sweep`, `analyze_c2`, `plot_c1`, `plot_c3`, `smoke` |
| `configs/` | `b2_static`, `c1_gd`, `c1_spectral`, `c2_momentum`, `c2_ns_steps`, `c2_weight_decay`, `c3_full_muon`, `smoke` |
| `scripts/run_all_gpu.sh` | ordered runbook for a single GPU |

## Setup and conventions

* **DLN.** Depth 3, widths (8, 8, 8, 8). Target `A* = U diag(4, 2, 1, 0.5, 0, 0, 0, 0) Vᵀ` with seeded
  random orthogonal `U, V`. Inputs `X ∈ R^{8×n}` (n = 1024) are Gaussian and exactly whitened, targets
  `Y = A* X` are noiseless, so the empirical loss `L_n = (1/2n) Σ‖y − Wx‖²` equals `½‖W − A*‖²_F`
  identically (unit-variance Gaussian NLL, so `nβ` is meaningful). Minibatch SGLD uses real subsets.
* **Init.** `W_l ~ N(0, σ²)`, `σ = 0.01` by default (`model.init_scale`); the same seed gives the same
  init for every optimizer.
* **Optimizers.** GD = `torch.optim.SGD`. Spectral GD = `SpectralOptimizer(momentum=0, orth="exact",
  weight_decay=0)`; the exact polar factor is `U Vᵀ` restricted to numerically nonzero singular values.
  Full Muon = `momentum=0.95, nesterov, orth="ns", ns_steps=5, weight_decay>0`, following Keller
  Jordan's `muon.py` (lerp momentum, Frobenius normalisation + 1e-7, transpose for tall matrices,
  `p *= 1 − lr·wd`). Two deliberate differences: Newton–Schulz runs in float32 (bf16 adds its own
  noise floor, which would confound the k-sweep) and the `sqrt(rows/cols)` factor is off (square layers).
* **LLC estimators** (column `estimator` in `llc.csv`):
  * `sgld` — devinterp rule `Δw = −(ε/2)[nβ∇L_batch + γ(w − w_t)] + √ε ξ`, chains vectorised.
  * `mala` — full-batch Metropolis-adjusted Langevin on a ladder of 8 log-spaced `nβ` around
    `n/log n` (devinterp's default), step size adapted to 57 % acceptance. Headline row = rung
    nearest `n/log n`; `mala_ladder` = the slope fit `λ̂(β) = λ − (m−1)/log nβ` giving λ, m̂ with
    bootstrap errors, plus the thermodynamic-integration route.
  * `hessian` — `½ Σ_h>0 nβh/(nβh + γ)`, the A1 prediction for a quadratic landscape, from the exact
    Hessian; also the count of negative eigenvalues (saddle diagnostic).
  * `surrogate` — MALA on the surrogate problem whose *true map is the checkpoint's own product*
    `W_t`. See the caveat below.
  * Diagnostics on every row: `var_over_mean` (≈1 in the scale-free regime), `mean2_over_var`
    (≤ λ_quad by Cauchy–Schwarz), acceptance, step size.
* **Resolution axis.** `γ/nβ` is treated as a measurement axis: `llc.gamma_sweep` runs every estimator at
  each γ (`resolution_sweep_checkpoints: all` by default) and the figures show `λ̂(t)` as a family over γ.

## The Aoyagi table (A3)

`uv run python -m muon_llc.theory.aoyagi --widths 8 8 8 8`

| rank r | 0 | 1 | 2 | 3 | 4 | … | 8 |
|---|---|---|---|---|---|---|---|
| λ | 21.5 | 24.0 | 26.0 | 28.0 | 29.5 | | 32 (= 192/2 − …, regular 8×8 map: 64/2) |

Steps are only 1.5–2.5, so the chain counts default to error bars well under 0.5 at a fixed
resolution. **Provenance caveat:** arxiv and the journal sites were unreachable when this was
written, so the formula in `theory/aoyagi.py` is reconstructed and verified against independent
special cases (depth 1 regular, depth 2 = Aoyagi–Watanabe reduced-rank regression including the
parity term, hand-computed scalar toys, continuity across the Σ boundary). Check it against Aoyagi
(2024) Theorem 1 / Furman & Lau (2024) before quoting the multiplicity `m`.

## Two things the smoke test taught us (read before interpreting λ̂(t))

1. **Basin escape at saddles.** A checkpoint on the rank-r plateau is a *saddle* of the true loss with
   a basin `½ s_{r+1}²` below it. At `nβ = n/log n ≈ 148` the tempered posterior escapes into it
   unless the localiser is strong: the direct estimate reads `≈ nβ · (loss drop) ≈ −1500` at
   early checkpoints. Escape is prevented when `γ/nβ ≳ s_{r+1}^{4/3}/3`, which is the same scale at
   which the *learned* directions (Hessian eigenvalues `~ s_r^{4/3}`) start to be resolved. No single γ
   works for every plateau; the family of curves over γ is the result. The `surrogate` estimator
   sidesteps the artefact by making `w_t` a global minimum of a surrogate loss; it measures the
   learning coefficient of the current function's fibre (Aoyagi's λ at the effective rank of `W_t`),
   which is the quantity the A4 staircase prediction is really about, but it is *not* the LLC of the
   actual loss at `w_t`. Both are reported.
2. **B2 must use rank-r targets.** The static check constructs, for each r, the problem whose true map
   is the top-r truncation `A_r` (same `U, V`, data), so the balanced point is a global minimum. With
   the full-rank target the same point is a saddle and shows the escape above.

## Experiments

* **B2** `static_dln`: λ̂ vs `γ/nβ` for r = 0..4 with the Aoyagi lines; picks the γ with the smallest mean
  deviation (`runs/b2_static/b2_summary.txt`, `b2_resolution.png`).
* **C1** `c1_gd.yaml`, `c1_spectral.yaml` → `train`, `estimate`, `plot_c1`. Figures in `runs/c1/`:
  `singular_values.png` (measured `s_i(t)` with the A4 aligned prediction dashed),
  `c1_overlay_<est>.png` (λ̂(t) at the headline γ, both optimizers, Aoyagi staircase, predicted step
  times, Hessian soft-rank), `c1_family_<est>.png` (λ̂(t) for every γ), `ladder_*.png`
  (λ̂ vs nβ per checkpoint; flat = scale-free), `resolution_*.png`.
* **C2** `sweep` + `analyze_c2` (singular-value observables by default; set `sweep.llc: true` to also
  estimate λ at a few checkpoints per grid point):
  * Newton–Schulz `k ∈ {1,2,3,5,10}` × `κ ∈ {2..128}`: lag ratio `[t½(s_r)/t½(s_1)] / (s_r/s_1)^{1/L}`
    (1 under uniform growth, ≫1 under saddle-to-saddle, ∞ if `s_r` never reaches half its target) and
    the uniformity `s_r/s_1` at `t½(s_r)`; breakdown `κ*(k)` where the lag exceeds 2; fit of
    `log κ*` vs k against `log a = 1.237`. Already visible at tiny scale: with k = 1 and κ = 8 the two
    smallest singular values do not move until the two largest saturate (the staircase is back),
    with k = 5 all four grow together.
  * Weight decay `λ_wd` × depth `L ∈ {2,3,4}` × `η`: final `s_i` vs `min(s_i, λ_wd^{−L})`; log-log slope
    vs `−L`; independence of `η`.
  * Momentum `μ ∈ {0,…,0.99}` × init scale × `η`: residual amplitude at saturation vs `η` (predicted
    slope 1) and uniformity vs μ.
* **C3** `c3_full_muon.yaml` → `plot_c3` overlays full Muon with the two C1 curves.

## Output formats

`runs/<name>/config.json`, `trajectory.csv` (`step, loss, sv_1..sv_8, norm_1..norm_L`),
`checkpoints.npz` (`steps`, `thetas` flat), `llc.csv` (one row per checkpoint × estimator × γ),
`ladder.csv` (per-rung WBIC values), figures. Sweeps: `runs/<sweep>/<point>/…` and `summary.csv`.
`--wandb` (`wandb.enabled: true`, `uv sync --extra wandb`) logs the training scalars live; off by default.

## GPU notes

`torch` from PyPI ships CUDA 12.x wheels on Linux, so `uv sync` on the vast.ai box is enough (check
`torch.cuda.is_available()`); `device: auto` picks CUDA. Training does not benefit much from the GPU;
the samplers do (chains × rungs are batched). `scripts/run_all_gpu.sh --quick` is a first pass with
fewer chains; increase `llc.mala.n_chains` / `n_draws` to shrink error bars.
