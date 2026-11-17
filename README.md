# muon-llc: learning dynamics under Muon through singular learning theory

Muon replaces the gradient of each weight matrix by its orthogonal polar factor, so every
singular direction of the update moves at the same speed. Gradient descent does the opposite: it
learns the strongest directions first and sits on plateaus in between. Singular learning theory
(SLT) reads those plateaus as critical points of the loss whose degeneracy is measured by the
local learning coefficient (LLC, λ), and reads training as a sequence of transitions between them.

This repository asks what Muon does to that picture:

1. Does Muon change **which** critical points a trajectory visits, and in what order?
2. Does λ along a Muon trajectory still move the way SLT predicts (up when the loss goes down)?
3. Which ingredient of Muon (orthogonalisation, Newton–Schulz, momentum, weight decay) is
   responsible for what?

It answers them in small models where both the dynamics and λ are known exactly, so that an
estimate can be checked against theory before it is trusted.

## Testbeds

| | Deep linear networks (`dln`) | Toy model of superposition (`tms`) |
|---|---|---|
| Model | `W = W_3 W_2 W_1`, widths (8, 8, 8, 8), 192 parameters | `f(x) = ReLU(WᵀWx + b)`, W 2×6, 18 parameters |
| Why | GD dynamics are closed-form (one singular value at a time); λ is known for every rank (Aoyagi 2024) | critical points are known k-gons; reproduces Chen et al. 2023 on dynamical vs Bayesian transitions |
| Optimizers | GD, spectral GD (idealised Muon), Muon, and one-ingredient-at-a-time ablations | SGD, AdamW, spectral GD, Muon (Muon on W, AdamW on b) |
| SLT observables | λ̂(t) by SGLD, MALA β-ladder (λ and multiplicity), Hessian soft-rank, surrogate-target λ̂ | λ̂ on trajectories and at critical points (SGLD), NUTS posterior occupancy, free-energy crossings |
| Report | [`results/dln/REPORT.md`](results/dln/REPORT.md) | [`results/tms/REPORT.md`](results/tms/REPORT.md) |

## Findings

<!-- FINDINGS:BEGIN -->
Results are being regenerated on the unified code; this section is filled in from the reports
once the runs finish.
<!-- FINDINGS:END -->

## Quickstart

Everything runs through [uv](https://docs.astral.sh/uv/). `torch` is taken from the PyTorch
CUDA 13.0 index on Windows and Linux (PyPI's Windows wheel is CPU-only); change `cu130` in
`pyproject.toml` for another CUDA runtime. Without a GPU everything still runs on CPU.

```bash
uv sync --extra dev --extra mcmc
```

```bash
uv run pytest
```

```bash
uv run python -m muon_llc.dln.experiments.smoke
```

```bash
uv run python -m muon_llc.tms.experiments.smoke
```

The two smoke commands run a reduced version of each testbed end to end (about a minute each)
and write to `runs/dln/smoke/` and `runs/tms/smoke/`.

## Reproducing the results

```bash
uv run python scripts/run_all.py
```

runs every stage in order and resumes where it stopped if interrupted. Stages can be run one
at a time with `--stages`:

| Stage | What it does | Output |
|---|---|---|
| `dln` | B2 static calibration, C1 GD vs spectral GD, C2 ablations, C3 full Muon | `runs/dln/`, `results/dln/` |
| `tms-arms` | 16 optimizer arms × 200 seeds | `runs/tms/arms/` |
| `tms-llc` | λ̂ along the trajectories of the four headline arms | `runs/tms/arms/*/llc.npz` |
| `tms-k1` | λ̂ at the 18 critical points (Table K.1 of Chen et al.) | `results/tms/table_k1.csv` |
| `tms-posterior` | NUTS posterior: occupancy vs n, posterior-initialised runs | `results/tms/occupancy*.csv`, `runs/tms/post/` |
| `tms-analysis` | transition tables, Bayesian-antecedent verdicts, figures | `results/tms/` |

`--quick` runs all of it at reduced size into `runs/quick/` in a few minutes. Raw runs live in
`runs/` (not tracked); `results/` holds the tables, figures and reports that are.

Tiny models are bound by per-operation overhead, not arithmetic, so the GPU only pays where the
batch is large. Sampling (thousands of chains in one tensor) runs on the GPU; single-run
training loops and NUTS (JAX has no CUDA build on native Windows) run on the CPU. The runbook
picks the device per stage and records it with the timings in `runs/logs/timings.json`.

### On a rented Linux GPU

Nothing in the code is tied to one machine: the lockfile resolves for Windows, Linux and macOS,
the CUDA runtime comes with the wheels (the box only needs an NVIDIA driver), and paths, seeds
and outputs are the same everywhere. Random numbers are drawn on the CPU and moved to the
device, so a seed gives the same initial point, data and minibatch order on any hardware.
Rounding still differs between devices, and training amplifies it: SGD, AdamW and exact spectral
GD agree between CPU and GPU to 1e-13 over the lengths tested, but Muon's float32 Newton–Schulz
drifts (1e-9 after 50 steps, 4e-3 after 1500). Per-seed Muon trajectories are therefore
comparable only within one device; counts and distributions over seeds are comparable across.

```bash
uv sync --extra dev --extra mcmc-cuda
```

```bash
uv run python scripts/run_all.py --benchmark
```

```bash
uv run python scripts/run_all.py
```

`mcmc-cuda` adds JAX's CUDA build, which exists for Linux only, so NUTS runs on the GPU there.
`--benchmark` times a few steps of each stage on CPU and GPU and prints which device to use for
each; the result is saved to `runs/logs/benchmark.json` and used by later runs unless
`--train-device` or `--sample-device` says otherwise. The defaults that suit a desktop card
under Windows are not necessarily right for a larger one under Linux. Copy `results/` back when
it finishes (`runs/` too if the raw trajectories are wanted).

## Method

**Optimizers** (`muon_llc/optim.py`). `SpectralOptimizer` exposes every ingredient of Muon
separately: momentum (lerp, optional Nesterov), orthogonalisation by exact polar factor or by
k Newton–Schulz steps, decoupled weight decay. Spectral GD is the special case with no momentum,
exact polar factor and no decay; Muon is momentum 0.95 + Nesterov + Newton–Schulz k = 5,
following Keller Jordan's reference implementation. Two deliberate differences from it:
Newton–Schulz runs in float32 rather than bfloat16 (whose noise floor would confound the
k-sweep), and the `sqrt(rows/cols)` factor is off by default.

**Local learning coefficient** (`muon_llc/slt/`). All estimators sample the localised tempered
posterior `p(w) ∝ exp(−nβ L_n(w) − γ/2 ‖w − w₀‖²)` around a checkpoint `w₀` and use
`λ̂ = nβ (E[L_n(w)] − L_n(w₀))`, with `nβ = n / log n` unless stated.

- `sgld.py`: SGLD with the devinterp update rule. One call handles many centres, each with its
  own γ, nβ and step size, so a whole trajectory and resolution sweep is one batched run.
- `mala.py`: full-batch Metropolis-adjusted Langevin on a ladder of inverse temperatures, with
  adapted step sizes. No discretisation bias; the ladder gives λ and the multiplicity from the
  slope of λ̂(β) against 1/log nβ.
- `estimators.py`: the WBIC form above, the ladder fit, thermodynamic integration, and the
  diagnostics reported with every estimate (variance/mean of nβΔL, acceptance, chain spread).
- `problem.py`: the small interface a testbed implements to use the samplers.

**Reading λ̂ at a saddle.** A checkpoint on a plateau is a saddle of the loss. A weakly localised
chain falls into the basin below and reports a large negative number; a strongly localised one
cannot resolve the directions it should count. No single γ works for every plateau, so γ/nβ is
treated as a measurement axis and λ̂(t) is reported as a family of curves over it. The DLN report
discusses this at length; the TMS runs use the settings of Chen et al. and their chain-health test.

## Repository layout

```
src/muon_llc/
  optim.py          Muon family, SGD, AdamW behind one build_optimizer(cfg)
  slt/              samplers and LLC estimators shared by every testbed
  config.py         YAML configs with `extends` and --set overrides, device, run directories
  style.py          one figure style and optimizer palette
  dln/              model, train, llc, io, plots, theory/ (Aoyagi λ, GD and spectral dynamics,
                    Hessian soft-rank, Muon scaling laws), experiments/
  tms/              model, potential, critical_points, train, llc, posterior, transitions,
                    analysis, plots, experiments/
configs/dln/        one YAML per experiment, all extending base.yaml
configs/tms/        arms.json: the optimizer arms
scripts/run_all.py  the runbook
results/<testbed>/  REPORT.md, tables, figures (tracked)
runs/<testbed>/     raw runs (not tracked)
tests/              theory checks, optimizer semantics, sampler checks on problems with known λ
```

## Extending

**A new testbed** is a subpackage with the same shape as `dln/` and `tms/`:

1. `model.py` with a problem class holding the data and a flat parameterisation, implementing
   `slt.problem.Problem` (`n`, `dim`, `device`, `dtype`, batched `loss`, `loss_and_grad`,
   `minibatch_indices`). Losses must broadcast over leading dimensions so chains stay
   independent; that is all the samplers need.
2. `train.py` using `optim.build_optimizer`, so every optimizer arm is a config entry.
3. `experiments/` drivers, configs under `configs/<testbed>/`, outputs under `runs/<testbed>/`,
   tracked summaries and a `REPORT.md` under `results/<testbed>/`, a stage in `scripts/run_all.py`.

For models too large for full-batch losses, `run_sgld` already takes a minibatch size and records
the minibatch loss alongside; the full-batch loss it also records is the part to replace by a
held-out estimate.

**A new observable** goes in `slt/` if it only needs the `Problem` interface (a restricted or
weight-refined LLC is `run_sgld` on a problem that exposes a subset of the parameters), and in
the testbed otherwise. **A new optimizer** is a branch in `optim.build_optimizer`.

## References

- M. Aoyagi (2024). Consideration on the learning efficiency of multiple-layered neural networks
  with linear units. *Neural Networks* 172.
- Z. Chen, E. Lau, J. Mendel, S. Wei, D. Murfet (2023). Dynamical versus Bayesian phase
  transitions in a toy model of superposition. arXiv 2310.06301.
- E. Lau, Z. Furman, G. Wang, D. Murfet, S. Wei (2025). The local learning coefficient: a
  singularity-aware complexity measure. AISTATS. arXiv 2308.12108.
- K. Jordan et al. (2024). Muon: an optimizer for hidden layers in neural networks.
- S. Watanabe (2009). *Algebraic Geometry and Statistical Learning Theory*. Cambridge.
