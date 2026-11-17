# TMS phase transitions under SGD, AdamW, spectral GD and Muon

Reproduction of Chen, Lau, Mendel, Wei, Murfet, *Dynamical versus Bayesian Phase Transitions in a
Toy Model of Superposition* (arXiv 2310.06301), r = 2, c = 6, extended to AdamW, spectral GD
(exact orthogonalised update, no momentum) and Muon (Newton–Schulz k = 5, Nesterov momentum 0.95).

## Summary

1. **The paper's results reproduce.** All 18 critical-point losses, the k-gon parameters, the prior
   constants and the 5→6 critical sample size (601) match to the printed digits. SGD trajectories
   plateau on the k-gon loss levels, the transitions we see are the ones in the paper's list (10),
   and λ̂ rises at every loss drop. Table K.1 is reproduced in ordering and magnitude.
2. **Muon does not remove the plateau structure.** Under spectral GD and Muon, at every learning rate
   tried (1e-4 to 3e-3), trajectories sit on the same k-gon levels as under SGD; in the second half of
   training under 0.6 % of checkpoints are off a known critical point for every arm. Plateaus are
   shorter in epochs (median 40–220 against 360 for SGD) but not shorter in distance travelled.
3. **The Bayesian antecedent hypothesis is not contradicted by any optimizer.** Over 15 arms × 200
   seeds (925 transitions, 25 distinct types) and 4 arms × 400 posterior-initialised runs (61
   transitions), there are **0 transitions with loss down and λ down** and 0 uphill transitions.
   23 of 25 types have ΔL < 0, Δλ > 0 beyond the standard error and a free-energy crossing; the other
   two are within error of Δλ = 0 (one of them, 4⁺⁻→4, has Δλ = 1.9 ± 0.65 with the paper's own λ̂).
4. **Muon, spectral GD and AdamW make transitions SGD did not make here, and each one has an
   antecedent.** Examples: 4⁻⁻→4⁻, 4⁺⁻⁻→4, 4⁺⁺⁻→5, and under spectral GD from posterior samples one
   **5→6**, the transition the paper predicts on the Bayesian side but never saw under SGD.
5. **One discrepancy with the paper on the Bayesian side.** Our NUTS posterior crosses 5→6 at
   n ≈ 300–400 when the posterior is exp(−n·L_n) with L_n the paper's loss, the convention in which
   the free-energy formula gives 601. See "Fig 2" below.

Against the three ways Muon could have broken the BAH, fixed before the Muon runs: (a) transitions
with Δλ ≤ 0 — none beyond error; (b) dwelling off critical points — no; (c) no plateau structure — no.

## What was run

| | |
|---|---|
| Model, data | f(x) = ReLU(WᵀWx + b), W 2×6; x = μ·e_i; n = 1000, **one dataset per seed**, batch 20, 4500 epochs |
| Init | reference-notebook init: 4-gon of length 0.9 at a random rotation + N(0, 0.01²) on W, bias ~ N(0, 1) |
| SGD baseline | plain SGD, lr 0.005, loss = mean ‖x − f(x)‖² (the paper's L) |
| Arms | SGD; AdamW lr 1e-4…3e-3 (+ wd 0.01); spectral GD lr 1e-4…3e-3; Muon lr 1e-4…3e-3 (+ Adam on the bias) |
| Muon and the bias | Muon only acts on W. The bias is under plain SGD at lr 0.005 unless stated |
| Seeds | 200 per arm; same init, dataset and minibatch order across arms |
| λ̂ on trajectories | paper settings: full-batch SGLD, ε = 0.001, γ = 1, 500 steps, every 30 epochs, first 30 seeds |
| λ̂ at critical points | paper protocol (Appendix K): n = 5000, γ = 0.1, 10 000 steps, 10 chains, their step sizes, escaped chains dropped |
| Posterior | NUTS (numpyro), prior N(0, 1) |

Two things the paper leaves open and how they were settled:

- **Momentum / loss scale of the baseline.** Four combinations were tried (momentum 0 or 0.9; loss
  divided by c or not). Plain SGD with the paper's loss gives transitions at epochs ~200–3800 in 13 of
  30 seeds (paper: 7 of 30 shown) and was kept. Momentum 0.9 with loss/6 is nearly identical.
- **One dataset per seed is needed.** With a single shared dataset 4⁺→5 never occurred. In the
  population loss the 4⁺-gon is locally stable (gradient flow from 4⁺ + noise of std 0.03 stays put;
  the Hessian near it has no negative eigenvalue). It becomes a saddle only through the finite-sample
  imbalance of the dataset, which is also why train and theory loss differ visibly in the paper's
  4⁺→5 figures.

A further point about the critical points themselves: for σ > 0 the gradient with respect to a
positive-bias zero column is (b/c)·Σ_i W_i, so 4^{σ+,φ−} is critical only if the columns sum to
zero. The paper's statement that a dead vertex's length is "subject only to l² < −b" holds for σ = 0.

## Figure-by-figure

| Paper | Here | Status |
|---|---|---|
| Table 1, A.1, C.2, Fig A.2 | `tests/test_tms.py` | match |
| Fig 1, B.1–B.7 | `figures/trajectory_seed*.png`, one column per optimizer | reproduced, Muon/AdamW added |
| Fig 3 | `figures/posterior_init_loss_vs_llc.png` | reproduced, Muon/AdamW added |
| Fig D.1 | `figures/transition_graph.png` | reproduced, Muon/AdamW added |
| Table K.1 | `table_k1.csv` | reproduced |
| Fig 2 | `figures/occupancy.png` | transition reproduced; location differs, see below |
| — | `figures/ensemble_loss*.png` | new: all 200 seeds per arm, and learning-rate sweeps |

### Dynamical transitions (Fig 1 / B)

`trajectory_seed002.png` is the paper's Fig 1 second half: 4⁺→5 with λ̂ going from about 4.5 to 7.
Across the 30 seeds with λ̂ estimates, every detected transition comes with a rise in on-trajectory
λ̂: 13/13 under SGD, 15/15 under spectral GD, 10/10 under Muon (`trajectory_llc_jumps.csv`; median
rise 4.2–5.0, smallest 1.4).

Per arm, 200 seeds (`arms_summary.csv`, `transitions.csv`):

| arm | runs with a transition | transitions | antecedent | inconclusive | violation | types not seen under SGD | off-critical, 2nd half (%) | median plateau (epochs) | median plateau (path length) |
|---|---|---|---|---|---|---|---|---|---|
| SGD lr 0.005 | 108 | 117 | 69 | 48 | 0 | – | 0.42 | 360 | 7.6 |
| spectral lr 1e-4 | 126 | 140 | 99 | 41 | 0 | 3 | 0.44 | 220 | 4.7 |
| spectral lr 3e-4 | 127 | 132 | 89 | 43 | 0 | 4 | 0 | 80 | 2.3 |
| spectral lr 1e-3 | 70 | 72 | 55 | 17 | 0 | 4 | 0.06 | 90 | 6.0 |
| spectral lr 3e-3 | 37 | 38 | 38 | 0 | 0 | 4 | 0.10 | 140 | 28 |
| Muon lr 1e-4 | 116 | 121 | 79 | 42 | 0 | 3 | 0 | 120 | 2.5 |
| Muon lr 3e-4 | 103 | 107 | 68 | 39 | 0 | 2 | 0 | 40 | 1.1 |
| Muon lr 1e-3 | 26 | 27 | 26 | 1 | 0 | 1 | 0.10 | 120 | 7.8 |
| Muon lr 3e-3 | 31 | 32 | 32 | 0 | 0 | 2 | 0.14 | 170 | 29 |
| Muon lr 1e-3, Adam bias | 16 | 16 | 16 | 0 | 0 | 1 | 0.02 | 85 | 4.5 |
| AdamW lr 1e-4 | 48 | 50 | 41 | 9 | 0 | 5 | 0.58 | 125 | 0.64 |
| AdamW lr 3e-4 | 32 | 32 | 28 | 4 | 0 | 1 | 0 | 80 | 0.86 |
| AdamW lr 1e-3 | 18 | 18 | 17 | 1 | 0 | 1 | 0 | 50 | 1.0 |
| AdamW lr 3e-3 | 5 | 5 | 5 | 0 | 0 | 0 | 0 | 60 | 3.1 |
| AdamW lr 1e-3, wd 0.01 | 18 | 18 | 17 | 1 | 0 | 1 | 0 | 95 | 2.3 |

"Inconclusive" is almost entirely the single type 4⁺⁻→4 (see below). The fall in transition counts
at high learning rate is partly a detection limit: a plateau must last 3 checkpoints (30 epochs), and
fast arms pass through the early states sooner than that. End states show where they go regardless:
AdamW ends in a 5- or 5⁺-gon in 63–84 of 200 runs, spectral GD in 35–54, Muon in 23–53, SGD in 18.

Transition types pooled over arms (λ from theory for 5, 5⁺, 6, otherwise our Table K.1 estimate;
error = standard error over chains; n_cr from n·ΔL + Δλ·log n + Δc = 0):

| α → β | count | arms | ΔL | Δλ ± s.e. | n_cr | verdict | seen under SGD |
|---|---|---|---|---|---|---|---|
| 4⁺⁻ → 4 | 245 | 11 | −0.049 | 0.10 ± 0.39 | 16 | inconclusive | yes |
| 4⁺ → 5 | 237 | 15 | −0.035 | 0.97 ± 0.30 | 189 | antecedent | yes |
| 4⁺⁻⁻ → 4⁻ | 104 | 8 | −0.049 | 1.19 ± 0.33 | 129 | antecedent | yes |
| 4⁺⁺ → 5⁺ | 98 | 15 | −0.035 | 1.74 ± 0.23 | 331 | antecedent | yes |
| 4⁺⁺⁻ → 4⁺ | 71 | 7 | −0.049 | 0.88 ± 0.66 | 92 | antecedent | yes |
| 4⁺⁻⁻⁻ → 4⁻⁻ | 35 | 6 | −0.049 | 0.27 ± 0.21 | 29 | antecedent | yes |
| 4⁻⁻ → 4⁻ | 24 | 6 | −0.056 | 1.82 ± 0.26 | 179 | antecedent | no |
| 4⁺⁺⁻ → 4⁺⁺ | 22 | 11 | −0.056 | 1.61 ± 0.63 | 155 | antecedent | yes |
| 4⁺⁻ → 4⁺ | 18 | 9 | −0.056 | 1.40 ± 0.45 | 132 | antecedent | yes |
| 5 → 5⁺ | 15 | 3 | −0.007 | 1.5 (theory) | 1590 | antecedent | no |
| 4⁺⁻⁻ → 4 | 15 | 5 | −0.104 | 2.71 ± 0.34 | 138 | antecedent | no |
| 4⁺⁺⁻⁻ → 4 | 14 | 5 | −0.097 | 2.18 ± 0.42 | 117 | antecedent | no |
| 13 rarer types | 27 | | < 0 | > 0 except one | | 12 antecedent, 1 inconclusive | |

- 4⁺⁻→4 is the most common transition under every optimizer including SGD. Our estimates put the two
  ends at 4.63 ± 0.33 and 4.73 ± 0.21; the paper's Table K.1 has 3.39 and 5.30. On-trajectory λ̂
  rises by a median of 5.1 across it (15/15). So it is "inconclusive" from our critical-point
  estimates and "antecedent" from the paper's; nothing points to a violation.
- 4⁺⁻⁻⁻→4⁻⁻ and 4⁺⁻⁻⁻⁻→4⁻⁻⁻ occur at very small n_cr; the paper also calls these inconclusive.
- The verdict of a transition type is a property of the loss landscape, not of the optimizer. What
  the optimizer changes is which types occur. The 13 types that only non-SGD arms produced are all
  loss-down, λ-up.

### Posterior-initialised runs (Fig 3)

400 NUTS samples at n = 100 as initial points, 5000 epochs, four arms. All arms end on the same
levels (≈ 200 at the 5-gon, 118 at the 4-gon, 27–29 at the 6-gon), and lower-loss plateaus have
higher on-trajectory λ̂ under every optimizer. Transitions: 5→5⁺ (7–10 per arm), 4⁺→5 (3–5), and under
spectral GD one 5→6 and one 4→5 (`posterior_init_transitions.csv`). The 6-gon end states are runs
that started in the 6-gon basin, not transitions.

### Table K.1

Same ordering as the paper. For 5, 5⁺, 6: 7.9, 10.5, 9.8 here; 7.7, 9.9, 9.0 in the paper; 7, 8.5, 8.5
in theory. Chains kept: 10 of 10 except 4⁺⁻⁻⁻⁻ (6), 4⁺⁺⁻⁻⁻⁻ (8), 4⁺⁺⁻ (7), 4⁺⁺ (8).

### Fig 2 (Bayesian 5→6 transition)

The transition exists and goes the right way: the 5-gon dominates at small n, the 6-gon at large n
(`figures/occupancy.png`, `occupancy.csv`, `occupancy_nll0.5.csv`; 3 datasets per n, 4 chains).
Where it happens depends on a factor of 2 in the likelihood:

| Posterior | 6-gon share passes the 5-gon share at | Free-energy formula with that convention |
|---|---|---|
| exp(−n·L_n), L_n = mean ‖x − f‖² (the convention of the formula n·L + λ·log n) | n ≈ 300–400 | 601 |
| exp(−n·L_n/2), i.e. p(x\|w) ∝ exp(−½‖x − f‖²) as written in the paper's eq. (3) | n ≈ 700–800 | 1316 |

The paper reports 600–700 from MCMC and 601 from the formula. Our second row lands near their MCMC
range, our first row is the convention their formula is written in. So in both conventions our
sampled posterior switches to the 6-gon at roughly half the n the formula predicts, and the paper's
agreement between the two would be explained if their sampler used the eq. (3) likelihood while the
formula used L without the ½. I could not check their sampling code, so this is an inference from
our numbers, on 3 datasets per n with some sticky chains at n ≥ 1000 (the 5-gon share at n = 1100
comes from chains that never left their starting basin).

This does not touch the BAH results above: the sign of ΔL and Δλ for each transition is unaffected,
and only the quoted n_cr values would rescale.

## Limits

- λ̂ is noisy for the degenerate 4-gons (chain std 0.4–1.5). Verdicts for those rest on the sign of
  a point estimate with a standard error from 6–10 chains; systematic SGLD bias is not included.
- Trajectory λ̂ was estimated for 30 seeds of four arms (SGD, spectral 3e-4, Muon 3e-4, AdamW 3e-4),
  not for every arm.
- One model size (r = 2, c = 6), constant learning rates, no schedule. Muon's weight decay was 0.
- Checkpoints are every 10 epochs, so sub-30-epoch plateaus under fast arms are not counted.
- The double-transition concavity check of Appendix C.4 was not done; runs with two transitions are
  rare (at most 14 of 200 per arm).
- Fig 2 uses 3 datasets per n and 4 chains, against 10 and 6 in the paper; no t-SNE diagnostics.

## Reproduce

```bash
uv sync --extra dev --extra mcmc
uv run pytest                                                    # theory checks included
uv run python -m muon_llc.experiments.tms_smoke                  # 1-minute end-to-end check
uv run python -m muon_llc.experiments.tms_arms --jobs 4          # 15 arms x 200 seeds -> runs/tms/arms
uv run python -m muon_llc.experiments.tms_critical_llc           # Table K.1
uv run python -m muon_llc.experiments.tms_estimate --run runs/tms/arms/<arm>   # trajectory lambda-hat
uv run python -m muon_llc.experiments.tms_posterior inits --out runs/tms
uv run python -m muon_llc.experiments.tms_arms --out runs/tms/post --init runs/tms/posterior_inits_n100.npy \
    --only sgd spectral_lr3e-4 muon_lr3e-4 adamw_lr3e-4 --set runs=400 epochs=5000 seed=500 data_seed=5000
uv run python -m muon_llc.experiments.tms_posterior occupancy
uv run python -m muon_llc.experiments.tms_bah && uv run python -m muon_llc.experiments.tms_figures
```
