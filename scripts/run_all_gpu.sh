#!/usr/bin/env bash
# Phase C runbook for a single GPU box (e.g. vast.ai). Usage:
#   bash scripts/run_all_gpu.sh            # full run
#   bash scripts/run_all_gpu.sh --quick    # fewer chains / draws / grid points for a first pass
#
# Rough wall-clock on one consumer GPU (the models are tiny; the cost is chains x betas x
# checkpoints x gammas): B2 ~2 min, C1 training ~2 min, C1 estimation ~10-20 min (7 gammas x
# 3 estimators x ~45 checkpoints x 2 runs), C2 sweeps ~1-2 h (training only; add llc: true in
# the sweep configs to also estimate lambda at a few checkpoints per point), C3 ~5 min.
# All outputs land in runs/; rsync that directory off the box.
set -euo pipefail
cd "$(dirname "$0")/.."

QUICK=""
if [[ "${1:-}" == "--quick" ]]; then
  QUICK="--set llc.sgld.n_chains=8 llc.sgld.n_draws=500 llc.mala.n_chains=8 llc.mala.n_draws=500 llc.checkpoints={max:16}"
fi

log() { printf '\n[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

log "environment"
uv sync
uv run python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
uv run pytest -q

log "A3 table"
uv run python -m muon_llc.theory.aoyagi --widths 8 8 8 8

log "B2 static calibration"
uv run python -m muon_llc.experiments.static_dln --config configs/b2_static.yaml
cat runs/b2_static/b2_summary.txt

log "C1 training: GD and spectral GD"
uv run python -m muon_llc.experiments.train --config configs/c1_gd.yaml $QUICK
uv run python -m muon_llc.experiments.train --config configs/c1_spectral.yaml $QUICK

log "C1 estimation (sgld + mala ladder + surrogate, resolution sweep at every checkpoint)"
uv run python -m muon_llc.experiments.estimate --run runs/c1_gd
uv run python -m muon_llc.experiments.estimate --run runs/c1_spectral
uv run python -m muon_llc.experiments.plot_c1 --runs runs/c1_gd runs/c1_spectral --out runs/c1

log "C2 ablations"
uv run python -m muon_llc.experiments.sweep --config configs/c2_ns_steps.yaml
uv run python -m muon_llc.experiments.analyze_c2 --sweep runs/c2_ns_steps --kind ns
uv run python -m muon_llc.experiments.sweep --config configs/c2_weight_decay.yaml
uv run python -m muon_llc.experiments.analyze_c2 --sweep runs/c2_weight_decay --kind wd
uv run python -m muon_llc.experiments.sweep --config configs/c2_momentum.yaml
uv run python -m muon_llc.experiments.analyze_c2 --sweep runs/c2_momentum --kind momentum

log "C3 full Muon"
uv run python -m muon_llc.experiments.train --config configs/c3_full_muon.yaml $QUICK
uv run python -m muon_llc.experiments.estimate --run runs/c3_full_muon
uv run python -m muon_llc.experiments.plot_c3 --runs runs/c1_gd runs/c1_spectral runs/c3_full_muon --out runs/c3

log "done: see runs/c1, runs/c2_*, runs/c3, runs/b2_static"
