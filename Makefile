.PHONY: sync test lint smoke table b2 c1 c2 c3 all

sync:
	uv sync --extra dev

test:
	uv run pytest

lint:
	uv run ruff check src tests

table:
	uv run python -m muon_llc.theory.aoyagi --widths 8 8 8 8

smoke:
	uv run python -m muon_llc.experiments.smoke

b2:
	uv run python -m muon_llc.experiments.static_dln --config configs/b2_static.yaml

c1:
	uv run python -m muon_llc.experiments.train --config configs/c1_gd.yaml
	uv run python -m muon_llc.experiments.train --config configs/c1_spectral.yaml
	uv run python -m muon_llc.experiments.estimate --run runs/c1_gd
	uv run python -m muon_llc.experiments.estimate --run runs/c1_spectral
	uv run python -m muon_llc.experiments.plot_c1 --runs runs/c1_gd runs/c1_spectral --out runs/c1

c2:
	uv run python -m muon_llc.experiments.sweep --config configs/c2_ns_steps.yaml
	uv run python -m muon_llc.experiments.analyze_c2 --sweep runs/c2_ns_steps --kind ns
	uv run python -m muon_llc.experiments.sweep --config configs/c2_weight_decay.yaml
	uv run python -m muon_llc.experiments.analyze_c2 --sweep runs/c2_weight_decay --kind wd
	uv run python -m muon_llc.experiments.sweep --config configs/c2_momentum.yaml
	uv run python -m muon_llc.experiments.analyze_c2 --sweep runs/c2_momentum --kind momentum

c3:
	uv run python -m muon_llc.experiments.train --config configs/c3_full_muon.yaml
	uv run python -m muon_llc.experiments.estimate --run runs/c3_full_muon
	uv run python -m muon_llc.experiments.plot_c3 --runs runs/c1_gd runs/c1_spectral runs/c3_full_muon --out runs/c3

all: b2 c1 c2 c3
