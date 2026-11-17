# Shortcuts for the commands in README.md. Everything runs through uv; on Windows without make,
# use the commands directly.
.PHONY: sync test lint smoke table quick all dln tms

sync:
	uv sync --extra dev --extra mcmc

test:
	uv run pytest

lint:
	uv run ruff check src tests scripts

table:
	uv run python -m muon_llc.dln.theory.aoyagi --widths 8 8 8 8

smoke:
	uv run python -m muon_llc.dln.experiments.smoke
	uv run python -m muon_llc.tms.experiments.smoke

# every stage at reduced size, outputs under runs/quick/
quick:
	uv run python scripts/run_all.py --quick

dln:
	uv run python scripts/run_all.py --stages dln

tms:
	uv run python scripts/run_all.py --stages tms-arms tms-llc tms-k1 tms-posterior tms-analysis

all:
	uv run python scripts/run_all.py
