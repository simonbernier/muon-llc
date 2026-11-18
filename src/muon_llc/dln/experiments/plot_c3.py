"""C3: λ̂(t) for every trained optimizer on one figure (GD, AdamW, spectral GD, full Muon), plus
singular values.

    uv run python -m muon_llc.dln.experiments.plot_c3 \
        --runs runs/dln/c1_gd runs/dln/c1_adamw runs/dln/c1_spectral runs/dln/c3_full_muon --out runs/dln/c3
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .plot_c1 import make_c1_figures


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--estimator", default=None)
    a = p.parse_args()
    make_c1_figures([Path(r) for r in a.runs], Path(a.out), a.estimator)


if __name__ == "__main__":
    main()
