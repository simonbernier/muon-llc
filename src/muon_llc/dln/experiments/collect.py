"""Copy the tables and figures of the DLN experiments from runs/dln/ into results/dln/.

    uv run python -m muon_llc.dln.experiments.collect [--runs-root runs/dln] [--out results/dln]

runs/ is scratch space (ignored by git); results/ is what gets committed. The layout mirrors the
run directories, one directory per experiment, holding only the small files written directly in
it (tables, summaries, resolved configs, figures):

    results/dln/
        b2_static/         b2.csv, b2_ladder.csv, b2_summary.txt, b2_resolution.png, config.json
        b2_long/           the same with long chains
        b2_gibbs/          the same with the Gibbs sampler
        c1_gd/             config.json, trajectory.csv, llc.csv, ladder.csv      (one per trained run:
        c1_adamw/            c1_gd, c1_adamw, c1_spectral, c3_full_muon)
        c1_spectral/
        c3_full_muon/
        c1_adamw_lr/       summary.csv (AdamW learning-rate scan, training only)
        c1/                C1 figures (plot_c1): singular_values.png, c1_overlay_*.png, c1_family_*.png,
                           ladder_*.png, resolution_*.png
        c3/                the same figures for all four optimizers (plot_c3)
        c2_ns_steps/       summary.csv, ns_breakdown.csv, ns_summary.txt, ns_*.png
        c2_weight_decay/   summary.csv, wd_summary.txt, wd_cap*.png
        c2_momentum/       summary.csv, momentum_summary.txt, momentum_*.png

Not copied: parameter checkpoints (`checkpoints.npz`, any other binary) and subdirectories, which
for a sweep are its per-grid-point run directories (their observables are the rows of
`summary.csv`). The smoke-test directory is skipped. Files already in the destination are
overwritten; nothing is deleted.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

KEEP_SUFFIXES = {".csv", ".txt", ".json", ".png", ".pdf", ".svg"}
SKIP_DIRS = {"smoke"}
EXPECTED = ["b2_static", "b2_long", "b2_gibbs", "b3_offset", "c1_gd", "c1_adamw", "c1_spectral", "c1",
            "c1_adamw_lr", "c2_ns_steps", "c2_weight_decay", "c2_momentum", "c3_full_muon", "c3"]


def collect(runs_root: Path, out: Path, skip=SKIP_DIRS, verbose: bool = True) -> dict[str, list[str]]:
    """Returns {experiment: [file names copied]} for every experiment directory that had any."""
    runs_root, out = Path(runs_root), Path(out)
    if not runs_root.is_dir():
        raise FileNotFoundError(f"no run directory {runs_root}")
    copied: dict[str, list[str]] = {}
    for d in sorted(p for p in runs_root.iterdir() if p.is_dir() and p.name not in skip):
        files = sorted(f for f in d.iterdir() if f.is_file() and f.suffix.lower() in KEEP_SUFFIXES)
        if not files:
            continue
        dest = out / d.name
        dest.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy2(f, dest / f.name)
        copied[d.name] = [f.name for f in files]
    if verbose:
        for name, files in copied.items():
            size = sum((out / name / f).stat().st_size for f in files) / 1024
            print(f"{name:18s} {len(files):3d} files {size:8.0f} kB   {' '.join(files[:6])}"
                  f"{' ...' if len(files) > 6 else ''}")
        missing = [e for e in EXPECTED if e not in copied]
        if missing:
            print(f"not found under {runs_root} (not run yet?): {' '.join(missing)}")
        print(f"{sum(len(v) for v in copied.values())} files -> {out}")
    return copied


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", default="runs/dln")
    ap.add_argument("--out", default="results/dln")
    ap.add_argument("--skip", nargs="*", default=sorted(SKIP_DIRS), help="experiment directories to leave out")
    args = ap.parse_args(argv)
    collect(Path(args.runs_root), Path(args.out), set(args.skip))


if __name__ == "__main__":
    main()
