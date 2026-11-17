"""Run every optimizer arm of a JSON arms file (see configs/tms/arms.json) in parallel processes.
Same seeds, datasets and minibatch order for every arm.

    python -m muon_llc.tms.experiments.arms --config configs/tms/arms.json --out runs/tms/arms --jobs 8

Each arm is one single-threaded process. `--jobs` defaults to two thirds of the logical cores
(`default_jobs`); `scripts/run_all.py --benchmark` says which device is faster on a given machine.
Exits non-zero if an arm fails.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def default_jobs() -> int:
    """Two thirds of the logical cores: on a 6-core / 12-thread CPU 8 jobs finished the 16 arms
    as fast as 12 and about 15 % faster than 6, and it leaves room for whatever else is running."""
    return max(1, (os.cpu_count() or 2) * 2 // 3)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/tms/arms.json")
    ap.add_argument("--out", default="runs/tms/arms")
    ap.add_argument("--jobs", type=int, default=default_jobs(), help="arms trained at the same time")
    ap.add_argument("--device", default="auto", help="auto (cuda if available) | cpu | cuda")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--set", nargs="*", default=[], help="override common keys: key=value")
    ap.add_argument("--init", default=None, help="npy of initial points instead of the k-gon init")
    args = ap.parse_args(argv)
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    common = cfg["common"]
    for kv in args.set:
        k, v = kv.split("=", 1)
        common[k] = json.loads(v)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def run(item):
        name, opt = item
        d = out / name
        if (d / "run.npz").exists():
            return name, "cached"
        cmd = [sys.executable, "-m", "muon_llc.tms.experiments.train", "--name", name,
               "--opt", json.dumps(opt), "--out", str(d), "--device", args.device]
        for k, v in common.items():
            cmd += [f"--{k.replace('_', '-')}", str(v)]
        if args.init:
            cmd += ["--init", args.init]
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "log.txt", "w", encoding="utf-8") as f:
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
        return name, f"exit {rc}"

    items = [(k, v) for k, v in cfg["arms"].items() if args.only is None or k in args.only]
    failed = []
    with ThreadPoolExecutor(args.jobs) as ex:
        for name, status in ex.map(run, items):
            print(name, status, flush=True)
            if status not in ("cached", "exit 0"):
                failed.append(name)
    if failed:
        sys.exit(f"arms failed (see log.txt in each directory): {' '.join(failed)}")


if __name__ == "__main__":
    main()
