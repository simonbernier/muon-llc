"""Run every optimizer arm of a JSON arms file (see configs/tms_arms.json) in parallel processes.
Same seeds, datasets and minibatch order for every arm.

    python -m muon_llc.experiments.tms_arms --config configs/tms_arms.json --out runs/tms/arms --jobs 4
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/tms_arms.json")
    ap.add_argument("--out", default="runs/tms/arms")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--set", nargs="*", default=[], help="override common keys: key=value")
    ap.add_argument("--init", default=None, help="npy of initial points instead of the k-gon init")
    args = ap.parse_args(argv)
    cfg = json.loads(Path(args.config).read_text())
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
        cmd = [sys.executable, "-m", "muon_llc.experiments.tms_train", "--name", name,
               "--opt", json.dumps(opt), "--out", str(d)]
        for k, v in common.items():
            cmd += [f"--{k.replace('_', '-')}", str(v)]
        if args.init:
            cmd += ["--init", args.init]
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "log.txt", "w") as f:
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
        return name, f"exit {rc}"

    items = [(k, v) for k, v in cfg["arms"].items() if args.only is None or k in args.only]
    with ThreadPoolExecutor(args.jobs) as ex:
        for name, status in ex.map(run, items):
            print(name, status, flush=True)


if __name__ == "__main__":
    main()
