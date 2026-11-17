"""Train R batched TMS runs under one optimizer and save checkpoints.

    python -m muon_llc.tms.experiments.train --name sgd --opt '{"name":"gd","lr":0.005}' \
        --runs 30 --epochs 4500 --out runs/tms/sgd

`--init` is `kgon` (reference-notebook init) or a path to an .npy of shape (R, 18).
`--device` moves the parameters, the data and the step loop; initial points, datasets and the
minibatch order are drawn on the CPU either way, so a seed gives the same run up to rounding.
The Newton-Schulz arms orthogonalise in float32 and are chaotic enough that this rounding grows
to visible differences between devices after about a thousand steps. Which device is faster
depends on the machine (`scripts/run_all.py --benchmark`): the tensors are tiny, so a step costs
per-operation overhead rather than arithmetic.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from ...config import get_device
from ..model import init_kgon, make_problem
from ..train import train_batched
from ..transitions import label_runs, seq_str, summarize


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True)
    ap.add_argument("--opt", required=True, help="JSON optimizer config")
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=4500)
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--c", type=int, default=6)
    ap.add_argument("--loss-scale", type=float, default=1.0)
    ap.add_argument("--init", default="kgon")
    ap.add_argument("--init-k", type=int, default=4)
    ap.add_argument("--init-length", type=float, default=0.9)
    ap.add_argument("--init-noise", type=float, default=0.01)
    ap.add_argument("--bias-std", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0, help="init seed base and minibatch order seed")
    ap.add_argument("--data-seed", type=int, default=1000, help="run j uses dataset seed data_seed + j")
    ap.add_argument("--shared-data", action="store_true", help="one dataset for all runs")
    ap.add_argument("--ckpt-every", type=int, default=10)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--device", default="auto", help="auto (cuda if available) | cpu | cuda")
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)
    device = get_device({"device": args.device})
    opt = json.loads(args.opt)
    problem = make_problem(args.n, args.c, seed=args.data_seed, loss_scale=args.loss_scale,
                           n_datasets=None if args.shared_data else args.runs, device=device)
    if args.init == "kgon":
        theta0 = init_kgon(args.runs, args.c, args.init_k, args.init_length, args.init_noise,
                           args.bias_std, args.seed)
    else:
        theta0 = torch.tensor(np.load(args.init)[: args.runs], dtype=torch.float64)
    t0 = time.time()
    run = train_batched(problem, theta0, opt, args.epochs, args.batch_size, args.seed, args.ckpt_every)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "run.npz", epochs=run.epochs, thetas=run.thetas.astype(np.float32),
                        path_len=run.path_len, steps_per_epoch=run.steps_per_epoch)
    cfg = vars(args) | {"opt": opt, "device": str(device), "seconds": time.time() - t0}
    (out / "config.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    labels, _ = label_runs(run.thetas)
    s = summarize(labels, run.path_len)
    print(f"[{args.name}] {time.time() - t0:.0f}s on {device}  runs with transition "
          f"{s['runs_with_transition']}/{s['n_runs']}  other {s['frac_other']:.2f} (late {s['frac_other_late']:.2f})")
    print("  transitions:", {f"{a}->{b}": n for (a, b), n in s["transitions"].most_common()})
    print("  finals:", dict(s["finals"].most_common()))
    for j, ps in enumerate(s["sequences"][:30]):
        if len(ps) > 1:
            print(f"   run {j}: {seq_str(ps)}  at epochs {[int(run.epochs[p.start]) for p in ps]}")


if __name__ == "__main__":
    main()
