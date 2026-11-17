"""lambda-hat along training trajectories with the paper's settings (full-batch SGLD,
eps = 0.001, gamma = 1, 500 steps, every 30 epochs). Writes llc.npz into the run directory."""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from ...config import get_device
from ..analysis import load_run, problem_of
from ..llc import sgld_llc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--n-runs", type=int, default=30)
    ap.add_argument("--stride", type=int, default=3, help="checkpoint stride (3 x 10 epochs = 30 epochs)")
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--burnin", type=int, default=0)
    ap.add_argument("--chains", type=int, default=2)
    ap.add_argument("--chunk", type=int, default=600)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--tag", default="llc")
    ap.add_argument("--device", default="auto", help="auto (cuda if available) | cpu | cuda")
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)
    device = get_device({"device": args.device})
    run = load_run(args.run)
    ck = np.arange(0, len(run["epochs"]), args.stride)
    rr = np.arange(min(args.n_runs, run["thetas"].shape[1]))
    centres = run["thetas"][ck][:, rr]  # (k, r, D)
    k, r, D = centres.shape
    di = np.tile(rr, k)
    prob = problem_of(run["cfg"], device=device)
    t0 = time.time()
    res = sgld_llc(prob, centres.reshape(k * r, D), eps=args.eps, gamma=args.gamma, n_steps=args.steps,
                   n_burnin=args.burnin, n_chains=args.chains, chunk=args.chunk,
                   data_index=di if prob.per_run else None, seed=0)
    np.savez_compressed(run["dir"] / f"{args.tag}.npz", ck_idx=ck, run_idx=rr,
                        llc=res.llc.reshape(k, r), llc_chain=res.llc_chain.reshape(k, r, -1),
                        frac_below=res.frac_below.reshape(k, r, -1), nbeta=res.nbeta,
                        eps=args.eps, gamma=args.gamma, steps=args.steps)
    print(f"[{run['name']}] {k}x{r} centres in {time.time() - t0:.0f}s on {device}; "
          f"median final llc {np.median(res.llc.reshape(k, r)[-1]):.2f}")


if __name__ == "__main__":
    main()
