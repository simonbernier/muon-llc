"""Table K.1: lambda-hat at each known critical point with the paper's protocol.

n = 5000, gamma = 0.1, 10 000 SGLD steps, 10 chains, a step size per loss group; chains with more
than 5 % of draws below the critical point's loss are discarded. Writes table_k1.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ...config import get_device
from ..critical_points import PAPER_LLC_HAT, THEORY_LLC, critical_points
from ..llc import sgld_llc
from ..model import make_problem

# SGLD step sizes of Table K.1, by number of dead vertices phi (k >= 5 separately)
PAPER_EPS = {4: 5e-7, 3: 1e-6, 2: 1e-6, 1: 5e-6, 0: 1e-5, "k5+": 5e-5}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results/tms")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--gamma", type=float, default=0.1)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--chains", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", default="auto", help="auto (cuda if available) | cpu | cuda")
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)
    problem = make_problem(args.n, seed=args.seed + 17, device=get_device({"device": args.device}))
    rows = []
    cps = critical_points()
    groups: dict[float, list] = {}
    for cp in cps:
        eps = PAPER_EPS["k5+"] if cp.k >= 5 else PAPER_EPS[cp.phi]
        groups.setdefault(eps, []).append(cp)
    for eps, members in groups.items():
        res = sgld_llc(problem, np.stack([m.theta for m in members]), eps=eps, gamma=args.gamma,
                       n_steps=args.steps, n_chains=args.chains, seed=args.seed, below_tol=0.0)
        mean, std, cnt = res.healthy(0.05)
        for i, m in enumerate(members):
            p_hat, p_std = PAPER_LLC_HAT[m.name]
            rows.append(dict(name=m.name, loss=m.loss, llc_hat=mean[i], llc_std=std[i],
                             chains_kept=int(cnt[i]), llc_all_chains=res.llc[i], sgld_eps=eps,
                             paper_llc_hat=p_hat, paper_std=p_std,
                             theory_llc=THEORY_LLC.get(m.name, np.nan)))
            print(rows[-1], flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).sort_values("loss", ascending=False)
    df.to_csv(out / "table_k1.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
