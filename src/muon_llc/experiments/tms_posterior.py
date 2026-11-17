"""Posterior experiments (optimizer-independent).

  occupancy : Fig. 2 (right). For each n and seed, NUTS-sample the posterior and record the
              share of samples in each region W_{k,sigma}.  -> occupancy.csv
  inits     : Fig. 3. Posterior samples at n = 100 to be used as initial points. -> inits.npy
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from ..tms.model import make_problem
from ..tms.posterior import classify_hull, sample_posterior

PHASES = ["4", "4+", "5", "5+", "6", "other"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["occupancy", "inits"])
    ap.add_argument("--out", default="results/tms")
    ap.add_argument("--ns", type=int, nargs="*", default=list(range(100, 1201, 100)))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--chains", type=int, default=6)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--samples", type=int, default=2000)
    ap.add_argument("--thin", type=int, default=10)
    ap.add_argument("--nll-scale", type=float, default=1.0)
    ap.add_argument("--n-inits", type=int, default=400)
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.mode == "inits":
        p = make_problem(100, seed=4242)
        s = sample_posterior(p.feat.numpy(), p.mu.numpy(), n_chains=args.chains, n_warmup=args.warmup,
                             n_samples=5000, thin=10, seed=1, nll_scale=args.nll_scale)
        flat = s.reshape(-1, s.shape[-1])
        rng = np.random.default_rng(0)
        pick = flat[rng.choice(len(flat), size=args.n_inits, replace=False)]
        np.save(out / "posterior_inits_n100.npy", pick)
        print("saved", pick.shape, Counter(classify_hull(pick)).most_common())
        return
    path = out / f"occupancy{args.tag}.csv"
    rows = pd.read_csv(path).to_dict("records") if path.exists() else []
    done = {(r["n"], r["seed"]) for r in rows}
    for n in args.ns:
        for sd in range(args.seed0, args.seed0 + args.seeds):
            if (n, sd) in done:
                continue
            t0 = time.time()
            p = make_problem(n, seed=10_000 + 97 * sd + n)
            s = sample_posterior(p.feat.numpy(), p.mu.numpy(), n_chains=args.chains, n_warmup=args.warmup,
                                 n_samples=args.samples, thin=args.thin, seed=sd, nll_scale=args.nll_scale)
            cnt = Counter(classify_hull(s))
            tot = sum(cnt.values())
            per_chain = [Counter(classify_hull(s[k])) for k in range(s.shape[0])]
            rows.append(dict(n=n, seed=sd, nll_scale=args.nll_scale, seconds=time.time() - t0,
                             **{ph: cnt.get(ph, 0) / tot for ph in PHASES},
                             chains_all_6=sum(pc.get("6", 0) == sum(pc.values()) for pc in per_chain)))
            pd.DataFrame(rows).to_csv(path, index=False)
            print(rows[-1], flush=True)


if __name__ == "__main__":
    main()
