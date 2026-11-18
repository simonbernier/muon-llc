"""Runbook: regenerate every result of the repository on one machine (Windows or Linux).

    uv run python scripts/run_all.py                      # everything, in order
    uv run python scripts/run_all.py --stages tms-arms tms-llc
    uv run python scripts/run_all.py --quick              # a few minutes, outputs under runs/quick/
    uv run python scripts/run_all.py --force --stages tms-k1
    uv run python scripts/run_all.py --benchmark          # time each kind of work on cpu and cuda
    uv run python scripts/run_all.py --dry-run            # print the commands, run nothing

Stages (run in this order whatever the order on the command line):

    dln            B2 static calibration (short and long chains); B3, the same points sampled by
                   long MALA chains from two kinds of start and by a Gibbs sampler; the four trained runs (C1 GD,
                   AdamW and spectral GD, C3 full Muon: train, then estimate); the three C2 sweeps
                   and their analyses, the AdamW learning-rate scan; figures; collect the tables
                   and figures into results/dln/
    tms-arms       every optimizer arm of configs/tms/arms.json            -> runs/tms/arms/
    tms-llc        lambda-hat along the trajectories of the headline arms  -> runs/tms/arms/<arm>/llc.npz
    tms-k1         Table K.1 (lambda-hat at the known critical points)     -> results/tms/table_k1.csv
    tms-posterior  NUTS posterior samples as initial points, the posterior-initialised arms and
                   their lambda-hat (runs/tms/post/), occupancy for both likelihood conventions
                   (results/tms/occupancy*.csv)
    tms-analysis   transition / Bayesian-antecedent tables for the arms and for the
                   posterior-initialised arms, trajectory_llc_jumps.csv, all figures -> results/tms/

How it runs. A stage is a few chains of steps; chains run side by side, the steps of a chain one
after the other. Every step is `python -m <module> ...` in its own process. Output is copied to
the console and to runs/logs/<stage>.log; wall-clock and device of every stage and step are
appended to runs/logs/timings.json.

Resuming. A step that finished leaves a stamp in runs/logs/done/. It is skipped when its stamp
and all of its outputs exist, so an interrupted run picks up where it stopped (inside a step,
`tms.experiments.arms` also keeps the arms that are already trained and the occupancy scan keeps
its finished rows). The stamp is required because results/ is committed: output files that came
with the checkout do not count as done. `--force` deletes the outputs of the selected stages and
runs them again.

Devices. There are two kinds of work: training (`--train-device`) and SGLD / MALA sampling
(`--sample-device`). Each flag takes cpu, cuda or auto (cuda when available). Left out, the
device comes from the last `--benchmark` on this machine (runs/logs/benchmark.json) and, failing
that, from the fallback: train on cpu, sample on cuda when available. The fallback is what was
measured on the first machine (RTX 4070 SUPER, 6 cores / 12 threads, Windows 11): a training
step acts on a few thousand numbers and costs per-operation overhead, 1.8-2.4 ms on the CPU
against 2.2-4.2 ms on the GPU for TMS and 0.3-1.1 ms against 1.2-6.2 ms for the DLN, while the
samplers batch thousands of chains and are 7-19 times faster on the GPU. Per-operation GPU
overhead is specific to the driver model (Windows WDDM here; usually lower on Linux) and the
number of cores per GPU shifts the balance too, so run `--benchmark` on a new machine. The
devices used and where the choice came from are recorded in timings.json.

`--jobs` is the number of TMS arms trained at the same time; the default is two thirds of the
logical cores (8 on the first machine, where 8 and 12 jobs were equally fast and 6 was 15 % slower).

NUTS (numpyro / JAX) uses whatever backend JAX finds: the CPU on native Windows, where JAX has no
CUDA build, a GPU on Linux if a CUDA jaxlib is installed. `--nuts-platform cpu|gpu` forces one.
The backend each NUTS step used is in its log line and in timings.json.

`--quick` shrinks every stage (fewer seeds, epochs, chains, grid points) so that the whole
pipeline runs in a few minutes, and redirects everything, including what would go to results/,
to runs/quick/. Its numbers are only good for checking that the pipeline runs.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGES = ["dln", "tms-arms", "tms-llc", "tms-k1", "tms-posterior", "tms-analysis"]
HEADLINE = ["sgd", "spectral_lr3e-4", "muon_lr3e-4", "adamw_lr3e-4"]
ARMS_CONFIG = Path("configs/tms/arms.json")
DLN_CONFIGS = Path("configs/dln")
DLN_RUNS = ["c1_gd", "c1_adamw", "c1_spectral", "c3_full_muon"]  # trained, then estimated; plot order
# sweep -> kind of analysis (analyze_c2), None for a sweep whose summary.csv is the result
DLN_SWEEPS = {"c2_ns_steps": "ns", "c2_weight_decay": "wd", "c2_momentum": "momentum", "c1_adamw_lr": None}
DLN_B3_ARMS = ["A", "B", "C", "C2", "D", "D2"]  # configs/dln/b3_offset.yaml, one process per arm
BENCHMARK_FILE = Path("runs/logs/benchmark.json")  # machine-local (runs/ is not committed)
FALLBACK_DEVICE = {"train": "cpu", "sample": "auto"}  # used when no flag and no benchmark; see docstring
DLN = "muon_llc.dln.experiments."
TMS = "muon_llc.tms.experiments."

# Fig. 2 as committed: 3 datasets per n, 4 chains, 150 kept draws per chain (1500 draws thinned by 10);
# the scan with the eq. (3) likelihood (nll_scale 0.5) was run on a coarser grid of n.
OCCUPANCY = {
    "": ["--seeds", "3", "--chains", "4", "--samples", "1500"],
    "_nll0.5": ["--seeds", "3", "--chains", "4", "--samples", "1500", "--nll-scale", "0.5",
                "--ns", "200", "400", "600", "700", "800", "1000", "1200"],
}
# Posterior-initialised arms (REPORT, "Reproduce") and lambda-hat along them: every run, every 100 epochs.
POST_SET = ["runs=400", "epochs=5000", "seed=500", "data_seed=5000"]
POST_LLC = ["--n-runs", "400", "--stride", "10"]

QUICK_DLN_LLC = ["llc.nbeta_ladder.n=5", "llc.gamma_sweep=[1.0, 10.0, 100.0, 1000.0]", "llc.checkpoints={max: 8}",
                 "llc.sgld.n_chains=6", "llc.sgld.n_burnin=100", "llc.sgld.n_draws=400",
                 "llc.mala.n_chains=6", "llc.mala.n_burnin=200", "llc.mala.n_draws=400", "llc.mala.n_boot=100",
                 "llc.gibbs.n_chains=6", "llc.gibbs.n_burnin=50", "llc.gibbs.n_draws=150", "llc.gibbs.n_boot=100"]
QUICK_DLN_B2 = ["static.ranks=[0, 2, 4]", "static.gammas=[1.0, 10.0, 100.0]", "static.nbeta_multipliers=[1.0]",
                "llc.nbeta_ladder.n=5", "llc.mala.n_chains=6", "llc.mala.n_burnin=200", "llc.mala.n_draws=400",
                "llc.mala.n_boot=100", "llc.gibbs.n_chains=6", "llc.gibbs.n_burnin=50", "llc.gibbs.n_draws=150",
                "llc.gibbs.n_boot=100"]
QUICK_DLN_B3 = ["static.ranks=[0, 2]", "static.nbeta_multipliers=[1.0]", "offset.n_chains=6",
                *[f"offset.arms.{a}.{k}={v}" for a in DLN_B3_ARMS
                  for k, v in (("n_burnin", 200), ("n_draws", 400), ("thin", 1))]]
QUICK_DLN_STEPS = {"c1_gd": 8000, "c1_adamw": 2000, "c1_spectral": 2000, "c3_full_muon": 2000}
QUICK_DLN_SWEEP = {
    "c2_ns_steps": ["train.steps=2000", "sweep.grid.optimizer.ns_steps=[1, 5]", "sweep.grid.data.kappa=[2, 32]"],
    "c2_weight_decay": ["train.steps=2000", "sweep.grid.optimizer.weight_decay=[0.5, 0.8, 1.2]",
                        "sweep.grid.model.depth=[3]", "sweep.grid.optimizer.lr=[0.003, 0.01]"],
    "c2_momentum": ["train.steps=2000", "sweep.grid.optimizer.momentum=[0.0, 0.95]",
                    "sweep.grid.model.init_scale=[0.01]", "sweep.grid.optimizer.lr=[0.003, 0.01]"],
    "c1_adamw_lr": ["train.steps=2000", "sweep.grid.optimizer.lr=[0.003, 0.01]"],
}
QUICK_ARMS_SET = ["runs=16", "epochs=300"]
QUICK_POST_SET = ["runs=32", "epochs=200", "seed=500", "data_seed=5000"]


@dataclass
class Step:
    name: str
    module: str
    args: list
    outputs: list = field(default_factory=list)  # all must exist (with the stamp) for the step to be skipped
    needs: list = field(default_factory=list)  # inputs made by an earlier stage
    copies: list = field(default_factory=list)  # (source, destination) copied after the command succeeds
    device: str = "cpu"  # "jax" = NUTS: replaced by the backend the step reports

    @property
    def cmd(self) -> list[str]:
        return [sys.executable, "-m", self.module, *[str(a) for a in self.args]]


@dataclass
class Ctx:
    quick: bool
    train_device: str
    sample_device: str
    device_source: dict
    jobs: int
    nuts_platform: str | None
    dln: Path
    tms: Path
    res: Path
    logs: Path


def arm_names() -> list[str]:
    return list(json.loads((ROOT / ARMS_CONFIG).read_text(encoding="utf-8"))["arms"])


def default_jobs() -> int:
    """Two thirds of the logical cores (the same rule as `muon_llc.tms.experiments.arms`)."""
    return max(1, (os.cpu_count() or 2) * 2 // 3)


# ----------------------------------------------------------------------------------------- stages
# A stage is a list of groups run in order; a group is a list of chains run side by side; a chain
# is a list of steps run in order.

def stage_dln(c: Ctx) -> list[list[list[Step]]]:
    d = c.dln
    train_set = ["--set", f"device={c.train_device}"]
    sample_set = ["--set", f"device={c.sample_device}"]
    # the two calibrations one after the other; each trained run is its own chain (train, then
    # estimate), so the estimates share the GPU: a sampler step is bound by drawing its noise on
    # the CPU, one core per process
    chains = [[Step(f"{name}", DLN + "static_dln",
                    ["--config", DLN_CONFIGS / f"{name}.yaml", "--runs-root", d, *sample_set,
                     *(QUICK_DLN_B2 if c.quick else [])],
                    [d / name / f for f in ("b2.csv", "b2_summary.txt", "b2_resolution.png")],
                    device=c.sample_device) for name in ("b2_static", "b2_long", "b2_gibbs")]]
    b3 = ["--config", DLN_CONFIGS / "b3_offset.yaml", "--runs-root", d, *sample_set,
          *(QUICK_DLN_B3 if c.quick else [])]
    for arm in DLN_B3_ARMS:
        chains.append([Step(f"b3_offset_{arm}", DLN + "offset_study", [*b3, "--arms", arm],
                            [d / "b3_offset" / f"offset_{arm}.csv"], device=c.sample_device)])
    for name in DLN_RUNS:
        quick = [f"train.steps={QUICK_DLN_STEPS[name]}", "train.n_checkpoints=12", *QUICK_DLN_LLC] if c.quick else []
        chains.append([
            Step(f"train_{name}", DLN + "train",
                 ["--config", DLN_CONFIGS / f"{name}.yaml", "--runs-root", d, *train_set, *quick],
                 [d / name / f for f in ("config.json", "trajectory.csv", "checkpoints.npz")],
                 device=c.train_device),
            Step(f"estimate_{name}", DLN + "estimate", ["--run", d / name, "--device", c.sample_device],
                 [d / name / "llc.csv", d / name / "ladder.csv"], device=c.sample_device),
        ])
    for name, kind in DLN_SWEEPS.items():  # training only
        chain = [Step(f"sweep_{name}", DLN + "sweep",
                      ["--config", DLN_CONFIGS / f"{name}.yaml", "--runs-root", d, *train_set,
                       *(QUICK_DLN_SWEEP[name] if c.quick else [])],
                      [d / name / "summary.csv"], device=c.train_device)]
        if kind is not None:
            chain.append(Step(f"analyze_{name}", DLN + "analyze_c2", ["--sweep", d / name, "--kind", kind],
                              [d / name / f"{kind}_summary.txt"]))
        chains.append(chain)
    plots = [Step("plot_c1", DLN + "plot_c1", ["--runs", d / "c1_gd", d / "c1_spectral", "--out", d / "c1"],
                  [d / "c1" / "singular_values.png"]),
             Step("plot_c3", DLN + "plot_c3", ["--runs", *[d / r for r in DLN_RUNS], "--out", d / "c3"],
                  [d / "c3" / "singular_values.png"]),
             Step("b3_offset_report", DLN + "offset_study", [*b3, "--report"],
                  [d / "b3_offset" / "offset.csv", d / "b3_offset" / "offset_summary.txt"])]
    collect = Step("collect", DLN + "collect", ["--runs-root", d, "--out", c.res / "dln"],
                   [c.res / "dln" / "b2_static" / "b2.csv", c.res / "dln" / "c1" / "singular_values.png",
                    c.res / "dln" / "c2_ns_steps" / "summary.csv"])
    return [chains, [plots], [[collect]]]


def stage_tms_arms(c: Ctx) -> list[list[list[Step]]]:
    out = c.tms / "arms"
    args = ["--config", ARMS_CONFIG, "--out", out, "--jobs", c.jobs, "--device", c.train_device]
    if c.quick:
        args += ["--set", *QUICK_ARMS_SET]
    return [[[Step("arms", TMS + "arms", args, [out / a / "run.npz" for a in arm_names()],
                   device=c.train_device)]]]


def _estimate(c: Ctx, root: Path, arm: str, extra: list, quick: list) -> Step:
    return Step(f"estimate_{root.name}_{arm}", TMS + "estimate",
                ["--run", root / arm, "--device", c.sample_device, *(quick if c.quick else extra)],
                [root / arm / "llc.npz"], needs=[root / arm / "run.npz"], device=c.sample_device)


def stage_tms_llc(c: Ctx) -> list[list[list[Step]]]:
    quick = ["--n-runs", "8", "--steps", "100", "--chains", "1"]
    return [[[_estimate(c, c.tms / "arms", arm, [], quick) for arm in HEADLINE]]]


def stage_tms_k1(c: Ctx) -> list[list[list[Step]]]:
    args = ["--out", c.res / "tms", "--device", c.sample_device]
    if c.quick:
        args += ["--n", "1000", "--steps", "300", "--chains", "4"]
    return [[[Step("table_k1", TMS + "critical_llc", args, [c.res / "tms" / "table_k1.csv"],
                   device=c.sample_device)]]]


def stage_tms_posterior(c: Ctx) -> list[list[list[Step]]]:
    t, res, post = c.tms, c.res / "tms", c.tms / "post"
    inits = t / "posterior_inits_n100.npy"
    nuts = ["--platform", c.nuts_platform] if c.nuts_platform else []
    main = [
        Step("inits", TMS + "posterior",
             ["inits", "--out", t, *nuts,
              *(["--chains", "2", "--warmup", "100", "--n-inits", "32"] if c.quick else [])],
             [res / inits.name], copies=[(inits, res / inits.name)], device="jax"),
        Step("post_arms", TMS + "arms",
             ["--config", ARMS_CONFIG, "--out", post, "--init", inits, "--only", *HEADLINE,
              "--set", *(QUICK_POST_SET if c.quick else POST_SET), "--jobs", min(c.jobs, len(HEADLINE)),
              "--device", c.train_device],
             [post / a / "run.npz" for a in HEADLINE], device=c.train_device),
        *[_estimate(c, post, arm, POST_LLC, ["--n-runs", "32", "--stride", "5", "--steps", "100", "--chains", "1"])
          for arm in HEADLINE],
    ]
    chains = [main]
    for tag, full in OCCUPANCY.items():
        quick = ["--ns", "100", "300", "--seeds", "1", "--chains", "2", "--warmup", "100", "--samples", "200",
                 *(["--nll-scale", "0.5"] if tag else [])]
        name = f"occupancy{tag}.csv"
        chains.append([Step(f"occupancy{tag}", TMS + "posterior",
                            ["occupancy", "--out", t, *(["--tag", tag] if tag else []), *nuts,
                             *(quick if c.quick else full)],
                            [res / name], copies=[(t / name, res / name)], device="jax")])
    return [chains]


def stage_tms_analysis(c: Ctx) -> list[list[list[Step]]]:
    t, res = c.tms, c.res / "tms"
    table = res / "table_k1.csv"
    return [[[
        Step("bah_arms", TMS + "bah", ["--arms", t / "arms", "--out", res, "--table", table],
             [res / f for f in ("arms_summary.csv", "transitions.csv", "trajectory_llc_jumps.csv")],
             needs=[t / "arms" / a / "run.npz" for a in arm_names()] + [t / "arms" / a / "llc.npz" for a in HEADLINE]),
        Step("bah_post", TMS + "bah",
             ["--arms", t / "post", "--out", res, "--table", table, "--prefix", "posterior_init_", "--no-jumps"],
             [res / "posterior_init_arms_summary.csv", res / "posterior_init_transitions.csv"],
             needs=[t / "post" / a / "run.npz" for a in HEADLINE]),
        Step("figures", TMS + "figures",
             ["--arms", t / "arms", "--post", t / "post", "--out", res, "--headline", *HEADLINE],
             # the transition graph is not required: a reduced run may have no transition at all
             [res / "figures" / f for f in ("ensemble_loss.png", "occupancy.png", "posterior_init_loss_vs_llc.png")]),
    ]]]


BUILDERS = {"dln": stage_dln, "tms-arms": stage_tms_arms, "tms-llc": stage_tms_llc, "tms-k1": stage_tms_k1,
            "tms-posterior": stage_tms_posterior, "tms-analysis": stage_tms_analysis}


# ----------------------------------------------------------------------------------------- runner

class Runner:
    JAX_LINE = re.compile(r"jax backend: (\w+)")

    def __init__(self, ctx: Ctx, force: bool, dry_run: bool):
        self.ctx, self.force, self.dry_run = ctx, force, dry_run
        self.lock = threading.Lock()
        self.procs: set[subprocess.Popen] = set()
        self.log = None
        self.env = os.environ | {"PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1",
                                 "MPLBACKEND": "Agg"}

    def say(self, text: str) -> None:
        with self.lock:
            print(text, flush=True)
            if self.log is not None:
                self.log.write(text + "\n")
                self.log.flush()

    def stamp(self, stage: str, step: Step) -> Path:
        return ROOT / self.ctx.logs / "done" / f"{stage}__{step.name}.json"

    def run_step(self, stage: str, step: Step, tag: str) -> dict:
        rec = dict(name=step.name, device=step.device, seconds=0.0, status="ok")
        stamp = self.stamp(stage, step)
        shown = " ".join(["python", *step.cmd[1:]])
        if self.dry_run:
            self.say(f"{tag}{shown}")
            return rec | {"status": "dry-run"}
        if not self.force and stamp.exists() and all((ROOT / o).exists() for o in step.outputs):
            done = json.loads(stamp.read_text(encoding="utf-8"))
            self.say(f"{tag}skip {step.name}: done {done['finished']} on {done.get('device', '?')}")
            return rec | {"status": "skipped", "device": done.get("device", step.device)}
        missing = [str(p) for p in step.needs if not (ROOT / p).exists()]
        if missing:
            self.say(f"{tag}FAILED {step.name}: missing input {missing[0]}"
                     f"{f' (+{len(missing) - 1} more)' if len(missing) > 1 else ''}; run the earlier stages first")
            return rec | {"status": "failed"}
        if self.force:
            for p in [*step.outputs, *[src for src, _ in step.copies]]:
                (ROOT / p).unlink(missing_ok=True)
        stamp.unlink(missing_ok=True)
        self.say(f"{tag}start {step.name} [{step.device}]: {shown}")
        t0 = time.time()
        proc = subprocess.Popen(step.cmd, cwd=ROOT, env=self.env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace")
        self.procs.add(proc)
        for line in proc.stdout:
            self.say(f"{tag}{line.rstrip()}")
            found = self.JAX_LINE.search(line) if step.device == "jax" else None
            if found:
                rec["device"] = f"jax:{found.group(1)}"
        code = proc.wait()
        self.procs.discard(proc)
        rec["seconds"] = round(time.time() - t0, 1)
        if code == 0:
            for src, dst in step.copies:
                (ROOT / dst).parent.mkdir(parents=True, exist_ok=True)
                (ROOT / dst).write_bytes((ROOT / src).read_bytes())
            lost = [str(o) for o in step.outputs if not (ROOT / o).exists()]
            if lost:
                code = f"0 but {lost[0]} was not written"
        if code != 0:
            self.say(f"{tag}FAILED {step.name} after {rec['seconds']:.0f}s (exit {code})")
            return rec | {"status": "failed"}
        stamp.parent.mkdir(parents=True, exist_ok=True)
        stamp.write_text(json.dumps({"finished": datetime.now().isoformat(timespec="seconds"),
                                     "seconds": rec["seconds"], "device": rec["device"], "cmd": step.cmd[1:]},
                                    indent=1), encoding="utf-8")
        self.say(f"{tag}done {step.name} in {rec['seconds']:.0f}s on {rec['device']}")
        return rec

    def run_chain(self, stage: str, chain: list[Step], tagged: bool) -> list[dict]:
        out = []
        for step in chain:
            out.append(self.run_step(stage, step, f"[{step.name}] " if tagged else ""))
            if out[-1]["status"] == "failed":
                break
        return out

    def run_stage(self, stage: str) -> dict:
        c = self.ctx
        groups = BUILDERS[stage](c)
        logs = ROOT / c.logs
        started = datetime.now().isoformat(timespec="seconds")
        t0 = time.time()
        steps: list[dict] = []
        if not self.dry_run:
            logs.mkdir(parents=True, exist_ok=True)
        with open(os.devnull if self.dry_run else logs / f"{stage}.log", "a", encoding="utf-8") as self.log:
            self.say(f"\n===== {stage}  {started}  train on {c.train_device} ({c.device_source['train']}), sample on "
                     f"{c.sample_device} ({c.device_source['sample']}), jobs {c.jobs}"
                     f"{'  (quick)' if c.quick else ''} =====")
            for chains in groups:
                with ThreadPoolExecutor(len(chains)) as ex:
                    results = list(ex.map(lambda ch: self.run_chain(stage, ch, len(chains) > 1), chains))
                steps += [r for chain in results for r in chain]
                if any(r["status"] == "failed" for r in steps):
                    break
            status = ("failed" if any(r["status"] == "failed" for r in steps)
                      else "dry-run" if self.dry_run
                      else "skipped" if all(r["status"] == "skipped" for r in steps) else "ok")
            seconds = round(time.time() - t0, 1)
            self.say(f"===== {stage}: {status} in {seconds:.0f}s =====")
        self.log = None
        rec = dict(stage=stage, started=started, seconds=seconds, status=status, quick=c.quick,
                   train_device=c.train_device, sample_device=c.sample_device, device_source=c.device_source,
                   jobs=c.jobs, platform=platform.platform(), steps=steps)
        if not self.dry_run:
            path = logs / "timings.json"
            hist = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
            path.write_text(json.dumps(hist + [rec], indent=1), encoding="utf-8")
        return rec


# ------------------------------------------------------------------------------ devices, benchmark

def cuda_available() -> bool:
    import torch

    return torch.cuda.is_available()


def choose_device(kind: str, flag: str | None) -> tuple[str, str]:
    """(device, where the choice came from) for kind 'train' or 'sample'. Decided when the runbook
    starts: flag > last --benchmark on this machine > fallback; 'auto' = cuda when available."""
    if flag is not None:
        name, source = flag, "flag"
    else:
        path = ROOT / BENCHMARK_FILE
        saved = json.loads(path.read_text(encoding="utf-8")).get("recommended", {}) if path.exists() else {}
        name, source = (saved[kind], "benchmark") if kind in saved else (FALLBACK_DEVICE[kind], "fallback")
    if name == "auto" or (name == "cuda" and source == "benchmark"):  # a saved choice must still be usable
        name = "cuda" if cuda_available() else "cpu"
    return name, source


def _time_per_step(fn, steps: int) -> float:
    """Milliseconds per step of fn() (which runs `steps` steps), after the caller's warm-up."""
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return 1e3 * (time.perf_counter() - t0) / steps


def benchmark(jobs: int) -> dict:
    """Time a few steps of each kind of work on every available device, extrapolate to the full
    run, print the table and save the recommendation to runs/logs/benchmark.json.

    Training is timed in one single-threaded process, as each arm runs. With `jobs` arms at once a
    CPU run scales with the cores while all arms on a GPU share it, so when the two devices are
    close the CPU is the better choice; the full-run column divides the arm-seconds by `jobs` and
    is a lower bound for both. DLN lambda-hat estimation uses the same batched samplers as the TMS
    estimates and is not timed separately."""
    import numpy as np
    import torch

    from muon_llc.config import load_config
    from muon_llc.dln.experiments.common import build_problem, resolve_derived
    from muon_llc.dln.train import train as dln_train
    from muon_llc.tms.critical_points import critical_points
    from muon_llc.tms.llc import sgld_llc
    from muon_llc.tms.model import init_kgon, make_problem
    from muon_llc.tms.train import train_batched

    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    arms_cfg = json.loads((ROOT / ARMS_CONFIG).read_text(encoding="utf-8"))
    common, arms = arms_cfg["common"], arms_cfg["arms"]
    R, n, bs = common["runs"], common["n"], common["batch_size"]
    spe = n // bs
    rows = []  # (label, kind, {device: ms per step}, steps in the full run, parallel processes)

    def kind_of(opt: dict) -> str:
        return {"gd": "sgd", "adamw": "adamw"}.get(opt["name"], f"{opt['name']} ({opt.get('orth', 'exact')})")

    # TMS training: one arm of each optimizer kind, all its seeds, a few epochs
    by_kind: dict[str, list[str]] = {}
    for name, opt in arms.items():
        by_kind.setdefault(kind_of(opt), []).append(name)
    theta0 = init_kgon(R, common.get("c", 6), 4, common["init_length"], common["init_noise"], common["bias_std"], 0)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    for kind, names in by_kind.items():
        ms = {}
        for dev in devices:
            p = make_problem(n, seed=common["data_seed"], n_datasets=R, device=dev)
            train_batched(p, theta0, arms[names[0]], 1, bs, 0, 10)  # warm-up
            ms[dev] = _time_per_step(lambda: train_batched(p, theta0, arms[names[0]], 4, bs, 0, 10), 4 * spe)
        rows.append((f"TMS training, {kind} x{len(names)} arms", "train", ms,
                     len(names) * common["epochs"] * spe, jobs))
    torch.set_num_threads(threads)

    # DLN training: the three trained runs and the sweeps, grouped by update rule
    def dln_kind(opt: dict) -> str:
        return opt["name"] if opt["name"] in ("gd", "adamw") else f"spectral ({opt.get('orth', 'exact')})"

    dln_steps: dict[str, int] = {}
    dln_cfg: dict[str, dict] = {}
    for name in [*DLN_RUNS, *DLN_SWEEPS]:
        cfg = load_config(ROOT / DLN_CONFIGS / f"{name}.yaml")
        points = int(np.prod([len(v) for v in cfg.get("sweep", {}).get("grid", {}).values()]))
        kind = dln_kind(cfg["optimizer"])
        dln_steps[kind] = dln_steps.get(kind, 0) + points * int(cfg["train"]["steps"])
        dln_cfg.setdefault(kind, cfg)
    for kind, cfg in dln_cfg.items():
        ms = {}
        for dev in devices:
            c = resolve_derived(load_config(ROOT / DLN_CONFIGS / "base.yaml", [f"device={dev}"]))
            prob = build_problem(c)

            def run(k: int, prob=prob, c=c, opt=cfg["optimizer"]):
                return dln_train(prob, opt, k, float(c["model"]["init_scale"]), int(c["model"]["seed"]),
                                 log_every=int(c["train"]["log_every"]), checkpoint_steps=[0, k])

            run(40)
            ms[dev] = _time_per_step(lambda: run(400), 400)
        rows.append((f"DLN training, {kind}", "train", ms, dln_steps[kind], 1 + len(DLN_SWEEPS)))

    # TMS sampling: trajectory lambda-hat (one chunk of 600 centres x 2 chains, n = 1000, one dataset per run)
    # and Table K.1 (a step-size group of 4 critical points x 10 chains, n = 5000)
    cps = np.stack([cp.theta for cp in critical_points()])
    n_ckpt = len(range(0, common["epochs"] // 10 + 1, 3))
    traj_steps = len(HEADLINE) * 500 * -(-n_ckpt * 30 // 600)
    post_steps = len(HEADLINE) * 500 * -(-len(range(0, 501, 10)) * 400 // 600)
    centres = np.resize(cps, (600, cps.shape[1]))
    for label, steps_full, make, kw in (
        ("TMS trajectory lambda-hat (arms + posterior-init)", traj_steps + post_steps,
         lambda dev: make_problem(n, seed=0, n_datasets=30, device=dev),
         dict(centres=centres, n_chains=2, data_index=np.arange(600) % 30)),
        ("TMS Table K.1", 5 * 10_000, lambda dev: make_problem(5000, seed=17, device=dev),
         dict(centres=cps[:4], n_chains=10, eps=1e-6, gamma=0.1)),
    ):
        ms = {}
        for dev in devices:
            p = make(dev)
            sgld_llc(p, n_steps=3, seed=0, **kw)
            ms[dev] = _time_per_step(lambda: sgld_llc(p, n_steps=20, seed=0, **kw), 20)
        rows.append((label, "sample", ms, steps_full, 1))

    try:
        import jax

        jax_backend = f"{jax.default_backend()} {jax.devices()}"
    except ImportError:
        jax_backend = "not installed (uv sync --extra mcmc)"

    total = {kind: {dev: 0.0 for dev in devices} for kind in ("train", "sample")}
    print(f"\n{'work':52s} " + " ".join(f"{d + ' ms/step':>14s} {d + ' full':>10s}" for d in devices))
    detail = []
    for label, kind, ms, steps_full, par in rows:
        full = {dev: ms[dev] * steps_full / 1e3 / par for dev in devices}
        for dev in devices:
            total[kind][dev] += full[dev]
        print(f"{label:52s} " + " ".join(f"{ms[d]:14.3f} {full[d] / 60:8.1f} m" for d in devices))
        detail.append(dict(work=label, kind=kind, ms_per_step=ms, full_run_steps=steps_full, parallel=par,
                           full_run_seconds=full))
    recommended = {kind: min(devices, key=lambda d: total[kind][d]) for kind in total}
    for kind, flag in (("train", "--train-device"), ("sample", "--sample-device")):
        label = {"train": "all training", "sample": "all sampling"}[kind]
        print(f"{label:52s} " + " ".join(f"{'':14s} {total[kind][d] / 60:8.1f} m" for d in devices)
              + f"   -> {flag} {recommended[kind]}")
    print(f"NUTS (numpyro): jax backend {jax_backend}")
    print("full = extrapolation to the full run with ideal scaling over parallel processes; see benchmark().")
    out = dict(recommended=recommended, measured=datetime.now().isoformat(timespec="seconds"),
               platform=platform.platform(), cpu_count=os.cpu_count(), torch=torch.__version__,
               gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, jax_backend=jax_backend,
               jobs=jobs, total_seconds=total, detail=detail)
    path = ROOT / BENCHMARK_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"saved {BENCHMARK_FILE.as_posix()}: later runs use these devices unless a flag says otherwise")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", nargs="+", choices=STAGES, default=STAGES, metavar="STAGE",
                    help=f"subset of: {' '.join(STAGES)} (default: all)")
    ap.add_argument("--quick", action="store_true", help="reduced sizes, everything under runs/quick/")
    ap.add_argument("--force", action="store_true", help="delete the outputs of the selected stages and rerun")
    ap.add_argument("--train-device", choices=["auto", "cpu", "cuda"], default=None,
                    help="device for training (default: last --benchmark, else cpu)")
    ap.add_argument("--sample-device", choices=["auto", "cpu", "cuda"], default=None,
                    help="device for SGLD / MALA sampling (default: last --benchmark, else cuda if available)")
    ap.add_argument("--nuts-platform", choices=["cpu", "gpu"], default=None,
                    help="JAX platform for NUTS (default: whatever JAX finds)")
    ap.add_argument("--jobs", type=int, default=default_jobs(),
                    help="TMS arms trained in parallel (default: two thirds of the logical cores)")
    ap.add_argument("--keep-going", action="store_true", help="continue with the next stage after a failure")
    ap.add_argument("--benchmark", action="store_true",
                    help="time each kind of work on cpu and cuda, save the recommendation, and exit")
    ap.add_argument("--dry-run", action="store_true", help="print the commands of the selected stages and exit")
    args = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    if args.benchmark:
        benchmark(args.jobs)
        return 0
    train, train_src = choose_device("train", args.train_device)
    sample, sample_src = choose_device("sample", args.sample_device)
    base = Path("runs/quick") if args.quick else Path("runs")
    ctx = Ctx(args.quick, train, sample, {"train": train_src, "sample": sample_src}, args.jobs, args.nuts_platform,
              base / "dln", base / "tms", base / "results" if args.quick else Path("results"), base / "logs")
    runner = Runner(ctx, args.force, args.dry_run)
    summary = []
    try:
        for stage in [s for s in STAGES if s in args.stages]:
            summary.append(runner.run_stage(stage))
            if summary[-1]["status"] == "failed" and not args.keep_going:
                break
    except KeyboardInterrupt:
        for proc in list(runner.procs):
            proc.kill()
        print("\ninterrupted; run the same command again to resume", flush=True)
        return 130
    print("\nstage            status    seconds")
    for rec in summary:
        print(f"{rec['stage']:16s} {rec['status']:9s} {rec['seconds']:7.0f}")
    not_run = [s for s in STAGES if s in args.stages and s not in {r["stage"] for r in summary}]
    if not_run:
        print(f"not run: {' '.join(not_run)}")
    return 1 if any(r["status"] == "failed" for r in summary) else 0


if __name__ == "__main__":
    sys.exit(main())
