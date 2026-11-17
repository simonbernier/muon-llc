"""From checkpoints to phase sequences and dynamical transitions.

A trajectory is labelled at every checkpoint with `critical_points.classify`. A *plateau* is a
maximal run of consecutive checkpoints with the same label (not "other") of at least `min_ckpts`
checkpoints; short interruptions by "other" (up to `gap` checkpoints) do not split it. The phase
sequence of a run is its ordered list of plateaus and a dynamical transition is a consecutive
pair. Dwell is reported in checkpoints, optimizer steps and parameter path length, because an
optimizer with a normalised step does not slow down near a critical point.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from .critical_points import classify


@dataclass
class Plateau:
    name: str
    start: int  # checkpoint index
    end: int  # inclusive
    path: float  # path length travelled inside the plateau


def label_runs(thetas: np.ndarray, **kw) -> tuple[np.ndarray, np.ndarray]:
    """thetas (K, R, D) -> labels (K, R) object array, theory loss (K, R)."""
    K, R, D = thetas.shape
    names, L = classify(thetas.reshape(K * R, D), c=D // 3, return_loss=True, **kw)
    return np.array(names, dtype=object).reshape(K, R), L.reshape(K, R)


def plateaus(labels: np.ndarray, path_len: np.ndarray | None = None, min_ckpts: int = 3,
             gap: int = 2) -> list[Plateau]:
    """labels: (K,) for one run."""
    K = len(labels)
    segs: list[list] = []  # [name, start, end]
    for t in range(K):
        lab = labels[t]
        if lab == "other":
            continue
        if segs and segs[-1][0] == lab and t - segs[-1][2] <= gap + 1:
            segs[-1][2] = t
        else:
            segs.append([lab, t, t])
    out = []
    for name, s, e in segs:
        if e - s + 1 >= min_ckpts:
            if out and out[-1].name == name:  # same phase re-entered after a short excursion
                out[-1].end = e
            else:
                out.append(Plateau(name, s, e, 0.0))
    if path_len is not None:
        for p in out:
            p.path = float(path_len[p.end] - path_len[p.start])
    return out


def summarize(labels: np.ndarray, path_len: np.ndarray | None = None, min_ckpts: int = 3,
              gap: int = 2) -> dict:
    """labels (K, R). Returns sequences, transition counts, final states, share of 'other'."""
    K, R = labels.shape
    seqs, trans, finals = [], Counter(), Counter()
    dwell: dict[str, list] = {}
    for j in range(R):
        ps = plateaus(labels[:, j], None if path_len is None else path_len[:, j], min_ckpts, gap)
        seqs.append(ps)
        for a, b in zip(ps[:-1], ps[1:]):
            trans[(a.name, b.name)] += 1
        for p in ps[:-1]:  # completed plateaus only
            dwell.setdefault(p.name, []).append((p.end - p.start + 1, p.path))
        finals[ps[-1].name if ps and ps[-1].end >= K - 1 - gap else "other"] += 1
    return dict(
        sequences=seqs, transitions=trans, finals=finals, dwell=dwell,
        frac_other=float((labels == "other").mean()),
        frac_other_late=float((labels[K // 2:] == "other").mean()),
        runs_with_transition=int(sum(len(s) > 1 for s in seqs)), n_runs=R,
    )


def seq_str(ps: list[Plateau]) -> str:
    return " -> ".join(p.name for p in ps) if ps else "(none)"
