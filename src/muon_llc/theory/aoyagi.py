"""Learning coefficient (RLCT) of a deep linear network, after Aoyagi (2024).

Setting. A DLN with L weight matrices W_L ... W_1, layer widths
H_0 = M (input), H_1, ..., H_{L-1} (hidden), H_L = N (output), realising a
linear map of rank r (the "true" map). Regression with Gaussian noise. The
learning coefficient lambda and multiplicity m govern the free energy
F_n = n L_n(w*) + lambda log n - (m - 1) log log n + O(1).

Formula (as implemented here; see the note below).
    Delta_i = H_i - r,  i = 0..L, sorted ascending.
    Choose Sigma = the (ell + 1) smallest Deltas, ell >= 1, such that
        max_{i in Sigma} Delta_i  <=  S / ell  <=  min_{i not in Sigma} Delta_i,
    where S = sum_{i in Sigma} Delta_i, and let a = S mod ell. Then
        lambda = a (ell - a) / (4 ell) + (r (M + N) - r^2) / 2
                 + S^2 / (4 ell) - (1/4) sum_{i in Sigma} Delta_i^2
        m      = 1 if a == 0 else 2.

NOTE ON PROVENANCE. arxiv.org and the journal sites were unreachable from the
machine this was written on, so the formula above is reconstructed from memory
and verified against independent special cases in tests/test_aoyagi.py:
  * L = 1 (a regular model): lambda = M N / 2 for every rank r;
  * L = 2 reproduces the reduced-rank-regression result of Aoyagi & Watanabe
    (2005), including the +1/8 parity correction and the boundary cases;
  * scalar toys whose RLCT can be computed by hand (e.g. widths (1,1,1),
    r = 0 gives lambda = 1/2, m = 2 for the normal crossing (ab)^2);
  * the value is continuous across the boundary of the Sigma condition, so
    the selection is robust to ties.
Please cross-check against Aoyagi (2024) Theorem 1 / Furman & Lau (2024)
before quoting the multiplicity m; lambda is the quantity Phase C uses.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence


@dataclass(frozen=True)
class LearningCoefficient:
    lam: Fraction
    multiplicity: int
    ell: int
    a: int
    sigma: tuple[int, ...]  # sorted Delta values in Sigma

    @property
    def value(self) -> float:
        return float(self.lam)


def _candidate(deltas_sorted: Sequence[int], ell: int) -> tuple[bool, Fraction, int]:
    """Check the Sigma condition for the (ell+1) smallest deltas; return (ok, S/ell, S)."""
    sigma = deltas_sorted[: ell + 1]
    rest = deltas_sorted[ell + 1 :]
    S = sum(sigma)
    avg = Fraction(S, ell)
    ok = Fraction(max(sigma)) <= avg and (not rest or avg <= Fraction(min(rest)))
    return ok, avg, S


def dln_learning_coefficient(widths: Sequence[int], rank: int) -> LearningCoefficient:
    """Learning coefficient of a DLN with the given widths (H_0, ..., H_L) and true rank."""
    widths = [int(h) for h in widths]
    if len(widths) < 2:
        raise ValueError("need at least one weight matrix, i.e. len(widths) >= 2")
    if any(h < 1 for h in widths):
        raise ValueError("widths must be positive")
    M, N = widths[0], widths[-1]
    r = int(rank)
    if r < 0 or r > min(widths):
        raise ValueError(f"rank must satisfy 0 <= r <= min(widths) = {min(widths)}, got {r}")

    deltas = sorted(h - r for h in widths)
    base = Fraction(r * (M + N) - r * r, 2)

    # Degenerate case: some Delta_i = 0, i.e. the true map already saturates a layer.
    # The extra (singular) contribution vanishes: the model is locally regular in the
    # remaining directions and lambda = (r(M+N) - r^2)/2. (Check: L=1, r=min(M,N) -> MN/2.)
    if deltas[0] == 0:
        return LearningCoefficient(base, 1, 0, 0, (0,))

    chosen = None
    for ell in range(1, len(deltas)):
        ok, _, S = _candidate(deltas, ell)
        if ok:
            chosen = (ell, S)
            break
    if chosen is None:  # should not happen; fall back to all layers
        ell = len(deltas) - 1
        chosen = (ell, sum(deltas))
    ell, S = chosen
    sigma = tuple(deltas[: ell + 1])
    a = S % ell
    lam = (
        Fraction(a * (ell - a), 4 * ell)
        + base
        + Fraction(S * S, 4 * ell)
        - Fraction(sum(d * d for d in sigma), 4)
    )
    m = 1 if a == 0 else 2
    return LearningCoefficient(lam, m, ell, a, sigma)


def staircase(widths: Sequence[int], max_rank: int | None = None) -> list[LearningCoefficient]:
    """lambda(r) for r = 0..max_rank (default min(widths))."""
    if max_rank is None:
        max_rank = min(widths)
    return [dln_learning_coefficient(widths, r) for r in range(max_rank + 1)]


def reduced_rank_regression_aoyagi_watanabe(M: int, H: int, N: int, r: int) -> Fraction:
    """Independent implementation of the L = 2 case (Aoyagi & Watanabe 2005), used in tests.

    Cases, for input M, hidden H, output N, true rank r:
      (1) N + r <= M + H, M + r <= N + H, H + r <= M + N, M+H+N+r even:
            [2(H+r)(M+N) - (M-N)^2 - (H+r)^2] / 8
      (2) same but M+H+N+r odd: add 1/8
      (3) M + H < N + r:  [H M - H r + N r] / 2   (the {B = 0}-type component dominates)
      (4) N + H < M + r:  [H N - H r + M r] / 2
      (5) M + N < H + r:  [M N] / 2
    """
    if M + H < N + r:
        return Fraction(H * M - H * r + N * r, 2)
    if N + H < M + r:
        return Fraction(H * N - H * r + M * r, 2)
    if M + N < H + r:
        return Fraction(M * N, 2)
    val = Fraction(2 * (H + r) * (M + N) - (M - N) ** 2 - (H + r) ** 2, 8)
    if (M + H + N + r) % 2 == 1:
        val += Fraction(1, 8)
    return val


def _main() -> None:
    p = argparse.ArgumentParser(description="Aoyagi DLN learning-coefficient table (A3)")
    p.add_argument("--widths", type=int, nargs="+", default=[8, 8, 8, 8])
    p.add_argument("--max-rank", type=int, default=None)
    args = p.parse_args()
    print(f"widths = {args.widths}   (L = {len(args.widths) - 1} weight matrices)")
    print(f"{'r':>3} {'lambda':>10} {'m':>3} {'ell':>4} {'a':>3}  Sigma")
    for r, lc in enumerate(staircase(args.widths, args.max_rank)):
        print(f"{r:>3} {lc.value:>10.4f} {lc.multiplicity:>3} {lc.ell:>4} {lc.a:>3}  {lc.sigma}")
    d = sum(args.widths[i] * args.widths[i + 1] for i in range(len(args.widths) - 1))
    print(f"(parameter count d = {d}, regular value d/2 = {d / 2})")


if __name__ == "__main__":
    _main()
