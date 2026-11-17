from fractions import Fraction
from itertools import product

import pytest

from muon_llc.dln.theory.aoyagi import (
    dln_learning_coefficient,
    reduced_rank_regression_aoyagi_watanabe,
    staircase,
)


@pytest.mark.parametrize("M,N", [(1, 1), (3, 5), (8, 8), (2, 7)])
def test_depth_one_is_regular(M, N):
    for r in range(0, min(M, N) + 1):
        assert dln_learning_coefficient([M, N], r).lam == Fraction(M * N, 2)


def test_depth_two_matches_aoyagi_watanabe():
    for M, H, N in product(range(1, 7), range(1, 7), range(1, 7)):
        for r in range(0, min(M, H, N) + 1):
            got = dln_learning_coefficient([M, H, N], r).lam
            want = reduced_rank_regression_aoyagi_watanabe(M, H, N, r)
            assert got == want, (M, H, N, r, got, want)


@pytest.mark.parametrize(
    "widths,r,lam,m",
    [
        ([1, 1, 1], 0, Fraction(1, 2), 2),  # (ab)^2: normal crossing, lambda 1/2, m 2
        ([2, 1, 2], 0, Fraction(1), 2),  # |u|^2 s^2 ... two-dim factors: 1 with m 2
        ([1, 2, 1], 0, Fraction(1, 2), 1),  # (a.b)^2 with a,b in R^2: smooth hypersurface
        ([1, 1, 1, 1], 0, Fraction(1, 2), 3),  # (abc)^2: three crossing hyperplanes, m 3
        ([1, 1, 1, 1, 1], 0, Fraction(1, 2), 4),  # (abcd)^2: m 4
        ([2, 1, 1, 2], 0, Fraction(1, 2), None),
    ],
)
def test_scalar_toys(widths, r, lam, m):
    lc = dln_learning_coefficient(widths, r)
    assert lc.lam == lam
    if m is not None:
        assert lc.multiplicity == m


def test_target_table_8_8_8_8():
    vals = [lc.value for lc in staircase([8, 8, 8, 8], 4)]
    assert vals == [21.5, 24.0, 26.0, 28.0, 29.5]
    assert dln_learning_coefficient([8, 8, 8, 8], 8).value == 32.0  # regular 8x8 map: 64/2


def test_monotone_in_rank():
    for widths in ([8, 8, 8, 8], [4, 6, 5, 7], [3, 3, 3]):
        vals = [lc.value for lc in staircase(widths)]
        assert all(b >= a for a, b in zip(vals, vals[1:]))


def test_boundary_choice_of_sigma_is_continuous():
    # (1,2,1): both Sigma={0,2} (ell=1) and Sigma=all (ell=2) sit on the boundary and agree.
    from muon_llc.dln.theory.aoyagi import _candidate

    deltas = sorted([1, 2, 1])
    ok1, _, S1 = _candidate(deltas, 1)
    ok2, _, S2 = _candidate(deltas, 2)
    assert ok1 and ok2
    assert dln_learning_coefficient([1, 2, 1], 0).lam == Fraction(1, 2)


def test_bad_inputs():
    with pytest.raises(ValueError):
        dln_learning_coefficient([8], 0)
    with pytest.raises(ValueError):
        dln_learning_coefficient([8, 8], 9)
