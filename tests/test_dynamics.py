import numpy as np

from muon_llc.theory.dynamics import (
    crossing_times,
    effective_rank,
    gd_aligned_trajectory,
    gd_escape_time_estimate,
    predicted_lambda_curve,
    spectral_saturation_time,
    spectral_trajectory,
)
from muon_llc.theory.muon_scaling import (
    NS_SMALL_SIGMA_GAIN,
    fit_loglog,
    ns_flattening_threshold,
    predicted_final_singular_values,
    wd_product_cap,
)


def test_gd_learns_in_decreasing_order_with_separated_escape_times():
    s = [4.0, 2.0, 1.0, 0.5]
    t, traj = gd_aligned_trajectory(s, depth=3, alpha=0.02, lr=0.05, steps=200000, record_every=10)
    cross = crossing_times(t, traj, threshold=0.25)
    assert np.all(np.diff(cross) > 0)  # s_1 first, s_4 last
    assert np.allclose(traj[-1], s, rtol=1e-3)
    # escape times roughly inversely proportional to s_i (leading order in alpha)
    est = np.array([gd_escape_time_estimate(si, 3, 0.02, 0.05) for si in s])
    ratio = cross / est
    assert np.all(ratio > 0.5) and np.all(ratio < 2.0), ratio


def test_spectral_all_singular_values_grow_together():
    s = [4.0, 2.0, 1.0, 0.5]
    t, traj = spectral_trajectory(s, depth=3, alpha=0.02, lr=0.005, steps=2000)
    thr = 0.1
    cross = crossing_times(t, traj, thr)
    assert np.all(cross == cross[0])  # single jump time for every direction
    assert np.allclose(traj[-1], s)
    sat = spectral_saturation_time(4.0, 3, 0.02, 0.005)
    assert abs(traj[int(sat) + 2, 0] - 4.0) < 1e-9


def test_predicted_lambda_curves():
    widths = [8, 8, 8, 8]
    s = [4.0, 2.0, 1.0, 0.5]
    t, traj = spectral_trajectory(s, 3, 0.02, 0.005, 2000)
    lam = predicted_lambda_curve(t, traj, 0.1, widths)
    assert set(np.unique(lam)) == {21.5, 29.5}  # single jump
    t2, traj2 = gd_aligned_trajectory(s, 3, 0.02, 0.05, 200000, record_every=10)
    lam2 = predicted_lambda_curve(t2, traj2, 0.1, widths)
    assert set(np.unique(lam2)) == {21.5, 24.0, 26.0, 28.0, 29.5}  # staircase
    assert effective_rank(traj2[-1], 0.1) == 4


def test_scaling_law_helpers():
    assert np.isclose(ns_flattening_threshold(2), NS_SMALL_SIGMA_GAIN**-2)
    assert wd_product_cap(0.5, 3) == 8.0
    assert np.allclose(predicted_final_singular_values([4, 2, 1, 0.5], 1.0, 3), [1, 1, 1, 0.5])
    b, A, r2 = fit_loglog([1, 2, 4, 8], [3, 12, 48, 192])
    assert np.isclose(b, 2.0) and np.isclose(A, 3.0) and r2 > 0.999
