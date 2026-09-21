"""Tests for the direct simulation, including the analytic margin cases.

The margin kappa* is the distance from the origin to the convex hull of the labelled points, so
small hand-built configurations have exactly computable answers. Those come first: they pin the
identity that the whole simulation rests on, before any sampling is involved.
"""
from __future__ import annotations

import numpy as np
import pytest

from mancap import analytic, simulate, synth


# ----------------------------------------------------------------------------------------------
# kappa* against hand computation
# ----------------------------------------------------------------------------------------------

def test_margin_single_point():
    """One point: the hull is the point, so kappa* is its norm."""
    z = np.array([[3.0], [4.0]])
    assert simulate.max_margin(z).kappa_star == pytest.approx(5.0)


def test_margin_segment():
    """Two points: the hull is a segment, and kappa* is the origin's distance to it."""
    # (1,0) and (0,1): nearest hull point is (0.5, 0.5), distance 1/sqrt(2).
    Z = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert simulate.max_margin(Z).kappa_star == pytest.approx(1.0 / np.sqrt(2.0), abs=1e-10)

    # (1,1) and (1,-1): nearest hull point is (1,0), distance 1. Here the foot of the
    # perpendicular is interior to the segment, the other case of the projection.
    Z = np.array([[1.0, 1.0], [1.0, -1.0]])
    assert simulate.max_margin(Z).kappa_star == pytest.approx(1.0, abs=1e-10)


def test_margin_is_zero_when_origin_is_inside_the_hull():
    """A triangle containing the origin is inseparable, so kappa* = 0."""
    Z = np.array([[1.0, -1.0, 0.0], [-0.5, -0.5, 1.0]])
    res = simulate.max_margin(Z)
    assert res.kappa_star == pytest.approx(0.0, abs=1e-6)
    assert not simulate.is_separable(Z)


def test_margin_matches_independent_optimizer():
    """Gilbert's algorithm vs SLSQP on the smooth simplex form. Different algorithms, same (+)."""
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(60):
        N = int(rng.integers(3, 12))
        total = int(rng.integers(4, 16))
        Z = rng.standard_normal((N, total))
        a = simulate.max_margin(Z)
        b = simulate.max_margin_qp(Z)
        assert a.converged, f"Gilbert did not converge, gap={a.gap:.2e}"
        worst = max(worst, abs(a.kappa_star - b))
    assert worst < 1e-6, f"margin disagreement {worst:.2e}"


def test_margin_certificate_is_returned_and_tight():
    """The gap must be reported, and the two bounds must bracket the answer."""
    rng = np.random.default_rng(1)
    Z = rng.standard_normal((8, 20)) + 3.0  # well separated from the origin
    res = simulate.max_margin(Z)
    assert res.converged
    # The gap must be at machine precision RELATIVE to the objective it is a difference of,
    # which is kappa^2. An absolute bound here would be a bound on the data's scale, not on the
    # solver's accuracy.
    assert res.gap <= 1e-12 * max(1.0, res.kappa_star ** 2)
    assert res.n_iter >= 1
    assert res.kappa_lower <= res.kappa_star
    assert res.kappa_lower == pytest.approx(res.kappa_star, rel=1e-6)
    # w must actually achieve the claimed margin.
    assert float(np.min(Z.T @ res.w)) == pytest.approx(res.kappa_star, rel=1e-6)


def test_separability_agrees_with_positive_margin():
    """The exact LP test and the margin bounds must agree. Two unrelated decisions.

    The comparison uses the certified bounds rather than the raw value. `kappa_lower > 0` PROVES
    separability, and `kappa_star == 0` proves the opposite, so the LP verdict must lie between
    them; a case where the bounds straddle zero is undecided by the margin routine and is
    skipped rather than asserted on. In practice the pairwise steps close the gap to machine
    precision and nothing is skipped.
    """
    rng = np.random.default_rng(2)
    undecided = 0
    for _ in range(60):
        N = int(rng.integers(3, 10))
        total = int(rng.integers(3, 25))
        Z = rng.standard_normal((N, total))
        lp = simulate.is_separable(Z)
        res = simulate.max_margin(Z)
        if res.kappa_lower > 1e-9:
            assert lp, "margin proves separability but the LP says otherwise"
        elif res.kappa_star < 1e-9:
            assert not lp, "margin is zero but the LP found a separating hyperplane"
        else:
            undecided += 1
    assert undecided <= 3, f"{undecided}/60 instances left undecided by the margin bounds"


def test_margin_scales_linearly():
    """kappa* is a distance, so scaling the data scales it by the same factor."""
    rng = np.random.default_rng(3)
    Z = rng.standard_normal((6, 12)) + 1.5
    base = simulate.max_margin(Z).kappa_star
    assert simulate.max_margin(3.7 * Z).kappa_star == pytest.approx(3.7 * base, rel=1e-8)


# ----------------------------------------------------------------------------------------------
# Machinery around the scan
# ----------------------------------------------------------------------------------------------

def test_crossing_interpolates_and_reports_failure():
    a = np.array([1.0, 2.0, 3.0])
    assert simulate.crossing(a, np.array([1.0, 0.5, 0.0]), 0.5) == pytest.approx(2.0)
    assert simulate.crossing(a, np.array([1.0, 0.75, 0.5]), 0.5) == pytest.approx(3.0)
    # Never crosses: must be NaN, not silently clipped to an endpoint.
    assert np.isnan(simulate.crossing(a, np.array([1.0, 0.9, 0.8]), 0.5))
    assert np.isnan(simulate.crossing(a, np.array([0.4, 0.3, 0.2]), 0.5))


def test_extrapolate_threshold_recovers_a_known_line():
    """A synthetic 1/N trend must be recovered exactly."""
    Ns = [50, 100, 200, 400]
    true_intercept, true_slope = 2.0, -12.0
    thr = [true_intercept + true_slope / N for N in Ns]
    ex = simulate.extrapolate_threshold(Ns, thr)
    assert ex["intercept"] == pytest.approx(true_intercept, abs=1e-9)
    assert ex["slope"] == pytest.approx(true_slope, abs=1e-9)
    assert ex["resid"] < 1e-9


def test_labelled_points_folds_labels():
    Xs = [np.ones((3, 2)), np.full((3, 4), 2.0)]
    Z = simulate.labelled_points(Xs, np.array([1.0, -1.0]))
    assert Z.shape == (3, 6)
    assert np.all(Z[:, :2] == 1.0)
    assert np.all(Z[:, 2:] == -2.0)


# ----------------------------------------------------------------------------------------------
# The headline validation, kept small enough to run in CI
# ----------------------------------------------------------------------------------------------

@pytest.mark.slow
def test_point_threshold_is_cover_bound():
    """Direct simulation of points must find alpha_c = 2 without using the theory anywhere.

    This is the test that catches a misapplied theory rather than a miscoded one. It builds
    points, labels them randomly, and asks an exact LP whether a separating hyperplane exists.
    Finite N smears the transition, so the threshold is extrapolated in 1/N.
    """
    make = lambda P, N, r: synth.points(P, N, r)
    alphas = np.arange(1.4, 2.7, 0.1)
    Ns, thr = [25, 50, 100], []
    for N in Ns:
        scan = simulate.threshold_scan(
            make, N, alphas, n_seeds=40, rng=np.random.default_rng(100 + N),
            compute_margin=False,
        )
        thr.append(simulate.crossing(scan.alphas, scan.frac_sep, 0.5))
    ex = simulate.extrapolate_threshold(Ns, thr)
    assert ex["intercept"] == pytest.approx(analytic.point_capacity(0.0), abs=0.12), (
        f"extrapolated alpha_c = {ex['intercept']:.3f}, thresholds {thr}"
    )


@pytest.mark.slow
def test_segment_threshold_is_below_point_threshold():
    """Segments must be measurably harder to separate than points, by direct simulation.

    A qualitative but theory-free statement: giving each category extent can only reduce the
    number of categories a readout can handle. Uses segments because their convex hull is exact
    with two sample points, so there is no sampling bias to confound the comparison.
    """
    alphas = np.arange(0.6, 2.4, 0.15)
    N = 50
    pts = simulate.threshold_scan(
        lambda P, N_, r: synth.points(P, N_, r), N, alphas, n_seeds=30,
        rng=np.random.default_rng(11), compute_margin=False,
    )
    segs = simulate.threshold_scan(
        lambda P, N_, r: synth.segments(P, N_, half_length=0.6, M=2, rng=r), N, alphas,
        n_seeds=30, rng=np.random.default_rng(12), compute_margin=False,
    )
    a_pts = simulate.crossing(pts.alphas, pts.frac_sep, 0.5)
    a_seg = simulate.crossing(segs.alphas, segs.frac_sep, 0.5)
    assert a_seg < a_pts, f"segments {a_seg:.3f} should be harder than points {a_pts:.3f}"
