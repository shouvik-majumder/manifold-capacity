"""Tests for capacity, anchor radius and anchor dimension.

Note on tolerances: capacity is a Monte Carlo average, so a test that hard-codes a tolerance is
either loose enough to be useless or tight enough to be flaky. Instead these tests use the
`f_sem` that `analyze_manifold` reports, and assert agreement within a stated number of standard
errors. That makes the tolerance self-calibrating in `n_t` and turns "is it close?" into a
statistical statement.
"""
from __future__ import annotations

import numpy as np
import pytest

from mancap import analytic, build_frames, synth
from mancap.capacity import (
    ManifoldResult,
    analyze_manifold,
    anchor_dimension,
    anchor_radius,
    ball_capacity_mc,
    combine,
)


def within_sem(result, expected: float, n_sigma: float = 4.0) -> bool:
    """Is the measured capacity within n_sigma of expected, given the Monte Carlo error?

    alpha = 1/f_mean, so the relative error of alpha equals the relative error of f_mean, i.e.
    f_sem/f_mean. This converts the reported standard error on F into one on alpha.
    """
    rel = result.f_sem / result.f_mean
    return abs(result.alpha - expected) <= n_sigma * expected * rel


# ----------------------------------------------------------------------------------------------
# The fixed point of the whole library
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("kappa", [0.0, 0.25, 0.5, 1.0])
def test_point_capacity_end_to_end(kappa):
    """Point manifolds must reproduce alpha_0(kappa), and alpha = 2 at kappa = 0.

    Run through the full public path (synthetic points, frame construction, capacity) so this
    tests the preprocessing as well as the estimator. A one-point manifold has zero offsets, so
    its frame is degenerate by construction and must still yield the right answer.
    """
    rng = np.random.default_rng(0)
    Xs = synth.points(P=40, N=120, rng=rng)
    frames = build_frames(Xs).frames
    res = analyze_manifold(frames[0], kappa=kappa, n_t=40000, rng=np.random.default_rng(1))
    assert within_sem(res, analytic.point_capacity(kappa)), (
        f"alpha={res.alpha:.4f} vs theory {analytic.point_capacity(kappa):.4f}, "
        f"rel sem {res.f_sem / res.f_mean:.4f}"
    )


def test_points_have_no_radius_or_dimension():
    """A point has no extent and no preferred directions, so R_M = D_M = 0 exactly."""
    rng = np.random.default_rng(2)
    frames = build_frames(synth.points(P=20, N=80, rng=rng)).frames
    res = analyze_manifold(frames[0], kappa=0.0, n_t=2000, rng=np.random.default_rng(3))
    assert res.radius == pytest.approx(0.0, abs=1e-12)
    assert res.dimension == pytest.approx(0.0, abs=1e-12)


def test_internal_consistency_of_the_two_f_routes():
    """F from the solver objective and F recomputed through the anchor must agree to ~1e-12.

    These are different computations: one is ||V-T||^2 straight from the minimizer, the other
    goes through the anchor point and the dual relation. Agreement tests the anchors, which are
    what R_M and D_M are built from, and a discrepancy here would invalidate the geometry even
    when capacity looks right.
    """
    rng = np.random.default_rng(4)
    for maker in [
        lambda r: synth.balls(6, 100, D=5, radius=0.3, M=40, rng=r),
        lambda r: synth.segments(6, 100, half_length=0.4, M=8, rng=r),
        lambda r: synth.rings(6, 100, radius=0.25, M=24, rng=r),
        lambda r: synth.ellipsoids(6, 100, radii=np.array([0.6, 0.05, 0.05, 0.05]), M=40, rng=r),
    ]:
        for frame in build_frames(maker(rng)).frames:
            res = analyze_manifold(frame, kappa=0.1, n_t=300, rng=np.random.default_rng(5))
            assert res.consistency < 1e-10, f"consistency {res.consistency:.2e}"


# ----------------------------------------------------------------------------------------------
# Anchor geometry is not naive geometry
# ----------------------------------------------------------------------------------------------

def test_anchor_dimension_of_a_needle_is_near_one():
    """An ellipsoid with one long axis and many short ones has D_M ~ 1, not D.

    This is the test that distinguishes anchor geometry from plain geometry. The cloud genuinely
    occupies D dimensions -- `ranks` confirms it -- but the classifier only ever recruits support
    vectors near the ends of the long axis, so the dimension it effectively presents is about 1.
    An implementation that reported D_M near D here would be computing the wrong thing while
    still producing plausible capacities.
    """
    rng = np.random.default_rng(6)
    radii = np.array([0.5] + [0.01] * 9)  # one long axis, nine nearly flat ones
    Xs = synth.ellipsoids(8, 150, radii=radii, M=200, rng=rng)
    fs = build_frames(Xs)
    assert fs.ranks[0] == 10, "the cloud really is 10-dimensional"
    res = analyze_manifold(fs.frames[0], kappa=0.0, n_t=2000, rng=np.random.default_rng(7))
    assert res.dimension < 2.0, f"D_M={res.dimension:.2f} should be near 1 for a needle"


def test_anchor_dimension_grows_with_isotropic_dimension():
    """For isotropic balls D_M should rise with D. A monotone trend, not a precise value."""
    rng = np.random.default_rng(8)
    dims = [2, 5, 15]
    got = []
    for D in dims:
        Xs = synth.balls(8, 200, D=D, radius=0.3, M=30 * D, rng=rng)
        res = analyze_manifold(
            build_frames(Xs).frames[0], kappa=0.0, n_t=1500, rng=np.random.default_rng(9)
        )
        got.append(res.dimension)
    assert got[0] < got[1] < got[2], f"D_M not monotone in D: {got}"


def test_larger_manifolds_have_lower_capacity():
    """Capacity must fall monotonically as the radius grows. The basic qualitative claim."""
    rng = np.random.default_rng(10)
    alphas = []
    for radius in [0.05, 0.2, 0.5, 1.0]:
        Xs = synth.balls(8, 200, D=4, radius=radius, M=120, rng=rng)
        res = [
            analyze_manifold(f, kappa=0.0, n_t=1200, rng=np.random.default_rng(11))
            for f in build_frames(Xs).frames
        ]
        alphas.append(combine(res))
    assert all(a > b for a, b in zip(alphas, alphas[1:])), f"not monotone: {alphas}"
    assert alphas[0] < 2.05, "a small manifold should not exceed the point capacity"


# ----------------------------------------------------------------------------------------------
# Invariances that must hold for the frame convention to mean what it claims
# ----------------------------------------------------------------------------------------------

def test_global_scale_invariance():
    """Multiplying every activation by a constant must not change anything.

    The frame divides offsets by the center norm, so capacity depends on the RATIO of extent to
    center distance. If this test failed, R_M would not be the dimensionless quantity it is
    documented to be, and capacities from differently normalised layers would not be comparable
    -- which is exactly the comparison the method exists to make.
    """
    rng = np.random.default_rng(12)
    Xs = synth.balls(6, 120, D=4, radius=0.3, M=60, rng=rng)
    t_vec = np.random.default_rng(13).standard_normal((61, 400))

    a = build_frames(Xs).frames
    b = build_frames([17.3 * X for X in Xs]).frames
    for fa, fb in zip(a, b):
        ra = analyze_manifold(fa, kappa=0.1, t_vec=t_vec[: fa.shape[0]])
        rb = analyze_manifold(fb, kappa=0.1, t_vec=t_vec[: fb.shape[0]])
        assert ra.alpha == pytest.approx(rb.alpha, rel=1e-9)
        assert ra.radius == pytest.approx(rb.radius, rel=1e-9)


def test_combine_is_harmonic_not_arithmetic():
    """Total capacity pools F, i.e. it is the harmonic mean of the per-manifold capacities."""
    fake = [
        ManifoldResult(alpha=a, radius=0, dimension=0, D=1, f_mean=1 / a, f_sem=0,
                       frac_interior=0, consistency=0, n_active_mean=1)
        for a in [1.0, 4.0]
    ]
    assert combine(fake) == pytest.approx(1.6)  # harmonic mean of 1 and 4, not 2.5


# ----------------------------------------------------------------------------------------------
# The ball: a closed-form inner solution as an oracle
# ----------------------------------------------------------------------------------------------

def test_sampled_ball_approaches_closed_form_ball():
    """Densely sampled ball manifolds must converge to the exact ball capacity from above.

    Uses `inner.solve_ball` (a cone projection, no iteration) as the oracle. Capacity comes out
    too HIGH when M is small, because the sampled hull is an inscribed polytope and therefore
    easier to separate. The test asserts both the direction and the convergence.
    """
    D, R = 3, 0.3
    exact_alpha, exact_R, exact_D = ball_capacity_mc(
        D, radius=R, kappa=0.0, n_t=20000, rng=np.random.default_rng(14)
    )
    # subtract_global_mean=False so that the manifolds really have radius R rather than the
    # inflated R/sqrt(1-1/P); otherwise this test would be comparing a radius-0.32 sampled ball
    # against a radius-0.30 exact one and the discrepancy would be attributed to sampling.
    rng = np.random.default_rng(15)
    prev = np.inf
    for M in [20, 100, 600]:
        Xs = synth.balls(8, 200, D=D, radius=R, M=M, rng=rng)
        res = [
            analyze_manifold(f, kappa=0.0, n_t=1500, rng=np.random.default_rng(16))
            for f in build_frames(Xs, subtract_global_mean=False).frames
        ]
        alpha = combine(res)
        assert alpha > exact_alpha - 0.05, "sampled capacity fell below the exact ball"
        assert alpha < prev + 1e-9, f"capacity not decreasing with M: {alpha} vs {prev}"
        prev = alpha
    assert prev == pytest.approx(exact_alpha, rel=0.15), (
        f"densely sampled {prev:.3f} vs exact ball {exact_alpha:.3f}"
    )


def test_ball_radius_recovers_the_geometric_radius_at_low_dimension():
    """For a D=1 ball (a segment through the center) R_M should track the true half-length.

    At D = 1 the anchor is always one of the two endpoints, so the anchor radius and the
    geometric radius coincide. This pins the normalisation of R_M to something externally known,
    which no internal consistency check can do.
    """
    for R in [0.1, 0.3, 0.6]:
        _, R_M, D_M = ball_capacity_mc(1, radius=R, kappa=0.0, n_t=20000,
                                       rng=np.random.default_rng(17))
        assert R_M == pytest.approx(R, rel=0.05), f"R_M={R_M:.4f} for true radius {R}"
        assert D_M == pytest.approx(1.0, abs=0.05)


def test_radius_and_dimension_helpers_on_hand_built_anchors():
    """Direct unit tests of Eqs. 28 and 29 on anchors with a known answer."""
    # Two anchors at +/- r along a single shape axis, center coordinate 1. Their mean is 0, so
    # each deviates by r, giving R_M = r.
    r = 0.4
    anchors = np.array([[r, -r], [1.0, 1.0]])
    assert anchor_radius(anchors) == pytest.approx(r)

    # Anchor perfectly aligned with t in a D=1 shape space: mean cosine 1, so D_M = D = 1.
    t = np.array([[2.0, -3.0], [0.5, 0.5]])
    aligned = np.array([[1.0, -1.0], [1.0, 1.0]])
    assert anchor_dimension(t, aligned) == pytest.approx(1.0)
    # Anti-aligned gives mean cosine -1, and D_M squares it, so still 1.
    anti = np.array([[-1.0, 1.0], [1.0, 1.0]])
    assert anchor_dimension(t, anti) == pytest.approx(1.0)
