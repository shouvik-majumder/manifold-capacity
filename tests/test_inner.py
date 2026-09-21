"""Tests for the inner minimization.

The strategy is to never test a solver against itself. Each test either checks a solver against
a closed form, against an algebraically unrelated solver, or against the KKT conditions, which
are a statement about what the answer must satisfy rather than about how it was found.
"""
from __future__ import annotations

import numpy as np
import pytest

from mancap.inner import solve_active_set, solve_ball, solve_slsqp


def random_manifold(rng, D=None, M=None, scale=0.4):
    """A random manifold frame: D shape rows plus the row of ones."""
    D = int(rng.integers(1, 7)) if D is None else D
    M = int(rng.integers(2, 10)) if M is None else M
    return np.vstack([rng.standard_normal((D, M)) * scale, np.ones((1, M))])


# ----------------------------------------------------------------------------------------------
# KKT: what the answer must satisfy, independent of how it was computed
# ----------------------------------------------------------------------------------------------

def test_active_set_satisfies_kkt():
    """Primal feasibility, dual feasibility, stationarity and complementary slackness.

    These four conditions are necessary and sufficient for the optimum of a convex QP, so a
    solution satisfying all of them IS the optimum. This test does not compare against anything;
    it verifies the defining property directly.
    """
    rng = np.random.default_rng(0)
    for _ in range(400):
        S = random_manifold(rng)
        t = rng.standard_normal(S.shape[0]) * rng.uniform(0.5, 3.0)
        kappa = float(rng.uniform(0.0, 0.5))
        sol = solve_active_set(S, t, kappa=kappa)

        slack = S.T @ sol.v - kappa
        assert np.all(slack > -1e-8), "primal infeasible"
        assert np.all(sol.mu >= -1e-10), "dual infeasible (negative multiplier)"
        # Stationarity: V - T must be the nonnegative combination S @ mu.
        assert np.allclose(sol.v - t, S @ sol.mu, atol=1e-9), "stationarity violated"
        # Complementary slackness: mu_j > 0 only where the constraint is tight.
        assert np.all(sol.mu[slack > 1e-7] < 1e-9), "complementary slackness violated"


def test_anchor_reconstructs_the_solution():
    """V = T + lambda * anchor with lambda = sum(mu). The relation R_M and D_M depend on."""
    rng = np.random.default_rng(1)
    for _ in range(200):
        S = random_manifold(rng)
        t = rng.standard_normal(S.shape[0]) * 2.0
        sol = solve_active_set(S, t, kappa=0.2)
        if sol.interior:
            continue
        lam = sol.mu.sum()
        assert np.allclose(sol.v, t + lam * sol.anchor, atol=1e-9)
        # The anchor is a convex combination of manifold points, so its center coordinate, which
        # is 1 for every point, must also be 1.
        assert sol.anchor[-1] == pytest.approx(1.0, abs=1e-9)


def test_interior_case_is_exact():
    """When T is already feasible the solution is T itself, with F exactly zero."""
    rng = np.random.default_rng(2)
    # A tiny manifold far along the positive center direction: almost any T with a large center
    # component is feasible.
    S = np.vstack([rng.standard_normal((3, 5)) * 1e-3, np.ones((1, 5))])
    t = np.array([0.0, 0.0, 0.0, 5.0])
    sol = solve_active_set(S, t)
    assert sol.interior
    assert sol.f == 0.0
    assert np.array_equal(sol.v, t)


# ----------------------------------------------------------------------------------------------
# Solver against algebraically unrelated solver
# ----------------------------------------------------------------------------------------------

def test_active_set_matches_slsqp():
    """Dual active set vs primal SLSQP. No shared algebra: one forms the Gram matrix and does
    combinatorial pivoting, the other runs a smooth line search on the primal."""
    rng = np.random.default_rng(3)
    worst_f = worst_anchor = 0.0
    for _ in range(300):
        S = random_manifold(rng)
        t = rng.standard_normal(S.shape[0]) * rng.uniform(0.5, 3.0)
        kappa = float(rng.uniform(0.0, 0.4))
        a = solve_active_set(S, t, kappa=kappa)
        b = solve_slsqp(S, t, kappa=kappa)
        worst_f = max(worst_f, abs(a.f - b.f))
        if not a.interior and not b.interior:
            worst_anchor = max(worst_anchor, float(np.max(np.abs(a.anchor - b.anchor))))
    assert worst_f < 1e-8, f"objective disagreement {worst_f:.2e}"
    assert worst_anchor < 1e-6, f"anchor disagreement {worst_anchor:.2e}"


# ----------------------------------------------------------------------------------------------
# Solver against closed form
# ----------------------------------------------------------------------------------------------

def test_point_manifold_closed_form():
    """For a point manifold (no shape rows) F(T) = max(0, kappa - T)^2 exactly."""
    rng = np.random.default_rng(4)
    S = np.ones((1, 1))
    for kappa in [0.0, 0.3, 1.0]:
        for _ in range(200):
            t = rng.standard_normal(1)
            sol = solve_active_set(S, t, kappa=kappa)
            assert sol.f == pytest.approx(max(0.0, kappa - t[0]) ** 2, abs=1e-12)


def test_ball_closed_form_constraint_is_correct():
    """The cone reduction v_c - R||v_s|| >= kappa must equal the true minimum over the ball.

    Checks the algebra of `solve_ball` against brute force over sampled ball points. Note the
    asymmetry, which is the whole story of sampling in this method: finitely many samples can
    only MISS the worst direction, never invent a worse one, so the brute-force minimum is an
    upper bound on the analytic one. Requiring equality is therefore only legitimate at low D,
    where samples do cover the sphere; at D = 6 the bound is one-sided and that is what gets
    asserted.
    """
    rng = np.random.default_rng(5)
    R, kappa, M = 0.35, 0.1, 20000
    for D, tight in [(1, True), (2, True), (6, False)]:
        for _ in range(60):
            t = rng.standard_normal(D + 1) * 1.5
            sol = solve_ball(t, radius=R, kappa=kappa)
            v = sol.v
            analytic_min = v[-1] - R * np.linalg.norm(v[:-1])
            P = rng.standard_normal((D, M))
            P /= np.linalg.norm(P, axis=0, keepdims=True)
            pts = np.vstack([R * P, np.ones((1, M))])
            brute_min = float(np.min(pts.T @ v))
            assert brute_min >= analytic_min - 1e-9, "sampling found a WORSE point than the ball"
            if tight:
                assert analytic_min == pytest.approx(brute_min, abs=5e-3)
            # The returned V must be feasible for the true ball, and tight when a constraint
            # binds -- that is what makes it the projection onto the cone boundary.
            assert analytic_min >= kappa - 1e-9, "returned V is infeasible for the ball"
            if not sol.interior:
                assert analytic_min == pytest.approx(kappa, abs=1e-9)


def test_sampled_hull_relaxes_the_ball_one_directionally():
    """Sampling a ball gives an inscribed polytope, so F can only DECREASE, never increase.

    This is the sharpest available statement about sampling bias: the sampled constraint set is a
    strict relaxation of the ball's, so F_sampled <= F_exact must hold for every single draw, not
    merely on average. A positive violation is a solver bug. The gap itself is a real property of
    the estimator and must shrink as M grows.
    """
    rng = np.random.default_rng(6)
    R, D = 0.3, 4
    prev_gap = np.inf
    for M in [10, 100, 1000]:
        gaps = []
        for _ in range(60):
            t = rng.standard_normal(D + 1)
            exact = solve_ball(t, radius=R)
            P = rng.standard_normal((D, M))
            P /= np.linalg.norm(P, axis=0, keepdims=True)
            S = np.vstack([R * P, np.ones((1, M))])
            samp = solve_active_set(S, t)
            assert samp.f - exact.f < 1e-9, "sampled manifold gave a LARGER F than the ball"
            gaps.append(exact.f - samp.f)
        gap = float(np.mean(gaps))
        assert gap < prev_gap, "sampling bias failed to shrink with more samples"
        prev_gap = gap


# ----------------------------------------------------------------------------------------------
# Invariances
# ----------------------------------------------------------------------------------------------

def test_rotation_invariance():
    """F depends only on the geometry, so rotating the frame and T together changes nothing.

    The center row must be preserved by the rotation, so the rotation acts on the D shape
    coordinates only.
    """
    rng = np.random.default_rng(7)
    S = random_manifold(rng, D=5, M=8)
    t = rng.standard_normal(6)
    Q, _ = np.linalg.qr(rng.standard_normal((5, 5)))
    R = np.eye(6)
    R[:5, :5] = Q
    a = solve_active_set(S, t, kappa=0.15)
    b = solve_active_set(R @ S, R @ t, kappa=0.15)
    assert a.f == pytest.approx(b.f, abs=1e-10)
    assert np.allclose(R @ a.anchor, b.anchor, atol=1e-9)


def test_duplicate_and_interior_points_are_free():
    """Adding duplicated points, or points inside the hull, cannot change the answer.

    The inner problem sees only the convex hull, so this is a defining property rather than a
    nice-to-have. It is also the practical reason capacity is insensitive to how densely a cloud
    is sampled in its interior, and sensitive only to how well its extremes are covered.
    """
    rng = np.random.default_rng(8)
    S = random_manifold(rng, D=3, M=6)
    t = rng.standard_normal(4) * 2
    base = solve_active_set(S, t, kappa=0.1)

    dup = np.concatenate([S, S[:, :3]], axis=1)
    assert solve_active_set(dup, t, kappa=0.1).f == pytest.approx(base.f, abs=1e-10)

    # A convex combination of existing points lies inside the hull.
    w = rng.random(S.shape[1])
    w /= w.sum()
    inside = np.concatenate([S, (S @ w).reshape(-1, 1)], axis=1)
    assert solve_active_set(inside, t, kappa=0.1).f == pytest.approx(base.f, abs=1e-10)
