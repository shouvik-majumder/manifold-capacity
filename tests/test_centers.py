"""Tests for the correlated-centers correction.

Layered the same way as the rest of the package: the gradient against finite differences, the
optimizer against problems with a known answer, and the subspace search against a PLANTED
subspace, so success means "it found the right directions" rather than "the number went down".
"""
from __future__ import annotations

import numpy as np
import pytest

from mancap import build_frames, center_correlation, synth
from mancap.centers import (
    check_gradient,
    find_center_subspace,
    residual_correlation,
    square_corrcoeff_cost,
    stiefel_minimize,
)


def planted(P=24, N=120, K=3, strength=0.85, seed=0):
    rng = np.random.default_rng(seed)
    C, U = synth.correlated_centers(P, N, rng, K=K, strength=strength)
    Xs = synth.balls_with_centers(C, D=4, radius=0.2, M=25, rng=rng)
    return Xs, U, rng


# ----------------------------------------------------------------------------------------------
# The cost function and its gradient
# ----------------------------------------------------------------------------------------------

def test_gradient_matches_finite_differences():
    """The one piece that could be silently wrong: a bad factor still yields a plausible subspace."""
    rng = np.random.default_rng(1)
    X = rng.standard_normal((12, 11))
    for K in [1, 2, 4]:
        err = check_gradient(X, K, rng)
        assert err < 1e-6, f"K={K}: gradient error {err:.2e}"


def test_cost_is_invariant_to_the_basis_of_V():
    """Only the SUBSPACE matters, so right-multiplying V by an orthogonal matrix changes nothing.

    If this failed, the optimizer would be chasing a particular basis rather than a subspace, and
    the choice of K would be meaningless.
    """
    rng = np.random.default_rng(2)
    X = rng.standard_normal((15, 14))
    V = np.linalg.qr(rng.standard_normal((14, 3)))[0]
    Q = np.linalg.qr(rng.standard_normal((3, 3)))[0]
    c1, _ = square_corrcoeff_cost(V, X, grad=False)
    c2, _ = square_corrcoeff_cost(V @ Q, X, grad=False)
    assert c1 == pytest.approx(c2, rel=1e-10)


def test_cost_floor_is_the_diagonal():
    """Diagonal terms are identically 1, so the cost can never fall below P/2."""
    rng = np.random.default_rng(3)
    P = 10
    X = rng.standard_normal((P, 9))
    V = np.linalg.qr(rng.standard_normal((9, 2)))[0]
    cost, _ = square_corrcoeff_cost(V, X, grad=False)
    assert cost >= P / 2 - 1e-9


def test_residual_correlation_bounds():
    """Uncorrelated in high dimension is near 0; identical centers give 1."""
    rng = np.random.default_rng(4)
    X = rng.standard_normal((20, 300))
    assert residual_correlation(None, X) < 0.15

    same = np.repeat(rng.standard_normal((1, 30)), 12, axis=0)
    assert residual_correlation(None, same) == pytest.approx(1.0, abs=1e-9)


# ----------------------------------------------------------------------------------------------
# The optimizer
# ----------------------------------------------------------------------------------------------

def test_stiefel_minimize_keeps_the_constraint_exactly():
    rng = np.random.default_rng(5)
    X = rng.standard_normal((18, 17))
    V0 = np.linalg.qr(rng.standard_normal((17, 4)))[0]
    V, _, info = stiefel_minimize(
        lambda V: square_corrcoeff_cost(V, X, grad=True), V0, max_iter=300
    )
    assert np.allclose(V.T @ V, np.eye(4), atol=1e-10), "orthogonality lost"
    assert info["n_iter"] >= 1


def test_stiefel_minimize_finds_the_top_subspace_of_a_quadratic():
    """On a problem with a known answer -- maximise trace(V^T A V) -- it must find the top
    eigenvectors.

    This validates the optimizer independently of the correlation cost, which has no closed-form
    optimum to compare against. Minimising -trace(V^T A V) over the Stiefel manifold has the
    span of A's top-K eigenvectors as its solution, so the recovered subspace can be checked
    exactly by principal angles.
    """
    rng = np.random.default_rng(6)
    n, K = 12, 3
    B = rng.standard_normal((n, n))
    A = B @ B.T
    evals, evecs = np.linalg.eigh(A)
    target = evecs[:, -K:]

    def cost_grad(V):
        return -float(np.trace(V.T @ A @ V)), -2.0 * (A @ V)

    V0 = np.linalg.qr(rng.standard_normal((n, K)))[0]
    V, cost, _ = stiefel_minimize(cost_grad, V0, max_iter=2000, tol=1e-10)

    # Principal angles: the singular values of target^T V are all 1 iff the spans coincide.
    sv = np.linalg.svd(target.T @ V, compute_uv=False)
    assert np.allclose(sv, 1.0, atol=1e-5), f"principal angles {np.arccos(np.clip(sv, -1, 1))}"
    assert cost == pytest.approx(-float(np.sum(evals[-K:])), rel=1e-7)


# ----------------------------------------------------------------------------------------------
# The subspace search, against a planted answer
# ----------------------------------------------------------------------------------------------

def test_finds_the_planted_subspace():
    """Recovering the planted directions, not merely lowering a number.

    Measured by principal angles between the recovered basis and the planted one. Getting the
    rank right AND the directions right is the real test; residual correlation could fall by
    projecting out almost anything.
    """
    Xs, U, _ = planted(P=24, N=120, K=3, strength=0.9, seed=7)
    res = find_center_subspace(Xs, n_restarts=3, rng=np.random.default_rng(8))

    assert res.residual < 0.6 * res.baseline, (
        f"residual {res.residual:.4f} vs baseline {res.baseline:.4f}"
    )
    assert res.K >= 2, f"planted rank 3, recovered {res.K}"

    # The planted subspace should be largely contained in the recovered one.
    k = min(res.K, U.shape[1])
    sv = np.linalg.svd(U.T @ res.basis, compute_uv=False)[:k]
    assert float(np.mean(sv)) > 0.8, f"principal-angle cosines {sv}"


def test_uncorrelated_centers_leave_little_to_remove():
    """With isotropic centers the baseline is already low and the correction gains little.

    The correction must not invent structure. If it reported a large improvement here, it would
    be fitting noise, and every capacity computed through it on real data would be suspect.
    """
    rng = np.random.default_rng(9)
    Xs = synth.balls(24, 400, D=4, radius=0.2, M=25, rng=rng)
    res = find_center_subspace(Xs, max_K=4, n_restarts=2, rng=np.random.default_rng(10))
    assert res.baseline < 0.2, f"isotropic baseline unexpectedly high: {res.baseline:.3f}"
    assert res.residual > 0.2 * res.baseline, "correction removed too much from random centers"


def test_projection_reduces_measured_center_correlation():
    """End to end: projecting the subspace out must lower the diagnostic in `frames`."""
    Xs, _, _ = planted(P=24, N=120, K=3, strength=0.9, seed=11)
    before = center_correlation(Xs)
    res = find_center_subspace(Xs, n_restarts=2, rng=np.random.default_rng(12))
    B = res.basis
    after = center_correlation([X - B @ (B.T @ X) for X in Xs])
    assert after["mean_abs_cos"] < before["mean_abs_cos"]
    assert after["participation"] > before["participation"]


def test_build_frames_accepts_the_subspace():
    """The correction has to reach the frames, and the frames must stay well formed."""
    Xs, _, _ = planted(seed=13)
    res = find_center_subspace(Xs, max_K=4, n_restarts=2, rng=np.random.default_rng(14))
    fs = build_frames(Xs, center_subspace=res.basis)
    for f in fs.frames:
        assert np.allclose(f[-1, :], 1.0)
        assert np.all(np.isfinite(f))


def test_build_frames_rejects_a_mismatched_subspace():
    Xs, _, _ = planted(seed=15)
    with pytest.raises(ValueError, match="N rows"):
        build_frames(Xs, center_subspace=np.zeros((7, 2)))


def test_too_few_manifolds_is_refused():
    rng = np.random.default_rng(16)
    Xs = synth.balls(3, 50, D=2, radius=0.2, M=8, rng=rng)
    with pytest.raises(ValueError, match="at least 4"):
        find_center_subspace(Xs)
