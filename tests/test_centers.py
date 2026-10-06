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
    null_reduction,
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
    """A wrong factor in the gradient would still yield a plausible subspace; check it independently."""
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
    """Recovering the planted RANK and the planted DIRECTIONS, not merely lowering a number.

    Residual correlation could fall by projecting out almost anything, so the test is principal
    angles against the known planted basis plus an exact rank match.
    """
    Xs, U, _ = planted(P=60, N=120, K=3, strength=0.9, seed=7)
    res = find_center_subspace(Xs, n_restarts=3, rng=np.random.default_rng(8))

    assert res.K == 3, f"planted rank 3, recovered {res.K} (argmin would give {res.K_argmin})"
    assert res.reduction > 0.5, f"only removed {res.reduction:.2f} of the correlation"
    sv = np.linalg.svd(U.T @ res.basis, compute_uv=False)
    assert float(np.min(sv)) > 0.9, f"principal-angle cosines {sv}"


def test_elbow_beats_argmin_at_recovering_the_rank():
    """The reason the selection rule is not argmin.

    Residual correlation keeps creeping down with K, so the outright minimum over-estimates the
    rank -- and over-projection removes directions the classifier could have used, biasing
    capacity downward. The elbow rule must find a strictly smaller, correct rank.
    """
    Xs, _, _ = planted(P=60, N=120, K=3, strength=0.9, seed=17)
    res = find_center_subspace(Xs, n_restarts=3, rng=np.random.default_rng(18))
    assert res.K == 3
    assert res.K_argmin > res.K, (
        f"argmin {res.K_argmin} should over-estimate the planted rank 3"
    )


def test_uncorrelated_centers_get_no_projection_at_all():
    """The correction must not invent structure.

    With isotropic centers there is nothing to remove, so the routine must return an empty
    subspace and the correction becomes a no-op. Fitting a subspace here would remove real
    directions and reduce capacity.
    """
    rng = np.random.default_rng(9)
    Xs = synth.balls(30, 200, D=4, radius=0.2, M=25, rng=rng)
    res = find_center_subspace(Xs, max_K=6, n_restarts=2, rng=np.random.default_rng(10))
    assert res.K == 0, f"projected rank {res.K} onto isotropic centres"
    assert res.basis.shape[1] == 0
    assert res.residual == pytest.approx(res.baseline)


def test_null_reduction_is_positive_and_usable_as_a_threshold():
    """Chance-level reduction is not zero, which is why a fixed threshold is the wrong guard.

    Fitting K directions to P centres always removes some correlation. The null measures how
    much, for this P and N, so that only reductions above it count as evidence.
    """
    nr = null_reduction(P=30, N=150, n_null=2, max_K=6, rng=np.random.default_rng(20))
    assert 0.0 < nr < 0.6, f"implausible chance-level reduction {nr:.3f}"


def test_null_guard_abstains_on_isotropic_data():
    """With n_null > 0 the threshold is measured rather than assumed, and must abstain here."""
    rng = np.random.default_rng(21)
    Xs = synth.balls(30, 150, D=4, radius=0.2, M=20, rng=rng)
    res = find_center_subspace(
        Xs, max_K=6, n_restarts=2, n_null=2, rng=np.random.default_rng(22)
    )
    assert res.K == 0, f"null guard failed to abstain, projected rank {res.K}"


def test_argmin_behaviour_is_still_reachable():
    """k_tol=0, min_reduction=0 reproduces the reference's rule, for comparison with published
    numbers."""
    Xs, _, _ = planted(P=40, N=120, K=3, strength=0.9, seed=23)
    res = find_center_subspace(
        Xs, n_restarts=2, k_tol=0.0, min_reduction=0.0, rng=np.random.default_rng(24)
    )
    assert res.K == res.K_argmin


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
