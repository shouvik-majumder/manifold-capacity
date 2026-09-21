"""Tests for frame construction and the center-correlation diagnostic."""
from __future__ import annotations

import numpy as np
import pytest

from mancap import analytic, build_frames, center_correlation, synth


def test_frame_shape_and_center_row():
    """Every frame must end in a row of ones: that row IS the center coordinate convention."""
    rng = np.random.default_rng(0)
    Xs = synth.balls(5, 100, D=4, radius=0.3, M=12, rng=rng)
    fs = build_frames(Xs)
    for f, X in zip(fs.frames, Xs):
        assert np.allclose(f[-1, :], 1.0)
        assert f.shape[1] == X.shape[1]
        # Dimension reduced to at most M, not left at the ambient N.
        assert f.shape[0] <= X.shape[1] + 1


def test_reduction_preserves_geometry():
    """Dimension reduction is a change of basis, so all pairwise distances must survive."""
    rng = np.random.default_rng(1)
    Xs = synth.balls(4, 200, D=6, radius=0.4, M=15, rng=rng)
    full = build_frames(Xs, reduce_dim=False)
    for mode in [True, "qr"]:
        red = build_frames(Xs, reduce_dim=mode)
        for a, b in zip(full.frames, red.frames):
            Da = a[:-1].T @ a[:-1]
            Db = b[:-1].T @ b[:-1]
            assert np.allclose(Da, Db, atol=1e-10), f"reduce_dim={mode} changed the Gram matrix"


def test_rank_reduction_gives_the_true_shape_dimension():
    """Default reduction must leave exactly `rank` shape rows, not M of them.

    A segment sampled at M points has rank 1 however many points are used. The reference's QR
    reduction leaves min(N, M) rows, i.e. unreachable padding; the SVD reduction leaves 1.
    """
    rng = np.random.default_rng(30)
    Xs = synth.segments(6, 200, half_length=0.3, M=5, rng=rng)
    svd_frames = build_frames(Xs, reduce_dim=True).frames
    qr_frames = build_frames(Xs, reduce_dim="qr").frames
    for f in svd_frames:
        assert f.shape[0] == 2, f"expected rank 1 plus centre row, got {f.shape[0]}"
    for f in qr_frames:
        assert f.shape[0] == 6, "qr mode should reproduce the reference's M+1 rows"


def test_padding_leaves_alpha_and_radius_but_corrupts_dimension():
    """The reason rank reduction is the default, asserted rather than asserted-to-be-obvious.

    Padding the frame with a direction the manifold cannot reach must leave alpha and R_M
    untouched -- the padded row of S is zero, so that coordinate of V is unconstrained and
    contributes nothing -- while D_M is diluted. For a rank-1 manifold seen in a 2-dimensional
    shape space the expected value is 2*(2/pi)^2 = 0.8106 instead of the correct 1.
    """
    from mancap.capacity import analyze_manifold

    rng = np.random.default_rng(31)
    L, n_t = 0.3, 40000
    t = rng.standard_normal((2, n_t))

    exact = analyze_manifold(np.array([[+L, -L], [1.0, 1.0]]), kappa=0.0, t_vec=t)
    padded = analyze_manifold(
        np.array([[+L, -L], [0.0, 0.0], [1.0, 1.0]]),
        kappa=0.0,
        t_vec=np.vstack([t[0:1], rng.standard_normal((1, n_t)), t[1:2]]),
    )

    assert padded.alpha == pytest.approx(exact.alpha, rel=1e-12)
    assert padded.radius == pytest.approx(exact.radius, rel=1e-12)
    assert exact.dimension == pytest.approx(1.0, abs=0.01)
    assert padded.dimension == pytest.approx(2 * (2 / np.pi) ** 2, abs=0.02)


def test_raw_radius_matches_construction_without_global_mean():
    """Without global-mean subtraction, the reported radius must equal the nominal one exactly.

    A surface-sampled ball has every point at exactly `radius` from its center, so the RMS
    offset divided by the center norm is the radius. This pins the normalisation of the frame to
    an externally known number.

    Offsets are measured from the SAMPLE mean of the M points, not from the true center, so the
    RMS offset is short by the usual finite-sample factor sqrt(1 - 1/M) -- 0.25% at M = 200.
    Asserting the corrected value rather than the nominal one keeps the test exact instead of
    hiding a known bias inside a loose tolerance.
    """
    rng = np.random.default_rng(2)
    M = 200
    for R in [0.1, 0.35, 0.8]:
        Xs = synth.balls(8, 300, D=5, radius=R, M=M, rng=rng, center_norm=1.0)
        fs = build_frames(Xs, subtract_global_mean=False)
        expected = R * np.sqrt(1.0 - 1.0 / M)
        assert np.allclose(fs.raw_radii, expected, rtol=0.01), f"{fs.raw_radii} vs {expected}"


def test_global_mean_subtraction_inflates_radii_as_predicted():
    """With global-mean subtraction, radii grow by 1/sqrt(1 - 1/P). Documented, not a bug.

    Subtracting the grand mean shortens every center vector, and the frame divides offsets by
    the center norm, so radii are inflated by a factor that depends only on the NUMBER of
    manifolds. Asserting the predicted factor turns a surprising 7% discrepancy into a checked
    property, and guards against someone later "fixing" it.
    """
    rng = np.random.default_rng(20)
    R = 0.2
    for P in [8, 60]:
        Xs = synth.balls(P, 400, D=5, radius=R, M=150, rng=rng, center_norm=1.0)
        measured = build_frames(Xs).raw_radii.mean()
        predicted = R / np.sqrt(1.0 - 1.0 / P)
        assert measured == pytest.approx(predicted, rel=0.02), (
            f"P={P}: measured {measured:.4f}, predicted {predicted:.4f}"
        )


def test_rank_reports_true_shape_dimension():
    rng = np.random.default_rng(3)
    for D in [1, 3, 9]:
        Xs = synth.balls(4, 150, D=D, radius=0.3, M=5 * D + 10, rng=rng)
        assert np.all(build_frames(Xs).ranks == D)


def test_center_norms_are_reported():
    """Unequal center norms must be visible, since a single capacity averages over them."""
    rng = np.random.default_rng(4)
    Xs = synth.balls(6, 120, D=3, radius=0.2, M=20, rng=rng, center_norm=1.0)
    Xs = [X * (1.0 + 0.5 * i) for i, X in enumerate(Xs)]
    fs = build_frames(Xs)
    assert fs.center_norms.max() / fs.center_norms.min() > 1.5


def test_single_manifold_is_refused_with_an_explanation():
    """One manifold's center IS the global mean, so the frame cannot be normalised.

    Failing loudly matters here: silently producing a zero-norm center would give a frame full of
    infinities or, worse, a plausible-looking number from a degenerate normalisation.
    """
    rng = np.random.default_rng(5)
    Xs = synth.balls(1, 50, D=2, radius=0.3, M=10, rng=rng)
    with pytest.raises(ValueError, match="global origin"):
        build_frames(Xs)


def test_mismatched_ambient_dimension_is_refused():
    with pytest.raises(ValueError, match="ambient dimension"):
        build_frames([np.zeros((10, 3)), np.zeros((11, 3))])


# ----------------------------------------------------------------------------------------------
# Center correlation: measuring the assumption rather than assuming it
# ----------------------------------------------------------------------------------------------

def test_random_centers_look_uncorrelated():
    """Isotropic centers in high dimension: mean |cos| ~ sqrt(2/(pi N)), participation ~ P-1.

    This is the regime the replica theory assumes and the regime the synthetic validation runs
    in, so establishing it explicitly is what licenses comparing those results to theory.
    """
    rng = np.random.default_rng(6)
    P, N = 30, 400
    Xs = synth.balls(P, N, D=3, radius=0.2, M=10, rng=rng)
    info = center_correlation(Xs)
    expected_cos = np.sqrt(2.0 / (np.pi * N))
    assert info["mean_abs_cos"] < 4 * expected_cos
    assert info["participation"] > 0.6 * (P - 1)


def test_collinear_centers_are_detected():
    """Centers along one direction must give participation ~ 1 and a high mean |cos|.

    This is the pathology real representations exhibit, and the reason an uncorrected capacity
    from real data is biased. The diagnostic has to fire loudly on it.
    """
    rng = np.random.default_rng(7)
    N, P = 200, 20
    axis = rng.standard_normal((N, 1))
    axis /= np.linalg.norm(axis)
    Xs = []
    for p in range(P):
        c = axis * (1.0 + 0.1 * p)
        Xs.append(c + 0.02 * rng.standard_normal((N, 8)))
    info = center_correlation(Xs)
    assert info["participation"] < 2.0, f"participation {info['participation']:.2f}"
    assert info["n_95"] <= 2
    assert info["mean_abs_cos"] > 0.8


def test_center_correlation_spectrum_is_normalised():
    rng = np.random.default_rng(8)
    Xs = synth.balls(12, 100, D=2, radius=0.2, M=6, rng=rng)
    info = center_correlation(Xs)
    assert np.isclose(info["spectrum"].sum(), 1.0)
