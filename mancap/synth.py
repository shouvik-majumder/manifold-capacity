"""Synthetic manifolds with known geometry, for validation.

Every generator returns a list of P arrays of shape (N, M), the format `frames.build_frames`
consumes. Centers are drawn isotropically at random, which satisfies the general-position
assumption of the theory, so results are comparable to theory without the center
correction.

Sampling: the inner minimisation sees a manifold through the convex hull of its sampled
points. A ball sampled with M surface points presents an inscribed polytope, so its
measured capacity is higher than the ball's, and the gap closes slowly with M when D is
large. scripts/02_validate_balls_and_segments.py measures this bias as a function of M and
D. `ball(..., fill=True)` samples the interior instead; interior points lie inside the hull
of the surface points and contribute little to capacity.
"""
from __future__ import annotations

import numpy as np


def _random_centers(P: int, N: int, rng: np.random.Generator, norm: float = 1.0) -> np.ndarray:
    """P center vectors, isotropically random with fixed norm. Shape (N, P)."""
    C = rng.standard_normal((N, P))
    C /= np.linalg.norm(C, axis=0, keepdims=True)
    return C * norm


def correlated_centers(
    P: int,
    N: int,
    rng: np.random.Generator,
    K: int = 3,
    strength: float = 0.8,
    norm: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """P centers with a planted K-dimensional shared component. Shape (N, P), plus its basis.

    Built as  center = strength * (shared K-dim component) + sqrt(1 - strength^2) * (isotropic),
    then renormalised. This mimics the structure of real representations, in which a few
    directions carry most of the between-category variance.

    The planted basis is returned so that `centers.find_center_subspace` can be tested on
    whether it recovers this subspace, rather than only on whether the residual correlation
    decreases.

    Args:
        K: Dimension of the planted shared subspace.
        strength: 0 gives isotropic centers, 1 puts them entirely inside the subspace.

    Returns:
        (centers (N, P), planted basis (N, K) orthonormal).
    """
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must lie in [0, 1]")
    U = _random_subspace(N, K, rng)
    shared = U @ rng.standard_normal((K, P))
    shared /= np.linalg.norm(shared, axis=0, keepdims=True)
    iso = rng.standard_normal((N, P))
    iso /= np.linalg.norm(iso, axis=0, keepdims=True)
    C = strength * shared + np.sqrt(1.0 - strength ** 2) * iso
    C /= np.linalg.norm(C, axis=0, keepdims=True)
    return C * norm, U


def balls_with_centers(
    centers: np.ndarray,
    D: int,
    radius: float,
    M: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Balls placed at supplied centers, so center geometry and manifold geometry are decoupled.

    Used with `correlated_centers` to build data where the theory's general-position assumption
    is violated in a controlled, known way while every manifold's shape stays identical.
    """
    N, P = centers.shape
    scale = float(np.mean(np.linalg.norm(centers, axis=0)))
    out = []
    for p in range(P):
        U = _random_subspace(N, D, rng)
        S = rng.standard_normal((D, M))
        S /= np.linalg.norm(S, axis=0, keepdims=True)
        out.append(centers[:, [p]] + scale * radius * (U @ S))
    return out


def _random_subspace(N: int, D: int, rng: np.random.Generator) -> np.ndarray:
    """An orthonormal basis for a random D-dimensional subspace of R^N. Shape (N, D)."""
    A = rng.standard_normal((N, D))
    Q, _ = np.linalg.qr(A)
    return Q[:, :D]


def points(P: int, N: int, rng: np.random.Generator, center_norm: float = 1.0) -> list[np.ndarray]:
    """P manifolds that are single points. The case where capacity must equal alpha_0(kappa).

    Note that `frames.build_frames` cannot normalise a one-point manifold's offsets (they are
    identically zero, giving D = 0 after reduction), which is correct: a point has no shape, and
    its frame is the single row of ones. This is handled, and it is the end-to-end test of the
    kappa = 0, alpha = 2 result.
    """
    C = _random_centers(P, N, rng, center_norm)
    return [C[:, [p]].copy() for p in range(P)]


def balls(
    P: int,
    N: int,
    D: int,
    radius: float,
    M: int,
    rng: np.random.Generator,
    center_norm: float = 1.0,
    fill: bool = False,
) -> list[np.ndarray]:
    """P D-dimensional balls of the given radius, each in its own random subspace.

    Args:
        radius: Radius in units of the center norm, matching the frame convention, so a radius
            of 0.2 means the ball's extent is 20% of its distance from the origin.
        M: Points sampled per manifold. On the surface by default; see the module docstring for
            why this biases capacity upward.
        fill: Sample the interior (uniformly by volume) instead of the surface.
    """
    C = _random_centers(P, N, rng, center_norm)
    out = []
    for p in range(P):
        U = _random_subspace(N, D, rng)
        S = rng.standard_normal((D, M))
        S /= np.linalg.norm(S, axis=0, keepdims=True)
        if fill:
            # r = u^(1/D) gives uniform density by volume in D dimensions.
            S *= rng.random(M) ** (1.0 / D)
        out.append(C[:, [p]] + center_norm * radius * (U @ S))
    return out


def segments(
    P: int,
    N: int,
    half_length: float,
    M: int,
    rng: np.random.Generator,
    center_norm: float = 1.0,
) -> list[np.ndarray]:
    """P line segments, sampled uniformly. The D = 1 case, where the hull is exact with M = 2.

    Because a segment's convex hull is determined by its two endpoints, capacity here is
    independent of M for M >= 2. That makes segments the cleanest test that the sampling bias
    seen with balls really is a hull effect and not a bug in the estimator.
    """
    C = _random_centers(P, N, rng, center_norm)
    out = []
    for p in range(P):
        U = _random_subspace(N, 1, rng)
        s = np.linspace(-1.0, 1.0, M).reshape(1, M)
        out.append(C[:, [p]] + center_norm * half_length * (U @ s))
    return out


def rings(
    P: int,
    N: int,
    radius: float,
    M: int,
    rng: np.random.Generator,
    center_norm: float = 1.0,
) -> list[np.ndarray]:
    """P circles (1-spheres) of the given radius, each in its own random 2-plane.

    A ring is a D = 2 manifold whose points all sit at the same distance from the center, so
    its hull is a regular M-gon. Included as a reference geometry: cyclic variables such as
    weekdays are represented on circles in some language-model activations.
    """
    C = _random_centers(P, N, rng, center_norm)
    theta = np.linspace(0.0, 2.0 * np.pi, M, endpoint=False)
    circle = np.stack([np.cos(theta), np.sin(theta)], axis=0)  # (2, M)
    out = []
    for p in range(P):
        U = _random_subspace(N, 2, rng)
        out.append(C[:, [p]] + center_norm * radius * (U @ circle))
    return out


def ellipsoids(
    P: int,
    N: int,
    radii: np.ndarray,
    M: int,
    rng: np.random.Generator,
    center_norm: float = 1.0,
) -> list[np.ndarray]:
    """P ellipsoids with the given per-axis radii, each in its own random subspace.

    The case that separates R_M and D_M from the geometric radius and dimension. An ellipsoid
    with one long axis and many short ones has a large geometric radius and dimension, but the
    classifier recruits anchors almost exclusively along the long axis, so D_M should come out
    near 1 while the rank is D. An implementation that reports D_M near D here is computing
    the geometric rather than the anchor dimension.
    """
    radii = np.asarray(radii, dtype=np.float64)
    D = radii.size
    C = _random_centers(P, N, rng, center_norm)
    out = []
    for p in range(P):
        U = _random_subspace(N, D, rng)
        S = rng.standard_normal((D, M))
        S /= np.linalg.norm(S, axis=0, keepdims=True)
        out.append(C[:, [p]] + center_norm * (U @ (radii[:, None] * S)))
    return out
