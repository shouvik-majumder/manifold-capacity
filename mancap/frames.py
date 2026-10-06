"""Turning raw activations into the (D+1)-dimensional frames the theory expects.

The replica result is stated for a manifold in its own coordinates: D shape directions plus
one center direction, with the center coordinate equal to 1 for every point. `build_frames`
performs this change of coordinates in four steps.

1. Subtract the global mean of all samples from all manifolds. This assumes the readout has
   no bias term, so that the origin of the classifier's coordinate system is the grand mean.

2. Per manifold, compute its center c and the offsets X - c.

3. Divide the offsets by ||c||. R_M is then dimensionless (a radius of 0.3 means an extent of
   30% of the distance from the origin) and capacity depends only on the ratio of extent to
   center norm, so rescaling all activations changes nothing. Step 1 shrinks the center
   norms, so step 3 inflates every radius by about 1/sqrt(1 - 1/P) relative to a generator's
   nominal radius (7% at P = 8, under 1% at P = 60). Pass subtract_global_mean=False when
   the nominal radius is the quantity of interest.

4. Reduce the shape dimension to the rank of the offsets, then append the row of ones. The
   reference implementation reduces to M rows with a thin QR instead, which leaves at least
   one direction in which the manifold has zero extent. That padding leaves alpha and R_M
   unchanged and lowers D_M: for a rank-1 manifold seen in a 2-dimensional shape space,
   0.81 instead of 1 (the predicted value is 2 (2/pi)^2 = 0.8106). Pass reduce_dim="qr" to
   reproduce the reference's behaviour.

Correlations between manifold centers, which the theory assumes absent, are measured by
`center_correlation` and corrected by `centers.find_center_subspace`; `build_frames` accepts
the resulting subspace through `center_subspace`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.linalg import qr


@dataclass
class FrameSet:
    """Manifolds in their own (D+1)-frames, plus diagnostics.

    frames        List of (D_i+1, M_i) arrays, ready for `capacity.analyze_manifold`.
    center_norms  ||c_i|| for each manifold, after global-mean subtraction. Unequal norms mean
                  the manifolds lie at different distances from the origin, so a single
                  capacity number averages over heterogeneous geometry.
    raw_radii     The geometric radius sqrt(mean ||x - c||^2) / ||c|| of each cloud, reported
                  alongside R_M, which measures only the part of the cloud recruited as anchors.
    ranks         Numerical rank of each cloud's offsets, i.e. the shape dimension D_i.
    """

    frames: list[np.ndarray]
    center_norms: np.ndarray
    raw_radii: np.ndarray
    ranks: np.ndarray


def build_frames(
    Xs: Sequence[np.ndarray],
    subtract_global_mean: bool = True,
    reduce_dim: bool | str = True,
    rank_tol: float | None = None,
    center_subspace: np.ndarray | None = None,
) -> FrameSet:
    """Build the (D+1)-frames for a list of manifolds.

    Args:
        Xs: Sequence of P arrays, each (N, M_i): N features, M_i sampled points. N must match
            across manifolds; M_i may differ.
        subtract_global_mean: Subtract the mean over all points of all manifolds first. Leave
            True unless the data are already centered. Note that it inflates radii by
            1/sqrt(1 - 1/P); see the module docstring.
        reduce_dim: How to restrict each frame to the directions the manifold spans. True
            (default) reduces to the numerical rank of the offsets via SVD, which keeps D_M
            unbiased. "qr" reproduces the reference implementation's thin QR to M rows, which
            biases D_M downward; provided for comparison against published numbers. False keeps
            all N rows and is only useful for testing invariances.
        rank_tol: Absolute singular-value threshold for the rank. Default follows numpy's
            matrix_rank convention, max(shape) * eps * largest singular value.
        center_subspace: Optional (N, K) orthonormal basis, from
            `centers.find_center_subspace`, projected out of every manifold before the frame is
            built. This is the correlated-centers correction. None is appropriate for synthetic
            data with random centers; `center_correlation` measures whether real data need it.

    Returns:
        FrameSet.
    """
    Xs = [np.asarray(X, dtype=np.float64) for X in Xs]
    if len({X.shape[0] for X in Xs}) != 1:
        raise ValueError("all manifolds must share the ambient dimension N")

    if subtract_global_mean:
        origin = np.concatenate(Xs, axis=1).mean(axis=1, keepdims=True)
    else:
        origin = np.zeros((Xs[0].shape[0], 1))

    frames: list[np.ndarray] = []
    center_norms, raw_radii, ranks = [], [], []

    if center_subspace is not None:
        center_subspace = np.asarray(center_subspace, dtype=np.float64)
        if center_subspace.shape[0] != Xs[0].shape[0]:
            raise ValueError("center_subspace must have N rows, matching the ambient dimension")

    for X in Xs:
        X0 = X - origin
        if center_subspace is not None:
            # Remove the shared center structure. Applied to the whole manifold, not just its
            # center, so that the frame is built inside the null space rather than having the
            # subspace subtracted after the fact.
            X0 = X0 - center_subspace @ (center_subspace.T @ X0)
        c = X0.mean(axis=1, keepdims=True)
        cn = float(np.linalg.norm(c))
        # Compare against the data's own scale rather than against exact zero: with a single
        # manifold the center cancels the global mean algebraically, but floating point leaves a
        # residue of order 1e-16 * ||X||, which would sail through a `cn == 0` test and produce a
        # frame scaled by that residue -- enormous, finite, and completely meaningless.
        scale = float(np.linalg.norm(X0)) / max(np.sqrt(X0.size), 1.0)
        if cn <= 1e-10 * max(scale, 1.0):
            raise ValueError(
                "a manifold center coincides with the global origin, so the frame cannot be "
                "normalised; this happens with a single manifold (its center IS the global "
                "mean) -- pass subtract_global_mean=False or supply more manifolds"
            )
        offsets = X0 - c
        S = offsets / cn

        raw_radii.append(float(np.sqrt(np.mean(np.sum(S ** 2, axis=0)))))
        center_norms.append(cn)

        U, sv, _ = np.linalg.svd(S, full_matrices=False)
        tol = (
            rank_tol if rank_tol is not None
            else (sv[0] * max(S.shape) * np.finfo(float).eps if sv.size else 0.0)
        )
        k = int(np.sum(sv > tol))
        ranks.append(k)

        if reduce_dim == "qr":
            if S.shape[0] > S.shape[1]:
                Q, _ = qr(S, mode="economic")
                S = Q.T @ S
        elif reduce_dim:
            # Project onto the left singular vectors that carry nonzero extent. Purely a change
            # of basis within the manifold's own span; k = 0 (a single point) correctly yields a
            # frame consisting of the center row alone, hence D = 0.
            S = U[:, :k].T @ S

        m = S.shape[1]
        frames.append(np.concatenate([S, np.ones((1, m))], axis=0))

    return FrameSet(
        frames=frames,
        center_norms=np.array(center_norms),
        raw_radii=np.array(raw_radii),
        ranks=np.array(ranks),
    )


def center_correlation(Xs: Sequence[np.ndarray]) -> dict[str, float | np.ndarray]:
    """How strongly the manifold centers violate the theory's general-position assumption.

    The replica calculation treats the P manifold centers as effectively random relative to one
    another. In real representations a few directions carry most of the between-category
    variance, so the centers are concentrated in a low-dimensional subspace and an uncorrected
    capacity estimate is biased. This function measures the violation.

    Returns a dict with:
        mean_abs_cos   Mean |cosine| between distinct center directions (after removing their
                       common mean). Random centers in high dimension give ~sqrt(2/(pi N)).
        participation  Participation ratio of the center covariance eigenvalues,
                       (sum s^2)^2 / sum s^4: an effective number of center dimensions, close to
                       P-1 for spread-out centers and close to 1 for collinear ones.
        n_95           Number of components needed for 95% of the center variance.
        spectrum       Normalised eigenvalue spectrum of the center covariance.
    """
    Xs = [np.asarray(X, dtype=np.float64) for X in Xs]
    centers = np.stack([X.mean(axis=1) for X in Xs], axis=1)  # (N, P)
    C = centers - centers.mean(axis=1, keepdims=True)

    s = np.linalg.svd(C, compute_uv=False)
    ev = s ** 2
    if ev.sum() <= 0:
        raise ValueError("all manifold centers are identical")
    ev = ev / ev.sum()

    norms = np.linalg.norm(C, axis=0)
    good = norms > 0
    U = C[:, good] / norms[good]
    gram = U.T @ U
    off = gram[~np.eye(gram.shape[0], dtype=bool)]

    return {
        "mean_abs_cos": float(np.mean(np.abs(off))) if off.size else 0.0,
        "participation": float(ev.sum() ** 2 / np.sum(ev ** 2)),
        "n_95": int(np.searchsorted(np.cumsum(ev), 0.95) + 1),
        "spectrum": ev,
    }
