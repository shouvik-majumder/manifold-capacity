"""Turning raw activations into the (D+1)-dimensional frames the theory expects.

WHY A FRAME CHANGE IS NEEDED AT ALL
-----------------------------------
The replica result is stated for a manifold described *in its own coordinates*: D shape
directions plus one center direction, with the center coordinate equal to 1 for every point.
Raw data does not arrive that way. It arrives as P clouds of activation vectors in R^N, all
sharing whatever global offset the network happens to have. `build_frames` performs the change
of coordinates, and every choice it makes is a modelling decision worth stating explicitly.

THE FOUR STEPS, AND WHAT EACH ASSUMES
-------------------------------------
1. Subtract the global mean of all samples from all manifolds.
   Assumption: the origin of the classifier's coordinate system is the grand mean, i.e. the
   readout has no bias term of its own. For neural data with a large shared baseline this step
   is doing real work -- without it, every manifold sits far from the origin in the same
   direction and capacity is dominated by that shared offset rather than by the categories.

2. Per manifold, compute its center c and the shape offsets X - c.

3. Divide the offsets by ||c||.
   This is what makes R_M dimensionless: a radius of 0.3 means "the manifold's extent is 30% of
   its distance from the origin". It also means capacity depends on the RATIO of extent to
   center norm, not on either separately, which is why rescaling all activations changes
   nothing -- a useful invariance to test.

   A consequence worth knowing before comparing numbers to a generator's nominal radius: step 1
   SHRINKS the center norms, so step 3 INFLATES every radius. For P manifolds with isotropic
   unit-norm centers, the grand mean has squared norm about 1/P and each center loses about the
   same, leaving ||c|| ~ sqrt(1 - 1/P), so radii come out roughly a factor 1/sqrt(1 - 1/P)
   too large -- about 7% at P = 8, under 1% at P = 60. This is correct behaviour, not bias: the
   classifier really does see the manifolds relative to the grand mean. But it means a test that
   builds radius-0.1 balls and asserts R_M = 0.1 will fail for small P, and the failure is the
   test's fault. With few manifolds, or when the nominal radius is the quantity of interest,
   pass subtract_global_mean=False.

4. Reduce the shape dimension to the manifold's actual RANK, then append the row of ones.
   The offsets of M points are centered, so they span at most M-1 dimensions no matter how
   large N is. Keeping N rows would leave the frame padded with directions the manifold cannot
   reach. Restricting to the column space removes them without changing any geometry.

   This step is where mancap deliberately DIFFERS from the reference implementation, which
   reduces to M rows with a thin QR. For M sample points the offsets have rank at most M-1, so
   reducing to M always leaves at least one direction in which the manifold has zero extent, and
   that padding is not harmless:

       alpha  unaffected -- the padded row of S is zero, so that coordinate of V is
              unconstrained, V = T there, and it contributes nothing to F.
       R_M    unaffected -- anchor deviations are zero in the padded direction.
       D_M    CHANGED    -- D_M = D * (mean cos)^2 uses the full shape-space norms of both the
              Gaussian vector and the anchor. Padding raises D while diluting the cosine, and
              the dilution wins.

   Measured on a segment (rank 1), which must have D_M = 1 exactly: the rank-reduced frame gives
   1.000000, while a frame padded by one direction gives 0.8115 -- matching the predicted
   2*(2/pi)^2 = 0.8106 for a rank-1 manifold seen in a 2-dimensional shape space. The bias is
   worst when M is small relative to the true dimension, which is precisely the regime real data
   lands in. Pass reduce_dim="qr" to reproduce the reference's behaviour exactly.

WHAT THIS MODULE DELIBERATELY DOES NOT DO
-----------------------------------------
It does not correct for correlations between manifold CENTERS. The theory assumes the centers
are in general position; real representations badly violate this (a handful of directions carry
most of the between-category variance). The reference implementation handles it with a Stiefel
manifold optimization that finds the low-rank center structure and projects it out. That is a
separate, heavier piece of machinery and it belongs in its own module with its own validation.
Until it exists, capacity numbers computed from real data with strongly correlated centers are
biased, and `center_correlation` below is provided so that the size of the problem is visible
rather than assumed away. On synthetic data with random centers -- which is what the validation
suite uses -- the correction is unnecessary by construction.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.linalg import qr


@dataclass
class FrameSet:
    """Manifolds in their own (D+1)-frames, plus the diagnostics needed to trust them.

    frames        List of (D_i+1, M_i) arrays, ready for `capacity.analyze_manifold`.
    center_norms  ||c_i|| for each manifold, after global-mean subtraction. Wildly unequal
                  norms mean the manifolds live at very different distances from the origin, so
                  a single capacity number is averaging over heterogeneous geometry.
    raw_radii     The plain geometric radius sqrt(mean ||x - c||^2) / ||c|| of each cloud. This
                  is the naive quantity R_M is often confused with; reporting both makes the
                  difference between "how big is the cloud" and "how big is the part the
                  classifier sees" visible.
    ranks         Numerical rank of each cloud's offsets, i.e. the true shape dimension D_i.
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
            True unless you have already centered the data and know why. Note that it inflates
            radii by 1/sqrt(1 - 1/P); see the module docstring.
        reduce_dim: How to restrict each frame to the directions the manifold can actually reach.
            True (default) reduces to the numerical RANK of the offsets via SVD, which is the
            only choice that makes D_M correct. "qr" reproduces the reference implementation's
            thin QR to M rows, which leaves at least one unreachable direction and biases D_M
            downward; provided for exact comparison against published numbers. False keeps all N
            rows, which biases D_M far more and is only useful for testing invariances.
        rank_tol: Absolute singular-value threshold for the rank. Default follows numpy's
            matrix_rank convention, max(shape) * eps * largest singular value.
        center_subspace: Optional (N, K) orthonormal basis, from
            `centers.find_center_subspace`, projected out of every manifold before the frame is
            built. This is the correlated-centers correction: it removes the directions carrying
            shared structure between category centers, which the theory assumes absent. Leaving
            it None is correct for synthetic data with random centers and WRONG for real
            representations; `center_correlation` below says which situation you are in.

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
    """How badly the manifold centers violate the theory's general-position assumption.

    The replica calculation treats the P manifold centers as effectively random relative to one
    another. Real representations do not comply: a few directions carry most of the
    between-category variance, so the centers are concentrated in a low-dimensional subspace.
    When that happens, an uncorrected capacity estimate is biased, and this function measures by
    how much rather than leaving it implicit.

    Returns a dict with:
        mean_abs_cos   Mean |cosine| between distinct center directions (after removing their
                       common mean). Random centers in high dimension give ~sqrt(2/(pi N));
                       anything much larger signals shared structure.
        participation  Participation ratio of the center covariance eigenvalues,
                       (sum s^2)^2 / sum s^4. This is an effective number of center dimensions:
                       close to P-1 means the centers are spread out as the theory assumes,
                       close to 1 means they all lie along one direction.
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
