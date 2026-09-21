"""Correlated manifold centers: measuring the violation, and correcting for it.

THE PROBLEM
-----------
The replica calculation treats the P manifold centers as being in general position -- effectively
random relative to one another, so that in high dimension they are nearly orthogonal. Real
representations do not comply. In a network layer, a few directions carry most of the
between-category variance, so the centers are concentrated in a low-dimensional subspace and are
strongly correlated. Capacity computed without accounting for this is biased, and the bias is not
small.

THE CORRECTION (Cohen, Chung, Lee & Sompolinsky, Nat. Commun. 2020)
--------------------------------------------------------------------
Find the K-dimensional subspace that carries the shared center structure, project it out of all
the manifold data, and run the ordinary analysis on what remains. "Carries the shared structure"
is made precise by a cost function: after removing the subspace, how correlated are the residual
centers? Writing X for the centers in an orthonormal basis of their own span, and V for an
orthonormal basis of the candidate subspace,

    residual Gram      A = X X^T - (XV)(XV)^T
    residual norms^2   c0 = diag(X X^T) - rowsum((XV)^2)
    cost(V)            = (1/2) sum_mn A_mn^2 / (c0_m c0_n)

which is the sum of SQUARED RESIDUAL CORRELATION COEFFICIENTS. Minimising it over the Stiefel
manifold {V : V^T V = I_K} gives the subspace whose removal leaves the centers as uncorrelated as
possible. K is then chosen as the rank that minimises the residual correlation.

WHAT IS DIFFERENT HERE FROM THE REFERENCE IMPLEMENTATION
--------------------------------------------------------
Same cost function, same criterion, two changes:

  * The gradient is computed in closed form at O(P^2 K + N P K) instead of by materialising a
    (P, P, N, K) tensor. For P = 100 centers in N = 99 dimensions at K = 10 the reference's
    tensor is ~80 MB per evaluation; the identity below needs none of it. Both reduce to

        dcost/dV = -2 X^T (PF1 @ c) + 2 X^T diag(PF2 @ c0) @ c
        PF1 = A / (c0 c0^T),   PF2 = A^2 / (c0 c0^T)^2

    and `check_gradient` verifies it against finite differences.
  * The Stiefel optimization is a self-contained Riemannian gradient descent with QR retraction
    and Armijo backtracking, so pymanopt (whose `pymanopt.solvers` API was removed years ago) is
    not needed.

HONEST STATUS OF THIS MODULE
----------------------------
The cost function, its gradient, and the optimizer are each tested directly. Whether applying
this correction makes the theory match reality on correlated data is a separate question, and it
is answered by direct simulation in `scripts/03_validate_correlated_centers.py`: build manifolds
with deliberately correlated centers, measure the empirical separability threshold, and compare
it against the corrected and uncorrected predictions. Do not trust the correction on real data
without looking at that figure first.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np


# ----------------------------------------------------------------------------------------------
# The cost function and its gradient
# ----------------------------------------------------------------------------------------------

def square_corrcoeff_cost(
    V: np.ndarray, X: np.ndarray, grad: bool = True
) -> tuple[float, np.ndarray | None]:
    """Sum of squared residual correlation coefficients after removing the subspace V.

    Args:
        V: (N, K) with V^T V = I_K. The candidate subspace.
        X: (P, N) centers expressed in an orthonormal basis of their own span.
        grad: Also return the Euclidean gradient with respect to V.

    Returns:
        (cost, gradient or None).

    Note on the constant: the diagonal terms are identically 1 (a residual is perfectly
    correlated with itself), so the cost can never fall below P/2. What matters is the excess
    over that floor, which is what `residual_correlation` reports instead.
    """
    X = np.asarray(X, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    P, N = X.shape
    if V.shape[0] != N:
        raise ValueError(f"V has {V.shape[0]} rows but X has {N} columns")

    C = X @ X.T                      # (P, P) Gram of the centers
    c = X @ V                        # (P, K) components inside the subspace
    c0 = np.diagonal(C).reshape(P, 1) - np.sum(c ** 2, axis=1, keepdims=True)  # residual norms^2
    A = C - c @ c.T                  # residual Gram
    B = c0 @ c0.T                    # outer product of residual norms^2
    cost = float(np.sum(A ** 2 / B) / 2.0)

    if not grad:
        return cost, None

    # dcost/dV, derived by differentiating A and c0 through V and collecting terms. Equivalent
    # to the reference's four-index sum, without building the (P, P, N, K) tensor.
    PF1 = A / B
    PF2 = (A ** 2) / (B ** 2)
    u = PF2 @ c0                     # (P, 1)
    gradient = -2.0 * (X.T @ (PF1 @ c)) + 2.0 * (X.T @ (u * c))
    return cost, gradient


def residual_correlation(V: np.ndarray | None, X: np.ndarray) -> float:
    """Mean absolute off-diagonal correlation of the residual centers.

    This is the interpretable number: 0 means the residual centers are uncorrelated, which is the
    regime the theory assumes. Pass V=None for the uncorrected baseline. It is the quantity used
    to select K, and the one worth reporting alongside any capacity from real data.
    """
    X = np.asarray(X, dtype=np.float64)
    P = X.shape[0]
    R = X if V is None else X - (X @ V) @ np.asarray(V).T
    n = np.linalg.norm(R, axis=1, keepdims=True)
    n = np.where(n > 0, n, 1.0)
    C = (R @ R.T) / (n @ n.T)
    return float((np.sum(np.abs(C)) - P) / (P * (P - 1)))


def check_gradient(
    X: np.ndarray, K: int, rng: np.random.Generator, eps: float = 1e-6
) -> float:
    """Largest absolute discrepancy between the analytic gradient and finite differences.

    The closed-form gradient is the one piece of this module that could be silently wrong: a
    mistaken factor would still produce a plausible subspace, just not the optimal one. Finite
    differences are the only check that does not share its algebra.
    """
    N = X.shape[1]
    V = np.linalg.qr(rng.standard_normal((N, K)))[0]
    _, g = square_corrcoeff_cost(V, X, grad=True)
    num = np.zeros_like(V)
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            Vp, Vm = V.copy(), V.copy()
            Vp[i, j] += eps
            Vm[i, j] -= eps
            cp, _ = square_corrcoeff_cost(Vp, X, grad=False)
            cm, _ = square_corrcoeff_cost(Vm, X, grad=False)
            num[i, j] = (cp - cm) / (2 * eps)
    return float(np.max(np.abs(num - g)))


# ----------------------------------------------------------------------------------------------
# Optimization on the Stiefel manifold
# ----------------------------------------------------------------------------------------------

def _retract(V: np.ndarray) -> np.ndarray:
    """QR retraction back onto the Stiefel manifold, with a positive-diagonal sign convention."""
    Q, R = np.linalg.qr(V)
    return Q * np.sign(np.where(np.diagonal(R) == 0, 1.0, np.diagonal(R)))


def stiefel_minimize(
    cost_grad: Callable[[np.ndarray], tuple[float, np.ndarray]],
    V0: np.ndarray,
    max_iter: int = 500,
    tol: float = 1e-8,
    max_backtrack: int = 40,
) -> tuple[np.ndarray, float, dict]:
    """Minimise a function over {V : V^T V = I} by Riemannian gradient descent.

    The Euclidean gradient is projected onto the tangent space at V,

        rgrad = G - V sym(V^T G),

    a step is taken along the negative Riemannian gradient, and the result is pulled back onto the
    manifold by a QR retraction. Step length comes from Armijo backtracking, which makes the
    routine robust without any tuning. The orthogonality constraint is maintained exactly at every
    iterate, not approximately, so the returned V can be used directly as a projector.

    Returns:
        (V, cost, info) with info carrying the iteration count, final Riemannian gradient norm,
        and whether the tolerance was met.
    """
    V = _retract(np.asarray(V0, dtype=np.float64))
    cost, G = cost_grad(V)
    step = 1.0
    it = 0
    gnorm = np.inf

    for it in range(1, max_iter + 1):
        sym = (V.T @ G + G.T @ V) / 2.0
        rgrad = G - V @ sym
        gnorm = float(np.linalg.norm(rgrad))
        if gnorm <= tol:
            break

        improved = False
        for _ in range(max_backtrack):
            V_new = _retract(V - step * rgrad)
            cost_new, G_new = cost_grad(V_new)
            # Armijo: accept a decrease proportional to the step and the squared gradient norm.
            if cost_new <= cost - 1e-4 * step * gnorm ** 2:
                V, cost, G = V_new, cost_new, G_new
                step *= 2.0  # grow, so a long flat stretch is not walked in tiny steps
                improved = True
                break
            step *= 0.5
        if not improved:
            break

    return V, cost, {"n_iter": it, "grad_norm": gnorm, "converged": bool(gnorm <= tol)}


# ----------------------------------------------------------------------------------------------
# Finding the center subspace
# ----------------------------------------------------------------------------------------------

@dataclass
class CenterSubspace:
    """The subspace carrying shared center structure, and the evidence for choosing it.

    basis        (N, K) orthonormal basis in the AMBIENT space, ready to project out.
    K            Chosen rank.
    residual     Mean absolute off-diagonal residual correlation at the chosen K.
    baseline     The same quantity before any correction. The ratio residual/baseline is how
                 much of the center correlation the projection actually removed.
    by_K         Residual correlation for every K tried, so the choice of K is visible as a curve
                 rather than asserted. A flat curve means no low-rank structure exists and the
                 correction is doing nothing useful.
    K_argmin     The rank that minimises residual correlation outright. Reported alongside K
                 because they differ, and the difference is the point -- see `find_center_subspace`.
    reduction    (baseline - min(by_K)) / baseline: the fraction of center correlation that the
                 best subspace removes. Small values mean there is no low-rank structure to find.
    """

    basis: np.ndarray
    K: int
    residual: float
    baseline: float
    by_K: np.ndarray
    K_argmin: int = 0
    reduction: float = 0.0
    span: np.ndarray = field(repr=False, default=None)


def null_reduction(
    P: int,
    N: int,
    n_null: int = 3,
    max_K: int | None = None,
    n_restarts: int = 2,
    rng: np.random.Generator | None = None,
) -> float:
    """Reduction in center correlation achievable on ISOTROPIC centers: the chance level.

    Fitting a K-dimensional subspace to P centers always removes some correlation, whether or not
    any is there -- with P = 60 centers in 59 dimensions the search removes about 10% of the
    baseline from purely isotropic centers. A fixed threshold cannot distinguish that from real
    structure, because the chance level depends on P, N and the ranks tried.

    This runs the identical search on matched isotropic centers and returns the largest reduction
    it manages. Use it as the threshold: only reductions above the null are evidence of structure.

    Returns:
        The maximum reduction over `n_null` isotropic draws.
    """
    rng = np.random.default_rng() if rng is None else rng
    out = []
    for _ in range(n_null):
        C = rng.standard_normal((N, P))
        C /= np.linalg.norm(C, axis=0, keepdims=True)
        fake = [C[:, [p]] for p in range(P)]
        r = find_center_subspace(
            fake, max_K=max_K, n_restarts=n_restarts, min_reduction=0.0, rng=rng
        )
        out.append(r.reduction)
    return float(max(out))


def find_center_subspace(
    Xs: Sequence[np.ndarray],
    max_K: int | None = None,
    n_restarts: int = 3,
    max_iter: int = 500,
    k_tol: float = 0.05,
    min_reduction: float = 0.10,
    n_null: int = 0,
    rng: np.random.Generator | None = None,
    verbose: bool = False,
) -> CenterSubspace:
    """Locate the low-rank structure shared by the manifold centers.

    CHOOSING K IS WHERE THIS GOES WRONG, AND THE OBVIOUS RULE IS THE WRONG ONE
    --------------------------------------------------------------------------
    Residual correlation decreases almost monotonically in K -- there is always one more
    direction whose removal helps a little -- so taking the outright argmin (which is what the
    reference implementation does) systematically over-estimates the rank. Measured on data with
    a PLANTED rank-3 component and P = 60 manifolds, argmin selects K = 8-10, and at zero planted
    correlation it still selects 10. That is not free: every removed direction is one the
    classifier could have used, so over-projection biases capacity DOWNWARD. In the validation
    run it made the corrected prediction undershoot the simulated capacity at every correlation
    strength.

    Two guards, both on by default:

      k_tol         Take the SMALLEST K that gets within `k_tol` of the best achievable
                    reduction, rather than the K that achieves it. This finds the elbow of the
                    curve, which is where the planted rank actually sits.
      min_reduction If the best subspace removes less than this fraction of the baseline
                    correlation, there is no low-rank structure worth removing, and an EMPTY
                    subspace is returned (K = 0, the correction becomes a no-op). Without this,
                    the routine invents structure in isotropic centers and the correction costs
                    capacity for nothing.

    The default min_reduction of 0.10 is a heuristic, and a weak one: the chance-level reduction
    depends on P, N and the ranks tried, not on a universal constant. With P = 60 centers the
    search removes about 10% of the baseline from purely isotropic centers, right at the
    threshold. Passing `n_null > 0` replaces the heuristic with a measured null -- the same search
    run on matched isotropic centers -- which is the defensible version and what real data
    deserves. It costs `n_null` extra searches.

    Set k_tol=0 and min_reduction=0 to reproduce the reference's argmin behaviour.

    Args:
        Xs: The manifolds, each (N, M_i). Only their centers are used.
        max_K: Largest rank to try. Defaults to the number of components holding 95% of the
            center variance, plus a margin, capped at P-2.
        n_restarts: Random restarts per K. The cost is not convex on the Stiefel manifold, so the
            best of several runs is kept.
        max_iter: Iterations per restart.
        k_tol: Elbow tolerance, as a fraction of the best achievable reduction.
        min_reduction: Below this fractional reduction, return no subspace at all.
        n_null: If > 0, measure the chance-level reduction on matched isotropic centers and use
            it in place of `min_reduction`. Slower, and correct.
        rng: Random generator.
        verbose: Print the residual correlation at each K.

    Returns:
        CenterSubspace. Always inspect `by_K`, `K_argmin` and `reduction` before trusting a
        corrected capacity: they say whether the structure being removed is real.
    """
    rng = np.random.default_rng() if rng is None else rng
    Xs = [np.asarray(X, dtype=np.float64) for X in Xs]
    centers = np.stack([X.mean(axis=1) for X in Xs], axis=1)  # (N, P)
    P = centers.shape[1]
    if P < 4:
        raise ValueError("center-correlation analysis needs at least 4 manifolds")

    # Express the centers in an orthonormal basis of their own span, which has dimension at most
    # P-1 after mean subtraction. All the optimization happens in that small space.
    Xb = (centers - centers.mean(axis=1, keepdims=True)).T  # (P, N)
    q, _ = np.linalg.qr(Xb.T)                               # (N, min(N, P))
    span = q[:, : P - 1]                                    # (N, P-1)
    X = Xb @ span                                           # (P, P-1)

    baseline = residual_correlation(None, X)

    if max_K is None:
        sv = np.linalg.svd(X, compute_uv=False)
        ev = sv ** 2 / np.sum(sv ** 2)
        n95 = int(np.searchsorted(np.cumsum(ev), 0.95) + 1)
        max_K = int(min(max(n95 + 5, 2), P - 2))

    best_by_K, V_by_K = [], []
    for K in range(1, max_K + 1):
        best_V, best_res = None, np.inf
        for _ in range(n_restarts):
            V0 = np.linalg.qr(rng.standard_normal((P - 1, K)))[0]
            V, _, _ = stiefel_minimize(
                lambda V: square_corrcoeff_cost(V, X, grad=True), V0, max_iter=max_iter
            )
            res = residual_correlation(V, X)
            if res < best_res:
                best_res, best_V = res, V
        best_by_K.append(best_res)
        V_by_K.append(best_V)
        if verbose:
            print(f"  K={K:>3}  residual correlation {best_res:.5f}  (baseline {baseline:.5f})")
        # Stop once three consecutive ranks have failed to improve on the running best.
        if K > 4 and min(best_by_K[-3:]) > min(best_by_K[:-3]):
            break

    by_K = np.array(best_by_K)
    K_argmin = int(np.argmin(by_K)) + 1
    best = float(by_K.min())
    reduction = (baseline - best) / baseline if baseline > 0 else 0.0

    if n_null > 0:
        min_reduction = null_reduction(
            P=centers.shape[1], N=centers.shape[0], n_null=n_null, max_K=max_K,
            n_restarts=max(1, n_restarts - 1), rng=rng,
        )
        if verbose:
            print(f"  null reduction (chance level): {min_reduction:.4f}")

    # Floor the threshold at zero. A negative `reduction` means the best subspace left the
    # centres MORE correlated than they started -- which happens on isotropic data, where the
    # residual-correlation curve rises with K -- and must always abstain, whatever a noisy null
    # happened to return.
    if reduction <= max(min_reduction, 0.0):
        # No low-rank structure worth removing. Returning an empty basis makes the projection a
        # no-op, which is the honest answer, rather than removing directions to chase noise.
        return CenterSubspace(
            basis=np.zeros((centers.shape[0], 0)), K=0, residual=baseline, baseline=baseline,
            by_K=by_K, K_argmin=K_argmin, reduction=float(reduction), span=span,
        )

    # Elbow: the smallest K that gets within k_tol of the best achievable reduction.
    threshold = best + k_tol * (baseline - best)
    K_best = int(np.argmax(by_K <= threshold)) + 1

    return CenterSubspace(
        basis=span @ V_by_K[K_best - 1],
        K=K_best,
        residual=float(by_K[K_best - 1]),
        baseline=baseline,
        by_K=by_K,
        K_argmin=K_argmin,
        reduction=float(reduction),
        span=span,
    )
