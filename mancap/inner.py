"""The inner minimisation of the replica theory.

Chung, Lee and Sompolinsky (2018) reduce the P-manifold classification problem to a
single-manifold problem averaged over a Gaussian field. For one manifold, written in its own
(D+1)-dimensional frame as columns S (D shape coordinates plus a center coordinate equal to
1 for every point), and one Gaussian vector T,

    F(T) = min ||V - T||^2     subject to    S_j . V >= kappa   for every point j        (*)

and the capacity is alpha_M(kappa)^{-1} = <F(T)>_T with T ~ N(0, I_{D+1}). For isolated
points (D = 0) this gives F(T) = max(0, kappa - T)^2 and alpha = 2 at kappa = 0 (Cover 1965).

The constraint is imposed on the M sampled points and therefore on their convex hull:
interior points do not change F, points that extend the hull reduce capacity, and the
number of samples per manifold is a parameter whose effect has to be measured (see `synth`
and scripts/02_validate_balls_and_segments.py).

Dual form
---------
Writing V = T + S mu with mu >= 0, (*) becomes a nonnegative quadratic program

    min_{mu >= 0}   mu^T G mu + 2 mu^T b,      G = S^T S,   b = S^T T - kappa * 1          (**)

G depends only on the manifold and is computed once per manifold. The gradient
(G mu + b)_j = S_j . V - kappa is the slack of constraint j, so the KKT conditions read as
"active constraints are tight, inactive constraints are satisfied". Only b depends on T,
through one matrix product. The reference implementation solves the primal with cvxopt once
per draw.

Anchor point
------------
At the solution V = T + lambda s_tilde with lambda = sum(mu) and s_tilde = S mu / lambda, a
convex combination of the manifold points with mu_j > 0. Averages of the anchor's length and
orientation over draws give R_M and D_M.

Sign convention
---------------
This module uses S . V >= kappa; the reference implementation uses S . V <= -kappa. Since
F_ref(T) = F_ours(-T) and T is symmetric, capacity, R_M and D_M are unchanged; a
draw-by-draw comparison must negate T.

Solvers
-------
solve_active_set   Exact, finitely terminating active-set method on the dual (**).
solve_slsqp        scipy's SLSQP on the primal (*); an independent code path for
                   cross-checking.
solve_ball         Closed form for a D-dimensional ball of radius R, a second-order cone
                   projection.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class InnerSolution:
    """The outcome of one inner minimization.

    v         The minimizer V, shape (D+1,).
    f         The objective ||V - T||^2 at the solution.
    mu        Dual multipliers, shape (M,). Nonzero only on active constraints.
    anchor    The anchor point s_tilde, shape (D+1,). When the solution is interior (no
              constraint is active, V = T, f = 0) the anchor is not determined by the
              minimization; we then report the single manifold point with the smallest
              projection onto T, which is the point that would bind first. This matches the
              reference implementation's behaviour.
    interior  True when V = T, i.e. T already satisfies every constraint.
    n_active  Number of constraints with mu > 0.
    """

    v: np.ndarray
    f: float
    mu: np.ndarray
    anchor: np.ndarray
    interior: bool
    n_active: int


def _interior_anchor(S: np.ndarray, t: np.ndarray) -> np.ndarray:
    """The manifold point that would bind first: the smallest projection onto t.

    Only the D shape coordinates matter for the comparison, because every point shares the same
    center coordinate, so that term is a constant offset in the projection.
    """
    d = S.shape[0] - 1
    j = int(np.argmin(t[:d] @ S[:d, :])) if d > 0 else 0
    return S[:, j].copy()


def solve_active_set(
    S: np.ndarray,
    t: np.ndarray,
    kappa: float = 0.0,
    G: np.ndarray | None = None,
    tol: float = 1e-11,
    max_outer: int = 200,
) -> InnerSolution:
    """Exact active-set solution of the dual nonnegative QP (**).

    This is the Lawson-Hanson NNLS active-set scheme generalised from a least-squares objective
    to a Gram-matrix objective. It maintains a "free set" of indices allowed to be positive and
    solves the equality-constrained subproblem on that set exactly, adding the most violated
    constraint each outer iteration and dropping any index that would go negative. Because each
    outer iteration strictly decreases the objective and there are finitely many subsets, it
    terminates; in practice the free set never exceeds D+1 members, since that is the most
    linearly independent active constraints the space can support.

    Args:
        S: Manifold in its own frame, shape (D+1, M). Last row is the center coordinate.
        t: Gaussian vector, shape (D+1,).
        kappa: Margin.
        G: Optional precomputed Gram matrix S^T S, shape (M, M). Pass it in when looping over
           many t for the same manifold -- it is the expensive part and does not depend on t.
        tol: Violation below which a constraint counts as satisfied.
        max_outer: Guard against cycling on degenerate problems.

    Returns:
        InnerSolution.
    """
    S = np.asarray(S, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).ravel()
    d1, m = S.shape
    if G is None:
        G = S.T @ S
    b = S.T @ t - kappa

    # b_j = S_j . T - kappa is the slack at mu = 0. If every slack is nonnegative, T itself is
    # feasible and there is nothing to do.
    if np.all(b >= -tol):
        return InnerSolution(
            v=t.copy(), f=0.0, mu=np.zeros(m), anchor=_interior_anchor(S, t),
            interior=True, n_active=0,
        )

    mu = np.zeros(m)
    free = np.zeros(m, dtype=bool)

    for _ in range(max_outer):
        # Half-gradient of the dual objective is the constraint slack; a negative slack is a
        # violated constraint. w_j > 0 means "constraint j is violated by w_j".
        w = -(G @ mu + b)
        w[free] = -np.inf  # already free; never re-add
        j = int(np.argmax(w))
        if w[j] <= tol:
            break
        free[j] = True

        # Inner loop: solve on the current free set, dropping indices that want to go negative.
        for _ in range(m + 1):
            idx = np.flatnonzero(free)
            if idx.size == 0:
                break
            # Unconstrained minimum of the dual over the free set: G[P,P] z = -b[P].
            # lstsq rather than solve because manifold points are linearly dependent in general
            # (they are centered deviations, so rank <= M-1) and the submatrix can be singular.
            z, *_ = np.linalg.lstsq(G[np.ix_(idx, idx)], -b[idx], rcond=None)
            if np.all(z >= -tol):
                mu[:] = 0.0
                mu[idx] = np.maximum(z, 0.0)
                break
            # Move as far towards z as nonnegativity allows, then release whatever hit zero.
            cur = mu[idx]
            neg = z < -tol
            denom = cur - z
            with np.errstate(divide="ignore", invalid="ignore"):
                ratios = np.where(neg & (denom > 0), cur / denom, np.inf)
            step = float(np.min(ratios))
            if not np.isfinite(step):
                # Cannot make progress on this free set; drop the offending indices outright.
                free[idx[neg]] = False
                continue
            cur_new = cur + step * (z - cur)
            mu[:] = 0.0
            mu[idx] = np.maximum(cur_new, 0.0)
            free[idx[cur_new <= tol]] = False

    lam = float(mu.sum())
    v = t + S @ mu
    f = float(np.sum((v - t) ** 2))
    if lam <= tol:
        return InnerSolution(
            v=t.copy(), f=0.0, mu=np.zeros(m), anchor=_interior_anchor(S, t),
            interior=True, n_active=0,
        )
    anchor = (S @ mu) / lam
    return InnerSolution(
        v=v, f=f, mu=mu, anchor=anchor, interior=False, n_active=int(np.sum(mu > tol))
    )


def solve_slsqp(S: np.ndarray, t: np.ndarray, kappa: float = 0.0) -> InnerSolution:
    """Primal solution via scipy's SLSQP, for cross-checking.

    Works on the primal (*) with a sequential quadratic programming line search and does not
    form the Gram matrix, so it shares no algebra with solve_active_set. Slow. The multipliers
    it reports are not reliable, so the anchor is recovered geometrically from V - T.
    """
    from scipy.optimize import minimize

    S = np.asarray(S, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).ravel()
    m = S.shape[1]

    res = minimize(
        fun=lambda v: float(np.sum((v - t) ** 2)),
        x0=t.copy(),
        jac=lambda v: 2.0 * (v - t),
        constraints=[{
            "type": "ineq",
            "fun": lambda v: S.T @ v - kappa,
            "jac": lambda v: S.T,
        }],
        method="SLSQP",
        options={"maxiter": 500, "ftol": 1e-14},
    )
    v = res.x
    f = float(np.sum((v - t) ** 2))
    dv = v - t
    lam = float(np.linalg.norm(dv))
    if lam < 1e-9:
        return InnerSolution(
            v=t.copy(), f=0.0, mu=np.zeros(m), anchor=_interior_anchor(S, t),
            interior=True, n_active=0,
        )
    # V - T = lambda * s_tilde, and the anchor's center coordinate is 1 by construction, which
    # fixes the scale: lambda is the center component of V - T.
    scale = dv[-1]
    anchor = dv / scale if abs(scale) > 1e-12 else dv / lam
    active = S.T @ v - kappa < 1e-7
    return InnerSolution(
        v=v, f=f, mu=np.full(m, np.nan), anchor=anchor, interior=False,
        n_active=int(np.sum(active)),
    )


def solve_ball(t: np.ndarray, radius: float, kappa: float = 0.0) -> InnerSolution:
    """Closed form for a full D-dimensional ball manifold of radius R.

    For the ball S = {(s, 1) : ||s|| <= R}, the infinitely many constraints S . V >= kappa
    collapse to a single second-order cone constraint. Writing V = (v_s, v_c) with v_s the D
    shape components and v_c the center component, the worst point of the ball is the one
    pointing opposite to v_s, so

        min over the ball of S . V  =  v_c - R ||v_s||   >=  kappa

    Projecting T onto {(v_s, v_c) : v_c - R ||v_s|| >= kappa} is the projection onto a shifted,
    scaled second-order cone, which has three cases (already inside; project onto the boundary;
    collapse to the apex). This gives F(T) with no iteration at all, so it is a genuine analytic
    oracle for a manifold that is not a point.

    Args:
        t: Gaussian vector, shape (D+1,), last entry the center coordinate.
        radius: Ball radius, in units of the center norm.
        kappa: Margin.

    Returns:
        InnerSolution. The anchor is the ball point that binds, i.e. -R * unit(v_s) with center
        coordinate 1.
    """
    t = np.asarray(t, dtype=np.float64).ravel()
    ts, tc = t[:-1], float(t[-1])
    r = float(np.linalg.norm(ts))
    R = float(radius)

    def _anchor_from(direction: np.ndarray, norm: float) -> np.ndarray:
        a = np.zeros_like(t)
        if norm > 0:
            a[:-1] = -R * direction / norm
        a[-1] = 1.0
        return a

    if tc - R * r >= kappa:  # already feasible
        return InnerSolution(v=t.copy(), f=0.0, mu=np.zeros(0), anchor=_anchor_from(ts, r),
                             interior=True, n_active=0)

    # Shift so the cone has its apex at the origin: y = v_c - kappa must satisfy y >= R * ||v_s||.
    y_t = tc - kappa
    # Apex case: if the shifted point lies in the polar cone, the projection is the apex itself.
    if R * y_t + r <= 0:
        v = np.zeros_like(t)
        v[-1] = kappa
        f = float(np.sum((v - t) ** 2))
        return InnerSolution(v=v, f=f, mu=np.zeros(0), anchor=_anchor_from(ts, r),
                             interior=False, n_active=1)

    # Boundary case: project onto y = R * ||v_s||. Along the radial direction this reduces to a
    # 2D projection of (r, y_t) onto the line y = R x, giving x* = (r + R y_t) / (1 + R^2).
    x_star = (r + R * y_t) / (1.0 + R * R)
    y_star = R * x_star
    v = np.empty_like(t)
    v[:-1] = (x_star * ts / r) if r > 0 else 0.0
    v[-1] = y_star + kappa
    f = float(np.sum((v - t) ** 2))
    return InnerSolution(v=v, f=f, mu=np.zeros(0), anchor=_anchor_from(ts, r),
                         interior=False, n_active=1)
