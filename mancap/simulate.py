"""Direct simulation of the separability threshold.

The other checks in this package compare implementations of the replica formulas with each
other or with closed forms derived from them, and would not detect an error shared by all
of them (a misplaced factor, a wrong margin normalisation). This module does not use the
theory: it builds P manifolds in R^N, assigns random binary labels, and searches for a
separating hyperplane. Sweeping the load alpha = P/N locates the empirical threshold, which
is then compared with the prediction of `capacity.analyze_manifold`.

Margin
------
With z_i = y_i x_i (the label folded into the point), the largest margin a unit-norm readout
can achieve is

    kappa* = max_{||w|| <= 1} min_i <w, z_i>  =  min_{lambda in simplex} || Z lambda ||       (+)

by minimax duality: the distance from the origin to the convex hull of the labelled points.
One geometric computation per dataset gives the margin exactly, a single sweep over P yields
the whole curve alpha(kappa), and kappa* = 0 exactly when the origin lies inside the hull.
(+) is solved by Gilbert's algorithm (Frank-Wolfe with exact line search on the simplex)
with pairwise steps, and cross-checked against SLSQP. For kappa = 0, `is_separable` uses an
exact linear-programming feasibility test.

Finite size
-----------
The replica prediction holds for N, P -> infinity at fixed P/N. At finite N the transition
is smeared over a window of width ~1/sqrt(N), so `threshold_scan` returns the
separable-fraction curve at each N and `extrapolate_threshold` fits the 1/N trend.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


# ----------------------------------------------------------------------------------------------
# The margin of a labelled dataset
# ----------------------------------------------------------------------------------------------

def labelled_points(Xs: Sequence[np.ndarray], labels: np.ndarray) -> np.ndarray:
    """Fold manifold labels into the points: Z = [y_p * x for x in manifold p]. Shape (N, total).

    After this, separability of the manifolds by a readout w is the single statement
    <w, z_i> > 0 for all i, and the manifold structure survives only through which points are
    present -- which is all the margin depends on.
    """
    if len(Xs) != len(labels):
        raise ValueError("one label per manifold required")
    return np.concatenate([X * y for X, y in zip(Xs, labels)], axis=1)


def is_separable(Z: np.ndarray, tol_scale: float = 1.0) -> bool:
    """Exact test for whether a unit-norm readout can put every labelled point on the positive
    side, by linear programming.

    Separability with *some* positive margin is scale free: if <w, z_i> > 0 for all i then w can
    be rescaled so that <w, z_i> >= 1 for all i. So the question is the feasibility of the linear
    system Z^T w >= 1, which HiGHS decides exactly rather than approximately. This is the
    authority for the kappa = 0 threshold.

    Args:
        Z: Labelled points, shape (N, total).
        tol_scale: Right-hand side of the inequalities. Any positive value gives the same answer;
            exposed only so that badly scaled data can be conditioned.

    Returns:
        True when a separating readout exists.
    """
    from scipy.optimize import linprog

    n, total = Z.shape
    res = linprog(
        c=np.zeros(n),
        A_ub=-Z.T,
        b_ub=-np.full(total, float(tol_scale)),
        bounds=[(None, None)] * n,
        method="highs",
    )
    # status 0 = optimal (hence feasible), 2 = infeasible, 3 = unbounded (cannot occur with c=0).
    return bool(res.status == 0)


@dataclass
class MarginResult:
    """kappa* with its certificate.

    kappa_star  The largest margin a unit-norm readout achieves, as an upper bound: every
                iterate is a point of the convex hull, so ||p|| can only overestimate the
                distance to it. 0 when inseparable.
    kappa_lower A lower bound from the duality gap. The gap bounds the suboptimality of the
                squared objective f = kappa^2/2, so kappa*_true >= sqrt(max(0, kappa_star^2 -
                2*gap)). A positive kappa_lower proves separability.
    w           The optimal readout (unit norm), or zeros when inseparable.
    gap         Frank-Wolfe duality gap at termination, in units of the squared objective
                0.5*||p||^2.
    n_iter      Iterations used. Hitting the cap with a large gap means the answer is an upper
                bound only.
    converged   gap <= tol.
    """

    kappa_star: float
    kappa_lower: float
    w: np.ndarray
    gap: float
    n_iter: int
    converged: bool


def _best_step(p: np.ndarray, d: np.ndarray, gamma_max: float) -> tuple[float, float]:
    """Exact line search for min ||p + gamma d||^2 over gamma in [0, gamma_max].

    Returns (gamma, decrease), where decrease is the reduction in 0.5*||p||^2. Returning the
    decrease lets the caller choose between candidate directions on the basis of actual progress
    rather than a heuristic.
    """
    dd = float(d @ d)
    if dd <= 0 or gamma_max <= 0:
        return 0.0, 0.0
    pd = float(p @ d)
    gamma = min(max(-pd / dd, 0.0), gamma_max)
    decrease = -(gamma * pd + 0.5 * gamma * gamma * dd)
    return gamma, decrease


def max_margin(
    Z: np.ndarray,
    max_iter: int = 20000,
    tol: float = 1e-12,
) -> MarginResult:
    """kappa*, the largest margin any unit-norm readout achieves, via identity (+).

    Minimises ||Z lambda|| over the simplex. Two candidate directions are considered at every
    iteration and the one that decreases the objective more is taken:

      TOWARD step (Gilbert / Frank-Wolfe): move from the current hull point towards the vertex
        with the most negative gradient. Converges as O(1/k).

      PAIRWISE step (Mitchell-Dem'yanov-Malozemov): move weight directly from the worst vertex
        in the support to the best vertex outside it. This can remove a vertex from the
        support, and gives linear convergence.

    Pure Gilbert steps stall at a gap around 1e-5 on ordinary random instances, which leaves
    kappa* visibly above zero on inseparable data; with pairwise steps the gap reaches machine
    precision and the sign test agrees with the exact LP.

    Args:
        Z: Labelled points, shape (N, total).
        max_iter: Iteration cap; each iteration is one matrix-vector product.
        tol: Convergence tolerance, relative to the objective scale ||p||^2 (with an absolute
            floor of 1 so that near-inseparable data, where ||p|| -> 0, is held to an absolute
            standard). The gap is a difference of quantities of size ||p||^2, so its
            floating-point floor is eps * ||p||^2; an absolute tolerance below that could never
            be met.

    Returns:
        MarginResult, carrying both bounds on kappa* and the certificate.
    """
    Z = np.asarray(Z, dtype=np.float64)
    n, total = Z.shape

    # Start at the vertex closest to the origin.
    sq = np.sum(Z ** 2, axis=0)
    j0 = int(np.argmin(sq))
    lam = np.zeros(total)
    lam[j0] = 1.0
    p = Z[:, j0].copy()  # p = Z lambda, the current hull point

    gap = float("inf")
    thresh = tol
    it = 0
    for it in range(1, max_iter + 1):
        g = Z.T @ p  # gradient of 0.5*||p||^2 with respect to lambda
        j_to = int(np.argmin(g))
        pp = float(p @ p)
        # Frank-Wolfe gap: <p, p - z_j> = <p,p> - g_j. Rigorous bound on suboptimality.
        gap = pp - float(g[j_to])
        thresh = tol * max(1.0, pp)
        if gap <= thresh:
            break

        cand = []
        d_to = Z[:, j_to] - p
        gamma, dec = _best_step(p, d_to, 1.0)
        cand.append(("toward", j_to, -1, d_to, gamma, dec))

        support = np.flatnonzero(lam > 0.0)
        if support.size > 1:
            j_aw = int(support[np.argmax(g[support])])
            d_pw = Z[:, j_to] - Z[:, j_aw]
            gamma, dec = _best_step(p, d_pw, float(lam[j_aw]))
            cand.append(("pair", j_to, j_aw, d_pw, gamma, dec))

        kind, j_t, j_a, d, gamma, dec = max(cand, key=lambda c: c[5])
        if gamma <= 0.0 or dec <= 0.0:
            break

        if kind == "toward":
            lam *= 1.0 - gamma
            lam[j_t] += gamma
        else:
            lam[j_t] += gamma
            lam[j_a] -= gamma
            if lam[j_a] <= 0.0:
                lam[j_a] = 0.0
        p = p + gamma * d

    kappa_star = float(np.linalg.norm(p))
    kappa_lower = float(np.sqrt(max(0.0, kappa_star ** 2 - 2.0 * max(gap, 0.0))))
    w = p / kappa_star if kappa_star > 0 else np.zeros(n)
    return MarginResult(
        kappa_star=kappa_star, kappa_lower=kappa_lower, w=w, gap=gap, n_iter=it,
        converged=bool(gap <= thresh),
    )


def max_margin_qp(Z: np.ndarray) -> float:
    """kappa* by a general-purpose optimizer on the simplex, as an independent check.

    Solves the same problem (+) as `max_margin`, with SLSQP on the smooth objective
    0.5*||Z lambda||^2 under the simplex constraints.

    The primal form, maximising min_i <w, z_i> over the unit ball, is not used because it is a
    maximin of linear functions and non-differentiable at the optimum; SLSQP stalls short of
    the solution there by around 1e-2. The simplex form has no such kink.
    """
    from scipy.optimize import minimize

    Z = np.asarray(Z, dtype=np.float64)
    total = Z.shape[1]
    G = Z.T @ Z
    lam0 = np.full(total, 1.0 / total)

    res = minimize(
        fun=lambda lam: 0.5 * float(lam @ G @ lam),
        x0=lam0,
        jac=lambda lam: G @ lam,
        bounds=[(0.0, 1.0)] * total,
        constraints=[{
            "type": "eq",
            "fun": lambda lam: float(np.sum(lam) - 1.0),
            "jac": lambda lam: np.ones_like(lam),
        }],
        method="SLSQP",
        options={"maxiter": 5000, "ftol": 1e-16},
    )
    return float(np.sqrt(max(2.0 * res.fun, 0.0)))


# ----------------------------------------------------------------------------------------------
# Sweeping the load to find the threshold
# ----------------------------------------------------------------------------------------------

@dataclass
class ThresholdScan:
    """Result of scanning the load at one ambient dimension.

    N          Ambient dimension.
    alphas     Loads P/N tested.
    P          Number of manifolds at each load.
    frac_sep   Fraction of random-label draws that were separable at kappa = 0.
    margin     Mean kappa* at each load (0 once inseparable).
    margin_sd  Standard deviation of kappa* across draws.
    """

    N: int
    alphas: np.ndarray
    P: np.ndarray
    frac_sep: np.ndarray
    margin: np.ndarray
    margin_sd: np.ndarray


def threshold_scan(
    make_manifolds: Callable[[int, int, np.random.Generator], Sequence[np.ndarray]],
    N: int,
    alphas: Sequence[float],
    n_seeds: int = 20,
    rng: np.random.Generator | None = None,
    compute_margin: bool = True,
) -> ThresholdScan:
    """Measure separability as a function of load for one ambient dimension.

    Args:
        make_manifolds: Callable (P, N, rng) -> list of (N, M) arrays. Pass a lambda closing over
            the geometry you want, e.g. `lambda P, N, r: synth.balls(P, N, D=5, radius=0.3,
            M=50, rng=r)`.
        N: Ambient dimension.
        alphas: Loads P/N to test. P is rounded to the nearest integer, minimum 1.
        n_seeds: Independent draws of both the manifolds and the labels at each load. The
            separable fraction is a binomial estimate, so its standard error is
            sqrt(f(1-f)/n_seeds) -- with 20 seeds that is at best 0.11, which is enough to
            locate a crossing but not to resolve a few percent. Raise it for final figures.
        rng: Random generator.
        compute_margin: Also compute kappa* (needed for capacity at nonzero margin; costs more).

    Returns:
        ThresholdScan.
    """
    rng = np.random.default_rng() if rng is None else rng
    alphas = np.asarray(alphas, dtype=np.float64)
    Ps = np.maximum(1, np.round(alphas * N).astype(int))

    frac, mean_m, sd_m = [], [], []
    for P in Ps:
        sep, margins = [], []
        for _ in range(n_seeds):
            Xs = make_manifolds(int(P), N, rng)
            labels = rng.choice([-1.0, 1.0], size=len(Xs))
            Z = labelled_points(Xs, labels)
            sep.append(is_separable(Z))
            if compute_margin:
                margins.append(max_margin(Z).kappa_star)
        frac.append(float(np.mean(sep)))
        mean_m.append(float(np.mean(margins)) if margins else np.nan)
        sd_m.append(float(np.std(margins)) if margins else np.nan)

    return ThresholdScan(
        N=N,
        alphas=alphas,
        P=Ps,
        frac_sep=np.array(frac),
        margin=np.array(mean_m),
        margin_sd=np.array(sd_m),
    )


def crossing(alphas: np.ndarray, values: np.ndarray, level: float = 0.5) -> float:
    """Load at which a monotonically decreasing curve crosses `level`, by linear interpolation.

    Used on `frac_sep` with level 0.5 to locate the kappa = 0 threshold, and on `margin` with
    level kappa to locate the threshold at a nonzero margin. Returns NaN when the curve does
    not cross within the scanned range.

    A curve that lands exactly on the level counts as having crossed there; the comparison is
    strict (`> level`) so that a scan ending precisely at the threshold reports it.
    """
    alphas = np.asarray(alphas, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    above = values > level
    if above.all() or (~above).all():
        return float("nan")
    i = int(np.argmax(~above))  # first index below the level
    if i == 0:
        return float("nan")
    a0, a1 = alphas[i - 1], alphas[i]
    v0, v1 = values[i - 1], values[i]
    if v0 == v1:
        return float(0.5 * (a0 + a1))
    return float(a0 + (v0 - level) * (a1 - a0) / (v0 - v1))


def extrapolate_threshold(Ns: Sequence[int], thresholds: Sequence[float]) -> dict[str, float]:
    """Extrapolate finite-N thresholds to N -> infinity by a linear fit in 1/N.

    The leading finite-size correction to a perceptron-style capacity is O(1/N), so the
    measured threshold against 1/N should be a straight line whose intercept is the
    thermodynamic value. A large slope means the scanned N were too small for the intercept to
    be reliable.

    Returns:
        dict with `intercept` (the extrapolated threshold), `slope`, and `resid` (RMS residual).
    """
    x = 1.0 / np.asarray(Ns, dtype=np.float64)
    y = np.asarray(thresholds, dtype=np.float64)
    good = np.isfinite(y)
    if good.sum() < 2:
        return {"intercept": float("nan"), "slope": float("nan"), "resid": float("nan")}
    A = np.vstack([np.ones(good.sum()), x[good]]).T
    coef, *_ = np.linalg.lstsq(A, y[good], rcond=None)
    pred = A @ coef
    return {
        "intercept": float(coef[0]),
        "slope": float(coef[1]),
        "resid": float(np.sqrt(np.mean((y[good] - pred) ** 2))),
    }
