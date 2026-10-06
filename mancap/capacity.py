"""Manifold capacity, anchor radius and anchor dimension from the inner minimisation.

alpha_M  Capacity: the largest load P/N at which a single linear readout separates the
         manifolds under random binary labels with every point on the correct side of the
         margin kappa. Isolated points give alpha = 2 at kappa = 0 (Cover 1965).

R_M      Anchor radius: the spread of the anchor points (the support vectors selected by the
         Gaussian directions) about their mean, in units of the center norm. This differs
         from the geometric radius of the point cloud.

D_M      Anchor dimension: the effective number of shape dimensions spanned by the anchors.
         Bounded by the shape dimension D and usually well below it.

In the small-radius regime alpha_M ~ alpha_0(kappa + R_M sqrt(D_M)), so R_M and D_M
decompose the capacity.

Equations: capacity, Eqs. 16-17; R_M, Eq. 28; D_M, Eq. 29 of Chung, Lee and Sompolinsky
(2018).

F is taken from the solver objective ||V - T||^2 and also recomputed from the anchor,

    lambda = max(kappa - T . s_tilde, 0) / ||s_tilde||^2,      F = lambda^2 ||s_tilde||^2.

The two agree at the solution; `analyze_manifold` reports their discrepancy as a
consistency check on the anchors, which R_M and D_M depend on.

Capacities combine through their reciprocals: alpha^{-1} is a Gaussian mean of F, so

    alpha_total = 1 / mean_over_manifolds( 1 / alpha_mu ).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .inner import solve_active_set, solve_ball


@dataclass
class ManifoldResult:
    """Per-manifold capacity and geometry.

    alpha      Capacity of this manifold alone (reciprocal of its mean F).
    radius     Anchor radius R_M.
    dimension  Anchor dimension D_M.
    D          Ambient shape dimension of the frame (the "D" in D+1), for context on D_M.
    f_mean     Mean of F over Gaussian draws. This, not alpha, is the additive quantity.
    f_sem      Standard error of f_mean over draws, so the Monte Carlo noise in alpha is
               visible rather than implied. alpha's relative error equals f_mean's.
    frac_interior  Fraction of draws for which T was already feasible (F = 0). If this is near
               1 the manifold is tiny and alpha is being estimated from a handful of draws, so
               n_t needs raising; if it is near 0 the manifold is large.
    consistency    Max absolute difference between F from the solver objective and F recomputed
               through the anchor. Should be ~1e-12. A large value means the anchors are wrong
               even if alpha looks plausible.
    n_active_mean  Mean number of active constraints, i.e. how many sample points combine to
               form a typical anchor. 1 means anchors are always vertices of the hull; higher
               means anchors are genuinely virtual points on faces.
    """

    alpha: float
    radius: float
    dimension: float
    D: int
    f_mean: float
    f_sem: float
    frac_interior: float
    consistency: float
    n_active_mean: float
    anchors: np.ndarray | None = field(default=None, repr=False)


def analyze_manifold(
    sD1: np.ndarray,
    kappa: float = 0.0,
    n_t: int = 200,
    rng: np.random.Generator | None = None,
    t_vec: np.ndarray | None = None,
    keep_anchors: bool = False,
) -> ManifoldResult:
    """Capacity, radius and dimension of a single manifold given in its own (D+1)-frame.

    Args:
        sD1: Shape (D+1, M). Columns are manifold points in the frame built by
            `mancap.frames`: the first D rows are shape coordinates in units of the center
            norm, and the last row is the center coordinate, equal to 1 for every point.
        kappa: Margin. 0 reproduces the classic capacity.
        n_t: Number of Gaussian draws. The Monte Carlo error on alpha falls as 1/sqrt(n_t);
            `f_sem` in the result reports it, so this does not have to be guessed.
        rng: Random generator. Pass one for reproducibility.
        t_vec: Optional explicit Gaussian vectors, shape (D+1, n_t). Supplying the same ones to
            two implementations makes their comparison exact rather than statistical, which is
            how the reference cross-check is done.
        keep_anchors: Also return the raw anchor matrix, for plotting the anchor distribution.

    Returns:
        ManifoldResult.
    """
    sD1 = np.asarray(sD1, dtype=np.float64)
    d1, m = sD1.shape
    D = d1 - 1
    if t_vec is None:
        rng = np.random.default_rng() if rng is None else rng
        t_vec = rng.standard_normal((d1, n_t))
    else:
        t_vec = np.asarray(t_vec, dtype=np.float64)
        n_t = t_vec.shape[1]

    # The Gram matrix is the expensive part and is independent of T, so build it once.
    G = sD1.T @ sD1

    anchors = np.empty((d1, n_t))
    f_solver = np.empty(n_t)
    interior = np.zeros(n_t, dtype=bool)
    n_active = np.zeros(n_t)

    for i in range(n_t):
        sol = solve_active_set(sD1, t_vec[:, i], kappa=kappa, G=G)
        anchors[:, i] = sol.anchor
        f_solver[i] = sol.f
        interior[i] = sol.interior
        n_active[i] = sol.n_active

    f_anchor = _f_from_anchor(t_vec, anchors, kappa)
    consistency = float(np.max(np.abs(f_solver - f_anchor))) if n_t else 0.0

    f_mean = float(np.mean(f_anchor))
    f_sem = float(np.std(f_anchor, ddof=1) / np.sqrt(n_t)) if n_t > 1 else float("nan")
    alpha = float("inf") if f_mean <= 0 else 1.0 / f_mean

    return ManifoldResult(
        alpha=alpha,
        radius=anchor_radius(anchors),
        dimension=anchor_dimension(t_vec, anchors),
        D=D,
        f_mean=f_mean,
        f_sem=f_sem,
        frac_interior=float(np.mean(interior)),
        consistency=consistency,
        n_active_mean=float(np.mean(n_active)),
        anchors=anchors if keep_anchors else None,
    )


def _f_from_anchor(t_vec: np.ndarray, anchors: np.ndarray, kappa: float) -> np.ndarray:
    """F recomputed through the anchor, Eqs. 16-17.

    With the convention S . V >= kappa and V = T + lambda s_tilde, the active constraint gives
    lambda = (kappa - T . s_tilde) / ||s_tilde||^2, clipped at zero because lambda is a
    nonnegative multiplier, and then F = lambda^2 ||s_tilde||^2.
    """
    ts = np.sum(t_vec * anchors, axis=0)
    s_sq = np.sum(anchors ** 2, axis=0)
    excess = np.maximum(kappa - ts, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = np.where(s_sq > 0, excess / s_sq, 0.0)
    return lam ** 2 * s_sq


def anchor_radius(anchors: np.ndarray) -> float:
    """Anchor radius R_M, Eq. 28.

    The spread of the anchor points about their own mean, measured in the D shape coordinates
    and expressed in units of the anchor's center coordinate. Note that this is a variance
    across Gaussian draws: it asks "how much does the recruited support vector move as the
    classification direction changes", which is the sense in which the manifold has an extent
    that costs capacity.
    """
    ds0 = anchors - anchors.mean(axis=1, keepdims=True)
    center = anchors[-1, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        ds = np.where(np.abs(center) > 0, ds0[:-1, :] / center, 0.0)
    return float(np.sqrt(np.mean(np.sum(ds ** 2, axis=0))))


def anchor_dimension(t_vec: np.ndarray, anchors: np.ndarray) -> float:
    """Anchor dimension D_M, Eq. 29.

    D times the squared mean cosine between the Gaussian vector and the anchor, both restricted
    to the D shape coordinates. The intuition: in D dimensions a random direction has typical
    overlap 1/sqrt(D) with any fixed direction, so if the anchor tracked the Gaussian field in
    all D directions equally the mean cosine would be 1/sqrt(D_eff) and this expression returns
    D_eff. It is the mean of the cosine and then squared, not the mean of the squared cosine;
    that ordering is what makes it an *effective* dimension rather than a normalisation.
    """
    D = anchors.shape[0] - 1
    if D == 0:
        return 0.0
    t_s, a_s = t_vec[:D, :], anchors[:D, :]
    t_norm = np.linalg.norm(t_s, axis=0)
    a_norm = np.linalg.norm(a_s, axis=0)
    good = (t_norm > 0) & (a_norm > 0)
    if not np.any(good):
        return 0.0
    cos = np.sum(t_s[:, good] * a_s[:, good], axis=0) / (t_norm[good] * a_norm[good])
    return float(D * np.mean(cos) ** 2)


def combine(results: list[ManifoldResult]) -> float:
    """Total capacity over several manifolds: the reciprocal of the mean reciprocal.

    Capacity is not additive and does not average. alpha^{-1} is a mean of F, so pooling
    manifolds means pooling their F values, which is the harmonic mean of the alphas.
    """
    inv = np.array([1.0 / r.alpha for r in results if np.isfinite(r.alpha)])
    if inv.size == 0:
        return float("inf")
    return float(1.0 / np.mean(inv))


def ball_capacity_mc(
    D: int,
    radius: float,
    kappa: float = 0.0,
    n_t: int = 20000,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Capacity, R_M and D_M of a D-dimensional ball, using the closed-form inner solution.

    The inner minimisation for a ball is a second-order cone projection with an analytic
    answer (`inner.solve_ball`), so the only Monte Carlo is the Gaussian average. This is the
    reference curve for the sampled-manifold pipeline.

    Returns:
        (alpha, R_M, D_M)
    """
    rng = np.random.default_rng() if rng is None else rng
    t_vec = rng.standard_normal((D + 1, n_t))
    anchors = np.empty((D + 1, n_t))
    f = np.empty(n_t)
    for i in range(n_t):
        sol = solve_ball(t_vec[:, i], radius=radius, kappa=kappa)
        anchors[:, i] = sol.anchor
        f[i] = sol.f
    f_mean = float(np.mean(f))
    alpha = float("inf") if f_mean <= 0 else 1.0 / f_mean
    return alpha, anchor_radius(anchors), anchor_dimension(t_vec, anchors)
