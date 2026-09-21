"""Closed forms that the numerical pipeline must reproduce.

These are the fixed points of the whole library. Every one of them is an independent statement
about what the answer has to be, derived on paper rather than measured, so a disagreement always
means the code is wrong and never means the data is interesting.

THE POINT CAPACITY (Cover 1965; Gardner 1988 for the margin)
------------------------------------------------------------
For isolated points the inner minimization collapses to one dimension, F(T) = max(0, kappa - T)^2,
and the Gaussian average can be done in closed form:

    alpha_0(kappa)^{-1} = integral_{-inf}^{kappa} (kappa - t)^2 phi(t) dt
                        = (1 + kappa^2) Phi(kappa) + kappa phi(kappa)

with phi and Phi the standard normal density and CDF. At kappa = 0 this is 1/2, so alpha_0 = 2.
Large positive kappa makes separation hard and alpha_0 falls to 0; large negative kappa makes it
trivial and alpha_0 diverges.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def point_capacity(kappa: float | np.ndarray) -> float | np.ndarray:
    """alpha_0(kappa), the capacity of isolated points at margin kappa. Exact.

    alpha_0(kappa) = 1 / [ (1 + kappa^2) Phi(kappa) + kappa phi(kappa) ]

    This is the single most important number in the library: it is the kappa = 0 value of 2 that
    fixes all the normalisation conventions, and it is the function the low-rank approximation
    below evaluates at a shifted margin.
    """
    k = np.asarray(kappa, dtype=np.float64)
    inv = (1.0 + k ** 2) * norm.cdf(k) + k * norm.pdf(k)
    out = np.where(inv > 0, 1.0 / np.where(inv > 0, inv, 1.0), np.inf)
    return float(out) if np.ndim(kappa) == 0 else out


def point_capacity_quad(kappa: float) -> float:
    """alpha_0(kappa) by numerical quadrature, as an independent check of the closed form.

    Same quantity as `point_capacity`, computed by integrating (kappa - t)^2 against the
    Gaussian density rather than by the analytic antiderivative. Agreement confirms the
    algebra of the closed form; it is used only in the tests.
    """
    from scipy.integrate import quad

    val, _ = quad(
        lambda t: (kappa - t) ** 2 * norm.pdf(t), -np.inf, kappa, epsabs=1e-13, epsrel=1e-13
    )
    return 1.0 / val if val > 0 else np.inf


def margin_for_capacity(alpha: float) -> float:
    """Invert alpha_0: the margin at which point capacity equals `alpha`.

    Needed to compare the theory against a simulation, because a simulation naturally measures
    "what margin is achievable at this load" while the theory states "what load is separable at
    this margin". Monotonicity of alpha_0 makes the inverse well defined.
    """
    from scipy.optimize import brentq

    if alpha <= 0:
        raise ValueError("capacity must be positive")
    return float(brentq(lambda k: point_capacity(k) - alpha, -6.0, 30.0, xtol=1e-13))


# ----------------------------------------------------------------------------------------------
# Units
# ----------------------------------------------------------------------------------------------

def standardise_margin(kappa_measured: float | np.ndarray, N: int) -> float | np.ndarray:
    """Convert a measured geometric margin into the theory's units: kappa_theory = sqrt(N) * kappa.

    THIS CONVERSION IS EASY TO MISS AND SILENTLY WRONG WITHOUT IT.

    A simulation measures the margin geometrically: with a unit-norm readout w and unit-norm
    points x, the achieved margin is min_i y_i <w, x_i>, a number that shrinks as ~1/sqrt(P).
    The replica theory's kappa is not that number. Its Gaussian field T is the STANDARDISED
    overlap: for random unit w, the overlap <w, x> has standard deviation 1/sqrt(N), so
    T = sqrt(N) <w, x> and therefore

        kappa_theory = sqrt(N) * kappa_measured.

    Verified by scaling collapse rather than by derivation alone: measuring kappa* across loads
    at N = 50, 100, 200, 400 and rescaling by sqrt(N) makes the four curves fall on top of each
    other and on the inverse of alpha_0, from alpha = 0.3 (kappa = 1.53) to alpha = 1.9
    (kappa = 0.03). Without the factor, a comparison of theory to simulation at nonzero margin
    disagrees by orders of magnitude -- and the disagreement looks like a capacity error rather
    than a units error, which is what makes it dangerous.

    See `scripts/01_validate_points.py` for the collapse figure.
    """
    return np.sqrt(N) * np.asarray(kappa_measured)


def low_rank_approx(kappa: float, radius: float, dimension: float) -> float:
    """The standard small-radius approximation alpha_M ~ alpha_0 of an effective margin.

        alpha_M(kappa)  ~  alpha_0( (kappa + R_M sqrt(D_M)) / sqrt(1 + R_M^2) )

    The reading is that a manifold behaves like a point that has been handed a harder margin:
    being large (R_M) and being spread over many directions (D_M) both push the effective margin
    up, and capacity down. This is the formula that makes R_M and D_M worth reporting at all --
    it is the bridge from geometry back to capacity.

    IMPORTANT: this is an asymptotic approximation, valid for small radius, and it is quoted in
    several slightly different normalisations in the literature. It is NOT used anywhere in the
    estimation path. `scripts/02_validate_balls.py` measures its accuracy against the exact ball
    capacity across R and D rather than assuming it, and the test suite only asserts that the
    two agree in the regime where the approximation is supposed to hold. Treat a discrepancy at
    large R_M as expected behaviour of the approximation, not as a bug.
    """
    eff = (kappa + radius * np.sqrt(dimension)) / np.sqrt(1.0 + radius ** 2)
    return point_capacity(eff)
