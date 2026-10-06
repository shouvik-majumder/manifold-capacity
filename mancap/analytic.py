"""Closed-form results used to validate the numerical pipeline.

Point capacity (Cover 1965; Gardner 1988 for the margin)
--------------------------------------------------------
For isolated points the inner minimisation is one-dimensional, F(T) = max(0, kappa - T)^2,
and the Gaussian average has a closed form:

    alpha_0(kappa)^{-1} = integral_{-inf}^{kappa} (kappa - t)^2 phi(t) dt
                        = (1 + kappa^2) Phi(kappa) + kappa phi(kappa)

with phi and Phi the standard normal density and CDF. At kappa = 0 this gives alpha_0 = 2.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def point_capacity(kappa: float | np.ndarray) -> float | np.ndarray:
    """alpha_0(kappa), the capacity of isolated points at margin kappa. Exact.

    alpha_0(kappa) = 1 / [ (1 + kappa^2) Phi(kappa) + kappa phi(kappa) ]

    The kappa = 0 value of 2 fixes the normalisation conventions used throughout the package.
    The low-rank approximation below evaluates this function at a shifted margin.
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

    A simulation measures the margin geometrically: with a unit-norm readout w and unit-norm
    points x, the achieved margin is min_i y_i <w, x_i>. The replica theory's kappa is the
    standardised overlap: for random unit w, <w, x> has standard deviation 1/sqrt(N), so
    T = sqrt(N) <w, x> and

        kappa_theory = sqrt(N) * kappa_measured.

    The convention is checked by a scaling collapse in scripts/01_validate_points.py: kappa*
    measured across loads at N = 50 to 400 and rescaled by sqrt(N) falls on a single curve
    matching the inverse of alpha_0. Without the factor, theory and simulation disagree by
    orders of magnitude at nonzero margin.
    """
    return np.sqrt(N) * np.asarray(kappa_measured)


def low_rank_approx(kappa: float, radius: float, dimension: float) -> float:
    """The small-radius approximation alpha_M ~ alpha_0 of an effective margin.

        alpha_M(kappa)  ~  alpha_0( (kappa + R_M sqrt(D_M)) / sqrt(1 + R_M^2) )

    A manifold behaves like a point with a harder margin: a larger radius R_M and a larger
    dimension D_M both raise the effective margin and lower capacity. This relation connects
    R_M and D_M to capacity.

    It is an asymptotic approximation, valid for small radius, and appears in several
    normalisations in the literature. It is not used in the estimation path;
    scripts/02_validate_balls_and_segments.py measures its accuracy against the exact ball
    capacity, and the tests assert agreement only in the small-radius regime.
    """
    eff = (kappa + radius * np.sqrt(dimension)) / np.sqrt(1.0 + radius ** 2)
    return point_capacity(eff)
