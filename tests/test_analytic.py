"""Tests for the closed forms."""
from __future__ import annotations

import numpy as np
import pytest

from mancap import analytic


def test_cover_bound():
    """alpha_0(0) = 2. The number every convention in the package is fixed by."""
    assert analytic.point_capacity(0.0) == pytest.approx(2.0, abs=1e-12)


@pytest.mark.parametrize("kappa", [-1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0, 2.0, 3.0])
def test_closed_form_matches_quadrature(kappa):
    """The analytic antiderivative vs numerical integration of the defining integral."""
    assert analytic.point_capacity(kappa) == pytest.approx(
        analytic.point_capacity_quad(kappa), rel=1e-9
    )


def test_monotone_decreasing_in_margin():
    """Demanding a bigger margin can only reduce capacity."""
    k = np.linspace(-2.0, 4.0, 200)
    a = analytic.point_capacity(k)
    assert np.all(np.diff(a) < 0)


def test_vectorised_and_scalar_agree():
    k = np.array([0.0, 0.5, 1.0])
    vec = analytic.point_capacity(k)
    assert isinstance(vec, np.ndarray)
    for i, kk in enumerate(k):
        assert vec[i] == pytest.approx(analytic.point_capacity(float(kk)))


def test_low_rank_approx_reduces_to_points():
    """With no radius and no dimension, the approximation must return alpha_0 exactly."""
    for kappa in [0.0, 0.3, 1.0]:
        assert analytic.low_rank_approx(kappa, radius=0.0, dimension=0.0) == pytest.approx(
            analytic.point_capacity(kappa), rel=1e-12
        )


def test_low_rank_approx_is_monotone_in_radius_and_dimension():
    """Bigger and more spread out must both cost capacity, in the approximation as in reality."""
    base = analytic.low_rank_approx(0.0, radius=0.2, dimension=4.0)
    assert analytic.low_rank_approx(0.0, radius=0.4, dimension=4.0) < base
    assert analytic.low_rank_approx(0.0, radius=0.2, dimension=9.0) < base
