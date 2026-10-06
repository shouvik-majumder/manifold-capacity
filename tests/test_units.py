"""Tests for the margin unit convention.

The theory's kappa is a STANDARDISED overlap, not a geometric one. A simulation naturally
measures the geometric margin min_i y_i <w, x_i> with unit-norm w and x; the theory's kappa is
sqrt(N) times that. Getting this wrong does not produce an obviously broken number -- it produces
a plausible capacity that is wrong by orders of magnitude, attributed to the wrong cause. These
tests pin the convention.
"""
from __future__ import annotations

import numpy as np
import pytest

from mancap import analytic, simulate, synth


def test_margin_inverse_round_trips():
    for alpha in [0.2, 0.5, 1.0, 1.5, 1.9, 2.5, 4.0]:
        k = analytic.margin_for_capacity(alpha)
        assert analytic.point_capacity(k) == pytest.approx(alpha, rel=1e-9)


def test_margin_inverse_at_cover_bound():
    """Capacity 2 must correspond to zero margin."""
    assert analytic.margin_for_capacity(2.0) == pytest.approx(0.0, abs=1e-9)


def test_standardise_margin_is_sqrt_n():
    assert analytic.standardise_margin(0.1, 100) == pytest.approx(1.0)
    assert np.allclose(analytic.standardise_margin(np.array([0.1, 0.2]), 400),
                       np.array([2.0, 4.0]))


@pytest.mark.slow
def test_margin_collapses_across_N_onto_the_theory():
    """sqrt(N) * kappa* is N-independent and matches the theory.

    Measures the achievable margin for point manifolds at several loads and several ambient
    dimensions. After the sqrt(N) rescaling the curves must fall on top of each other and on
    alpha_0 inverted. Solver-to-solver comparisons cannot detect a wrong margin convention;
    this collapse does.
    """
    loads = [0.5, 1.0, 1.6]
    Ns = [50, 200, 400]
    for alpha in loads:
        expected = analytic.margin_for_capacity(alpha)
        got = []
        for N in Ns:
            rng = np.random.default_rng(int(100 * alpha) + N)
            P = max(1, int(round(alpha * N)))
            ks = []
            for _ in range(10):
                Xs = synth.points(P, N, rng)
                y = rng.choice([-1.0, 1.0], size=len(Xs))
                ks.append(simulate.max_margin(simulate.labelled_points(Xs, y)).kappa_star)
            got.append(float(np.mean(analytic.standardise_margin(np.array(ks), N))))
        # Collapse: the N-dependence must be small compared with the value itself.
        assert np.std(got) < 0.1 + 0.15 * abs(expected), f"alpha={alpha}: no collapse, {got}"
        assert np.mean(got) == pytest.approx(expected, abs=0.12), (
            f"alpha={alpha}: standardised margin {np.mean(got):.3f} vs theory {expected:.3f}"
        )
