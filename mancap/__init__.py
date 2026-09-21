"""mancap -- manifold capacity by replica mean-field theory, reimplemented and validated.

An implementation of the mean-field theoretic manifold analysis (MFTMA) of Chung, Lee &
Sompolinsky, "Classification and Geometry of General Perceptual Manifolds", Phys. Rev. X 8,
031003 (2018), and its correlated-centers extension in Cohen, Chung, Lee & Sompolinsky,
"Separability and geometry of object manifolds in deep neural networks", Nat. Commun. 11, 746
(2020).

THE QUESTION IT ANSWERS
-----------------------
Given P sets of activation vectors -- one set per category, each set a cloud of points in R^N --
how many such categories could a single linear readout separate at once, under arbitrary +/-1
labels? The answer is a load alpha = P/N, the manifold capacity, and it is a single number
summarising how linearly usable a representation is. It decomposes into an anchor radius R_M and
an anchor dimension D_M, which say whether capacity was lost because the clouds are large or
because they are spread over many directions.

WHY REIMPLEMENT IT
------------------
The reference implementation is correct but slow and undeployable: the inner optimization calls
cvxopt once per Gaussian sample in a Python loop, and the center-correlation step runs a Stiefel
manifold optimization for 20000 iterations at every candidate rank. Its dependencies (autograd,
cvxopt, pymanopt with a long-renamed API) no longer install cleanly. This package reformulates
the inner problem in its dual, where it is a nonnegative quadratic program whose Gram matrix
does not depend on the Gaussian sample, which makes it exactly solvable by a finite active-set
method now and batchable on a GPU later.

HOW IT IS VALIDATED
-------------------
Capacity is a number with no error bars attached, computed from a Monte Carlo average of the
solution of an optimization problem. It is very easy to produce a plausible-looking wrong one.
So the package is organised around independent checks rather than around the estimator:

  analytic.point_capacity   Closed form for points. alpha = 2 at kappa = 0 fixes all conventions.
  inner.solve_ball          Closed-form inner solution for balls -- an exact oracle on a
                            manifold that is neither a point nor low dimensional.
  inner.solve_slsqp         A second, algebraically unrelated solver for the same problem.
  capacity.analyze_manifold Returns `consistency`, the gap between the objective computed two
                            ways, so the anchors are tested and not just the capacity.
  simulate                  Direct simulation: build the manifolds, assign random labels, and
                            actually search for a separating hyperplane. Sweep P/N to find where
                            separability breaks, and compare that to the theory. This is the only
                            test that does not assume the theory is correctly transcribed.

Layout
------
inner      The inner minimization (active set on the dual, SLSQP, closed-form ball).
capacity   alpha_M, R_M, D_M from the inner solutions; how to combine manifolds.
frames     Raw activations -> the (D+1)-frames the theory expects; center-correlation diagnostics.
synth      Synthetic manifolds with known geometry: points, balls, segments, rings, ellipsoids.
centers    The correlated-centers correction, which real representations need.
analytic   Closed forms the pipeline must reproduce.
simulate   Direct simulation of the separability threshold.
"""
from __future__ import annotations

from . import analytic, capacity, centers, frames, inner, simulate, synth
from .capacity import ManifoldResult, analyze_manifold, combine
from .frames import FrameSet, build_frames, center_correlation

__all__ = [
    "analytic",
    "capacity",
    "centers",
    "frames",
    "inner",
    "simulate",
    "synth",
    "ManifoldResult",
    "analyze_manifold",
    "combine",
    "FrameSet",
    "build_frames",
    "center_correlation",
]

__version__ = "0.1.0"
