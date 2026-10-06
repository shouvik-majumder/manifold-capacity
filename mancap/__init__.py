"""mancap: manifold capacity by mean-field theoretic manifold analysis (MFTMA).

An implementation of the replica mean-field theory of Chung, Lee and Sompolinsky,
"Classification and Geometry of General Perceptual Manifolds", Phys. Rev. X 8, 031003
(2018), and of the correlated-centers correction of Cohen, Chung, Lee and Sompolinsky,
"Separability and geometry of object manifolds in deep neural networks", Nat. Commun. 11,
746 (2020). Written as a learning reproduction; the authors' reference implementation is
github.com/schung039/neural_manifolds_replicaMFT.

Given P sets of activation vectors in R^N, one per category, the manifold capacity
alpha = P/N is the largest load at which a single linear readout separates the categories
under arbitrary binary labels. It decomposes into an anchor radius R_M and an anchor
dimension D_M.

Differences from the reference implementation
---------------------------------------------
The inner minimisation is solved in its dual form, a nonnegative quadratic program whose
Gram matrix does not depend on the Gaussian sample, by an active-set method; the reference
solves the primal with cvxopt. The correlated-centers correction uses a closed-form gradient
and a self-contained Stiefel optimiser in place of autograd and pymanopt. Frames are reduced
to the rank of the manifold offsets rather than to the number of samples, and the rank K of
the center subspace is chosen by an elbow rule with an optional measured null. Per-manifold
results agree with the reference draw by draw (scripts/00_crosscheck_reference.py).

Validation
----------
Closed forms (analytic), a closed-form inner solution for balls (inner.solve_ball), an
independent SLSQP solver, an internal consistency check on the anchors, and direct
simulation of the separability threshold (simulate), which does not use the replica
formulas.

Layout
------
inner      Inner minimisation: active set on the dual, SLSQP, closed-form ball.
capacity   alpha_M, R_M, D_M from the inner solutions; combining manifolds.
frames     Raw activations to (D+1)-frames; center-correlation diagnostics.
synth      Synthetic manifolds with known geometry.
centers    Correlated-centers correction.
analytic   Closed forms and the margin convention.
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
