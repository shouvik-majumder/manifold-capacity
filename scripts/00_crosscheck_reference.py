"""VALIDATION 0 -- cross-check mancap against the original cvxopt/MFTMA implementation.

Why this is exact rather than statistical: the reference and mancap are handed the SAME manifold
frame and the SAME Gaussian vectors. The only wrinkle is the sign convention -- the reference
imposes S . V <= -kappa while mancap imposes S . V >= +kappa -- and the two are related by
V -> -V, T -> -T. So the reference evaluated at -T must equal mancap evaluated at +T, for every
single draw, not merely on average. Agreement draw by draw is far stronger than agreement on a
Monte Carlo average, where two different errors could cancel.

WHAT THIS DOES AND DOES NOT ESTABLISH
-------------------------------------
It establishes that mancap's reformulation of the inner problem into its dual, and its
active-set solver, return exactly what the published cvxopt implementation returns for the
per-manifold quantities alpha_M, R_M and D_M. It does NOT establish that either is a correct
application of the theory -- both could share a transcription error. That is what
`01_validate_points.py` and the direct-simulation validations are for. It also does not cover
the correlated-centers correction (`fun_FA` in the reference), which mancap does not implement
yet.

SETUP -- this needs its own environment
---------------------------------------
The reference depends on cvxopt, autograd, and pymanopt's long-renamed API (`pymanopt.solvers`
was removed years ago), which will not coexist with a current scientific stack. Build a
throwaway environment, and stub out the pymanopt import, which is only needed for the
center-correlation step this script does not exercise:

    conda create -n mftmaref -c conda-forge python=3.11 "numpy<2" scipy cvxopt autograd -y
    conda activate mftmaref
    pip install -e /path/to/manifold-capacity --no-deps
    git clone https://github.com/schung039/neural_manifolds_replicaMFT
    # copy mftma/manifold_analysis_correlation.py next to this script as ref_mftma.py,
    # then comment out its three `from pymanopt...` lines
    python 00_crosscheck_reference.py

Last run 2026-09-21: worst relative disagreement 1.2e-12 (alpha), 1.4e-10 (R_M), 3.3e-08 (D_M),
over D = 1..12 with and without margin. See 00b_benchmark_reference.py for the timing comparison
(5-12x faster on CPU, single threaded).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import ref_mftma  # noqa: E402

from mancap.capacity import analyze_manifold  # noqa: E402


def make_frame(rng, D, M, radius):
    """A frame in the same convention both implementations expect: D shape rows, then ones."""
    S = rng.standard_normal((D, M))
    S /= np.linalg.norm(S, axis=0, keepdims=True)
    S *= radius
    S -= S.mean(axis=1, keepdims=True)  # centered offsets, as build_frames produces
    return np.vstack([S, np.ones((1, M))])


def main():
    rng = np.random.default_rng(0)
    rows = []

    cases = [
        # (D, M, radius, kappa)
        (1, 2, 0.30, 0.0),
        (1, 2, 0.30, 0.2),
        (2, 8, 0.25, 0.0),
        (2, 8, 0.25, 0.3),
        (3, 12, 0.40, 0.0),
        (5, 30, 0.30, 0.0),
        (5, 30, 0.30, 0.25),
        (8, 60, 0.20, 0.1),
        (12, 80, 0.35, 0.0),
    ]

    for (D, M, radius, kappa) in cases:
        sD1 = make_frame(rng, D, M, radius)
        n_t = 400
        t_vec = rng.standard_normal((D + 1, n_t))

        # mancap on +T (its own convention: S . V >= kappa)
        mine = analyze_manifold(sD1, kappa=kappa, n_t=n_t, t_vec=t_vec)

        # reference on -T (its convention: S . V <= -kappa)
        a_ref, R_ref, D_ref = ref_mftma.each_manifold_analysis_D1(
            sD1, kappa, n_t, t_vec=-t_vec
        )

        rows.append({
            "D": D, "M": M, "radius": radius, "kappa": kappa,
            "alpha_mancap": mine.alpha, "alpha_ref": float(a_ref),
            "R_mancap": mine.radius, "R_ref": float(R_ref),
            "D_mancap": mine.dimension, "D_ref": float(D_ref),
            "consistency": mine.consistency,
        })

        def rel(a, b):
            return abs(a - b) / max(abs(b), 1e-12)

        print(f"D={D:3d} M={M:3d} R={radius:.2f} k={kappa:.2f} | "
              f"alpha {mine.alpha:8.5f} vs {float(a_ref):8.5f} (rel {rel(mine.alpha, a_ref):.2e}) | "
              f"R_M {mine.radius:7.5f} vs {float(R_ref):7.5f} (rel {rel(mine.radius, R_ref):.2e}) | "
              f"D_M {mine.dimension:7.5f} vs {float(D_ref):7.5f} (rel {rel(mine.dimension, D_ref):.2e})")

    worst = {
        "alpha": max(abs(r["alpha_mancap"] - r["alpha_ref"]) / abs(r["alpha_ref"]) for r in rows),
        "R_M": max(abs(r["R_mancap"] - r["R_ref"]) / max(abs(r["R_ref"]), 1e-12) for r in rows),
        "D_M": max(abs(r["D_mancap"] - r["D_ref"]) / max(abs(r["D_ref"]), 1e-12) for r in rows),
    }
    print("\nworst relative disagreement:")
    for k, v in worst.items():
        print(f"  {k:6s} {v:.3e}")

    out = {"rows": rows, "worst_relative": worst}
    Path(__file__).with_name("compare_results.json").write_text(json.dumps(out, indent=2))
    print("\nwrote compare_results.json")


if __name__ == "__main__":
    main()
