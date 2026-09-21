"""Timing: mancap's active-set dual vs the reference's cvxopt primal, same inputs."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import ref_mftma  # noqa: E402

from mancap.capacity import analyze_manifold  # noqa: E402


def make_frame(rng, D, M, radius=0.3):
    S = rng.standard_normal((D, M))
    S /= np.linalg.norm(S, axis=0, keepdims=True)
    S *= radius
    S -= S.mean(axis=1, keepdims=True)
    return np.vstack([S, np.ones((1, M))])


rng = np.random.default_rng(0)
print(f"{'D':>4} {'M':>5} {'n_t':>6} {'mancap (s)':>11} {'cvxopt (s)':>11} {'speedup':>8}")
for D, M, n_t in [(5, 30, 200), (10, 60, 200), (20, 100, 200), (40, 200, 200)]:
    sD1 = make_frame(rng, D, M)
    t_vec = rng.standard_normal((D + 1, n_t))

    t0 = time.perf_counter()
    analyze_manifold(sD1, kappa=0.0, n_t=n_t, t_vec=t_vec)
    t_mine = time.perf_counter() - t0

    t0 = time.perf_counter()
    ref_mftma.each_manifold_analysis_D1(sD1, 0.0, n_t, t_vec=-t_vec)
    t_ref = time.perf_counter() - t0

    print(f"{D:>4} {M:>5} {n_t:>6} {t_mine:>11.3f} {t_ref:>11.3f} {t_ref / t_mine:>7.1f}x")
