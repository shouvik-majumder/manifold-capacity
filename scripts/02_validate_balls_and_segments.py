"""Validation 2: manifolds with extent. Exact balls, sampling bias, and a theory-free threshold.

Validation 1 established the point limit, where the inner minimisation collapses to one
variable. This script validates the parts of the theory that only exist once a manifold has
extent.

(A) Do alpha, R_M and D_M behave correctly for a manifold with known geometry?
    Uses `inner.solve_ball`, the closed-form cone projection, so the manifold is an exact
    ball and the only Monte Carlo is the Gaussian average. Expected: alpha falls with
    radius; R_M tracks the true radius at low D; D_M rises with D but stays below it.

(B) How much does finite sampling bias capacity?
    The estimator sees only the convex hull of the sampled points, so a ball sampled with M
    points is an inscribed polytope with higher capacity than the ball. Measured against the
    exact ball as a function of M and D.

(C) Is the low-rank approximation alpha_M ~ alpha_0((kappa + R sqrt(D))/sqrt(1+R^2))
    accurate? It is not used in the estimation path; this measures its range of validity.

(D) Does the theory predict the empirical threshold for a manifold with extent? Direct
    simulation for segments: a segment's convex hull is exactly its two endpoints, so M = 2
    represents the manifold with no sampling bias, which separates the question of the
    theory's correctness from the question of sampling adequacy.

Frame convention
----------------
Theory curves are computed with subtract_global_mean=False so that a manifold built with
nominal radius R has radius R; with the global mean subtracted, radii are inflated by
1/sqrt(1 - 1/P), and P changes along a load scan. The simulation separates through the
origin with no bias term, matching this convention.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from mancap import analytic, build_frames, simulate, synth
from mancap.capacity import analyze_manifold, ball_capacity_mc, combine

ROOT = Path(__file__).resolve().parents[1]


def part_a(radii, dims, n_t, seed=0):
    """Exact ball geometry and capacity via the closed-form inner solution."""
    rows = []
    for D in dims:
        for R in radii:
            a, R_M, D_M = ball_capacity_mc(
                D, radius=R, kappa=0.0, n_t=n_t, rng=np.random.default_rng(seed + D)
            )
            rows.append({"D": D, "radius": R, "alpha": a, "R_M": R_M, "D_M": D_M})
    return rows


def part_b(D_list, M_list, radius, n_t, P=12, N=400, seed=0):
    """Sampling bias: sampled-ball capacity against the exact ball, versus M and D."""
    rows = []
    for D in D_list:
        exact, exact_R, exact_D = ball_capacity_mc(
            D, radius=radius, kappa=0.0, n_t=20000, rng=np.random.default_rng(seed + 77 + D)
        )
        for M in M_list:
            rng = np.random.default_rng(seed + 1000 * D + M)
            Xs = synth.balls(P, N, D=D, radius=radius, M=M, rng=rng)
            res = [
                analyze_manifold(f, kappa=0.0, n_t=n_t, rng=np.random.default_rng(seed + M))
                for f in build_frames(Xs, subtract_global_mean=False).frames
            ]
            rows.append({
                "D": D, "M": M, "alpha_sampled": combine(res), "alpha_exact": exact,
                "R_sampled": float(np.mean([r.radius for r in res])),
                "R_exact": exact_R,
                "D_sampled": float(np.mean([r.dimension for r in res])),
                "D_exact": exact_D,
            })
    return rows


def part_c(rows_a):
    """Accuracy of the low-rank approximation against the exact ball numbers from part A."""
    out = []
    for r in rows_a:
        approx = analytic.low_rank_approx(0.0, radius=r["R_M"], dimension=r["D_M"])
        out.append({
            "D": r["D"], "radius": r["radius"], "alpha_exact": r["alpha"],
            "alpha_approx": float(approx),
            "rel_error": float(abs(approx - r["alpha"]) / r["alpha"]),
        })
    return out


def part_d(half_lengths, N, alphas, n_seeds, n_t, seed=0):
    """Theory vs direct simulation for segments, whose hull is exact with two sample points."""
    rows = []
    for L in half_lengths:
        # Theory: build one segment, analyse it. Every segment is congruent, so one suffices.
        rng = np.random.default_rng(seed)
        Xs = synth.segments(12, 400, half_length=L, M=2, rng=rng)
        res = [
            analyze_manifold(f, kappa=0.0, n_t=n_t, rng=np.random.default_rng(seed + 5 + i))
            for i, f in enumerate(build_frames(Xs, subtract_global_mean=False).frames)
        ]
        theory = combine(res)

        scan = simulate.threshold_scan(
            lambda P, NN, r: synth.segments(P, NN, half_length=L, M=2, rng=r),
            N, alphas, n_seeds=n_seeds, rng=np.random.default_rng(seed + 400 + int(L * 100)),
            compute_margin=False,
        )
        emp = simulate.crossing(scan.alphas, scan.frac_sep, 0.5)
        rows.append({
            "half_length": L, "theory": theory, "empirical": emp, "N": N,
            "R_M": float(np.mean([r.radius for r in res])),
            "D_M": float(np.mean([r.dimension for r in res])),
            "frac_sep": scan.frac_sep.tolist(),
        })
        print(f"  L={L:.2f}  theory alpha={theory:.4f}   simulated alpha_c={emp:.4f}   "
              f"(R_M={rows[-1]['R_M']:.3f}, D_M={rows[-1]['D_M']:.3f})")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-t", type=int, default=4000)
    ap.add_argument("--n-seeds", type=int, default=40)
    ap.add_argument("--N", type=int, default=100, help="ambient dim for the simulation in part D")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.n_t, args.n_seeds, args.N = 1200, 20, 60

    t0 = time.time()
    radii = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5]
    dims = [1, 2, 5, 10, 20, 50]

    print("PART A -- exact ball capacity and geometry (closed-form inner solution)")
    rows_a = part_a(radii, dims, n_t=max(args.n_t, 20000))
    print(f"  {'D':>4} {'R':>6} {'alpha':>9} {'R_M':>8} {'D_M':>8}")
    for r in rows_a:
        if r["radius"] in (0.1, 0.3, 1.0):
            print(f"  {r['D']:>4} {r['radius']:>6.2f} {r['alpha']:>9.4f} "
                  f"{r['R_M']:>8.4f} {r['D_M']:>8.3f}")

    print("\nPART B -- sampling bias of the estimator vs the exact ball")
    rows_b = part_b([2, 5, 20], [10, 30, 100, 300, 1000], radius=0.3, n_t=args.n_t)
    print(f"  {'D':>4} {'M':>6} {'alpha_samp':>11} {'alpha_exact':>12} {'bias %':>8} "
          f"{'D_M samp':>9} {'D_M exact':>10}")
    for r in rows_b:
        bias = 100 * (r["alpha_sampled"] - r["alpha_exact"]) / r["alpha_exact"]
        print(f"  {r['D']:>4} {r['M']:>6} {r['alpha_sampled']:>11.4f} "
              f"{r['alpha_exact']:>12.4f} {bias:>+8.1f} {r['D_sampled']:>9.3f} "
              f"{r['D_exact']:>10.3f}")

    print("\nPART C -- accuracy of the low-rank approximation")
    rows_c = part_c(rows_a)
    for D in dims:
        sub = [r for r in rows_c if r["D"] == D]
        small = [r["rel_error"] for r in sub if r["radius"] <= 0.2]
        large = [r["rel_error"] for r in sub if r["radius"] >= 1.0]
        print(f"  D={D:>3}  mean rel error: R<=0.2 -> {np.mean(small):.3f}   "
              f"R>=1.0 -> {np.mean(large):.3f}")

    print("\nPART D -- THEORY-FREE: segments, theory vs direct simulation")
    alphas = np.concatenate([np.arange(0.3, 1.2, 0.1), np.arange(1.2, 2.6, 0.15)])
    rows_d = part_d([0.1, 0.3, 0.6, 1.0], args.N, alphas, args.n_seeds, args.n_t)

    out = {
        "part_a": rows_a, "part_b": rows_b, "part_c": rows_c, "part_d": rows_d,
        "alphas_d": alphas.tolist(), "radii": radii, "dims": dims,
        "settings": vars(args), "seconds": time.time() - t0,
    }
    (ROOT / "data").mkdir(exist_ok=True)
    (ROOT / "data" / "validate_balls_segments.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote data/validate_balls_segments.json  ({out['seconds']:.1f}s)")
    _figure(out)


def _figure(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(12.5, 9))
    cmap = plt.get_cmap("viridis")
    dims = out["dims"]

    # (A) capacity vs radius, by dimension --------------------------------------------------
    for i, D in enumerate(dims):
        sub = [r for r in out["part_a"] if r["D"] == D]
        ax[0, 0].plot([r["radius"] for r in sub], [r["alpha"] for r in sub], "o-", ms=4,
                      color=cmap(i / max(len(dims) - 1, 1)), label=f"D={D}")
    ax[0, 0].axhline(2.0, ls=":", color="#d62728", lw=1.2)
    ax[0, 0].text(0.06, 2.06, "point limit $\\alpha_0(0)=2$", fontsize=8, color="#d62728")
    ax[0, 0].set_xscale("log")
    ax[0, 0].set_xlabel("ball radius $R$ (units of centre norm)")
    ax[0, 0].set_ylabel("capacity $\\alpha_M$")
    ax[0, 0].set_title("(A) Exact ball capacity: extent and dimension both cost capacity",
                       fontsize=10)
    ax[0, 0].legend(frameon=False, fontsize=8, ncol=2)

    # (B) sampling bias ---------------------------------------------------------------------
    for i, D in enumerate([2, 5, 20]):
        sub = [r for r in out["part_b"] if r["D"] == D]
        M = [r["M"] for r in sub]
        bias = [100 * (r["alpha_sampled"] - r["alpha_exact"]) / r["alpha_exact"] for r in sub]
        ax[0, 1].plot(M, bias, "o-", ms=5, color=cmap(i / 2.0), label=f"D={D}")
    ax[0, 1].axhline(0, ls="--", color="0.4", lw=1)
    ax[0, 1].set_xscale("log")
    ax[0, 1].set_xlabel("sample points per manifold, $M$")
    ax[0, 1].set_ylabel("capacity overestimate (%)")
    ax[0, 1].set_title("(B) Finite sampling inflates capacity\nbadly, and worse at high D",
                       fontsize=10)
    ax[0, 1].legend(frameon=False, fontsize=8)

    # (C) low-rank approximation accuracy ---------------------------------------------------
    for i, D in enumerate(dims):
        sub = [r for r in out["part_c"] if r["D"] == D]
        ax[1, 0].plot([r["radius"] for r in sub], [100 * r["rel_error"] for r in sub], "o-",
                      ms=4, color=cmap(i / max(len(dims) - 1, 1)), label=f"D={D}")
    ax[1, 0].set_xscale("log")
    ax[1, 0].set_yscale("log")
    ax[1, 0].set_xlabel("ball radius $R$")
    ax[1, 0].set_ylabel("relative error of approximation (%)")
    ax[1, 0].set_title("(C) The low-rank formula is good only at small radius\n"
                       "(measured, not assumed)", fontsize=10)
    ax[1, 0].legend(frameon=False, fontsize=8, ncol=2)

    # (D) theory vs simulation for segments -------------------------------------------------
    d = out["part_d"]
    th = [r["theory"] for r in d]
    em = [r["empirical"] for r in d]
    L = [r["half_length"] for r in d]
    ax[1, 1].plot(L, th, "o-", ms=7, color="0.25", label="replica theory")
    ax[1, 1].plot(L, em, "s", ms=9, mfc="none", mew=2, color="#d62728",
                  label=f"direct simulation, N={d[0]['N']}")
    for x, a, b in zip(L, th, em):
        ax[1, 1].plot([x, x], [a, b], "-", color="0.8", lw=1, zorder=0)
    ax[1, 1].set_xlabel("segment half-length (units of centre norm)")
    ax[1, 1].set_ylabel("capacity $\\alpha$")
    ax[1, 1].set_title("(D) Theory-free check with EXTENT: segments\n"
                       "(hull exact at M=2, so no sampling bias)", fontsize=10)
    ax[1, 1].legend(frameon=False, fontsize=8)

    for a in ax.ravel():
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Validation 2: manifolds with extent -- exact geometry, sampling bias, "
                 "and an independent threshold", fontsize=12)
    fig.tight_layout()
    (ROOT / "figures").mkdir(exist_ok=True)
    p = ROOT / "figures" / "02_validate_balls_segments.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"wrote {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
