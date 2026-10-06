"""Validation 3: does the correlated-centers correction improve the prediction?

The replica theory assumes the P manifold centers are in general position; real
representations violate this. The correction of Cohen et al. (2020) finds the low-rank
subspace carrying the shared center structure and projects it out. The unit tests of
`mancap.centers` show that it recovers a planted subspace; this script asks whether
removing that subspace makes the theory predict the simulated separability threshold.

  1. Build P manifolds whose centers have a planted K-dimensional shared component, with a
     strength from 0 (isotropic) to near 1 (centers almost entirely inside a 3-dimensional
     subspace).
  2. Predict capacity without and with the correction.
  3. Measure capacity by direct simulation: random labels, exact linear-programming test for
     a separating hyperplane, sweep the load, find the threshold.
  4. Compare the two predictions with the measurement as the correlation grows.

Caveat: the center subspace is estimated from the P centers present, so it depends on P,
while the simulation sweeps P along the load axis. The theory is evaluated at a single
representative P. Because the planted structure is defined pairwise, the correlation between
any two centers is independent of P, so this is a mild approximation.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from mancap import build_frames, center_correlation, simulate, synth
from mancap.capacity import analyze_manifold, combine
from mancap.centers import find_center_subspace

ROOT = Path(__file__).resolve().parents[1]


def make_correlated(P, N, rng, K, strength, D, radius, M):
    C, U = synth.correlated_centers(P, N, rng, K=K, strength=strength)
    return synth.balls_with_centers(C, D=D, radius=radius, M=M, rng=rng), U


def theory_capacity(Xs, kappa, n_t, seed, corrected, n_restarts=3, n_null=3):
    """Capacity predicted with or without the center-subspace projection.

    n_null > 0 makes the structure threshold a measured chance level rather than a fixed
    constant. This matters for the strength = 0 row: fitting a subspace to isotropic centres
    removes about 10% of their correlation by chance, so a fixed 10% threshold would project a
    rank-8 subspace out of data that has none and make the corrected prediction undershoot.
    """
    sub = None
    info = {}
    if corrected:
        res = find_center_subspace(
            Xs, n_restarts=n_restarts, n_null=n_null, rng=np.random.default_rng(seed + 99)
        )
        sub = res.basis if res.K > 0 else None
        info = {"K": res.K, "K_argmin": res.K_argmin, "reduction": res.reduction,
                "residual": res.residual, "baseline": res.baseline,
                "by_K": res.by_K.tolist()}
    frames = build_frames(Xs, center_subspace=sub).frames
    out = [
        analyze_manifold(f, kappa=kappa, n_t=n_t, rng=np.random.default_rng(seed + i))
        for i, f in enumerate(frames)
    ]
    return combine(out), float(np.mean([r.radius for r in out])), \
        float(np.mean([r.dimension for r in out])), info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=120)
    ap.add_argument("--P-theory", type=int, default=60)
    ap.add_argument("--D", type=int, default=4)
    ap.add_argument("--radius", type=float, default=0.25)
    ap.add_argument("--M", type=int, default=40)
    ap.add_argument("--K-planted", type=int, default=3)
    ap.add_argument("--n-t", type=int, default=3000)
    ap.add_argument("--n-seeds", type=int, default=40)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.N, args.P_theory, args.n_t, args.n_seeds = 70, 30, 1000, 20

    strengths = [0.0, 0.4, 0.7, 0.85, 0.95]
    alphas = np.concatenate([np.arange(0.3, 1.2, 0.1), np.arange(1.2, 2.4, 0.15)])
    t0 = time.time()
    rows = []

    print(f"planted rank {args.K_planted}, balls D={args.D} R={args.radius} M={args.M}, "
          f"N={args.N}\n")
    print(f"{'strength':>9} {'|cos|':>7} {'partic':>7} {'uncorr':>8} {'corrected':>10} "
          f"{'simulated':>10} {'K':>3} {'K_argmin':>9} {'reduc':>6}")

    for s in strengths:
        rng = np.random.default_rng(int(1000 * s) + 7)
        Xs, _ = make_correlated(args.P_theory, args.N, rng, args.K_planted, s,
                                args.D, args.radius, args.M)
        diag = center_correlation(Xs)

        a_un, R_un, D_un, _ = theory_capacity(Xs, 0.0, args.n_t, seed=11, corrected=False)
        a_co, R_co, D_co, info = theory_capacity(Xs, 0.0, args.n_t, seed=11, corrected=True)

        scan = simulate.threshold_scan(
            lambda P, NN, r: make_correlated(P, NN, r, args.K_planted, s, args.D,
                                             args.radius, args.M)[0],
            args.N, alphas, n_seeds=args.n_seeds,
            rng=np.random.default_rng(int(2000 * s) + 3), compute_margin=False,
        )
        emp = simulate.crossing(scan.alphas, scan.frac_sep, 0.5)

        rows.append({
            "strength": s, "mean_abs_cos": diag["mean_abs_cos"],
            "participation": diag["participation"],
            "alpha_uncorrected": a_un, "alpha_corrected": a_co, "alpha_simulated": emp,
            "R_uncorrected": R_un, "R_corrected": R_co,
            "D_uncorrected": D_un, "D_corrected": D_co,
            "subspace": info, "frac_sep": scan.frac_sep.tolist(),
        })
        print(f"{s:>9.2f} {diag['mean_abs_cos']:>7.3f} {diag['participation']:>7.1f} "
              f"{a_un:>8.4f} {a_co:>10.4f} {emp:>10.4f} {info.get('K', 0):>3} "
              f"{info.get('K_argmin', 0):>9} {info.get('reduction', 0):>6.3f}")

    err_un = [abs(r["alpha_uncorrected"] - r["alpha_simulated"]) for r in rows
              if np.isfinite(r["alpha_simulated"])]
    err_co = [abs(r["alpha_corrected"] - r["alpha_simulated"]) for r in rows
              if np.isfinite(r["alpha_simulated"])]
    print(f"\nmean |theory - simulation|:  uncorrected {np.mean(err_un):.4f}   "
          f"corrected {np.mean(err_co):.4f}")
    verdict = ("the correction improves agreement" if np.mean(err_co) < np.mean(err_un)
               else "the correction does NOT improve agreement in this regime")
    print(f"VERDICT: {verdict}")

    out = {"rows": rows, "alphas": alphas.tolist(), "settings": vars(args),
           "mean_err_uncorrected": float(np.mean(err_un)),
           "mean_err_corrected": float(np.mean(err_co)),
           "verdict": verdict, "seconds": time.time() - t0}
    (ROOT / "data").mkdir(exist_ok=True)
    (ROOT / "data" / "validate_correlated_centers.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote data/validate_correlated_centers.json  ({out['seconds']:.1f}s)")
    _figure(out)


def _figure(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = out["rows"]
    s = [r["strength"] for r in rows]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))

    ax[0].plot(s, [r["alpha_simulated"] for r in rows], "s-", ms=9, mfc="none", mew=2,
               color="#d62728", label="direct simulation (ground truth)")
    ax[0].plot(s, [r["alpha_uncorrected"] for r in rows], "o--", ms=6, color="0.55",
               label="theory, uncorrected")
    ax[0].plot(s, [r["alpha_corrected"] for r in rows], "o-", ms=6, color="#1f77b4",
               label="theory, centers corrected")
    ax[0].set_xlabel("planted center-correlation strength")
    ax[0].set_ylabel("capacity $\\alpha$")
    ax[0].set_title("Does the correction track reality?", fontsize=11)
    ax[0].legend(frameon=False, fontsize=8)

    ax[1].plot(s, [r["mean_abs_cos"] for r in rows], "o-", color="#2ca02c")
    ax[1].set_xlabel("planted center-correlation strength")
    ax[1].set_ylabel("mean $|\\cos|$ between centres")
    ax[1].set_title("How badly the assumption is violated", fontsize=11)
    axb = ax[1].twinx()
    axb.plot(s, [r["participation"] for r in rows], "s--", color="#9467bd", ms=5)
    axb.set_ylabel("participation ratio of centres", color="#9467bd")
    axb.tick_params(axis="y", colors="#9467bd")

    for r in rows:
        by_K = r["subspace"].get("by_K")
        if by_K:
            ax[2].plot(range(1, len(by_K) + 1), np.array(by_K) / r["subspace"]["baseline"],
                       "o-", ms=4, label=f"strength {r['strength']:.2f}")
    ax[2].axhline(1.0, ls=":", color="0.6")
    ax[2].set_xlabel("rank $K$ of removed subspace")
    ax[2].set_ylabel("residual correlation / baseline")
    ax[2].set_title("How $K$ is chosen\n(flat = no low-rank structure to remove)", fontsize=10)
    ax[2].legend(frameon=False, fontsize=7)

    for a in ax:
        a.spines[["top"]].set_visible(False)
    fig.suptitle("Validation 3: the correlated-centres correction, judged against simulation",
                 fontsize=12)
    fig.tight_layout()
    (ROOT / "figures").mkdir(exist_ok=True)
    p = ROOT / "figures" / "03_validate_correlated_centers.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"wrote {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
