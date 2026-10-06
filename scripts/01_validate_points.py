"""Validation 1: points. The closed form, the estimator, and a theory-free simulation.

Three routes to the capacity of isolated point manifolds should agree:

  (a) the closed form alpha_0(kappa) = 1 / [(1+kappa^2) Phi(kappa) + kappa phi(kappa)],
  (b) the Monte Carlo estimator in `capacity.analyze_manifold`, run through the full public
      path including frame construction,
  (c) direct simulation: build P points in R^N, label them at random, and ask an exact
      linear program whether a separating hyperplane exists; sweep the load, find where
      separability breaks, extrapolate to N -> infinity.

(a) and (b) share the theory, so their agreement does not test its transcription. (c) never
evaluates a replica formula, so agreement between (c) and (a) tests the application of the
theory. The expected answer is Cover's alpha_c = 2 at kappa = 0.

Figure
------
Left panel: the three routes against margin. The simulated points carry error bars from the
binomial spread of the separable fraction.

Right panel: separability against load at several N. The transition sharpens as N grows,
with a width of order 1/sqrt(N); the second trace shows the measured threshold against 1/N,
extrapolating to 2.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from mancap import analytic, build_frames, simulate, synth
from mancap.capacity import analyze_manifold, combine

ROOT = Path(__file__).resolve().parents[1]


def estimator_curve(kappas, n_t, seed=0):
    """Route (b): the Monte Carlo estimator through the full public path.

    Each margin gets independent Gaussian draws, so that the reported z-scores are separate
    tests rather than one fluctuation seen through a slowly varying function.
    """
    rng = np.random.default_rng(seed)
    Xs = synth.points(P=40, N=150, rng=rng)
    frames = build_frames(Xs).frames
    out = []
    for j, k in enumerate(kappas):
        res = [
            analyze_manifold(
                f, kappa=float(k), n_t=n_t,
                rng=np.random.default_rng(seed + 1000 * (j + 1) + i),
            )
            for i, f in enumerate(frames[:4])
        ]
        rel = float(np.mean([r.f_sem / r.f_mean for r in res]))
        out.append((combine(res), rel))
    return np.array([o[0] for o in out]), np.array([o[1] for o in out])


def margin_collapse(Ns, alphas, n_seeds, seed=0):
    """Route (c) at nonzero margin, as a scaling collapse.

    A simulation measures the margin geometrically: with unit-norm readout and unit-norm
    points, kappa_measured = min_i y_i <w, x_i>. The theory's kappa is the standardised overlap,
    kappa_theory = sqrt(N) * kappa_measured (see `analytic.standardise_margin`).

    Measuring kappa* across loads at several N and multiplying by sqrt(N) should place all the
    curves on top of each other and on the inverse of alpha_0. This tests the margin convention,
    which agreement between solvers cannot.
    """
    out = {}
    for N in Ns:
        rng = np.random.default_rng(seed + N)
        mean_k, sd_k = [], []
        for alpha in alphas:
            P = max(1, int(round(alpha * N)))
            ks = []
            for _ in range(n_seeds):
                Xs = synth.points(P, N, rng)
                labels = rng.choice([-1.0, 1.0], size=len(Xs))
                ks.append(simulate.max_margin(simulate.labelled_points(Xs, labels)).kappa_star)
            ks = analytic.standardise_margin(np.array(ks), N)
            mean_k.append(float(np.mean(ks)))
            sd_k.append(float(np.std(ks)))
        out[N] = {"mean": mean_k, "sd": sd_k}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-t", type=int, default=20000, help="Gaussian draws for the estimator")
    ap.add_argument("--n-seeds", type=int, default=60, help="label draws per load")
    ap.add_argument("--Ns", type=int, nargs="+", default=[25, 50, 100, 200])
    ap.add_argument("--margin-N", type=int, default=100,
                    help="ambient dim for the nonzero-margin curve; kept below the largest N "
                         "because that scan needs max_margin at every load, not just an LP")
    ap.add_argument("--quick", action="store_true", help="small settings for a smoke run")
    args = ap.parse_args()
    if args.quick:
        args.n_t, args.n_seeds, args.Ns, args.margin_N = 4000, 25, [25, 50], 50

    t0 = time.time()
    kappas = np.array([0.0, 0.1, 0.25, 0.5, 0.75, 1.0])

    print("ROUTE (a) closed form and (b) Monte Carlo estimator")
    theory = analytic.point_capacity(kappas)
    est, rel = estimator_curve(kappas, args.n_t)
    print(f"  {'kappa':>7} {'theory':>10} {'estimator':>10} {'rel sem':>9} {'z':>7}")
    for k, th, es, rl in zip(kappas, theory, est, rel):
        z = (es - th) / (th * rl) if rl > 0 else np.nan
        print(f"  {k:>7.2f} {th:>10.4f} {es:>10.4f} {rl:>9.4f} {z:>+7.2f}")
    max_z = float(np.max(np.abs((est - theory) / (theory * rel))))
    print(f"  worst deviation: {max_z:.2f} standard errors")

    print("\nROUTE (c) direct simulation, finite-size scan at kappa = 0")
    alphas = np.arange(1.3, 2.9, 0.1)
    scans, thr = [], []
    for N in args.Ns:
        s = simulate.threshold_scan(
            lambda P, NN, r: synth.points(P, NN, r), N, alphas, n_seeds=args.n_seeds,
            rng=np.random.default_rng(1000 + N), compute_margin=False,
        )
        a50 = simulate.crossing(s.alphas, s.frac_sep, 0.5)
        scans.append(s)
        thr.append(a50)
        print(f"  N={N:4d}  alpha_c = {a50:.4f}")
    ex = simulate.extrapolate_threshold(args.Ns, thr)
    print(f"  extrapolated N -> inf: {ex['intercept']:.4f}  "
          f"(slope {ex['slope']:+.2f}, rms resid {ex['resid']:.4f})   THEORY 2.0000")

    print("\nROUTE (c) at nonzero margin: scaling collapse of sqrt(N) * kappa*")
    margin_alphas = np.array([0.3, 0.5, 0.8, 1.0, 1.3, 1.6, 1.9])
    margin_Ns = [N for N in args.Ns if N >= 50] or args.Ns
    coll = margin_collapse(margin_Ns, margin_alphas, max(12, args.n_seeds // 4))
    kappa_theory = [analytic.margin_for_capacity(a) for a in margin_alphas]
    print(f"  {'alpha':>7} {'kappa_th':>9} |" + "".join(f"{f'N={N}':>10}" for N in margin_Ns))
    for j, a in enumerate(margin_alphas):
        row = f"  {a:>7.2f} {kappa_theory[j]:>9.4f} |"
        row += "".join(f"{coll[N]['mean'][j]:>10.4f}" for N in margin_Ns)
        print(row)
    dev = max(
        abs(coll[N]["mean"][j] - kappa_theory[j])
        for N in margin_Ns for j in range(len(margin_alphas))
    )
    print(f"  worst absolute deviation from the theoretical margin: {dev:.4f}")

    out = {
        "kappas": kappas.tolist(),
        "theory": theory.tolist(),
        "estimator": est.tolist(),
        "estimator_rel_sem": rel.tolist(),
        "worst_z": max_z,
        "Ns": list(args.Ns),
        "thresholds": thr,
        "extrapolation": ex,
        "alphas": alphas.tolist(),
        "frac_sep": [s.frac_sep.tolist() for s in scans],
        "margin_collapse": {str(N): coll[N] for N in margin_Ns},
        "margin_alphas": margin_alphas.tolist(),
        "margin_kappa_theory": kappa_theory,
        "margin_worst_deviation": float(dev),
        "settings": vars(args),
        "seconds": time.time() - t0,
    }
    (ROOT / "data").mkdir(exist_ok=True)
    (ROOT / "data" / "validate_points.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote data/validate_points.json  ({out['seconds']:.1f}s)")

    _figure(out)


def _figure(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kap = np.array(out["kappas"])
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.4))

    # --- capacity vs margin, three routes -----------------------------------------------------
    kk = np.linspace(-0.2, 1.15, 300)
    ax[0].plot(kk, analytic.point_capacity(kk), "-", lw=2, color="0.25",
               label="closed form $\\alpha_0(\\kappa)$")
    est = np.array(out["estimator"])
    rel = np.array(out["estimator_rel_sem"])
    ax[0].errorbar(kap, est, yerr=est * rel * 1.96, fmt="o", ms=6, capsize=3,
                   color="#1f77b4", label="replica estimator (95% MC)")
    ax[0].axhline(2.0, ls=":", color="0.6", lw=1)
    ax[0].set_xlabel("margin $\\kappa$")
    ax[0].set_ylabel("capacity $\\alpha$")
    ax[0].set_title("Closed form vs Monte Carlo estimator")
    ax[0].legend(frameon=False, fontsize=8)

    # --- finite size: separability curves -----------------------------------------------------
    alphas = np.array(out["alphas"])
    cmap = plt.get_cmap("viridis")
    for i, (N, frac) in enumerate(zip(out["Ns"], out["frac_sep"])):
        c = cmap(i / max(len(out["Ns"]) - 1, 1))
        ax[1].plot(alphas, frac, "o-", ms=3.5, color=c, label=f"N={N}")
    ax[1].axvline(2.0, ls="--", color="#d62728", lw=1.5, label="theory $\\alpha_c=2$")
    ax[1].axhline(0.5, ls=":", color="0.6", lw=1)
    ax[1].set_xlabel("load $\\alpha = P/N$")
    ax[1].set_ylabel("fraction separable")
    ax[1].set_title("Transition sharpens with N\n(finite-size smearing, not noise)", fontsize=10)
    ax[1].legend(frameon=False, fontsize=8)

    # --- finite size: extrapolation -----------------------------------------------------------
    Ns = np.array(out["Ns"], dtype=float)
    thr = np.array(out["thresholds"])
    x = 1.0 / Ns
    ax[2].plot(x, thr, "o", ms=7, color="#1f77b4")
    ex = out["extrapolation"]
    xs = np.linspace(0, x.max() * 1.05, 50)
    ax[2].plot(xs, ex["intercept"] + ex["slope"] * xs, "-", color="0.35", lw=1.5,
               label=f"fit: {ex['intercept']:.3f} {ex['slope']:+.2f}/N")
    ax[2].axhline(2.0, ls="--", color="#d62728", lw=1.5, label="theory $\\alpha_c=2$")
    ax[2].plot([0], [ex["intercept"]], "*", ms=16, color="#1f77b4", zorder=5)
    ax[2].set_xlabel("$1/N$")
    ax[2].set_ylabel("measured $\\alpha_c$")
    ax[2].set_title("Extrapolation to the thermodynamic limit", fontsize=10)
    ax[2].legend(frameon=False, fontsize=8)

    # --- margin scaling collapse ---------------------------------------------------------------
    ma = np.array(out["margin_alphas"])
    kt = np.array(out["margin_kappa_theory"])
    aa = np.linspace(ma.min() * 0.9, ma.max() * 1.05, 120)
    ax[3].plot(aa, [analytic.margin_for_capacity(a) for a in aa], "-", lw=2, color="0.25",
               label="theory: $\\alpha_0^{-1}(\\alpha)$")
    for i, (N, d) in enumerate(sorted(out["margin_collapse"].items(), key=lambda kv: int(kv[0]))):
        c = cmap(i / max(len(out["margin_collapse"]) - 1, 1))
        ax[3].errorbar(ma, d["mean"], yerr=d["sd"], fmt="o", ms=5, capsize=2, color=c,
                       label=f"N={N}")
    ax[3].set_xlabel("load $\\alpha = P/N$")
    ax[3].set_ylabel("$\\sqrt{N}\\,\\kappa^*$ (standardised margin)")
    ax[3].set_title("Margin units: the curves collapse across N\n"
                    "(without $\\sqrt{N}$ they disagree by orders of magnitude)", fontsize=10)
    ax[3].legend(frameon=False, fontsize=8)

    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Validation 1: point manifolds reproduce Cover's bound, and the margin "
                 "convention is verified by scaling collapse", fontsize=12)
    fig.tight_layout()
    (ROOT / "figures").mkdir(exist_ok=True)
    p = ROOT / "figures" / "01_validate_points.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"wrote {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
