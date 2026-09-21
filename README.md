# mancap — manifold capacity, reimplemented and validated

An implementation of **mean-field theoretic manifold analysis** (MFTMA): Chung, Lee &
Sompolinsky, *Classification and Geometry of General Perceptual Manifolds*,
[Phys. Rev. X 8, 031003 (2018)](https://doi.org/10.1103/PhysRevX.8.031003), and its
correlated-centers extension in Cohen, Chung, Lee & Sompolinsky, *Separability and geometry of
object manifolds in deep neural networks*,
[Nat. Commun. 11, 746 (2020)](https://doi.org/10.1038/s41467-020-14578-5).

Everything here is free and local. No paid API, no gated weights, no GPU required for the
current stage.

## The question

Given P sets of activation vectors — one cloud per category — how many such categories could a
single linear readout separate at once, under arbitrary ±1 labels? The answer is a load
α = P/N, the **manifold capacity**, and it is one number summarising how linearly usable a
representation is. It decomposes into an **anchor radius** R_M and an **anchor dimension** D_M,
which say *why* capacity is what it is: a representation can lose capacity by its categories
being large, or by their being spread over many directions, and those have different causes.

Isolated points give α = 2 (Cover 1965). Anything with extent gives less.

## Why reimplement it

The reference implementation ([schung039/neural_manifolds_replicaMFT](https://github.com/schung039/neural_manifolds_replicaMFT))
is correct but hard to deploy and slow:

| | Reference | mancap |
|---|---|---|
| Inner minimization | `cvxopt` on the primal, one call per Gaussian draw, in a Python loop | Active set on the **dual**, where the Gram matrix is independent of the draw |
| Speed (CPU, single thread) | — | **5–12× faster**, same numbers to 1e-12 |
| Dependencies | `autograd`, `cvxopt`, `pymanopt` (pre-rename API) | numpy + scipy |
| Frame reduction | thin QR to M rows | SVD to the manifold's true **rank** (see below) |
| Center correlation | `pymanopt` Stiefel optimization; gradient via a (P, P, N, K) tensor | self-contained Riemannian descent; closed-form gradient at O(P²K) |

The inner problem is: given a manifold's frame S and a Gaussian vector T, find the nearest V
that keeps every manifold point on the correct side of the margin,

```
F(T) = min ||V − T||²   s.t.   S_j · V ≥ κ  for every point j
α_M(κ)⁻¹ = ⟨ F(T) ⟩_T
```

Writing V = T + Sμ with μ ≥ 0 turns it into a nonnegative quadratic program

```
min_{μ ≥ 0}  μᵀGμ + 2μᵀ(SᵀT − κ1),      G = SᵀS
```

whose gradient is literally the constraint slack, and whose Gram matrix G does not depend on T —
so it is built once per manifold and reused for every draw. That is what makes the active-set
solver exact and fast, and what will make a batched GPU version possible.

## Validation

Capacity is a Monte Carlo average of the solution of an optimization problem, reported as a
single number with no error bars. It is very easy to produce a plausible wrong one. So the
package is organised around independent checks rather than around the estimator.

| Check | What it rules out | Status |
|---|---|---|
| `analytic.point_capacity` closed form vs quadrature | algebra error in the closed form | agrees to 1e-9 |
| Points end-to-end vs α₀(κ) | error in the estimator or the frame code | within Monte Carlo error |
| Active set vs SLSQP on the primal | solver bug (no shared algebra) | 1e-12 on F, 1e-14 on anchors |
| KKT conditions verified directly | any solver bug, without a second solver | exact |
| `inner.solve_ball` cone projection | error on a manifold that is not a point | closed form, no iteration |
| Two routes to F (objective, and via the anchor) | **wrong anchors** behind a right capacity | `consistency` < 1e-10 |
| Reference implementation, draw by draw | any deviation from the published method | **1e-12** on α, D = 1…12 |
| Gradient of the center-correlation cost vs finite differences | a wrong subspace that still looks plausible | < 1e-6 |
| Stiefel optimizer on a problem with a known optimum | optimizer bug | recovers top eigenvectors exactly |
| Subspace search against a *planted* subspace | a plausible-but-wrong subspace | exact rank, cosines > 0.95 |
| Chance-level (null) reduction on isotropic centers | inventing structure that is not there | abstains, K = 0 |
| **Direct simulation** | **a misapplied theory that all the above share** | see below |

The last row is the one that matters. Everything else validates the *implementation* of the
replica formulas against other implementations of the same formulas; all of it agrees perfectly
if the theory has been transcribed wrongly. `mancap.simulate` never evaluates a replica formula:
it builds the manifolds, assigns random labels, and actually searches for a separating
hyperplane, using the identity

> the best margin any unit-norm readout achieves is the **distance from the origin to the convex
> hull** of the labelled points,

so one geometric computation per dataset gives the margin exactly — no training, no learning
rate, no stopping criterion — and a single sweep over the load yields the whole α(κ) curve.

Measured: **α_c = 2.047 → 1.993 → 2.000 → 1.992** at N = 25, 50, 100, 200, extrapolating in 1/N
to **1.98** against Cover's 2. The transition visibly sharpens with N; that broadening is
finite-size smearing of a phase transition, not noise, which is why a single N would not be
enough to detect a few percent of error.

For manifolds **with extent**, segments are the right test — their convex hull is exact at
M = 2, so there is no sampling bias to confound the comparison:

| segment half-length | 0.10 | 0.30 | 0.60 | 1.00 |
|---|---|---|---|---|
| replica theory | 1.763 | 1.449 | 1.177 | 0.994 |
| direct simulation | 1.758 | 1.475 | 1.177 | 1.000 |

## Three findings that matter for applying this to real data

**0. The theory's margin κ is a *standardised* overlap, not a geometric one.** A simulation
measures `min_i y_i ⟨w, x_i⟩` with unit-norm readout and points — a quantity that shrinks as
1/√P. The theory's Gaussian field is its standardised version, so

```
κ_theory = √N · κ_measured
```

Without that factor the theory and a simulation disagree by orders of magnitude at nonzero
margin, and **the disagreement looks like a capacity error rather than a units error**. Verified
by scaling collapse rather than by derivation: rescaling κ* by √N makes the curves at
N = 50, 100, 200, 400 fall on top of each other and on α₀⁻¹, from α = 0.3 (κ = 1.53) to α = 1.9
(κ = 0.03). This is exactly the class of error that agreement between solvers can never catch,
and it is the reason the direct simulation exists.

**1. Finite sampling inflates capacity, severely, and worse in high dimensions.** The inner
problem sees a manifold only through the convex hull of its sampled points, so a ball sampled
with M points is really an inscribed polytope — smaller, easier to separate. This is one-directional
(verified at machine precision: F_sampled ≤ F_exact for every single draw) and it closes slowly:

| | M = 10 | M = 30 | M = 100 | M = 300 | M = 1000 |
|---|---|---|---|---|---|
| D = 2 | +5% | +1% | +3% | +5% | +2% |
| D = 5 | +32% | +19% | +9% | −1% | +3% |
| D = 20 | +163% | +125% | +99% | +80% | **+64%** |

At D = 20, a thousand samples per manifold still overestimates capacity by 64%, and the anchor
dimension is recovered as 10.0 rather than 20. (The D = 2 row is Monte Carlo noise around zero,
not bias: ten points already cover a circle's hull.) A concept
manifold in a language model's residual stream is high dimensional, and a realistic dataset
supplies tens to hundreds of prompts per concept. **Capacity numbers from real data are not
comparable to theory unless the sample count is swept**, and the honest deliverable is the trend
in M, not a single number.

**2. The frame must be reduced to the manifold's rank, not to M.** For M sample points the
offsets have rank at most M−1, so the reference's QR reduction to M rows always leaves at least
one direction in which the manifold has zero extent. α and R_M are provably unaffected by that
padding (the padded row of S is zero, so that coordinate of V is unconstrained), but **D_M is
not**: on a segment, which must have D_M = 1 exactly, the rank-reduced frame gives 1.000000 and
a frame padded by one direction gives 0.8115 — matching the predicted 2(2/π)² = 0.8106. The bias
is worst when M is small relative to the true dimension, i.e. exactly the regime real data lands
in. `reduce_dim="qr"` reproduces the reference's behaviour for comparison against published
numbers.

**3. The correlated-centers correction works, and it is not optional.** The theory assumes the P
category centers are in general position; real representations concentrate them in a few
directions. `mancap.centers` reimplements the correction — find the low-rank subspace carrying
the shared structure, project it out — with a closed-form gradient (O(P²K) instead of a
(P, P, N, K) tensor) and a self-contained Stiefel optimizer, so `pymanopt` is not needed.

Whether the correction makes the theory *predict reality* is a separate question from whether
the optimizer finds the subspace, and only simulation can answer it. Planting a rank-3 shared
component in the centers and sweeping its strength (P = 60, N = 120):

| planted strength | 0.00 | 0.40 | 0.70 | 0.85 | 0.95 |
|---|---|---|---|---|---|
| mean \|cos\| between centers | 0.07 | 0.11 | 0.25 | 0.36 | 0.44 |
| theory, uncorrected | 1.189 | 1.194 | 1.191 | 1.188 | 1.189 |
| theory, corrected | 1.189 | 1.120 | 1.013 | 0.856 | 0.605 |
| **direct simulation** | **1.172** | **1.160** | **1.025** | **0.886** | **0.626** |
| selected K (argmin would give) | 0 (10) | 7 (9) | 3 (10) | 3 (8) | 3 (8) |

The uncorrected prediction is **flat** — it cannot see the correlation at all — while the true
capacity falls by half. Mean absolute error against simulation: **0.216 uncorrected → 0.024
corrected**, a 9× improvement. At zero correlation the routine correctly declines to project
anything, so the corrected and uncorrected columns coincide there by construction.

### Choosing K is where this goes wrong, and the obvious rule is the wrong one

Residual correlation decreases almost monotonically in K — there is always one more direction
whose removal helps a little — so taking the outright **argmin**, which is what the reference
implementation does, systematically over-estimates the rank: on data with a *planted rank 3* it
selects K = 8–10, and at zero planted correlation it still selects 10 (see the last row above).
Over-projection is not free — every removed direction is one the classifier could have used — so
it biases capacity **downward**. With argmin the corrected column undershot the simulation at
every strength and the mean error was 0.063; the two guards below cut that to 0.024.

Two guards, both on by default:

- **Elbow instead of argmin** (`k_tol`): take the smallest K within 5% of the best achievable
  reduction. This recovers the planted rank exactly (K = 3) at strength ≥ 0.7, with
  principal-angle cosines 0.95–0.995 against the planted basis.
- **A measured null** (`n_null`): fitting K directions to P centers removes correlation whether
  or not any is there — about 10% of the baseline for P = 60 isotropic centers, which is right at
  any plausible fixed threshold. `centers.null_reduction` runs the identical search on matched
  isotropic centers and returns the chance level, so only reductions above it count as evidence.
  With it, the strength-0 case correctly returns **K = 0 and projects nothing**.

`k_tol=0, min_reduction=0` reproduces the reference's behaviour for comparison with published
numbers. Always inspect `by_K`, `K_argmin` and `reduction` before trusting a corrected capacity.

## Layout

```
mancap/
  inner.py      the inner minimization: active set on the dual, SLSQP, closed-form ball
  capacity.py   α_M, R_M, D_M from the inner solutions; how to combine manifolds
  frames.py     raw activations → (D+1)-frames; center-correlation diagnostics
  synth.py      synthetic manifolds with known geometry: points, balls, segments, rings, ellipsoids
  centers.py    the correlated-centers correction: cost, closed-form gradient, Stiefel optimizer
  analytic.py   closed forms, and the margin unit convention
  simulate.py   direct simulation of the separability threshold
scripts/
  00_crosscheck_reference.py       draw-by-draw comparison against the published code
  00b_benchmark_reference.py       timing comparison
  01_validate_points.py            closed form vs estimator vs direct simulation
  02_validate_balls_and_segments.py  exact balls, sampling bias, and a theory-free threshold
  03_validate_correlated_centers.py  does the correction make theory match simulation?
tests/                             82 tests
```

## Setup

```powershell
conda create -n mancap -c conda-forge python=3.12 numpy scipy matplotlib pytest scikit-learn tqdm -y
conda activate mancap
pip install -e D:\dev\manifold-capacity --no-deps
pytest                              # 82 tests, ~2.5 min
```

Run the validations (each writes JSON to `data/` and a figure to `figures/`):

```bash
python scripts/01_validate_points.py
```

```bash
python scripts/02_validate_balls_and_segments.py
```

```bash
python scripts/03_validate_correlated_centers.py
```

Add `--quick` to either for a fast smoke run. The reference cross-check needs its own
environment; see the docstring of `scripts/00_crosscheck_reference.py`.

## What is not done yet

- **Correlated centers at realistic scale.** The correction is implemented and validated against
  simulation (finding 3 above), but only at P ~ 60 manifolds in N ~ 120 dimensions. Real data
  will have far larger N and a center-correlation structure that is not a clean planted subspace.
  `frames.center_correlation` reports the size of the problem and `CenterSubspace.by_K` shows
  whether low-rank structure exists at all; look at both before trusting a corrected number.
- **Batched GPU solver.** The dual formulation is designed for it — only `SᵀT` depends on the
  draw — but the current solver is a per-draw active set in numpy.
- **Application to language model activations.** Deliberately last. Note the caveat above: the
  replica result is a P, N → ∞ limit, so with N = 2304 and a handful of concepts the number is
  an extrapolation rather than a measurement. Sizing that properly is the first task of that
  stage.

## References

- Chung, Lee & Sompolinsky (2018), [PRX 8, 031003](https://doi.org/10.1103/PhysRevX.8.031003) — the theory; capacity Eqs. 16–17, radius Eq. 28, dimension Eq. 29.
- Cohen, Chung, Lee & Sompolinsky (2020), [Nat. Commun. 11, 746](https://doi.org/10.1038/s41467-020-14578-5) — correlated centers, and the DNN application.
- Chung & Abbott (2021), *Neural population geometry: an approach for understanding biological and artificial neural networks*, [Curr. Opin. Neurobiol. 70, 137](https://doi.org/10.1016/j.conb.2021.10.010) — the overview.
- Cover (1965), *Geometrical and statistical properties of systems of linear inequalities* — α = 2.
- Reference implementation: [schung039/neural_manifolds_replicaMFT](https://github.com/schung039/neural_manifolds_replicaMFT).
