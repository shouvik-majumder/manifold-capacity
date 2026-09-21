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
| Center correlation | Stiefel optimization, 20000 iterations per candidate rank | not yet implemented |

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

Measured: **α_c = 2.03 → 2.01 → 2.01 → 2.02** at N = 25, 50, 100, 200, extrapolating in 1/N to
**2.03** against Cover's 2. The transition visibly sharpens with N; that broadening is
finite-size smearing of a phase transition, not noise, which is why a single N would not be
enough to detect a few percent of error. For manifolds *with extent*, segments (whose convex
hull is exact at M = 2, so there is no sampling bias to confound the comparison) agree with
theory to a few percent across half-lengths 0.1 to 1.0.

## Two findings that matter for applying this to real data

**1. Finite sampling inflates capacity, severely, and worse in high dimensions.** The inner
problem sees a manifold only through the convex hull of its sampled points, so a ball sampled
with M points is really an inscribed polytope — smaller, easier to separate. This is one-directional
(verified at machine precision: F_sampled ≤ F_exact for every single draw) and it closes slowly:

| | M = 10 | M = 30 | M = 100 | M = 300 | M = 1000 |
|---|---|---|---|---|---|
| D = 2 | ~0% | ~0% | ~0% | ~0% | ~0% |
| D = 5 | +23% | +20% | +4% | +4% | +7% |
| D = 20 | +155% | +137% | +91% | +79% | **+70%** |

At D = 20, a thousand samples per manifold still overestimates capacity by 70%. A concept
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

## Layout

```
mancap/
  inner.py      the inner minimization: active set on the dual, SLSQP, closed-form ball
  capacity.py   α_M, R_M, D_M from the inner solutions; how to combine manifolds
  frames.py     raw activations → (D+1)-frames; center-correlation diagnostics
  synth.py      synthetic manifolds with known geometry: points, balls, segments, rings, ellipsoids
  analytic.py   closed forms the pipeline must reproduce
  simulate.py   direct simulation of the separability threshold
scripts/
  00_crosscheck_reference.py       draw-by-draw comparison against the published code
  00b_benchmark_reference.py       timing comparison
  01_validate_points.py            closed form vs estimator vs direct simulation
  02_validate_balls_and_segments.py  exact balls, sampling bias, and a theory-free threshold
tests/                             62 tests
```

## Setup

```powershell
conda create -n mancap -c conda-forge python=3.12 numpy scipy matplotlib pytest scikit-learn tqdm -y
conda activate mancap
pip install -e D:\dev\manifold-capacity --no-deps
pytest                              # 62 tests, ~1 min
```

Run the validations (each writes JSON to `data/` and a figure to `figures/`):

```bash
python scripts/01_validate_points.py
```

```bash
python scripts/02_validate_balls_and_segments.py
```

Add `--quick` to either for a fast smoke run. The reference cross-check needs its own
environment; see the docstring of `scripts/00_crosscheck_reference.py`.

## What is not done yet

- **Correlated centers.** The theory assumes the P manifold centers are in general position.
  Real representations badly violate this — a few directions carry most of the between-category
  variance — and the reference corrects for it with a Stiefel-manifold optimization that finds
  and projects out the low-rank center structure. Until that is implemented and validated,
  capacity from real data with correlated centers is biased. `frames.center_correlation`
  measures the size of the problem (mean |cos| between centers, participation ratio, components
  for 95% of center variance) so it is visible rather than assumed away. Synthetic validation
  runs in the uncorrelated regime by construction, where no correction is needed.
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
