# mancap: manifold capacity, reimplemented and validated

A small, self-contained re-implementation of **mean-field theoretic manifold analysis** (MFTMA)
from

> Chung, Lee & Sompolinsky, *Classification and Geometry of General Perceptual Manifolds*,
> [Phys. Rev. X 8, 031003 (2018)](https://doi.org/10.1103/PhysRevX.8.031003)
>
> Cohen, Chung, Lee & Sompolinsky, *Separability and geometry of object manifolds in deep neural
> networks*, [Nat. Commun. 11, 746 (2020)](https://doi.org/10.1038/s41467-020-14578-5)

written for learning and experimenting. It is not the authors' code and is not affiliated with
them; the reference implementation is
[schung039/neural_manifolds_replicaMFT](https://github.com/schung039/neural_manifolds_replicaMFT).

## What it does

Given P clouds of activation vectors, one per category, *manifold capacity* α = P/N measures how
many categories per dimension a single linear readout can separate under arbitrary labels. It
decomposes into an anchor **radius** R_M and anchor **dimension** D_M, which say why capacity is
what it is. Isolated points give α = 2 (Cover 1965); anything with extent gives less.

`mancap` computes α, R_M and D_M from raw activations, including the correlated-centers
correction of Cohen et al., using only numpy and scipy.

## Additions beyond the reference implementation

- **Faster exact solver.** The inner minimisation is solved as a nonnegative QP on its dual,
  whose Gram matrix does not depend on the Gaussian draw, with an active-set method: 5-12x faster
  on CPU and identical to the reference draw by draw (to 1e-12).
- **No legacy dependencies.** The correlated-centers correction uses a closed-form gradient and a
  self-contained Stiefel optimiser instead of `cvxopt`, `autograd` and old `pymanopt`.
- **Rank-based frame reduction**, which keeps D_M unbiased: a segment gives D_M = 1.000 rather
  than 0.81 with the reference's padding (`reduce_dim="qr"` reproduces the reference).
- **Guarded choice of the correlated-centers rank K**: an elbow rule plus a measured null, so
  uncorrelated centers give K = 0 rather than an over-projected subspace.
- **Validation against direct simulation.** `mancap.simulate` never evaluates a replica formula:
  the best achievable margin is the distance from the origin to the convex hull of the labelled
  points, so the theory can be checked against the actual separability threshold.

## Validation

| Check | Result |
|---|---|
| Points: estimator vs closed form α₀(κ) | within Monte Carlo error |
| Points: direct simulation, extrapolated in 1/N | α_c = 1.98 (theory: 2) |
| Segments, half-length 0.1 / 0.3 / 0.6 / 1.0: theory | 1.763 / 1.449 / 1.177 / 0.994 |
| Segments: direct simulation | 1.758 / 1.475 / 1.177 / 1.000 |
| Correlated centers: mean error against simulation | 0.216 uncorrected, 0.024 corrected |
| Active set vs SLSQP; KKT conditions; gradient vs finite differences | agree to 1e-6 or better |

Two findings that matter when applying this to real data:

- **The theory's margin is standardised**: κ_theory = √N · κ_measured. Without that factor,
  theory and simulation disagree by orders of magnitude at nonzero margin.
- **Finite sampling inflates capacity, badly in high dimensions.** A manifold sampled with M
  points is seen as their convex hull; for a 20-dimensional ball even M = 1000 overestimates
  capacity by 64%. Report the trend in M rather than a single number.

![Validation with points](figures/01_validate_points.png)
![Manifolds with extent](figures/02_validate_balls_segments.png)
![Correlated centers](figures/03_validate_correlated_centers.png)

## Setup

```bash
git clone git@github.com:shouvik-majumder/manifold-capacity.git
cd manifold-capacity
pip install -e ".[sim,plot,dev]"
pytest                                  # 82 tests, a few minutes
```

## Usage

```python
import numpy as np
import mancap

# Xs: one (N features, M samples) array per category
rng = np.random.default_rng(0)
Xs = [rng.standard_normal((50, 20)) * 0.1 + rng.standard_normal((50, 1)) for _ in range(10)]

frames = mancap.build_frames(Xs)
results = [mancap.analyze_manifold(f, rng=rng) for f in frames.frames]
print(mancap.combine(results), results[0].radius, results[0].dimension)
```

Validation scripts (each writes JSON to `data/` and a figure to `figures/`; `--quick` for a fast
run):

```bash
python scripts/01_validate_points.py
python scripts/02_validate_balls_and_segments.py
python scripts/03_validate_correlated_centers.py
```

`scripts/00_crosscheck_reference.py` compares against the reference implementation draw by draw;
it needs a separate environment, described in its docstring.

## Layout

```
mancap/inner.py      inner minimisation: dual active set, SLSQP, closed-form ball
mancap/capacity.py   α_M, R_M, D_M from the inner solutions
mancap/frames.py     raw activations to manifold frames; center-correlation diagnostics
mancap/centers.py    correlated-centers correction
mancap/simulate.py   direct simulation of the separability threshold
mancap/synth.py      synthetic manifolds with known geometry
mancap/analytic.py   closed forms and the margin convention
```

## Limitations

The replica result is a P, N → ∞ limit, and the correlated-centers correction has been validated
only at P ≈ 60, N ≈ 120 with a planted low-rank structure.

## License

MIT; see [LICENSE](LICENSE).
