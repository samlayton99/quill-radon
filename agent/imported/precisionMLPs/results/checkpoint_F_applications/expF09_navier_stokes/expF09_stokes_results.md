# expF09 Stage A -- Stokes via frozen-geometry collocation

**Status:** drafted (single seed). 12-cell W/lambda sweep. First stage of the
Navier-Stokes program.

## TL;DR

- The multi-field `solve_system` core solves the **primitive velocity-pressure
  Stokes system** on `[-1,1]^2` to the **fp64 floor**: best velocity rel L2
  **7.7e-14** (W=1600, lambda=0.2), with the incompressibility residual
  `max|div u|` at **8.2e-11** and pressure at 3.4e-11 -- zero training, one
  min-norm least-squares solve.
- Velocity converges spectrally in width: `9.4e-12 (W=576) -> 3.6e-13 (1024) ->
  7.7e-14 (1600)`. `lambda=0.2` is the sweet spot; `lambda=0.3` stalls at
  ~1e-10 (too-wide kernels lose conditioning on the coupled system).
- **Pressure recovers a few orders above velocity** (3e-11 vs 8e-14) -- expected,
  it is the constraint variable -- but is still excellent; no inf-sup pathology
  appeared with the min-norm SVD truncation and a one-row pressure gauge.
- This validates the shared `solve_system` core (multi-field coupled collocation
  + gauge) for the rest of the program: Stage B (steady NS / Kovasznay), C
  (unsteady / Taylor-Green), D (FNO benchmark).

## Question

Does the training-free collocation-lstsq recipe extend from scalar PDEs to a
**coupled multi-field system with a constraint** -- incompressible Stokes,
`-nu lap u + grad p = f`, `div u = 0` -- and still reach the fp64 floor? This is
the linear precursor to Navier-Stokes and the first test of the shared solver.

## Experiment design

- **Core:** `core_system.solve_system(fields, equations, W, lambda)` -- each of
  `u, v, p` shares the frozen Radon geometry + degree-3 poly tail; equations
  (momentum-x, momentum-y, continuity, velocity BCs, pressure gauge) are field-
  block collocation constraints stacked into one min-norm lstsq (rcond 1e-13),
  each block scaled to O(1). Only <=2nd derivatives. Scalar primitives vendored
  from expF08.
- **Problem:** manufactured divergence-free field from `psi = sin(pi x) sin(pi y)`:
  `u* = pi sin(pi x) cos(pi y)`, `v* = -pi cos(pi x) sin(pi y)`
  (`u*_x + v*_y = 0` exactly), `p* = cos(pi x) cos(pi y)`, `nu = 1`. Forcing
  `f = -nu lap u* + grad p*`, all hand-coded and FD-verified. Error vs the exact
  field on a fresh 151^2 grid; `max|div u|` at 5000 fresh interior points.
- **Sweep:** W in {576, 1024, 1600, 2304} x lambda in {0.2, 0.25, 0.3},
  8000 interior collocation points, seed 0.
- **Reproduce:**
  `uv run --extra dev python experiments/expF09_navier_stokes/run_stokes.py`
  (`--smoke`, `--plot`).

## Results

Best-lambda velocity convergence (full grid in `data.json`,
`stokes_convergence.png`):

| W | velocity rel L2 | pressure rel L2 | max \|div u\| | best lambda | t_solve |
|---|---|---|---|---|---|
| 576 | 9.4e-12 | 1.7e-9 | 5.8e-9 | 0.20 | 10 s |
| 1024 | 3.6e-13 | 1.1e-10 | 2.2e-10 | 0.20 | 101 s |
| 1600 | **7.7e-14** | 3.4e-11 | 8.2e-11 | 0.20 | 165 s |
| 2304 | 1.1e-13 | 4.5e-11 | 1.1e-10 | 0.20 | 148 s |

- **Velocity hits the fp64 floor** (~1e-13) by W=1600; the slight uptick at
  W=2304 is the usual gamma-scaling roundoff at large width (seen throughout the
  repo) -- the smallest width that resolves is best.
- **`lambda` is the dominant knob**: at fixed W=1600, `lambda=0.2 -> 7.7e-14`,
  `0.25 -> 8.2e-13`, `0.3 -> 1.4e-10`. Wider kernels lose conditioning on the
  coupled saddle-type system faster than on scalar problems.
- **Incompressibility** is satisfied pointwise to ~1e-10 at fresh points (not a
  fitted quantity -- the constraint rows enforce it in the collocation sense and
  it holds off-grid).

## Regime vs neural operators (accuracy-ceiling framing)

This is a **training-free, per-instance physics solve** -- it is handed the
equations and BCs and returns the solution; it does **not** learn a solution map
from data. The honest comparison against data-driven neural operators is
multi-axis:

| axis | data-driven NO (FNO/DeepONet) | this solve |
|---|---|---|
| accuracy (known PDE) | ~1e-2 typical | **7.7e-14** (fp64 floor) |
| training data | required | **none** |
| cost | train-once (hrs) + ~ms/inference | ~150 s / instance, no training |
| amortized over many inputs | **wins** | pays a solve each time |
| works with no known equation | **wins** | needs the PDE |

So on a *known* PDE this sets the accuracy ceiling (~12 orders below a trained
operator) at a per-instance cost of ~150 s; the neural operator's value is
amortized inference and the no-equation regime, not accuracy. The full
accuracy-vs-cost head-to-head on real benchmark instances lands at **Stage D**
(the FNO NS benchmark).

## Conclusions

1. **The recipe extends to coupled constrained systems.** Incompressible Stokes
   in primitive variables reaches the fp64 floor with the same frozen-geometry +
   one-lstsq machinery, only <=2nd derivatives, plus a pressure gauge.
2. **No inf-sup trouble surfaced** -- the min-norm SVD truncation handled the
   velocity-pressure pair; pressure recovers a few orders above velocity but is
   stable and accurate.
3. **The shared `solve_system` core is validated** for the nonlinear (Newton),
   unsteady (space-time), and benchmark stages that follow.

## Open questions / next

- **Stage B -- steady NS (Kovasznay).** Add `(u.grad)u` via Newton (expF06
  pattern); exact Kovasznay verifier; expect ~1e-7 nonlinear floor.
- **Pressure gauge choice** (point value vs zero-mean) and its effect on the
  pressure floor -- a small ablation.
- **Conditioning of the coupled system at large W** -- quantify why `lambda`
  tolerance narrows vs scalar problems.
