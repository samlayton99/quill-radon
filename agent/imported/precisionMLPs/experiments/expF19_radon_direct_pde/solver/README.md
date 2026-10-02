# Constructed neural PDE solver: supported interfaces

## Equation declarations: current general interface

`GeneralProblem` takes mathematical equations, a domain, and constraints.
The same damped Gauss–Newton / matrix-free LSMR engine handles scalar and vector
fields, arbitrary declared derivative multi-indices, nonlinear constitutive
laws, measurements and unknown physical parameters. It does not dispatch on
equation names. The [measured checkpoint](../../../docs/quill_general_solver_checkpoint.md)
records thirteen forward/inverse tests, a four-input transient Navier–Stokes
test, exact-polynomial controls, and limitations.

```python
import sympy as sp
from solver import ResidualDomain, GeneralProblem, Condition
from solver.general_symbolic import DifferentialResidual

x, y = sp.symbols('x y')
u = sp.Function('u')(x, y)
equation = DifferentialResidual(
    (x, y), (u,), -sp.diff(u, x, 2) - sp.diff(u, y, 2) + u**3 - 1,
)
boundary = DifferentialResidual((x, y), (u,), u)
problem = GeneralProblem(
    ResidualDomain([[-1, 1], [-1, 1]]), equation,
    [Condition('zero_boundary', boundary)],
)
solution = problem.solve(tolerance=1e-6)
# Always inspect solution.status and the fresh residual/constraint checks.
```

Initial conditions use `Condition(..., location='slice', axis=..., value=...)`.
Boundary selectors distinguish spatial walls from the initial/final coordinate
faces of a space-time box. Fixed observation points use `points=...`; unknown
parameters are declared to every residual object and initialized/bounded in
`GeneralProblem`. Optional symbolic compilation discovers derivative orders
and expands product rules; custom Torch callbacks remain available through
`ResidualDeclaration` / `ResidualBlock`. Weak integrals and inequalities use
explicit aggregation and `relation='le'/'ge'` in that lower-level interface.

The default features are corrected QUILL encodings of total-degree Legendre
products. This is a shared-bank/product architecture, not a flat Radon network.
Increasing total degree increases field coefficients independently of increasing
the tanh center count used to encode each one-dimensional bank. Both resolution
knobs must be reported. Actual tanh derivatives enter the residual; the
`polynomial` backend is a matched control, never silently substituted.

High precision: `solve_declared` automatically requests the engine's
`high_accuracy` mode when tolerance is below 1e-9. `feature_options` can set
`centers`, `lam`, `encoding_tolerance`, and `evaluation='anchored'`. The automatic
high-accuracy encoding gate is a worst-mode calibration policy, not a field
error guarantee. A failure to encode a finer basis can report
`construction_limit`. `solver_options` exposes Krylov budgets, bounded-block
preconditioning, defect correction and checkpoint callbacks. Resource limits
still apply unless explicitly increased. Solver residual, independent solution
error, and floating-point epsilon are different quantities.

`solver/general_sparse.py` supplies equation-independent coefficient support
selection, all-coordinate support growth, feature restriction and coefficient
embedding. Optional derivative-aware support scores use exact Legendre
derivative bounds. Discarded expansion bounds do not certify PDE solution
error; fresh residual and field checks remain necessary. Sparse precision
studies must initialize only from solved coefficients, never from reference
decompositions.

## Earlier specialized interfaces

The earlier research prototype uses declared equation classes. It constructs neural
features, then solves for physical expansion coordinates from the PDE. Nonlinear
evolution still requires numerical integration or nonlinear iteration. There is
no universal Radon formula that returns every nonlinear PDE solution.

Run scripts from the repository root using `.venv/bin/python`. Import `solver`
from scripts in `experiments/expF19_radon_direct_pde`, or put that directory on
`PYTHONPATH`. Set `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
VECLIB_MAXIMUM_THREADS=1` for the reported one-thread protocol.

```python
import numpy as np
from solver import PeriodicEvolution, solve

problem = PeriodicEvolution(
    initial=lambda x: np.sin(x) + .2*np.cos(2*x),
    rhs=lambda t, x, u, ux, uxx, p: -u*ux + p['nu']*uxx,
    parameters={'nu': .05},
)
solution = solve(problem, t_end=.5, adaptive=True, tolerance=1e-6,
                 maximum_modes=128, rtol=1e-10, atol=1e-12)
values = solution.evaluate(np.linspace(-np.pi, np.pi, 101), .5)
neurons = solution.readouts(.5)
```

The forward example supplies only initial data and dynamics. The constructed
readouts are not fitted to a numerical reference solution. Its time-dependent
spatial snapshots are not one fixed spacetime MLP.

| Declaration | Supported equation/domain | Algorithm / representation |
|---|---|---|
| `PeriodicEvolution` | Scalar smooth periodic 1D, user RHS depending on values and first/second spatial derivatives | Actual QUILL fields, explicit quadrature, DOP853 or BDF |
| `ReactionDiffusion1D` | Periodic Allen–Cahn or quadratic reaction with diffusion | Actual QUILL + BDF; sampled physical-invariant guard defaults on |
| `BoxProblem` | 2D rectangular Dirichlet, positive constant diffusion, nonnegative cubic reaction | Shared QUILL banks, exact boundary factors, product gates, preconditioned residual iteration |
| `AnnulusProblem` | Positive smooth scalar diffusion and cubic reaction on an annulus; Dirichlet or inner Robin / outer Dirichlet | Polar coordinates, shared QUILL banks, product gates, Newton-CG with small PDE preconditioner |
| `RidgeLinear` | Whole-space constant-coefficient heat/advection, supplied ridge-decomposed initial data | Independent exact profile propagation followed by explicit QUILL encoding |
| `Conservation1D` | Burgers flux, periodic or outflow 1D | Conventional Rusanov finite volume + SSPRK3; field is derivative of a ReLU primitive, **not smooth QUILL** |

`solve(problem, **options)` dispatches explicitly; unsupported equations raise.
Return objects still differ by backend: smooth generic, box, annulus, and ridge
solves have `.evaluate`; specialized reaction/conservation return state/history
dictionaries. This is not yet a standardized production package.

For smooth QUILL encoding, N is interior centers, h is interior spacing, and
each side receives ceil(sqrt(N)) corrected halo centers. Generic periodic
`Resolution` chooses lambda from .15/.20/.25/.30 using second-derivative basis
encoding error. Box/domain examples use .2, informed by the preceding derivative
sweep; the specialized stiff harness uses .25. These are measured choices, not
claims of universal optimal bandwidth. Angular and within-direction resolution
remain separate error sources. Products/shared banks change the architecture
and avoid duplicating full ridge networks on tensor-structured domains.

## Status and validation

- Completing time integration is not a spatial error certificate. Inspect tail,
  amplitude, invariant, and refinement results. Generic adaptive evolution
  requires successive-resolution agreement, an independently sampled physical
  residual, and tighter time integration. `solve_verified` is also public;
  `solve(..., adaptive=True)` uses it. A negative control in
  `verification_study.py` shows why resolution agreement alone can miss an
  unresolved forcing component. Irregular temporal probes plus accepted-step
  samples avoid systematic aliasing of periodic forces at evenly spaced
  validation times. Time tolerance is a separate setting.
- Box and annulus solve checks include independent physical residual points.
  A converged projected nonlinear system can still be underresolved. None of
  the sampled diagnostics certifies a continuum error bound.
- `EvolutionSolution.residual` is a spatial semidiscrete residual, not a
  derivative through the time interpolant. Time error requires tolerance or
  step refinement.
- Amplitude stops do not establish the exact blow-up time. Positivity and
  maximum-principle checks are sampled, not guaranteed between samples.
- High-dimensional nonlinear sweeps in `general_dimensions.py` are explicitly
  **classical Galerkin controls plus neural compilation**, not native QUILL
  high-dimensional dynamics. Supplied-ridge linear solves are direct.

## Sparse measurements / inverse parameters

`calibrate(forward, values, initial, bounds=..., sigma=..., jacobian=...)` fits a
small physical parameter vector. It does not fit thousands of readout weights.
The returned Jacobian rank diagnoses local identifiability. Local Gaussian
covariance assumes known independent observation noise and a correct forward
model; it is disabled at active parameter bounds and rank deficiency. It is not
a Bayesian posterior, a model-discrepancy model, or a general unknown-field
inverse solver. Examples: `general_inverse_uq.py` and `native_inverse.py`.

The separate `ProfileForward`, `ProfileObservations`, `fit_profile` and
`fit_profile_refined` interfaces recover unknown periodic initial profiles for
viscous Burgers or heat. Their unknowns are a declared finite set of physical
profile coefficients, not the full neural readout. The study uses 17 unknowns
and 48 scalar measurements, a smoothness prior, and known noise to choose its
strength. An independent forward refinement at the inferred profile can
override an apparently successful data fit with `forward_underresolved`.
`fit_profile_refined` repeats the fit at higher forward resolution and returns
that final model along with the estimate, avoiding evaluation with a stale
coarse model. It does not guarantee identifiability of the earlier field.

## Adaptive directions and native 3D Navier–Stokes

`AdaptiveRidgeProblem` and `solve_adaptive_ridges` evolve smooth periodic 2D
reaction–diffusion equations. They select directions from a finite integer
frequency dictionary using explicit quadrature of the PDE increment. Each
active ridge receives the smallest tested center count passing independent
value and Laplacian encoding checks, with a bandwidth sweep and corrected
sqrt(N) halos. This is adaptive spectral selection through constructed neural
features, not continuous learned-angle optimization. Candidate analysis
arrays are still stored, so sparse output does not imply sparse discovery cost.

Use `rollback=True` to save a window's start state and replay that window
when significant omitted directions appear during it. Merely adding those
directions afterward can permanently lose their earlier contribution. The
replay count is bounded; exhaustion raises. The selection tolerance is an
increment budget, not a global error bound; independent cutoff, time and
physical-residual checks remain necessary. `adaptive_ridge_study.py` includes
matched fixed-neural and classical controls and this failure case.

`PeriodicNS` and `solve_ns_refined` support smooth incompressible flow on the
periodic cube [-pi, pi]^3. Supply initial velocity and an optional
`forcing(t, points)` returning an array of shape `(len(points), 3)`.
The solver uses shared one-dimensional QUILL banks and exact product gates,
with explicit quadrature and Fourier pressure projection. Actual constructed
values and derivatives enter every Runge–Kutta stage; there is no global
readout fit. This changes the earlier flat-ridge architecture. The low tanh
count must be reported alongside the growing tensor coefficient count.

Native NS checks the prescribed initial field on an independent shifted grid,
then checks physical pressure-projected residual, divergence and energy at
recorded trajectory states, as well as dt/2 agreement. Independent irregular
temporal force probes detect unresolved transients that disappear before the
final time. `solve_ns_refined` repeats complete runs
from the original initial condition and also checks successive field agreement.
Its memory estimate is not measured process RSS; the time budget is checked
between attempts, not enforced as a hard deadline. A `converged_sampled` result
is finite numerical evidence, not a continuum regularity or blowup certificate.

The September 2026 forced-NS paper is audited in
`adaptive_followup/moonshot_source_audit.json`. Its full corrected numerical
forcing evaluator is **not implemented**. `moonshot_geometry.py` is a separate
synthetic shrinking-vortex representation probe with prescribed scales, not a
PDE solve or reproduction of that paper. Fixed finite Galerkin resolution with
bounded smooth forcing cannot literally develop infinite speed while preserving
its energy identity; progressively concentrated solutions require resolution
or geometry adaptation.

## Reproduce

New outputs are in
`results/checkpoint_F_applications/expF19_radon_direct_pde/general_solver/`.

```sh
.venv/bin/python experiments/expF19_radon_direct_pde/general_api.py
.venv/bin/python experiments/expF19_radon_direct_pde/general_domains.py
.venv/bin/python experiments/expF19_radon_direct_pde/general_hard.py
.venv/bin/python experiments/expF19_radon_direct_pde/general_dimensions.py
.venv/bin/python experiments/expF19_radon_direct_pde/general_inverse_uq.py
.venv/bin/python -m pytest -q tests/test_quill_general_solver.py
```

The PINN baseline uses DeepXDE 1.15.0, PyTorch 2.4.1, float64, four tanh hidden
layers, Glorot initialization, exact Dirichlet output factors, 15,000 Adam steps
at 1e-3, and nominal 5,000 L-BFGS iterations. Actual completed library iteration
counts are recorded because the library can overshoot that nominal budget.
No reference values enter training. Two widths and two seeds are a bounded
comparison, not a best-possible PINN architecture/optimizer search.

For the recorded run the extra dependencies were isolated in a temporary
directory rather than changing the project environment:

```sh
UV_CACHE_DIR=/private/tmp/codex-quill-uv-cache uv pip install --python .venv/bin/python --target /private/tmp/codex-quill-deps --no-deps deepxde==1.15.0 scikit-optimize==0.10.2 pyaml==26.7.0
.venv/bin/python experiments/expF19_radon_direct_pde/general_baseline.py --seed 0 --width 32 --points 1024
.venv/bin/python experiments/expF19_radon_direct_pde/general_baseline.py --seed 1 --width 32 --points 1024
.venv/bin/python experiments/expF19_radon_direct_pde/general_baseline.py --seed 0 --width 64 --points 2048
.venv/bin/python experiments/expF19_radon_direct_pde/general_compare.py
```

Numerical references, compilation-only cases, manufactured forcing tests,
architecture changes, and measured timing scope are disclosed in the metrics.
All substantive results are also presented in the conversation.

The adaptive follow-up outputs are in the sibling `adaptive_followup/` results
directory. Reproduce using `verification_study.py`, `adaptive_ridge_study.py`,
`native_ns_extended.py`, `inverse_profile_study.py`, and `moonshot_geometry.py`
under this experiment directory. Each study records its numerical controls and
timing scope. These interfaces remain separate explicitly scoped entry points;
the common dispatcher does not infer an arbitrary PDE class.
