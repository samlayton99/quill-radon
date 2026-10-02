# Direct Radon PDE construction (Codex)

The latest implementation is the **equation-declared residual solver**, documented
in [the general solver checkpoint](../../docs/quill_general_solver_checkpoint.md)
and [the API](solver/README.md). It uses constructed QUILL product features and
one matrix-free nonlinear coefficient solver without named-PDE dispatch. It
does solve for coefficients numerically; the historical no-readout-solve claim
below applies only to the original scripts in that section. Current precision
studies are `general_ns4d_precision.py`, `general_ns4d_sparse_precision.py`,
and `general_ns_polynomial_precision.py`. Reference-only capacity diagnostics
in `general_ns4d_floor.py` are explicitly separate from PDE solves.

The consolidated explanation, measured comparisons, limitations, and evidence
index are in [the QUILL PDE checkpoint](../../docs/quill_pde_checkpoint.md).

Results, configurations, figures and numerical evidence are in
`results/checkpoint_F_applications/expF19_radon_direct_pde/`.
The substantive findings are delivered in the conversation; the files are
reproducibility records.

Run from the repository root with `.venv/bin/python` and
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1`.

- `linear.py`: fixed two-input advection/Airy networks; time-dependent heat readouts.
- `linear_numerical.py`: evolve sampled initial Radon profiles numerically; no interior solution oracle.
- `linear_wave.py`: construct a fixed four-input network for the full-space 3D wave IVP.
- `linear_verify.py`: independent ordinary-MLP/autograd verification.
- `nonlinear_pde.py`: Cole–Hopf Burgers and dealiased 3D Taylor–Green Navier–Stokes evolution, followed by explicit ridge-to-tanh encoding.
- `nonlinear_validation.py`: finer NS reference, time refinement, exact positive controls, and center-bandwidth aliasing checks.
- `scaling.py --mode all`: dimension and center/direction allocation sweeps.
- `scaling_structured.py`: high-order sphere quadrature in four/five dimensions.
- `scaling_adapted.py`: known-metric directional adaptation control.
- `scaling_controls.py`: equal-budget allocation and leading-readout controls.
- `inverse.py`: recover six diffusion coefficients from measurements; regularized backward heat.

None of the original forward scripts listed above trains a network or solves a least-squares readout. The
inverse diffusion experiment explicitly uses nonlinear least squares for six
physical parameters; its readouts are always constructed.

The forward algorithm has two separate jobs: solve/evolve the PDE, and encode
its directional profiles. Constant-coefficient linear PDEs admit independent
profile propagators. Generic Navier–Stokes requires coupled numerical evolution;
the current implementation uses a Fourier Galerkin solver. Its spatial neural
snapshots are not a single fixed four-input network. The wave experiment is.

These experiments cover smooth whole-space IVPs and periodic fluid flow.
They do not establish arbitrary-boundary, turbulent, or unrestricted
high-dimensional PDE performance. Angular and center errors, numerical evolution
error, and inverse-data instability are measured separately.

## QUILL correction and allocation follow-up

The original runs fixed lambda at .25 and used long uncorrected halo intervals.
The additive `quill_review/` results supersede that geometry-selection protocol:

- `quill_boundary.py`: current September 26 construction, including the effective
  boundary moments and stable scaled partial fractions. Existing halo readouts
  receive an explicit correction; no least squares or new neurons are used.
- `quill_sweep.py`: 3D Gaussian lambda sweeps, independent angular/center floors,
  heldout allocation predictions, and lambda selection for value versus
  Laplacian accuracy. Here N is interior centers, h=2/(N-1), and each side has
  ceil(sqrt(N)) halo centers. Count the full cost M*(N+2*ceil(sqrt(N))).
- `quill_ns_sweep.py`: scalar and saved Navier–Stokes encoding checks, including
  first/second derivatives and boundary points. This reuses the Fourier states;
  it does not implement neural time evolution.

Authoritative history: `docs/appendix_notes/paper_v5_replacements/organized_v7_package/sections/7_appendix/01_Construction/09_26_sl.tex`,
`docs/lambda_theorem_compatibility/choosing_optimal_lambda/`, and
`results/checkpoint_H_highdim/expH05_direction_cliff_2d/expH05_results.md`.
The paper and standalone boundary diagnostic use interval-cell counts; the
allocation and NS integrations explicitly use interior-center counts instead.
Reported relative norms are empirical sample norms, not certified continuous
domain bounds. The empirical max-of-floors rule is tested, not assumed exact.

## Native dynamics follow-up

`native_dynamics/` contains new experiments that compute their evolution from
the initial condition and PDE itself. They do not load the earlier solutions.
Conventional solvers are run separately afterward as error/cost references.

- `native_tanh.py`: periodic nonlinear Burgers with actual corrected QUILL
  tanh inference and analytic derivatives at every RK stage. The update is
  theta_dot = E A F(S theta). A is explicit trigonometric quadrature; E encodes
  the entire trigonometric basis with QUILL. No training or readout solve.
  This is modified spectral evolution in neural coordinates, with O(PQ)
  dense work/storage, not a demonstrated speedup or universal PDE construction.
- `native_burgers.py`: an independent cubic B-spline/ReLU^3 control. Explicit
  geometric convolution maps PDE values to coefficient increments without FFT
  or a linear solve. Classical spline collocation, second-order in this test.
- `native_ns_pilot.py`: short smooth 3D Taylor–Green evolution through actual
  QUILL fields, using explicit Fourier closure and the Leray pressure projector.
  Cached neural basis evaluations reduce repeated inference cost. It is a
  modified pseudospectral method, with independent reference refinements.
- `native_ns_refine.py`: add the K=5, 577-direction native Navier–Stokes case.
- `native_taylor.py`: construct the native K=5 Navier–Stokes time polynomial
  directly from its coefficient recurrence, without time marching. Independent
  RK references are used afterward. This is a classical Taylor method for the
  finite spatially projected polynomial ODE, with a local convergence scope.

The original complex-shift QUILL density cannot be applied directly to its
own tanh network: the evaluation points lie on that network's complex poles.
The real-data finite-basis bridge in the native experiments avoids this trap.
Readout updates encode the PDE increment, never repeatedly re-encode the full
state. These methods still integrate an ODE numerically and retain spatial
resolution, stability, boundary, and nonlinear-coupling requirements.

## PINN capability tests

New results are under `capabilities/`. Run each script with the same single-thread
Python invocation above; references do not enter the forward construction.

- `native_inverse.py`: 24 synthetic Burgers measurements infer viscosity and two
  initial-condition amplitudes. Actual QUILL fields are evolved; a 24-by-3
  sensitivity Jacobian is used in the outer physical-parameter least-squares
  problem. Includes five seeds at each nonzero noise level, withheld space/time
  tests, tangent checks, and an unidentifiable initial-only observation control.
- `native_elliptic.py`: nonlinear 2D Dirichlet problem on a square, with a sine
  basis explicitly represented by QUILL ridges. Updates use the actual neural
  residual and scalar division by reference shifted-Laplacian eigenvalues.
  No readout fit or matrix solve. A nonlinear iteration
  failure control removes the shift. Lambda is swept against second-derivative
  encoding error, with R=ceil(sqrt(N)) halos per side and corrected boundaries.

These are numerical PDE/inverse algorithms, not a universal closed-form Radon
solution. The elliptic reference uses higher-resolution classical sine
discretization. Projected residual, off-grid PDE residual, solution error, and
boundary error are separate metrics. `coefficient_updates` excludes the final
convergence check; `residual_evaluations` includes it. Saved validation also
reports the local iteration Jacobian spectra and a forbidden-solve/FFT guard.

## General solver follow-up

`solver/` now provides explicit problem declarations and a common `solve`
dispatcher. Its [interface and reproduction record](solver/README.md) identifies
the supported equations, architecture choices, numerical solves, failure
statuses, inverse uncertainty assumptions, and maintained DeepXDE comparison.
New experiments are `general_*.py`, with metrics and figures under
`general_solver/` in the experiment results directory. This follow-up extends
the earlier scope to a curved annulus, mixed boundaries, high diffusion
contrast, stiff fronts, conservative shocks, and blow-up diagnostics.
