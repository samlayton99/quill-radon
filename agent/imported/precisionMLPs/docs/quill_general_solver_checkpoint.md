# A common constructed-feature PDE solver

October 1, 2026. This is the measured follow-up to
`quill_pde_checkpoint.md`. The conversation contains the substantive report;
this file preserves the implementation, definitions, evidence, and limits.

## What changed

The new path accepts equations and constraints rather than an equation-class
name. One residual engine handles all the new forward and inverse experiments.
It has no Burgers, diffusion, wave, plate, or Navier–Stokes dispatch. It does not
load a reference solution, numerically evolve a reference trajectory, or train
on interior solution labels in forward tests. This differs materially from the
earlier Fourier-evolution-then-encoding experiments, which remain documented.

The implementation **does numerically solve for expansion coefficients**. It
uses damped Gauss–Newton and matrix-free LSMR, plus small block factorizations.
It is not a formula that constructs arbitrary PDE solutions without solving.
The successful general basis is a product architecture built from corrected
QUILL one-dimensional feature banks, not the original flat Radon ridge network.

## Mathematical objects and algorithm

Let the bounded domain be Omega in R^d. Time, when present, is one of these d
coordinates. The unknown field u maps Omega to R^q. Optional unknown physical
parameters form a vector p in R^s. The caller declares residual equations

\[
F(x,u(x),D u(x),D^2 u(x),\ldots,p)=0,
\]

along with boundary/initial constraints, optional measurements and gauges.
The PDE and its intended solution conditions remain necessary problem data.

For degree m, the constructed scalar functions phi_j:Omega->R form a finite
feature bank with P=binomial(m+d,d) entries. Each is a total-degree product of
one-dimensional Legendre modes, with the modes explicitly encoded by corrected
QUILL tanh banks. The unknown coefficient array A lies in R^(P x q), and

\[
u_A(x)=\sum_{j=1}^{P}\phi_j(x) A_{j,:}.
\]

For a derivative multi-index alpha, the cached basis matrix B_alpha has one row
per collocation point and one column per feature:

\[
(B_\alpha)_{ij}=D^\alpha\phi_j(x_i),\qquad
D^\alpha u_A(x_i)=(B_\alpha A)_{i,:}.
\]

These are derivatives of the actual constructed tanh/product features, not
polynomial derivatives substituted while evaluating a different neural field.
N counts interior centers; each side has ceil(sqrt(N)) corrected halos. N and
lambda are selected from an encoding sweep against independent per-mode value
and derivative checks through the highest derivative order requested by the
equation. This calibration sees the basis, not the unknown PDE solution.
The default normalized per-mode scaled tolerance is 1e-8; physical-coordinate
derivative scaling is applied afterward. Higher derivative absolute errors can
therefore be appreciable even with small scaled calibration errors.

Stack all sampled equations and constraints into a residual vector
r(z) in R^K, where z=(vec(A),p) in R^(Pq+s). The local Jacobian J maps
coefficient/parameter increments to residual increments. At a nonlinear step,
the engine approximately solves

\[
\min_\delta\;\|r(z)+J\delta\|_2^2+\mu\|\delta\|_2^2,
\]

in scaled/preconditioned coordinates, followed by backtracking and parameter
bound projection. Torch differentiates each pointwise residual with respect
to its local field derivatives and parameters. Those local sensitivities and
B_alpha implement J times a vector and its adjoint. The global Jacobian and
global normal-equation matrix are not assembled. LSMR remains a **global
iterative coefficient solve**, not a collection of independent local solves.

The default starts with diagonal scaling. If Krylov iterations are expensive,
or the solve reaches above-target stationarity, it switches to regularized
feature-parity Gram blocks of at most 64 unknowns. These small Cholesky blocks
precondition the same equation-independent iteration. They are reused until
sensitivity changes warrant rebuilding. No inverse Laplacian or fluid pressure
projector is supplied.

The outer policy starts at low degree and embeds the best prior coefficients
at increasing degree. It checks fresh equation and constraint samples and
successive field agreement. If a degree jump exceeds the basis-cache budget,
it tries an intermediate degree. Time/resolution/memory limits are reported
as failures rather than successful PDE solutions. The checks are empirical;
they do not imply a continuum error theorem or uniqueness.

The cached basis derivative matrices can be larger than the hypothetical dense
residual Jacobian. Matrix-free Jacobian products are therefore not, by
themselves, evidence of a memory reduction. Their actual storage is reported.

## Interface

`solver/general_symbolic.py` supplies `DifferentialResidual`: symbolic scalar or
vector equations are expanded and compiled into differentiable Torch callbacks.
Derivative multi-indices are discovered automatically, including variable
coefficient product rules. Unknown physical parameters retain their gradients.
Strings and undeclared functions/variables are rejected. SymPy is imported only
when the optional symbolic frontend is used.

`GeneralProblem(domain, equation, conditions, parameter_initial=...,
parameter_bounds=...)` builds fresh collocation sets and calls the shared
policy. `Condition` supports boundary samples, interior samples, fixed data
points, coordinate slices for initial conditions, and selectors. For custom
sampling, normal derivatives, integral/weak forms, or non-symbolic constitutive
laws, `ResidualDeclaration` and `ResidualBlock` provide the lower-level API.
Aggregation precedes the inequality hinge for weak/entropy conditions.

Domains can be boxes or box-clipped implicit regions. Boundary sampling uses
projection and includes clipping faces. It is not certified uniform surface
quadrature or guaranteed discovery of tiny disconnected boundary pieces.

## Measured smooth/inverse results

All rows below use the same automatic policy: default degree ladder through
64, tolerance 1e-6, six interior samples per scalar feature with a minimum of
256, fresh checks, automatic preconditioning, and finite time/memory budgets.
The equation declarations differ in their physical conditions, not in solver
selection. The field errors below use an additional independent reference
audit after the solve. Local times include continuation but may benefit from
reusable construction caches and differ in concurrent machine activity; they
are not controlled speed comparisons.

| Problem | Independent relative field L2 | Local seconds | Unknowns |
|---|---:|---:|---:|
| Nonlinear diffusion | 1.40e-10 | 0.22 | 153 |
| Perforated implicit domain, variable anisotropy/cubic reaction | 9.13e-11 | 0.24 | 153 |
| Fourth-order clamped plate | 1.02e-13 | 0.36 | 91 |
| Unforced viscous Burgers IVP | 1.38e-8 | 8.47 | 561 |
| Wave IVP | 4.50e-9 | 0.51 | 231 |
| Steady 2D incompressible Navier–Stokes | 1.70e-8 | 1.83 | 693 |
| 3D ellipsoidal domain | 1.66e-11 | 2.38 | 455 |
| Unknown diffusivity with 24 noiseless observations | 2.44e-9 | 0.22 | 232 |
| Gradient-dependent diffusion | 3.09e-12 | 0.46 | 153 |
| Mixed Robin/Dirichlet boundaries | 6.05e-11 | 0.082 | 91 |
| Coupled reaction–diffusion | 3.34e-10 | 0.27 | 306 |
| Thin convection–diffusion boundary layer | 1.20e-8 | 0.77 | 45 |
| Smooth diffusion contrast about 1,000 | 5.77e-8 | 2.30 | 153 |

Eight of these thirteen use manufactured forcing: a known field defines a
forcing term for correctness testing. The field is not an interior training
label. The other five are Burgers, wave, steady Navier–Stokes, the boundary
layer, and inverse diffusivity. The inverse parameter's relative error is
1.34e-10 in this noiseless identifiable test; this is not a noisy inverse-field
recovery or identifiability guarantee. Fixed measurement fit is explicitly
separated from fresh physical-residual checks.

The matched **exact-polynomial** version of the same feature bank and solver
obtains similar errors on all thirteen. These controls support the importance
of the basis and coefficient solve, and do not establish a QUILL/tanh accuracy
or runtime advantage over classical polynomial collocation. The old DeepXDE
comparisons are not new matched comparisons against this implementation.

## Four-input transient 3D Navier–Stokes

### Precision follow-up, October 1

The accuracy-first follow-up separates four quantities: approximation-space
resolution, QUILL encoding accuracy, coefficient-solve accuracy, and independent
solution error. A small residual alone is insufficient: in the degree-twelve
ABC test, pressure improved by roughly a factor of twelve while the residual
changed little. All field errors below are evaluated after coefficient discovery.

The full equation-only public interface, starting with zero coefficients, solves
a manufactured nonlinear transient three-dimensional Navier–Stokes control with
actual QUILL features to velocity relative L2 error 1.36e-15 and pressure
1.88e-15. Fresh equation residual is 1.18e-15. Degrees four and six agree to
1.08e-18 in the public combined-field check. It automatically selects 257
interior centers per coordinate, lambda .2, 17 halos per side, and anchored
evaluation. The control has nonzero convection, viscosity, pressure gradients,
and vorticity. Its low polynomial degree is favorable: sufficiently resolved
all-face velocity traces determine its velocity in the small trial space.
The equations remain necessary for pressure. The low-degree result is not
evidence of arbitrary Navier–Stokes accuracy.

A stronger control starts from zero at degree ten, where nonzero incompressible
velocity perturbations can vanish on every spatial face and on the initial
slice. With the same automatic policy it takes four accepted updates to reach
1.32e-15 training residual. Degree-twelve continuation then passes immediately:
velocity relative L2 is 1.31e-15, pressure 3.54e-15, an independent equation
check is 1.95e-15, and successive-field difference is 3.73e-16. The selected
degree-twelve bank uses N=193, lambda .22, and 14 halos per side. This result
does require the PDE to determine otherwise unconstrained interior motion;
it is still the same smooth manufactured physical solution. Its maximum
sampled pressure error is 1.18e-14, so the claim is roundoff-scale accuracy,
not a uniform one-machine-epsilon bound. The complete public solve takes
387 seconds on this Mac.

For the cube half-width a, put B=(x^2-a^2)(y^2-a^2)(z^2-a^2). The degree-ten
velocity perturbation t B ((x^2-a^2)y, -(y^2-a^2)x, 0) has zero divergence,
zero spatial trace, and zero initial value. It is not included as solution
data. This supplies an explicit check that the cold trial space leaves
genuine incompressible interior freedom for the PDE to resolve.

On the separate unforced ABC problem, tightening the same engine gives:

| Trial space and solve | Velocity relative L2 | Pressure relative L2 |
|---|---:|---:|
| Original actual-QUILL adaptive solve | 3.84e-8 | 3.21e-6 |
| Degree ten, tight polynomial control | 1.50e-11 | 5.78e-10 |
| Degree twelve, improved trace sampling | 1.19e-14 | 4.22e-13 |
| Degree fourteen, 1,795 selected polynomial features | 2.46e-15 | 9.13e-14 |
| Degree fourteen, 2,701 actual-QUILL product features | 5.74e-16 | 1.50e-14 |

The degree-twelve and 1,795-feature polynomial runs did not attain the requested
2e-15 residual target; their plateau/iteration-limit statuses are retained.
The 2,701-feature actual-QUILL refinement attained fresh equation residual
8.69e-16 after one 445-second correction. Its support was expanded only from
the earlier PDE-solved support, with no reference-based selection. The final
time has velocity error 1.26e-15 but pressure error 1.22e-13: a spacetime L2
number must not hide this endpoint limitation. A subsequent tighter correction
was interrupted at Sam's pause request and preserved the same accepted state;
it supplies no additional improvement.

Additional fresh sets confirm the stronger manufactured control's spacetime
numbers; its final-time pressure error is 1.42e-14, compared with 2.85e-15
for the earlier low-degree control on the same points. Extra degrees of
freedom can worsen near-roundoff pressure conditioning. The precision studies
were guided by independent reference-capacity audits, although reference
coefficients never enter the PDE initialization or support selection. Thus
these are controlled research refinements, not blind automatic benchmark runs.

Reference-only capacity checks are stored separately. Analytic Legendre
coefficients, encoded with actual QUILL at degree fourteen and N=257, represent
the ABC velocity at 4.87e-16 and pressure at 1.39e-15. Those coefficients are
never used to initialize the PDE solve or select its support. This establishes
capacity for this benchmark, not discovery by the solver.

Precision improvements include removing absolute early-stopping floors,
linear defect refinement, useful-step retention after soft time budgets,
cancellation-resistant evaluation of halo contributions, and bounded-block
preconditioning. The halos still follow R=ceil(sqrt(N)). Increasing N beyond
the accurate encoding range can increase roundoff; N=257 was better than
N=1025 in the degree-fourteen reference-only comparison.

There are two independent resolution counts. For total polynomial degree k
in d inputs the full scalar feature count is P=binomial(k+d,d). N instead
controls the accuracy of each one-dimensional constructed mode. In four
inputs, a flat spatial/initial trace has binomial(k+3,3) polynomial degrees
of freedom. The sampling policy now grows with that trace count; the older
square-root-of-volume-sample rule grew too slowly. With six interior samples
per feature and eight cached derivatives, the interior basis cache alone is
384 P^2 bytes. This is about 3.60 GB at k=14, before boundary arrays. Sparse
support reduces that cost but can omit tiny terms essential near roundoff.

For analytic solutions admitting exponentially accurate polynomial
approximation, k proportional to log(1/error) suggests P proportional to
log(1/error)^d, before encoding, conditioning, sampling, and roundoff costs.
This is a conditional approximation law, not a measured universal PDE-solver
law. Fixed double precision cannot support arbitrarily many accurate digits.

Reproducible precision experiments are `general_symbolic_ns_precision.py`,
`general_ns4d_precision.py`, `general_ns4d_sparse_precision.py`, and the
explicitly reference-only `general_ns4d_floor.py` in expF19. Measurements and
coefficients live in the existing `general_residual/` result directory.

The same engine solves for three velocities and pressure simultaneously on
[-.5,.5]^3 x [0,1], with viscosity .1 and smooth ABC/Beltrami initial/boundary
data. It receives only the unforced momentum equation, incompressibility,
initial velocity, velocity traces on six spatial faces, and a pressure gauge
p(0,0,0,t)=0. There is no final-time condition, interior solution label,
time-marching backend, or pressure projection.

The independent reference has curl U=U, Delta U=-U, and
u(x,t)=exp(-nu t) U(x). Its advective term is a pressure gradient, so this is
a favorable smooth nonlinear-flow test, not turbulent energy transfer. The
reference supplies the prescribed conditions and post-solve error evaluation.

- Four inputs, four fields, selected total degree 10.
- 1,001 scalar product features and **4,004 unknown field coefficients**.
- 396 shared tanh units plus product gates; tanh count is not total model size.
- 137.39 seconds locally; cached basis jets about 399 MB.
- Space-time velocity relative L2: **3.84e-8**.
- Space-time pressure relative L2: **3.21e-6**.
- Final-time velocity/pressure: **3.29e-7 / 2.77e-5**.
- Fresh momentum component RMS about 3e-9; divergence RMS 7.07e-10.

The automatic degree attempts were 4, 8, 12, then 10 after degree 12 exceeded
the cache budget. Preconditioner setup took about 71 seconds. The method is
still limited by coefficient discovery, not only by inference cost. Small
residuals can coexist with larger field errors, as the pressure results show.

## Failures that changed the method

Cold high-degree solves sometimes performed worse than lower degree because
conditioning deteriorated. Reusing a resolved lower-degree solution and
switching the common preconditioner addressed the observed cases. The contrast
test previously spent 180.69 seconds without passing its checks, with field
error 6.53e-5; automatic block preconditioning reduced the observed run to
2.30 seconds with error 5.77e-8. The thin boundary layer still failed its
residual check at degree 40; generic continuation resolved it at degree 44.
Both failures remain in saved metrics.

The initial domain sampler assigned successive Sobol points to faces modulo
face count, correlating sequence structure with face identity. This was fixed
using independently scrambled face samples. Implicit-domain sampling now also
includes exposed bounding-box faces. Regression tests cover both defects.

## Discontinuities: a positive mechanism and a real scope limit

The same residual engine accepts integrated conservation and entropy
inequalities. In a restricted translating/spreading piecewise-linear front
family, it recovered a downward Burgers shock's speed as .5 without supplying
the speed. For an upward step, conservation alone allowed a nonphysical
expansion shock. Its exact weak residual was about 9.6e-17 while L1 solution
error remained .2. Adding the entropy condition selected the spreading
rarefaction with L1 error about 2.5e-6 at t=.8, limited by prescribed smoothing.
The corrected solve took six Gauss–Newton steps.

This is a controlled demonstration of the **same engine** using a mathematically
appropriate solution definition. It is not general automatic shock discovery:
the front family and exact cell integrals are supplied. A finite collection
of quadratic-entropy checks is not a universal entropy-solution certificate.
The smooth QUILL/product default has not been shown to handle arbitrary
shock interactions or blowup. These are important PDE classes, not cases to
relabel pathological merely to claim general success.

## Evidence and reproduction

New data live in
`results/checkpoint_F_applications/expF19_radon_direct_pde/general_residual/`.
`general_solver_summary.json/.png/.pdf` collate the fourteen tests and thirteen
polynomial controls. Full histories preserve failed stages, sample counts,
encoding sweeps, storage, and iteration counts.

From the repository root, with
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1`:

```sh
.venv/bin/python experiments/expF19_radon_direct_pde/general_adaptive_study.py
.venv/bin/python experiments/expF19_radon_direct_pde/general_adaptive_study.py --backend polynomial
.venv/bin/python experiments/expF19_radon_direct_pde/general_holdout_study.py
.venv/bin/python experiments/expF19_radon_direct_pde/general_holdout_study.py --backend polynomial
.venv/bin/python experiments/expF19_radon_direct_pde/general_ns4d_study.py
.venv/bin/python experiments/expF19_radon_direct_pde/general_weak_study.py
.venv/bin/python experiments/expF19_radon_direct_pde/general_solver_summary.py
```

Tests cover constructed derivatives, domain coverage, Jacobian/adjoint actions,
local callback independence, parameter bounds, inequality aggregation,
preconditioner behavior, budget failures, symbolic compilation, and end-to-end
equation-only declarations. The general-solver theory companion records the
connections to existing variational, weak-PINN, and residual-solver literature.

The established result is a common implementation with strong accuracy on
these tests. Unestablished claims include generic high-dimensional tractability,
arbitrary shock discovery, turbulent/long-horizon accuracy, all nonlinear
branches, inverse identifiability, unit-insensitive defaults, continuum error
certificates, and superiority to tuned PINNs or classical solvers. No existing
result here proves that the remaining difficulties occur only in pathological
or uninteresting problems.
