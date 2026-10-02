# A common constructed-feature PDE engine

Design and literature review, October 1, 2026. This note specifies a common
algorithm and its falsifiable requirements. It does not report new solver
experiments. Existing measurements remain in `quill_pde_checkpoint.md`.

The proposed change is substantial: stop selecting a named PDE backend and
instead accept a domain, a differential equation, constraints, and optional
observations. Construct an approximation space independently of the solution,
then minimize the declared equations in that space with one matrix-free
nonlinear solver. This is a genuine generalization of the software interface.
Whether it is a reliable general numerical method depends on the residual norm,
the approximation space, and independent resolution checks.

## 1. The objects and the common residual

Let `z` denote a point in a bounded domain `D` contained in R^d. One coordinate
can be time. The unknown field `u : D -> R^q` has `q` components. Optional
unknown physical parameters form a vector `p` in R^s. A differential jet means
the list of field values and derivatives actually requested by the equation;
for an equation of order two this is `(u, Du, D2u)`.

Use deterministic scalar basis functions `phi_j : D -> R`, and coefficients
`a` in R^(qP), to write each component as

\[
u_{a,k}(z)=\sum_{j=1}^{P}a_{kj}\phi_j(z).
\]

The QUILL encoder can supply neural approximations of analytic polynomial or
ridge profiles used to build these functions. Tensor products and compact
windows change the architecture from a single flat tanh layer; report them.
No unknown solution values enter this construction. A global total-degree
polynomial space is a useful first implementation, but not a scalable final
space: its size is binomial(d+p,p), and corners or narrow layers can converge
slowly. Overlapping local spaces with a partition of unity are the natural
next implementation.

The user supplies a differentiable map

\[
F:(z,u,Du,D^2u,p)\mapsto\mathbb R^r,
\]

with unused derivatives omitted. It returns the equation residual at a point.
Boundary, initial, interface, periodic-pair, gauge, and observation conditions
are additional residual blocks. At numerical integration points, stack their
weighted residuals into a vector

\[
R(\theta)\in\mathbb R^m,\qquad \theta=(a,p)\in\mathbb R^n.
\]

Quadrature weights approximate a declared integral norm. Observation weights
use the measurement noise model where one is known. Physical units and scales
must be retained when interpreting each block; a dimensionless total loss is
not itself a bound on field error. Dirichlet hard constraints can be supported
when a valid lifting is supplied, but arbitrary smooth distance factors should
not silently be assumed available on every domain.

An equation callback is not a solution oracle. A reference solution may be
used only for held-out benchmark evaluation. Unknown physical fields, such as
conductivity, can be represented by additional components and constrained by
data, priors, or positivity parameterizations. That expands the residual; it
does not make an unidentifiable inverse unique.

## 2. One numerical update for every residual

At the current iterate theta, let `J : R^n -> R^m` be the derivative of `R`.
For a proposed coefficient change `delta`, the first-order prediction is

\[
R(\theta+\delta)\approx R(\theta)+J\delta.
\]

Choose a diagonal or local-block scaling matrix `D` on parameter changes.
The damped Gauss--Newton subproblem is

\[
\min_\delta \frac12\|J\delta+R\|_2^2+
\frac{\mu}{2}\|D\delta\|_2^2.
\]

This is equivalent to an augmented linear least-squares problem with operator
`[J; sqrt(mu) D]` and right-hand side `[-R; 0]`. LSMR only needs products with
that operator and its transpose. It need not form `J`, `J^T J`, an SVD, or a
dense readout matrix. Automatic differentiation supplies local jet derivatives;
basis evaluation supplies changes in the field jets. With local bases these
products can be streamed by spatial batches rather than caching every feature
at every point. This reduces storage, not the mathematical dimension of the
unknown or the condition number of the PDE.

Accept a step using the ratio of actual to predicted residual reduction.
Increase damping or backtrack after a bad prediction; decrease damping after
a reliable one. Use inexact inner tolerances so early nonlinear iterations do
not oversolve an inaccurate linearization. Column or local-block scaling is a
generic starting preconditioner. It is not a proof of mesh-independent Krylov
convergence: high-order derivatives and mixed physical scales can still demand
operator-aware multilevel preconditioning.

LSMR and matrix-free damped least squares are established numerical tools. The
authors' implementation documents the operator interface and regularization:
[Fong and Saunders, LSMR](https://www-leland.stanford.edu/group/SOL/software/lsmr/).

The algorithmic distinction from existing expF19 solvers is that this same
update acts on every declared residual. The distinction from ordinary PINNs is
that the approximation geometry is constructed, initially leaving a structured
coefficient problem rather than training all feature parameters with Adam.
There is still a potentially large numerical coefficient solve; calling it
matrix-free must not obscure that fact.

## 3. What connects residual size to solution error

There are three separate questions: whether the space can approximate the
solution, whether the numerical optimizer finds it, and whether residuals
determine that solution stably. The first two cannot replace the third.

For a linear, uniquely constrained problem, suppose the complete continuum
operator `L : X -> Y` satisfies

\[
\|v\|_X\le C\|Lv\|_Y
\]

for admissible homogeneous perturbations `v`. If `Lu=f`, then

\[
L(u_a-u)=Lu_a-f,
\qquad
\|u_a-u\|_X\le C\|Lu_a-f\|_Y.
\]

That is the essential missing condition in the phrase "small PDE loss means
correct solution." The choice of residual space `Y`, boundary conditions,
and nullspace constraints all matter. Discrete collocation adds quadrature and
sampling error, which this continuum inequality does not automatically bound.

For a nonlinear operator, a comparable statement is local. Let `u_star` solve
`F(u_star)=0`; let `e=u-u_star`; suppose the derivative at `u_star` satisfies
`||e||_X <= C ||F'(u_star)e||_Y`. If the derivative is Lipschitz with constant
`L` on the segment between the two fields, the integral remainder gives

\[
F(u)=F'(u_\star)e+r,
\qquad \|r\|_Y\le \tfrac L2\|e\|_X^2.
\]

Therefore

\[
\|e\|_X\le C\|F(u)\|_Y+\tfrac{CL}{2}\|e\|_X^2.
\]

Inside a neighborhood where `CL ||e||_X <= 1`, move the quadratic contribution
to the left to obtain `||e||_X <= 2C ||F(u)||_Y`. Outside that neighborhood this
argument says nothing about competing branches or global convergence. A
successful optimizer does not establish the neighborhood or the stability
constant. Sampling and refinement are practical evidence, not a substitute
for those hypotheses.

Least-squares finite-element analysis studies precisely these norm and
coercivity questions. Even a linear Stokes example can lose the intended
optimal behavior without appropriate residual weights:
[Bochev and Gunzburger, velocity-pressure-stress least squares](https://www.sciencedirect.com/science/article/abs/pii/004578259500826M).

## 4. Strong, weak, and entropy residuals belong in the same interface

A pointwise second-derivative residual cannot describe every important PDE
solution. This is a mathematical issue, not a reason to retain named backends.
Allow a residual block to be a local integral over a domain or its boundary.
The solver then sees the same vector `R` and the same Jacobian products.

For a scalar conservation law `u_t + div f(u) = s`, a smooth test function `v`
with compact support in the open spacetime domain gives the weak condition

\[
\int\bigl(u\,v_t+f(u)\cdot\nabla v+s\,v\bigr)\,dz=0.
\]

Initial, exterior-boundary and subdomain-interface terms must be included when
the support touches them. Local test functions and conservative boundary-flux
integrals permit sharp fronts without differentiating the front itself.
Increasing test-space and quadrature resolution independently prevents the
trial space from exploiting blind spots in finitely many tests.

Weak satisfaction alone does not select the physical solution after a shock.
For scalar laws, the Kruzhkov entropy family supplies a systematic additional
declaration: for every constant `k`,

\[
\partial_t|u-k|+
\nabla\cdot\left[\operatorname{sign}(u-k)(f(u)-f(k))\right]
\le \operatorname{sign}(u-k)s
\]

in the distributional sense. Nonnegative local test functions turn violations
into residual inequality blocks. Sampling finitely many `k` and tests is an
approximation, so refine both. For systems, a declared physical entropy pair
may be needed; no generic callback can infer every admissibility principle
from a black-box pointwise residual.

[hp-VPINNs](https://arxiv.org/abs/2003.05385) develops local polynomial test spaces
and domain decomposition. [Weak RFM](https://arxiv.org/abs/2505.00508) makes a
related fixed-feature weak formulation. [wPINNs](https://arxiv.org/abs/2207.08483)
directly addresses entropy solutions with weak entropy residuals. These support
the proposed architecture; they do not establish success of our implementation.

### Exact shock counterexample

Take inviscid Burgers on the line, `u_t + u u_x=0`, and smooth the entropy shock
from 1 to 0 moving at speed 1/2:

\[
u_\varepsilon(x,t)=U(y),\quad
y=(x-t/2)/\varepsilon,\quad U(y)=(1-\tanh y)/2.
\]

As epsilon decreases, this converges in local `L1` to the correct shock. But
the strong residual is

\[
R_\varepsilon=(U-1/2)U'/\varepsilon
=\frac{\tanh y\,\operatorname{sech}^2 y}{4\varepsilon}.
\]

Changing variables `dx=epsilon dy` gives

\[
\int_{\mathbb R}R_\varepsilon^2\,dx
=\frac1{16\varepsilon}\int_{\mathbb R}
\tanh^2y\,\operatorname{sech}^4y\,dy.
\]

Set `z=tanh y`, so `dz=sech^2 y dy`. The integral becomes
`integral[-1,1] z^2(1-z^2) dz = 4/15`, hence

\[
\boxed{\int_{\mathbb R}R_\varepsilon^2\,dx=\frac1{60\varepsilon}.}
\]

The correct sharper approximation has a worse strong `L2` residual. If finite
point samples miss that layer, the measured loss can instead look artificially
excellent. Both outcomes show why the formulation and independent quadrature
must change. This calculation is derived here; it is not a reported run.

## 5. Generic controls with no named-equation dispatch

| Ordinary difficulty | Required common-engine response |
|---|---|
| Long or chaotic evolution | Declare time; solve overlapping causal slabs with interface traces, reject and refine failed slabs. Monolithic residual minimization remains optional. |
| Stiffness | Implicit residual solve on each slab, trust-region damping, scale checks, and slab subdivision. No explicit CFL restriction is introduced, but poor conditioning remains possible. |
| Neumann or pressure nullspace | Declare integral gauge or known nullspace; detect near-null directions and report nonunique field/parameter modes. |
| Multiple nonlinear branches | Initial guess or continuation path is part of the problem declaration; report branch dependence. Generic homotopy is a tool, not a uniqueness theorem. |
| Corners, interfaces, boundary layers | Local polynomial/ridge spaces and residual-driven h/p refinement; supply interface physics when discontinuous coefficients require it. |
| Shocks | Conservation/weak residual and entropy declaration, plus front-sensitive independent quadrature. |
| Unknown parameters/fields | Append to the same parameter vector; require data or declared priors and test local identifiability. |
| High dimension | Sparse adaptive spaces or supplied structure; count candidate search and coefficient dimensions. There is no generic polynomial-cost promise. |
| Discretization blind spots | New validation points, quadrature/test enrichment, basis refinement, causal/time refinement, per-block checks, and physical constraints when declared. |

The causal idea has precedent in
[Wang, Sankaran, and Perdikaris](https://arxiv.org/abs/2203.07404); local overlapping
neural spaces have precedent in
[FBPINNs](https://arxiv.org/abs/2107.07871). Our proposed feature construction and
Krylov solve can use those organizational ideas without training each local
basis.

## 6. Tests that distinguish generality from a renamed dispatcher

Freeze the engine and numerical defaults, then change only declaration data:
domain samples, equation callback, boundary/initial constraints, unknowns, and
observations. The engine must never branch on an equation name or see its
reference solution. Benchmark analytic references are kept outside that
declaration.

1. Nonlinear variable-coefficient elliptic PDE on a nonrectangular domain.
2. Coupled vector PDE with a gauge, including mixed derivatives.
3. Advection-dominated spacetime IVP, then increase advection or time length.
4. Stiff reaction-diffusion with a narrow developing layer.
5. Same equation with an unknown physical coefficient and sparse observations.
6. Shock law with strong and weak-entropy declarations: strong formulation must
   not silently pass the sharp limit, while the weak formulation must select
   the entropy solution rather than an expansion shock.
7. Deliberately underconstrained Neumann or inverse case: return a meaningful
   ambiguity diagnostic rather than a high-confidence arbitrary answer.
8. High-frequency forcing absent at training nodes: fresh validation must fail
   it before the final solution is reported as resolved.

Separate feature-encoding error, discrete nonlinear solve error, independent
PDE residual, and solution/refinement error. Record both activation counts and
coefficient counts, Krylov products, memory, and wall time. Compare against the
same unencoded polynomial basis to isolate what QUILL changes. Success on this
suite would support a common general-purpose engine. It would not establish
uniform efficiency or convergence over all ordinary nonlinear PDEs.

The strongest near-term claim we can aim to earn is: **one constructed-feature
residual solver handles several structurally different PDEs and inverse
problems without PDE-specific numerical branches, and it detects important
failures rather than accepting a small training loss.**

## 7. New standalone weak/entropy experiment

`experiments/expF19_radon_direct_pde/general_shock_analysis.py` tests the proposed
residual ingredients in small constructed front families. This is not an
experiment with the general engine or evidence of automatic front discovery.
Its outputs are
`results/checkpoint_F_applications/expF19_radon_direct_pde/general_residual/shock_diagnostics.json`
and `shock_diagnostics.png`.

First, the same conservation-integral code receives quadratic and cubic scalar
flux callbacks. It fits the unknown center and speed of a narrow tanh front
from initial mass and spacetime conservation balances. A two-variable damped
Gauss--Newton solve recovers the speeds 1/2 and 1/3 within 2.7e-12. The formulas
for these speeds appear only in post-fit evaluation, not the residual.

For either flux, both downward and upward fronts satisfy those conservation
balances to roughly 1e-12. The quadratic-entropy production measured on an
independent rectangle is -1/12 for the physical downward shock and +1/12 for
the inadmissible upward shock. Thus conservation alone does not select the
physical branch.

Second, add an unknown spreading rate to an upward piecewise-linear ramp.
Its center, propagation speed, and spreading are the three unknowns. The
initial ramp width is 1e-5; initial mass determines its center. Each snapshot
is exactly a difference of two ReLUs. For the quadratic flux, use the same
Gauss--Newton optimizer and initialization with or without entropy-inequality
blocks:

| Residual declaration | Fitted spreading rate | Independent weak balance norm | L1 error at t=.8 versus limiting step-data rarefaction |
|---|---:|---:|---:|
| Conservation only | 0 | 1.61e-6 | 0.19998 |
| Conservation plus entropy | 1.000000000013 | 1.05e-12 | 2.50e-6 |

The remaining 2.50e-6 is consistent with the prescribed finite initial ramp
width: the reference has a discontinuous step. The fitted entropy branch
solves the finite-width ramp evolution within the restricted family. Its
maximum sampled entropy violation is zero. The conservation-only fit settles
on the sharp wrong branch, and its independent entropy violation is .0658.

Third, the tanh width sweep verifies the exact strong-residual calculation:
as epsilon decreases from .1 to .0003, the independent weak cell-balance norm
decreases from .07756 to .000335, whereas the strong L2 norm increases from
.408 to 7.45. Quadrature agrees with 1/(60 epsilon) for the squared strong norm
to below 8e-16 relative discrepancy. These numbers establish a concrete route
to meaningful residuals around shocks; they do not prove that finite entropy
tests will select every system's admissible solution.
