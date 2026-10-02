# General constructed-neural residual solver: paused at Sam's request

Coordinator: root. Started October 1, 2026. This extends expF19; it does not
supersede its measured results or claim a universal PDE solution formula.

Objective: remove named-equation dispatch and hand-selected spectral operators.
Use one constructed feature family, differential-residual interface, and
matrix-free nonlinear solver across equations, domains, boundaries, observations,
and inverse parameters. Required mathematical problem data remain explicit.

Ownership:

- root: generic domains, benchmark declarations, independent validation, synthesis.
- general_features: `solver/general_features.py`, feature tests.
- native_quill_theory: `solver/general_residual.py`, engine tests.
- general_solver_theory: literature/theory and weak conservation/entropy analysis.

All workers share the repository and preserve other edits. Outputs go in
`results/checkpoint_F_applications/expF19_radon_direct_pde/general_residual/`.

Current design: total-degree polynomial feature banks constructed by corrected
QUILL; pointwise differential jets; arbitrary residual blocks and optional
quadrature aggregation; damped Gauss–Newton with matrix-free LSMR. This is an
iterative numerical coefficient solve, not a claim of no numerical solving.
No target values enter forward solves except prescribed boundary/initial data
and forcing. Independent exact/classical references are evaluation only.

Status: implemented and tested. Thirteen smooth/inverse declarations passed the
common adaptive policy, followed by a four-input transient 3D Navier–Stokes
solve through the same engine. Symbolic equations and a high-level automatic
sampling interface are integrated. Weak/entropy constraints were tested through
the same engine in a restricted front family. Exact-polynomial controls perform
similarly; there is no established QUILL advantage or universal convergence.

Measured findings, architecture changes, failures, scope, and reproduction are
in `docs/quill_general_solver_checkpoint.md`. Remaining research questions are
automatic local/shock representation, high-dimensional cost, general nonlinear
convergence, and reliable solution-error estimation. The experiment checkpoint
is complete; the ambition of solving essentially every PINN is not established.

Sam requested a pause and a full in-chat recap on October 1. All numerical
experiments are stopped. The final polish attempt was interrupted at its last
saved accepted state, without claiming improvement from incomplete work.

The stronger manufactured 3D transient Navier–Stokes control starts from zero
at degree ten through the unchanged public automatic interface. Actual QUILL
reaches 1.31e-15 velocity and 3.54e-15 pressure relative spacetime L2 error;
fresh final-time pressure error is 1.42e-14. Degree-twelve agreement passes.
The unforced ABC case with 2,701 actual-QUILL product features reaches velocity
5.74e-16 and pressure 1.50e-14 in spacetime; final-time pressure remains
1.22e-13. Pressure's endpoint accuracy is the clearest unresolved precision
issue. Sam considers 1e-14 essentially at the desired scale.

Reference-only capacity diagnostics are separate: analytic coefficients are
forbidden as solver initializations or support-selection inputs. They informed
research-level decisions about refinement, so this precision study is not a
blind automatic adaptivity benchmark. There is no universal PDE convergence or
field-error certificate.

Precision work also adds tighter engine stopping/refinement controls, fast
bounded-block Gram construction, cancellation-resistant feature evaluation,
boundary sample scaling by trace dimension, and optional sparse mode growth.
Root owns sparse ABC runs; quill_allocation_history independently audits them;
native_quill_theory owns engine fixes and the public manufactured precision
control. All workers are now frozen. Latest combined verification: 93 tests
passed, followed only by runner/report changes and independent saved-field
audits. No commits or publication were requested.
