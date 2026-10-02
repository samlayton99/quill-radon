# expF09 -- Navier-Stokes via frozen-geometry collocation (program tracker)

Training-free, per-instance collocation-lstsq solver extended to incompressible
Navier-Stokes in **primitive velocity-pressure `(u, v, p)`** variables. Positioned
as the **accuracy-ceiling reference** against data-driven neural operators (we
solve each instance from the equations; we do not learn a solution map).

Design spec: `docs/superpowers/specs/2026-07-16-navier-stokes-collocation-design.md`

## Stages

| Stage | Problem | New machinery | Exact verifier | Status |
|---|---|---|---|---|
| **A** | Stokes (linear) | `solve_system` + pressure gauge | manufactured div-free | **done** -- velocity **7.7e-14** (fp64 floor), div ~8e-11 |
| **B** | Steady NS | Newton on the system (expF06 pattern) | Kovasznay flow (exact) | roadmapped |
| **C** | Unsteady NS | time as a coordinate + IC block | Taylor-Green (box Dirichlet, no periodicity) | roadmapped |
| **D** | FNO NS benchmark | Fourier ridges + periodic torus + data | dataset + trained FNO | roadmapped |

Each stage gets its own detailed spec + plan at its gate. Stages A-C live on the
box `[-1,1]^2` with tanh ridges; periodicity (Fourier ridges) is needed only for
Stage D.

## Code (Stage A)

- `experiments/expF09_navier_stokes/core_system.py` -- vendored scalar
  primitives + multi-field `solve_system` / `eval_field`.
- `experiments/expF09_navier_stokes/stokes.py` -- manufactured Stokes problem.
- `experiments/expF09_navier_stokes/run_stokes.py` -- W/lambda sweep.
- Stage-A writeup: `expF09_stokes_results.md`.

Reproduce Stage A:
`uv run --extra dev python experiments/expF09_navier_stokes/run_stokes.py`
