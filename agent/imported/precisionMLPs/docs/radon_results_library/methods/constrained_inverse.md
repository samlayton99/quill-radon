# Physics-constrained native inverse wrapper

**Use when a few unknown physical parameters must be recovered from observations and the native forward PDE solver can validate each candidate.** It wraps the [native solver](native_solver.md); it is not a second independent forward method. [Clean and noisy rankings](../rankings.md).

## What it changes

A joint finite-penalty inverse solve can lower measurement error by tolerating a physically inaccurate field. The reduced wrapper instead does the following:

1. Propose physical parameters, such as viscosity.
2. Freeze them and solve the forward PDE from physics and prescribed initial/boundary data. Sensor measurements do not enter this inner solve.
3. Check physical feasibility using native residuals, a richer representation where required, and the ordinary exported MLP. Reject or stop if the forward gate fails.
4. Only then compare predictions with sensors using the declared measurement scale/noise model.
5. Update the small physical parameter vector. The current implementation uses finite differences of validated forward solutions and a small sensor-by-parameter fit.

This lets measurement noise remain in the data discrepancy without requiring the PDE itself to become noisy. It does not remove nonidentifiability, model mismatch, prior choices or forward-solver failures.

## Existing evidence

| Experiment | Result | Limit |
|---|---|---|
| Easy diffusion, no noise | Physical parameter .7 recovered within about 9e-14; raw PDE RMS about 2.6e-14 | Deliberately easy polynomial control |
| Same diffusion with sigma=.002 | Matched the noise-dependent parameter optimum within about 9e-14; estimate differs from truth by 5.8e-5 | Numerical accuracy and statistical error are different |
| Wall-driven NS, 16 velocity locations, no noise | Viscosity .0299999996305 for reference .03; relative error 1.23e-8; momentum RMS 2.82e-9 | Native tolerance 1e-8; synthetic measurements from a finer same-method solution |
| Same NS with sigma=.001 | Twelve completed forward evaluations; best feasible candidate .0300838817693 | Interrupted at user's organization request; no completed inverse convergence or final untouched audit |
| Harder Burgers forward gates | No feasible forward solve at tested budgets | No valid reduced-inverse answer; do not report a parameter inferred from a failed forward |

The original noiseless NS inverse process recorded 1.38 GB peak RSS, including an avoidable unbatched reference audit. A later bounded re-audit used 466 MB and reproduced values. That does not retroactively establish a sub-512-MB peak for the original inverse solve.

## Where earlier inverse methods still matter

D01 uses an analytical heat forward model and is useful for that operator. D02 evolves native reduced Burgers coordinates with parameter sensitivities. D03 recovers a restricted initial profile with regularization. D04 stabilizes classical backward heat before neural compilation. These answer different inverse questions and have different architecture/dependency qualifications; the [inverse evidence page](../evidence/inverse_problems.md) ranks them by class.

A noise-aware point estimate is not calibrated uncertainty quantification. The current wrapper supports a small parameter vector (1–16), not arbitrary coefficient fields, unknown geometry or discovery of the PDE itself. Each parameter trial can require an expensive forward solve.

## Source of truth

- [All inverse evidence and figures](../evidence/inverse_problems.md).
- [Implementation](../../../experiments/expF19_radon_direct_pde/solver/route2_inverse.py).
- [Diffusion controls](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_battletest/inverse/reduced_diffusion_precision/summary.json).
- [Completed wall-NS result](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_battletest/inverse/wall_ns_reduced_inverse/noise_0/metrics.json).
- [Interrupted noisy NS status](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_battletest/inverse/wall_ns_reduced_inverse/noise_0.001/interruption_status.json).
