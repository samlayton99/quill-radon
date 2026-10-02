# Generic residual engine and conditioning follow-up

The common solver now supports nonlinear equations, data, initial/boundary conditions, parameter priors, gauges, weak/integral residuals, and soft inequalities through declarations. There are no named-equation branches in `solver/general_residual.py`.

This is an iterative global coefficient and parameter solve. Constructed QUILL features supply the representation; damped Gauss–Newton and LSMR supply the numerical solve. It does not eliminate nonlinear iteration or solve a general PDE by a direct coefficient formula.

## Public contract

`ResidualBlock(name, points, function, derivatives, weight=1, scale=1, aggregation=None, relation='eq')` declares a block. The callback receives Torch points, a dictionary of declared derivative jets (Q-by-fields), and independent parameter rows (Q-by-parameters). It must be pointwise across rows. An additional random autograd pullback checks this contract by default.

Dense or sparse aggregation G is applied after the callback. `relation='le'` uses max(G r,0); `relation='ge'` uses min(G r,0); `eq` leaves G r unchanged. The active mask is included in both Jacobian products. With M output rows, residual normalization is sqrt(weight/M)/scale, so weight multiplies mean squared scaled residual. Scale can be scalar or per-equation. These are soft least-squares inequalities: inconsistent declarations can have a stationary solution with nonzero violation, which is reported explicitly.

`ResidualProblem(features, blocks, fields=1, parameter_initial=(), parameter_bounds=None)` accepts bounds as (lower, upper). `solve_residual` returns coefficients, global parameters, a field evaluator, metrics, and iteration history. Derivative orders are delegated to the feature backend, rather than capped at four. Gauges and all needed constraints must be declared; uniqueness and identifiability are not certified by a small residual.

Local nonlinear derivatives are computed once per linearization with Torch. LSMR uses only Jacobian-vector and transpose products through cached basis jets. The full residual Jacobian and global normal matrix are never assembled. Nonfinite residuals, iteration/resource exhaustion, stagnation, failed line searches, and stationary residuals above the declared target have separate statuses. The convergence claim covers declared samples/aggregates, not continuum accuracy.

## Why more modes made some earlier results worse

Legendre products are orthogonal on their reference box under the appropriate measure. Restricting them to an irregular domain changes that geometry, and high-order differential residuals strongly couple modes across degrees. Diagonal column normalization controls magnitude but cannot remove these correlations. The earlier high-degree failures were conditioning failures; adding features alone did not fix them.

The optional `preconditioner='block'` groups mode indices by coordinate parity, splitting groups to at most 64 variables. It constructs only small diagonal Gram blocks through J/J-transpose products, regularizes them, and uses inverse-triangular right preconditioning. No full Jacobian panel or global Gram matrix is stored. Factors are reused across nonlinear steps, with refresh on large sensitivity changes or a requested interval.

`preconditioner='auto'` starts with diagonal scaling. It switches to bounded blocks if LSMR reaches its iteration limit or uses at least twice the unknown count (minimum eight), or before accepting a stationary point above the residual target. One generic extra polishing step is allowed before first-order stationarity stops an above-target solve. These rules do not inspect the equation or case name. The solver's default remains diagonal to preserve prior experiments; callers can request auto.

## Matched p=12 results

All runs used the same equation declarations, samples, QUILL features, initial coefficients, tolerance 1e-9, and maximum 35 nonlinear iterations. Truth was used only to declare the study's manufactured data and independently validate the output. Timings include basis-cache preparation and preconditioner construction inside the solver. Shared feature/problem construction is separately recorded in each JSON file.

| Case | Preconditioner | Solve time | GN iterations | LSMR iterations | Independent relative field error | Status |
|---|---|---:|---:|---:|---:|---|
| Clamped biharmonic plate | diagonal | 1.221 s | 35 | 12,740 | 1.801e-7 | iteration_limit |
| Clamped biharmonic plate | block | .0216 s | 2 | 24 | 3.369e-13 | converged |
| Clamped biharmonic plate | auto | .0572 s | 2 | 388 | 3.369e-13 | converged |
| Nonlinear 3D ellipsoid | diagonal | 47.25 s | 35 | 35,000 | 3.171e-6 | iteration_limit |
| Nonlinear 3D ellipsoid | block | 1.386 s | 3 | 699 | 1.385e-8 | stationary |
| Nonlinear 3D ellipsoid | auto | 2.899 s | 4 | 2,051 | 1.915e-9 | converged |

Automatic switching includes the cost of the initial unsuccessful diagonal Krylov solve. It is more expensive than knowing in advance that the plate needs block preconditioning, but makes that decision from solver behavior.

For the auto plate solve, fresh residual RMS values were 2.77e-11 (equation), 1.41e-15 (clamped value), and 4.04e-15 (clamped normal derivative). For the auto ellipsoid they were 2.13e-10 (equation) and 1.06e-9 (Dirichlet). The latter slightly exceeds the training tolerance on different points, illustrating why the engine's sampled convergence status does not replace independent physical validation.

The plate had 91 unknowns. Its four factors had maximum size 28 and used 16,856 bytes; construction cost .0075 s. The ellipsoid had 455 unknowns. Its nine factors had maximum size 64 and used 196,296 bytes; construction cost .486 s. Setup required 91/455 J-products and the same number of transpose products; these are included in total counters, beyond the LSMR iteration counts.

Matrix-free does not mean the current implementation minimizes memory. The plate cached 1.36 MB of basis jets, versus .576 MB for a hypothetical assembled Jacobian. The ellipsoid cached 34.49 MB, versus 9.65 MB for a hypothetical Jacobian. The metadata therefore calls the latter `unassembled_dense_jacobian_bytes`, not memory saved. Peak process RSS was 378 MB. Chunked basis evaluation remains a separate possible improvement.

## Verification and reproduction

Sixteen focused tests cover matrix-free adjoints and finite differences for identity/dense/sparse aggregation; inequality active masks; inconsistent soft constraints; nonlinear actual-QUILL fields; an inverse parameter with both preconditioners; explicit integral gauges; nonlocal-callback rejection; budget/nonfinite states; bounded block factors and their transpose; automatic switching; and feature-supported fifth derivatives. Total product counters include all linearizations and preconditioner work, rather than only the final often-unused Jacobian.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_quill_general_residual.py
.venv/bin/python experiments/expF19_radon_direct_pde/general_precondition_study.py
.venv/bin/python experiments/expF19_radon_direct_pde/general_precondition_study.py --methods auto
```

The study runs one CPU thread and writes `precondition_*` JSON/NPZ files beside this report. Earlier failed diagonal results remain preserved. These two conditioning controls support the generic implementation change; they are not a comprehensive solver benchmark or a claim that one preconditioner resolves every residual system.
