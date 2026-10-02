# Native coefficient access: controlled diffusion diagnosis, October 2

This investigation was authorized after the October 1 organization pause. The complete chronological record is [current findings/scratchpad](../current_findings/scratchpad.md). Every numerical invocation records a setup, result or failure, and a source snapshot. Results live in [route2_gap_diagnosis](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_gap_diagnosis/). Production solver modules were not changed.

## What is established

Near-float64 **field accuracy is accessible through the native streamed PDE solver**, with an ordinary one-hidden-layer tanh network, for the existing paired manufactured diffusion control. The successful recipe uses a compatible coordinate basis and enough accurate inner work. The earlier plateau was not an intrinsic accuracy limit of this network.

The adaptive pair reaches sub-1e-15 ordinary relative field error in both cases, with asymmetric acceptance outcomes:

| Run | Contrast | Field relative L2 | Raw PDE RMS | Boundary RMS | Driver status |
|---|---:|---:|---:|---:|---|
| G024 | 1000 | 7.63e-16 | 7.49e-13 | 7.66e-16 | All sampled checks passed |
| G025 | 10 | 9.74e-16 | 5.48e-13 | 9.48e-16 | Time budget; held-out normalized PDE 1.66e-13 exceeds 1e-13 |

This is a sampled verification result for a smooth linear 2D equation, not a proof about the full domain, nonlinear PDEs, inverse inference, high dimensions, or Navier–Stokes. Absolute raw PDE residual and relative field error are different quantities. G025's ordinary postfit audit is available even though its driver did not proceed to its gated ordinary-export validation.

## Problem and information allowed in the solve

On the unit square, solve

\[
-\nabla\cdot(a_\kappa\nabla u)=g_\kappa,
\qquad
a_\kappa(x,y)=\exp\!\left[\tfrac12\log(\kappa)\sin(2\pi x)\sin(2\pi y)\right].
\]

The two contrasts are \(\kappa=10,1000\). The verification field is

\[
u_*(x,y)=\exp[0.3\sin(\pi xy)]+0.2\sin(2\pi x+0.4\pi y).
\]

The declared inputs are its analytically derived forcing \(g_\kappa\) and prescribed Dirichlet boundary values. No interior field labels, reference coefficients, or external numerical PDE solution enter fitting. Interior field values are evaluated only afterward. Both contrasts share this target, seed 0, and domain. These are diagnostic controls developed using previous results, not a sealed benchmark on an unseen PDE family.

Independent audits use 2,048 held-out field points and 257 raw PDE points. The driver separately checks 257 fresh normalized PDE points, 512 boundary points, ordinary exported-network arithmetic, and agreement across successive resolutions. The final field audit and derivatives use PyTorch's ordinary `Linear/Tanh/Linear` evaluation and automatic differentiation.

## Why the earlier plateau was misleading

The older normalized output labeled degree 32 performed **zero corrections at degree 32**. It embedded a degree-24 solution and passed the loose scaled stopping test. Tighter controls instead spent 533/564 seconds at degree 24 and never reached degree 32. A label saying degree 32 was therefore not evidence of solving that resolution.

The dense reference controls make the distinction quantitative. At degree 24, normalized ideal and actual-neural fits both give about 8.00e-11 relative field error. At degree 32, a normalized actual-neural fit followed by residual refinement reaches 3.75e-15 (G009). This reference uses a dense matrix and SVD **only to diagnose what a fixed representation permits**. Its arrays and coefficients are never imported into a native run.

## The coordinate issue

Let \(P\) be the number of independent readout coordinates and \(Q\) the number of scalar residual entries. A coefficient vector \(c\in\mathbb R^P\) determines the flat neural readout through the existing analytical Radon map. The linear PDE/BC residual is mathematically \(r(c)=Ac-b\), where \(A:\mathbb R^P\to\mathbb R^Q\). The native solver applies this operator without retaining the full matrix. Its correction model uses the corresponding analytically evaluable ideal profiles, while trial acceptance evaluates the actual neural residual.

Two coordinate charts describe the same ideal total-degree polynomial space. The enclosing-disk chart is well organized on the enclosing disk; the samples occupy only the square inside it. The box-Legendre chart is matched to that square. A formal change of basis preserves the ideal space but can radically change its numerical conditioning. Finite neural encoding and rounding mean the two implementations should still be audited separately.

At degree 32, the matched ideal-column, unit-column-scaled controls P004/P006 reduce the condition from about 1.23e11 (disk) to 4.82e4 (box). Separately, the unscaled actual disk diagnostic G004 has condition 3.03e12 and the unscaled ideal box diagnostic G015 has condition 1.03e5; that latter comparison changes both chart and actual/ideal column evaluation. Under the matched native parity grouping, the worst local block condition drops from 5.14e10 to 3.72e3. In disk coordinates, the preconditioner's regularized Gram construction suppresses nearly dependent block directions; QR improves early convergence but not final accuracy. In box coordinates, existing Gram and QR give essentially identical preconditioned condition (1.42e3) and accuracy. Replacing the production preconditioner is therefore not supported by these controls.

P007 verifies the box ideal streamed operator against explicit columns: random forward-product discrepancy 3.24e-16, transpose-product discrepancy 2.34e-15, and solved-vector discrepancy 4.35e-16. The successful dense iterative control can therefore meaningfully test the native correction operator. It does not benchmark native speed or memory.

## The tested native recipe

The adaptive pair G024/G025 shares these frozen settings:

1. Start from zero with box-Legendre readout coordinates. The geometry is supplied by the existing Radon constructor.
2. Use degrees 4, 8, 12, 16, 24, 32, 36; 257 interior centers per direction; 17 corrected halo centers per side; lambda 0.2. In 2D the direction count is degree plus one.
3. Use the existing equation normalization based on declared differential sensitivities at zero jets. Boundary constraints retain their original weighting.
4. Use the existing streamed ideal correction operator and local block preconditioner. Disable the loose inexact-inner schedule; use the high-accuracy inner tolerance (1e-15 here), up to 4,000 Krylov iterations and four outer iterations per degree. Damping is 1e-30, effectively zero for this linear control.
5. Evaluate every trial with the actual neural residual. If the ideal correction stalls, request refinement without claiming stationarity of the exact objective. Continue from the preceding native solution at the next degree.
6. Accept success only when independent normalized residual and ordinary-export checks pass and successive fields agree. No target field error is supplied to that decision.

The stopping tolerance is 1e-13, with a 420-second soft total budget, 64-point batches, six residual samples per coordinate before boundary rows, and a 1,024 MiB named-array limit. Process RSS is measured separately. Degree 36 has 703 readout coordinates and 10,767 ordinary tanh neurons. Polynomial algebra is used for coefficient conversion and correction coordinates; it is not substituted into the deployed forward model.

This is a **combined recipe**. We did not independently ablate damping and inexact scheduling in a factorial design, and do not attribute the entire gain to either one alone. Negligible damping is appropriate to the controlled linear residual; it is not yet a safe general nonlinear default. No production defaults were changed.

## Evidence that should remain visible

- G013/G014: limiting coarse-stage work lets the native disk solver reach degree 32, but field errors remain about 1.5e-12/1.6e-12.
- G017/G018: the box chart with the same short inner cap still gives about 1.2e-12/1.6e-12. Better coordinates alone do not finish the solve.
- G019: starting directly at degree 32 with the loose schedule and only eight outer steps is poor: field error 1.39e-5 and readout L1 2.13e8. A large allowed inner cap is ineffective when the inner stopping rule ends early.
- G020/G021: tight fixed-degree native solves give 4.68e-15/2.23e-15 field error for contrast 1000/10. Both drivers return `resolution_limit`; G020's inner solve passed its training tolerance, while G021 reached its outer cap. Neither fixed-degree run passed the held-out PDE goal.
- G024: the full ladder advances through coarse approximate-model stalls, rejects degree 32 on independent normalized PDE RMS 4.19e-13, then reaches degree 36. Held-out normalized PDE RMS is 6.06e-14 and the successive-field difference is 3.91e-15. Both pass their checks.
- G025: the same full ladder reaches degree 36 and field 9.74e-16, but its normalized held-out PDE RMS is 1.66e-13 and it exhausts the soft time budget. Preserve `time_budget`, not convergence; the raw PDE RMS is 5.48e-13. No further work was run to force this paired outcome to pass.
- G022/G023: more degree is not automatically better. Degree-36 box diagnostics with residual refinement give field 1.13e-15/raw PDE 1.74e-12. Degree 40 remains worse after three refinements. Earlier degree-40 disk controls have readout L1 up to 2.82e5 and lose digits in ordinary evaluation. These are not certificates of an ultimate floor.

## Memory and limits

The successful native path stores neither a global \(Q\times P\) Jacobian nor a global factorization. Its existing construction, batched operator products, and bounded local preconditioners retain their previous asymptotic costs; this investigation adds no new dimension-dependent dense solve to production. The dense controls intentionally do store the diagnostic matrix and are classified separately.

G024 used 353.2 MiB peak process RSS and 263.8 seconds; G025 used 363.1 MiB and 478.4 seconds in the current M4 Mac mini workspace. The latter exceeded its 420-second soft budget while completing an in-progress correction. Some controls ran concurrently as separate single-thread processes; elapsed times are not isolated speed comparisons or laptop benchmarks. This is not a new high-dimensional scaling measurement.

The next justified work is to test the coordinate/inner-work policy on another PDE family and dimension, then design a nonlinear damping/refinement policy from those results. Higher precision in derivatives and stable high-degree conversion remain separate issues. The paired diffusion result does not resolve the documented Helmholtz, Allen–Cahn, low-viscosity flow, or inverse-identifiability failures.

## Records

- [Chronological scratchpad](../current_findings/scratchpad.md)
- [Machine-readable run ledger](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_gap_diagnosis/run_ledger.csv)
- [Native comparison figure](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_gap_diagnosis/coefficient_access_progress.png)
- [G024 protocol](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_gap_diagnosis/G024_box_native_adaptive_tight_c1000/protocol.json) and [complete metrics](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_gap_diagnosis/G024_box_native_adaptive_tight_c1000/result.json)
- [G025 protocol](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_gap_diagnosis/G025_box_native_adaptive_tight_c10/protocol.json) and [complete metrics](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_gap_diagnosis/G025_box_native_adaptive_tight_c10/result.json)
- [P006 preconditioner evidence](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_gap_diagnosis/P006_box_ideal_native_parity/evidence.md)
