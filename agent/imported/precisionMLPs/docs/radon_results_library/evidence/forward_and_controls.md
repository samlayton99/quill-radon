# Forward PDE results and controls, organized by utility

This page organizes completed saved evidence. **No new experiments, solver changes, or performance measurements were made for this review.** It covers the forward stress suite, equation normalization, stricter stopping, the initial-plane control, and the most relevant earlier controls. The [historical catalogue](../../radon_method_catalogue.md) retains the full method inventory and asymptotics. Inverse and higher-dimensional fluid results have separate evidence owners.

**October 2 supplement:** the subsequently authorized [coefficient-access investigation](coefficient_access_2026_10_02.md) resolves the fixed-space field-accuracy gap for the paired diffusion control. Adaptive native runs reach ordinary relative field errors 7.63e-16/9.74e-16; the contrast-1000 run passes its sampled driver checks and contrast 10 ends at its time budget. The original results below remain historical and the other stress-suite failures remain unresolved.

The current deployment requirement is one hidden layer: \(u(x)=b+\sum_{j=1}^{B}v_j\tanh(w_j\cdot x+\beta_j)\), where \(x\in\mathbb R^d\), \(B\) is the neuron count, and the readouts \(v_j\) can be vectors for multiple fields. Analytic coordinates used during construction do not change this requirement: the exported model must actually evaluate that sum. Product gates, polynomial evaluators, deeper networks, and routed time-slab collections do not qualify. Additive neuron banks and affine input changes can qualify because they compile into this sum.

## What to keep and why

| Utility | Evidence-backed decision | What it does not establish |
| --- | --- | --- |
| Main platform for native PDE work | Keep the streamed native residual solver with analytical ridge coordinates. It discovers coefficients from equations and prescribed conditions and exports an ordinary flat tanh network. Earlier smooth controls reach roundoff; the broader suite exposes specific failures. | A blind solver that reliably reaches the floor on every smooth PDE. |
| Strongest diagnosis | Keep the paired diffusion comparison and its cross-operator capacity witness. The same accurate network works for both operators, while the unnormalized cold solves differ by ten orders of magnitude in field accuracy. | Identical finite-dimensional minimizers for different residual norms, or a proof that every failure is a conditioning problem. |
| Useful generic component | Keep operator-derived residual normalization as an option inside the same solver. It removes the catastrophic contrast penalty in this test. | A new independent solver, a raw-residual accuracy certificate, or universally improved precision. |
| Useful failed control | Keep exact-initial-plane neuron pairing. It preserves the flat architecture and the algebraic initial constraint, but the tested Allen–Cahn solves still fail badly. | A completed cure for time-dependent nonlinear problems. |
| Important negative result | Preserve the stricter-tolerance diffusion extension. Spending longer on intermediate resolutions prevented timely refinement and did not improve final accuracy. | A representation floor or a fundamental inability to reach better accuracy. |
| Narrow positive control | Keep the matched-frame iteration as evidence that no-least-squares native PDE iteration can work for a specially compatible operator. | A generic inverse for arbitrary PDE operators. |

These are method components and diagnostic controls. Counting normalization, diagonal scaling, block scaling, and different stopping tolerances as independent competing methods would inflate the inventory.

## How the forward suite was run

All six cases use two inputs: two spatial coordinates for Helmholtz/diffusion, or space and time for Allen–Cahn. The native solver receives PDE residuals and prescribed boundary/initial data, starts from zero at its first resolution, and continues only from its own previous-resolution coefficients. No interior reference values or externally evolved solution enter fitting. The final architecture is `Linear/Tanh/Linear`.

The shared scalar geometry has 257 interior centers, 17 halos per side, and lambda 0.2. The degree ladder is 4, 8, 12, 16, 24, 32, 40, 48. In the two-dimensional adapter, degree \(p\) uses \(M=p+1\) directions, \(B=291M\) neurons, and \(P=(p+1)(p+2)/2\) solver coordinates. Neuron count and coordinate count are different quantities.

- **V1 screening:** six cases, two seeds, 60-second soft budgets, default 20 outer iterations. All twelve runs ended at their time budgets.
- **V2 accuracy extension:** all six cases, seed 0 selected uniformly, cold restarts, 600-second soft budgets and 60 outer iterations. Geometry, data policy, tolerances, and sampling are unchanged. Every retained V2 model has degree 32, 9,603 neurons, and 561 coordinates.
- **Normalized suite:** the same six cases and 600-second policy, with a fixed equation-derived scale applied to PDE rows; boundary/initial rows are unchanged. This changes the residual norm, not the physical equation or network family.

Field error below means relative \(L^2\) on 2,048 held-out points. Raw PDE RMS means the original unscaled equation evaluated with ordinary Torch differentiation on a separate set of 257 points. Neither is a uniform continuum bound. Actual time includes fitting, export, and audits; inner solves can overrun a soft limit, and concurrent machine work prevents interpreting these times as isolated speed benchmarks.

Protocols and complete results: [V1 protocol](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_battletest/forward/protocol.json), [V1 all-seed summary](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_battletest/forward/summary.json), [V2 protocol](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_battletest/forward/accuracy_v2/protocol.json), [V2 summary](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_battletest/forward/accuracy_v2/summary.json), [normalization protocol](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_battletest/normalized/protocol.json), [normalization summary](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_battletest/normalized/summary.json).

## Completed results, including failures

The V1 range includes both seeds. V2 and normalized columns are the prescribed seed 0 runs. Each V2/normalized value links to its full per-case record.

| Case | V1 field error, both seeds | V2 field error | V2 raw PDE RMS | Normalized field error | Normalized raw PDE RMS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Helmholtz, frequency 2 | 2.67e-6–1.26e-3 | [3.57e-10](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_battletest/forward/accuracy_v2/helmholtz_n2_s0.json) | 1.69e-7 | [3.56e-11](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_battletest/normalized/helmholtz_n2_s0.json) | 1.27e-7 |
| Helmholtz, frequency 4 | 156–160 | [0.0322](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_battletest/forward/accuracy_v2/helmholtz_n4_s0.json) | 13.85 | [0.00443](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_battletest/normalized/helmholtz_n4_s0.json) | 1.62 |
| Diffusion, contrast 10 | 3.19e-8–4.38e-8 | [6.00e-14](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_battletest/forward/accuracy_v2/diffusion_c10_s0.json) | 3.82e-11 | [3.26e-11](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_battletest/normalized/diffusion_c10_s0.json) | 1.15e-7 |
| Diffusion, contrast 1000 | 0.00458–0.00467 | [0.00432](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_battletest/forward/accuracy_v2/diffusion_c1000_s0.json) | 0.00226 | [8.03e-11](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_battletest/normalized/diffusion_c1000_s0.json) | 1.15e-7 |
| Allen–Cahn, diffusion 0.01 | 0.886–0.887 | [0.693](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_battletest/forward/accuracy_v2/allen_cahn_d001_s0.json) | 0.0169 | [0.562](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_battletest/normalized/allen_cahn_d001_s0.json) | 0.0644 |
| Allen–Cahn, diffusion 0.0001 | 0.940–0.943 | [0.804](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_battletest/forward/accuracy_v2/allen_cahn_d00001_s0.json) | 0.0184 | [0.663](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_battletest/normalized/allen_cahn_d00001_s0.json) | 0.0760 |

Only V2 contrast 10 passed the unnormalized sampled checks; its function error is near double-precision roundoff, but its PDE residual is not 1e-14. V2 actual times range from600.6 to 789.9 seconds and peak process RSS from 316.9 to 339.9 MiB. All six cases therefore fit comfortably in memory here, but several remain inaccurate.

The normalized suite reports sampled-check success for both diffusion cases and Helmholtz frequency 2. Those checks apply to the scaled PDE: raw residuals remain approximately 1e-7. Diffusion contrast 10/1000 took 143.3/173.0 seconds; normalized Helmholtz frequency 2 took 1362.3 seconds, exceeding its soft budget and retaining `budget_exhausted` at the inner-solver level despite final sampled-check success. Frequency4 and both Allen–Cahn cases remain time-budget exits. Their improved field errors do not erase their large remaining errors. Normalization makes the easier diffusion case less accurate at its chosen stopping point.

Figure: [V2 fields and errors](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_battletest/forward/accuracy_v2/hard_fields.png). The older [cross-campaign comparison figure](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_battletest/comparison/forward_comparison.png) was generated before two normalized rows were added; use the six per-case records above as the complete evidence.

### What the cases actually test

Helmholtz uses the published DeepXDE square Dirichlet case, with frequency 4 as the declared harder extension. Its analytic target is used only for evaluation; separability is not exploited by the solver. Allen–Cahn uses the original PINN periodic setup with initial data \(x^2\cos(\pi x)\), plus the higher-diffusion control. The manufactured diffusion control uses the same nonseparable target \(u_*(x,y)=\exp(0.3\sin(\pi xy))+0.2\sin(2\pi x+0.4\pi y)\) at both contrasts, with \(a(x,y)=\exp[\tfrac12\log(C)\sin(2\pi x)\sin(2\pi y)]\) in \(-\nabla\cdot(a\nabla u)=f\). Target values enter prescribed boundary data and the manufactured forcing, never an interior fitting loss. These are verification controls, not a claim of reproducing the full PINNacle protocol.

The analytic evaluator gives exact-form references for Helmholtz and diffusion. Allen–Cahn references were generated only after fitting, with independent Fourier collocation and DOP853 at 512/1024 grid points. Their relative discrepancies are 2.29e-6 and 9.28e-6 for the two diffusion values: adequate to diagnose errors above 0.5, inadequate to validate a precision floor. The initial periodic values agree at the endpoints, but initial first derivatives do not; uniform corner-derivative accuracy is not a suitable expectation. Source declarations and reference quarantine are explicit in the [forward runner](../../../experiments/expF19_radon_direct_pde/route2_battletest_forward.py). Published case links are recorded in its frozen protocols.

## Strongest control: capacity is present, solving can still fail

The V2 contrast 10 model was evaluated unchanged under contrast 1000 physics. This is a post-fit capacity witness, not a warm start or an additional fit. The same 9,603-neuron network achieves contrast 1000 PDE RMS 3.92e-11 on 2,048 fresh points; a second 257-point check gives 9.16e-11, and 512 boundary points give 3.26e-13 RMS. Its worst PDE error on the larger audit is 7.08e-10.

Thus the contrast 1000 baseline's 0.00432 field error cannot be explained by absence of an accurate function in that network space. The operator changes the residual geometry and the difficulty of finding coefficients. Normalization improves the cold-solve field error to 8.03e-11 and reduces the readout absolute-sum norm from 5.27e6 to 3.74. These observations support conditioning and cancellation as important mechanisms; they do not provide a complete causal decomposition.

Exact evidence: [cross-operator capacity witness](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_battletest/forward/accuracy_v2/diffusion_capacity_witness.json). No witness coefficients entered either baseline solve.

## Stricter stopping is not automatically better

The separately frozen precision extension applied identical normalization and tolerance 1e-13 to both diffusion contrasts, with cold starts, the same geometry, and 600 seconds/60 outer iterations. Both exhausted time at degree 24, with 7,275 neurons and 325 coordinates:

| Contrast | Held-out field relative L2 | Raw PDE RMS | Actual seconds | Final readout absolute-sum norm |
| --- | ---: | ---: | ---: | ---: |
| 10 | 3.26e-11 | 1.15e-7 | 613.1 | 3.73 |
| 1000 | 8.03e-11 | 1.15e-7 | 660.3 | 3.72 |

Both had approximately 268 MiB peak process RSS and 9.73 MiB maximum estimated named arrays. The estimates exclude unspecified callback/internal-library workspace and are not hard RSS caps. Intermediate resolutions repeatedly consumed 60 outer iterations. Degree24 then consumed approximately 1,500 inner LSMR iterations and 533–564 seconds, leaving no budget for degree 32. This identifies a stopping/refinement weakness; it is not a capacity ceiling. The broader normalized suite reaches degree 32 but can retain essentially the same field after its own stopping decision.

Sources: [precision protocol](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_battletest/forward/normalized_diffusion_precision_v3/protocol.json), [contrast 10 full history](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_battletest/forward/normalized_diffusion_precision_v3/diffusion_c10_s0.json), [contrast 1000 full history](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_battletest/forward/normalized_diffusion_precision_v3/diffusion_c1000_s0.json), [figure](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_battletest/forward/normalized_diffusion_precision_v3/forward_summary.png).

## Initial-plane control: valid architecture, unsuccessful solve

The control adds an analytical encoding of the prescribed initial data and represents a correction as \(F(x,t)-F(x,0)\). Pairing actual tanh neurons makes the correction zero at the initial plane in exact arithmetic; concatenating the paired and initial-data banks still yields one ordinary hidden layer. Only the supplied initial data are encoded—no future solution is used.

This is a separate geometry/control protocol, not an otherwise identical timed ablation: it uses 20 outer iterations, different coefficient grouping, and paired neurons. Both final models have degree 40, 24,153 neurons, 820 coordinates and end with `budget_exhausted` around 608–609 seconds. Peak RSS is 445–449 MiB.

| Allen–Cahn diffusion | Field relative L2 | Raw PDE RMS | Ordinary initial-data RMS | Readout absolute-sum norm |
| --- | ---: | ---: | ---: | ---: |
| 0.01 | 0.506 | 0.0839 | 5.73e-6 | 4.12e11 |
| 0.0001 | 0.637 | 0.102 | 5.75e-6 | 3.80e11 |

The positive phase reappears, but global field errors remain 50–64%. Huge readouts make finite-precision cancellation consequential: symbolic initial-plane cancellation does not give a machine-precision initial condition in the exported ordinary sum. Periodic derivative RMS is about 0.064 in both. Retain this as an architectural mechanism with an unsuccessful current optimization result, not a solved Allen–Cahn case.

Sources: [control protocol](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_battletest/initial_plane/protocol.json), [summary and ordinary checks](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_battletest/initial_plane/summary.json), [higher-diffusion history](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_battletest/initial_plane/allen_cahn_d001.json), [stiff-case history](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_battletest/initial_plane/allen_cahn_d00001.json), [figure](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_battletest/initial_plane/initial_plane_control.png), [implementation](../../../experiments/expF19_radon_direct_pde/route2_initial_plane_profiles.py).

## Earlier positive controls worth preserving

These remain valuable demonstrations within the same research hierarchy, rather than evidence that the harder suite must work unchanged.

| Control | Completed result | Utility and restriction | Exact evidence |
| --- | --- | --- | --- |
| Streamed independent disk coordinates, block preconditioner, catalogue E10 | Smooth manufactured forward field 4.31e-16; ordinary PDE RMS 4.15e-15, max 4.03e-14; 4,947 neurons, 153 coordinates, 59.2 seconds | Establishes that native solving and genuine flat export can reach roundoff on a nontrivial smooth control. It is a narrow 2D problem. | [forward record](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_generality/streamed_solver/smooth_p16_n257_disk.json) |
| Same model with diagonal scaling, catalogue E11 | Field 3.68e-16; PDE RMS 4.18e-15; 185.6 seconds and 1,701 Krylov iterations, versus 532 for blocks | Same solver/model, alternative setup tradeoff. Final saved field is accurate although the solver returned budget exhaustion. | [diagonal record](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_generality/streamed_solver/smooth_p16_n257_disk_diagonal.json) |
| Redundant directional coordinates, longer solve, catalogue E09 | Field 1.47e-15; PDE RMS 6.86e-14, compared with initial field 1.3e-8 | Demonstrates under-solving rather than an immediate capacity floor. Independent angular coordinates improve this same family. | [long run](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_generality/streamed_solver/smooth_p16_n257_streamed_long.json) |
| Matched-frame iteration, catalogue E08 | Tightened degree 16 result: field 1.26e-16 and PDE RMS 1.49e-15 after 46 updates | Native no-LS iteration works when the frame diagonalizes the reference operator. The tested diffusion is boundary-degenerate with natural flux on a disk, not arbitrary Dirichlet physics. | [tightened run](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_generality/frame_iteration/p16_quad2_tighter.json), [independent checks](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_generality/frame_iteration/independent_checks.json) |

## Method families serve different problem classes

The [rankings by problem class](../rankings.md) contain the top-three recommendations. There is no global ordering across analytical function construction, fitting observed samples, forward PDEs, and inverse problems: the supplied information and the job to be done differ. This evidence page records the applicable uses and limits.

- **Unknown PDE solution: streamed native residual solving in analytical ridge coordinates.** One flat MLP, equation/condition declarations, bounded feature tiles, no external solution fit. Operator scaling, coordinate redundancy removal, block/diagonal preconditioning, and initial-plane pairing belong under this family. The six-case evidence limits its current robustness.
- **Known analytical profiles: corrected analytical Radon construction, catalogue A20/A21.** This no-fit construction supplies a reference for achievable representation accuracy and the scalar bank used by the native solver. Access to known profiles is essential; it does not itself discover unknown PDE solutions or establish reconstruction from noisy observations. Square-root halos alone are insufficient without the contour/boundary correction. See [A20/A21 theory, cost, and evidence](../../radon_method_catalogue.md).
- **Supported periodic evolution: native reduced ridge-coordinate dynamics, catalogue B09/B10 and related inverse wrappers.** Fixed-time outputs are flat MLPs and no external trajectory supplies training labels. It requires native integration/projection, uses problem-specific operator structure, and does not automatically provide one flat network of space and time. See [reduced dynamics](../../radon_method_catalogue.md) and [native ridge Navier–Stokes](../../radon_method_catalogue.md).

Dimensional overhead must be assessed separately from approximation size. Let \(Q\) be the physics-point count and \(J\) the derivative-channel count. Streaming avoids storing a full \(Q\times P\) feature/Jacobian cache. It does not make repeated physics products, coefficient transforms, preconditioner setup, or iteration counts free. An ordinary dense export stores \(O(dB)\) weights and its batch inference uses \(O(EdB)\) arithmetic for \(E\) inputs; shared-direction implementations can reduce repeated projections before export. A full tensor angular rule can already have \(M\sim(p+1)^{d-1}\). The two-input successes here cannot certify that construction/solving adds no further costly dimensional dependence. The [catalogue cost definitions](../../radon_method_catalogue.md) explicitly include encoder, map, preconditioner and iteration costs.

Outside the current qualifying architecture/native-PDE set: product-gate and exact-polynomial controls, the deep-tanh compiler, routed collections of time-slab networks, and external numerical solves followed by neural encoding. They remain useful comparisons, but fail either the current one-hidden-layer deployment requirement or native solution-discovery requirement. The matched-frame iteration remains a strong specialized branch; it has not replaced the broader residual solver. A known-function compiler can still be useful in its own problem class without qualifying as a native PDE solver.

## Evidence integrity and remaining limits

- Frozen protocols and failures remain intact. This organization did not rerun, select a new seed, change a tolerance, or overwrite an experiment.
- V1 includes both seeds; longer campaigns use the uniformly chosen seed 0 only. Broad seed robustness is not established by those extensions.
- Error claims are specific to field values, derivatives, raw/scaled PDE residuals, and sampling locations. Near-roundoff field error is not automatically a PDE floor.
- The [V3 provenance note](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_battletest/forward/normalized_diffusion_precision_v3/runtime_provenance_note.json) preserves a limitation: the older driver recorded source-file hashes at completion, while already loaded solver modules preceded a later default-off disk edit. No historical startup fingerprint is invented. Future fits record startup and completion snapshots separately.
- No universal error-versus-neuron law or dimension-independent iteration bound follows from these endpoints. The evidence supports a useful native method with identifiable failures, not a solved general-PINN claim.
