# Inverse problems: useful methods, completed evidence, and limits

This page organizes existing evidence only. No solve, test, model evaluation, or new figure was produced for this inventory. The noisy wall-driven Navier–Stokes run was stopped at the user's organization-only request; its saved candidates remain available, but it is not a completed inverse result.

The current **reduced native inverse** is an extension of the main streamed native PDE solver, not an independent forward solver. Its useful distinction is that observations choose among physically validated neural solutions. The exported field remains one ordinary `Linear/Tanh/Linear` network. Its main limitation is equally direct: if the underlying native forward problem cannot be resolved, the inverse must stop as unresolved.

## Which methods to use

These are utility rankings within each problem class, not claims that three generic methods have been validated. Specialized alternatives are retained because they sometimes have stronger evidence on their supported problem than the general wrapper does.

### Clean inverse data

| Rank | Method and appropriate use | Completed evidence | Restriction |
| --- | --- | --- | --- |
| 1 | **Reduced native inverse:** a few unknown physical parameters, with each trial field found by the native PDE solver | Wall-driven NS viscosity relative error 1.23e-8 from 32 sensor scalars; easy diffusion parameter matches its analytic optimum within 9.04e-14 | Depends on successful forward validation; NS measurements use a finer same-method reference. Not a universal inverse solver. |
| 2 | **D02, native Burgers inverse with tangent equations:** viscosity and two initial-condition amplitudes | Refined viscosity relative error 1.14e-11; withheld field error 2.81e-13 | Specialized native numerical time integration. A flat spatial snapshot is supported; this is not a single flat network of the entire space-time solution. |
| 3 | **D01, analytical heat-tensor inverse:** six tensor entries when the explicit heat forward family applies | Noiseless tensor relative error 2.8e-9 from 36 observations | Uses its supported analytical forward formula. It does not discover arbitrary unknown PDE fields. |

For an unknown *initial function*, rather than a few physical parameters, **D03** is the more relevant specialized alternative: it recovered a 17-coefficient early-time Burgers initial profile to 1.09e-12 after forward refinement. Its finite-dimensional prior and time integration are substantive parts of that result.

### Noisy inverse data

| Rank | Method and appropriate use | Completed evidence | Restriction |
| --- | --- | --- | --- |
| 1 | **Reduced native inverse:** preferred general formulation when forward validation is feasible | Noisy diffusion reaches its statistical optimum within 9.04e-14 while ordinary PDE RMS stays 2.61e-14 | This completed noisy control is easy. Noisy NS is interrupted, and the harder independent Burgers forward gates failed. The ranking reflects formulation and architectural fit, not broad empirical dominance. |
| 2 | **D02, tangent-based native parameter inverse:** a few Burgers parameters | At 0.1%/1% observation noise, median viscosity errors were 1.21%/12.1%; field errors 0.0328%/0.324% | Noise amplifies through parameter sensitivity. The conditional uncertainty experiments do not establish global identifiability. |
| 3 | **D03, regularized initial-profile inverse:** noisy recovery of an initial function in a declared prior family | At 0.5% noise, early-profile errors 2.18–4.29%; late-profile errors 10.18–10.69% | Late observations lose information about the initial state. Regularization and forward refinement are required; no noise-floor breakthrough is claimed. |

For **backward heat specifically**, use the specialized **D04 discrepancy-cutoff baseline** before assuming a generic neural inverse is preferable. Its recovered initial-field error was 0.001624, versus 2.73e173 without regularization. The subsequent tanh encoding error was 4.20e-16; that is a *conversion* floor, not an inverse-solution floor. D04 is classical spectral inversion followed by neural compilation, so it does not qualify as native neural solution discovery. D01 also has a useful noisy heat-tensor control: 1% noise produced 1.2–5.3% tensor error across eight draws.

Exact earlier evidence: [D01 tensor results](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/inverse/tensor_metrics.json), [D02 results](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/capabilities/inverse_results.md), [D02 uncertainty controls](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/general_solver/inverse_uq_metrics.json), [D03 profile results](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/adaptive_followup/inverse_profile_metrics.json), [D04 backward-heat results](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/inverse/backward_heat_metrics.json). The [historical catalogue](../../radon_method_catalogue.md#d01-analytical-heat-tensor-inverse-wrapper) retains their construction and memory costs.

## What the reduced native inverse actually does

Let \(\eta\in\mathbb R^k\) be the unknown physical parameters, \(c\in\mathbb R^P\) the current neural readout coordinates, and \(F(c,\eta)\in\mathbb R^Q\) the sampled PDE, boundary, initial-condition, and gauge residuals. Let \(s(c)\in\mathbb R^S\) be the sensor predictions and \(y\in\mathbb R^S\) the observations. The dimensions \(P\) and \(Q\) can increase during native resolution refinement; \(k\) and \(S\) stay fixed for an inverse problem.

The joint finite-penalty formulation minimizes

\[
\tfrac12\|F(c,\eta)\|^2+\tfrac12\|W(s(c)-y)\|^2,
\]

where \(W\in\mathbb R^{S\times S}\) is the diagonal observation-scaling matrix, including any noise and sample-count normalization. With noise, a physically exact state usually still has a nonzero sensor gradient. A finite physics weight therefore permits the optimizer to improve the data term by leaving the physical solution set. This explains a possible source of bias; it does not by itself explain every observed optimization failure.

The reduced formulation instead computes a physically validated state \(c_\eta\) at each trial parameter and minimizes

\[
J(\eta)=\tfrac12\|W(s(c_\eta)-y)\|^2.
\]

The implementation follows this sequence:

1. Freeze the candidate parameters in a copied physics declaration. The inner solver has no remaining unknown physical parameters and receives no sensor loss.
2. Solve the PDE and prescribed conditions with the native neural residual. Check fresh residual points, the declared resolution policy, and ordinary exported-network derivatives.
3. Score observations only after those forward checks pass. A failed forward is **unresolved**, not evidence that its parameter value is physically wrong.
4. Estimate the small sensor-response matrix by finite differences of validated native forwards, then make a bounded parameter update and line search. The current API supports 1–16 scalar parameters.
5. A previous native field may provide an initial guess. Optional truncation keeps named coordinates through the greatest fixed-ladder degree below the current validated degree. This copies coefficients algebraically, preserves the chart, and requires fresh checks at the new parameter; it never fits reference field values.

This is a reduced PDE-constrained estimation method with a native neural inner solver. The reduction principle itself is established methodology, not a new inverse-optimization theorem. The useful addition here is the connection to the streamed Radon/QI coordinate solver and its flat tanh export. See the [implementation](../../../experiments/expF19_radon_direct_pde/solver/route2_inverse.py) and the standard [reduced-problem formulation](https://dolfin-adjoint.github.io/dolfin-adjoint/documentation/maths/2-problem.html).

`data_stationary` means the sampled local projected-gradient criterion passed. It does not establish uniqueness, global optimality, or a continuum error bound. In particular, a scalar finite-difference response has numerical rank one whenever it is merely nonzero; that is not an identifiability certificate. An optional step-size agreement check exists, but was not enabled for the archived NS result.

## Completed wall-driven Navier–Stokes inverse

The problem is steady incompressible flow on the unit disk, with zero body force, pressure gauge \(p(0)=0\), and prescribed tangential wall speed \(1+0.25\cos(2\theta)+0.15\sin(3\theta)\). Only viscosity is unknown. The inverse starts at \(\nu=0.05\), with zero neural readouts, and receives velocity at 16 interior locations: **32 scalar measurements and no pressure observations**. Bounds are \([0.005,0.2]\).

| Quantity | Completed noiseless result |
| --- | ---: |
| Reference viscosity | 0.03 |
| Recovered viscosity | 0.029999999630522526 |
| Relative viscosity error | 1.23e-8 |
| Untouched velocity relative L2 error | 6.34e-10 |
| Ordinary held-out momentum RMS / maximum | 2.82e-9 / 1.09e-7 |
| Ordinary divergence RMS | 4.50e-10 |
| Ordinary wall-velocity RMS | 8.89e-12 |
| Native forward solves | 15, all validated |
| Inverse elapsed time | 1,889 s, about 31.5 minutes |
| Final network | 11,931 tanh neurons; three outputs; degree 40 |
| Readout-coordinate unknowns per final forward | 2,583 |
| Final status | `data_stationary` |

The geometry uses 41 directions, 257 interior centers and 17 halo centers per side, with lambda 0.2. The requested physical RMS tolerance was 1e-8. This is a successful physical inverse at that tolerance, **not a machine-floor inverse**. Reported wall time includes concurrent machine work and is not an isolated speed benchmark.

**Reference caveat:** the measurements come from a finer native p56/N513 solution of the same method. Only measurement values, locations, and known noise scale enter the inverse; reference coefficients and full fields never initialize it. Separate samples and a finer model reduce some discretization bias, but do not eliminate the same-method “inverse crime” concern. This is neither experimental data nor independent-discretization inverse validation. The finer reference was a native continuation/re-encoding check, not a new independently solved truth.

Evidence: [frozen protocol](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_battletest/inverse/wall_ns_reduced_inverse/protocol.json), [complete metrics/history](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_battletest/inverse/wall_ns_reduced_inverse/noise_0/metrics.json), [ordinary model](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_battletest/inverse/wall_ns_reduced_inverse/noise_0/ordinary_mlp.pt), [held-out audit arrays](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_battletest/inverse/wall_ns_reduced_inverse/noise_0/heldout_audit.npz), [existing figure](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_battletest/inverse/wall_ns_reduced_inverse/wall_ns_inverse.png), [reference record](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_hardening/physical_flow/lowviscosity_resolution_check_disk_stirring_nu0.03_p56_n513.json).

### Noisy NS is interrupted, not failed or converged

The separate noise-0.001 case also started cold under the same protocol. At the user's pause, 12 forwards had completed and passed validation. The current forward was interrupted with exit code 130. The best completed feasible sensor candidate had viscosity **0.030083881769285767**, degree 40, sensor objective **0.66334**, and maximum component RMS across its ordinary physics gate **2.04e-9**. That gate used 1,025 interior points and 132 wall points.

No final inverse convergence assessment or untouched final field audit was completed. Outer acceptance history was not separately checkpointed, so the saved candidate must not be promoted to a finalized estimate. The existing NS figure contains the completed noiseless result only.

Exact saved status: [interruption record](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_battletest/inverse/wall_ns_reduced_inverse/noise_0.001/interruption_status.json), [best completed feasible candidate](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_battletest/inverse/wall_ns_reduced_inverse/noise_0.001/forward_009_native.npz), [completed-forward log](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_battletest/inverse/wall_ns_reduced_inverse/noise_0.001/forward_progress.json).

## Noise need not cap physical consistency: the easy control

For \(-\nu\Delta u=2(2-x^2-y^2)\) on the square with zero boundary data, nine sensors identify a scalar viscosity. The reference viscosity 0.7 is used only to generate measurements and audit; fitting starts at 1.1. The polynomial solution family makes this an easy verification control.

| Observation noise | Recovered viscosity | Difference from statistical optimum | Ordinary PDE RMS / maximum |
| --- | ---: | ---: | ---: |
| 0 | 0.6999999999999096 | 9.04e-14 | 2.57e-14 / 8.59e-14 |
| 0.002 | 0.7000579912474776 | 9.04e-14 | 2.61e-14 / 8.97e-14 |

The noisy optimum differs from the true viscosity by 5.80e-5. That statistical error coexists with nearly roundoff-level physical consistency. Each case used 19 validated forwards, about 5.2 s, degree 6 and 2,037 neurons; recorded process RSS was 301–312 MB. The conditional field error, measured against the exact field at the *estimated* viscosity, is about 1e-15. It must not be confused with error against the true noisy-data-generating field.

Evidence: [precision protocol](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_battletest/inverse/reduced_diffusion_precision/protocol.json), [complete two-case summary](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_battletest/inverse/reduced_diffusion_precision/summary.json). The analytic statistical optimum was computed only for the audit.

## Independent Burgers evidence: the broader method still fails

The independent campaign fits viscosity in \(u_t+uu_x-\nu u_{xx}=0\), with \(x\in[-1,1]\), \(t\in[0,1]\), initial value \(-\sin(\pi x)\), and zero endpoint values. An analytically evaluated Cole–Hopf integral generates 32 sensor measurements and held-out truth. It never supplies coefficients, interior labels beyond those sensors, or initialization to the solver. Its quadrature-resolution comparison was about 2.6e-15 or better before the pause; this is a reference evaluation, not an externally evolved training trajectory.

The joint finite-penalty baseline included 14 cases across two viscosities, multiple seeds/starts, and noise 0, 0.001, and 0.01. **All 14 ended at their 60-second soft budgets.** The viscosity-0.1 cases had roughly 0.3–1.2% parameter error; viscosity-0.02 cases had 207–220% error. Every outcome remains in the [full baseline table](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_battletest/inverse/summary.csv) and [figure](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_battletest/inverse/inverse_campaign.png).

A frozen four-case extension increased the common budget to 600 s and the outer cap to 60, retaining zero readout starts, initial viscosity 0.04, N257, lambda 0.2, and the same generic degree ladder through 64. All four still returned `time_budget`:

| True viscosity | Noise | Recovered viscosity | Relative parameter error | Held-out field relative L2 | Ordinary PDE RMS | Returned degree |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.1 | 0 | 0.10000546 | 5.46e-5 | 1.95e-5 | 1.79e-3 | 40 |
| 0.1 | 0.001 | 0.09986853 | 1.31e-3 | 6.50e-4 | 7.11e-4 | 40 |
| 0.02 | 0 | 0.05338369 | 1.67 | 0.168 | 6.74e-2 | 32 |
| 0.02 | 0.001 | 0.05336531 | 1.67 | 0.168 | 6.76e-2 | 32 |

Actual times were 600–612 s; process RSS was 359–370 MB. More time helped the easier regime substantially but did not resolve the harder one. The actual-neural versus ideal-coordinate PDE discrepancies were about 1.1e-6–2.3e-6, much smaller than the remaining physics errors. Encoding error alone therefore does not explain these failures.

Separately, **physics-only forward gates block reduced inversion here**. With no sensors and viscosity fixed, the same 600-second policy returned PDE RMS 0.0859 at viscosity 0.04 and 0.1668 at viscosity 0.02; field errors were 8.86% and 26.7%. Both retained degree 32 as their best state. The reduced method correctly has no qualified inverse estimate to emit when these prerequisites fail.

A single predeclared 60-second control disabled inexact inner solves. It stopped at degree 14 with about 9.99% field error and PDE RMS 0.0822, worse than the corresponding baseline. This is evidence about allocating a short budget, not evidence that accurate linear solves are inherently harmful. The forward-gate histories often used only one or two Krylov iterations per outer step; loose stopping is a plausible issue, but its causal contribution has not been isolated.

Evidence: [600-second protocol](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_battletest/inverse/accuracy_extension/protocol.json), [four-case table](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_battletest/inverse/accuracy_extension/summary.csv), [existing accuracy figure](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_battletest/inverse/accuracy_extension/inverse_campaign.png), [physics-only gate summary](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_battletest/inverse/burgers_reduced_forward_gates/summary.json), [gate figure](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_battletest/inverse/burgers_reduced_forward_gates/forward_gate_summary.png), [tight-inner control directory](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_battletest/inverse/tight_control/).

## Actual cost and memory

Let \(T_{\mathrm{forward}}\) be the cost of one validated native forward, including continuation and checks; \(I\) the inverse iteration count; \(k\) the unknown parameter count; and \(L\) the number of line-search forward trials per iteration. Central finite differences cost approximately

\[
O\!\left(I\left[(2k+L)T_{\mathrm{forward}}+Sk^2+k^3\right]\right),
\]

plus the initial forward. Optional finite-difference step checks add forwards. Sensor-response storage is \(O(Sk)\); small parameter algebra uses \(O(k^2)\). The wrapper otherwise retains a bounded number of current/trial/native-guess states and the underlying forward workspace. It does not introduce a dense neuron-by-observation Jacobian. An ordinary model with \(B\) hidden neurons, \(d\) inputs, and \(F\) outputs still stores \(O(B(d+F))\) weights and evaluates a batch of \(E\) inputs in \(O(EB(d+F))\) arithmetic.

This does **not** establish inference-order construction. Forward resolution, encoder transforms, preconditioner setup, and iteration growth remain inside \(T_{\mathrm{forward}}\). For example, the box encoder can require \(O(MpP)\) conversion work for \(M\) directions and maximum coordinate degree \(p\), despite never retaining the full \(M\times p\times P\) map. Bounded memory is not a proof that all avoidable dimensional compute overhead has been removed.

The separate [blind NS forward gate](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_battletest/inverse/wall_ns_forward_gate/metrics.json) used **587 Krylov iterations, 46 outer iterations, and 159 residual evaluations** across degrees 8–32, taking 213 s. Its final coefficients matched the initial inverse forward. Aggregate iteration counts for all 15 original inverse forwards were not fully archived; multiplying this one count by 15 would be unjustified.

The original noiseless inverse recorded **1,384,611,840 bytes peak process RSS**. That run included an unbatched post-solve reference audit; the 31,863-neuron reference evaluated at 2,049 points creates about 522 MB for one activation array alone. This explains a major audit allocation but does not erase the measured process peak. A separate solver-only peak was not recorded.

An already-completed [bounded re-audit](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_battletest/inverse/wall_ns_reduced_inverse/noise_0/bounded_reaudit.json) used 64-row reference batches and **466,239,488 bytes peak RSS**, performed zero solves, and reproduced momentum RMS exactly; the reference batching difference was 6.55e-15. That is an isolated audit measurement, **not evidence that the original inverse solver stayed below 512 MB**. Future audit batching and logging improvements do not retroactively change the original record.

## Architecture, provenance, and what remains unproved

All current reduced-wrapper exports are a single hidden tanh layer. Ideal polynomial coordinates accelerate corrections or preconditioning, but the primal physics residual, accepted field, ordinary derivative audit, and deployed model use the actual tanh network. There is no window product, stitched time-slab field, deep compiler, or external PDE trajectory in these results. Earlier D02/D03 time-integrated methods retain their snapshot qualification and must not be relabeled as the current whole-domain solver.

The completed wall-NS result establishes a useful small-parameter inverse with sparse observations; the noisy diffusion control establishes separation of statistical error from physical consistency. Neither supplies a general nonlinear inverse theorem. Harder independently referenced Burgers cases remain unresolved; noisy NS was paused; global identifiability and continuum guarantees remain absent. There is no measured universal neuron/error law or bound on forward iteration growth.

The [loaded-source notes](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_battletest/inverse/wall_ns_reduced_inverse/loaded_source_notes.json) preserve a reporting-only bug in early inverse histories: `normalized_sensor_rms` had an extra division by the square root of the scalar observation count. Objectives, updates, and stopping were unaffected. Recover the correct normalized RMS as \(\sqrt{2J}\) from the recorded sensor objective rather than reusing that historical display field.

Reproduction entry points: [general inverse API](../../../experiments/expF19_radon_direct_pde/solver/route2_inverse.py), [Burgers campaign](../../../experiments/expF19_radon_direct_pde/route2_battletest_inverse.py), [wall-NS inverse](../../../experiments/expF19_radon_direct_pde/route2_battletest_inverse_ns.py), [Burgers feasibility gates](../../../experiments/expF19_radon_direct_pde/route2_battletest_inverse_burgers_gate.py). These links identify existing implementations; this archival task did not execute them.
