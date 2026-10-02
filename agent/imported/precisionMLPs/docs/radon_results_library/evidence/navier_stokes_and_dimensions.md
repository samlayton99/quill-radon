# Navier–Stokes, dimensional scaling, and encoding limits

This is an inventory of saved evidence, assembled on 2026-10-01. No new solve, model evaluation, or audit was performed for this document. The strongest results use one hidden tanh layer with the original physical inputs. They establish useful native PDE solves on smooth cases, not a general machine-precision Navier–Stokes solver.

## Method and source requirements

The deployed object is an ordinary `Linear/Tanh/Linear` network. Radon directions and QI centers specify its hidden features; analytical coordinates organize its readout. Polynomials used to organize coordinates or approximate a correction do **not** appear in the deployed forward pass. The nonlinear residual, accepted state, and exported-model checks use the neural function. The main solver uses streamed products and an approximate Jacobian built from ideal coordinate functions, with actual-neural checks and bounded preconditioning.

Manufactured benchmarks provide forcing and prescribed boundary/initial values. Their interior exact solution is reserved for validation. Physical-flow benchmarks provide zero body force and wall motion, with no interior solution labels. Saved native PDE iterates may initialize later stages. None of the headline native results below uses an external PDE trajectory or a known-function readout construction as initialization. The explicit known-target capacity study below is separate and must remain labeled accordingly.

Single-hidden-layer architecture and unchanged input dimension are satisfied. **The high-dimensional tensor direction count still grows exponentially with dimension.** Removing a dense readout solve does not remove that representation cost; these results do not satisfy an unrestricted “no dimensional blowup” requirement.

## Strongest measured three-, four-, and five-input results

Field errors in this table are relative L2 errors against held-out exact values. PDE residuals are raw RMS values, not normalized relative errors. Bytes are decimal MB/GB.

| Case | Ordinary network | Held-out field error | Raw PDE residual | Important limit |
|---|---|---|---|---|
| Steady, interacting manufactured 3D NS, ν=0.2 | 84,099 neurons; 3,876 readout coordinates; 5.38 MB model | Velocity **7.93e-15**; pressure **1.56e-14** | Momentum RMS **8.94e-16**, max **1.19e-14**; divergence RMS **1.03e-16** | Near-floor smooth forced test; pressure slightly exceeds 1e-14. Outer driver returned `resolution_limit`, although the inner solve converged. |
| Time-dependent 3D NS: three spatial inputs plus time, ν=0.2 | 751,689 neurons; 19,380 coordinates; 54.12 MB model | Velocity **1.84e-15**; pressure **6.37e-14** | Momentum RMS **9.60e-16**, max **4.02e-14**; divergence RMS **1.17e-16** | All-corner maximum field error **1.32e-13**; corner momentum max **1.46e-13**. Budget exit, not uniform 1e-14 accuracy. |
| Five-input semilinear elliptic control, −Δu+u³=f | 1,003,833 neurons; 1,287 coordinates; 56.21 MB model | Scalar field **2.73e-9** | PDE RMS **1.73e-8**, max **9.40e-8** | Corner maximum field error **5.37e-7**. Smooth full-rank manufactured scaling control; not a hard real-world 5D success. |

The 3D and space-time NS checks use 2,048 value points, 1,024 derivative points, all box corners, and larger boundary/initial-condition checks. These are sampled checks, not continuum certificates. The final 3D correction took **211 s** with roughly **450 MB** process peak RSS; the final space-time polish took **2,000 s** with **1.08 GB** peak RSS. Both started from earlier native iterates: these are final-stage costs, not cold end-to-end training times.

The five-input test is a cold run with degrees 4→6→8, 129 interior centers per direction, λ=0.25, and the full tensor angular rule. It uses u=exp(xᵀBx) only to declare forcing/boundary values and validate; B is dense, symmetric, rank five. Its independent ordinary-network audit has 257 interior values, 97 PDE points, 257 boundary points, and all 32 corners. The complete solve took **917 s**, audit **16 s**, peak RSS **476 MB**. Accuracy improved as follows:

| Degree | Neurons | Relative field error | Raw PDE RMS | Corner maximum error |
|---|---:|---:|---:|---:|
| 4 | 95,625 | 3.32e-5 | 8.88e-5 | 9.41e-4 |
| 6 | 367,353 | 3.42e-7 | 1.31e-6 | 2.43e-5 |
| 8 | 1,003,833 | 2.73e-9 | 1.73e-8 | 5.37e-7 |

The baseline was manually interrupted after 929 s inside an unfinished actual-Jacobian safeguard at degree 6. A separate cold control changed only the optional `refine_on_ideal_stall` policy: an uncertified stalled approximate correction permits trying the next resolution. Shared earlier states were verified bitwise identical. The control reached degree 8 and returned normally at its soft time budget. This policy is a resource-allocation choice, **not** a stationarity or representation-capacity certificate. The p8 stage used only 17 inner iterations but took 619 s; separate component timings were not instrumented.

Evidence and reproducibility:

- 3D: [saved solve](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_hardening/ns/trig_p16_n257_native_ideal_correction.json), [expanded audit](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_hardening/ns/trig_p16_n257_native_ideal_correction_extended_audit.json), [accuracy figure](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_hardening/ns/native_3d_accuracy_ladder.png).
- Space-time: [saved solve](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_hardening/ns/transient_trig_p16_n129_native_fullgroups.json), [expanded audit of its final accepted checkpoint](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_hardening/ns/transient_trig_p16_n129_native_fullgroups_iteration2_native_extended_audit.json), [accuracy figure](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_hardening/ns/native_4d_accuracy_ladder.png).
- Five inputs: [control result and full stage history](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_hardening/five_dimensional_refinement_control/results.json), [comparison/provenance](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_hardening/five_dimensional_refinement_control/comparison_provenance.json), [baseline disposition](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_hardening/five_dimensional/baseline_summary.json), [accuracy/cost figure](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_hardening/five_dimensional_refinement_control/five_dimensional_accuracy_cost.png).
- Scripts: [NS declaration and runs](../../../experiments/expF19_radon_direct_pde/route2_hardening_ns.py), [expanded ordinary-AD audit](../../../experiments/expF19_radon_direct_pde/route2_ns_extended_audit.py), [five-input experiment and figure reproduction](../../../experiments/expF19_radon_direct_pde/route2_native_five_dimensional.py).

## Physical flows: successes and harder failures

These steady 2D flows have **no known truth error**. The measurements below are independent ordinary-network PDE and wall residuals. Small residuals plus resolution agreement are useful evidence, but do not establish machine-precision solution error or uniqueness. All disk cases use prescribed tangential wall motion, zero forcing, and a pressure gauge.

| Saved physical case | Neurons | Momentum RMS / maximum | Divergence RMS | Outcome and utility |
|---|---:|---:|---:|---|
| Disk stirring, ν=0.1, p40/N257 | 11,931 | **3.93e-15 / 3.11e-14** | 3.56e-16 | Strongest compact physical-flow demonstration; native continuation, one final correction, 295 s. |
| Disk stirring, ν=0.03, p48/N257 | 14,259 | **6.71e-15 / 1.97e-13** | 1.29e-15 | Lower-viscosity success by RMS; maximum residual is materially larger. Final native correction 551 s. |
| Counterrotating wall, ν=0.1, p64/N513, recurrence control | 36,335 | **3.20e-11 / 1.16e-9** | 3.85e-11 | Completed budget exit after 879 s; useful harder case, not floor. |
| Counterrotating wall, ν=0.03, same resolution/control | 36,335 | **7.78e-3 / 5.06e-1** | 8.37e-3 | Completed budget exit after 610 s; substantial failure at this protocol. |
| Regularized lid cavity, ν=0.1, p32/N257, enclosing-disk coordinates | 9,603 | **1.14e-4 / 4.94e-3** | 5.92e-5 | No floor; near-corner momentum RMS 5.42e-3 and max 1.43e-2. Domain/trace difficulty remains. |

The successful stirring cases were also re-encoded at p48/N513 and p56/N513. Those checks performed **zero optimization iterations**; they test the continued state in a richer declared representation and on fresh residual points, not a separately optimized richer solution. The measured momentum RMS remained 4.25e-15 and 7.53e-15 respectively. The p32 cavity Box-coordinate variant attained RMS 9.40e-5 but had larger encoding discrepancies; neither cavity variant solved the difficult corners.

The latest recurrence campaign changes both the implementation and the preconditioner budget. It therefore gives **no isolated end-to-end recurrence speedup claim**. At inventory time, only its two counterrotating cases have completed result files; ν=0.01 stirring has a preflight/checkpoint, and ν=0.003 is listed in the protocol. Neither is a completed result here.

- Successful physical flows: [ν=0.1 result](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_hardening/physical_flow/highdegree_ideal_disk_stirring_nu0.1_p40_n257.json), [ν=0.03 result](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_hardening/physical_flow/lowviscosity_refined_disk_stirring_nu0.03_p48_n257.json), [p48 re-encoding check](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_hardening/physical_flow/resolution_check_disk_stirring_nu0.1_p48_n513.json), [p56 re-encoding check](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_hardening/physical_flow/lowviscosity_resolution_check_disk_stirring_nu0.03_p56_n513.json).
- Harder outcomes: [recurrence campaign summary](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_battletest/flows_recurrence/summary.json), [frozen protocol](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_battletest/flows_recurrence/protocol.json), [ν=0.1 full result](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_battletest/flows_recurrence/recurrence_disk_counterrotating_nu0.1_p64_n513.json), [ν=0.03 full result](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_battletest/flows_recurrence/recurrence_disk_counterrotating_nu0.03_p64_n513.json), [enclosing-disk cavity result](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_hardening/physical_flow/enclosing_disk_native_cavity_nu0.1_p32_n257.json).
- View/reproduce: [physical-flow figure](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_hardening/physical_flow/disk_resolution_flow_and_residual.png), [original flow runner](../../../experiments/expF19_radon_direct_pde/route2_physical_flow.py), [recurrence campaign runner](../../../experiments/expF19_radon_direct_pde/route2_physical_recurrence_study.py), [figure script](../../../experiments/expF19_radon_direct_pde/route2_physical_figures.py).

## Dimension forecasts: architecture counts, not successful solves

Let d be the number of inputs, p the maximum coordinate degree, N the interior centers per direction, F the output count, and R=ceil(√N) the halo count per side in these runs. The full tensor rule has M=(p+1)^(d−1) directions, B=M(N+2R) hidden neurons, and F·binomial(p+d,d) readout coordinates. The exported float64 model occupies 8[B(d+1+F)+F] bytes. These are exact counts for this rule; total solver RSS is larger.

At p16, N257, F4:

| Inputs d | Neurons B | Ordinary model storage | Evidence status |
|---|---:|---:|---|
| 3 | 84,099 | 5.38 MB | Measured native NS result above. |
| 4 | 1,429,683 | 102.94 MB | This N257 architecture was attempted; strongest final space-time result uses smaller N129. |
| 5 | 24,304,611 | 1.94 GB | Count only; measured 5D success above is p8, N129, F1. |
| 6 | 413,178,387 | 36.36 GB | Count only; exceeds the 16 GB test machine before solver workspace. |

[Full saved count table](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_hardening/dimension_footprint.json). Do not read these rows as a precision scaling law. Neither the required p nor the attainable floating-point accuracy is determined by dimension alone.

## Encoding and recurrence controls

**Encoding limits.** A saved native Helmholtz p32 solution already had 3.22% ideal-coordinate field error. Holding those learned coordinates fixed, increasing N from 257 to 1025 at λ=0.2 reduced ordinary-versus-ideal encoding error from 3.14e-6 to 2.49e-8; it did not repair the original field error. Readout L1 fell from 1.85e10 to 6.65e7, with over 99.78% of L1 in halo neurons. Broad halo continuation of high-degree profiles and large learned coefficients are a practical failure mechanism.

A separate explicit Fourier/Bessel construction used the known target only to audit capacity. At p64/N513, ideal field error was 1.95e-15 and stable evaluation of the encoded neural sum gave 2.19e-15, but ordinary tanh inference gave **9.91e-12**. This is evidence of an ordinary-arithmetic deployment limit even with a bounded analytical extension; it is **not** a native PDE solution or initialization. [Fixed-native encoding results](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_hardening/hard_encoding/fixed_native_helmholtz.json), [explicit capacity results](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_hardening/hard_encoding/analytic_capacity_helmholtz.json), [figure](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_hardening/hard_encoding/encoding_limits.png), [script](../../../experiments/expF19_radon_direct_pde/route2_hard_encoding_sweep.py).

**Disk recurrence.** Shared Jacobi-family recurrences accelerate ideal-coordinate products while leaving actual tanh features and exported weights unchanged. The isolated p64, 128-row, three-output microbenchmark measured **2.62× forward / 2.57× adjoint** speedup, with bounded batch workspace. This is a reusable implementation component, not a new PDE method. The alternative complete-row panel version used more memory without improving that benchmark. [Saved timings](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_hardening/disk_recurrence/benchmark_p16_p32_p64.json), [opt-in implementation](../../../experiments/expF19_radon_direct_pde/route2_disk_recurrence.py).

## Utility within the problem-class hierarchy

See [rankings by problem class](../rankings.md). There is no global ranking across known-function construction, blind PDEs and inverse estimation. The native actual-residual/ideal-correction method is the main forward-PDE platform; disk, affine-disk and factored-box implementations are variants within it. The all-actual-Jacobian path is a fidelity baseline and safeguard within that same family. Explicit known-function construction has a separate primary use, not merely a supporting role for PDE diagnosis.

The physical recurrence campaign was stopped at the user's organization request after two completed counterrotation budget exits. The viscosity .01 stirring checkpoint is interrupted, and the .003 case was unstarted; neither is a completed success or numerical failure. [Administrative status](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_battletest/flows_recurrence/interruption_status.json).

Recurrences, tensor contractions, parity blocks, bounded batching, warm continuation, and the optional ideal-stall refinement policy belong under these methods as components or controls. Counting each as a separate successful solver would obscure what the evidence actually compares.
