# Current findings: representation-to-solve gap

Coordinator: root. Started 2026-10-02. **Chronological working log; preliminary observations are preserved.** User explicitly authorized continuing experiments after the October 1 organization pause. Every numerical run must append its setup before launch and its outcome afterward, including failures and interruptions. Convincing findings will later be promoted into the results library with links back here.

**October 2 first batch completed:** 25 G-runs and seven P-controls are recorded below. The supported findings have been promoted to [coefficient-access evidence](../evidence/coefficient_access_2026_10_02.md); the chronological entries remain unchanged. The adaptive native pair reaches ordinary field errors 7.63e-16/9.74e-16; only contrast 1000 passes all driver checks, while contrast 10 ends at its soft time budget. All runs in this batch have finished. The broader nonlinear/high-dimensional transfer question remains open.

## Question and fixed requirements

Why can the current single-hidden-layer Radon network represent an accurate field while native PDE coefficient discovery sometimes fails? Separate (1) iterative access to a fixed least-squares objective, (2) objective/sampling/constraint limitations, and (3) ordinary-network arithmetic/encoding. Final deployment remains one Linear/Tanh/Linear network on original inputs. No external PDE solution, known interior target values or reference coefficients initialize the blind solve. Numerical help is permitted; a dense small diagnostic is not a proposed scalable solver.

## First controlled investigation

Use the existing paired manufactured diffusion equations (contrasts 10 and 1000), unchanged target, domain, native feature construction and deterministic collocation seeds. Freeze degree rather than conflate solver quality with the adaptive ladder. Compare actual encoded neural columns with ideal analytical columns, and raw with already-existing equation-normalized residuals. A small dense QR/SVD reference is allowed **only as a diagnostic control**; its memory/compute and rank cutoff are recorded. Interior truth is evaluation-only; forcing and boundary conditions are prescribed inputs.

Questions: does an accurately solved small reference recover the good field? Does it disagree with the archived native coefficients? Is the gap already present in ideal coordinates, or introduced by neural encoding/cancellation? Do normalization and rank cutoffs change the answer?

## Provenance and ownership

- Root owns this log and the diagnostic runner/results. An explorer is auditing the existing stopping logic read-only.
- Existing production solver code and archived results remain unchanged during the first comparison.
- Local context installation checks passed; cross-machine context sync failed transport. Work uses local evidence without claiming refreshed remote state.
- Runtime probe: workspace Python, NumPy 2.5.1, SciPy 1.18.0, Torch 2.4.1. Current machine is the existing Mac mini workspace; record actual RSS/timing, not laptop estimates.
- Results: `results/checkpoint_F_applications/expF19_radon_direct_pde/route2_gap_diagnosis/`.

## Run log

Entries below are appended by the diagnostic runner. A start without a finish is not a completed result. Setup/import and saved-file inspections are not counted as new scientific experiments.

### G001_p24_c1000_actual_raw — START 2026-10-02 07:11:01 UTC

Fixed-degree diffusion_c1000, p=24, N=257, lambda=0.2, actual columns, raw weighting, SVD cutoff=1e-14. Physics/BC only; dense reference diagnostic, no production solver change. Artifacts: `results/checkpoint_F_applications/expF19_radon_direct_pde/route2_gap_diagnosis/G001_p24_c1000_actual_raw`.

**FINISH 2026-10-02 07:11:02 UTC — complete.** Ordinary field relL2 2.86728e-09; ideal 2.86728e-09; raw ordinary PDE RMS 8.06261e-09; boundary RMS 3.68893e-09. Rank 325/325; condition 1.66e+08; readout L1 3.75. 0.9s; process peak 353.9 MiB. Interpretation pending paired controls; no automatic floor claim.

### Read-only audit — 2026-10-02

The earlier normalized outputs labeled degree 32 did **zero corrections at degree 32**: they embedded degree-24 coefficients, then passed the loose scaled stopping/refinement check. The tighter V3 runs instead spent 533/564 seconds and 1509/1505 Krylov steps at degree 24, never reaching degree 32. Therefore the archived 3.26e-11 / 8.03e-11 field errors are not evidence of a fitted p32 accuracy floor. Source: saved stage histories and explorer audit of route2.py/general_residual.py. Ideal corrections solve a different linearization from the actual-neural objective; guarded actual corrections can consume the remaining budget. No production change yet.

### G002_p24_c1000_actual_normalized — START 2026-10-02 07:11:56 UTC

Fixed-degree diffusion_c1000, p=24, N=257, lambda=0.2, actual columns, normalized weighting, SVD cutoff=1e-14. Physics/BC only; dense reference diagnostic, no production solver change. Artifacts: `results/checkpoint_F_applications/expF19_radon_direct_pde/route2_gap_diagnosis/G002_p24_c1000_actual_normalized`.

**FINISH 2026-10-02 07:11:57 UTC — complete.** Ordinary field relL2 8.00135e-11; ideal 8.00135e-11; raw ordinary PDE RMS 2.98526e-08; boundary RMS 9.04389e-11. Rank 325/325; condition 1.36e+08; readout L1 3.72. 0.8s; process peak 353.7 MiB. Interpretation pending paired controls; no automatic floor claim.

### G003_p32_c1000_actual_raw — START 2026-10-02 07:11:58 UTC

Fixed-degree diffusion_c1000, p=32, N=257, lambda=0.2, actual columns, raw weighting, SVD cutoff=1e-14. Physics/BC only; dense reference diagnostic, no production solver change. Artifacts: `results/checkpoint_F_applications/expF19_radon_direct_pde/route2_gap_diagnosis/G003_p32_c1000_actual_raw`.

**FINISH 2026-10-02 07:12:00 UTC — complete.** Ordinary field relL2 8.47761e-13; ideal 8.47741e-13; raw ordinary PDE RMS 8.08566e-12; boundary RMS 1.27797e-12. Rank 561/561; condition 3e+11; readout L1 3.73. 2.2s; process peak 421.5 MiB. Interpretation pending paired controls; no automatic floor claim.

### G004_p32_c1000_actual_normalized — START 2026-10-02 07:12:01 UTC

Fixed-degree diffusion_c1000, p=32, N=257, lambda=0.2, actual columns, normalized weighting, SVD cutoff=1e-14. Physics/BC only; dense reference diagnostic, no production solver change. Artifacts: `results/checkpoint_F_applications/expF19_radon_direct_pde/route2_gap_diagnosis/G004_p32_c1000_actual_normalized`.

**FINISH 2026-10-02 07:12:03 UTC — complete.** Ordinary field relL2 3.49739e-13; ideal 3.49807e-13; raw ordinary PDE RMS 1.03098e-10; boundary RMS 5.10214e-13. Rank 561/561; condition 3.03e+12; readout L1 3.75. 2.2s; process peak 421.2 MiB. Interpretation pending paired controls; no automatic floor claim.

### G005_p32_c10_actual_raw — START 2026-10-02 07:12:04 UTC

Fixed-degree diffusion_c10, p=32, N=257, lambda=0.2, actual columns, raw weighting, SVD cutoff=1e-14. Physics/BC only; dense reference diagnostic, no production solver change. Artifacts: `results/checkpoint_F_applications/expF19_radon_direct_pde/route2_gap_diagnosis/G005_p32_c10_actual_raw`.

**FINISH 2026-10-02 07:12:06 UTC — complete.** Ordinary field relL2 3.77936e-13; ideal 3.779e-13; raw ordinary PDE RMS 5.23408e-12; boundary RMS 9.57078e-13. Rank 561/561; condition 3.51e+11; readout L1 3.74. 2.2s; process peak 421.6 MiB. Interpretation pending paired controls; no automatic floor claim.

### G006_p32_c1000_ideal_raw — START 2026-10-02 07:12:07 UTC

Fixed-degree diffusion_c1000, p=32, N=257, lambda=0.2, ideal columns, raw weighting, SVD cutoff=1e-14. Physics/BC only; dense reference diagnostic, no production solver change. Artifacts: `results/checkpoint_F_applications/expF19_radon_direct_pde/route2_gap_diagnosis/G006_p32_c1000_ideal_raw`.

**FINISH 2026-10-02 07:12:08 UTC — complete.** Ordinary field relL2 7.75365e-13; ideal 7.7537e-13; raw ordinary PDE RMS 8.03511e-12; boundary RMS 1.27044e-12. Rank 561/561; condition 3e+11; readout L1 3.73. 0.5s; process peak 420.7 MiB. Interpretation pending paired controls; no automatic floor claim.

### G007_p24_c1000_ideal_normalized — START 2026-10-02 07:12:09 UTC

Fixed-degree diffusion_c1000, p=24, N=257, lambda=0.2, ideal columns, normalized weighting, SVD cutoff=1e-14. Physics/BC only; dense reference diagnostic, no production solver change. Artifacts: `results/checkpoint_F_applications/expF19_radon_direct_pde/route2_gap_diagnosis/G007_p24_c1000_ideal_normalized`.

**FINISH 2026-10-02 07:12:09 UTC — complete.** Ordinary field relL2 8.00243e-11; ideal 8.00243e-11; raw ordinary PDE RMS 2.98514e-08; boundary RMS 9.04437e-11. Rank 325/325; condition 1.36e+08; readout L1 3.72. 0.3s; process peak 353.9 MiB. Interpretation pending paired controls; no automatic floor claim.

### G008_p32_c1000_actual_refine — START 2026-10-02 07:13:50 UTC

Fixed-degree diffusion_c1000, p=32, N=257, lambda=0.2, actual columns, raw weighting, SVD cutoff=1e-14. Physics/BC only; dense reference diagnostic, no production solver change. Artifacts: `results/checkpoint_F_applications/expF19_radon_direct_pde/route2_gap_diagnosis/G008_p32_c1000_actual_refine`.

**FINISH 2026-10-02 07:13:57 UTC — complete.** Ordinary field relL2 7.90229e-13; ideal 7.90198e-13; raw ordinary PDE RMS 5.09181e-12; boundary RMS 9.97196e-13. Rank 561/561; condition 3e+11; readout L1 3.73. 6.8s; process peak 422.1 MiB. Interpretation pending paired controls; no automatic floor claim.

### G009_p32_c1000_normalized_refine — START 2026-10-02 07:15:00 UTC

Fixed-degree diffusion_c1000, p=32, N=257, lambda=0.2, actual columns, normalized weighting, SVD cutoff=1e-14. Physics/BC only; dense reference diagnostic, no production solver change. Artifacts: `results/checkpoint_F_applications/expF19_radon_direct_pde/route2_gap_diagnosis/G009_p32_c1000_normalized_refine`.

### G010_p40_c1000_actual_raw — START 2026-10-02 07:15:01 UTC

Fixed-degree diffusion_c1000, p=40, N=257, lambda=0.2, actual columns, raw weighting, SVD cutoff=1e-14. Physics/BC only; dense reference diagnostic, no production solver change. Artifacts: `results/checkpoint_F_applications/expF19_radon_direct_pde/route2_gap_diagnosis/G010_p40_c1000_actual_raw`.

### G011_p40_c1000_actual_normalized — START 2026-10-02 07:15:02 UTC

Fixed-degree diffusion_c1000, p=40, N=257, lambda=0.2, actual columns, normalized weighting, SVD cutoff=1e-14. Physics/BC only; dense reference diagnostic, no production solver change. Artifacts: `results/checkpoint_F_applications/expF19_radon_direct_pde/route2_gap_diagnosis/G011_p40_c1000_actual_normalized`.

### G012_p32_c1000_column_scaled — START 2026-10-02 07:15:04 UTC

Fixed-degree diffusion_c1000, p=32, N=257, lambda=0.2, actual columns, raw weighting, SVD cutoff=1e-14. Physics/BC only; dense reference diagnostic, no production solver change. Artifacts: `results/checkpoint_F_applications/expF19_radon_direct_pde/route2_gap_diagnosis/G012_p32_c1000_column_scaled`.

**FINISH 2026-10-02 07:15:08 UTC — complete.** Ordinary field relL2 3.75037e-15; ideal 3.76321e-15; raw ordinary PDE RMS 1.08271e-11; boundary RMS 8.24601e-15. Rank 561/561; condition 3.03e+12; readout L1 3.75. 8.0s; process peak 422.3 MiB. Interpretation pending paired controls; no automatic floor claim.

**FINISH 2026-10-02 07:15:11 UTC — complete.** Ordinary field relL2 7.90017e-13; ideal 7.90017e-13; raw ordinary PDE RMS 5.1074e-12; boundary RMS 9.96937e-13. Rank 561/561; condition 3.08e+10; readout L1 3.73. 7.2s; process peak 426.9 MiB. Interpretation pending paired controls; no automatic floor claim.

**FINISH 2026-10-02 07:15:15 UTC — complete.** Ordinary field relL2 1.59378e-12; ideal 3.10334e-13; raw ordinary PDE RMS 3.89694e-09; boundary RMS 2.80032e-12. Rank 859/861; condition 1.92e+14; readout L1 4.75e+04. 14.2s; process peak 547.7 MiB. Interpretation pending paired controls; no automatic floor claim.

**FINISH 2026-10-02 07:15:16 UTC — complete.** Ordinary field relL2 2.81047e-11; ideal 5.23565e-14; raw ordinary PDE RMS 3.27256e-08; boundary RMS 3.01969e-11. Rank 853/861; condition 1.87e+15; readout L1 2.82e+05. 14.1s; process peak 542.2 MiB. Interpretation pending paired controls; no automatic floor claim.

### G013_bounded_native_diffusion_c10 — START 2026-10-02 07:17:02 UTC

Cold native diffusion_c10; normalized PDE, stage cap 6 outer / 80 Krylov, 180.0s soft total budget, degree ladder4–40, ideal-stall refinement enabled. Shared policy; no dense reference state enters solve. Artifacts: `results/checkpoint_F_applications/expF19_radon_direct_pde/route2_gap_diagnosis/G013_bounded_native_diffusion_c10`.

### G014_bounded_native_diffusion_c1000 — START 2026-10-02 07:17:02 UTC

Cold native diffusion_c1000; normalized PDE, stage cap 6 outer / 80 Krylov, 180.0s soft total budget, degree ladder4–40, ideal-stall refinement enabled. Shared policy; no dense reference state enters solve. Artifacts: `results/checkpoint_F_applications/expF19_radon_direct_pde/route2_gap_diagnosis/G014_bounded_native_diffusion_c1000`.

### Interpretation after G001–G012 (provisional)

- p24 normalized dense actual and ideal solves both give about8.00e-11 field error, matching the earlier native plateau. This supports refining that space instead of spending the whole budget polishing it.
- p32 normalized actual solve with recomputed-native-residual refinement (G009) reaches **3.75e-15 ordinary field relative error**, boundary RMS8.25e-15, but raw PDE RMS remains1.08e-11. This is a blind PDE/BC coefficient solve: no interior target data. The dense control is diagnostic, not the scalable method.
- Raising p to40 at N257 worsens ordinary arithmetic. Normalized G011 has ideal field5.24e-14 but ordinary field2.81e-11 and readoutL1~2.82e5. More coordinates are not automatically safer.
- G008+ independently compare assembled columns against the native operator: relative discrepancies~1e-13 atp32, ~1e-11 atp40; RHS exactly agrees. These are finite-arithmetic discrepancies, not silently different equations.
- G009–G012 overlapped as separate single-thread processes; their elapsed times are not isolated speed benchmarks. G013/G014 are the next paired native-policy control and also run concurrently. No production default has changed.

**FINISH 2026-10-02 07:20:13 UTC — time_budget.** Returned p32; ordinary field relL2 1.50186e-12; raw PDE RMS 6.97395e-10; readout L1 4.521; 190.9s, 373.0MiB. All stage exits/inner counts retained. A budget exit is not a precision-floor certificate.

**FINISH 2026-10-02 07:20:18 UTC — time_budget.** Returned p32; ordinary field relL2 1.62312e-12; raw PDE RMS 3.56474e-09; readout L1 6.288; 196.3s, 373.4MiB. All stage exits/inner counts retained. A budget exit is not a precision-floor certificate.

### P001_actual_contiguous64 — START 2026-10-02 07:20:31 UTC

Fixed p32 diffusion_c1000, N257, lambda0.2, actual columns, normalized PDE and unit column scaling. Same matrix and contiguous 64-coordinate blocks: diagonal, regularized Gram, and guarded QR. LSMR cap 800; no target values in solve. Dense diagnostic only. Artifacts: `results/checkpoint_F_applications/expF19_radon_direct_pde/route2_gap_diagnosis/P001_actual_contiguous64`.

P001_actual_contiguous64 / diagonal: 800 LSMR iterations, algebraic residual 6.9349e-05, ordinary field relL2 4.37635e-06, raw PDE RMS 0.00433892, transformed condition 1.233e+11. Setup 0.08s, solve 0.56s. Finite iteration diagnostic; low objective alone is not a field certificate.

P001_actual_contiguous64 / gram_ridge: 800 LSMR iterations, algebraic residual 0.00020647, ordinary field relL2 1.06706e-05, raw PDE RMS 0.0105164, transformed condition 6.629e+09. Setup 0.08s, solve 0.56s. Finite iteration diagnostic; low objective alone is not a field certificate.

P001_actual_contiguous64 / qr: 800 LSMR iterations, algebraic residual 0.000206159, ordinary field relL2 1.06497e-05, raw PDE RMS 0.0105071, transformed condition 6.629e+09. Setup 0.11s, solve 0.56s. Finite iteration diagnostic; low objective alone is not a field certificate.

**P001_actual_contiguous64 FINISH 2026-10-02 07:20:35 UTC — complete.** Three preconditioner variants retained; total 4.81s, process peak 428.8MiB. Dense matrix diagnostics deliberately exceed scalable-storage policy; factor storage alone is recorded separately.

### G015_box_p32_c1000_ideal_normalized — START 2026-10-02 07:21:15 UTC

Fixed-degree diffusion_c1000, p=32, N=257, lambda=0.2, ideal columns, normalized weighting, SVD cutoff=1e-14. Physics/BC only; dense reference diagnostic, no production solver change. Artifacts: `results/checkpoint_F_applications/expF19_radon_direct_pde/route2_gap_diagnosis/G015_box_p32_c1000_ideal_normalized`.

### P002_actual_contiguous64_long — START 2026-10-02 07:21:15 UTC

Fixed p32 diffusion_c1000, N257, lambda0.2, actual columns, normalized PDE and unit column scaling. Same matrix and contiguous 64-coordinate blocks: diagonal, regularized Gram, and guarded QR. LSMR cap 10000; no target values in solve. Dense diagnostic only. Artifacts: `results/checkpoint_F_applications/expF19_radon_direct_pde/route2_gap_diagnosis/P002_actual_contiguous64_long`.

### Additional controlled question: coordinate chart

G015 uses the existing box-Legendre chart at the same total degree32, same PDE/BC locations and normalized objective. It uses ideal columns only for this diagnostic and still audits ordinary deployed tanh afterward. In exact polynomial algebra this spans the same degree space as the enclosing-disk chart, but conditioning on the physical box can differ. A good ideal solve with bad exported neural accuracy would isolate a chart-to-network construction gap. This is not a new production method and not permission to deploy polynomials.

**FINISH 2026-10-02 07:21:17 UTC — complete.** Ordinary field relL2 2.62208e-13; ideal 2.62124e-13; raw ordinary PDE RMS 7.15652e-11; boundary RMS 4.43709e-13. Rank 561/561; condition 1.03e+05; readout L1 4.01. 2.1s; process peak 389.3 MiB. Interpretation pending paired controls; no automatic floor claim.

P002_actual_contiguous64_long / diagonal: 10000 LSMR iterations, algebraic residual 3.45017e-06, ordinary field relL2 1.22086e-07, raw PDE RMS 0.000185322, transformed condition 1.233e+11. Setup 0.09s, solve 7.09s. Finite iteration diagnostic; low objective alone is not a field certificate.

P002_actual_contiguous64_long / gram_ridge: 10000 LSMR iterations, algebraic residual 7.15288e-06, ordinary field relL2 3.30554e-07, raw PDE RMS 0.000421471, transformed condition 6.629e+09. Setup 0.08s, solve 7.07s. Finite iteration diagnostic; low objective alone is not a field certificate.

P002_actual_contiguous64_long / qr: 10000 LSMR iterations, algebraic residual 7.13145e-06, ordinary field relL2 3.29212e-07, raw PDE RMS 0.000420342, transformed condition 6.629e+09. Setup 0.11s, solve 7.09s. Finite iteration diagnostic; low objective alone is not a field certificate.

**P002_actual_contiguous64_long FINISH 2026-10-02 07:21:40 UTC — complete.** Three preconditioner variants retained; total 24.49s, process peak 428.8MiB. Dense matrix diagnostics deliberately exceed scalable-storage policy; factor storage alone is recorded separately.

### P003_actual_production280 — START 2026-10-02 07:21:49 UTC

Fixed p32 diffusion_c1000, N257, lambda0.2, actual columns, normalized PDE and unit column scaling. Same matrix and contiguous 280-coordinate blocks: diagonal, regularized Gram, and guarded QR. LSMR cap 800; no target values in solve. Dense diagnostic only. Artifacts: `results/checkpoint_F_applications/expF19_radon_direct_pde/route2_gap_diagnosis/P003_actual_production280`.

P003_actual_production280 / diagonal: 800 LSMR iterations, algebraic residual 6.9349e-05, ordinary field relL2 4.37635e-06, raw PDE RMS 0.00433892, transformed condition 1.233e+11. Setup 0.09s, solve 0.56s. Finite iteration diagnostic; low objective alone is not a field certificate.

P003_actual_production280 / gram_ridge: 800 LSMR iterations, algebraic residual 9.84885e-05, ordinary field relL2 4.69821e-06, raw PDE RMS 0.00604627, transformed condition 3.986e+07. Setup 0.10s, solve 0.56s. Finite iteration diagnostic; low objective alone is not a field certificate.

P003_actual_production280 / qr: 800 LSMR iterations, algebraic residual 9.50979e-05, ordinary field relL2 4.5707e-06, raw PDE RMS 0.00574365, transformed condition 3.986e+07. Setup 0.15s, solve 0.56s. Finite iteration diagnostic; low objective alone is not a field certificate.

**P003_actual_production280 FINISH 2026-10-02 07:21:54 UTC — complete.** Three preconditioner variants retained; total 4.84s, process peak 427.8MiB. Dense matrix diagnostics deliberately exceed scalable-storage policy; factor storage alone is recorded separately.

### G016_box_p32_c1000_ideal_refine — START 2026-10-02 07:22:38 UTC

Fixed-degree diffusion_c1000, p=32, N=257, lambda=0.2, ideal columns, normalized weighting, SVD cutoff=1e-14. Physics/BC only; dense reference diagnostic, no production solver change. Artifacts: `results/checkpoint_F_applications/expF19_radon_direct_pde/route2_gap_diagnosis/G016_box_p32_c1000_ideal_refine`.

### G018_box_native_diffusion_c1000 — START 2026-10-02 07:22:39 UTC

Cold native diffusion_c1000; coordinates=box; normalized PDE, stage cap 6 outer / 80 Krylov, 180.0s soft total budget, degree ladder4–40, ideal-stall refinement enabled. Shared policy; no dense reference state enters solve. Artifacts: `results/checkpoint_F_applications/expF19_radon_direct_pde/route2_gap_diagnosis/G018_box_native_diffusion_c1000`.

### G017_box_native_diffusion_c10 — START 2026-10-02 07:22:39 UTC

Cold native diffusion_c10; coordinates=box; normalized PDE, stage cap 6 outer / 80 Krylov, 180.0s soft total budget, degree ladder4–40, ideal-stall refinement enabled. Shared policy; no dense reference state enters solve. Artifacts: `results/checkpoint_F_applications/expF19_radon_direct_pde/route2_gap_diagnosis/G017_box_native_diffusion_c10`.

**FINISH 2026-10-02 07:22:43 UTC — complete.** Ordinary field relL2 3.88329e-15; ideal 3.77098e-15; raw ordinary PDE RMS 1.07708e-11; boundary RMS 8.32332e-15. Rank 561/561; condition 1.03e+05; readout L1 3.75. 5.3s; process peak 389.2 MiB. Interpretation pending paired controls; no automatic floor claim.

### P004_ideal_native_parity — START 2026-10-02 07:23:06 UTC

Fixed p32 diffusion_c1000, N257, lambda0.2, ideal columns, normalized PDE and unit column scaling. Same matrix, auto groups capped at 1024 coordinates: diagonal, regularized Gram, and guarded QR. LSMR cap 800; no target values in solve. Dense diagnostic only. Artifacts: `results/checkpoint_F_applications/expF19_radon_direct_pde/route2_gap_diagnosis/P004_ideal_native_parity`.

P004_ideal_native_parity / diagonal: 800 LSMR iterations, algebraic residual 6.92405e-05, ordinary field relL2 4.35007e-06, raw PDE RMS 0.00432917, transformed condition 1.233e+11. Setup 0.10s, solve 0.59s. Finite iteration diagnostic; low objective alone is not a field certificate.

P004_ideal_native_parity / gram_ridge: 800 LSMR iterations, algebraic residual 5.37545e-06, ordinary field relL2 1.25854e-07, raw PDE RMS 0.000239469, transformed condition 1.395e+05. Setup 0.10s, solve 0.57s. Finite iteration diagnostic; low objective alone is not a field certificate.

P004_ideal_native_parity / qr: 800 LSMR iterations, algebraic residual 1.76543e-08, ordinary field relL2 4.66003e-10, raw PDE RMS 4.70131e-07, transformed condition 1419. Setup 0.14s, solve 0.57s. Finite iteration diagnostic; low objective alone is not a field certificate.

**P004_ideal_native_parity FINISH 2026-10-02 07:23:09 UTC — complete.** Three preconditioner variants retained; total 3.76s, process peak 429.9MiB. Dense matrix diagnostics deliberately exceed scalable-storage policy; factor storage alone is recorded separately.

### P001–P003 grouping clarification — 2026-10-02 07:23:22 UTC

P001/P002 used 64-coordinate contiguous blocks, and P003 used 280-coordinate contiguous blocks. These match the low-level preconditioner grouping with field_parity=none, but **do not match the native preset**, which sets block_size1024 (capped at P//2) and field_parity=auto. Their filenames/protocol snapshots are retained. P004 is the native parity-grouped ideal-column comparison. Do not interpret P001–P003 as production-default ablations.

### P005_ideal_native_parity_long — START 2026-10-02 07:23:57 UTC

Fixed p32 diffusion_c1000, N257, lambda0.2, ideal columns, normalized PDE and unit column scaling. Same matrix, auto groups capped at 1024 coordinates: diagonal, regularized Gram, and guarded QR. LSMR cap 4000; no target values in solve. Dense diagnostic only. Artifacts: `results/checkpoint_F_applications/expF19_radon_direct_pde/route2_gap_diagnosis/P005_ideal_native_parity_long`.

P005_ideal_native_parity_long / diagonal: 4000 LSMR iterations, algebraic residual 6.89136e-06, ordinary field relL2 3.81131e-07, raw PDE RMS 0.000317534, transformed condition 1.233e+11. Setup 0.10s, solve 3.05s. Finite iteration diagnostic; low objective alone is not a field certificate.

P005_ideal_native_parity_long / gram_ridge: 4000 LSMR iterations, algebraic residual 6.73182e-09, ordinary field relL2 6.27189e-11, raw PDE RMS 4.4201e-07, transformed condition 1.395e+05. Setup 0.10s, solve 3.06s. Finite iteration diagnostic; low objective alone is not a field certificate.

P005_ideal_native_parity_long / qr: 2175 LSMR iterations, algebraic residual 1.73184e-08, ordinary field relL2 4.64935e-10, raw PDE RMS 4.25983e-07, transformed condition 1419. Setup 0.15s, solve 1.66s. Finite iteration diagnostic; low objective alone is not a field certificate.

**P005_ideal_native_parity_long FINISH 2026-10-02 07:24:07 UTC — complete.** Three preconditioner variants retained; total 9.82s, process peak 429.9MiB. Dense matrix diagnostics deliberately exceed scalable-storage policy; factor storage alone is recorded separately.

**FINISH 2026-10-02 07:24:08 UTC — resolution_limit.** Returned p40; ordinary field relL2 1.23591e-12; raw PDE RMS 6.18535e-10; readout L1 11.92; 88.9s, 375.1MiB. All stage exits/inner counts retained. A budget exit is not a precision-floor certificate.

**FINISH 2026-10-02 07:24:19 UTC — resolution_limit.** Returned p32; ordinary field relL2 1.5532e-12; raw PDE RMS 3.29297e-09; readout L1 13.66; 100.3s, 348.4MiB. All stage exits/inner counts retained. A budget exit is not a precision-floor certificate.

### P001–P005 conclusion — 2026-10-02 07:25:40 UTC

Correctly matching native parity grouping reveals block condition5.14e10: QR improves the formal whitened condition1.39e5→1.42e3 and at800 iterations improves field error1.26e-7→4.66e-10. But longer P005 reverses the final field comparison: Gram4000 gives6.27e-11; QR stops2175 at4.65e-10, both raw PDE about4e-7. **QR is an early-convergence improvement, not a solved precision gap.** Applying ill-conditioned triangular inverses may limit numerical accuracy; that mechanism remains unisolated. Domain-compatible coordinates should be checked before production integration. Detailed evidence: `results/checkpoint_F_applications/expF19_radon_direct_pde/route2_gap_diagnosis/P005_ideal_native_parity_long/evidence.md`. No production edits.

### G019_box_native_p32_long_c1000 — START 2026-10-02 07:28:13 UTC

Cold native diffusion_c1000; coordinates=box; normalized PDE, stage cap 8 outer / 1000 Krylov, 240.0s soft total budget, degrees=(32,), ideal-stall refinement enabled. Shared policy; no dense reference state enters solve. Artifacts: `results/checkpoint_F_applications/expF19_radon_direct_pde/route2_gap_diagnosis/G019_box_native_p32_long_c1000`.

### P006_box_ideal_native_parity — START 2026-10-02 07:28:21 UTC

Fixed p32 diffusion_c1000, N257, lambda0.2, box chart, ideal columns, normalized PDE and unit column scaling. Same matrix, auto groups capped at 1024 coordinates: diagonal, regularized Gram, and guarded QR. LSMR cap 4000; no target values in solve. Dense diagnostic only. Artifacts: `results/checkpoint_F_applications/expF19_radon_direct_pde/route2_gap_diagnosis/P006_box_ideal_native_parity`.

P006_box_ideal_native_parity / diagonal: 4000 LSMR iterations, algebraic residual 1.00663e-07, ordinary field relL2 1.76879e-08, raw PDE RMS 2.32596e-06, transformed condition 4.82e+04. Setup 0.10s, solve 2.89s. Finite iteration diagnostic; low objective alone is not a field certificate.

P006_box_ideal_native_parity / gram_ridge: 716 LSMR iterations, algebraic residual 5.10198e-13, ordinary field relL2 3.30119e-14, raw PDE RMS 4.48416e-11, transformed condition 1419. Setup 0.10s, solve 0.52s. Finite iteration diagnostic; low objective alone is not a field certificate.

P006_box_ideal_native_parity / qr: 725 LSMR iterations, algebraic residual 5.08938e-13, ordinary field relL2 3.50995e-14, raw PDE RMS 3.88384e-11, transformed condition 1419. Setup 0.14s, solve 0.52s. Finite iteration diagnostic; low objective alone is not a field certificate.

**P006_box_ideal_native_parity FINISH 2026-10-02 07:28:27 UTC — complete.** Three preconditioner variants retained; total 6.01s, process peak 418.3MiB. Dense matrix diagnostics deliberately exceed scalable-storage policy; factor storage alone is recorded separately.

**FINISH 2026-10-02 07:28:39 UTC — resolution_limit.** Returned p32; ordinary field relL2 1.38781e-05; raw PDE RMS 0.0252664; readout L1 2.127e+08; 25.9s, 324.3MiB. All stage exits/inner counts retained. A budget exit is not a precision-floor certificate.

### P006 interpretation — 2026-10-02 07:29:06 UTC

Matched box-Legendre chart makes QR unnecessary in this control. Same ideal normalized PDE matrix/parity blocks: Gram converges in716 LSMR steps to ordinary field3.30e-14, raw PDE4.48e-11; QR takes725 steps, field3.51e-14, raw PDE3.88e-11. Both transformed conditions are1418.689, matching the ideal disk-QR formal condition but avoiding its field plateau~4.65e-10. Plain column scaling without blocks remains at1.77e-8 field after4000 steps. This supports domain-compatible coordinates plus existing bounded Gram blocks; no production QR integration warranted from this case. These are dense diagnostic products, not production timing. Native streamed validation remains separate.

### G020_box_native_p32_tight_c1000 — START 2026-10-02 07:29:31 UTC

Cold native diffusion_c1000; coordinates=box; normalized PDE, stage cap 4 outer / 4000 Krylov, 240.0s soft total budget, degrees=(32,), exact_inner=True, damping=1e-30, ideal-stall refinement enabled. Shared policy; no dense reference state enters solve. Artifacts: `results/checkpoint_F_applications/expF19_radon_direct_pde/route2_gap_diagnosis/G020_box_native_p32_tight_c1000`.

### P007_box_native_operator_agreement — START 2026-10-02 07:30:16 UTC

Parent-requested follow-up audit only: compare assembled ideal p32 normalized c1000 box columns against native ideal matvec and adjoint; no fitting or target values. Artifacts: `results/checkpoint_F_applications/expF19_radon_direct_pde/route2_gap_diagnosis/P007_box_native_operator_agreement`.

**P007 FINISH 2026-10-02 07:30:17 UTC — complete.** Random ideal native-vs-assembled matvec relative 3.24e-16; adjoint 2.34e-15; native inner-product identity 0; P006 solved-vector product discrepancy 4.35e-16. 0.88s. Operator agreement control only; no new solve.

**FINISH 2026-10-02 07:30:56 UTC — resolution_limit.** Returned p32; ordinary field relL2 4.68431e-15; raw PDE RMS 1.09497e-11; readout L1 3.755; 84.8s, 329.7MiB. All stage exits/inner counts retained. A budget exit is not a precision-floor certificate.

### G021_box_native_p32_tight_c10 — START 2026-10-02 07:31:35 UTC

Cold native diffusion_c10; coordinates=box; normalized PDE, stage cap 4 outer / 4000 Krylov, 240.0s soft total budget, degrees=(32,), exact_inner=True, damping=1e-30, ideal-stall refinement enabled. Shared policy; no dense reference state enters solve. Artifacts: `results/checkpoint_F_applications/expF19_radon_direct_pde/route2_gap_diagnosis/G021_box_native_p32_tight_c10`.

### G022_box_p36_c1000_refine — START 2026-10-02 07:32:25 UTC

Fixed-degree diffusion_c1000, p=36, N=257, lambda=0.2, ideal columns, normalized weighting, SVD cutoff=1e-14. Physics/BC only; dense reference diagnostic, no production solver change. Artifacts: `results/checkpoint_F_applications/expF19_radon_direct_pde/route2_gap_diagnosis/G022_box_p36_c1000_refine`.

### G023_box_p40_c1000_refine — START 2026-10-02 07:32:26 UTC

Fixed-degree diffusion_c1000, p=40, N=257, lambda=0.2, ideal columns, normalized weighting, SVD cutoff=1e-14. Physics/BC only; dense reference diagnostic, no production solver change. Artifacts: `results/checkpoint_F_applications/expF19_radon_direct_pde/route2_gap_diagnosis/G023_box_p40_c1000_refine`.

**FINISH 2026-10-02 07:32:32 UTC — complete.** Ordinary field relL2 1.12595e-15; ideal 1.04656e-15; raw ordinary PDE RMS 1.73514e-12; boundary RMS 1.26063e-15. Rank 703/703; condition 1.05e+05; readout L1 3.75. 7.1s; process peak 447.8 MiB. Interpretation pending paired controls; no automatic floor claim.

**FINISH 2026-10-02 07:32:36 UTC — complete.** Ordinary field relL2 1.08042e-14; ideal 9.16271e-15; raw ordinary PDE RMS 7.44238e-11; boundary RMS 5.40203e-15. Rank 861/861; condition 1.39e+05; readout L1 3.89. 9.5s; process peak 522.0 MiB. Interpretation pending paired controls; no automatic floor claim.

### G024_box_native_adaptive_tight_c1000 — START 2026-10-02 07:34:10 UTC

Cold native diffusion_c1000; coordinates=box; normalized PDE, stage cap 4 outer / 4000 Krylov, 420.0s soft total budget, degrees=(4, 8, 12, 16, 24, 32, 36), exact_inner=True, damping=1e-30, ideal-stall refinement enabled. Shared policy; no dense reference state enters solve. Artifacts: `results/checkpoint_F_applications/expF19_radon_direct_pde/route2_gap_diagnosis/G024_box_native_adaptive_tight_c1000`.

### G025_box_native_adaptive_tight_c10 — START 2026-10-02 07:34:11 UTC

Cold native diffusion_c10; coordinates=box; normalized PDE, stage cap 4 outer / 4000 Krylov, 420.0s soft total budget, degrees=(4, 8, 12, 16, 24, 32, 36), exact_inner=True, damping=1e-30, ideal-stall refinement enabled. Shared policy; no dense reference state enters solve. Artifacts: `results/checkpoint_F_applications/expF19_radon_direct_pde/route2_gap_diagnosis/G025_box_native_adaptive_tight_c10`.

**FINISH 2026-10-02 07:34:23 UTC — resolution_limit.** Returned p32; ordinary field relL2 2.23279e-15; raw PDE RMS 2.12328e-12; readout L1 3.756; 167.7s, 332.5MiB. All stage exits/inner counts retained. A budget exit is not a precision-floor certificate.

### Interpretation and next paired controls — 2026-10-02 07:35:05 UTC

G020 reproduces the diagnostic field floor using the actual streamed solver: ordinary field4.6843e-15, raw PDE1.09497e-11, boundary8.84e-15, 1381 Krylov iterations, 84.84s, 329.72MiB process RSS. No dense reference arrays or coefficients enter this solve. The network has 9603 tanh neurons and561 readout coordinates. The inner training residual passes1e-13, but independent scaled PDE4.11e-13 does not, so the driver correctly returns resolution_limit. This closes fixed-space coefficient access only for this controlled linear diffusion case, not the raw PDE floor or general PINNs.

Box coordinates plus tight undamped linear correction are a **combined recipe**, not a fully factorial attribution to damping alone. G019 shows the same box chart with the default loose inner schedule and only8 outer steps can still fail badly; its huge readoutL1 is retained. P006/P007 isolate good conditioning/accurate streamed ideal products in box coordinates.

G022/G023 show increasing the ideal degree is not automatically a numerical improvement: boxp36 with three actual residual refinements gives ordinary field1.13e-15/rawPDE1.74e-12, while p40 is still1.08e-14/raw7.44e-11 after three refinements. These are dense diagnostic controls, not scalable wins or certified floors.

G024/G025 now use the same tight recipe through the full native degree ladder4,8,12,16,24,32,36, cold zero, both contrasts, without importing any dense control state. This tests resolution selection and continuation instead of hand-returning the single p32 reference space. Settings are frozen identically for the pair before the outcomes. Future FINISH log entries include explicit run IDs; the generated run ledger links all earlier interleaved entries to result/protocol timestamps.

**FINISH 2026-10-02 07:38:34 UTC — sampled_residual_checks_passed.** Returned p36; ordinary field relL2 7.63115e-16; raw PDE RMS 7.48903e-13; readout L1 3.751; 263.8s, 353.2MiB. All stage exits/inner counts retained. A budget exit is not a precision-floor certificate.

### Saved-result assembly — 2026-10-02 07:41:52 UTC

Wrote `run_ledger.csv`, `run_ledger.json`, and `coefficient_access_progress.png/.pdf` from 44 saved result/variant rows. No fitting or new scientific evaluation. Timings are not isolated performance benchmarks. The ledger makes earlier interleaved START/FINISH entries unambiguous through their result paths.

**FINISH 2026-10-02 07:42:10 UTC — time_budget.** Returned p36; ordinary field relL2 9.73715e-16; raw PDE RMS 5.47934e-13; readout L1 3.751; 478.4s, 363.1MiB. All stage exits/inner counts retained. A budget exit is not a precision-floor certificate.

### Saved-result assembly — 2026-10-02 07:44:51 UTC

Wrote `run_ledger.csv`, `run_ledger.json`, and `coefficient_access_progress.png/.pdf` from 44 saved result/variant rows. No fitting or new scientific evaluation. Timings are not isolated performance benchmarks. The ledger makes earlier interleaved START/FINISH entries unambiguous through their result paths.

### Final ordinary-model artifact export — START 2026-10-02 07:45:36 UTC

Reconstructing the already audited native G024/G025 final model parameters from their saved coefficients. No fitting or additional accuracy claim; exported form is Linear/Tanh/Linear.

**Final ordinary-model artifact export FINISH 2026-10-02 07:45:37 UTC.** Both saved single-hidden-layer parameter archives and shape/checksum manifests are written. These are the same construction/export path used by the independent field audit.

### Batch consolidation — 2026-10-02 07:46:58 UTC

All32 named numerical invocations have results; the ledger contains44 result/variant rows and no pending entries. G024 is contrast1000: field7.63115e-16, rawPDE7.48903e-13, sampled checks passed,263.76s,353.20MiB. G025 is contrast10: field9.73715e-16, rawPDE5.47934e-13, time_budget; normalized heldoutPDE1.65534e-13 exceeds1e-13,478.39s,363.14MiB. Both are degree36/10767-neuron flat tanh models.

An independent read-only source/results review found no interior-target leakage or architecture mismatch. It corrected two presentation points: use matched ideal/column-scaled condition numbers (disk1.2326e11 versus box4.8197e4), and describe the recipe as sufficient in these tests, not mathematically necessary. These corrections are included in the evidence page. Production source hashes match the startup records. All newly added documentation links resolve. No general-solver guarantee or new dimension-scaling result is claimed.

Promoted evidence: `docs/radon_results_library/evidence/coefficient_access_2026_10_02.md`. Figure and complete run ledger: `results/checkpoint_F_applications/expF19_radon_direct_pde/route2_gap_diagnosis/`. Final ordinary weights and shape/checksum manifests are preserved in both adaptive run folders.

Next untested questions: transfer the stable-coordinate/accurate-inner recipe to a different PDE family and dimension; make nonlinear damping and resolution allocation automatic; separate derivative/encoding precision from solve precision. No numerical process from this batch remains running.

## Higher-dimensional and harder-equation transfer batch — START 2026-10-02 08:05:08 UTC

User authorized battle testing the improved recipe beyond paired diffusion. Coordinator root owns NS and consolidation. Workers own scalar3D/4D/5D and hard2D respectively, with separate files/output prefixes under `route2_transfer_2026_10_02/`. No production solver edit is planned. Cross-machine context sync again failed transport; local checks passed, so this uses local evidence only.

Shared invariants: one flat ordinary tanh MLP, box coordinates, N257/lambda.2/corrected sqrt(N)halo, tensor directions, native actual-residual/ideal-correction block preconditioning, exact inner schedule, maxinner4000, independent ordinary audits, cold zero then native continuation only. Linear class uses damping1e-30; nonlinear class uses standard1e-6 with existing line search, declared before outcomes. Maxouter8 accommodates nonlinear updates. Fixed tolerance1e-13;1GiB named-array and2million-neuron caps; single-thread childruns, at most three simultaneous solver processes. No symmetry reduction or sparse active directions.

Each group freezes equation cases, degree ladder, sampling, and softtime budget before outcomes. Scalar dimensional tests are smooth full-rank coupled semilinear equations, not hardrealworld5D claims. Hard2D includes Helmholtz and Allen-Cahn, with external numerical reference used only for postfit AC verification. Root NS includes a smooth3D control, moreoscillatory low-viscosity manufactured3D/space-time examples, and a driven3D cavity withzero bodyforce/no known interiortruth. Failures, encoding loss, and budget/resource rejection remain outcomes. Numerical solve budgets are soft; an in-progress inner solve can overrun them.

### D001_semilinear_3d — START 2026-10-02 08:06:03 UTC

Frozen 3D full-rank coupled semilinear control. B=[[0.0207015115293407, -0.002080734182735712, -0.004949962483002227], [-0.002080734182735712, 0.02273178189568194, 0.00480085143325183], [-0.004949962483002227, 0.00480085143325183, 0.02944434869057662]]; box coordinates, tensor angular rule, N257/lambda .2/halo17, p=(4, 6, 8, 10, 12); zero native start; fixed normalization; 8 outer/4000 inner, exact inner, damping1e-6; 600s soft budget; 1024MiB named arrays, 2million neurons. Forecast rejected degrees []. Interior truth confined to post-fit audit. Artifacts: `results/checkpoint_F_applications/expF19_radon_direct_pde/route2_transfer_2026_10_02/D001_semilinear_3d`.

### H001_helmholtz_n2_startup_failure — START / FINISH recorded 2026-10-02T08:07:12.242574+00:00

Child invocation aborted before imports and before numerical work, with OMP Error #179 (cannot open shared memory in sandbox). No equation solve or tuning occurred. Same protocol will be retried with local execution access; this failed invocation is preserved.

### H002_helmholtz_n4_startup_failure — START / FINISH recorded 2026-10-02T08:07:12.242574+00:00

Child invocation aborted before imports and before numerical work, with OMP Error #179 (cannot open shared memory in sandbox). No equation solve or tuning occurred. Same protocol will be retried with local execution access; this failed invocation is preserved.

### H003_allen_cahn_d001_startup_failure — START / FINISH recorded 2026-10-02T08:07:12.242574+00:00

Child invocation aborted before imports and before numerical work, with OMP Error #179 (cannot open shared memory in sandbox). No equation solve or tuning occurred. Same protocol will be retried with local execution access; this failed invocation is preserved.

### H004_allen_cahn_d00001_startup_failure — START / FINISH recorded 2026-10-02T08:07:12.242574+00:00

Child invocation aborted before imports and before numerical work, with OMP Error #179 (cannot open shared memory in sandbox). No equation solve or tuning occurred. Same protocol will be retried with local execution access; this failed invocation is preserved.

### H001_helmholtz_n2 — START 2026-10-02 08:07:19 UTC

Harder 2D transfer `helmholtz_n2`. Frozen before outcomes: box coordinates, N257, lambda .2, degree ladder (4, 8, 12, 16, 24, 32, 36, 40), exact inner tolerance, ideal/block-parity native correction, 8 outer / 4000 inner iterations per stage, 480s soft total budget, 1GiB named arrays, 2 million neurons. Generic PDE normalization. Damping 1e-30 from linear/nonlinear class policy. Cold zero; no interior truth or external solve enters fit. Artifacts: `results/checkpoint_F_applications/expF19_radon_direct_pde/route2_transfer_2026_10_02/H001_helmholtz_n2`.

### N001_ns3_smooth — START 2026-10-02 08:07:52 UTC

NS transfer, {'family': 'trigonometric', 'frequency': 0.7, 'viscosity': 0.2, 'transient': False}, inputs=3. Frozen shared nonlinear policy: box/N257/lambda.2, exact inner, maxinner4000,maxouter8,damping1e-6,tol1e-13,600s soft budget. No interiortruth or reference coefficients in solve. Case/protocol/preflight: `results/checkpoint_F_applications/expF19_radon_direct_pde/route2_transfer_2026_10_02/N001_ns3_smooth`.

**D001_semilinear_3d FINISH 2026-10-02 08:08:08 UTC — resolution_limit.** 3D, returned p12, 169 directions, 49179 ordinary tanh neurons. Ordinary field relL2 4.86487e-15; raw PDE RMS 3.88762e-14; boundary max 8.70415e-14; corner max 2.19824e-13; readout L1 0.46601. 124.7s including audit; peak RSS 317.9MiB. Full stage memory/work and rejected levels saved; budget/resolution exits are not precision certificates.

### D002_semilinear_4d — START 2026-10-02 08:08:14 UTC

Frozen 4D full-rank coupled semilinear control. B=[[0.0207015115293407, -0.002080734182735712, -0.004949962483002227, -0.0032682181043180597], [-0.002080734182735712, 0.020065115229015273, 0.00480085143325183, -0.0007275001690430677], [-0.004949962483002227, 0.00480085143325183, 0.024111015357243283, 0.004219269793662461], [-0.0032682181043180597, -0.0007275001690430677, 0.004219269793662461, 0.02921170259838308]]; box coordinates, tensor angular rule, N257/lambda .2/halo17, p=(4, 6, 8, 10, 12); zero native start; fixed normalization; 8 outer/4000 inner, exact inner, damping1e-6; 600s soft budget; 1024MiB named arrays, 2million neurons. Forecast rejected degrees []. Interior truth confined to post-fit audit. Artifacts: `results/checkpoint_F_applications/expF19_radon_direct_pde/route2_transfer_2026_10_02/D002_semilinear_4d`.

**Harder2D startup qualification — 2026-10-02T08:08:39.715092+00:00**. Earlier aborts occurred during restricted execution with an OpenMP shared-memory error. The unchanged script runs after approved local-access retry, but the root cause was not isolated; a sandbox cause is a hypothesis, not a proven diagnosis. No numerical setting changed between these attempts.

**N001_ns3_smooth FINISH 2026-10-02 08:10:45 UTC — sampled_residual_checks_passed.** Degree20, neurons128331; ordinary velocity relL2=1.3757824661478147e-14, pressure=1.1842974606011681e-14; raw momentum RMS=1.11248e-15, divergence=7.43035e-17; 172.1s, 522.8MiB. Unknown cavity field error remains unknown. Full validation/failure gates retained.

### N002_ns3_bubble — START 2026-10-02 08:11:32 UTC

NS transfer, {'family': 'bubble', 'frequency': 1.5, 'viscosity': 0.03, 'transient': False}, inputs=3. Frozen shared nonlinear policy: box/N257/lambda.2, exact inner, maxinner4000,maxouter8,damping1e-6,tol1e-13,600s soft budget. No interiortruth or reference coefficients in solve. Case/protocol/preflight: `results/checkpoint_F_applications/expF19_radon_direct_pde/route2_transfer_2026_10_02/N002_ns3_bubble`.

**H001_helmholtz_n2 FINISH 2026-10-02 08:13:16 UTC — resolution_limit.** Returned degree 40, 11931 neurons / 861 coordinates. Ordinary field relative L2 7.45427e-12; raw PDE RMS 1.8849e-09; 356.5s solve / 356.6s including audit; 370.0MiB peak process RSS. Reference only after frozen fit; Allen-Cahn reference uncertainty and endpoint compatibility caveat retained. All stage exits preserved.

### H002_helmholtz_n4 — START 2026-10-02 08:13:17 UTC

Harder 2D transfer `helmholtz_n4`. Frozen before outcomes: box coordinates, N257, lambda .2, degree ladder (4, 8, 12, 16, 24, 32, 36, 40), exact inner tolerance, ideal/block-parity native correction, 8 outer / 4000 inner iterations per stage, 480s soft total budget, 1GiB named arrays, 2 million neurons. Generic PDE normalization. Damping 1e-30 from linear/nonlinear class policy. Cold zero; no interior truth or external solve enters fit. Artifacts: `results/checkpoint_F_applications/expF19_radon_direct_pde/route2_transfer_2026_10_02/H002_helmholtz_n4`.

### N001_ns3_smooth_saved_audit — START 2026-10-02 08:14:59 UTC

Read-only saved ordinary-model corner/near-edge PDE checks and ideal-coordinate comparison. No refit, new initialization, or target data supplied to a solver. Separate audit file, original metrics unchanged.

**N001_ns3_smooth_saved_audit FINISH 2026-10-02 08:15:01 UTC.** Near-edge momentum RMS/max=2.21869e-15/7.20604e-15; ordinary-vs-ideal field max=4.77396e-15; 1.8s. Original fit unchanged.

**H002_helmholtz_n4 FINISH 2026-10-02 08:16:18 UTC — resolution_limit.** Returned degree 36, 10767 neurons / 703 coordinates. Ordinary field relative L2 0.000864968; raw PDE RMS 0.566152; 180.9s solve / 181.0s including audit; 366.0MiB peak process RSS. Reference only after frozen fit; Allen-Cahn reference uncertainty and endpoint compatibility caveat retained. All stage exits preserved.

### H003_allen_cahn_d001 — START 2026-10-02 08:16:19 UTC

Harder 2D transfer `allen_cahn_d001`. Frozen before outcomes: box coordinates, N257, lambda .2, degree ladder (4, 8, 12, 16, 24, 32, 36, 40), exact inner tolerance, ideal/block-parity native correction, 8 outer / 4000 inner iterations per stage, 480s soft total budget, 1GiB named arrays, 2 million neurons. Generic PDE normalization. Damping 1e-06 from linear/nonlinear class policy. Cold zero; no interior truth or external solve enters fit. Artifacts: `results/checkpoint_F_applications/expF19_radon_direct_pde/route2_transfer_2026_10_02/H003_allen_cahn_d001`.

**D002_semilinear_4d FINISH 2026-10-02 08:19:00 UTC — time_budget.** 4D, returned p10, 1331 directions, 387321 ordinary tanh neurons. Ordinary field relL2 1.11271e-11; raw PDE RMS 3.48007e-11; boundary max 2.08556e-10; corner max 1.13384e-09; readout L1 0.85223. 645.9s including audit; peak RSS 340.8MiB. Full stage memory/work and rejected levels saved; budget/resolution exits are not precision certificates.

### D003_semilinear_5d — START 2026-10-02 08:19:08 UTC

Frozen 5D full-rank coupled semilinear control. B=[[0.0207015115293407, -0.002080734182735712, -0.004949962483002227, -0.0032682181043180597, 0.0014183109273161313], [-0.002080734182735712, 0.018731781895681938, 0.00480085143325183, -0.0007275001690430677, -0.004195357645382263], [-0.004949962483002227, 0.00480085143325183, 0.02144434869057662, 0.004219269793662461, -0.0037984395642941065], [-0.0032682181043180597, -0.0007275001690430677, 0.004219269793662461, 0.025211702598383078, 0.0020404103090669603], [0.0014183109273161313, -0.004195357645382263, -0.0037984395642941065, 0.0020404103090669603, 0.03895601405931737]]; box coordinates, tensor angular rule, N257/lambda .2/halo17, p=(4, 6, 8, 10, 12); zero native start; fixed normalization; 8 outer/4000 inner, exact inner, damping1e-6; 600s soft budget; 1024MiB named arrays, 2million neurons. Forecast rejected degrees [10, 12]. Interior truth confined to post-fit audit. Artifacts: `results/checkpoint_F_applications/expF19_radon_direct_pde/route2_transfer_2026_10_02/D003_semilinear_5d`.

**H003_allen_cahn_d001 FINISH 2026-10-02 08:21:18 UTC — resolution_limit.** Returned degree 36, 10767 neurons / 703 coordinates. Ordinary field relative L2 0.532759; raw PDE RMS 0.0713086; 298.6s solve / 298.7s including audit; 366.9MiB peak process RSS. Reference only after frozen fit; Allen-Cahn reference uncertainty and endpoint compatibility caveat retained. All stage exits preserved.

### H004_allen_cahn_d00001 — START 2026-10-02 08:21:19 UTC

Harder 2D transfer `allen_cahn_d00001`. Frozen before outcomes: box coordinates, N257, lambda .2, degree ladder (4, 8, 12, 16, 24, 32, 36, 40), exact inner tolerance, ideal/block-parity native correction, 8 outer / 4000 inner iterations per stage, 480s soft total budget, 1GiB named arrays, 2 million neurons. Generic PDE normalization. Damping 1e-06 from linear/nonlinear class policy. Cold zero; no interior truth or external solve enters fit. Artifacts: `results/checkpoint_F_applications/expF19_radon_direct_pde/route2_transfer_2026_10_02/H004_allen_cahn_d00001`.

**N002_ns3_bubble FINISH 2026-10-02 08:22:27 UTC — time_budget.** Degree20, neurons128331; ordinary velocity relL2=9.208450479115385e-10, pressure=2.451722181971846e-10; raw momentum RMS=8.72857e-11, divergence=1.45179e-11; 654.6s, 572.0MiB. Unknown cavity field error remains unknown. Full validation/failure gates retained.

### N002_ns3_bubble_saved_audit — START 2026-10-02 08:23:02 UTC

Read-only saved ordinary-model corner/near-edge PDE checks and ideal-coordinate comparison. No refit, new initialization, or target data supplied to a solver. Separate audit file, original metrics unchanged.

### N003_ns4_transient — START 2026-10-02 08:23:03 UTC

NS transfer, {'family': 'trigonometric', 'frequency': 1.5, 'viscosity': 0.05, 'transient': True}, inputs=4. Frozen shared nonlinear policy: box/N257/lambda.2, exact inner, maxinner4000,maxouter8,damping1e-6,tol1e-13,600s soft budget. No interiortruth or reference coefficients in solve. Case/protocol/preflight: `results/checkpoint_F_applications/expF19_radon_direct_pde/route2_transfer_2026_10_02/N003_ns4_transient`.

**N002_ns3_bubble_saved_audit FINISH 2026-10-02 08:23:04 UTC.** Near-edge momentum RMS/max=1.50083e-08/1.09466e-07; ordinary-vs-ideal field max=1.00732e-13; 2.2s. Original fit unchanged.
