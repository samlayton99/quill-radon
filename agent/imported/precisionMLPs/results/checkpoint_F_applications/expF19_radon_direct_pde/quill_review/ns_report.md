# Current QUILL halo correction on saved nonlinear PDE profiles

This is an **encoding-only** follow-up. The saved 48³ Fourier Navier–Stokes state is unchanged; no PDE evolution, optimization, least-squares solve, or target-data fit was repeated. The reference is that same saved finite Fourier state. We tested the current-paper explicit boundary correction from `quill_boundary.py`, not merely the older infinite-line complex-shift density with a shorter halo.

The convention is Sam's: N counts interior centers, h=2L/(N−1), R=ceil(sqrt(N)) centers per side, and total neurons M(N+2R). The boundary encoder therefore receives n_cells=N−1 and explicit halo=R. NS uses the periodic cube [-π,π]³, shared projected core [-π√3,π√3], and 2361 primitive Fourier directions unless pruning is explicitly stated. The seven shared output heads are velocity3, its PDE-supplied time derivative3, and pressure. No claim is made that this is a fixed four-input space-time MLP.

## Controlled matched comparison

All NS errors below use **identical** 128 fresh Sobol interior points, 8 corners, and 48 face points, and the same full Fourier state in every error denominator. Derivatives are derivatives of the actual constructed tanh network.

| Geometry/readout choice | Neurons | Velocity relative error | Velocity Laplacian relative error | Pressure relative error |
|---|---:|---:|---:|---:|
| Earlier λ=.25, physical halo3, 425 centers/direction | 1,003,425 | 1.65e-15 | 8.38e-14 | 7.38e-14 |
| Current correction, N128,R12,λ=.25 | 358,872 | 2.24e-15 | 6.53e-14 | 9.60e-15 |
| Current correction, N192,R14,λ=.20 | 519,420 | 2.62e-15 | 1.61e-15 | 2.68e-15 |
| Current correction, M1365,N128,R12,λ=.20, mode pruning1e-12 | 207,480 | 3.20e-11 | 2.47e-9 | 7.82e-11 |

Thus the tested compact full-spectrum construction uses64.2% fewer neurons while retaining value precision. The derivative-focused construction uses48.2% fewer neurons and improves Laplacian error by about52×. The balanced construction uses79.3% fewer neurons, intentionally accepts a spectral-pruning error comparable to the measured PDE-grid velocity error, and does not retain machine-precision derivatives. These are best choices among the measured settings, not global optima.

The correction is essential: simply shortening the halo to sqrt(N) without changing readouts leaves approximately1e-4–1e-3 errors in the full NS state. At N192,λ=.20, using exact direction-specific projected intervals L_v=π||v||₁ instead of the common worst-case interval gave2.38e-15 velocity and1.65e-15 Laplacian errors, comparable to the common interval; it did not earn an additional neuron saving because N was held fixed.

The broad scalar screening grid was λ∈{.15,.20,.25,.30,.40}, N∈{64,96,128,192,256}, tested on frequency1 and8 sine waves plus the finite Fourier Burgers profile. The full NS check used9 targeted settings at λ=.20/.25. For the same finite Burgers Fourier profile, the old309-neuron long-halo encoding has8.63e-16 value error; the corrected N128,R12,λ=.25 construction has8.48e-16 with152 neurons. Both compare with the same Fourier profile; neither number should be confused with total error against the Cole–Hopf continuum solution.

## The two-floor allocation prediction was actually tested

We measured center errors with all2361 directions and measured direction errors directly by pruning the Fourier spectrum before any neural encoding. The latter is a directional spectral-support test, not uniform angular quadrature. Independently obtained curves predicted crossed settings using max(direction error, center error), under the same full-state norm denominator. The 3×3 held-out combinations used:

- M393,813,1365, obtained by coefficient cutoffs1e-8,1e-10,1e-12.
- N96,128,192 at λ=.20 with the required sqrt(N) halos.

Prediction error was at most0.80% for velocity and10.32% for the Laplacian across these nine cells. This empirical success does not turn the maximum rule into a general exact identity; the exact general rule is the sum of error vectors, with a triangle-inequality bound.

At M1365,N128 the measured velocity error was3.1984e-11; the independent floor prediction was3.1969e-11. Increasing N to192 costs300,300 instead of207,480 neurons and changes velocity error only to3.1969e-11, because directional truncation already limits it. Reducing N to96 uses158,340 neurons but degrades velocity error to5.5052e-10. This is a directly verified useful balance in this experiment.

The compact model's physical momentum residual RMS on the current endpoint-inclusive points is4.46e-10, versus3.46e-10 for full-spectrum models. Its divergence is1.30e-15. Gradient relative error3.28e-10 and Laplacian relative error2.47e-9 make the deliberate derivative tradeoff explicit.

The old64³ comparison state was not saved, so we did **not** rerun the fluid solver. On the identical128 old validation points, the saved full48³ neural model had measured error3.6554e-11 relative to64³. The new balanced model differs from that saved full48³ neural model by3.0833e-11 relative to its norm. The triangle inequality therefore yields an **upper bound6.7387e-11** relative to64³. This is a bound, not a newly measured end-to-end error.

## Numerical controls and artifacts

- Errors include cube corners and faces, avoiding an interior-only assessment of boundary corrections.
- Velocity/readout vectors preserve the per-direction orthogonality responsible for structural zero divergence; roundoff-size defects persist after the linear correction.
- Periodicity, maximum face/corner error, weight L1/max norms, pressure, gradient, Laplacian, and residual differences are in `ns_quill_metrics.json`.
- Increasing contour quadrature4× on12 strongest NS profiles changes outputs by at most1.39e-16 and second derivatives by1.11e-16; maximum weight change6.94e-16.
- Full NS grid took about17s; there was no new fluid time integration.

Chat figure: `ns_quill_headline.png`. It shows matched cost/error comparisons and the crossed M,N predictions. Supplement: `ns_quill_comparison.png`. Scalar screening: `ns_scalar_sweep.json`; full comparisons: `ns_quill_metrics.json`; old matched baseline: `ns_baseline_comparison.json`; allocation check: `ns_allocation_check.json`; quadrature check: `ns_quadrature_check.json`. Explicit networks: `ns_quill_compact_snapshot.npz`, `ns_quill_snapshot.npz` (derivative-focused), and `ns_quill_balanced_snapshot.npz`.

Reproduce using one OpenMP/BLAS thread:

```
.venv/bin/python experiments/expF19_radon_direct_pde/quill_ns_sweep.py --stage all
```

This requires the pre-existing saved NS states in `../nonlinear/`. The entry point runs scalar screening, saved-state export, matched baselines, allocation checks, and figures without evolving the fluid.
