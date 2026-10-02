# Longer native QUILL Navier–Stokes evolution

The constructed solver now reaches T=1 on the periodic cube [-pi,pi]^3 for the interacting Taylor–Green initial condition, at viscosities .05 and .01. Both selected solutions pass independent initial-data, intermediate physical-residual, divergence, energy, time-step, and spatial-refinement checks. These are sampled numerical checks, not continuum error certificates.

## Method and architecture

`solver/ns_extended.py` supplies `PeriodicNS` and `solve_ns_refined`. Initial data use `initial(points)->(N,3)`; optional forcing uses `forcing(t,points)->(N,3)`. Viscosity is positive. The domain is the prescribed periodic cube, and smooth data are assumed. There is no arbitrary-geometry or universal PINN backend here.

Each axis uses a QUILL-encoded bank containing cos(mx), sin(mx), m=1,...,K, plus the exact constant. The velocity is a tensor readout of products of three bank outputs. Actual encoded values and analytic first/second derivatives enter every nonlinear RHS. Real-data Fourier analysis, Leray projection, and RK4 evolve the compressed coefficients. No classical solution is supplied to the forward solve, and no readout least-squares solve is performed.

This explicitly changes the previous flat ridge architecture: it has shared one-dimensional banks and product gates. N=257 interior centers gives 867 tanh activations across three axes, while the compressed vector state still has 3(2K+1)^3 coefficients: 36,501 at K=11 and 59,049 at K=13. Fourier closure and global pressure projection remain essential. The method is a constructed neural reparameterization of a spectral method.

## Resolution and accuracy

The fixed-resolution runs used K=3,5,7, grids Q=3K+3, dt=.005, T=1. Every native run was completed before its conventional comparison trajectories were constructed. All three fixed cutoffs were underresolved at relative physical residual tolerance .001.

| Viscosity | K=3 error vs K=13 | K=5 error vs K=13 | K=7 error vs K=13 |
|---|---:|---:|---:|
| .05 | .0160145 | .00137594 | .000154737 |
| .01 | .0268979 | .00352025 | .000633557 |

The refinement wrapper cold-runs increasing cutoffs from the original IC. It requires relative physical residual <=.001 and successive-field change <=.0001, in addition to the time and conservation checks. It selected:

| Viscosity | Final K | Maximum sampled physical residual | Successive-field change | Independent finer comparison | dt vs dt/2 difference |
|---|---:|---:|---:|---:|---:|
| .05 | 11 | 7.529e-5 | 1.969e-5 | 2.380e-6 vs K=13 | 1.922e-12 |
| .01 | 13 | 1.854e-4 | 3.130e-5 | 7.225e-6 vs K=15 | 4.084e-12 |

The physical residual is evaluated on independent shifted grids, with the pressure eliminated by the full diagnostic-grid Leray projector, without truncating that projector to the evolution cutoff. Its normalization is the largest RMS convection, viscous term, or forcing. Accepted solutions were re-integrated with the final guards at 21 times, including endpoints. Their final coefficient arrays exactly match the earlier runs, so the separately recorded actual dt/2 comparisons remain applicable. `ns_guarded_validation.json` records this provenance explicitly.

Classical reference refinement at dt=.0025 gave K=11 to K=13 differences 2.380e-6 and 2.910e-5 for viscosities .05 and .01. For .01, the subsequent K=13 to K=15 change was 7.225e-6. Thus the finer comparisons are numerical references with visible spatial uncertainty, not exact continuum errors. The reference K=11 dt=.0025 versus .00125 differences were 1.20e-13 and 2.55e-13. No behavior beyond T=1 was tested.

Mean kinetic energy decreased from .125 to .0918722842 (.05) and .1174809339 (.01). The selected final maximum speeds were .8342 and .9342; maximum vorticities were 1.4561 and 1.5515. Generated vertical velocities in the K=7 runs had RMS .06085 and .08061, confirming substantial nonlinear interaction. Divergence and instantaneous energy-balance defects stayed near floating-point roundoff; full per-time maxima are in the guarded validation file.

## Cost and the necessary classical control

All numerical jobs used one CPU thread and stayed below 800 MB. The initial fixed-cutoff/reference harness took 231 seconds with peak RSS 345 MB; the largest later reference/refinement process recorded 382 MB. Most total study time was spent producing fresh higher-resolution comparison trajectories.

The selected native construction cost was about 4–5 ms. Before the final intermediate-time guards were added, measured setup plus forward times were about 4.91 s at K=11 and 8.65 s at K=13, versus 9.96 s and 18.60 s for this FFT implementation. Time-refinement checks and reference solves are excluded from those forward times and reported separately. Cold adaptive reruns including their dt/2 checks cost 27.8 s and 52.4 s. The final guarded rerun costs are recorded separately in `ns_guarded_validation.json`; the earlier matched timing observations were preserved rather than relabeled.

The decisive control uses exact trigonometric banks with the same tensor contractions. It took 4.35 s and 6.89 s, with final fields differing from native QUILL by 1.35e-15 and 1.47e-15. Native/ordinary tensor forward-cost ratios were 1.13 and 1.26. Off-grid evaluation of 256 points took 35.5/56.0 ms for QUILL versus 34.5/54.5 ms for exact trigonometric products. The advantage over this FFT implementation comes from the arithmetic layout at these small resolutions; it is not evidence of an intrinsic neural speedup.

## Guard failures and positive controls

`native_ns_extended.py --check-only` is the reproducible guard and coordinate-check entry point. Together with the main harness, it verifies the following:

- An aliased sin(9y) initial field at K=3 has an on-grid representation discrepancy of only 1.58e-15 but independent initial error 1.414. The solver rejects it before stepping.
- A divergent sin(x)e_x IC is rejected rather than silently accepting its Leray projection. Its retained-mode projection change is 1.0.
- Increasing K=3 to 5 to 7 recovers a sin(5y)e_x IC after the first attempt reports `initial_underresolved`.
- The transient force t(1-t)sin(30x)e_z vanishes at the final time, but intermediate physical residuals expose it. K=3/5 refinement returns `resolution_exhausted`.
- The force sin(10*pi*t)^2 sin(30x)e_z vanishes at every regularly recorded time in the regression. Its regular-time residual is 5.62e-16, but 16 independent stratified jittered-time forcing probes give mismatch 1.0, so it is rejected. Probe times are saved.
- A pure gradient force is accepted: full-grid Leray projection removes the pressure contribution rather than requiring the raw force to fit the divergence-free state.
- A known smooth manufactured forced flow gives relative error 3.44e-12. Excessive dt is rejected by the stability guard. A spatially unresolved steady force is rejected even when the numerical state scarcely changes.
- Actual neural values, gradients and Laplacians agree with an independent trigonometric calculation for random solenoidal coefficients to 2.11e-15, 4.44e-15 and 4.09e-14. The RHS also agrees with the earlier independent coordinate implementation to 1.08e-15.

Intermediate and jittered checks remain finite samples. They can miss sufficiently narrow or adversarially placed events. No global stability, identifiability, or all-time residual bound is claimed.

## Reproduction and artifacts

Run from the repository root with the working project interpreter:

```sh
.venv/bin/python experiments/expF19_radon_direct_pde/native_ns_extended.py
.venv/bin/python experiments/expF19_radon_direct_pde/native_ns_extended.py --refine-only
.venv/bin/python experiments/expF19_radon_direct_pde/native_ns_extended.py --product-control-only
.venv/bin/python experiments/expF19_radon_direct_pde/native_ns_extended.py --guard-recheck-only
.venv/bin/python experiments/expF19_radon_direct_pde/native_ns_extended.py --check-only
.venv/bin/python experiments/expF19_radon_direct_pde/native_ns_extended.py --plot-slices-only
```

The JSON files contain full numerical data and measured cost scopes. NPZ files contain compressed coefficients and explicit encoder parameters. `ns_adaptive_convergence.png` shows the stopping criteria; `ns_flow_slices.png` evaluates the actual constructed field at z=pi/4. The earlier fixed-cutoff and energy plots are retained.

## Blowup limitation

For exact fixed-K solenoidal Fourier Galerkin dynamics with positive viscosity and bounded forcing, the energy identity bounds the finite-dimensional state over every finite time interval. All fixed-K norms are equivalent, so a fixed-cutoff simulation cannot establish continuum finite-time blowup. The QUILL representation slightly perturbs exact Galerkin cancellation, which is why its energy and divergence defects are explicitly monitored. Large unresolved peaks must trigger refinement or resource exhaustion, not a claim that blowup was demonstrated.

The September 2026 forced-blowup construction was not instantiated or reproduced. Its complete forcing callback has not been supplied; these tests use Taylor–Green dynamics and explicitly identified manufactured controls only.
