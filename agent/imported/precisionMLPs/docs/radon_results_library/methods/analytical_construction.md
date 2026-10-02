# Analytical QUILL–Radon construction

**Use when the function, or an appropriate directional representation of it, is available.** Historical IDs A10–A20 describe successive versions; A20 is the corrected finite-halo construction. See [per-problem rankings](../rankings.md) and the [original construction audit](../../radon_catalogue_construction_audit.md).

## What is constructed

For input x in R^d, let theta_m be a unit direction, omega_m its angular quadrature weight, and g_m a scalar ridge profile. The supplied representation is approximated by

\[
f(x)\approx\sum_{m=1}^M\omega_m g_m(\theta_m^T x).
\]

Each known g_m is then encoded as a scalar tanh network with centers c_k and widths gamma_k. Combining the networks produces

\[
\widehat f(x)=b+\sum_{m=1}^M\sum_{k=1}^{H}
v_{mk}\tanh\{\gamma_k(\theta_m^T x-c_k)\}.
\]

This is one hidden layer on the original inputs. H=N+2R counts N interior centers and R halo centers per side. In the corrected experiments R=ceil(sqrt(N)). Angular weights are absorbed into v_mk.

The useful fact is that **the v_mk can be constructed from known profiles without a global readout fit**. The finite-contour version evaluates the analytical density, computes endpoint correction moments, redistributes the correction through existing halo readouts using explicit partial fractions, and reanchors the output bias. No global SVD or least-squares discovery of the readouts is required.

This does not mean arbitrary real samples automatically supply analytically continued ridge profiles. Obtaining those profiles, performing their quadrature, and satisfying the required analytic-domain assumptions have their own costs. For arbitrary sampled functions, consult the sample-only class in the rankings.

## What survived the experiments

| Version | Present use | Evidence |
|---|---|---|
| A20 corrected finite contour, existing sqrt(N) halo | Preferred analytical constructor | 3D anisotropic Gaussian: 244,800 neurons; held-out relative value error 6.40e-16, gradient 1.49e-15, Laplacian 7.87e-15 |
| A14 complex-shift construction with a long plain halo | Independent successful older construction; useful reference | Three 3D Gaussian-family examples reached roughly 4.24e-15 to 8.78e-15 with sufficient angular resolution |
| A11 angular interpolation with shared scalar profile LS | Hybrid for profiles supplied through Fourier/radial information | Repairs nearest-direction snapping; scalar profile fitting remains explicit, not a no-fit claim |
| A12 derivative samples alone | Leading-order intuition and control | Smoothing bias prevents the requested floor in tested settings |
| A13/A19 short or naive sqrt(N) halo | Failed shortcut | Same 3D example with naive halo: value error 3.96e-5 rather than 6.40e-16 |

The original high-dimensional result deserves its own scope statement. A18 constructed supplied four-ridge Fourier targets through ambient dimension 256 with 804 neurons and about 5.89e-16 error. This is a substantial construction result **for known sparse directional content**. The generic angular construction still pays for resolution over the sphere. A15's dense 4D/5D experiments and A16's approximate Sobol rule do not establish arbitrary unstructured high-dimensional precision at fixed cost.

## Resolution and cost

Two distinct resolutions matter: M directions and N centers within each direction. Refining one while the other dominates gives a plateau. The useful allocation principle is to balance angular and scalar errors; the historical measured rates and conditional spectral model are recorded in the [full catalogue](../../radon_method_catalogue.md).

The corrected halo fixes a major boundary error, but lambda is not a universal constant for every derivative order and degree. For example, larger N at fixed lambda need not monotonically improve second-derivative accuracy. The contour boundary estimate is a bound for that error component, not the whole PDE residual.

Storage can be close to the represented neurons plus bounded contour/profile work. The **implemented** contour setup and partial fractions have extra arithmetic; current code is not proven strictly linear in neuron count. Supplied sparse ridges and a shared profile bank are particularly favorable. Full formulas, including hidden preprocessing, are in [requirements and costs](../requirements_and_costs.md) and the historical A20 row.

## Source of truth

- [Boundary method and derivation](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/quill_review/boundary_method.md).
- [Boundary metrics](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/quill_review/boundary_metrics.json), [validation](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/quill_review/boundary_validation.json), [selected allocation results](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/quill_review/allocation_optimized_selected.json).
- [Boundary comparison figure](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/quill_review/boundary_comparison.png), [allocation figure](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/quill_review/allocation_summary.png).
- [Implementation](../../../experiments/expF19_radon_direct_pde/quill_boundary.py), [allocation study](../../../experiments/expF19_radon_direct_pde/quill_sweep.py).

No new construction or sweep was run for this library.
