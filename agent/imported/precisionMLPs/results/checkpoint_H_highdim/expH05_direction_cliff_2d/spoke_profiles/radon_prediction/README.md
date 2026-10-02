# Independent Radon predictions for the solved spoke coefficients

September 29, 2026. The interior coefficients closely follow the independently
constructed radial ridge profiles, after undoing finite tanh smoothing. This is
strong empirical agreement for these targets, not recovery guaranteed by the
higher-dimensional approximation theory. The asymmetric composition target
does not recover the profiles from the direction-snapping construction.

The [forward-evaluation follow-up](forward_check/README.md) is the stronger test:
it supplies the theoretical profiles to actual tanh blocks, isolates angular
and finite-band errors, and projects the construction into the solver's retained
coefficient space. It also constructs a more accurate angular interpolation for
composition on the same directions; this reaches the numerical floor without
a joint 2D fit. The coefficient discrepancies below should be read with that
follow-up, rather than treated as failures of function approximation.

## Which higher-dimensional theory

The relevant source is [`docs/ridge_quadrature_theory.md`](../../../../../docs/ridge_quadrature_theory.md),
the August 31 checkpoint H operating theory. It describes the shallow model used
in the 3D/4D experiments: Fourier-polar ridge representation, angular
discretization, a 1D tanh approximation on each direction, and a joint readout
solve. Its Step 4 explicitly distinguishes existence of an accurate set of
profiles from the profiles selected by least squares.

The September 3 local review at `/Users/sam/Downloads/review_compositional_ridge_QI.md`,
Section 2, proposes a sharper angular-quadrature certificate using spherical
harmonics and spherical designs. It distinguishes smooth angular spectra from
Fourier mass concentrated on lines. `/Users/sam/Downloads/depth_theory_response.md`,
Section 3.2, retains both constructions: angular quadrature for the smooth
background and direction snapping for line atoms. These sections concern the
shallow higher-dimensional lift; the separate KAN/composition discussion is not
needed for this comparison. The proposed spherical-design certificate does not
automatically apply to the Fibonacci directions used by the 3D implementation.

## Prediction and meaning of the derivative

For centered coordinates u = x - x0, write the network as

\[
\widehat f(x_0+u)=b+\sum_m g_m(v_m\cdot u),\qquad
g_m(t)=\sum_j a_{mj}\tanh[\gamma(t-c_{mj})].
\]

Let q_m be an independently constructed theoretical ridge contribution, including
its angular quadrature weight. The leading coefficient prediction is

\[
a_{mj}^{(0)}=\frac h2 q'_m(c_{mj}).
\]

For suitable whole-space targets, q_m is a weighted **filtered** Radon projection,
not a spatial slice f(x0 + t v_m). In dimension d the filter has Fourier multiplier
|omega|^(d-1). In 3D it is minus the second derivative of the plane-integral Radon
projection, so the leading tanh coefficient density involves minus its third
derivative, with the normalization and angular weight included. See
[Radon inversion, equations 9.A.2–9.A.5](https://www.math.utoronto.ca/courses/apm346h1/20181/PDE-textbook/Chapter9/S9.A.html).

The spatial directional derivative receives contributions from every spoke:

\[
D_v\widehat f(x_0+u)=\sum_m(v\cdot v_m)g'_m(v_m\cdot u).
\]

Finite tanh width smooths a coefficient density with
K_gamma(t) = gamma sech²(gamma t)/2. Its Fourier multiplier is
A omega / sinh(A omega), A = pi/(2 gamma). Undoing this continuum smoothing gives

\[
a_{mj}^{\rm corr}=\frac h2
\mathcal F^{-1}\!\left[
\frac{\sinh(A\omega)}{A\omega}\widehat{q'_m}(\omega)
\right](c_{mj}).
\]

For real analytic q_m on the needed complex strip, the equivalent expression is
h Im[q_m(c_mj + i A)]/(2 A). This is a continuum bandwidth correction. It does
not include every effect of the discrete center grid, finite collar, or truncated
SVD, and it is not asserted to be the exact cardinal-QI readout formula.

For radial targets, the normalized theoretical profile G satisfies
mean_theta G(r cos theta) = F(r). The script uses its Abel inverse; for Gaussian,
Runge and the decaying spatial packet this is the filtered-Radon profile.
The undamped concentric cosine has no ordinary convergent Radon line integral;
its comparison uses the analytic radial ridge identity instead. The periodic
composition target uses its Bessel Fourier series and nearest-line snapping.
There is no fitted scale, offset, profile parameter, or regression in a prediction.

## Measurements

The historical run saved error summaries but no coefficients. These comparisons
use the reproduced solutions saved by `spoke_profiles.py`, with the historical
half-step directions. M = 32; N = 64 or 128; h = 1/N; gamma h = 0.25. The training
disk has radius 0.4 around x0 = (0.35, -0.25). The table pools all 32 spokes over
|c_mj| < 0.3. Discrepancy means ||a - a_pred||_2 / ||a||_2, not output error.

| Target | 2048: corrected discrepancy | 4096: leading derivative | 4096: corrected discrepancy | 4096: corrected correlation |
|---|---:|---:|---:|---:|
| Gaussian bump | 5.63% | 3.68% | 3.76% | 0.999295 |
| Radial Runge | 0.429% | 2.67% | 0.0808% | 0.9999997 |
| Concentric waves | 3.71% | 8.51% | 0.735% | 0.999973 |
| Spatial packet | 1.18% | 25.4% | 0.239% | 0.9999972 |
| Asymmetric composition, snapped Fourier prediction | 263% | 275% | 278% | 0.308338 |

The 4096-neuron output errors are 1.3e-14 to 1.3e-13 on the saved 20,000-point
interior test set. The corrected radial coefficient discrepancy decreases from
2048 to 4096 in every case, but these two widths do not establish convergence.
For the Gaussian, the bandwidth correction alone does not explain the remaining
discrepancy. Its cause has not been isolated.

The outer collar differs substantially: pooling **all** centers at width 4096
gives corrected discrepancies of 58.4%, 11.7%, 39.2%, and 31.8% for Gaussian,
Runge, waves and packet, respectively. The interior claim must not be extended
to the collar.

For the historical directions, the composition's snapped Fourier ridge sum
already has 1.52% relative output error on an independent 400-point check,
before discretizing its 1D profiles. The joint network fit reaches 1.34e-14.
Thus the solve is using different profiles from this explicit construction.
This is compatible with the existence argument in Step 4; it is evidence
against interpreting that argument as a coefficient-recovery theorem.

The axis-aligned fits used by the interactive view give the same qualitative
conclusion. Their width-4096 corrected discrepancies are 3.74%, 0.0728%, 0.662%,
0.203%, and 278%. Their exact values differ because the directions were rotated
by half an angular step. No 3D coefficient comparison was performed here.

## Files and checks

- `recorded/comparison.json`: full measurements for the historical geometry.
- `comparison.json`: the corresponding axis-aligned geometry.
- Each directory contains `predictions_N64.npz`, `predictions_N128.npz`, and
  `independent_radon_coefficients.png` / `.pdf`.
- Predictions reconstruct the four radial targets with 256 directions to
  maximum absolute error below 2.5e-15 on an independent check. The composition's
  unsnapped Fourier series agrees with its analytic target to 1.4e-15.
- Increasing the radial quadrature from 64 to 128 nodes changes the complex
  profiles/derivatives by less than 8e-13. The tanh-kernel inverse multiplier
  check is accurate to 1.2e-16.
- The interactive view displays the independent coefficient predictions in its
  “Independent theory prediction” mode. Its embedded prediction arrays use
  float32 for display; the stored comparisons and this table use float64.

Reproduce from the repository root:

```sh
MPLCONFIGDIR=/tmp/precision-spokes-mpl .venv/bin/python experiments/expH05_direction_cliff_2d/spoke_radon_prediction.py --angle-rule recorded
MPLCONFIGDIR=/tmp/precision-spokes-mpl .venv/bin/python experiments/expH05_direction_cliff_2d/spoke_radon_prediction.py --angle-rule endpoint
```
