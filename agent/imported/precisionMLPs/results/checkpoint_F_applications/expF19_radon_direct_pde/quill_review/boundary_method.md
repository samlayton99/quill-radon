# Current QUILL boundary correction: implementation and checks

The prescribed rational correction works in FP64. Merely cutting off the complex-shift lattice after `ceil(sqrt(N))` halo neurons did **not** implement the newer construction. The correction changes the readouts of existing outer halo neurons; it adds no neurons and performs no fit.

Source: `docs/appendix_notes/paper_v5_replacements/organized_v7_package/sections/7_appendix/01_Construction/09_26_sl.tex`, equations `density`, `mu`, `numinus`, `nuplus`, `scaled-pf`. Code: `experiments/expF19_radon_direct_pde/quill_boundary.py`. This is the current finite-contour construction, not a heuristic fit of omitted tails.

## Precise geometry and coefficient recipe

Here N means interval **cells**. On `[L,U]`, h=(U−L)/N, there are N+1 interior centers, R=ceil(sqrt(N)) halo neurons per side, and W=N+1+2R total neurons. The center sequence is `L+j*h`, j=−R,…,N+R. Gamma=lambda/h. The API allows explicit halo overrides when the caller uses an interior-point convention instead.

This scalar diagnostic sweep deliberately reproduces the paper's cell-count convention. It does **not** use Sam's interior-point-count convention in every row: 128 cells means 129 interior points and both conventions give R=12, but 256 cells means 257 interior points, for which this diagnostic uses R=16 while Sam's rule gives R=17. The main `quill_sweep.py` integration and updated PDE exports pass the explicit interior-point halo count, so their geometry follows Sam's convention exactly. Do not describe every scalar diagnostic row as using that convention.

Let d=pi/(2 gamma), let a and b be the outermost half-cell endpoints, and let x0=L. For a prescribed analytic profile f, the core density and weights are

\[
 \rho(z)=\frac{f(z+id)-f(z-id)}{2id},\qquad
 w_j^{\rm base}=\frac h2\rho(c_j).
\]

The baseline network is anchored at f(L). Its truncation error includes two boundary functions, whose power-series coefficients are g=mu−nu. For each integer ell=1,…,m with m=floor((R+1)/2), define, at endpoint e=a with sign s=+1 or e=b with sign s=−1,

\[
 \mu_\ell=\frac1d\int_0^d
 \operatorname{Re}\{f(e+iy)e^{s2i\ell\gamma y}\}\,dy,
\]
\[
 \nu_\ell=2(-1)^\ell\int_0^\sigma
 \frac{\operatorname{Im}\{\rho(e+iy)e^{s2i\ell\gamma y}\}}
 {1+e^{2\pi y/h}}\,dy.
\]

These real integrals exploit conjugate symmetry. The implementation subtracts f(e) before calculating mu, since its exact integral is zero. This avoids injecting quadrature error from the constant part.

For the current paper's contour, sigma=2kd with k=ceil(delta/(8d)). The implementation verifies d≤delta/8 and physical halo≤delta. An entire profile can use delta=max(8d,physical halo). A profile with complex singularities must instead supply a genuinely valid analytic tube radius; the code cannot infer this radius from a callable.

Put zeta=exp(−2 lambda), r_i=zeta^(i−1/2), and p_k=product_{j=1}^k(1−zeta^j). Let e_k denote the elementary symmetric polynomial. The stable partial-fraction formula is

\[
 c_i=(-1)^i\frac{\zeta^{[i(i+1)-1]/2}}{p_{i-1}p_{m-i}}
 \sum_{\ell=1}^{m}
 e_{\ell-1}(r_1,\ldots,\widehat r_i,\ldots,r_m)g_\ell.
\]

We add c_i/2 to the i-th outermost left readout and subtract c_i/2 from the i-th outermost right readout, then recalculate the bias by anchoring at L. The prescribed rational denominator has poles corresponding exactly to these existing tanh centers. The algorithm is a fixed algebraic map from contour moments to readouts; no linear-system solve, least squares, or SVD appears.

The exact-arithmetic paper bound for this corrected boundary contribution is proportional to h exp(−lambda R²/4), under its stated analytic and geometric assumptions. This is a bound on the boundary component, not a bound on the entire FP64 network error: lattice replicas, contour remainder, coefficient calculation, and arithmetic remain.

## Measurements

`boundary_metrics.json` contains 360 rows: 3 targets × 5 widths × 8 bandwidths × 3 methods. Methods are plain sqrt(N) halo, corrected sqrt(N) halo, and a long plain halo with at least ceil(24/lambda) neurons per side. Predictions are evaluated on 2,201 real points, including 200 additional points approaching the interval endpoints. Values and first derivatives are measured. This is verification using known targets, not data fitting.

At lambda=.25:

| Target / cells | Plain sqrt(N), max value error | Corrected sqrt(N), max value error |
|---|---:|---:|
| sin(2 pi x), N=128 | 1.14e−4 | 4.44e−16 |
| sin(2 pi x), N=256 | 3.15e−5 | 5.55e−16 |

For sin(12 pi x), corrected relative L2 is 2.19e−14 at N=128 and 2.11e−15 at N=256. The shifted Gaussian exp(−3(x−.15)²) reaches relative L2 1.10e−16 at N=128. First derivative errors at N=128 are respectively 1.16e−15, 1.13e−13, and 8.90e−16.

The bandwidth sweep has an interior optimum. For sin(2 pi x), N=128, corrected relative L2 errors are:

| lambda | .10 | .15 | .20 | .25 | .30 | .40 |
|---|---:|---:|---:|---:|---:|---:|
| Error | 5.04e−8 | 2.56e−11 | 7.85e−15 | 2.27e−16 | 9.99e−15 | 2.50e−11 |

At smaller width N=32 the best tested lambda for this sine is .40, not .25. Small lambda makes the finite boundary correction and coefficient magnitudes harder; large lambda permits larger lattice replicas. No universal optimal bandwidth is claimed.

Independent implementation check: the prior 60-decimal mpmath construction at N=512, lambda=.25 agrees with our weights within 2.50e−16 and correction coefficients within 5.00e−16. Three simultaneous vector outputs agree with their targets within 1.67e−15. Quadrupling the contour quadrature order changes predictions by at most 4.44e−16. A constant profile has exactly zero readout weights. Details and reproducible assertions are in `boundary_validation.json` and `validate_reference()`.

The complete scalar sweep took 8.4 seconds on one CPU thread, excluding plotting and the independent precision check. This does not benchmark a large directional PDE export; other experiment workers measure that separately.
