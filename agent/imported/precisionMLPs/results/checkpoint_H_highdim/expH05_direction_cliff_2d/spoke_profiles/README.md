# Spoke functions in the 2D floor solutions

Diagnostic requested September 29, 2026. The six-panel gallery is in
[`six_panel_views/spoke_gallery.pdf`](six_panel_views/spoke_gallery.pdf), with separate
PNG files for the Gaussian bump, fast concentric waves, and asymmetric composition.

Each gallery figure has the requested layout: all spoke functions drawn as radial
3D wires, the actual target surface, then individual 2D profiles in the x, y,
45-degree, and 135-degree directions. Solid curves are the reconstructed spoke
functions; dashed curves are spatial slices of the target. All six panels use the
same vertical scale within a figure.

## What is plotted

Write the fitted network around the data center x0 = (0.35, -0.25) as

\[
\widehat f(x_0+u)=b+\sum_{m=1}^{M}g_m(v_m\cdot u),\qquad
g_m(t)=\sum_{j=1}^{N}a_{mj}\tanh\bigl(\gamma(t-t_{mj})\bigr).
\]

Here v_m is a unit direction, t_mj is a center measured relative to v_m dot x0,
and a_mj is its solved readout coefficient. The gallery plots

\[
G_m(t)=M[g_m(t)-g_m(0)]+\widehat f(x_0).
\]

This removes the 1/M amplitude scaling and gives every spoke the same value at
the center. It preserves the exact reconstruction identity

\[
\widehat f(x_0+u)=\frac1M\sum_m G_m(v_m\cdot u).
\]

The left 3D panel places the curve at (t v_m,x, t v_m,y, G_m(t)). This is a display
of the spoke parameters. Its height is not the fitted 2D surface evaluated at
that location. The right panel is the analytic target on the radius-0.36 disk,
the same interior domain used for scoring. Axis coordinates are offsets from x0.

## Reproduction and accuracy

The original experiment saved error summaries, not coefficients. Three configurations
were reproduced using the original samples, tanh features, and truncated SVD:
16 directions × 128 centers, 32 × 64, and 32 × 128. Their solved coefficients are
now saved as NPZ files alongside this note.

The shared direction generator changed in commit b8ffdf0 (September 1): it now
includes the axes, whereas the recorded August results use a half-step angular
offset. `spoke_profiles.py` explicitly reproduces the historical convention without
changing the shared implementation. Its four split-study targets reproduce the
recorded errors to floating-point rounding; the 4096-column solve has recorded
retained rank 3122.

The gallery uses an additional fit with the current endpoint directions so the
four displayed snapshots lie exactly on the requested axes and diagonals. Those
solutions are in `endpoint_direction_check/`; they use 32 directions × 128 centers.

| Gallery target | Relative L2 error on 20,000 scored points |
|---|---:|
| Gaussian bump | 4.22e-14 |
| Fast concentric waves | 1.23e-13 |
| Asymmetric composition | 3.86e-14 |

These are precise fits on the specified disk, not a claim of global precision.
The normalized-profile reconstruction agrees with the original network to
1.11e-15 maximum absolute difference on 100 independent points. Grouped/flat
network evaluation, analytic/finite-difference spoke derivatives, and the analytic
Gaussian Radon reconstruction were also checked. Full numbers are in `summary.json`
and `six_panel_views/metadata.json`.

## What the coefficients form

Exactly,

\[
g'_m(t)=\gamma\sum_j a_{mj}\operatorname{sech}^2(\gamma(t-t_{mj})).
\]

For smooth coefficient profiles on a uniform center grid with spacing h,
the narrow-kernel approximation gives 2 a_mj / h approximately equal to
g'_m(t_mj), just as in the 1D experiment. This approximation is not the same as
the exact expression above: at finite gamma, differentiation smooths the
coefficient profile with K_gamma(t) = gamma sech²(gamma t)/2.

In the historical 4096-neuron fits, pooling all spokes over |t| < 0.3:

| Target | Correlation of 2a/h with g' | Relative norm discrepancy |
|---|---:|---:|
| Gaussian bump | 0.999987 | 1.04% |
| Radial Runge | 0.999899 | 2.73% |
| Fast waves | 0.999816 | 9.25% |
| Composition | 0.999943 | 1.77% |
| Spatial packet | 0.998742 | 33.88% |

The shapes agree closely, but the high-frequency packet needs a substantial
finite-bandwidth correction. The coefficient collar also has boundary structure;
it is shown in `raw_coefficients_and_widths.png`, rather than hidden in these metrics.

This table compares coefficients with the derivative of the **same solved
component**. It does not independently validate Radon theory. The separate
[Radon prediction comparison](radon_prediction/README.md) constructs profiles from
the target alone and measures their agreement with the solved coefficients.

## Relation to Radon inversion

For a sufficiently regular, decaying target on the whole plane, define its Radon
projection Rf(theta,s) as the integral over the line with normal v_theta and
offset s. With Fourier convention exp(-i omega s), filtered backprojection gives

\[
q_\theta(s)=\frac{1}{2\pi}\Lambda Rf(\theta,s),\quad
\widehat{\Lambda h}(\omega)=|\omega|\widehat h(\omega),\quad
f(x)=\int_0^\pi q_\theta(v_\theta\cdot x)\,d\theta.
\]

Thus the canonical 1D functions are filtered projections. Their derivatives,
with angular and center-spacing factors included, are the continuum counterparts
of the tanh coefficients. This follows from the standard
[filtered-backprojection formula, page 4](https://people.ucsc.edu/~fmonard/UWSummer2011/NotesRayTransforms.pdf).

For a Gaussian exp(-||x-a||²/sigma²), the canonical normalized profile is
pi q_theta(s) = 1 - 2 z D(z), where z = (s-v_theta dot a)/sigma and D is Dawson's
integral. The formula was checked by angular reconstruction to 1.11e-16 absolute
error. After centering the profiles, the learned Gaussian spokes differ from
this canonical expression by 3.64% in relative norm over |t| <= 0.36;
`gaussian_radon_comparison.png` shows the comparison.

A fit restricted to a disk does not uniquely specify a whole-plane extension or
canonical Radon decomposition. Low-degree contributions can also cancel across
directions. For example, replacing g_m(t) by g_m(t)+alpha_m t leaves the total
unchanged whenever sum_m alpha_m v_m = 0. This is an identity for unrestricted
ridge profiles; the finite tanh dictionary can approximate such redistributions
on the fitted disk. Consequently, reaching the numerical floor does not by
itself establish that each fitted spoke equals the canonical Radon profile.

## Regeneration

From the repository root, using the existing Python environment:

```sh
OPENBLAS_NUM_THREADS=6 VECLIB_MAXIMUM_THREADS=6 MPLCONFIGDIR=/tmp/precision-spokes-mpl .venv/bin/python experiments/expH05_direction_cliff_2d/spoke_profiles.py
OPENBLAS_NUM_THREADS=6 VECLIB_MAXIMUM_THREADS=6 MPLCONFIGDIR=/tmp/precision-spokes-mpl .venv/bin/python experiments/expH05_direction_cliff_2d/spoke_profiles.py --angle-rule endpoint
MPLCONFIGDIR=/tmp/precision-spokes-mpl .venv/bin/python experiments/expH05_direction_cliff_2d/spoke_gallery.py
```

Existing NPZ solutions are reused. Add `--plot-only` to regenerate the first
script's plots without solving any missing configurations.
