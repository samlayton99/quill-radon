# Why the composition spoke comparison differs

September 29, 2026. This checks the axis-aligned 4096-neuron solution currently
shown in the interactive comparison. The difference cannot be repaired by one
normalization factor. Some individual spoke shapes are close after rescaling,
but the explicit Fourier construction concentrates its coefficients on a few
directions and the solved network distributes them broadly.

The subsequent [forward test](../forward_check/README.md) makes this explanation
quantitative: over 99.99% of the squared parameter discrepancy is discarded by
the training SVD, and a sharper angular construction reaches the output floor
on the same 32 directions. That follow-up supersedes any interpretation that
the atomic spectrum itself forces the approximation gap.

The target has the exact factorization

\[
e^{\sin(\pi x)\cos(\pi y)}
=e^{\frac12\sin(\pi(x+y))}
 e^{\frac12\sin(\pi(x-y))}.
\]

Its global Fourier spectrum consists of discrete frequencies. The leading ones
point along the two diagonals; higher terms also occupy other directions.
The plotted reference groups these Fourier frequencies onto their nearest
available spoke. This is a different construction from sampling a smooth
filtered-Radon angular density, as used for Gaussian, Runge and packet.

## Scaling checks

For a coefficient array a and reference p, the diagnostic scale minimizes
||a - alpha p||_2. It is alpha = (a dot p)/(p dot p). This fitted scale is used
only to test the normalization hypothesis; the interactive prediction remains
unscaled and independent of the solution.

| Region | Best global multiplier | Discrepancy before | Discrepancy after |
|---|---:|---:|---:|
| Interior, −0.3 < t < 0.3 | 0.120738 | 277.85% | 93.32% |
| All centers, −0.5 < t < 0.5 | 0.121659 | 187.13% | 97.52% |

Within the interior, individually rescaling the 0-degree reference by 0.20345
leaves 6.57% discrepancy; at 45 degrees the best multiplier is 0.10165 and the
remaining discrepancy is 24.83%; at 135 degrees it is 0.11929 and 13.22%.
At 22.5 degrees the reference is essentially zero while the network has a
substantial profile. Thus the visual impression of similar shapes on selected
spokes is real, but there is no common missing multiplier.

The two diagonal spokes carry 91.67% of the reference's squared coefficient
norm but only 10.41% of the observed interior coefficient norm. These are shares
of coefficient norm, not shares of the represented function's energy: different
spokes are correlated and can cancel.

## Normalization and reconstruction checks

The unsnapped Fourier series reconstructs the analytic target with maximum
absolute error 1.78e-15 on 500 independent points in [-2,2]^2. The product
factorization agrees with the analytic target to 1.12e-15 on those points.
This checks the frequency phases, amplitudes, and Fourier normalization without
using any solved readout.

After snapping the frequencies to 32 available directions, the continuous
ridge sum has relative output error 6.97e-4 on 2,000 independent points in the
radius-0.36 disk. Its best output scale, allowing a free bias, is 1.0000394
(bias -5.43e-5), rather than the 0.120738 required by the coefficient comparison.
The network itself has saved output error 3.86e-14 on its 20,000-point test set.

## Interpretation

The joint fit observes the function only on the training disk. Its truncated
SVD selects a minimum-norm readout in the retained singular subspace; it does
not enforce the global Fourier allocation across directions. Locally, different
ridge collections can approximate the same function. For example, on evenly
spaced directions in 2D,

\[
b\cdot u=\sum_{m=1}^{M}\frac{2}{M}(b\cdot v_m)(v_m\cdot u),
\]

so a linear contribution can be spread across every spoke instead of carried
on one direction. A finite tanh dictionary approximates such identities on the
disk. Higher-degree polynomial contributions also admit redundant ridge
representations when enough directions are present. These freedoms explain why
coefficient recovery requires more than accurate reconstruction. They are
consistent with the measured redistribution, but this diagnostic does not
derive the exact profiles selected by the SVD from first principles.

This is precisely the distinction in
[Step 4 of the checkpoint H theory](../../../../../../docs/ridge_quadrature_theory.md):
the construction supplies a candidate approximant, while the joint projection
may choose different profiles. The experiment does not show the 1D derivative
relation breaking. It shows that the candidate Fourier profiles are different
from the profiles selected by this local solve. The close radial comparisons
remain empirical observations, not a universal recovery result.

`diagnostic.json` contains all per-direction checks. The PNG/PDF
`scaling_vs_direction_allocation` compares the direction distributions and the
45-degree profile after its best amplitude adjustment. The diagnostic is
reproducible with `experiments/expH05_direction_cliff_2d/composition_spoke_diagnostic.py`.
