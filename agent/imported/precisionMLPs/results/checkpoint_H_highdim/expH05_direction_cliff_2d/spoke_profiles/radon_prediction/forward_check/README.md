# Forward evaluation of the theoretical spoke construction

September 29, 2026. This follows up the coefficient comparison with actual
network evaluation. The large raw coefficient discrepancy is mostly in the
numerical nullspace of the joint dictionary. Separately, composition exposes an
accuracy limitation of nearest-direction snapping. A higher-order angular
interpolation construction reaches the numerical floor on the same geometry.

These are axis-aligned, 32-direction reproductions, matching the interactive
view. All output errors below are relative L2 on the original independent
20,000-point test set: radius 0.36 around (0.35, -0.25), seed 1. The training
disk has radius 0.4. These measurements concern this local approximation task.

## What was actually supplied to the network

The target-derived profiles q_m are fixed before any scalar conversion. For the
four radial targets they are the profiles used in the prior comparison; for
composition the initial construction groups exact Fourier atoms onto their
nearest available direction. Four stages are evaluated separately:

1. Sum the prescribed continuous profiles directly.
2. Insert the earlier derivative-based coefficient predictions into the finite
   tanh network. Set the single bias by exact matching at the disk center.
3. Accurately represent each prescribed q_m with its existing 1D tanh block:
   independently solve q_m(t) on 8N+1 uniform scalar samples in [-0.4, 0.4].
   This uses the theoretical profiles as targets, not the 2D function values.
   The direction allocation and profile shapes are fixed by the construction.
4. Compare with the saved joint 2D fit on the same directions, centers and widths.

| Target, width 4096 | Continuous theoretical ridge sum | Theoretical profiles converted to tanh | Saved joint 2D fit |
|---|---:|---:|---:|
| Gaussian | 1.89e-16 | 4.00e-15 | 4.21e-14 |
| Runge | 1.81e-16 | 2.45e-15 | 1.03e-14 |
| Concentric waves | 1.36e-15 | 1.37e-14 | 1.23e-13 |
| Spatial packet | 3.54e-16 | 1.16e-14 | 3.80e-14 |
| Composition, nearest-direction snapping | 6.98e-4 | 6.98e-4 | 3.84e-14 |

For composition the 1D conversion itself differs from its prescribed continuous
ridge sum by only 3.41e-14. The 6.98e-4 error is already in the angular construction;
it is not a failure of the tanh blocks or a missing coefficient normalization.

The earlier continuum-corrected coefficients, copied directly into the original
finite band, produce errors 2.24e-7, 7.03e-8, 3.79e-6, and 2.02e-7 for Gaussian,
Runge, waves, and packet. Thus an interior coefficient match does not imply
machine-precision reconstruction after truncating the coefficient tails.
Keeping the same spacing and gamma but adding 80 centers at each end of every
spoke reduces those errors to 6.22e-15, 3.32e-15, 5.38e-14, and 1.91e-15.
That halo check uses **9216 neurons**, not 4096. Independent scalar conversion
achieves the first table's errors with the original **4096 neurons**, by adjusting
the finite-band readouts while preserving the prescribed profiles on the data band.

The leading h q'_m/2 prediction without the tanh-width correction is less accurate
still; every stage and both widths (2048, 4096) are recorded in `comparison.json`.
Neither the continuum bandwidth correction nor the scalar least-squares
conversion is being mislabeled as the exact finite cardinal-stencil algorithm.

## Where the coefficient discrepancy goes

Let A be the actual training feature matrix with its bias column. If the solver
retains right singular vectors V_tau, define P_tau = V_tau V_tau^T. This is a
projection in parameter space, determined by the geometry, sample locations,
and numerical cutoff. It uses no target values. The fitted vector and the
constructed vector obey

\[
a_{\rm fit}=A_\tau^+ y,\qquad
a_{\rm fit}-P_\tau a_T=A_\tau^+(y-Aa_T).
\]

Consequently the direct difference contains two distinct pieces:

\[
a_T-a_{\rm fit}=(I-P_\tau)a_T+A_\tau^+(Aa_T-y).
\]

The first is discarded by the numerical solver. The second reflects the
construction's function error, amplified by the retained pseudoinverse. Even
near-floor function errors need not produce identical coefficients in a
poorly conditioned dictionary.

For the 4096-neuron snapped composition construction, the matrix has retained
rank 3123 out of 4097 columns. **99.9913% of the squared constructed-versus-solved
readout-and-bias difference lies in its discarded singular subspace.**
Projecting the constructed readout changes the output by 2.51e-14 relative L2
on the independent test set. Its interior coefficient discrepancy drops from
277.85% to **2.63%**, without a fitted amplitude or any change to the prescribed
function. The remaining function error is still 6.98e-4.

The corresponding projected interior coefficient discrepancies for Gaussian,
Runge, waves and packet are 0.0994%, 0.0340%, 0.0550%, and 0.0213%. The stored
observed solutions themselves lie in the retained subspace to approximately
6e-15 relative. Thus the projection test directly measures the effect discussed
in Step 4 of the checkpoint H theory; it does not merely invoke nonuniqueness
as a possible explanation.

## Why snapping loses accuracy, and a sharper construction

A composition Fourier term at frequency (3 pi, pi) points at 18.43495 degrees.
Nearest-line snapping sends it to 16.875 degrees, a 1.55995-degree error. The
coefficient magnitude of each such complex Fourier atom is about 0.008228.
Other similarly important off-grid atoms are also moved. The resulting phase
errors account for a real approximation loss before any tanh discretization.
The snapped sum's absolute phase bound at radius 0.36 is 0.007284.

However, snapping is only one possible angular approximation. For a fixed
frequency magnitude rho and fixed u in the disk, the plane wave
exp(i rho v(theta) dot u) is smooth as a function of its direction theta,
including when the Fourier measure itself is discrete. Approximate this
angular dependence by trigonometric interpolation on the 64 oriented nodes
theta_j = j pi/32. The even-node cardinal function is

\[
L_j(\theta)=\frac{\sin[M(\theta-\theta_j)]}
 {2M\tan[(\theta-\theta_j)/2]},\qquad M=32,
\]

with its limiting value at a node. Replace each wave by

\[
e^{i\rho v(\theta)\cdot u}\approx
\sum_{j=0}^{2M-1}L_j(\theta)e^{i\rho v(\theta_j)\cdot u}.
\]

Weight these terms by the target's analytic Fourier coefficients and group
antipodal nodes. This produces **32 prescribed real 1D profiles** on exactly the
same available directions. The angular weights may be signed; they are known
cardinal interpolation weights, not learned amplitudes. This construction is
an additional test derived from the Fourier-polar representation. It is **not**
the nearest-line construction stated in Step 2 of the operating notes, and it
does not require interpreting an atomic Fourier measure as a smooth density.

| Composition construction | Relative output error |
|---|---:|
| Continuous angular interpolation | 3.37e-16 |
| Its 2048-neuron tanh representation | 4.10e-15 |
| Its 4096-neuron tanh representation | 3.41e-14 |

The 4096 conversion uses only independent 1D solves against the prescribed
profiles. There is no joint fit to the 2D target. The difference between the
2048 and 4096 results is at the floating-point/conditioning floor and is not a
claim that more neurons reduce approximation capacity.

Projecting this sharper 4096-neuron construction gives output error 3.02e-14;
the projection changes its output by 2.68e-14. Its interior coefficients still
differ from the saved learned ones by **2.55%**, and its full readout-and-bias
vector differs by 6.19%. Thus the remaining few-percent coefficient difference
cannot be attributed solely to the old snapping error: it persists when both
functions are already at the numerical floor. The retained-space formula above
explains why weak singular directions can amplify near-floor function differences.
For this construction too, 99.9913% of the original squared parameter difference
is in the discarded singular subspace.

This establishes a sharper conclusion than the earlier explanation: the
periodic target's atomic spectrum does not force the 6.98e-4 error. That error
belongs to the nearest-direction approximation. Likewise, a different raw
coefficient array does not by itself imply a different function. The relevant
comparisons are forward error and, when predicting the selected readout,
the construction expressed in the solver's retained coefficient space.

## Reproduction

`theory_forward_check.py --project` produces `comparison.json`, independent
scalar readouts, and their retained-space projection. The projection applies
the actual training SVD and its recorded cutoff 1e-13.

`composition_angular_interpolation.py --project` produces
`angular_interpolation.json`, both interpolation readouts, and a projection
check for the 4096-neuron case. The figure
`composition_construction_errors.png` / `.pdf` displays the forward comparison.
Both scripts live in `experiments/expH05_direction_cliff_2d/` and use the already
saved solutions and analytic target formulas. Numerical artifacts include the
unrounded errors and independent output checks.
