# Pure MLP repair: two implemented routes

Coordinator: root. October 1, 2026. Sam explicitly resumed work with two
allowed routes after identifying the architectural change in the previous
general PDE solver. The product-feature baseline remains preserved and is
not evidence for an ordinary MLP.

Acceptance requirements:

- The evaluated/exported forward model uses affine layers and stated
  elementwise activations. No external product gates or polynomial evaluation.
- Ordinary tanh/GELU MLPs are the main target. Squared-ReLU is at most a
  separately labeled result unless Sam explicitly accepts that activation.
- Every PDE residual uses the actual neural function and its derivatives.
- Initial coefficients are zero or prescribed-condition-only. No solution
  from an exact-polynomial or other separate PDE solver may initialize a run.
- Analytically constructing fixed basis/readout coordinate maps is allowed;
  the remaining numerical coefficient solve must be disclosed.
- Check value and first/second derivative accuracy independently, including
  an ordinary exported PyTorch MLP. Do not hide stabilized arithmetic that is
  absent from that forward implementation.

Ownership (shared working tree; preserve other workers):

- native_quill_theory: route 1 depth/compiler, `solver/pure_mlp.py`,
  `pure_mlp_depth_study.py`, and dedicated tests/results. Replace products with
  explicitly constructed square subnetworks and affine polarization wiring.
- quill_allocation_history: route 2 flat Radon/ridge theory and prototype,
  `ridge_pinn_*` files, dedicated tests/results. Construct a fixed analytic
  readout map and solve physics directly through the actual flat tanh MLP.
- root: higher-dimensional ridge-frame analysis, integration, architecture
  audits, source verification, and in-chat synthesis.

Initial scope is a controlled small nonlinear two-input PDE and rigorous
architecture/derivative tests. No new large Navier–Stokes launch until a route
passes these gates. Relevant primary literature is being checked. Context
sync failed transport; local context check passed and local guidance is used.

## Completed October 1, 2026

Both architecture gates passed. These are new cold-start PDE experiments;
the old product-feature Navier–Stokes results are not reused as their evidence.

1. Depth route: `solver/pure_mlp.py` exports only Linear/Tanh operations.
   QUILL square profiles approximate multiplication through polarization.
   With N=257 interior centers, R=17 per side and lambda=.2, the small
   nonlinear elliptic experiment has ordinary exported-network relative L2
   error 6.85e-16 and raw Torch-AD PDE RMS 3.56e-15. It starts with zero
   readout, solves 15 coordinates, and has 9,312 tanh units. Actual dense
   weights occupy 3.367 MB; the separate PDE/audit process peaked at 758 MB.
   Earlier cell-count halo convention outputs are explicitly archived.

2. Flat ridge route: `solver/ridge_pinn_features.py` uses a known angular
   reproducing identity and explicit QUILL profile encodings. The actual
   model is Linear(2,5529)/Tanh/Linear(5529,1). For the nonlinear disk PDE,
   190 solved readout coordinates and five GN steps from zero give ordinary
   relative field error 4.05e-16 and raw Torch-AD PDE RMS 5.26e-15. There is
   no polynomial PDE warmstart. Independent polynomial controls are solely
   comparisons, never initializers. Saved Sequential state dictionaries and
   figures are in the F19 `ridge_pinn/` result folder.

3. Root's multidimensional extension: `solver/ball_ridge_features.py`
   constructs box-based readout coordinates through algebraic ball moments
   and Gegenbauer ridge profiles. The evaluated/exported model contains no
   products or polynomials. A cold four-input/four-output manufactured
   Navier–Stokes control at degree4, N129, lambda=.25 exports 19,125 tanh
   neurons and solves 280 coordinates. Ordinary-network relative errors:
   velocity 7.67e-14, pressure 2.51e-14; raw AD momentum RMS components
   approximately 1.90e-14, 1.22e-14, 1.22e-14. This is a low-degree precision
   control: all-face traces already determine the degree4 velocity, and it
   is not the harder ABC benchmark. Stable grouped arithmetic is more
   accurate than the flattened float64 summation. Do not quote its ~1e-15
   field errors as ordinary-MLP performance.

4. The unforced Burgers IVP remains unresolved at the requested floor.
   Disk ridge coordinates become ill-conditioned on its space-time square:
   sampled basis condition at p28 grows to ~3.09e9, versus ~3.98 on the disk.
   This is not a PDE-Jacobian condition estimate. A stronger disk-coordinate
   run reached 2.44e-7 space-time / 1.19e-5 final-time relative errors, with
   iteration_limit. The explicit box-coordinate follow-up at p24 reached
   ordinary space-time error 1.05e-7 and final-time error 8.54e-6 in a bounded
   run, also iteration_limit. Its fresh physical residual RMS is 2.94e-7.
   p20 exact-polynomial and actual-tanh controls both remain near 3.05e-6.

## What is explicit, and what is solved

For z in R^d, let h(z) in R^(1+B) contain a constant and B prescribed tanh
ridge activations. The fixed matrix E in R^((1+B) x P) maps P coordinates
to an output bias plus ordinary neuron readouts. For a scalar PDE the unknown
is a in R^P and u_a(z)=h(z)^T E a. For q fields, a is P-by-q. E is constructed
from known basis functions, not from an unknown solution. Physics determines
a through the generic nonlinear residual solve R(a)=0, using matrix-free
linearized least squares. This remains a numerical physics solve.

On the unit ball B^d, write C_n=C_n^(d/2), and let sphere measure have total
mass one. The identity used is

    integral_S C_n(omega.x) C_n(omega.nu) d_sigma(omega) = C_n(nu.x).

For known polynomial coordinate phi_k of degree at most p, the coefficient
of directional profile C_n(omega_m.x) is

    E_profile[m,n,k] = q_m (2n+d)/d
                        * E_uniform_ball[phi_k(y) C_n(omega_m.y)].

The normalized ball monomial moments are zero if any exponent is odd. For
alpha=2k they equal product_i (1/2)_(k_i)/(d/2+1)_(sum k_i). This constructs
the map algebraically. Angular cubature exact through degree2p completes
polynomial reproduction before the univariate QUILL approximation. Root's
separate 2–4D known-function study checks capacity only, not a PDE solve.

The high-degree moment implementation initially exposed integer overflow
and cancellation. Factorial products now use Python integers; exact
degree/parity zeros are enforced; degrees above8 use 50 or more decimal
digits during map construction. On this ARM Mac, numpy.longdouble has only
float64 precision. Exported networks and PDE solves remain float64. The
known degree24 box basis still has sampled normalized derivative errors up
to 2.14e-5, so precision of that coordinate map is not universally solved.

Relevant primary foundations:
- https://www.math.auckland.ac.nz/~waldron/Preprints/Ball-polys/ballpolys.pdf
- https://people.math.sc.edu/pencho/Publications/KKLPP-Radon-2010.pdf
- https://arxiv.org/abs/2104.08938

Final relevant checks: 53 passed (depth architecture/jets/export, ridge
construction, ball identities/high-degree arithmetic, generic residual
engine). All bounded experiments finished. Remaining research work is
operator/domain-aware conditioning and dependable high-degree construction,
followed by harder genuine-MLP Navier–Stokes runs. These are limitations,
not completed universal-solver claims. Squared-ReLU is permitted only as a
separately labeled comparison; no RePU PDE result was run or claimed.
