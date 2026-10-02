# Direct Radon construction: dimension and allocation results

Completed 2026-09-30. These are actual finite tanh network evaluations, not a dense least-squares existence certificate. No training, coefficient fit, fitted amplitude, or sample-by-neuron matrix is used. The analytic target and its precision matrix are known. Consequently these experiments test representation/encoding cost, not discovery of an unknown PDE solution.

## Conclusions to communicate in chat

- The old three-input dense-solve barrier is bypassed. A genuinely full-rank rotated anisotropic 4D Gaussian reaches **8.15e-15 relative L2** on 1,024 independent validation points, with 32,768 directions and 201 centers (6,586,368 neurons). The independent validation run takes **30.3 seconds** for constructing two targets, continuous-profile evaluation, and finite-network evaluation. With 121 centers, the 128-point experiment still reaches 6.91e-15 using **3,964,928 neurons** in **7.0 seconds**.
- At 5D, 331,776 directions and 81 centers (**26,873,856 neurons**) reach **6.45e-11**, taking **20.1 seconds** for two targets and 128 evaluation points. Raising centers to 201 costs 66,686,976 neurons and 39.5 seconds but leaves **6.43e-11**, because angular cubature controls the error. This highest-direction case has 128 evaluation points; it has not received a separate fresh-point validation. A cheaper 13,172,736-neuron 5D dictionary was independently validated on 1,024 fresh points at **1.19e-7**.
- High ambient dimension is possible but not automatically precise. At 16,384 target-independent Sobol sphere directions and 201 centers (3,293,184 neurons), the median full-rank anisotropic errors are about **0.063%, 0.079%, 0.265%, 1.17%, 2.34%** in dimensions 4,5,8,16,32. Exact numbers and the two scramble realizations are in `dimension.json`. These are angular errors: center conversion remains at roughly 1e-15 to 1e-14.
- Analytically adapting directions with the known Gaussian metric reduces the 32D error to **0.44–0.61%**, and supports a 64D example at **0.46–0.57%**. This is a known-geometry control, not learned geometry. It still has 3,293,184 ordinary tanh neurons.
- Four *known* Fourier ridge directions need only **804 neurons** and reach 5.89e-16 in 256 input dimensions. There is no universal maximum input dimension; directional complexity and availability of the directions matter.
- At an equal budget near 52,000 neurons, the 3D anisotropic Gaussian gives 9.97e-4 at (M,N)=(64,811), 6.36e-7 at (256,203), **2.50e-10 at (576,89)**, then worsens to 1.07e-5 at (1024,49). At a budget near 208,000 neurons, **(1600,129) gives 1.92e-15**, whereas (256,811) remains 6.36e-7 and (4096,49) fails at 1.07e-5. Excessively coarse centers can make the deconvolved coefficients catastrophically large, not merely impose a modest error floor.
- The finite-width correction matters: ordinary q' coefficient samples have **18.2% error in 2D** and **22.7% in 3D** on the anisotropic case with the same geometry; the corrected constructions are at 2.1e-15 and 7.3e-14 respectively.

## Domains, targets, controls, timings

Each target is exp(-x^T B x). Isotropic B=3I. Anisotropic B=R diag(eigenvalues) R^T, with eigenvalues geometrically spaced between 1.5 and 6 and a reproducibly generated orthogonal R. This is full-rank anisotropy, not a low-dimensional ridge hidden in many input coordinates.

All evaluation points lie in the unit Euclidean ball. Their directions are independent normalized Gaussian draws and their radii are equally spaced from 0 to 1. Thus the score weights radius uniformly, **not volume uniformly**; this avoids concentration near the boundary hiding interior errors as dimension increases. These 128-point sets are for reporting, never fitting or selecting coefficients. Separate 1,024-point checks use a fresh direction seed. No uniform-norm guarantee is claimed from these samples.

The primary sweep uses a uniform circle rule in 2D, Gauss–Legendre by periodic-angle sphere cubature in 3D, and normalized inverse-normal Sobol sphere samples in higher dimensions. Higher-dimensional QMC uses two independent scrambles. Because changing the rule confounds a naive dimension comparison, a second study uses the appropriate recursive Gauss–Jacobi sphere rule in 4D and 5D. These tensor rules have M=n^(d-1) antipodally reduced directions at even order n. All outputs evaluate the fixed global neuron dictionary; there is no radial input norm feature or other symmetry shortcut.

Centers span [-4,4] (including halos outside the evaluation ball), h=8/(N-1), gamma=0.25/h. All computations use float64, one requested BLAS/OMP thread. Time measurements are wall time on the shared Mac, hence not a dedicated-machine performance benchmark. `total_seconds` includes target geometry preparation, coefficients, continuous-profile evaluation, and finite tanh evaluation for both targets. Explicit allocation-free giant-layer timings are not implied. Evaluating many more points remains proportional to point count times neuron count.

`compressed_one_target_MiB` counts double-precision directions M*d, readouts M*N, shared center grid N, gamma and bias. A fully untied dense first layer would instead store about 8*M*N*(d+2) bytes. The 5D M=331776,N=201 case needs about **521.4 MiB per target** in the shared-direction format, or **3561.5 MiB** as an ordinary untied layer. `process_peak_MiB` is macOS `ru_maxrss` (bytes converted to MiB), the peak of the whole Python process including imports, intermediate arrays and both targets, not model size and not total computer RAM. That largest N=201 push peaked at **2197.8 MiB**. Chunking keeps transient feature arrays bounded; there is no matrix factorization.

## General-dimensional profile derivation

Let d>=2, B be a positive definite d by d real matrix, mu in R^d, and f:R^d->R be exp(-(x-mu)^T B (x-mu)). Let v be a unit vector and nu the uniform probability measure on the sphere. Define s_v=v^T B^(-1)v and

q_v(t)=det(B)^(-1/2) s_v^(-d/2) 1F1(d/2;1/2;-(t-v^T mu)^2/s_v).

Then f(x)=integral q_v(v^T x) dnu(v).

Proof with Fourier convention fhat(xi)=integral exp(-i xi^T x)f(x)dx: the transform is pi^(d/2) det(B)^(-1/2) exp(-xi^T B^(-1)xi/4) exp(-i xi^T mu). Split inverse Fourier integration into rho>=0 and v on the sphere. Opposite directions cancel the imaginary part. The radial integral is

integral_0^infinity rho^(d-1) exp(-s rho^2/4) cos(rho t) dr
=2^(d-1) s^(-d/2) Gamma(d/2) 1F1(d/2;1/2;-t^2/s).

To see this, expand cosine, integrate each Gaussian moment, use Gamma(d/2+k)=Gamma(d/2)(d/2)_k and (2k)!=4^k k!(1/2)_k. The sphere area 2 pi^(d/2)/Gamma(d/2) and inverse Fourier constants cancel exactly, leaving the stated prefactor. The same formula covers even d, where the filtered Radon profile is nonlocal and has algebraic tails; no odd-dimensional differentiation formula is incorrectly reused.

For B=bI, q_v(t)=1F1(d/2;1/2;-bt^2), independent of direction. The sphere average of (v^T x)^(2k) is |x|^(2k)(1/2)_k/(d/2)_k. Substituting the series recovers exp(-b|x|^2), independently checking normalization.

## Tanh encoding and its assumptions

The normalized derivative kernel k_gamma(t)=(gamma/2)sech^2(gamma t) has Fourier multiplier A omega/sinh(A omega), where A=pi/(2gamma). For a real analytic profile q whose shifted values and Fourier decay permit this operation, the density

rho(c)=Im q(c+iA)/A

has Fourier multiplier i sinh(A omega)/A applied to q. Therefore k_gamma*rho=q'. An integrated representation, anchored at t=0, is

q(t)=q(0)+(1/2) integral rho(c)[tanh(gamma(t-c))-tanh(-gamma c)] dc.

Discretizing with spacing h and angular weights w_m gives a_mj=(h w_m/2)rho_m(c_j), plus an explicitly prescribed bias. Finite h and finite halos introduce numerical approximation errors. The Gaussian profiles here are entire, so the imaginary shift is defined; this is not an unrestricted recipe for profiles with nearby singularities or nonsmooth PDE solutions. The inverse smoothing factor grows exponentially with frequency, making coarse sampling and noisy profile estimates dangerous.

## Allocation theory and observed scope

For direct construction, the error vector is exactly angular quadrature error plus conversion error. The triangle inequality applies. The previously observed least-squares max-of-two-floors law must not be promoted to a universal theorem for these direct coefficients. Here, once one component dominates, total error follows that component as expected.

For analytic integrands under a spectrally accurate tensor angular rule, a conditional model is e_ang approximately C exp(-a M^(1/(d-1))) and e_center approximately D exp(-bN). Balancing the exponents under neuron budget P=MN predicts M proportional to P^((d-1)/d) and N proportional to P^(1/d), with problem-dependent constants. It is an asymptotic allocation model, not a rule for QMC, nonsmooth targets, or tiny unstable N. With algebraic angular convergence instead, the balance differs. The measured fixed-budget U-shaped curves demonstrate why neither "all directions" nor "all centers" works.

A separate grid selector compares predictions under coarsening in M and N, without target values (`allocation_selection.json`). It rejects very coarse center grids, but it is only a consistency heuristic and often retains more centers than the oracle fixed-budget minimum. It is not claimed to be a rigorous adaptive error estimator or optimal allocator.

## Reproduction and files

Run with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1:

- experiments/expF19_radon_direct_pde/scaling.py --mode all
- experiments/expF19_radon_direct_pde/scaling_structured.py
- experiments/expF19_radon_direct_pde/scaling_adapted.py
- experiments/expF19_radon_direct_pde/scaling_controls.py

The extra high-order N=81/121 cells in `structured_allocation.json` use `scaling.cell(d,M,N,vw=scaling_structured.sphere(d,order),rule='Gauss-Jacobi sphere')` at (d,order)=(4,32),(5,24). All figures and JSON records live beside this report. Principal chat figures: `structured_dimensions.png`, `equal_budget.png`, `dimension_scaling.png`, `adapted_geometry.png`.
