# Nonlinear PDEs from initial conditions, with explicit tanh readouts

This is an additive Codex experiment. No interior solution labels, gradient training, least-squares regression, SVD, or Gauss–Newton enters these runs. Burgers uses its special Cole–Hopf reduction. Navier–Stokes uses a classical Fourier Galerkin time integrator; the resulting directional profiles are encoded directly into a tanh network. The Navier–Stokes result is **not** a closed-form solution of arbitrary nonlinear equations and is **not** one fixed four-input space-time MLP. It is an evolving family of three-input spatial MLPs whose readouts depend on time.

## Direct encoding formula and why it is still the same construction

On a periodic spatial box, write a real vector field as

\[
u(x)=\bar u+2\operatorname{Re}\sum_{k\in\mathcal K_+}\widehat u_k e^{ik\cdot x}.
\]

Here \(x\in[-\pi,\pi]^d\), \(k\in\mathbb Z^d\), \(\widehat u_k\in\mathbb C^o\) is the coefficient vector for the \(o\) output fields, and \(\mathcal K_+\) contains one member of every pair \(\{k,-k\}\). Reality supplies the conjugate coefficient. Group wavevectors with the same primitive integer direction. A group has unit direction \(v\) and one-dimensional profile

\[
q_v(s)=2\operatorname{Re}\sum_{k\parallel v}\widehat u_k e^{i|k|s},
\qquad u(x)=\bar u+\sum_vq_v(v\cdot x).
\]

These are exact discrete Fourier ridge profiles. They replace a continuous sphere integral because this task is periodic; no assertion that the ordinary whole-space Radon plane integral of a periodic field converges is required.

For equally spaced centers \(c_j\), spacing \(h\), slope \(\gamma\), and \(A=\pi/(2\gamma)\), the derivative of a tanh expansion is convolution with \(K_\gamma(s)=\gamma\operatorname{sech}^2(\gamma s)/2\). Its Fourier multiplier is \(\widehat K_\gamma(\omega)=A\omega/\sinh(A\omega)\). If the continuous readout density is \(\rho_v\), we need \(K_\gamma*\rho_v=q'_v\). Therefore every Fourier component of the density is

\[
\widehat\rho_v(\omega)
=\frac{i\omega\widehat q_v(\omega)}{\widehat K_\gamma(\omega)}
=\frac{i\sinh(A\omega)}{A}\widehat q_v(\omega).
\]

Trapezoidal quadrature gives the explicit real readout vectors

\[
a_{vj}=h\operatorname{Re}\sum_{k\parallel v}
\frac{i\sinh(A|k|)}A\widehat u_k e^{i|k|c_j}.
\]

The bias sets the known profile value at the origin:

\[
b=\bar u+\sum_vq_v(0)-\sum_{v,j}a_{vj}\tanh(-\gamma c_j).
\]

This is precisely the finite-width-corrected ridge-profile construction. All these operations are explicit sums. The unknown target value is never supplied at the origin: that value is computed from the PDE-produced Fourier state.

For velocity, Fourier incompressibility is \(k\cdot\widehat u_k=0\). Collinear grouping preserves \(v\cdot a_{vj}=0\) for **each neuron**, so

\[
\nabla\cdot\sum_{v,j}a_{vj}\tanh[\gamma(v\cdot x-c_j)]
=\sum_{v,j}\gamma(v\cdot a_{vj})\operatorname{sech}^2[\gamma(v\cdot x-c_j)]=0.
\]

Finite halos do not spoil this structural divergence constraint. The measured defects are roundoff.

## Nonlinear Burgers: an explicit PDE reduction really is possible

Equation \(u_t+uu_x=\nu u_{xx}\), periodic domain \([-\pi,\pi]\), initial condition \(u(x,0)=\sin x\), \(\nu=.15\), evaluation time \(t=.8\). Set \(u=-2\nu\partial_x\log\psi\). Substitution reduces the nonlinear equation to \(\psi_t=\nu\psi_{xx}\). The initial condition supplies \(\psi(x,0)=\exp[\cos x/(2\nu)]\), up to an irrelevant positive constant. Its known Bessel Fourier coefficients are multiplied by \(e^{-\nu k^2t}\). Then \(-2\nu\psi_x/\psi\) supplies the solution without fitting. Fourier coefficients of that resulting smooth profile are calculated by FFT, then the above formula supplies the neural readouts.

At \(h=.04\), \(\gamma=6.25\), 309 neurons in one direction:

- Actual tanh network relative solution error: **5.83e-15** against the Cole–Hopf evaluation.
- Physical PDE residual RMS from the network's own spatial derivatives and encoded time derivative: **2.25e-13**.
- Independent dealiased Fourier RK4 evolution agrees with Cole–Hopf to **6.38e-15**.
- Cole–Hopf's directly evaluated algebraic PDE residual RMS: **2.20e-16**.

The FFT is computing coefficients of a solution already determined by the initial-value formula; it is not a regression on unknown interior labels. This special nonlinear PDE has a linearizing transform. Generic Navier–Stokes does not.

## The center-resolution warning is real and predicted

Setting \(\gamma h=.25\) alone is insufficient. A center lattice aliases frequencies modulo \(2\pi/h\). For one Fourier component \(q_k e^{iks}\), the infinite lattice introduces image frequencies \(\omega_\ell=k+2\pi\ell/h\), with function coefficient ratio

\[
\frac{\text{image coefficient at }\omega_\ell}{q_k}
=\frac{k}{\omega_\ell}\frac{\widehat K_\gamma(\omega_\ell)}{\widehat K_\gamma(k)}
=\frac{\sinh(Ak)}{\sinh(A\omega_\ell)}.
\]

Thus including frequencies above center Nyquist can create *amplified low-frequency aliases*, not merely lose a small unresolved high-frequency tail. This is independent of halo truncation. In the Burgers test, retaining modes through \(k=49\) with \(h=.16\) (Nyquist19.6) gives relative error **1.29e4**, even after extending the halo. Cutting to \(k\le19\) and choosing a sufficient halo gives **9.94e-7**. With \(h=.08\), cutting to \(k\le39\) gives **1.52e-12**. With \(h=.04\), all retained modes resolve and the error is **5.79e-15**. A practical construction must choose the center spacing from the directional spectrum, and choose the halo in units of \(1/\gamma\). The Nyquist condition is necessary but the weighted alias formula, not Nyquist alone, controls high precision.

## Actual interacting three-dimensional Navier–Stokes

Solve on the periodic cube \([-\pi,\pi]^3\):

\[
u_t+(u\cdot\nabla)u+\nabla p=\nu\Delta u,\quad\nabla\cdot u=0,
\]

with \(\nu=.05\), no forcing, and initial velocity

\[
u_0=(\sin x\cos y\cos z,-\cos x\sin y\cos z,0).
\]

This is the three-dimensional Taylor–Green initial-value problem, not the closed-form two-dimensional decaying vortex. By \(t=.5\), 1.266% of kinetic energy lies in modes absent initially, and .629% is in the third velocity component, which was identically zero initially.

The Fourier coefficients obey

\[
\frac{d\widehat u_k}{dt}
=-\nu|k|^2\widehat u_k
-iP_k\sum_{p+q=k}(\widehat u_p\cdot q)\widehat u_q,
\quad P_k=I-kk^\top/|k|^2.
\]

The actual implementation uses the equivalent rotational form \(P(u\times\operatorname{curl}u)\), strict two-thirds dealiasing, and explicit RK4. Pressure is recovered from the divergence constraint, with zero spatial mean. Nonlinear pressure products used for residual verification are evaluated on a doubled grid to avoid product aliases. The time-derivative readouts are obtained by applying the same linear encoding map to the Fourier ODE right-hand side, so they are the derivative of the evolving network while its selected mode set and geometry are fixed.

Why independent spoke evolution fails: products between modes on different directions generate the new vector \(k=p+q\), generally on a new direction. The heat operator is diagonal on spokes; nonlinear advection couples spokes. The construction removes the global least-squares system. It does not remove this physical coupling or the need to integrate it.

| Fourier grid | Shared tanh neurons | Velocity error vs refined grid | Tanh encoding error | Physical momentum RMS | max divergence |
|---|---:|---:|---:|---:|---:|
| 16³ | 337,025 | 9.78e-5 | 1.69e-15 | 2.78e-4 | 9.54e-16 |
| 24³ | 668,525 | 4.06e-6 | 1.75e-15 | 1.68e-5 | 1.53e-15 |
| 32³ | 801,125 | 5.10e-8 | 1.41e-15 | 3.14e-7 | 1.83e-15 |
| 48³ | 1,003,425 | 3.66e-11 | 1.58e-15 | 3.67e-10 | 2.03e-15 |

16–32 use 48³ as the comparison; 48 uses64³. All checks use fresh spatial points. Center spacing .04, gamma6.25, 425 centers per direction including halos. The stored seven-output network includes velocity3, velocity-time-derivative3, and pressure; the latter four are diagnostics/physics outputs. The full48³ readout occupies56.2MB uncompressed. Evolving48³ to t=.5 with200 time steps took11s on this Mac;64³ refinement took27s; encoding48³ and evaluating128 fresh points including all derivatives took.57s. These timings are task-local observations, not general hardware benchmarks.

Additional checks:

- Fixed-grid RK4 differences on halving dt: 4.98e-10,3.08e-11,1.91e-12 (ratios16.2 and16.1).
- Exact initial projected nonlinear TG acceleration agrees to1.25e-15.
- Embedded exact2D TG positive control: velocity3.61e-16,pressure8.15e-16.
- Discrete kinetic-energy balance defect6.94e-18.
- Neural initial-condition error1.51e-15.
- Periodic velocity face mismatch at most1.85e-15; spatial-gradient mismatch2.28e-15; pressure mismatch7.22e-16.

The state comparison and fresh full physical residual decrease together under spatial refinement. Tiny neural encoding error alone would not establish PDE accuracy; this experiment measures both.

## Practical directional sparsity

Allocating a full center string only to active modes reduces exported network size. Dropping a set of conjugate Fourier pairs gives the explicit uniform velocity bound \(2\sum_{k\,\mathrm{dropped}}\|\widehat u_k\|\), plus the encoding error. No exact solution is needed to calculate that bound.

At the48³ final state, pruning by the maximum absolute velocity/pressure mode coefficient gives:

| Mode cutoff | Directions | Neurons | Velocity error vs full Fourier state |
|---|---:|---:|---:|
| 1e-4 |49|20,825|1.18e-3|
| 1e-6 |177|75,225|1.17e-5|
| 1e-8 |393|167,025|2.51e-7|
| 1e-10 |813|345,525|2.35e-9|
| 1e-12 |1365|580,125|3.01e-11|
| 1e-14 |2037|865,725|2.46e-13|

The full PDE error still includes Fourier-grid and time errors. These numbers quantify only the additional export/pruning error. Directions arise from physics-generated Fourier support, not uniformly sampling the sphere.

## Scope

This establishes an initial-data-only, no-training, no-least-squares route to neural representations of a short-time smooth 3D incompressible flow, with controlled evolution and representation errors. It does not establish arbitrary-boundary Navier–Stokes, turbulent high-Reynolds DNS, long-time stability of a neural evolution discretization, a closed-form nonlinear PDE solver, or a globally fixed4D MLP. Periodic Fourier evolution is a mature classical solver; the new project-specific ingredient is the explicit, bandwidth-corrected conversion to the QI/Radon ridge network and its diagnostic/error accounting.

## Sources and reproduction

The pseudospectral method is grounded in [Mortensen and Langtangen,2016](https://arxiv.org/abs/1602.03638) and its [official spectralDNS implementation](https://github.com/spectralDNS/spectralDNS). This experiment is independently written from the equations; spectralDNS was consulted as a method reference, not imported or copied. The periodic Cole–Hopf formulation is also used in [Pelinovsky,2012](https://arxiv.org/abs/1204.3905).

Run with one BLAS/OpenMP thread:

```
.venv/bin/python experiments/expF19_radon_direct_pde/nonlinear_pde.py
.venv/bin/python experiments/expF19_radon_direct_pde/nonlinear_validation.py
.venv/bin/python experiments/expF19_radon_direct_pde/nonlinear_postprocess.py
```

Metrics: `burgers_metrics.json`, `ns_metrics.json`, `validation_metrics.json`, `boundary_sparse_metrics.json`, `burgers_bandwidth_ablation.json`. Figures: `nonlinear_summary.png`, `navier_stokes_3d.png`, `nonlinear_validated.png`. Saved explicit readouts: `burgers.npz`, `ns_snapshot.npz`, `ns_snapshot48.npz`. Figures and conclusions should be presented in chat; this file is only a reproducibility backup.
