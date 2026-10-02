# Direct Radon/QI PDE construction: linear cases

No readout least squares, no optimizer, and no interior solution labels enter any construction here. There are three distinct algorithms, and they must not be conflated:

1. `linear.py` uses the supplied IC and an analytic PDE propagator to construct coefficients. Advection and Airy are fixed two-input tanh MLPs. Heat uses spatial tanh features with explicitly time-dependent readouts.
2. `linear_numerical.py` samples the initial filtered Radon profiles and evolves them with a one-dimensional FFT heat solver. It does not call the analytic interior profiles. Neural readouts are then direct transforms of the evolved Fourier coefficients. This is a classical spectral PDE solver plus explicit neural compilation.
3. `linear_wave.py` gives a genuine fixed four-input tanh MLP for a full-space three-spatial-dimensional wave IVP. Its evaluation has ordinary affine input weights, tanh, and a fixed linear output; no time-dependent weights or time stepping.

All reference values are evaluated at independent Sobol points. Whole-space Cauchy conditions are assumed for advection, Airy, 3D heat, and 3D wave; no artificial bounded-domain boundary conditions are silently inferred. The 1D heat test uses the supplied two sine-mode initial condition, with compatible zero Dirichlet boundaries. These are deliberately smooth, constant-coefficient linear PDEs, not a claim to have solved general nonlinear PDEs.

## Equations and derivation

For a scalar profile q:R→R with analytic continuation into the required complex strip, define a=pi/(2 gamma). The inverse smoothing density is

    d_gamma(s) = Im q(s + i a)/a.

It is not simply q'(s). In Fourier variables its multiplier is

    i omega sinh(a omega)/(a omega).

To derive this, the normalized tanh derivative K_gamma(s)=(gamma/2) sech²(gamma s) has Fourier multiplier a omega/sinh(a omega). Differentiating a continuous tanh synthesis with coefficient density d_gamma/2 gives K_gamma*d_gamma. Multiplying the two Fourier multipliers gives i omega qhat, hence the derivative is q'. The additive constant is supplied from q(0). Sampling this density at c_j with uniform spacing h gives a_j=(h/2)d_gamma(c_j). Truncation, quadrature aliasing, and roundoff remain; the continuous identity is not a finite-width exactness theorem. For general nonanalytic/noisy q, inverse smoothing is unbounded and requires a spectral cutoff. The numerical experiment uses a fixed relative cutoff 1e-14 on the initial profile spectra, never on true interior errors.

For advection u_t+u_x=0 with initial q_0(x), put s=x-t and construct q_0(s). Each tanh neuron itself satisfies the PDE, whatever its readout. Initial-condition accuracy is therefore indispensable; a zero PDE residual does not imply the correct IVP solution.

For Airy u_t+u_xxx=0, an initial sine frequency k evolves as sin(kx+k³t). For each of the two supplied frequencies, construct q(s)=A sin(s) along direction (k,k³). This is a fixed two-input MLP. The third derivative amplifies floating-point cancellation: value errors can be ~1e-16 while scaled PDE residual is ~1e-12 to 1e-10.

For heat u_t = div(D grad u) in R³ with constant symmetric positive-definite D, define the plane Radon transform

    Ru(v,s,t) = integral over {x: v dot x=s} of u(x,t),   ||v||=1.

Integration by parts (or Fourier slicing) gives

    R[partial_i partial_j u] = v_i v_j partial_s² Ru.

Consequently Ru_t=(v^T D v) Ru_ss. The normalized inversion profile is

    q_v(s,t) = -(1/(2 pi)) partial_s² Ru(v,s,t),
    u(x,t) = average over S² of q_v(v dot x,t).

The same PDE holds for q_v:

    (q_v)_t = (v^T D v) (q_v)_ss.

Thus each directional profile is an independent one-dimensional heat problem. If the supplied IC includes A_0 exp[-(x-mu)^T P (x-mu)], let

    B(t)=P^{-1}+4tD,  S_v(t)=v^T B(t) v,  y=s-v dot mu.

Its initial Radon transform is Gaussian. Solving its scalar heat equation gives

    q_v(s,t) = A_0 / [sqrt(det P) S_v(t)^(3/2)]
               * (1-2y²/S_v(t)) exp[-y²/S_v(t)].

This formula is derived from IC and the operator; no interior observations enter. In `linear_numerical.py`, replace this analytic propagation by FFT of q_v(s,0) and multiply mode omega by exp[-t(v^T D v)omega²]. The offset derivative, Fourier multiplier and explicit coefficient law then produce a(t). This eliminates a global neural solve but does not eliminate the underlying PDE calculation.

For the wave equation u_tt=c² Delta u with initial displacement f and zero initial velocity, its filtered profiles obey (q_v)_tt=c²(q_v)_ss. D'Alembert gives

    q_v(s,t) = [q_v(s-ct,0)+q_v(s+ct,0)]/2.

Compile q_v(s,0) once with coefficient a_vj. Each compiled pair becomes

    (a_vj/2) tanh(gamma[v dot x - ct - c_j])
    + (a_vj/2) tanh(gamma[v dot x + ct - c_j]).

The fixed input weight vectors are gamma(v,-c) and gamma(v,+c). Each neuron satisfies the wave PDE exactly because ||v||=1. This is a conventional four-input, one-hidden-layer tanh MLP. Its independent reference uses the Kirchhoff formula

    u(x,t) = average_{omega in S²} [f(x+ct omega)
                                  + ct omega dot grad f(x+ct omega)],

integrated in physical space, not using Radon profiles.

## Critical numerical distinction

The continuous directional approximation solves the constant-coefficient heat or wave PDE even with too few angular directions. Accordingly, a very small PDE residual can coexist with the wrong initial field. The heat sweep shows this directly: angular errors fall from percent level to ~1e-15, while the PDE residual is already ~1e-13 at the coarsest angle count. Initial-condition checks and nested angular refinement must accompany residual checks.

The error budget is: initial-profile computation + one-dimensional PDE evolution + directional quadrature + finite tanh compilation + truncation/boundary error + roundoff. More centers repair the offset/compilation error; they do not repair insufficient directions. More directions do not repair inaccurate one-dimensional profiles.

## Recorded outputs

- `metrics.json`: F01-like advection, Airy and heat; analytic 3D heat propagation; all widths and residuals.
- `numerical_evolution_metrics.json`: sampled-IC FFT heat evolution, including 1,048,576-neuron evaluator.
- `wave_metrics.json`: genuine fixed four-input wave MLP and independent Kirchhoff reference refinement.
- `linear_ivp_convergence.png`, `radon_heat3d.png`, `radon_wave3d_fixed.png`: chat-ready figures.
- Compressed network parameters: actual directly generated coefficients, geometry and biases.

The heat evaluator's analytic time derivative was checked against central differences of the evaluated network, relative difference 1.16e-10 at step 1e-5. The two reference derivatives were not equated in the code to force the heat residual to zero.

## Literature

The derivative intertwining is classical. Donsub Rim, *Dimensional splitting of hyperbolic partial differential equations using the Radon transform*, SIAM J. Sci. Comput. 40(6), 2018, explicitly develops Radon dimensional splitting: https://arxiv.org/abs/1705.03609 . The contribution here is its combination with the independently specified tanh inverse-smoothing coefficient construction, and these reproducible precision checks; the dimensional splitting is not a newly discovered principle.

An independent dense PyTorch MLP/autograd check verifies the factorized evaluators: the fixed 4-input wave network (32,128 neurons for this check) matches the factorized output to 5.93e-16 relative error and gives autograd PDE RMS 4.59e-15. The 583-neuron Airy MLP matches to 6.28e-16 and has autograd PDE RMS 2.06e-11. Stored in `independent_autograd_checks.json`. `config.json` records all ICs, seeds, domains, widths, quadrature counts and spectrum cutoff. No plots or scripts depend on unsaved external state.
