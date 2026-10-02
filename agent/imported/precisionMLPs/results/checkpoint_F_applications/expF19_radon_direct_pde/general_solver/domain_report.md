# Constructed neural elliptic solver on an annulus

The reusable backend solves `-div(a grad u) + beta u^3 = f` on a curved, nonconvex annulus. Actual QUILL values and derivatives enter every nonlinear Galerkin residual. No interior solution labels, external numerical solution, or neuron-readout least-squares fit determine the field. A small global PDE coefficient matrix is explicitly factored for preconditioning; this is not a claim of avoiding all linear algebra.

## Public interface

Code: `experiments/expF19_radon_direct_pde/solver/elliptic.py`.

```python
from solver.elliptic import Annulus, AnnulusProblem, solve_annulus

problem = AnnulusProblem(
    domain=Annulus(inner=.4, outer=1.2),
    coefficient=a,                 # a(xy), positive scalar diffusion
    coefficient_gradient=grad_a,    # Cartesian gradient, shape (n, 2)
    forcing=f,                     # prescribed forcing values
    beta=5.,
    boundary_lift=g,                # g, g_r, g_theta, g_rr, g_theta_theta
)
# Optional: a*d_n(u) + alpha(theta)*u = h(theta) on the inner circle.
# problem.inner_robin = lambda theta: (alpha(theta), h(theta))
solution = solve_annulus(problem, radial_degree=10, angular_degree=9,
                         n_centers=241, lam=.2, residual_tolerance=1e-5)
values = solution.evaluate(xy)
fields = solution.evaluate(xy, derivatives=True)
residual = solution.residual(xy)
assert solution.status == 'converged'
```

The default lift is zero. Outer Dirichlet data always come from the lift. With no Robin callback, its inner trace also supplies Dirichlet data. A failed solve returns a failure status and diagnostics; it does not silently mark the approximate field converged. Queries outside the annulus are rejected, as are disks, nonfinite radii, non-annular geometry, and unsupported inner boundary options.

`metrics['converged']` and `metrics['nonlinear_converged']` refer only to the discrete nonlinear Galerkin equations. Public `solution.status` also checks strong PDE residuals on 256 independent area-uniform Sobol points, and Robin residuals at independent boundary angles when present. The indicator is RMS residual divided by `max(1, RMS prescribed data)`; a value above `residual_tolerance` returns `underresolved`. Setting that tolerance to `None` disables the check and returns `nonlinear_converged` after a successful discrete solve. Finite residual samples can miss localized errors and are not a certified continuum error bound. These distinctions are recorded in `convergence_scope`.

## Representation and equations

The architecture includes a polar map, two shared QUILL polynomial banks, explicit sine/cosine coordinate features, a boundary factor, and product gates. It is not a single hidden-layer tanh network. The radial bank approximates Legendre polynomials in normalized radius. The angular bank approximates Chebyshev polynomials evaluated at `cos(theta)`; the identity `sin(m theta)=sin(theta) U_(m-1)(cos(theta))` gives exactly periodic angular features even before considering encoding accuracy.

For two Dirichlet circles, the radial factor is `(r-ri)(ro-r)/(ro-ri)^2`; for inner Robin and outer Dirichlet, it is `(ro-r)/(ro-ri)`. Multiplication makes the homogeneous correction vanish exactly on the prescribed Dirichlet circles. The known lift supplies nonzero boundary values. Inner Robin data enter through the usual boundary integral in the weak equation, so their pointwise residual is measured separately.

The nonlinear residual and its Newton Jacobian use the actual encoded basis. Newton-CG applies Jacobian products without constructing a nonlinear Jacobian matrix. A Cholesky factorization of the small diffusion/Robin matrix supplies the default preconditioner. The finest example has 190 PDE coefficients and a 190-by-190 preconditioning matrix, with 546 hidden tanh neurons in its two shared banks.

Geometry-aware boundary trial functions have established precedents; they are not a new consequence of Radon inversion. See [Sukumar and Srivastava, exact boundary imposition](https://arxiv.org/abs/2104.08426).

## Measurements

The annulus has radii .4 and 1.2. The main problem uses `a=exp(.8x)`, `beta=5`, and zero Dirichlet values. Prescribed manufactured forcing comes from a smooth analytic expression with infinitely many radial/angular expansion coefficients. Its interior values are used only for validation. All reported norms are empirical norms on 1,024 fresh area-uniform Sobol points.

| PDE coefficients | Tanh neurons | Relative value error | Relative off-grid strong residual | Setup + solve/check seconds | Status |
|---:|---:|---:|---:|---:|---|
| 28 | 306 | 1.11e-2 | 1.34e-2 | .00926 | underresolved |
| 66 | 342 | 1.85e-5 | 3.40e-5 | .01346 | underresolved |
| 120 | 442 | 9.25e-9 | 2.58e-8 | .02279 | converged |
| 190 | 546 | 2.19e-12 | 8.92e-12 | .03760 | converged |

All four discrete nonlinear systems converge in five Newton updates; the two coarser fields fail the separate spatial residual diagnostic. Dirichlet trace errors are approximately 1.6e-15. A finer integration rule changes the finest field by 1.96e-14. Direct Cartesian finite differences of the actual neural evaluator check its analytic gradient and Laplacian to approximately 9e-8 and 2.5e-8 at step 2e-4, consistent with the finite-difference truncation error.

At 190 coefficients:

| Additional case | Value error | Strong residual | Newton updates | Total seconds |
|---|---:|---:|---:|---:|
| Different nonzero Dirichlet data on the two circles | 2.28e-12 | 8.93e-12 | 5 | .03626 |
| Diffusion contrast 10,000, beta=50, nonzero Dirichlet data | 3.91e-12 | 1.21e-12 | 8 | .04887 |
| Inner Robin, outer Dirichlet | 2.21e-12 | 2.56e-11 | 6 | .04160 |

The Robin relative residual is 2.37e-12 on 503 fresh angular points. The high-contrast case uses up to 42 CG iterations per Newton step with the diffusion preconditioner. A diagonal preconditioner limited to 30 CG iterations fails at its first step and returns `cg_iteration_limit`; this is a resource-limited control, not evidence that the underlying PDE is unsolvable.

Matched classical Legendre/Fourier Galerkin solutions agree with the neural fields to approximately 1e-15, except the high-contrast comparison at 2.25e-14. Their measured total times, including the same independent residual check, are lower: .00990 seconds for the finest main case, and .02388 seconds for the high-contrast case. No speed superiority over the matching classical method is claimed. The complete recorded harness, including validation and plotting, took .98 seconds with 463 MB peak process memory on one CPU thread.

## Supported scope and reproduction

This backend supports annuli, smooth positive scalar diffusion, a nonnegative cubic reaction coefficient, prescribed Dirichlet data, and optional inner Robin data with nonnegative Robin coefficient. It does not implement arbitrary CAD geometry, pure Neumann problems, shocks, high-dimensional domains, or nonmonotone reaction laws. A coefficient gradient is required for the strong residual diagnostic.

```sh
.venv/bin/python experiments/expF19_radon_direct_pde/general_domains.py
```

`domain_metrics.json` stores all native/classical comparisons and the failure control. `domain_state.npz` stores the finest homogeneous field coefficients, both explicit QUILL banks, and validation values. `domain_comparison.png` shows the nonconvex domain, refinement, and nonlinear convergence. Existing prototype solvers were not changed.
