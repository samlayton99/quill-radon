"""Constructed neural Galerkin solver on an annulus with boundary data.

The nonlinear residual uses actual QUILL-encoded polynomial values and
derivatives. A polar map, an exact radial boundary factor, angular sine/cosine
features, and product gates extend the architecture beyond a single tanh layer.
Newton-CG solves the small nonlinear PDE coefficient system; there is no
sampled-solution readout fit. A diffusion Cholesky preconditioner is optional.
Supported equation: -div(a grad(u)) + beta*u**3 = f, a>0, beta>=0.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable
import time

import numpy as np
from numpy.polynomial import Legendre, Chebyshev, Polynomial
from numpy.polynomial.legendre import leggauss
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse.linalg import LinearOperator, cg
from scipy.stats import qmc

try:
    from ..quill_boundary import encode
except ImportError:
    from quill_boundary import encode

Array = np.ndarray


@dataclass(frozen=True)
class Annulus:
    inner: float = .4
    outer: float = 1.2

    def __post_init__(self):
        try:
            inner, outer = float(self.inner), float(self.outer)
        except (TypeError, ValueError) as error:
            raise ValueError('Annulus radii must be finite numbers with 0 < inner < outer') from error
        if not np.isfinite([inner, outer]).all() or not 0 < inner < outer:
            raise ValueError('Annulus requires finite radii with 0 < inner < outer; disks are unsupported')
        object.__setattr__(self, 'inner', inner)
        object.__setattr__(self, 'outer', outer)

    def polar(self, xy):
        xy = np.asarray(xy, float)
        return np.linalg.norm(xy, axis=-1), np.arctan2(xy[..., 1], xy[..., 0])

    def quadrature(self, radial_order, angular_order):
        z, w = leggauss(radial_order)
        r = (self.inner+self.outer)/2 + (self.outer-self.inner)*z/2
        theta = 2*np.pi*np.arange(angular_order)/angular_order
        rr, tt = np.meshgrid(r, theta, indexing='ij')
        weights = np.broadcast_to(w[:, None]*(self.outer-self.inner)/2*rr*2*np.pi/angular_order, rr.shape)
        xy = np.column_stack((rr.ravel()*np.cos(tt.ravel()), rr.ravel()*np.sin(tt.ravel())))
        return xy, weights.ravel()


def zero_lift(r, theta):
    """Return value, radial derivative, angular derivative, rr, theta-theta."""
    return tuple(np.zeros_like(r) for _ in range(5))


@dataclass
class AnnulusProblem:
    domain: Annulus
    coefficient: Callable[[Array], Array]
    coefficient_gradient: Callable[[Array], Array]
    forcing: Callable[[Array], Array]
    beta: float = 5.
    boundary_lift: Callable = zero_lift
    name: str = 'annulus_cubic_diffusion'
    inner_robin: Callable | None = None

    def __post_init__(self):
        if not isinstance(self.domain, Annulus):
            raise NotImplementedError('This backend supports Annulus geometry only')
        if not np.isfinite(self.beta) or self.beta < 0:
            raise ValueError('This monotone Newton-CG backend requires finite beta >= 0')


class ConstructedBasis:
    def __init__(self, domain, radial_degree, angular_degree, backend='quill',
                 n_centers=193, lam=.2, inner_boundary='dirichlet'):
        self.domain = domain
        self.inner_boundary = inner_boundary
        if inner_boundary not in ('dirichlet', 'robin'):
            raise ValueError('Supported inner conditions are dirichlet or robin; outer condition is Dirichlet')
        self.radial_degree, self.angular_degree = int(radial_degree), int(angular_degree)
        if self.radial_degree != radial_degree or self.angular_degree != angular_degree or radial_degree < 1 or angular_degree < 1:
            raise ValueError('Positive integer radial and angular degrees required')
        radial_degree, angular_degree = self.radial_degree, self.angular_degree
        if backend not in ('quill', 'classical'):
            raise ValueError('backend must be quill or classical')
        self.backend = backend
        self.radial_polynomials = [Legendre.basis(j) for j in range(radial_degree)]
        self.angular_polynomials = [Chebyshev.basis(j) for j in range(1, angular_degree+1)]
        second_kind = [Polynomial([1.]), Polynomial([0., 2.])]
        for j in range(2, angular_degree):
            second_kind.append(Polynomial([0., 2.])*second_kind[-1]-second_kind[-2])
        self.angular_polynomials += second_kind[:angular_degree]
        self.enc_radial = self.enc_angular = None
        if backend == 'quill':
            halo = int(np.ceil(np.sqrt(n_centers)))
            def make(polynomials):
                return encode(lambda z: np.stack([p(z) for p in polynomials], axis=-1),
                              n_centers-1, lam=lam, halo=halo, interval=(-1., 1.))
            self.enc_radial = make(self.radial_polynomials)
            self.enc_angular = make(self.angular_polynomials)
        self.size = radial_degree*(2*angular_degree+1)
        self.neurons = 0 if backend == 'classical' else len(self.enc_radial.centers)+len(self.enc_angular.centers)

    def _radial(self, r):
        width = self.domain.outer-self.domain.inner
        q = (2*r-self.domain.inner-self.domain.outer)/width
        dq = 2/width
        if self.backend == 'quill':
            p, dp, d2p = (self.enc_radial.evaluate(q, derivative=j) for j in range(3))
        else:
            p, dp, d2p = (np.stack([v.deriv(j)(q) for v in self.radial_polynomials], axis=-1) for j in range(3))
        if self.inner_boundary == 'dirichlet':
            mask = (r-self.domain.inner)*(self.domain.outer-r)/width**2
            dm = (self.domain.inner+self.domain.outer-2*r)/width**2
            d2m = -2/width**2
        else:
            mask = (self.domain.outer-r)/width
            dm = np.full_like(r, -1/width)
            d2m = 0.
        return (mask[:, None]*p,
                dm[:, None]*p+mask[:, None]*dp*dq,
                d2m*p+2*dm[:, None]*dp*dq+mask[:, None]*d2p*dq*dq)

    def _angular(self, theta):
        m = np.arange(1, self.angular_degree+1)
        if self.backend == 'classical':
            angle = theta[:, None]*m
            return (np.column_stack((np.ones(len(theta)), np.cos(angle), np.sin(angle))),
                    np.column_stack((np.zeros(len(theta)), -m*np.sin(angle), m*np.cos(angle))),
                    np.column_stack((np.zeros(len(theta)), -m*m*np.cos(angle), -m*m*np.sin(angle))))
        c, s = np.cos(theta), np.sin(theta)
        h, dh, d2h = (self.enc_angular.evaluate(c, derivative=j) for j in range(3))
        ct, su = h[:, :len(m)], h[:, len(m):]
        ct1, su1 = dh[:, :len(m)], dh[:, len(m):]
        ct2, su2 = d2h[:, :len(m)], d2h[:, len(m):]
        c, s = c[:, None], s[:, None]
        values = np.column_stack((np.ones(len(theta)), ct, s*su))
        first = np.column_stack((np.zeros(len(theta)), -s*ct1, c*su-s*s*su1))
        second = np.column_stack((np.zeros(len(theta)), s*s*ct2-c*ct1,
                                  -s*su-3*s*c*su1+s**3*su2))
        return values, first, second

    def evaluate(self, xy):
        r, theta = self.domain.polar(xy)
        radial, dr, drr = self._radial(r)
        angular, dt, dtt = self._angular(theta)
        def tensor(a, b):
            return np.einsum('ip,ia->ipa', a, b).reshape(len(r), -1)
        value = tensor(radial, angular)
        derivative_r = tensor(dr, angular)
        derivative_t = tensor(radial, dt)
        laplacian = tensor(drr, angular)+derivative_r/r[:, None]+tensor(radial, dtt)/r[:, None]**2
        return value, derivative_r, derivative_t, laplacian


@dataclass
class AnnulusSolution:
    problem: AnnulusProblem
    basis: ConstructedBasis
    coefficients: Array
    metrics: dict = field(default_factory=dict)

    @property
    def status(self):
        return self.metrics.get('status', 'converged' if self.metrics['converged'] else self.metrics['failure'])

    def evaluate(self, xy, derivatives=False):
        xy = np.atleast_2d(np.asarray(xy, float))
        if xy.shape[1] != 2 or not np.all(np.isfinite(xy)):
            raise ValueError('Queries must be finite Cartesian points with shape (n, 2)')
        r, theta = self.problem.domain.polar(xy)
        if np.any(r < self.problem.domain.inner-1e-12) or np.any(r > self.problem.domain.outer+1e-12):
            raise ValueError('Query lies outside the constructed annulus')
        p, pr, pt, lap = self.basis.evaluate(xy)
        g, gr, gt, grr, gtt = self.problem.boundary_lift(r, theta)
        value = g+p@self.coefficients
        if not derivatives:
            return value
        ur, ut = gr+pr@self.coefficients, gt+pt@self.coefficients
        gradient = np.column_stack((np.cos(theta)*ur-np.sin(theta)*ut/r,
                                    np.sin(theta)*ur+np.cos(theta)*ut/r))
        return dict(value=value, gradient=gradient,
                    laplacian=grr+gr/r+gtt/r**2+lap@self.coefficients)

    def residual(self, xy):
        field = self.evaluate(xy, True)
        return (-self.problem.coefficient(xy)*field['laplacian']
                -np.sum(self.problem.coefficient_gradient(xy)*field['gradient'], axis=1)
                +self.problem.beta*field['value']**3-self.problem.forcing(xy))


def solve_annulus(problem, radial_degree=8, angular_degree=6, backend='quill',
                  n_centers=193, lam=.2, quadrature_orders=None, newton_tol=1e-11,
                  max_newton=30, max_cg=None, preconditioner='diffusion',
                  residual_tolerance=1e-5, residual_check_points=256):
    """Solve the weak nonlinear PDE with actual constructed neural basis fields.

    Diffusion preconditioning factors a size P*(2M+1) PDE matrix once.
    Jacobian products remain matrix-free; no global neuron-readout fit occurs.
    Failure is returned in metrics instead of silently accepting the iterate.
    The converged flag refers to the nonlinear Galerkin equations. Public status
    also checks independent sampled strong/Robin residuals when enabled. This
    sampling diagnostic is not a certified continuum error bound.
    """
    started = time.perf_counter()
    if residual_tolerance is not None and (not np.isfinite(residual_tolerance) or residual_tolerance <= 0):
        raise ValueError('residual_tolerance must be positive and finite, or None to disable the check')
    if int(residual_check_points) != residual_check_points or residual_check_points < 16:
        raise ValueError('residual_check_points must be an integer >= 16')
    residual_check_points = int(residual_check_points)
    boundary_kind = 'robin' if problem.inner_robin is not None else 'dirichlet'
    basis = ConstructedBasis(problem.domain, radial_degree, angular_degree, backend, n_centers, lam, boundary_kind)
    nr, nt = quadrature_orders or (3*radial_degree+12, 6*angular_degree+12)
    xy, weight = problem.domain.quadrature(nr, nt)
    r, theta = problem.domain.polar(xy)
    p, pr, pt, lap = basis.evaluate(xy)
    a = problem.coefficient(xy)
    if np.min(a) <= 0 or not np.all(np.isfinite(a)):
        raise ValueError('Diffusion coefficient must be finite and positive at quadrature nodes')
    g, gr, gt, _, _ = problem.boundary_lift(r, theta)
    wa = weight*a
    diffusion = pr.T@(wa[:, None]*pr)+pt.T@((wa/r**2)[:, None]*pt)
    load = pr.T@(wa*gr)+pt.T@(wa*gt/r**2)-p.T@(weight*problem.forcing(xy))
    if problem.inner_robin is not None:
        bt = 2*np.pi*np.arange(nt)/nt
        br = np.full(nt, problem.domain.inner)
        bx = np.column_stack((br*np.cos(bt), br*np.sin(bt)))
        boundary_basis = basis.evaluate(bx)[0]
        alpha, target = problem.inner_robin(bt)
        alpha = np.broadcast_to(np.asarray(alpha, float), bt.shape)
        target = np.broadcast_to(np.asarray(target, float), bt.shape)
        if np.min(alpha) < 0:
            raise ValueError('This monotone backend requires nonnegative Robin alpha')
        boundary_weight = problem.domain.inner*2*np.pi/nt
        boundary_lift = problem.boundary_lift(br, bt)[0]
        diffusion += boundary_basis.T@((boundary_weight*alpha)[:, None]*boundary_basis)
        load += boundary_basis.T@(boundary_weight*(alpha*boundary_lift-target))
    scale = max(float(np.linalg.norm(load)), np.finfo(float).eps)
    if preconditioner == 'diffusion':
        factor = cho_factor(diffusion, lower=True, check_finite=False)
        pre = LinearOperator((basis.size, basis.size), matvec=lambda v: cho_solve(factor, v, check_finite=False))
    elif preconditioner == 'diagonal':
        pre_diagonal = np.diag(diffusion).copy()
        pre = LinearOperator((basis.size, basis.size), matvec=lambda v: v/pre_diagonal)
    else:
        raise ValueError('preconditioner must be diffusion or diagonal')
    setup_seconds = time.perf_counter()-started
    beta = problem.beta
    def residual(b):
        u = g+p@b
        return diffusion@b+load+beta*p.T@(weight*u**3)
    b = np.zeros(basis.size)
    history, cg_iterations = [], []
    converged, failure = False, None
    start_solve = time.perf_counter()
    for iteration in range(max_newton+1):
        res = residual(b)
        norm = float(np.linalg.norm(res))
        history.append(norm/scale)
        if norm/scale < newton_tol:
            converged = True
            break
        if iteration == max_newton:
            failure = 'newton_iteration_limit'
            break
        u = g+p@b
        reaction_diagonal = 3*beta*weight*u*u
        jac = LinearOperator((basis.size, basis.size), matvec=lambda v: diffusion@v+p.T@(reaction_diagonal*(p@v)))
        count = [0]
        def count_step(_):
            count[0] += 1
        step, status = cg(jac, -res, M=pre, rtol=1e-10, atol=0.,
                          maxiter=max_cg or max(100, 3*basis.size), callback=count_step)
        cg_iterations.append(count[0])
        if status != 0:
            failure = 'cg_iteration_limit' if status > 0 else 'cg_breakdown'
            break
        damping = 1.
        for _ in range(25):
            candidate = b+damping*step
            if np.linalg.norm(residual(candidate)) <= (1-1e-4*damping)*norm:
                b = candidate
                break
            damping /= 2
        else:
            failure = 'newton_line_search_failure'
            break
    solve_seconds = time.perf_counter()-start_solve
    metrics = dict(name=problem.name, backend=backend, converged=converged, nonlinear_converged=converged, failure=failure,
                   convergence_scope='converged/nonlinear_converged flags mean nonlinear Galerkin convergence only; status additionally applies independent sampled physical residual checks, not a certified spatial error bound',
                   boundary_conditions=f'inner {boundary_kind}; outer Dirichlet',
                   radial_degree=radial_degree, angular_degree=angular_degree,
                   latent_coefficients=basis.size, tanh_neurons=basis.neurons,
                   n_centers=n_centers if backend == 'quill' else 0, lam=lam,
                   quadrature_orders=[nr, nt], quadrature_points=len(xy),
                   setup_seconds=setup_seconds, solve_seconds=solve_seconds,
                   total_seconds=time.perf_counter()-started, newton_updates=len(history)-1,
                   relative_weak_residual=history[-1], residual_history=history,
                   cg_iterations=cg_iterations, preconditioner=preconditioner,
                   preconditioner_matrix_shape=list(diffusion.shape),
                   coefficient_min=float(np.min(a)), coefficient_max=float(np.max(a)),
                   working_array_bytes=sum(v.nbytes for v in (p, pr, pt, lap, diffusion)),
                   architecture='Polar coordinate features; QUILL polynomial banks; exact boundary mask; product gates; small nonlinear Galerkin coefficient system')
    solution = AnnulusSolution(problem, basis, b, metrics)
    metrics['status'] = failure if not converged else 'nonlinear_converged'
    metrics['residual_tolerance'] = residual_tolerance
    metrics['residual_check'] = 'disabled' if residual_tolerance is None else 'not_run_nonlinear_failure'
    if converged and residual_tolerance is not None:
        check_started = time.perf_counter()
        z = qmc.Sobol(2, scramble=True, seed=78103).random_base2(int(np.ceil(np.log2(residual_check_points))))[:residual_check_points]
        cr = np.sqrt(problem.domain.inner**2+(problem.domain.outer**2-problem.domain.inner**2)*z[:, 0])
        ct = 2*np.pi*z[:, 1]
        checks = np.column_stack((cr*np.cos(ct), cr*np.sin(ct)))
        strong = solution.residual(checks)
        forcing_rms = float(np.sqrt(np.mean(problem.forcing(checks)**2)))
        strong_rms = float(np.sqrt(np.mean(strong**2)))
        indicator = strong_rms/max(1., forcing_rms)
        metrics.update(residual_check='independent_area_uniform_Sobol', residual_check_points=residual_check_points,
                       strong_residual_rms=strong_rms, strong_residual_indicator=indicator,
                       residual_normalization='RMS residual divided by max(1, RMS prescribed data)',
                       strong_residual_check_note='Finite samples can miss localized errors; refine resolution/quadrature for validation')
        if problem.inner_robin is not None:
            theta = 2*np.pi*(np.arange(residual_check_points)+.37)/residual_check_points
            normals = -np.column_stack((np.cos(theta), np.sin(theta)))
            inner = -problem.domain.inner*normals
            fields = solution.evaluate(inner, True)
            alpha, target = problem.inner_robin(theta)
            boundary_residual = problem.coefficient(inner)*np.sum(normals*fields['gradient'], axis=1)+alpha*fields['value']-target
            boundary_indicator = float(np.sqrt(np.mean(boundary_residual**2))/max(1., np.sqrt(np.mean(np.asarray(target)**2))))
            metrics['robin_residual_indicator'] = boundary_indicator
            indicator = max(indicator, boundary_indicator)
        metrics['status'] = 'converged' if np.isfinite(indicator) and indicator <= residual_tolerance else 'underresolved'
        metrics['residual_check_seconds'] = time.perf_counter()-check_started
    metrics['total_seconds'] = time.perf_counter()-started
    return solution
