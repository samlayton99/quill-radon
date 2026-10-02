"""Recover an unknown periodic initial profile through constructed PDE dynamics.

The inverse unknowns are 1+2P physical Fourier profile coefficients, not neural
readouts. Each coefficient trajectory b(t) gives readouts explicitly as E b(t).
The QUILL backend evaluates its actual tanh network and derivatives at every RK
stage. Trigonometric features are a separately labeled matched-cost control.

Known Gaussian noise selects the H2 smoothness penalty by the discrepancy
principle (or local prediction-risk estimation). This is a prior, not recovered information. A small
observation Jacobian is used for diagnostics; there is no large readout solve.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import chi2
from .base import QuillBasis1D, Resolution
from native_tanh import modes


@dataclass(frozen=True)
class ProfileObservations:
    x: np.ndarray
    times: np.ndarray
    values: np.ndarray
    sigma: float

    def __post_init__(self):
        x, t, y = (np.asarray(a, float).copy() for a in (self.x, self.times, self.values))
        if x.ndim != 1 or t.ndim != 1 or y.shape != (len(t), len(x)):
            raise ValueError('Expected x[sensor], times[time], values[time,sensor]')
        if not len(t) or np.any(t < 0) or np.any(np.diff(t) <= 0):
            raise ValueError('Observation times must be nonnegative and strictly increasing')
        if (not all(np.all(np.isfinite(a)) for a in (x, t, y)) or
                not np.isfinite(self.sigma) or self.sigma <= 0):
            raise ValueError('Finite observations and a positive known noise scale required')
        for name, value in (('x', x), ('times', t), ('values', y)):
            object.__setattr__(self, name, value)


class ProfileForward:
    """u_t + u u_x = nu u_xx on [-pi,pi], or its heat-equation control."""

    def __init__(self, profile_modes=8, evolution_modes=24, nu=.15, dt=.003,
                 backend='quill', equation='burgers'):
        began = time.perf_counter()
        if (not isinstance(profile_modes, (int, np.integer)) or
                not isinstance(evolution_modes, (int, np.integer)) or
                not 1 <= profile_modes <= evolution_modes):
            raise ValueError('Require 1 <= profile modes <= evolution modes')
        if not np.isfinite(nu) or not np.isfinite(dt) or nu <= 0 or dt <= 0 or equation not in ('burgers', 'heat'):
            raise ValueError('Positive diffusivity/timestep and supported equation required')
        self.p, self.k, self.nu, self.dt = profile_modes, evolution_modes, nu, dt
        self.backend, self.equation = backend, equation
        self.dimension = 2*profile_modes+1
        self.calls = self.rhs_calls = 0
        # Same quadrature and reduced coordinates in both backends.
        q = 6*evolution_modes+4
        self.x = -np.pi+2*np.pi*np.arange(q)/q
        self.A = modes(self.x, self.k).T/q
        self.A[1:] *= 2
        self.initial_map = np.zeros((2*self.k+1, self.dimension))
        self.initial_map[0, 0] = 1
        for j in range(1, self.p+1):
            self.initial_map[j, j] = 1
            self.initial_map[self.k+j, self.p+j] = 1
        if backend == 'quill':
            self.basis = QuillBasis1D((-np.pi, np.pi), Resolution(modes=self.k))
            self.U, self.Ux, self.Uxx = self.basis.U, self.basis.Ux, self.basis.Uxx
            self.geometry = self.basis.metadata
        elif backend == 'trigonometric':
            self.basis = None
            self.U, self.Ux, self.Uxx = self._exact_matrices(self.x)
            self.geometry = {'neurons': 0, 'reduced_coordinates': 2*self.k+1}
        else:
            raise ValueError('backend must be quill or trigonometric')
        self.setup_seconds = time.perf_counter()-began

    def _exact_matrices(self, x):
        x = np.atleast_1d(np.asarray(x, float))
        w = np.arange(1, self.k+1)
        z = x[:, None]*w
        return (modes(x, self.k),
                np.column_stack([np.zeros(len(x)), -w*np.sin(z), w*np.cos(z)]),
                np.column_stack([np.zeros(len(x)), -w*w*np.cos(z), -w*w*np.sin(z)]))

    def matrix(self, x):
        if self.basis is not None:
            return self.basis.matrix(x)
        if np.any(np.abs(x) > np.pi+1e-12):
            raise ValueError('Query outside [-pi,pi]')
        return self._exact_matrices(x)[0]

    def integrate(self, profile, times, tangents=False):
        profile, times = np.asarray(profile, float), np.asarray(times, float)
        if profile.shape != (self.dimension,) or not np.all(np.isfinite(profile)):
            raise ValueError('Invalid initial-profile coefficient vector')
        if times.ndim != 1 or not len(times) or np.any(times < 0) or np.any(np.diff(times) < 0):
            raise ValueError('Output times must be sorted and nonnegative')
        self.calls += 1
        state = np.zeros((2*self.k+1, self.dimension+1 if tangents else 1))
        state[:, 0] = self.initial_map@profile
        if tangents:
            state[:, 1:] = self.initial_map

        def rhs(z):
            self.rhs_calls += 1
            u, ux, uxx = self.U@z, self.Ux@z, self.Uxx@z
            r = self.nu*uxx
            if self.equation == 'burgers':
                r -= u[:, :1]*ux
                if tangents:
                    r[:, 1:] -= u[:, 1:]*ux[:, :1]
            return self.A@r

        outputs, current = [], 0.
        for target in times:
            count = int(np.ceil((target-current)/self.dt))
            if count:
                step = (target-current)/count
                for _ in range(count):
                    a = rhs(state)
                    b = rhs(state+step*a/2)
                    c = rhs(state+step*b/2)
                    d = rhs(state+step*c)
                    state += step*(a+2*b+2*c+d)/6
            if not np.all(np.isfinite(state)):
                raise FloatingPointError('Forward/tangent evolution became nonfinite')
            outputs.append(state.copy())
            current = target
        return np.stack(outputs)

    def predict(self, profile, x, times, tangents=False):
        return np.einsum('xd,tdp->txp', self.matrix(x),
                         self.integrate(profile, times, tangents))

    def readouts(self, profile, t):
        if self.basis is None:
            raise ValueError('Classical control has no neural readout')
        state = self.integrate(profile, np.array([t]))[0, :, 0]
        weights = self.basis.E@state
        return dict(centers=self.basis.enc.centers.copy(), gamma=self.basis.enc.gamma,
                    readout=weights[:-1], bias=float(weights[-1]))


def fit_profile(model, observations, initial=None, regularize=True,
                maximum_alpha=1e6, discrepancy_bisections=5, max_nfev=250,
                regularization_rule='discrepancy', refinement_check=True):
    """Fit an unknown profile; choose regularization using only known noise.

    Minimize ||(pred-y)/sigma||² + alpha ||D p||², with D=(0,k²,k²).
    Default: choose alpha so whitened RSS approximates observation count.
    Optional SURE chooses alpha minimizing RSS+2*effective_df-observation_count
    over a fixed logarithmic grid. This is an exact unbiased prediction-risk
    formula for fixed linear smoothers; using the fitted nonlinear tangent is
    a local approximation, not a global guarantee or risk estimate for the IC.
    Finite bounds +/-2 on each profile coefficient are a disclosed loose prior.
    A fit cannot report data_consistent if even alpha=0 fails a 99.9% residual
    test. Neither alpha choice nor fitting receives withheld reference fields.
    """
    began = time.perf_counter()
    if regularization_rule not in ('sure', 'discrepancy'):
        raise ValueError('Regularization rule must be sure or discrepancy')
    if observations.values.size <= model.dimension:
        raise ValueError('Need more scalar observations than profile coefficients')
    x, times, y, sigma = observations.x, observations.times, observations.values, observations.sigma
    basis = model.matrix(x)
    penalty = np.r_[0., np.arange(1, model.p+1)**2, np.arange(1, model.p+1)**2]
    start = np.zeros(model.dimension) if initial is None else np.asarray(initial, float).copy()
    cache, path = {}, []
    before = model.calls

    def fields(p):
        if 'p' not in cache or not np.array_equal(p, cache['p']):
            state = model.integrate(p, times, True)
            values = np.einsum('xd,tdp->txp', basis, state)
            cache.update(p=p.copy(), residual=(values[:, :, 0]-y).ravel()/sigma,
                         jacobian=values[:, :, 1:].reshape(-1, model.dimension)/sigma,
                         states=state[:, :, 0].copy())
        return cache

    def at_alpha(alpha, guess):
        def fun(p):
            return np.r_[fields(p)['residual'], np.sqrt(alpha)*penalty*p]
        def jac(p):
            return np.vstack([fields(p)['jacobian'], np.diag(np.sqrt(alpha)*penalty)])
        opt = least_squares(fun, guess, jac=jac, bounds=(-2., 2.),
                            ftol=2e-9, xtol=2e-9, gtol=2e-9, max_nfev=max_nfev)
        data = fields(opt.x)
        free = opt.active_mask == 0
        augmented = np.vstack([data['jacobian'][:, free], np.diag(np.sqrt(alpha)*penalty)[:, free]])
        q, _ = np.linalg.qr(augmented, mode='reduced')
        effective_df = float(np.sum(q[:y.size]**2))
        discrepancy = float(np.dot(data['residual'], data['residual']))
        row = dict(alpha=float(alpha), discrepancy=float(np.dot(data['residual'], data['residual'])),
                   optimizer_success=bool(opt.success), evaluations=opt.nfev,
                   active_coefficients=int(np.sum(opt.active_mask != 0)),
                   local_effective_df=effective_df,
                   local_sure=discrepancy+2*effective_df-y.size)
        path.append(row)
        return opt, row

    selected, selected_row = at_alpha(0., start)
    unregularized = selected.x.copy()
    best_data_discrepancy = selected_row['discrepancy']
    unregularized_success = bool(selected.success)
    n, p = y.size, model.dimension
    mismatch_threshold = float(chi2.ppf(.999, n-p))
    inconsistent = unregularized_success and best_data_discrepancy > mismatch_threshold
    lower, upper = None, None
    if regularize and regularization_rule == 'sure' and not inconsistent:
        candidates = [(selected, selected_row)] if selected.success else []
        continuation = selected.x
        for alpha in np.logspace(-3, 6, 10):
            if alpha > maximum_alpha:
                break
            current, row = at_alpha(float(alpha), continuation)
            continuation = current.x
            if current.success:
                candidates.append((current, row))
        if candidates:
            selected, selected_row = min(candidates, key=lambda z: z[1]['local_sure'])
    elif regularize and regularization_rule == 'discrepancy' and not inconsistent and best_data_discrepancy < n:
        for alpha in np.r_[np.logspace(-3, 6, 10)]:
            if alpha > maximum_alpha:
                break
            current, row = at_alpha(float(alpha), selected.x)
            if row['discrepancy'] >= n:
                upper = (float(alpha), current, row)
                break
            lower = (float(alpha), current, row)
            selected, selected_row = current, row
        if upper is not None:
            low = lower[0] if lower is not None else 0.
            high = upper[0]
            candidates = [(selected, selected_row), (upper[1], upper[2])]
            for _ in range(discrepancy_bisections):
                alpha = np.sqrt(low*high) if low > 0 else high/10
                current, row = at_alpha(alpha, selected.x)
                candidates.append((current, row))
                if row['discrepancy'] < n:
                    low = alpha
                else:
                    high = alpha
            selected, selected_row = min(candidates, key=lambda z: abs(z[1]['discrepancy']-n))
    data = fields(selected.x)
    # This SVD is only the small physical observation map (48 by 17 in study).
    singular = np.linalg.svd(data['jacobian'], compute_uv=False)
    rank = int(np.sum(singular > singular[0]*1e-10))
    info_rank = int(np.sum(singular >= 1.))
    status = ('observation_inconsistent' if inconsistent else
              'optimizer_failed' if not selected.success else
              'prior_dominated_directions' if info_rank < model.dimension else 'data_consistent')
    forward_check = {'performed': False}
    if refinement_check and selected.success:
        refined = ProfileForward(profile_modes=model.p, evolution_modes=2*model.k,
                                 nu=model.nu, dt=min(model.dt/2, .5/(model.nu*(2*model.k)**2)),
                                 backend=model.backend, equation=model.equation)
        probe = np.linspace(-np.pi, np.pi, 129)
        coarse_values = model.matrix(probe)@data['states'].T
        fine_values = refined.predict(selected.x, probe, times)[:, :, 0].T
        gap = float(np.sqrt(np.mean((coarse_values-fine_values)**2)))
        limit = .1*sigma
        forward_check = dict(performed=True, refined_modes=2*model.k,
                             refined_dt=refined.dt, spatial_probe_points=len(probe),
                             check_times=len(times), rms_difference=gap,
                             threshold=.1*sigma, fraction_of_noise=gap/sigma,
                             status='agreed_on_probe_points' if gap <= limit else 'underresolved',
                             scope='Empirical resolution agreement at inferred profile; not an error certificate')
        if gap > limit:
            status = 'forward_underresolved'
    report = dict(status=status, coefficients=selected.x.tolist(), unregularized_coefficients=unregularized.tolist(),
                  alpha=selected_row['alpha'], discrepancy=selected_row['discrepancy'],
                  observations=int(n), unknown_profile_coefficients=int(p),
                  alpha_zero_discrepancy=best_data_discrepancy,
                  unregularized_optimizer_success=unregularized_success,
                  selected_optimizer_success=bool(selected.success),
                  active_coefficients=int(np.sum(selected.active_mask != 0)),
                  inconsistency_threshold_99_9=mismatch_threshold,
                  weighted_jacobian_singular_values=singular.tolist(),
                  numerical_rank=rank, directions_with_unit_scale_snr_over_one=info_rank,
                  path=path, forward_tangent_solves=model.calls-before,
                  fit_seconds=time.perf_counter()-began,
                  forward_refinement=forward_check,
                  regularization=f'H2 profile seminorm; {regularization_rule} uses known noise, no heldout truth',
                  regularization_selection_scope='Local prediction-risk estimate for observations; nonlinear approximation, not initial-profile risk or universal guarantee.',
                  uncertainty_scope='The smoothness penalty supplies unrecovered information; no global posterior claim.')
    return selected.x, report


def fit_profile_refined(model, observations, maximum_modes=128, **fit_options):
    """Refine a failed forward check and return the matching model and profile.

    The return value includes the final forward model so callers cannot silently
    evaluate a refined coefficient estimate with a stale coarse representation.
    Refinement never increases the number of inverse profile unknowns.
    """
    if maximum_modes < model.k:
        raise ValueError('Maximum modes cannot be below initial forward resolution')
    began = time.perf_counter()
    guess = fit_options.pop('initial', None)
    history = []
    while True:
        estimate, report = fit_profile(model, observations, initial=guess, **fit_options)
        history.append(dict(modes=model.k, dt=model.dt, status=report['status'],
                            fit_seconds=report['fit_seconds'], forward_refinement=report['forward_refinement']))
        if report['status'] != 'forward_underresolved' or 2*model.k > maximum_modes:
            break
        guess = estimate
        model = ProfileForward(profile_modes=model.p, evolution_modes=2*model.k,
                               nu=model.nu, dt=min(model.dt/2, .5/(model.nu*(2*model.k)**2)),
                               backend=model.backend, equation=model.equation)
    report['refinement_attempts'] = history
    report['all_refinements_seconds'] = time.perf_counter()-began
    report['final_forward_modes'] = model.k
    report['final_forward_dt'] = model.dt
    return model, estimate, report
