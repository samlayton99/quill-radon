"""Periodic 3-D NS with shared QUILL banks and explicit product gates.

This is a constructed tensor-product neural spectral method, not a flat ridge
network or a general-geometry solver. Every native RHS evaluates actual QUILL
values and analytic derivatives. Real-data Fourier analysis and Leray projection
close the finite system; no target trajectory or readout least-squares is used.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
import numpy as np

try:
    from ..quill_boundary import encode
except ImportError:
    from quill_boundary import encode


def _apply_axis(matrix, array, axis):
    return np.moveaxis(np.tensordot(matrix, array, axes=(1, axis)), 0, axis)


def _tensor(matrices, coefficients):
    out = coefficients
    for axis, matrix in enumerate(matrices):
        out = _apply_axis(matrix, out, axis)
    return out


def _complex_coefficients(b, inverse=False):
    out = np.asarray(b, complex)
    for axis in range(3):
        x = np.moveaxis(out, axis, 0)
        y = np.empty_like(x)
        y[0] = x[0]
        if inverse:
            y[1::2] = x[1::2] + x[2::2]
            y[2::2] = 1j*(x[1::2] - x[2::2])
        else:
            y[1::2] = (x[1::2] - 1j*x[2::2])/2
            y[2::2] = (x[1::2] + 1j*x[2::2])/2
        out = np.moveaxis(y, 0, axis)
    return out.real if inverse else out


def taylor_green(points):
    x, y, z = np.asarray(points).T
    return np.column_stack((np.sin(x)*np.cos(y)*np.cos(z),
                            -np.cos(x)*np.sin(y)*np.cos(z), np.zeros(len(x))))


@dataclass
class NSResult:
    model: 'PeriodicNS'
    coefficients: np.ndarray
    time: float
    metrics: dict
    history: list

    @property
    def status(self):
        return self.metrics['status']

    def evaluate(self, points, derivatives=False):
        return self.model.evaluate(self.coefficients, points, derivatives)


class PeriodicNS:
    """Prescribed cubic torus [-pi,pi]^3; positive viscosity, smooth data.

    ``initial(points)`` and ``forcing(t, points)`` return arrays of shape (N,3).
    Initial velocity and forcing are projected to divergence-free retained
    Fourier modes. The actual constructed field is solenoidal only to encoder
    accuracy. The ``fft`` architecture is a conventional comparison backend.
    """

    def __init__(self, cutoff=5, grid=None, viscosity=.05,
                 architecture='quill_product', n_centers=257, lam=.2):
        began = time.perf_counter()
        if int(cutoff) != cutoff or cutoff < 1:
            raise ValueError('cutoff must be a positive integer')
        self.cutoff = int(cutoff)
        self.grid = int(grid) if grid is not None else 3*self.cutoff+3
        if self.grid <= 3*self.cutoff or (grid is not None and self.grid != grid):
            raise ValueError('grid must be an integer > 3*cutoff for quadratic dealiasing')
        if not np.isfinite(viscosity) or viscosity <= 0:
            raise ValueError('viscosity must be finite and positive')
        if architecture not in ('quill_product', 'fft', 'trig_product'):
            raise ValueError("architecture must be 'quill_product', 'fft', or 'trig_product'")
        self.viscosity, self.architecture = float(viscosity), architecture
        self.length = 2*self.cutoff+1
        self.frequency = np.r_[0, np.column_stack((np.arange(1, self.cutoff+1),
                                                   -np.arange(1, self.cutoff+1))).ravel()]
        self.real_frequency = np.abs(self.frequency)
        self.wave = np.array(np.meshgrid(*([self.frequency]*3), indexing='ij'))
        self.norm2 = np.sum(self.wave**2, axis=0)
        self.phase = (-1.)**np.sum(self.wave, axis=0)
        self.index = np.ix_(*([self.frequency % self.grid]*3))
        self.axis = -np.pi+2*np.pi*np.arange(self.grid)/self.grid
        self.points = np.array(np.meshgrid(*([self.axis]*3), indexing='ij')).reshape(3, -1).T
        self.encoder = None
        if architecture == 'quill_product':
            if int(n_centers) != n_centers or n_centers < 17:
                raise ValueError('n_centers must be an integer >=17')
            rho = np.arange(1, self.cutoff+1)
            def profile(z):
                angle = np.asarray(z)[..., None]*rho
                return np.stack((np.cos(angle), np.sin(angle)), axis=-1).reshape(np.shape(z)+(2*self.cutoff,))
            self.encoder = encode(profile, int(n_centers)-1, lam=lam,
                                  interval=(-np.pi, np.pi))
        self.bank = [self._bank(self.axis, d) for d in range(3)]
        self.build = dict(seconds=time.perf_counter()-began,
                          architecture=architecture,
                          architecture_note=('Three shared 1D QUILL banks, explicit products, tensor readout; not a flat ridge tanh network' if self.encoder else ('Conventional FFT pseudospectral comparison in equivalent real tensor coordinates' if architecture == 'fft' else 'Exact trigonometric product basis with the same tensor contractions as the native backend')),
                          hidden_tanh_neurons=0 if self.encoder is None else 3*len(self.encoder.centers),
                          unique_encoder_parameter_banks=0 if self.encoder is None else 1,
                          scalar_modes=self.length**3, velocity_coefficients=3*self.length**3,
                          basis_cache_bytes=sum(x.nbytes for x in self.bank),
                          n_centers=n_centers if self.encoder else 0, lam=lam if self.encoder else None)
        if self.encoder:
            dense = np.linspace(-np.pi, np.pi, 513)
            self.build['basis_value_error'] = float(np.max(abs(self._bank(dense, 0)-self._exact_bank(dense, 0))))
            self.build['basis_first_error'] = float(np.max(abs(self._bank(dense, 1)-self._exact_bank(dense, 1))))
            self.build['basis_second_error'] = float(np.max(abs(self._bank(dense, 2)-self._exact_bank(dense, 2))))
        self.build['seconds'] = time.perf_counter()-began

    def _exact_bank(self, x, derivative=0):
        k = np.arange(1, self.cutoff+1)
        p = np.asarray(x)[:, None]*k
        out = np.empty((len(x), self.length))
        out[:, 0] = 1 if derivative == 0 else 0
        if derivative == 0:
            out[:, 1::2], out[:, 2::2] = np.cos(p), np.sin(p)
        elif derivative == 1:
            out[:, 1::2], out[:, 2::2] = -k*np.sin(p), k*np.cos(p)
        else:
            out[:, 1::2], out[:, 2::2] = -k*k*np.cos(p), -k*k*np.sin(p)
        return out

    def _bank(self, x, derivative=0):
        if self.encoder is None:
            return self._exact_bank(x, derivative)
        out = np.empty((len(x), self.length))
        out[:, 0] = 1 if derivative == 0 else 0
        out[:, 1:] = self.encoder.evaluate(x, derivative)
        return out

    def _project(self, h):
        dot = np.sum(np.moveaxis(self.wave, 0, -1)*h, axis=-1)
        scale = np.divide(dot, self.norm2, out=np.zeros_like(dot), where=self.norm2 > 0)
        return h-np.moveaxis(self.wave, 0, -1)*scale[..., None]

    def analyze(self, values, project=True):
        values = np.asarray(values, float)
        if values.shape != (self.grid**3, 3) or not np.all(np.isfinite(values)):
            raise ValueError(f'field must return finite ({self.grid**3},3) values')
        h = np.fft.fftn(values.reshape((self.grid,)*3+(3,)), axes=(0, 1, 2))/self.grid**3
        h = h[self.index]*self.phase[..., None]
        if project:
            h = self._project(h)
        return _complex_coefficients(h, inverse=True)

    def _fft_fields(self, b, derivatives=True):
        h = np.zeros((self.grid,)*3+(3,), complex)
        h[self.index] = _complex_coefficients(b)*self.phase[..., None]
        def physical(hh):
            return (np.fft.ifftn(hh, axes=(0, 1, 2))*self.grid**3).real.reshape(-1, 3)
        u = physical(h)
        if not derivatives:
            return u
        freq = np.fft.fftfreq(self.grid, 1/self.grid)
        wave = np.array(np.meshgrid(freq, freq, freq, indexing='ij'))
        grad = np.stack([physical(1j*wave[d, ..., None]*h) for d in range(3)], axis=2)
        lap = physical(-np.sum(wave*wave, axis=0)[..., None]*h)
        return u, grad, lap

    def fields(self, b, axes=None, derivatives=True):
        if axes is None and self.architecture == 'fft':
            return self._fft_fields(b, derivatives)
        banks = [[self.bank[d] for d in range(3)] for _ in range(3)] if axes is None else [[self._bank(x, d) for d in range(3)] for x in axes]
        base = [bank[0] for bank in banks]
        u = _tensor(base, b).reshape(-1, 3)
        if not derivatives:
            return u
        grad, lap = [], np.zeros_like(u)
        for axis in range(3):
            first, second = base.copy(), base.copy()
            first[axis], second[axis] = banks[axis][1], banks[axis][2]
            grad.append(_tensor(first, b).reshape(-1, 3))
            lap += _tensor(second, b).reshape(-1, 3)
        return u, np.stack(grad, axis=2), lap

    def evaluate(self, b, points, derivatives=False):
        points = np.asarray(points, float)
        if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
            raise ValueError('points must be finite (N,3) coordinates')
        if np.any(abs(points) > np.pi+1e-12):
            raise ValueError('evaluation points must lie in [-pi,pi]^3; wrap periodic coordinates explicitly')
        banks = [[self._bank(points[:, axis], d) for d in range(3 if derivatives else 1)] for axis in range(3)]
        def contract(m):
            return np.einsum('ni,nj,nk,ijkc->nc', *m, b, optimize=True)
        base = [bank[0] for bank in banks]
        u = contract(base)
        if not derivatives:
            return u
        grad, lap = [], np.zeros_like(u)
        for axis in range(3):
            first, second = base.copy(), base.copy()
            first[axis], second[axis] = banks[axis][1], banks[axis][2]
            grad.append(contract(first))
            lap += contract(second)
        return dict(value=u, gradient=np.stack(grad, axis=2), laplacian=lap)

    @staticmethod
    def _force(forcing, t, points):
        if forcing is None:
            return np.zeros((len(points), 3))
        out = np.asarray(forcing(float(t), points), float)
        if out.shape != (len(points), 3) or not np.all(np.isfinite(out)):
            raise ValueError('forcing(t,points) must return finite (N,3) values')
        return out

    def rhs(self, t, b, forcing=None):
        u, grad, lap = self.fields(b)
        physical = -np.einsum('nj,nij->ni', u, grad)+self.viscosity*lap+self._force(forcing, t, self.points)
        if not np.all(np.isfinite(physical)):
            raise FloatingPointError('Nonfinite physical RHS during integration')
        return self.analyze(physical)

    def initial_diagnostics(self, initial, b=None, sampled_initial=None):
        """Check prescribed IC independently; never silently accept its projection."""
        raw = np.asarray(initial(self.points) if sampled_initial is None else sampled_initial, float)
        retained = self.analyze(raw, project=False)
        b = self.analyze(raw) if b is None else b
        unprojected_values = self._fft_fields(retained, False)
        projected_values = self._fft_fields(b, False)
        q = 4*self.cutoff+5
        axes = [-np.pi+2*np.pi*(np.arange(q)+shift)/q for shift in (.23, .41, .67)]
        points = np.array(np.meshgrid(*axes, indexing='ij')).reshape(3, -1).T
        prescribed = np.asarray(initial(points), float)
        if prescribed.shape != (len(points), 3) or not np.all(np.isfinite(prescribed)):
            raise ValueError('initial(points) must return finite (N,3) values on independent coordinates')
        represented = self.fields(b, axes, False)
        rms = lambda a: float(np.sqrt(np.mean(np.sum(a*a, axis=1))))
        source_norm = rms(prescribed)
        absolute = rms(represented-prescribed)
        training_scale = max(rms(raw), 1e-30)
        return dict(sample_points=len(points), rms_absolute_error=absolute,
                    relative_error=absolute/max(source_norm, 1e-30), source_rms=source_norm,
                    retained_mode_truncation_relative=rms(unprojected_values-raw)/training_scale,
                    retained_leray_projection_change_relative=rms(projected_values-unprojected_values)/training_scale,
                    scope='Independent shifted-grid prescribed IC versus actual constructed field; separate on-grid truncation and Leray changes are diagnostic, not continuum certificates')

    @staticmethod
    def _full_project(values, q):
        h = np.fft.fftn(values.reshape((q,)*3+(3,)), axes=(0, 1, 2))
        freq = np.fft.fftfreq(q, 1/q)
        wave = np.moveaxis(np.array(np.meshgrid(*([freq]*3), indexing='ij')), 0, -1)
        norm2 = np.sum(wave*wave, axis=-1)
        dot = np.sum(wave*h, axis=-1)
        pressure = wave*np.divide(dot, norm2, out=np.zeros_like(dot), where=norm2 > 0)[..., None]
        return np.fft.ifftn(h-pressure, axes=(0, 1, 2)).real.reshape(-1, 3)

    def forcing_diagnostics(self, forcing, final_time, count=16):
        """Independent temporal probes detect unresolved transient forcing.

        Full-grid Leray projection removes pressure gradients. These finite
        deterministic-jitter probes can miss narrower unsampled events; they
        are not an all-time bound or a time-integration error certificate.
        """
        if forcing is None:
            return dict(times=[], max_relative_error=0., max_absolute_rms=0., seconds=0.)
        began = time.perf_counter()
        q = 4*self.cutoff+5
        axes = [-np.pi+2*np.pi*(np.arange(q)+shift)/q for shift in (.29, .43, .71)]
        points = np.array(np.meshgrid(*axes, indexing='ij')).reshape(3, -1).T
        times = final_time*(np.arange(count)+np.random.default_rng(49271).uniform(.1, .9, count))/count
        rms = lambda a: float(np.sqrt(np.mean(np.sum(a*a, axis=1))))
        relative, absolute = [], []
        for t in times:
            raw = self._force(forcing, t, points)
            full = self._full_project(raw, q)
            retained = self.analyze(self._force(forcing, t, self.points))
            actual = self.fields(retained, axes, False)
            difference = rms(actual-full)
            absolute.append(difference)
            relative.append(difference/max(rms(raw), 1e-14))
        return dict(times=times.tolist(), grid=q,
                    max_relative_error=max(relative), max_absolute_rms=max(absolute),
                    seconds=time.perf_counter()-began,
                    scope='Jittered times and independent shifted spatial grid; compares solenoidal forcing after full diagnostic-grid pressure projection. Finite samples can miss narrow events.')

    def diagnostics(self, t, b, forcing=None, grid=None):
        """Independent shifted-grid projected strong residual and energy balance.

        Pressure is eliminated on the diagnostic grid using its full FFT Leray
        projector (not truncated to the evolution cutoff). This is a sampled
        physical residual, not a certificate for the continuum PDE.
        """
        q = int(grid or (4*self.cutoff+5))
        if q <= 3*self.cutoff:
            raise ValueError('diagnostic grid must exceed 3*cutoff')
        axes = [-np.pi+2*np.pi*(np.arange(q)+shift)/q for shift in (.19, .37, .61)]
        points = np.array(np.meshgrid(*axes, indexing='ij')).reshape(3, -1).T
        u, grad, lap = self.fields(b, axes)
        ut = self.fields(self.rhs(t, b, forcing), axes, False)
        force = self._force(forcing, t, points)
        convection = np.einsum('nj,nij->ni', u, grad)
        physical = -convection+self.viscosity*lap+force
        projected = self._full_project(physical, q)
        rms = lambda a: float(np.sqrt(np.mean(np.sum(a*a, axis=-1))))
        residual = rms(ut-projected)
        scale = max(rms(convection), rms(self.viscosity*lap), rms(force), 1e-14)
        energy = float(.5*np.mean(np.sum(u*u, axis=1)))
        dissipation = float(self.viscosity*np.mean(np.sum(grad*grad, axis=(1, 2))))
        power = float(np.mean(np.sum(u*force, axis=1)))
        energy_derivative = float(np.mean(np.sum(u*ut, axis=1)))
        return dict(grid=q, points=q**3, strong_residual_rms=residual,
                    strong_residual_relative=residual/scale, strong_residual_scale=scale,
                    residual_scope='Independent shifted grid; full-grid pressure projection, sampled physical residual, no certified error bound',
                    divergence_max=float(np.max(abs(np.trace(grad, axis1=1, axis2=2)))),
                    energy=energy, dissipation=dissipation, forcing_power=power,
                    energy_derivative=energy_derivative,
                    energy_balance_scale=max(dissipation, rms(u)*rms(force), 1e-14),
                    instantaneous_energy_balance=energy_derivative+dissipation-power)

    def solve(self, initial=taylor_green, forcing=None, final_time=1., dt=.005,
              residual_tolerance=1e-3, max_cfl=2., history_stride=10,
              verify_time=True, time_tolerance=1e-6,
              divergence_tolerance=1e-8, energy_tolerance=1e-8,
              initial_tolerance=1e-6, initial_atol=1e-10):
        if not np.isfinite(final_time) or final_time <= 0 or not np.isfinite(dt) or dt <= 0:
            raise ValueError('final_time and dt must be finite and positive')
        if residual_tolerance is not None and (not np.isfinite(residual_tolerance) or residual_tolerance <= 0):
            raise ValueError('residual_tolerance must be positive or None')
        if not np.isfinite(max_cfl) or max_cfl <= 0:
            raise ValueError('max_cfl must be finite and positive')
        if int(history_stride) != history_stride or history_stride < 1:
            raise ValueError('history_stride must be a positive integer')
        if any(not np.isfinite(tol) or tol <= 0 for tol in (time_tolerance, divergence_tolerance, energy_tolerance, initial_tolerance, initial_atol)):
            raise ValueError('time, divergence, energy, and initial tolerances must be finite and positive')
        began = time.perf_counter()
        sampled_initial = np.asarray(initial(self.points), float)
        b = self.analyze(sampled_initial)
        initial_check = self.initial_diagnostics(initial, b, sampled_initial)
        forcing_check = self.forcing_diagnostics(forcing, final_time)
        steps = int(np.ceil(final_time/dt))
        dt = final_time/steps
        status, t, history = 'completed', 0., []
        completed = 0
        diagnostic_history = []
        history_diagnostic_seconds = 0.
        # At least eight temporal intervals when the step count permits it;
        # every recorded state receives an independent physical-space check.
        diagnostic_stride = min(history_stride, max(1, steps//8))
        viscous_number = dt*self.viscosity*3*self.cutoff**2
        def snapshot():
            nonlocal history_diagnostic_seconds
            u, grad, _ = self.fields(b)
            curl = np.column_stack((grad[:, 2, 1]-grad[:, 1, 2],
                                    grad[:, 0, 2]-grad[:, 2, 0],
                                    grad[:, 1, 0]-grad[:, 0, 1]))
            coefficients = _complex_coefficients(b)
            modal_energy = np.sum(abs(coefficients)**2, axis=-1)
            tail_start = max(1, int(np.ceil(.75*self.cutoff)))
            tail = np.max(abs(self.wave), axis=0) >= tail_start
            check_start = time.perf_counter()
            physical = self.diagnostics(t, b, forcing)
            physical['energy_balance_relative'] = abs(physical['instantaneous_energy_balance'])/physical['energy_balance_scale']
            diagnostic_history.append(dict(time=t, **physical))
            history_diagnostic_seconds += time.perf_counter()-check_start
            return dict(time=t, energy=float(.5*np.mean(np.sum(u*u, axis=1))),
                        dissipation=float(self.viscosity*np.mean(np.sum(grad*grad, axis=(1, 2)))),
                        forcing_power=float(np.mean(np.sum(u*self._force(forcing, t, self.points), axis=1))),
                        cfl=float(dt*self.cutoff*np.max(np.sum(abs(u), axis=1))),
                        max_speed=float(np.max(np.linalg.norm(u, axis=1))),
                        max_vorticity=float(np.max(np.linalg.norm(curl, axis=1))),
                        strong_residual_relative=physical['strong_residual_relative'],
                        strong_residual_rms=physical['strong_residual_rms'],
                        divergence_max=physical['divergence_max'],
                        energy_balance_relative=physical['energy_balance_relative'],
                        physical_check_grid=physical['grid'],
                        high_frequency_energy_fraction=float(np.sum(modal_energy[tail])/max(float(np.sum(modal_energy)), 1e-30)),
                        high_frequency_start=tail_start)
        history.append(snapshot())
        if initial_check['relative_error'] > initial_tolerance and initial_check['rms_absolute_error'] > initial_atol:
            status = 'initial_underresolved'
        elif viscous_number > 2.5 or history[-1]['cfl'] > max_cfl:
            status = 'stability_guard'
        else:
            for n in range(steps):
                try:
                    with np.errstate(over='ignore', invalid='ignore'):
                        k1 = self.rhs(t, b, forcing)
                        k2 = self.rhs(t+dt/2, b+dt*k1/2, forcing)
                        k3 = self.rhs(t+dt/2, b+dt*k2/2, forcing)
                        k4 = self.rhs(t+dt, b+dt*k3, forcing)
                except FloatingPointError:
                    status = 'nonfinite_state'
                    break
                b += dt*(k1+2*k2+2*k3+k4)/6
                completed, t = n+1, (n+1)*dt
                if not np.all(np.isfinite(b)):
                    status = 'nonfinite_state'
                    break
                if completed % diagnostic_stride == 0 or completed == steps:
                    history.append(snapshot())
                    if history[-1]['cfl'] > max_cfl:
                        status = 'stability_guard'
                        break
        evolution_seconds = time.perf_counter()-began
        diagnostic_start = time.perf_counter()
        diagnostics = (None if status == 'nonfinite_state' else
                       diagnostic_history[-1] if diagnostic_history[-1]['time'] == t else self.diagnostics(t, b, forcing))
        if diagnostics is not None and diagnostic_history[-1]['time'] != t:
            diagnostics['energy_balance_relative'] = abs(diagnostics['instantaneous_energy_balance'])/diagnostics['energy_balance_scale']
            diagnostic_history.append(dict(time=t, **diagnostics))
        diagnostic_seconds = time.perf_counter()-diagnostic_start
        failures = []
        if residual_tolerance is not None and forcing_check['max_relative_error'] > residual_tolerance:
            failures.append('forcing_representation')
        if diagnostics is not None:
            energy_scale = diagnostics['energy_balance_scale']
            diagnostics['energy_balance_relative'] = abs(diagnostics['instantaneous_energy_balance'])/energy_scale
            if max(d['divergence_max'] for d in diagnostic_history) > divergence_tolerance:
                failures.append('divergence')
            if max(d['energy_balance_relative'] for d in diagnostic_history) > energy_tolerance:
                failures.append('energy_balance')
            if residual_tolerance is not None and max(d['strong_residual_relative'] for d in diagnostic_history) > residual_tolerance:
                failures.append('spatial_residual')
        time_difference, verification_seconds = None, 0.
        if completed == steps and verify_time:
            verification_start = time.perf_counter()
            finer = self.solve(initial, forcing, final_time, dt/2,
                               residual_tolerance=None, max_cfl=max_cfl,
                               history_stride=history_stride*2, verify_time=False,
                               time_tolerance=time_tolerance,
                               divergence_tolerance=divergence_tolerance,
                               energy_tolerance=energy_tolerance,
                               initial_tolerance=initial_tolerance, initial_atol=initial_atol)
            if finer.metrics['reached_final_time']:
                delta = self.fields(b-finer.coefficients, derivatives=False)
                fine_values = self.fields(finer.coefficients, derivatives=False)
                time_difference = float(np.linalg.norm(delta)/max(np.linalg.norm(fine_values), 1e-30))
                if time_difference > time_tolerance:
                    failures.append('time_refinement')
            else:
                failures.append('time_refinement_failed')
            verification_seconds = time.perf_counter()-verification_start
        if status == 'completed':
            if failures:
                status = 'underresolved' if set(failures) <= {'spatial_residual', 'time_refinement', 'forcing_representation'} else 'diagnostic_failure'
            else:
                status = 'converged_sampled' if verify_time and residual_tolerance is not None else 'completed_unverified'
        metrics = dict(status=status, reached_final_time=completed == steps, steps=completed,
                       requested_steps=steps, dt=dt, viscosity=self.viscosity,
                       viscous_stability_number=viscous_number,
                       evolution_seconds=evolution_seconds,
                       diagnostic_seconds=diagnostic_seconds,
                       history_diagnostic_seconds=history_diagnostic_seconds,
                       physical_diagnostic_times=[d['time'] for d in diagnostic_history],
                       max_sampled_strong_residual_relative=max(d['strong_residual_relative'] for d in diagnostic_history),
                       max_sampled_divergence=max(d['divergence_max'] for d in diagnostic_history),
                       max_sampled_energy_balance_relative=max(d['energy_balance_relative'] for d in diagnostic_history),
                       time_verification_seconds=verification_seconds,
                       time_refinement_relative=time_difference,
                       diagnostic_failures=failures,
                       initial_condition=initial_check,
                       forcing_representation=forcing_check,
                       initial_tolerance=initial_tolerance, initial_atol=initial_atol,
                       total_seconds=self.build['seconds']+time.perf_counter()-began,
                       residual_tolerance=residual_tolerance, diagnostics=diagnostics,
                       time_tolerance=time_tolerance, divergence_tolerance=divergence_tolerance,
                       energy_tolerance=energy_tolerance,
                       convergence_scope='Independent prescribed IC check, jittered-time forcing representation checks, successful integration, optional dt/2 comparison, physical residual/divergence/energy checks at recorded intermediate times and final time; finite space-time samples can miss narrow events and do not certify continuum accuracy or global 3D regularity')
        return NSResult(self, b, t, metrics, history)


@dataclass
class NSRefinementResult:
    solution: NSResult | None
    status: str
    attempts: list
    metrics: dict

    def evaluate(self, points, derivatives=False):
        if self.solution is None:
            raise RuntimeError('No resolution fit the requested resource limits')
        return self.solution.evaluate(points, derivatives)


def solve_ns_refined(initial=taylor_green, forcing=None, viscosity=.05,
                     final_time=1., dt=.005, cutoffs=(3, 5, 7, 9, 11, 13),
                     residual_tolerance=1e-3, field_tolerance=1e-4,
                     time_tolerance=1e-6, n_centers=257,
                     initial_tolerance=1e-6, initial_atol=1e-10,
                     max_seconds=120., max_estimated_bytes=800_000_000):
    """Cold rerun refinement, requiring physical and successive-field checks.

    Every attempt starts from the prescribed IC: an earlier final state is not
    used as a replacement IC. The memory limit is a conservative allocation
    estimate, not an operating-system RSS guarantee. Wall time is checked
    between attempts, so the last authorized attempt can exceed the budget.
    """
    from scipy.stats import qmc
    if not cutoffs or any(int(k) != k or k < 1 for k in cutoffs) or any(b <= a for a,b in zip(cutoffs, cutoffs[1:])):
        raise ValueError('cutoffs must be a nonempty strictly increasing sequence of positive integers')
    if any(not np.isfinite(x) or x <= 0 for x in (field_tolerance, max_seconds, max_estimated_bytes)):
        raise ValueError('field tolerance and resource budgets must be positive and finite')
    if residual_tolerance is None:
        raise ValueError('adaptive refinement requires a physical residual tolerance')
    began = time.perf_counter()
    points = 2*np.pi*(qmc.Sobol(3, scramble=True, seed=13719).random_base2(8)-.5)
    previous, result, attempts, status = None, None, [], 'resolution_exhausted'
    for cutoff in cutoffs:
        estimated = 1600*(4*cutoff+5)**3+32*(2*cutoff+1)**3
        if time.perf_counter()-began >= max_seconds or estimated > max_estimated_bytes:
            status = 'resource_exhausted'
            break
        start = time.perf_counter()
        model = PeriodicNS(cutoff, 3*cutoff+3, viscosity, n_centers=n_centers)
        result = model.solve(initial, forcing, final_time, dt,
                             residual_tolerance=residual_tolerance,
                             time_tolerance=time_tolerance, verify_time=True,
                             initial_tolerance=initial_tolerance, initial_atol=initial_atol)
        values = result.evaluate(points)
        difference = None if previous is None else float(np.linalg.norm(values-previous)/max(np.linalg.norm(values), 1e-30))
        attempts.append(dict(cutoff=cutoff, grid=model.grid, build=model.build,
                             metrics=result.metrics, field_refinement_relative=difference,
                             estimated_workspace_bytes=estimated,
                             attempt_seconds=time.perf_counter()-start))
        if result.status == 'converged_sampled' and difference is not None and difference <= field_tolerance:
            status = 'converged_sampled'
            break
        if result.status == 'initial_underresolved':
            previous = None
            continue
        if not result.metrics['reached_final_time'] or result.status == 'diagnostic_failure':
            status = result.status
            break
        previous = values
    return NSRefinementResult(result, status, attempts,
                              dict(seconds=time.perf_counter()-began,
                                   field_tolerance=field_tolerance,
                                   residual_tolerance=residual_tolerance,
                                   time_tolerance=time_tolerance,
                                   convergence_scope='dt/2, final sampled physical residual and change between cold resolution reruns; no continuum error certificate',
                                   sample_points=len(points)))
