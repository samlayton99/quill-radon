"""Small native QUILL Navier--Stokes evolution, with explicit Fourier closure.

State a=E b consists of QUILL readouts. Every stage evaluates the actual tanh
field and its derivatives before Fourier analysis of the PDE right-hand side:
    b' = A_K F(S E b),   a' = E A_K F(S a).
E is prescribed by quill_boundary.encode on entire sine/cosine basis profiles.
Cached S E, grad(S E), lap(S E) exploit fixed geometry; no target solution,
training, interpolation solve, or least-squares solve constructs the dynamics.
This is also explicitly a modified pseudospectral method in coordinates b.
"""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_name] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/native-ns-mpl')
import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.stats import qmc
from quill_boundary import encode

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/checkpoint_F_applications/expF19_radon_direct_pde/native_dynamics'


def relative(a, b):
    return float(np.linalg.norm(a-b)/np.linalg.norm(b))


def canonical(k):
    nz = np.flatnonzero(k)
    return bool(len(nz) and k[nz[0]] > 0)


def initial(points):
    x, y, z = points.T
    return np.column_stack((np.sin(x)*np.cos(y)*np.cos(z),
                            -np.cos(x)*np.sin(y)*np.cos(z), np.zeros(len(x))))


class Coordinates:
    def __init__(self, cutoff, grid, viscosity):
        self.cutoff, self.grid, self.viscosity = cutoff, grid, viscosity
        assert grid > 3*cutoff, 'Quadratic spectral product needs grid > 3K'
        allk = np.array(np.meshgrid(*([np.arange(-cutoff, cutoff+1)]*3), indexing='ij')).reshape(3, -1).T
        self.k = np.array([k for k in allk if canonical(k)])
        self.index = tuple((self.k % grid).T)
        self.minus = tuple((-self.k % grid).T)
        self.phase = (-1.)**np.sum(self.k, axis=1)
        self.norm = np.linalg.norm(self.k, axis=1)
        self.directions = np.repeat(self.k/self.norm[:, None], 2, axis=0)
        self.count = 1 + 2*len(self.k)
        axis = -np.pi + 2*np.pi*np.arange(grid)/grid
        self.points = np.array(np.meshgrid(axis, axis, axis, indexing='ij')).reshape(3, -1).T

    def analyze(self, fields, project=True):
        h = np.fft.fftn(fields.reshape(self.grid, self.grid, self.grid, 3), axes=(0, 1, 2))/self.grid**3
        c = h[self.index]*self.phase[:, None]
        if project:
            c -= self.k*np.sum(self.k*c, axis=1)[:, None]/self.norm[:, None]**2
        b = np.empty((self.count, 3))
        b[0] = h[0, 0, 0].real
        b[1::2], b[2::2] = 2*c.real, -2*c.imag
        return b

    def synthesize_spectral(self, b, derivatives=False):
        h = np.zeros((self.grid, self.grid, self.grid, 3), complex)
        c = (b[1::2]-1j*b[2::2])/2
        h[self.index] = c*self.phase[:, None]
        h[self.minus] = c.conj()*self.phase[:, None]
        h[0, 0, 0] = b[0]
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

    def spectral_rhs(self, b):
        u, grad, lap = self.synthesize_spectral(b, True)
        rhs = -np.einsum('nj,nij->ni', u, grad) + self.viscosity*lap
        return self.analyze(rhs)

    def heldout_spectral(self, b, points):
        phase = points@self.k.T
        return b[0] + np.cos(phase)@b[1::2] + np.sin(phase)@b[2::2]


class NeuralCoordinates(Coordinates):
    def __init__(self, cutoff, grid, viscosity, n_interior=193, lam=.2):
        start = time.perf_counter()
        super().__init__(cutoff, grid, viscosity)
        self.groups = []
        grouped = {}
        for index, k in enumerate(self.k):
            primitive = tuple(k//np.gcd.reduce(np.abs(k)))
            grouped.setdefault(primitive, []).append(index)
        self.value = np.ones((len(self.points), self.count))
        self.first = np.zeros_like(self.value)
        self.second = np.zeros_like(self.value)
        worst_value = worst_first = worst_second = 0.
        total_neurons = 0
        for primitive, indices in grouped.items():
            indices = np.array(indices)
            v = np.array(primitive, float)
            v /= np.linalg.norm(v)
            rho = self.norm[indices]
            def profile(z, rho=rho):
                angle = np.asarray(z)[..., None]*rho
                return np.stack((np.cos(angle), np.sin(angle)), axis=-1).reshape(np.shape(z)+(2*len(rho),))
            halfwidth = np.pi*np.sum(np.abs(v))
            enc = encode(profile, n_interior-1, lam=lam, interval=(-halfwidth, halfwidth),
                         halo=int(np.ceil(np.sqrt(n_interior))))
            columns = np.column_stack((1+2*indices, 2+2*indices)).ravel()
            s = self.points@v
            for derivative, destination in [(0, self.value), (1, self.first), (2, self.second)]:
                destination[:, columns] = enc.evaluate(s, derivative)
            theta = s[:, None]*rho
            exact = np.stack((np.cos(theta), np.sin(theta)), axis=-1).reshape(len(s), -1)
            first = np.stack((-rho*np.sin(theta), rho*np.cos(theta)), axis=-1).reshape(len(s), -1)
            second = np.stack((-rho*rho*np.cos(theta), -rho*rho*np.sin(theta)), axis=-1).reshape(len(s), -1)
            worst_value = max(worst_value, float(np.max(abs(self.value[:, columns]-exact))))
            worst_first = max(worst_first, float(np.max(abs(self.first[:, columns]-first))))
            worst_second = max(worst_second, float(np.max(abs(self.second[:, columns]-second))))
            self.groups.append((v, enc, columns))
            total_neurons += len(enc.centers)
        self.build = dict(seconds=time.perf_counter()-start, neurons=total_neurons,
                          directions=len(self.groups), latent_scalar_columns=self.count,
                          n_interior=n_interior, lam=lam,
                          cached_bytes=self.value.nbytes+self.first.nbytes+self.second.nbytes,
                          worst_basis_value_error=worst_value,
                          worst_basis_directional_derivative_error=worst_first,
                          worst_basis_laplacian_error=worst_second)

    def fields(self, b):
        u = self.value@b
        grad = np.stack([self.first[:, 1:]@(b[1:]*self.directions[:, d, None]) for d in range(3)], axis=2)
        return u, grad, self.second@b

    def rhs(self, b):
        u, grad, lap = self.fields(b)
        physical = -np.einsum('nj,nij->ni', u, grad) + self.viscosity*lap
        return self.analyze(physical)

    def direct(self, b, points, derivatives=False):
        """Evaluate collapsed actual tanh readouts, independent of cached S E."""
        u = np.broadcast_to(b[0], (len(points), 3)).copy()
        grad = np.zeros((len(points), 3, 3))
        lap = np.zeros_like(u)
        for v, enc, columns in self.groups:
            weight = enc.weights@b[columns]
            bias = enc.bias@b[columns]
            s = points@v
            z = enc.gamma*(s[:, None]-enc.centers)
            t = np.tanh(z)
            u += t@weight + bias
            if derivatives:
                rr = np.exp(-2*np.abs(z))
                sech2 = 4*rr/(1+rr)**2
                first = enc.gamma*sech2
                grad += (first@weight)[:, :, None]*v
                lap += (-2*enc.gamma*t*first)@weight
        return (u, grad, lap) if derivatives else u

    def export(self, b):
        """Ordinary fixed-geometry tanh readouts, without latent Fourier state."""
        return dict(directions=np.array([v for v, _, _ in self.groups]),
                    centers=np.array([enc.centers for _, enc, _ in self.groups]),
                    gamma=np.array([enc.gamma for _, enc, _ in self.groups]),
                    weights=np.array([enc.weights@b[columns] for _, enc, columns in self.groups]),
                    bias=b[0]+sum(enc.bias@b[columns] for _, enc, columns in self.groups))


def integrate(rhs, start, end, dt):
    began = time.perf_counter()
    steps = int(np.ceil(end/dt))
    dt = end/steps
    b = start.copy()
    for _ in range(steps):
        k1 = rhs(b)
        k2 = rhs(b+dt*k1/2)
        k3 = rhs(b+dt*k2/2)
        k4 = rhs(b+dt*k3)
        b += dt*(k1+2*k2+2*k3+k4)/6
    return b, dict(seconds=time.perf_counter()-began, dt=dt, steps=steps, rhs_calls=4*steps)


def run(out, end=.05, dt=.002, viscosity=.05):
    out.mkdir(parents=True, exist_ok=True)
    points = 2*np.pi*(qmc.Sobol(3, scramble=True, seed=1941).random_base2(8)-.5)
    # Fresh cube faces supplement interior evaluations.
    rng = np.random.default_rng(124)
    faces = rng.uniform(-np.pi, np.pi, (48, 3))
    for i in range(48):
        faces[i, (i//8)%3] = np.pi*(-1 if (i//4)%2 else 1)
    points = np.vstack((points, faces))
    records = []
    models = []
    # Both native runs finish before conventional comparison solves are started.
    for cutoff in (2, 3):
        model = NeuralCoordinates(cutoff, 12, viscosity)
        b0 = model.analyze(initial(model.points))
        initial_error = relative(model.direct(b0, points), initial(points))
        b, timing = integrate(model.rhs, b0, end, dt)
        u, grad, lap = model.fields(b)
        direct, dgrad, dlap = model.direct(b, model.points[::37], True)
        record = dict(cutoff=cutoff, grid=model.grid, build=model.build,
                      native=timing, initial_relative_error=initial_error,
                      direct_vs_cache_relative=relative(direct, u[::37]),
                      direct_gradient_vs_cache_relative=relative(dgrad, grad[::37]),
                      direct_laplacian_vs_cache_relative=relative(dlap, lap[::37]),
                      divergence_max=float(np.max(abs(np.trace(grad, axis1=1, axis2=2)))),
                      initial_energy=float(.5*np.mean(np.sum(model.value@b0*(model.value@b0), axis=1))),
                      final_energy=float(.5*np.mean(np.sum(u*u, axis=1))),
                      newly_generated_mode_norm=float(np.linalg.norm((b-b0)[np.linalg.norm(b0, axis=1)<1e-13])),
                      velocity_change_relative=relative(u, model.value@b0),
                      generated_vertical_velocity_rms=float(np.sqrt(np.mean(u[:, 2]**2))))
        print(json.dumps(dict(native_completed=record)), flush=True)
        records.append(record)
        models.append((model, b0, b))

    # References are verification only: no native stage consumes their states.
    reference = Coordinates(7, 24, viscosity)
    ref0 = reference.analyze(initial(reference.points))
    bref, reftime = integrate(reference.spectral_rhs, ref0, end, dt/2)
    finer = Coordinates(9, 30, viscosity)
    bfine, fine_time = integrate(finer.spectral_rhs, finer.analyze(initial(finer.points)), end, dt/2)
    ref_values = finer.heldout_spectral(bfine, points)
    reference_difference = relative(reference.heldout_spectral(bref, points), ref_values)

    for record, (model, b0, b) in zip(records, models):
        ordinary, ordinary_time = integrate(model.spectral_rhs, b0, end, dt)
        refined, refined_time = integrate(model.rhs, b0, end, dt/2)
        actual = model.direct(b, points)
        conventional = model.heldout_spectral(ordinary, points)
        record.update(same_cutoff_spectral=ordinary_time,
                      native_time_refinement=refined_time,
                      native_vs_same_cutoff_spectral_relative=relative(actual, conventional),
                      native_vs_dt_half_relative=relative(actual, model.direct(refined, points)),
                      native_vs_cutoff9_relative=relative(actual, ref_values),
                      spectral_same_cutoff_vs_cutoff9_relative=relative(conventional, ref_values),
                      native_evolution_cost_ratio=record['native']['seconds']/ordinary_time['seconds'])
        np.savez_compressed(out/f'ns_native_K{model.cutoff}.npz', coefficients=b, initial_coefficients=b0,
                            k=model.k, evaluation_points=points, native_velocity=actual,
                            same_cutoff_spectral_velocity=conventional, reference_velocity=ref_values)
        np.savez_compressed(out/f'ns_native_network_K{model.cutoff}.npz', **model.export(b))
        print(json.dumps(dict(comparison=record)), flush=True)

    metadata = dict(method='a = E b; b_t = A_K F(S E b); explicit QUILL E, neural derivatives, Fourier/Leray closure',
                    interpretation='Modified pseudospectral method in compressed coordinates; no external PDE solution is used by native dynamics',
                    equation='3D periodic incompressible Navier-Stokes',
                    initial_condition='(sin(x) cos(y) cos(z), -cos(x) sin(y) cos(z), 0)',
                    domain='[-pi,pi]^3', viscosity=viscosity, final_time=end,
                    validation_points=len(points), references_created_after_native_runs=True,
                    reference=dict(cutoff7=reftime, cutoff9=fine_time, cutoff7_vs9_relative=reference_difference),
                    limitations=['Short smooth flow only; low spatial cutoff',
                                 'No speed or high-dimensional advantage claimed',
                                 'Finite tanh fields are only approximately periodic; Fourier closure uses their periodic samples',
                                 'Dealiasing is exact for retained trigonometric products, approximate for QUILL fields with tiny non-Fourier tails'],
                    rows=records)
    (out/'ns_native_metrics.json').write_text(json.dumps(metadata, indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), constrained_layout=True)
    k = [r['cutoff'] for r in records]
    axes[0].semilogy(k, [r['native_vs_cutoff9_relative'] for r in records], 'o-', label='Total vs K=9')
    axes[0].semilogy(k, [r['native_vs_same_cutoff_spectral_relative'] for r in records], 's-', label='Neural vs same K')
    axes[0].semilogy(k, [r['native_vs_dt_half_relative'] for r in records], '^-', label='dt vs dt/2')
    axes[0].set(title='Separate numerical errors', xlabel='Fourier cutoff K', ylabel='Relative velocity error')
    axes[0].legend(fontsize=8)
    axes[1].bar(np.array(k)-.18, [r['native']['seconds'] for r in records], .36, label='Native evolution')
    axes[1].bar(np.array(k)+.18, [r['same_cutoff_spectral']['seconds'] for r in records], .36, label='Spectral evolution')
    axes[1].set(title='Evolution cost (one CPU thread)', xlabel='Fourier cutoff K', ylabel='Seconds')
    axes[1].legend(fontsize=8)
    axes[2].bar(k, [r['build']['seconds'] for r in records])
    axes[2].set(title='One-time QUILL/caching cost', xlabel='Fourier cutoff K', ylabel='Seconds')
    for ax in axes:
        ax.set_xticks(k)
        ax.grid(alpha=.2)
    fig.suptitle('Native QUILL: interacting 3D Taylor–Green flow, T=.05')
    fig.savefig(out/'ns_native_comparison.png', dpi=180)
    plt.close(fig)
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    run(args.out)
