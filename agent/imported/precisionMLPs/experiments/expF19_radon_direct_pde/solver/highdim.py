"""Explicit ridge IVPs and a bounded-dimensional nonlinear scaling control.

The linear IVP uses only declared initial ridge profiles and a constant PDE
symbol. The nonlinear control is conventional dealiased Fourier Galerkin
evolution, followed by explicit QUILL encoding; it is NOT native neural
evolution. These distinct methods must not be conflated in reports.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import itertools
import math
import sys
import time

import numpy as np
from scipy.fft import fftn, ifftn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quill_boundary import encode


@dataclass
class RidgeSolution:
    directions: np.ndarray
    centers: np.ndarray  # (directions, centers)
    gamma: np.ndarray
    weights: np.ndarray
    bias: float
    metadata: dict
    time_weights: np.ndarray | None = None
    time_bias: float = 0.

    def evaluate(self, x, derivative_order=2):
        """Actual tanh evaluation; returns values and requested derivatives."""
        x = np.atleast_2d(np.asarray(x, dtype=float))
        value = np.full(len(x), self.bias)
        gradient = np.zeros_like(x) if derivative_order >= 1 else None
        laplacian = np.zeros(len(x)) if derivative_order >= 2 else None
        dt = np.full(len(x), self.time_bias) if self.time_weights is not None else None
        limits = self.metadata.get("projection_limits")
        for i in range(len(self.directions)):
            v, g = self.directions[i], self.gamma[i]
            projected = x @ v
            if limits is not None and np.any(abs(projected) > limits[i]*(1+1e-12)):
                raise ValueError("Point is outside the explicitly encoded ridge interval")
            z = g * (projected[:, None] - self.centers[i])
            t = np.tanh(z)
            value += t @ self.weights[i]
            if dt is not None:
                dt += t @ self.time_weights[i]
            if derivative_order:
                r = np.exp(-2*np.abs(z)); s = 4*r/(1+r)**2
                first = (g*s) @ self.weights[i]
                gradient += first[:, None]*v
                if derivative_order >= 2:
                    laplacian += (-2*g*g*t*s) @ self.weights[i]
        out = {"value": value}
        if gradient is not None:
            out["gradient"] = gradient
        if laplacian is not None:
            out["laplacian"] = laplacian
        if dt is not None:
            out["time_derivative"] = dt
        return out

    @property
    def neurons(self):
        return int(self.weights.size)

    @property
    def storage_bytes(self):
        arrays = [self.directions, self.centers, self.gamma, self.weights]
        if self.time_weights is not None:
            arrays.append(self.time_weights)
        return sum(a.nbytes for a in arrays)


def solve_ridge_linear(profiles, directions, t_end, diffusivity,
                       velocity=None, max_frequency=12, n_interior=129,
                       lam=.2, period=2., quadrature_points=None):
    """Solve u_t + b.grad(u) = nu*Delta(u) on R^d from ridge ICs.

    Initial data is explicitly supplied as sum_j profiles[j](v_j.x), with
    UNIT v_j and period-periodic profiles. Structure is supplied, not learned.
    Trapezoidal analysis of INITIAL DATA supplies Fourier amplitudes. The PDE
    supplies their exact heat/advection multiplier; no interior solution labels.
    Returned spatial network has time-dependent readouts, not fixed time input.
    """
    start = time.perf_counter()
    directions = np.asarray(directions, dtype=float)
    if len(profiles) != len(directions) or directions.ndim != 2:
        raise ValueError("Need one profile per direction")
    if not np.allclose(np.linalg.norm(directions, axis=1), 1., atol=1e-12):
        raise ValueError("Directions must be unit vectors")
    if diffusivity < 0 or t_end < 0:
        raise ValueError("This forward solver requires nonnegative nu and time")
    velocity = np.zeros(directions.shape[1]) if velocity is None else np.asarray(velocity)
    k = np.arange(-max_frequency, max_frequency+1)
    omega = 2*np.pi*k/period
    q = quadrature_points or (8*max_frequency+8)
    samples = period*np.arange(q)/q-period/2
    analysis = np.exp(-1j*omega[:, None]*samples)/q
    centers, gammas, weights, time_weights = [], [], [], []
    bias = time_bias = 0.
    for f, v in zip(profiles, directions):
        initial = analysis @ np.asarray(f(samples))
        symbol = -diffusivity*omega**2 - 1j*(velocity @ v)*omega
        evolved = initial*np.exp(t_end*symbol)
        # Real-valued IC quadrature supplies conjugate coefficients. Taking real
        # on the real axis is implicit; keep the analytic continuation entire.
        def profile(z):
            waves = np.exp(1j*np.asarray(z)[..., None]*omega)
            return np.stack([waves @ evolved, waves @ (symbol*evolved)], axis=-1)
        enc = encode(profile, n_interior-1, lam,
                     interval=(-period/2, period/2),
                     halo=math.ceil(math.sqrt(n_interior)))
        centers.append(enc.centers); gammas.append(enc.gamma)
        weights.append(enc.weights[:, 0]); time_weights.append(enc.weights[:, 1])
        bias += float(enc.bias[0]); time_bias += float(enc.bias[1])
    metadata = dict(method="initial-profile quadrature plus exact constant-PDE semigroup",
                    ambient_dimension=int(directions.shape[1]),
                    supplied_directions=len(directions), max_frequency=max_frequency,
                    initial_quadrature_points=q, n_interior=n_interior,
                    halo_per_side=math.ceil(math.sqrt(n_interior)), lam=lam,
                    t_end=t_end, diffusivity=diffusivity,
                    setup_seconds=time.perf_counter()-start,
                    structure="provided ridge decomposition; not inferred or learned",
                    projection_limits=[period/2]*len(directions),
                    domain="whole space; evaluation projection must stay inside encoded interval")
    return RidgeSolution(directions, np.asarray(centers), np.asarray(gammas),
                         np.asarray(weights), bias, metadata,
                         np.asarray(time_weights), time_bias)


class PeriodicReactionDiffusion:
    """Conventional FFT Galerkin control u_t=nu*Delta(u)+u^2 on T^d.

    Retains |k_j|<=K, uses Q>3K for quadratic dealiasing. Initial function is
    called only at t=0. No neural/LS claim is made for this evolution routine.
    """
    def __init__(self, dimension, cutoff, diffusivity=.05, memory_limit_mb=800):
        self.dimension, self.cutoff = int(dimension), int(cutoff)
        self.diffusivity = float(diffusivity)
        self.q = 2*math.ceil((3*self.cutoff+1)/2)
        self.shape = (self.q,)*self.dimension
        self.points = self.q**self.dimension
        self.estimated_workspace_bytes = 16*self.points*14
        if self.estimated_workspace_bytes > memory_limit_mb*1e6:
            raise MemoryError("Requested Galerkin workspace exceeds explicit cap")
        # fftfreq*Q can return 3.0000000000000004. Round BEFORE the cutoff
        # comparison; otherwise a K=3 calculation silently becomes K=2.
        k1 = np.rint(np.fft.fftfreq(self.q)*self.q).astype(int)
        self.k1 = k1
        self.k2 = np.zeros(self.shape)
        self.mask = np.ones(self.shape, bool)
        for j in range(self.dimension):
            sl = [1]*self.dimension; sl[j] = self.q
            kj = k1.reshape(sl)
            self.k2 += kj*kj
            self.mask &= abs(kj) <= self.cutoff

    def rhs(self, coefficients):
        physical = ifftn(coefficients, workers=1).real*self.points
        nonlinear = fftn(physical*physical, workers=1)/self.points
        return (nonlinear-self.diffusivity*self.k2*coefficients)*self.mask

    def initial(self, initial_function):
        axis = 2*np.pi*np.arange(self.q)/self.q-np.pi
        coordinates = np.meshgrid(*([axis]*self.dimension), indexing="ij", sparse=True)
        # Fourier coefficients use x=2*pi*j/Q; shift the IC evaluation itself
        # to this coordinate convention, not coefficients after the fact.
        coordinates = [a+np.pi for a in coordinates]
        physical = np.broadcast_to(initial_function(coordinates), self.shape)
        return fftn(physical, workers=1)/self.points*self.mask

    def solve(self, initial_function, t_end=.1, dt=.01):
        started = time.perf_counter()
        state = self.initial(initial_function)
        steps = math.ceil(t_end/dt); dt = t_end/steps
        for _ in range(steps):
            a = self.rhs(state); b = self.rhs(state+dt*a/2)
            c = self.rhs(state+dt*b/2); d = self.rhs(state+dt*c)
            state += dt/6*(a+2*b+2*c+d)
        return state, dict(dimension=self.dimension, cutoff=self.cutoff,
                           quadrature_side=self.q, quadrature_points=self.points,
                           retained_box_modes=(2*self.cutoff+1)**self.dimension,
                           workspace_estimate_mb=self.estimated_workspace_bytes/1e6,
                           state_mb=state.nbytes/1e6, steps=steps, dt=dt,
                           seconds=time.perf_counter()-started,
                           method="classical dealiased Fourier Galerkin, explicit RK4",
                           t_end=t_end, diffusivity=self.diffusivity)

    def sparse_coefficients(self, state, threshold=0.):
        index = np.argwhere(self.mask & (abs(state) > threshold))
        frequencies = self.k1[index].astype(int)
        coefficients = state[tuple(index.T)]
        return frequencies, coefficients


def evaluate_fourier(x, frequencies, coefficients, derivatives=False, chunk=256):
    x = np.asarray(x)
    value = np.zeros(len(x), complex)
    grad = np.zeros_like(x, dtype=complex)
    lap = np.zeros(len(x), complex)
    for start in range(0, len(frequencies), chunk):
        k = frequencies[start:start+chunk]; c = coefficients[start:start+chunk]
        wave = np.exp(1j*x @ k.T)*c
        value += wave.sum(axis=1)
        if derivatives:
            grad += wave @ (1j*k)
            lap -= wave @ np.sum(k*k, axis=1)
    return dict(value=value.real, gradient=grad.real, laplacian=lap.real)


@lru_cache(maxsize=16)
def _encoding_bank(max_integer_frequency, n_interior, lam):
    modes = np.arange(1, max_integer_frequency+1)*np.pi
    def profile(z):
        z = np.asarray(z)[..., None]*modes
        return np.concatenate([np.cos(z), np.sin(z)], axis=-1)
    return encode(profile, n_interior-1, lam, interval=(-1., 1.),
                  halo=math.ceil(math.sqrt(n_interior)))


def encode_fourier_ridges(frequencies, coefficients, n_interior=257, lam=.2,
                          coefficient_threshold=1e-11):
    """Compile a real periodic Fourier field into corrected tanh ridges.

    The encoder NEVER solves the PDE. Explicit filtering bound is the discarded
    Fourier l1 mass. Standard cube [-pi,pi]^d; per-direction projected intervals
    use pi*||v||_1. Shared normalized trig bank makes this practical in 5D.
    """
    start = time.perf_counter()
    frequencies = np.asarray(frequencies, dtype=int)
    coefficients = np.asarray(coefficients)
    d = frequencies.shape[1]
    mean = float(coefficients[np.all(frequencies == 0, axis=1)].real.sum())
    groups = {}
    dropped_l1 = 0.
    for k, c in zip(frequencies, coefficients):
        nonzero = np.flatnonzero(k)
        if not len(nonzero):
            continue
        if abs(c) < coefficient_threshold:
            dropped_l1 += abs(c)
            continue
        if k[nonzero[0]] < 0:
            continue
        g = np.gcd.reduce(abs(k)); p = tuple(k//g)
        groups.setdefault(p, []).append((int(g), c))
    if not groups:
        return RidgeSolution(np.empty((0, d)), np.empty((0, 0)), np.empty(0),
                             np.empty((0, 0)), mean, dict(dropped_l1=dropped_l1))
    maxfreq = int(max(g*sum(abs(i) for i in p) for p, modes in groups.items() for g, _ in modes))
    bank = _encoding_bank(maxfreq, n_interior, lam)
    directions, centers, gammas, weights = [], [], [], []
    bias = mean
    for p, modes in groups.items():
        p = np.asarray(p); norm = np.linalg.norm(p); v = p/norm
        length = np.pi*np.sum(abs(p))/norm
        trig = np.zeros(2*maxfreq)
        for g, c in modes:
            i = g*np.sum(abs(p))-1
            trig[i] += 2*c.real
            trig[maxfreq+i] -= 2*c.imag
        directions.append(v); centers.append(length*bank.centers)
        gammas.append(bank.gamma/length); weights.append(bank.weights@trig)
        bias += bank.bias@trig
    metadata = dict(method="explicit Fourier-to-QUILL encoding AFTER PDE evolution",
                    directions=len(groups), n_interior=n_interior,
                    halo_per_side=math.ceil(math.sqrt(n_interior)), lam=lam,
                    max_normalized_integer_frequency=maxfreq,
                    discarded_fourier_l1_bound=float(dropped_l1),
                    coefficient_threshold=coefficient_threshold,
                    projection_limits=[float(np.pi*np.sum(abs(v))) for v in directions],
                    setup_seconds=time.perf_counter()-start,
                    domain="[-pi,pi]^d; direction-dependent projected intervals")
    return RidgeSolution(np.array(directions), np.array(centers), np.array(gammas),
                         np.array(weights), float(bias), metadata)


def interaction_support_count(dimension, degree):
    """Number of integer modes with l1 norm <= degree, exact combinatorics.

    Starting from constant + all coordinate cosines, repeated multiplication
    generates this support. Counts possible modes, not a claim all are large.
    """
    return sum(2**j*math.comb(dimension, j)*math.comb(degree, j)
               for j in range(min(dimension, degree)+1))
