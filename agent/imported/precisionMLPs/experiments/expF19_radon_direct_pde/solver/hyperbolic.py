"""Conservative Burgers backend: Rusanov flux and SSPRK3 cell-average evolution.

This is a conventional finite-volume weak solver, not global smooth QUILL
collocation. Its piecewise-constant state is the derivative of a ReLU primitive.
"""
from __future__ import annotations
import time
import numpy as np


def reconstruct(x, edges, values):
    """Evaluate the finite-volume piecewise-constant reconstruction."""
    j = np.searchsorted(edges, np.asarray(x), side="right")-1
    return values[np.clip(j, 0, len(values)-1)]


def relu_primitive(edges, values):
    """Return ReLU knots/readouts for the exact integral of the cell state."""
    return dict(knots=np.asarray(edges), weights=np.r_[values[0], np.diff(values), -values[-1]])


def solve_burgers(initial, x_bounds=(-1., 1.), cells=256, t_end=.5,
                  boundary="periodic", cfl=.4, save_times=None):
    if cells < 4 or t_end < 0 or not 0 < cfl <= .5:
        raise ValueError("Need cells>=4, t_end>=0 and 0<cfl<=0.5")
    if boundary not in ("periodic", "outflow"):
        raise ValueError("boundary must be periodic or outflow")
    edges = np.linspace(*x_bounds, cells+1)
    h = edges[1]-edges[0]
    if h <= 0: raise ValueError("x_bounds must increase")
    x = (edges[:-1]+edges[1:])/2
    u = np.asarray(initial(x) if callable(initial) else initial, dtype=float).copy()
    if u.shape != (cells,) or not np.all(np.isfinite(u)):
        raise ValueError("Initial cell values must be a finite vector of length cells")
    targets = np.unique(np.r_[0., t_end] if save_times is None else np.r_[0., save_times, t_end])
    if targets[0] < 0 or targets[-1] > t_end: raise ValueError("Invalid save times")
    mass0 = float(h*np.sum(u))
    entropy0 = float(.5*h*np.sum(u*u))
    flux_budget = entropy_budget = 0.
    rhs_calls = steps = 0
    snapshots, records = [], []
    started = time.perf_counter()

    def rhs(v):
        nonlocal rhs_calls
        rhs_calls += 1
        pad = np.r_[v[-1], v, v[0]] if boundary == "periodic" else np.r_[v[0], v, v[-1]]
        left, right = pad[:-1], pad[1:]
        speed = np.maximum(abs(left), abs(right))
        flux = .25*(left*left+right*right)-.5*speed*(right-left)
        entropy_flux = (left**3+right**3)/6-.25*speed*(right*right-left*left)
        return -(flux[1:]-flux[:-1])/h, flux[0]-flux[-1], entropy_flux[0]-entropy_flux[-1]

    t = 0.
    for target in targets:
        while t < target-4*np.finfo(float).eps:
            dt = min(target-t, cfl*h/max(float(np.max(abs(u))), 1e-14))
            a, fa, qa = rhs(u)
            u1 = u+dt*a
            b, fb, qb = rhs(u1)
            u2 = .75*u+.25*(u1+dt*b)
            c, fc, qc = rhs(u2)
            u = u/3+2*(u2+dt*c)/3
            flux_budget += dt*(fa+fb+4*fc)/6
            entropy_budget += dt*(qa+qb+4*qc)/6
            t += dt
            steps += 1
            if not np.all(np.isfinite(u)):
                raise FloatingPointError("Finite-volume trajectory became nonfinite")
        snapshots.append(u.copy())
        records.append(dict(t=float(target), mass=float(h*np.sum(u)), entropy=float(.5*h*np.sum(u*u)),
                            mass_balance_residual=float(h*np.sum(u)-mass0-flux_budget),
                            entropy_balance_defect=float(.5*h*np.sum(u*u)-entropy0-entropy_budget)))
    metrics = dict(backend="finite_volume_rusanov_ssprk3", steps=steps, rhs_calls=rhs_calls,
                   seconds=time.perf_counter()-started, cells=cells, h=float(h), boundary=boundary,
                   mass_initial=mass0, mass_final=float(h*np.sum(u)),
                   mass_balance_residual=records[-1]["mass_balance_residual"],
                   entropy_initial=entropy0, entropy_final=float(.5*h*np.sum(u*u)),
                   entropy_balance_defect=records[-1]["entropy_balance_defect"],
                   coefficient_count=cells, representation="cell values; derivative of finite ReLU primitive")
    return dict(x=x, edges=edges, values=u, time=float(t_end), times=targets,
                solution=np.array(snapshots), history=records, status="reached_t_end", metrics=metrics,
                neural_primitive=relu_primitive(edges, u))
