"""Stiff reaction-diffusion in actual QUILL coordinates, with validity events.

The production backend uses BDF and small implicit state solves. No neural
readout solve or solution labels enter. A separate classical Strang method is
available as an independent validation reference, with no QUILL claim.
"""
from __future__ import annotations
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_key] = "1"
import time
import numpy as np
from scipy.integrate import solve_ivp
from native_tanh import construct, features


def solve_reaction_diffusion(initial, x_bounds=(-np.pi, np.pi), cells=512, t_end=.2,
                              diffusivity=.1, reaction="allen_cahn", reaction_rate=100.,
                              modes=32, centers=None, max_step=.01, rtol=1e-8, atol=1e-10,
                              amplitude_limit=1e3, tail_limit=.05, save_times=None,
                              invariant_tolerance=None):
    if reaction not in ("quadratic", "allen_cahn"):
        raise ValueError("reaction must be quadratic or allen_cahn")
    if diffusivity < 0 or reaction_rate <= 0 or t_end <= 0 or amplitude_limit <= 0:
        raise ValueError("Invalid physical parameters or validity threshold")
    if invariant_tolerance is not None and invariant_tolerance < 0:
        raise ValueError("invariant_tolerance must be nonnegative or None")
    a, b = x_bounds
    if b <= a: raise ValueError("x_bounds must increase")
    started = time.perf_counter()
    model = construct(modes, 16*modes+1 if centers is None else centers)
    model.pop("theta0")
    scale = 2*np.pi/(b-a)
    quadrature_x = (model["x"]+np.pi)/scale+a
    samples = np.asarray(initial(quadrature_x) if callable(initial) else initial, dtype=float)
    if samples.shape != quadrature_x.shape or not np.all(np.isfinite(samples)):
        raise ValueError("initial must supply one finite value per quadrature node")
    A = model["A"]
    U = model["phi"]@model["E"]
    Uxx = scale**2*(model["dxx"]@model["E"])
    b0 = A@samples
    x = a+(b-a)*np.arange(cells)/cells
    z = (x-a)*scale-np.pi
    basis = features(z, model["enc"])[0]@model["E"]
    lower_bound = min(-1., float(np.min(samples))) if reaction == "allen_cahn" else (0. if np.min(samples) >= 0 else -np.inf)
    upper_bound = max(1., float(np.max(samples))) if reaction == "allen_cahn" else np.inf
    tail = np.r_[np.arange(1+modes*3//4, 1+modes), np.arange(1+modes+modes*3//4, 1+2*modes)]

    def tail_ratio(state):
        return float(np.linalg.norm(state[tail])/max(np.linalg.norm(state), 1e-30))

    def rhs(t, state):
        u = U@state
        nonlinear = reaction_rate*(u*u if reaction == "quadratic" else u-u**3)
        return A@(diffusivity*(Uxx@state)+nonlinear)

    def jac(t, state):
        u = U@state
        diagonal = reaction_rate*(2*u if reaction == "quadratic" else 1-3*u*u)
        return A@(diffusivity*Uxx+diagonal[:, None]*U)

    def amplitude_event(t, state):
        return amplitude_limit-float(np.max(abs(U@state)))

    def resolution_event(t, state):
        return (np.inf if tail_limit is None else tail_limit-tail_ratio(state))

    def invariant_event(t, state):
        if invariant_tolerance is None: return np.inf
        values = np.r_[basis@state, U@state]
        return min(float(np.min(values))-lower_bound, upper_bound-float(np.max(values)))+invariant_tolerance

    amplitude_event.terminal = resolution_event.terminal = invariant_event.terminal = True
    amplitude_event.direction = resolution_event.direction = invariant_event.direction = -1
    if amplitude_event(0., b0) <= 0:
        raise ValueError("Initial state exceeds amplitude_limit")
    if resolution_event(0., b0) <= 0:
        raise ValueError("Initial state exceeds tail_limit; refine modes")
    setup_seconds = time.perf_counter()-started
    solve_started = time.perf_counter()
    result = solve_ivp(rhs, (0., t_end), b0, method="BDF", jac=jac, rtol=rtol, atol=atol,
                       max_step=max_step, events=(amplitude_event, resolution_event, invariant_event), dense_output=True)
    solve_seconds = time.perf_counter()-solve_started
    status = "reached_t_end" if result.success else "integrator_failure"
    if len(result.t_events[0]): status = "amplitude_limit"
    if len(result.t_events[1]): status = "underresolved"
    if len(result.t_events[2]): status = "invariant_violation"
    final_t = float(result.t[-1])
    final_state = result.y[:, -1]
    if not np.all(np.isfinite(final_state)):
        raise FloatingPointError("Integrator returned a nonfinite state")
    output_times = np.linspace(0., final_t, 51) if save_times is None else np.asarray(save_times)
    output_times = np.unique(np.r_[0., output_times[(output_times >= 0)&(output_times <= final_t)], final_t])
    states = result.sol(output_times)
    solution = (basis@states).T
    values = basis@final_state
    all_values = np.r_[values, U@final_state]
    history = [dict(t=float(t), max_abs=float(np.max(abs(U@state))),
                    mean=float(np.mean(U@state)), tail_ratio=tail_ratio(state))
               for t, state in zip(result.t, result.y.T)]
    metrics = dict(backend="actual_quill_reduced_bdf", modes=modes, reduced_coordinates=2*modes+1,
                   neurons=len(model["enc"].centers), quadrature_points=model["Q"],
                   setup_seconds=setup_seconds, seconds=solve_seconds,
                   nfev=result.nfev, njev=result.njev, implicit_factorizations=result.nlu,
                   accepted_steps=len(result.t)-1, final_time=final_t,
                   max_abs=float(np.max(abs(values))), quadrature_max_abs=float(np.max(abs(U@final_state))),
                   minimum_value=float(np.min(values)),
                   maximum_principle_excess=(float(max(0., np.max(abs(values))-max(1., np.max(abs(samples)))))
                                             if reaction == "allen_cahn" else None),
                   positive_initial_data=bool(np.min(samples) > 0),
                   invariant_tolerance=invariant_tolerance, invariant_monitor_points=cells+model["Q"],
                   invariant_sampled_violation=float(max(0., lower_bound-np.min(all_values),
                                                          np.max(all_values)-upper_bound)),
                   final_tail_ratio=tail_ratio(final_state), amplitude_limit=amplitude_limit, tail_limit=tail_limit,
                   rtol=rtol, atol=atol, max_step=max_step, diffusivity=diffusivity,
                   reaction=reaction, reaction_rate=reaction_rate,
                   integrator_message=result.message,
                   representation="theta=E b; actual tanh feature and derivative matrices; BDF state solves, no readout fit")
    return dict(x=x, values=values, time=final_t, times=output_times, solution=solution,
                history=history, status=status, metrics=metrics, coefficients=final_state,
                network_weights=model["E"]@final_state)


def classical_split_reference(initial, cells=2048, t_end=.2, diffusivity=.1,
                               reaction="allen_cahn", reaction_rate=100., max_step=.0001):
    """Independent periodic finite-difference heat + exact reaction Strang split."""
    x = -np.pi+2*np.pi*np.arange(cells)/cells
    h = 2*np.pi/cells
    u = np.asarray(initial(x), dtype=float).copy()
    steps = int(np.ceil(t_end/max_step))
    dt = t_end/steps
    k = np.fft.fftfreq(cells, 1/cells)
    heat = np.exp(-2*diffusivity*dt*np.sin(np.pi*k/cells)**2/h**2)
    # heat factor is a HALF-step for the discrete Laplacian -4 sin²/h².
    for _ in range(steps):
        u = np.fft.ifft(heat*np.fft.fft(u)).real
        if reaction == "allen_cahn":
            u = u/np.sqrt(u*u+(1-u*u)*np.exp(-2*reaction_rate*dt))
        elif reaction == "quadratic":
            denominator = 1-reaction_rate*dt*u
            if np.min(denominator) <= 0: raise FloatingPointError("Reference reaction pole")
            u = u/denominator
        else:
            raise ValueError("Unknown reference reaction")
        u = np.fft.ifft(heat*np.fft.fft(u)).real
    return dict(x=x, values=u, time=t_end, steps=steps, dt=dt,
                backend="classical_fd_heat_exact_reaction_strang")
