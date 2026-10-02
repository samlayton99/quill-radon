"""Native periodic spline-kernel Burgers evolution; no learned/readout solve.

The evolving state is the kernel coefficient vector a. Every RK stage evaluates
u, u_x, u_xx from that same vector and maps the PDE RHS back with a fixed,
explicit geometric convolution. There is no FFT or linear solve in NativeBurgers.
The separate Fourier solver below is a cost baseline and post-run error meter.

This is classical cubic-spline collocation with an exact ReLU^3 realization.
It is NOT the tanh QUILL boundary construction or a new numerical method.
"""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_key] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/codex-native-burgers-mpl")

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time
from unittest.mock import patch

import numpy as np
from scipy.ndimage import convolve1d

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/checkpoint_F_applications/expF19_radon_direct_pde/native_dynamics"


def initial_condition(x):
    return np.sin(x) + .2 * np.cos(2 * x)


def relative_l2(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def beta3(s, derivative=0):
    """Centered cardinal cubic B-spline and its first two exact derivatives."""
    q = np.abs(np.asarray(s))
    if derivative == 0:
        return np.where(q < 1, (4 - 6*q*q + 3*q**3)/6,
                        np.where(q < 2, (2-q)**3/6, 0.))
    if derivative == 1:
        return np.sign(s) * np.where(q < 1, -2*q + 1.5*q*q,
                                    np.where(q < 2, -.5*(2-q)**2, 0.))
    if derivative == 2:
        return np.where(q < 1, -2 + 3*q, np.where(q < 2, 2-q, 0.))
    raise ValueError("Only derivatives 0, 1, 2 are supported")


class NativeBurgers:
    """Coefficient ODE for a periodic cubic-kernel representation."""

    def __init__(self, n, nu=.05, inverse_radius=28):
        if n < 8:
            raise ValueError("At least eight periodic centers are required")
        self.n = n
        self.nu = nu
        self.h = 2*np.pi/n
        self.x = -np.pi + self.h*np.arange(n)
        self.inverse_radius = inverse_radius
        self.rho = -(2 - np.sqrt(3.))
        offsets = np.arange(-inverse_radius, inverse_radius+1)
        self.inverse_weights = np.sqrt(3.) * self.rho**np.abs(offsets)
        self.inverse_tail_l1_bound = float(
            2*np.sqrt(3.)*abs(self.rho)**(inverse_radius+1)/(1-abs(self.rho)))
        self.initial_coefficients = self.analyze(initial_condition(self.x))

    def analyze(self, values):
        """Fixed local convolution, not a target-dependent inverse/solve."""
        return convolve1d(values, self.inverse_weights, mode="wrap")

    def node_fields(self, a):
        previous = np.roll(a, 1)
        following = np.roll(a, -1)
        return ((previous + 4*a + following)/6,
                (following-previous)/(2*self.h),
                (previous-2*a+following)/self.h**2)

    def rhs(self, a):
        u, ux, uxx = self.node_fields(a)
        return self.analyze(-u*ux + self.nu*uxx)

    def evaluate(self, a, x, derivative=0):
        """Actual periodic kernel expansion, exploiting its four active terms."""
        q = (np.asarray(x)+np.pi)/self.h
        left = np.floor(q).astype(np.int64)
        ans = np.zeros_like(q, dtype=float)
        for offset in (-1, 0, 1, 2):
            j = left + offset
            ans += a[j % self.n]*beta3(q-j, derivative)
        return ans / self.h**derivative

    def dense_relu3(self, a, x):
        """Independent literal finite ReLU^3 realization on [-pi, pi].

        This audit evaluator is numerically worse than the compact beta3 kernel:
        large positive cubic terms cancel. Production uses the equivalent stable
        compact formula. It is not used in initialization or evolution.
        """
        q = (np.asarray(x)+np.pi)/self.h
        result = np.zeros_like(q)
        for j in range(-1, self.n+2):
            s = q-j
            kernel = sum(w*np.maximum(s+k, 0.)**3
                         for k, w in zip((2, 1, 0, -1, -2), (1, -4, 6, -4, 1)))/6
            result += a[j % self.n]*kernel
        return result

    def run(self, end=.5, dt=.001, history=True):
        steps = int(np.ceil(end/dt))
        dt = end/steps
        a = self.initial_coefficients.copy()
        records = []
        started = time.perf_counter()
        for step in range(steps):
            k1 = self.rhs(a)
            k2 = self.rhs(a+.5*dt*k1)
            k3 = self.rhs(a+.5*dt*k2)
            k4 = self.rhs(a+dt*k3)
            a += dt*(k1+2*k2+2*k3+k4)/6
            if history and (step % max(1, steps//20) == 0 or step == steps-1):
                u = self.node_fields(a)[0]
                records.append(dict(t=(step+1)*dt, mean=float(np.mean(u)),
                                    energy=float(.5*np.mean(u*u)), max_abs=float(np.max(abs(u)))))
        if not np.all(np.isfinite(a)):
            raise FloatingPointError("Native coefficient trajectory became nonfinite")
        return a, dict(steps=steps, dt=dt, evolution_seconds=time.perf_counter()-started,
                       history=records)


def fft_baseline(n, end=.5, nu=.05, dt=.001):
    """Independent classical pseudospectral solver, used only as a baseline.

    The quadratic term is dealiased by padding to 2N before multiplying, then
    truncating back to the N-mode state. The Nyquist coefficient is removed.
    """
    x = -np.pi + 2*np.pi*np.arange(n)/n
    k = np.fft.fftfreq(n, 1/n)
    integer_k = k.astype(int)
    keep = np.abs(k) < n/2
    state = np.fft.fft(initial_condition(x))/n
    state *= keep

    def rhs(c):
        padded = np.zeros(2*n, dtype=complex)
        padded[integer_k % (2*n)] = c
        u = np.fft.ifft(padded*(2*n)).real
        nonlinear = np.fft.fft(u*u)/(2*n)
        return (-.5j*k*nonlinear[integer_k % (2*n)]-nu*k*k*c)*keep

    steps = int(np.ceil(end/dt))
    dt = end/steps
    started = time.perf_counter()
    for _ in range(steps):
        k1 = rhs(state)
        k2 = rhs(state+.5*dt*k1)
        k3 = rhs(state+.5*dt*k2)
        k4 = rhs(state+dt*k3)
        state += dt*(k1+2*k2+2*k3+k4)/6
    return state, dict(steps=steps, dt=dt, evolution_seconds=time.perf_counter()-started)


def eval_fourier(c, x):
    k = np.fft.fftfreq(len(c), 1/len(c))
    return np.real(np.exp(1j*(np.asarray(x)+np.pi)[:, None]*k[None, :])@c)


def eval_fourier_uniform(c, point_count, offset=0.):
    """FFT evaluation on a uniform shifted grid, for an honest fast baseline."""
    if point_count < len(c):
        raise ValueError("This evaluation helper only upsamples")
    k = np.fft.fftfreq(len(c), 1/len(c)).astype(int)
    padded = np.zeros(point_count, dtype=complex)
    padded[k % point_count] = c*np.exp(1j*k*offset)
    return np.fft.ifft(padded*point_count).real


def timed(call, repeats=3):
    values = []
    answer = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        answer = call()
        values.append(time.perf_counter()-t0)
    return answer, dict(median_seconds=float(np.median(values)), samples_seconds=values)


def validate():
    """Focused checks of the representation, PDE mapping and no-solve contract."""
    sim = NativeBurgers(64)
    rng = np.random.default_rng(1901)
    a = rng.normal(size=64)
    values = sim.node_fields(a)[0]
    inverse_error = float(np.max(abs(sim.analyze(values)-a)))
    probe = rng.uniform(-np.pi, np.pi, 200)
    node_error = max(float(np.max(abs(sim.evaluate(a, sim.x, d)-sim.node_fields(a)[d])))
                     for d in range(3))
    eps = 1e-5
    first_fd = (sim.evaluate(a, probe+eps)-sim.evaluate(a, probe-eps))/(2*eps)
    second_fd = (sim.evaluate(a, probe+eps, 1)-sim.evaluate(a, probe-eps, 1))/(2*eps)
    derivative_error = max(relative_l2(first_fd, sim.evaluate(a, probe, 1)),
                           relative_l2(second_fd, sim.evaluate(a, probe, 2)))
    smooth_a = sim.initial_coefficients
    relu_error = float(np.max(abs(sim.dense_relu3(smooth_a, probe)-sim.evaluate(smooth_a, probe))))
    node_rhs = sim.node_fields(sim.rhs(smooth_a))[0]
    u, ux, uxx = sim.node_fields(smooth_a)
    rhs_error = float(np.max(abs(node_rhs+u*ux-sim.nu*uxx)))
    period_error = float(np.max(abs(sim.evaluate(a, probe+2*np.pi)-sim.evaluate(a, probe))))
    def prohibited(*args, **kwargs):
        raise AssertionError("Native path called a prohibited FFT or linear solve")
    names = [(np.fft, key) for key in ("fft", "ifft", "rfft", "irfft", "fftn", "ifftn")]
    names += [(np.linalg, key) for key in ("solve", "inv", "pinv", "lstsq", "svd")]
    from contextlib import ExitStack
    with ExitStack() as stack:
        for module, name in names:
            stack.enter_context(patch.object(module, name, prohibited))
        trial = NativeBurgers(32)
        result, _ = trial.run(end=.01, dt=.002)
        trial.evaluate(result, probe)
    assert inverse_error < 5e-14
    assert node_error < 2e-10
    assert derivative_error < 2e-6
    assert relu_error < 2e-9
    assert rhs_error < 5e-14
    assert period_error < 2e-13
    return dict(inverse_stencil_roundtrip_linf=inverse_error, node_derivative_formula_linf=node_error,
                analytic_derivatives_vs_finite_difference_rel_l2=derivative_error,
                literal_relu3_vs_compact_kernel_linf=relu_error,
                coefficient_rhs_vs_physical_rhs_linf=rhs_error,
                periodicity_linf=period_error, native_path_with_fft_and_linalg_disabled="passed",
                inverse_tail_l1_bound=sim.inverse_tail_l1_bound)


def plot_results(out, rows, x, snapshots, reference, temporal, ablation):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullLocator
    plt.rcParams.update({"font.size":10, "axes.spines.top":False, "axes.spines.right":False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ax = axes[0, 0]
    ax.plot(x, initial_condition(x), color=".65", label="Prescribed initial condition")
    ax.plot(x, reference, color="black", label="Independent Fourier reference")
    ax.plot(x[::30], snapshots[64][::30], "o", ms=3, mfc="none", label="Native cubic kernel, N=64")
    ax.set(xlabel="x", ylabel="u(x, 0.5)")
    ax = axes[0, 1]
    for n in (32, 64, 128, 256):
        ax.plot(x, snapshots[n]-reference, label=f"N={n}")
    ax.set(xlabel="x", ylabel="Native kernel error at t=0.5")
    ax = axes[1, 0]
    ax.loglog([r["n"] for r in rows], [r["solution_rel_l2"] for r in rows], "o-", label="Native cubic kernel")
    ax.loglog([r["n"] for r in rows], [r["fft_solution_rel_l2"] for r in rows], "s--", label="Fourier baseline")
    anchor = rows[0]["solution_rel_l2"]
    ax.loglog([r["n"] for r in rows], [anchor*(rows[0]["n"]/r["n"])**2 for r in rows], ":", color=".5", label="Second-order guide")
    ax.set(xlabel="Number of spatial coefficients", ylabel="Relative L2 error", ylim=(1e-14, 1e-2))
    ax.set_xticks([r["n"] for r in rows], [str(r["n"]) for r in rows])
    ax.xaxis.set_minor_locator(NullLocator())
    ax = axes[1, 1]
    ax.semilogy([1000*r["native_total_timing"]["median_seconds"] for r in rows],
              [r["solution_rel_l2"] for r in rows], "o-", label="Native cubic kernel")
    ax.semilogy([1000*r["fft_total_timing"]["median_seconds"] for r in rows],
              [r["fft_solution_rel_l2"] for r in rows], "s--", label="Fourier baseline")
    ax.set(xlabel="Setup + evolution wall time (ms; median of 3)", ylabel="Relative L2 error", ylim=(1e-14, 1e-2))
    for ax in axes.ravel():
        ax.grid(alpha=.2)
        ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.02), ncol=1 if ax is axes[0, 0] else 2, fontsize=8)
    fig.subplots_adjust(hspace=.7, wspace=.28, top=.84, bottom=.07)
    fig.savefig(out/"burgers_native_summary.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    for row in rows:
        history = row["trajectory"]["history"]
        axes[0].plot([v["t"] for v in history], [v["energy"] for v in history], label=f"N={row['n']}")
    axes[0].set(xlabel="Time", ylabel="Mean kinetic energy")
    axes[1].loglog([r["dt"] for r in temporal], [r["difference_from_finest_rel_l2"] for r in temporal], "o-")
    axes[1].set(xlabel="RK4 time step", ylabel="Difference from dt=0.0025 trajectory", title="Fixed N=64; temporal refinement")
    axes[1].set_xticks([r["dt"] for r in temporal], [f"{r['dt']:.3g}" for r in temporal])
    axes[1].xaxis.set_minor_locator(NullLocator())
    axes[2].semilogy([r["radius"] for r in ablation], [r["solution_rel_l2"] for r in ablation], "o-")
    axes[2].set(xlabel="Explicit inverse-stencil radius", ylabel="Solution relative L2 error", title="Fixed N=64; deconvolution ablation")
    axes[0].legend(loc="lower center", bbox_to_anchor=(.5, 1.02), ncol=4, fontsize=8)
    for ax in axes: ax.grid(alpha=.2)
    fig.subplots_adjust(top=.8, bottom=.15, wspace=.28)
    fig.savefig(out/"burgers_native_checks.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    checks = validate()
    if args.validate_only:
        print(json.dumps(checks, indent=2))
        return
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    x = np.linspace(-np.pi, np.pi, 2048, endpoint=False)+.137*(2*np.pi/2048)
    snapshots = {}
    rows = []
    # All native solutions are generated before the reference is constructed.
    for n in (16, 32, 64, 128, 256):
        dt = min(.001, .12*(2*np.pi/n)**2/.05)
        def native_run():
            sim = NativeBurgers(n)
            a, meta = sim.run(dt=dt)
            return sim, a, meta
        (sim, a, meta), native_time = timed(native_run)
        values = sim.evaluate(a, x)
        snapshots[n] = values
        residual = sim.evaluate(sim.rhs(a), x)+values*sim.evaluate(a, x, 1)-sim.nu*sim.evaluate(a, x, 2)
        _, inference_time = timed(lambda: sim.evaluate(a, x), repeats=10)
        rows.append(dict(n=n, h=sim.h, dt=meta["dt"], native_total_timing=native_time,
                         trajectory=meta, offgrid_residual_rms=float(np.sqrt(np.mean(residual**2))),
                         offgrid_initial_rel_l2=relative_l2(sim.evaluate(sim.initial_coefficients, x), initial_condition(x)),
                         kernel_inference_timing=inference_time, inverse_radius=sim.inverse_radius,
                         coefficient_storage_bytes=a.nbytes, inverse_stencil_bytes=sim.inverse_weights.nbytes,
                         mean_drift=float(abs(np.mean(sim.node_fields(a)[0])-np.mean(initial_condition(sim.x))))))
        np.savez_compressed(args.out/f"burgers_native_n{n}.npz", coefficients=a, n=n, nu=sim.nu,
                            time=.5, inverse_radius=sim.inverse_radius)
        print(json.dumps({"native_complete":n, "seconds":native_time["median_seconds"]}), flush=True)
    ref_a, ref_meta = fft_baseline(512, dt=.0001)
    ref_b, _ = fft_baseline(768, dt=.00005)
    truth = eval_fourier(ref_b, x)
    reference_difference = relative_l2(eval_fourier(ref_a, x), truth)
    for row in rows:
        n = row["n"]
        (c, fft_meta), fft_time = timed(lambda: fft_baseline(n, dt=row["dt"]))
        _, direct_fft_eval = timed(lambda: eval_fourier(c, x), repeats=5)
        fast_values, fast_fft_eval = timed(lambda: eval_fourier_uniform(c, len(x), x[0]+np.pi), repeats=10)
        uniform_check = float(np.max(abs(fast_values-eval_fourier(c, x))))
        assert uniform_check < 2e-14
        row.update(solution_rel_l2=relative_l2(snapshots[n], truth),
                   solution_linf=float(np.max(abs(snapshots[n]-truth))),
                   fft_solution_rel_l2=relative_l2(eval_fourier(c, x), truth),
                   fft_total_timing=fft_time, fft_trajectory=fft_meta,
                   fft_direct_inference_timing=direct_fft_eval,
                   fft_uniform_inference_timing=fast_fft_eval,
                   fft_uniform_vs_direct_linf=uniform_check)
    sim = NativeBurgers(64)
    temporal = []
    fine, _ = sim.run(dt=.0025, history=False)
    fine_values = sim.evaluate(fine, x)
    for dt in (.04, .02, .01, .005):
        a, meta = sim.run(dt=dt, history=False)
        temporal.append(dict(dt=meta["dt"], difference_from_finest_rel_l2=relative_l2(sim.evaluate(a, x), fine_values)))
    # The deliberately truncated inverse is a mechanism ablation, not a fit.
    ablation = []
    for radius in (0, 2, 6, 12, 28):
        sim = NativeBurgers(64, inverse_radius=radius)
        a, _ = sim.run(dt=.001)
        ablation.append(dict(radius=radius, inverse_tail_bound=sim.inverse_tail_l1_bound,
                             solution_rel_l2=relative_l2(sim.evaluate(a, x), truth)))
    output = dict(protocol="Native cubic B-spline coefficient evolution, explicit fixed geometric convolution, RK4. No FFT/solve/solution labels in native path; Fourier baseline is verification only.",
                  equation="u_t + u u_x = nu u_xx", initial_condition="sin(x) + 0.2 cos(2x)",
                  domain="periodic [-pi, pi]", nu=.05, end_time=.5, checks=checks,
                  reference=dict(n=768, dt=.00005, refinement_rel_l2=reference_difference,
                                 coarse_n=512, coarse_dt=ref_meta["dt"]),
                  rows=rows, temporal_refinement=temporal, inverse_radius_ablation=ablation,
                  environment=dict(python=platform.python_version(), numpy=np.__version__,
                                   platform=platform.platform(), threads=1),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  elapsed_seconds=time.perf_counter()-started)
    (args.out/"burgers_native_metrics.json").write_text(json.dumps(output, indent=2)+"\n")
    with (args.out/"burgers_native_metrics.jsonl").open("w") as stream:
        for row in rows: stream.write(json.dumps(row)+"\n")
    np.savez_compressed(args.out/"burgers_native_predictions.npz", x=x, reference=truth,
                        **{f"native_{n}":v for n,v in snapshots.items()})
    plot_results(args.out, rows, x, snapshots, truth, temporal, ablation)
    print(json.dumps(dict(reference_refinement=reference_difference,
                         rows=[{k:r[k] for k in ("n", "solution_rel_l2", "fft_solution_rel_l2", "offgrid_residual_rms")} for r in rows],
                         temporal_refinement=temporal, ablation=ablation), indent=2))


if __name__ == "__main__":
    main()
