"""Sparse-data inversion of three physical parameters with actual QUILL dynamics.

Only the 24 scalar observations enter fitting. The independent FFT generator
and withheld dense fields are outside the inverse solver. scipy least_squares
fits (nu, alpha, beta), never neural readouts. QUILL geometry is fixed.
"""
from __future__ import annotations
import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/codex-native-inverse-mpl")

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import time
import numpy as np
from scipy.optimize import least_squares
from native_tanh import construct, features
from native_burgers import fft_baseline, eval_fourier

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/checkpoint_F_applications/expF19_radon_direct_pde/capabilities"


@dataclass(frozen=True)
class Sensors:
    x: np.ndarray
    times: np.ndarray
    values: np.ndarray


class QuillForward:
    """Compressed exact image(E) dynamics of the actual tanh network.

    E maps reduced coordinates b to the neural readouts theta. U=Phi E and
    its derivative counterparts are products of actual tanh feature matrices,
    not substituted trigonometric functions. Only associativity is used.
    """

    def __init__(self, k=32, n=513, dt=.002):
        started = time.perf_counter()
        self.model = construct(k, n)
        m = self.model
        # The shared factory also returns its own example IC. It is never used
        # here: initialization is parameterized exclusively by (alpha, beta).
        m.pop("theta0")
        self.A = m["A"]
        self.U = m["phi"]@m["E"]
        self.Ux = m["dx"]@m["E"]
        self.Uxx = m["dxx"]@m["E"]
        self.ic_basis = m["A"]@np.column_stack([np.sin(m["x"]), np.cos(2*m["x"])])
        self.dt = dt
        self.calls = 0
        self.rhs_calls = 0
        self.setup_seconds = time.perf_counter()-started

    def evaluate_basis(self, x):
        return features(x, self.model["enc"])[0]@self.model["E"]

    def integrate(self, parameters, times, sensitivities=True):
        """Return b and, optionally, exact tangent RK4 sensitivities at each time."""
        nu, alpha, beta = parameters
        times = np.asarray(times)
        if np.any(times < 0) or np.any(np.diff(times) < 0):
            raise ValueError("Output times must be nonnegative and sorted")
        self.calls += 1
        state = np.zeros((self.ic_basis.shape[0], 4 if sensitivities else 1))
        state[:, 0] = self.ic_basis@np.array([alpha, beta])
        if sensitivities:
            state[:, 2:] = self.ic_basis

        def rhs(z):
            self.rhs_calls += 1
            u, ux, uxx = self.U@z, self.Ux@z, self.Uxx@z
            g = -u[:, :1]*ux + nu*uxx
            if sensitivities:
                g[:, 1:] -= u[:, 1:]*ux[:, :1]
                g[:, 1] += uxx[:, 0]
            return self.A@g

        outputs = []
        current = 0.
        for target in times:
            if target > current:
                count = int(np.ceil((target-current)/self.dt))
                step = (target-current)/count
                for _ in range(count):
                    k1 = rhs(state)
                    k2 = rhs(state+.5*step*k1)
                    k3 = rhs(state+.5*step*k2)
                    k4 = rhs(state+step*k3)
                    state += step*(k1+2*k2+2*k3+k4)/6
            outputs.append(state.copy())
            current = target
        answer = np.stack(outputs)
        if not np.all(np.isfinite(answer)):
            raise FloatingPointError("Nonfinite QUILL forward/tangent trajectory")
        return answer

    def predict(self, parameters, x, times, sensitivities=False):
        states = self.integrate(parameters, times, sensitivities)
        return np.einsum("xd,tdp->txp", self.evaluate_basis(x), states)


def fit_parameters(forward, sensors, initial=(.09, .8, .05)):
    """Fit exactly three physical parameters; no reference field is accepted."""
    started = time.perf_counter()
    basis = forward.evaluate_basis(sensors.x)
    cache = {}
    calls_before, rhs_before = forward.calls, forward.rhs_calls

    def compute(p):
        if "p" not in cache or not np.array_equal(p, cache["p"]):
            state = forward.integrate(p, sensors.times, sensitivities=True)
            fields = np.einsum("xd,tdp->txp", basis, state)
            cache.update(p=p.copy(), residual=(fields[:, :, 0]-sensors.values).ravel(),
                         jacobian=fields[:, :, 1:].reshape(-1, 3))
        return cache

    result = least_squares(lambda p: compute(p)["residual"], np.array(initial),
                           jac=lambda p: compute(p)["jacobian"],
                           bounds=([.005, .4, -.5], [.2, 1.6, .7]),
                           x_scale=[.05, 1., .2], ftol=1e-11, xtol=1e-11, gtol=1e-11,
                           max_nfev=40)
    elapsed = time.perf_counter()-started
    singular_values = np.linalg.svd(result.jac, compute_uv=False)
    # This SVD has only three columns: it diagnoses physical identifiability.
    return result.x, dict(parameters=result.x.tolist(), fit_seconds=elapsed,
                         forward_tangent_solves=forward.calls-calls_before,
                         rhs_calls=forward.rhs_calls-rhs_before,
                         optimizer_nfev=result.nfev, optimizer_njev=result.njev,
                         optimizer_success=bool(result.success), optimizer_status=result.message,
                         sensor_residual_rms=float(np.sqrt(np.mean(result.fun**2))),
                         physical_jacobian_shape=list(result.jac.shape),
                         physical_jacobian_singular_values=singular_values.tolist(),
                         physical_jacobian_rank=int(np.sum(singular_values > singular_values[0]*1e-10)))


def verify_tangents(forward, sensors):
    p = np.array([.065, .9, .1])
    analytic = forward.predict(p, sensors.x, sensors.times, True)[:, :, 1:]
    errors = []
    for j, step in enumerate((1e-5, 1e-5, 1e-5)):
        direction = np.zeros(3)
        direction[j] = step
        plus = forward.predict(p+direction, sensors.x, sensors.times)[:, :, 0]
        minus = forward.predict(p-direction, sensors.x, sensors.times)[:, :, 0]
        fd = (plus-minus)/(2*step)
        errors.append(float(np.linalg.norm(fd-analytic[:, :, j])/np.linalg.norm(analytic[:, :, j])))
    assert max(errors) < 2e-7
    # Verify compression is solely reassociation of the full neural RHS.
    rng = np.random.default_rng(834)
    b = rng.normal(size=forward.ic_basis.shape[0])*.01
    m = forward.model
    theta = m["E"]@b
    full = m["A"]@(-(m["phi"]@theta)*(m["dx"]@theta)+p[0]*(m["dxx"]@theta))
    compressed = forward.A@(-(forward.U@b)*(forward.Ux@b)+p[0]*(forward.Uxx@b))
    reassociation = float(np.linalg.norm(full-compressed)/np.linalg.norm(full))
    assert reassociation < 2e-12
    return dict(tangent_relative_errors=errors, compressed_vs_full_neural_rhs_relative_error=reassociation)


def independent_data():
    """Generate synthetic measurements and separately retained withheld truth."""
    sensors_x = -np.pi+2*np.pi*(np.arange(8)+.37)/8
    times = np.array([.15, .3, .5])
    future_times = np.array([.15, .3, .5, .7])
    dense_x = np.linspace(-np.pi, np.pi, 1024, endpoint=False)+.137*2*np.pi/1024
    states = []
    for t in future_times:
        state, _ = fft_baseline(384, end=float(t), nu=.05, dt=.0001)
        states.append(state)
    observations = np.array([eval_fourier(state, sensors_x) for state in states[:3]])
    withheld = np.array([eval_fourier(state, dense_x) for state in states])
    refined, _ = fft_baseline(512, end=.7, nu=.05, dt=.00005)
    ref_diff = float(np.linalg.norm(eval_fourier(refined, dense_x)-withheld[-1])/np.linalg.norm(withheld[-1]))
    return Sensors(sensors_x, times, observations), dense_x, future_times, withheld, ref_diff


def plot_results(rows, identifiability, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    labels = ["No noise", "0.1% noise", "1% noise"]
    levels = [0., .001, .01]
    colors = ["#247ba0", "#f18f01", "#8f3985"]
    for pidx, label in enumerate(("Viscosity", "sin(x) amplitude", "cos(2x) amplitude")):
        for level_index, level in enumerate(levels):
            selected = [r for r in rows if r["noise_fraction"] == level and r["k"] == 32]
            y = [abs(r["parameter_relative_errors"][pidx]) for r in selected]
            jitter = np.linspace(-.045, .045, len(y)) if len(y)>1 else np.array([0.])
            axes[0].scatter(level_index+(pidx-1)*.17+jitter, np.maximum(y, 1e-10),
                            color=colors[pidx], s=28, label=label if level_index == 0 else None)
    axes[0].set(yscale="log", ylabel="Absolute relative parameter error", xticks=range(3), xticklabels=labels)
    axes[0].legend(loc="lower center", bbox_to_anchor=(.5, 1.02), fontsize=8, ncol=1)
    for level_index, level in enumerate(levels):
        selected = [r for r in rows if r["noise_fraction"] == level and r["k"] == 32]
        jitter = np.linspace(-.06, .06, len(selected)) if len(selected)>1 else np.array([0.])
        axes[1].scatter(level_index+jitter, [r["withheld_space_relative_l2"] for r in selected],
                        marker="o", color="#247ba0", label="Unobserved spatial points, t≤0.5" if level_index == 0 else None)
        axes[1].scatter(level_index+jitter, [r["future_time_relative_l2"] for r in selected],
                        marker="x", color="#f18f01", label="Unobserved future time, t=0.7" if level_index == 0 else None)
    axes[1].set(yscale="log", ylabel="Withheld field relative L2 error", xticks=range(3), xticklabels=labels)
    axes[1].legend(loc="lower center", bbox_to_anchor=(.5, 1.02), fontsize=8)
    for data, label, marker in ((rows[0], "Three observation times", "o"),
                                (identifiability, "Initial-condition observations only", "s")):
        singular = np.array(data["physical_jacobian_singular_values"])
        axes[2].semilogy([1, 2, 3], np.maximum(singular, 1e-14), marker+"-", label=label)
    axes[2].annotate("Exactly zero: viscosity is unidentified", (3, 1e-14), xytext=(1.15, 1e-11), fontsize=8,
                      arrowprops={"arrowstyle":"->", "color":".4"})
    axes[2].set(xlabel="Singular-value index", ylabel="Physical observation Jacobian singular value", xticks=[1, 2, 3])
    axes[2].legend(loc="lower center", bbox_to_anchor=(.5, 1.02), fontsize=8)
    for ax in axes:
        ax.grid(alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(top=.76, bottom=.16, wspace=.35)
    fig.savefig(out/"inverse_summary.png", dpi=180)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    began = time.perf_counter()
    sensors, dense_x, check_times, withheld, ref_diff = independent_data()
    truth = np.array([.05, 1., .2])
    forward = QuillForward()
    validation = verify_tangents(forward, sensors)
    data_rms = float(np.sqrt(np.mean(sensors.values**2)))
    rows = []
    for fraction in (0., .001, .01):
        for seed in ([0] if fraction == 0 else range(5)):
            noise = np.random.default_rng(seed).normal(size=sensors.values.shape)*fraction*data_rms
            observed = Sensors(sensors.x.copy(), sensors.times.copy(), sensors.values+noise)
            estimate, report = fit_parameters(forward, observed)
            # Withheld dense truth is first accessed after fitting returns.
            prediction = forward.predict(estimate, dense_x, check_times)[:, :, 0]
            report.update(k=32, neurons=len(forward.model["enc"].centers), noise_fraction=fraction,
                          noise_seed=seed, noise_standard_deviation=fraction*data_rms,
                          setup_seconds=forward.setup_seconds,
                          parameter_relative_errors=((estimate-truth)/truth).tolist(),
                          withheld_space_relative_l2=float(np.linalg.norm(prediction[:3]-withheld[:3])/np.linalg.norm(withheld[:3])),
                          future_time_relative_l2=float(np.linalg.norm(prediction[3]-withheld[3])/np.linalg.norm(withheld[3])))
            rows.append(report)
            print(json.dumps({k:report[k] for k in ["noise_fraction", "noise_seed", "parameters", "fit_seconds", "withheld_space_relative_l2"]}), flush=True)
    # Resolve the noiseless discretization floor with higher QUILL bandwidth.
    refined = QuillForward(64, 1025, dt=.001)
    estimate, report = fit_parameters(refined, sensors)
    prediction = refined.predict(estimate, dense_x, check_times)[:, :, 0]
    report.update(k=64, neurons=len(refined.model["enc"].centers), noise_fraction=0., noise_seed=0,
                  noise_standard_deviation=0., setup_seconds=refined.setup_seconds,
                  parameter_relative_errors=((estimate-truth)/truth).tolist(),
                  withheld_space_relative_l2=float(np.linalg.norm(prediction[:3]-withheld[:3])/np.linalg.norm(withheld[:3])),
                  future_time_relative_l2=float(np.linalg.norm(prediction[3]-withheld[3])/np.linalg.norm(withheld[3])))
    rows.append(report)
    refined_estimate = estimate.copy()
    refined_prediction = prediction.copy()
    ic_data = Sensors(sensors.x, np.array([0.]), (np.sin(sensors.x)+.2*np.cos(2*sensors.x))[None, :])
    unidentifiable = []
    for initial_nu in (.02, .09, .17):
        estimate, info = fit_parameters(forward, ic_data, initial=(initial_nu, .8, .05))
        prediction = forward.predict(estimate, dense_x, check_times)[:, :, 0]
        info.update(initial_viscosity=initial_nu,
                    future_time_relative_l2=float(np.linalg.norm(prediction[3]-withheld[3])/np.linalg.norm(withheld[3])))
        unidentifiable.append(info)
    output = dict(protocol="Three-physical-parameter inverse problem. Actual QUILL state compressed by exact reassociation. Observations alone enter residual; no neural readout fit.",
                  parameters=["nu", "alpha", "beta"], truth=truth.tolist(),
                  initial_guess=[.09, .8, .05], bounds=[[.005, .4, -.5], [.2, 1.6, .7]],
                  equation="u_t+u u_x=nu u_xx, periodic [-pi,pi]", initial_condition_family="alpha sin(x)+beta cos(2x)",
                  sensors=dict(x=sensors.x.tolist(), times=sensors.times.tolist(), clean_values=sensors.values.tolist(), count=sensors.values.size),
                  measurements="Synthetic data from independent dealiased Fourier RK4, N384 dt1e-4. No future reference field enters fitting.",
                  reference_refinement_at_t07_relative_l2=ref_diff, tangent_validation=validation,
                  rows=rows, initial_condition_only_controls=unidentifiable,
                  reduced_state_count=forward.U.shape[1], tangent_state_shape=[forward.U.shape[1], 4],
                  jacobian_shape=[24, 3], native_dt=forward.dt, threads=1,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  elapsed_seconds=time.perf_counter()-began)
    (OUT/"inverse_metrics.json").write_text(json.dumps(output, indent=2)+"\n")
    np.savez_compressed(OUT/"inverse_withheld.npz", x=dense_x, times=check_times, reference=withheld,
                        refined_estimate=refined_estimate, refined_prediction=refined_prediction)
    plot_results(rows, unidentifiable[1], OUT)
    print(json.dumps(dict(validation=validation,refined=rows[-1],unidentifiable=unidentifiable),indent=2))


if __name__ == "__main__":
    main()
