"""Explicit QUILL complex-shift readouts with the current rational halo correction.

Implements 09_26_sl.tex equations density, mu, numinus, nuplus, scaled-pf.
N counts interval CELLS: N+1 interior centers, ceil(sqrt(N)) halo centers
on each side. No least-squares solve, SVD, or target-output interpolation.

``profile(z)`` must evaluate the prescribed analytic scalar/vector ridge
profile at complex z and return shape z.shape or z.shape+(n_outputs,).
The caller must ensure analyticity over the complex neighborhood used.
The default analytic_delta is appropriate only for entire profiles.
"""
from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import expit


@dataclass
class Encoding:
    centers: np.ndarray
    gamma: float
    weights: np.ndarray
    bias: np.ndarray
    baseline_weights: np.ndarray
    baseline_bias: np.ndarray
    correction_coefficients: np.ndarray
    boundary_moments: np.ndarray
    interval: tuple[float, float]
    n_cells: int
    halo_per_side: int
    lam: float
    analytic_delta: float
    sigma: float
    quadrature_order: int

    def evaluate(self, x, derivative=0, corrected=True):
        """Evaluate the actual tanh network (or its analytic x derivatives)."""
        x = np.asarray(x, dtype=float)
        z = self.gamma * (x[..., None] - self.centers)
        t = np.tanh(z)
        if derivative == 0:
            phi = t
        else:
            r = np.exp(-2 * np.abs(z))
            sech2 = 4 * r / (1 + r) ** 2
            if derivative == 1:
                phi = self.gamma * sech2
            elif derivative == 2:
                phi = -2 * self.gamma**2 * t * sech2
            elif derivative == 3:
                phi = self.gamma**3 * (4 * t*t * sech2 - 2 * sech2*sech2)
            else:
                raise ValueError("Only derivatives 0, 1, 2, 3 are implemented")
        weights = self.weights if corrected else self.baseline_weights
        result = phi @ weights
        if derivative == 0:
            result = result + (self.bias if corrected else self.baseline_bias)
        return result


def _scaled_partial_fraction(moments, lam):
    """Stable geometric-node formula; never evaluate a cancelling polynomial."""
    m = len(moments)
    zeta = np.exp(-2 * lam)
    r = np.exp(-2 * lam * (np.arange(1, m + 1) - .5))
    p = np.r_[1., np.cumprod(-np.expm1(-2 * lam * np.arange(1, m + 1)))]
    result = np.empty_like(moments)
    for i0 in range(m):
        elementary = np.array([1.])
        for j in range(m):
            if j != i0:
                elementary = np.convolve(elementary, [1., r[j]])
        i = i0 + 1
        factor = (-1.)**i * zeta**((i*(i+1)-1)/2) / (p[i-1]*p[m-i])
        result[i0] = factor * np.tensordot(elementary, moments, axes=(0, 0))
    return result


def encode(profile, n_cells, lam=.25, interval=(-1., 1.), halo=None,
           analytic_delta=None, quadrature_order=None):
    """Return baseline AND paper-corrected readouts in the same fixed geometry.

    For non-entire profiles supply analytic_delta (a valid local tube radius).
    Reject paper-inadmissible d>delta/8 or physical halo>delta. These are
    sufficient theorem conditions, not asserted necessary conditions.
    Entire profiles may use the default dynamically chosen admissible tube.
    """
    n = int(n_cells)
    if n != n_cells or n < 2 or lam <= 0:
        raise ValueError("n_cells must be an integer >=2 and lam positive")
    lo, hi = map(float, interval)
    if hi <= lo:
        raise ValueError("interval must increase")
    h = (hi-lo)/n
    radius = math.ceil(math.sqrt(n)) if halo is None else int(halo)
    if radius < 1:
        raise ValueError("Need at least one halo neuron per side")
    gamma = lam/h
    d = np.pi/(2*gamma)
    physical_halo = (radius+.5)*h
    delta = max(8*d, physical_halo) if analytic_delta is None else float(analytic_delta)
    if d > delta/8 * (1+1e-14) or physical_halo > delta*(1+1e-14):
        raise ValueError("Chosen geometry exceeds supplied analytic tube")
    k = max(1, math.ceil(delta/(8*d)-1e-14))
    sigma = 2*k*d
    centers = lo + h*np.arange(-radius, n+radius+1)
    a, b = centers[0]-h/2, centers[-1]+h/2
    m = (radius+1)//2
    nq = max(96, 8*m*k+32) if quadrature_order is None else int(quadrature_order)
    nodes, quad_weights = leggauss(nq)
    unit_y, quad_weights = (nodes+1)/2, quad_weights/2
    ell = np.arange(1, m+1)

    def density(z):
        return (np.asarray(profile(z+1j*d))-np.asarray(profile(z-1j*d)))/(2j*d)

    base = np.real(h/2*density(centers.astype(complex)))
    correction = []
    moment_arrays = []
    for endpoint, sign in [(a, 1), (b, -1)]:
        # Subtract f(endpoint) before integrating: its exact mu moment is zero.
        f_y = np.asarray(profile(endpoint+1j*d*unit_y))-np.asarray(profile(endpoint+0j))
        phase_mu = np.exp(sign*1j*np.pi*ell[:, None]*unit_y)
        mu = np.real((phase_mu*quad_weights) @ f_y)
        rho_y = density(endpoint+1j*sigma*unit_y)
        phase_nu = np.exp(sign*2j*ell[:, None]*gamma*sigma*unit_y)
        nu_weights = quad_weights*expit(-2*np.pi*sigma*unit_y/h)
        nu = 2*sigma*np.imag((phase_nu*nu_weights) @ rho_y)
        nu = nu * ((-1.)**ell).reshape((-1,)+(1,)*(nu.ndim-1))
        moments = mu-nu
        moment_arrays.append(moments)
        correction.append(_scaled_partial_fraction(moments, lam))
    correction = np.asarray(correction)
    weights = base.copy()
    weights[:m] += correction[0]/2
    weights[-m:] -= correction[1, ::-1]/2
    anchor = np.real(np.asarray(profile(lo+0j)))
    bias = anchor - np.tanh(gamma*(lo-centers)) @ weights
    baseline_bias = anchor - np.tanh(gamma*(lo-centers)) @ base
    return Encoding(centers, gamma, weights, bias, base, baseline_bias,
                    correction, np.asarray(moment_arrays), (lo, hi), n,
                    radius, float(lam), delta, sigma, nq)


def run_diagnostics(out):
    """Deterministic dense real-domain checks and selected precision references."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out.mkdir(parents=True, exist_ok=True)
    x = np.r_[np.linspace(-1, 1, 2001), -1+np.geomspace(1e-12, .1, 100),
              1-np.geomspace(1e-12, .1, 100)]
    targets = {
        "sine2pi": (lambda z: np.sin(2*np.pi*z),
                     lambda x: 2*np.pi*np.cos(2*np.pi*x)),
        "sine12pi": (lambda z: np.sin(12*np.pi*z),
                      lambda x: 12*np.pi*np.cos(12*np.pi*x)),
        "gaussian": (lambda z: np.exp(-3*(z-.15)**2),
                     lambda x: -6*(x-.15)*np.exp(-3*(x-.15)**2)),
    }
    rows = []
    start = time.perf_counter()
    for name, (f, fp) in targets.items():
        for n in [32, 64, 128, 256, 512]:
            for lam in [.10, .15, .20, .25, .30, .40, .50, .75]:
                enc = encode(f, n, lam)
                long = encode(f, n, lam, halo=max(enc.halo_per_side, math.ceil(24/lam)))
                y, yp = f(x), fp(x)
                for method, network, corrected in [("plain_sqrtN", enc, False),
                                                    ("corrected_sqrtN", enc, True),
                                                    ("plain_long_halo", long, False)]:
                    pred, dp = network.evaluate(x, corrected=corrected), network.evaluate(x, 1, corrected)
                    err = pred-y
                    weights = network.weights if corrected else network.baseline_weights
                    rows.append(dict(target=name, n_cells=n, lam=lam, method=method,
                                     neurons=len(network.centers), halo=network.halo_per_side,
                                     relative_l2=float(np.linalg.norm(err)/np.linalg.norm(y)),
                                     linf=float(np.max(abs(err))),
                                     derivative_relative_l2=float(np.linalg.norm(dp-yp)/np.linalg.norm(yp)),
                                     weight_l1=float(np.sum(abs(weights))),
                                     weight_max=float(np.max(abs(weights)))))
    metadata = dict(seconds=time.perf_counter()-start, n_points=len(x), rows=rows,
                    n_convention="interval cells; interior centers=N+1; R=ceil(sqrt(N)) per side",
                    method="Current-paper complex contour moments and scaled partial fractions; no fit")
    (out/"boundary_metrics.json").write_text(json.dumps(metadata, indent=2))
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for col, name in enumerate(targets):
        for method, style in [("plain_sqrtN", "--o"), ("corrected_sqrtN", "-o"), ("plain_long_halo", ":o")]:
            selected = [r for r in rows if r["target"] == name and r["lam"] == .25 and r["method"] == method]
            axes[0,col].loglog([r["n_cells"] for r in selected], [r["relative_l2"] for r in selected], style, label=method)
            selected = [r for r in rows if r["target"] == name and r["n_cells"] == 256 and r["method"] == method]
            axes[1,col].semilogy([r["lam"] for r in selected], [r["relative_l2"] for r in selected], style, label=method)
        axes[0,col].set(title=name+"; lambda=0.25", xlabel="N interval cells", ylabel="Relative value error")
        axes[1,col].set(title=name+"; N=256", xlabel="lambda", ylabel="Relative value error")
        for ax in axes[:,col]:
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
    fig.savefig(out/"boundary_comparison.png", dpi=170)
    plt.close(fig)
    validation = validate_reference()
    (out/"boundary_validation.json").write_text(json.dumps(validation, indent=2))
    print(json.dumps({"seconds": metadata["seconds"], "rows": len(rows)}))


def validate_reference():
    """Independent high-precision historical implementation and vector checks."""
    import importlib.util
    source = Path(__file__).parents[1]/"expD06_fixed_center_scales/construction_reference.py"
    spec = importlib.util.spec_from_file_location("quill_historical_reference", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ref = module.sine_coefficients(n=512, dps=60)
    enc = encode(lambda z: np.sqrt(2)*np.sin(2*np.pi*z), 512, .25,
                 analytic_delta=.25, quadrature_order=256)
    x = np.linspace(-1, 1, 501)
    f = lambda z: np.stack([np.sin(2*np.pi*z), np.cos(3*np.pi*z),
                            np.exp(-3*(z-.15)**2)], axis=-1)
    vector = encode(f, 128, .25)
    refined = encode(f, 128, .25, quadrature_order=384)
    constant = encode(lambda z: np.ones_like(z)*3.25, 128)
    answer = dict(
        reference_source=str(source), reference_decimal_precision=60,
        weight_max_difference=float(np.max(abs(ref["c"][1:]-enc.weights))),
        bias_difference=float(abs(ref["c"][0]-enc.bias)),
        correction_max_difference=float(np.max(abs(ref["correction_coefficients"]-enc.correction_coefficients))),
        vector_linf=np.max(abs(vector.evaluate(x)-f(x)), axis=0).tolist(),
        quadrupled_quadrature_prediction_difference=float(np.max(abs(vector.evaluate(x)-refined.evaluate(x)))),
        constant_linf=float(np.max(abs(constant.evaluate(x)-3.25))),
        constant_max_weight=float(np.max(abs(constant.weights))),
    )
    assert answer["weight_max_difference"] < 2e-13
    assert max(answer["vector_linf"]) < 2e-13
    assert answer["quadrupled_quadrature_prediction_difference"] < 2e-13
    assert answer["constant_max_weight"] == 0
    return answer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results/checkpoint_F_applications/expF19_radon_direct_pde/quill_review"))
    args = parser.parse_args()
    run_diagnostics(args.out)
