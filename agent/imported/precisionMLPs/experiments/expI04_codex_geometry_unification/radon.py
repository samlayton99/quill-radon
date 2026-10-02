"""Independent 3D Radon-to-tanh encoding and an explicit coefficient gauge.

No least-squares fit enters either experiment.  The only target information
used is its analytic Gaussian formula (hence its analytic plane integrals).
Run with BLAS threads=1; figures/measurements live beside the experiment report.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/codex-radon-mpl")
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.spatial.transform import Rotation
from scipy.stats import qmc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/checkpoint_I_depth_theory/expI04_codex_geometry_unification/radon"


def sphere_rule(nz):
    """Normalized tensor spherical cubature, merging antipodal duplicates."""
    z, wz = leggauss(nz)
    phi = 2 * np.pi * np.arange(2 * nz) / (2 * nz)
    zz, pp = np.meshgrid(z, phi, indexing="ij")
    rr = np.sqrt(1 - zz * zz)
    v = np.stack((rr * np.cos(pp), rr * np.sin(pp), zz), -1).reshape(-1, 3)
    weights = np.broadcast_to(wz[:, None] / (4 * nz), zz.shape).ravel()
    keep = v[:, 2] > 0
    return v[keep], 2*weights[keep]


def targets():
    rotation = Rotation.from_rotvec([.43, -.29, .61]).as_matrix()
    anis = rotation @ np.diag([1.5, 3., 6.]) @ rotation.T
    return {
        "isotropic": [(1., 3. * np.eye(3), np.zeros(3))],
        "anisotropic": [(1., anis, np.zeros(3))],
        "shifted mixture": [(.8, anis, np.array([.17, -.11, .09])),
                            (-.35, 2.2 * np.eye(3), np.array([-.19, .15, -.1]))],
    }


def target(x, components):
    answer = np.zeros(len(x))
    for amplitude, precision, center in components:
        y = x - center
        answer += amplitude * np.exp(-np.einsum("ni,ij,nj->n", y, precision, y))
    return answer


def profile(t, v, components, derivative=False):
    """q_v(t) = - Rf(v,t)'' / (2 pi), for normalized sphere integration."""
    answer = np.zeros(np.broadcast_shapes(t.shape, (len(v), 1)), dtype=np.result_type(t, float))
    for amplitude, precision, center in components:
        s = np.einsum("ni,ij,nj->n", v, np.linalg.inv(precision), v)[:, None]
        y = t - (v @ center)[:, None]
        prefactor = amplitude / (np.sqrt(np.linalg.det(precision)) * s ** 1.5)
        if derivative:
            answer += prefactor * (-6*y/s + 4*y**3/s**2) * np.exp(-y*y/s)
        else:
            answer += prefactor * (1 - 2*y*y/s) * np.exp(-y*y/s)
    return answer


def rel(pred, truth):
    return float(np.linalg.norm(pred-truth)/np.linalg.norm(truth))


def encode(v, weights, centers, gamma, components, corrected):
    h = float(centers[1]-centers[0])
    a = np.pi/(2*gamma)
    density = (np.imag(profile(centers[None, :]+1j*a, v, components))/a
               if corrected else profile(centers[None, :], v, components, True))
    coefficient = .5*h*weights[:, None]*density
    # A prescribed integration constant, not a regression or fitted scale.
    value_at_zero = weights @ profile(np.zeros((len(v), 1)), v, components)[:, 0]
    bias = value_at_zero - np.sum(coefficient * np.tanh(-gamma*centers)[None, :])
    return coefficient, bias, density


def evaluate(x, v, centers, gamma, coefficient, bias):
    # Limit transient memory, sharing tanh evaluations among all three targets.
    if coefficient.ndim == 3:
        answer = np.broadcast_to(bias, (len(x), len(bias))).copy()
    else:
        answer = np.full(len(x), bias)
    for start in range(0, len(v), 16):
        p = x @ v[start:start+16].T
        features = np.tanh(gamma*(p[:, :, None]-centers[None, None, :]))
        if coefficient.ndim == 3:
            answer += np.einsum("nmc,mct->nt", features, coefficient[start:start+16])
        else:
            answer += np.einsum("nmc,mc->n", features, coefficient[start:start+16])
    return answer


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    # Fresh nontraining points: no fitting samples exist in this experiment.
    x = 1.2*(qmc.Sobol(3, scramble=True, seed=20260930).random_base2(9)-.5)
    cases = targets()
    truths = np.column_stack([target(x, comps) for comps in cases.values()])
    h = .04
    gamma = .25/h
    centers = h*np.arange(-100, 101)
    rows = []
    saved = {}
    for nz in [4, 8, 12, 20, 32, 48]:
        v, angular_weights = sphere_rule(nz)
        weights, biases, continuous = [], [], []
        for name, comps in cases.items():
            weights.append(encode(v, angular_weights, centers, gamma, comps, True)[0])
            biases.append(encode(v, angular_weights, centers, gamma, comps, True)[1])
            continuous.append(angular_weights @ profile((x@v.T).T, v, comps))
        weights = np.stack(weights, -1)
        y = evaluate(x, v, centers, gamma, weights, np.array(biases))
        yc = np.column_stack(continuous)
        for k, name in enumerate(cases):
            row = dict(nz=nz, directions=len(v), neurons=len(v)*len(centers), target=name,
                       continuous_rel_l2=rel(yc[:, k], truths[:, k]),
                       tanh_rel_l2=rel(y[:, k], truths[:, k]),
                       tanh_conversion_rel_l2=rel(y[:, k], yc[:, k]))
            rows.append(row)
            print(row, flush=True)
        if nz == 48:
            saved = dict(v=v, angular_weights=angular_weights, coefficient=weights,
                         bias=np.array(biases), predictions=y, continuous=yc)
    leading_w, leading_b = [], []
    for comps in cases.values():
        w, b, _ = encode(saved['v'], saved['angular_weights'], centers, gamma, comps, False)
        leading_w.append(w); leading_b.append(b)
    leading_y = evaluate(x, saved['v'], centers, gamma, np.stack(leading_w, -1), np.array(leading_b))
    leading_errors = dict(zip(cases, [rel(leading_y[:, k], truths[:, k]) for k in range(len(cases))]))

    # An exact nonidentifiability example: two orthonormal frames span the
    # same r^2 through sum_k (v_k dot x)^2.  No numerical nullspace is invoked.
    rotation = Rotation.from_rotvec([.6, -.3, .4]).as_matrix()
    frames = np.vstack((np.eye(3), rotation))
    qtruth = np.sum(x*x, axis=1)
    gauge_rows = []
    gauge_predictions = []
    gauge_coefficients = []
    for tau in [0., 1., 5.]:
        frame_amplitudes = np.r_[np.full(3, 1-tau), np.full(3, tau)]
        coefficients = frame_amplitudes[:, None]*h*centers[None, :]
        bias = -np.sum(coefficients*np.tanh(-gamma*centers)[None, :])
        y = evaluate(x, frames, centers, gamma, coefficients, bias)
        exact = np.sum(frame_amplitudes[None, :]*(x@frames.T)**2, axis=1)
        recovered_quadratic = np.einsum('m,mi,mj->ij', frame_amplitudes, frames, frames)
        gauge_rows.append(dict(tau=tau, tanh_rel_l2=rel(y, qtruth),
                               exact_polynomial_rel_l2=rel(exact, qtruth),
                               invariant_quadratic_tensor_error=float(np.linalg.norm(recovered_quadratic-np.eye(3))),
                               coefficient_l2=float(np.linalg.norm(coefficients)), bias=float(bias)))
        gauge_predictions.append(y); gauge_coefficients.append(coefficients)
    gauge_delta = gauge_coefficients[2]-gauge_coefficients[0]
    coefficient_change = float(np.linalg.norm(gauge_delta)/np.linalg.norm(gauge_coefficients[0]))
    function_change = rel(gauge_predictions[2], gauge_predictions[0])
    metrics = dict(protocol="No joint or scalar least squares; analytic target-to-readout encoding.",
                   x_domain="512 independent Sobol points in [-0.6,0.6]^3", h=h, gamma=gamma,
                   lambda_value=h*gamma, centers=len(centers), centers_interval=[-4, 4],
                   rows=rows, leading_errors_at_2304_directions=leading_errors,
                   corrected_max_abs=np.max(np.abs(saved['predictions']-truths), axis=0).tolist(),
                   gauge=gauge_rows, gauge_relative_coefficient_change=coefficient_change,
                   gauge_relative_function_change=function_change,
                   elapsed_seconds=time.monotonic()-started)
    (OUT/'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
    np.savez_compressed(OUT/'construction.npz', x=x, truths=truths, centers=centers,
                        gamma=gamma, **saved)

    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    colors=['#2865b0', '#d65b35', '#2e8e65']
    for k, (name, comps) in enumerate(cases.items()):
        rr = [r for r in rows if r['target']==name]
        ax[0].loglog([r['directions'] for r in rr], [r['tanh_rel_l2'] for r in rr], 'o-', c=colors[k], label=name)
        ax[0].loglog([r['directions'] for r in rr], [r['continuous_rel_l2'] for r in rr], ':', c=colors[k], alpha=.75)
    ax[0].set(xlabel='Number of sphere directions', ylabel='Relative L2 error', title='3D: target → coefficients, no fitting')
    ax[0].legend(fontsize=9); ax[0].grid(alpha=.2)
    select_v = np.array([[1., 0, 0]])
    comps = cases['anisotropic']
    mask = np.abs(centers)<1.5
    corrected = encode(select_v, np.ones(1), centers, gamma, comps, True)[2][0]
    leading = profile(centers[None, :], select_v, comps, True)[0]
    ax[1].plot(centers[mask], leading[mask], color='#222222', label="Filtered Radon derivative q′")
    ax[1].plot(centers[mask], corrected[mask], '--', color='#ae3f94', label='Finite-width correction')
    ax[1].set(xlabel='Center c', ylabel='Readout / (h × angular weight / 2)', title='What a spoke readout encodes')
    ax[1].legend(fontsize=9); ax[1].grid(alpha=.2)
    positions=np.arange(len(cases)); width=.35
    final=[r['tanh_rel_l2'] for r in rows if r['nz']==48]
    ax[2].bar(positions-width/2, list(leading_errors.values()), width, color='#999999', label="Leading q′ samples")
    ax[2].bar(positions+width/2, final, width, color='#2865b0', label='Bandwidth-corrected')
    ax[2].set_yscale('log'); ax[2].set_xticks(positions, ['Isotropic', 'Anisotropic', 'Shifted mix.'])
    ax[2].set(ylabel='Relative L2 error', title='Same geometry, same analytic target')
    ax[2].legend(fontsize=9); ax[2].grid(axis='y', alpha=.2)
    fig.savefig(OUT/'radon_3d_prediction.png', dpi=170); fig.savefig(OUT/'radon_3d_prediction.pdf'); plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    inner=np.abs(centers)<1
    for k, tau in enumerate([0., 1., 5.]):
        ax[0].plot(centers[inner], gauge_coefficients[k][0, inner], c=colors[k], label=f'τ = {tau:g}')
        ax[1].plot(centers[inner], gauge_coefficients[k][3, inner], c=colors[k])
    ax[0].set(xlabel='Center c', ylabel='Readout', title='One axis-frame spoke')
    ax[1].set(xlabel='Center c', ylabel='Readout', title='One rotated-frame spoke')
    ax[0].legend(); ax[0].grid(alpha=.2); ax[1].grid(alpha=.2)
    order=np.argsort(qtruth)
    for k,tau in enumerate([0.,1.,5.]):
        ax[2].plot(qtruth[order], np.abs(gauge_predictions[k][order]-qtruth[order])+1e-18, '.', markersize=2, c=colors[k], label=f'τ = {tau:g}')
    ax[2].set_yscale('log'); ax[2].set(xlabel='Target ‖x‖²', ylabel='Absolute prediction error', title='All encode the same function')
    ax[2].grid(alpha=.2)
    fig.suptitle('An exact directional ambiguity: (1−τ) Σ xᵢ² + τ Σ (R x)ᵢ² = ‖x‖²', fontsize=14)
    fig.savefig(OUT/'coefficient_gauge.png', dpi=170); fig.savefig(OUT/'coefficient_gauge.pdf'); plt.close(fig)
    print(json.dumps({'leading_errors':leading_errors,'gauge':gauge_rows,'seconds':metrics['elapsed_seconds']},indent=2),flush=True)


if __name__ == '__main__':
    main()
