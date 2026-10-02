"""Sparse-anchor coefficient inference and its exact data-free ambiguity.

For homogeneous Dirichlet data and -kappa div(a grad u)=f, the family
u_kappa=(kappa_true/kappa)*u_true satisfies the same physics for every
positive kappa. Sparse observations supply the missing amplitude scale.
Noiseless and iid Gaussian-noise controls use identical features/options.
"""
from __future__ import annotations

import json
import resource
import sys
import time

import route2_constraints_common as common
import numpy as np


def run():
    started = time.perf_counter()
    features = common.new_features()
    points = common.annulus_points(12,55109)
    exact = common.truth(points)
    rows = []
    for initial in (.6,1.,2.7):
        rows.append(common.run_case(f'inverse_no_anchors_initial_{initial:g}',features,beta=0,inverse=True,
                                     boundary_kind='dirichlet',parameter_initial=initial,save_model=initial==1.))
    rows.append(common.run_case('inverse_exact_anchors',features,beta=0,inverse=True,boundary_kind='dirichlet',
                                 anchors=(points,exact),save_model=True))
    for fraction in (.005,.02):
        sigma = fraction*common.FIELD_SCALE
        for replicate in range(3):
            measurements = exact+sigma*np.random.default_rng(19079+replicate).normal(size=len(points))
            rows.append(common.run_case(f'inverse_noise_{fraction:g}_rep{replicate}',features,beta=0,inverse=True,
                                         boundary_kind='dirichlet',anchors=(points,measurements),noise_sigma=sigma))
    summary = dict(provenance=common.summary_provenance(),cases=rows,
                   exact_nonidentifiability='For all kappa in [.25,4], u=(1.7/kappa)*u_true satisfies the same no-anchor forcing and zero Dirichlet data. Residual convergence cannot identify kappa.',
                   noise_model='Independent zero-mean Gaussian noise; sigma=.005 or.02 times declared field unit .1; three fixed noise seeds per level, not enough to claim coverage calibration',
                   total_seconds=time.perf_counter()-started,
                   process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024))
    (common.OUT/'inverse_summary.json').write_text(json.dumps(summary,indent=2))
    plot(rows)
    return summary


def plot(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(1,3,figsize=(12.4,3.6))
    unobserved = [r for r in rows if r['observation_count']==0]
    axes[0].plot([r['initial_parameter'] for r in unobserved],[r['kappa_estimate'] for r in unobserved],'o-',label='No anchors')
    axes[0].axhline(common.KAPPA,color='black',linestyle='--',label='Generating coefficient')
    axes[0].set(xlabel='Initial coefficient',ylabel='Recovered coefficient',title='Same PDE, multiple admissible answers')
    axes[0].legend(fontsize=8)
    groups = [('Exact',[r for r in rows if r['observation_count'] and not r['noise_sigma_absolute']]),
              ('σ = 0.0005',[r for r in rows if abs(r['noise_sigma_absolute']-.0005)<1e-12]),
              ('σ = 0.002',[r for r in rows if abs(r['noise_sigma_absolute']-.002)<1e-12])]
    for index,(_,values) in enumerate(groups):
        axes[1].plot(np.full(len(values),index),[r['kappa_estimate'] for r in values],'o',color='tab:blue')
        axes[2].semilogy(np.full(len(values),index),[r['ordinary_forward_relative_l2'] for r in values],'o',color='tab:blue')
    axes[1].axhline(common.KAPPA,color='black',linestyle='--')
    for axis in axes[1:]:
        axis.set_xticks(range(len(groups)),[g[0] for g in groups])
    axes[1].set(ylabel='Recovered coefficient',title='Twelve observations identify amplitude')
    axes[2].set(ylabel='Held-out relative field error',title='Noise limits field reconstruction')
    fig.tight_layout()
    fig.savefig(common.OUT/'inverse_constraints.png',dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    run()
