"""Flat tanh variable-coefficient PDEs on an annulus with mixed/integral constraints."""
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
    rows = [common.run_case('forward_robin_physics_only',features,save_model=True),
            common.run_case('forward_robin_exact_anchors',features,anchors=(points,exact),save_model=True),
            common.run_case('forward_neumann_integral',features,beta=0,boundary_kind='neumann_mean',save_model=True)]
    # Same physical model and block normalization at both noise levels.
    # Perfect known physics does not need these labels; degradation is an
    # expected, explicitly retained negative control for naive data fusion.
    for fraction in (.005,.02):
        sigma = fraction*common.FIELD_SCALE
        for replicate in range(3):
            measurements = exact+sigma*np.random.default_rng(19079+replicate).normal(size=len(points))
            rows.append(common.run_case(f'forward_robin_noise_{fraction:g}_rep{replicate}',features,
                                         anchors=(points,measurements),noise_sigma=sigma))
    summary = dict(provenance=common.summary_provenance(),cases=rows,
                   total_seconds=time.perf_counter()-started,
                   process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024))
    (common.OUT/'forward_constraints_summary.json').write_text(json.dumps(summary,indent=2))
    plot(rows)
    return summary


def plot(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(1,3,figsize=(12.4,3.6))
    saved = np.load(common.OUT/'forward_robin_physics_only.npz')
    points = saved['held_points']
    for axis,data,title in zip(axes[:2],(saved['ordinary_prediction'],saved['ordinary_prediction']-saved['held_truth']),
                               ('Variable diffusion + cubic reaction','Ordinary tanh MLP field error')):
        im = axis.scatter(points[:,0],points[:,1],c=data,s=8,cmap='viridis' if axis is axes[0] else 'coolwarm')
        axis.set_aspect('equal')
        axis.set(xlabel='x',ylabel='y',title=title,xlim=(-1.05,1.05),ylim=(-1.05,1.05))
        fig.colorbar(im,ax=axis,shrink=.8)
    groups = [('No labels',[rows[0]]),('Exact anchors',[rows[1]]),('σ = 0.0005',[r for r in rows if abs(r['noise_sigma_absolute']-.0005)<1e-12]),
              ('σ = 0.002',[r for r in rows if abs(r['noise_sigma_absolute']-.002)<1e-12])]
    for index,(_,values) in enumerate(groups):
        axes[2].semilogy(np.full(len(values),index),[r['ordinary_forward_relative_l2'] for r in values],'o',color='tab:blue')
    axes[2].set_xticks(range(len(groups)),[g[0] for g in groups],rotation=15)
    axes[2].set(ylabel='Held-out relative field error',title='Known physics: noisy labels can hurt')
    fig.tight_layout()
    fig.savefig(common.OUT/'forward_constraints.png',dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    run()
