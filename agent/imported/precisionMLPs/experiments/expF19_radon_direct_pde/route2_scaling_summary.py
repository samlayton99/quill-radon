"""Collect measured cold PDE runs and explicit (not measured) cost projections."""
import json
import math
import os
from pathlib import Path
import numpy as np
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/route2-scaling-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from route2_scaling_study import OUT


def estimate(d,p,n=257,fields=1,angular='tensor'):
    coordinates=math.comb(p+d,d)
    lower=math.comb(p+d-1,d-1)
    directions=(p+1)**(d-1)
    if angular=='sparse_low_order':
        if p==2: directions=d*d
        elif p==3: directions=sum(2**(k-1)*math.comb(d,k) for k in range(1,min(d,3)+1))
        else: raise ValueError('Only degree2 and3 sparse rules are implemented')
    k=n+2*math.ceil(math.sqrt(n))
    neurons=directions*k
    q=max(128,4*coordinates);qb=max(64,2*coordinates)
    model_bytes=8*(neurons*(d+1+fields)+fields)
    map_bytes=8*directions*(p+1)*coordinates
    jet_bytes=8*coordinates*((d+1)*q+qb)
    # Existing engine is matrix-free in the full Jacobian; block size64.
    block_bytes=8*(coordinates*fields)*min(64,coordinates*fields)
    return dict(dimension=d,degree=p,centers=n,fields=fields,angular_rule=angular,
                coefficient_count=coordinates,solved_unknowns=coordinates*fields,
                angular_lower_bound_for_full_degree_p_space=lower,
                directions=directions,neurons=neurons,dense_model_bytes=model_bytes,
                profile_map_bytes=map_bytes,cached_feature_jet_bytes=jet_bytes,
                block_preconditioner_approx_bytes=block_bytes,
                counted_persistent_arrays_bytes=model_bytes+map_bytes+jet_bytes+block_bytes,
                scope='Formula projection, not an allocated/measured run; excludes workspaces, runtime, input and local derivative arrays',
                sampling_assumption='Qint=max(128,4P), Qbc=max(64,2P), value plus d diagonal second derivatives',
                jacobian='Global dense Jacobian is not stored; coefficient-feature jets are stored')


def main():
    rows=[]
    for path in OUT.glob('d*.json'):
        row=json.loads(path.read_text())
        row.setdefault('boundary_sampling','legacy_interleaved')
        row.setdefault('angular_weights_total_variation',row.get('solver',{}).get('feature_metrics',{}).get('angular_weights_total_variation',1.))
        rows.append(row)
    rows.sort(key=lambda r:(r['family'],r['dimension'],r['degree'],r['centers']))
    projections=[estimate(d,p) for p in (2,4,8,12,16) for d in range(2,11)]
    projections += [estimate(d,p,angular='sparse_low_order') for p in (2,3) for d in (5,10,20,30)]
    summary=dict(hardware='Apple M4 Mac mini, 10 cores, 16 GB; all numerical processes single-threaded',
        tested_scope='Manufactured semilinear elliptic PDE -Delta u+u^3=s; zero start; actual flat tanh model',
        caveats=['Quadratic/cubic controls measure low-degree feasibility; they do not establish arbitrary high-dimensional accuracy.',
                 'All-face Dirichlet traces strongly constrain low-degree spaces, and exactly determine degree2/3 polynomials for d>=2. These high-dimensional runs are architecture/conditioning controls, not difficult interior-dynamics demonstrations.',
                 'Held-out solution truth is used only for validation; source and boundary conditions are manufactured.',
                 'Active-axis reduction is a declared prior, not learned manifold discovery.',
                 'Projected memory counts omit temporary arrays and Python/Torch runtime overhead; measured RSS is reported separately.',
                 'Earlier interleaved Sobol boundary sampling missed half of each face. Final sparse high-dimensional and active-axis controls were rerun with independent sequences per face.'],
        sparse_cubature_reference='https://arxiv.org/pdf/math/0509101, equations24 and29 (Stroud/Smolyak rules)',
        measured=rows,formula_projections=projections)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(1,3,figsize=(16.4,4.6),constrained_layout=True)
    tensor=[r for r in rows if r['family']=='quadratic' and r['degree']==4 and r['centers']==129]
    sparse=[r for r in rows if r['family']=='dense_quadratic' and 'sparse' in r.get('angular_rule','')]
    cubic=[r for r in rows if r['family']=='dense_cubic']
    for selected,color,label,marker in [(tensor,'#365e9c','Full degree4 / product directions','o'),
                                        (sparse,'#b15829','Full degree2 / sparse directions','s'),
                                        (cubic,'#34866e','Full degree3 / sparse directions','D')]:
        ax[0].semilogy([r['dimension'] for r in selected],[r['seconds'] for r in selected],marker=marker,color=color,label=label)
    ax[0].set(xlabel='Ambient input dimension',ylabel='Measured seconds, construction + solve + audit',title='Actual cold nonlinear PDE controls')
    ax[0].legend(fontsize=8,loc='lower right')
    ax[0].grid(alpha=.2)
    d=np.arange(2,21)
    ax[1].semilogy(d,3.**(d-1),color='#8a8a8a',label='Previous degree2: 3^(d−1)')
    ax[1].semilogy(d,d*d,color='#b15829',label='Implemented degree2: d²')
    ax[1].semilogy(d,[sum(2**(k-1)*math.comb(int(j),k) for k in range(1,min(int(j),3)+1)) for j in d],color='#34866e',label='Implemented degree3: O(d³)')
    ax[1].set(xlabel='Ambient input dimension',ylabel='Directions (formula counts)',title='Low-order angular overhead is avoidable')
    ax[1].legend(fontsize=8);ax[1].grid(alpha=.2)
    for p,color in [(4,'#365e9c'),(8,'#a96f27'),(12,'#94417b')]:
        selection=[r for r in projections if r['degree']==p and r['angular_rule']=='tensor']
        ax[2].semilogy([r['dimension'] for r in selection],[r['counted_persistent_arrays_bytes']/2**30 for r in selection],marker='o',color=color,label=f'Degree {p}')
    ax[2].axhline(16,color='black',ls='--',lw=1,label='Entire host RAM: 16 GiB')
    ax[2].set(xlabel='Ambient input dimension',ylabel='Counted persistent arrays, GiB',title='Projected storage, N=257; excludes workspaces')
    ax[2].set_ylim(1e-4,1e8);ax[2].legend(fontsize=8,loc='upper left');ax[2].grid(alpha=.2)
    fig.suptitle('Route2 scaling: low-degree manufactured controls reach 20D; full high-resolution spaces remain expensive',fontsize=12)
    fig.savefig(OUT/'route2_dimension_scaling.png',dpi=180)
    plt.close(fig)
    # Small, auditable numeric table: estimates are not benchmark claims.
    selected=[estimate(d,p) for d,p in [(3,8),(4,8),(4,12),(4,16),(5,8),(5,12),(6,8),(8,4),(10,4)]]
    print(json.dumps(dict(selected_memory_projections=selected,
                         figure=str(OUT/'route2_dimension_scaling.png'),
                         measured_runs=len(rows)),indent=2))


if __name__=='__main__': main()
