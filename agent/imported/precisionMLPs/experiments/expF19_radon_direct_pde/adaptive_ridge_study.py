"""Controlled adaptive directional PDE study; actual neural RHS and baselines."""
from __future__ import annotations
import os
for name in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","VECLIB_MAXIMUM_THREADS"]:os.environ.setdefault(name,"1")
os.environ.setdefault("MPLCONFIGDIR","/private/tmp/codex-adaptive-ridge-mpl")
from pathlib import Path
import argparse,json,time
import numpy as np
from solver.adaptive_ridges import AdaptiveRidgeProblem,solve_adaptive_ridges
from solver.highdim import PeriodicReactionDiffusion,evaluate_fourier

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/checkpoint_F_applications/expF19_radon_direct_pde/adaptive_followup'


def rel(a,b):return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-30))


def cases():
    return {
        'sparse':dict(initial=lambda x:.2+.12*np.cos(x[:,0])+.002*np.cos(8*x[:,0]),
                      grid_initial=lambda x:.2+.12*np.cos(x[0])+.002*np.cos(8*x[0]),
                      nu=.5,K=8,T=1.,description='Invariant x-only solution; tiny fast harmonic decays and can be dropped'),
        'interacting':dict(initial=lambda x:.25+.17*np.cos(3*x[:,0])+.17*np.cos(3*x[:,1])+.1*np.cos(2*x[:,0]-2*x[:,1]),
                          grid_initial=lambda x:.25+.17*np.cos(3*x[0])+.17*np.cos(3*x[1])+.1*np.cos(2*x[0]-2*x[1]),
                          nu=.01,K=10,T=.6,description='Three initial directions generate interacting mixed-frequency directions'),
    }


def reference(case,x,cutoff,dt=.001):
    before=time.perf_counter()
    model=PeriodicReactionDiffusion(2,cutoff,case['nu'])
    state,meta=model.solve(case['grid_initial'],case['T'],dt)
    k,c=model.sparse_coefficients(state)
    evaluated=evaluate_fourier(x,k,c,True)
    meta['evaluation_seconds']=time.perf_counter()-before-meta['seconds']
    return evaluated,meta


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    rows=[];histories={};grids={};references={}
    x=np.random.default_rng(742).uniform(-np.pi,np.pi,(768,2))
    axis=np.linspace(-np.pi,np.pi,61)
    image_x=np.stack(np.meshgrid(axis,axis,indexing='ij'),axis=-1).reshape(-1,2)
    # All reference values are computed separately and never enter the solver.
    settings=[('fixed_quill',dict(adaptive=False,backend='quill',tolerance=1e-5)),
              ('adaptive_quill',dict(adaptive=True,backend='quill',tolerance=1e-5)),
              ('adaptive_classical',dict(adaptive=True,backend='classical',tolerance=1e-5)),
              ('tight_quill',dict(adaptive=True,backend='quill',tolerance=1e-6)),
              ('loose_quill',dict(adaptive=True,backend='quill',tolerance=1e-4)),
              ('half_dt_quill',dict(adaptive=True,backend='quill',tolerance=1e-5,dt=.0025)),
              ('half_adapt_quill',dict(adaptive=True,backend='quill',tolerance=1e-5,adapt_interval=.025)),
              ('no_drop_quill',dict(adaptive=True,backend='quill',tolerance=1e-5,drop_fraction=0.))]
    for name,case in cases().items():
        truth,refmeta=reference(case,x,24)
        fine,finemeta=reference(case,x,32,dt=.0005)
        references[name]=dict(reference=refmeta,finer_reference=finemeta,
                              value_disagreement=rel(truth['value'],fine['value']),
                              laplacian_disagreement=rel(truth['laplacian'],fine['laplacian']))
        outputs={};solutions={};supports={}
        case_settings=settings+([('K14_tight_quill',dict(adaptive=True,backend='quill',tolerance=1e-6,cutoff=14)),
                                 ('K18_tight_quill',dict(adaptive=True,backend='quill',tolerance=1e-6,cutoff=18)),
                                 ('K18_tight_classical',dict(adaptive=True,backend='classical',tolerance=1e-6,cutoff=18)),
                                 ('K18_half_adapt',dict(adaptive=True,backend='quill',tolerance=1e-6,cutoff=18,adapt_interval=.025)),
                                 ('K18_each_step',dict(adaptive=True,backend='quill',tolerance=1e-6,cutoff=18,adapt_interval=.005)),
                                 ('K18_each_half_step',dict(adaptive=True,backend='quill',tolerance=1e-6,cutoff=18,adapt_interval=.0025,dt=.0025)),
                                 ('K18_fixed_quill',dict(adaptive=False,backend='quill',tolerance=1e-6,cutoff=18)),
                                 ('K18_replay_quill',dict(adaptive=True,backend='quill',tolerance=1e-6,cutoff=18,rollback=True)),
                                 ('K18_replay_classical',dict(adaptive=True,backend='classical',tolerance=1e-6,cutoff=18,rollback=True))]
                                if name=='interacting' else [])
        for label,original_kw in case_settings:
            kw=original_kw.copy();cutoff=kw.pop('cutoff',case['K'])
            p=AdaptiveRidgeProblem(case['initial'],lambda u:u*u,case['nu'],cutoff,name)
            options=dict(t_end=case['T'],dt=.005,adapt_interval=.05)
            options.update(kw)
            sol=solve_adaptive_ridges(p,**options)
            ts=time.perf_counter();value,lap=sol.evaluate(x,True);evaltime=time.perf_counter()-ts
            ts=time.perf_counter();residual=sol.residual(x);restime=time.perf_counter()-ts
            rhs=case['nu']*lap+value**2
            row=dict(case=name,label=label,**sol.metrics,
                     heldout_value_error=rel(value,fine['value']),
                     heldout_laplacian_error=rel(lap,fine['laplacian']),
                     offgrid_relative_pde_residual=float(np.linalg.norm(residual)/np.linalg.norm(rhs)),
                     offgrid_max_pde_residual=float(np.max(abs(residual))),
                     heldout_evaluation_seconds=evaltime,heldout_residual_seconds=restime,
                     initial_candidate_directions=case['K'],
                     reference_validation_seconds=refmeta['seconds']+finemeta['seconds']+refmeta['evaluation_seconds']+finemeta['evaluation_seconds'])
            row['total_with_heldout_validation_seconds']=row['total_seconds']+evaltime+restime
            # Metadata field above is not a direction count; report actual IC
            # support and primitive directions from coefficient extraction.
            nm=sol.basis.nmodes
            initial=sol.basis.A@case['initial'](sol.basis.x)
            initactive=np.hypot(initial[1:1+nm],initial[1+nm:])>1e-10
            row.pop('initial_candidate_directions')
            row['initial_modes']=int(initactive.sum())
            row['initial_directions']=len(set(tuple(k//np.gcd.reduce(abs(k))) for k in sol.basis.k[initactive]))
            rows.append(row);histories[name+'_'+label]=sol.history
            outputs[label]=value;supports[label]=sol.active.copy()
            # Keep only the two fields used for the final image/export check;
            # retaining every K18 analysis dictionary wastes gigabytes.
            if label in ['fixed_quill','adaptive_quill']:solutions[label]=sol
            print(name,label,json.dumps({k:row[k] for k in ['total_seconds','final_modes','added','dropped','heldout_value_error','offgrid_relative_pde_residual','geometry']}),flush=True)
        for row in rows:
            if row['case']!=name:continue
            row['difference_from_fixed_native']=rel(outputs[row['label']],outputs['fixed_quill'])
            if row['label']=='adaptive_quill':
                row['difference_from_same_adaptive_classical']=rel(outputs['adaptive_quill'],outputs['adaptive_classical'])
                row['same_adaptive_classical_identical_support']=bool(np.array_equal(supports['adaptive_quill'],supports['adaptive_classical']))
            if row['label']=='K18_tight_quill':
                row['difference_from_same_adaptive_classical']=rel(outputs['K18_tight_quill'],outputs['K18_tight_classical'])
            if row['label']=='K18_replay_quill':
                row['difference_from_same_adaptive_classical']=rel(outputs['K18_replay_quill'],outputs['K18_replay_classical'])
                row['difference_from_fixed_same_cutoff']=rel(outputs['K18_replay_quill'],outputs['K18_fixed_quill'])
                row['same_adaptive_classical_identical_support']=bool(np.array_equal(supports['K18_replay_quill'],supports['K18_replay_classical']))
        base=solutions['adaptive_quill'];geo=base.basis.geometry(base.b,base.active)
        # Independent reconstruction from exported summed tanh blocks.
        raw=np.full(len(x),geo['bias'])
        for block in geo['rows']:
            raw+=np.tanh(block['gamma']*((x@block['direction'])[:,None]-block['centers']))@block['weights']+block['bias']
        exporterror=rel(raw,outputs['adaptive_quill'])
        assert exporterror<1e-11
        for row in rows:
            if row['case']==name and row['label']=='adaptive_quill':row['exported_geometry_difference']=exporterror
        grids[name]=dict(axis=axis,adaptive=base.evaluate(image_x).reshape(len(axis),len(axis)),
                         fixed=solutions['fixed_quill'].evaluate(image_x).reshape(len(axis),len(axis)),
                         active_k=base.basis.k[base.active],
                         active_amplitude=np.hypot(base.b[1:1+base.basis.nmodes],base.b[1+base.basis.nmodes:])[base.active],
                         directions=np.array([block['direction'] for block in geo['rows']]),
                         centers=np.array([block['N'] for block in geo['rows']]))
        save(rows,histories,references,grids)
    # Invariant-case test: adaptation must not manufacture off-axis directions.
    assert np.all(grids['sparse']['active_k'][:,1]==0)
    # Mixing-case test: more directions must be discovered than supplied.
    mixing=[r for r in rows if r['case']=='interacting' and r['label']=='adaptive_quill'][0]
    assert mixing['geometry']['directions']>mixing['initial_directions']
    # Candidate cutoff is not an accuracy certificate: refine it independently.
    mixing_rows={r['label']:r for r in rows if r['case']=='interacting'}
    assert mixing_rows['K18_tight_quill']['offgrid_relative_pde_residual']<mixing_rows['tight_quill']['offgrid_relative_pde_residual']*.1
    assert mixing_rows['K18_replay_quill']['heldout_value_error']<mixing_rows['K18_tight_quill']['heldout_value_error']*.1
    # A missing high-frequency IC must be rejected before evolution.
    underresolved=AdaptiveRidgeProblem(lambda z:np.sin(17*z[:,0]),lambda u:u*u,.01,4)
    try:solve_adaptive_ridges(underresolved,t_end=.02)
    except ValueError as error:assert 'underresolved' in str(error)
    else:raise AssertionError('An unrepresented IC was silently accepted')
    hidden=AdaptiveRidgeProblem(lambda z:.3+.2*np.cos(3*z[:,0]),lambda u:u*u,.01,4)
    hidden_sol=solve_adaptive_ridges(hidden,t_end=.002,dt=.001,tolerance=1e-8)
    v,lap=hidden_sol.evaluate(x,True);r=hidden_sol.residual(x)
    hidden_residual=float(np.linalg.norm(r)/np.linalg.norm(.01*lap+v*v))
    assert hidden_residual>.05
    references['negative_control']=dict(initial_represented=True,max_frequency=4,generated_frequency=6,
        independent_physical_residual=hidden_residual,status=hidden_sol.status,
        interpretation='Candidate-only selection does not certify global accuracy; actual off-grid residual exposes missing nonlinear mode')
    save(rows,histories,references,grids)
    plot()


def save(rows,histories,references,grids):
    (OUT/'adaptive_metrics.json').write_text(json.dumps(dict(rows=rows,histories=histories,references=references,
        equation='u_t=nu*Delta u+u^2, periodic[-pi,pi]^2',
        distinction='QUILL fields and analytic Laplacians generate each nonlinear RHS; Fourier quadrature selects increments without LS.',
        caution='Classical comparator uses IDENTICAL adaptive algorithm, with exact trigonometric features. Spectral thresholding is not a novel geometry theory.'),indent=2))
    data={}
    for name,row in grids.items():
        for key,v in row.items():data[name+'_'+key]=v
    np.savez_compressed(OUT/'adaptive_fields.npz',**data)


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter,FixedLocator,ScalarFormatter
    r=json.loads((OUT/'adaptive_metrics.json').read_text())
    rows=r['rows'];data=np.load(OUT/'adaptive_fields.npz')
    fig,ax=plt.subplots(2,3,figsize=(15,8.5),constrained_layout=True)
    for i,name in enumerate(['sparse','interacting']):
        subset={a['label']:a for a in rows if a['case']==name}
        for label in ['adaptive_quill','tight_quill','loose_quill']:
            h=r['histories'][name+'_'+label]
            ax[i,0].step([a['time'] for a in h],[a['active_modes'] for a in h],where='post',label=label.replace('_quill',''))
        ax[i,0].axhline(subset['fixed_quill']['candidate_modes'],color='.5',ls='--',label='Fixed dense')
        ax[i,0].set(xlabel='Physical time',ylabel='Active frequency pairs',title=name+': mode growth')
        ax[i,0].set_yscale('log')
        ax[i,0].legend(fontsize=8)
        labels=['fixed_quill','adaptive_quill','adaptive_classical','tight_quill','loose_quill']
        if name=='interacting':labels+=['K18_replay_quill','K18_fixed_quill','K18_replay_classical']
        for label in labels:
            a=subset[label]
            ax[i,1].scatter(a['total_seconds'],a['heldout_value_error'],s=65,label=label.replace('_',' '))
        ax[i,1].set(xscale='log',yscale='log',xlabel='Setup + evolution + adaptation (s)',ylabel='Held-out relative solution error',title='Actual cost against independent reference')
        ax[i,1].xaxis.set_minor_formatter(NullFormatter())
        ax[i,1].legend(fontsize=8)
        k=data[name+'_active_k'];amplitude=data[name+'_active_amplitude']
        c=ax[i,2].scatter(k[:,0],k[:,1],c=np.log10(np.maximum(amplitude,1e-12)),s=35,cmap='viridis')
        fig.colorbar(c,ax=ax[i,2],label='log10 final mode amplitude')
        ax[i,2].set(xlabel='kx',ylabel='ky',title='Discovered frequency directions',aspect='equal')
        if name=='sparse':ax[i,2].set(ylim=(-1,1),xlim=(0,5),yticks=[-1,0,1])
        for j in [0,1]:ax[i,j].grid(alpha=.2)
    fig.savefig(OUT/'adaptive_selection.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(1,3,figsize=(14,4.6),constrained_layout=True)
    for name in ['sparse','interacting']:
        subset={a['label']:a for a in rows if a['case']==name}
        labels=['loose_quill','adaptive_quill','tight_quill']
        ax[0].loglog([subset[s]['geometry']['neurons'] for s in labels],[subset[s]['heldout_value_error'] for s in labels],'-o',label=name)
        ref=subset['adaptive_quill']
        ax[1].bar([name+' native',name+' classical'],[ref['total_seconds'],subset['adaptive_classical']['total_seconds']])
        checks=['half_dt_quill','half_adapt_quill','no_drop_quill']
        ax[2].semilogy(range(len(checks)),[subset[s]['difference_from_fixed_native'] for s in checks],'-o',label=name)
    ax[0].set(xlabel='Actual exported tanh neurons',ylabel='Relative solution error',title='Geometry savings versus accuracy');ax[0].legend()
    ax[1].set(yscale='log',ylabel='Total solver seconds',title='Does QUILL beat the same spectral algorithm?');ax[1].tick_params(axis='x',rotation=25)
    ax[2].set(xticks=[0,1,2],xticklabels=['Half timestep','Half adaptation gap','No drops'],ylabel='Difference from fixed native solver',title='Independent adaptivity controls');ax[2].legend()
    for a in ax:a.grid(axis='y',alpha=.2)
    fig.savefig(OUT/'adaptive_controls.png',dpi=170);plt.close(fig)
    mixing={a['label']:a for a in rows if a['case']=='interacting'}
    fig,ax=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    labels=['tight_quill','K14_tight_quill','K18_tight_quill']
    ax[0].semilogy([mixing[s]['max_frequency'] for s in labels],[mixing[s]['heldout_value_error'] for s in labels],'-o',label='Solution error')
    ax[0].semilogy([mixing[s]['max_frequency'] for s in labels],[mixing[s]['offgrid_relative_pde_residual'] for s in labels],'-o',label='Physical PDE residual')
    ax[0].set(xlabel='Candidate frequency cutoff K',ylabel='Relative error / residual',title='Cutoff refinement exposes adaptation delay');ax[0].legend()
    labels=['K18_tight_quill','K18_half_adapt','K18_each_step','K18_each_half_step']
    ax[1].loglog([mixing[s]['adapt_interval'] for s in labels],[mixing[s]['heldout_value_error'] for s in labels],'-o')
    replay=mixing['K18_replay_quill']
    ax[1].scatter(replay['adapt_interval'],replay['heldout_value_error'],marker='*',s=170,label='Same .05 interval with replay')
    ax[1].legend(fontsize=8)
    ax[1].xaxis.set_major_locator(FixedLocator([.0025,.005,.025,.05]));ax[1].xaxis.set_major_formatter(ScalarFormatter())
    ax[1].xaxis.set_minor_formatter(NullFormatter())
    ax[1].set(xlabel='Time between direction searches',ylabel='Relative solution error',title='More frequent discovery resolves delayed modes')
    for a in ax:a.grid(alpha=.2)
    fig.savefig(OUT/'adaptive_refinement.png',dpi=170);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plot',action='store_true')
    args=p.parse_args();plot() if args.plot else main()
