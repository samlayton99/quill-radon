"""Same-problem accuracy, cold solve cost, and field evaluation comparisons."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/codex-general-compare-mpl')
from pathlib import Path
import time,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from solver.tensor_box import BoxProblem,solve_box
from native_elliptic import classical,classical_values,force


def main():
    out=Path(__file__).resolve().parents[2]/'results/checkpoint_F_applications/expF19_radon_direct_pde/general_solver'
    out.mkdir(parents=True,exist_ok=True)
    xy=np.random.default_rng(1901).uniform(0,np.pi,(2048,2))
    records=[];solutions=[]
    for backend in ['quill','classical']:
        for k in [4,8,12,16,20]:
            s=solve_box(BoxProblem(force),modes=k,backend=backend)
            y=s.evaluate(xy)
            timings=[]
            for _ in range(5):
                start=time.perf_counter();s.evaluate(xy);timings.append(time.perf_counter()-start)
            records.append(dict(**s.metrics,evaluate_2048_median_seconds=float(np.median(timings))))
            solutions.append((s,y))
    # Reference values are not inputs to any native solve.
    ref,mn,_=classical(28);truth=classical_values(xy,ref,mn)
    for r,(s,y) in zip(records,solutions):
        r['relative_l2']=float(np.linalg.norm(y-truth)/np.linalg.norm(truth))
        r['heldout_relative_residual']=float(np.linalg.norm(s.residual(xy))/np.linalg.norm(force(xy)))
    pinn=[json.loads(p.read_text()) for p in sorted(out.glob('pinn_*_metrics.json'))]
    flat=json.loads((out.parent/'capabilities/elliptic_metrics.json').read_text())['rows']
    result=dict(constructed=records,pinn=pinn,prior_flat_ridge=flat,
                comparison='Same PDE/forcing and independent 2048 points. Constructed tensor architecture differs from flat ridges and PINN; classical controls use exact sine banks. No full architecture/hyperparameter search.',
                timing='Single cold solve including 521-point physical residual check, plus 5 query timings. Library imports/reference generation excluded; PINN includes diagnostic callbacks. Concurrent machine activity may affect wall times.')
    (out/'comparison_metrics.json').write_text(json.dumps(result,indent=2))
    fig,ax=plt.subplots(1,3,figsize=(16,4.8),constrained_layout=True)
    for backend,label,marker in [('quill','Constructed shared QUILL banks','o'),('classical','Classical sine basis','s')]:
        rows=[r for r in records if r['backend']==backend]
        ax[0].loglog([r['total_with_check_seconds'] for r in rows],[r['relative_l2'] for r in rows],'-'+marker,label=label)
        ax[1].loglog([r['evaluate_2048_median_seconds'] for r in rows],[r['relative_l2'] for r in rows],'-'+marker,label=label)
    for i,r in enumerate(pinn):
        label='DeepXDE Adam + L-BFGS' if i==0 else None
        ax[0].loglog(r['total_seconds'],r['relative_l2'],'x',ms=9,color='#ba3d4f',label=label)
        ax[1].loglog(r['evaluate_2048_median_seconds'],r['relative_l2'],'x',ms=9,color='#ba3d4f',label=label)
        history=r['history']
        ax[2].semilogy([h['seconds'] for h in history],[h['relative_l2'] for h in history],label=f"PINN width{r['width']} seed{r['seed']}")
    ax[0].set(title='Cold solve + independent residual check',xlabel='Seconds',ylabel='Held-out relative L2 error')
    ax[1].set(title='Evaluate field at 2,048 new points',xlabel='Seconds (median of 5)',ylabel='Held-out relative L2 error')
    ax[2].set(title='Maintained-library PINN convergence',xlabel='Training seconds',ylabel='Held-out relative L2 error')
    for a in ax:a.grid(alpha=.2);a.legend(fontsize=8)
    fig.savefig(out/'comparison.png',dpi=165)
    print(json.dumps([{k:r[k] for k in ['backend','modes','neurons','total_with_check_seconds','evaluate_2048_median_seconds','relative_l2','heldout_relative_residual','status']} for r in records],indent=2))

if __name__=='__main__':main()
