"""Actual QUILL Legendre value/derivative precision versus geometry.

No PDE or target data enter this experiment. The prescribed features alone
are encoded; exact Legendre formulas provide independent validation.
"""
import json
import math
import os
from pathlib import Path
import time

os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/quill-feature-precision-mpl')
import numpy as np
from numpy.polynomial import legendre
from quill_boundary import encode
from solver.general_features import _evaluate_encoding, _exact_legendre, ConstructedFeatures

OUT = Path(__file__).resolve().parents[2]/'results/checkpoint_F_applications/expF19_radon_direct_pde/general_residual'


def sweep(evaluation):
    x = np.unique(np.r_[np.cos(np.pi*np.arange(258)/257),
                        2*np.mod(np.arange(1,132)*np.sqrt(2),1)-1])
    rows = []
    for degree in [8,12,16]:
        exact = [_exact_legendre(x,degree,k) for k in range(3)]
        for n in [65,129,257,513,1025]:
            for lam in [.10,.15,.20,.25,.30]:
                started = time.perf_counter()
                bank = encode(lambda z:legendre.legvander(z,degree),n-1,lam,
                              halo=math.ceil(math.sqrt(n)))
                bank.evaluation_mode = evaluation
                bank.anchor_x = -1.
                bank.anchor_value = (-1.)**np.arange(degree+1)
                checks = []
                for k in range(3):
                    pred = _evaluate_encoding(bank,x,k)
                    err = np.max(abs(pred-exact[k]),axis=0)
                    scale = np.maximum(1.,np.max(abs(exact[k]),axis=0))
                    checks.append(dict(order=k,maximum_absolute=float(err.max()),
                                       maximum_scaled=float(np.max(err/scale)),
                                       per_mode_absolute=err.tolist()))
                row = dict(degree=degree,N_interior=n,lam=lam,evaluation=evaluation,
                           halo=bank.halo_per_side,
                           neurons=len(bank.centers),weight_l1_max=float(np.max(np.sum(abs(bank.weights),axis=0))),
                           checks=checks,maximum_scaled=max(a['maximum_scaled'] for a in checks),
                           seconds=time.perf_counter()-started)
                rows.append(row)
            best=min(rows[-5:],key=lambda a:a['maximum_scaled'])
            print(json.dumps({'degree':degree,'N':n,'evaluation':evaluation,
                              'best_lambda':best['lam'],
                              'best_scaled_error':best['maximum_scaled']}),flush=True)
            filename = 'feature_precision_sweep.json' if evaluation=='standard' else 'feature_precision_anchored.json'
            (OUT/filename).write_text(json.dumps(rows,indent=2))
    return rows


def precision_diagnostic():
    """Separate summation, FP64 function evaluation and coefficient errors."""
    import mpmath as mp
    degree,n,lam = 16,513,.2
    bank = encode(lambda z:legendre.legvander(z,degree),n-1,lam,
                  halo=math.ceil(math.sqrt(n)))
    bank.evaluation_mode = 'anchored'
    bank.anchor_x = -1.
    bank.anchor_value = (-1.)**np.arange(degree+1)
    x = np.array([-1.,-.999,-.8,-.23,0.,.123,.77,.99,1.])
    rows=[]
    with mp.workdps(55):
        c=list(map(mp.mpf,bank.centers)); gamma=mp.mpf(bank.gamma)
        weights=[[mp.mpf(w) for w in row] for row in bank.weights]
        for order in range(3):
            high=[]
            truth=[]
            standard=_evaluate_encoding(bank,x,order)
            compensated=[]
            for i,point in enumerate(x):
                xx=mp.mpf(point)
                tt=[mp.tanh(gamma*(xx-cc)) for cc in c]
                if order==0:
                    phi=[t-mp.tanh(gamma*(-1-cc)) for t,cc in zip(tt,c)]
                elif order==1:
                    phi=[gamma*(1-t*t) for t in tt]
                else:
                    phi=[-2*gamma*gamma*t*(1-t*t) for t in tt]
                high.append([mp.fsum(ph*ww[j] for ph,ww in zip(phi,weights))+
                             (mp.mpf((-1)**j) if order==0 else 0)
                             for j in range(degree+1)])
                truth.append([mp.diff(lambda z:mp.legendre(j,z),xx,order)
                              for j in range(degree+1)])
                # fsum improves summation but keeps FP64 elementary functions.
                z=bank.gamma*(point-bank.centers)
                t=np.tanh(z);r=np.exp(-2*abs(z));s=4*r/(1+r)**2
                if order==1: fp=bank.gamma*s
                elif order==2: fp=-2*bank.gamma**2*t*s
                else: fp=t-np.tanh(bank.gamma*(-1-bank.centers))
                compensated.append([math.fsum(fp*bank.weights[:,j])+
                                    ((-1.)**j if order==0 else 0)
                                    for j in range(degree+1)])
            def errors(pred):
                return [float(max(abs(mp.mpf(pred[i][j])-truth[i][j])
                                  for i in range(len(x)))) for j in range(degree+1)]
            rows.append(dict(order=order,fp64_anchored_error=errors(standard),
                             fp64_fsum_error=errors(compensated),
                             high_precision_evaluation_of_fp64_coefficients_error=errors(high)))
    result=dict(degree=degree,N_interior=n,lam=lam,reference_dps=55,
                points=x.tolist(),checks=rows,
                scope='Nine-point diagnostic, not a uniform bound; coefficients are unchanged FP64')
    (OUT/'feature_precision_arithmetic.json').write_text(json.dumps(result,indent=2))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    standard=sweep('standard')
    anchored=sweep('anchored')
    precision_diagnostic()
    selections=[]
    for degree in [8,12,16]:
        for tolerance in [1e-13,1e-14]:
            try:
                features=ConstructedFeatures([[-1.,1.]],degree,
                    evaluation='anchored',encoding_tolerance=tolerance)
                selections.append(dict(degree=degree,tolerance=tolerance,
                                       metrics=features.metrics))
            except ValueError as error:
                selections.append(dict(degree=degree,tolerance=tolerance,error=str(error)))
    (OUT/'feature_precision_adaptive.json').write_text(json.dumps(selections,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),constrained_layout=True)
    for ax,degree in zip(axes,[8,12,16]):
        for rows,method in [(standard,'Standard sum'),(anchored,'Anchored tanh differences')]:
            best=[min((r for r in rows if r['degree']==degree and r['N_interior']==n),
                      key=lambda r:r['maximum_scaled']) for n in [65,129,257,513,1025]]
            ax.loglog([r['N_interior'] for r in best],
                      [r['maximum_scaled']/np.finfo(float).eps for r in best],'-o',label=method)
        ax.axhline(1,color='black',ls=':',label='One FP64 epsilon')
        ax.set(title=f'Legendre modes 0–{degree}',xlabel='Interior centers per axis',
               ylabel='Worst per-mode scaled error / epsilon')
        ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.suptitle('Actual QUILL values and first/second derivatives; best of five tested bandwidths')
    fig.savefig(OUT/'feature_precision_summary.png',dpi=175)


if __name__=='__main__':
    main()
