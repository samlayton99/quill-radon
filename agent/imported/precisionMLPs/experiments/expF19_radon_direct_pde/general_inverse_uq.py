"""Conditional linearized uncertainty and a model-mismatch negative control."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/codex-general-uq-mpl')
from pathlib import Path
import json
import numpy as np
from native_inverse import QuillForward,independent_data
from solver import calibrate


def main():
    out=Path(__file__).resolve().parents[2]/'results/checkpoint_F_applications/expF19_radon_direct_pde/general_solver'
    out.mkdir(parents=True,exist_ok=True)
    sensors,_,_,_,_=independent_data()
    model=QuillForward();basis=model.evaluate_basis(sensors.x)
    cache={};truth=np.array([.05,1.,.2]);rows=[]
    def fields(p):
        if 'p' not in cache or not np.array_equal(p,cache['p']):
            states=model.integrate(p,sensors.times,True)
            f=np.einsum('xd,tdp->txp',basis,states)
            cache.update(p=p.copy(),y=f[:,:,0].ravel(),jac=f[:,:,1:].reshape(-1,3))
        return cache
    predict=lambda p:fields(p)['y']
    jac=lambda p:fields(p)['jac']
    scale=float(np.sqrt(np.mean(sensors.values**2)))
    for level in [.001,.01]:
        for seed in range(30):
            observed=sensors.values.ravel()+np.random.default_rng(100+seed).normal(size=24)*level*scale
            fit=calibrate(predict,observed,[.09,.8,.05],bounds=([.005,.4,-.5],[.2,1.6,.7]),sigma=level*scale,jacobian=jac)
            std=fit['standard_deviation'];error=fit['parameters']-truth
            rows.append(dict(noise_fraction=level,seed=seed,parameters=fit['parameters'].tolist(),
                             standard_deviation=std.tolist(),covered95=(abs(error)<=1.96*std).tolist(),
                             reduced_chi_squared=float(np.sum(fit['residual']**2)/21),
                             seconds=fit['seconds'],success=fit['success'],rank=fit['rank'],
                             active_mask=fit['active_mask'].tolist()))
    # Deliberate mismatch: sensor offset absent from the inverse model.
    biased=sensors.values.ravel()+.02
    wrong=calibrate(predict,biased,[.09,.8,.05],bounds=([.005,.4,-.5],[.2,1.6,.7]),sigma=.001*scale,jacobian=jac)
    summary=[]
    for level in [.001,.01]:
        selected=[r for r in rows if r['noise_fraction']==level]
        summary.append(dict(noise_fraction=level,repeats=len(selected),coverage95_counts=np.sum([r['covered95'] for r in selected],axis=0).tolist(),
                            median_sd=np.median([r['standard_deviation'] for r in selected],axis=0).tolist(),
                            median_reduced_chi_squared=float(np.median([r['reduced_chi_squared'] for r in selected])),
                            median_fit_seconds=float(np.median([r['seconds'] for r in selected]))))
    result=dict(method='Local Gaussian covariance from known-noise weighted physical Jacobian; nonlinear forward tangents',
                truth=truth.tolist(),summary=summary,rows=rows,
                mismatch=dict(unmodeled_sensor_offset=.02,assumed_noise_fraction=.001,
                              parameters=wrong['parameters'].tolist(),reduced_chi_squared=wrong['reduced_chi_squared'],
                              observation_consistency=wrong['observation_consistency']),
                scope='Conditional uncertainty for 3 physical parameters; 30 realizations per noise level, with the same random draws rescaled between levels. Not a full Bayesian posterior or arbitrary model discrepancy treatment.')
    (out/'inverse_uq_metrics.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))

if __name__=='__main__':main()
