"""Reproducible component benchmark; no PDE convergence or solution claims."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key,'1')
os.environ.setdefault('KMP_USE_SHM','0')
import json
from pathlib import Path
import time
import tracemalloc
import numpy as np
from route2_box_profiles import BoxProfileOperator


def benchmark(degree,repetitions=5):
    features=BoxProfileOperator([[-1.,1.]]*4,degree,centers=33)
    orders=((0,0,0,0),(1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1),
            (2,0,0,0),(0,2,0,0),(0,0,2,0))
    rng=np.random.default_rng(20261003+degree)
    points=rng.uniform(-.7,.7,(128,4));coefficients=rng.normal(size=(features.size,4))
    cotangents={d:rng.normal(size=(len(points),4)) for d in orders}
    plan=features.ideal_tensor_plan(4,128,orders)
    def panel_forward():
        prepared=features.prepare_ideal_columns(points,orders)
        values={d:np.zeros((len(points),4)) for d in orders}
        for start in range(0,features.size,64):
            indices=np.arange(start,min(start+64,features.size))
            panel=features.selected_ideal_columns(points,indices,orders,prepared=prepared)
            for d in orders:values[d]+=panel[d]@coefficients[indices]
        return values
    def panel_adjoint():
        prepared=features.prepare_ideal_columns(points,orders)
        gradient=np.zeros_like(coefficients)
        for start in range(0,features.size,64):
            indices=np.arange(start,min(start+64,features.size))
            panel=features.selected_ideal_columns(points,indices,orders,prepared=prepared)
            gradient[indices]=sum(panel[d].T@cotangents[d] for d in orders)
        return gradient
    def tensor_forward():
        state=features.prepare_ideal_tensor_forward(coefficients)
        values={d:np.empty((len(points),4)) for d in orders}
        for start in range(0,len(points),plan['rows']):
            sl=slice(start,min(start+plan['rows'],len(points)))
            batch=features.forward_ideal_tensor_jets(state,points[sl],orders)
            for d in orders:values[d][sl]=batch[d]
        return values
    def tensor_adjoint():
        state=features.prepare_ideal_tensor_adjoint(4)
        for start in range(0,len(points),plan['rows']):
            sl=slice(start,min(start+plan['rows'],len(points)))
            features.accumulate_ideal_tensor_jets({d:v[sl] for d,v in cotangents.items()},points[sl],state)
        return features.finish_ideal_tensor_adjoint(state)
    row=dict(dimension=4,degree=degree,fields=4,points=len(points),coordinates=features.size,plan=plan)
    results={}
    for name,function in [('panel_forward',panel_forward),('tensor_forward',tensor_forward),
                          ('panel_adjoint',panel_adjoint),('tensor_adjoint',tensor_adjoint)]:
        elapsed=[]
        for _ in range(repetitions):
            started=time.perf_counter();results[name]=function();elapsed.append(time.perf_counter()-started)
        row[name+'_seconds']=float(np.median(elapsed))
        tracemalloc.start();function();_,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
        row[name+'_peak_traced_bytes']=peak
    row.update(forward_speedup=row['panel_forward_seconds']/row['tensor_forward_seconds'],
        adjoint_speedup=row['panel_adjoint_seconds']/row['tensor_adjoint_seconds'],
        forward_relative_difference=max(float(np.linalg.norm(results['tensor_forward'][d]-results['panel_forward'][d])/
            np.linalg.norm(results['panel_forward'][d])) for d in orders),
        adjoint_relative_difference=float(np.linalg.norm(results['tensor_adjoint']-results['panel_adjoint'])/
            np.linalg.norm(results['panel_adjoint'])))
    return row


if __name__=='__main__':
    results=dict(scope='Single-thread ideal-coordinate component benchmark; panels already reuse univariate tables. No full-solve speed claim.',
                 cases=[benchmark(degree) for degree in (12,16)])
    output=Path(__file__).resolve().parents[2]/'results/checkpoint_F_applications/expF19_radon_direct_pde/route2_hardening/ideal_tensor_benchmark.json'
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2),flush=True)
