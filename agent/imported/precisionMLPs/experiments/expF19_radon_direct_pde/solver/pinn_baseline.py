"""Maintained DeepXDE baseline; no locally reimplemented optimizer/training loop."""
from __future__ import annotations
import os
os.environ['DDE_BACKEND']='pytorch'
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS']:
    os.environ[name]='1'
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/codex-general-pinn-mpl')
import sys
from pathlib import Path
import time,json
import numpy as np


def train_square(output,seed=0,width=32,adam_steps=15000,lbfgs_steps=5000,points=1024):
    """Known PDE/BC only; reference values are used exclusively for diagnostics."""
    dependency_dir=Path('/private/tmp/codex-quill-deps')
    if dependency_dir.exists():
        sys.path.insert(0,str(dependency_dir))
    import torch
    import deepxde as dde
    torch.set_num_threads(1)
    dde.config.set_default_float('float64')
    dde.config.set_random_seed(seed)
    from native_elliptic import classical,classical_values,force
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    tag=f'pinn_w{width}_seed{seed}'
    # This separately computed truth never enters data.PDE, loss, or optimizer.
    ref,refmn,_=classical(28)
    rng=np.random.default_rng(1901)
    held=rng.uniform(0,np.pi,(2048,2))
    truth=classical_values(held,ref,refmn)
    history=[]
    started=time.perf_counter()
    def pde(x,y):
        lap=dde.grad.hessian(y,x,i=0,j=0)+dde.grad.hessian(y,x,i=1,j=1)
        f=3*torch.sin(x[:,0:1])*torch.sin(x[:,1:2])+.3*torch.sin(3*x[:,0:1])*torch.sin(2*x[:,1:2])
        return -lap+5*y**3-f
    geometry=dde.geometry.Rectangle([0,0],[np.pi,np.pi])
    data=dde.data.PDE(geometry,pde,[],num_domain=points,num_boundary=0,
                      train_distribution='Hammersley',num_test=1024)
    net=dde.nn.FNN([2]+[width]*4+[1],'tanh','Glorot normal')
    net.apply_feature_transform(lambda x:2*x/np.pi-1)
    net.apply_output_transform(lambda x,y:torch.sin(x[:,0:1])*torch.sin(x[:,1:2])*y)
    model=dde.Model(data,net)
    setup_seconds=time.perf_counter()-started

    class Trace(dde.callbacks.Callback):
        def __init__(self,stage):
            super().__init__();self.stage=stage;self.last=-1
        def on_epoch_end(self):
            step=self.model.train_state.step
            if step==self.last or step%1000:
                return
            self.last=step
            y=self.model.predict(held)[:,0]
            row=dict(stage=self.stage,step=int(step),seconds=time.perf_counter()-started,
                     relative_l2=float(np.linalg.norm(y-truth)/np.linalg.norm(truth)))
            history.append(row)
            with (output/(tag+'_trace.jsonl')).open('a') as f:f.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True)

    model.compile('adam',lr=1e-3)
    model.train(iterations=adam_steps,display_every=1000,callbacks=[Trace('adam')],verbose=0)
    adam_seconds=time.perf_counter()-started-setup_seconds
    adam_pred=model.predict(held)[:,0]
    adam_error=float(np.linalg.norm(adam_pred-truth)/np.linalg.norm(truth))
    dde.optimizers.config.set_LBFGS_options(maxcor=50,ftol=1e-15,gtol=1e-10,
                                           maxiter=lbfgs_steps,maxfun=lbfgs_steps*2,maxls=50)
    lbfgs_started=time.perf_counter()
    model.compile('L-BFGS')
    model.train(display_every=1000,callbacks=[Trace('lbfgs')],verbose=0)
    lbfgs_seconds=time.perf_counter()-lbfgs_started
    total_seconds=time.perf_counter()-started
    prediction=model.predict(held)[:,0]
    residual=model.predict(held,operator=pde)[:,0]
    timings=[]
    for _ in range(5):
        t=time.perf_counter();model.predict(held);timings.append(time.perf_counter()-t)
    report=dict(method='DeepXDE Adam then L-BFGS',deepxde_version=dde.__version__,torch_version=torch.__version__,
                seed=seed,width=width,hidden_layers=4,precision='float64',threads=1,
                train_points=points,hard_boundary='sin(x)sin(y) output multiplier',
                adam_requested_steps=adam_steps,lbfgs_requested_steps=lbfgs_steps,
                total_completed_steps=int(model.train_state.step),setup_seconds=setup_seconds,
                adam_seconds=adam_seconds,lbfgs_seconds=lbfgs_seconds,total_seconds=total_seconds,
                adam_relative_l2=adam_error,relative_l2=float(np.linalg.norm(prediction-truth)/np.linalg.norm(truth)),
                heldout_relative_residual=float(np.linalg.norm(residual)/np.linalg.norm(force(held))),
                parameters=sum(p.numel() for p in net.parameters()),
                evaluate_2048_median_seconds=float(np.median(timings)),history=history,
                reference_policy='K28 classical square solver used only for diagnostics; no solution labels in training',
                limits='Two architecture settings and limited seeds, not an exhaustive best-PINN search. Wall times include diagnostic callbacks.')
    (output/(tag+'_metrics.json')).write_text(json.dumps(report,indent=2))
    np.savez_compressed(output/(tag+'_heldout.npz'),x=held,truth=truth,prediction=prediction,residual=residual)
    torch.save(net.state_dict(),output/(tag+'_weights.pt'))
    print(json.dumps({k:v for k,v in report.items() if k!='history'}),flush=True)
    return report
