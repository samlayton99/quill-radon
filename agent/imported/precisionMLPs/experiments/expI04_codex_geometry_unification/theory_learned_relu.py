"""A trained deep-ReLU test of independent target-curvature readout prediction."""
import os
os.environ["OMP_NUM_THREADS"]="1"
os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ.setdefault("MPLCONFIGDIR","/private/tmp/codex_geometry_theory_mpl")
from pathlib import Path
import json
import argparse
import numpy as np
from scipy.linalg import lstsq
from scipy.stats import qmc
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"results/checkpoint_I_depth_theory/expI04_codex_geometry_unification/theory"
torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
TARGET="oscillatory"

def f(x):
    if TARGET=="quadratic":
        return x*x+.2*x
    return np.sin(np.pi*x)+.2*np.cos(3*np.pi*x)

def coefficients(model):
    p=[q.detach().numpy().copy() for q in model.parameters()]
    w1,b1,w2,b2,w3,b3=p
    return w1.ravel(),b1,w2,b2,w3.ravel(),b3.item()

def features(x,w1,b1,w2,b2):
    return np.maximum(np.maximum(x[:,None]*w1+b1,0)@w2.T+b2,0)

def analyze(model,seed,width):
    w1,b1,w2,b2,v,b=coefficients(model)
    first=-b1/w1
    first=first[(first>-1)&(first<1)]
    breaks=np.unique(np.r_[-1,first,1])
    roots=[]; slopes=[]; neurons=[]
    for left,right in zip(breaks[:-1],breaks[1:]):
        middle=(left+right)/2
        active=(w1*middle+b1)>0
        a=w2[:,active]@w1[active]
        d=w2[:,active]@b1[active]+b2
        with np.errstate(divide="ignore",invalid="ignore"):
            rt=-d/a
        valid=(rt>left+1e-10)&(rt<right-1e-10)&(abs(a)>1e-10)
        for j in np.where(valid)[0]:
            roots.append(rt[j]);slopes.append(abs(a[j]));neurons.append(j)
    roots=np.array(roots);slopes=np.array(slopes);neurons=np.array(neurons,dtype=int)
    full=np.unique(np.r_[breaks,roots])
    samples=f(full)
    ds=np.diff(samples)/np.diff(full)
    target_alpha=np.diff(ds)
    if TARGET=="quadratic":
        # Exact for x²+0.2x, even when neighboring learned knots almost collide.
        target_alpha=np.diff(full)[:-1]+np.diff(full)[1:]
    slots=np.searchsorted(full,roots)-1
    independent_prediction=target_alpha[slots]/slopes
    actual=v[neurons]
    pred_per_neuron=np.full(len(v),np.nan)
    for j in np.unique(neurons):
        use=neurons==j
        # Geometry-only weighted average: least-squares scalar matching of
        # prescribed target curvature charges; no function-feature fit.
        pred_per_neuron[j]=np.sum(slopes[use]*target_alpha[slots[use]])/np.sum(slopes[use]**2)
    used=np.isfinite(pred_per_neuron)
    x=np.linspace(-1,1,16385)
    H=features(x,w1,b1,w2,b2)
    target=f(x)
    yh=H@v+b
    A=np.column_stack((np.ones_like(x),H))
    solved=lstsq(A,target,cond=1e-12)[0]
    rel=lambda a,b:float(np.linalg.norm(a-b)/np.linalg.norm(b))
    # The dense grid includes the original training points. Keep it as a
    # resolution diagnostic, and measure held-out error independently as well.
    x_holdout=2*qmc.Sobol(d=1,scramble=True,seed=17000+seed).random_base2(12).ravel()-1
    H_holdout=features(x_holdout,w1,b1,w2,b2)
    y_holdout=f(x_holdout)
    holdout_overlap=np.intersect1d(x_holdout,np.linspace(-1,1,1025)).size
    metrics={"seed":seed,"width":width,"test_relative_l2":rel(yh,target),"readout_floor_rel_l2":rel(A@solved,target),
             "independent_sobol_relative_l2":rel(H_holdout@v+b,y_holdout),
             "independent_sobol_readout_floor_rel_l2":rel(H_holdout@solved[1:]+solved[0],y_holdout),
             "independent_sobol_points":len(x_holdout),"independent_sobol_seed":17000+seed,
             "independent_sobol_training_point_overlap":int(holdout_overlap),
             "dense_grid_contains_training_points":True,
             "first_layer_knots":len(first),"new_second_layer_crossings":len(roots),
             "neurons_with_new_crossings":int(used.sum()),
             "multiple_crossing_neurons":int(sum(np.sum(neurons==j)>1 for j in range(len(v)))),
             "target_curvature_raw_coefficient_rel_error":rel(pred_per_neuron[used],v[used]),
             "target_curvature_refit_coefficient_rel_error":rel(pred_per_neuron[used],solved[1:][used]),
             "per_crossing_coefficient_rel_error":rel(independent_prediction,actual),
             "new_crossing_charge_sign_agreement":float(np.mean(np.sign(independent_prediction)==np.sign(actual))),
             "max_function_error":float(abs(yh-target).max()),
             "min_knot_gap":float(np.diff(full).min())}
    order=np.argsort(roots)
    fig,ax=plt.subplots(1,3,figsize=(15,4.2))
    ax[0].plot(x,target,"k--",label="target")
    ax[0].plot(x,yh,label="ordinary trained ReLU")
    ax[0].set(xlabel="x",ylabel="function",title=f"Learned solution: relative L2 {metrics['test_relative_l2']:.2e}")
    ax[0].legend(fontsize=8)
    ax[1].scatter(roots[order],actual[order],s=20,label="actual readout (one dot per crossing)")
    ax[1].scatter(roots[order],independent_prediction[order],s=18,marker="x",label="target-curvature prediction")
    ax[1].set(xlabel="new physical crossing",ylabel="raw readout",title="A good function fit need not match local charge prediction")
    ax[1].legend(fontsize=8)
    ax[2].plot(x,abs(yh-target),label="trained residual")
    ax[2].plot(x,abs(A@solved-target),label="same-geometry LS residual")
    ax[2].set(xlabel="x",ylabel="absolute residual",title="Readout refit checks the fixed-geometry floor")
    ax[2].legend(fontsize=8)
    fig.tight_layout();fig.savefig(OUT/f"learned_relu_{TARGET}_width{width}_seed{seed}.png",dpi=180);plt.close(fig)
    np.savez(OUT/f"learned_relu_{TARGET}_width{width}_seed{seed}.npz",w1=w1,b1=b1,w2=w2,b2=b2,v=v,b=b,
             roots=roots,slopes=slopes,neurons=neurons,coefficient_prediction=independent_prediction,
             per_neuron_prediction=pred_per_neuron,refit=solved)
    return metrics

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--width",type=int,default=128)
    parser.add_argument("--steps",type=int,default=12000)
    parser.add_argument("--target",choices=("oscillatory","quadratic"),default="oscillatory")
    args=parser.parse_args()
    TARGET=args.target
    allm=[]
    for seed in (0,1):
        torch.manual_seed(seed)
        model=torch.nn.Sequential(torch.nn.Linear(1,args.width),torch.nn.ReLU(),torch.nn.Linear(args.width,args.width),
                                  torch.nn.ReLU(),torch.nn.Linear(args.width,1))
        for layer in model:
            if isinstance(layer,torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.uniform_(layer.bias,-.2,.2)
        x=torch.linspace(-1,1,1025)[:,None]
        y=torch.from_numpy(f(x.numpy()))
        optim=torch.optim.Adam(model.parameters(),lr=1e-3)
        for step in range(args.steps):
            optim.zero_grad(set_to_none=True)
            loss=((model(x)-y)**2).mean()
            loss.backward();optim.step()
        print(f"seed{seed} adam mse {loss.item():.6e}",flush=True)
        optim=torch.optim.LBFGS(model.parameters(),lr=1,max_iter=1500,max_eval=2200,
                               tolerance_grad=1e-12,tolerance_change=1e-15,
                               history_size=50,line_search_fn="strong_wolfe")
        def closure():
            optim.zero_grad(set_to_none=True)
            loss=((model(x)-y)**2).mean()
            loss.backward();return loss
        optim.step(closure)
        metrics=analyze(model,seed,args.width)
        metrics.update({"target":TARGET,"adam_steps":args.steps,"adam_lr":1e-3,"lbfgs_max_iter":1500,
                        "training_points":1025,"test_points":16385,"dtype":"float64",
                        "initialization":"Xavier uniform weights, all biases uniform[-0.2,0.2]"})
        allm.append(metrics)
        print(json.dumps(metrics),flush=True)
    (OUT/f"learned_relu_{TARGET}_width{args.width}_metrics.json").write_text(json.dumps(allm,indent=2)+"\n")
