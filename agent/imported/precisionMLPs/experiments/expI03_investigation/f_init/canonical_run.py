"""Exact-midpoint fixed-bank confirmation and mixer-prior control.

Three existing real-data splits, three independent training/validation partitions,
one common update schedule; validation selects the checkpoint and test only scores.
"""
import json
import math
import time
import argparse
import numpy as np
import torch
from probe import ROOT,OUT,SimpleMLP2,qi_ridge_init_layer_
from fixed_qi_mlp import FixedQIMLP


@torch.no_grad()
def occupancy(model,x):
    rows=[]
    for proj,bank in zip(model.projections,model.banks):
        p=proj(x)
        rows.append(dict(outside_fraction=(p.abs()>1).double().mean().item(),max_abs=p.abs().max().item(),
                         median_std=p.std(0).median().item(),weight_norm=proj.weight.norm().item()))
        x=bank(p).flatten(1)
    return rows


def main(resolution_sweep=False):
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1);torch.set_num_interop_threads(1)
    rows=[];steps=5000
    configs=["dense_std","dense_qi2","block_random","block_polynomial"]
    if resolution_sweep:configs=["block_n4","block_n8","block_n16","block_n32"]
    for name in ["airfoil","parkinsons","bike_sharing"]:
        d=torch.load(ROOT/f"data/cache_all20/{name}.pt",weights_only=True)
        budget=12*(d["d_in"]+132+2)+132+1
        w=round((-(d["d_in"]+3)+math.sqrt((d["d_in"]+3)**2+4*(budget-1)))/2)
        for seed in [0,1,2]:
            gen=torch.Generator().manual_seed(1000+seed)
            order=torch.randperm(len(d["xtr"]),generator=gen)
            valid=order[:max(128,len(order)//5)];train=order[len(valid):][:4096]
            x,xv,xt=d["xtr"][train].double(),d["xtr"][valid].double(),d["xte"].double()
            ym,ys=d["ytr"][train].double().mean(),d["ytr"][train].double().std()
            y,yv,yt=[(a.double()-ym)/ys for a in (d["ytr"][train],d["ytr"][valid],d["yte"])]
            batches=torch.randint(len(x),(steps,128),generator=gen)
            for config in configs:
                t0=time.perf_counter();torch.manual_seed(seed)
                initdiag=None
                if config.startswith("block"):
                    if resolution_sweep:
                        n=int(config.split("_n")[1]);b=d["d_in"]+n+2
                        estimate=(-b+math.sqrt(b*b+4*n*1799))/(2*n)
                        choices=[max(1,math.floor(estimate)),max(1,math.ceil(estimate))]
                        m=min(choices,key=lambda m:abs(n*m*m+b*m+1-1800))
                        model=FixedQIMLP(d["d_in"],channels=(m,m),centers=(n,n))
                        initdiag=model.initialize_(x,mixer_init="random",seed=seed)
                    else:
                        model=FixedQIMLP(d["d_in"])
                        initdiag=model.initialize_(x,mixer_init=config.split("_")[1],seed=seed)
                    buffers={k:v.clone() for k,v in model.named_buffers()}
                else:
                    model=SimpleMLP2(d["d_in"],w,1,activation="tanh")
                    if config=="dense_qi2":
                        qi_ridge_init_layer_(model.fc1,x,centers_per_dir=math.isqrt(w),uniform_centers=False)
                        with torch.no_grad():qi_ridge_init_layer_(model.fc2,model.act(model.fc1(x)),centers_per_dir=math.isqrt(w),uniform_centers=False)
                opt=torch.optim.Adam(model.parameters(),lr=1e-3)
                trace=[];best_valid=float("inf");best_test=None;best_step=None
                for step,idx in enumerate(batches):
                    opt.zero_grad(set_to_none=True)
                    loss=(model(x[idx]).squeeze(-1)-y[idx]).square().mean();loss.backward();opt.step()
                    if (step+1)%100==0:
                        with torch.no_grad():
                            ve=(model(xv).squeeze(-1)-yv).square().mean().item();te=(model(xt).squeeze(-1)-yt).square().mean().item()
                            tr=(model(x).squeeze(-1)-y).square().mean().item()
                        trace.append(dict(step=step+1,train_mse=tr,valid_mse=ve,test_mse=te))
                        if ve<best_valid:best_valid,best_test,best_step=ve,te,step+1
                enddiag=None
                if config.startswith("block"):
                    assert all(torch.equal(buffers[k],v) for k,v in model.named_buffers())
                    enddiag=occupancy(model,x)
                r=dict(task=name,seed=seed,config=config,params=sum(p.numel() for p in model.parameters()),
                       n_train=len(x),n_valid=len(xv),n_test=len(xt),steps=steps,batch=128,lr=1e-3,
                       channels=list(model.channels) if initdiag else None,centers=list(model.center_counts) if initdiag else None,
                       dense_width=w if initdiag is None else None,initial=initdiag,final=enddiag,
                       selected_test_mse=best_test,selected_valid_mse=best_valid,selected_step=best_step,
                       final_test_mse=trace[-1]["test_mse"],trace=trace,seconds=time.perf_counter()-t0)
                rows.append(r);(OUT/("resolution_rows.json" if resolution_sweep else "canonical_rows.json")).write_text(json.dumps(rows,indent=2))
                print(name,seed,config,"P",r["params"],"test",best_test,"seconds",r["seconds"],flush=True)
    if resolution_sweep:plot_resolution(rows)
    else:plot(rows)


def plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cfgs=["dense_std","dense_qi2","block_random","block_polynomial"]
    fig,axs=plt.subplots(1,3,figsize=(12,4));summary=[]
    for ax,task in zip(axs,["airfoil","parkinsons","bike_sharing"]):
        for i,cfg in enumerate(cfgs):
            rs=[r for r in rows if r["task"]==task and r["config"]==cfg]
            v=[r["selected_test_mse"] for r in rs]
            ax.bar(i,np.mean(v),alpha=.65,color=f"C{i}");ax.scatter([i]*len(v),v,c="k",s=18,zorder=3)
            summary.append(dict(task=task,config=cfg,params=rs[0]["params"],mean=float(np.mean(v)),std=float(np.std(v))))
        ax.set_xticks(range(4),["Dense default","Dense QI init","Block random mixer","Block I02 polynomial"],rotation=35,ha="right",fontsize=8)
        ax.set_title(task);ax.set_ylabel("Test MSE / train target variance");ax.set_ylim(bottom=0)
    fig.suptitle("Exact fixed midpoint banks: matched parameters, 3 seeds, 5000 Adam updates")
    fig.tight_layout();fig.savefig(OUT/"canonical_comparison.png",dpi=160);plt.close(fig)
    (OUT/"canonical_summary.json").write_text(json.dumps(summary,indent=2))


def plot_resolution(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,3,figsize=(12,4));summary=[]
    for ax,task in zip(axs,["airfoil","parkinsons","bike_sharing"]):
        means=[]
        for n in [4,8,16,32]:
            rs=[r for r in rows if r["task"]==task and r["centers"][0]==n]
            v=np.array([r["selected_test_mse"] for r in rs]);means.append(v.mean())
            ax.scatter([n]*len(v),v,c="k",s=18,zorder=3)
            summary.append(dict(task=task,centers=n,channels=rs[0]["channels"][0],params=rs[0]["params"],mean=float(v.mean()),std=float(v.std())))
        ax.plot([4,8,16,32],means,marker="o");ax.set_xscale("log",base=2)
        labels=[f"{n}\nM={next(s['channels'] for s in summary if s['task']==task and s['centers']==n)}" for n in [4,8,16,32]]
        ax.set_xticks([4,8,16,32],labels);ax.set_xlabel("Centers per channel (M = channel count)")
        ax.set_title(task);ax.set_ylabel("Test MSE / train target variance");ax.set_ylim(bottom=0)
    fig.suptitle("Spend parameters on channels or bank resolution? Approximately 1800 parameters")
    fig.tight_layout();fig.savefig(OUT/"resolution_comparison.png",dpi=160);plt.close(fig)
    (OUT/"resolution_summary.json").write_text(json.dumps(summary,indent=2))


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--resolution-sweep",action="store_true")
    main(ap.parse_args().resolution_sweep)
