"""Bounded F04 audit: preserved historical data plus fair depth/initialization probes.

Run with project .venv/bin/python; all output is isolated from historical artifacts.
This is an initializer/architecture diagnostic, not a new optimizer proposal.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/precision-mpl-cache")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/expF04_qi_init_real_data"))
from model import SimpleMLP, SimpleMLP2, qi_ridge_init_layer_

OUT = ROOT / "results/checkpoint_I_depth_theory/investigation_20260906/f_init"
HIST = ROOT / "results/checkpoint_F_applications/expF04_qi_init_real_data"


def historical():
    one = json.loads((HIST / "all20/all20_rows.json").read_text())
    two = json.loads((HIST / "all20_2layers/all20_2l_rows_N256.json").read_text())
    uni = json.loads((HIST / "all20_2layers_uniform/all20_2lu_rows_N256.json").read_text())
    summary = []
    for name in sorted({r["name"] for r in one}):
        a = lambda arm: float(np.mean([r["eval"] for r in one if r["name"] == name and r["arm"] == arm]))
        b = lambda rows, scheme, key: float(np.mean([r[key] for r in rows if r["name"] == name and r["act"] == "tanh" and r["scheme"] == scheme]))
        summary.append(dict(name=name, kind=next(r["kind"] for r in one if r["name"] == name),
                            one_default=a("baseline"), one_qi=a("scaled_psqrt"),
                            two_default=b(two,"baseline","final_eval"),
                            two_qi1=b(two,"qi1","final_eval"), two_qi2=b(two,"qi2","final_eval"),
                            two_qi2_best=b(two,"qi2","best_eval"),
                            two_uniform_qi2=b(uni,"qi2","final_eval")))
    (OUT / "historical_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def features(model, x):
    h = model.act(model.fc1(x))
    return model.act(model.fc2(h)) if isinstance(model, SimpleMLP2) else h


def diagnostic(model, x, y, xv, yv, xt, yt):
    """Frozen readout regularization picked on validation; test untouched by selection."""
    with torch.no_grad():
        h, hv, ht = [features(model, z) for z in (x, xv, xt)]
        mu, sd = h.mean(0), h.std(0).clamp_min(1e-8)
        h, hv, ht = [(z - mu) / sd for z in (h, hv, ht)]
        U, s, Vh = torch.linalg.svd(h, full_matrices=False)
        rhs = U.T @ y
        candidates = []
        for alpha in [1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1.]:
            w = Vh.T @ (s / (s*s + alpha*h.shape[0]) * rhs)
            ve = (hv @ w - yv).square().mean().item()
            candidates.append((ve, alpha, (ht @ w - yt).square().mean().item()))
        ve, alpha, te = min(candidates)
        pre = model.fc1(x)
        return dict(frozen_valid_mse=ve, frozen_test_mse=te, ridge_alpha=alpha,
                    first_saturated_fraction=(pre.abs() > 3).double().mean().item(),
                    first_mean_tanh_derivative=(1-torch.tanh(pre).square()).mean().item(),
                    feature_rank_1e6=int((s > s[0]*1e-6).sum()))


def bundle_spread(model, p):
    """Mean sine of angle from each first-layer row to its bundle's first row."""
    w = model.fc1.weight.detach()
    vals = []
    for start in range(0, w.shape[0], p):
        u = w[start:start+p]
        u = u/u.norm(dim=1, keepdim=True).clamp_min(1e-30)
        vals.extend((1-(u @ u[0]).square()).clamp_min(0).sqrt().tolist())
    return float(np.mean(vals))


def experiment(steps=1000, block_budget=False):
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    rows = []
    width = 128
    configs = ["one_std", "one_qi", "two_match_std", "two_match_qi1", "two_match_qi2",
               "two_same_qi1", "two_same_qi2"]
    if block_budget:
        configs=["two_match_block_std", "two_match_block_qi2"]
    for name in ["airfoil", "parkinsons", "bike_sharing"]:
        d = torch.load(ROOT / f"data/cache_all20/{name}.pt", weights_only=True)
        d_in = d["d_in"]
        # Single-layer P=(d+2)W+1; two-layer P=w²+(d+3)w+1.
        match_width = round((-(d_in+3)+math.sqrt((d_in+3)**2+4*(d_in+2)*width))/2)
        if block_budget:
            m=math.ceil(width/math.isqrt(width))
            budget=m*(d_in+width+2)+width+1
            match_width=round((-(d_in+3)+math.sqrt((d_in+3)**2+4*(budget-1)))/2)
        for seed in [0, 1, 2]:
            gen = torch.Generator().manual_seed(1000+seed)
            order = torch.randperm(len(d["xtr"]), generator=gen)
            valid = order[:max(128,len(order)//5)]
            train = order[len(valid):][:4096]
            x,xv,xt = d["xtr"][train].double(), d["xtr"][valid].double(),d["xte"].double()
            ym,ys = d["ytr"][train].double().mean(), d["ytr"][train].double().std()
            y,yv,yt = [(a.double()-ym)/ys for a in (d["ytr"][train],d["ytr"][valid],d["yte"])]
            batches = torch.randint(len(x),(steps,128),generator=gen)
            for config in configs:
                t0 = time.perf_counter()
                torch.manual_seed(seed)
                w = match_width if "match" in config else width
                cls = SimpleMLP if config.startswith("one") else SimpleMLP2
                model = cls(d_in,w,1,activation="tanh")
                p=math.isqrt(w)
                if "qi" in config:
                    qi_ridge_init_layer_(model.fc1,x,centers_per_dir=p,uniform_centers=False)
                    if config.endswith("qi2"):
                        with torch.no_grad():
                            qi_ridge_init_layer_(model.fc2,model.act(model.fc1(x)),centers_per_dir=p,uniform_centers=False)
                initial=diagnostic(model,x,y,xv,yv,xt,yt)
                opt=torch.optim.Adam(model.parameters(),lr=1e-3)
                trace=[]
                best_valid=float("inf")
                best_test=None
                for step, idx in enumerate(batches):
                    opt.zero_grad(set_to_none=True)
                    loss=(model(x[idx]).squeeze(-1)-y[idx]).square().mean()
                    loss.backward(); opt.step()
                    if (step+1)%100==0 or step+1==steps:
                        with torch.no_grad():
                            ve=(model(xv).squeeze(-1)-yv).square().mean().item()
                            te=(model(xt).squeeze(-1)-yt).square().mean().item()
                            tr=(model(x).squeeze(-1)-y).square().mean().item()
                        trace.append(dict(step=step+1,train_mse=tr,valid_mse=ve,test_mse=te))
                        if ve < best_valid:
                            best_valid,best_test=ve,te
                final=diagnostic(model,x,y,xv,yv,xt,yt)
                r=dict(task=name,seed=seed,config=config,width=w,p=p,
                       params=sum(a.numel() for a in model.parameters()),n_train=len(x),n_valid=len(xv),n_test=len(xt),
                       steps=steps,initial=initial,final=final,trace=trace,
                       final_test_mse=trace[-1]["test_mse"],selected_valid_mse=best_valid,selected_test_mse=best_test,
                       first_bundle_spread=bundle_spread(model,p) if "qi" in config else None,
                       seconds=time.perf_counter()-t0)
                rows.append(r)
                (OUT / ("block_dense_rows.json" if block_budget else "probe_rows.json")).write_text(json.dumps(rows,indent=2))
                print(name,seed,config,"P",r["params"],"final",round(r["final_test_mse"],5),"selected",round(best_test,5),"sec",round(r["seconds"],1),flush=True)
    return rows


def figures(hist, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(12,5))
    names=[r["name"] for r in hist]
    y=np.arange(len(names))
    axs[0].barh(y-.2,[r["one_default"]/r["one_qi"] for r in hist],height=.4,label="1 hidden layer: default / QI")
    axs[0].barh(y+.2,[r["two_default"]/r["two_qi2"] for r in hist],height=.4,label="2 hidden layers: default / QI both")
    axs[0].set_yticks(y,names);axs[0].axvline(1,color="k",lw=.8)
    axs[0].set_xlabel("Final evaluation loss ratio (larger favors QI)")
    axs[0].legend(fontsize=8);axs[0].set_title("Historical F04: initialization alone gives modest gains")
    axs[1].barh(y,[r["one_qi"]/r["two_qi2"] for r in hist])
    axs[1].set_yticks(y,names);axs[1].axvline(1,color="k",lw=.8)
    axs[1].set_xlabel("Final loss: 1-layer QI / 2-layer QI")
    axs[1].set_title("Historical depth comparison adds 65,792 parameters")
    fig.tight_layout();fig.savefig(OUT/"historical_ratios.png",dpi=150);plt.close(fig)
    if not rows:return
    fig,axs=plt.subplots(1,3,figsize=(14,4),sharey=False)
    configs=list(dict.fromkeys(r["config"] for r in rows))
    for ax,task in zip(axs,["airfoil","parkinsons","bike_sharing"]):
        for i,cfg in enumerate(configs):
            rs=[r for r in rows if r["task"]==task and r["config"]==cfg]
            vals=[r["selected_test_mse"] for r in rs]
            ax.bar(i,np.mean(vals),alpha=.65)
            ax.scatter([i]*len(vals),vals,color="k",s=14,zorder=3)
        ax.set_xticks(range(len(configs)),configs,rotation=50,ha="right",fontsize=8)
        ax.set_yscale("log");ax.set_title(task);ax.set_ylabel("Test MSE / train target variance")
    fig.suptitle("Depth and QI: matched updates and validation selection, 3 seeds")
    fig.tight_layout();fig.savefig(OUT/"controlled_depth.png",dpi=150);plt.close(fig)


if __name__ == "__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--steps",type=int,default=1000);ap.add_argument("--analyze-only",action="store_true");ap.add_argument("--block-budget",action="store_true")
    args=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    h=historical()
    r=json.loads((OUT/"probe_rows.json").read_text()) if args.analyze_only and (OUT/"probe_rows.json").exists() else ([] if args.analyze_only else experiment(args.steps,args.block_budget))
    if not args.block_budget:figures(h,r)
