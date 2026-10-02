"""Does keeping F04's initial ridge sharing constrain its subsequent learning?

Reuses probe.py's data splits, seeds, batches, width and Adam hyperparameters.
The initial function is exactly equal to the free model; only direction sharing
persists during training. Biases and readout remain free, making this a narrower
intervention than adopting an I02 fixed bank.
"""
import json
import math
import time
import argparse

import torch
import torch.nn.functional as F

from probe import ROOT, OUT, SimpleMLP, SimpleMLP2, qi_ridge_init_layer_, diagnostic


class TiedLinear(torch.nn.Module):
    def __init__(self, linear, p):
        super().__init__()
        n = linear.weight.shape[0]
        sizes = [min(p, n-start) for start in range(0,n,p)]
        self.register_buffer("mapping",torch.arange(len(sizes)).repeat_interleave(torch.tensor(sizes)))
        self.directions=torch.nn.Parameter(linear.weight.detach()[::p].clone())
        self.bias=torch.nn.Parameter(linear.bias.detach().clone())

    @property
    def weight(self):
        return self.directions[self.mapping]

    def forward(self,x):
        return F.linear(x,self.weight,self.bias)


class FixedBankLinear(TiedLinear):
    """Fixed bank offsets; only one shared projection offset is learned per ridge.

    This is a conventional projection -> fixed 1D bank: a change to the free
    projection moves occupancy, while the bank centers/inner scale stay fixed.
    For uniform initialization these offsets are independent of data ranges.
    """
    def __init__(self, linear, p):
        super().__init__(linear,p)
        offset=self.bias.detach().clone()
        del self.bias
        self.register_buffer("bank_offset",offset)
        self.projection_bias=torch.nn.Parameter(torch.zeros(len(self.directions)))

    def forward(self,x):
        bias=self.bank_offset+self.projection_bias[self.mapping]
        return F.linear(x,self.weight,bias)


def run(fixed_banks=False, uniform=False):
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    rows=[]
    width,steps=128,5000
    for name in ["airfoil","parkinsons","bike_sharing"]:
        d=torch.load(ROOT/f"data/cache_all20/{name}.pt",weights_only=True)
        for seed in [0,1,2]:
            gen=torch.Generator().manual_seed(1000+seed)
            order=torch.randperm(len(d["xtr"]),generator=gen)
            valid=order[:max(128,len(order)//5)];train=order[len(valid):][:4096]
            x,xv,xt=d["xtr"][train].double(),d["xtr"][valid].double(),d["xte"].double()
            ym,ys=d["ytr"][train].double().mean(),d["ytr"][train].double().std()
            y,yv,yt=[(a.double()-ym)/ys for a in (d["ytr"][train],d["ytr"][valid],d["yte"])]
            batches=torch.randint(len(x),(steps,128),generator=gen)
            for config in (["two_same_qi2"] if fixed_banks else ["one_qi","two_same_qi2"]):
                t0=time.perf_counter();torch.manual_seed(seed)
                cls=SimpleMLP if config=="one_qi" else SimpleMLP2
                model=cls(d["d_in"],width,1,activation="tanh")
                p=math.isqrt(width)
                qi_ridge_init_layer_(model.fc1,x,centers_per_dir=p,uniform_centers=uniform)
                if config=="two_same_qi2":
                    with torch.no_grad():
                        qi_ridge_init_layer_(model.fc2,model.act(model.fc1(x)),centers_per_dir=p,uniform_centers=uniform)
                with torch.no_grad():
                    pred0=model(x[:128]).clone()
                layer_cls=FixedBankLinear if fixed_banks else TiedLinear
                model.fc1=layer_cls(model.fc1,p)
                if config=="two_same_qi2":model.fc2=layer_cls(model.fc2,p)
                with torch.no_grad():
                    delta=(model(x[:128])-pred0).abs().max().item()
                    assert delta == 0,delta
                opt=torch.optim.Adam(model.parameters(),lr=1e-3)
                trace=[];best_valid=float("inf");best_test=None
                for step,idx in enumerate(batches):
                    opt.zero_grad(set_to_none=True)
                    loss=(model(x[idx]).squeeze(-1)-y[idx]).square().mean()
                    loss.backward();opt.step()
                    if (step+1)%100==0:
                        with torch.no_grad():
                            ve=(model(xv).squeeze(-1)-yv).square().mean().item()
                            te=(model(xt).squeeze(-1)-yt).square().mean().item()
                        trace.append(dict(step=step+1,valid_mse=ve,test_mse=te))
                        if ve<best_valid:best_valid,best_test=ve,te
                r=dict(task=name,seed=seed,config=config,fixed_banks=fixed_banks,uniform=uniform,params=sum(a.numel() for a in model.parameters()),
                       init_max_difference=delta,trace=trace,selected_test_mse=best_test,
                       final_test_mse=trace[-1]["test_mse"],seconds=time.perf_counter()-t0)
                filename=("fixed_uniform_rows.json" if uniform else "fixed_sampled_rows.json") if fixed_banks else "tied_rows.json"
                rows.append(r);(OUT/filename).write_text(json.dumps(rows,indent=2))
                print(name,seed,config,"tied",best_test,"params",r["params"],"seconds",r["seconds"],flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--fixed-banks",action="store_true");ap.add_argument("--uniform",action="store_true")
    a=ap.parse_args();run(a.fixed_banks,a.uniform)
