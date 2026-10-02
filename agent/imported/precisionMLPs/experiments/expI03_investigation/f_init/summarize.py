"""Render the completed F initialization audit without running more training."""
import json
import numpy as np
from probe import OUT

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

full=json.loads((OUT/"probe_rows.json").read_text())
dense=json.loads((OUT/"block_dense_rows.json").read_text())
tied=json.loads((OUT/"tied_rows.json").read_text())
sampled=json.loads((OUT/"fixed_sampled_rows.json").read_text())
uniform=json.loads((OUT/"fixed_uniform_rows.json").read_text())
groups=[("Dense, default",dense,"two_match_block_std"),
        ("Dense, QI both",dense,"two_match_block_qi2"),
        ("Block, sampled bank",sampled,"two_same_qi2"),
        ("Block, uniform bank",uniform,"two_same_qi2")]
fig,axs=plt.subplots(1,3,figsize=(12,4))
summ=[]
for ax,task in zip(axs,["airfoil","parkinsons","bike_sharing"]):
    for i,(label,rows,cfg) in enumerate(groups):
        rs=[r for r in rows if r["task"]==task and r["config"]==cfg]
        v=np.array([r["selected_test_mse"] for r in rs])
        ax.bar(i,v.mean(),alpha=.65,color=f"C{i}")
        ax.scatter([i]*len(v),v,color="k",s=18,zorder=3)
        summ.append(dict(task=task,model=label,params=rs[0]["params"],test_mse_mean=float(v.mean()),test_mse_std=float(v.std())))
    ax.set_xticks(range(4),[g[0] for g in groups],rotation=35,ha="right",fontsize=8)
    ax.set_ylim(bottom=0);ax.set_title(task);ax.set_ylabel("Test MSE / train target variance")
fig.suptitle("F-inspired fixed banks: 3 seeds, approximately equal parameter counts")
fig.tight_layout();fig.savefig(OUT/"fixed_bank_comparison.png",dpi=160);plt.close(fig)
(OUT/"fixed_bank_summary.json").write_text(json.dumps(summ,indent=2))

fig,axs=plt.subplots(1,3,figsize=(11,3.5))
for ax,task in zip(axs,["airfoil","parkinsons","bike_sharing"]):
    for i,cfg in enumerate(["one_std","one_qi"]):
        rs=[r for r in full if r["task"]==task and r["config"]==cfg]
        initial=[r["initial"]["frozen_test_mse"] for r in rs]
        final=[r["final"]["frozen_test_mse"] for r in rs]
        ax.bar(i-.18,np.mean(initial),width=.36,color="C0",label="Initial frozen fit" if i==0 else None)
        ax.bar(i+.18,np.mean(final),width=.36,color="C1",label="Trained frozen fit" if i==0 else None)
    ax.set_xticks([0,1],["Standard init","QI init"]);ax.set_title(task)
    ax.set_ylabel("Test MSE / train target variance");ax.legend(fontsize=8)
fig.suptitle("High-dimensional QI init does not start with a better frozen dictionary")
fig.tight_layout();fig.savefig(OUT/"initial_dictionary.png",dpi=160);plt.close(fig)
print(json.dumps(summ,indent=2))
