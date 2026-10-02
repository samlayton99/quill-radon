import sys,json,os,tempfile
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR",str(Path(tempfile.gettempdir())/"precisionmlps-mpl"))
import numpy as np,torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
sys.path.insert(0,str(Path(__file__).resolve().parent))
from route2_battletest_inverse_burgers_gate import OUT,oracle,predict
root=OUT/'burgers_reduced_forward_gates'
rows=[json.loads(p.read_text()) for p in sorted(root.glob('nu*/metrics.json'))]
(root/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
fig,axes=plt.subplots(1,2,figsize=(11,4.5))
for ax,row in zip(axes,rows):
 nu=row['viscosity'];data=torch.load(root/f'nu{nu:g}'/'ordinary_mlp.pt',weights_only=True)
 model=torch.nn.Sequential(torch.nn.Linear(2,data['hidden']),torch.nn.Tanh(),torch.nn.Linear(data['hidden'],1)).double()
 model.load_state_dict(data['state_dict'])
 for index,t in enumerate((.1,.5,.9)):
  x=np.column_stack([np.linspace(-1,1,501),np.full(501,t)])
  ax.plot(x[:,0],predict(model,x)[:,0],color=f'C{index}',lw=1.7)
  ax.plot(x[:,0],oracle(x,nu)[:,0],color=f'C{index}',ls='--',lw=1)
 ax.set(xlabel='x',ylabel='u',title=f"ν={nu:g}: field error {100*row['heldout_relative_l2']:.1f}%, PDE RMS {row['ordinary_pde_rms']:.3f}")
 ax.grid(alpha=.25)
handles=[Line2D([],[],color=f'C{i}',label=f't={t:g}') for i,t in enumerate((.1,.5,.9))]
handles += [Line2D([],[],color='black',label='Native MLP'),Line2D([],[],color='black',ls='--',label='Cole–Hopf audit reference')]
fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.92),ncol=5,frameon=False,fontsize=9)
fig.suptitle('Physics-only Burgers gates: both remained unresolved after 600 seconds',y=.99)
fig.text(.5,.015,'Zero readouts initially. No sensors or reference field entered either solve. These failures block reduced inverse scoring.',ha='center',fontsize=9)
fig.subplots_adjust(left=.07,right=.985,top=.76,bottom=.16,wspace=.25)
fig.savefig(root/'forward_gate_summary.png',dpi=180,bbox_inches='tight')
