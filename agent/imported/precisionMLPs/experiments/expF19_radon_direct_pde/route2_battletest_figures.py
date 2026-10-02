"""Comparison figures from frozen, already evaluated campaigns; never fits a model."""
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/codex-route2-battle-figures')
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'results/checkpoint_F_applications/expF19_radon_direct_pde/route2_battletest'
OUT=BASE/'comparison'
CASES=['helmholtz_n2','helmholtz_n4','diffusion_c10','diffusion_c1000',
       'allen_cahn_d001','allen_cahn_d00001']
LABELS=['Helmholtz\nn = 2','Helmholtz\nn = 4','Diffusion\ncontrast 10',
        'Diffusion\ncontrast 1,000','Allen–Cahn\nD = 0.01','Allen–Cahn\nD = 0.0001']


def forward_comparison():
    campaigns=[('60 s screening',BASE/'forward','#a5a9b4'),
               ('600 s, unchanged equations',BASE/'forward/accuracy_v2','#7160ba'),
               ('600 s, operator-normalized equations',BASE/'normalized','#18877b')]
    fig,ax=plt.subplots(figsize=(12.6,5.5))
    rows=[]
    for index,(label,path,color) in enumerate(campaigns):
        for k,name in enumerate(CASES):
            file=path/(name+'_s0.json')
            if not file.exists(): continue
            row=json.loads(file.read_text()); value=row.get('heldout_relative_l2')
            if value is None: continue
            rows.append(dict(campaign=label,case=name,relative_l2=value,
                             raw_pde_rms=row['raw_pde_rms'],status=row['status'],
                             seconds=row['all_fit_export_seconds']))
            x=k+(index-1)*.24
            ax.bar(x,value,bottom=1e-15,width=.21,color=color,
                   label=label if k==0 else None)
            ax.text(x,value*1.55,f'{value:.1e}',ha='center',va='bottom',fontsize=8,rotation=90)
    ax.set_yscale('log');ax.set_ylim(1e-15,3e4)
    ax.set_ylabel('Held-out relative L2 field error (lower is better)')
    ax.set_xticks(np.arange(len(CASES)),LABELS)
    ax.grid(axis='y',which='major',alpha=.18);ax.set_axisbelow(True)
    ax.set_title('One fixed solver across six forward problems',pad=65,fontsize=15)
    # Explicit handles preserve the legend if an early case has not completed.
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=c,label=l) for l,_,c in campaigns],
              loc='lower center',bbox_to_anchor=(.5,1.02),ncol=3,frameon=False,fontsize=9)
    fig.text(.5,.012,'All seed 0; no interior solution labels. Times are soft budgets. '
             'Blank bars mean pending evaluation; field error is separate from PDE residual.',
             ha='center',fontsize=9)
    fig.subplots_adjust(left=.09,right=.99,bottom=.16,top=.76)
    fig.savefig(OUT/'forward_comparison.png',dpi=180)
    fig.savefig(OUT/'forward_comparison.pdf');plt.close(fig)
    (OUT/'forward_comparison.json').write_text(json.dumps(rows,indent=2)+'\n')


def diffusion_fields():
    name='diffusion_c1000'
    before=BASE/'forward/accuracy_v2';after=BASE/'normalized'
    files=[p/(name+'_s0.npz') for p in (before,after)]
    reference=before/(name+'_plot_reference.npz')
    if not all(p.exists() for p in files+[reference]):return
    a,b=(np.load(p) for p in files);r=np.load(reference)
    truth=r['truth'].reshape(-1);grid=r['grid']; n=int(np.sqrt(len(truth)))
    errors=[abs(a['grid_prediction'].reshape(-1)-truth),abs(b['grid_prediction'].reshape(-1)-truth)]
    fig,axes=plt.subplots(1,3,figsize=(12.5,4.4))
    gx=grid[:,0].reshape(n,n);gy=grid[:,1].reshape(n,n)
    im=axes[0].pcolormesh(gx,gy,truth.reshape(n,n),shading='nearest',cmap='viridis')
    fig.colorbar(im,ax=axes[0],shrink=.8);axes[0].set_title('Target used only for audit')
    from matplotlib.colors import LogNorm
    for ax,error,title in zip(axes[1:],errors,['Original solve: absolute error','Normalized equations: absolute error']):
        im=ax.pcolormesh(gx,gy,np.maximum(error,1e-13).reshape(n,n),shading='nearest',
                        cmap='magma',norm=LogNorm(1e-13,1e-1))
        fig.colorbar(im,ax=ax,shrink=.8);ax.set_title(title)
    for ax in axes:ax.set_xlabel('x');ax.set_ylabel('y')
    fig.suptitle('Diffusion contrast 1,000: same target, same 9,603 neurons, different equation scaling',fontsize=13)
    fig.text(.5,.02,'The error panels share the same logarithmic color scale. '
             'Forcing and boundary data only entered the solves.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.06,1,.9]);fig.savefig(OUT/'diffusion_fields.png',dpi=180);plt.close(fig)


def initial_plane_figure():
    # Re-evaluate existing ordinary exports only; the numerical reference was
    # already quarantined after fitting by the independent forward campaign.
    import torch
    torch.set_num_threads(1)
    from route2_battletest_forward import evaluate_raw
    folder=BASE/'initial_plane'; summary=json.loads((folder/'summary.json').read_text())
    fig,axes=plt.subplots(2,3,figsize=(12,7.4))
    for row,spec in enumerate(summary):
        name=spec['case']; saved=torch.load(folder/(name+'.pt'),map_location='cpu',weights_only=True)
        model=torch.nn.Sequential(torch.nn.Linear(2,saved['width']),torch.nn.Tanh(),
                                  torch.nn.Linear(saved['width'],1)).double()
        model.load_state_dict(saved['state_dict']);model.requires_grad_(False)
        ref=np.load(BASE/'forward'/(name+'_plot_reference.npz'))
        grid=ref['grid'];truth=ref['truth'].reshape(-1);pred,_=evaluate_raw(model,grid)
        n=int(np.sqrt(len(grid)));pred=pred.reshape(-1)
        # Use the saved coordinates explicitly: horizontal time, vertical space.
        panels=[truth,pred,abs(pred-truth)]
        for col,values in enumerate(panels):
            ax=axes[row,col]
            im=ax.pcolormesh(grid[:,1].reshape(n,n),grid[:,0].reshape(n,n),values.reshape(n,n),
                            shading='nearest',cmap='coolwarm' if col<2 else 'magma',
                            vmin=-1 if col<2 else 0,vmax=1 if col<2 else 2)
            fig.colorbar(im,ax=ax,shrink=.8)
            if row==0:ax.set_title(['Independent reference','Prescribed IC + native MLP','Absolute error'][col],pad=12)
            ax.set_xlabel('Time');ax.set_ylabel('Space x')
        diffusion='.01' if row==0 else '.0001'
        axes[row,1].text(.5,1.015,f'D = {diffusion}; field error {spec["heldout_relative_l2"]:.1%}',
                        transform=axes[row,1].transAxes,ha='center',va='bottom',fontsize=10)
    # Put column names in a separate figure row; no titles collide with per-case labels.
    for ax in axes[0]:ax.set_title('')
    fig.subplots_adjust(top=.84,bottom=.08,hspace=.35,wspace=.36)
    for j,title in enumerate(['Independent reference','Prescribed IC + native MLP','Absolute error']):
        pos=axes[0,j].get_position();fig.text((pos.x0+pos.x1)/2,.89,title,ha='center',fontsize=11)
    fig.suptitle('Preserving initial data alone does not resolve the later dynamics',y=.97,fontsize=14)
    fig.savefig(folder/'initial_plane_control.png',dpi=180);plt.close(fig)


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    forward_comparison();diffusion_fields();initial_plane_figure()
