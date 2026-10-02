"""Figures from saved native PDE solves; no training or fitted references."""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/quill-hardening-mpl')
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle
from route2_hardening_disk import exact

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/checkpoint_F_applications/expF19_radon_direct_pde/route2_hardening'


def load(path):
    return json.loads(path.read_text())


def disk_figure(stem='parity_cold_p14_n257'):
    directory=OUT/'disk_ns'
    result=load(directory/(stem+'.json'))
    state=torch.load(directory/(stem+'_torch.pt'),weights_only=True,map_location='cpu')
    model=torch.nn.Sequential(torch.nn.Linear(2,len(state['0.bias']),dtype=torch.float64),
        torch.nn.Tanh(),torch.nn.Linear(len(state['0.bias']),3,dtype=torch.float64))
    model.load_state_dict(state);model.requires_grad_(False)
    line=np.linspace(-1,1,129);xx,yy=np.meshgrid(line,line)
    mask=xx*xx+yy*yy<=1
    points=torch.tensor(np.column_stack((xx[mask],yy[mask])),dtype=torch.float64)
    values=torch.cat([model(p) for p in points.split(128)]).numpy()
    target=exact(points).numpy()
    speed=np.full(xx.shape,np.nan);speed[mask]=np.linalg.norm(values[:,:2],axis=1)
    error=np.full(xx.shape,np.nan);error[mask]=np.linalg.norm(values[:,:2]-target[:,:2],axis=1)
    u=np.full(xx.shape,np.nan);v=u.copy();u[mask]=values[:,0];v[mask]=values[:,1]
    fig,axes=plt.subplots(1,3,figsize=(16,4.9),gridspec_kw={'width_ratios':[1,1,1.1]})
    im=axes[0].pcolormesh(xx,yy,speed,cmap='viridis',shading='auto',rasterized=True)
    axes[0].streamplot(line,line,np.ma.masked_invalid(u),np.ma.masked_invalid(v),
                       color='white',density=.85,linewidth=.6,arrowsize=.7)
    fig.colorbar(im,ax=axes[0],shrink=.8,label='Speed')
    im=axes[1].pcolormesh(xx,yy,np.maximum(error,1e-16),cmap='magma',shading='auto',
                        norm=LogNorm(1e-16,3e-14),rasterized=True)
    fig.colorbar(im,ax=axes[1],shrink=.8,label='Absolute velocity-vector error')
    for ax in axes[:2]:
        ax.add_patch(Circle((0,0),1,fill=False,color='#555555',lw=1))
        ax.set(xlim=(-1.03,1.03),ylim=(-1.03,1.03),xlabel='x',ylabel='y',aspect='equal')
    axes[0].set_title('Ordinary tanh network: no-slip flow')
    axes[1].set_title('Error against analytical check')
    history=result['history']
    steps=[r['iteration'] for r in history]
    residual=[r['maximum_scaled_block_rms'] for r in history]
    axes[2].semilogy(steps,residual,'o-',color='#1565c0',lw=2)
    axes[2].axhline(2e-14,color='#666666',ls='--',lw=1)
    axes[2].set(xlabel='Newton correction',ylabel='Largest scaled physics-block RMS',
                title='Solve starts from zero coefficients',xticks=steps,ylim=(1e-16,20))
    axes[2].grid(alpha=.2)
    audit=result['ordinary_audit']
    fig.suptitle('Native nonlinear 2D Navier–Stokes · 4,365 tanh neurons · 360 independent coefficients',fontsize=14)
    fig.text(.5,.015,
        f"Fresh-point ordinary-network check: velocity relative L2 {audit['velocity_relative_l2']:.2e}; "
        f"pressure {audit['pressure_relative_l2']:.2e}; momentum RMS {audit['momentum_rms']:.2e}.\n"
        'Prescribed body force + zero wall velocity + pressure gauge; no interior labels or external PDE solve. Dashed line: solve tolerance.',
        ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.1,1,.94))
    fig.savefig(OUT/'native_ns_disk.png',dpi=180,bbox_inches='tight')
    fig.savefig(OUT/'native_ns_disk.pdf',bbox_inches='tight')
    plt.close(fig)


def resolution_figure():
    names=['pilot_p6_n257','pilot_p10_n257','refined_p14_n257']
    rows=[load(OUT/'disk_ns'/(name+'.json')) for name in names]
    fig,ax=plt.subplots(figsize=(8,4.8))
    for key,label,marker in [('velocity_relative_l2','Velocity relative L2','o'),
                              ('pressure_relative_l2','Pressure relative L2','s'),
                              ('momentum_relative_rms','Momentum relative RMS','^')]:
        ax.semilogy([r['features']['neurons'] for r in rows],
                     [r['ordinary_audit'][key] for r in rows],marker+'-',label=label,lw=2)
    ax.axhline(1e-14,color='#999999',ls='--',lw=1)
    ax.set(xlabel='Tanh neurons (257 interior centers per direction)',ylabel='Held-out error',
           title='Native no-slip Navier–Stokes: refinement reaches the arithmetic floor')
    ax.grid(alpha=.2)
    ax.legend(loc='lower center',bbox_to_anchor=(.5,1.02),ncol=3,frameon=False,fontsize=9)
    ax.set_title('Native no-slip Navier–Stokes: refinement reaches the arithmetic floor',pad=44)
    fig.text(.5,.015,'Degrees 6 → 10 → 14; each continuation starts from the preceding native PDE iterate.\n'
             'Degree 6 and 10 were time-limited: this is an observed run sequence, not an asymptotic rate claim.',
             ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.12,1,1))
    fig.savefig(OUT/'native_ns_resolution.png',dpi=180,bbox_inches='tight')
    plt.close(fig)


def extensions_figure():
    inverse=load(OUT/'inverse_ns/inverse_ideal_refresh_p14_n257.json')
    trajectory=load(OUT/'inverse_ns/inverse_ideal_refresh_p14_n257_progress.json')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    axes[0].plot([r['iteration'] for r in trajectory],[r['viscosity'] for r in trajectory],
                 'o-',color='#00796b',lw=2,label='Recovered viscosity')
    axes[0].axhline(.3,color='#555555',ls='--',label='Prescribed viscosity')
    axes[0].set(xlabel='Newton correction',ylabel='Viscosity',ylim=(.13,.32),
                title='Inverse problem: five velocity locations')
    forward=load(OUT/'disk_ns/parity_cold_p14_n257.json')
    lower=load(OUT/'disk_ns/lower_viscosity_cold_p14_n257.json')
    for row,label,marker in [(forward,'Forward, viscosity 0.3','o'),
                            (lower,'Forward, viscosity 0.03','s'),
                            (inverse,'Unknown viscosity + fields','^')]:
        hist=row['history']
        axes[1].semilogy([r['iteration'] for r in hist],
                          [r['maximum_scaled_block_rms'] for r in hist],marker+'-',label=label)
    axes[1].set(xlabel='Newton correction',ylabel='Largest scaled physics-block RMS',
                ylim=(1e-16,20),title='All three start from zero readouts')
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(loc='lower center',bbox_to_anchor=(.5,1.02),frameon=False,fontsize=9)
        ax.set_title(ax.get_title(),pad=70)
    fig.text(.5,.012,
        'Ordinary exported-network checks: velocity relative L2 is 3.5e−15 in all three cases.\n'
        'Manufactured steady 2D tests; the inverse case uses ten scalar measurements, with prescribed forcing held fixed.',
        ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.13,1,1))
    fig.savefig(OUT/'native_ns_extensions.png',dpi=180,bbox_inches='tight')
    fig.savefig(OUT/'native_ns_extensions.pdf',bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':
    torch.set_num_threads(1)
    disk_figure();resolution_figure();extensions_figure()
