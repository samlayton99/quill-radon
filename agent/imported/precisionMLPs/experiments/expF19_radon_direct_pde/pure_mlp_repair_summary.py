"""Plot existing ordinary-network audits; never launches or fits a PDE."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]/'results/checkpoint_F_applications/expF19_radon_direct_pde'


def main():
    rows=[json.loads(p.read_text()) for p in (ROOT/'ridge_pinn').glob('ridge_pinn_*_p*_m*_n*.json')]
    fig,axes=plt.subplots(1,3,figsize=(14,4.4),constrained_layout=True)
    specs=[('resolution','coefficient_count','Solved readout coordinates P','Increase the approximation space'),
           ('centers','interior_centers_per_direction','Centers per direction N','Resolve each ridge profile'),
           ('directions','directions','Directions M','Resolve the angular content')]
    for ax,(label,key,xlabel,title) in zip(axes,specs):
        selected=sorted([r for r in rows if r.get('label')==label and r['features']['backend']=='quill'],key=lambda r:r['features'][key])
        x=[r['features'][key] for r in selected]
        y=[r['errors']['ordinary_flat_mlp_relative_l2'] for r in selected]
        ax.semilogy(x,y,'o-',color='#6548a3',lw=2,markersize=6)
        ax.set(xlabel=xlabel,ylabel='Relative L2 solution error',title=title)
        ax.grid(alpha=.2)
        if label=='resolution':
            ax.annotate(f'{y[-1]:.1e}',(x[-1],y[-1]),xytext=(-48,17),textcoords='offset points')
        else:
            ax.text(.97,.96,'Fixed profile degree p = 12',ha='right',va='top',transform=ax.transAxes,fontsize=10)
    fig.suptitle('Nonlinear PDE solved from zero by a flat tanh MLP\nErrors measured on independent points using ordinary network evaluation',fontsize=14)
    out=ROOT/'pure_mlp_repair';out.mkdir(exist_ok=True)
    fig.savefig(out/'flat_network_resolution.png',dpi=180);plt.close(fig)


if __name__=='__main__':main()
