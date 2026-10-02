"""Render saved common-engine runs, including matched polynomial controls.

No solvers run here. Error bars are not implied: each entry is one deterministic
study, checked against a reference at independent points. Timings are local
observations including adaptive solves, not a controlled speed comparison.
"""
import json
import os
from pathlib import Path
import numpy as np
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/quill-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/checkpoint_F_applications/expF19_radon_direct_pde/general_residual'
CASES=[
    ('nonlinear_diffusion','Nonlinear diffusion'),
    ('irregular_variable_diffusion','Irregular domain + variable diffusion'),
    ('biharmonic_plate','Fourth-order plate equation'),
    ('burgers_ivp','Unforced viscous Burgers IVP'),
    ('wave_ivp','Wave IVP'),
    ('steady_navier_stokes','Steady 2D Navier–Stokes'),
    ('three_dimensional_ellipsoid','3D ellipsoidal domain'),
    ('inverse_diffusivity','Inverse diffusivity, 24 observations'),
    ('quasilinear_gradient_diffusion','Gradient-dependent diffusion'),
    ('mixed_robin','Mixed Dirichlet / Robin boundaries'),
    ('coupled_reaction_diffusion','Coupled reaction–diffusion'),
    ('convection_boundary_layer','Thin convection–diffusion layer'),
    ('high_contrast_diffusion','Diffusion contrast ≈ 1,000'),
]


def main():
    rows=[]
    for case,label in CASES:
        d=json.loads((OUT/f'adaptive_auto_{case}_quill.json').read_text())
        control=json.loads((OUT/f'adaptive_auto_{case}_polynomial.json').read_text())
        rows.append(dict(case=case,label=label,
            relative_l2=d['validation']['relative_l2'],
            polynomial_relative_l2=control['validation']['relative_l2'],
            seconds=d['metrics']['seconds'],degree=d['degree_selected'],
            unknowns=d['solver']['unknowns'],status=d['metrics']['status'],
            manufactured_forcing=d['manufactured_forcing'],
            validation=d['validation'],note=d['note']))
    ns=json.loads((OUT/'ns4d_metrics.json').read_text())
    rows.append(dict(case='transient_navier_stokes_3d',label='Transient 3D Navier–Stokes (4 inputs)',
        relative_l2=ns['validation']['velocity_relative_l2'],
        pressure_relative_l2=ns['validation']['pressure_relative_l2'],
        final_pressure_relative_l2=ns['validation']['final_pressure_relative_l2'],
        seconds=ns['solve_seconds'],degree=ns['selected_degree'],
        unknowns=ns['unknown_coefficients'],status=ns['status'],
        manufactured_forcing=False,note=ns['limitations']))
    result=dict(scope='13 common-policy tests plus smooth four-input 3D Navier–Stokes',
        error_definition='Independent empirical relative L2, combined across fields except transient NS shown separately',
        timing_scope='Local observations including construction/continuation; caches and machine activity differ. Not a speed benchmark.',
        geometry='Total-degree Legendre products explicitly represented using QUILL tanh banks; product gates included',
        rows=rows)
    (OUT/'general_solver_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,(ax,tm)=plt.subplots(1,2,figsize=(14,9.7),sharey=True,
                            gridspec_kw={'width_ratios':[2.25,1]})
    y=np.arange(len(rows));blue='#1565c0';gold='#c47700';gray='#9ca3af'
    for axis in (ax,tm):
        axis.set_xscale('log');axis.grid(axis='x',color='#e5e7eb',zorder=0)
        axis.set_ylim(len(rows)-.5,-.7)
        for i in range(0,len(rows),2):axis.axhspan(i-.45,i+.45,color='#f5f7fa',zorder=-1)
    ax.scatter([r['polynomial_relative_l2'] for r in rows[:-1]],y[:-1]+.1,
               marker='x',s=45,color=gray,label='Exact-polynomial control',zorder=3)
    ax.scatter([r['relative_l2'] for r in rows],y-.08,s=44,color=blue,
               label='QUILL construction',zorder=4)
    ax.scatter(rows[-1]['pressure_relative_l2'],y[-1]+.12,marker='D',s=48,
               color=gold,label='3D NS pressure',zorder=4)
    ax.set_yticks(y,[r['label'] for r in rows]);ax.tick_params(axis='y',length=0,pad=10)
    ax.set_xlim(1e-14,1e-4);ax.set_xticks([1e-14,1e-12,1e-10,1e-8,1e-6,1e-4])
    ax.set_xlabel('Relative solution error on fresh points  ·  smaller is better',labelpad=12)
    ax.set_title('Solution accuracy',loc='left',fontweight='bold',pad=16)
    handles,labels=ax.get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.64,.102),
               ncol=3,frameon=False,fontsize=10)
    tm.hlines(y,.05,[r['seconds'] for r in rows],color='#bfdbfe',lw=3,zorder=2)
    tm.scatter([r['seconds'] for r in rows],y,s=34,color=blue,zorder=3)
    for i,r in enumerate(rows):
        label=(f"{r['seconds']:.0f} s" if r['seconds']>=10 else f"{r['seconds']:.2g} s")
        tm.annotate(label,(r['seconds'],i),xytext=(6,0),
                    textcoords='offset points',va='center',fontsize=10)
    tm.set_xlim(.05,500);tm.set_xticks([.1,1,10,100]);tm.tick_params(axis='y',length=0)
    tm.set_xlabel('Observed local runtime (seconds)',labelpad=12)
    tm.set_title('Observed solve time',loc='left',fontweight='bold',pad=16)
    fig.suptitle('One residual solver across 14 PDE / inverse tests',x=.03,ha='left',
                 fontsize=19,fontweight='bold',y=.98)
    fig.text(.03,.932,'No named-equation solver dispatch. No interior solution labels in forward solves.',fontsize=11)
    fig.text(.03,.04,'3D transient NS: blue = velocity, gold = pressure over space–time. Final-time pressure error: 2.8 × 10⁻⁵.\n'
             'Smooth / resolved tests, several with manufactured forcing. Polynomial controls perform similarly.\n'
             'Timings are individual local runs, not evidence of a speed advantage over PINNs or classical solvers.',fontsize=10,color='#374151')
    fig.subplots_adjust(left=.35,right=.975,top=.875,bottom=.22,wspace=.12)
    fig.savefig(OUT/'general_solver_summary.png',dpi=180)
    fig.savefig(OUT/'general_solver_summary.pdf')
    print(json.dumps({'cases':len(rows),'statuses':sorted(set(r['status'] for r in rows)),
                      'figure':str(OUT/'general_solver_summary.png')},indent=2))


if __name__=='__main__':main()
