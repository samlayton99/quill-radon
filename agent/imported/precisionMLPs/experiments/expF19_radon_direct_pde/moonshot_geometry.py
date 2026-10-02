"""Representation-only probe inspired by published concentrating-core scales.

NOT the paper's Navier--Stokes solution or its smooth forcing. The synthetic
axisymmetric Gaussian swirl is divergence-free, has a growing peak and shrinking
energy, but lacks the paper's axial flow, profiles and oscillatory corrections.
This isolates whether center/width adaptation can represent a shrinking core.
"""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/codex-moonshot-mpl')
from pathlib import Path
import time,json
import numpy as np
from scipy.stats import qmc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from quill_boundary import encode

OUT=Path(__file__).resolve().parents[2]/'results/checkpoint_F_applications/expF19_radon_direct_pde/adaptive_followup'
SOURCE='https://cdn.openai.com/pdf/32d9f210-8b73-45e0-91bc-82a30aef8a9a/navier-stokes.pdf'


def profile(z):
    z=np.asarray(z)
    g=np.exp(-z*z/2)
    return np.stack([g,z*g],axis=-1)


def field(points,tau,h=.005):
    scale=np.array([tau**.5,tau**.5,tau**(.5-h)])
    xyz=points/scale
    g=np.exp(-np.sum(xyz*xyz,axis=1)/2)
    amplitude=tau**(-.5-h)
    return amplitude*np.column_stack([-xyz[:,1]*g,xyz[:,0]*g,np.zeros(len(points))])


def evaluate(points,enc_r,enc_z,scales,amplitude):
    x=enc_r.evaluate(points[:,0]/scales[0]);y=enc_r.evaluate(points[:,1]/scales[1])
    z=enc_z.evaluate(points[:,2]/scales[2])
    return amplitude*np.column_stack([-x[:,0]*y[:,1]*z[:,0],x[:,1]*y[:,0]*z[:,0],np.zeros(len(points))])


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    core=8*(qmc.Sobol(3,scramble=True,seed=603).random_base2(10)-.5)
    global_points=2*np.pi*(qmc.Sobol(3,scramble=True,seed=604).random_base2(10)-.5)
    rows=[];h=.005;began=time.perf_counter()
    normalized=encode(profile,384,.2,interval=(-8,8),halo=int(np.ceil(np.sqrt(385))))
    normalized_seconds=time.perf_counter()-began
    for tau in [1.,.1,.01,.001,.0001]:
        lengths=np.array([tau**.5,tau**.5,tau**(.5-h)])
        # Keep core and global checks separate, so uniform samples cannot miss the core.
        points=core*lengths
        points=points[np.max(abs(points),axis=1)<=np.pi]
        truth=field(points,tau,h);amplitude=tau**(-.5-h)
        for n in [193,385,769,1537]:
            start=time.perf_counter()
            with np.errstate(over='ignore',invalid='ignore'):
                radial=encode(lambda z:profile(z/lengths[0]),n-1,.2,interval=(-np.pi,np.pi),halo=int(np.ceil(np.sqrt(n))))
                axial=encode(lambda z:profile(z/lengths[2]),n-1,.2,interval=(-np.pi,np.pi),halo=int(np.ceil(np.sqrt(n))))
                values=evaluate(points,radial,axial,np.ones(3),amplitude)
            with np.errstate(over='ignore',invalid='ignore'):
                error=float(np.linalg.norm(values-truth)/np.linalg.norm(truth))
            rows.append(dict(tau=tau,method='fixed_global_grid',interior_centers=n,neurons=2*len(radial.centers)+len(axial.centers),
                             core_relative_error=error if np.isfinite(error) else None,
                             finite=bool(np.isfinite(error)),setup_and_core_evaluation_seconds=time.perf_counter()-start))
        start=time.perf_counter()
        values=evaluate(points,normalized,normalized,lengths,amplitude)
        expected_peak=amplitude*np.exp(-.5)
        peak_point=np.array([[lengths[0],0,0]])
        peak=float(np.linalg.norm(evaluate(peak_point,normalized,normalized,lengths,amplitude)))
        global_truth=field(global_points,tau,h)
        global_values=evaluate(global_points,normalized,normalized,lengths,amplitude)
        rows.append(dict(tau=tau,method='scaled_geometry',interior_centers=385,neurons=3*len(normalized.centers),
                         core_relative_error=float(np.linalg.norm(values-truth)/np.linalg.norm(truth)),
                         global_absolute_error_over_peak=float(np.max(abs(global_values-global_truth))/expected_peak),
                         peak=peak,exact_peak=expected_peak,
                         exact_energy_R3=float(.5*np.pi**1.5*amplitude**2*np.prod(lengths)),
                         radial_width=float(lengths[0]),axial_width=float(lengths[2]),
                         physical_gamma=float(normalized.gamma/lengths[0]),
                         # Axisymmetric pure swirl has no tangential convective
                         # or pressure term. At r=ell_r,z=0 this exact synthetic
                         # profile therefore REQUIRES the following singular
                         # tangential force for viscosity .05. It is not the
                         # paper's smooth-forced construction.
                         required_tangential_force_at_peak_nu_005=float(
                             expected_peak*((.5+h+3*.05)/tau+.05/lengths[2]**2)),
                         evaluation_seconds=time.perf_counter()-start))
    output=dict(rows=rows,source=SOURCE,source_scope='Only core length and velocity exponents from Section2.1/3.1; all Gaussian profiles are our synthetic control.',
                h=h,normalized_construction_seconds=normalized_seconds,
                no_PDE_solve=True,no_paper_forcing_implemented=True,
                synthetic_force_warning='For this axisymmetric pure swirl, tangential convection and pressure vanish. At r=ell_r,z=0 the required force is peak*((.5+h+3*nu)/tau+nu/ell_z^2), which diverges. Thus the representation probe does not reproduce smooth-forced NS blowup.',
                interpretation='Known coordinate rescaling can preserve representation accuracy at fixed neuron count. It does not discover the coordinates or reconstruct oscillatory NS correction fields.')
    (OUT/'moonshot_geometry_metrics.json').write_text(json.dumps(output,indent=2,allow_nan=False))
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),constrained_layout=True)
    for n in [193,385,769,1537]:
        r=[x for x in rows if x['method']=='fixed_global_grid' and x['interior_centers']==n]
        axes[0].loglog([x['tau'] for x in r],[min(10.,x['core_relative_error']) if x['core_relative_error'] is not None else 10. for x in r],'-o',label=f'Fixed grid N={n}')
    r=[x for x in rows if x['method']=='scaled_geometry']
    axes[0].loglog([x['tau'] for x in r],[x['core_relative_error'] for x in r],'-o',lw=2.5,label='Scaled geometry N=385')
    axes[0].set(xlabel='Time remaining, tau (smaller = narrower)',ylabel='Core relative field error',title='Synthetic shrinking vortex: representation only')
    axes[0].set_ylim(3e-16,100.)
    axes[0].text(.04,.93,'Errors above 10 clipped; nonfinite encodings shown at 10',transform=axes[0].transAxes,fontsize=7)
    axes[0].invert_xaxis();axes[0].legend(fontsize=8)
    axes[1].loglog([x['tau'] for x in r],[x['peak'] for x in r],'-o',label='Constructed peak speed')
    axes[1].loglog([x['tau'] for x in r],[x['exact_energy_R3'] for x in r],'-s',label='Analytic kinetic energy')
    axes[1].set(xlabel='Time remaining, tau',title='Growing speed with shrinking energy')
    axes[1].invert_xaxis();axes[1].legend(fontsize=8)
    sample=np.linspace(-.3,.3,501)
    for tau in [.1,.01,.001]:
        lengths=np.array([tau**.5,tau**.5,tau**(.5-h)])
        values=evaluate(np.column_stack([sample,np.zeros(len(sample)),np.zeros(len(sample))]),normalized,normalized,lengths,tau**(-.5-h))
        axes[2].plot(sample,values[:,1],label=f'tau={tau}')
    axes[2].set(xlabel='x at y=z=0',ylabel='Tangential velocity',title='Actual constructed kernel field')
    axes[2].legend(fontsize=8)
    for ax in axes:ax.grid(alpha=.2)
    fig.savefig(OUT/'moonshot_geometry.png',dpi=180)
    print(json.dumps(output,indent=2))


if __name__=='__main__':run()
