"""Compile frozen transfer artifacts; does not fit or evaluate any model."""
from __future__ import annotations
import os
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/codex-route2-transfer-mpl')
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/checkpoint_F_applications/expF19_radon_direct_pde/route2_transfer_2026_10_02'
LABELS = {
    'D001_semilinear_3d': 'Nonlinear elliptic · 3 inputs',
    'D002_semilinear_4d': 'Nonlinear elliptic · 4 inputs',
    'D003_semilinear_5d': 'Nonlinear elliptic · 5 inputs',
    'H001_helmholtz_n2': 'Helmholtz · 2 oscillations',
    'H002_helmholtz_n4': 'Helmholtz · 4 oscillations',
    'H003_allen_cahn_d001': 'Allen–Cahn · diffusion 0.01',
    'H004_allen_cahn_d00001': 'Allen–Cahn · diffusion 0.0001',
    'N001_ns3_smooth': 'Steady 3D Navier–Stokes · smooth',
    'N002_ns3_bubble': 'Steady 3D Navier–Stokes · wall bubble',
    'N003_ns4_transient': 'Transient 3D Navier–Stokes · 4 inputs',
    'N004_ns3_cavity': 'Physical 3D driven cavity · no field truth',
}


def load(path):
    return json.loads(path.read_text())


def compile_rows():
    rows = []
    hashes = []
    for run_id, label in LABELS.items():
        folder = OUT/run_id
        result = load(folder/'result.json')
        protocol = load(folder/'protocol.json')
        audit = result.get('ordinary_audit', result)
        field = audit.get('ordinary_relative_l2', audit.get('velocity_relative_l2'))
        pde = audit.get('ordinary_pde_rms', audit.get('momentum_rms'))
        stages = result.get('stage_history', result.get('history', []))
        row = dict(run_id=run_id, label=label, status=result['status'],
            dimension=result.get('dimension', 2), degree=result['degree'],
            neurons=result['neurons'], directions=result.get('directions', result['degree']+1),
            readout_coordinates=result.get('readout_coordinates', result.get('coordinates')),
            ordinary_field_relative_l2=field, raw_pde_rms=pde,
            pressure_relative_l2=audit.get('pressure_relative_l2'),
            divergence_rms=audit.get('divergence_rms'),
            peak_rss_mib=result.get('peak_rss_bytes', result.get('process_peak_rss_bytes'))/2**20,
            seconds=result['seconds'], solve_seconds=result['solve_seconds'],
            field_truth='unavailable' if field is None else ('independent numerical reference' if run_id.startswith(('H003','H004')) else 'analytical manufactured'),
            source_path=str((folder/'result.json').relative_to(ROOT)),
            corner_field_max=audit.get('corner_max', audit.get('corner_max_abs')),
            ideal_field_relative_l2=audit.get('ideal_relative_l2'),
            readout_l1=result.get('ordinary_readout_l1', audit.get('readout_l1')))
        if (folder/'additional_audit.json').exists():
            extra = load(folder/'additional_audit.json')
            row.update(near_edge_pde_rms=extra['checks']['near_edges']['momentum_rms'],
                corner_pde_rms=extra['checks']['corners']['momentum_rms'],
                ordinary_vs_ideal_field_max=extra['ordinary_vs_ideal_max'],
                ideal_field_relative_l2=extra.get('ideal_velocity_relative_l2'))
        if 'reference' in result:
            row['reference_resolution_gap']=result['reference'].get('reference_resolution_difference_relative_l2')
        rows.append(row)
        for filename, expected in protocol.get('source_hashes', protocol.get('source_sha256', {})).items():
            # Snapshots include runner and shared modules. This is a comparison
            # to recorded hashes, not a claim to cover every imported file.
            actual = hashlib.sha256((ROOT/filename).read_bytes()).hexdigest()
            hashes.append(dict(run_id=run_id, path=filename, unchanged=actual==expected))
    return rows, hashes


def overview(rows):
    plt.rcParams.update({'font.size':11, 'axes.spines.top':False, 'axes.spines.right':False})
    fig, axes = plt.subplots(1,3,figsize=(15,7.4),sharey=True,
        gridspec_kw={'width_ratios':[1.4,1.4,1]})
    palette=['#247bba']*3+['#ba542c']*4+['#2b8062']*4
    for i,row in enumerate(rows):
        marker='o' if row['status']=='sampled_residual_checks_passed' else 'X'
        for ax,key in zip(axes[:2],['ordinary_field_relative_l2','raw_pde_rms']):
            value=row[key]
            if value is not None:
                ax.scatter(value,i,s=65,marker=marker,color=palette[i],zorder=3)
            else:
                ax.text(.97,i,'Unknown',ha='right',va='center',transform=ax.get_yaxis_transform(),color='#555555')
        axes[2].barh(i,row['peak_rss_mib'],height=.48,color=palette[i],alpha=.8)
        axes[2].text(row['peak_rss_mib']+8,i,f"{row['peak_rss_mib']:.0f} MiB",va='center',fontsize=9)
    axes[0].set_yticks(range(len(rows)),[r['label'] for r in rows])
    axes[0].invert_yaxis()
    axes[0].set_title('Relative field error\n(velocity for Navier–Stokes)')
    axes[1].set_title('Raw PDE residual RMS\n(momentum for Navier–Stokes)')
    axes[2].set_title('Peak process memory\n(solve + ordinary-model audit)')
    for ax in axes[:2]:
        ax.set_xscale('log');ax.set_xlim(1e-17,2)
        ax.set_xticks([1e-16,1e-12,1e-8,1e-4,1]);ax.grid(axis='x',alpha=.22)
    axes[0].axvline(1e-14,color='#777777',ls='--',lw=1)
    axes[0].text(1e-14,-.68,'1e−14 goal',ha='center',fontsize=9,color='#555555')
    axes[2].set_xlim(0,max(r['peak_rss_mib'] for r in rows)*1.32)
    axes[2].set_xlabel('MiB');axes[2].grid(axis='x',alpha=.18)
    for ax in axes:
        for border in (2.5,6.5):ax.axhline(border,color='#bbbbbb',lw=.7)
    fig.suptitle('Frozen transfer test: accuracy does not transfer uniformly',fontsize=17,x=.57,y=.985)
    fig.text(.31,.035,'● All sampled solver gates passed    × Gates did not pass (budget or resolution limit)',fontsize=10)
    fig.text(.31,.011,'Raw residuals use different equation scales; compare them within each problem. One seed per case; no interior truth used in fitting.',fontsize=9)
    fig.subplots_adjust(left=.31,right=.985,top=.87,bottom=.105,wspace=.23)
    fig.savefig(OUT/'transfer_overview.png',dpi=180)
    fig.savefig(OUT/'transfer_overview.pdf')
    plt.close(fig)


def difficult_fields():
    # Scatter the held-out points directly: no interpolation and no extra solve.
    keys=['H002_helmholtz_n4','H003_allen_cahn_d001','H004_allen_cahn_d00001']
    fig,axes=plt.subplots(3,3,figsize=(12,10),layout='constrained')
    for i,key in enumerate(keys):
        data=np.load(OUT/key/'field_audit.npz')
        x=data['points'];truth=data['reference'].ravel();pred=data['prediction'].ravel()
        vmax=max(abs(truth).max(),abs(pred).max());err=pred-truth
        for j,(values,title,lim) in enumerate([(truth,'Reference',vmax),(pred,'Ordinary tanh model',vmax),(err,'Model − reference',max(abs(err).max(),1e-16))]):
            ax=axes[i,j]
            sc=ax.scatter(x[:,0],x[:,1],c=values,s=8,cmap='RdBu_r',vmin=-lim,vmax=lim,rasterized=True)
            fig.colorbar(sc,ax=ax,shrink=.8)
            if i==0:ax.set_title(title)
            ax.set_xlabel('x');ax.set_ylabel('y' if i==0 else 'time')
            if j==0:ax.text(0,1.035,LABELS[key],transform=ax.transAxes,fontsize=10)
    fig.suptitle('The difficult failures are visible in the field, not only in a tolerance flag',fontsize=14)
    fig.savefig(OUT/'difficult_fields.png',dpi=160)
    fig.savefig(OUT/'difficult_fields.pdf')
    plt.close(fig)


def main():
    rows,hashes=compile_rows()
    (OUT/'summary.json').write_text(json.dumps(dict(rows=rows,source_hash_checks=hashes,
        all_recorded_sources_unchanged=all(r['unchanged'] for r in hashes),
        scope='Frozen one-seed transfer campaign; no external reference solve supplies fitted coefficients. Postfit audits only. Shared-host timings.'),indent=2)+'\n')
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/'summary.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    overview(rows);difficult_fields()
    print(json.dumps(dict(cases=len(rows),all_recorded_sources_unchanged=all(r['unchanged'] for r in hashes))))


if __name__=='__main__':main()
