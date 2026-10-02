"""Read saved diagnostics and native runs; make a ledger and comparison figure.

No fitting, reevaluation, or new reference data are performed here.
"""
from __future__ import annotations

import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/codex-route2-gap-mpl')
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from route2_gap_diagnosis import OUT, ROOT, dump, log, stamp


def collect():
    rows=[]
    for folder in sorted(OUT.iterdir()):
        if not folder.is_dir() or not folder.name.startswith(('G','P')):
            continue
        protocol=json.loads((folder/'protocol.json').read_text()) if (folder/'protocol.json').exists() else {}
        path=folder/'result.json'
        if not path.exists():
            failed=folder/'failure.json'
            rows.append(dict(run_id=folder.name,status='failed' if failed.exists() else 'pending',
                source=str((failed if failed.exists() else folder).relative_to(ROOT))))
            continue
        result=json.loads(path.read_text())
        config=protocol or result.get('config',{})
        options=config.get('options',{})
        variants=result.get('variants',[result])
        for variant in variants:
            rows.append(dict(run_id=folder.name,variant=variant.get('kind',''),
                case=config.get('case',result.get('case','diffusion_c1000' if folder.name.startswith('P') else '')),
                coordinate_chart=config.get('coordinates',options.get('coordinates',result.get('coordinate_chart',''))),
                degree=result.get('degree',config.get('degree','')),centers=config.get('centers',options.get('centers','')),
                lambda_value=config.get('lam',options.get('lam','')),status=result.get('status',''),
                solver_status=result.get('solver_status',''),
                field_relative_l2=variant.get('ordinary_relative_l2',''),raw_pde_rms=variant.get('ordinary_pde_rms',''),
                boundary_rms=variant.get('boundary_rms',''),ideal_field_relative_l2=variant.get('ideal_relative_l2',''),
                ordinary_vs_ideal_relative_l2=variant.get('ordinary_vs_ideal_relative_l2',''),
                readout_l1=variant.get('readout_l1',''),condition=variant.get('preconditioned_condition',result.get('condition','')),
                krylov_iterations=variant.get('lsmr_iterations',sum(h.get('solver_work',{}).get('lsmr_iterations',0) for h in result.get('history',[]))),
                neurons=result.get('neurons',''),coordinates=result.get('coordinates',''),
                seconds=result.get('seconds',''),peak_rss_mib=result.get('process_peak_rss_bytes',0)/1024**2,
                scope='native streamed' if 'inner_history' in result else 'dense diagnostic' if folder.name[:1] in ('G','P') else '',
                source=str(path.relative_to(ROOT))))
    return rows


def main():
    began=stamp()
    rows=collect()
    columns=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/'run_ledger.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(rows)
    dump(OUT/'run_ledger.json',dict(generated=began,rows=rows,
        note='Saved-result assembly only. Timings include overlapping single-thread processes and are not isolated benchmarks.'))
    by_id={r['run_id'].split('_')[0]:r for r in rows if not r.get('variant')}
    native_groups=[('Bounded work\ndisk coordinates','G013','G014'),
                   ('Bounded work\nbox coordinates','G017','G018'),
                   ('Tight solve\nfixed degree 32','G021','G020'),
                   ('Tight solve\nadaptive degree','G025','G024')]
    available=[g for g in native_groups if all(by_id.get(k,{}).get('field_relative_l2') for k in g[1:])]
    fig,axes=plt.subplots(1,2,figsize=(14,5.6),layout='constrained')
    colors=['#2864b6','#bd4e34']
    for ax,key,title in zip(axes,['field_relative_l2','raw_pde_rms'],
            ['Function accuracy','Raw PDE residual — separate quantity']):
        for i,label in enumerate(['Diffusion contrast 10','Diffusion contrast 1000']):
            y=[float(by_id[g[i+1]][key]) for g in available]
            ax.plot(range(len(y)),y,'o-',color=colors[i],label=label,lw=2,ms=7)
            for j,val in enumerate(y):
                other=float(by_id[available[j][2-i]][key])
                ax.annotate(f'{val:.1e}',(j,val),xytext=(0,11 if val>=other else -17),
                    textcoords='offset points',ha='center',fontsize=9,color=colors[i])
        ax.set_yscale('log');ax.set_xticks(range(len(available)),[g[0] for g in available],fontsize=9)
        ax.grid(axis='y',which='major',alpha=.22);ax.spines[['top','right']].set_visible(False)
        ax.set_title(title,fontsize=13,pad=14);ax.margins(x=.12,y=.28)
    axes[0].set_ylabel('Relative L2 field error on 2,048 held-out points')
    axes[0].axhline(1e-14,color='#444444',ls='--',lw=1,label='Requested field-accuracy level: 1e-14')
    axes[1].set_ylabel('Unscaled PDE RMS on 257 held-out points')
    axes[0].legend(loc='upper right',fontsize=9)
    fig.suptitle('Native Radon coefficient discovery: same equations, different solve policies\n'
        'Ordinary one-hidden-layer tanh evaluation; PDE + boundary fitting only',fontsize=14)
    for suffix in ['png','pdf']:
        fig.savefig(OUT/f'coefficient_access_progress.{suffix}',dpi=180)
    plt.close(fig)
    log(f'\n### Saved-result assembly — {began}\n\nWrote `run_ledger.csv`, `run_ledger.json`, '
        f'and `coefficient_access_progress.png/.pdf` from {len(rows)} saved result/variant rows. '
        'No fitting or new scientific evaluation. Timings are not isolated performance benchmarks. '
        'The ledger makes earlier interleaved START/FINISH entries unambiguous through their result paths.')
    print(json.dumps(dict(ledger_rows=len(rows),native_groups=len(available),figure=str(OUT/'coefficient_access_progress.png'))))


if __name__=='__main__':main()
