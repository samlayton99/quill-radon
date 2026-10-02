"""Plot saved ordinary-network results; no training or new PDE solves."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from route2_burgers_study import OUT,raw_values,reference


def main():
    paths=sorted(OUT.glob('route2_burgers_*_p*_n*_l*_s*.json'))
    results={p.stem:json.loads(p.read_text()) for p in paths}
    baseline=sorted([(k,r) for k,r in results.items() if '_global_' in k],key=lambda kv:kv[1]['degree'])
    fig,ax=plt.subplots(1,2,figsize=(12,4.5))
    ax[0].semilogy([r['degree'] for _,r in baseline],[r['global_raw_relative_l2'] for _,r in baseline],'-o',label='One ordinary tanh MLP')
    for key,r in results.items():
        if '_slabs_' in key:
            ax[0].semilogy(r['degree'],r['global_raw_relative_l2'],'s',color='#438c55',markersize=8,label=f'{r["intervals"]} time-slab MLPs')
        elif '_faces_' in key:
            ax[0].semilogy(r['degree'],r['global_raw_relative_l2'],'*',color='#d66b2b',markersize=12,label='Face PDE control (budget exhausted)')
    ax[0].set(xlabel='Maximum coordinate degree',ylabel='Raw exported model relative L2',title='Unforced Burgers IVP, physics and IC/BC only')
    times=np.linspace(0,1,41);curves=[]
    for key,r in results.items():
        if r['degree'] not in (32,36) and '_slabs_' not in key:continue
        nets=[dict(np.load(OUT/(key+f'_slab{s}.npz'))) for s in range(r['intervals'])]
        errors=[]
        for t in times:
            slab=min(int(t*r['intervals']),r['intervals']-1)
            points=np.column_stack([np.linspace(-.5,.5,257),np.full(257,t)])
            pred=raw_values(nets[slab],points);truth=reference(points)
            errors.append(float(np.linalg.norm(pred-truth)/np.linalg.norm(truth)))
        label=f'p={r["degree"]}'+(f', {r["intervals"]} MLPs' if r['intervals']>1 else ', one MLP')
        if '_faces_' in key:label+=' + face PDE (budget exhausted)'
        color=('#d66b2b' if '_faces_' in key else '#438c55' if r['intervals']>1 else '#2765a5' if r['degree']==36 else '#71a6d2')
        ax[1].semilogy(times,errors,label=label,color=color)
        curves.append(dict(stem=key,times=times.tolist(),relative_errors=errors))
    ax[1].set(xlabel='Physical time',ylabel='Raw exported model spatial relative L2',title='Fresh spatial grid, including both endpoints')
    for a in ax:a.grid(alpha=.2);a.legend(fontsize=8)
    fig.suptitle('ν = 0.1; x ∈ [−0.5, 0.5], t ∈ [0, 1]; ordinary float64 tanh exports')
    fig.tight_layout();fig.savefig(OUT/'route2_burgers_summary.png',dpi=190);plt.close(fig)
    (OUT/'route2_burgers_time_validation.json').write_text(json.dumps(curves,indent=2))


if __name__=='__main__':main()
