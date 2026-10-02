"""Check scaling versus angular redistribution in the composition comparison."""
from __future__ import annotations

import json
import numpy as np
import matplotlib.pyplot as plt
import spoke_radon_prediction as R


def scale_check(actual, predicted):
    a=actual.ravel();p=predicted.ravel()
    scale=float(a@p/(p@p))
    return dict(best_scale=scale,
                raw_relative_error=float(np.linalg.norm(a-p)/np.linalg.norm(a)),
                scaled_relative_error=float(np.linalg.norm(a-scale*p)/np.linalg.norm(a)))


def main():
    out=R.OUT/'composition_diagnostic';out.mkdir(parents=True,exist_ok=True)
    z=np.load(R.OUT/'predictions_N128.npz')
    k=z['keys'].tolist().index('composition')
    observed=z['actual'][:,:,k];predicted=z['corrected'][:,:,k]
    t=z['centers'];V=z['directions']
    result={'target':'exp(sin(pi*x)*cos(pi*y))','width':4096,
            'geometry':'axis-aligned directions, matching the interactive view','regions':{}}
    for label,limit in [('interior',.3),('all_centers',.5)]:
        keep=np.abs(t)<limit
        rows=[dict(angle=j*180/32,**scale_check(observed[j,keep[j]],predicted[j,keep[j]])) for j in range(32)]
        shares=[]
        for arr in [observed,predicted]:
            energy=np.sum(np.where(keep,arr,0)**2,axis=1)
            shares.append(energy/energy.sum())
        result['regions'][label]=dict(limit=limit,global_scale_check=scale_check(observed[keep],predicted[keep]),
            per_direction=rows,observed_coefficient_energy_share=shares[0].tolist(),
            predicted_coefficient_energy_share=shares[1].tolist(),
            observed_diagonal_share=float(shares[0][[8,24]].sum()),
            predicted_diagonal_share=float(shares[1][[8,24]].sum()))

    # This checks the normalization on a larger domain, without a tanh fit.
    x=np.random.default_rng(129).uniform(-2,2,size=(500,2))
    xi,coef,_,_=R.composition_atoms(V)
    fourier=np.real(np.exp(1j*((x-R.S.original.X0)@xi.T))@coef)
    truth=R.S.original.f_composition(x)
    factorized=np.exp(.5*np.sin(np.pi*(x[:,0]+x[:,1])))*np.exp(.5*np.sin(np.pi*(x[:,0]-x[:,1])))
    result['fourier_normalization_max_error']=float(np.max(np.abs(fourier-truth)))
    result['factorization_max_error']=float(np.max(np.abs(factorized-truth)))
    assert result['fourier_normalization_max_error']<1e-13
    assert result['factorization_max_error']<1e-13
    x=R.S.original.ball(2000,.36,R.S.original.X0,np.random.default_rng(482))
    snapped=R.atom_profiles(((x-R.S.original.X0)@V.T).T,V)[0].sum(axis=0)
    truth=R.S.original.f_composition(x)
    scale,bias=np.linalg.lstsq(np.column_stack([snapped,np.ones(len(x))]),truth,rcond=None)[0]
    result['snapped_function_relative_error']=float(np.linalg.norm(snapped-truth)/np.linalg.norm(truth))
    result['best_function_affine_scale']=float(scale)
    result['best_function_affine_bias']=float(bias)
    result['source_solution_relative_error']=float(np.load(R.S.OUT/'endpoint_direction_check'/'solution_M32_N128.npz')['rel_l2'][k])

    # Separate the distribution across directions from similarity within a spoke.
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    region=result['regions']['interior'];angles=np.arange(32)*180/32
    axes[0].bar(angles-1.2,100*np.array(region['observed_coefficient_energy_share']),width=2.4,color='#2878b5',label='Observed')
    axes[0].bar(angles+1.2,100*np.array(region['predicted_coefficient_energy_share']),width=2.4,color='#d76b1b',label='Fourier construction')
    axes[0].set(xlabel='Spoke direction (degrees)',ylabel='Share of squared coefficient norm (%)',title='The weights occupy different directions',xticks=[0,45,90,135,180])
    axes[0].legend(frameon=False)
    j=8;keep=np.abs(t[j])<.3;row=region['per_direction'][j]
    axes[1].plot(t[j,keep],row['best_scale']*predicted[j,keep],color='#d76b1b',lw=2,label=f"Theory × {row['best_scale']:.4f} (fitted for this check)")
    axes[1].scatter(t[j,keep],observed[j,keep],s=14,color='#2878b5',label='Observed coefficients',zorder=3)
    axes[1].set(xlabel='Center t along the 45-degree spoke',ylabel='Readout coefficient',title=f"Even after rescaling: {100*row['scaled_relative_error']:.1f}% discrepancy")
    axes[1].ticklabel_format(axis='y',style='sci',scilimits=(0,0));axes[1].legend(frameon=False,fontsize=9)
    for ax in axes:ax.grid(axis='y',alpha=.15)
    fig.suptitle('Asymmetric composition · 4096 neurons · interior centers |t| < 0.3')
    fig.savefig(out/'scaling_vs_direction_allocation.png',dpi=170)
    fig.savefig(out/'scaling_vs_direction_allocation.pdf');plt.close(fig)
    (out/'diagnostic.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='regions'},indent=2))
    for name,region in result['regions'].items():print(name,region['global_scale_check'])


if __name__=='__main__':main()
