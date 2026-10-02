"""A higher-order alternative to snapping Fourier atoms onto nearest spokes.

Interpolate the plane-wave angular dependence on the 64 oriented nodes of the
32-line dictionary. This construction uses analytic Fourier coefficients and
geometry alone, then independently approximates each prescribed 1D profile.
It is an additional construction, not the nearest-line certificate in the notes.
"""
from __future__ import annotations

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import spoke_radon_prediction as R
from theory_forward_check import OUT,evaluate,metric


def angular_coefficients(V):
    M=len(V);xi,c,_,_=R.composition_atoms(V)
    theta=np.arctan2(xi[:,1],xi[:,0]);nodes=np.arctan2(V[:,1],V[:,0])
    def cardinal(delta):
        d=(delta+np.pi)%(2*np.pi)-np.pi
        aligned=np.abs(d)<1e-13
        denominator=2*M*np.tan(np.where(aligned,1.,d)/2)
        result=np.sin(M*d)/denominator
        result[aligned]=1.
        # Enforce the exact nodal identity rather than retain sine roundoff.
        other_nodes=(np.abs(d*M/np.pi-np.rint(d*M/np.pi))<1e-13)&~aligned
        result[other_nodes]=0.
        return result
    delta=theta[None,:]-nodes[:,None]
    plus=cardinal(delta);minus=cardinal(delta-np.pi)
    partition=(plus+minus).sum(axis=0)
    partition_error=float(np.max(np.abs(partition-1)))
    # Exact cardinal functions sum to one. Restore that identity in fp64.
    plus/=partition;minus/=partition
    rho2=np.rint(np.sum((xi/np.pi)**2,axis=1)).astype(int)
    unique,index=np.unique(rho2,return_inverse=True);rho=np.pi*np.sqrt(unique)
    positive=np.zeros((M,len(rho)),complex);negative=positive.copy()
    for a in range(len(c)):
        positive[:,index[a]]+=plus[:,a]*c[a]
        negative[:,index[a]]+=minus[:,a]*c[a]
    return rho,positive,negative,partition_error


def profile(t,data):
    rho,positive,negative,_=data
    values=np.empty_like(t)
    for j in range(len(t)):
        e=np.exp(1j*t[j,:,None]*rho)
        values[j]=np.real(e@positive[j]+e.conj()@negative[j])
    return values


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',action='store_true')
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    sol=np.load(R.S.OUT/'endpoint_direction_check'/'solution_M32_N128.npz')
    V=sol['directions'];data=angular_coefficients(V)
    test=R.S.original.ball(20000,.36,R.S.original.X0,np.random.default_rng(1))
    truth=R.S.original.f_composition(test)[:,None]
    continuous=np.concatenate([profile(((chunk-R.S.original.X0)@V.T).T,data).sum(axis=0)
                               for chunk in np.array_split(test,40)])[:,None]
    result={'construction':'trigonometric interpolation of plane-wave angle; no 2D fit',
            'direction_lines':32,'oriented_interpolation_nodes':64,
            'partition_identity_roundoff_before_correction':data[-1],
            'continuous_error':metric(continuous,truth),'cells':[]}
    for N in [64,128]:
        sol=np.load(R.S.OUT/'endpoint_direction_check'/f'solution_M32_N{N}.npz')
        gamma=float(sol['gamma']);centers=sol['centers']-(V@R.S.original.X0)[:,None]
        t=np.linspace(-.4,.4,8*N+1)
        prescribed=profile(np.broadcast_to(t,(32,len(t))),data)
        w,b,info=R.S.original.solve_many(np.tanh(gamma*(t[:,None]-centers[0])),prescribed.T)
        weights=w.T[:,:,None];bias=np.array([b.sum()])
        pred=np.concatenate([evaluate(chunk,V,centers,gamma,weights,bias) for chunk in np.array_split(test,40)])
        cell={'N':N,'width':32*N,'one_dimensional_fit_rank':info['rank'],
              'network_error':metric(pred,truth),'one_dimensional_conversion_error':metric(pred,continuous)}
        if args.project and N==128:
            print('Projecting angular-interpolation readout into the retained SVD space',flush=True)
            model=R.S.original.RecenteredRidge(32,N,R.S.original.X0,.4)
            train=R.S.original.ball(8*32*N,.4,R.S.original.X0,np.random.default_rng(0))
            A=np.empty((len(train),32*N+1));A[:,:-1]=model.features(train);A[:,-1]=1
            U,s,Vt=np.linalg.svd(A,full_matrices=False)
            keep=s>float(sol['rcond'])*s[0];basis=Vt[keep]
            constructed=np.r_[weights.ravel(),bias]
            k=R.S.KEYS.index('composition');observed=np.r_[sol['w'][:,:,k].ravel(),sol['bias'][k]]
            projected=basis.T@(basis@constructed)
            difference=constructed-observed;retained=basis.T@(basis@difference)
            cell['projection']={'rank':int(keep.sum()),
                'discarded_fraction_of_squared_coefficient_difference':float(np.linalg.norm(difference-retained)**2/np.linalg.norm(difference)**2),
                'raw_coefficient_relative_difference':float(np.linalg.norm(difference)/np.linalg.norm(observed)),
                'projected_coefficient_relative_difference':float(np.linalg.norm(projected-observed)/np.linalg.norm(observed))}
            pw=projected[:-1].reshape(32,N,1);pb=projected[-1:]
            mask=np.abs(centers)<.3
            cell['projection']['interior_coefficient_relative_difference']=float(np.linalg.norm((pw[:,:,0]-sol['w'][:,:,k])[mask])/np.linalg.norm(sol['w'][:,:,k][mask]))
            del U,s,Vt,basis,A,train
            projected_y=np.concatenate([evaluate(chunk,V,centers,gamma,pw,pb) for chunk in np.array_split(test,40)])
            cell['projection']['network_error']=metric(projected_y,truth)
            cell['projection']['function_change']=metric(projected_y,pred)
            np.savez_compressed(OUT/'angular_interpolation_projected_N128.npz',w=pw,bias=pb,centers=centers,directions=V)
        result['cells'].append(cell)
        np.savez_compressed(OUT/f'angular_interpolation_N{N}.npz',w=weights,bias=bias,centers=centers,directions=V)
    (OUT/'angular_interpolation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    # A single scientific comparison figure for the current question.
    forward=json.loads((OUT/'comparison.json').read_text())['cells'][1]
    k=R.S.KEYS.index('composition')
    stages=['Snapped profiles','Same profiles\nas tanh blocks','Angular interpolation\nas tanh blocks','Joint 2D solve']
    errors=[forward['target_errors']['continuous']['rel_l2'][k],
            forward['target_errors']['independent_1d']['rel_l2'][k],
            result['cells'][1]['network_error']['rel_l2'][0],
            forward['target_errors']['joint_observed']['rel_l2'][k]]
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(9.5,4.7),layout='constrained')
    ax.scatter(range(4),errors,s=70,color=['#d76b1b','#d76b1b','#26866d','#2878b5'],zorder=3)
    ax.set(yscale='log',ylim=(1e-15,1e-2),xticks=range(4),xticklabels=stages,
           ylabel='Relative output error',title='Composition: angular interpolation closes the approximation gap')
    ax.grid(axis='y',alpha=.2)
    for j,e in enumerate(errors):ax.annotate(f'{e:.2e}',(j,e),xytext=(0,10),textcoords='offset points',ha='center')
    fig.savefig(OUT/'composition_construction_errors.png',dpi=170)
    fig.savefig(OUT/'composition_construction_errors.pdf');plt.close(fig)


if __name__=='__main__':main()
