"""Evaluate theoretical spoke constructions, then compare their SVD projections.

All comparisons use the axis-aligned 32-spoke dictionary already displayed.
Independent scalar fits approximate prescribed theoretical profiles; they never
fit the multivariate target. The final projection changes only the readout's
representation in the retained subspace, without re-fitting the target.
"""
from __future__ import annotations

import argparse
import json
import time
import numpy as np
import spoke_radon_prediction as R

OUT=R.OUT/'forward_check'
KEYS=R.S.KEYS


def profiles(t,V):
    values=np.empty((*t.shape,len(KEYS)))
    for k,key in enumerate(KEYS):
        values[:,:,k]=R.atom_profiles(t,V)[0] if key=='composition' else R.radial_profiles(key,t,V)[0]/len(V)
    return values


def coefficients(t,V,gamma):
    values=np.empty((*t.shape,len(KEYS)));a=np.pi/(2*gamma)
    for k,key in enumerate(KEYS):
        values[:,:,k]=R.atom_profiles(t,V,gamma)[2] if key=='composition' else np.imag(R.radial_profiles(key,t+1j*a,V)[0])/(len(V)*a)
    return values/(8*gamma)  # h/2, since gamma*h = 1/4


def metric(pred,truth):
    return {'rel_l2':(np.linalg.norm(pred-truth,axis=0)/np.linalg.norm(truth,axis=0)).tolist(),
            'max_abs':np.max(np.abs(pred-truth),axis=0).tolist()}


def coefficient_metric(w,observed,mask):
    return (np.linalg.norm((w-observed)[mask],axis=0)/np.linalg.norm(observed[mask],axis=0)).tolist()


def evaluate(x,V,centers,gamma,w,bias):
    result=np.zeros((len(x),w.shape[-1]))+bias
    for j in range(len(V)):
        result+=np.tanh(gamma*((x-R.S.original.X0)@V[j,None].T-centers[j]))@w[j]
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--project',action='store_true')
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    x=R.S.original.ball(20000,.36,R.S.original.X0,np.random.default_rng(1))
    truth=np.column_stack([target[3](x) for target in R.S.TARGETS])
    result={'keys':KEYS,'evaluation':'same 20000-point radius-0.36 set, seed 1','cells':[]}
    for N in [64,128]:
        started=time.monotonic();sol=dict(np.load(R.S.OUT/'endpoint_direction_check'/f'solution_M32_N{N}.npz'))
        V=sol['directions'];M=len(V);gamma=float(sol['gamma']);h=1/N
        centers=sol['centers']-(V@R.S.original.X0)[:,None]
        q0=profiles(np.zeros((M,1)),V)[:,0]
        cache=np.load(R.OUT/f'predictions_N{N}.npz')
        methods={name:cache[name] for name in ['leading','corrected']}
        biases={name:q0.sum(axis=0)-np.einsum('mn,mnk->k',np.tanh(-gamma*centers),w) for name,w in methods.items()}

        # Each prescribed profile is solved independently on the projection band.
        # The feature matrix is shared; one scalar SVD handles all 160 RHS.
        train_t=np.linspace(-.4,.4,8*N+1)
        prescribed=profiles(np.broadcast_to(train_t,(M,len(train_t))),V)
        phi=np.tanh(gamma*(train_t[:,None]-centers[0]))
        w,b,info=R.S.original.solve_many(phi,prescribed.transpose(1,0,2).reshape(len(train_t),-1))
        methods['independent_1d']=w.reshape(N,M,len(KEYS)).transpose(1,0,2)
        biases['independent_1d']=b.reshape(M,len(KEYS)).sum(axis=0)
        methods['joint_observed']=sol['w'];biases['joint_observed']=sol['bias']

        # Same h and gamma, but 80 extra centers at each end: isolate halo loss.
        halo=80
        extended=np.broadcast_to(centers[0,0]+h*np.arange(-halo,N+halo),(M,N+2*halo)).copy()
        halo_w=coefficients(extended,V,gamma)
        halo_bias=q0.sum(axis=0)-np.einsum('mn,mnk->k',np.tanh(-gamma*extended),halo_w)
        pred={name:[] for name in methods};pred['continuous']=[];pred['corrected_extended_halo']=[]
        for chunk in np.array_split(x,40):
            tt=((chunk-R.S.original.X0)@V.T).T
            pred['continuous'].append(profiles(tt,V).sum(axis=0))
            # Batch the ordinary-width methods to avoid re-evaluating features.
            combined=np.concatenate([methods[name] for name in methods],axis=-1)
            combined_bias=np.concatenate([biases[name] for name in methods])
            evaluated=evaluate(chunk,V,centers,gamma,combined,combined_bias)
            for i,name in enumerate(methods):pred[name].append(evaluated[:,i*len(KEYS):(i+1)*len(KEYS)])
            pred['corrected_extended_halo'].append(evaluate(chunk,V,extended,gamma,halo_w,halo_bias))
        pred={name:np.concatenate(parts) for name,parts in pred.items()}
        cell={'N':N,'width':M*N,'independent_1d_rank':info['rank'],
              'extended_halo_width':M*(N+2*halo),
              'target_errors':{name:metric(values,truth) for name,values in pred.items()},
              'construction_errors':{name:metric(values,pred['continuous']) for name,values in pred.items() if name!='continuous'},
              'coefficient_discrepancy_interior':{name:coefficient_metric(w,sol['w'],np.abs(centers)<.3) for name,w in methods.items()}}
        if args.project and N==128:
            print('Projecting independently constructed readout into the retained SVD space',flush=True)
            model=R.S.original.RecenteredRidge(M,N,R.S.original.X0,.4)
            train=R.S.original.ball(8*M*N,.4,R.S.original.X0,np.random.default_rng(0))
            A=np.empty((len(train),M*N+1));A[:,:-1]=model.features(train);A[:,-1]=1
            U,s,Vt=np.linalg.svd(A,full_matrices=False)
            keep=s>float(sol['rcond'])*s[0];basis=Vt[keep]
            observed=np.vstack([sol['w'].reshape(M*N,-1),sol['bias']])
            constructed=np.vstack([methods['independent_1d'].reshape(M*N,-1),biases['independent_1d']])
            projected=basis.T@(basis@constructed)
            delta=constructed-observed;retained=basis.T@(basis@delta)
            discarded=delta-retained
            cell['projection']={'rank':int(keep.sum()),
                'discarded_fraction_of_squared_coefficient_difference':(np.sum(discarded**2,axis=0)/np.sum(delta**2,axis=0)).tolist(),
                'raw_coefficient_difference':(np.linalg.norm(delta,axis=0)/np.linalg.norm(observed,axis=0)).tolist(),
                'projected_coefficient_difference':(np.linalg.norm(projected-observed,axis=0)/np.linalg.norm(observed,axis=0)).tolist(),
                'observed_retained_residual':(np.linalg.norm(observed-basis.T@(basis@observed),axis=0)/np.linalg.norm(observed,axis=0)).tolist(),
                'constructed_coefficient_norm':np.linalg.norm(constructed,axis=0).tolist(),
                'projected_coefficient_norm':np.linalg.norm(projected,axis=0).tolist(),
                'observed_coefficient_norm':np.linalg.norm(observed,axis=0).tolist()}
            pw=projected[:-1].reshape(M,N,-1);pb=projected[-1]
            del U,s,Vt,basis,A,train
            projected_y=np.concatenate([evaluate(chunk,V,centers,gamma,pw,pb) for chunk in np.array_split(x,40)])
            cell['target_errors']['projected_theory']=metric(projected_y,truth)
            cell['construction_errors']['projected_theory']=metric(projected_y,pred['continuous'])
            cell['projection']['function_change_from_projection']=metric(projected_y,pred['independent_1d'])
            cell['projection']['interior_coefficient_difference']=coefficient_metric(pw,sol['w'],np.abs(centers)<.3)
            np.savez_compressed(OUT/'projected_theory_N128.npz',w=pw,bias=pb,centers=centers,directions=V,keys=np.array(KEYS))
        np.savez_compressed(OUT/f'prescribed_profile_fit_N{N}.npz',w=methods['independent_1d'],bias=biases['independent_1d'],centers=centers,directions=V,keys=np.array(KEYS))
        cell['seconds']=time.monotonic()-started;result['cells'].append(cell)
        (OUT/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
        print('N',N,'seconds',cell['seconds'],flush=True)
        for name,metrics in cell['target_errors'].items():print(name,dict(zip(KEYS,metrics['rel_l2'])),flush=True)
        if 'projection' in cell:print('projection',cell['projection'],flush=True)


if __name__=='__main__':main()
