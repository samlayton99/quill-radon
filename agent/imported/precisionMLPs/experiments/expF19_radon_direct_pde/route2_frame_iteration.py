"""Native tanh residual correction using an explicit ball-frame inverse.

Proof-of-mechanism ONLY: the specially matched, boundary-degenerate operator
L=-div((I-xx^T)grad)+alpha on the unit disk has eigenvalues n(n+2)+alpha.
No global fit, LS, Krylov solve, dense coordinate map, or reference warmstart.
Actual tanh values/derivatives determine each residual. Known polynomials
are test functions in quadrature analysis, never a substitute forward model.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key,'1')
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/quill-route2-mpl')
import json,time
from pathlib import Path
import numpy as np
import torch
from numpy.polynomial.legendre import leggauss
from scipy.stats import qmc
from solver.ball_ridge_features import gegenbauer_profiles
from route2_directional_operator import DirectionalProfileOperator

OUT=Path(__file__).resolve().parents[2]/'results/checkpoint_F_applications/expF19_radon_direct_pde/route2_generality/frame_iteration'
ALPHA=2.;BETA=1.
ORDERS=[(0,0),(1,0),(0,1),(2,0),(1,1),(0,2)]


def quadrature(radial,angular):
    z,w=leggauss(radial);r=np.sqrt((z+1)/2)
    theta=2*np.pi*np.arange(angular)/angular
    points=np.stack([r[:,None]*np.cos(theta),r[:,None]*np.sin(theta)],axis=-1).reshape(-1,2)
    return points,np.repeat(w/(2*angular),angular)


def target_jets(points):
    x,y=points.T
    e=np.exp(.4*x);c=np.cos(1.2*y);s=np.sin(1.2*y)
    return {(0,0):.1+.25*e*c+.05*x*y,
            (1,0):.1*e*c+.05*y,(0,1):-.3*e*s+.05*x,
            (2,0):.04*e*c,(1,1):-.12*e*s+.05,(0,2):-.36*e*c}


def linear_part(points,j):
    x,y=points.T
    return (x*x-1)*j[(2,0)]+2*x*y*j[(1,1)]+(y*y-1)*j[(0,2)]+3*(x*j[(1,0)]+y*j[(0,1)])+ALPHA*j[(0,0)]


def source(points):
    j=target_jets(points)
    return linear_part(points,j)+BETA*np.tanh(j[(0,0)])


def neural_field_and_operator(operator,theta,points):
    """Apply L to neural ridges with three scalar jets, avoiding a d-by-d Hessian."""
    if np.any(operator.midpoint) or np.any(operator.scale!=1):
        raise ValueError('The ball operator requires physical unit-ball coordinates.')
    beta,b=operator.unpack(theta)
    points=operator._points(points)
    u=np.full(len(points),beta);lu=np.full(len(points),ALPHA*beta)
    for m,direction in enumerate(operator.directions):
        readout=operator.weights@b[m]
        anchor=operator.anchor_values@b[m]
        for start in range(0,len(points),operator.block_size):
            rows=slice(start,min(start+operator.block_size,len(points)))
            s=points[rows]@direction
            if np.any(abs(s)>1+2e-12):
                raise ValueError('Point outside the encoded projection band.')
            z=operator.encoding.gamma*(s[:,None]-operator.encoding.centers)
            h=operator._basis(z,0)@readout+anchor
            first=operator._basis(z,1)@readout
            second=operator._basis(z,2)@readout
            u[rows]+=h
            lu[rows]+=-(1-s*s)*second+(operator.dimension+1)*s*first+ALPHA*h
    return u,lu


def analyze(residual,points,weights,operator):
    # No Q-by-coordinate or M-by-mode-by-P matrix is built or cached.
    b=np.zeros((len(operator.directions),operator.degree))
    for start in range(0,len(points),128):
        x=points[start:start+128];v=weights[start:start+128]*residual[start:start+128]
        profiles=gegenbauer_profiles(x@operator.directions.T,2,operator.degree)[:,:,1:]
        b+=np.einsum('q,qmn->mn',v,profiles)
    b*=np.arange(2,operator.degree+2)[None,:]/len(operator.directions)
    return float(weights@residual),b


def ordinary_jets(model,points):
    result={key:[] for key in ORDERS}
    for batch in np.array_split(points,max(1,int(np.ceil(len(points)/64)))):
        x=torch.tensor(batch,dtype=torch.float64,requires_grad=True)
        u=model(x)[:,0];g=torch.autograd.grad(u.sum(),x,create_graph=True)[0]
        gx=torch.autograd.grad(g[:,0].sum(),x,retain_graph=True)[0]
        gy=torch.autograd.grad(g[:,1].sum(),x)[0]
        for key,value in zip(ORDERS,[u,g[:,0],g[:,1],gx[:,0],gx[:,1],gy[:,1]]):
            result[key].append(value.detach().numpy())
    return {key:np.concatenate(value) for key,value in result.items()}


def iterate(operator,points,w,rhs,stopping=2e-14,max_iterations=70):
    """The update sees only prescribed forcing and the current neural residual."""
    theta=np.zeros(operator.size);history=[]
    modes=np.arange(1,operator.degree+1)
    for iteration in range(max_iterations):
        u,lu=neural_field_and_operator(operator,theta,points)
        residual=lu+BETA*np.tanh(u)-rhs
        rms=float(np.sqrt(w@(residual*residual)))
        mean,b=analyze(residual,points,w,operator)
        correction=operator.pack(mean/ALPHA,b/(ALPHA+modes*(modes+2)))
        projected=float(np.linalg.norm(correction))
        history.append(dict(iteration=iteration,physical_residual_rms=rms,coordinate_correction_norm=projected))
        theta-=correction
        if projected<stopping:break
    return theta,history


def run(degree,centers=257,lam=.2,quadrature_scale=1,stopping=2e-14,tag=None):
    began=time.perf_counter();m=degree+1
    angles=np.pi*np.arange(m)/m
    operator=DirectionalProfileOperator(np.column_stack([np.cos(angles),np.sin(angles)]),degree,centers=centers,lam=lam)
    points,w=quadrature(quadrature_scale*max(10,degree//2+4),quadrature_scale*max(32,2*degree+8))
    rhs=source(points)  # Prescribed manufactured forcing; no solution labels.
    theta,history=iterate(operator,points,w,rhs,stopping=stopping)
    elapsed=time.perf_counter()-began
    z=qmc.Sobol(2,scramble=True,seed=783).random_base2(10)
    r=np.sqrt(z[:,0]);angle=2*np.pi*z[:,1]
    held=np.column_stack([r*np.cos(angle),r*np.sin(angle)])
    model=operator.torch_model(theta);j=ordinary_jets(model,held)
    exact=target_jets(held)[(0,0)];residual=linear_part(held,j)+BETA*np.tanh(j[(0,0)])-source(held)
    # A separate quadrature and analytic reference validate, never correct.
    row=dict(degree=degree,directions=m,centers=centers,halo_per_side=int(np.ceil(np.sqrt(centers))),
       neurons=model[0].out_features,directional_coordinates=operator.size,points=len(points),
       lam=lam,quadrature_scale=quadrature_scale,stopping=stopping,
       iterations=len(history),status='projected_correction_small' if history[-1]['coordinate_correction_norm']<stopping else 'iteration_limit',
       architecture='Linear/Tanh/Linear',method='actual neural PDE residual -> explicit frame analysis -> divide by known operator eigenvalues',
       no_least_squares=True,no_pde_matrix_factorization=True,no_dense_coordinate_map=True,reference_used_in_updates=False,
       ordinary_relative_l2=float(np.linalg.norm(j[(0,0)]-exact)/np.linalg.norm(exact)),
       ordinary_pde_rms=float(np.sqrt(np.mean(residual**2))),ordinary_pde_maximum=float(abs(residual).max()),
       setup_and_iterations_seconds=elapsed,total_seconds=time.perf_counter()-began,history=history,
       scope='Special ball operator with natural zero normal flux from boundary-degenerate diffusion; NOT ordinary Poisson or general nonlinear PDE solver',
       exact_space_contraction_bound=BETA/ALPHA,
       discrete_claim='Exact Galerkin/frame version contracts by beta/alpha; tanh encoding and quadrature perturb it. Raw neural residual audited independently.')
    OUT.mkdir(parents=True,exist_ok=True)
    label=f'p{degree}' if tag is None else tag
    (OUT/f'{label}.json').write_text(json.dumps(row,indent=2)+'\n')
    torch.save(model.state_dict(),OUT/f'{label}_ordinary_mlp.pt')
    np.savez_compressed(OUT/f'{label}.npz',theta=theta,points=held,prediction=j[(0,0)],target=exact)
    print(json.dumps({k:row[k] for k in ['degree','neurons','iterations','ordinary_relative_l2','ordinary_pde_rms','total_seconds']}),flush=True)
    return row


def main():
    torch.set_num_threads(1)
    rows=[run(p) for p in [4,8,12,16]]
    (OUT/'summary.json').write_text(json.dumps(dict(alpha=ALPHA,beta=BETA,rows=rows),indent=2)+'\n')
    checks=[run(16,quadrature_scale=2,tag='p16_quad2'),
            run(16,quadrature_scale=2,stopping=2e-16,tag='p16_quad2_tighter'),
            run(16,centers=129,stopping=2e-16,tag='p16_n129'),
            run(16,centers=513,stopping=2e-16,tag='p16_n513')]
    (OUT/'independent_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),constrained_layout=True)
    for row in rows:
        axes[0].semilogy([h['iteration'] for h in row['history']],[h['coordinate_correction_norm'] for h in row['history']],label=f"p={row['degree']}")
    axes[0].set(xlabel='Correction iteration',ylabel='Correction norm',title='Explicit directional correction; no global solve');axes[0].legend()
    axes[1].semilogy([r['degree'] for r in rows],[r['ordinary_relative_l2'] for r in rows],'-o',label='Ordinary MLP field error')
    axes[1].semilogy([r['degree'] for r in rows],[r['ordinary_pde_rms'] for r in rows],'-s',label='Ordinary AD PDE residual')
    axes[1].set(xlabel='Approximation order',title='Independent held-out checks');axes[1].legend(fontsize=8)
    for a in axes:a.grid(alpha=.2)
    fig.savefig(OUT/'frame_iteration.png',dpi=180)

if __name__=='__main__':main()
