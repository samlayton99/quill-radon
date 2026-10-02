"""Full 3D transient incompressible NS declared to the generic residual engine.

Four inputs (x,y,z,t), four unknown fields (u,v,w,p). Smooth ABC/Beltrami data
on a compact box. No NS-specific solver, Leray projection, time stepping, or
interior/final solution labels enter the solve. This is a smooth limited-domain
test, not a turbulence or general Navier--Stokes regularity claim.
"""
from __future__ import annotations
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(name,'1')
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/quill-ns4d-mpl')
from pathlib import Path
import json,time
import numpy as np
import torch
from solver.general_domains import ResidualDomain
from solver.general_residual import ResidualBlock
from solver.general_adaptive import ResidualDeclaration,solve_declared,check_residuals
from general_residual_study import exact_jets,clean

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/checkpoint_F_applications/expF19_radon_direct_pde/general_residual'
NU=.1;A=.7;B=.5;C=.3
ZERO=(0,0,0,0)
X=(1,0,0,0);Y=(0,1,0,0);Z=(0,0,1,0);T=(0,0,0,1)
XX=(2,0,0,0);YY=(0,2,0,0);ZZ=(0,0,2,0)
DERIVATIVES=(ZERO,X,Y,Z,T,XX,YY,ZZ)
DOMAIN=ResidualDomain([[-.5,.5],[-.5,.5],[-.5,.5],[0,1]])


def reference(points):
    """Analytic data/validation reference; never an interior fitting target."""
    x,y,z,t=points.unbind(-1)
    base=torch.stack((A*torch.sin(z)+C*torch.cos(y),
                      B*torch.sin(x)+A*torch.cos(z),
                      C*torch.sin(y)+B*torch.cos(x)),dim=1)
    decay=torch.exp(-NU*t)
    velocity=decay[:,None]*base
    pressure=-.5*decay**2*(torch.sum(base*base,dim=1)-(A*A+B*B+C*C))
    return torch.cat((velocity,pressure[:,None]),dim=1)


def ns_residual(points,jets,parameters):
    u=jets[ZERO][:,:3]
    gradient=torch.stack((jets[X][:,:3],jets[Y][:,:3],jets[Z][:,:3]),dim=2)
    convection=torch.einsum('qj,qij->qi',u,gradient)
    pressure_gradient=torch.stack((jets[X][:,3],jets[Y][:,3],jets[Z][:,3]),dim=1)
    laplacian=jets[XX][:,:3]+jets[YY][:,:3]+jets[ZZ][:,:3]
    momentum=jets[T][:,:3]+convection+pressure_gradient-NU*laplacian
    divergence=jets[X][:,0]+jets[Y][:,1]+jets[Z][:,2]
    return torch.cat((momentum,divergence[:,None]),dim=1)


def make_blocks(count,seed):
    interior=DOMAIN.interior(count,seed)
    blocks=[ResidualBlock('momentum_and_divergence',interior,ns_residual,DERIVATIVES)]
    n=max(96,int(np.sqrt(count))*12)
    boundary=DOMAIN.boundary(n,seed+13)
    mask=abs(boundary.normals[:,3])<.5
    spatial=boundary.points[mask]
    normals=boundary.normals[mask,:3]
    faces=np.argmax(abs(normals),axis=1)*2+(normals.sum(axis=1)>0)
    assert len(np.unique(faces))==6
    spatial_target=reference(torch.as_tensor(spatial))[:,:3].detach()
    blocks.append(ResidualBlock('spatial_velocity',spatial,
        lambda x,j,p:j[ZERO][:,:3]-spatial_target,(ZERO,)))
    initial=DOMAIN.interior(n,seed+104729);initial[:,3]=0.
    initial_target=reference(torch.as_tensor(initial))[:,:3].detach()
    blocks.append(ResidualBlock('initial_velocity',initial,
        lambda x,j,p:j[ZERO][:,:3]-initial_target,(ZERO,)))
    gauge=DOMAIN.interior(max(32,n//8),seed+29989);gauge[:,:3]=0.
    blocks.append(ResidualBlock('pressure_gauge',gauge,lambda x,j,p:j[ZERO][:,3],(ZERO,)))
    return blocks


def validate_reference():
    points=DOMAIN.interior(79,51821)
    jets=exact_jets(reference,points,DERIVATIVES)
    residual=ns_residual(torch.as_tensor(points),jets,torch.zeros((len(points),0)))
    maximum=float(torch.max(abs(residual)))
    assert maximum<2e-15
    gradient=torch.stack((jets[X][:,:3],jets[Y][:,:3],jets[Z][:,:3]),dim=2)
    curl=torch.stack((gradient[:,2,1]-gradient[:,1,2],
                      gradient[:,0,2]-gradient[:,2,0],
                      gradient[:,1,0]-gradient[:,0,1]),dim=1)
    return dict(maximum_pde_residual=maximum,
                maximum_beltrami_defect=float(torch.max(abs(curl-jets[ZERO][:,:3]))),
                maximum_laplacian_defect=float(torch.max(abs(jets[XX][:,:3]+jets[YY][:,:3]+jets[ZZ][:,:3]+jets[ZERO][:,:3]))))


def validate_solution(solution):
    points=DOMAIN.interior(1009,71813)
    target=reference(torch.as_tensor(points)).numpy();predicted=solution.evaluate(points)
    relative=lambda a,b:float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-30))
    final=points.copy();final[:,3]=1.
    final_truth=reference(torch.as_tensor(final)).numpy();final_pred=solution.evaluate(final)
    checks=check_residuals(solution,make_blocks(1201,772313))
    jets={d:torch.as_tensor(solution.evaluate(points,d)) for d in DERIVATIVES}
    residual=ns_residual(torch.as_tensor(points),jets,torch.zeros((len(points),0))).numpy()
    truth_jets=exact_jets(reference,points,(X,Y,Z,T))
    return dict(velocity_relative_l2=relative(predicted[:,:3],target[:,:3]),
                pressure_relative_l2=relative(predicted[:,3],target[:,3]),
                final_velocity_relative_l2=relative(final_pred[:,:3],final_truth[:,:3]),
                final_pressure_relative_l2=relative(final_pred[:,3],final_truth[:,3]),
                velocity_max_error=float(np.max(abs(predicted[:,:3]-target[:,:3]))),
                pressure_max_error=float(np.max(abs(predicted[:,3]-target[:,3]))),
                velocity_gradient_relative_l2=relative(np.stack([jets[d][:,:3].numpy() for d in [X,Y,Z]]),
                    np.stack([truth_jets[d][:,:3].numpy() for d in [X,Y,Z]])),
                velocity_time_derivative_relative_l2=relative(jets[T][:,:3].numpy(),truth_jets[T][:,:3].numpy()),
                physical_residual_component_rms=np.sqrt(np.mean(residual**2,axis=0)).tolist(),
                physical_residual_component_max=np.max(abs(residual),axis=0).tolist(),fresh_block_checks=checks)


def main():
    torch.set_num_threads(1);OUT.mkdir(parents=True,exist_ok=True)
    reference_check=validate_reference()
    declaration=ResidualDeclaration(DOMAIN,make_blocks,fields=4)
    began=time.perf_counter()
    # Standard public defaults, shared with all declared PDEs. No per-NS
    # optimizer choice or special pressure solve is supplied.
    result=solve_declared(declaration,tolerance=1e-6,
                          max_seconds=180,maximum_basis_cache_mb=512)
    solve_seconds=time.perf_counter()-began
    before=time.perf_counter();validation=validate_solution(result.solution)
    validation_seconds=time.perf_counter()-before
    features=result.problem.features
    output=dict(problem='3D time-dependent incompressible Navier-Stokes, ABC/Beltrami data',
        inputs=['x','y','z','time'],fields=['velocity_x','velocity_y','velocity_z','pressure'],
        viscosity=NU,abc_constants=[A,B,C],bounds=DOMAIN.bounds.tolist(),
        data='Initial velocity and velocity on six spatial faces; p(0,0,0,t)=0. No interior or final-time solution labels.',
        formulation='Unforced four-field global spacetime residual; no NS-specific solver, time marcher, or Leray projection',
        limitations='Smooth low-frequency solution on a compact cube; exact Dirichlet traces supplied as problem data. This does not test turbulence, large domains, or high Reynolds number.',
        status=result.status,selected_degree=features.degree,features=features.metrics,
        scalar_features=features.size,unknown_coefficients=features.size*4,
        solve_seconds=solve_seconds,validation_seconds=validation_seconds,
        reference_checks=reference_check,adaptive_metrics=result.metrics,
        adaptive_history=result.history,validation=validation)
    (OUT/'ns4d_metrics.json').write_text(json.dumps(clean(output),indent=2)+'\n')
    np.savez_compressed(OUT/'ns4d_coefficients.npz',coefficients=result.coefficients,
                        multiindices=features.multiindices,bounds=features.bounds)
    plot(result,output)
    print(json.dumps(clean({k:output[k] for k in ['status','selected_degree','scalar_features','unknown_coefficients','solve_seconds','validation_seconds','reference_checks','validation']}),indent=2),flush=True)


def plot(solution,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    axis=np.linspace(-.5,.5,51)
    x,y=np.meshgrid(axis,axis,indexing='ij')
    points=np.column_stack((x.ravel(),y.ravel(),np.zeros(x.size),np.ones(x.size)))
    truth=reference(torch.as_tensor(points)).numpy()[:,:3]
    predicted=solution.evaluate(points)[:,:3]
    arrays=[np.linalg.norm(truth,axis=1),np.linalg.norm(predicted,axis=1),
            np.log10(np.maximum(np.linalg.norm(predicted-truth,axis=1),1e-16))]
    titles=['Reference speed, t=1 and z=0','Generic residual solve speed','log10 velocity-vector error']
    fig,axes=plt.subplots(1,3,figsize=(12,3.7),constrained_layout=True)
    for i,(ax,array,title) in enumerate(zip(axes,arrays,titles)):
        options={} if i==2 else dict(vmin=arrays[0].min(),vmax=arrays[0].max())
        m=ax.pcolormesh(x,y,array.reshape(x.shape),shading='auto',**options)
        fig.colorbar(m,ax=ax);ax.set(xlabel='x',ylabel='y',title=title,aspect='equal')
    fig.suptitle('4-input / 4-field Navier–Stokes: '+output['status']+', degree '+str(output['selected_degree']))
    fig.savefig(OUT/'ns4d_final_slice.png',dpi=180);plt.close(fig)


if __name__=='__main__':main()
