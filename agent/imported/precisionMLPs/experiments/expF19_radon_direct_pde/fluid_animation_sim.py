"""Native QUILL Navier--Stokes trajectory and passive material visualization.

No reference field enters dynamics. Particles are passive: periodic cubic
interpolation of actual constructed velocities is used only for visualization.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import argparse, json, time
import numpy as np
from scipy.ndimage import map_coordinates
from solver.ns_extended import PeriodicNS, taylor_green

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/checkpoint_F_applications/expF19_radon_direct_pde/fluid_animation'


def initial(points):
    x,y,z=np.asarray(points).T
    abc=np.column_stack((.8*np.sin(z)+np.cos(y), .6*np.sin(x)+.8*np.cos(z),
                         np.sin(y)+.6*np.cos(x)))
    harmonic=np.column_stack((np.sin(2*y+.4),np.sin(2*z-.7),np.sin(2*x+.9)))
    return 1.35*taylor_green(points)+.65*abc+.12*harmonic


def wrap(points):return (points+np.pi)%(2*np.pi)-np.pi


def seeds():
    a=np.linspace(-2.25,2.25,17)
    y,z=np.meshgrid(a,a,indexing='ij')
    first=np.column_stack((np.full(y.size,-1.3),y.ravel(),z.ravel()))
    second=np.column_stack((y.ravel(),np.full(y.size,1.2),z.ravel()))
    particles=np.random.default_rng(146).uniform(-2.9,2.9,(160,3))
    return np.vstack((first,second,particles))


def interpolate(velocity,points,grid):
    coordinates=((wrap(points)+np.pi)*grid/(2*np.pi)).T
    array=velocity.reshape(grid,grid,grid,3)
    return np.column_stack([map_coordinates(array[...,j],coordinates,order=3,
                           mode='grid-wrap',prefilter=True) for j in range(3)])


def evolve(cutoff=13,steps=480,record=True):
    began=time.perf_counter();model=PeriodicNS(cutoff=cutoff,viscosity=.075,n_centers=257,lam=.2)
    b=model.analyze(initial(model.points));check=model.initial_diagnostics(initial,b)
    if check['relative_error']>1e-8:raise RuntimeError('Initial representation failed')
    T=4.;dt=T/steps;pos=seeds() if record else None
    positions=[];speeds=[];times=[];states=[];diagnostics=[];samples=[]
    probe=np.random.default_rng(628).uniform(-np.pi,np.pi,(96,3))
    maximum_cfl=0.;interpolation_checks=[]

    def stage(state,points):
        u,grad,lap=model.fields(state)
        physical=-np.einsum('nj,nij->ni',u,grad)+model.viscosity*lap
        rate=model.analyze(physical)
        return rate,interpolate(u,points,model.grid) if points is not None else None

    for n in range(steps+1):
        if n%(steps//12)==0:
            d=model.diagnostics(n*dt,b);d['time']=n*dt;diagnostics.append(d)
            if not np.isfinite(d['strong_residual_relative']):raise RuntimeError('Nonfinite diagnostics')
            print(json.dumps({'K':cutoff,'time':n*dt,'residual':d['strong_residual_relative'],'energy':d['energy']}),flush=True)
        if n%(steps//4)==0:samples.append(model.evaluate(b,probe))
        if record and n%(steps//120)==0:
            u=model.fields(b,derivatives=False);v=interpolate(u,pos,model.grid)
            positions.append(wrap(pos).astype(np.float32));speeds.append(np.linalg.norm(v,axis=1).astype(np.float32))
            times.append(n*dt);states.append(b.astype(np.float32))
            maximum_cfl=max(maximum_cfl,float(dt*cutoff*np.max(np.sum(abs(u),axis=1))))
            if n%(steps//4)==0:
                exact=model.evaluate(b,wrap(pos[:48]))
                interpolation_checks.append(float(np.linalg.norm(v[:48]-exact)/np.linalg.norm(exact)))
        if n==steps:break
        k1,v1=stage(b,pos)
        k2,v2=stage(b+dt*k1/2,None if pos is None else pos+dt*v1/2)
        k3,v3=stage(b+dt*k2/2,None if pos is None else pos+dt*v2/2)
        k4,v4=stage(b+dt*k3,None if pos is None else pos+dt*v3)
        b+=dt*(k1+2*k2+2*k3+k4)/6
        if pos is not None:pos=wrap(pos+dt*(v1+2*v2+2*v3+v4)/6)
        if not np.isfinite(b).all():raise RuntimeError('Nonfinite state')
        if maximum_cfl>1.5:raise RuntimeError('CFL guard')
    metadata=dict(cutoff=cutoff,dt=dt,steps=steps,final_time=T,viscosity=model.viscosity,
                  forcing='none',initial='1.35 Taylor-Green + .65 ABC(A=.8,B=.6,C=1) + .12 cyclic second harmonic',
                  build=model.build,initial_check=check,diagnostics=diagnostics,
                  max_physical_residual=max(d['strong_residual_relative'] for d in diagnostics),
                  max_divergence=max(d['divergence_max'] for d in diagnostics),
                  max_relative_energy_defect=max(abs(d['instantaneous_energy_balance'])/d['energy_balance_scale'] for d in diagnostics),
                  max_cfl=maximum_cfl,particle_interpolation_relative_checks=interpolation_checks,
                  seconds=time.perf_counter()-began,
                  method='Actual QUILL fields and analytic gradients/Laplacians at every RK4 stage; Fourier quadrature and pressure projection; no readout fit or reference trajectory.',
                  tracer_method='Passive RK4 particles with periodic cubic interpolation of actual QUILL fields; visualization interpolation does not affect fluid dynamics.')
    return dict(metadata=metadata,positions=np.asarray(positions),speeds=np.asarray(speeds),times=np.asarray(times),
                coefficients=np.asarray(states),final=b,samples=np.asarray(samples))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--cutoff',type=int,default=13)
    parser.add_argument('--reference',action='store_true');parser.add_argument('--time-check',action='store_true')
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    record=not(args.reference or args.time_check)
    result=evolve(args.cutoff,960 if args.time_check else 480,record)
    name='time_check' if args.time_check else 'reference' if args.reference else 'trajectory'
    np.savez_compressed(OUT/(name+'.npz'),**{k:v for k,v in result.items() if k!='metadata'})
    (OUT/(name+'_metrics.json')).write_text(json.dumps(result['metadata'],indent=2))
    print(json.dumps({'saved':name,'seconds':result['metadata']['seconds'],'max_residual':result['metadata']['max_physical_residual']}),flush=True)


if __name__=='__main__':main()
