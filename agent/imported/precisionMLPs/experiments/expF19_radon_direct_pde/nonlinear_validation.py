"""Independent checks and finer reference for nonlinear_pde.py."""
from nonlinear_pde import *

def main():
    start=time.perf_counter(); OUT.mkdir(exist_ok=True,parents=True)
    metrics=json.loads((OUT/'ns_metrics.json').read_text())
    sim=PeriodicNS(64,.05); state,meta=sim.run(.5,.0025)
    x=2*np.pi*(qmc.Sobol(3,scramble=True,seed=1843).random_base2(9)-.5)
    k,c,mean=sim.modes(state,1e-17); ref=fourier_values(x,k,c,mean)
    stored=np.load(OUT/'ns_spectral_states.npz'); s48=PeriodicNS(48,.05); h48=stored['state48']
    k48,c48,m48=s48.modes(h48,1e-17); pred48=fourier_values(x,k48,c48,m48)
    info=dict(reference64=meta,rel_48_vs_64=rel(pred48,ref))
    # Refined network: same construction and independent heldout residual.
    k,c,mean=ns_fields(s48,h48,1e-14)
    encstart=time.perf_counter();net=encode_fourier(k,c,mean,h=.04,halo=3.)
    y,g,l=eval_network(x[:128],net,True);sy,sg,sl=fourier_values(x[:128],k,c,mean,True)
    residual=y[:,3:6]+np.einsum('nd,nod->no',y[:,:3],g[:,:3,:])+g[:,6,:]-.05*l[:,:3]
    sr=sy[:,3:6]+np.einsum('nd,nod->no',sy[:,:3],sg[:,:3,:])+sg[:,6,:]-.05*sl[:,:3]
    info['encode48']=dict(grid=48,directions=len(net['directions']),centers=len(net['centers']),
        neurons=len(net['directions'])*len(net['centers']),retained_conjugate_mode_pairs=len(k),
        velocity_rel_l2_vs_spectral=rel(y[:,:3],sy[:,:3]),velocity_rel_l2_vs_64=rel(y[:,:3],ref[:128]),
        pressure_rel_l2=rel(y[:,6],sy[:,6]),velocity_gradient_rel_l2=rel(g[:,:3,:],sg[:,:3,:]),
        divergence_max=float(np.max(np.abs(np.trace(g[:,:3,:],axis1=1,axis2=2)))),
        physical_residual_rms=float(np.sqrt(np.mean(residual**2))),spectral_physical_residual_rms=float(np.sqrt(np.mean(sr**2))),
        coefficient_storage_mb=net['coefficient'].nbytes/1e6,encode_and_128_eval_seconds=time.perf_counter()-encstart)
    np.savez_compressed(OUT/'ns_snapshot48.npz',**net,t=.5,nu=.05)
    print('REFINE',info['rel_48_vs_64'],info['encode48'],flush=True)
    # RK4 temporal convergence differences on a fixed spatial subspace.
    s32=PeriodicNS(32,.05); steps={}
    for dt in [.02,.01,.005]: steps[dt]=s32.run(.5,dt)[0]
    fine=stored['state32']
    info['temporal']=dict(rel_dt02_vs_dt01=rel(steps[.02],steps[.01]),rel_dt01_vs_dt005=rel(steps[.01],steps[.005]),
                          rel_dt005_vs_dt0025=rel(steps[.005],fine))
    # At t=0, exact TG projected nonlinear acceleration is elementary.
    n=16;s=PeriodicNS(n,.05);xx,yy,zz=np.meshgrid(2*np.pi*np.arange(n)/n,2*np.pi*np.arange(n)/n,2*np.pi*np.arange(n)/n,indexing='ij')
    exactnl=np.array([-.125*np.sin(2*xx)*np.cos(2*zz),-.125*np.sin(2*yy)*np.cos(2*zz),
                       .125*(np.cos(2*xx)+np.cos(2*yy))*np.sin(2*zz)])
    numericalnl=s.physical(s.rhs(s.u0)+s.nu*s.k2*s.u0)
    info['initial_tg_nonlinear_acceleration_rel_l2']=rel(numericalnl,exactnl)
    # Exact embedded 2D Taylor-Green checks diffusion evolution, projection, pressure sign.
    u0=np.array([np.sin(xx)*np.cos(yy),-np.cos(xx)*np.sin(yy),np.zeros_like(xx)])
    s.u0=np.fft.fftn(u0,axes=(1,2,3))/n**3;hf,_=s.run(.5,.01)
    info['embedded_2d_tg_evolution_rel_l2']=rel(s.physical(hf),np.exp(-.05)*u0)
    pk,pc,pm=ns_fields(s,hf,1e-17)
    checkx=x[:128];actual=fourier_values(checkx,pk,pc,pm)[:,6]
    expected=.25*(np.cos(2*checkx[:,0])+np.cos(2*checkx[:,1]))*np.exp(-.1)
    info['embedded_2d_tg_pressure_rel_l2']=rel(actual,expected)
    # Energy identity for the interacting solution (no expected closed form).
    h32=fine;rhs=s32.rhs(h32)
    dissipation=s32.nu*np.sum(s32.k2*np.sum(np.abs(h32)**2,axis=0))
    derivative=np.real(np.sum(np.conj(h32)*rhs))
    info['energy_balance_absolute_defect']=float(abs(derivative+dissipation))
    info['elapsed_seconds']=time.perf_counter()-start
    (OUT/'validation_metrics.json').write_text(json.dumps(info,indent=2)+'\n')
    print(json.dumps(info,indent=2),flush=True)
if __name__=='__main__':main()
