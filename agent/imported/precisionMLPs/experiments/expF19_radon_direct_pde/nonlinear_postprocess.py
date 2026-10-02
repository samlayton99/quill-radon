"""Chat figures, sparse-profile ablation and boundary checks."""
from nonlinear_pde import *

def make_bandwidth_ablation():
    x=np.linspace(-np.pi,np.pi,1025); truth=burgers_solution(x,.8,.15)
    grid=2*np.pi*np.arange(2048)/2048
    fh=np.fft.fft(burgers_solution(grid,.8,.15)[:,:2],axis=0)/2048
    rows=[]
    for h in [.16,.08,.04]:
        for cut in [None,int(np.floor(np.pi/h))]:
            kk=np.arange(1,256);keep=np.max(np.abs(fh[kk]),axis=1)>1e-14
            if cut is not None: keep &= kk<=cut
            k=kk[keep,None];c=fh[k[:,0]]
            for halo in [3.,18/(.25/h)]:
                net=encode_fourier(k,c,np.real(fh[0]),h=h,halo=halo)
                y=eval_network(x[:,None],net)[:,0]
                rows.append(dict(h=h,cutoff=cut,halo=halo,modes=len(k),max_mode=int(k.max()),
                    neurons=len(net['centers']),rel_l2=rel(y,truth[:,0]),
                    max_abs_readout=float(np.max(np.abs(net['coefficient'])))))
    (OUT/'burgers_bandwidth_ablation.json').write_text(json.dumps(rows,indent=2)+'\n')


def main():
    make_bandwidth_ablation()
    b=json.loads((OUT/'burgers_metrics.json').read_text());ns=json.loads((OUT/'ns_metrics.json').read_text());val=json.loads((OUT/'validation_metrics.json').read_text())
    npz=np.load(OUT/'ns_snapshot48.npz');net={k:npz[k] for k in ['directions','centers','gamma','coefficient','bias']}
    surface=2*np.pi*(qmc.Sobol(2,scramble=True,seed=72).random_base2(5)-.5)
    bc=[]
    for d in range(3):
        xm=np.zeros((32,3));xp=np.zeros_like(xm);others=[q for q in range(3) if q!=d]
        xm[:,others]=surface;xp[:,others]=surface;xm[:,d]=-np.pi;xp[:,d]=np.pi
        ym,gm,_=eval_network(xm,net,True);yp,gp,_=eval_network(xp,net,True)
        bc.append(dict(axis=d,max_velocity_periodic_mismatch=float(np.max(np.abs(ym[:,:3]-yp[:,:3]))),
                       max_velocity_gradient_periodic_mismatch=float(np.max(np.abs(gm[:,:3,:]-gp[:,:3,:]))),
                       max_pressure_periodic_mismatch=float(np.max(np.abs(ym[:,6]-yp[:,6])))))
    s=PeriodicNS(16,.05);k,c,mean=ns_fields(s,s.u0,1e-14);initial=encode_fourier(k,c,mean,h=.04,halo=3)
    x=2*np.pi*(qmc.Sobol(3,scramble=True,seed=1442).random_base2(8)-.5)
    exact=np.c_[np.sin(x[:,0])*np.cos(x[:,1])*np.cos(x[:,2]),-np.cos(x[:,0])*np.sin(x[:,1])*np.cos(x[:,2]),np.zeros(len(x))]
    initial_error=rel(eval_network(x,initial)[:,:3],exact)
    # Drop small Fourier contributions before allocating whole center strings.
    states=np.load(OUT/'ns_spectral_states.npz');s=PeriodicNS(48,.05)
    k,c,mean=ns_fields(s,states['state48'],1e-16);base=fourier_values(x,k,c,mean)
    sparse=[]
    for tol in [1e-4,1e-6,1e-8,1e-10,1e-12,1e-14]:
        keep=np.max(np.abs(c[:,[0,1,2,6]]),axis=1)>tol
        smallnet=encode_fourier(k[keep],c[keep],mean,h=.04,halo=3)
        y=eval_network(x,smallnet)
        bound=float(2*np.sum(np.linalg.norm(c[~keep,:3],axis=1)))
        sparse.append(dict(mode_threshold=tol,directions=len(smallnet['directions']),neurons=len(smallnet['directions'])*len(smallnet['centers']),
             velocity_rel_l2_vs_full=rel(y[:,:3],base[:,:3]),pressure_rel_l2_vs_full=rel(y[:,6],base[:,6]),
             uniform_velocity_absolute_error_bound=bound))
    extra=dict(periodic_boundary=bc,initial_condition_rel_l2=initial_error,sparse_export=sparse)
    (OUT/'boundary_sparse_metrics.json').write_text(json.dumps(extra,indent=2)+'\n')
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(1,3,figsize=(16,4.4),constrained_layout=True)
    enc=ns['encoding']+[val['encode48']]
    ax[0].semilogy([r['grid'] for r in enc],[r.get('velocity_rel_l2_vs_64',r.get('velocity_rel_l2_vs_48')) for r in enc],'o-',label='Total velocity error*',c='#2865b0')
    ax[0].semilogy([r['grid'] for r in enc],[r['physical_residual_rms'] for r in enc],'s--',label='Physical momentum RMS',c='#cf6531')
    ax[0].semilogy([r['grid'] for r in enc],[r['velocity_rel_l2_vs_spectral'] for r in enc],'o-',label='Tanh conversion alone',c='#26855b')
    ax[0].set(xlabel='Fourier grid per spatial dimension',ylabel='Error',title='Interacting 3D Navier–Stokes, t = 0.5')
    ax[0].legend(fontsize=9);ax[0].text(.03,.04,'*16–32 vs 48³; 48 vs 64³',transform=ax[0].transAxes,fontsize=8)
    row=json.loads((OUT/'burgers_bandwidth_ablation.json').read_text())
    safe=[r for r in row if r['cutoff'] is not None and r['halo']!=3.]
    ax[1].loglog([r['neurons'] for r in safe],[r['rel_l2'] for r in safe],'o-',c='#2865b0',label='Resolve frequencies + sufficient halo')
    unsafe=[r for r in row if r['cutoff'] is None and r['halo']!=3.]
    ax[1].loglog([r['neurons'] for r in unsafe],[r['rel_l2'] for r in unsafe],'x--',c='#cf6531',label='Keep unresolved frequencies')
    ax[1].set(xlabel='Neurons',ylabel='Relative L2 error',title='Burgers: center resolution matters')
    ax[1].legend(fontsize=8)
    ax[2].loglog([r['neurons'] for r in sparse],[r['velocity_rel_l2_vs_full'] for r in sparse],'o-',label='Velocity',c='#2865b0')
    ax[2].loglog([r['neurons'] for r in sparse],[r['pressure_rel_l2_vs_full'] for r in sparse],'s--',label='Pressure',c='#cf6531')
    ax[2].set(xlabel='Explicit tanh neurons',ylabel='Relative error vs full Fourier state',title='Allocate directions to active modes')
    ax[2].legend(fontsize=9)
    for a in ax:a.grid(alpha=.2)
    fig.savefig(OUT/'nonlinear_validated.png',dpi=180);plt.close(fig)
    print(json.dumps(extra,indent=2))
if __name__=='__main__':main()
