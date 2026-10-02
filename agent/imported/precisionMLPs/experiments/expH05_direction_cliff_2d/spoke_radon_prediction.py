"""Independent Radon/Fourier predictions for saved 2D spoke coefficients.

No profile or amplitude is fitted to the solved coefficients. Radial profiles use
the Abel inverse of angular averaging; for decaying radial targets this is the
normalized filtered Radon profile. Composition uses its analytic Fourier atoms
and the direction-snapping construction in the local higher-dimensional notes.
"""
from __future__ import annotations

import json
import argparse
from pathlib import Path
import numpy as np
from scipy.special import dawsn, iv
from numpy.polynomial.legendre import leggauss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import spoke_profiles as S

OUT=S.OUT/"radon_prediction"


def gaussian_profile(s,sigma):
    z=np.asarray(s)/sigma
    g=1-2*z*dawsn(z)
    gp=((4*z*z-2)*dawsn(z)-2*z)/sigma
    return g,gp


def runge_profile(s):
    z=np.sqrt(8)*np.asarray(s)
    den=1+z*z
    g=1/den-z*np.arcsinh(z)/den**1.5
    gp=np.sqrt(8)*(-3*z/den**2+(2*z*z-1)*np.arcsinh(z)/den**2.5)
    return g,gp


def radial_abel(s,key,order=96):
    """G(s)=integral sin(a)[F(s sin(a))+s sin(a) F'(s sin(a))] da.

    For any even analytic radial F, mean_theta G(r cos(theta))=F(r).
    Gaussian-damped packet: standard filtered-Radon profile. Undamped cosine:
    analytic ridge identity, without assuming ordinary Radon integrals exist.
    """
    nodes,weights=leggauss(order)
    sine=np.sin((nodes+1)*np.pi/4)
    weights=weights*np.pi/4
    r=np.asarray(s)[...,None]*sine
    if key=='fast_waves':
        k=6*np.pi/np.sqrt(2)
        f=np.cos(k*r);fp=-k*np.sin(k*r);fpp=-k*k*f
    else:
        a=1/(2*.18**2);k=10*np.pi/np.sqrt(2)
        e=.8*np.exp(-a*r*r);c=np.cos(k*r);sn=np.sin(k*r)
        f=e*c;fp=e*(-2*a*r*c-k*sn)
        fpp=e*((4*a*a*r*r-2*a-k*k)*c+4*a*k*r*sn)
    return np.sum(weights*sine*(f+r*fp),axis=-1),np.sum(weights*sine**2*(2*fp+r*fpp),axis=-1)


def radial_profiles(key,t,V,cone=False,order=96):
    """M times the theoretical per-spoke contribution and its derivative."""
    if cone:
        nodes,weights=leggauss(16)
        offsets=nodes*np.pi/(2*len(V));weights=weights/2
    else:
        offsets=np.array([0.]);weights=np.array([1.])
    base=np.arctan2(V[:,1],V[:,0])
    total=np.zeros_like(t,dtype=np.result_type(t,float));deriv=total.copy()
    for delta,weight in zip(offsets,weights):
        v=np.column_stack([np.cos(base+delta),np.sin(base+delta)])
        if key=='gauss_bump':
            s=t+(v@(S.original.X0-S.original.A_BUMP))[:,None]
            g,gp=gaussian_profile(s,.5)
        elif key=='radial_runge':
            s=t+(v@(S.original.X0-S.original.A_RAD))[:,None]
            g,gp=runge_profile(s)
        elif key=='fast_waves':
            s=t+(v@(S.original.X0-S.original.A_RAD))[:,None]
            g,gp=radial_abel(s,key,order)
        elif key=='spatial_packet':
            g,gp=radial_abel(t,key,order)
            s=t+(v@(S.original.X0-S.original.A_BUMP))[:,None]
            b,bp=gaussian_profile(s,1.)
            g=g+b;gp=gp+bp
        else:
            raise ValueError(key)
        total+=weight*g;deriv+=weight*gp
    return total,deriv


def composition_atoms(V,cutoff=18):
    """exp(sin(pi*x)*cos(pi*y)) as a product of two Bessel Fourier series."""
    m,n=np.meshgrid(np.arange(-cutoff,cutoff+1),np.arange(-cutoff,cutoff+1),indexing='ij')
    m=m.ravel();n=n.ravel()
    xi=np.pi*np.column_stack([m+n,m-n])
    coef=iv(np.abs(m),.5)*iv(np.abs(n),.5)*np.exp(-.5j*np.pi*(m+n))
    coef=coef*np.exp(1j*(xi@S.original.X0))
    dot=xi@V.T
    assignment=np.argmax(np.abs(dot),axis=1)
    sign=np.sign(dot[np.arange(len(dot)),assignment])
    omega=sign*np.linalg.norm(xi,axis=1)
    return xi,coef,assignment,omega


def atom_profiles(t,V,gamma=None):
    _,coef,assignment,omega=composition_atoms(V)
    g=np.zeros_like(t);gp=g.copy();p=g.copy()
    a=None if gamma is None else np.pi/(2*gamma)
    for j in range(len(V)):
        keep=assignment==j;om=omega[keep];c=coef[keep]
        e=np.exp(1j*t[j,:,None]*om)
        g[j]=np.real(e@c)
        gp[j]=np.real(e@(1j*om*c))
        if a is not None:
            z=a*om
            factor=np.ones_like(z);nz=np.abs(z)>1e-12
            factor[nz]=np.sinh(z[nz])/z[nz]
            p[j]=np.real(e@(1j*om*c*factor))
    return g,gp,p


def metrics(actual,pred,mask):
    a=actual[mask];p=pred[mask]
    return dict(relative_error=float(np.linalg.norm(a-p)/np.linalg.norm(a)),
                correlation=float(np.corrcoef(a,p)[0,1]),
                predicted_to_actual_norm=float(np.linalg.norm(p)/np.linalg.norm(a)))


def check_reconstruction(key,V,cone=False,n=400):
    x=S.original.ball(n,.36,S.original.X0,np.random.default_rng(991))
    t=(x-S.original.X0)@V.T
    if key=='composition':
        pred=atom_profiles(t.T,V)[0].sum(axis=0)
    else:
        pred=radial_profiles(key,t.T,V,cone)[0].mean(axis=0)
    truth=S.TARGETS[S.KEYS.index(key)][3](x)
    return dict(rel_l2=float(np.linalg.norm(pred-truth)/np.linalg.norm(truth)),
                max_abs=float(np.max(np.abs(pred-truth))))


def main():
    global OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--angle-rule',choices=['recorded','endpoint'],default='recorded')
    args=parser.parse_args()
    source=S.OUT if args.angle_rule=='recorded' else S.OUT/'endpoint_direction_check'
    if args.angle_rule=='recorded':OUT=OUT/'recorded'
    OUT.mkdir(parents=True,exist_ok=True)
    output={'comparison':'independent predictions; no regression, amplitude fit, or spoke shifts',
            'angle_rule':args.angle_rule,
            'source_directory':str(source),
            'mask':'all 32 spokes, |t| < 0.3; error denominator is norm of solved coefficients',
            'cells':[], 'validation':{}, 'prediction_types':{
                'gauss_bump':'filtered Radon / angular quadrature',
                'radial_runge':'filtered Radon / angular quadrature',
                'spatial_packet':'filtered Radon / angular quadrature',
                'fast_waves':'analytic radial ridge identity; ordinary Radon integral diverges',
                'composition':'Fourier-atom direction snapping; no smooth angular density'}}
    exports={}
    for N in [64,128]:
        sol=dict(np.load(source/f'solution_M32_N{N}.npz'))
        V=sol['directions'];M=len(V);h=1/N;gamma=float(sol['gamma']);a=np.pi/(2*gamma)
        t=sol['centers']-(V@S.original.X0)[:,None]
        leading=np.zeros_like(sol['w']);corrected=leading.copy();snapped=leading.copy()
        rows=[]
        for k,key in enumerate(S.KEYS):
            actual=sol['w'][:,:,k]
            if key=='composition':
                g,d,p=atom_profiles(t,V,gamma)
                leading[:,:,k]=h*d/2;corrected[:,:,k]=h*p/2;snapped[:,:,k]=corrected[:,:,k]
                theory_derivative=d
            else:
                g,d=radial_profiles(key,t,V)
                gc,_=radial_profiles(key,t+1j*a,V)
                leading[:,:,k]=h*d/(2*M)
                corrected[:,:,k]=h*np.imag(gc)/(2*M*a)
                # Cone averaging is only needed for smooth spectral densities.
                if key in ['gauss_bump','radial_runge']:
                    cone_complex,_=radial_profiles(key,t+1j*a,V,cone=True)
                    snapped[:,:,k]=h*np.imag(cone_complex)/(2*M*a)
                else:snapped[:,:,k]=corrected[:,:,k]
                theory_derivative=d/M
            fitted_d=np.empty_like(actual)
            for j in range(M):
                z=gamma*(t[j,:,None]-t[j,None,:]);fitted_d[j]=gamma*(1-np.tanh(z)**2)@actual[j]
            row=dict(target=key,width=M*N,rel_l2=float(sol['rel_l2'][k]),
                     leading=metrics(actual,leading[:,:,k],np.abs(t)<.3),
                     tanh_corrected=metrics(actual,corrected[:,:,k],np.abs(t)<.3),
                     corrected_all_centers=metrics(actual,corrected[:,:,k],np.ones_like(t,dtype=bool)),
                     component_derivative=metrics(fitted_d,theory_derivative,np.abs(t)<.3),
                     predicted_function_error=check_reconstruction(key,V))
            if key in ['gauss_bump','radial_runge']:
                row['snapped_cone_corrected']=metrics(actual,snapped[:,:,k],np.abs(t)<.3)
                row['snapped_cone_function_error']=check_reconstruction(key,V,cone=True)
            rows.append(row)
        output['cells'].extend(rows)
        np.savez_compressed(OUT/f'predictions_N{N}.npz',leading=leading,corrected=corrected,
                            snapped=snapped,actual=sol['w'],centers=t,directions=V,keys=np.array(S.KEYS))
        exports[N]=(sol,t,leading,corrected)
        print('width',M*N,flush=True)
        for row in rows:print(row['target'], 'leading',row['leading'], 'corrected',row['tanh_corrected'],flush=True)
    # Quadrature and analytic identities checked independently of any network.
    s=np.linspace(-.7,.7,171)+.04j
    for key in ['fast_waves','spatial_packet']:
        g1,d1=radial_abel(s,key,64);g2,d2=radial_abel(s,key,128)
        err=float(max(np.max(np.abs(g1-g2)),np.max(np.abs(d1-d2))))
        output['validation'][key+'_quadrature_64_128']=err
        assert err<2e-11
    vd=S.original.even_directions(2,256)
    for key in ['gauss_bump','radial_runge','fast_waves','spatial_packet']:
        error=check_reconstruction(key,vd,n=100)
        output['validation'][key+'_angular_reconstruction']=error
        assert error['max_abs']<2e-12
    xi,coef,_,_=composition_atoms(vd)
    x=S.original.ball(100,.36,S.original.X0,np.random.default_rng(802))
    pred=np.real(np.exp(1j*((x-S.original.X0)@xi.T))@coef)
    truth=S.original.f_composition(x)
    err=float(np.max(np.abs(pred-truth)))
    output['validation']['composition_unsnapped_fourier_max_abs']=err
    assert err<1e-13
    # The inverse-kernel factor should undo the exact tanh derivative multiplier.
    gamma=32.;omega=np.array([1.,7.,23.]);a=np.pi/(2*gamma)
    khat=a*omega/np.sinh(a*omega)
    correction=np.sinh(a*omega)/(a*omega)
    output['validation']['kernel_factor_product_max_abs']=float(np.max(np.abs(khat*correction-1)))
    # Plot raw solved coefficients against independent predictions.
    sol,t,leading,corrected=exports[128]
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(3,2,figsize=(12,11),layout='constrained')
    for r,key in enumerate(['gauss_bump','radial_runge','spatial_packet']):
        k=S.KEYS.index(key)
        for c,j in enumerate([0,16]):
            ax=axes[r,c];keep=np.abs(t[j])<.3
            ax.scatter(t[j,keep],sol['w'][j,keep,k],s=16,c='#252525',label='Solved coefficients')
            ax.plot(t[j,keep],leading[j,keep,k],c='#e67e22',ls='--',lw=1.6,label='Radon derivative × h/2')
            ax.plot(t[j,keep],corrected[j,keep,k],c='#2874a6',lw=1.6,label='Radon + tanh bandwidth correction')
            angle=np.degrees(np.arctan2(sol['directions'][j,1],sol['directions'][j,0]))
            ax.set_title(f'{S.TARGETS[k][1]}, {angle:.4g} degrees')
            ax.set_xlabel('center t along spoke');ax.set_ylabel('readout coefficient a')
            ax.grid(alpha=.18);ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
            if r==0 and c==0:ax.legend(fontsize=8)
    fig.suptitle('Do solved coefficients follow the independent Radon prediction?\n'
                 f'4096 neurons; {args.angle_rule} angles; no fitted prediction parameters',fontsize=15)
    fig.savefig(OUT/'independent_radon_coefficients.png',dpi=165)
    fig.savefig(OUT/'independent_radon_coefficients.pdf');plt.close(fig)
    (OUT/'comparison.json').write_text(json.dumps(output,indent=2)+'\n')
    print('validation',output['validation'],flush=True)


if __name__=='__main__':main()
