"""Weak/entropy front-family declarations solved by the generic GN/LSMR engine.

The family is a clipped linear ramp (two ReLUs per time slice), not a general
QUILL shock representation. Three physical parameters are center, propagation
speed, and spreading rate. Only prescribed initial mass and spacetime balances
enter optimization. Exact solution formulas appear only in held-out validation.
"""
from __future__ import annotations
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(name,'1')
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/quill-weak-engine-mpl')
from pathlib import Path
import json,math,time
import numpy as np
import torch
from solver.general_features import ConstructedFeatures
from solver.general_residual import ResidualBlock,ResidualProblem,solve_residual,linearize_residual,residual_vector

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/checkpoint_F_applications/expF19_radon_direct_pde/general_residual'
CELLS=np.array([(-.12,.16,.04,.51),(.02,.37,.11,.83),(.14,.43,.20,.77),
                (-.31,.12,.05,.62),(.25,.64,.21,.93),(-.6,.6,.17,.71)])
HELDOUT=np.array([(-.073,.213,.057,.533),(.093,.583,.213,.977),
                  (-.4,.62,.032,.822),(-2.,2.,.137,.719)])
ZERO=(0,0,0,0)


def _rational_moment(q,j):
    """Integral_0^1 z^j/(1+q*z)^j dz, stable at q=0, for j<=3.

    A convergent power series removes the removable singularity at q=0.
    Its 32-term truncation at |q|<.2 is below float64 roundoff here.
    """
    if j==0:return torch.ones_like(q)
    small=abs(q)<.2
    qs=torch.where(small,q,torch.zeros_like(q))
    series=torch.zeros_like(q)
    for k in range(32):
        series=series+((-1.)**k*math.comb(j+k-1,k)/(j+k+1))*qs**k
    a=torch.where(small,torch.ones_like(q),q);r=1+a
    if j==1:exact=(a-torch.log1p(a))/a**2
    elif j==2:exact=(a-2*torch.log1p(a)+1-1/r)/a**3
    elif j==3:exact=(a-3*torch.log1p(a)+2.5-3/r+.5/r**2)/a**4
    else:raise ValueError('Only powers up to three are used')
    return torch.where(small,series,exact)


def spatial_power(xl,xr,t,theta,p,left,right,epsilon):
    """Exact integral_xl^xr u(x,t)^p dx for a clipped linear front."""
    pos,speed,spread=theta.unbind(-1)
    width=epsilon+spread*t;edge=pos+speed*t-width/2;delta=right-left
    def primitive(x):
        y=(x-edge)/width;r=torch.clamp(y,0.,1.)
        result=left**p*(x-edge)
        for j in range(1,p+1):
            result=result+width*math.comb(p,j)*left**(p-j)*delta**j*(r**(j+1)/(j+1)+torch.relu(y-1))
        return result
    return primitive(xr)-primitive(xl)


def temporal_power(x,t0,t1,theta,p,left,right,epsilon):
    """Exact piecewise rational flux integral, with differentiable transitions.

    Split at crossings of the CURRENT represented edges, not target features.
    Thus an arbitrarily thin front cannot fall between fixed quadrature nodes.
    """
    pos,speed,spread=theta.unbind(-1)
    crossings=[]
    for sign in (-1.,1.):
        velocity=speed+sign*spread/2
        stationary=abs(velocity)<1e-13
        safe=torch.where(stationary,torch.ones_like(velocity),velocity)
        crossing=(x-pos-sign*epsilon/2)/safe
        crossing=torch.where(stationary,t0,crossing)
        crossings.append(torch.minimum(torch.maximum(crossing,t0),t1))
    edges=torch.sort(torch.stack([t0,t1,*crossings],dim=-1),dim=-1).values
    result=torch.zeros_like(t0)
    for i in range(3):
        a,b=edges[:,i],edges[:,i+1];length=b-a
        wa=epsilon+spread*a;wb=epsilon+spread*b
        ya=.5+(x-pos-speed*a)/wa;yb=.5+(x-pos-speed*b)/wb
        middle=.5+(x-pos-speed*(a+b)/2)/(epsilon+spread*(a+b)/2)
        q=(wb-wa)/wa
        u0=left+(right-left)*ya
        h=(right-left)*(yb-ya)*(1+q)
        inside=torch.zeros_like(a)
        for j in range(p+1):inside=inside+math.comb(p,j)*u0**(p-j)*h**j*_rational_moment(q,j)
        integral=length*inside
        integral=torch.where(middle<=0,length*left**p,integral)
        integral=torch.where(middle>=1,length*right**p,integral)
        result=result+integral
    return result


def cell_balance(points,theta,left,right,epsilon,entropy=False):
    xl,xr,t0,t1=points.unbind(-1)
    p=2 if entropy else 1
    if epsilon==0:
        # Exact sharp-front limit, used as a weak-only negative control. Speed
        # is bounded positive in this study, so each boundary crosses once.
        pos,speed,_=theta.unbind(-1)
        def spatial(t,p):
            left_length=torch.clamp(pos+speed*t-xl,min=0.)
            left_length=torch.minimum(left_length,xr-xl)
            return left**p*left_length+right**p*(xr-xl-left_length)
        def temporal(x,p):
            before=torch.clamp((x-pos)/speed-t0,min=0.)
            before=torch.minimum(before,t1-t0)
            return right**p*before+left**p*(t1-t0-before)
        return (spatial(t1,p)-spatial(t0,p))/p+(temporal(xr,p+1)-temporal(xl,p+1))/(p+1)
    top=spatial_power(xl,xr,t1,theta,p,left,right,epsilon)/p
    bottom=spatial_power(xl,xr,t0,theta,p,left,right,epsilon)/p
    right_flux=temporal_power(xr,t0,t1,theta,p+1,left,right,epsilon)/(p+1)
    left_flux=temporal_power(xl,t0,t1,theta,p+1,left,right,epsilon)/(p+1)
    return top-bottom+right_flux-left_flux


def make_problem(left,right,enforce_entropy,epsilon=1e-5,downward=False,start=None):
    # Coordinates describe cells (xl,xr,t0,t1). The polynomial constant is
    # unused by the front and pinned to zero; it does not represent a PDE field.
    features=ConstructedFeatures([(-2,2),(-2,2),(0,1),(0,1)],0,backend='polynomial')
    cells=np.array([(-2,2,0,.2),(-2,2,.2,.47),(-2,2,.47,.8)]) if downward else CELLS
    def weak(points,jets,p):return cell_balance(points,p,left,right,epsilon)
    blocks=[ResidualBlock('conservation',cells,weak,(ZERO,))]
    initial=np.array([(-2,2,0,0)])
    def mass(points,jets,p):
        if epsilon==0:return (left-right)*p[:,0]
        return spatial_power(points[:,0],points[:,1],points[:,2],p,1,left,right,epsilon)-2*(left+right)
    blocks.append(ResidualBlock('initial_mass',initial,mass,(ZERO,)))
    blocks.append(ResidualBlock('unused_constant_gauge',initial,lambda x,j,p:j[ZERO][:,0],(ZERO,)))
    if downward or epsilon==0:
        # The downward demonstration is a translating narrow front. Expansion
        # tests below permit unknown spreading and test entropy branch selection.
        blocks.append(ResidualBlock('fixed_width_family',initial,lambda x,j,p:p[:,2],(ZERO,)))
    if enforce_entropy:
        entropy_cells=np.vstack([CELLS,[-2,2,.11,.73]])
        blocks.append(ResidualBlock('entropy',entropy_cells,
            lambda x,j,p:cell_balance(x,p,left,right,epsilon,True),(ZERO,),relation='le'))
    start=([.15,.1,0.] if downward else [0.,.5,.001]) if start is None else start
    return ResidualProblem(features,blocks,parameter_initial=start,
                           parameter_bounds=([-.2,.05,0.],[.2,.95,2.]))


def values(x,t,theta,left,right,epsilon):
    pos,speed,spread=theta
    if epsilon+spread*t==0:return np.where(x<pos+speed*t,left,right).astype(float)
    return left+(right-left)*np.clip(.5+(x-pos-speed*t)/(epsilon+spread*t),0,1)


def heldout_l1(theta,left,right,epsilon,downward,t=.8):
    """Exact integration of absolute piecewise-linear validation differences.

    The reference is used only after fitting. Split at both represented and
    reference edges; a uniform grid would miss a front narrower than its mesh.
    """
    pos,speed,spread=theta;center=pos+speed*t;width=epsilon+spread*t
    edges=sorted(set([-.5,1.5,center-width/2,center+width/2]+([.5*t] if downward else [0.,t])))
    edges=[z for z in edges if -.5<=z<=1.5]
    def difference(z):
        reference=(z<.5*t).astype(float) if downward else np.clip(z/t,0.,1.)
        return values(z,t,theta,left,right,epsilon)-reference
    total=0.
    for a,b in zip(edges[:-1],edges[1:]):
        length=b-a;mid=(a+b)/2
        v=difference(np.array([mid-length/4,mid+length/4]))
        slope=(v[1]-v[0])/(length/2);middle=(v[0]+v[1])/2
        vl,vr=middle-slope*length/2,middle+slope*length/2
        if vl*vr>=0:total+=length*(abs(vl)+abs(vr))/2
        else:total+=length*(vl*vl+vr*vr)/(2*(abs(vl)+abs(vr)))
    return float(total)


def fit(label,left,right,entropy=False,downward=False,epsilon=1e-5,tolerance=1e-10,start=None):
    began=time.perf_counter();problem=make_problem(left,right,entropy,epsilon,downward,start)
    solution=solve_residual(problem,max_iterations=80,tolerance=tolerance,lsmr_tolerance=1e-12,
                            gradient_tolerance=1e-13,step_tolerance=1e-14,
                            max_residual_evaluations=1000,scaling_probes=32)
    p=torch.as_tensor(np.broadcast_to(solution.parameters,(len(HELDOUT),3)).copy())
    points=torch.as_tensor(HELDOUT)
    conservation=cell_balance(points,p,left,right,epsilon).numpy()
    entropy_values=cell_balance(points,p,left,right,epsilon,True).numpy()
    row=dict(label=label,left=left,right=right,epsilon=epsilon,entropy_enforced=entropy,
             parameters=solution.parameters.tolist(),status=solution.status,
             heldout_weak_l2=float(np.linalg.norm(conservation)),
             heldout_weak_max=float(np.max(abs(conservation))),
             heldout_maximum_entropy_violation=float(np.maximum(entropy_values,0).max()),
             heldout_entropy_balances=entropy_values.tolist(),
             heldout_l1_t0_8=heldout_l1(solution.parameters,left,right,epsilon,downward),
             total_seconds=time.perf_counter()-began,metrics=solution.metrics,history=solution.history)
    return row


def width_control():
    points=torch.as_tensor(HELDOUT)
    p=torch.tensor(np.broadcast_to([0.,.5,0.],(len(HELDOUT),3)).copy())
    rows=[]
    for epsilon in [1e-3,1e-5,1e-7,1e-9]:
        weak=cell_balance(points,p,0,1,epsilon).numpy()
        entropy=cell_balance(points,p,0,1,epsilon,True).numpy()
        rows.append(dict(epsilon=epsilon,heldout_weak_l2=float(np.linalg.norm(weak)),
                         maximum_entropy_violation=float(np.maximum(entropy,0).max()),
                         heldout_l1=heldout_l1([0,.5,0],0,1,epsilon,False)))
    return rows


def check_integrals():
    # Independent split Gauss quadrature from the old standalone diagnostic.
    from general_shock_analysis import RampFront,control_volume_balance
    rows=[]
    for theta in ([.017,.43,0.],[.017,.43,.001],[.017,.43,.91],[0,.5,1.]):
        p=torch.tensor(np.broadcast_to(theta,(len(HELDOUT),3)).copy(),requires_grad=True)
        points=torch.as_tensor(HELDOUT)
        for entropy in (False,True):
            actual=cell_balance(points,p,0,1,1e-5,entropy)
            field=RampFront(1e-5,*theta)
            density=(lambda u:u*u/2) if entropy else (lambda u:u)
            flux=(lambda u:u**3/3) if entropy else (lambda u:u*u/2)
            expected=np.array([control_volume_balance(field,density,flux,c) for c in HELDOUT])
            error=float(np.max(abs(actual.detach().numpy()-expected)))
            assert error<5e-10,(theta,entropy,error)
            gradient=torch.autograd.grad(actual.sum(),p)[0].numpy()
            assert np.isfinite(gradient).all()
            rows.append(dict(parameters=list(theta),entropy=entropy,maximum_quadrature_disagreement=error))
    # Check the actual generic engine products on feasible interior parameters.
    problem=make_problem(0,1,True,start=[.017,.43,.12])
    c=np.zeros((1,1));p=problem.parameter_initial
    linear=linearize_residual(problem,c,p)
    rng=np.random.default_rng(121);v=rng.normal(size=4);w=rng.normal(size=len(linear.residual))
    lhs=w@linear.operator.matvec(v);rhs=v@linear.operator.rmatvec(w)
    h=2e-6
    fd=(residual_vector(problem,c+h*v[:1,None],p+h*v[1:])-residual_vector(problem,c-h*v[:1,None],p-h*v[1:]))/(2*h)
    error=float(np.linalg.norm(fd-linear.operator.matvec(v))/np.linalg.norm(fd))
    assert abs(lhs-rhs)<1e-10
    assert error<2e-5,error
    return dict(integral_comparisons=rows,adjoint_error=float(abs(lhs-rhs)),directional_derivative_error=error)


def main():
    torch.set_num_threads(1);OUT.mkdir(parents=True,exist_ok=True)
    checks=check_integrals()
    rows=[fit('downward_front',1,0,downward=True),
          fit('expansion_conservation_only',0,1),
          fit('expansion_conservation_and_entropy',0,1,entropy=True),
          fit('expansion_sharp_conservation_only',0,1,epsilon=0.,start=[.1,.3,0.])]
    assert rows[0]['status']=='converged' and abs(rows[0]['parameters'][1]-.5)<1e-9
    assert rows[1]['heldout_l1_t0_8']>.19 and rows[1]['heldout_maximum_entropy_violation']>.06
    assert rows[2]['status']=='converged' and abs(rows[2]['parameters'][2]-1)<1e-8
    assert rows[2]['heldout_l1_t0_8']<3e-6 and rows[2]['heldout_weak_l2']<1e-9
    assert rows[3]['status']=='converged' and rows[3]['heldout_l1_t0_8']>.19
    output=dict(scope='Restricted piecewise-linear front family, not automatic QUILL shock discovery',
        engine='The same solve_residual Gauss-Newton/LSMR engine as the smooth PDE studies',
        fitting_inputs='Initial step mass, spacetime conservation, optional quadratic entropy inequalities; no target speed or solution samples',
        quadrature='Exact piecewise power/rational antiderivatives; stable power series for removable small-spreading singularities',
        checks=checks,rows=rows,width_control=width_control(),
        caveats='A finite-width ramp has an O(epsilon) weak defect on front-cutting cells. Conservation-only strict solve may hit iteration limit at that floor; width refinement can pass weak tolerances while retaining the nonphysical expansion shock. One quadratic entropy and finitely many cells select the intended branch within this family, not all possible weak solutions.')
    (OUT/'weak_engine_metrics.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps([{k:v for k,v in row.items() if k not in ['metrics','history']} for row in rows],indent=2))
    plot(output)


def plot(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    x=np.linspace(-.3,1.1,3000);t=.8
    ax[0].plot(x,(x<.5*t).astype(float),'k--',label='Entropy shock, validation only')
    ax[1].plot(x,np.clip(x/t,0,1),'k--',label='Rarefaction, validation only')
    for row,color in zip(output['rows'][:3],['#2369bc','#c64b48','#239359']):
        j=0 if row['label']=='downward_front' else 1
        label='Fitted front' if j==0 else ('Conservation + entropy' if row['entropy_enforced'] else 'Conservation only')
        ax[j].plot(x,values(x,t,row['parameters'],row['left'],row['right'],row['epsilon']),label=label,color=color)
        ax[2].semilogy([h['iteration'] for h in row['history']],
            [max(h['maximum_scaled_block_rms'],1e-16) for h in row['history']],label=label,color=color)
    for i in [0,1]:ax[i].set(xlabel='x',ylabel='u(x, .8)');ax[i].legend(fontsize=8)
    ax[0].set_title('Speed learned from conservation')
    ax[1].set_title('Entropy rejects the wrong weak branch')
    ax[2].set(xlabel='Generic GN iteration',ylabel='Maximum scaled block RMS',title='Same engine, different declarations')
    ax[2].set_xscale('symlog',linthresh=1)
    ax[2].legend(fontsize=8)
    for a in ax:a.grid(alpha=.2)
    fig.savefig(OUT/'weak_engine_comparison.png',dpi=180);plt.close(fig)


if __name__=='__main__':main()
