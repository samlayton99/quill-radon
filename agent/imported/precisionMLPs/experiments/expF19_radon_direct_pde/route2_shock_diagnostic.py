"""Exact one-neuron shock capacity/physics diagnostic, not a PDE solve.

For stationary inviscid Burgers, -tanh(gamma*x) approaches the admissible
compression; +tanh(gamma*x) approaches a non-admissible expansion. Both
satisfy the limiting flux jump. Entropy separates them.
"""
import json
from pathlib import Path
import numpy as np
from scipy.integrate import quad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parents[2]/"results/checkpoint_F_applications/expF19_radon_direct_pde/route2_generality/shock_diagnostic"

def main():
    gamma=np.logspace(0,4,41)
    # Integral over [-1,1]: use a stable log-cosh identity.
    l1=2*(np.log(2.)-np.log1p(np.exp(-2*gamma)))/gamma
    # ||u*u_x||_2^2: t=tanh(gamma*x) makes integral elementary.
    t=np.tanh(gamma)
    strong=np.sqrt(2*gamma*(t**3/3-t**5/5))
    assert np.allclose(l1[-1]*gamma[-1],2*np.log(2))
    assert np.allclose(strong[-1]/np.sqrt(gamma[-1]),np.sqrt(4/15))
    # Quadrature independent checks at moderate widths.
    for g,err,res in zip(gamma[::10],l1[::10],strong[::10]):
        check=2*quad(lambda z: 1-np.tanh(z),0,min(g,40),epsabs=1e-12)[0]/g
        assert abs(check-err)<1e-10
    row=dict(scope="Capacity and entropy diagnostic only; not a solved shock IVP",
      equation="u_t + (u^2/2)_x = 0",networks="u=+-tanh(gamma*x)",
      compression=dict(left=1,right=-1,flux_jump=0,quadratic_entropy_flux_jump=-2/3),
      expansion=dict(left=-1,right=1,flux_jump=0,quadratic_entropy_flux_jump=2/3),
      explanation="Entropy inequality requires jump q(right)-q(left)<=0 for a stationary shock; q=u^3/3. Uniform approximation error is at least1 for any continuous approximation to this jump. Smooth MLPs can converge in L1, while their strong residual diverges.",
      gamma=gamma.tolist(),l1_error=l1.tolist(),strong_residual_l2=strong.tolist())
    OUT.mkdir(parents=True,exist_ok=True);(OUT/"metrics.json").write_text(json.dumps(row,indent=2)+"\n")
    fig,axes=plt.subplots(1,2,figsize=(10,3.5),constrained_layout=True)
    x=np.linspace(-1,1,1001)
    axes[0].plot(x,-np.tanh(30*x),label="Compression: entropy admissible")
    axes[0].plot(x,np.tanh(30*x),label="Expansion: entropy violated")
    axes[0].set(xlabel="x",ylabel="u",title="Same zero limiting flux jump")
    axes[0].legend(fontsize=8)
    axes[1].loglog(gamma,l1,label="L1 field error")
    axes[1].loglog(gamma,strong,label="Strong PDE residual L2")
    axes[1].set(xlabel="Steepness gamma",title="Sharper shock; larger strong residual")
    axes[1].legend(fontsize=8)
    for ax in axes:ax.grid(alpha=.2)
    fig.savefig(OUT/"shock_limitation.png",dpi=170)
    print("Shock capacity/entropy identities verified; this is not a PDE solve.")

if __name__=="__main__":main()
