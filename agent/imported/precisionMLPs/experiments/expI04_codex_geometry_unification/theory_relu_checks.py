"""Independent target-to-readout checks for deep 1-D ReLU charts.

No training or fitting enters the coefficient predictor. Dense least squares is
an independent reference for the frozen geometry. NumPy/SciPy only, one thread.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/codex_geometry_theory_mpl")
from pathlib import Path
import json
import numpy as np
from scipy.linalg import lstsq
from scipy.integrate import cumulative_trapezoid
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/checkpoint_I_depth_theory/expI04_codex_geometry_unification/theory"
OUT.mkdir(parents=True, exist_ok=True)

def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))

def target(z):
    return np.sin(np.pi*z) + .15*np.cos(2*np.pi*z)

def target2(z):
    return -np.pi**2*np.sin(np.pi*z) - .6*np.pi**2*np.cos(2*np.pi*z)

def regr_features(z, centers, gammas):
    return np.column_stack((np.ones_like(z), z, np.maximum(z[:, None]-centers, 0)*gammas))

def interp_readout(nodes, values, gammas):
    slopes = np.diff(values)/np.diff(nodes)
    coeff = np.diff(slopes)/gammas
    return np.r_[values[0]-slopes[0]*nodes[0], slopes[0], coeff]

def run_chart():
    # Strictly monotone PL chart, represented exactly by a first ReLU layer.
    xk = np.array([-1., -.67, -.2, .35, 1.])
    zk = np.array([-1., -.5, 0., .5, 1.])
    x = np.linspace(-1, 1, 16385)
    z = np.interp(x, xk, zk)
    y = target(z)
    all_metrics = []
    saved = None
    for n in (17, 33, 65, 129):
        nodes = np.linspace(-1, 1, n)
        c = nodes[1:-1]
        h = nodes[1]-nodes[0]
        gamma = 1.5 + .65*np.sin(7*c + .3)
        A = regr_features(z, c, gamma)
        # Gaussian quadrature on each common latent cell integrates in the
        # original input measure dx=dz/g'. Avoid spurious coefficient changes
        # from a uniform-x sampling grid that misses many physical knots.
        breaks = np.unique(np.r_[nodes,zk])
        leg,legw = np.polynomial.legendre.leggauss(16)
        zq = ((breaks[:-1,None]+breaks[1:,None])/2
              +(breaks[1:,None]-breaks[:-1,None])*leg/2).ravel()
        wq = ((breaks[1:,None]-breaks[:-1,None])*legw/2).ravel()
        cell = np.clip(np.searchsorted(zk,zq)-1,0,len(zk)-2)
        wq *= np.diff(xk)[cell]/np.diff(zk)[cell]
        Aq = regr_features(zq,c,gamma)
        solved = lstsq(Aq*np.sqrt(wq[:,None]),target(zq)*np.sqrt(wq),
                       cond=1e-13,lapack_driver="gelsd")[0]
        pred_interp = interp_readout(nodes, target(nodes), gamma)
        pred_qi = h*target2(c)/gamma
        # Exact infinite-uniform-grid L2 multiplier, no target-fit dependence.
        def mult(w):
            return np.sinc(w*h/(2*np.pi))**4 / ((2+np.cos(w*h))/3)
        pred_spectral = h*(-np.pi**2*mult(np.pi)*np.sin(np.pi*c)
                            -.6*np.pi**2*mult(2*np.pi)*np.cos(2*np.pi*c))/gamma
        # Density jumps in the input fitting norm and interval boundaries spoil
        # exact translation symmetry nearby. State their exclusion explicitly.
        distance = np.min(abs(c[:, None]-zk[None, :]), axis=1)
        clean = distance > 3.01*h
        m = {"nodes": n, "hidden_knots": len(c),
             "ls_function_rel_l2": rel(A@solved, y),
             "interpolant_function_rel_l2": rel(A@pred_interp, y),
             "all_coeff_interpolation_rel_error": rel(pred_interp[2:], solved[2:]),
             "all_coeff_qi_rel_error": rel(pred_qi, solved[2:]),
             "interior_coefficients": int(clean.sum())}
        if np.any(clean):
            m.update({"interior_coeff_qi_rel_error": rel(pred_qi[clean], solved[2:][clean]),
                      "interior_coeff_spectral_rel_error": rel(pred_spectral[clean], solved[2:][clean]),
                      "interior_coeff_interpolation_rel_error": rel(pred_interp[2:][clean], solved[2:][clean])})
        all_metrics.append(m)
        if n == 65:
            saved = (x,z,y,c,gamma,A,solved,pred_interp,pred_qi,clean)
    x,z,y,c,gamma,A,solved,pred_interp,pred_qi,clean = saved
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    xc = np.interp(c, zk, xk)
    ax[0].plot(x, z, color="black", lw=2, label="first-layer chart g(x)")
    ax[0].scatter(xc,c,c=gamma,s=18,cmap="viridis",label="second-layer thresholds")
    ax[0].set(xlabel="input x", ylabel="latent z", title="Uniform latent knots become nonuniform input knots")
    ax[0].legend(fontsize=8)
    ax[1].plot(xc, solved[2:], "o", ms=4,label="independent function-LS readout")
    ax[1].plot(xc, pred_qi, "-", lw=1.5,label="prediction: h F''(c) / gamma")
    ax[1].plot(xc[~clean],solved[2:][~clean],"o",mfc="none",mec="gray",ms=7,label="near chart kink / boundary")
    ax[1].set(xlabel="physical threshold x",ylabel="raw readout",title="Target-derived coefficients through the chart")
    ax[1].legend(fontsize=8)
    ax[2].plot(x,y,"k--",lw=2,label="target F(g(x))")
    ax[2].plot(x,A@pred_interp,label="predicted interpolation coefficients")
    ax[2].plot(x,A@solved,lw=1,label="LS reference")
    ax[2].set(xlabel="input x",ylabel="function",title="The independent prediction is a usable function")
    ax[2].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT/"relu_chart_prediction.png",dpi=180); plt.close(fig)
    return all_metrics

def run_fold():
    x = np.linspace(-1,1,16385)
    z = abs(x) # first layer ReLU(x)+ReLU(-x)
    nodes = np.linspace(0,1,33)
    c = nodes[1:-1]
    A = regr_features(z,c,np.ones_like(c))
    even = np.cos(np.pi*x)
    odd = np.sin(np.pi*x)
    ve = lstsq(A,even,cond=1e-13)[0]
    vo = lstsq(A,odd,cond=1e-13)[0]
    vi = interp_readout(nodes,np.cos(np.pi*nodes),np.ones_like(c))
    # For a symmetric uniform physical grid, target curvature charges at +/-c.
    h=nodes[1]-nodes[0]
    alpha_even_plus=(np.cos(np.pi*(c+h))-2*np.cos(np.pi*c)+np.cos(np.pi*(c-h)))/h
    alpha_even_minus=alpha_even_plus.copy()
    alpha_odd_plus=(np.sin(np.pi*(c+h))-2*np.sin(np.pi*c)+np.sin(np.pi*(c-h)))/h
    alpha_odd_minus=-alpha_odd_plus
    fig,ax=plt.subplots(1,3,figsize=(15,4.2))
    ax[0].plot(x,z,color="black")
    for cc in c[::5]:
        ax[0].plot([-cc,cc],[cc,cc],"o-",ms=4,alpha=.7)
    ax[0].set(xlabel="input x",ylabel="latent |x|",title="One latent knot has two physical crossings")
    ax[1].plot(c,alpha_even_plus,label="even target: either crossing")
    ax[1].plot(c,alpha_odd_plus,label="odd target: + crossing")
    ax[1].plot(c,alpha_odd_minus,"--",label="odd target: - crossing")
    ax[1].set(xlabel="latent knot c",ylabel="target-predicted readout",title="One coefficient cannot satisfy opposite charges")
    ax[1].legend(fontsize=8)
    ax[2].plot(x,even,"k--",label="even target")
    ax[2].plot(x,A@vi,label="predicted even interpolant")
    ax[2].plot(x,odd,"--",color="firebrick",label="odd target")
    ax[2].plot(x,A@vo,color="firebrick",label="best odd fit: zero")
    ax[2].set(xlabel="input x",ylabel="function",title="Folding removes information; readout cannot repair it")
    ax[2].legend(fontsize=8)
    fig.tight_layout();fig.savefig(OUT/"relu_fold_obstruction.png",dpi=180);plt.close(fig)
    return {"even_ls_rel_l2":rel(A@ve,even),"even_predicted_interpolant_rel_l2":rel(A@vi,even),
            "even_coeff_interpolant_vs_ls_rel_error":rel(vi[2:],ve[2:]),
            "odd_ls_rel_l2":rel(A@vo,odd),"odd_prediction_max_abs":float(abs(A@vo).max()),
            "even_charge_mismatch":float(np.linalg.norm(alpha_even_plus-alpha_even_minus)),
            "odd_charge_mismatch_relative":rel(alpha_odd_plus,alpha_odd_minus)}

def run_density():
    x=np.linspace(-1,1,100001)
    a=100.; center=.2
    y=np.exp(-a*(x-center)**2)
    y2=(4*a*a*(x-center)**2-2*a)*y
    density=abs(y2)**.4 + 1e-4
    cumulative=cumulative_trapezoid(density,x,initial=0)
    cumulative/=cumulative[-1]
    ms=[]
    for n in (17,33,65,129):
        uniform=np.linspace(-1,1,n)
        adaptive=np.interp(np.linspace(0,1,n),cumulative,x)
        fu=np.exp(-a*(uniform-center)**2)
        fa=np.exp(-a*(adaptive-center)**2)
        eu=rel(np.interp(x,uniform,fu),y)
        ea=rel(np.interp(x,adaptive,fa),y)
        ms.append({"nodes":n,"uniform_rel_l2":eu,"curvature_density_rel_l2":ea,"improvement":eu/ea})
    n=33
    uniform=np.linspace(-1,1,n)
    adaptive=np.interp(np.linspace(0,1,n),cumulative,x)
    yu=np.interp(x,uniform,np.exp(-a*(uniform-center)**2))
    ya=np.interp(x,adaptive,np.exp(-a*(adaptive-center)**2))
    fig,ax=plt.subplots(1,3,figsize=(15,4.2))
    ax[0].plot(x,y,"k",label="target")
    ax[0].scatter(uniform,np.full(n,-.07),s=8,label="uniform knots")
    ax[0].scatter(adaptive,np.full(n,-.13),s=8,label="predicted density knots")
    ax[0].set(xlabel="x",ylabel="function",title="Uniform spacing spends knots away from the signal")
    ax[0].legend(fontsize=8)
    ax[1].plot(x,abs(y-yu),label="uniform")
    ax[1].plot(x,abs(y-ya),label="curvature-density prediction")
    ax[1].set(xlabel="x",ylabel="absolute error",title="Same 33 knots; target values only; no LS")
    ax[1].legend(fontsize=8)
    ax[2].loglog([m["nodes"] for m in ms],[m["uniform_rel_l2"] for m in ms],"o-",label="uniform")
    ax[2].loglog([m["nodes"] for m in ms],[m["curvature_density_rel_l2"] for m in ms],"o-",label="predicted density")
    ax[2].set(xlabel="knots",ylabel="relative L2 error",title="The density prediction improves every resolution")
    ax[2].legend(fontsize=8)
    fig.tight_layout();fig.savefig(OUT/"uniformity_counterexample.png",dpi=180);plt.close(fig)
    return ms

if __name__ == "__main__":
    metrics={"monotone_chart":run_chart(),"folded_chart":run_fold(),"density":run_density()}
    (OUT/"metrics.json").write_text(json.dumps(metrics,indent=2)+"\n")
    print(json.dumps(metrics,indent=2))
