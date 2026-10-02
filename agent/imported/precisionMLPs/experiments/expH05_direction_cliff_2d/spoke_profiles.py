"""Inspect individual 1D spoke functions in reproduced expH05 floor solutions.

Uses the original geometry, samples, SVD solver, and cutoff. This is a diagnostic,
not a new parameter sweep. Saved coefficients permit plotting without another solve.
Run: OPENBLAS_NUM_THREADS=6 VECLIB_MAXIMUM_THREADS=6 MPLCONFIGDIR=/tmp/precision-spokes-mpl \
     .venv/bin/python experiments/expH05_direction_cliff_2d/spoke_profiles.py
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import dawsn

import run as original

OUT = original.RESULTS_DIR / "spoke_profiles"
KEYS = ["gauss_bump", "radial_runge", "fast_waves", "composition", "spatial_packet"]
TARGETS = [next(t for t in original.TARGETS if t[0] == key) for key in KEYS]
R = 0.4


def recorded_model(M, N):
    """Reproduce the half-step fan in the August results, before commit b8ffdf0.

    The shared generator now includes axes. Keep that intentional project change;
    explicitly restore the historical geometry only in this diagnostic.
    """
    theta = .5 * np.pi / M + np.arange(M) * np.pi / M
    V = np.column_stack([np.cos(theta), np.sin(theta)])
    for i in range(M):
        if V[i, np.argmax(np.abs(V[i]))] < 0:
            V[i] *= -1
    model = original.RecenteredRidge.__new__(original.RecenteredRidge)
    dirs, centers, gammas = [], [], []
    for v in V:
        T = original.MARGIN * R * float(np.linalg.norm(v))
        h = 2*T/N
        t = -T + (np.arange(N)+.5)*h
        dirs.append(np.repeat(v[None,:],N,axis=0))
        centers.append(float(v @ original.X0)+t)
        gammas.append(np.full(N,original.LAMBDA/h))
    model.unique_directions = V
    model.directions = np.vstack(dirs)
    model.centers = np.concatenate(centers)
    model.gammas = np.concatenate(gammas)
    return model


def fit(M, N, recorded=True):
    path = OUT / f"solution_M{M}_N{N}.npz"
    if path.exists():
        return dict(np.load(path))
    start = time.monotonic()
    model = recorded_model(M, N) if recorded else original.RecenteredRidge(M, N, original.X0, R)
    x = original.ball(8 * M * N, R, original.X0, np.random.default_rng(0))
    y = np.column_stack([t[3](x) for t in TARGETS])
    print(f"Solving M={M}, N={N}, width={M*N}", flush=True)
    w, b, info = original.solve_many(model.features(x), y)
    xt = original.ball(20000, .9 * R, original.X0, np.random.default_rng(1))
    yt = np.column_stack([t[3](xt) for t in TARGETS])
    pred = np.vstack([model.features(z) @ w + b for z in np.array_split(xt, 20)])
    errors = np.linalg.norm(pred - yt, axis=0) / np.linalg.norm(yt, axis=0)
    result = dict(w=w.reshape(M, N, len(KEYS)), bias=b, directions=model.unique_directions,
                  centers=model.centers.reshape(M, N), gamma=model.gammas[0],
                  M=M, N=N, r=R, keys=np.array(KEYS), rel_l2=errors,
                  max_abs=np.max(np.abs(pred-yt), axis=0), rank=info["rank"],
                  x0=original.X0, rcond=original.RCOND,
                  angle_rule="half-step, as recorded before b8ffdf0" if recorded else "endpoint, axes included",
                  seconds=time.monotonic()-start)
    # Independent grouped reconstruction must agree with the flat network.
    grouped = np.zeros((100, len(KEYS))) + b
    for j, v in enumerate(result["directions"]):
        grouped += np.tanh(float(result["gamma"]) *
                           (xt[:100] @ v[:, None] - result["centers"][j])) @ result["w"][j]
    result["grouped_max_difference"] = np.max(np.abs(grouped-pred[:100]))
    assert result["grouped_max_difference"] < 1e-12
    np.savez_compressed(path, **result)
    print(f"Saved {path.name}: {dict(zip(KEYS, errors))}, {result['seconds']:.1f}s", flush=True)
    return result


def profiles(sol, t):
    """t is offset from each direction's projection of the data-ball center."""
    M, N = int(sol["M"]), int(sol["N"])
    c = sol["centers"] - (sol["directions"] @ original.X0)[:, None]
    z = float(sol["gamma"]) * (t[None, :, None] - c[:, None, :])
    phi = np.tanh(z)
    g = np.einsum("msn,mnk->msk", phi, sol["w"])
    deriv = np.einsum("msn,mnk->msk", float(sol["gamma"]) * (1-phi*phi), sol["w"])
    g0 = np.einsum("mn,mnk->mk", np.tanh(-float(sol["gamma"])*c), sol["w"])
    return g, deriv, g0, c


def gaussian_canonical(t, v):
    """pi times the filtered Radon profile, so angular averaging recovers f.

    Fourier convention exp(-i omega s): q=(1/2pi) Lambda Rf.
    For f=exp(-||x-a||^2/sigma^2), pi*q=1-2 z Dawson(z).
    """
    z = (t + v @ (original.X0-original.A_BUMP)) / .5
    g = 1-2*z*dawsn(z)
    d = ((4*z*z-2)*dawsn(z)-2*z)/.5
    return g, d


def plot_and_measure(solutions):
    fine = solutions[(32,128)]
    M = int(fine["M"])
    selected = [0,8,16,23]
    colors = ["#2166ac", "#d95f02", "#1b9e77", "#984ea3"]
    t = np.linspace(-.5,.5,1001)
    g, dg, g0, centers = profiles(fine,t)
    h = 1/int(fine["N"])
    angles = np.rad2deg(np.arctan2(fine["directions"][:,1],fine["directions"][:,0]))
    stats = {"configurations":[], "derivative_law":[], "width_changes":[]}
    for (m,n), sol in solutions.items():
        stats["configurations"].append(dict(M=m,N=n,width=m*n,rank=int(sol["rank"]),
            rel_l2=dict(zip(KEYS,sol["rel_l2"].tolist())),
            grouped_max_difference=float(sol["grouped_max_difference"])))
    # Coefficients and analytic derivative evaluated at precisely the same centers.
    dcenters = np.empty_like(fine["w"])
    for j in range(M):
        z = float(fine["gamma"]) * (centers[j,:,None]-centers[j,None,:])
        dcenters[j] = float(fine["gamma"])*(1-np.tanh(z)**2) @ fine["w"][j]
    for k,key in enumerate(KEYS):
        mask = np.abs(centers)<.3
        a = (2*fine["w"][:,:,k]/h)[mask]
        d = dcenters[:,:,k][mask]
        stats["derivative_law"].append(dict(target=key,interior_abs_t=.3,
            correlation=float(np.corrcoef(a,d)[0,1]),relative_discrepancy=float(np.linalg.norm(a-d)/np.linalg.norm(d))))
    # Analytic derivative versus a symmetric difference, independently of coefficient law.
    eps = 1e-5
    gp = profiles(fine,np.array([-.2,0,.2])+eps)[0]
    gm = profiles(fine,np.array([-.2,0,.2])-eps)[0]
    d = profiles(fine,np.array([-.2,0,.2]))[1]
    stats["derivative_finite_difference_relative_error"] = float(np.linalg.norm((gp-gm)/(2*eps)-d)/np.linalg.norm(d))
    assert stats["derivative_finite_difference_relative_error"] < 1e-6
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig, ax = plt.subplots(2,4,figsize=(17,8),sharex=True,layout="constrained")
    for col,k in enumerate([1,2,3,4]):
        for j,color in zip(selected,colors):
            inside = np.abs(centers[j])<=.36
            ax[0,col].plot(t,M*dg[j,:,k],color=color,lw=1.6)
            ax[0,col].scatter(centers[j,inside],2*M*fine["w"][j,inside,k]/h,color=color,s=9,alpha=.7)
            ax[1,col].plot(t,M*(g[j,:,k]-g0[j,k]),color=color,label=f"{angles[j]:.1f}°")
        ax[0,col].set_title(f"{TARGETS[k][1]}\nrelative error {fine['rel_l2'][k]:.1e}")
        ax[1,col].set_xlabel("spoke coordinate t, relative to data center")
        for row in range(2):
            ax[row,col].set_xlim(-.36,.36)
            ax[row,col].axhline(0,color=".75",lw=.6)
            # Autoscale y to the displayed (scored) region only.
            view = np.abs(t)<=.36
            vals = (M*dg[selected,:,k] if row==0 else M*(g[selected,:,k]-g0[selected,k,None]))[:,view]
            if row == 0:
                dots = (2*M*fine["w"][:,:,k]/h)[selected]
                vals = np.concatenate([vals.ravel(), dots[np.abs(centers[selected])<=.36]])
            lo,hi = np.min(vals),np.max(vals)
            ax[row,col].set_ylim(lo-.08*(hi-lo),hi+.08*(hi-lo))
            ax[row,col].grid(alpha=.15)
    ax[0,0].set_ylabel("dots: 2 M a / h; lines: M g′(t)")
    ax[1,0].set_ylabel("spoke function M [g(t) − g(0)]")
    ax[1,3].legend(ncol=2,fontsize=9)
    fig.suptitle("What each spoke represents — 4096 tanh neurons = 32 directions × 128 centers\n"
                 "All profiles multiplied by 32; constants removed from bottom row. Only the scored interior is shown.",fontsize=14)
    fig.savefig(OUT/"spokes_4096.png",dpi=160)
    plt.close(fig)
    # Raw coefficients, including the collar, with matching-spoke width comparison.
    fig, ax = plt.subplots(2,5,figsize=(19,7.5),layout="constrained")
    coarse = solutions[(32,64)]
    gc,dc,gc0,cc = profiles(coarse,t)
    for k in range(5):
        for j,color in zip(selected,colors):
            ax[0,k].plot(centers[j],fine["w"][j,:,k],color=color,lw=1,label=f"{angles[j]:.1f}°")
            ax[1,k].plot(t,M*(g[j,:,k]-g0[j,k]),color=color,lw=1.5)
            ax[1,k].plot(t,M*(gc[j,:,k]-gc0[j,k]),color=color,lw=1,ls="--")
        ax[0,k].set_title(TARGETS[k][1])
        ax[0,k].axvspan(-.5,-.4,color=".9");ax[0,k].axvspan(.4,.5,color=".9")
        ax[0,k].set_xlim(-.5,.5);ax[1,k].set_xlim(-.36,.36)
        view=np.abs(t)<=.36
        vals=np.concatenate([M*(g[selected,:,k]-g0[selected,k,None])[:,view],M*(gc[selected,:,k]-gc0[selected,k,None])[:,view]])
        lo,hi=vals.min(),vals.max();ax[1,k].set_ylim(lo-.05*(hi-lo),hi+.05*(hi-lo))
        ax[1,k].set_xlabel("t")
        delta=(g[:,:,k]-g0[:,k,None])-(gc[:,:,k]-gc0[:,k,None])
        stats["width_changes"].append(dict(target=KEYS[k],relative_centered_profile_change=float(np.linalg.norm(delta[:,view])/np.linalg.norm((g[:,:,k]-g0[:,k,None])[:,view]))))
    ax[0,0].set_ylabel("raw readout coefficient a");ax[1,0].set_ylabel("M [g(t) − g(0)]")
    ax[0,4].legend(fontsize=8)
    fig.suptitle("Top: actual coefficients, including the unobserved collar (gray)\nBottom: same spokes at width 4096 (solid) and 2048 (dashed)",fontsize=14)
    fig.savefig(OUT/"raw_coefficients_and_widths.png",dpi=160);plt.close(fig)
    # Gaussian provides an analytic filtered-Radon comparison without FFT/window choices.
    fig, ax = plt.subplots(2,4,figsize=(16,7.5),layout="constrained")
    for col,j in enumerate(selected):
        can,can_d=gaussian_canonical(t,fine["directions"][j])
        can0=gaussian_canonical(np.array([0.]),fine["directions"][j])[0][0]
        slice_y=original.f_gauss_bump(original.X0+t[:,None]*fine["directions"][j])
        slice0=original.f_gauss_bump(original.X0[None,:])[0]
        ax[0,col].plot(t,M*(g[j,:,0]-g0[j,0]),label="fitted spoke × M")
        ax[0,col].plot(t,can-can0,ls="--",label="filtered Radon × π")
        ax[0,col].plot(t,slice_y-slice0,ls=":",label="spatial slice of f")
        ax[1,col].plot(t,M*dg[j,:,0],label="fitted spoke derivative × M")
        ax[1,col].plot(t,can_d,ls="--",label="filtered Radon derivative × π")
        inside = np.abs(centers[j])<=.36
        ax[1,col].scatter(centers[j,inside],2*M*fine["w"][j,inside,0]/h,s=10,label="scaled coefficients")
        ax[0,col].set_title(f"Gaussian bump, θ={angles[j]:.1f}°")
        for row in range(2):
            ax[row,col].set_xlim(-.36,.36);ax[row,col].grid(alpha=.15)
            view = np.abs(t)<=.36
            curves = [M*(g[j,:,0]-g0[j,0]),can-can0,slice_y-slice0] if row==0 else [M*dg[j,:,0],can_d]
            vals = np.concatenate([curve[view] for curve in curves])
            lo,hi=vals.min(),vals.max();ax[row,col].set_ylim(lo-.07*(hi-lo),hi+.07*(hi-lo))
        ax[1,col].set_xlabel("t")
    ax[0,0].set_ylabel("profile with value at t=0 removed")
    ax[1,0].set_ylabel("derivative")
    ax[0,3].legend(fontsize=8);ax[1,3].legend(fontsize=8)
    fig.suptitle("Gaussian: fitted spokes closely resemble filtered Radon profiles\nGaussian fit on a radius-0.4 ball; width 4096; relative error " + f"{fine['rel_l2'][0]:.1e}",fontsize=14)
    fig.savefig(OUT/"gaussian_radon_comparison.png",dpi=160);plt.close(fig)
    # Verify canonical angular reconstruction on independent points.
    directions=original.even_directions(2,512)
    xx=original.ball(100,.9*R,original.X0,np.random.default_rng(37))
    zz=(xx-original.A_BUMP)@directions.T/.5
    canonical_pred=np.mean(1-2*zz*dawsn(zz),axis=1)
    stats["canonical_gaussian_max_error"]=float(np.max(np.abs(canonical_pred-original.f_gauss_bump(xx))))
    assert stats["canonical_gaussian_max_error"]<1e-13
    can=np.stack([gaussian_canonical(t,v)[0] for v in fine["directions"]])
    can0=np.array([gaussian_canonical(np.array([0.]),v)[0][0] for v in fine["directions"]])
    view=np.abs(t)<=.36
    learned=M*(g[:,:,0]-g0[:,0,None])[:,view]
    reference=(can-can0[:,None])[:,view]
    stats["gaussian_canonical_centered_profile_relative_difference"]=float(np.linalg.norm(learned-reference)/np.linalg.norm(reference))
    (OUT/"summary.json").write_text(json.dumps(stats,indent=2)+"\n")
    print(json.dumps(stats,indent=2),flush=True)


def main():
    global OUT
    ap=argparse.ArgumentParser()
    ap.add_argument("--plot-only",action="store_true")
    ap.add_argument("--angle-rule",choices=["recorded","endpoint"],default="recorded",
                    help="Original recorded half-step fan, or the current fan with exact axes/diagonals.")
    args=ap.parse_args()
    if args.angle_rule == "endpoint":
        OUT = OUT / "endpoint_direction_check"
    OUT.mkdir(parents=True,exist_ok=True)
    configs=[(32,64),(32,128),(16,128)]
    if args.plot_only:
        solutions={c:dict(np.load(OUT/f"solution_M{c[0]}_N{c[1]}.npz")) for c in configs}
    else:
        solutions={c:fit(*c,recorded=args.angle_rule=="recorded") for c in configs}
    plot_and_measure(solutions)


if __name__=="__main__":
    main()
