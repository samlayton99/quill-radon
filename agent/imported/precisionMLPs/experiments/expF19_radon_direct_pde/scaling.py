"""Direct Gaussian Radon encoding across dimensions; no training or LS.

CPU: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python .../scaling.py
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import resource
import time

os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/codex-f19-scaling-mpl')
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import hyp1f1, ndtri
from scipy.stats import qmc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/checkpoint_F_applications/expF19_radon_direct_pde/scaling'


def directions(d, m, seed=0, structured=True):
    if d == 2 and structured:
        t = np.pi*np.arange(m)/m
        return np.column_stack([np.cos(t), np.sin(t)]), np.full(m, 1/m)
    if d == 3 and structured:
        nz = 2*max(2, int(np.sqrt(m)/2))
        z, w = leggauss(nz)
        phi = np.pi*np.arange(2*nz)/nz
        z, phi = np.meshgrid(z, phi, indexing='ij')
        r = np.sqrt(1-z*z)
        v = np.stack([r*np.cos(phi), r*np.sin(phi), z], -1).reshape(-1, 3)
        weights = np.broadcast_to(w[:, None]/(4*nz), z.shape).ravel()
        keep = v[:, 2]>0
        return v[keep], 2*weights[keep]
    u = qmc.Sobol(d, scramble=True, seed=seed).random_base2(int(np.log2(m)))
    v = ndtri(np.clip(u, 1e-14, 1-1e-14))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v, np.full(len(v), 1/len(v))


def points(d, n=128, seed=915):
    # Uniform radius, not uniform volume: avoid concentration hiding interior errors.
    rng = np.random.default_rng(seed+d)
    x = rng.normal(size=(n, d))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    x *= np.linspace(0, 1, n)[:, None]
    return x


def targets(d):
    rng = np.random.default_rng(819+d)
    rot, _ = np.linalg.qr(rng.normal(size=(d, d)))
    eig = np.geomspace(1.5, 6., d)
    return {'isotropic': 3*np.eye(d), 'anisotropic': (rot*eig)@rot.T}


def profile(t, v, b):
    d = v.shape[1]
    s = np.einsum('mi,ij,mj->m', v, np.linalg.inv(b), v)[:, None]
    pref = np.exp(-.5*np.linalg.slogdet(b)[1]-.5*d*np.log(s))
    return pref*hyp1f1(.5*d, .5, -t*t/s)


def encode(v, w, centers, gamma, b):
    h = centers[1]-centers[0]
    shift = np.pi/(2*gamma)
    coef = .5*h*w[:, None]*profile(centers[None, :]+1j*shift, v, b).imag/shift
    zero = w@profile(np.zeros((len(v), 1)), v, b)[:, 0]
    bias = zero-np.sum(coef*np.tanh(-gamma*centers)[None, :])
    return coef, bias


def evaluate(x, v, centers, gamma, coef, bias):
    ans = np.broadcast_to(bias, (len(x), len(bias))).copy()
    for k in range(0, len(v), 32):
        p = x@v[k:k+32].T
        a = np.tanh(gamma*(p[:, :, None]-centers[None, None, :]))
        ans += np.einsum('pmc,mct->pt', a, coef[k:k+32], optimize=True)
    return ans


def relative(y, truth):
    return float(np.linalg.norm(y-truth)/np.linalg.norm(truth))


def cell(d, m, n, seed=0, x=None, structured=True, vw=None, rule=None):
    began = time.perf_counter()
    v, w = directions(d, m, seed, structured) if vw is None else vw
    x = points(d) if x is None else x
    cases = targets(d)
    centers = np.linspace(-4, 4, n)
    gamma = .25/(centers[1]-centers[0])
    coef, bias, truth, continuous = [], [], [], []
    for b in cases.values():
        # Chunk complex hypergeometric values; no dense sample-by-neuron matrix.
        c=np.empty((len(v),len(centers))); a=0.
        for k in range(0,len(v),512):
            cc,aa=encode(v[k:k+512],w[k:k+512],centers,gamma,b)
            c[k:k+512]=cc; a+=aa
        coef.append(c); bias.append(a)
        truth.append(np.exp(-np.einsum('ni,ij,nj->n', x, b, x)))
        y = np.zeros(len(x))
        for k in range(0, len(v), 512):
            y += w[k:k+512]@profile((x@v[k:k+512].T).T, v[k:k+512], b)
        continuous.append(y)
    built = time.perf_counter()
    coef = np.stack(coef, -1)
    predicted = evaluate(x, v, centers, gamma, coef, np.asarray(bias))
    end = time.perf_counter()
    rows = []
    for k, name in enumerate(cases):
        rows.append(dict(d=d, directions=len(v), centers=n, neurons=len(v)*n,
            target=name, scramble=seed, rule=rule or ('structured' if d<4 and structured else 'Sobol-normal sphere'),
            tanh_rel_l2=relative(predicted[:, k], truth[k]),
            angular_rel_l2=relative(continuous[k], truth[k]),
            conversion_rel_l2=relative(predicted[:, k], continuous[k]),
            max_abs=float(np.max(np.abs(predicted[:, k]-truth[k]))),
            build_and_profile_seconds=built-began, eval_seconds=end-built,
            total_seconds=end-began, evaluated_points=len(x),
            compressed_one_target_MiB=8*(len(v)*d+len(v)*n+n+2)/2**20,
            explicit_one_target_MiB=8*len(v)*n*(d+2)/2**20,
            process_peak_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20))
    return rows, predicted


def dimension_run(pilot=False):
    rows = []
    dims = [2, 3, 4, 5, 8, 16, 32]
    for d in dims:
        ms = [64, 256] if pilot else ([16, 64, 256, 1024] if d<4 else [64, 256, 1024, 4096, 16384])
        for seed in ([0] if d<4 else [0, 1]):
            for m in ms:
                rr, _ = cell(d, m, 201, seed)
                rows.extend(rr)
                print(json.dumps(rr), flush=True)
                (OUT/('pilot.json' if pilot else 'dimension.json')).write_text(json.dumps(rows, indent=2))
    return rows


def allocation_run():
    rows = []
    predictions = {}
    for d in [2, 3, 4]:
        for m in [16, 64, 256, 1024]:
            for n in [41, 81, 121, 201]:
                rr, pred = cell(d, m, n, 0)
                rows.extend(rr)
                predictions[d, m, n] = pred
                print(json.dumps(rr), flush=True)
                (OUT/'allocation.json').write_text(json.dumps(rows, indent=2))
    # Geometry selection observes refinement differences only, never exact target values.
    selection = []
    for d in [2, 3, 4]:
        for budget in [20000, 60000, 220000]:
            candidates = []
            for m0, m in [(16,64), (64,256), (256,1024)]:
                for n0, n in [(41,81),(81,121),(121,201)]:
                    actual_m = len(directions(d,m)[0])
                    if actual_m*n>budget:
                        continue
                    p = predictions[d,m,n]
                    # Absolute prediction changes: both angular and center refinement.
                    score = float(np.linalg.norm(p-predictions[d,m0,n])+np.linalg.norm(p-predictions[d,m,n0]))
                    candidates.append((score,m,n))
            _, m, n = min(candidates)
            rr = [r for r in rows if r['d']==d and r['directions']==len(directions(d,m)[0]) and r['centers']==n]
            selection.append(dict(d=d,budget=budget,m=m,n=n,rows=rr))
    (OUT/'allocation_selection.json').write_text(json.dumps(selection, indent=2))
    return rows


def sparse_run():
    rows=[]
    for d in [3, 16, 64, 256]:
        rng=np.random.default_rng(909+d)
        v=rng.normal(size=(4,d)); v/=np.linalg.norm(v,axis=1,keepdims=True)
        amp=np.array([1.,.7,-.5,.3]); omega=np.array([2.,3.,5.,7.]); phase=np.array([.1,-.3,.4,.2])
        centers=np.linspace(-4,4,201); h=centers[1]-centers[0]; gamma=.25/h; a=np.pi/(2*gamma)
        # Exact inverse of the normalized sech^2 convolution on each Fourier frequency.
        density=amp[:,None]*np.sinh(a*omega[:,None])/a*np.cos(omega[:,None]*centers+phase[:,None])
        coef=.5*h*density
        bias=np.sum(amp*np.sin(phase))-np.sum(coef*np.tanh(-gamma*centers))
        x=points(d,n=1024)
        began=time.perf_counter(); y=evaluate(x,v,centers,gamma,coef[:,:,None],np.array([bias]))[:,0]
        truth=np.sum(amp[None,:]*np.sin((x@v.T)*omega+phase),axis=1)
        rows.append(dict(d=d,directions=4,centers=201,neurons=804,rel_l2=relative(y,truth),seconds=time.perf_counter()-began,max_abs=float(np.max(abs(y-truth))),directions_known=True))
    (OUT/'sparse.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows),flush=True)
    return rows


def figures():
    rows=json.loads((OUT/'dimension.json').read_text())
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(1,3,figsize=(15,4.6),constrained_layout=True)
    colors=plt.cm.viridis(np.linspace(.05,.9,7))
    for d,color in zip([2,3,4,5,8,16,32],colors):
        for j,target in enumerate(['isotropic','anisotropic']):
            r=[r for r in rows if r['d']==d and r['target']==target]
            ms=sorted(set(t['directions'] for t in r))
            med=[np.median([t['tanh_rel_l2'] for t in r if t['directions']==m]) for m in ms]
            lo=[min(t['tanh_rel_l2'] for t in r if t['directions']==m) for m in ms]
            hi=[max(t['tanh_rel_l2'] for t in r if t['directions']==m) for m in ms]
            ax[j].loglog(ms,med,'o-',label=f'd={d}',c=color);ax[j].fill_between(ms,lo,hi,color=color,alpha=.15)
        r=[r for r in rows if r['d']==d and r['target']=='anisotropic' and r['scramble']==0]
        ax[2].loglog([t['neurons'] for t in r],[t['total_seconds'] for t in r],'o-',c=color,label=f'd={d}')
    for a in ax:a.grid(alpha=.2)
    ax[0].set(xlabel='Directions M',ylabel='Relative L2 error',title='Isotropic Gaussian; actual tanh network')
    ax[1].set(xlabel='Directions M',ylabel='Relative L2 error',title='Rotated anisotropic Gaussian')
    ax[2].set(xlabel='Neurons M × 201',ylabel='Seconds / cell (two targets)',title='Construction + 128-point evaluation')
    ax[0].legend(ncol=2,fontsize=9)
    fig.suptitle('No training, no least squares; unit-radius domains. d≤3 structured cubature; d≥4 two Sobol scrambles.',fontsize=12)
    fig.savefig(OUT/'dimension_scaling.png',dpi=170);plt.close(fig)
    rows=json.loads((OUT/'allocation.json').read_text())
    fig,ax=plt.subplots(1,3,figsize=(15,4.5),constrained_layout=True)
    for a,d in zip(ax,[2,3,4]):
        for n in [41,81,121,201]:
            r=[r for r in rows if r['d']==d and r['target']=='anisotropic' and r['centers']==n]
            a.loglog([t['neurons'] for t in r],[t['tanh_rel_l2'] for t in r],'o-',label=f'N={n}')
        a.set(title=f'{d} input dimensions',xlabel='Neurons M × N',ylabel='Relative L2 error');a.grid(alpha=.2);a.legend(fontsize=9)
    fig.suptitle('Direct construction allocation: coarse centers eventually impose their own floor',fontsize=13)
    fig.savefig(OUT/'allocation.png',dpi=170);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['pilot','dimension','allocation','sparse','figures','all'],default='all');args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.mode in ['pilot','dimension','all']:dimension_run(args.mode=='pilot')
    if args.mode in ['allocation','all']:allocation_run()
    if args.mode in ['sparse','all']:sparse_run()
    if args.mode in ['figures','all']:figures()
