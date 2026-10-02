"""PDE dimensional scaling: supplied ridge structure vs nonlinear mode growth.

Run with one BLAS thread. Results are independent of the prior known-function
encoding benchmark. Forward methods receive ICs and PDEs, never interior labels.
"""
from __future__ import annotations
import os
for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ.setdefault(key, "1")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/codex-general-dimensions-mpl")
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
from solver.highdim import (solve_ridge_linear, PeriodicReactionDiffusion,
                           evaluate_fourier, encode_fourier_ridges,
                           interaction_support_count)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/"results/checkpoint_F_applications/expF19_radon_direct_pde/general_solver"


def rel(a, b):
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(b), 1e-300))


def structured():
    rows = []
    modes = np.array([1, 3, 7])
    amp = np.array([.7, .2, .08])
    phase = np.array([.3, -.2, .6])
    t, nu = .15, .03
    def profile(s):
        return np.cos(np.asarray(s)[..., None]*(modes*np.pi)+phase)@amp
    # Two IC Fourier truncations verify recovery from initial quadrature rather
    # than supplying exact coefficient amplitudes to the forward method.
    for d in [2, 3, 5, 10, 20, 50, 100]:
        rng = np.random.default_rng(173+d)
        v = rng.normal(size=(3, d)); v /= np.linalg.norm(v, axis=1)[:, None]
        drift = rng.normal(size=d); drift *= .2/np.linalg.norm(drift)
        x = rng.normal(size=(512, d)); x /= np.linalg.norm(x, axis=1)[:, None]
        x *= np.linspace(.02, 1, len(x))[:, None]
        for k in ([5, 9] if d in [2, 20] else [9]):
            net = solve_ridge_linear([profile]*3, v, t, nu, drift,
                                     max_frequency=k, n_interior=193, lam=.2)
            before = time.perf_counter(); pred = net.evaluate(x)
            elapsed = time.perf_counter()-before
            # Independent exact verifier created only after forward propagation.
            truth = np.zeros(len(x)); gradient = np.zeros_like(x)
            lap = np.zeros(len(x)); initial = np.zeros(len(x))
            for direction in v:
                s = x@direction
                omega = modes*np.pi
                angle = (s[:, None]-(drift@direction)*t)*omega+phase
                damp = amp*np.exp(-nu*omega**2*t)
                truth += np.cos(angle)@damp
                gradient += (-np.sin(angle)@(omega*damp))[:, None]*direction
                lap -= np.cos(angle)@(omega**2*damp)
                initial += profile(s)
            physical_dt = nu*pred['laplacian']-pred['gradient']@drift
            rows.append(dict(**net.metadata, neurons=net.neurons,
                             storage_mb=net.storage_bytes/1e6,
                             evaluate_seconds=elapsed, evaluation_points=len(x),
                             initial_direction_span_rank=int(np.linalg.matrix_rank(v)),
                             evaluation_domain="unit Euclidean ball in R^d; fixed maximum radius, not full cube",
                             sampling="512 independent angular directions and uniformly spaced radii; not volume uniform",
                             value_error=rel(pred['value'], truth),
                             gradient_error=rel(pred['gradient'], gradient),
                             laplacian_error=rel(pred['laplacian'], lap),
                             relative_pde_residual=rel(pred['time_derivative'], physical_dt),
                             solution_change=rel(truth, initial)))
            print("LINEAR", json.dumps(rows[-1]), flush=True)
    payload = dict(rows=rows, PDE="u_t+b.grad u=nu*Delta u on R^d",
                   initial_data="three supplied unit ridges; each cos(pi*s), cos(3pi*s), cos(7pi*s)",
                   interpretation="Ambient dimension is cheap only because a rank<=3 initial decomposition is supplied.")
    (OUT/"dimension_linear.json").write_text(json.dumps(payload, indent=2))
    return payload


def initial_nonlinear(coordinates):
    d = len(coordinates)
    return .3+(.1/math.sqrt(d))*sum(np.cos(x) for x in coordinates)


def nonlinear():
    rows = []; seed = 828
    for d in [2, 3, 4, 5]:
        rng = np.random.default_rng(seed+d)
        x = rng.uniform(-np.pi, np.pi, (128, d))
        states = []
        # Highest K is a spatial refinement reference, not an exact solution.
        for k in [1, 2, 3, 4]:
            model = PeriodicReactionDiffusion(d, k, .05)
            state, metadata = model.solve(initial_nonlinear, t_end=.2, dt=.01)
            modes, coeff = model.sparse_coefficients(state)
            evaluated = evaluate_fourier(x, modes, coeff, derivatives=True)
            rhs_modes, rhs_coeff = model.sparse_coefficients(model.rhs(state))
            dt_value = evaluate_fourier(x, rhs_modes, rhs_coeff)['value']
            physical_rhs = .05*evaluated['laplacian']+evaluated['value']**2
            initial_modes, initial_coeff = model.sparse_coefficients(model.initial(initial_nonlinear), 1e-12)
            active, _ = model.sparse_coefficients(state, 1e-10)
            rows.append(dict(**metadata, initial_active_modes=len(initial_modes),
                             final_active_modes_1e10=len(active),
                             generated_modes_1e10=int(sum(np.count_nonzero(kk)>1 for kk in active)),
                             offgrid_pde_residual_relative=rel(dt_value, physical_rhs),
                             offgrid_pde_residual_rms=float(np.sqrt(np.mean((dt_value-physical_rhs)**2))),
                             minimum_sample_value=float(evaluated['value'].min()),
                             maximum_sample_value=float(evaluated['value'].max())))
            states.append((model, state, evaluated, len(rows)-1))
            print("NONLINEAR", json.dumps(rows[-1]), flush=True)
        reference = states[-1][2]
        for model, state, evaluated, index in states[:-1]:
            rows[index].update(value_error_vs_K4=rel(evaluated['value'], reference['value']),
                               gradient_error_vs_K4=rel(evaluated['gradient'], reference['gradient']),
                               laplacian_error_vs_K4=rel(evaluated['laplacian'], reference['laplacian']))
        model, state, evaluated, index = states[-1]
        refined, timing = model.solve(initial_nonlinear, t_end=.2, dt=.005)
        modes, coeff = model.sparse_coefficients(refined)
        timestep_ref = evaluate_fourier(x, modes, coeff)
        rows[index].update(time_halving_difference=rel(evaluated['value'], timestep_ref['value']),
                           time_halving_seconds=timing['seconds'])
        if d == 5:
            spatial = PeriodicReactionDiffusion(d, 5, .05)
            higher, higher_stats = spatial.solve(initial_nonlinear, t_end=.2, dt=.005)
            km, cm = spatial.sparse_coefficients(higher)
            higher_value = evaluate_fourier(x, km, cm)['value']
            rows[index].update(spatial_K4_to_K5_at_half_dt=rel(timestep_ref['value'], higher_value),
                               K5_validation_stats=higher_stats)
        # Neural compile is timed separately and does not enter PDE evolution.
        model, state, evaluated, index = states[-2]
        modes, coeff = model.sparse_coefficients(state)
        net = encode_fourier_ridges(modes, coeff, n_interior=257, lam=.2,
                                    coefficient_threshold=1e-10)
        started = time.perf_counter(); pred = net.evaluate(x); evtime = time.perf_counter()-started
        rows[index].update(neural_encoding={k:v for k,v in net.metadata.items() if k!='projection_limits'}, neurons=net.neurons,
                           neural_storage_mb=net.storage_bytes/1e6,
                           neural_evaluation_seconds=evtime,
                           neural_value_difference=rel(pred['value'], evaluated['value']),
                           neural_gradient_difference=rel(pred['gradient'], evaluated['gradient']),
                           neural_laplacian_difference=rel(pred['laplacian'], evaluated['laplacian']))
        print("ENCODING", d, json.dumps(rows[index]), flush=True)
        # Save each dimension before the next potentially expensive run.
        (OUT/"dimension_nonlinear.json").write_text(json.dumps(dict(rows=rows,
            PDE="u_t=0.05*Delta u+u^2 on periodic [-pi,pi]^d", t_end=.2,
            initial="0.3+(0.1/sqrt(d))*sum_j cos(x_j)",
            distinction="Conventional nonlinear PDE evolution; subsequent explicit neural compilation. No native neural dynamics claim.",
            reference="K=4 is spatial refinement reference, not exact truth"), indent=2))
    return rows


def counts():
    rows = []
    for d in [2, 3, 4, 5, 10, 20, 50, 100]:
        for k in [1, 2, 3, 4, 6, 10]:
            q = 2*math.ceil((3*k+1)/2)
            rows.append(dict(dimension=d, cutoff=k,
                             retained_box_modes=(2*k+1)**d,
                             quadrature_points=q**d,
                             estimated_workspace_gb=14*16*q**d/1e9,
                             l1_support_modes=interaction_support_count(d, k)))
    (OUT/"dimension_support_counts.json").write_text(json.dumps(dict(rows=rows,
        support_meaning="l1 support of products of <=degree coordinate cosines; possible modes, not all significant",
        workspace_meaning="14 complex128 grid-sized arrays, conservative implementation estimate"), indent=2))
    return rows


def frontier():
    """Measured six-dimensional cap and stronger nonlinear-evolution controls."""
    rows=[]; limit_mb=800
    for d,tend,cutoffs in [(6,.2,[1,2,3]),(3,.8,[3,4,5]),(4,.8,[3,4,5]),(5,.8,[3,4,5])]:
        x=np.random.default_rng(919+d).uniform(-np.pi,np.pi,(96,d))
        stored=[]
        for k in cutoffs:
            model=PeriodicReactionDiffusion(d,k,.05,memory_limit_mb=limit_mb)
            state,meta=model.solve(initial_nonlinear,t_end=tend,dt=.01)
            km,cm=model.sparse_coefficients(state)
            fields=evaluate_fourier(x,km,cm,True)
            kt,ct=model.sparse_coefficients(model.rhs(state))
            dtvalue=evaluate_fourier(x,kt,ct)['value']
            rhs=.05*fields['laplacian']+fields['value']**2
            active,_=model.sparse_coefficients(state,1e-10)
            row=dict(**meta,active_modes_1e10=len(active),
                     offgrid_pde_residual_relative=rel(dtvalue,rhs),
                     mean_sample_value=float(np.mean(fields['value'])),
                     max_sample_value=float(np.max(fields['value'])))
            rows.append(row);stored.append((fields,model,state,row))
            print('FRONTIER',json.dumps(row),flush=True)
        ref=stored[-1][0]
        for fields,model,state,row in stored[:-1]:
            row['value_difference_next_reference']=rel(fields['value'],ref['value'])
            row['laplacian_difference_next_reference']=rel(fields['laplacian'],ref['laplacian'])
            row['reference_cutoff']=cutoffs[-1]
        if d==6:
            try:PeriodicReactionDiffusion(6,4,.05,memory_limit_mb=limit_mb)
            except MemoryError:
                stored[-1][3]['next_refinement_rejected']=True
                stored[-1][3]['next_workspace_estimate_mb']=14*16*14**6/1e6
        else:
            # Time refinement at K4 separates time and spatial discretization.
            fields,model,state,row=stored[-2]
            finer,meta=model.solve(initial_nonlinear,t_end=tend,dt=.005)
            km,cm=model.sparse_coefficients(finer)
            fineval=evaluate_fourier(x,km,cm)['value']
            row['time_halving_difference']=rel(fields['value'],fineval)
            row['time_halving_seconds']=meta['seconds']
        (OUT/'dimension_frontier.json').write_text(json.dumps(dict(rows=rows,
            workspace_cap_mb=limit_mb,
            cap_meaning='deliberate per-experiment budget, not physical laptop capacity',
            method='classical Fourier Galerkin dimensional-limit control, not native QUILL dynamics',
            reference='largest permitted cutoff is a refinement reference, not exact truth'),indent=2))
    return rows


def validate():
    # Constant nonlinear IC has exact Riccati solution independent of dimension.
    for d in [2, 3, 5]:
        model = PeriodicReactionDiffusion(d, 1)
        state, _ = model.solve(lambda x: .3, t_end=.2, dt=.005)
        k, c = model.sparse_coefficients(state)
        y = evaluate_fourier(np.zeros((3, d)), k, c)['value']
        assert max(abs(y-.3/(1-.3*.2))) < 2e-13
    for d in [2, 3, 5]:
        for k in [1, 2, 3, 4]:
            model = PeriodicReactionDiffusion(d, k)
            assert int(model.mask.sum()) == (2*k+1)**d
    # Combinatorial support agrees with brute enumeration in small dimensions.
    import itertools
    for d in [1, 2, 3]:
        for degree in [1, 2, 3]:
            brute = sum(sum(abs(i) for i in k) <= degree
                        for k in itertools.product(range(-degree, degree+1), repeat=d))
            assert brute == interaction_support_count(d, degree)
    # Encoder orientation, grouping, constant and Fourier normalization check.
    k = np.array([[0,0],[1,2],[-1,-2],[2,4],[-2,-4]])
    c = np.array([.3, .2+.1j, .2-.1j, -.02j, .02j])
    net = encode_fourier_ridges(k,c,257,.2,0.)
    x = np.random.default_rng(23).uniform(-np.pi,np.pi,(100,2))
    truth = evaluate_fourier(x,k,c,True); pred=net.evaluate(x)
    assert rel(pred['value'],truth['value']) < 1e-10
    assert rel(pred['gradient'],truth['gradient']) < 1e-9
    return dict(constant_reaction_exact=True, support_combinatorics=True,
                grouped_encoder_value_error=rel(pred['value'],truth['value']),
                grouped_encoder_gradient_error=rel(pred['gradient'],truth['gradient']))


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    a=json.loads((OUT/"dimension_linear.json").read_text())['rows']
    b=json.loads((OUT/"dimension_nonlinear.json").read_text())['rows']
    fig,ax=plt.subplots(1,3,figsize=(15,4.6),constrained_layout=True)
    a=[r for r in a if r['max_frequency']==9]
    for key,label in [('value_error','Value'),('gradient_error','Gradient'),('laplacian_error','Laplacian')]:
        ax[0].loglog([r['ambient_dimension'] for r in a],[r[key] for r in a],'-o',label=label)
    ax[0].set(xlabel='Ambient dimension',ylabel='Relative error',title='Direct linear PDE: 663 neurons; supplied ridges',ylim=(1e-16,1e-12))
    ax[0].legend()
    for d in [2,3,4,5]:
        group=[r for r in b if r['dimension']==d and r['cutoff']<4]
        ax[1].semilogy([r['cutoff'] for r in group],[r['value_error_vs_K4'] for r in group],'-o',label=f'd={d}')
    ax[1].set(xlabel='Fourier cutoff per coordinate K',ylabel='Difference from K=4',title='Interacting nonlinear PDE: spatial refinement')
    ax[1].set_xticks([1,2,3])
    ax[1].legend()
    for k in [2,3,4]:
        group=[r for r in b if r['cutoff']==k]
        ax[2].semilogy([r['dimension'] for r in group],[r['seconds'] for r in group],'-o',label=f'K={k}')
    ax[2].set(xlabel='Spatial dimension',ylabel='Evolution seconds',title='Nonlinear dense-grid cost still grows rapidly')
    ax[2].set_xticks([2,3,4,5])
    ax[2].legend()
    for axis in ax:axis.grid(alpha=.2)
    fig.savefig(OUT/'dimension_pde_results.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(11,4.6),constrained_layout=True)
    d=np.array([2,3,4,5,10,20,50,100])
    for degree in [2,4,6,10]:
        ax[0].loglog(d,[interaction_support_count(int(x),degree) for x in d],'-o',label=f'Product degree {degree}')
    ax[0].set(xlabel='Dimension',ylabel='Possible nonzero Fourier modes',title='Multiplication fills new directions')
    ax[0].legend()
    group=[r for r in b if r['cutoff']==3]
    ax[1].semilogy([r['dimension'] for r in group],[r['initial_active_modes'] for r in group],'-o',label='Initial')
    ax[1].semilogy([r['dimension'] for r in group],[r['final_active_modes_1e10'] for r in group],'-o',label='T=.2, K=3; |coefficient| > 1e-10')
    frontier_path=OUT/'dimension_frontier.json'
    if frontier_path.exists():
        frontier_rows=json.loads(frontier_path.read_text())['rows']
        strong=[r for r in frontier_rows if r['t_end']==.8 and r['cutoff']==3]
        ax[1].semilogy([r['dimension'] for r in strong],[r['active_modes_1e10'] for r in strong],'-o',label='T=.8, K=3; |coefficient| > 1e-10')
    ax[1].set(xlabel='Dimension',ylabel='Active modes',title='Measured growth in u_t = 0.05 Delta u + u^2')
    ax[1].legend()
    for axis in ax:axis.grid(alpha=.2)
    fig.savefig(OUT/'dimension_mode_growth.png',dpi=170);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['all','linear','nonlinear','frontier','plot','validate'],default='all')
    args=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if args.mode in ['all','validate']:
        result=validate();(OUT/'dimension_validation.json').write_text(json.dumps(result,indent=2));print(result,flush=True)
    if args.mode in ['all','linear']:structured()
    if args.mode in ['all','nonlinear']:nonlinear();counts()
    if args.mode in ['all','frontier']:frontier()
    if args.mode in ['all','plot']:plot()
