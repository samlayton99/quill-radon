"""Hard-problem checks for honest conservative and native-QUILL backends."""
from __future__ import annotations
import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/codex-general-hard-mpl")
from pathlib import Path
import json
import time
import numpy as np
from scipy.optimize import brentq
from solver.hyperbolic import solve_burgers, reconstruct
from solver.reaction import solve_reaction_diffusion, classical_split_reference

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/checkpoint_F_applications/expF19_radon_direct_pde/general_solver"


def smooth_entropy(x, t):
    """Entropy solution for u0=-sin(pi*x), periodic [-1,1], after shock formation.

    For x>0 the entropy characteristic originates at y in [x,1], solving
    x=y-t sin(pi*y). Its symmetric negative-x branch has opposite velocity.
    The only entropy shock is stationary at x=0; the value exactly there is
    assigned zero, which does not affect L1 or the weak solution.
    """
    result = []
    for z in np.asarray(x):
        ax = abs(z)
        if ax < 1e-15 or abs(ax-1) < 1e-15:
            result.append(0.)
        else:
            y = brentq(lambda y:y-t*np.sin(np.pi*y)-ax, ax, 1., xtol=1e-14)
            result.append(-np.sign(z)*np.sin(np.pi*y))
    return np.array(result)


def hyperbolic_suite():
    cases = {
        "riemann_shock": (lambda x:np.where(x < 0, 1., 0.), .4, "outflow",
                          lambda x:np.where(x < .2, 1., 0.)),
        "riemann_rarefaction": (lambda x:np.where(x < 0, -1., 1.), .4, "outflow",
                                lambda x:np.clip(x/.4, -1., 1.)),
        "smooth_shock_formation": (lambda x:-np.sin(np.pi*x), .5, "periodic",
                                  lambda x:smooth_entropy(x, .5)),
    }
    rows, displays = [], {}
    for name, (ic, end, boundary, exact) in cases.items():
        x = np.linspace(-1., 1., 16384, endpoint=False)+1/16384
        truth = exact(x)
        display = {"x":x, "truth":truth, "states":{}}
        for n in (64, 128, 256, 512, 1024):
            result = solve_burgers(ic, cells=n, t_end=end, boundary=boundary,
                                   save_times=np.linspace(0., end, 21))
            pred = reconstruct(x, result["edges"], result["values"])
            linf = float(np.max(abs(pred-truth)))
            if name != "riemann_rarefaction":
                jump = .2 if name == "riemann_shock" else 0.
                probe = np.array([jump-1e-12, jump+1e-12])
                linf = max(linf, float(np.max(abs(reconstruct(probe,result["edges"],result["values"])-exact(probe)))))
            record = dict(case=name, n=n, t_end=end, l1=float(2*np.mean(abs(pred-truth))),
                          linf=linf, **result["metrics"], status=result["status"])
            assert abs(record["mass_balance_residual"]) < 5e-11
            assert record["entropy_balance_defect"] <= 1e-10
            rows.append(record)
            display["states"][n] = pred
        displays[name] = display
    return rows, displays


def reaction_suite():
    ic = lambda x:.3*np.sin(x)+.05*np.cos(2*x)
    ac_rows, ac_states = [], []
    for k in (16, 32, 64, 128, 256):
        result = solve_reaction_diffusion(ic, modes=k, cells=1024, t_end=.1,
                                         diffusivity=.1, reaction_rate=100., tail_limit=.05)
        ac_states.append(result)
        ac_rows.append(dict(k=k, status=result["status"], **result["metrics"]))
    # Independent references are built after all native Allen-Cahn solutions.
    coarse = classical_split_reference(ic, cells=8192, t_end=.1, max_step=5e-5)
    fine = classical_split_reference(ic, cells=16384, t_end=.1, max_step=2.5e-5)
    truth = fine["values"][::16]
    reference_difference = float(np.linalg.norm(coarse["values"]-fine["values"][::2])/np.linalg.norm(fine["values"][::2]))
    for row, result in zip(ac_rows, ac_states):
        row["relative_l2_vs_independent_reference"] = (
            float(np.linalg.norm(result["values"]-truth)/np.linalg.norm(truth))
            if result["status"] == "reached_t_end" else None)
    ac_refined_time = solve_reaction_diffusion(ic, modes=128, cells=1024, t_end=.1,
                                              diffusivity=.1, reaction_rate=100., rtol=1e-10,
                                              atol=1e-12, max_step=.005, tail_limit=.05)
    time_difference = float(np.linalg.norm(ac_refined_time["values"]-ac_states[3]["values"])/np.linalg.norm(ac_refined_time["values"]))
    homogeneous = []
    for rtol in (1e-6, 1e-8, 1e-10):
        result = solve_reaction_diffusion(lambda x:np.ones_like(x), modes=16, cells=128,
                                         t_end=1.1, diffusivity=.05, reaction="quadratic", reaction_rate=1.,
                                         amplitude_limit=1000., rtol=rtol, atol=rtol*.01, max_step=.02)
        row = dict(status=result["status"], exact_threshold_time=.999,
                   threshold_time_error=abs(result["time"]-.999), **result["metrics"])
        assert result["status"] == "amplitude_limit"
        assert np.all(np.isfinite(result["values"]))
        homogeneous.append(row)
    blowup_rows, blowup_states = [], []
    for k in (16, 32, 64, 128):
        for cap in (100., 1000., 10000.):
            result = solve_reaction_diffusion(lambda x:1+.2*np.cos(x), modes=k, cells=1024,
                                             t_end=1.05, diffusivity=.05, reaction="quadratic", reaction_rate=1.,
                                             amplitude_limit=cap, tail_limit=.05, max_step=.01,
                                             rtol=1e-9, atol=1e-11)
            assert result["status"] in ("underresolved", "amplitude_limit")
            assert np.all(np.isfinite(result["values"]))
            blowup_rows.append(dict(k=k, status=result["status"],
                                    local_reaction_pole_proxy=result["time"]+1/result["metrics"]["quadrature_max_abs"],
                                    **result["metrics"]))
            blowup_states.append(result)
    refined_blowup = solve_reaction_diffusion(lambda x:1+.2*np.cos(x), modes=128, cells=1024,
                                             t_end=1.05, diffusivity=.05, reaction="quadratic", reaction_rate=1.,
                                             amplitude_limit=10000., tail_limit=.05, max_step=.005,
                                             rtol=1e-11, atol=1e-13)
    return dict(allen_cahn=ac_rows, allen_cahn_reference_refinement=reference_difference,
                allen_cahn_time_refinement=time_difference, homogeneous=homogeneous,
                blowup=blowup_rows, nonconstant_true_blowup_time_bounds=[1/1.2, 1.],
                nonconstant_time_refinement=dict(status=refined_blowup["status"],
                    refined_threshold_time=refined_blowup["time"],
                    threshold_time_difference=abs(refined_blowup["time"]-blowup_states[-1]["time"]),
                    rtol=1e-11,atol=1e-13,max_step=.005)), ac_states, truth, blowup_states


def plots(hyper, displays, reaction, ac_states, ac_truth, blowup_states):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullLocator
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    titles = {"riemann_shock":"Moving entropy shock", "riemann_rarefaction":"Entropy rarefaction",
              "smooth_shock_formation":"Shock from smooth initial data"}
    for ax, (name, data) in zip(axes, displays.items()):
        ax.plot(data["x"], data["truth"], "k--", lw=1.5, label="Exact entropy solution")
        for n in (64, 256, 1024): ax.plot(data["x"], data["states"][n], label=f"N={n}")
        ax.set(title=titles[name], xlabel="x", ylabel="u(x,T)")
        ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.13), ncol=2, fontsize=8)
        ax.grid(alpha=.2)
    fig.subplots_adjust(top=.7, bottom=.14, wspace=.28)
    fig.savefig(OUT/"hard_hyperbolic_profiles.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for name in displays:
        rows = [r for r in hyper if r["case"] == name]
        ns = [r["n"] for r in rows]
        axes[0].loglog(ns, [r["l1"] for r in rows], "o-", label=titles[name])
        axes[1].semilogx(ns, [r["linf"] for r in rows], "o-", label=titles[name])
        axes[2].plot(ns, [r["entropy_balance_defect"] for r in rows], "o-", label=titles[name])
    for ax in axes:
        ax.set_xticks([64,128,256,512,1024], ["64","128","256","512","1024"])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlabel("Cells")
        ax.grid(alpha=.2)
    axes[0].set_ylabel("L1 error (domain integral)")
    axes[1].set_ylabel("Maximum pointwise error")
    axes[2].set_ylabel("Entropy change minus boundary flux")
    axes[2].set_xscale("log", base=2)
    axes[2].set_xticks([64,128,256,512,1024], ["64","128","256","512","1024"])
    axes[2].axhline(0, color=".5", ls=":")
    axes[0].legend(loc="lower center", bbox_to_anchor=(1.7, 1.05), ncol=3, fontsize=9)
    fig.subplots_adjust(top=.8, bottom=.17, wspace=.28)
    fig.savefig(OUT/"hard_hyperbolic_refinement.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot(ac_states[-1]["x"], ac_truth, "k--", label="Independent splitting reference")
    for idx in (1,2,3):
        result=ac_states[idx]
        axes[0].plot(result["x"],result["values"],label=f"Actual QUILL K={result['metrics']['modes']}")
    axes[0].set(xlabel="x",ylabel="Allen–Cahn u(x,0.1)")
    axes[0].legend(loc="lower center",bbox_to_anchor=(.5,1.02),fontsize=8,ncol=2)
    valid=[r for r in reaction["allen_cahn"] if r["status"]=="reached_t_end"]
    axes[1].loglog([r["k"] for r in valid],[r["relative_l2_vs_independent_reference"] for r in valid],"o-",label="Actual QUILL vs independent reference")
    axes[1].axhline(reaction["allen_cahn_reference_refinement"],ls=":",color=".4",label="Reference refinement difference")
    axes[1].set(xlabel="Maximum retained frequency K",ylabel="Relative L2 difference")
    axes[1].set_xticks([32,64,128,256],["32","64","128","256"])
    axes[1].xaxis.set_minor_locator(NullLocator())
    axes[1].legend(loc="lower center",bbox_to_anchor=(.5,1.02),fontsize=8)
    for idx in (2,5,8,11):
        result=blowup_states[idx]
        hist=result["history"]
        line, = axes[2].plot([v["t"] for v in hist],[1/v["max_abs"] for v in hist],label=f"K={result['metrics']['modes']}; {result['status']}")
        axes[2].plot(hist[-1]["t"],1/hist[-1]["max_abs"],
                     "o" if result["status"] == "amplitude_limit" else "x",color=line.get_color(),ms=6)
    axes[2].set(xlabel="Time (near the growth singularity)",ylabel="1 / max |u|",xlim=(.832,.840),ylim=(0,.008))
    axes[2].legend(loc="lower center",bbox_to_anchor=(.5,1.02),fontsize=7,ncol=1)
    for ax in axes: ax.grid(alpha=.2)
    fig.subplots_adjust(top=.73,bottom=.16,wspace=.3)
    fig.savefig(OUT/"hard_reaction_and_blowup.png",dpi=180)
    plt.close(fig)


def invariant_suite():
    rows = []
    cases = [("allen_cahn", 32, lambda x:.3*np.sin(x)+.05*np.cos(2*x), .1, .1, 100.),
             ("quadratic", 64, lambda x:1+.2*np.cos(x), 1.05, .05, 1.)]
    for reaction, k, initial, end, nu, rate in cases:
        result = solve_reaction_diffusion(initial, reaction=reaction, reaction_rate=rate,
                    diffusivity=nu, modes=k, t_end=end, cells=1024, amplitude_limit=10000,
                    invariant_tolerance=.001, rtol=1e-9, atol=1e-11)
        assert result["status"] == "invariant_violation"
        rows.append(dict(status=result["status"], **result["metrics"]))
    (OUT/"hard_invariant_guards.json").write_text(json.dumps(rows, indent=2)+"\n")
    return rows


def plot_invariant_guards(guards, reaction):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    unguarded_ac = next(r for r in reaction["allen_cahn"] if r["k"] == 32)
    unguarded_blowup = next(r for r in reaction["blowup"] if r["k"] == 64 and r["amplitude_limit"] == 10000)
    values = [(unguarded_ac["maximum_principle_excess"], guards[0]["maximum_principle_excess"]),
              (max(0.,-unguarded_blowup["minimum_value"]), max(0.,-guards[1]["minimum_value"]))]
    for ax, violation, title in zip(axes, values, ("Allen–Cahn K=32", "Positive-data growth K=64")):
        ax.bar([0,1], violation, color=["#bb5566", "#4477aa"])
        ax.set(yscale="log", xticks=[0,1], xticklabels=["Tail guard only", "+ invariant guard"],
               ylabel="Output-grid bound violation", title=title, ylim=(1e-4, 1.))
        ax.axhline(.001, color=".4", linestyle=":")
        ax.grid(axis="y", alpha=.2)
    fig.subplots_adjust(left=.1, bottom=.18, top=.85, wspace=.4)
    fig.savefig(OUT/"hard_guard_comparison.png", dpi=180)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    began=time.perf_counter()
    hyper, displays=hyperbolic_suite()
    print("Conservative shock/rarefaction refinements complete",flush=True)
    reaction, ac_states, truth, blowup_states=reaction_suite()
    guards=invariant_suite()
    output=dict(hyperbolic=hyper,reaction=reaction,invariant_guards=guards,threads=1,elapsed_seconds=time.perf_counter()-began,
                note="FV weak conservative backend is conventional; stiff backend is actual QUILL with BDF implicit state solves. Threshold stops are not exact singularity times.")
    (OUT/"hard_metrics.json").write_text(json.dumps(output,indent=2)+"\n")
    plots(hyper,displays,reaction,ac_states,truth,blowup_states)
    plot_invariant_guards(guards,reaction)
    np.savez_compressed(OUT/"hard_reaction_fields.npz",x=ac_states[-1]["x"],reference=truth,
                        **{f"quill_K{r['metrics']['modes']}":r["values"] for r in ac_states})
    print(json.dumps(dict(allen_cahn=reaction["allen_cahn"],homogeneous=reaction["homogeneous"],
                         blowup=[{k:r[k] for k in ("k","amplitude_limit","status","final_time","max_abs","local_reaction_pole_proxy")} for r in reaction["blowup"]]),indent=2))


if __name__=="__main__":main()
