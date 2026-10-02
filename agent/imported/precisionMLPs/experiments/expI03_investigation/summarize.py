"""Regenerate compact figures/tables from the completed investigation records."""
from pathlib import Path
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/precision-mpl-cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

OUT = Path(__file__).resolve().parents[2] / "results/checkpoint_I_depth_theory/investigation_20260906"
ARMS = ["raw", "poly3", "quadratic", "quadratic_free", "mlp"]
LABELS = {"raw": "I02\nmixer\n539 params", "poly3": "Cubic\nsubspace\n191 params",
          "quadratic": "Quadratic\nprior\n179 params", "quadratic_free": "Quadratic\nreleased\n539 params", "mlp": "Dense\nMLP\n568 params"}
COLORS = dict(zip(ARMS, ["#607d8b", "#b37928", "#157a6e", "#7257a2", "#202b36"]))


def read(name):
    return json.loads((OUT / f"{name}.json").read_text())


def summarize(runs):
    result = {}
    for metric in ("train", "test", "full_test"):
        values = [r["final"][metric] for r in runs]
        result[metric] = dict(median=float(np.median(values)), min=min(values), max=max(values), values=values)
    result.update(seeds=[r["seed"] for r in runs], params=runs[0]["final"]["actual_params"],
                  seconds=[r["time"] for r in runs])
    return result


def main():
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    rows = read("confirm_d3")
    release = read("quadratic_release") if (OUT / "quadratic_release.json").exists() else {}
    rows.update(release)
    summary = {}
    targets = ["fast_waves", "composition", "product_peak", "random_ridges"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.4))
    table = ["# Analytic confirmation: final errors", "",
             "Three new seeds (3, 4, 5); medians with seed minima and maxima. All errors are relative L2. "
             "Inner test samples use 90% of the training-ball radius; full-ball samples include its outer shell. "
             "No held-out errors select a checkpoint. See the main report for protocol and compute limits.", "",
             "| Target | Arm | Parameters | Inner test median [min, max] | Full ball median [min, max] |", "|---|---|---:|---:|---:|"]
    for ax, target in zip(axes.flat, targets):
        arms = [a for a in ARMS if any(r["arm"] == a and r["target"] == target for r in rows.values())]
        summary[target] = {}
        for i, arm in enumerate(arms):
            runs = sorted([r for r in rows.values() if r["target"] == target and r["arm"] == arm], key=lambda r: r["seed"])
            s = summarize(runs)
            summary[target][arm] = s
            for offset, metric, marker in [(-.10, "test", "o"), (.10, "full_test", "^")]:
                values = s[metric]["values"]
                jitter = np.linspace(-.035, .035, len(values))
                ax.scatter(i + offset + jitter, values, color=COLORS[arm], s=27,
                           marker=marker, alpha=.8, zorder=3)
                ax.plot([i + offset - .07, i + offset + .07], [s[metric]["median"]]*2,
                        color=COLORS[arm], lw=2.3)
            a, b = s["test"], s["full_test"]
            table.append(f"| {target} | {arm} | {s['params']} | {a['median']:.6g} [{a['min']:.6g}, {a['max']:.6g}] | {b['median']:.6g} [{b['min']:.6g}, {b['max']:.6g}] |")
        ax.set_yscale("log")
        ax.set_xticks(range(len(arms)), [LABELS[a] for a in arms], fontsize=9)
        ax.set_title(target.replace("_", " ").capitalize(), loc="left", fontweight="bold")
        ax.set_ylabel("Independent test relative L2")
        ax.set_ylim({"fast_waves": (1e-4, 1), "composition": (5e-4, .02),
                     "product_peak": (.005, .06), "random_ridges": (.01, 1)}[target])
        ax.grid(axis="y", alpha=.2)
        ax.set_xlim(-.5, len(arms)-.5)
    fig.legend(handles=[Line2D([], [], color="#444444", marker="o", ls="", label="Inner test: each seed"),
                        Line2D([], [], color="#444444", marker="^", ls="", label="Full ball: each seed"),
                        Line2D([], [], color="#444444", lw=2, label="Seed median")],
               loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, .95), h_pad=2.5)
    fig.savefig(OUT / "analytic_comparison.png", dpi=180)
    plt.close(fig)
    (OUT / "analytic_summary.json").write_text(json.dumps(summary, indent=2))
    (OUT / "analytic_tables.md").write_text("\n".join(table) + "\n")

    ablation = read("quadratic_init_ablation")
    resolution = read("quadratic_resolution")
    variants = [(ablation, "poly2", "Random quadratic\nN=32"),
                (rows, "quadratic", "Positive quadratic\nN=32"),
                (resolution, "poly2", "Random quadratic\nN=64"),
                (resolution, "quadratic", "Positive quadratic\nN=64")]
    if len(release) == 9:
        variants.append((release, "quadratic_free", "Positive init, released\nN=32"))
    fig, ax = plt.subplots(figsize=(9.3, 4.6))
    for seed, color in zip((3, 4, 5), ("#517ca4", "#b75d37", "#50785c")):
        vals = [source[f"fast_waves|{seed}|{arm}"]["final"]["test"] for source, arm, _ in variants]
        ax.semilogy(range(len(vals)), vals, "o-", color=color, label=f"Seed {seed}", lw=1.2, ms=5)
    ax.set_xticks(range(len(variants)), [v[2] for v in variants], fontsize=9)
    ax.set(ylabel="Fast waves: independent inner-test relative L2",
           ylim=(1e-4, 1))
    ax.grid(axis="y", alpha=.2)
    fig.suptitle("The quadratic initialization matters more than doubling bank resolution", y=.995)
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(.5, .95))
    fig.tight_layout(rect=(0, 0, 1, .85))
    fig.savefig(OUT / "quadratic_controls.png", dpi=180)
    plt.close(fig)

    refined = read("confirm_d3_refine")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    refinement_summary = {}
    for key, run in refined.items():
        arm = key.split("|")[-1]
        curve = run["curve"]
        for ax, metric, title in zip(axes, ("test", "full_test"), ("Inner test", "Full training ball")):
            ax.semilogy([r["calls"] for r in curve], [r[metric] for r in curve],
                        color=COLORS[arm], label=arm)
            ax.set(xlabel="Additional objective/gradient evaluations", ylabel="Independent test relative L2", title=title)
            ax.set_ylim(1e-5, 1)
            ax.grid(alpha=.2)
        refinement_summary[arm] = {"initial": curve[0], "final": curve[-1],
                                   **{k: v for k, v in run.items() if k not in ("curve", "final")}}
    fig.suptitle("Fast waves, seed 3: L-BFGS with decreasing readout ridge", y=.995)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(.5, .95))
    fig.tight_layout(rect=(0, 0, 1, .83))
    fig.savefig(OUT / "refinement_comparison.png", dpi=180)
    plt.close(fig)
    (OUT / "refinement_summary.json").write_text(json.dumps(refinement_summary, indent=2))


if __name__ == "__main__":
    main()
