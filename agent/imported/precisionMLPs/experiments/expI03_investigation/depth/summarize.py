"""Refresh depth probe tables and diagnostic figures from saved JSON."""
import json
from pathlib import Path
import numpy as np
import run


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    screen = json.loads((run.OUT / "screen.json").read_text())
    lines = ["| target | arm | median test error | seed range | n |",
             "|---|---|---:|---:|---:|"]
    for target in ["fast_waves", "composition"]:
        for arm in run.ARMS:
            vals = [r["final"] for r in screen.values() if r["target"] == target and r["arm"] == arm]
            if vals:
                lines.append(f"| {target} | {arm} | {np.median(vals):.3g} | {min(vals):.3g}–{max(vals):.3g} | {len(vals)} |")
    (run.OUT / "tables.md").write_text("\n".join(lines)+"\n")
    run.plot(screen, run.OUT / "screen_trajectories.png")
    for name in ["a2_matched", "sample_count_control"]:
        path = run.OUT / f"{name}.json"
        if path.exists():
            run.plot(json.loads(path.read_text()), run.OUT / f"{name}_trajectories.png")

    # Compare the exact common starting function to later mesh escape.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for arm in ["depth2", "random3", "identity3", "identity3_small"]:
        row = screen[f"fast_waves|0|{arm}"]
        curve = row["trajectory"]
        steps = [p["step"] for p in curve]
        axes[0].plot(steps, [p["test"] for p in curve], label=arm)
        axes[1].plot(steps, [max(abs(v) for q in p["pres"][1:] for v in q["mean"]) for p in curve], label=arm)
    axes[0].set(yscale="log", ylabel="Relative test L2", xlabel="Adam step")
    axes[1].axhline(1, ls=":", color="black", label="Mesh boundary")
    axes[1].set(ylabel="Largest absolute channel mean", xlabel="Adam step")
    for ax in axes:
        ax.grid(alpha=.2)
    run.title_and_shared_legend(fig, axes,
        "Same initial function does not prevent later mesh escape (fast waves, seed 0)")
    fig.savefig(run.OUT / "escape_diagnostic.png", dpi=160)
    plt.close(fig)

    for name in ["continuation", "continuation_preserved"]:
        p = run.OUT / f"{name}.json"
        if not p.exists():
            continue
        rows = json.loads(p.read_text())
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
        for key, row in rows.items():
            curve = row["trajectory"]
            steps = [p["step"] for p in curve]
            axes[0].plot(steps, [p["test"] for p in curve], label=key)
            axes[1].plot(steps, [max(abs(v) for q in p["pres"][1:] for v in q["mean"]) for p in curve], label=key)
        axes[0].set(yscale="log", ylabel="Relative test L2", xlabel="Additional Adam steps")
        axes[1].set(ylabel="Largest absolute channel mean", xlabel="Additional Adam steps")
        for ax in axes:
            ax.grid(alpha=.2)
        subtitle = " (initial objective preserved)" if name.endswith("preserved") else ""
        run.title_and_shared_legend(fig, axes,
            "Adding exact identity layers to the reproduced successful depth-two fit" + subtitle)
        fig.savefig(run.OUT / f"{name}_trajectories.png", dpi=160)
        plt.close(fig)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
