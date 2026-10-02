"""Tables for the expI02 writeup, straight from the study JSON files. Usage: run.py is not needed; python tables.py."""
import json
import math
from pathlib import Path

RES = Path(__file__).resolve().parents[2] / "results" / "checkpoint_I_depth_theory" / "expI02_block_prior"


def load(name):
    p = RES / f"{name}.json"
    return json.load(open(p)) if p.exists() else {}


def gmean(v):
    v = [x for x in v if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return math.exp(sum(math.log(max(x, 1e-300)) for x in v) / len(v)) if v else float("nan")


def a1_table():
    A1 = load("a1")
    targets = ["gauss_bump", "fast_waves", "composition", "three_bumps"]
    rungs = sorted({k.split("|")[1] for k in A1 if k.count("|") == 2}, key=lambda r: (0 if "+" not in r else 1, r))
    print("| rung | " + " | ".join(t.replace("_", " ") for t in targets) + " | geo. mean | +GN geo. mean | seeds (geo. mean per seed) |")
    print("|---|" + "---|" * (len(targets) + 3))
    for r in rungs:
        row = [A1.get(f"{t}|{r}|0") for t in targets]
        if not all(row):
            continue
        fin = [x["final"] for x in row]; gn = [x.get("final_gn", float("nan")) for x in row]
        seeds = sorted({int(k.split("|")[2]) for k in A1 if k.count("|") == 2 and k.split("|")[1] == r})
        per_seed = [gmean([A1[f"{t}|{r}|{s}"]["final"] for t in targets if f"{t}|{r}|{s}" in A1]) for s in seeds]
        print(f"| {r} | " + " | ".join(f"{v:.1e}" for v in fin) + f" | {gmean(fin):.1e} | {gmean(gn):.1e} | " + ", ".join(f"{v:.1e}" for v in per_seed) + " |")
    for t in targets:
        o = A1.get(f"{t}|oracle")
        print(f"oracle floor {t}: {o['err']:.1e} (channel {o['chan']:.1e})" if o else f"oracle floor {t}: none")


def b1_table():
    B1 = load("b1")
    names = [n for n in ["gauss_bump", "fast_waves", "composition", "product_peak", "three_bumps", "random_ridges"] if any(k.startswith(n + "|") for k in B1)]
    print("| target | block (median, min..max) | +GN (seed 0) | shallow | MLP same params | MLP same FLOPs | block params / units |")
    print("|---|---|---|---|---|---|---|")
    import statistics
    for n in names:
        rows = [B1[k] for k in B1 if k.startswith(n + "|")]
        def cell(arm):
            v = [r[arm]["final"] for r in rows]
            return f"{statistics.median(v):.1e} ({min(v):.0e}..{max(v):.0e})"
        gn = [r["block"].get("final_gn") for r in rows if "final_gn" in r["block"]]
        c = rows[0]["block"]["counts"]
        print(f"| {n} | {cell('block')} | {gn[0] if gn else float('nan'):.1e} | {cell('shallow')} | {cell('mlp_params')} | {cell('mlp_flops')} | {c['params']} / {c['units']} |")


def b2_table():
    B2 = load("b2")
    print("| dataset | n, d | ridge | block | MLP same params | block rcond |")
    print("|---|---|---|---|---|---|")
    import statistics
    for n in ["friedman1", "lorenz_map", "kin8nm", "concrete"]:
        rows = [B2[k] for k in B2 if k.startswith(n + "|")]
        if not rows:
            continue
        f = lambda key: f"{statistics.median([r[key]['rmse'] if isinstance(r[key], dict) else r[key] for r in rows]):.3f}"
        rc = sorted({f"{r['block']['rcond']:.0e}" for r in rows})
        print(f"| {n} | {rows[0]['n']}, {rows[0]['d']} | {f('ridge')} | {f('block')} | {f('mlp_params')} | {', '.join(rc)} |")


def b3_table():
    B3 = load("b3")
    print("| model | params | test accuracy (trained head) | test accuracy (LS refit head) |")
    print("|---|---|---|---|")
    for k in ["logistic|0", "small|block|0", "small|mlp_tanh|0", "small|mlp_relu|0", "large|block|0", "large|mlp_tanh|0", "large|mlp_relu|0"]:
        r = B3.get(k)
        if r:
            print(f"| {k.rsplit('|', 1)[0]} | {r['counts']['params']} | {100 * (1 - r['final_err']):.1f} | {100 * (1 - r['final_err_solved']):.1f} |")


def b4_table():
    B4 = load("b4")
    print("| PDE | block: Adam | block: operator solve | MLP: Adam | MLP: operator solve | block params / MLP width |")
    print("|---|---|---|---|---|---|")
    for n in ["poisson1d", "poisson2d", "burgers"]:
        r = B4.get(f"{n}|0")
        if r:
            b, m = r["block"], r["mlp_params"]
            print(f"| {n} | {b['final']:.1e} | {b.get('final_solved', float('nan')):.1e} | {m['final']:.1e} | {m.get('final_solved', float('nan')):.1e} | {b['counts']['params']} / {m['w']} |")


def a2_table():
    A2 = load("a2")
    for name in ["fast_waves", "composition", "product_peak"]:
        rows = [(k, v) for k, v in A2.items() if k.startswith(name + "|") and isinstance(v, dict) and "final" in v]
        if not rows:
            continue
        print(f"\n{name}: " + "; ".join(f"{k.split('|', 1)[1]}={v['final']:.1e}@{v['counts']['params']}p" for k, v in rows if "counts" in v))
        o = A2.get(f"{name}|oracle")
        if o:
            print(f"  oracle floor {o['err']:.1e}")


if __name__ == "__main__":
    for fn in [a1_table, b1_table, b2_table, b3_table, b4_table, a2_table]:
        print(f"\n### {fn.__name__}")
        try:
            fn()
        except Exception as e:
            print("  (not available:", e, ")")
