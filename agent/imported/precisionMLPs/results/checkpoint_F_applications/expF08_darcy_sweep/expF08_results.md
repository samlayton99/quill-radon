# expF08 -- Darcy precision sweep (tanh collocation on the FNO benchmark)

**Status:** drafted. Single-seed sweep, 975 solves, ~2 h CPU.

## TL;DR

The expF01 tanh collocation core, applied to the standard FNO **darcy_421**
benchmark, reaches the **fp64 floor (3.0e-14 rel L2)** on a smooth manufactured
Darcy control -- so the machine-precision claim holds for the Darcy *operator*.
On the **16 rough benchmark instances** it does **not** reach that floor: the
best per-instance error is **2.8e-3 median rel L2** (vs the dataset reference),
which *beats a trained FNO (~1e-2)* but sits ~11 orders above the smooth-control
floor. The bottleneck is **coefficient roughness**, not offset resolution or
width -- Gaussian pre-smoothing of the coefficient (sigma 0 -> 4 px) monotonically
drives the error from 7.2e-2 down to 2.8e-3. This independently reproduces
expF05's spline-ridge finding ("the rough-coefficient bottleneck is not
offset-resolution-limited") with a different basis.

## Question

Does the frozen-geometry + one-lstsq recipe that hits the fp64 floor on smooth
manufactured PDEs (expF01) transfer to a *real operator-learning benchmark* --
Darcy flow `-div(a grad u) = 1`, `u = 0` on the boundary -- where the
coefficient `a` is a rough, gridded field the recipe never saw at design time?
Two sub-questions: (1) is the *method* exact on the Darcy operator when the
coefficient is smooth (control)? (2) how far does raw error fall on the rough
benchmark instances, and what limits it?

## Experiment design

- **Operator.** Non-divergence form on `[-1,1]^2`: `-a lap u - grad a . grad u = f`,
  `u = 0` on the boundary. The benchmark's `-div_s(a grad_s u) = 1` on `[0,1]^2`
  maps under `p = 2s-1` to forcing `f = 1/4`. Solver terms
  `L = -a*lap - grad a . grad` (see `darcy_problems.py`).
- **Smooth control.** `a = 3 + exp(sin pi x sin pi y)`,
  `u* = sin pi x sin pi y + 0.5 sin 2pi x sin pi y`, forcing manufactured so
  `L[u*] = f`. Every hand-coded derivative and the forcing identity are
  FD-verified (`verify_control`, tolerance 2e-4). Error is vs the *exact* `u*`.
- **Rough instances.** First 16 instances of `darcy_test_421` (421x421,
  cell-centered grid). The gridded `a` is fit by a cubic `RectBivariateSpline`
  surrogate (exposing `a, a_x, a_y` analytically), optionally Gaussian
  pre-smoothed by `sigma in {0,1,2,4}` px before fitting. Error is rel L2 vs the
  dataset reference `u` (itself numerical, ~1e-4-1e-6 accurate -- a floor on
  this metric); the raw **PDE residual** `|L u_hat - f|` at 20k fresh interior
  points is the precision-side metric.
- **Sweep.** Widths `W in {256,576,1024,1600,2304}`, `lambda in {0.2,0.25,0.3}`;
  control x (W,lambda) and 16 instances x 4 sigma x (W,lambda) = 975 solves,
  one min-norm `lstsq` (rcond 1e-13) each, no training. Single seed (42).
- **Reproduce.** `python experiments/expF08_darcy_sweep/run_darcy.py`
  (`--smoke` for a 2-instance subset; `--plot` to regenerate figures;
  `--darcy-path PATH` to point at the npz). Core vendored from expF01 as
  `core.py`.

## Results

**Smooth control -- spectral convergence to the fp64 floor** (best-of-lambda):

| W    | rel L2 (vs exact) | Linf     | max PDE residual |
|------|-------------------|----------|------------------|
| 256  | 1.6e-6            | 8.3e-6   | 4.2e-3           |
| 576  | 1.4e-10           | 2.3e-9   | 2.6e-5           |
| 1024 | 4.9e-13           | 5.0e-12  | 4.2e-7           |
| 1600 | 1.1e-13           | 1.6e-12  | 1.7e-7           |
| 2304 | **3.0e-14**       | 7.1e-13  | 2.3e-7           |

The Darcy operator with a smooth coefficient is solved to machine precision,
exactly like expF01's 2D steady problems. **The machine-precision claim lives
here.**

**Rough instances -- coefficient roughness is the wall.** Median over the 16
instances of each instance's best rel L2 (min over W, lambda), per pre-smoothing
level:

| sigma (px) | median rel L2 (vs reference) | median PDE res (med / max) |
|-----------:|------------------------------|----------------------------|
| 0 (raw)    | 7.2e-2                       | 3.8e-2 / 12.7              |
| 1          | 3.0e-2                       | 2.4e-2 / 12.6              |
| 2          | 1.3e-2                       | 1.5e-2 / 9.2               |
| 4          | **2.8e-3**                   | 6.7e-3 / 3.2               |

Per-instance best (sigma=4): median 2.8e-3, min 1.9e-3, max 3.5e-3 across the 16
instances -- tight. Error falls monotonically with pre-smoothing and is roughly
width-saturated by W=2304; the **PDE residual never approaches the floor** (max
stays O(1)), so the rough coefficient -- not conditioning at the offset scale --
is what caps the error. `error_vs_width.png` shows the control curve plunging to
3e-14 while every rough-`a` curve plateaus above 1e-3; `error_heatmaps.png`
shows the error on instance 0 concentrating where `a` varies fastest and
shrinking uniformly as sigma grows.

## Conclusions

1. **The recipe transfers to the Darcy operator exactly** -- 3.0e-14 on the
   smooth control, no training, one lstsq. Whatever limits the benchmark
   instances is the *coefficient*, not the method.
2. **On rough real instances it is accurate but not machine-precise**:
   2.8e-3 median rel L2, **better than a trained FNO (~1e-2)** at zero training
   cost, but ~11 orders above the operator floor. Two compounding causes: the
   spline surrogate of a rough `a` has large/oscillatory gradients feeding the
   `grad a . grad u` term, and the dataset reference is itself only ~1e-4-1e-6
   accurate, flooring the rel-L2 metric.
3. **Pre-smoothing is the dominant knob** (sigma 0->4 buys 1.4 orders); W and
   lambda are second-order once the coefficient is resolved. More neurons do not
   help past ~W=1024 on rough instances.
4. **Cross-experiment agreement.** expF05 (B-spline ridges, residual-guided knot
   adaptivity) hit the same ~7.5e-2 rough-Darcy stall at sigma=0 and concluded
   the bottleneck "is not offset-resolution-limited." expF08 reaches the same
   conclusion from the tanh side and adds the sigma-continuation that quantifies
   the roughness cost. **The FOSLS/first-order-system escalation is therefore
   about coefficient roughness handling, not basis resolution** -- and is not
   needed to beat the FNO baseline.

## Open questions

- **FOSLS / mixed form.** Recast as `{q - a grad u = 0, div q = -f}` so the
  rough coefficient enters algebraically (multiplying `q`) instead of through
  `grad a`. Pre-registered as the escalation if interface-concentrated error is
  the obstacle; expF08 shows the error *is* interface-concentrated, so this is
  the natural next experiment.
- **Reference-accuracy floor.** Quantify how much of the 2.8e-3 is the dataset
  reference vs the solver by comparing against a high-order FD/FEM solve of the
  *same* spline-surrogate coefficient (removes the surrogate + reference
  mismatch, isolating the solver).
- **Seeds.** Single seed here; the control floor and the sigma trend are stable
  by construction, but the per-instance spread wants error bars.
- **No-smoothing precision.** Whether an interface-aware basis (or domain
  decomposition across the coefficient's level sets) can recover precision on
  the *raw* rough coefficient without the sigma=4 pre-smooth.
