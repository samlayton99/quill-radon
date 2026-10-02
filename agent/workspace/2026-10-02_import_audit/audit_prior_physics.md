# Pre-Radon physics program (checkpoint F, expF01-F18 + cdeng/): audit

Agent-side document (preliminary). Read-only audit subagent, Oct 2, 2026. "Checked" = re-read from saved json. Paths relative to `agent/imported/precisionMLPs/`.

## Summary
- **Method.** Every pre-Radon PDE result: frozen tanh ridge dictionary (2-D: sqrt(W) directions x sqrt(W) QI offsets; d=3: expH06 mesh) + often a degree 3-5 monomial block + ONE dense global least-squares solve over all collocation/BC/IC/data rows (truncated SVD/QR, rcond 1e-13 then 1e-15), damped Gauss-Newton for nonlinear PDEs. Direct ancestor of the Radon construction, minus an analytic readout.
- **Positives.** All 18 smooth manufactured zoo problems (1-D, 2-D, linear and nonlinear) reach ~1e-13 to 1e-16 (nonlinear needs expF03 cascade init + rcond 1e-15). No-oracle width selector picked the oracle width on 3/3 held-out problems. BWLer suite: wave 1.6e-13 (80x better than BWLer); convection/reaction trail BWLer; Burgers nu=0.01/pi fails (0.29). Stokes 7.7e-14. 2-D+t NS 1.07e-11. Trained PINNs sit at 1e-2 to 1e-6; their refit certificate shows the trained readout, not the features, is the bottleneck.
- **The memory wall** ("couldn't get more than 3 inputs"): d=2 one field W=9216 (27.6k x 9.2k, ~2 GB, 3 min); d=3 (expF18) 12.3k unknowns x ~46k rows, 7 GB peak after augmented-QR rewrite (naive swapped at 12 GB on 16 GB M4), accuracy still falling -> memory stopped it; d=4 (expF12) only via tensor-product top-K basis, capped ~sqrt(eps) 1.5e-6. `expF01_linear_de_zoo/PINN_FEASIBILITY.md`: O((1/h)^d) features, cubic dense solve.
- **Other negatives.** Collocation minimizer displaced from best fit (reaction, NS) by up to 2 orders. Shocks and rough coefficients fail for every method. Tuned spectral beats QI on smooth problems; tuned ELM ties or loses by 2-100x. At matched columns tensor-QI ties/beats Radon in 2-D but is a product architecture.
- **Never attempted in checkpoint F:** inverse problems, noisy data, d >= 4 with one hidden layer.
- **expF19 imports none of this and reruns none of these benchmarks**: its numbers are not comparable to any pre-Radon number.

## 1. What the program established
**Positives.** expF01: all 9 linear problems at floor (1-D 2.7e-15 to 8.3e-15; steady 9.2e-15 to 1.2e-13; space-time 1.1e-13 to 3.1e-13). expF02/F03: all 9 nonlinear with cascade init + rcond 1e-15 (Monge-Ampere 1.0e-15, eikonal 6.6e-15, viscous Burgers 6.5e-15, KdV 2.8e-13, inviscid Burgers 5.8e-12). expF03 no-oracle protocol: frozen per-category recipe, width ladder, nested-width self-consistency selector -- the anti-overfitting template Sam asks for. Diagnostics: a small PDE residual certifies nothing (ill-posed dispersive placement matched well-posed residual to two digits while true errors differed by 1e4); tuning signals must include condition rows (u=0 trap); nested-width difference tracks true error. expF13 BWLer suite (table below). Stokes 7.7e-14 (expF09); 2-D+t drifting Taylor-Green NS d=3 (expF18) velocity 1.07e-11, pressure 8.6e-10 -- strongest pre-Radon physics result; 3-D+t Beltrami d=4 tensor architecture 1.5e-6 (expF12). PINN refit certificate: darcy_man trained 4e-4 vs refit 8.6e-9 (checked).

**Walls.** The dense global lstsq wall above. Collocation objective displaced from best fit: reaction GN from oracle fit walks 1.6e-6 -> 9.4e-5; NS at 2048 oracle coefficients have higher stacked residual than the solve, GN lands 100x worse in velocity (mechanism: spiky fit error at corners/fronts amplified by gamma^r under the operator -- directly relevant to any PINN-style Radon method). Finite optimal width / gamma^r drift for high-order linear 1-D (order 3: 2.7e-15 at W=173, ~1e-11 at W=1845); nonlinear drift was a solver artifact. Burgers nu=0.01/pi flat 0.03-0.3 for every method; rough Darcy 7.5e-2 raw.

## 2. Recommended benchmark / baseline set for paper 2
| # | Benchmark | d | Source | Oracle | Pre-Radon number | Why |
|---|---|---|---|---|---|---|
| 1 | BWLer five: convection c40/c80, reaction, wave, Burgers nu=0.01/pi | 2 | `experiments/expF13_bwler_suite/problems.py` + `ref/` | closed forms ~1e-50; Burgers Cole-Hopf 75-node GH 2.7e-15 | BWLer Table 2; expF13 W=9216; expF17 matched-column table | external citable precision-PINN suite; Burgers the honest shock negative |
| 2 | poisson_man (square minus 4 holes) | 2 | same | exact | expF13 1.1e-6; expF17 Radon 1.4e-7, tensor 1.7e-9 | non-tensor domain where ridges should win. Drop poisson_cg (COMSOL ref only 2.5e-5) |
| 3 | darcy_man (variable-coefficient elliptic) | 2 | `experiments/expF08_darcy_sweep/darcy_problems.py` | exact | 3.0e-14; expF17 Radon 7.2e-15; PINN 4e-4, refit 8.6e-9 | dramatic PINN-vs-solve gap |
| 4 | linear + nonlinear zoo (18) + 3 held-out | 1-2 | expF01/F02 `problems.py`, expF03 `heldout_problems.py` | manufactured, FD-verified | expF03 W* tables | breadth for plug-and-play; frozen held-out set exists |
| 5 | 2-D+t NS drifting Taylor-Green | 3 | `experiments/expF18_ns_spacetime/problem.py` | exact, non-gradient advection | 1.07e-11 at 12.3k unknowns, 7 GB, 20 min | the d=3 memory-wall reference |
| 6 | Stokes manufactured | 2 | `experiments/expF09_navier_stokes/stokes.py` | exact | 7.7e-14 | coupled system / pressure gauge |
| 7 | 3-D+t Beltrami | 4 | `experiments/expF12_tensor_ns3d/beltrami.py` | exact | tensor+GN 1.5e-6; PINNs 2e-2 | d=4 point. Caveat: advection is a pure gradient (weak nonlinearity) |
| 8 | rough Darcy darcy_421 | 2 | needs cluster data | unmeasured | 7.5e-2 | optional negative; blocked on data |
| 9 | dysts chaotic ODEs | 1 | `experiments/expF14_dysts_chaos/` | mpmath 30 dps | QI ~1e-13 | optional; 1-D input, not a Radon test |

**Baselines to carry:** trained PINN (expF03 `baselines_pinn.py` 3x128 tanh Adam+L-BFGS; expF17 `f17/pinn.py` 1-layer matched width + **refit certificate**); ELM/random features tuned by residual (closest one-hidden-layer competitor); tensor Chebyshev/Fourier spectral LS (expF17 `spectral`); Kansa RBF (one row); FD (low priority); published BWLer Table 2; the pre-Radon global-lstsq QI solve itself (its memory footprint is what paper 2 must beat).

**Protocol pieces worth reusing:** expF03 part 2 (no-oracle selector + held-out set solved once); expF17 SPEC sections 13-18 (matched columns, method-owned sampling, seeds perturb only the dictionary, geometric mean and GSD, equal declared tuning effort); oracle-vs-dynamic two-regime presentation; oracle footprint rule (perimeter rows must pin halo columns); rcond 1e-15, never form Phi^T Phi, block scaling, condition weights, cascade init.

## 3. Per-experiment entries (tags for paper 2)
- **expF01 linear DE zoo -- CORE.** 9 problems; Radon tensor ridges + degree-3 poly; one min-norm lstsq. All 9 at floor (checked). Dispersive placement bug; gamma^r drift; ELM spot check (19-685x, later revised to 2-100x). Draft pending Sam. `PINN_FEASIBILITY.md` states the d >= 3 wall.
- **expF02 nonlinear DE zoo -- CORE.** Logistic, Bratu, Blasius, eikonal, Monge-Ampere, third-order, inviscid/viscous Burgers, KdV; damped GN. Most rescued in expF03.
- **expF03 ablations, no-oracle selection, baselines -- CORE.** Cascade init dominates; selector matches oracle width 12/18 dev and 3/3 held-out; spectral wins 7/9 linear; ELM ties 1-D, loses 2-100x 2-D; PINN 1.6e-6 to 8e-4; nonlinear spectral+Newton 9/9 machine eps, ELM+Newton 1e-10 to 1e-15, PINN fails Monge-Ampere (0.37). `common.py` loads F01/F02 problems via importlib. Skip `cache/` (87M).
- **expF04 -- IRRELEVANT** (real-data MLP init).
- **expF05 B-spline ridges -- HISTORY** (spline floor ~2e-4 vs tanh 3e-14; rough Darcy stall). Data on cluster only.
- **expF06 steady 2-D Burgers -- HISTORY.** "Nonlinear floor ~1e-7" contradicted by expF03/expF18 (artifact of rcond 1e-13 + cold start).
- **expF07 lstsq finisher on trained PINN -- SUPPORT (weak).** Deep PINN + correction violates one hidden layer.
- **expF08 Darcy -- SUPPORT.** Smooth control 3.0e-14; rough median 7.2e-2. Hard-coded `/scr/cdeng` path.
- **expF09 Stokes -- SUPPORT.** 7.7e-14.
- **expF10/F11 neural operators, FNO init -- IRRELEVANT.**
- **expF12 3-D unsteady NS tensor basis -- SUPPORT (negative wall).** Product architecture; GN in top-K product-SVD basis; 1.5e-6 (supervised ceiling 5.6e-9); "Kronecker-pinv trap": product-basis precision pays conditioning to the 4th power. Cited json absent here.
- **expF13 BWLer suite -- CORE.**

  | problem | expF13 | BWLer |
  |---|---|---|
  | wave | 1.55e-13 | 1.3e-11 |
  | convection c40 | 6.2e-12 | 2e-13 |
  | convection c80 | 1.03e-9 | 1.1e-12 |
  | reaction (W=6400) | 4.56e-7 | 6.9e-11 |
  | Burgers | 0.287 | 4.6e-3 |
  | Poisson-CG | 9.4e-5 | 1.1e-2 |
  | poisson_man | 1.05e-6 | -- |

- **expF14 dysts chaos -- SUPPORT (optional).** Lorenz 1.13e-13 at W=693; Newton warm start is a DOP853 fit (external solver).
- **expF15/F16 (cdeng/) tensor-QI vs Radon vs BWLer -- SUPPORT.** Oracle ceilings. Constructive rank-2 tensor-QI for convection/wave from characteristics (no fit); sin(x+y)/sqrt(1+x^2): rank-2 tensor QI 2.41e-15 vs oracle-tuned 2-D Radon lstsq 7.55e-14 at 1156 coefficients. Nearest prior art to the known-function construction, but product architecture.
- **expF17 cross-method scaling suite -- CORE.** 16 tasks x methods (qi_radon, qi_tensor, elm, spectral, "bwler" = Chebyshev nodal, rbf, 1-layer PINN with refit); oracle and dynamic regimes; widths 121-4096; 3 seeds; 4594 cells. Dynamic QI-Radon at 4096: c40 2.2e-12, wave 1.1e-12, darcy_man 7.2e-15, reaction 2.7e-6, poisson_man 1.4e-7, Burgers 0.28. **No expF17_results.md** (awaits Sam's review).
- **expF18 2-D+t NS -- CORE.** Dynamic velocity 1.8e-3 (256), 5.7e-6 (512), 8.2e-9 (1024), 2.3e-9 (2048), 1.07e-11 (4096 units/field); oracle 2.8e-4 to 1.1e-12; error at corners and t=2 face; memory wall 12.3k unknowns at 7 GB. Imports `experiments/expH06_ridge_hierarchy/h06/core.py`.

## 4. Overclaims and contradictions
1. `expF_results.md` stale: "geometry wins 19x-685x" revised by expF03 to ties (1-D) and 2-100x (2-D); spectral beats QI 7/9 and 9/9; no F12-F18 entries.
2. expF06 "nonlinear floor ~1e-7" was an artifact; expF07 inherits it.
3. expF03 PINN device: writeup says CPU fp64, but every PINN record has `device_adam: "mps"` (Adam in fp32; only L-BFGS fp64). Likely does not change the conclusion; text is wrong.
4. expF13 Poisson-CG "below reference noise" false (reference good to 2.5e-5).
5. expF08 reference accuracy asserted not measured; interface-concentration statements contradict each other and expF05; "beats trained FNO" only after sigma=4 pre-smoothing.
6. expF12 "the moonshot works": 1.5e-6, tensor product, Beltrami advection is a pure gradient; cited json absent.
7. Raw data for expF05-F12 absent on this machine (run on Catherine's cluster).
8. expF17 "BWLer (explicit)" is a Chebyshev nodal basis, not BWLer; qi_tensor is a product basis; tuning on full set without validation split (Sam's ruling 17.1); darcy_orig never ran; no writeup.
9. expF18: w_mult=3 "inherited by every width and seed" but status.md shows seed 0 at 256/512 with 1.0; 2048 rung weak (N=12 gives 1.2e-10); "no training" still means a GN fit.
10. cdeng: expF15 BWLer table is fit-vs-solve (apples to oranges); Burgers scoring via .mat interpolation (1.9e-3 error); dysts references DOP853 rtol 1e-13.
11. Requirement violations if reused as "our method": global dense lstsq; polynomial block is not a tanh neuron; expF07 deep; expF12 and tensor-QI product architectures; expF14 RK warm start; Burgers nu-continuation.
