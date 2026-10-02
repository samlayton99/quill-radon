# expF03 -- ablations, deployable model selection, and baselines

**Status: all four parts complete -- conclusions proposed, pending Sam.**

Turns the expF01/expF02 feasibility results into a method: which design choices carry the precision (part 1), how to pick a config without the exact solution (part 2), how it compares to other solvers on the linear zoo (part 3), and whether those other solvers reach precision on the nonlinear zoo when given our machinery (part 4).

**Data map.** `experiments/expF03_ablation_and_baselines/`: `common.py` (consolidated solver), `run.py` (CLI), `baselines*.py`, `baselines_nonlinear.py`, `heldout_problems.py`. Results grouped by part:
- `part1_ablations/` -- `ablate.json`, `refine.json`, `rescue.json`; `ablate_summary.png`, `ablate_shapes.png`, `rescue.png`.
- `part2_selection/` -- `select*.json`; `selection_oracle_vs_deployed.png`, `selector_signal_vs_error.png`, `selection_trajectories.png`.
- `part3_baselines/` -- `baselines_<method>.json`; `baseline_bars.png` (+ `baseline_bars_fair.png`), `baseline_error_vs_width.png`, `baseline_tradeoff.png`.
- `part4_nonlinear_baselines/` -- `nl_<method>.json`; `nl_baseline_bars.png` (+ `nl_baseline_bars_fair.png`).
- `cache/*.npz` -- eval-grid predictions per solve (gitignored; feeds the part-2 selector).

---

# Part 1 -- oracle ablations

## TL;DR

- **Newton initialization is the dominant nonlinear knob.** A coarse-to-fine *cascade* init (converge at $W/4$, refit, polish) improves four of six 2D nonlinear problems by 3--5 orders at fixed width and $\lambda=0.25$. The expF02 hard cases were Newton-basin artifacts, not representation limits.
- **All four rescue targets improved** at flat $\lambda=0.25$, seed-checked where noted: eikonal $4.8\times10^{-10}\to6.6\times10^{-15}$; stress order-3 $9.2\times10^{-12}\to1.3\times10^{-14}$; viscous Burgers @$2304$ $1.3\times10^{-7}\to6.5\times10^{-15}$; inviscid Burgers $9.8\times10^{-6}\to5.8\times10^{-12}$.
- **The nonlinear finite-optimal-width drift was a solver artifact** -- under cascade + `rcond=1e-15` error now decreases through $W=2304$ on viscous Burgers, KdV, and (with block down-weighting) stress. The **linear 1D order-3 drift is real**: the best knob leaves $4.6\times10^{-12}$ at $N=1024$ vs $2.7\times10^{-15}$ at the optimal width.
- **Knobs are not additive.** Two individually-winning knobs frequently lose to the better single knob (cascade+LM is 3 orders worse than cascade alone on the eikonal); winners must be validated jointly.
- **$\lambda=0.25$ flat survives.** Basin 0.20--0.30, cliff at $\ge0.35$; flat 0.25 stays within 20x (usually 4x) of the per-problem optimum. Apparent extreme $\lambda$-sensitivity under a bad init is Newton-basin luck. Small $\lambda$ (0.10) is a second handle on the 1D drift (34--49x).
- Everything here is **oracle-selected** (winner chosen by true error). Part 2 tests what survives without it.

## Design

All solves use the expF01/expF02 machinery consolidated into `common.py`; a **reproduction gate** requires it to reproduce recorded expF01/expF02 cells before any run (it does, 0.93--1.06x). Method is probe-then-refine, not a factorial: round 1 varies **one axis at a time** around the default at probe widths (1D $N\in\{32,1024\}$; 2D $1024$ linear / $576$ nonlinear); round 2 runs targeted combos of the winners; a rescue round attacks the four hard problems. $\lambda=0.25$ flat except on the $\lambda$ axis. Metrics as in expF01 (rel $L_2$ on a fixed fine grid; $L_\infty$ logged). Every solve also logs rank, retained singular values, coefficient norms, train residual, fresh-point physical residual, and Newton history -- part 2's raw material.

**Knobs** (default in parentheses): `lam` (0.25, $\lambda=\gamma h$); `rcond` ($10^{-13}$, SVD truncation vs $\sigma_{\max}$); `w_mult` (1.0, multiplier on the $\sqrt{n_{\rm pde}/n_{\rm blk}}$ condition-block weight); `halo` ($\max(70,0.4N)$, 1D exterior centers); `collar` (1.25 disk / 1.6 square, Radon offset half-width); `oversample` (4x/5x, collocation rows per neuron); `poly_degree` (3, monomial supplement); `equilibrate` (off, unit-norm columns pre-SVD); `max_newton` (25), `rcond_tighten` (off).

**Newton solver.** Damped Gauss-Newton on $r(a)=\sum_i s_i(D_i\Phi)a + N(Du_a)-f$. *Backtracking* (default, expF02's solver): full step, halve up to 8x until the stacked residual drops. *Levenberg-Marquardt* (`lm`): adaptive Tikhonov damping $\min_\delta\|[J;\sqrt\mu I]\delta+[r;0]\|$, $\mu$ shrunk 3x on accept / grown 4x on reject; also regularizes a rank-deficient $J$.

**Initialization ladder** (each = one or a few *linear* solves): `zero`; `classical` (problem-specific linear pre-solve -- harmonic extension for the eikonal, $\Delta u=2\sqrt f$ for Monge-Ampere); `bcfit` (min-norm lstsq on the condition rows alone); `linpart` (nonlinear term deleted, e.g. Burgers $\to$ heat); `cascade` (solve nonlinear at $W/4$, fit the full-width dictionary to it, Newton-polish); `picard` (freeze one bilinear factor, 2--3 fixed-point steps); `homotopy` (scale the nonlinearity $s:0\to1$); *(rescue only)* `continuation` (add artificial dissipation, shrink to 0).

## Results

### Hard problems, before and after

"Before" is expF02's best cell (which swept $\lambda$); "after" is flat $\lambda=0.25$ with the stated config. Oracle-selected; seeds checked where shown.

| problem | before | after | config |
|---|---|---|---|
| eikonal | $4.8\times10^{-10}$ | $6.6\times10^{-15}$ @$1024$ (seeds $7.9\times10^{-15}$, $1.1\times10^{-14}$) | cascade + rcond $10^{-15}$ |
| Monge-Ampere | $1.2\times10^{-14}$ (tuned $\lambda$) | $1.2\times10^{-13}$ @$576$ | collar 2.0 + poly 5 + rcond $10^{-15}$ |
| stress order-3 | $9.2\times10^{-12}$ | $1.3\times10^{-14}$ @$2304$ (3 seeds $1.26$--$1.28\times10^{-14}$) | cascade + rcond $10^{-15}$ + w_mult 0.1 |
| viscous Burgers | $1.3\times10^{-7}$ @$2304$ | $6.5\times10^{-15}$ @$2304$ | cascade + rcond $10^{-15}$ |
| KdV | $4.7\times10^{-8}$ @$2304$ | $2.8\times10^{-13}$ @$2304$ | cascade + rcond $10^{-15}$ |
| inviscid Burgers | $9.8\times10^{-6}$ | $5.8\times10^{-12}$ @$2304$ | LM + `max_newton=60` |

Under the new configs the width trend is monotone through $W=2304$ (viscous Burgers $3.4\times10^{-12}\to1.0\times10^{-12}\to6.5\times10^{-15}$ over $576/1024/2304$; KdV similar). Stress still drifts under cascade+rcond alone ($2.0\times10^{-13}$@$1024\to8.9\times10^{-11}$@$2304$); `w_mult=0.1` removes it.

### Per-knob findings

- **init (nonlinear): dominant.** At $W=576$, switching only the initializer: eikonal $2.7\times10^{-8}\to3.1\times10^{-13}$ (cascade), stress $2.1\times10^{-6}\to1.2\times10^{-11}$ (cascade), viscous Burgers $1.0\times10^{-6}\to1.9\times10^{-12}$ (bcfit), KdV $3.7\times10^{-8}\to4.0\times10^{-12}$ (cascade). `zero` diverges on Monge-Ampere (its linearization at $u=0$ vanishes). `picard`/`linpart`/`homotopy` never won.
- **rcond (nonlinear): large.** $10^{-15}$ gains 4--5 orders on viscous Burgers/KdV @$576$. Linear side is two-sided: `rcond=0` helps 1D order-2 drift 13x but costs 2 orders on order-3 @$32$. No universal setting.
- **poly degree + collar (2D steady): large.** Degree 5 gains up to 109x (screened Poisson $2.2\times10^{-12}\to2.0\times10^{-14}$); removing the block loses 2--4 orders (Monge-Ampere $3.1\times10^{-10}\to4\times10^{-6}$). Collar 1.6--2.0 beats 1.25; collar 1.0 hurts up to 3 orders.
- **w_mult: real, two-sided.** Down-weighting ($0.03$--$0.1$) is the best knob for the linear 1D drift (46x on order-2 @$1024$: $1.4\times10^{-12}\to3.0\times10^{-14}$) and completes the stress rescue, but *hurts* the logistic IVP (its single IC row is the only inhomogeneous data).
- **halo: load-bearing but binary.** `halo=0` loses 6--9 orders ($2.5\times10^{-15}\to2\times10^{-5}$ on order-3); the default is already adequate.
- **lam: basin at 0.25.** Under winning configs the basin is 0.20--0.30, $\lambda\ge0.35$ costs 1--3 orders. The extreme $\lambda$-sensitivity of *default-init* nonlinear solves collapses under cascade init -- which retroactively explains why expF02's tuned-$\lambda$ cells beat its flat-$\lambda$ ones. Small $\lambda$ (0.10) gains 34--49x on the 1D drift, consistent with the drift being $\gamma^r=(\lambda N/2)^r$ roundoff.
- **LM: a rescue tool, not a default.** The only knob that moves inviscid Burgers ($2.8\times10^{-5}\to1.0\times10^{-7}$@$576$), but *diverges on the logistic IVP* and costs an order on the other 1D ODEs.
- **Small / null:** oversampling (<7x), `equilibrate` (3x at best -- a notable *negative* result, since it targets the $\gamma^r$ drift directly), `rcond_tighten` (nothing), continuation (loses everywhere, 5--25x slower).
- **Non-additivity.** cascade+LM $1.5\times10^{-10}$ vs cascade $2.6\times10^{-13}$ (eikonal); collar1.6+poly5 $2.0\times10^{-13}$ vs poly5 $2.0\times10^{-14}$ (screened Poisson). Not universal -- the union cascade+rcond+poly5+collar1.6 dominates plain cascade+rcond on all three steady nonlinear problems (Monge-Ampere $2.8\times10^{-10}\to8.1\times10^{-14}$).

### Figures

- **`ablate_summary.png`** -- problem $\times$ knob heatmap; cell $=\log_{10}$(best setting on that axis / default), blue = knob helps, red = every tried setting hurts. Read for the init/rcond/poly block lighting up on nonlinear-2D rows and the near-white linear space-time rows (defaults already right).
- **`ablate_shapes.png`** -- one panel per knob with a shape worth seeing; per panel, the median effect over problems (line) and full range (band) for linear vs nonlinear, normalized to default (<1 = better), each panel on its own scale. Read for the *form* of each knob: basin ($\lambda$, collar), cliff (halo), two-sided (rcond, w_mult), flat (oversample), cascade dipping below 1 (init). Single-value/null knobs (equilibrate, lm, rcond_tighten) are omitted here -- they live in the heatmap.
- **`rescue.png`** -- per hard problem: horizontal bars of every lever (green = best) over the default, plus Newton-residual histories; the continuation trails show the stall LM avoids.

### Load-bearing caveats

- The **linear 1D order-3 drift stands**: best single knob @$1024$ is `w_mult=0.03` ($6.3\times10^{-11}\to4.6\times10^{-12}$), still 3 orders above the $2.7\times10^{-15}$ optimum at $W=173$; no combo beats it. Width selection, not configuration, handles this.
- **Advection (linear space-time) is width-limited**: @$1024$ no knob moves it >3x from $6.4\times10^{-7}$; expF01 already showed it needs $W\approx4096$ for $3\times10^{-13}$.
- Single collocation seed except the stated 3-seed rescue checks (spread $\le2$x). Probe widths only (part 2 supplies full curves). LM used one damping schedule; its logistic divergence may be schedule-specific.

## Conclusions (part 1)

*Proposed, pending Sam.* On these 18 problems at flat $\lambda=0.25$: (1) the expF02 hard cases were solver-basin problems, not representation problems -- changing only the Newton init and rcond, no $\lambda$ retuning, no training, brings eikonal, stress, viscous Burgers, and KdV to $\le3\times10^{-13}$ at the tested widths; (2) inviscid Burgers additionally needs LM + more iterations, reaching $5.8\times10^{-12}$; (3) the knobs that must be right are init, rcond, the polynomial block, collar, and condition-block weighting, and they interact, so a deployable recipe must be validated as a package -- part 2.

---

# Part 2 -- deployable model selection

## TL;DR

- **The full no-oracle protocol works.** A frozen per-category recipe + width ladder + nested-width selector, with $u^*$ used nowhere, matches the oracle width on 12/18 development problems and lands within 3.5x of the oracle error on 17/18 (the one 12x miss is between two machine-eps solutions).
- **Held-out: 3 for 3, zero regret.** On three unseen problems -- damped wave (a two-IC condition type absent from the zoo), variable-coefficient convection-diffusion, Fisher-KPP -- the frozen selector picks the oracle width exactly, reaching $1.6\times10^{-13}$, $3.1\times10^{-14}$, $2.0\times10^{-15}$.
- **The selector catches the $\gamma^r$ drift**: 1D order-3 gets $W^*=32$ (the true optimum $2.5\times10^{-15}$) where fixed $W=1024$ costs 4 orders. Fixed-width loses up to 5 orders (advection).
- **The deployed recipe sometimes beats the old oracle**: Monge-Ampere reaches $1.0\times10^{-15}$ @$2304$ vs expF02's $\lambda$-tuned $1.2\times10^{-14}$. The stall-triggered LM fallback fired only on inviscid Burgers ($5.8\times10^{-12}$, unattended).
- **The tuning signal must include the condition rows.** The homogeneous-problem $u\equiv0$ trap (below) is the sharpest result: PDE residual alone endorses an exactly-wrong solution.

## Design

**The deployed protocol** (all frozen before the held-out run):
1. *Recipe* (per setup $\times$ category, from part 1): linear 1D and space-time use repo defaults; linear steady-2D adds poly 5 + collar 1.6; nonlinear 2D uses cascade + rcond $10^{-15}$, steady additionally poly 5 + collar 1.6. $\lambda=0.25$ flat.
2. *Ladder*: $W\in\{8..1024\}$ (1D, doubling) / $\{144,256,576,1024,2304\}$ (2D; linear space-time to 4096), 3 seeds for 2D.
3. *Selector*: on a shared grid, $\delta_k=$ median-over-seeds of $\|u_{W_{k+1}}-u_{W_k}\|/\|u_{W_{k+1}}\|$; estimate $\mathrm{est}(W_k)=\max(\delta_{k-1},\delta_k)$; $W^*=\arg\min\mathrm{est}$, ties to smaller width.
4. *Stall fallback* (nonlinear): if the Newton residual stalls, re-solve with LM + 60 iterations, keep it iff its residual is lower. No $u^*$.

Compared against the same-ladder oracle, fixed $W=1024$, and a residual-pick control. **Leakage control:** the 18 zoo problems are the development set; the three held-out problems (`heldout_problems.py`, FD-verified) were solved once, after everything froze.

## Results

| | dev (18) | held-out (3) |
|---|---|---|
| $W^*$ = oracle width | 12 / 18 | **3 / 3** |
| regret $\le3.5$ x | 17 / 18 | 3 / 3 (zero regret) |
| worst regret | 12x, at machine eps ($2.6\times10^{-15}$ vs $2.1\times10^{-16}$) | -- |

- **Fixed-width loses badly**: $W=1024$ costs 5 orders on advection ($6.6\times10^{-7}$ vs $1.9\times10^{-12}$@$W^*=4096$), 4 on 1D order-3, 3 on 1D order-2.
- **The estimate is usable, not just the argmin**: $\mathrm{est}$ sits within ~1 order above the true error across the ladders (conservative almost everywhere), so the protocol reports an error bar.
- **Deployed quality at $W^*$** (median over seeds): all nine linear $2.1\times10^{-14}$--$2.3\times10^{-12}$; nonlinear -- logistic/Bratu/Blasius at machine eps, Monge-Ampere $1.0\times10^{-15}$, eikonal $4.9\times10^{-14}$ (selector took 576; oracle's 1024 gives $1.4\times10^{-14}$), stress $1.1\times10^{-12}$ (the one-recipe-per-category price: it gives up the problem-specific `w_mult=0.1` that reached $1.3\times10^{-14}$), viscous Burgers $6.5\times10^{-15}$, KdV $2.0\times10^{-13}$, inviscid Burgers $1.4\times10^{-11}$ via the automatic LM fallback.

### Which logged signals can drive parameter tuning (post-hoc over ~900 solves)

Within-problem Spearman between each $u^*$-free signal and the true rel $L_2$ (median over problem groups; $+1$ = ranks configs correctly):

| signal | across configs | across the width ladder |
|---|---|---|
| fresh-point PDE residual | **+0.79** | **+0.94** |
| train (in-sample) residual | +0.68 | -- |
| $\|a\|$ (tanh coefficients) | +0.34 | +0.80 (sign-flips on 1D) |
| smallest retained $\sigma$ | +0.38 | -- |
| rank | $-0.11$ | -- |

Acting on it -- pick the config with the smallest fresh residual per problem -- gives **median regret 1.4x, $\le$10x on 20 of 21 groups**, and one catastrophic failure that is the most instructive result: LM-diverged logistic returned $u\equiv0$, an *exact* solution of the homogeneous ODE $u'=3u(1-u)$, so its PDE residual is literally zero while the solution is 100% wrong; the entire misfit sits in the IC row ($|u(-1)-u^*(-1)|=4.7\times10^{-2}$). **The tuning signal must be the stacked residual including condition rows** -- every homogeneous problem has trivial exact branches only the conditions rule out. `fresh_bc_err` is now logged alongside `fresh_res_phys`.

Two signals, complementary jobs, and nothing else in the logged set earns a place: the **condition-inclusive fresh residual** is the cheap *relative ranker* for knob tuning at fixed width; the **nested-width $\delta$** is the *absolute, conservative error estimate* and width selector (and the only one that also flags misspecification, per expF01). Coefficient norms, rank, and the singular spectrum are not usable tuners here. One negative result: rcond-sensitivity does **not** track $\gamma^r$ (spans 1.2x--520x, no clean relation) -- there is no formula for the right rcond, but the whole rcond axis costs one SVD, so the rule is *sweep it and pick by the stacked fresh residual*.

### Figures

- **`selection_oracle_vs_deployed.png`** (deliverable) -- grouped bars per problem (18 dev + 3 held-out, marked): true rel $L_2$ at the oracle width, the selector's $W^*$, fixed $W=1024$, and the residual control; log scale. Read for green (selector) tracking gray (oracle), orange (fixed) losing up to 5 orders.
- **`selector_signal_vs_error.png`** -- $\mathrm{est}$ vs true rel $L_2$ over every ladder point, log-log with $y{=}x$; the cloud sits on-to-above the diagonal (conservative).
- **`selection_trajectories.png`** -- per problem: true error and $\mathrm{est}$ vs width, the shaded band = est$-$true gap, red line = selected $W^*$, ★ = held-out. Shows the knee-picking mechanics, including the V-shaped drift cases.

## Conclusions (part 2)

*Proposed, pending Sam.* With a frozen per-category recipe and the nested-width selector, the whole pipeline -- geometry, solve, width choice, error estimate -- runs without $u^*$ and, on three unseen problems including a condition type absent from development, picks the oracle width every time while reporting a correct conservative error bar. The oracle selection in expF01/expF02's tables was a presentation choice, not a requirement of the method. Caveat: dev-set regret stats are honest but in-sample; the 3-problem held-out set is the unbiased evidence, and it is small.

---

# Part 3 -- baselines

## TL;DR

- **Tuned Chebyshev spectral least squares is the method to beat**, winning 7 of 9 -- typically ~1 order, at equal or smaller DOF. On smooth, low-dim, nice-geometry problems a well-implemented spectral method is the gold standard; our floor is comparable ($10^{-13}$--$10^{-15}$) but not better.
- **The old 19x--685x "geometry wins" number needs revision.** That expF01 spot-check held random features at our untuned $\lambda=0.25$. Against *residual-tuned* random features (Dong & Yang's protocol) at their oracle width: **ties at the 1D floor, 2--100x in our favor on 2D**. Structured geometry still wins every 2D problem, but the honest margin is 1--2 orders, not 2.5--3.
- **Kansa-RBF is conditioning-limited far above everyone** ($10^{-5}$--$10^{-10}$): its optimal flat shape parameter is unreachable in fp64 (Larsson-Fornberg), verified directly.
- **The trained PINN lands exactly where the literature says** ($1.6\times10^{-6}$ best, $10^{-4}$--$10^{-3}$ typical) -- **8--11 orders behind the solve-based methods at 10--100x the wall-clock** (69--420 s vs $\le$4 s spectral, $\le$24 s ours). The project's thesis in one column: the gap is the optimizer, not the architecture -- the matched single-layer net does no better than 3x128.
- **FD converges at its fixed algebraic order** ($10^{-10}$ at dof 1845 on the easiest 1D problem, $10^{-4}$--$10^{-6}$ on space-time) -- never competitive on smooth problems at these budgets, as expected.

## Design

Nine linear problems (expF01 zoo). Every baseline implemented from the literature, not memory (WebSearch due diligence, sources below), FD-verified (`verify_bases()` checks each basis's derivative rows before any run), and given a *matched tuning budget* selected by the same $u^*$-free signal we use (the stacked fresh residual). Ours is reported at the **no-oracle deployed $W^*$ from part 2**; baselines at their **oracle-best cell** (best width/variant/seed) -- the comparison is deliberately tilted toward the baselines.

- **Random features (ELM/PIELM)**: $\tanh(w\cdot p+b)$, $w,b\sim U[-R,R]$ per Dong & Li (CMAME 2021); $R$ tuned by minimizing the collocation residual, Dong & Yang's (JCP 2022) own protocol (signal-picked $R$ matched the oracle $R$ on 7/9). Literature-default $R{=}1$ and tuned; 3 seeds.
- **Kansa RBF**: multiquadric and Gaussian; random centers with a denser boundary ring (Fedoseyev 2002); shape parameter from Franke's $c=1.25D/\sqrt N$, tuned over a multiplier grid (picked Gaussian on all nine). Pushing $c$ flatter collapses the solve (fp64 wall), so the tuned values are the genuine double-precision optimum (Larsson & Fornberg 2003).
- **Chebyshev spectral LS**: 1D at Chebyshev-Gauss-Lobatto nodes (Trefethen), extra low-degree rungs since it saturates by degree ~30; square tensor-CGL; disk total-degree Chebyshev on scattered points (Boyd & Yu 2011). Rows per-row equilibrated (spectral rows scale $k^{2r}$). Two of *our own* harness bugs were caught and fixed in the baseline's favor during bring-up (uniform-node collocation; block-scaling).
- **Trained PINN**: tanh MLPs in torch float64; literature-standard 3$\times$128 (Adam lr $10^{-3}$ exp decay, $\lambda_{bc}=100$, then strong-Wolfe L-BFGS; 3 seeds) and architecture-matched single layer at $W^*$. MPS (fp32) was only 1.25x faster than CPU fp64 for these small nets, below the threshold to pay -- CPU fp64 used.
- **Finite differences**: Fornberg-weight stencils (orders 2, 4), 1D + global space-time; disk skipped. Differentiation matrices verified by convergence rate.

Every solve logs assembly/solve/training wall-clock, DOF, and matrix shape.

## Results

Ours at the no-oracle deployed $W^*$ vs each baseline's oracle-best cell:

| problem | QI (deployed) | ELM (oracle) | RBF (oracle) | spectral (oracle, dof) | PINN (best) | FD4 (best) |
|---|---|---|---|---|---|---|
| 1D o1 | $2.1\times10^{-14}$ | $3.6\times10^{-14}$ | $5.9\times10^{-9}$ | $\mathbf{3.8\times10^{-15}}$ (33) | $1.8\times10^{-5}$ | $1.0\times10^{-10}$ |
| 1D o2 | $6.6\times10^{-15}$ | $\mathbf{5.3\times10^{-15}}$ | $1.4\times10^{-10}$ | $1.7\times10^{-13}$ (33) | $4.2\times10^{-5}$ | $6.4\times10^{-8}$ |
| 1D o3 | $2.5\times10^{-15}$ | $\mathbf{1.5\times10^{-15}}$ | $2.8\times10^{-9}$ | $2.7\times10^{-13}$ (33) | $1.6\times10^{-6}$ | $6.5\times10^{-9}$ |
| steady o1 | $4.9\times10^{-14}$ | $3.7\times10^{-13}$ | $5.3\times10^{-8}$ | $\mathbf{8.1\times10^{-15}}$ (561) | $4.5\times10^{-4}$ | -- |
| steady o2 | $3.8\times10^{-14}$ | $7.2\times10^{-14}$ | $6.2\times10^{-9}$ | $\mathbf{6.2\times10^{-15}}$ (253) | $2.5\times10^{-5}$ | -- |
| steady o3 | $2.9\times10^{-13}$ | $4.8\times10^{-12}$ | $3.3\times10^{-7}$ | $\mathbf{2.8\times10^{-14}}$ (253) | $3.6\times10^{-4}$ | -- |
| time o1 | $1.9\times10^{-12}$ | $1.3\times10^{-10}$ | $3.1\times10^{-6}$ | $\mathbf{2.0\times10^{-14}}$ (2304) | $7.9\times10^{-4}$ | $2.8\times10^{-4}$ |
| time o2 | $4.0\times10^{-13}$ | $3.4\times10^{-11}$ | $6.1\times10^{-6}$ | $\mathbf{3.7\times10^{-14}}$ (2304) | $4.7\times10^{-4}$ | $2.2\times10^{-6}$ |
| time o3 | $2.3\times10^{-12}$ | $2.6\times10^{-11}$ | $2.8\times10^{-5}$ | $\mathbf{6.1\times10^{-13}}$ (576) | $7.6\times10^{-5}$ | $7.6\times10^{-6}$ |

(FD skips the disk by design. Best-cell wall-clock: spectral $\le$4 s, ours $\le$24 s, PINN 69--420 s.)

- **Spectral wins 7/9** (all six 2D + 1D order 1), ~1 order at equal/smaller DOF; the two 1D exceptions are roundoff-limited cases where both sit at $10^{-13}$--$10^{-15}$. The expected outcome for smooth problems on nice domains -- the honest content of expF01's scope disclaimer. What QI retains over spectral: scattered points, one mechanism across interval/disk/square, and the Newton path to nonlinear problems (parts 1--2).
- **Geometry vs random features, measured fairly**: ties at the 1D floor; 2--100x for the geometry on 2D. expF01's 19x--685x compared *untuned* features and overstates the deployable gap; both numbers should be cited with their conditions.
- **RBF-Kansa is not competitive in fp64**, and verifiably not from under-tuning: its error minimizes at the flattest shape parameter fp64 supports (the uncertainty-principle limit). RBF-QR exists precisely for this; out of scope.
- **The PINN column is the thesis in one table.** Trained networks land 8--11 orders above every solve-based method at 10--100x the cost, exactly at the literature's well-tuned level -- so not a strawman -- and the matched single layer does no better than 3x128: the architecture is not the bottleneck, the optimizer is.
- **FD is the algebraic-convergence control** ($10^{-10}$ only on the easiest 1D problem at the largest grid). Its 1D order-2/3 curves also turn up at fine grids -- the $h^{-r}$ stencil conditioning, the FD analogue of our $\gamma^r$ drift.

### Figures

- **`baseline_bars.png`** (deliverable) -- 3x3, one panel per problem: log rel $L_2$ bar per method; ours at the no-oracle deployed $W^*$, baselines at oracle-best. The "who wins where" chart.
- **`baseline_bars_fair.png`** -- the same chart with QI *also* at oracle-best (best width + config), so both sides get oracle access. It flips only the two 1D drift cases (oracle width picks the pre-drift $W$) -- spectral still wins 7/9. The deployed-vs-oracle gap for QI on the linear zoo is small because the part-2 selector already lands on the oracle width; oracle access does not overtake spectral on its smooth home turf.
- **`baseline_error_vs_width.png`** -- 3x3 (rows = order, cols = category), rel $L_2$ vs DOF, one line per method (tuned variant, median over seeds), log-log, $10^{-13}$ reference. Read for three convergence classes -- exponential (ours, spectral, tuned ELM), exponential-until-conditioning (RBF), algebraic (FD) -- and the PINN points floating 6+ orders above.
- **`baseline_tradeoff.png`** -- best rel $L_2$ vs wall-clock per method per problem, log-log: the lstsq methods cluster at seconds/$10^{-13}$-ish; PINN sits minutes/$10^{-4}$.

## Conclusions (part 3)

*Proposed, pending Sam.* On nine smooth linear problems with every baseline implemented per its literature and given a matched-or-better tuning budget: (1) the solve-based methods separate from the trained PINN by 8--11 orders at a fraction of the cost -- the gap the QI program attributes to optimization, measured directly; (2) tuned Chebyshev spectral LS is the strongest method on these smooth, nice-domain problems and beats our deployed config on 7/9 -- our contribution is not out-precising spectral on its home turf, but reaching the same $10^{-13}$--$10^{-15}$ class with scattered points, one mechanism across domains, and a Newton path to nonlinear problems plus a no-oracle protocol; (3) the fair margin over random features is ties at the 1D floor and 2--100x on 2D, revising expF01's 19x--685x spot-check.

---

# Part 4 -- baselines on the nonlinear zoo

## TL;DR

- **The nonlinear Gauss-Newton solver is basis-agnostic**, so the ELM / RBF / spectral dictionaries drop into *our* harness unchanged -- same Newton, cascade init, rcond $10^{-15}$. The question: with our good techniques, do the other bases reach precision on the nine expF02 problems?
- **Chebyshev spectral + our Newton reaches machine epsilon on all nine** ($10^{-16}$--$10^{-13}$), matching or beating our deployed QI everywhere -- the linear part-3 result carries over to the nonlinear zoo.
- **Random features (ELM) + our Newton reach precision on all nine**: machine-eps on the three 1D ODEs and KdV ($7.6\times10^{-15}$), $10^{-11}$--$10^{-13}$ on the steady disk and viscous Burgers, $8\times10^{-10}$ on inviscid Burgers. Given our Newton harness and cascade init, the random-feature dictionary matches the QI method across the nonlinear zoo -- the strong affirmative answer to "can we get the other methods to work?"
- **Kansa-RBF + our Newton stays conditioning-limited**: it converges everywhere but only to $10^{-7}$--$10^{-9}$ (1D ODEs, steady disk, inviscid Burgers) and to a qualitatively wrong solution ($10^{-2}$--$10^{-1}$) on viscous Burgers and KdV -- the fp64 shape-parameter wall from part 3, which Newton cannot lift.
- **Trained PINN stays at training level** ($10^{-4}$--$10^{-5}$, best $5.8\times10^{-7}$ on Blasius) and **essentially fails Monge-Ampere** (rel $L_2=0.37$) -- the fully-nonlinear det-Hessian operator defeats the optimizer. 8--11 orders behind the solve-based methods at 70--420 s per problem: the project thesis, reconfirmed on nonlinear PDEs.
- **Finite differences: not run** (nonlinear FD = Newton on the stencil grid, and the disk is excluded); greyed at full height as the algebraic-convergence control already characterized in part 3.

## Design

Smaller-scale than part 3. The nonlinear solve in `common.gauss_newton` needs only a derivative dictionary $D[\text{idx}] = \text{basis.rows}(pts, [(\text{idx},1)])$ plus the problem's pointwise nonlinearity $N$, both already produced by the baseline bases -- so ELM, RBF, and spectral run through the *identical* Newton machinery, cascade initialization, and rcond as the QI method (`baselines_nonlinear.py`). Each is given a short DOF ladder (1D $\{17,33,65,129\}$; 2D $\{150,300,600,1200\}$), the three initializers (`cascade`, `bcfit`, `zero`) with the deployable no-oracle signal (stacked fresh residual) choosing between them, and per-method scale tuning (ELM's $R$, RBF's shape parameter, by the same residual). Ours is the part-2 deployed $W^*$ (no oracle); baselines are at their oracle-best cell. The trained PINN uses the literature-standard 3$\times$128 net with the nonlinear residual (`baselines_pinn.py`), single seed. FD is greyed.

**Code & data.** `baselines_nonlinear.py`, `baselines_pinn.run_pinn_nonlinear`; `part4_nonlinear_baselines/nl_<method>.json`; figure `nl_baseline_bars.png`.

## Results

| problem | QI (deployed) | ELM+Newton | RBF+Newton | spectral+Newton | PINN | FD |
|---|---|---|---|---|---|---|
| logistic (1D o1) | $1.7\times10^{-15}$ | $2.6\times10^{-16}$ | $8.6\times10^{-10}$ | $\mathbf{1.1\times10^{-16}}$ | $7.1\times10^{-5}$ | n/a |
| Bratu (1D o2) | $2.6\times10^{-15}$ | $9.3\times10^{-16}$ | $2.9\times10^{-9}$ | $\mathbf{1.5\times10^{-16}}$ | $3.7\times10^{-5}$ | n/a |
| Blasius (1D o3) | $2.5\times10^{-16}$ | $5.1\times10^{-15}$ | $8.0\times10^{-8}$ | $\mathbf{1.2\times10^{-16}}$ | $5.8\times10^{-7}$ | n/a |
| eikonal (steady o1) | $4.9\times10^{-14}$ | $3.4\times10^{-14}$ | $5.9\times10^{-9}$ | $\mathbf{5.4\times10^{-16}}$ | $2.0\times10^{-5}$ | n/a |
| Monge-Ampere (steady o2) | $1.0\times10^{-15}$ | $8.1\times10^{-13}$ | $9.0\times10^{-9}$ | $\mathbf{2.1\times10^{-16}}$ | $3.7\times10^{-1}$ | n/a |
| stress (steady o3) | $1.1\times10^{-12}$ | $1.2\times10^{-11}$ | $3.4\times10^{-8}$ | $\mathbf{1.6\times10^{-16}}$ | $5.0\times10^{-4}$ | n/a |
| inviscid Burgers (time o1) | $1.4\times10^{-11}$ | $7.8\times10^{-10}$ | $2.3\times10^{-7}$ | $\mathbf{1.8\times10^{-13}}$ | $6.8\times10^{-4}$ | n/a |
| viscous Burgers (time o2) | $6.5\times10^{-15}$ | $1.3\times10^{-11}$ | $6.5\times10^{-3}$ | $\mathbf{2.0\times10^{-16}}$ | $1.5\times10^{-4}$ | n/a |
| KdV (time o3) | $2.1\times10^{-13}$ | $7.6\times10^{-15}$ | $1.4\times10^{-1}$ | $\mathbf{8.9\times10^{-15}}$ | $1.4\times10^{-4}$ | n/a |

- **Spectral** wins outright (machine eps on all nine, smallest DOF), as on the linear zoo.
- **ELM** reaches the QI precision class everywhere -- the interesting negative: the structured QI geometry is *not* uniquely required for the stiff space-time problems; a random-feature dictionary in the same Newton harness solves them too (KdV to $7.6\times10^{-15}$). Where the QI geometry was ahead of random features on the linear zoo (part 3's 2--100x on 2D), that margin largely closes here once both use cascade Newton.
- **RBF** is the lone dictionary our machinery cannot rescue: it converges but the fp64 shape-parameter conditioning caps it at $10^{-7}$--$10^{-9}$, and on the two stiffest space-time fronts it converges to a wrong solution ($10^{-2}$--$10^{-1}$).
- **PINN** reconfirms the thesis on nonlinear PDEs: $10^{-4}$--$10^{-5}$ at 70--420 s, and an outright failure on Monge-Ampere -- 8--11 orders behind the solve-based methods.

### Figure

- **`nl_baseline_bars.png`** -- 3x3, one panel per nonlinear problem: log rel $L_2$ bar per method; ours at the no-oracle deployed $W^*$, dictionary baselines at their oracle-best cell, PINN best-of-run. FD is greyed at full height (N/A). Read for the two clean tiers (QI/ELM/spectral at $\le10^{-11}$ vs RBF at $10^{-7}$--$10^{-1}$ and PINN at $10^{-4}$) and the RBF/PINN collapses on the two Burgers/KdV space-time fronts.
- **`nl_baseline_bars_fair.png`** -- the same chart with QI *also* at oracle-best (width + config). QI's part-1 config lifts the hard cases (stress $1.1\times10^{-12}\to1.3\times10^{-14}$, inviscid Burgers $1.4\times10^{-11}\to5.8\times10^{-12}$) to tie random features; spectral still reaches machine eps everywhere. The tier structure is unchanged -- the fair comparison confirms QI $\approx$ ELM $<$ spectral, not an artefact of handicapping QI.

## Conclusions (part 4)

*Proposed, pending Sam.* Handing the competitor dictionaries our nonlinear machinery -- Gauss-Newton, cascade initialization, rcond $10^{-15}$ -- makes the collocation approach reach machine precision on the whole nonlinear zoo *regardless of dictionary*, provided the dictionary is well-conditioned: spectral and random features both join the QI method at $\le10^{-11}$ on all nine problems, spectral at machine eps. The QI geometry's specific edge over random features, visible in part 3's linear margins, does not persist on the nonlinear zoo -- what carries the precision is the Newton harness plus cascade init, not the choice of well-conditioned basis. The two methods our machinery cannot lift are RBF (fp64 conditioning wall) and the trained PINN (optimizer-limited at $10^{-4}$, failing Monge-Ampere outright) -- the same two separations found on the linear zoo, now confirmed under nonlinearity.

---

## Open questions (parts 1--4)

- Can the eikonal's remaining gap and inviscid Burgers' last ~1.5 orders be closed at larger width, or is that the ceiling for first-order fully-nonlinear/hyperbolic problems?
- Why does equilibration do nothing when the $\gamma^r$ column-scale disparity is exactly what it removes?
- LM divergence on the logistic IVP: damping schedule or something structural?
- Can the nested-width $\delta$ signal also select `rcond` (and $\lambda$) at $W^*$, closing the last per-problem gaps (stress's 85x) without an oracle?
- Width selection may be better done by *ensembling the two independent selectors* rather than committing to one. Post-hoc, the fresh-residual pick (the current "control") and the nested-width $\delta$ pick each hit the oracle width on most dev problems and disagree on only 7/18, all at already-converged error ($\le3.6\times10^{-12}$); on every high-stakes width problem (the $\gamma^r$ drift, advection) they agree. So a deployable rule that takes the geometric midpoint of the two picks (arithmetic mean in log-width, snapped to the nearest ladder rung, ties to smaller) is *correct where the stakes are high* and *harmless where it acts* -- plausibly more robust than either selector alone without needing an oracle. Untested where it could actually matter: a stakes-level *wide* disagreement (picks many rungs apart with a large error gap), which the 18 dev problems never produce. Needs the held-out set and more problems before adoption; if adopted, guard it by logging both individual picks and flagging when they diverge by more than a couple rungs *and* their error estimates differ by more than ~10x. (The aggregate dev-set metric weakly *disfavors* the midpoint, but that signal is one machine-eps bump out of 18 and is not trustworthy.)
- The obvious hybrid: can the QI geometry adopt spectral-style structured collocation where the domain allows, or can signal-driven per-problem tuning recover the remaining order to spectral?
- Part 4 shows random features match the QI geometry on the nonlinear zoo once both use cascade Newton -- so where, if anywhere, does the structured geometry earn its keep beyond the linear 2D margins? (candidates: worse-conditioned operators, higher dimension, fewer collocation points.)
- Can RBF be lifted off its fp64 conditioning wall with stable evaluation (RBF-QR) inside this same Newton harness, or is it fundamentally out of the machine-precision regime?

## After Sam signs off

Fold the confirmed numbers into `expF_results.md` (replace the "ablations so far show that" placeholders) and `results.md` Checkpoint F, and add the 19x--685x revision to expF01's Ablations section (both numbers with their conditions).
