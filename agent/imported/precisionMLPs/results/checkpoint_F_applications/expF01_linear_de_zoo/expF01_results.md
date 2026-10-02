# expF01 -- linear differential-equation zoo: solve, don't train

**Status: draft -- conclusions pending Sam's sign-off.**

## TL;DR

- The QI-geometry collocation solve reaches the fp64 floor on **all nine** linear problems with zero training: 1D ODEs (floor by $W\approx200$), 2D steady PDEs on the disk, and space-time PDEs -- across orders 1-3 and across IVP / Dirichlet / Neumann / inflow / Cauchy conditions. **"Floor" throughout this repo means rel $L_2 \lesssim 10^{-13}$--$10^{-14}$** (the conditioning-limited level hit since Checkpoint A, not $\varepsilon_{\text{mach}}=2.2\times10^{-16}$); $L_\infty$ is reported alongside and runs $10^{-15}$--$10^{-11}$, the tail being $\gamma^r$ amplification on high-order operators.
- **The geometry is what buys the precision, not the frozen-feature trick.** Against a random-feature (ELM/PIELM) baseline at identical width and $\lambda$, the structured Radon/QI geometry wins by **19x--685x** on all six 2D problems. This is the Checkpoint C thesis reproducing in the PDE setting.
- **No oracle tuning is needed.** At a fixed $\lambda=0.25$ (no per-problem search), the 2D problems still reach $8.5\times10^{-15}$--$9\times10^{-12}$. The $\lambda$/width sweeps in the figure are a study of the basin, not a requirement of the method.
- **Condition placement is a first-class correctness requirement.** The dispersive case ($u_t+u_{xxx}=0$) first plateaued at $3\times10^{-8}$ because the well-posed split for leftward group velocity puts two conditions at the *right* edge and the run had them on the left. Fixing it gives $1.7\times10^{-13}$. **The PDE residual does NOT detect this** (it is identical for the right and wrong problem); **nested-width self-consistency does**, and needs no exact solution. See *Additional details*.
- High-order operators show a $\gamma^{r}$ roundoff drift: the order-3 ODE bottoms at $2.7\times10^{-15}$ at $W=173$ and rises to $\sim10^{-11}$ by $W=1845$. Optimal width is finite; more neurons is not better.
- Full suite (~130 lstsq solves + all figures/gifs) runs in ~7 minutes on CPU.

## Question

Does the frozen-geometry + lstsq recipe solve *actual* linear ODEs/PDEs -- across operator order (1/2/3), domain type (interval, disk, space-time), and condition type (IVP, Dirichlet, Neumann, inflow, Cauchy, IC) -- and to what precision as width grows?

## Experiment design

Model: $u(p) = \sum_m a_m \tanh(\gamma(w_m^\top p - t_m)) + \text{poly}(p)$, poly being all monomials of total degree $\le 3$. **The poly block is a low-frequency supplement, not "the kernel of $L$":** it coincides with $\ker L$ only for pure-derivative operators (for $-\Delta+4I$, constants are emphatically not annihilated -- their PDE-block entries are $4$). Its role is to supply the smooth low-order modes that a single sharp bandwidth $\gamma$ represents inefficiently, and it earns its place empirically (see *Ablations*). Geometry is frozen: 1D uses the uniform QI grid with halo $R=\max(70,0.4N)$ and $\gamma=\lambda/h$ at $\lambda=0.25$; 2D uses the expE01 Radon tensor ridges ($\sqrt W$ directions $\times$ $\sqrt W$ offsets, collar 1.25 disk / 1.6 square, $\gamma=\lambda/h_\text{ref}$, $h_\text{ref}=2.8/\sqrt W$).

For a linear operator $L=\sum_i s_i D_i$ the PDE rows are

$$(L\Phi)_{jm} \;=\; \sum_i s_i(p_j)\;\gamma^{o_i}\,\pi_i(w_m)\;\psi^{(o_i)}(Z_{jm}), \qquad Z_{jm}=\gamma(w_m^\top p_j - t_m),$$

with $o_i$ the order of $D_i$, $\pi_i(w_m)$ the product of direction components from the chain rule, and $\psi^{(o)}$ the $o$-th tanh derivative (a closed-form polynomial in $\tanh$; FD-verified). Condition rows are dictionary rows (or derivative-dictionary rows) at boundary/initial points. Everything is assembled into one system and solved by a single min-norm lstsq (`rcond=1e-13`), with the PDE block scaled to $O(1)$ by its max entry and each condition block weighted $\sqrt{n_\text{pde}/n_\text{block}}$.

The nine problems (exact solutions manufacture $f$ and the condition values):

| category | order | problem | conditions |
|---|---|---|---|
| 1D ODE | 1 | $u' + (2+\sin\pi x)u = f$ (variable coeff.) | IVP $u(-1)$ |
| 1D ODE | 2 | $u'' + 0.4u' + 100u = f$ (damped oscillator) | $u(-1)$, $u'(1)$ |
| 1D ODE | 3 | $u''' + 4u' = f$ (steady dispersion) | $u(-1), u'(-1), u(1)$ |
| 2D steady | 1 | $b\cdot\nabla u + u = f$ (transport + decay) | Dirichlet on the inflow arc |
| 2D steady | 2 | $-\Delta u + 4u = f$ (screened Poisson) | Dirichlet on the circle |
| 2D steady | 3 | $u_{xxx} + u_{yyy} + u = f$ (stress test) | Cauchy ($u$, $\partial_n u$) on the circle |
| space-time | 1 | $u_t + u_x = 0$ (traveling pulse) | IC + inflow |
| space-time | 2 | $u_t = 0.15\,u_{xx}$ (two decaying modes) | IC + Dirichlet both ends |
| space-time | 3 | $u_t + u_{xxx} = 0$ (two dispersive modes) | IC + 2 right / 1 left (see below) |

Space-time problems are posed on $(x,\tau)$ with $\tau=2t-1$ (so $\partial_t = 2\partial_\tau$) and solved as anisotropic 2D problems on the square: **time is a coordinate, the IC is a row block on the $t=0$ edge, and there is no time-stepping.** The three space-time solutions are classical (forcing $\equiv 0$), so those solves are driven *entirely* by the IC/BC rows -- if you kept only the PDE block, min-norm lstsq would return $a=0$.

Sweeps: 1D $N\in\{8,\dots,1024\}$ at fixed $\lambda=0.25$; 2D $W\in\{144,\dots,2304\}$ (space-time to 4096), with $\lambda$ swept on a 7-point grid at anchor width 1024 per problem and then refined $\pm 0.04$ around that optimum at every width (best kept per cell). Collocation: $4W$ uniform (1D) / $5W$ area-uniform random, single seed (2D); eval on grids strictly finer than the collocation spacing (1D $3\times$; 2D $241^2$). Metrics: rel $L_2=\|\hat u-u^*\|_2/\|u^*\|_2$ and absolute $L_\infty$. Every hand-coded forcing, derivative, and condition value is verified against finite differences at startup (`problems.verify_all`); the run refuses to start otherwise.

**Code & data.** `experiments/expF01_linear_de_zoo/` (`run.py`, `problems.py`). Data: `data.json`. Figures: `error_vs_width.png` (deliverable), `function_representations/{ode1d,pde2d_steady}/order{1,2,3}.png`, `function_representations/pde2d_time/order{1,2,3}.gif`.

## Results

Best cell per problem (rel $L_2$ / $L_\infty$):

| | order 1 | order 2 | order 3 |
|---|---|---|---|
| **1D ODE** | $6.4\times10^{-15}$ @ $W{=}923$ | $8.3\times10^{-15}$ @ $W{=}205$ | $2.7\times10^{-15}$ @ $W{=}173$ |
| **2D steady** | $8.3\times10^{-14}$ @ $W{=}2304$ | $9.2\times10^{-15}$ @ $W{=}2304$ | $1.2\times10^{-13}$ @ $W{=}2304$ |
| **space-time** | $3.1\times10^{-13}$ @ $W{=}4096$ | $1.1\times10^{-13}$ @ $W{=}4096$ | $1.7\times10^{-13}$ @ $W{=}4096$ |

- **Every problem reaches the fp64 floor.** No condition type resisted: inflow-only data, mixed Dirichlet/Neumann, and full Cauchy data all behave like plain Dirichlet.
- **1D is at the floor almost immediately** ($W\approx160$--$205$), then the second- and third-order problems *drift back up* with width ($\gamma^r$ roundoff amplification in the PDE block). Order 1 shows no drift.
- **2D descends steeply and cleanly.** Best $\lambda$ sits at $0.20$--$0.29$ and drifts down with width, as in expE01. Error fields are spatially uniform (no boundary/interior structure).
- **Space-time works exactly like a 2D steady problem.** Advection descends $6\times10^{-2}\to3\times10^{-13}$; heat $2\times10^{-3}\to1\times10^{-13}$; dispersion $3.8\times10^{-6}\to1.7\times10^{-13}$. There is no error growth in $t$: the slab is solved at once, so nothing accumulates.

### Figures

- **`error_vs_width.png`** (deliverable) -- $3\times3$ grid, rows = operator order, columns = {1D ODE, 2D steady, space-time}; each panel rel $L_2$ (solid) and $L_\infty$ (dashed) vs total width, log-log, with a $10^{-13}$ reference line. Read for: the floor in every panel, and the order-2/order-3 upward drift in the 1D column.
- **`function_representations/ode1d/order{1,2,3}.png`** -- exact vs solved overlay (left) and $|\hat u - u^*|$ on log scale (right) at each problem's best cell.
- **`function_representations/pde2d_steady/order{1,2,3}.png`** -- 3D surface of $u^*$ (left) and $\log_{10}|\hat u - u^*|$ heatmap on the disk (right); the order-2 map is uniform at $\sim10^{-14.5}$.
- **`function_representations/pde2d_time/order{1,2,3}.gif`** -- $u(x,t)$ animated over $t\in[0,1]$ (exact solid, solved dashed, locked axes) with the log-scale error profile beside it. Watch the error stay flat in $t$: no accumulation, because there is no rollout.

## Ablations and baselines

Preliminary, run as single spot-checks at $W=2304$ (2D) or $W=1024$ unless stated. **A full ablation study (expF03) should turn each of these into a proper sweep with seeds and error bars**; the numbers below are the direction, not the final word.

**Baseline: is it the geometry, or just the frozen-feature trick?** The core algorithm (freeze features, solve the readout by lstsq) is shared with Extreme Learning Machine collocation / PIELM / random-feature methods. So the load-bearing question is whether the *QI-structured* geometry does anything a random one wouldn't. Swapping the Radon tensor grid for random directions and offsets, everything else identical, $\lambda=0.25$ fixed:

| problem | Radon/QI | random features (best of 3 seeds) | gain |
|---|---|---|---|
| steady transport | $1.1\times10^{-13}$ | $2.1\times10^{-12}$ | 19x |
| screened Poisson | $8.5\times10^{-15}$ | $1.7\times10^{-12}$ | 200x |
| steady 3rd-order | $9.6\times10^{-14}$ | $2.2\times10^{-11}$ | 228x |
| advection | $9.1\times10^{-12}$ | $2.9\times10^{-9}$ | 317x |
| heat | $1.3\times10^{-12}$ | $8.6\times10^{-11}$ | 65x |
| Airy | $3.3\times10^{-12}$ | $2.3\times10^{-9}$ | 685x |

Random features are *not bad* ($10^{-9}$--$10^{-12}$, consistent with what the PIELM literature reports); the structured geometry is 1--2.5 orders better on every problem. This is the Checkpoint C result ("geometry is the whole game") reproducing in the PDE setting, and it is the single most important thing expF03 should nail down properly -- with seeds, across widths, and against further baselines (RBF centers, Chebyshev/spectral, and a trained PINN).

**No oracle tuning is required.** The headline table below selects $\lambda$ and reports the best width, both of which use $u^*$ and are therefore not available on a real problem. That selection is *not* what makes the method work: at a flat $\lambda=0.25$ with no search at all, the six 2D problems give $1.1\times10^{-13}$, $8.5\times10^{-15}$, $9.6\times10^{-14}$, $9.1\times10^{-12}$, $1.3\times10^{-12}$, $3.3\times10^{-12}$ (the "Radon/QI" column above). expF03 should report the no-oracle numbers as primary and treat the $\lambda$ sweep as a basin study.

**Polynomial supplement.** Removing the degree-3 block barely moves 1D or space-time, but degrades the steady-2D problems by roughly 1--3 orders. It is a real component of the method, not a cosmetic null-space completion. Note that steady-transport's manufactured solution happens to contain $0.3(x^2+y^2)$, exactly representable by that block -- but the result is not an artifact of it: a target with no polynomial component still reaches $1.0\times10^{-13}$ at $W=2304$.

**rcond is a regularization knob, not an innocuous constant.** The solve hardcodes `rcond=1e-13`. At $W=1024$ on the two order-3 problems:

| rcond | 1e-12 | 1e-13 | 1e-14 | 1e-15 |
|---|---|---|---|---|
| steady order-3 | $1.0\times10^{-11}$ | $2.6\times10^{-13}$ | $3.2\times10^{-13}$ | $2.8\times10^{-13}$ |
| space-time order-3 | $1.2\times10^{-11}$ | $3.0\times10^{-12}$ | $3.1\times10^{-13}$ | $1.3\times10^{-13}$ |

A decade of rcond costs up to two orders of accuracy, and $10^{-13}$ is not even optimal for the space-time case. expF03 must sweep it and log rank + retained singular values (the current code logs neither).

**Not yet ablated** (all belong in expF03): condition-block weighting (each block currently gets aggregate weight comparable to the entire PDE block -- defensible, not neutral); halo size; collocation oversampling ratio; seed variance (single seed here; spot checks move well under a decade but this needs error bars).

## Additional details

**The dispersion bug, and what actually caught it.** The first version of the space-time order-3 problem carried its extra Neumann condition on the *left* edge and plateaued at rel $L_2\approx3\times10^{-8}$, non-monotone in width. For $u_t+u_{xxx}=0$ with $u\sim e^{i(kx-\omega t)}$, the dispersion relation is $\omega=-k^3$, so the group velocity $\mathrm{d}\omega/\mathrm{d}k=-3k^2<0$: energy travels leftward, information enters from the right, and the well-posed split is **two conditions at the right edge, one at the left**:

| BC placement | $W=576$ | $W=1024$ | $W=2304$ |
|---|---|---|---|
| 2-left (ill-posed) | $1.2\times10^{-7}$ | $1.6\times10^{-8}$ | $3.5\times10^{-8}$ (plateau) |
| 2-right (well-posed) | $8.2\times10^{-11}$ | $3.0\times10^{-12}$ | $1.7\times10^{-13}$ |

**The residual does not detect this, and a tiny residual must not be read as a correct solution.** This is worth stating loudly because the opposite is the tempting inference. Comparing the ill-posed and well-posed systems at $W=1024$:

| | ill-posed (2-left) | well-posed (2-right) |
|---|---|---|
| scaled residual $\max|r|$ | $6.22\times10^{-11}$ | $6.14\times10^{-11}$ |
| rank | 762/1034 | 762/1034 |
| $\sigma_{\min}/\sigma_{\max}$ | $1.68\times10^{-17}$ | $1.40\times10^{-17}$ |
| **true error** | $3.2\times10^{-8}$ | $3.0\times10^{-12}$ |

The residuals are identical to two digits across a four-order difference in accuracy, and rank and the singular spectrum do not separate them either. The solver faithfully solved the wrong problem, and every internal signal said it had succeeded.

**What does work, without $u^*$: nested-width self-consistency.** Compare successive solutions to *each other* rather than to the truth:

| | $\|u_{1024}-u_{576}\|$ | $\|u_{2304}-u_{1024}\|$ | true error |
|---|---|---|---|
| ill-posed | $2.7\times10^{-7}$ | $4.1\times10^{-8}$ (stalling) | $3.2\times10^{-8}$ |
| well-posed | $1.2\times10^{-10}$ | $3.1\times10^{-12}$ (converging) | $3.0\times10^{-12}$ |

The successive difference tracks the true error to within a small factor and requires no exact solution. This is the deployable diagnostic, and it doubles as the width/$\lambda$ selector that replaces the oracle tuning above. expF03 should implement it as a standard emitted metric.

**Other caveats.** Single collocation seed. The anchored-$\lambda$ policy matched the full-sweep optimum wherever both were computed; the $\lambda$ basin is wide, as in expC03/expE01. eval $L_\infty$ on a finite grid is a lower bound. A small physical-residual note: because an order-$r$ operator amplifies function error by $\gamma^r$, the *physical* (unscaled) PDE residual of a solution at rel $L_2\sim10^{-13}$ can still be $\sim10^{-8}$ -- "solution at the floor" and "physical residual at the floor" are different statements and should not be conflated.

## Conclusions

*Proposed, pending Sam.* On nine smooth manufactured linear differential problems in one and two coordinates, the frozen QI/Radon geometry plus one least-squares solve reaches the repo's precision floor (rel $L_2\lesssim10^{-13}$) with no training, across orders 1--3 and every tested condition type. The precision is attributable to the *geometry*, not merely to freezing features: a random-feature baseline at identical width is 19x--685x worse, and no oracle tuning is needed to get there. Three operational rules come out of it: high-order operators have a finite optimal width ($\gamma^r$ roundoff drift); condition placement must respect the direction of information flow; and correctness must be judged by nested-width self-consistency, because the residual, the rank, and the singular spectrum all fail to distinguish a well-posed solve from an ill-posed one.

Scope, stated plainly: this is a controlled feasibility result on smooth manufactured problems, not a demonstration of competitiveness with established solvers. It does not establish behaviour on shocks, discontinuous coefficients, singular geometry, $d\ge3$, inverse/eigenvalue problems, or noisy data, and the dense global solve scales far worse than a sparse local stencil method.

## Open questions

- **expF03 -- the ablation and baseline study.** The single highest-value follow-up, scoped in *Ablations* above: no-oracle protocol as primary; random-feature / RBF / spectral / trained-PINN baselines; rcond, polynomial-block, weighting, halo, and seed ablations with error bars; rank and singular-value logging.
- **Deployable model selection.** Turn nested-width self-consistency into the standard width/$\lambda$ selector and verify it picks the finite optimal width that the $\gamma^r$ drift creates.
- **Stability theory.** QI gives expressivity (the dictionary contains an accurate approximant); it does not prove the collocation system inherits the PDE's stability. The 2D Radon geometry is empirical, not the 1D construction generalized by theorem.