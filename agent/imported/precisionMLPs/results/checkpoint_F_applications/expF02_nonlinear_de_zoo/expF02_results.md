# expF02 -- nonlinear differential-equation zoo: Newton, one lstsq per step

**Status: draft -- conclusions pending Sam's sign-off.**

## TL;DR

- Nonlinearity costs a handful of linear solves, not the method. Damped Gauss-Newton on the frozen QI geometry (each step a variable-coefficient linear PDE, solved by the same stacked lstsq) reaches **machine epsilon on all three 1D ODEs** ($\sim2\times10^{-16}$: logistic, Bratu, Blasius) and the **fp64 floor on Monge-Ampere** ($1.2\times10^{-14}$, a fully nonlinear 2D PDE) in 5 Newton steps.
- The other five land between $5\times10^{-10}$ and $10^{-5}$. Convergence is fast and uniform where it works: typically **5-7 Newton steps**, independent of width.
- **Two distinct failure modes, both visible as a *stalling* Newton residual.** (a) *Ill-posed condition placement*: the eikonal's full-circle Dirichlet over-determines its linearized transport operator (residual sticks at $1.4\times10^{-3}$); restricting to the inflow arc gains ~2 orders. (b) *Genuine Newton stagnation*: inviscid Burgers sticks with the residual at $1.1\times10^{-4}$, while its viscous sibling reaches $2.5\times10^{-10}$ -- the dissipation term is what rescues the iteration. Caveat that matters: a residual that stalls *far above* the floor is a real, $u^*$-free alarm, but a residual *at* the floor certifies nothing (expF01 shows an ill-posed linear solve whose residual is indistinguishable from the well-posed one). Convergence must be judged by nested-width self-consistency, not residual magnitude.
- **Hyperbolic/dispersive problems have a finite optimal width** and degrade beyond it (viscous Burgers $2.5\times10^{-10}$ @ $W{=}576 \to 1.3\times10^{-7}$ @ $2304$; not a $\lambda$ artifact, verified on a wider grid). Same $\gamma^r$ conditioning drift as expF01, amplified by the iteration.

## Question

Does the expF01 recipe survive nonlinearity? Specifically: with the geometry frozen, is Newton on the readout alone (one lstsq per step, no training, no learning rate) enough to solve classical nonlinear ODEs/PDEs, and where does it break?

## Experiment design

Identical geometry, dictionary, collocation, sweep, and metrics to expF01 (see that writeup). The only change is the solver. Write the residual as

$$r(a) \;=\; \underbrace{\textstyle\sum_i s_i\,(D_i\Phi)\,a}_{\text{linear part}} \;+\; N\big(D u_a\big) \;-\; f,$$

where $N$ is a pointwise smooth nonlinearity of derivative fields of $u$. Since the geometry is frozen, this is a nonlinear least-squares problem in the readout $a$ only. Each Gauss-Newton step linearizes the PDE around the current iterate,

$$J_k \;=\; \sum_i s_i (D_i\Phi) \;+\; \sum_{\text{fields}} \mathrm{diag}\!\big(\partial N/\partial(D_\ell u_k)\big)\,(D_\ell\Phi),$$

which is exactly a **variable-coefficient linear PDE for the correction** (coefficients read off the current iterate at the collocation points), then solves $\min_\delta \|J_k\delta + r_k\|_2$ by the same min-norm lstsq (`rcond=1e-13`, same block scaling and condition weights). Backtracking halves the step if the stacked residual grows. Condition rows are linear and unchanged across steps.

Newton starts from $a_0 = 0$ except where a classical initializer exists: the eikonal starts from the harmonic extension of its boundary data, and Monge-Ampere from the standard $\Delta u = 2\sqrt{f}$ pre-solve. Each is one extra linear solve.

The nine problems (exact solutions manufacture $f$ and the condition values; $f\equiv 0$ for logistic, viscous Burgers, and KdV, which are classical solutions):

| category | order | problem | conditions |
|---|---|---|---|
| 1D ODE | 1 | logistic $u' = 3u(1-u)$ | IVP $u(-1)$ |
| 1D ODE | 2 | Bratu $u'' + e^u = f$ (combustion) | Dirichlet both ends |
| 1D ODE | 3 | Blasius $u''' + \tfrac12 u u'' = f$ (boundary layer) | $u(-1), u'(-1), u(1)$ |
| 2D steady | 1 | eikonal $|\nabla u|^2 = f$ (fully nonlinear) | Dirichlet on the **inflow arc** |
| 2D steady | 2 | Monge-Ampere $u_{xx}u_{yy} - u_{xy}^2 = f$ (fully nonlinear, convex) | Dirichlet on the circle |
| 2D steady | 3 | $u_{xxx} + u_{yyy} + uu_x + u = f$ (stress test) | Cauchy on the circle |
| space-time | 1 | inviscid Burgers $u_t + uu_x = f$ | IC + inflow |
| space-time | 2 | viscous Burgers $u_t + uu_x = 0.25\,u_{xx}$ (Cole-Hopf front) | IC + Dirichlet both ends |
| space-time | 3 | KdV soliton $u_t + 6uu_x + u_{xxx} = 0$ | IC + 2 right / 1 left |

Condition placement follows expF01's lesson: the KdV's $u_{xxx}$ gives leftward group velocity, so two conditions sit on the right edge; the eikonal's linearization is transport, so its Dirichlet data sits on the inflow arc only (both discussed below).

**Code & data.** `experiments/expF02_nonlinear_de_zoo/` (`run.py`, `problems.py`). Data: `data.json` (per-cell errors and Newton-iteration counts). Figures: `error_vs_width.png` (deliverable), `function_representations/{ode1d,pde2d_steady}/order{1,2,3}.png`, `function_representations/pde2d_time/order{1,2,3}.gif`.

## Results

Best cell per problem (rel $L_2$, with Newton steps):

| | order 1 | order 2 | order 3 |
|---|---|---|---|
| **1D ODE** | logistic $1.8\times10^{-16}$ (23 it) | Bratu $2.5\times10^{-16}$ (7 it) | Blasius $2.1\times10^{-16}$ (6 it) |
| **2D steady** | eikonal $4.8\times10^{-10}$ (6 it) | Monge-Ampere $1.2\times10^{-14}$ (5 it) | stress $9.2\times10^{-12}$ (6 it) |
| **space-time** | inviscid Burgers $9.8\times10^{-6}$ (11 it) | viscous Burgers $2.5\times10^{-10}$ (6 it) | KdV $4.6\times10^{-10}$ (6 it) |

- **The 1D ODEs hit machine epsilon**, an order better than their linear counterparts in expF01. All three are classical: logistic growth, Bratu combustion, Blasius boundary-layer similarity.
- **Monge-Ampere reaches the fp64 floor in 5 Newton steps.** A fully nonlinear second-order PDE (the determinant of the Hessian) solved to $1.2\times10^{-14}$ with no training is the strongest single result here.
- **Newton is cheap and width-independent where it works:** 5-7 steps typically, not growing with $W$. The logistic case is the outlier (10-23 steps) because its sigmoid tail at the IC is nearly flat, so early steps make little progress.
- **Elliptic problems reach the floor; hyperbolic/dispersive ones plateau** ($10^{-10}$-ish) and then degrade with width. Full sweep: 7.1 minutes.

### Figures

- **`error_vs_width.png`** (deliverable) -- $3\times3$ grid, rows = operator order, columns = category; rel $L_2$ (solid) and $L_\infty$ (dashed) vs width, log-log, $10^{-13}$ reference line. Read for: the 1D column at machine epsilon, Monge-Ampere at the floor, and the non-monotone hyperbolic curves (bottom-right two panels).
- **`function_representations/ode1d/order{1,2,3}.png`** -- exact vs solved overlay and log-scale error, annotated with Newton-step count.
- **`function_representations/pde2d_steady/order{1,2,3}.png`** -- 3D surface of $u^*$ and $\log_{10}|\hat u - u^*|$ heatmap on the disk. The Monge-Ampere panel is uniform at the floor; the eikonal panel shows error organized along characteristics.
- **`function_representations/pde2d_time/order{1,2,3}.gif`** -- $u(x,t)$ animated with locked axes and the log-scale error profile. Order 2 is the Cole-Hopf front translating; order 3 is the KdV soliton.

## Additional details

**Two failure modes, both announced by a stalling residual.** The nonlinear setting gives a signal the linear setting does not: Newton either drives the residual toward the floor or it visibly sticks, and sticking is diagnostic without knowing $u^*$. (The converse does *not* hold -- expF01 shows a residual sitting *at* the floor on an ill-posed problem whose solution is four orders wrong, with rank and singular spectrum equally blind. A small residual certifies nothing; only self-consistency across widths does.)

*Ill-posed placement (eikonal).* Newton linearizes $|\nabla u|^2 = f$ to $2\nabla u_k\cdot\nabla\delta = -r$, a **first-order transport operator**, which admits data only on the inflow boundary. Prescribing Dirichlet on the entire circle over-determines the correction equation, and the residual stalls visibly:

```
full circle:  2.6e+00  3.3e-01  7.4e-03  1.4e-03  1.4e-03  1.4e-03   <-- stalls
```

Restricting to the inflow arc ($\nabla u\cdot n<0$) gains roughly two orders (rel $L_2$ $\sim10^{-7}\to\sim5\times10^{-10}$). Note that the *nonlinear* problem is consistent with full-circle data (the exact solution satisfies it); it is the *linearized* problem, which Newton actually solves, that is over-determined. This is a nonlinear-specific trap with no analogue in expF01.

*Genuine Newton stagnation (inviscid Burgers).* Here the residual itself sticks at $1.1\times10^{-4}$ (solution error $9.8\times10^{-6}$), so the iteration, not the problem statement, is at fault. The contrast with its viscous sibling is the whole story: adding $0.25\,u_{xx}$ takes the same nonlinearity from $10^{-5}$ to $2.5\times10^{-10}$. This is the classical vanishing-viscosity picture -- a nonlinear hyperbolic operator with no dissipation gives Gauss-Newton no elliptic regularization to work with. Continuation in $\nu\to0$ is the standard remedy and is untested here.

**Finite optimal width, sharpened by nonlinearity.** Viscous Burgers peaks at $W=576$ ($2.5\times10^{-10}$) and degrades to $1.3\times10^{-7}$ by $W=2304$; KdV likewise ($4.6\times10^{-10}\to4.7\times10^{-8}$); the order-3 steady case peaks at $576$ and rises to $10^{-8}$ while its Newton count climbs from 5 to 13. This is not a $\lambda$-window artifact: a wider $\lambda$ grid at the affected widths reproduces the same numbers. It is expF01's $\gamma^r$ roundoff drift, now feeding back into the iteration (a noisier Jacobian needs more steps and converges less far).

**Ablations and baselines (preliminary; expF03 should do this properly).** Everything in expF01's *Ablations* section applies here unchanged, since the geometry, dictionary, weighting, and `rcond=1e-13` are identical -- in particular the random-feature baseline (structured geometry wins 19x--685x on the linear problems) and the finding that a flat $\lambda=0.25$ needs no oracle search. The nonlinear-specific ablations that expF03 should add: Newton globalization (backtracking vs Levenberg-Marquardt vs continuation), the effect of the initializer (zero vs harmonic vs $\Delta u=2\sqrt f$), and whether the per-step `rcond` should tighten as the residual falls. As in expF01, the 2D $\lambda$/best-width selection here uses $u^*$ and is a basin study, not a deployable protocol; nested-width self-consistency is the replacement.

**Caveats.** Single collocation seed. Newton is started from zero (or a classical initializer) with backtracking only; no Levenberg-Marquardt damping or parameter continuation was tried, and both are the obvious next lever for the stagnating cases. Scope is the same as expF01: smooth manufactured problems, low dimension, dense global solve -- feasibility, not competitiveness with established nonlinear solvers.

## Conclusions

*Proposed, pending Sam.* Nonlinearity is not a barrier for the frozen-geometry solve: with the geometry fixed, Newton reduces every problem to a sequence of variable-coefficient linear solves of exactly the expF01 form, and it reaches machine epsilon on classical 1D nonlinear ODEs and the fp64 floor on Monge-Ampere in a handful of steps. What limits the remaining cases is not the representation but the iteration and the problem statement: linearized operators must be given well-posed data (the eikonal's inflow arc), and nonlinear hyperbolic operators without dissipation stagnate where their viscous counterparts do not.

## Open questions

- **Rescue the stagnating hyperbolic cases**: vanishing-viscosity continuation ($\nu\to0$) for inviscid Burgers, and Levenberg-Marquardt damping generally. Does either reach the floor?
- **Why does the eikonal stop at $10^{-9}$** rather than the floor, even with well-posed inflow data? Fully nonlinear first-order may be intrinsically harder, or the inflow arc may need the characteristic-consistent treatment at the tangency points.
- **Finite optimal width**: the a posteriori residual should be able to select it automatically, for both expF01 and expF02.
- Seed-average the 2D cells.
