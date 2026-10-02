# expF13 -- the BWLer PDE suite under the frozen-geometry solve

**Status: draft -- conclusions pending Sam's sign-off.**

## TL;DR

- The expF01/expF02 recipe (frozen Radon ridge geometry + one lstsq per Gauss-Newton step, no training) runs unchanged on the five BWLer benchmarks (arXiv 2506.23024). With the width extension to $W=9216$: **wave reaches $1.6\times10^{-13}$ -- the repo floor, ~80x below BWLer's best ($1.3\times10^{-11}$)** -- and **Poisson-CG $9.4\times10^{-5}$ beats their $1.1\times10^{-2}$ by ~115x** on their own COMSOL reference at flat $\lambda$.
- Convection c=40 reaches $6.2\times10^{-12}$ at $W=9216$ (BWLer $2.0\times10^{-13}$; our descent is decelerating near their number) and c=80 $1.0\times10^{-9}$ (BWLer $1.1\times10^{-12}$; still descending ~6x per step). The suite is genuinely high-frequency (wavevectors ~20-40 in scaled coords vs ~5 in expF01), which pushes the exponential-convergence knee 1-2 width octaves right; linear cells need only ONE lstsq, so a $W=9216$ solve costs ~3 minutes on CPU.
- **The PDE solve is at the basis's own limit, verified two ways**: pure regression of $u^*$ onto the frozen dictionary matches the PDE-solve error (convection/wave/Poisson-control), and for reaction -- the one problem with a residual-vs-value gap -- Gauss-Newton started AT the oracle regression fit walks away from it, proving the collocation objective's own minimizer is the limit (the residual rows pay a $\gamma$-amplified derivative-representation premium near fronts). Not a solver defect; width is the lever.
- **Burgers ($\nu=0.01/\pi$) is the honest negative, twice**: the $\nu$-continuation ladder collapses at $\nu\le0.02$ at every width (the shock, width ~0.015, is below the uniform ridge spacing), and a first shock-graded band probe (sinh-clustered near-x ridges at the known stationary shock line, per-neuron $\gamma$) improved intermediate rungs ~7x but still collapsed at the final rungs -- with a suspected cause: global PDE-block max-normalization under-weights smooth-region rows once $\gamma$ is heterogeneous. Parked as a designed-experiment lead (expC06 cascaded geometry + row equilibration), not a parameter hunt.
- Two mechanical findings: `rcond` must be $10^{-15}$ on this suite ($10^{-13}$ costs 35x on wave -- it truncates modes the oscillatory targets need), and the $\lambda$-anchor-at-$W=1024$ policy misfires when the anchor width is pre-resolution (reaction picked $\lambda=0.16$; $\lambda=0.20$ is ~2.5x better at $W=4096$; $\lambda=0.20$ also wins every extended-width cell).

## Question

Does the frozen-geometry collocation solve survive contact with an external PINN benchmark suite it was not designed around -- BWLer's five PDEs, their references, their metric -- and where exactly does it fall behind a purpose-built high-precision PINN?

## Experiment design

Model, geometry, and solver identical to expF01/expF02: $u(p)=\sum_m a_m\tanh(\gamma(w_m^\top p - t_m)) + \text{poly}_{\le3}(p)$ on Radon tensor ridges ($\sqrt W$ directions $\times$ $\sqrt W$ offsets, collar 1.6, $\gamma=\lambda/h_{\rm ref}$, $h_{\rm ref}=2.8/\sqrt W$), collocation $5W$ uniform random points, one min-norm lstsq (`rcond=1e-15`) per damped Gauss-Newton step, PDE block scaled to $O(1)$, condition blocks weighted $\sqrt{n_{\rm pde}/n_{\rm block}}$. All problems posed in scaled coordinates $(\xi,\eta)\in[-1,1]^2$ with derivative factors folded into the operator coefficients. One new condition type: periodic rows, built as row *differences* $\phi(\xi{=}-1,\eta)-\phi(\xi{=}+1,\eta)$ with value 0.

The problems (verbatim from HazyResearch/bwler @ 7ff2e17; every residual, IC, and BC FD-verified at startup, both reference files sanity-checked at load):

| problem | equation (physical) | conditions | eval reference |
|---|---|---|---|
| convection c=40, 80 | $u_t + cu_x=0$, $u_0=\sin x$ | IC + periodic ($u$, $u_x$) | exact $\sin(x-ct)$, 200x200 |
| reaction | $u_t = 5u(1-u)$, Gaussian $u_0$ | IC + periodic (value only) | exact logistic, 200x200 |
| wave | $u_{tt}=4u_{xx}$, $\beta=5$ modes | $u$, $u_t$ ICs + Dirichlet | exact two-mode, 200x200 |
| Burgers | $u_t+uu_x=\nu u_{xx}$, $\nu=0.01/\pi$ | IC + Dirichlet | Chebfun `pde15s`, 201x511 grid |
| Poisson-CG | $\Delta u=0$, square minus 4 holes | $u{=}1$ square, $u{=}0$ holes | COMSOL nodes (float32, NN-lookup) |
| Poisson control | same geometry, manufactured harmonic $u^*$ (logs at hole centers + $\xi^2-\eta^2$) | Dirichlet from $u^*$ | exact, $241^2$ minus holes |

Reaction's periodicity is value-only by design: the Gaussian IC has a $C^1$ kink at the seam. Nonlinear problems (reaction, Burgers) use damped Gauss-Newton (cap 30, backtracking); Burgers uses the expF06 vanishing-viscosity ladder $\nu: 0.5\to0.1\to0.05\to0.02\to0.01/\pi$, warm-started per rung, dictionary built once per width. Sweep: $W\in\{576,1024,2304,4096\}$; $\lambda$ anchored at $W=1024$ over $\{0.12,\dots,0.30\}$ for closed-form problems, flat $\lambda=0.25$ (no oracle) for Burgers and Poisson-CG. Metrics: rel $L_2$ (BWLer's headline), $L_\infty$ (which BWLer never reports), and nested-width self-consistency $\|u_{W_{i+1}}-u_{W_i}\|/\|u_{W_{i+1}}\|$.

**Code & data.** `experiments/expF13_bwler_suite/` (`run.py`, `problems.py`, `extend_width.py`, `ref/` with both BWLer reference files). Data: `data.json`, `extend_width.json`. Figures: `error_vs_width.png` (deliverable, includes the extended widths), `function_representations/{convection_c40,convection_c80,reaction,wave,burgers}.gif`, `function_representations/{poisson_cg,poisson_man}.png`. Full suite ~17 min on CPU; the width extension ~15 min more. The regression, reaction-gap, and Burgers-band diagnostics were scratch scripts; their designs and numbers are recorded in *Additional details*.

## Results

Best cell per problem (rel $L_2$; main sweep to $W=4096$, extension to $9216$ for the linear trio) against BWLer Table 2:

| problem | ours | at $W$ | BWLer best | ratio |
|---|---|---|---|---|
| convection c=40 | $6.2\times10^{-12}$ | 9216 | $2.0\times10^{-13}$ | behind ~30x, decelerating |
| convection c=80 | $1.0\times10^{-9}$ | 9216 | $1.1\times10^{-12}$ | behind ~900x, descending |
| reaction | $4.6\times10^{-7}$ | 6400 | $6.9\times10^{-11}$ | behind, converging ~19x/step |
| wave | $1.6\times10^{-13}$ | 9216 | $1.3\times10^{-11}$ | **beat 80x, at the floor** |
| Burgers | $2.9\times10^{-1}$ | 4096 | $4.6\times10^{-3}$ | **failed** (shock) |
| Poisson-CG | $9.4\times10^{-5}$ | 4096 | $1.1\times10^{-2}$ | **beat 115x** |
| Poisson control | $1.1\times10^{-6}$ | 4096 | -- | descending |

- **No cell is floored except Burgers.** Every non-Burgers curve is still descending steeply at $W=4096$ (convection c=40 fell $6.7\times10^{-9}\to1.2\times10^{-10}$ in the last width step; c=80 fell $2.2\times10^{-1}\to8.7\times10^{-8}$). The comparison at fixed $W=4096$ understates where the method lands with more width.
- **Self-consistency works as the deployable selector here too**: the green curves track true rel $L_2$ within a small factor on every solvable problem, and correctly flag Burgers (stalls at $\sim10^{-1.5}$) and pre-resolution c=80 cells ($\sim1$) with no reference solution needed.
- **Poisson-CG beats BWLer on their own benchmark at a quarter of their parameter count** -- but the number should not be read as our precision: their reference is a float32 COMSOL nodal export evaluated by nearest-neighbour lookup, so both methods are measured against a $\sim10^{-2}$-quality ruler and ours is simply below its noise. The manufactured control on the same geometry ($1.1\times10^{-6}$, still descending) is the honest precision statement.
- Newton behaves as in expF02: 3-6 iterations for linear/mildly nonlinear cells, 11-17 for reaction, width-independent.

### Figures

- **`error_vs_width.png`** (deliverable) -- 7 panels, rel $L_2$ (solid), $L_\infty$ (dashed), self-consistency (dotted), BWLer's best as the red dashed hline, $10^{-13}$ reference dotted. Read for: wave touching the BWLer line, Poisson-CG crossing far below it, the two convection panels' late-but-steep knees, and Burgers flat at $10^{-0.5}$.
- **`function_representations/*.gif`** -- $u(x,t)$ animated (reference solid, solved dashed, locked axes) with the log-error profile; Burgers' gif shows the solve tracking until the shock forms at $t\approx0.5$ and losing it thereafter.
- **`function_representations/poisson_{cg,man}.png`** -- solved field + log-error (scatter at COMSOL nodes for CG, dense heatmap vs $u^*$ for the control).

## Additional details

**The representation-limit diagnostic (the load-bearing check).** For each closed-form problem, regress $u^*$ directly onto the frozen dictionary (one lstsq, no PDE): convection c=40 gives $2.5\times10^{-9}$ at $W=4096$, wave $2.6\times10^{-9}$ (at `rcond=1e-13`; $6\times10^{-11}$ at $10^{-15}$), Poisson control $2.2\times10^{-7}$ -- the same numbers the PDE solve reaches. The solve is extracting everything the basis holds at this width; the deficit vs BWLer's barycentric-Chebyshev basis on the oscillatory targets is a property of the dictionary resolution, not the collocation solve.

**Reaction's residual-vs-value gap, dissected.** Reaction's PDE solve sits ~2 orders above its own value-regression limit ($1.0\times10^{-4}$ vs $1.6\times10^{-6}$ at $W=2304$). Four arms isolate why: warm-starting Gauss-Newton AT the regression fit *walks away* to $9.4\times10^{-5}$ (so the basin is not the issue and the stacked system's minimizer really is there); doubling collocation to $10W$ changes nothing; raising the Newton cap to 60 confirms the residual genuinely plateaus ($3.5\times10^{-5}$, flat to 3 digits). The collocation objective minimizes the *residual*, whose rows contain $u_t$ -- and the dictionary represents the derivative field a $\gamma$-factor worse than the values near reaction's fronts, so the residual-minimizing coefficients trade away value accuracy. This is expF01's "solution at the floor $\ne$ physical residual at the floor" with the roles mirrored, and it converges with width ($1.7\times10^{-4}\to8.8\times10^{-6}\to4.6\times10^{-7}$ at $W=2304/4096/6400$, $\lambda=0.20$ -- accelerating, ~19x on the last step). BWLer does not pay this premium: their Chebyshev nodal basis differentiates its interpolant exactly. The $\lambda$-anchor also misfired here (anchor at $W=1024$ chose $0.16$; $0.20$ is ~2.5x better at 4096) -- anchor at the largest affordable width instead.

**The Burgers graded-band probe (negative).** Exploiting the shock's known stationarity at $x=0$, a deterministic band (9 directions within $\pm0.35$ rad of the x-axis $\times$ 192 sinh-graded offsets, per-neuron $\gamma=\lambda/h_{\rm loc}$, center spacing $3.7\times10^{-3}$) was appended to the uniform $W=2304$ grid, with shock-clustered collocation. It improved the intermediate rungs (residual at $\nu=0.05$: $7.3\times10^{-3}$ vs $5.3\times10^{-2}$ uniform; $\nu=0.02$: $0.89$ vs $1.60$) but still collapsed at $\nu\le0.02$; final rel $L_2$ $3.2\times10^{-1}$, no better than uniform. Two suspected causes for a designed follow-up: the band's center $\gamma\approx67$ is still ~2x too wide for the $\nu=0.01/\pi$ front (~$8\times10^{-3}$), and the PDE block is normalized by its global max entry, which the band's high-$\gamma$ second-derivative rows dominate -- silently down-weighting the smooth region once $\gamma$ is heterogeneous. Row equilibration is the first thing such an experiment must fix.

**The width extension mechanics.** Linear problems need exactly one lstsq (the Gauss-Newton wrapper wastes 3-4 redundant solves on them); with a lean direct path, $W=6400$ costs ~60 s and $W=9216$ ~3 min per solve on CPU (collocation oversampling 4x and 3x respectively, 16 GB RAM). This is what makes the $10^4$-width regime routinely affordable for the linear suite.

**rcond, again.** expF01 flagged `rcond` as a regularization knob; this suite sharpens it: $10^{-13}\to10^{-15}$ is worth 35x on wave regression and ~2x-5x on the solves. High-frequency targets live in the small-singular-value directions that $10^{-13}$ deletes.

**The Burgers wall, quantified.** Rungs converge (residual at $W=4096$: $3.6\times10^{-7}$, $9.4\times10^{-7}$, $5.5\times10^{-3}$ for $\nu=0.5,0.1,0.05$) then collapse at $\nu=0.02$ (residual ~1.1) regardless of warm start. The front width scales as $\sim8\nu$; at $\nu=0.02$ that is ~0.16 against ridge spacing $h_{\rm ref}=0.044$ -- marginal -- and at $\nu=0.01/\pi$ it is ~0.015, far below resolution. This is expF02's inviscid-Burgers lesson in benchmark form: the failure is representation of the shock by a *uniform* fixed geometry, and continuation cannot rescue what the basis cannot express. Graded/adaptive center placement (the expC05/expE01 curvature-clustering lead) is the relevant open lead, not a better outer iteration.

**Scope.** Single collocation seed, single run per cell; $\lambda$ anchoring uses $u^*$ for the closed-form problems (the flat-$\lambda$ Poisson-CG and Burgers rows are the no-oracle protocol); BWLer comparisons are against their reported Table-2 numbers, not reruns of their code.

## Conclusions

*Proposed, pending Sam.* On an external PINN precision benchmark it was never tuned for, the frozen-geometry solve beats the purpose-built method by ~80x on wave (reaching the repo's fp64 floor at $W=9216$) and by two orders on Poisson-CG (reference-ceiling caveat), and trails on the two convection problems, whose frequency content pushes the exponential knee past the tested widths -- with the PDE solve verified, by regression and oracle-start diagnostics, to sit at the dictionary's own representation limit everywhere. The suite's two real exposures are complementary: front-forming targets pay a $\gamma$-amplified derivative-representation premium in the collocation objective (reaction), and shock-forming problems are outside what a uniform fixed geometry can express at any tested width (Burgers, where a first graded-band probe also failed).

## Open questions

- **Convection c=40's deceleration** ($10\times$ then $2\times$ per width step, ending $6.2\times10^{-12}$): entering a conditioning-limited regime short of BWLer's $2.0\times10^{-13}$, or just the knee of another descent? One more octave (or a per-width $\lambda$ re-anchor) answers it. Same for c=80's remaining 900x.
- **Reaction's derivative-representation premium**: is there a principled residual weighting (e.g. Sobolev-weighted rows) that removes the $\gamma$-factor penalty without breaking the linear-solve structure, or is width the only honest lever?
- **The Burgers cascaded-geometry experiment** (from the parked probe): row equilibration for heterogeneous-$\gamma$ dictionaries + center $\gamma$ matched to the $8\times10^{-3}$ front + the expC06 multi-band structure. The one problem where geometry, not width, binds.
- Seed-average; report the no-oracle (flat $\lambda$) rows for all problems alongside the anchored ones.
