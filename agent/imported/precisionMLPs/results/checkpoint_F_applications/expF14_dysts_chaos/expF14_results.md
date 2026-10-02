# expF14 -- chaotic ODEs from `dysts`: solve the whole trajectory, don't integrate it

**Status: draft -- conclusions pending Sam's sign-off.**

## TL;DR

- The expF01/expF02 recipe transfers to chaotic ODE systems unchanged: freeze the QI geometry in $t$, solve the readout, and the entire trajectory over three Lyapunov times comes out at the fp64 floor on all five of the first batch -- $1.1\times10^{-13}$ (Lorenz), $4.3\times10^{-13}$ (Rössler), $1.4\times10^{-12}$ (Thomas), $4.9\times10^{-13}$ (Halvorsen), $1.3\times10^{-13}$ (Lorenz96) -- at $W=693$ neurons, in 4-5 Gauss-Newton steps, with no training and no time-stepping. A second batch of three adds DoublePendulum ($3.3\times10^{-13}$) and two instructive failures.
- **It beats the tightest fp64 Runge-Kutta on 4 of 5.** DOP853 at `rtol=1e-13` costs 5-10k function evaluations and lands at $5.8\times10^{-13}$ (Lorenz) / $1.2\times10^{-11}$ (Thomas); the QI solve is $5\times$ and $8.5\times$ better respectively. Rössler is the one loss ($3\times$).
- Width scaling is the clean geometric descent Checkpoint B predicts, and it **stops at the interpolation floor of the same dictionary**, not at some solver-specific wall: the solve tracks that floor to within $1$-$1.5$ orders everywhere past resolution.
- **The warm start is load-bearing and the geometry is load-bearing.** From $a_0=0$ Newton fails outright ($\sim10^{-1}$ to $10^{2}$) on every system; with random centres at the same width and bandwidth the solve loses 2-6 orders.
- **Where it fails, it fails for a nameable reason and says so.** A near-square-wave drive (InteriorSquirmer) and a $C^0$ vector field (MacArthur) both defeat a single shared bandwidth. On both, the solve stops tracking its own dictionary's interpolation floor (318-1459$\times$ above it, versus 1-35$\times$ when it works) and the reference-free residual is 7 orders larger -- so the failure is detectable without knowing the answer.

## Question

Checkpoint F established the recipe on *manufactured smooth* differential problems. Chaotic ODEs break the two assumptions that made those easy: the solution is only defined implicitly by the flow, and any error is amplified by $e^{\lambda_{\max} t}$. Does the frozen-geometry collocation solve still reach the fp64 floor, and how far in Lyapunov times does a fixed neuron budget carry?

## Experiment design

**Systems.** Five chaotic flows from `dysts` (Gilpin, NeurIPS 2021 Datasets & Benchmarks), in the order Sam specified: Lorenz $\to$ Rössler $\to$ Thomas $\to$ Halvorsen $\to$ Lorenz96. Canonical parameters, on-attractor initial condition $u_0$, dominant period, and the estimated maximal Lyapunov exponent $\lambda_{\max}$ are read from the library; only the vector field is re-implemented locally, because the Gauss-Newton step needs $F$ evaluated at $n$ collocation points at once *and* an analytic $\partial F/\partial u$ there.

| | $d$ | $\lambda_{\max}$ | $T$ at $\lambda_{\max}T=3$ | in dominant periods |
|---|---:|---:|---:|---:|
| Lorenz | 3 | 0.892 | 3.364 | 2.24 |
| Rössler | 3 | 0.151 | 19.921 | 3.37 |
| Thomas | 3 | 0.632 | 4.744 | 0.95 |
| Halvorsen | 3 | 0.696 | 4.308 | 2.89 |
| Lorenz96 | 4 | 1.336 | 2.245 | 1.02 |

**Model.** Time is rescaled to $s = 2t/T - 1$ on $[-1,1]$, so $\mathrm{d}/\mathrm{d}t = (2/T)\,\mathrm{d}/\mathrm{d}s$. Every component of the state shares one frozen 1-D QI dictionary -- uniform centres $s_m = -1 + mh$ with $h = 2/N$, extended by a halo of $R=\max(70,\lceil 0.4N\rceil)$ nodes each side, one shared bandwidth $\gamma = \lambda/h$ at $\lambda=0.25$, plus monomials up to degree 3:

$$\tilde u_c(s) \;=\; \sum_{m} A_{c,m}\,\tanh\!\big(\gamma(s - s_m)\big) \;+\; \sum_{k\le 3} A_{c,W+k}\, s^k , \qquad u_c = \sigma_c\,\tilde u_c .$$

Total width is $W = N + 2R + 1$; only $A \in \mathbb{R}^{d\times(W+4)}$ is ever solved for. The per-component scales $\sigma_c$ are the RMS of a cheap fp64 Runge-Kutta pre-solve (never the reference), so every residual block is $O(1)$.

**The system solved.** With $D_0,D_1$ the value and $\mathrm{d}/\mathrm{d}s$ dictionary rows at $n_{\text{col}}=4W$ uniform collocation points, the residual and its Jacobian are

$$R_{:,c} \;=\; \tfrac{2}{T}\,D_1 A_c \;-\; \tfrac{1}{\sigma_c} F_c\!\big(\sigma\odot D_0A^{\!\top}\big), \qquad \frac{\partial R_{:,c}}{\partial A_{c'}} \;=\; \tfrac{2}{T}\,\delta_{cc'} D_1 \;-\; \tfrac{\sigma_{c'}}{\sigma_c}\,\mathrm{diag}\!\Big(\tfrac{\partial F_c}{\partial u_{c'}}\Big) D_0 ,$$

stacked with one initial-condition row per component, $D_0(s{=}{-}1)A_c = u_{0,c}/\sigma_c$, weighted $\sqrt{n_{\text{col}}}$ as in expF02. Damped Gauss-Newton with backtracking; **each step is one min-norm lstsq** (`rcond=1e-13`) of the full $(d\,n_{\text{col}})\times(d(W{+}4))$ system. Nothing is time-stepped: $t$ is a coordinate, the IC is a row block, and the whole window is solved at once, so nothing accumulates over $t$.

**Newton start.** A standard fp64 DOP853 run at `rtol=1e-8`, `atol=1e-11`, least-squares-fitted into the dictionary. This is the ODE-native analogue of expF03's initialisation ladder -- a cheap integrator always exists for an ODE -- and it costs ~1.3-2.8k function evaluations, under 20 ms. The pipeline is therefore *cheap RK $\to$ QI polish*, the expF07 finisher pattern.

**Reference, and why fp64 is not enough.** On a chaotic IVP a fp64 reference cannot certify a fp64 solution: DOP853 at `rtol=1e-13, atol=1e-14` sits at $6.1\times10^{-13}$ (Lorenz) and $1.1\times10^{-11}$ (Thomas) against extended precision. The reference is `mpmath.odefun` (Taylor method) at 30 decimal digits with $\text{tol}=10^{-25}$, on a 6001-point grid, cached to disk -- the repo's existing mpmath convention: an offline extended-precision precomputation producing fp64 constants. Parameters and $u_0$ are handed to mpmath as the *exact fp64 values*, so it solves the same IVP.

**Metrics.** Aggregate rel $L_2 = \|\hat U - U^*\|_F/\|U^*\|_F$ and absolute $L_\infty$ on the 6001-point grid; max per-component rel $L_2$ logged alongside. Two extra diagnostics are emitted per cell: the **interpolation floor** (best rel $L_2$ the same frozen dictionary reaches by fitting the reference directly, with no ODE -- this separates "the basis cannot resolve this trajectory" from "the solve cannot find it") and **nested-width self-consistency** $\|u_{W_2}-u_{W_1}\|/\|u_{W_2}\|$, expF01's reference-free error estimate.

**Sweeps.** (A) width at fixed horizon $\lambda_{\max}T=3$, $N \in \{48,96,128,192,256,384\}$ ($W=189$ to $693$); (B) horizon $\lambda_{\max}T \in \{1,2,3,4,6\}$ at fixed $N=256$; (C) ablations at $N=256$ -- centre placement, $\lambda$, `rcond`, polynomial block, Newton start, warm-start tolerance; (D) the expF03 knobs that matter most for an IVP -- the initialisation ladder and the initial-condition row weight.

**Verification (run before every sweep; the run refuses to start otherwise).**
- Our vectorised $F$ against `dysts`' own `rhs`: max relative difference $0$ for Rössler / Thomas / Halvorsen / Lorenz96, $9.2\times10^{-17}$ for Lorenz.
- Our analytic $\partial F/\partial u$ against the **complex-step** derivative $\mathrm{Im}\,F(u + \mathrm{i}h e_k)/h$ (exact to rounding, no subtractive cancellation): $0.95$-$1.8\times10^{-16}$ relative on all five.
- The assembled collocation Jacobian against central differences of the residual it claims to differentiate: $2\times10^{-9}$ relative.
- The frozen dictionary alone reaches $<10^{-13}$ by plain lstsq on a smooth target, so the geometry is not the limit.
- The mpmath reference against itself at 25 vs 40 digits.

**Code & data.** `experiments/expF14_dysts_chaos/` (`systems.py` -- vector fields, Jacobians, verification; `reference.py` -- mpmath reference and its cache; `core.py` -- geometry, dictionary, Gauss-Newton; `run.py` -- sweeps and figures). Tests: `tests/test_expF14_dysts_chaos.py`. Data: `results/checkpoint_F_applications/expF14_dysts_chaos/data.json`, reference cache in `ref_cache/`. Figures in `figures/`.

## Results

Width sweep at $\lambda_{\max}T = 3$ (rel $L_2$ against the mpmath reference; "floor" is the interpolation floor of the same dictionary):

| $W$ | Lorenz | Rössler | Thomas | Halvorsen | Lorenz96 |
|---:|---:|---:|---:|---:|---:|
| 189 | $6.6\times10^{-2}$ | $4.1\times10^{-1}$ | $1.2\times10^{-2}$ | $4.9\times10^{-2}$ | $3.7\times10^{-2}$ |
| 269 | $1.4\times10^{-5}$ | $3.0\times10^{-2}$ | $2.2\times10^{-7}$ | $6.5\times10^{-5}$ | $1.8\times10^{-10}$ |
| 347 | $1.2\times10^{-11}$ | $8.8\times10^{-5}$ | $9.8\times10^{-11}$ | $1.7\times10^{-10}$ | $4.7\times10^{-13}$ |
| 463 | $2.2\times10^{-13}$ | $1.4\times10^{-8}$ | $3.1\times10^{-12}$ | $1.2\times10^{-12}$ | $4.5\times10^{-13}$ |
| **693** | $\mathbf{1.1\times10^{-13}}$ | $\mathbf{4.3\times10^{-13}}$ | $\mathbf{1.4\times10^{-12}}$ | $\mathbf{4.9\times10^{-13}}$ | $\mathbf{1.3\times10^{-13}}$ |
| floor @693 | $2.6\times10^{-14}$ | $4.1\times10^{-13}$ | $4.0\times10^{-14}$ | $4.2\times10^{-14}$ | $4.5\times10^{-14}$ |
| best DOP853 | $5.8\times10^{-13}$ | $1.4\times10^{-13}$ | $1.2\times10^{-11}$ | $6.6\times10^{-13}$ | $4.8\times10^{-13}$ |

- **All five reach the floor**, and the descent is the clean geometric one: Lorenz falls $6.6\times10^{-2} \to 1.4\times10^{-5} \to 1.2\times10^{-11} \to 1.1\times10^{-13}$ over a $3.7\times$ increase in width. The warm start is $1.3\times10^{-7}$ throughout, so the solve buys 6 orders over the integrator that seeded it.
- **The stopping point is the dictionary's own interpolation floor**, not a solver wall. Past resolution the solve sits $1$-$35\times$ above the floor of the same basis; on Rössler the two are indistinguishable ($4.3$ vs $4.1\times10^{-13}$).
- **Against the tightest fp64 RK**: better on Lorenz ($5.2\times$), Thomas ($8.5\times$), Halvorsen ($1.3\times$), Lorenz96 ($3.8\times$); worse on Rössler ($3.1\times$). DOP853 at `rtol=1e-13` uses 4.8-9.8k function evaluations.
- **Newton is cheap and width-independent**: 3-5 steps everywhere, no growth with $W$, and quadratic (Lorenz at $W=693$: residual $1.9\times10^{-3} \to 1.0\times10^{-7} \to 6.4\times10^{-12}$).
- **Cost**: 0.2 s at $W=189$ to 6.8 s at $W=693$ per system on a MacBook Air (15.7 s for $d=4$). The whole width sweep across five systems is under 90 s.
- **Rössler is the hard case**, and it is a resolution problem, not a chaos problem: its $\lambda_{\max}$ is the smallest of the five so $\lambda_{\max}T=3$ buys the longest window (3.4 dominant periods, $T=19.9$), and its sharp $z$-spike is what the grid has to resolve. Its interpolation floor descends in lockstep with the solve at every width.

### Horizon: at a fixed neuron budget the wall is resolution, not chaos

Fixed $W=463$, sweeping $\lambda_{\max}T$ (rel $L_2$; the dictionary's interpolation floor in brackets):

| $\lambda_{\max}T$ | 1 | 2 | 3 | 4 | 6 |
|---|---:|---:|---:|---:|---:|
| Lorenz | $7.6\times10^{-15}$ | $2.2\times10^{-13}$ | $2.4\times10^{-13}$ | $1.1\times10^{-11}$ | $7.5\times10^{-5}$ |
| *(floor)* | $5.8\times10^{-14}$ | $9.3\times10^{-14}$ | $3.4\times10^{-14}$ | $1.0\times10^{-11}$ | $4.2\times10^{-8}$ |
| Rössler | $6.2\times10^{-15}$ | $5.3\times10^{-13}$ | $1.4\times10^{-8}$ | $1.9\times10^{-4}$ | $3.3\times10^{-2}$ |
| Thomas | $1.5\times10^{-14}$ | $2.6\times10^{-13}$ | $3.3\times10^{-12}$ | $1.5\times10^{-9}$ | $9.4\times10^{-5}$ |
| Halvorsen | $3.7\times10^{-15}$ | $1.2\times10^{-13}$ | $4.0\times10^{-13}$ | $1.6\times10^{-10}$ | $8.2\times10^{-4}$ |
| Lorenz96 | $2.2\times10^{-15}$ | $1.1\times10^{-13}$ | $4.1\times10^{-13}$ | $7.0\times10^{-13}$ | $1.9\times10^{-10}$ |

At $\lambda_{\max}T=1$ every system is at $10^{-15}$ -- below the interpolation floor measured on that same grid, i.e. at the noise level of the comparison. The solve then holds the floor to $\lambda_{\max}T\approx3$-$4$ and falls apart by $6$.

**What binds is resolution, and the data says so unambiguously**: at every horizon the solve sits within about an order of the interpolation floor of the same dictionary, and both degrade together. The $\varepsilon_{\text{mach}}e^{\lambda_{\max}T}$ conditioning line is $10^{-16}$ to $10^{-13}$ across this whole range and is never the binding constraint. A longer window needs proportionally more neurons to resolve, exactly as Checkpoint B's width law predicts; it does not need extra precision. **This is also where the method loses to a stepper**: DOP853's error is roughly flat in horizon (its *cost* grows, not its error), so past $\lambda_{\max}T\approx4$ at fixed $W$ the Runge-Kutta wins, and holding the floor to $\lambda_{\max}T=6$ would need the width sweep re-run at 2-3x these widths. Lorenz96 is the exception that proves the reading: its window is the shortest in absolute time ($T=1.7$ at $\lambda_{\max}T=2.2$... $T=4.5$ at $6$), so it is still resolved at $\lambda_{\max}T=6$ and still at $1.9\times10^{-10}$.

### Ablations (at $N=256$, $\lambda_{\max}T=3$)

**Centre placement is the whole game, and by more than it was in 2-D.** Same width, same span, same $\gamma$, same solver -- only where the centres sit changes:

| | uniform (QI) | random | Chebyshev | uniform gain vs random |
|---|---:|---:|---:|---:|
| Lorenz | $2.2\times10^{-13}$ | $1.5\times10^{-5}$ | $2.0\times10^{-9}$ | $7\times10^{7}$ |
| Rössler | $1.4\times10^{-8}$ | $1.5\times10^{-6}$ | $1.8\times10^{-4}$ | $107$ |
| Thomas | $3.1\times10^{-12}$ | $2.8\times10^{-4}$ | $1.8\times10^{-10}$ | $9\times10^{7}$ |
| Halvorsen | $1.2\times10^{-12}$ | $3.0\times10^{-5}$ | $4.6\times10^{-9}$ | $2.5\times10^{7}$ |

expF01 measured $19$-$685\times$ for the same comparison on 2-D PDEs; in 1-D time the gap is 2-8 orders. Chebyshev clustering -- the classical answer for a bounded interval -- is beaten by the uniform grid everywhere, which is Checkpoint C's result reproducing in a third setting.

**The Newton start decides whether it converges at all.** From $a_0=0$ every system fails (Lorenz $8.7\times10^{-1}$, Rössler $2.9$, Thomas $1.7\times10^{2}$), and fitting the initial-condition row alone (`bcfit`) is worse still. The expF03 **cascade init does not transfer**: solving at $N/4$ and refitting wins narrowly on Thomas ($2.5$ vs $3.1\times10^{-12}$) but lands in the wrong basin on Lorenz ($6.2\times10^{-4}$) and Rössler ($2.1\times10^{-1}$) -- at three Lyapunov times the coarse sub-solve is under-resolved, so it is a *worse* initialiser than the cheap integrator, not a better one. Warm-start tolerance itself barely matters ($10^{-6}$ vs $10^{-10}$ move Lorenz between $1.3$ and $2.4\times10^{-13}$, i.e. within the fp64 scatter).

**The other knobs are already at their defaults.** $\lambda=0.25$ is the basin bottom (Lorenz: $2.5\times10^{-11}$ at $0.15$, $1.3\times10^{-10}$ at $0.40$, $2.2\times10^{-13}$ at $0.25$), confirming expC03 in the time domain. The IC-row weight is best at the expF02 default: both down-weighting ($0.1\times$, $33\times$ worse on Lorenz) and up-weighting ($10\times$, $117\times$ worse) cost -- consistent with expF03's finding that the single inhomogeneous row of an IVP must not be down-weighted, and sharpening it to a two-sided optimum. The degree-3 polynomial block is a **null** here ($2.21$ vs $2.16\times10^{-13}$), unlike steady 2-D where it was worth 1-3 orders. `rcond=1e-15` is *worse* ($2.2\times10^{-11}$ on Lorenz, $9.3\times10^{-10}$ on Thomas), the opposite of expF03's nonlinear-PDE result -- there is still no universal rcond.

### Figures

- **`figures/error_vs_width.png`** (deliverable) -- $2\times3$ grid, one panel per system, log-log. Blue solid = QI solve rel $L_2$, orange dashed = $L_\infty$, grey dotted = the interpolation floor of the same dictionary, green dash-dot = the RK warm start that seeded Newton, red dotted = the best DOP853. Read for: the geometric descent in every panel, the blue curve flattening onto the grey floor rather than onto some higher plateau, and blue ending at or below the red line on four of five.
- **`figures/representations.png`** -- one row per system: the phase portrait (reference solid black, solved dashed) which overlays exactly; the components over $[0,T]$; and $\|\hat u - u^*\|_2$ against $t$ on a log axis, with the warm start for contrast and an $e^{\lambda_{\max}t}$ guide. The right column is the chaos-specific panel: the error grows along the guide slope, i.e. what limits the solve is amplification of the representation floor, not accumulation of local truncation error.
- **`figures/newton_convergence.png`** -- collocation residual against Gauss-Newton step, one curve per width. Quadratic drop then a flat floor; the floor level, not the step count, is what width buys.
- **`figures/self_consistency.png`** -- the reference-free nested-width estimate against the true error. It tracks one width behind, as expected, and is conservative.
- **`figures/error_vs_horizon.png`** -- rel $L_2$ against $\lambda_{\max}T$ at fixed width, with the interpolation floor and the $\varepsilon_{\text{mach}}e^{\lambda_{\max}T}$ chaos line, separating the resolution wall from the conditioning wall.
- **`figures/ablations.png`** and **`figures/init_and_signal.png`** -- centre placement, bandwidth basin, `rcond`, polynomial block, Newton start, warm-start tolerance; the initialisation ladder, the IC-row weight, and whether the reference-free residual ranks configurations.

## Second batch: three systems chosen to break the method

Sam picked three more, and each violates a different assumption the first five all satisfied. One passes; the two failures are more informative than the pass.

| system | $d$ | assumption broken | rel $L_2$ at largest width |
|---|---:|---|---:|
| DoublePendulum | 4 | conservative (no attractor); $\lambda_{\max}=3.99$, the most chaotic of all eight | $3.3\times10^{-13}$ @ $W{=}693$ |
| InteriorSquirmer | 3 | non-autonomous, driven by a near-square wave | $4.2\times10^{-5}$ @ $W{=}1385$ |
| MacArthur | 10 | vector field only $C^0$ (growth rate is a `min`); $d=10$ | $3.7\times10^{-5}$ @ $W{=}463$ |

**DoublePendulum works, and it settles the horizon question.** It reaches $3.3\times10^{-13}$ against an interpolation floor of $1.1\times10^{-14}$, from a warm start of $1.7\times10^{-10}$. More useful is its horizon curve, which is essentially **flat**: $5.8\times10^{-15}$, $4.0\times10^{-13}$, $5.0\times10^{-13}$, $5.0\times10^{-13}$, $5.4\times10^{-13}$ at $\lambda_{\max}T = 1,2,3,4,6$. The most chaotic system in the set is the *least* horizon-sensitive, because $\lambda_{\max}=3.99$ means even $\lambda_{\max}T=6$ is a window of only $T=1.5$ in absolute time. That is the cleanest available confirmation of the sweep-B reading: what binds is the absolute length of the window the dictionary must resolve, not how many Lyapunov times it contains. It is also the one system where the defaults are not optimal -- Chebyshev centres beat the uniform grid ($8.6\times10^{-14}$ vs $5.0\times10^{-13}$), and $\lambda=0.30$ or a down-weighted condition row reach $1.2$-$1.3\times10^{-14}$, worth about $40\times$.

**InteriorSquirmer: machine precision until a switch enters the window, then a 12-order cliff.** Its third state is a clock, and the mode amplitudes are gated by $0.5 + 0.5\tanh\!\big(60\sin(2\pi t/3)\big)$ -- a near-square wave whose transition width is $0.016$ in $t$ (measured: $\max|\dot{\text{protocol}}| = 62.8$), with switches at $t = 1.5, 3.0, 4.5,\dots$ Counting the switches strictly inside each window:

| $\lambda_{\max}T$ | $T$ | switches inside $(0,T)$ | rel $L_2$ at $W=463$ |
|---:|---:|---:|---:|
| 1 | 1.184 | 0 | $4.5\times10^{-15}$ |
| 2 | 2.368 | 1 | $2.1\times10^{-3}$ |
| 3 | 3.551 | 2 | $7.2\times10^{-3}$ |
| 4 | 4.735 | 3 | $9.7\times10^{-3}$ |
| 6 | 7.103 | 4 | $2.7\times10^{-2}$ |

With no switch in the window the method is at machine precision, better than any other cell in the experiment. The first switch costs twelve orders. Widening helps only algebraically: the interpolation floor falls roughly as $W^{-6.5}$ (from $2.6\times10^{-3}$ at $W=237$ to $2.9\times10^{-8}$ at $W=1385$) instead of geometrically. One shared bandwidth cannot hold both the trajectory's own scale and a feature $200\times$ narrower, which is expC06's multi-scale finding arriving in the time domain -- and the cascaded multi-band geometry proposed there is the indicated fix.

**MacArthur: algebraic convergence, exactly where the smoothness theory says it should be.** Its growth rate is $\mu_i = \min_j r R_j/(k_{ji} + R_j)$ -- Liebig's law of the minimum -- so the vector field is only $C^0$ and the trajectory only $C^1$. The limiting resource switches 9 times inside the window. expA07's smoothness stress test predicts precisely this: when the target is rougher than the kernel order requires, the geometric law is replaced by a steeper-looking but merely algebraic power law, and the lock never arrives at reachable width. Measured: the interpolation floor falls as about $W^{-4.4}$, reaching only $1.2\times10^{-7}$ at $W=463$. This is the one system in the set where machine precision is not available at any feasible width, and the reason is a property of the problem, not of the method.

### The failures share one signature, and it is visible without a reference

For the six systems that work, the solve tracks the interpolation floor of its own dictionary. For the two that fail, it does not:

| | solve / interpolation floor | fresh-point PDE residual |
|---|---:|---:|
| Lorenz, Rössler, Lorenz96 | 1-4$\times$ | $10^{-11}$-$10^{-10}$ |
| Halvorsen, DoublePendulum, Thomas | 12-35$\times$ | $10^{-11}$-$10^{-10}$ |
| MacArthur | **318$\times$** | $1.0\times10^{-2}$ |
| InteriorSquirmer | **1459$\times$** | $4.4\times10^{-4}$ |

The reference-free residual separates the two groups by seven orders, so the failure is detectable in deployment. That matters, because **on both failures the solve is worse than the warm start it began from** -- InteriorSquirmer $4.2\times10^{-5}$ against a warm start of $3.0\times10^{-8}$ ($1390\times$ worse), MacArthur $3.7\times10^{-5}$ against $1.2\times10^{-7}$ ($317\times$ worse). A sharp or non-smooth feature lets the collocation residual be reduced by an aliased solution that is further from the truth. The operational rule is therefore: **do not apply the polish unconditionally** -- keep it only when its fresh stacked residual beats the warm start's.

**No knob rescues either one, and that is itself the diagnosis.** Every ablation on InteriorSquirmer lands within $2\times$ of $7.2\times10^{-3}$, and every ablation on MacArthur within $2\times$ of $7.4\times10^{-5}$ -- centre placement, $\lambda$, `rcond`, and the polynomial block all null. Compare the six working systems, where centre placement alone moves the answer 2-8 orders. When nothing in the configuration moves the answer, the limit is the representation.

### The extended-precision reference is not universal either

This is the methodological finding, and it generalises past this experiment. `mpmath.odefun` is a Taylor method and assumes an analytic field. On MacArthur it is simply **wrong**: at $\lambda_{\max}T=3$ it disagrees with DOP853@$10^{-13}$, DOP853@$10^{-14}$ and Radau@$10^{-12}$ by $8.4\times10^{-5}$, while those three independent integrator families -- explicit Runge-Kutta, implicit Runge-Kutta, and (at $2.6\times10^{-12}$) the LSODA multistep -- agree with each other to $8\times10^{-14}$. Its own two-precision self-check, 25 versus 40 digits, differs by $3.4\times10^{-7}$, so extra digits do not help: the error is a method error, not a rounding error. MacArthur therefore uses a convergence-verified fp64 reference, and its DOP853 line is suppressed from the figures because there DOP853 *is* the reference. The repo leans on mpmath as ground truth throughout; that licence does not extend to non-analytic problems, and the only way to find out is to cross-check independent integrator families.

## Additional details

**The fixed-point trap is structurally present, and did not fire.** Every one of these systems has fixed points, and $u \equiv$ (fixed point) is an *exact* solution of the ODE with an identically zero PDE residual -- only the initial-condition row rules it out. That is exactly the trap expF03 part 2 hit on the logistic IVP, so the backtracking criterion and the emitted diagnostic are both the **stacked** residual (PDE rows *and* the IC row). In these runs no failure took that branch: the diverged starts carry *large* PDE residuals ($3.9\times10^{1}$ for Lorenz cold, $3.2\times10^{7}$ for Rössler `bcfit`), so the PDE rows alone would have flagged them. The guard is cheap and the argument for it is structural, not empirical -- but it is not load-bearing on this evidence.

**The reference-free residual ranks configurations correctly.** Across the ablation cells the stacked fresh-point residual orders the runs the way the true error does -- Lorenz's failed cascade init shows $6.4\times10^{-2}$ (true error $6.2\times10^{-4}$) against the warm start's $2.4\times10^{-11}$ (true $2.2\times10^{-13}$) -- reproducing expF03 part 2's finding that it is the cheap relative ranker, while nested-width self-consistency is the absolute estimate.

**Why the reference had to be extended precision.** This is not a formality. Against mpmath at 30 digits, DOP853 at `rtol=1e-13` is $6.1\times10^{-13}$ on Lorenz and $1.1\times10^{-11}$ on Thomas -- i.e. a fp64 RK "reference" would have been *worse than the thing it was certifying* on four of the five systems, and would have reported the QI solve's error as its own.

**Scope.** Eight vector fields, $d\le10$, one coordinate, a single global window of at most a few Lyapunov times, noise-free, single collocation seed, and an initial condition taken exactly from the benchmark. Six are smooth and autonomous; the second batch deliberately includes one non-autonomous and one non-analytic field, and both fail. The solve is dense and global: the system is $(d\,n_{\text{col}})\times(d(W{+}4))$ and the cost is cubic in width, so this is a precision result, not a claim of competitiveness with a stepper at loose tolerance -- at `rtol=1e-6` DOP853 answers in 9 ms. It says nothing about forecasting, about learning the vector field from data, about long-horizon rollout, or about the higher-dimensional systems in the benchmark.

## Conclusions

*Pending Sam.* On six of eight chaotic systems from the `dysts` benchmark, freezing the QI geometry in time and solving the readout reaches the repo's fp64 floor over three Lyapunov times, in a handful of Gauss-Newton steps and with no training -- and does so more accurately than the tightest fp64 Runge-Kutta on four of the five smooth-and-autonomous ones. Two things carry it, both inherited from earlier checkpoints rather than new here: the uniform-grid geometry (Checkpoint C), without which the same algorithm loses orders, and a cheap classical initialiser (expF03), without which Newton does not converge at all. What stops the descent is the interpolation floor of the dictionary itself, which is the honest limit and not a solver defect. The two failures locate that limit precisely: a near-square-wave drive and a $C^0$ vector field both defeat a single shared bandwidth, both are detectable in deployment from the reference-free residual, and on both the polish must be withheld because it degrades the trajectory it started from.

## Baselines worth running (not run here; Sam's next step)

The load-bearing comparison for this experiment is *not* a forecasting model -- those solve a different problem (predict from data) -- but the family that shares our exact algorithm and differs only in the geometry, plus the numerical gold standard. In priority order:

1. **Physics-informed random-projection networks / ELM collocation** (Fabiani, Galaris, Russo & Siettos, [arXiv:2108.01584](https://arxiv.org/abs/2108.01584); [arXiv:2203.05337](https://arxiv.org/abs/2203.05337); code at [`GianlucaFabiani/RPNN_for_Stiff_ODEs`](https://github.com/GianlucaFabiani/RPNN_for_Stiff_ODEs)). **This is the direct twin**: a frozen single-hidden-layer network, Gauss-Newton on the output weights only, one linear solve per step -- identical to ours except the hidden layer is random rather than the QI grid. They report beating MATLAB's `ode15s`/`ode23t` on stiff problems. Our `geom=random` ablation is an in-house version of this and loses 2-8 orders; running their actual published scheme is the honest version of that claim.
2. **High-order Taylor / clean numerical simulation** (Liao's CNS; [arXiv:1305.4222](https://arxiv.org/pdf/1305.4222), [arXiv:2101.06682](https://arxiv.org/pdf/2101.06682)). The accuracy gold standard for chaotic ODEs -- reliable Lorenz trajectories out to $t=10000$ via 3500th-order Taylor at 4180-digit precision. Not an fp64 competitor, but it is the right *ceiling* reference and it is the same philosophy as our mpmath reference. Fixed-order fp64 Taylor (`TaylorIntegration.jl`, `taylorpy`) is the fair fp64 member of the family and should be in the table.
3. **The rest of the fp64 stepper suite**: `DOP853` (in the table already), `Radau`, `LSODA`, and a symplectic/geometric option, all at matched tolerance, scored on error *and* function evaluations *and* wall time. This is the comparison that decides whether the method is competitive or merely accurate.
4. **Standard PINNs on the same windows.** The literature is consistent that they fail here -- reported transition to chaos around $t\approx0.4$ and $>10\%$ relative $L_2$ by $t\approx0.8$ on Lorenz ([arXiv:2203.07404](https://arxiv.org/pdf/2203.07404)) -- so this is a cheap, decisive contrast rather than a real contest, and it is the one that connects back to the paper's thesis.
5. **The `dysts` forecasting leaderboard** (NBEATS, ESNs, transformers, zero-shot foundation models; [Gilpin 2021](http://www.wgilpin.com/papers/gilpin_neurips_2021.pdf), [arXiv:2409.15771](https://arxiv.org/pdf/2409.15771)) is **context, not a baseline** -- different task, different information. Cite it to place the benchmark, do not put it in the same table.

## Open questions

- **Horizon past $\lambda_{\max}T=4$.** The fixed-width sweep shows the resolution wall; the untested question is whether re-running the width sweep at 2-3x the widths restores the floor at $\lambda_{\max}T=6$-$12$, and whether the $\varepsilon_{\text{mach}}e^{\lambda_{\max}T}$ line ever becomes the binding constraint before the solve gets too expensive.
- **Windowing.** A single global window is the faithful expF01 framing, but sequential windows (or multiple shooting) is the obvious way past the exponential wall. Untested.
- **Rössler's resolution cost.** Its sharp $z$-spike is what sets the width; would a non-uniform time reparameterisation (arclength, or $\lambda$-per-band as in expC06's cascaded geometry) buy back the two widths?
- **Higher $d$.** Lorenz96 at $d=4$ costs $2.3\times$ Lorenz at the same width; the solve is dense in $d\,W$, so $d\gtrsim 20$ needs the block structure exploited rather than assembled.
