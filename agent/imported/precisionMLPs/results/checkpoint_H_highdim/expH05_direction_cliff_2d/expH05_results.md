# expH05 -- the 2-D direction cliff, on nine functions

**Status:** draft -- conclusions pending Sam's sign-off.

## TL;DR

- Seven of the nine targets show the cliff: the error sits on a plateau while $M$ is too small, then falls 5 to 9 orders of magnitude across two or three steps of $M$, then stops at a floor near $10^{-14}$ relative $L_2$ (largest absolute error $\sim 10^{-13}$).
- The cliff moves right as the data ball grows. Over the $8\times$ range $r = 0.1 \to 0.8$ the first $M$ below $10^{-10}$ roughly doubles for the targets that get there at all (gauss bump and fast waves both go $8 \to 16$).
- Two targets are not resolution tests and behave differently: the product of sines is exactly a sum of two ridges at $45^\circ$ and $135^\circ$, so it is solved to $10^{-14}$ whenever the direction set happens to contain them ($M = 2, 6$) and badly otherwise -- a sawtooth, not a cliff; the degree-4 polynomial hits the floor at $M = 4$ at every radius, with no dependence on $r$ at all.
- The narrow radial Runge never reaches $10^{-10}$ at any radius within $M \le 16$; the composition, radial Runge and spatial packet reach it only at the smaller radii.
- Doubling the offsets per direction at the hardest setting ($r = 0.8$, $M = 16$: 128 to 256) leaves every error that is above the floor unchanged to within 9%. The direction count, not the along-direction resolution, is what binds.

- **Follow-up (the split, $r = 0.4$):** at a fixed unit budget $MN$ the error is the worse of two separate limits, one set by the direction count and one by the offsets per direction, and each shows up as a dead-flat band on the map. The best split is where they cross, and the optimal path doubles $M$ and $N$ in alternation, keeping $N/M$ inside $[1/2, 4]$ for three of the four targets and reaching 8 for the one strongly oscillatory target. That contrast does not follow the symmetric/asymmetric grouping the targets were picked for.

- **Follow-up 2 (the tradeoff at fine resolution):** the two floors have different shapes -- against the direction count the error falls exponentially or faster ($e^{-aM^q}$, $q = 0.92$ to $2.1$), against the offsets per direction it falls like a power law, roughly $N^{-10}$. Taking the larger of the two measured curves predicts every above-floor cell of the exact grid to within 21%, and the optimal split follows $M^* \propto B^\alpha$ with $\alpha = 0.27$ to $0.44$ (never the balanced $1/2$), which is what the two fitted floor laws predict.

## Question

expH04 saw the cliff on one target. Does a 2-D ridge network fitted by a single least-squares readout show the same plateau-then-collapse shape across functions of very different character, and where does the collapse sit as a function of the target and of the size of the region we are asking for precision on?

## Experiment design

**Domain and data.** Everything lives on $[-1,1]^2$. Training and test points are drawn uniformly in a ball about the off-center point $x_0 = (0.35, -0.25)$: $n_\text{train} = 8 \cdot (\text{units})$ points in the ball of radius $r$, $r \in \{0.1, 0.2, 0.4, 0.8\}$, seed 0. Uniform-in-ball means radius drawn as $r \cdot u^{1/2}$ with $u$ uniform, direction uniform on the circle.

**The model.** A single-hidden-layer tanh ridge network with the geometry fixed in advance and only the readout solved. The geometry is recentered on $x_0$: take $M$ evenly spaced angles on $[0,\pi)$ offset by half a step,

$$\theta_k = \frac{\pi}{2M} + \frac{k\pi}{M}, \qquad v_k = (\cos\theta_k, \sin\theta_k), \qquad k = 0, \dots, M-1,$$

with $M \in \{1, 2, 3, 4, 6, 8, 12, 16\}$ (the same half-step-offset set `even_directions` uses in expH01, so no direction lands on an axis). Along each direction the offsets span the band the data actually projects into, plus a 25% collar: $c = v_k \cdot x_0 + t$ with $t$ evenly spaced (cell-centered) over $[-T, T]$, $T = 1.25\,r$. The collar is the same margin expH01 and expE01 use, and it is load-bearing there: without it the two ends of each line are starved. The number of offsets is $n_\text{per} = 128$ on **every** line at **every** $M$ -- deliberately generous, so the along-direction resolution is never what limits the fit and the only thing changing across a panel is the direction count. Total units $= 128 M$, at most 2048. Widths come from the spacing, $h = 2T/n_\text{per}$ and $\gamma = \lambda/h$ with $\lambda = 0.25$ (the value expC03 settled on). The feature matrix is

$$\Phi_{ij} = \tanh\!\big(\gamma_j\,(v_j \cdot x_i - c_j)\big).$$

**The solve.** One truncated-SVD least squares on $[\Phi, \mathbf{1}]$ at $\texttt{rcond} = 10^{-13}$ (`h01suite.baseline._solve_svd`), singular values below $10^{-13}\sigma_{\max}$ dropped. Nothing is trained; nothing is iterated. Because $\Phi$ depends only on $(r, M)$ and not on the target, the SVD is taken once per $(r, M)$ and all nine right-hand sides are solved against it; `--check` verifies that this agrees with calling `_solve_svd` nine times (worst relative coefficient difference $6\times10^{-10}$, which is $\kappa \epsilon$ for this system).

**Scoring.** 20000 points uniform in the ball of radius $0.9\,r$ about $x_0$ -- the inner 90%, so the collar and the outermost shell of the data are never scored. Metrics are $\text{rel }L_2 = \|\hat f - f\|_2/\|f\|_2$ and $\max_x|\hat f - f|$ over those points. Note the denominator is the target's norm *on that ball*, which shrinks with $r$ for some targets; relative $L_2$ is therefore not perfectly comparable across radii for a target that is nearly constant on the smallest ball.

**The nine targets.** Plain numpy, no rotated coordinates (unlike the expH01 suite, which writes its families in the normalized coordinates $z$). Following the suite's convention the scaled radial distance is $\rho_a(x) = \|x - a\|_2/\sqrt{2}$. Three anchors are used: $a_1 = (0.2, 0.1)$ for the bumps, $a_2 = (0.3, -0.2)$ for the radial family, and $a_0 = x_0 = (0.35, -0.25)$ for the packet. $a_2$ sits at distance $0.071$ from $x_0$, so the center of the concentric waves is inside even the $r = 0.1$ ball; $a_0$ is the ball's own center, so the packet's oscillating core is always inside it. $a_1$ is $0.38$ away, outside the two smallest balls -- the bumps are the easy targets and do not need to be centered.

| # | name | formula |
|---|---|---|
| 1 | gauss bump | $\exp(-\|x - a_1\|^2/0.5^2)$ |
| 2 | product sines | $\sin(2\pi x)\sin(2\pi y)$ |
| 3 | composition | $\exp(\sin(\pi x)\cos(\pi y))$ |
| 4 | polynomial | $x^2 y - x y^3 + x y$ |
| 5 | slow concentric waves | $\cos(\pi \rho_{a_2})$ |
| 6 | radial Runge | $1/(1 + 16\rho_{a_2}^2)$ |
| 7 | fast concentric waves | $\cos(6\pi \rho_{a_2})$ |
| 8 | narrow radial Runge | $1/(1 + 144\rho_{a_2}^2)$ |
| 9 | spatial packet | $0.8\,e^{-(\rho_{a_0}/0.18)^2}\cos(10\pi\rho_{a_0}) + e^{-\rho_{a_1}^2/(2\cdot 0.5^2)}$ |

All are $O(1)$ on the cube. Targets 2 and 4 turned out not to be resolution tests -- see Results.

**The control.** At $r = 0.8$, $M = 16$, refit with $n_\text{per} = 256$ instead of 128 (4096 units, 32768 training points) and compare, to check that the plateau at the largest radius is a shortage of directions and not of offsets.

**Code & data.** `experiments/expH05_direction_cliff_2d/run.py` (`--plot` replots from the saved data, `--check` runs the solver equivalence check, `--control` runs the offsets control). Data: `results/checkpoint_H_highdim/expH05_direction_cliff_2d/data.json` (288 rows: function, $r$, $M$, units, $n_\text{train}$, rel $L_2$, max abs, rank, readout norm) and `control_n_per.json`. Figures: `figures/direction_cliff_2d.png`, `figures/cliff_summary_2d.png`, `figures/control_n_per.png`. Full sweep runtime 27 s on 6 threads; the control adds 40 s.

## Results

**The shape.** For seven of the nine targets the error against $M$ has three parts. A plateau, where adding a direction buys well under an order of magnitude. Then a collapse: on the last two or three steps of the $M$ grid the error falls 5 to 9 orders. Then a floor. Two examples. Gauss bump at $r = 0.1$ steps through $6\times10^{-2}$, $6\times10^{-3}$, $3\times10^{-4}$, $2\times10^{-6}$, $4\times10^{-9}$, $2\times10^{-12}$, $4\times10^{-14}$ -- per-step drops of $1.0, 1.3, 2.2, 2.7, 3.3, 3.3$ orders, so the descent accelerates rather than running at a fixed rate. Fast concentric waves at $r = 0.4$ is sharper still: $3\times10^{-1}$, $7\times10^{-2}$, $6\times10^{-4}$, $1\times10^{-6}$, $5\times10^{-13}$ over $M = 3, 4, 6, 8, 12$, which is 6.5 orders on the single step $M = 8 \to 12$.

Taking "a cliff" to mean a drop of at least 4 orders across two steps of the $M$ grid: gauss bump, slow waves, fast waves and the spatial packet clear it at every radius where they get there; composition and radial Runge clear it at $r \le 0.4$; the narrow radial Runge clears it only at $r = 0.1$ (4.6 orders over $M = 8 \to 16$) and even then never crosses $10^{-10}$. Where the criterion fails it is because the sweep stops before the collapse finishes, not because the curve is gentle.

**Where the cliff sits.** The smallest $M$ reaching relative $L_2 < 10^{-10}$ inside the ball:

| function | $r=0.1$ | $r=0.2$ | $r=0.4$ | $r=0.8$ |
|---|---|---|---|---|
| gauss bump | 8 | 12 | 12 | 16 |
| product sines | 2 | 2 | 2 | 2 |
| composition | 12 | 12 | -- | -- |
| polynomial | 4 | 4 | 4 | 4 |
| slow concentric waves | 4 | 6 | 6 | 8 |
| radial Runge | 8 | 12 | 16 | -- |
| fast concentric waves | 8 | 12 | 12 | 16 |
| narrow radial Runge | -- | -- | -- | -- |
| spatial packet | 8 | 12 | -- | -- |

("--" means never, up to $M = 16$.) The ordering matches how hard the targets look: the smooth bump and the slow waves are cheapest, the fast waves and the packet cost twice as many directions, and the narrow Runge -- whose complex pole sits a distance $1/12$ from the real plane in $\rho$ -- is out of reach everywhere within this budget. The threshold rises with the radius for every target that depends on $r$ at all, roughly doubling across the $8\times$ range. That is well short of the $M \propto r^{d-1} = r$ that expH04's plane-wave argument predicts at large $kr$; the $M$ grid here is coarse and stops at 16, so this is not a scaling measurement, and expH04 (which ran to $M = 96$) is the place to read one. For the one target the two experiments share in spirit -- fast concentric waves -- expH04 got thresholds $12, 12, 16, 24$ against our $8, 12, 12, 16$; same shape, ours slightly cheaper, consistent with scoring only the inner 90% and using 128 offsets per line instead of 48.

**The floor.** Where a target reaches it the floor is $3\times10^{-15}$ to $9\times10^{-14}$ relative $L_2$, with largest absolute error $10^{-14}$ to $10^{-13}$, and it does not improve with further directions -- several curves tick up slightly from $M = 12$ to $M = 16$. This is the accuracy of the single truncated-SVD solve on this geometry, not an approximation error.

**Two targets are measuring something else.** The product of sines is exactly a sum of two ridge functions: $\sin(2\pi x)\sin(2\pi y) = \tfrac12[\cos(2\pi(x-y)) - \cos(2\pi(x+y))]$, one term along $45^\circ$ and one along $135^\circ$. The half-step-offset direction set contains those two angles exactly when $M \equiv 2 \pmod 4$, i.e. at $M = 2$ and $M = 6$ on this grid -- and those are precisely the two $M$ where the error is $\sim 10^{-14}$ at every radius, while $M = 3$ and $M = 4$ sit at $10^{-3}$. The panel is a sawtooth, not a cliff, and it is the cleanest demonstration in the figure that what the cliff is about is whether the direction set contains the directions the target needs. It also means this target does not satisfy the "genuinely 2-D, not a sum of 1-D ridges" requirement and should be replaced if the design is reused. The polynomial is a milder version of the same thing: a degree-4 bivariate polynomial is (very nearly) in the span of ridge functions along a handful of directions no matter what region you look at, and it duly drops about 11 orders on the single step $M = 3 \to 4$ and reaches $10^{-14}$ at $M = 4$ for all four radii, with no $r$ dependence whatsoever.

**Readout norms.** The readout weight norm and the error move together across the whole sweep, over 24 orders of magnitude combined. Below the cliff the solve is in a huge-cancellation regime -- $\|w\|_2$ of $10^7$ to $10^9$, fitting the training points by differencing enormous numbers and generalizing badly inside the very ball it was trained on. At the cliff $\|w\|_2$ collapses to $O(10^{-1})$. Fast waves at $r = 0.1$: $\|w\|_2 = 4\times10^5$ at $M = 4$, $4\times10^1$ at $M = 6$, $10^{-1}$ at $M = 8$, while the error goes $6\times10^{-5} \to 6\times10^{-9} \to 2\times10^{-13}$. This is the same weight-blowup signature the 1-D work tracks, and here it is a usable diagnostic: the readout norm says whether the direction set is adequate without needing a reference solution.

**The control.** At $r = 0.8$ and $M = 16$, doubling the offsets per direction from 128 to 256 (2048 to 4096 units) leaves every error that has not already bottomed out unchanged to within 9%: narrow Runge $9.17\times10^{-3} \to 9.42\times10^{-3}$, spatial packet $1.033\times10^{-3} \to 1.044\times10^{-3}$, composition $7.36\times10^{-6} \to 7.46\times10^{-6}$, radial Runge $1.51\times10^{-6} \to 1.57\times10^{-6}$. The two targets sitting at the floor (polynomial, slow waves) move by 3.5x and 7.3x but stay inside the floor band, $10^{-14}$ to $10^{-13}$. The plateau at large $r$ is a shortage of directions.

### Figures

- **`figures/direction_cliff_2d.png`** -- the deliverable. A $3\times3$ grid, one panel per target, titled with its name and formula, in roughly increasing difficulty (reading order). $x$ is the number of directions $M$ on a log axis with ticks at the eight actual values $1, 2, 3, 4, 6, 8, 12, 16$; $y$ is relative $L_2$ inside the ball, fixed to $[10^{-15}, 10^{1}]$ on every panel so panels are directly comparable. Four lines per panel, one per data radius, viridis dark-to-light for $r = 0.1, 0.2, 0.4, 0.8$, markers at every $M$; the dotted grey horizontal line is $10^{-10}$. What to look for: the flat-then-plunge-then-flat shape and how far right the plunge sits; the ordering of the four lines once the descent starts (larger balls always need more directions; on the plateau the lines are bunched and can cross); the sawtooth in the product-sines panel and the one-step step in the polynomial panel, which are the two targets that are not resolution tests; and the bottom-middle panel, where all four lines are still descending at $M = 16$.
- **`figures/control_n_per.png`** -- the offsets control. One column per target (same order), $y$ = relative $L_2$ on a fixed $[10^{-15}, 10^{1}]$ axis, a large blue circle for 128 offsets per direction and a small orange square for 256, joined by a grey line; the grey band is the floor, $10^{-14}$ to $10^{-13}$. What to look for: the two markers sit on top of each other for every target that is above the floor. Doubling the along-direction resolution buys nothing at the largest radius.
- **`figures/cliff_summary_2d.png`** -- two summary panels. Left: the table above as a coloured grid, rows = the nine targets in the same order, columns = the four radii, cell text = the smallest $M$ below $10^{-10}$ (grey with "--" where it never happens), colour = that same $M$ on a log scale. Read it for the two flat rows (product sines, polynomial: no $r$ dependence), the rows that walk right as $r$ grows, and the fully grey row (narrow Runge). Right: every one of the 288 fits as a point, $x$ = relative $L_2$, $y$ = readout weight norm $\|w\|_2$, coloured by radius, with the $10^{-10}$ threshold marked. Read it for the single tight monotone band -- the solve pays for an inadequate direction set in readout magnitude, and the points at the error floor are exactly the points with $O(0.1)$ weights.

## Follow-up: spending the budget on directions or on offsets

**What was run.** The main sweep held the offsets per direction fixed at 128 and moved only $M$. This asks the other question: at a fixed number of units, is it better to buy directions or offsets? One data radius, $r = 0.4$, everything else identical (same ball about $x_0$, same recentered geometry, same 25% collar, same $\gamma = 0.25/h$, same single truncated-SVD readout, same $n_\text{train} = 8\,MN$ at seed 0, same scoring on the inner 90%).

The grid is powers of two, $M \in \{2,4,8,16,32,64\}$ against $N \in \{8,16,32,64,128\}$, keeping the 29 cells with $MN \le 4096$. That makes every budget $B = MN$ an **exact** anti-diagonal of cells rather than a line drawn between them: $B \in \{16, 32, 64, 128, 256, 512, 1024, 2048, 4096\}$ with $1, 2, 3, 4, 5, 5, 4, 3, 2$ cells on it (for instance $B = 1024$ is exactly $8\times128$, $16\times64$, $32\times32$, $64\times16$). The largest solve is $32768 \times 4097$. One SVD per cell serves all four targets: the two radial ones (fast concentric waves, radial Runge) and two asymmetric ones (composition, spatial packet). 29 cells, 130 s.

On each diagonal the best cell is the one with the lowest error, and a **tie** is any other cell on the same diagonal within a factor 2 of it -- a flat-bottomed diagonal, where the split hardly matters.

**Code & data.** `run.py --split-exact` runs it, `run.py --split-exact --plot` replots from `split_exact_2d.json` (116 rows: function, $M$, $N$, units, $n_\text{train}$, rel $L_2$, max abs, rank, weight norm). Figure: `figures/split_exact_2d.png`. An earlier, log-spaced $9\times9$ version of the same sweep (`figures/split_heatmap_2d.png`, `split_heatmap_2d.json`) is kept on disk; its iso-budget lines were drawn between cells rather than through them, and it is superseded by this one.

**Best cell on every exact budget.** Error is relative $L_2$ inside the ball.

| $B$ | cells | fast concentric waves | radial Runge | composition | spatial packet |
|---|---|---|---|---|---|
| 16 | 1 | $2{\times}8$, 6.2e-01 | $2{\times}8$, 4.1e-02 | $2{\times}8$, 3.7e-02 | $2{\times}8$, 2.7e-01 |
| 32 | 2 | $4{\times}8$, 6.0e-02 | $4{\times}8$, 2.2e-03 | $4{\times}8$, 4.5e-03 | $4{\times}8$, 2.0e-01 *(tie $2{\times}16$, 2.6e-01)* |
| 64 | 3 | $8{\times}8$, 1.4e-03 | $8{\times}8$, 2.7e-04 | $8{\times}8$, 2.5e-05 | $8{\times}8$, 1.2e-01 *(tie $4{\times}16$, 2.1e-01)* |
| 128 | 4 | $8{\times}16$, 4.1e-06 | $8{\times}16$, 6.6e-06 | $16{\times}8$, 1.3e-06 | $8{\times}16$, 3.4e-03 |
| 256 | 5 | $8{\times}32$, 1.6e-06 | $16{\times}16$, 8.2e-09 | $16{\times}16$, 2.3e-08 | $16{\times}16$, 7.4e-07 |
| 512 | 5 | $16{\times}32$, 1.7e-09 | $16{\times}32$, 8.5e-11 | $16{\times}32$, 6.0e-10 | $16{\times}32$, 1.5e-10 |
| 1024 | 4 | $16{\times}64$, 7.3e-13 | $32{\times}32$, 8.0e-12 | $32{\times}32$, 4.2e-11 | $32{\times}32$, 4.3e-11 |
| 2048 | 3 | $16{\times}128$, 2.9e-14 | $32{\times}64$, 7.4e-15 | $32{\times}64$, 1.5e-13 | $32{\times}64$, 3.0e-13 |
| 4096 | 2 | $32{\times}128$, 1.3e-13 | $64{\times}64$, 7.5e-15 *(tie $32{\times}128$, 1.4e-14)* | $32{\times}128$, 1.3e-14 | $32{\times}128$, 1.9e-14 |

Only four of the 36 (function, budget) pairs have a tie, so on this grid the best split is usually a clear winner rather than a flat bottom. Three of the four ties are at the tiny budgets where the packet has not started converging at all; the fourth is the genuine one, radial Runge at 4096, where $64\times64$ and $32\times128$ are both at the floor and the difference between them is meaningless.

**What the path does.** The optimal $(M, N)$ does not jump around: each doubling of the budget doubles exactly one of the two, and the path is nearly the same for all four targets --

$2{\times}8 \to 4{\times}8 \to 8{\times}8 \to 8{\times}16 \to 16{\times}16 \to 16{\times}32 \to 32{\times}32 \to 32{\times}64 \to 32{\times}128$,

with fast waves running one step ahead on offsets from $B = 256$ ($8{\times}32$, then $16{\times}64$, then $16{\times}128$), composition spending its $B = 128$ doubling on $M$ instead of $N$, and radial Runge taking $64\times64$ rather than $32\times128$ at the top. So the ratio $N/M$ stays inside $[1/2, 4]$ for the three slow-along-a-line targets and reaches 8 for fast waves. Neither axis is ever the right place to spend everything: the extreme cells of a diagonal are always the worst on it.

**The map is the larger of two separate floors.** That is why the path stays near the diagonal. Each axis has its own hard floor and the error is set by whichever is worse, which shows up as dead-flat bands. Along the rows: fast waves at $N = 8$ gives $1.44\times10^{-3}$, $1.43\times10^{-3}$, $1.43\times10^{-3}$ at $M = 8, 32, 64$ -- past $M = 8$ extra directions buy nothing at all while the offsets are starved. Along the columns: at $M = 2$ every target is unchanged to within 4% across all five values of $N$ ($6.2\times10^{-1}$ for fast waves at $N = 8$ and at $N = 128$), and at $M = 16$ radial Runge sits at $8.5\times10^{-11}$, $7.4\times10^{-11}$, $8.2\times10^{-11}$ for $N = 32, 64, 128$. Going from $M = 16$ to $M = 32$ at $N = 64$ then drops it from $7.4\times10^{-11}$ to $7.4\times10^{-15}$.

**Which side the optimum sits on.** Writing the split as $N/M$, the balanced split is 1. Fast concentric waves is the only one of the four that consistently wants more offsets than directions -- $N/M = 4$ at $B = 256$, then 2, 4, 8, 4 -- and it is also the only one that oscillates several times along a line through the ball. The other three sit at $N/M \in \{1, 2\}$ from $B = 256$ upward, including radial Runge, which is radially symmetric and behaves like the two asymmetric targets rather than like the other radial one. On this evidence the split preference tracks how much structure the target has *along* a direction versus *across* directions, not symmetry; one radial pair is far too small a sample to say anything about symmetry.

**A caveat on the floor.** Once a target is at the floor the map is noisy at the $10^{-14}$ level and more units can read slightly worse: fast waves is $2.9\times10^{-14}$ at $16\times128$ (2048 units) and $1.3\times10^{-13}$ at $32\times128$ (4096), and radial Runge is flat from 2048 to 4096. Only differences well above $10^{-14}$ should be read, and the best cell at the largest budget is not always the best cell overall.

### Figure

- **`figures/split_exact_2d.png`** -- four columns, one per target, titled with name and formula, radial pair first. **Top row:** a heat map of $\log_{10}$ relative $L_2$ inside the ball over directions $M$ ($x$) against offsets per direction $N$ ($y$), both log2-spaced with ticks at the grid values; one shared colour scale on the right fixed to $[-14, 0]$, dark = accurate; cells past the 4096-unit cap are blank. Nothing is drawn on top of the data. What to look for: the flat horizontal band along $N = 8$, where extra directions buy nothing because the offsets are starved, and the equally flat vertical bands at $M = 2$ and $M = 4$, where extra offsets buy nothing because the directions are; the dark corner opens up only where both are adequate, and it opens at a different place in each panel. **Bottom row:** the steepest-descent path. $x$ is the unit budget $B = MN$ on a log2 axis with a tick at each of the nine exact budgets. On the left log2 axis, blue circles are the optimal $M$ on that budget and red squares the optimal $N$; hollow markers of the same colour are tie cells (another cell on the same diagonal within a factor 2), so a flat-bottomed diagonal shows as a hollow marker beside a filled one. On the right log axis, fixed to $[10^{-15}, 10^{1}]$, the grey triangles are the best error achievable at that budget. Legend above each panel. What to look for: the blue and red staircases climbing in alternation and staying within a factor of a few of each other -- never one axis running away from the other -- and the grey curve falling roughly a decade per budget doubling until it flattens at the floor.

## Follow-up: the tradeoff curve at fine resolution, and its exponents

**What was run.** The exact-grid map above said the error surface looks like the larger of two independent floors: one set by the direction count, flat along $N$, and one set by the offsets per direction, flat along $M$. That was read off a $6\times5$ grid of powers of two. This measures the two floors directly on a fine grid, tests the max-of-two-floors model against every cell of the exact grid, and then measures how the optimal split moves with the budget. Same setup throughout as the split sweep: one data radius $r = 0.4$, ball about $x_0$, 25% collar, $\gamma = 0.25/h$, $n_\text{train} = 8MN$ at seed 0, one truncated-SVD readout at $\texttt{rcond} = 10^{-13}$, error scored on 20000 points in the inner 90% of the ball. One SVD per $(M, N)$ serves all four targets, and every $(M, N)$ is solved once and cached across both parts.

**Part A, the two floor curves.**

- $e_M(M)$: $N = 128$ fixed, $M \in \{2,3,4,5,6,8,10,12,14,16,20,24,28,32,40,48\}$ (up to 6144 units; the $48\times128$ solve is $49152 \times 6145$). $M = 64$ was not run: at $N = 128$ that is 8192 units and a $65536\times8193$ SVD, which is about eight minutes and 13 GB on its own, and the curve is already flat at the floor from $M = 20$ upward for all four targets.
- $e_N(N)$: $M = 48$ fixed, $N \in \{4,6,8,10,12,16,20,24,32,40,48,64,96,128\}$. 48 directions is past where the direction floor stops binding for all four targets, so this curve is offset-limited everywhere it is above the floor.

Each curve is fitted per target on its **leading contiguous run above $10^{-13}$** -- once a curve reaches the floor it bounces around $10^{-14}$, and individual later points read back above $10^{-13}$; including those would fit the noise. Three forms, all fitted in $\ln e$: exponential $e = A e^{-ax}$, power law $e = A x^{-p}$, stretched exponential $e = A e^{-a x^q}$ ($q$ by 1-D search, coefficients by linear least squares at each $q$). The reported residual is the RMS of $\log_{10}(\text{fit}) - \log_{10}(\text{measured})$, so 0.1 means the fit is typically within 25% of the measured error and 1.0 means it is typically off by a decade.

**Part A, the model test.** For each target, interpolate $e_M$ and $e_N$ piecewise-linearly in $(\log x, \log e)$ (clamped at both ends -- both curves are flat there), predict each of the 29 cells of `split_exact_2d.json` by $\max(e_M(M), e_N(N))$, and compare with the measured value. Reported as the ratio predicted/measured. The five cells at $N = 128$ lie on the $e_M$ curve itself and are not predictions, so they are excluded from the headline numbers.

**Part B, the iso-budget valleys.** Budgets $B \in \{64, 91, 128, 181, 256, 362, 512, 724, 1024, 1448, 2048, 2896, 4096\}$ (a $\sqrt{2}$ ladder). On each budget, about ten log-spaced integer $M$ between $\max(2, B/128)$ and $\min(48, B/8)$, with $N = \mathrm{round}(B/M)$; the actual $M \cdot N$ is recorded and is within 1% of $B$ everywhere. The lower limit on $M$ is what caps $N$ at 128, matching the $e_M$ sweep; the upper limit is the 48-direction cap. Per (target, budget) the analysis records the minimizer $M^*(B)$, $N^*(B)$, the best error, and the **valley width**, the range of $M$ on the ladder whose error is within $2\times$ of the minimum. The exponent $\alpha$ in $M^* \sim B^\alpha$ is an ordinary least-squares fit of $\log M^*$ on $\log B$ over the budgets whose best error is above $10^{-13}$, with the standard error from the residuals. The same exponent is then predicted from the part-A laws alone, by minimizing $\max(e_M(M), e_N(B/M))$ over $M$ on the same range for each of those budgets and fitting $\log M$ on $\log B$.

**Code & data.** `run.py --tradeoff` runs both parts, `run.py --tradeoff --plot` re-derives the fits, the model test, the valleys and the exponents from the saved cells and replots (nothing is re-solved). Data: `results/checkpoint_H_highdim/expH05_direction_cliff_2d/tradeoff_2d.json` (147 distinct $(M, N)$ cells with rel $L_2$, max abs, rank and readout norm per target, plus the two floor curves, the fits, the model test, the valleys and the exponents). Figure: `figures/tradeoff_2d.png`. 147 solves, 739 s on 6 threads. The 64 (target, $M$, $N$) combinations shared with `split_exact_2d.json` reproduce it exactly, bit for bit.

### Part A results: the two floors have different shapes

The two curves are not the same kind of decay. Against $M$ the fall is exponential or faster; against $N$ it is much closer to a power law. Best form and its parameters, with the residual in $\log_{10}$ error, and the two rejected forms for comparison:

| target | $e_M(M)$, $N = 128$ | resid | exp / power resid | $e_N(N)$, $M = 48$ | resid | exp / power resid |
|---|---|---|---|---|---|---|
| fast concentric waves | $e^{-0.336\,M^{1.80}}$ | 0.11 | 0.69 / 1.68 | $e^{-25.9\,N^{0.218}}$ | 0.35 | 1.12 / 0.48 |
| radial Runge | $e^{-1.57\,M^{0.97}}$ | 0.010 | 0.029 / 1.00 | $e^{-25.9\,N^{0.238}}$ | 0.48 | 0.99 / 0.57 |
| composition | $e^{-1.78\,M^{0.92}}$ | 0.34 | 0.34 / 0.96 | $e^{-182\,N^{0.043}}$ | 0.37 | 0.82 / 0.37 |
| spatial packet | $e^{-0.0667\,M^{2.10}}$ | 0.14 | 0.74 / 1.52 | $e^{-4.46\,N^{0.595}}$ | 0.73 | 0.90 / 1.13 |

Read the exponent $q$ rather than the prefactor. For $e_M$, radial Runge and composition sit at $q \approx 0.95$, which is a plain exponential in $M$ -- the stretched fit barely improves on it (0.010 vs 0.029 for radial Runge, a dead heat for composition), and the power law is off by a decade RMS. The two harder targets are steeper than exponential, $q = 1.8$ and $q = 2.1$. Radial Runge is the clean case: the plain exponential $e^{-1.43M}$ tracks eleven points over nine orders with a 7% typical residual.

For $e_N$ the picture is reversed. Three of the four fit a power law at least as well as anything else, and the stretched fit for those has drifted to small $q$ with a large $a$, which is the known degeneracy -- $Ae^{-aN^q}$ with $q \to 0$ *is* a power law, and for composition the two residuals are 0.368 and 0.369, the same fit written twice. Only the spatial packet has a genuinely intermediate $q = 0.60$. The measured power-law exponents are $p = 8.7$ to $12.9$, so along a direction the error falls roughly as $N^{-10}$, not exponentially. That is the more interesting half of the result and the residuals for it are the worst in the table (0.35--0.73): the $e_N$ curves have a visible shoulder -- radial Runge goes $7.9\times10^{-9}$, $4.3\times10^{-10}$, $1.3\times10^{-10}$ at $N = 16, 20, 24$ and then resumes falling -- which no single one of these three forms captures.

**The model test.** Predicting all 29 exact-grid cells by $\max(e_M(M), e_N(N))$, using the interpolated curves and no fit:

| target | cells above the floor and off the $e_M$ curve | median ratio | ratio range | worst factor |
|---|---|---|---|---|
| fast concentric waves | 21 | 1.01 | 0.83 -- 1.12 | 1.21 |
| radial Runge | 22 | 1.01 | 0.96 -- 1.11 | 1.11 |
| composition | 22 | 1.00 | 0.92 -- 1.05 | 1.09 |
| spatial packet | 22 | 1.01 | 0.97 -- 1.07 | 1.07 |

The model holds. Every cell above the floor is predicted to within 21%, and for three of the four targets to within 12%, by two one-dimensional curves and a maximum -- across nine orders of magnitude of error and with no free parameter fitted to the grid. The worst single cell in the whole test is fast waves at $8\times16$, predicted $3.4\times10^{-6}$ against a measured $4.1\times10^{-6}$. Including the cells at the floor the ratios reach $9.3$ (radial Runge), which is expected and not a failure of the model: at $10^{-14}$ the measured value is round-off, and a model built from two smooth curves has nothing to say about it.

### Part B results: the valleys and the exponent

Each budget gives a narrow V in $M$. Above the floor the minimum is sharp: on 5 to 7 of the 8 or 9 above-floor budgets per target, **no other point on the ten-point ladder is within $2\times$ of the minimum**, and the widest valley anywhere is a $1.3\times$ range in $M$. Moving a factor of two away from $M^*$ costs a median of $10^3$ (radial Runge and fast waves) to $10^{4.8}$ (spatial packet) on the too-many-directions side, and $10^{2.8}$ to $10^{5.1}$ on the too-few side; the cheapest single misstep anywhere is composition at $2M^*$, at $60\times$. Across a whole budget the spread between the best and the worst split runs 3 to 10 orders of magnitude. Getting the split right is worth much more than getting more units. The left arms are a second, independent check on the model: at a fixed $M$ left of the minimizer the error is direction-limited and should not care which budget it came from, and it does not -- across all budgets that reach a given $M$ on their left arm, the error varies by less than $1.25\times$ at every $M$ but one.

The minimizer moves with the budget as a clean power law, and the exponent is well below the balanced $\alpha = 1/2$ for every target -- the optimum buys offsets faster than it buys directions:

| target | budgets used | $\alpha$ (measured) | $\alpha$ predicted from the part-A laws | $N^*/M^*$ over that range |
|---|---|---|---|---|
| fast concentric waves | 64 -- 1024 (9) | $0.274 \pm 0.032$ | $0.254 \pm 0.008$ | 1.4 $\to$ 4.5 |
| radial Runge | 64 -- 724 (8) | $0.420 \pm 0.047$ | $0.413 \pm 0.009$ | 1.3 $\to$ 2.0 |
| composition | 64 -- 724 (8) | $0.343 \pm 0.057$ | $0.349 \pm 0.013$ | 0.7 $\to$ 2.0 |
| spatial packet | 64 -- 724 (8) | $0.437 \pm 0.058$ | $0.364 \pm 0.013$ | 1.1 $\to$ 2.0 |

$N^* \sim B^{1-\alpha}$ then runs from $B^{0.56}$ to $B^{0.73}$. Three of the four predicted exponents land inside one standard error of the measured one; the spatial packet's is 1.3 standard errors low. That is the same statement as the model test, made a second way: the crossing point of the two fitted floor laws is where the measured minimizer is.

The ordering across targets follows the shapes in the part-A table. Fast concentric waves has the steepest $e_M$ ($q = 1.8$) against an ordinary $e_N$, so directions are cheap for it and the optimum drifts hardest toward offsets ($\alpha = 0.27$, $N^*/M^*$ reaching 4.5). Radial Runge has the shallowest $e_M$ of the four and the largest $\alpha$. This is the quantitative version of the qualitative claim in the previous section, that the split tracks how much structure the target has along a direction versus across directions.

### Figure

- **`figures/tradeoff_2d.png`** -- three rows, four columns, one column per target in the same order as the split figure (the two radial targets first), titled with name and formula. **Row 1, the two floor curves.** $x$ is a count on a log2 axis, ticks at 2, 4, 8, 16, 32, 128; $y$ is relative $L_2$ fixed to $[10^{-15}, 10^{1}]$ on every panel. Blue circles are $e_M(M)$ at $N = 128$, red squares $e_N(N)$ at $M = 48$; the dashed line of each colour is that curve's best-fitting form, drawn only over the range it was fitted on, with the law itself in the legend above the panel. What to look for: the blue curve is steeper than the red on every panel, and both stop dead at the same $\sim10^{-14}$ floor; where a dashed line is invisible under its solid line the fit is good (radial Runge's blue), and where it bows away from the markers it is not (the red shoulders near $N = 20$). **Row 2, the valleys.** $x$ is the direction count $M$ on a log2 axis with $N \approx B/M$ implied; $y$ is the same fixed $[10^{-15}, 10^1]$. One line per budget, coloured by $\log B$ on the viridis colourbar at the right, a star at each minimizer. What to look for: every line is a narrow V, not a flat basin; the left arms of all budgets collapse onto one curve, because on the left of the valley the fit is direction-limited and the offsets it has are irrelevant; the stars walk steadily right as the budget grows; and the top budgets have their V bottoms flattened by the $10^{-14}$ floor. **Row 3, where the optimum sits.** $x$ is the budget $B$ on a log2 axis with a tick at each of the thirteen budgets, $y$ is $M^*$ and $N^*$ on a log2 axis fixed to $[1.5, 320]$. Filled blue circles are $M^*$, filled red squares $N^*$, hollow markers are the budgets whose best error is at the floor and which are therefore excluded from the fit; the solid lines are the fitted $B^\alpha$ and $B^{1-\alpha}$ with $\alpha$ and its standard error in the legend, and the grey dashed line is $\alpha$ predicted from the part-A laws. What to look for: the grey dashed line sitting on the blue one, and the red line above the blue one with a visibly larger slope on all four panels.

## Additional details

- **Confound in the $r$ sweep.** $n_\text{per}$ is held at 128 while the projection band $T = 1.25r$ grows with $r$, so the absolute along-direction spacing $h$ also grows with $r$. The offsets control shows this is not binding at $r = 0.8$, but the two axes are not fully separated by design.
- **Relative $L_2$ across radii.** The denominator is $\|f\|_2$ on the ball being scored. For a target that is nearly constant on the smallest ball (the gauss bump at $r = 0.1$, say) the denominator is large relative to the variation, which flatters the small-$r$ numbers a little. `max_abs` is recorded in `data.json` for anyone who wants the unnormalized version; the ratio $\max_x|\hat f - f| / \text{rel }L_2$ stays between 0.3 and 26 across all 288 fits.
- **Split sweep, grid edges.** The exact grid runs $M$ to 64 and $N$ to 128 under a 4096-unit cap, so the two largest budgets have only three and two cells on them and the $B = 4096$ diagonal cannot express much. No optimum is pinned to the $M$ edge below $B = 4096$.
- **Tradeoff ladder, edges.** Requiring $M \ge B/128$ caps $N$ at 128, so the ladder cannot express $N > 128$ and at the largest budgets $M$ is confined to a narrow range ($[32, 48]$ at $B = 4096$). Every budget in the exponent fits is well inside both limits except composition at $B = 64$ and $B = 91$, whose minimizers sit at the top $M$ of the ladder with $N^* = 8$; those two points push composition's $\alpha$ down slightly.
- **Tradeoff, $M = 64$.** $e_M$ stops at $M = 48$. At $N = 128$ that is already 6144 units and a $49152 \times 6145$ solve; $M = 64$ would be 8192 units, roughly eight minutes and 13 GB. Since all four curves are flat at the floor from $M = 20$ upward, the extra point would not constrain any fit.
- **Rank.** The truncated SVD keeps 82--86% of the columns at every $(r, M)$ (e.g. 1688 of 2049 at $M = 16$), essentially independent of $r$. The discarded directions are the tanh geometry's own redundancy, not a data effect.

## Conclusions

*Pending Sam's review.* What the data plainly shows: the plateau-collapse-floor shape reproduces on seven of nine targets, the collapse is 5--9 orders across two or three steps of $M$, the floor is $\sim10^{-14}$ relative $L_2$, the threshold rises with both the target's difficulty and the ball radius, and at the largest radius the binding constraint is measurably the direction count rather than the along-direction resolution. From the split follow-up: at fixed budget the two axes have independent floors, the map is the larger of them, and the best split sits where they cross -- for the one strongly oscillatory target on the more-offsets side, for the other three near balanced, with the optimal path doubling the two axes in alternation. From the fine tradeoff sweep: the two floors decay differently -- exponentially or faster in $M$, like a power law in $N$ -- the maximum of the two measured curves reproduces every above-floor cell of the exact grid to within 21%, the iso-budget valleys are sharp rather than flat-bottomed, and the optimal split follows $M^* \propto B^\alpha$ with $\alpha$ between 0.27 and 0.44, matching the crossing point of the two fitted laws.

## Open questions

- Two of the nine targets (product sines, polynomial) are exactly or nearly ridge-representable and so measure direction alignment rather than resolution. Should they be replaced, or kept as deliberate controls?
- The threshold roughly doubles over an $8\times$ range in $r$, which is far weaker than $M \propto r$. Is that the coarse $M$ grid and the $M \le 16$ ceiling, or is the $kr$ here simply too small for the asymptotic to apply?
- At a fixed budget the four targets do not split by symmetry but by whether their structure is along a direction or across directions. Is that the right axis, and does it survive on more targets?
- The optimal path alternates doublings of $M$ and $N$ and keeps them within a small factor of each other. Is that a rule worth building into a geometry-growing loop, or an artifact of a power-of-two grid?
- Why is the offset floor a power law in $N$ (roughly $N^{-10}$) when the direction floor is exponential in $M$? A power law in the along-direction resolution is not what a 1-D QI on a smooth target does, and the shoulder near $N = 20$ in three of the four $e_N$ curves suggests the curve is a mixture rather than one mechanism.
- $\alpha$ is below $1/2$ for all four targets here. Is that general, and does the max-of-two-floors picture -- fit two cheap 1-D curves, cross them, read off the split -- survive in $d \ge 3$, where the direction count is the expensive axis?
- The readout norm tracks the error over 24 orders without a reference solution. Is that reliable enough to use as the stopping test for a direction-adding loop?
