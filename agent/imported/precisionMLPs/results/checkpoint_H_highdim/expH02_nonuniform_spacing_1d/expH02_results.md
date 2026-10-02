# expH02 -- Smoothly non-uniform center spacing in 1-D, with $\lambda=0.25$ everywhere

**Status: conclusions approved by Sam (2026-08-28); residual follow-up added.**

## TL;DR

- Non-uniform spacing works, as long as it changes smoothly and each neuron's $\gamma_j$ is set from its own local spacing ($\gamma_jh_j=0.25$). With the half-Gaussian spacing (spacing changes $3\times$ across the interval, neighboring spacings within $1\%$ of each other) every width reaches machine precision ($\sim10^{-13}$) on all four functions, exactly like evenly spaced centers.
- The price of non-uniform spacing is set by the widest gap. The bimodal spacing (spacing changes $10\times$) reaches the same precision but needs $N\approx512$ centers where evenly spaced needs $128$; the widest gap in that geometry is about as wide as the evenly spaced gap at $N\approx32$-$40$, and the curves line up under that shift.
- What does not work is a spacing that jumps between neighbors and keeps jumping no matter how many centers there are. The fully non-uniform Beta(2,5) spacing has zero density at $x=1$, so the last gap there is $4\times$ its neighbor at every $N$; the error stalls at $3\times10^{-7}$ for $\sin2\pi x$ and $10^{-2}$ for the oscillatory functions. The rows one third and two thirds of the way toward it keep a uniform floor in their density (no jump, neighbor ratio $\le1.01$) and reach machine precision at $N=512$.
- Where the training data sit barely matters once there is enough of it: uniform data and data drawn from the center distribution give the same curves at $n=16W$ points. With $n=2W$ the error is $1$-$2$ orders worse and noisier. The exception is the Beta spacing, where center-distribution data leaves the right end empty and the error there is orders worse (the data-gap effect seen in expG03).

## Question

Keep $\lambda=\gamma_jh_j=0.25$ fixed and vary only the local spacing $h_j$, smoothly. Does the fixed-geometry least-squares fit still reach machine precision at the same width as evenly spaced centers?

## Experiment design

**Geometry.** Pick a "center distribution" $q$ on $[-1,1]$ and a parameter $s\in[0,1]$. The row's density is the mixture $q_s(x)=(1-s)\cdot\tfrac12+s\,q(x)$, and the interior centers are placed so that equal fractions of $q_s$ lie between neighbors:

$$c_j(s)=Q_s^{-1}(j/N),\qquad j=0,\dots,N,$$

with $Q_s^{-1}$ the inverse of $q_s$'s cumulative distribution. $s=0$ is exactly the standard evenly spaced grid; $s=1$ is the most regular set of centers whose smoothed histogram would look like $q$ (nothing is sampled at random); every in-between row keeps a uniform floor of $(1-s)/2$ in its density. (An earlier version interpolated the center *positions* instead, which follows a different in-between density; `center_placement.png` overlays both.) Halo: the standard number of extra centers beyond each end of the interval, $R=\texttt{default\_halo}(N,0.25)$, continuing the spacing at that end. Local spacing by central difference, $h_j=(c_{j+1}-c_{j-1})/2$ (one-sided at the two outermost halo centers), $\gamma_j=0.25/h_j$, neurons $\tanh(\gamma_j(x-c_j))$. Output weights and bias by least squares (truncated SVD, singular values below $10^{-13}$ of the largest dropped).

**Center distributions.** (i) `halfgauss`: density $\propto e^{-t^2/2}$ with $t$ running linearly from $-1.5$ at $x=-1$ to $0$ at $x=1$; spacing shrinks smoothly to the right. (ii) `bimodal`: $0.1\,\text{uniform}+0.9\,[\tfrac12N(-0.5,0.2^2)+\tfrac12N(0.5,0.2^2)]$. (iii) `beta`: Beta(2,5) on $(x+1)/2$, peaked near $x=-0.6$, zero density at both ends. Rows: $s\in\{0,\tfrac13,\tfrac23,1\}$. Checks: centers increasing, $\gamma_jh_j=0.25$ exactly; a histogram of the placed centers matches the intended density to within histogram noise (L1 distance $\approx0.02$ at $N=128$); largest ratio of neighboring spacings $h_{j+1}/h_j$ at $N=512$ is $1.00$ (halfgauss, all rows), $\le1.12$ (bimodal), $\le1.01$ (beta, $s\le\tfrac23$) and $4.45$ (beta, $s=1$). The Beta $s=1$ jump does not shrink with $N$.

**Functions.** $\sin2\pi x$, $1/(1+25x^2)$, $\sin8\pi x$, and $0.55\sin(\pi x-0.3)+e^{-((x-0.45)/0.16)^2}\sin(12\pi(x-0.45))$ (a slow wave with a short burst of high frequency near $x=0.45$).

**Data.** Four training sets per case: uniform on $[-1,1]$ and drawn from the row's own density $q_s$ ($x=Q_s^{-1}(u)$, $u\sim U(0,1)$), each with $n=2W$ and $n=16W$ points, $W=N+2R+1$. Widths $N\in\{32,64,128,256,512\}$. Error: relative $L_2$ on a 4001-point uniform grid over $[-1,1]$.

**Code & data.** `experiments/expH02_nonuniform_spacing_1d/run.py` (`--plot` to replot). Data `data.json`; the three error figures `spacing_{halfgauss,bimodal,beta}.png`, the three residual figures `residual_N128_{halfgauss,bimodal,beta}.png`, the center-placement figure `center_placement.png`, and the geometry check `diagnostics/centers_and_gammas.png`, under `results/checkpoint_H_highdim/expH02_nonuniform_spacing_1d/`.

## Results

Relative $L_2$ error at $N=512$, uniform data with $n=16W$:

| spacing | widest / narrowest gap | largest $h_{j+1}/h_j$ | $\sin2\pi x$ | Runge | $\sin8\pi x$ | burst |
|---|---|---|---|---|---|---|
| evenly spaced ($s=0$) | 1 | 1.00 | $6\times10^{-14}$ | $6\times10^{-14}$ | $1\times10^{-13}$ | $6\times10^{-14}$ |
| halfgauss, $s=1$ | 3.1 | 1.00 | $5\times10^{-14}$ | $7\times10^{-14}$ | $2\times10^{-13}$ | $6\times10^{-14}$ |
| bimodal, $s=1$ | 10.0 | 1.12 | $2\times10^{-14}$ | $6\times10^{-14}$ | $1\times10^{-13}$ | $4\times10^{-14}$ |
| beta, $s=\tfrac23$ | 5.9 | 1.01 | $1\times10^{-13}$ | $8\times10^{-14}$ | $9\times10^{-13}$ | $1\times10^{-13}$ |
| beta, $s=1$ | 262 | 4.45 | $3\times10^{-7}$ | $6\times10^{-9}$ | $1\times10^{-2}$ | $3\times10^{-2}$ |

- **Smooth changes in spacing cost nothing.** Every halfgauss row sits on the evenly spaced row's curve within a factor of about $2$ at every width.
- **The widest gap decides the width needed.** Bimodal at $s=1$ reaches $10^{-13}$ only at $N=512$; at $N=128$ it is $10^{-11}$ ($\sin2\pi x$) to $10^{-7}$ (burst). Beta at $s=\tfrac23$ (widest gap $6\times$ the narrowest) is at $10^{-13}$ by $N=512$ but $10^{-5}$ on the burst at $N=128$.
- **A spacing jump that never shrinks stalls the error.** Pure Beta(2,5) ($s=1$) stalls at every width, also when the error is measured on its own center distribution (stored in `data.json` under `eval="PX"`), so the problem is the geometry, not only missing data. The last gap at $x\to1$ is $\sim4\times$ its neighbor at every $N$ because the density vanishes like $(1-u)^4$, and the right halo is built from that oversized gap (very wide, nearly linear tanh units). The rows at $s=\tfrac13,\tfrac23$ have the same shape with a uniform floor of $\tfrac13,\tfrac16$ in the density, no jump, and they converge.
- **The error lives where the centers are sparse.** The signed residual at $N=128$ (`residual_N128_<distribution>.png`) is a ripple whose envelope tracks the local spacing: for bimodal it peaks at $x=0$ and at the edges (the gaps between and outside the two lobes), for Beta at the right end, and it is flat for the half-Gaussian. The four training sets give nearly the same residual curve, so the shape is set by the geometry, not by the data.
- **Amount of data matters more than where it is.** With $n=16W$, uniform and center-distribution data are indistinguishable except where the center distribution leaves a region empty (the Beta right end). With $n=2W$ the curves are $1$-$2$ orders above the limit and jittery.

### Figures

- **`spacing_<distribution>.png`** (the three deliverables) -- $4\times4$: rows = how far the centers have moved toward the non-uniform spacing ($s=0,\tfrac13,\tfrac23,1$); columns = the four functions; $x$ = width $N$; $y$ = relative $L_2$ error on the uniform grid, fixed $[10^{-15},10]$. Blue = uniform training data, red = training data drawn from the center distribution; dashed $n=2W$, solid $n=16W$. Read down a column: halfgauss stays at machine precision in every row; bimodal's descent shifts right as $s$ grows; beta converges at $s\le\tfrac23$ and stalls only at $s=1$.
- **`residual_N128_<distribution>.png`** -- $4\times4$ at $N=128$: rows = $s$, columns = the four functions; $y$ = signed residual fit $-$ true on a symmetric-log axis ($\pm10^{-15}$ linear near zero, fixed range $\pm1$), one line per training set (same colors and dashes as the error figures); grey ticks at $y=-0.6$ are the row's centers. Read the envelope against the ticks: the residual grows exactly where the ticks thin out.
- **`center_placement.png`** -- $4\times3$: rows = $s$, columns = the three center distributions. Ticks are the interior centers at $N=128$; grey is their histogram; red is the intended density $(1-s)\cdot\tfrac12+s\,q$; black dashed is the density that interpolating the center positions would have given instead. The panel text gives the largest neighbor-spacing ratio and the L1 distance between a smoothed histogram of the centers and the red curve. The histogram follows the red curve in every panel, and the ratio is $\le1.09$ everywhere except Beta at $s=1$.
- **`diagnostics/centers_and_gammas.png`** (a check of the geometry, not a result) -- rows = distributions, columns = $s$; the centers as ticks, $\gamma_j$ (blue, log scale) and the row's center density (red). Shows the spacing changes smoothly for halfgauss and bimodal, and that the Beta rows have a sparse, abruptly spaced right end.

## Additional details

The Beta $s=1$ failure mixes two defects: the neighbor-spacing jump at the vanishing-density end, and the halo built from that oversized end gap. Separating them (e.g. a halo built from the median interior spacing) was not done here.

## Conclusions

Approved by Sam:

- If the center distribution has enough centers, it has the right scaling law and reaches machine precision. The point of varying the centers is that each neuron locally needs the right $\gamma$: $\gamma_jh_j=0.25$ with $h_j$ the local spacing.
- More data is always more helpful than less data, in every case.
- Preliminary: matching the training data to the center distribution helps a little, but it is not definitive. In the width-scaling figures the two data placements are nearly indistinguishable (on the half-Gaussian spacing uniform data may suffer slightly at the larger widths, still to be seen). In the residual figures at $N=128$ the picture is sharper: on the Beta spacing the data that matches the centers pinches the residual tighter where the centers are dense, while in the sparse tail both placements are about the same, so the gain is swamped in the $L_2$ average by the tail. This is most visible at $n=2W$: on Beta, matched data with $n=2W$ actually does best (Runge in particular is at machine precision where the centers are dense and decent everywhere else), the reverse of what the width-scaling figures suggested; it just carries bad tails. On the bimodal spacing the $n=2W$ matched data also pinches slightly tighter than uniform. The half-Gaussian spacing is close to uniform, so nothing separates there.

## Open questions

- Measure the error limit as a function of the neighbor-spacing ratio (sweep the jump size at fixed $N$), and separate the end-gap halo effect from the interior jump.
- Does the same hold in 2-D along the offsets of a ridge direction (the suite's hotspot tasks are the natural test)?
- Since the widest gap decides the width, the useful question for an adaptive method is which regions can afford to be coarse: the suite's high-frequency-burst functions under data concentrated on or away from the burst.
