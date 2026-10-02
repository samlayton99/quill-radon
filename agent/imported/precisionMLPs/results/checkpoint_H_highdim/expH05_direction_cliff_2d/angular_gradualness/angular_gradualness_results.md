# expH05 / angular gradualness -- does the spoke density have to change gradually?

**Status: draft -- conclusions pending Sam's sign-off.**

## TL;DR

- Evenly spaced angles are a sharp optimum. Every other placement tested -- a smooth gradual density, a quarter-step jitter, and pure random -- costs about the same, and the three are nearly indistinguishable from each other. On fast concentric waves the even set is below $10^{-10}$ at $M=12$; all three perturbed sets need $M=24$.
- Gradualness is not the operative variable, which is the opposite of expH02's 1-D center result. The smooth gradual density has the most gradual spacing of the three perturbations (largest neighbouring-gap ratio $1.81$ at $M=32$) and is the *worst* of the three on radial Runge ($9.4\times10^{-12}$ versus jitter's $4.8\times10^{-13}$, whose neighbouring-gap ratio is $3.50$). What orders the placements is the widest angular gap, not how gradually the gaps change.
- Nothing stalls. The angular analogue of expH02's Beta(2,5) failure -- a density $\propto\sin^2\theta$ that vanishes at $\theta=0$ and leaves one gap $\sim\!4.3\times$ its neighbour at *every* $M$ (expH02's Beta ratio was $4.45$) -- reaches the $10^{-13}$ floor on two of the three functions by $M=24$ and is still falling on the third. It is not the worst curve at $M=32$; a perfectly smooth one-lobe density is, because its widest gap is larger.
- How much unevenness costs depends strongly on the function: nearly nothing on the composition (factor $1.4$-$3.4$ at $M=16$), one doubling of $M$ on fast waves, and one to five orders at $M=32$ on radial Runge, the one target that still separates the placements at the largest $M$ tested.

## Question

expH02 asked, for centers along a 1-D line, how non-uniform the spacing may be before the fixed-geometry least-squares fit stops reaching machine precision; smooth changes cost nothing and a spacing jump that did not shrink with $N$ stalled the error. This asks the same question about the *directions* of a 2-D ridge network: is a smoothly varying spoke density as good as evenly spaced spokes, and is there an angular analogue of the Beta(2,5) stall?

## Experiment design

**The fixed part.** Everything except the set of angles is expH05's $r=0.4$ setting, copied verbatim. Data uniform in the ball of radius $r=0.4$ about $x_0=(0.35,-0.25)$, $n_\text{train}=8\cdot\text{units}$ points at seed 0. Given $M$ angles $\theta_i$, the directions are $v_i=(\cos\theta_i,\sin\theta_i)$ and the ridge system is recentered on $x_0$: along each direction the offsets are $c = v_i\cdot x_0 + t$ with $t$ evenly spaced (cell-centered) over $[-T,T]$, $T=1.25\,r$ (the 25% collar), $n_\text{per}=128$ offsets on **every** spoke at **every** $M$, so along-direction resolution never binds. Widths from the spacing: $h=2T/n_\text{per}$, $\gamma=\lambda/h$, $\lambda=0.25$. Features $\Phi_{ij}=\tanh(\gamma_j(v_j\cdot x_i - c_j))$; one truncated-SVD least squares on $[\Phi,\mathbf 1]$ at $\texttt{rcond}=10^{-13}$. Units $=128M$, so at $M=32$ that is 4096 units and 32768 training points. Error is relative $L_2=\|\hat f-f\|_2/\|f\|_2$ on 20000 points uniform in the ball of radius $0.9\,r$, so the collar is never scored. Since $\Phi$ depends only on the angle set, one SVD serves all three targets.

**Angle space.** A direction is $v=(\cos\theta,\sin\theta)$ and $\theta$, $\theta+\pi$ are the same spoke, so the angles live on the circle $[0,\pi)$ and every density here is $\pi$-periodic. Gaps between neighbouring spokes are measured with the wrap-around gap $\theta_0+\pi-\theta_{M-1}$ included; the reported **neighbouring-gap ratio** is $\max_i\max(g_{i+1}/g_i,\;g_i/g_{i+1})$ over the circle (always $\ge1$; the even set gives exactly 1), the analogue of expH02's $\max_j h_{j+1}/h_j$. The **widest gap** is $\max_i g_i$, quoted in degrees; the even set's is $180^\circ/M$.

**Placement by inverse CDF (both figures).** For an angle density $q$ on $[0,\pi)$ and a non-uniformity level $s\in[0,1]$, the row's density is the mixture $q_s=(1-s)/\pi + s\,q$ and the spokes are placed at

$$\theta_i = Q_s^{-1}\!\Big(\frac{i+1/2}{M}\Big),\qquad i=0,\dots,M-1,$$

with $Q_s^{-1}$ the numeric inverse CDF of $q_s$ on a 20001-point grid. The half-step offset makes $s=0$ expH05's even set $\theta_i=(i+1/2)\pi/M$ (checked: the two agree to $4.4\times10^{-16}$, pure rounding), so the ladder starts from the reference and the $s=0$ row is computed once from the even set and reused.

- **Figure 1 placements**, four in all: (a) *even*, the reference; (b) *smooth gradual*, the mixture rule at $s=1$ with $q\propto 1+0.8\cos 2(\theta-\pi/4)$ -- smooth, never zero, ratio $9.0$ between its maximum and minimum density, and a largest neighbouring-gap ratio that runs from $3.09$ at $M=6$ down to $1.81$ at $M=32$; (c) *even + jitter*, each even angle shifted by $N(0,(0.25\,\pi/M)^2)$ (a quarter of a step) and wrapped mod $\pi$; (d) *random*, $M$ angles uniform on $[0,\pi)$. (c) and (d) use 3 seeds; the plot shows the median with a min/max band.
- **Figure 2 shapes**, at $s\in\{0,\tfrac13,\tfrac23,1\}$: (i) *one lobe*, $q\propto\exp(1.5\cos 2(\theta-\pi/4))$; (ii) *two lobes*, $q\propto\exp(1.5\cos 4(\theta-\pi/4))$, peaks at $45^\circ$ and $135^\circ$; (iii) *vanishing*, $q\propto\sin^2\theta$, which is zero at $\theta=0$ (equivalently $180^\circ$ -- the same spoke), the angular analogue of the endpoint where expH02's Beta(2,5) density vanished.

**Functions**, three, taken unchanged from expH05 with $\rho(x)=\|x-a\|_2/\sqrt2$ and $a=(0.3,-0.2)$: fast concentric waves $\cos(6\pi\rho)$ (symmetric, oscillatory), composition $\exp(\sin\pi x\cos\pi y)$ (asymmetric, smooth), radial Runge $1/(1+16\rho^2)$ (peaked). Directions $M\in\{4,6,8,12,16,24,32\}$.

**Code & data.** `experiments/expH05_direction_cliff_2d/angular_gradualness/run.py` (`--plot` replots from the saved data; self-contained -- the pieces of expH05 it uses are copied, not imported). Data `results/checkpoint_H_highdim/expH05_direction_cliff_2d/angular_gradualness/data.json` (357 rows: placement, $s$, seed, $M$, units, $n_\text{train}$, rel $L_2$, max abs, rank, readout norm, neighbouring-gap ratio, widest/narrowest gap ratio). Figures under `figures/`: `angle_spacing_vs_M.png`, `angle_spacing_h02_style.png`, `angle_placement.png`. 119 SVDs, 909 s total on 6 threads.

## Results

**Even is a sharp optimum, and it does not much matter how you leave it.** On fast concentric waves the even set falls to $4.8\times10^{-13}$ at $M=12$ and $3.0\times10^{-14}$ at $M=16$. At those same $M$ the smooth gradual density gives $2.7\times10^{-5}$ and $7.8\times10^{-8}$, the jitter $1.7\times10^{-5}$ and $4.1\times10^{-8}$, and pure random $3.4\times10^{-5}$ and $5.3\times10^{-8}$ -- five to six orders above even, and within a factor of about two of one another. All three reach the $\sim\!10^{-13}$ floor at $M=24$, so on this target the price of any unevenness is one doubling of $M$ ($12\to24$).

**The penalty depends on the function.** On the composition the placements barely separate: at $M=16$, even is $5.9\times10^{-10}$ and the three perturbations are $8.3\times10^{-10}$, $2.0\times10^{-9}$ and $2.0\times10^{-9}$, and everything is at the floor by $M=24$. Radial Runge is the discriminating target: even reaches $8.3\times10^{-11}$ at $M=16$ and $1.4\times10^{-14}$ at $M=32$, while at $M=32$ the smooth gradual density is at $9.4\times10^{-12}$, random at $1.7\times10^{-12}$ and jitter at $4.8\times10^{-13}$, and the worst case in the whole study, the one-lobe density at $s=1$, is at $1.7\times10^{-9}$.

**Gradualness is not what matters; the widest gap is.** The three figure-1 perturbations have very different neighbouring-gap ratios at $M=32$ -- smooth gradual $1.81$, jitter $3.50$ (median over seeds; worst seed $8.89$), random $174$ (worst seed $4834$) -- yet they land within a factor of 20 of each other on radial Runge, and in the wrong order for a gradualness story: the smoothest one is the worst. Their widest gaps run the other way and do track the errors: $24.9^\circ$ (smooth), $9.7^\circ$ (jitter), $22.3^\circ$ (random), against even's $5.6^\circ$. Sorting all thirteen placements at $M=32$ by widest gap orders their radial-Runge errors monotonically apart from two inversions:

| widest gap at $M=32$ | placement | neighbouring-gap ratio | radial Runge |
|---|---|---|---|
| $5.6^\circ$ | even | 1.00 | $1.4\times10^{-14}$ |
| $7.9$-$8.4^\circ$ | the three shapes at $s=1/3$ | 1.08-1.19 | $2.6$-$5.2\times10^{-14}$ |
| $9.7^\circ$ | jitter | 3.50 | $4.8\times10^{-13}$ |
| $13.1$-$16.4^\circ$ | the three shapes at $s=2/3$ | 1.24-1.40 | $0.9$-$1.7\times10^{-12}$ |
| $22.3^\circ$ | random | 174 | $1.7\times10^{-12}$ |
| $24.9^\circ$ | smooth gradual | 1.81 | $9.4\times10^{-12}$ |
| $30.7^\circ$ | two lobes, $s=1$ | 2.89 | $4.1\times10^{-12}$ |
| $37.0^\circ$ | one lobe, $s=1$ | 2.08 | $1.7\times10^{-9}$ |
| $48.6^\circ$ | vanishing, $s=1$ | 4.33 | $1.2\times10^{-10}$ |

The ordering holds only at fixed $M$; it is not a collapse. The even set at $M=8$ has a widest gap of $22.5^\circ$, almost exactly random's at $M=32$, but its radial-Runge error is $7.2\times10^{-6}$ against random's $1.7\times10^{-12}$. The extra spokes elsewhere still buy a great deal.

**No angular analogue of the Beta(2,5) stall.** The vanishing density $q\propto\sin^2\theta$ reproduces the structural feature that stalled expH02: because the density vanishes quadratically, the gap straddling $\theta=0$ stays about $4.3\times$ its neighbour at every $M$ ($4.00$, $4.13$, $4.21$, $4.28$, $4.33$ at $M=8,12,16,24,32$; expH02's Beta(2,5) ratio was $4.45$ and did not shrink either), while the two smooth lobed densities have ratios that fall with $M$ ($5.45\to2.08$ and $3.95\to2.89$ over $M=4\to32$). It does not stall: at $s=1$ it reaches $1.2\times10^{-13}$ on fast waves and $1.7\times10^{-14}$ on the composition by $M=24$, and on radial Runge it falls from $3.9\times10^{-9}$ at $M=24$ to $1.2\times10^{-10}$ at $M=32$ -- still descending where the sweep stops. Contrast expH02, where the Beta(2,5) centers sat at $3\times10^{-7}$ to $10^{-2}$ at every width. The two smooth lobed shapes converge as well, and one of them is worse than the vanishing one at $M=32$.

**Most of the damage happens early in $s$.** On fast waves at $M=12$ the even set is at $4.8\times10^{-13}$; moving only one third of the way to a non-uniform density already costs six orders ($6.3\times10^{-7}$ for one lobe, $\sim10^{-5}$ for the other two shapes), and going the rest of the way to $s=1$ costs about one more order. The three shapes lie almost on top of each other in nearly every panel.

### Figures

- **`figures/angle_spacing_vs_M.png`** -- one row, three panels, one per function. $x$ = number of directions $M$ (log$_2$ ticks at 4, 6, 8, 12, 16, 24, 32); $y$ = relative $L_2$ inside the ball on a fixed $[10^{-15},10]$ log axis. Four lines: black = even (reference), blue = smooth gradual density, green dashed = even + jitter, red dashed = uniform random; the two random placements show the median over 3 seeds with a light min/max band. What to look for: the black line separating from the other three and staying below them, and how little the other three separate from *each other*. The seed bands are narrow, so the random placements are not just an unlucky draw.
- **`figures/angle_spacing_h02_style.png`** -- $4\times3$, the expH02 analogue. Rows = non-uniformity level $s=0,\tfrac13,\tfrac23,1$; columns = the three functions; $x$ and $y$ as above. In each panel the thick grey line is the even reference ($s=0$) repeated for comparison, and the three coloured lines are the density shapes: blue = one lobe at $45^\circ$, orange = two lobes, red = vanishing at $0^\circ$. The top row is $s=0$, where all three shapes are the even set by construction, so the four lines coincide (a consistency check, not a result). What to look for: the gap between grey and the coloured lines opening as $s$ grows, and the red line -- the vanishing density -- sitting inside the bundle rather than peeling off, which is the point of the figure.
- **`figures/angle_placement.png`** -- $4\times4$ diagnostic at $M=16$, one panel per placement used. Top row: even, smooth gradual, jitter (seed 0), random (seed 0). Rows 2-4: the three figure-2 shapes across $s=0,\tfrac13,\tfrac23,1$. Blue ticks below the axis are the placed spokes over $0^\circ$ to $180^\circ$ ($0^\circ$ and $180^\circ$ are the same spoke); the red curve is the mixture density $q_s$ the spokes were placed by (a flat reference line for the jitter and random panels, which follow no density). Each panel prints its largest neighbouring-gap ratio and its widest/narrowest gap ratio. What to look for: the ticks thinning exactly where the red curve dips, the near-coincident pairs of ticks in the random panel (ratio 66) that the jitter panel does not have, and the wide empty stretch around $0^\circ$ in the bottom-right panel.

## Additional details

The analogy with expH02's Beta(2,5) is imperfect in one way worth stating. On the circle there is no boundary, so a vanishing density produces one wide gap and nothing else; in expH02 the Beta failure mixed the interior spacing jump with a halo built out of an oversized end gap, and the two were never separated. The angular experiment therefore tests the gap-jump half of that failure, not the halo half.

The sweep stops at $M=32$ (4096 units, 32768 training points, a $32768\times4097$ SVD). "Still falling" for the worst curves is a statement about the range measured, not a claim that they reach the floor.

## Conclusions

Pending Sam's review. What the data plainly shows:

- Evenly spaced spokes are strictly best at every $M$ on all three functions, and the three ways of leaving evenness tested here (smooth gradual density, quarter-step jitter, pure random) cost about the same.
- Among these angle sets, the widest angular gap orders the error at fixed $M$; the neighbouring-gap ratio does not.
- No angle placement tested stalls, including one whose neighbouring-gap ratio stays at $4.3$ at every $M$ -- the feature that stalled the 1-D center experiment.

## Open questions

- Does the widest-gap ordering hold at larger $M$, and does the worst case (one lobe at $s=1$, $1.7\times10^{-9}$ at $M=32$) eventually reach the floor?
- Radial Runge is the only target here that still separates the placements at $M=32$. Is that a property of peaked targets generally, or of this one?
- Spokes and offsets were varied separately (this study and expH05's split heat map). An adaptive method would move both; nothing here says what happens when a non-uniform angle set is paired with non-uniform offsets.
